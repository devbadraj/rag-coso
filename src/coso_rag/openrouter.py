"""Bounded paid calls; reserve worst-case cost before sending, cache successes.

Reservations survive unknown outcomes. There are no automatic paid retries.
SQLite transactions make concurrent browser sessions share the same budget.
"""

import hashlib
import json
import math
import os
import sqlite3
import time
from pathlib import Path

import httpx

from .config import API, MODEL, ROOT


class APIError(RuntimeError):
    pass


class BudgetExceeded(APIError):
    pass


class OpenRouter:
    INPUT_CAP = (
        API.input_price_cap
    )  # USD / million tokens, enforced in provider routing
    OUTPUT_CAP = API.output_price_cap

    def __init__(self, key=None, budget=None, directory=None, transport=None):
        self.key = os.getenv("OPENROUTER_API_KEY", "") if key is None else key
        self.budget = float(
            budget
            if budget is not None
            else os.getenv("COSO_BUDGET_USD", str(API.budget_usd))
        )
        if not 0 < self.budget < 1000:
            raise ValueError("Budget must be between $0 and $1000")
        self.directory = Path(directory or ROOT / ".local")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.database = self.directory / "usage.sqlite"
        self.transport = transport
        with self.connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS calls (id INTEGER PRIMARY KEY, cache_key TEXT, purpose TEXT, reserved REAL, cost REAL, state TEXT, created REAL, tokens TEXT, provider TEXT, generation TEXT)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, response TEXT)"
            )

    def connect(self):
        return sqlite3.connect(self.database, timeout=30)

    def usage(self):
        with self.connect() as conn:
            charged, reserved, count = conn.execute(
                "SELECT COALESCE(SUM(cost),0), COALESCE(SUM(CASE WHEN cost IS NULL THEN reserved ELSE 0 END),0), COUNT(*) FROM calls"
            ).fetchone()
            rows = conn.execute(
                "SELECT purpose,cost,reserved,state,provider,generation FROM calls ORDER BY id"
            ).fetchall()
        return {
            "charged_usd": round(charged, 8),
            "reserved_usd": round(reserved, 8),
            "limit_usd": self.budget,
            "calls": count,
            "history": [
                dict(
                    zip(
                        [
                            "purpose",
                            "cost",
                            "reserved",
                            "state",
                            "provider",
                            "generation",
                        ],
                        r,
                    )
                )
                for r in rows
            ],
        }

    def complete(
        self,
        messages,
        *,
        purpose,
        max_tokens=API.answer_tokens,
        schema=None,
        image_count=0,
    ):
        payload = {
            "model": MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "reasoning": {"enabled": False},
            "usage": {"include": True},
            "provider": {
                "sort": "price",
                "require_parameters": True,
                "max_price": {"prompt": self.INPUT_CAP, "completion": self.OUTPUT_CAP},
            },
        }
        if schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "result", "strict": True, "schema": schema},
            }
        # Images are resized to <=1600px by ingestion; reserve 50k tokens per image.
        # Count UTF-8 bytes for text rather than assuming English chars/token.
        text_bytes = sum(
            len(json.dumps(m, ensure_ascii=False).encode())
            for m in messages
            if isinstance(m.get("content"), str)
        )
        for m in messages:
            if isinstance(m.get("content"), list):
                text_bytes += sum(len(x.get("text", "").encode()) for x in m["content"])
        input_bound = (
            text_bytes
            + API.request_token_reserve
            + image_count * API.image_token_reserve
        )
        reservation = (
            input_bound * self.INPUT_CAP + max_tokens * self.OUTPUT_CAP
        ) / 1_000_000
        cache_key = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        with self.connect() as conn:
            cached = conn.execute(
                "SELECT response FROM cache WHERE key=?", (cache_key,)
            ).fetchone()
            if cached:
                result = json.loads(cached[0])
                result["_cached"] = True
                return result
        if not self.key:
            raise APIError(
                "Set OPENROUTER_API_KEY in .env to generate an answer. Retrieval works without a key."
            )
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            spent = conn.execute(
                "SELECT COALESCE(SUM(COALESCE(cost,reserved)),0) FROM calls"
            ).fetchone()[0]
            if spent + reservation > self.budget:
                raise BudgetExceeded(
                    f"Local ${self.budget:.2f} budget reached. No request sent; ${spent:.4f} already spent or reserved."
                )
            cursor = conn.execute(
                "INSERT INTO calls(cache_key,purpose,reserved,state,created) VALUES(?,?,?,'pending',?)",
                (cache_key, purpose, reservation, time.time()),
            )
            call_id = cursor.lastrowid
        try:
            with httpx.Client(
                timeout=API.timeout_seconds, transport=self.transport
            ) as client:
                response = client.post(
                    API.endpoint,
                    json=payload,
                    headers={
                        "Authorization": "Bearer " + self.key,
                        "X-Title": "COSO focused RAG assessment",
                    },
                )
                if response.status_code != 200:
                    raise APIError(
                        f"OpenRouter returned HTTP {response.status_code}. No automatic retry; the cost reservation remains until reconciled."
                    )
                result = response.json()
                if result.get("error") or not result.get("choices"):
                    raise APIError(
                        "Provider did not return a completion. No automatic retry."
                    )
                if result["choices"][0].get("finish_reason") != "stop":
                    raise APIError(
                        "Provider output was truncated or interrupted. No automatic retry."
                    )
                usage = result.get("usage", {})
                cost = usage.get("cost")
                if cost is not None and (
                    not isinstance(cost, (int, float))
                    or not math.isfinite(cost)
                    or cost < 0
                ):
                    raise APIError("Invalid billing metadata")
                with self.connect() as conn:
                    conn.execute(
                        "UPDATE calls SET cost=?,state=?,tokens=?,provider=?,generation=? WHERE id=?",
                        (
                            cost,
                            "complete" if cost is not None else "unreconciled",
                            json.dumps(usage),
                            result.get("provider"),
                            result.get("id"),
                            call_id,
                        ),
                    )
                    conn.execute(
                        "INSERT OR REPLACE INTO cache VALUES(?,?)",
                        (cache_key, json.dumps(result)),
                    )
                result["_cached"] = False
                return result
        except Exception as exc:
            with self.connect() as conn:
                conn.execute(
                    "UPDATE calls SET state='unknown' WHERE id=? AND state='pending'",
                    (call_id,),
                )
            if isinstance(exc, APIError):
                raise
            raise APIError(
                "OpenRouter request failed; reservation retained. Check connectivity before explicitly retrying."
            ) from None
