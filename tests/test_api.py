import json
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from coso_rag.config import artifact_paths
from coso_rag.openrouter import APIError, BudgetExceeded, OpenRouter

PATHS = artifact_paths()


def completion(cost=0.001):
    return {
        "id": "test-generation",
        "provider": "test",
        "choices": [{"finish_reason": "stop", "message": {"content": '{"ok":true}'}}],
        "usage": {"cost": cost, "prompt_tokens": 10, "completion_tokens": 5},
    }


def test_budget_blocks_before_network_and_cache_is_free(tmp_path):
    sent = []

    def respond(request):
        payload = json.loads(request.content)
        sent.append(payload)
        assert payload["provider"]["max_price"] == {"prompt": 0.3, "completion": 1.2}
        assert payload["max_tokens"] == 100
        return httpx.Response(200, json=completion())

    client = OpenRouter(
        key="test",
        budget=0.003,
        directory=tmp_path,
        transport=httpx.MockTransport(respond),
    )
    msg = [{"role": "user", "content": "hello"}]
    assert client.complete(msg, purpose="test", max_tokens=100)["_cached"] is False
    assert client.complete(msg, purpose="test", max_tokens=100)["_cached"] is True
    assert len(sent) == 1
    assert client.usage()["charged_usd"] == 0.001
    with pytest.raises(BudgetExceeded):
        client.complete(
            [{"role": "user", "content": "x" * 10000}],
            purpose="blocked",
            max_tokens=100,
        )
    assert len(sent) == 1


def test_unknown_request_keeps_reservation(tmp_path):
    def fail(request):
        return httpx.Response(503, json={"error": "unavailable"})

    client = OpenRouter(
        key="test", directory=tmp_path, transport=httpx.MockTransport(fail)
    )
    with pytest.raises(APIError):
        client.complete([{"role": "user", "content": "hello"}], purpose="failure")
    usage = client.usage()
    assert usage["charged_usd"] == 0
    assert usage["reserved_usd"] > 0
    assert usage["history"][0]["state"] == "unknown"


def test_nonfinite_billing_cost_keeps_its_reservation(tmp_path):
    client = OpenRouter(
        key="test",
        directory=tmp_path,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, content=json.dumps(completion(float("nan")))
            )
        ),
    )
    with pytest.raises(APIError, match="billing"):
        client.complete([{"role": "user", "content": "test"}], purpose="invalid-cost")
    assert client.usage()["charged_usd"] == 0
    assert client.usage()["reserved_usd"] > 0


def test_parallel_reservations_cannot_overrun_local_budget(tmp_path):
    import time

    sent = []

    def slow(request):
        sent.append(1)
        time.sleep(0.1)
        return httpx.Response(200, json=completion(0.0013))

    client = OpenRouter(
        key="test",
        budget=0.0015,
        directory=tmp_path,
        transport=httpx.MockTransport(slow),
    )

    def run(i):
        try:
            return client.complete(
                [{"role": "user", "content": str(i)}],
                purpose="parallel",
                max_tokens=100,
            )
        except BudgetExceeded:
            return "blocked"

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, range(4)))
    assert results.count("blocked") == 3
    assert len(sent) == 1
