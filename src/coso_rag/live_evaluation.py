"""Explicit paid evaluation; saves outputs for review, never claims semantic
correctness based solely on the citation validator passing.
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from .answering import answer
from .config import API, REPORTS
from .datasets import questions
from .openrouter import APIError, OpenRouter
from .retrieval import Retriever


def evaluate_live(limit=None, client=None):
    if limit is not None and limit < 1:
        raise ValueError("Answer evaluation limit must be positive")
    r = Retriever()
    client = client or OpenRouter()
    rows = [q for q in questions() if q["split"] == "test"]
    if limit is not None:
        rows = rows[:limit]
    trained = bool(r.model and r.model.get("activate_by_default"))
    # Preload query vectors once before worker threads. This also handles missing-
    # evidence questions, which the retrieval metric intentionally skips.
    for q in rows:
        r.query_vector(q["question"])
    r.save_query_cache()

    def run(q):
        try:
            result = answer(q["question"], r, client=client, trained=trained)
            result["evidence"] = [
                {key: source[key] for key in ("id", "doc_id", "page")}
                for source in result["evidence"]
            ]
            return {
                "id": q["id"],
                "question": q["question"],
                "expected": q["expected"],
                "answerable": q["answerable"],
                "validation": "pass",
                "result": result,
                "status_matches": (result["status"] == "answered") == q["answerable"],
            }
        except APIError as exc:
            return {
                "id": q["id"],
                "question": q["question"],
                "expected": q["expected"],
                "answerable": q["answerable"],
                "validation": "fail",
                "error": str(exc),
                "status_matches": False,
            }

    results = []
    # Queries in the supplied evaluation are cached vectors, so no model loading race.
    with ThreadPoolExecutor(max_workers=API.evaluation_workers) as pool:
        pending = [pool.submit(run, q) for q in rows]
        for future in as_completed(pending):
            item = future.result()
            results.append(item)
            print(
                f"{len(results)}/{len(rows)} {item['id']}: {item['validation']}",
                flush=True,
            )
    order = {row["id"]: index for index, row in enumerate(rows)}
    results.sort(key=lambda item: order[item["id"]])
    report = {
        "ranking": "trained" if trained else "baseline",
        "index_fingerprint": r.metadata["fingerprint"],
        "questions": len(rows),
        "citation_validation_passes": sum(x["validation"] == "pass" for x in results),
        "expected_status_matches": sum(x["status_matches"] for x in results),
        "results": results,
        "usage": {
            key: value for key, value in client.usage().items() if key != "history"
        },
        "evidence_storage": "Page references point to artifacts/index/pages.jsonl and the original PDFs; claims retain the exact supporting quotes.",
        "semantic_review": "Pending manual comparison with source and expected answer.",
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "answers-latest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    return {k: v for k, v in report.items() if k != "results"}
