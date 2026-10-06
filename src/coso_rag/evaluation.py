"""Compare original, pretrained and adapted rankings without changing weights."""

import json

from .config import REPORTS, SEARCH, artifact_paths
from .datasets import CHECK_SPLITS, questions, relevant, validate_splits
from .reranking import TextReranker
from .retrieval import Retriever


def metrics(retriever, rows, trained):
    details = []
    for question in rows:
        if not question["answerable"]:
            continue
        found = retriever.search(
            question["question"], trained=trained, k=SEARCH.candidate_pool * 2
        )
        rank = next(
            (i for i, chunk in enumerate(found, 1) if relevant(question, chunk)), None
        )
        details.append(
            {
                "id": question["id"],
                "rank": rank,
                "hit_at_5": rank is not None and rank <= 5,
                "candidate_hit": any(
                    relevant(question, chunk)
                    for chunk in retriever.candidates(question["question"])
                ),
                "top_ids": [chunk["id"] for chunk in found[: SEARCH.results]],
            }
        )
    denominator = max(len(details), 1)
    return {
        "questions": len(details),
        "hit_at_5": sum(row["hit_at_5"] for row in details) / denominator,
        "mrr": sum(1 / row["rank"] if row["rank"] else 0 for row in details)
        / denominator,
        "candidate_recall": sum(row["candidate_hit"] for row in details) / denominator,
        "details": details,
    }


def deployment_decision(model, splits):
    regressions = [
        {
            "split": split,
            "measure": measure,
            "baseline": modes["baseline"][measure],
            "trained": modes["trained"][measure],
        }
        for split, modes in splits.items()
        for measure in ("hit_at_5", "mrr")
        if modes["trained"][measure] + 1e-12 < modes["baseline"][measure]
    ]
    enough_checks = all(
        splits.get(split, {}).get("trained", {}).get("questions", 0)
        for split in CHECK_SPLITS
    )
    approved = (
        model.get("development_eligible", False) and enough_checks and not regressions
    )
    return {
        "status": "approved" if approved else "blocked",
        "rule": "Development improvement and no hit@5 or MRR regression against original search on any held-out split; all configured check sets required",
        "regressions": regressions,
        "weights_changed_by_evaluation": False,
    }


def evaluate(directory=None):
    paths = artifact_paths(directory)
    retriever = Retriever(paths.root)
    if retriever.model is None:
        raise ValueError("Train the reranker before evaluation")
    rows = questions()
    validate_splits(rows)
    results = {
        "index_fingerprint": retriever.metadata["fingerprint"],
        "split_counts": {
            split: sum(row["split"] == split for row in rows)
            for split in ("train", *CHECK_SPLITS)
        },
        "model": retriever.model["selection"],
        "splits": {},
    }
    pretrained = TextReranker(retriever.model, paths.root, pretrained=True)
    adapted = TextReranker(retriever.model, paths.root)
    adapted.cache = pretrained.cache
    for split in CHECK_SPLITS:
        checks = [row for row in rows if row["split"] == split]
        modes = {"baseline": metrics(retriever, checks, False)}
        retriever.text_reranker = pretrained
        modes["pretrained"] = metrics(retriever, checks, True)
        retriever.text_reranker = adapted
        modes["trained"] = metrics(retriever, checks, True)
        results["splits"][split] = modes
    decision = deployment_decision(retriever.model, results["splits"])
    retriever.model["deployment"] = decision
    retriever.model["activate_by_default"] = decision["status"] == "approved"
    results["deployment"] = decision
    results["default"] = (
        "trained" if retriever.model["activate_by_default"] else "baseline"
    )
    (paths.model / "reranker.json").write_text(json.dumps(retriever.model, indent=2))
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "retrieval.json").write_text(json.dumps(results, indent=2))
    retriever.save_query_cache()
    adapted.save_cache()
    return results
