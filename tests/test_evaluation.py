import json

import pytest

from coso_rag.config import DATASETS, REPORTS, artifact_paths
from coso_rag.datasets import questions
from coso_rag.evaluation import deployment_decision, metrics
from coso_rag.reranking import TextReranker
from coso_rag.retrieval import Retriever

PATHS = artifact_paths()


def test_search_and_reranking_match_all_saved_comparisons(monkeypatch):
    report = json.loads((REPORTS / "retrieval.json").read_text())
    rows = questions()
    retriever = Retriever()
    adapted = TextReranker(retriever.model)
    pretrained = TextReranker(retriever.model, pretrained=True)

    def prevent_uncached_inference():
        raise AssertionError("Expected the shipped evaluation pairs to be cached")

    monkeypatch.setattr(adapted, "load", prevent_uncached_inference)
    monkeypatch.setattr(pretrained, "load", prevent_uncached_inference)
    for split, modes in report["splits"].items():
        checks = [row for row in rows if row["split"] == split]
        for mode, expected in modes.items():
            retriever.text_reranker = pretrained if mode == "pretrained" else adapted
            assert metrics(retriever, checks, mode != "baseline") == expected


def test_split_and_training_labels_exclude_held_out_sources():
    qs = questions()
    lookup = {q["id"]: q for q in qs}
    groups = {
        s: {q["group"] for q in qs if q["split"] == s}
        for s in ["train", "dev", "test", "fresh", "neural_holdout"]
    }
    assert not any(groups[a] & groups[b] for a in groups for b in groups if a != b)
    heldout = {
        (q["doc_id"], p) for q in qs if q["split"] != "train" for p in q["gold_pages"]
    }
    chunks = {c["id"]: c for c in Retriever().chunks}
    for pair in [
        json.loads(line)
        for line in (DATASETS / "training-labels.jsonl").read_text().splitlines()
    ]:
        assert lookup[pair["question_id"]]["split"] == "train"
        c = chunks[pair["chunk_id"]]
        assert (c["doc_id"], c["page"]) not in heldout
        assert "PROPOSED" not in pair["reason"]


def test_deployment_blocks_regression_and_never_changes_fitted_weights():
    model = {"development_eligible": True, "adapter_sha256": "frozen"}
    snapshot = json.loads(json.dumps(model))
    splits = {
        s: {
            "baseline": {"questions": 10, "hit_at_5": 0.8, "mrr": 0.6},
            "trained": {"questions": 10, "hit_at_5": 0.8, "mrr": 0.7},
        }
        for s in ["dev", "test", "fresh", "neural_holdout"]
    }
    assert deployment_decision(model, splits)["status"] == "approved"
    assert (
        deployment_decision(
            model, {s: data for s, data in splits.items() if s != "neural_holdout"}
        )["status"]
        == "blocked"
    )
    splits["fresh"]["trained"]["mrr"] = 0.5
    decision = deployment_decision(model, splits)
    assert decision["status"] == "blocked"
    assert decision["regressions"] == [
        {"split": "fresh", "measure": "mrr", "baseline": 0.6, "trained": 0.5}
    ]
    assert model == snapshot
    assert deployment_decision(model, {"dev": splits["dev"]})["status"] == "blocked"


def test_python_api_respects_blocked_model_and_explicit_training_override():
    r = Retriever()
    r.model["activate_by_default"] = False
    q = next(q["question"] for q in questions() if q["id"] == "mobilization-1")
    assert r.search(q) == r.search(q, trained=False)
    assert r.search(q, trained=True) != r.search(q, trained=False)


def test_explicit_trained_search_rejects_a_missing_model():
    retriever = Retriever()
    retriever.model = None
    with pytest.raises(ValueError, match="compatible trained model"):
        retriever.search("What is the maximum advance?", trained=True)


def test_training_rejects_held_out_page_before_fitting(tmp_path, monkeypatch):
    from coso_rag import datasets, training

    q = next(q for q in questions() if q["id"] == "guarantee-1")
    c = next(
        c
        for c in Retriever().candidates(q["question"])
        if c["doc_id"] == q["doc_id"] and c["page"] in q["gold_pages"]
    )
    held = {**q, "id": "reserved", "group": "reserved-source", "split": "dev"}
    evaluation = tmp_path / "datasets"
    evaluation.mkdir()
    (evaluation / "training-labels.jsonl").write_text(
        json.dumps(
            {
                "question_id": q["id"],
                "chunk_id": c["id"],
                "label": 1,
                "reason": "Reviewed",
            }
        )
        + "\n"
    )
    monkeypatch.setattr(datasets, "DATASETS", evaluation)
    monkeypatch.setattr(datasets, "questions", lambda: [q, held])
    with pytest.raises(ValueError, match="Held-out source page"):
        training.train()
