"""Reviewed labels and source-group partitions used only for training/evaluation."""

import json

from .config import DATASETS
from .reranking import passage_text
from .retrieval import normalize

QUESTION_FILES = ("questions.jsonl", "fresh-questions.jsonl", "neural-holdout.jsonl")
CHECK_SPLITS = ("dev", "test", "fresh", "neural_holdout")


def questions():
    return [
        json.loads(line)
        for name in QUESTION_FILES
        for line in (DATASETS / name).read_text().splitlines()
        if line.strip()
    ]


def validate_splits(rows):
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Question IDs must be unique")
    group_splits = {}
    for row in rows:
        split = row["split"]
        if split not in ("train", *CHECK_SPLITS):
            raise ValueError(f"Unknown question split: {split}")
        group = row["group"]
        if group in group_splits and group_splits[group] != split:
            raise ValueError("Source groups overlap between splits")
        group_splits[group] = split


def relevant(question, chunk):
    anchors = question.get("evidence_anchors") or [question["evidence_anchor"]]
    return (
        question["answerable"]
        and chunk["doc_id"] == question["doc_id"]
        and any(normalize(anchor) in normalize(chunk["text"]) for anchor in anchors)
    )


def reviewed_examples(retriever, rows):
    validate_splits(rows)
    lookup = {row["id"]: row for row in rows}
    labels = [
        json.loads(line)
        for line in (DATASETS / "training-labels.jsonl").read_text().splitlines()
        if line.strip()
    ]
    heldout = {
        (row["doc_id"], page)
        for row in rows
        if row["split"] != "train"
        for page in row["gold_pages"]
    }
    candidates = {
        row["id"]: {
            chunk["id"]: chunk for chunk in retriever.candidates(row["question"])
        }
        for row in rows
        if row["split"] == "train"
    }
    examples, seen = [], set()
    for pair in labels:
        key = (pair["question_id"], pair["chunk_id"])
        if key in seen:
            raise ValueError("Duplicate training pair")
        seen.add(key)
        question = lookup.get(pair["question_id"])
        if question is None or question["split"] != "train":
            raise ValueError("Training label refers to an unknown or held-out question")
        chunk = candidates[question["id"]].get(pair["chunk_id"])
        if chunk is None:
            raise ValueError(
                "Labeled chunk is outside the candidate pool; review labels after changing retrieval"
            )
        if (chunk["doc_id"], chunk["page"]) in heldout:
            raise ValueError("Held-out source page leaked into training")
        reason = pair.get("reason", "")
        if pair["label"] not in (0, 1) or not reason.strip() or "PROPOSED" in reason:
            raise ValueError("Every training label needs explicit source review")
        examples.append(
            {**pair, "question": question["question"], "passage": passage_text(chunk)}
        )
    if not examples or {example["label"] for example in examples} != {0, 1}:
        raise ValueError("Training requires reviewed positive and negative examples")
    return examples
