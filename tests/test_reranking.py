import hashlib
import json

import pytest

from coso_rag.config import DATASETS, ROOT, artifact_paths

PATHS = artifact_paths()


def test_neural_checkpoint_is_real_partial_fine_tuning_and_matches_frozen_protocol():
    from safetensors.torch import load_file

    from coso_rag.reranking import adapter_path

    model = json.loads((PATHS.model / "reranker.json").read_text())
    assert model["kind"] == "cross_encoder"
    path = adapter_path(PATHS.model, model)
    state = load_file(str(path))
    assert len(state) == model["changed_tensors"] == 36
    assert sum(t.numel() for t in state.values()) == model["trainable_parameters"]
    assert all(name.startswith(tuple(model["trainable_prefixes"])) for name in state)
    assert all(norm > 0 for norm in model["weight_change_norms"].values())
    protocol = json.loads((ROOT / "evaluation/protocol.json").read_text())
    assert protocol["adapter_sha256"] == model["adapter_sha256"]
    assert (
        hashlib.sha256((ROOT / "evaluation/frozen-model.json").read_bytes()).hexdigest()
        == protocol["frozen_model_sha256"]
    )
    assert (
        hashlib.sha256((DATASETS / "training-labels.jsonl").read_bytes()).hexdigest()
        == model["training_labels_sha256"]
    )
    assert (
        hashlib.sha256((DATASETS / "neural-holdout.jsonl").read_bytes()).hexdigest()
        == protocol["neural_holdout_sha256"]
    )


def test_neural_checkpoint_rejects_wrong_hash_and_path(tmp_path):
    from coso_rag.reranking import ADAPTER_NAME, adapter_path

    (tmp_path / ADAPTER_NAME).write_bytes(b"checkpoint fixture")
    with pytest.raises(ValueError, match="hash mismatch"):
        adapter_path(
            tmp_path, {"adapter_file": ADAPTER_NAME, "adapter_sha256": "wrong"}
        )
    with pytest.raises(ValueError, match="checkpoint path"):
        adapter_path(tmp_path, {"adapter_file": "../outside.safetensors"})


def test_neural_pair_cache_is_invalidated_by_question_passage_and_checkpoint(tmp_path):
    from types import SimpleNamespace

    import torch

    from coso_rag.reranking import TextReranker

    calls = []

    class Batch(dict):
        def to(self, device):
            return self

    class Tokenizer:
        def __call__(self, queries, passages, **kwargs):
            calls.append((queries, passages, kwargs))
            return Batch(input_ids=torch.ones((len(queries), 2), dtype=torch.int64))

    class Model:
        def eval(self):
            return self

        def __call__(self, **features):
            return SimpleNamespace(
                logits=torch.arange(
                    len(features["input_ids"]), dtype=torch.float32
                ).reshape(-1, 1)
            )

    scorer = TextReranker(
        {"adapter_sha256": "checkpoint-1"},
        tmp_path,
        model=Model(),
        tokenizer=Tokenizer(),
    )
    c = {
        "title": "Contract",
        "text": "The maximum is ten percent.",
        "sections": ["CPWD Clause 2"],
    }
    assert scorer.score("What is the maximum?", [c]) == [0.0]
    assert scorer.score("What is the maximum?", [c]) == [0.0]
    assert len(calls) == 1
    assert calls[0][0] == ["What is the maximum?"]
    assert calls[0][1] == ["Contract\nCPWD Clause 2\nThe maximum is ten percent."]
    assert calls[0][2]["truncation"] == "only_second"
    scorer.score("What is the limit?", [c])
    scorer.score("What is the maximum?", [{**c, "text": "A different rule."}])
    scorer.metadata["adapter_sha256"] = "checkpoint-2"
    scorer.score("What is the maximum?", [c])
    assert len(calls) == 4
