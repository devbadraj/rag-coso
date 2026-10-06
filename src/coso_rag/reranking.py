"""Score question–passage pairs with the pinned base and optional adapted weights."""

import hashlib
import json
import math
import threading
from pathlib import Path

from .config import RERANKER, RERANKER_REVISION, SEARCH, TRAINING, artifact_paths

BASE_MODEL = RERANKER
BASE_REVISION = RERANKER_REVISION
ADAPTER_NAME = "text-reranker.safetensors"


def trainable_prefixes(model, last_layers):
    total_layers = model.config.num_hidden_layers
    if not 1 <= last_layers <= total_layers:
        raise ValueError("Trainable layer count is outside the base model")
    return tuple(
        f"{model.base_model_prefix}.encoder.layer.{layer}."
        for layer in range(total_layers - last_layers, total_layers)
    ) + ("classifier.",)


def passage_text(chunk):
    sections = ", ".join(chunk.get("sections", []))
    return "\n".join(part for part in (chunk["title"], sections, chunk["text"]) if part)


def adapter_path(directory, metadata):
    # Metadata may come from a shipped file. Never load outside its artifact directory.
    name = metadata.get("adapter_file")
    if name != ADAPTER_NAME:
        raise ValueError("Invalid text-reranker checkpoint path")
    path = Path(directory) / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["adapter_sha256"]:
        raise ValueError("Text-reranker checkpoint hash mismatch")
    return path


class TextReranker:
    def __init__(
        self, metadata, directory=None, *, model=None, tokenizer=None, pretrained=False
    ):
        import torch

        self.metadata = (
            {**metadata, "adapter_sha256": BASE_REVISION} if pretrained else metadata
        )
        self.paths = artifact_paths(directory)
        self.pretrained = pretrained
        self.max_length = metadata.get("max_length", SEARCH.max_tokens)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.lock = threading.RLock()
        self.model, self.tokenizer = model, tokenizer
        self.cache_path = self.paths.cache / "text-score-cache.json"
        self.cache = (
            json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        )

    def load(self):
        if self.model is not None:
            return
        from safetensors.torch import load_file
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if (
            self.metadata["base_model"] != BASE_MODEL
            or self.metadata["base_revision"] != BASE_REVISION
        ):
            raise ValueError("Unsupported text-reranker base revision")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
        model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL,
            revision=BASE_REVISION,
            use_safetensors=True,
            attn_implementation="eager",
        )
        if not self.pretrained:
            path = adapter_path(self.paths.model, self.metadata)
            state = load_file(str(path))
            prefixes = tuple(self.metadata["trainable_prefixes"])
            allowed = trainable_prefixes(model, len(prefixes) - 1)
            if prefixes != allowed:
                raise ValueError(
                    "Checkpoint trainable layers do not match its base model"
                )
            expected = {
                name
                for name, _ in model.named_parameters()
                if name.startswith(prefixes)
            }
            if set(state) != expected:
                raise ValueError(
                    "Checkpoint has incomplete or unexpected trainable tensors"
                )
            model.load_state_dict(state, strict=False)
        model.to(self.device).eval()
        self.model, self.tokenizer = model, tokenizer

    def score(self, question, candidates):
        import torch

        version = self.metadata.get("adapter_sha256", BASE_REVISION)
        keys = [
            hashlib.sha256(
                json.dumps(
                    [
                        BASE_REVISION,
                        version,
                        self.max_length,
                        question,
                        passage_text(c),
                    ],
                    ensure_ascii=False,
                ).encode()
            ).hexdigest()
            for c in candidates
        ]
        with self.lock:
            missing = [i for i, key in enumerate(keys) if key not in self.cache]
            if missing:
                self.load()
                self.model.eval()
                for start in range(0, len(missing), TRAINING.inference_batch_size):
                    indices = missing[start : start + TRAINING.inference_batch_size]
                    features = self.tokenizer(
                        [question] * len(indices),
                        [passage_text(candidates[i]) for i in indices],
                        padding=True,
                        truncation="only_second",
                        max_length=self.max_length,
                        return_tensors="pt",
                    ).to(self.device)
                    with torch.inference_mode():
                        values = (
                            self.model(**features)
                            .logits.flatten()
                            .detach()
                            .cpu()
                            .tolist()
                        )
                    for i, value in zip(indices, values):
                        if not math.isfinite(value):
                            raise RuntimeError(
                                "Text-reranker returned a nonfinite score"
                            )
                        self.cache[keys[i]] = value
            return [self.cache[key] for key in keys]

    def save_cache(self):
        with self.lock:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache))
