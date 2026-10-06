"""Partial fine-tuning on reviewed question–passage pairs; dev selects the epoch."""

import hashlib
import json
import math

import numpy as np

from .config import DATASETS, SEARCH, TRAINING, artifact_paths
from .reranking import (
    ADAPTER_NAME,
    BASE_MODEL,
    BASE_REVISION,
    TextReranker,
    trainable_prefixes,
)


def train(directory=None):
    from .datasets import questions, reviewed_examples
    from .evaluation import metrics
    from .retrieval import Retriever

    paths = artifact_paths(directory)
    directory = paths.root
    paths.model.mkdir(parents=True, exist_ok=True)
    retriever = Retriever(directory)
    rows = questions()
    examples = reviewed_examples(
        retriever, rows
    )  # Check leakage before downloading a model.
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(TRAINING.seed)
    rng = np.random.default_rng(TRAINING.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.set_num_threads(TRAINING.cpu_threads)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        revision=BASE_REVISION,
        use_safetensors=True,
        attn_implementation="eager",
    )
    prefixes = trainable_prefixes(model, TRAINING.last_layers)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(prefixes))
    initial = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }
    model.to(device)
    metadata = {
        "kind": "cross_encoder",
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "index_fingerprint": retriever.metadata["fingerprint"],
        "max_length": SEARCH.max_tokens,
        "training_pairs": len(examples),
        "positives": sum(e["label"] for e in examples),
        "negatives": sum(e["label"] == 0 for e in examples),
        "question_ids": sorted({e["question_id"] for e in examples}),
        "trainable_prefixes": list(prefixes),
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "training_labels_sha256": hashlib.sha256(
            (DATASETS / "training-labels.jsonl").read_bytes()
        ).hexdigest(),
        "learning_rate": TRAINING.learning_rate,
        "weight_decay": TRAINING.weight_decay,
        "batch_size": TRAINING.batch_size,
        "epochs": TRAINING.epochs,
        "seed": TRAINING.seed,
        "device": device,
        "loss": "BCEWithLogitsLoss",
        "attention_implementation": "eager",
        "activate_by_default": False,
        "deployment": {"status": "pending_evaluation"},
    }
    scorer = TextReranker(metadata, directory, model=model, tokenizer=tokenizer)
    scorer.device = device
    retriever.model, retriever.text_reranker = metadata, scorer
    dev = [q for q in rows if q["split"] == "dev"]
    baseline = metrics(retriever, dev, False)
    pretrained = metrics(retriever, dev, True)
    print(
        json.dumps(
            {
                "stage": "pretrained_development",
                "hit_at_5": pretrained["hit_at_5"],
                "mrr": pretrained["mrr"],
                "device": device,
            }
        ),
        flush=True,
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=TRAINING.learning_rate,
        weight_decay=TRAINING.weight_decay,
    )
    steps = TRAINING.epochs * math.ceil(len(examples) / TRAINING.batch_size)
    warmup = max(1, int(steps * TRAINING.warmup_fraction))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(
            (step + 1) / warmup, max(0, (steps - step) / max(steps - warmup, 1))
        ),
    )
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            metadata["negatives"] / metadata["positives"], device=device
        )
    )
    trials, best_key, best_state, selected_epoch = [], None, None, None
    for epoch in range(1, TRAINING.epochs + 1):
        model.train()
        losses = []
        order = rng.permutation(len(examples))
        for start in range(0, len(order), TRAINING.batch_size):
            batch = [examples[i] for i in order[start : start + TRAINING.batch_size]]
            features = tokenizer(
                [e["question"] for e in batch],
                [e["passage"] for e in batch],
                padding=True,
                truncation="only_second",
                max_length=SEARCH.max_tokens,
                return_tensors="pt",
            ).to(device)
            labels = torch.tensor(
                [e["label"] for e in batch], device=device, dtype=torch.float32
            )
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(**features).logits.flatten(), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                TRAINING.gradient_clip,
            )
            optimizer.step()
            scheduler.step()
            losses.append((float(loss.detach().cpu()), len(batch)))
        # A new identifier prevents reuse of scores from the previous checkpoint.
        scorer.metadata = {**metadata, "adapter_sha256": f"training-epoch-{epoch}"}
        result = metrics(retriever, dev, True)
        trial = {
            "epoch": epoch,
            "train_loss": sum(loss * n for loss, n in losses) / len(examples),
            "dev_mrr": result["mrr"],
            "dev_hit_at_5": result["hit_at_5"],
        }
        trials.append(trial)
        key = (result["mrr"], result["hit_at_5"], -epoch)
        if best_key is None or key > best_key:
            best_key, selected_epoch = key, epoch
            best_state = {
                name: p.detach().cpu().clone()
                for name, p in model.named_parameters()
                if p.requires_grad
            }
        print(json.dumps(trial), flush=True)
    path = paths.model / ADAPTER_NAME
    save_file(best_state, str(path))
    changed = {
        name: float(torch.linalg.vector_norm(value - initial[name]))
        for name, value in best_state.items()
    }
    if not any(changed.values()):
        raise RuntimeError("Training did not change any pretrained weights")
    metadata.update(
        {
            "adapter_file": ADAPTER_NAME,
            "adapter_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "adapter_bytes": path.stat().st_size,
            "selected_epoch": selected_epoch,
            "changed_tensors": len([v for v in changed.values() if v > 0]),
            "weight_change_norms": changed,
            "development_eligible": best_key[0] > baseline["mrr"]
            and best_key[1] >= baseline["hit_at_5"],
            "selection": {
                "criterion": "development MRR, hit@5, earlier epoch breaks ties; test sets are never scored during training",
                "dev_mrr": best_key[0],
                "baseline_dev_mrr": baseline["mrr"],
                "pretrained_dev_mrr": pretrained["mrr"],
                "pretrained_dev_hit_at_5": pretrained["hit_at_5"],
                "trials": trials,
            },
        }
    )
    (paths.model / "reranker.json").write_text(json.dumps(metadata, indent=2))
    retriever.save_query_cache()
    # Only the selected checkpoint's scores are shipped, not training-epoch scores.
    return {
        k: v
        for k, v in metadata.items()
        if k not in ("weight_change_norms", "question_ids")
    }
