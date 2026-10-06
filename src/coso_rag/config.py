"""Shared settings and paths. Secrets stay in the environment, not config.toml."""

import math
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env", override=False)
CONFIG = tomllib.loads((ROOT / "config.toml").read_text())
ARTIFACTS = Path(os.getenv("COSO_ARTIFACT_DIR", str(ROOT / "artifacts"))).resolve()
DATASETS = ROOT / "evaluation" / "datasets"
REPORTS = ROOT / "evaluation" / "reports"
DOCUMENTS = CONFIG["documents"]
EXAMPLES = CONFIG.get("examples", [])
MODEL = CONFIG["models"]["answer"]
EMBEDDING = CONFIG["models"]["embedding"]
EMBEDDING_REVISION = CONFIG["models"]["embedding_revision"]
RERANKER = CONFIG["models"]["reranker"]
RERANKER_REVISION = CONFIG["models"]["reranker_revision"]


def positive(*values):
    return all(math.isfinite(value) and value > 0 for value in values)


@dataclass(frozen=True)
class SearchSettings:
    candidate_pool: int
    results: int
    context_pages: int
    max_question_chars: int
    max_tokens: int
    embedding_batch_size: int
    bm25_k1: float
    bm25_b: float
    fusion_constant: int
    chunk_words: int
    chunk_overlap: int
    hindi_chunk_words: int
    hindi_chunk_overlap: int

    def __post_init__(self):
        if not 0 <= self.bm25_b <= 1 or not positive(self.bm25_k1):
            raise ValueError("Invalid BM25 settings")
        for words, overlap in (
            (self.chunk_words, self.chunk_overlap),
            (self.hindi_chunk_words, self.hindi_chunk_overlap),
        ):
            if not 0 <= overlap < words:
                raise ValueError("Chunk overlap must be smaller than chunk size")
        if not positive(
            self.candidate_pool,
            self.results,
            self.context_pages,
            self.max_question_chars,
            self.max_tokens,
            self.embedding_batch_size,
            self.fusion_constant,
        ):
            raise ValueError("Search sizes and limits must be positive")


@dataclass(frozen=True)
class TrainingSettings:
    learning_rate: float
    weight_decay: float
    batch_size: int
    inference_batch_size: int
    epochs: int
    warmup_fraction: float
    gradient_clip: float
    last_layers: int
    seed: int
    cpu_threads: int

    def __post_init__(self):
        if not positive(
            self.batch_size,
            self.inference_batch_size,
            self.epochs,
            self.last_layers,
            self.cpu_threads,
        ):
            raise ValueError("Training sizes must be positive")
        if (
            not positive(self.learning_rate, self.gradient_clip)
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
            or not 0 <= self.warmup_fraction < 1
        ):
            raise ValueError("Invalid training optimizer settings")


@dataclass(frozen=True)
class APISettings:
    endpoint: str
    budget_usd: float
    input_price_cap: float
    output_price_cap: float
    answer_tokens: int
    ocr_tokens: int
    timeout_seconds: int
    evaluation_workers: int
    image_max_pixels: int
    image_token_reserve: int
    request_token_reserve: int
    ocr_min_characters: int

    def __post_init__(self):
        if not self.endpoint.startswith("https://"):
            raise ValueError("API endpoint must use HTTPS")
        if not positive(
            self.budget_usd,
            self.input_price_cap,
            self.output_price_cap,
            self.answer_tokens,
            self.ocr_tokens,
            self.timeout_seconds,
            self.evaluation_workers,
            self.image_max_pixels,
            self.image_token_reserve,
            self.request_token_reserve,
            self.ocr_min_characters,
        ):
            raise ValueError("API prices, sizes and limits must be positive")


SEARCH = SearchSettings(**CONFIG["search"])
TRAINING = TrainingSettings(**CONFIG["training"])
API = APISettings(**CONFIG["api"])


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path

    @property
    def index(self):
        return self.root / "index"

    @property
    def model(self):
        return self.root / "model"

    @property
    def cache(self):
        return self.root / "cache"


def artifact_paths(directory=None):
    return ArtifactPaths(Path(directory or ARTIFACTS))


def document_config(doc_id):
    for document in DOCUMENTS:
        if document["id"] == doc_id:
            return document
    raise ValueError(f"Unknown document: {doc_id}")


def document_path(doc_id):
    document = document_config(doc_id)
    path = (ROOT / "data" / "documents" / document["file"]).resolve()
    if path.parent != (ROOT / "data" / "documents").resolve():
        raise ValueError("Document file must be inside data/documents")
    return path
