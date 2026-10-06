"""Semantic and keyword retrieval, followed by optional text-pair reranking."""

import hashlib
import json
import math
import re
import threading
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np

from .config import (
    EMBEDDING,
    EMBEDDING_REVISION,
    SEARCH,
    artifact_paths,
    document_config,
)
from .ingest import chunk_pages

STOPWORDS = set(
    "a an the is are was were to of in on at for and or be by with what which how does do can shall must this that as from it than not have under per its only any will more less according supplied document please tell me about".split()
)


def tokens(text):
    return [
        t
        for t in re.findall(
            r"[\w\u0900-\u097f]+(?:[.:-][\w\u0900-\u097f]+)*", text.lower()
        )
        if t not in STOPWORDS and len(t) > 1
    ]


def normalize(text):
    return " ".join(text.split()).casefold()


@lru_cache(maxsize=1)
def embedding_model():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING, revision=EMBEDDING_REVISION, device="cpu")
    model.max_seq_length = SEARCH.max_tokens
    return model


def encode(texts, prefix):
    return (
        embedding_model()
        .encode(
            [prefix + t for t in texts],
            normalize_embeddings=True,
            batch_size=SEARCH.embedding_batch_size,
            show_progress_bar=False,
        )
        .astype("float32")
    )


def build_index(pages, directory=None):
    paths = artifact_paths(directory)
    paths.index.mkdir(parents=True, exist_ok=True)
    chunks = []
    sections = {}
    for page in pages:
        document = document_config(page["doc_id"])
        hindi = document["language"].casefold() == "hindi"
        page_chunks = chunk_pages(
            [page],
            words=SEARCH.hindi_chunk_words if hindi else SEARCH.chunk_words,
            overlap=SEARCH.hindi_chunk_overlap if hindi else SEARCH.chunk_overlap,
        )
        pattern = document.get("section_pattern")
        if pattern:
            headings = list(re.finditer(pattern, page["text"]))
            label, provenance = sections.get(page["doc_id"], (None, None))
            prefix = document.get("section_prefix", "")
            for chunk in page_chunks:
                before = [
                    heading for heading in headings if heading.start() <= chunk["start"]
                ]
                if before:
                    heading = before[-1]
                    label = prefix + heading.group(1).strip()
                    provenance = {
                        "pdf_page": page["page"],
                        "heading": heading.group(0).strip(),
                    }
                local = [
                    heading
                    for heading in headings
                    if chunk["start"] < heading.start() < chunk["end"]
                ]
                chunk["sections"] = ([label] if label else []) + [
                    prefix + heading.group(1).strip() for heading in local
                ]
                chunk["section_source"] = provenance
            if headings:
                heading = headings[-1]
                sections[page["doc_id"]] = (
                    prefix + heading.group(1).strip(),
                    {"pdf_page": page["page"], "heading": heading.group(0).strip()},
                )
        chunks.extend(page_chunks)
    if not chunks:
        raise ValueError("No searchable text was extracted")
    vectors = encode(
        [chunk["title"] + "\n" + chunk["text"] for chunk in chunks], "passage: "
    )
    content = "".join(json.dumps(chunk, ensure_ascii=False) + "\n" for chunk in chunks)
    (paths.index / "chunks.jsonl").write_text(content)
    np.save(paths.index / "vectors.npy", vectors, allow_pickle=False)
    metadata = {
        "embedding": EMBEDDING,
        "revision": EMBEDDING_REVISION,
        "max_tokens": SEARCH.max_tokens,
        "chunks": len(chunks),
        "dimensions": int(vectors.shape[1]),
        "fingerprint": hashlib.sha256(content.encode()).hexdigest(),
        "pages_sha256": hashlib.sha256(
            (paths.index / "pages.jsonl").read_bytes()
        ).hexdigest(),
        "chunking": {
            "words": SEARCH.chunk_words,
            "overlap": SEARCH.chunk_overlap,
            "hindi_words": SEARCH.hindi_chunk_words,
            "hindi_overlap": SEARCH.hindi_chunk_overlap,
            "cross_pages": False,
        },
    }
    (paths.index / "index.json").write_text(json.dumps(metadata, indent=2))
    return metadata


class Retriever:
    def __init__(self, directory=None):
        self.paths = artifact_paths(directory)
        self.directory = self.paths.root
        self.lock = threading.RLock()
        self.chunks = [
            json.loads(line)
            for line in (self.paths.index / "chunks.jsonl").read_text().splitlines()
        ]
        self.vectors = np.load(self.paths.index / "vectors.npy", allow_pickle=False)
        self.metadata = json.loads((self.paths.index / "index.json").read_text())
        if (
            hashlib.sha256((self.paths.index / "chunks.jsonl").read_bytes()).hexdigest()
            != self.metadata["fingerprint"]
        ):
            raise ValueError("Index is stale: rebuild after changing chunks")
        page_bytes = (self.paths.index / "pages.jsonl").read_bytes()
        if (
            self.metadata.get("pages_sha256")
            and hashlib.sha256(page_bytes).hexdigest() != self.metadata["pages_sha256"]
        ):
            raise ValueError(
                "Page text is stale: rebuild after changing extracted pages"
            )
        self.pages = {
            (p["doc_id"], p["page"]): p
            for p in [json.loads(line) for line in page_bytes.decode().splitlines()]
        }
        if (
            self.metadata["embedding"] != EMBEDDING
            or self.metadata["revision"] != EMBEDDING_REVISION
            or self.metadata["max_tokens"] != SEARCH.max_tokens
        ):
            raise ValueError("Embedding settings changed: rebuild the index")
        if self.vectors.shape != (len(self.chunks), self.metadata["dimensions"]):
            raise ValueError("Vector dimensions do not match the index")
        self.counts = [
            Counter(tokens(c["title"] + " " + c["text"])) for c in self.chunks
        ]
        self.lengths = np.array([sum(c.values()) for c in self.counts])
        self.average_length = max(float(self.lengths.mean()), 1)
        self.postings = defaultdict(list)
        for i, count in enumerate(self.counts):
            for term, frequency in count.items():
                self.postings[term].append((i, frequency))
        self.model = None
        self.text_reranker = None
        model_path = self.paths.model / "reranker.json"
        if model_path.exists():
            model = json.loads(model_path.read_text())
            if (
                model["index_fingerprint"] == self.metadata["fingerprint"]
                and model.get("kind") == "cross_encoder"
            ):
                self.model = model
        self.cache_path = self.paths.cache / "query-vectors.json"
        self.query_cache = (
            json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        )

    def lexical(self, question):
        scores = np.zeros(len(self.chunks))
        n = len(self.chunks)
        for term in set(tokens(question)):
            postings = self.postings.get(term, [])
            idf = math.log(1 + (n - len(postings) + 0.5) / (len(postings) + 0.5))
            for idx, freq in postings:
                scores[idx] += (
                    idf
                    * freq
                    * (SEARCH.bm25_k1 + 1)
                    / (
                        freq
                        + SEARCH.bm25_k1
                        * (
                            1
                            - SEARCH.bm25_b
                            + SEARCH.bm25_b * self.lengths[idx] / self.average_length
                        )
                    )
                )
        return scores

    def query_vector(self, question):
        key = hashlib.sha256(
            json.dumps(
                [EMBEDDING_REVISION, SEARCH.max_tokens, question], ensure_ascii=False
            ).encode()
        ).hexdigest()
        with self.lock:
            if key not in self.query_cache:
                self.query_cache[key] = encode([question], "query: ")[0].tolist()
            return np.array(self.query_cache[key], dtype="float32")

    def save_query_cache(self):
        with self.lock:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.query_cache))

    def candidates(self, question, document=None, pool=SEARCH.candidate_pool):
        if not question.strip() or len(question) > SEARCH.max_question_chars:
            raise ValueError(
                f"Enter a question between 1 and {SEARCH.max_question_chars} characters"
            )
        if document is not None:
            document_config(document)
        dense = self.vectors @ self.query_vector(question)
        lexical = self.lexical(question)
        allowed = [
            i
            for i, c in enumerate(self.chunks)
            if document is None or c["doc_id"] == document
        ]
        dr = sorted(allowed, key=lambda i: (-float(dense[i]), i))[:pool]
        lr = sorted(allowed, key=lambda i: (-float(lexical[i]), i))[:pool]
        dense_ranks, lexical_ranks = (
            {i: r + 1 for r, i in enumerate(dr)},
            {i: r + 1 for r, i in enumerate(lr)},
        )
        result = []
        for index in sorted(set(dr + lr)):
            ranks = (dense_ranks.get(index), lexical_ranks.get(index))
            score = sum(
                1 / (SEARCH.fusion_constant + rank)
                for rank in ranks
                if rank is not None
            )
            result.append(
                {**self.chunks[index], "baseline_score": score, "score": score}
            )
        return sorted(result, key=lambda c: (-c["score"], c["id"]))

    def search(self, question, *, trained=None, document=None, k=SEARCH.results):
        if trained is None:
            trained = bool(self.model and self.model.get("activate_by_default"))
        if trained and self.model is None:
            raise ValueError(
                "No compatible trained model; run train and evaluate first"
            )
        candidates = self.candidates(question, document)
        if trained and self.model and self.model.get("kind") == "cross_encoder":
            from .reranking import TextReranker

            with self.lock:
                if self.text_reranker is None:
                    self.text_reranker = TextReranker(self.model, self.directory)
            scores = self.text_reranker.score(question, candidates)
            for c, score in zip(candidates, scores):
                c["score"] = score
        return sorted(candidates, key=lambda c: (-c["score"], c["id"]))[:k]
