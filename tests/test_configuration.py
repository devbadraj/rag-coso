import json
import shutil
from dataclasses import replace

import numpy as np
import pytest

from coso_rag import retrieval
from coso_rag.config import API, SEARCH, artifact_paths
from coso_rag.ingest import chunk_pages, extract


@pytest.mark.parametrize("words,overlap", [(0, 0), (20, 20), (20, -1)])
def test_invalid_chunk_settings_are_rejected(words, overlap):
    with pytest.raises(ValueError, match="overlap"):
        chunk_pages([], words=words, overlap=overlap)
    with pytest.raises(ValueError, match="overlap"):
        replace(SEARCH, chunk_words=words, chunk_overlap=overlap)


def test_nonfinite_price_caps_are_rejected():
    with pytest.raises(ValueError, match="API"):
        replace(API, input_price_cap=float("nan"))


def test_chunking_and_section_detection_use_document_configuration(
    tmp_path, monkeypatch
):
    paths = artifact_paths(tmp_path)
    paths.index.mkdir(parents=True)
    document = {
        "language": "Hindi",
        "section_pattern": r"(?m)^Section ([A-Z]+)$",
        "section_prefix": "Procedure ",
    }
    monkeypatch.setattr(retrieval, "document_config", lambda doc_id: document)
    monkeypatch.setattr(
        retrieval,
        "encode",
        lambda texts, prefix: np.ones((len(texts), 4), dtype="float32"),
    )
    pages = [
        {
            "doc_id": "custom",
            "title": "Custom source",
            "page": page,
            "text": ("Section IV\n" if page == 1 else "")
            + " ".join(f"word{i}" for i in range(220)),
            "method": "native",
            "uncertainties": [],
        }
        for page in (1, 2)
    ]
    (paths.index / "pages.jsonl").write_text(
        "".join(json.dumps(page) + "\n" for page in pages)
    )
    retrieval.build_index(pages, tmp_path)
    chunks = [
        json.loads(line)
        for line in (paths.index / "chunks.jsonl").read_text().splitlines()
    ]
    assert all(
        len(chunk["text"].split()) <= SEARCH.hindi_chunk_words for chunk in chunks
    )
    continuation = next(chunk for chunk in chunks if chunk["page"] == 2)
    assert continuation["sections"] == ["Procedure IV"]
    assert continuation["section_source"] == {"pdf_page": 1, "heading": "Section IV"}


def test_reorganized_ingestion_preserves_every_shipped_page(tmp_path):
    shipped = artifact_paths()
    paths = artifact_paths(tmp_path)
    shutil.copytree(shipped.cache / "ocr", paths.cache / "ocr")
    pages = extract(directory=tmp_path)
    expected = [
        json.loads(line)
        for line in (shipped.index / "pages.jsonl").read_text().splitlines()
    ]
    assert pages == expected


def test_ingestion_rejects_an_ocr_cache_from_another_source(tmp_path):
    shipped = artifact_paths()
    paths = artifact_paths(tmp_path)
    shutil.copytree(shipped.cache / "ocr", paths.cache / "ocr")
    cache = next((paths.cache / "ocr").glob("standard-8-*.json"))
    record = json.loads(cache.read_text())
    record["pdf_sha256"] = "wrong-source"
    cache.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="source page"):
        extract(directory=tmp_path)
