import hashlib
import json

import pytest

from coso_rag.config import artifact_paths, document_path
from coso_rag.datasets import questions, relevant
from coso_rag.ingest import chunk_pages, page_image
from coso_rag.retrieval import Retriever

PATHS = artifact_paths()


def test_chunks_preserve_exact_page_offsets_and_do_not_cross_pages():
    pages = [
        {
            "doc_id": "tender",
            "title": "Tender",
            "page": n,
            "text": " ".join(str(i) for i in range(420)),
            "method": "vision",
            "uncertainties": [],
        }
        for n in [1, 2]
    ]
    chunks = chunk_pages(pages)
    assert len({c["id"] for c in chunks}) == len(chunks)
    for c in chunks:
        p = pages[c["page"] - 1]
        assert c["text"] == p["text"][c["start"] : c["end"]]
        assert len(c["text"].split()) <= 180


def test_shipped_documents_index_and_evidence_are_consistent():
    manifest = json.loads((PATHS.index / "documents.json").read_text())
    assert sum(d["pages"] for d in manifest) == 142
    for doc in manifest:
        assert (
            hashlib.sha256(document_path(doc["id"]).read_bytes()).hexdigest()
            == doc["sha256"]
        )
    r = Retriever()
    for q in questions():
        if q["answerable"]:
            assert any(relevant(q, c) for c in r.chunks), q["id"]
    with pytest.raises(ValueError):
        document_path("../../secret")
    with pytest.raises(ValueError):
        page_image("tender", 100)


def test_hindi_and_numeric_retrieval_without_paid_api():
    r = Retriever()
    for q in [
        q
        for q in questions()
        if q["split"] == "test" and q["id"] in ["monsoon-1", "tribunal-1", "digital-1"]
    ]:
        results = r.search(q["question"], trained=bool(r.model["activate_by_default"]))
        assert any(relevant(q, c) for c in results), q["id"]
    results = r.search(
        "क्या कार्य पूरा करने की अवधि में मानसून शामिल है?", document="tender", trained=False
    )
    assert all(c["doc_id"] == "tender" for c in results)


def test_formula_correction_retains_original_provenance():
    pages = [
        json.loads(line)
        for line in (PATHS.index / "pages.jsonl").read_text().splitlines()
    ]
    page = next(p for p in pages if p["doc_id"] == "standard" and p["page"] == 8)
    original = json.loads(
        next((PATHS.cache / "ocr").glob("standard-8-*.json")).read_text()
    )
    assert "√(f_ck/f_y)" in original["text"]
    assert "√f_ck / f_y" in page["text"]
    assert page["corrections"]


def test_contract_clause_heading_survives_page_break():
    r = Retriever()
    continuation = next(c for c in r.chunks if c["id"] == "contract-p24-c1")
    assert continuation["sections"] == ["CPWD Clause 7"]
    assert continuation["section_source"] == {"pdf_page": 23, "heading": "Clause 7"}
