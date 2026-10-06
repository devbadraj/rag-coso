import base64
import hashlib
import json
import re

import pymupdf as fitz

from .config import API, DOCUMENTS, MODEL, ROOT, SEARCH, artifact_paths, document_path
from .openrouter import OpenRouter

OCR_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["text", "uncertainties"],
    "additionalProperties": False,
}


def clean_text(text, cleanup_pattern=None):
    if cleanup_pattern:
        text = re.sub(cleanup_pattern, "", text)
    text = re.sub(r"([a-zA-Z])-\s*\n\s*([a-z])", r"\1\2", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def page_image(doc_id, page_number):
    with fitz.open(document_path(doc_id)) as pdf:
        if not 1 <= page_number <= len(pdf):
            raise ValueError("Page outside document")
        page = pdf[page_number - 1]
        scale = min(2.0, API.image_max_pixels / max(page.rect.width, page.rect.height))
        return page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False).tobytes(
            "png"
        )


def extract(*, vision=False, client=None, directory=None):
    paths = artifact_paths(directory)
    paths.index.mkdir(parents=True, exist_ok=True)
    ocr_dir = paths.cache / "ocr"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    corrections_path = ROOT / "data" / "extraction-corrections.json"
    corrections = (
        json.loads(corrections_path.read_text()) if corrections_path.exists() else []
    )
    pages, manifest = [], []
    for doc in DOCUMENTS:
        path = document_path(doc["id"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with fitz.open(path) as pdf:
            manifest.append(
                {
                    **doc,
                    "sha256": digest,
                    "pages": len(pdf),
                    "bytes": path.stat().st_size,
                }
            )
            for idx, page in enumerate(pdf):
                number = idx + 1
                native = clean_text(
                    page.get_text(sort=False), doc.get("cleanup_pattern")
                )
                needs_vision = (
                    doc.get("vision_all_pages", False)
                    or number in doc.get("vision_pages", [])
                    or (
                        doc.get("auto_ocr", False)
                        and len(native) < API.ocr_min_characters
                    )
                )
                text, method, uncertainties = native, "native", []
                cached = ocr_dir / f"{doc['id']}-{number}-{digest[:12]}.json"
                if needs_vision:
                    if cached.exists():
                        record = json.loads(cached.read_text())
                    elif vision:
                        client = client or OpenRouter()
                        image = base64.b64encode(page_image(doc["id"], number)).decode()
                        prompt = (ROOT / "prompts/ocr.txt").read_text().strip()
                        result = client.complete(
                            [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": prompt},
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": "data:image/png;base64," + image
                                            },
                                        },
                                    ],
                                }
                            ],
                            purpose=f"ocr:{doc['id']}:{number}",
                            max_tokens=API.ocr_tokens,
                            schema=OCR_SCHEMA,
                            image_count=1,
                        )
                        value = json.loads(result["choices"][0]["message"]["content"])
                        record = {
                            **value,
                            "model": MODEL,
                            "pdf_sha256": digest,
                            "page": number,
                            "generation": result.get("id"),
                            "cost_usd": result.get("usage", {}).get("cost"),
                        }
                        cached.write_text(
                            json.dumps(record, ensure_ascii=False, indent=2)
                        )
                    else:
                        raise RuntimeError(
                            f"Missing OCR for {doc['id']} page {number}. Run ingest --vision with a key or use the shipped artifacts."
                        )
                    if (
                        record.get("pdf_sha256") != digest
                        or record.get("page") != number
                    ):
                        raise RuntimeError("OCR cache does not match the source page")
                    if not isinstance(record.get("text"), str) or not isinstance(
                        record.get("uncertainties"), list
                    ):
                        raise RuntimeError(
                            "OCR cache has invalid text or uncertainty data"
                        )
                    text, method, uncertainties = (
                        record["text"],
                        "vision",
                        record["uncertainties"],
                    )
                applied = []
                for correction in corrections:
                    if (
                        correction["doc_id"] == doc["id"]
                        and correction["page"] == number
                    ):
                        if correction["before"] not in text:
                            raise RuntimeError(
                                "Correction does not match extraction; review before rebuilding"
                            )
                        text = text.replace(correction["before"], correction["after"])
                        applied.append(correction["reason"])
                pages.append(
                    {
                        "doc_id": doc["id"],
                        "title": doc["title"],
                        "page": number,
                        "text": text,
                        "method": method,
                        "uncertainties": uncertainties,
                        "pdf_sha256": digest,
                        "native_characters": len(native),
                        "corrections": applied,
                    }
                )
    (paths.index / "pages.jsonl").write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pages)
    )
    (paths.index / "documents.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    return pages


def chunk_pages(pages, words=SEARCH.chunk_words, overlap=SEARCH.chunk_overlap):
    if not 0 <= overlap < words:
        raise ValueError("Chunk overlap must be smaller than chunk size")
    chunks = []
    for page in pages:
        tokens = list(re.finditer(r"\S+", page["text"]))
        if len(tokens) < 12:
            continue
        for start in range(0, len(tokens), words - overlap):
            end = min(start + words, len(tokens))
            a, b = tokens[start].start(), tokens[end - 1].end()
            text = page["text"][a:b]
            chunks.append(
                {
                    "id": f"{page['doc_id']}-p{page['page']}-c{start // (words - overlap) + 1}",
                    "doc_id": page["doc_id"],
                    "title": page["title"],
                    "page": page["page"],
                    "text": text,
                    "start": a,
                    "end": b,
                    "method": page["method"],
                    "uncertainties": page["uncertainties"],
                }
            )
            if end == len(tokens):
                break
    return chunks
