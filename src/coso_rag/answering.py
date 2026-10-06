import json

from .config import API, ROOT, SEARCH
from .openrouter import APIError, OpenRouter
from .retrieval import normalize

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["answered", "insufficient_evidence"]},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "citation_id": {"type": "string"},
                    "evidence_quote": {"type": "string"},
                },
                "required": ["text", "citation_id", "evidence_quote"],
                "additionalProperties": False,
            },
        },
        "missing_information": {"type": "string"},
    },
    "required": ["status", "claims", "missing_information"],
    "additionalProperties": False,
}

SYSTEM = (ROOT / "prompts/answer.txt").read_text().strip()


def validate_answer(value, evidence):
    if not isinstance(value, dict) or value.get("status") not in {
        "answered",
        "insufficient_evidence",
    }:
        raise APIError("Answer had an invalid structure")
    claims = value.get("claims")
    if not isinstance(claims, list) or len(claims) > 12:
        raise APIError("Answer had invalid claims")
    lookup = {c["id"]: c for c in evidence}
    if value["status"] == "insufficient_evidence":
        if (
            claims
            or not isinstance(value.get("missing_information"), str)
            or not value["missing_information"].strip()
        ):
            raise APIError("Invalid abstention")
    elif not claims:
        raise APIError("Answer contained no supported claims")
    for claim in claims:
        if not isinstance(claim, dict) or not all(
            isinstance(claim.get(k), str)
            for k in ["text", "citation_id", "evidence_quote"]
        ):
            raise APIError("Malformed citation")
        source = lookup.get(claim["citation_id"])
        if source is None:
            raise APIError("Answer referenced a source outside the retrieved evidence")
        quote = normalize(claim["evidence_quote"])
        if (
            len(quote) < 12
            or quote not in normalize(source["text"])
            or "[illegible]" in quote
        ):
            raise APIError(
                "Answer quote could not be verified against the cited excerpt"
            )
        if not claim["text"].strip():
            raise APIError("Empty claim")
    return value


def gather_evidence(question, retriever, *, trained=None, document=None):
    seeds = retriever.search(
        question, trained=trained, document=document, k=SEARCH.candidate_pool * 2
    )
    evidence = []
    seen = set()
    # Read distinct parent pages to preserve tables and clause qualifiers.
    for seed in seeds:
        key = (seed["doc_id"], seed["page"])
        if key in seen:
            continue
        page = retriever.pages[key]
        siblings = [c for c in retriever.chunks if (c["doc_id"], c["page"]) == key]
        sections = list(
            dict.fromkeys(s for c in siblings for s in c.get("sections", []))
        )
        evidence.append(
            {
                **page,
                "id": f"{key[0]}-p{key[1]}",
                "seed_chunk_id": seed["id"],
                "sections": sections,
                "section_source": siblings[0].get("section_source"),
                "score": seed["score"],
                "start": 0,
                "end": len(page["text"]),
            }
        )
        seen.add(key)
        if len(evidence) == SEARCH.context_pages:
            break
    return evidence


def answer(question, retriever, *, client=None, trained=None, document=None):
    if trained is None:
        trained = bool(retriever.model and retriever.model.get("activate_by_default"))
    evidence = gather_evidence(question, retriever, trained=trained, document=document)
    context = [
        {
            "citation_id": c["id"],
            "document": c["title"],
            "pdf_page": c["page"],
            "extraction": c["method"],
            "uncertainties": c["uncertainties"],
            "sections": c.get("sections", []),
            "section_source": c.get("section_source"),
            "text": c["text"],
        }
        for c in evidence
    ]
    client = client or OpenRouter()
    result = client.complete(
        [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": json.dumps(
                    {"question": question, "excerpts": context}, ensure_ascii=False
                ),
            },
        ],
        purpose="answer",
        max_tokens=API.answer_tokens,
        schema=ANSWER_SCHEMA,
    )
    try:
        value = json.loads(result["choices"][0]["message"]["content"])
    except (ValueError, KeyError, TypeError):
        raise APIError("Provider returned invalid JSON; no automatic retry") from None
    value = validate_answer(value, evidence)
    return {
        **value,
        "evidence": evidence,
        "ranking": "trained" if trained and retriever.model else "baseline",
        "search_scope": document or "all",
        "cost_usd": result.get("usage", {}).get("cost"),
        "cached": result.get("_cached", False),
        "generation": result.get("id"),
    }


def format_answer(value):
    if value["status"] == "insufficient_evidence":
        return "I couldn’t find sufficient evidence. " + value["missing_information"]
    lookup = {c["id"]: c for c in value["evidence"]}
    return "\n\n".join(
        f"{claim['text']} [{lookup[claim['citation_id']]['title']}, PDF page {lookup[claim['citation_id']]['page']}; {claim['citation_id']}]"
        for claim in value["claims"]
    )
