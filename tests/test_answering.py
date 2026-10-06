import json

import httpx
import pytest

from coso_rag.answering import answer, gather_evidence, validate_answer
from coso_rag.config import REPORTS, artifact_paths
from coso_rag.datasets import questions
from coso_rag.openrouter import APIError, OpenRouter
from coso_rag.retrieval import Retriever

PATHS = artifact_paths()


def test_compact_baseline_report_retains_verifiable_source_quotes():
    retriever = Retriever()
    report = json.loads((REPORTS / "answers-baseline.json").read_text())
    for row in report["results"]:
        if row["validation"] != "pass":
            continue
        result = row["result"]
        sources = [
            {**retriever.pages[(source["doc_id"], source["page"])], "id": source["id"]}
            for source in result["evidence"]
        ]
        assert validate_answer(result, sources) == result


@pytest.fixture
def mock_answer_client(tmp_path):
    def complete(request):
        payload = json.loads(request.content)
        excerpts = json.loads(payload["messages"][1]["content"])["excerpts"]
        source = excerpts[0]
        quote = " ".join(source["text"].split()[:10])
        value = {
            "status": "answered",
            "claims": [
                {
                    "text": quote,
                    "citation_id": source["citation_id"],
                    "evidence_quote": quote,
                }
            ],
            "missing_information": "",
        }
        return httpx.Response(
            200,
            json={
                "id": "fixture-generation",
                "choices": [
                    {"finish_reason": "stop", "message": {"content": json.dumps(value)}}
                ],
                "usage": {"cost": 0.0001},
            },
        )

    return OpenRouter(
        key="test",
        directory=tmp_path / "ledger",
        transport=httpx.MockTransport(complete),
    )


def test_complete_answer_pipeline_validates_and_caches_mocked_completion(
    mock_answer_client,
):
    retriever = Retriever()
    question = next(
        row["question"]
        for row in questions()
        if row["split"] == "test" and row["answerable"]
    )
    first = answer(question, retriever, client=mock_answer_client)
    cached = answer(question, retriever, client=mock_answer_client)
    assert first["status"] == cached["status"] == "answered"
    assert first["claims"] == cached["claims"]
    assert first["cached"] is False and cached["cached"] is True
    assert mock_answer_client.usage()["calls"] == 1
    assert first["claims"][0]["citation_id"] in {
        source["id"] for source in first["evidence"]
    }


def test_new_answer_evaluation_preserves_baseline_review(
    tmp_path, monkeypatch, mock_answer_client
):
    from coso_rag import live_evaluation

    reports = tmp_path / "reports"
    reports.mkdir()
    baseline = reports / "answers-baseline.json"
    baseline.write_text("previous manual review")
    monkeypatch.setattr(live_evaluation, "REPORTS", reports)
    live_evaluation.evaluate_live(limit=1, client=mock_answer_client)
    latest = json.loads((reports / "answers-latest.json").read_text())
    assert baseline.read_text() == "previous manual review"
    assert latest["citation_validation_passes"] == 1
    assert latest["results"][0]["result"]["evidence"]
    assert all(
        set(source) == {"id", "doc_id", "page"}
        for source in latest["results"][0]["result"]["evidence"]
    )
    assert "history" not in latest["usage"]


def test_fabricated_sources_quotes_and_invalid_abstentions_rejected():
    source = {
        "id": "contract-p16-c1",
        "text": "The compensation shall not exceed 10 percent of tendered value.",
    }
    value = {
        "status": "answered",
        "claims": [
            {
                "text": "Cap is 10 percent.",
                "citation_id": source["id"],
                "evidence_quote": source["text"],
            }
        ],
        "missing_information": "",
    }
    assert validate_answer(value, [source]) == value
    for field, replacement in [
        ("citation_id", "invented-p1-c1"),
        ("evidence_quote", "The compensation is unlimited."),
    ]:
        forged = json.loads(json.dumps(value))
        forged["claims"][0][field] = replacement
        with pytest.raises(APIError):
            validate_answer(forged, [source])
    with pytest.raises(APIError):
        validate_answer(
            {
                "status": "insufficient_evidence",
                "claims": value["claims"],
                "missing_information": "Unknown",
            },
            [source],
        )


def test_parent_page_context_keeps_tender_table_amount_and_deadline():
    r = Retriever()
    for q, required in [
        ("यात्री निवास निर्माण की अनुमानित लागत कितनी है?", "286.47"),
        (
            "What is the online bid submission deadline in the Rajasthan tender?",
            "15.04.2025",
        ),
    ]:
        evidence = gather_evidence(q, r, trained=False)
        assert len(evidence) <= 6
        assert len({c["id"] for c in evidence}) == len(evidence)
        assert any(required in c["text"] for c in evidence)
        for c in evidence:
            assert r.pages[(c["doc_id"], c["page"])]["text"] == c["text"]
