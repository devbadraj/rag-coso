"""Streamlit views. Search, model calls and evaluation stay in their own modules."""

import hashlib
import json

import streamlit as st

from .answering import answer, gather_evidence
from .config import (
    DOCUMENTS,
    EXAMPLES,
    REPORTS,
    ROOT,
    SEARCH,
    artifact_paths,
    document_config,
    document_path,
)
from .ingest import page_image
from .openrouter import APIError, OpenRouter
from .retrieval import Retriever

METHOD_NAMES = {
    "baseline": "Original search",
    "pretrained": "Text model before training",
    "trained": "Text model after training",
}


@st.cache_resource
def load_retriever(index_fingerprint, model_revision):
    """Both arguments identify the files that invalidate this shared resource."""
    return Retriever()


@st.dialog("Read the original page", width="large")
def show_page(doc_id, page):
    document = document_config(doc_id)
    st.caption(f"{document['display_name']} · Page {page}")
    st.image(page_image(doc_id, page), width="stretch")
    st.download_button(
        "Download this document",
        document_path(doc_id).read_bytes(),
        file_name=document["file"],
        mime="application/pdf",
    )


def render_sidebar(retriever, client):
    with st.sidebar:
        st.markdown("### Site Notes")
        st.caption("Help reading construction documents")
        st.divider()
        document = st.selectbox(
            "Which documents?",
            [None] + [item["id"] for item in DOCUMENTS],
            format_func=lambda value: (
                document_config(value)["display_name"] if value else "All documents"
            ),
        )
        with st.expander("Extra options"):
            trained = st.toggle(
                "Try search with training",
                value=bool(
                    retriever.model and retriever.model.get("activate_by_default")
                ),
                disabled=retriever.model is None,
                help="The reranker reads your question and each possible paragraph together. Turn it off to compare with original search.",
            )
            generate = st.toggle(
                "Write an answer",
                value=bool(client.key),
                disabled=not client.key,
                help="Turn this off to see document text without paying for a written answer.",
            )
            if not client.key:
                st.caption(
                    "To get written answers, add your OpenRouter key to .env and restart the app."
                )
        usage = client.usage()
        st.divider()
        st.metric("AI cost so far", f"${usage['charged_usd']:.4f}")
        st.text(f"Spending limit: ${usage['limit_usd']:.2f}")
        if usage["reserved_usd"]:
            st.caption(
                f"Unreported request costs: ${usage['reserved_usd']:.4f} set aside."
            )
        st.caption("We save previous answers to keep costs down.")
    return document, trained, generate


def render_result(result):
    if result["status"] == "insufficient_evidence":
        st.info(
            "I couldn’t find the answer in the pages I checked. "
            + result["missing_information"]
        )
    elif result["status"] == "retrieval_only":
        st.info(
            "Here’s the document text I found. Add an API key to get a written answer."
        )
    else:
        lookup = {chunk["id"]: chunk for chunk in result["evidence"]}
        for index, claim in enumerate(result["claims"]):
            source = lookup[claim["citation_id"]]
            name = document_config(source["doc_id"])["display_name"]
            st.write(claim["text"])
            if st.button(
                f"Read page {source['page']} · {name} ↗", key=f"claim-{index}"
            ):
                show_page(source["doc_id"], source["page"])
            with st.expander("Exact words from the document"):
                st.write(claim["evidence_quote"])
        if result["cached"]:
            cost_note = "Saved answer — no extra charge"
        elif result["cost_usd"] is not None:
            cost_note = f"This answer cost ${result['cost_usd']:.6f}"
        else:
            cost_note = (
                "The cost hasn’t been reported; money is set aside within your limit"
            )
        st.caption("Quotes checked against the document text. " + cost_note)
    if result.get("ranking"):
        with st.expander("How this answer was found"):
            st.write(
                f"We searched for useful text, then read up to {SEARCH.context_pages} pages to write the answer."
            )
            st.caption(
                "Search method: "
                + METHOD_NAMES.get(result["ranking"], result["ranking"])
            )
            scope = result.get("search_scope")
            st.caption(
                "Documents searched: "
                + (
                    document_config(scope)["display_name"]
                    if scope and scope != "all"
                    else "all documents"
                )
            )
    st.markdown("#### Pages we checked")
    st.caption("Open one to read the text or see the original page.")
    for index, source in enumerate(result["evidence"]):
        name = document_config(source["doc_id"])["display_name"]
        with st.expander(f"{name} · Page {source['page']}"):
            st.text(source["text"])
            if source["uncertainties"]:
                st.caption(
                    "Some words were hard to read: "
                    + "; ".join(source["uncertainties"])
                )
            if st.button("Open the original page", key=f"view-{index}"):
                show_page(source["doc_id"], source["page"])


def render_questions(retriever, client, document, trained, generate):
    st.caption(
        "Answers refer to the supplied editions and tender. Check the source before applying a rule."
    )
    if EXAMPLES:
        for column, example in zip(st.columns(len(EXAMPLES)), EXAMPLES):
            if column.button(
                example["label"], width="stretch", help=example["question"]
            ):
                st.session_state.pending = example["question"]
    question = st.chat_input(
        "Ask in English or Hindi…", max_chars=SEARCH.max_question_chars
    ) or st.session_state.pop("pending", None)
    if question:
        st.session_state.question = question
        st.session_state.result = None
        st.session_state.failure = None
        with st.spinner("Reading the documents…"):
            try:
                if generate:
                    st.session_state.result = answer(
                        question,
                        retriever,
                        client=client,
                        trained=trained,
                        document=document,
                    )
                else:
                    st.session_state.result = {
                        "status": "retrieval_only",
                        "evidence": gather_evidence(
                            question, retriever, trained=trained, document=document
                        ),
                    }
            except (APIError, OSError, ValueError, RuntimeError) as exc:
                st.session_state.failure = str(exc)
        st.rerun()
    if st.session_state.get("question"):
        st.markdown("#### " + st.session_state.question)
    if st.session_state.get("failure"):
        st.error(st.session_state.failure)
    if st.session_state.get("result"):
        render_result(st.session_state.result)
    elif not st.session_state.get("failure"):
        st.markdown("#### Try a question above, or type your own")
        st.write(
            "You can ask in English or Hindi. Each answer comes with a page you can check yourself."
        )


def render_documents(manifest):
    for document in manifest:
        with st.container(border=True):
            st.subheader(document_config(document["id"])["display_name"])
            st.caption(document["title"])
            st.caption(f"{document['language']} · {document['pages']} pages")
            st.download_button(
                "Download PDF",
                document_path(document["id"]).read_bytes(),
                file_name=document["file"],
                key=document["id"],
            )
            if st.button("Read the first page", key="first-" + document["id"]):
                show_page(document["id"], 1)
    st.caption("Page numbers count from the start of each PDF, including its cover.")


def render_comparison(modes):
    st.table(
        [
            {
                "Search method": METHOD_NAMES.get(name, name),
                "Questions with useful text found": f"{sum(row['hit_at_5'] for row in data['details'])} out of {data['questions']}",
                "Useful text came first": f"{sum(row['rank'] == 1 for row in data['details'])} out of {data['questions']}",
            }
            for name, data in modes.items()
        ]
    )


def render_answer_checks(path, title):
    if not path.exists():
        return
    report = json.loads(path.read_text())
    checks = report["results"]
    reviewed = [row for row in checks if row.get("manual_review")]
    st.subheader(title)
    st.caption(
        f"Recorded search method: {METHOD_NAMES.get(report['ranking'], report['ranking'])}. These are separate from the passage-ranking checks."
    )
    st.write(
        f"{report['citation_validation_passes']} of {report['questions']} responses passed citation and quote validation. That does not by itself establish answer correctness."
    )
    if reviewed:
        correct = sum(
            row["manual_review"].get("verdict") == "supported_correct_answer"
            for row in reviewed
        )
        answerable = sum(row["answerable"] for row in reviewed)
        declined = sum(
            row["manual_review"].get("verdict") == "correct_abstention"
            for row in reviewed
        )
        missing = sum(not row["answerable"] for row in reviewed)
        st.write(
            f"Manual review: {correct} of {answerable} answerable questions were correct; {declined} of {missing} missing-information questions were declined correctly."
        )
    else:
        st.caption("Manual answer review is pending.")


def render_evaluation(retriever):
    st.subheader("Did the extra training help?")
    if retriever.model:
        st.write(
            f"We adapted a language model to choose useful paragraphs using {retriever.model['training_pairs']} reviewed examples. It reads the question and paragraph together."
        )
    path = REPORTS / "retrieval.json"
    if not path.exists():
        st.info(
            "The check results haven’t been prepared yet. Run coso-rag evaluate to create them."
        )
        return
    report = json.loads(path.read_text())
    render_comparison(report["splits"]["test"])
    st.write(
        "The trained model is the default after passing comparison checks."
        if report["default"] == "trained"
        else "Original search remains the default because the trained model did not pass every check."
    )
    st.caption(
        "Useful text found means it appeared among the first five results. These scores measure finding evidence, not written-answer correctness."
    )
    heldout = report["splits"].get("neural_holdout")
    if heldout:
        st.write(
            f"We also set aside {heldout['baseline']['questions']} new questions before this training run."
        )
        render_comparison(heldout)
    st.write(
        "Compare the before-training and after-training rows separately: using a pretrained model does not mean extra training caused the whole improvement."
    )
    render_answer_checks(
        REPORTS / "answers-baseline.json", "Earlier baseline answer checks"
    )
    latest = REPORTS / "answers-latest.json"
    if latest.exists():
        render_answer_checks(latest, "Latest answer checks")
    else:
        st.caption(
            "Paid answer testing has not been repeated for the current search and prompt."
        )
    with st.expander("More about the checks"):
        counts = report["split_counts"]
        st.write(
            f"{sum(counts.values())} authored questions, with related source groups kept together."
        )
        st.table(
            [{"Purpose": split, "Questions": count} for split, count in counts.items()]
        )
        st.caption(
            "These are checks on this small corpus, not a benchmark for all construction documents."
        )


def main():
    st.set_page_config(
        page_title="Site Notes · Construction documents", page_icon="📖", layout="wide"
    )
    st.markdown(
        "<style>" + (ROOT / ".streamlit/style.css").read_text() + "</style>",
        unsafe_allow_html=True,
    )
    paths = artifact_paths()
    try:
        fingerprint = json.loads((paths.index / "index.json").read_text())[
            "fingerprint"
        ]
        model_path = paths.model / "reranker.json"
        revision = (
            hashlib.sha256(model_path.read_bytes()).hexdigest()
            if model_path.exists()
            else None
        )
        retriever = load_retriever(fingerprint, revision)
        manifest = json.loads((paths.index / "documents.json").read_text())
    except (OSError, ValueError) as exc:
        st.error(
            f"The documents aren’t ready: {exc}. Follow README.md to rebuild the index."
        )
        st.stop()
    client = OpenRouter()
    document, trained, generate = render_sidebar(retriever, client)
    st.title("What would you like to know?")
    st.write(
        "Ask about these construction documents. Get an answer and open the page it came from."
    )
    languages = ", ".join(dict.fromkeys(item["language"] for item in manifest))
    st.caption(
        f"{len(manifest)} documents · {sum(item['pages'] for item in manifest)} pages · {languages}"
    )
    ask_tab, docs_tab, eval_tab = st.tabs(
        ["Ask a question", "Read the documents", "How we checked it"]
    )
    with ask_tab:
        render_questions(retriever, client, document, trained, generate)
    with docs_tab:
        render_documents(manifest)
    with eval_tab:
        render_evaluation(retriever)
