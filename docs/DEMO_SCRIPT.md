# Site Notes — three-minute demo

This script was checked against the running interface, README, DESIGN.md and saved evaluation results. A separate original assignment brief was not available; DESIGN.md records the requirements as 2–3 construction documents, cited answers and a training component.

## Before recording

- Use a full-width browser window so the document filter and cost sidebar are visible.
- Rehearse both example questions once. Confirm written answers, citations and PDF viewing work before recording. Paid answers need OpenRouter credits.
- Under Extra options, leave Try search with training and Write an answer enabled. Use All documents.
- Record the app window only. Keep credentials and account details off-screen.
- Cut long loading pauses. Rehearsed answers may display Saved answer — no extra charge; describe them honestly as cached if asked.
- Have docs/DESIGN.md ready in a second tab or editor. Do not show terminal installation or training during this short demo.

## 0:00–0:20 — purpose and interface

Show: Ask a question home screen, including the sidebar and document/page count.

Say: “Hi, this is Site Notes, my COSO construction-document assistant. It helps users ask questions in English or Hindi and check the evidence behind an answer. The interface keeps the main task simple: ask a question, read the answer, and open its source.”

## 0:20–0:35 — document coverage

Show: Read the documents. Point to the three document cards.

Say: “The corpus contains three different documents: a ductile-detailing standard, the CPWD construction contract, and a scanned Hindi tender. Together they cover 142 pages. Users can read or download the original PDFs here.”

## 0:35–1:10 — English answer and verification

Show: Ask a question → How much can a delay cost? Wait for the answer. Open Exact words from the document, then a Read page citation button. Close the page viewer after showing the source.

Say: “I’ll start with the example asking about the cap on delay compensation under CPWD clause two. The answer includes a page citation and a supporting quote. Opening the original page makes the result easy to verify. The system checks that citations exist and quotes match the supplied text, though those checks alone do not prove that every interpretation is correct.”

## 1:10–1:35 — Hindi interaction

Show: Click क्या मानसून का समय गिना जाता है? Show the resulting Hindi answer and citation. Read the actual displayed answer briefly if time permits.

Say: “Next, I’m asking whether the completion period includes the monsoon. This question is in Hindi, and the answer is presented in Hindi with evidence from the tender. Both languages use the same search and citation workflow.”

## 1:35–2:05 — architecture and training

Show: docs/DESIGN.md, Question to answer and Training sections. Return to the app.

Say: “The pipeline combines multilingual E5 semantic search with BM25 keyword search. MiniLM reranks the candidate passages, and DeepSeek reads the selected source pages to write a cited answer. For the training component, I fine-tuned the reranker’s last two transformer blocks and classifier on 146 reviewed question–passage pairs. DeepSeek and the embedding model were not fine-tuned.”

## 2:05–2:40 — evaluation without overstating results

Show: How we checked it. Point to Useful text came first in the original-check table, then the twelve-question table.

Say: “On twenty original answerable questions, useful evidence ranked first for fourteen with original search and nineteen with either reranker. Both rerankers also achieved twelve out of twelve on questions reserved before training. Most of the improvement comes from the pretrained model; extra fine-tuning showed only a small gain on an earlier regression set. These measure retrieval, not final-answer accuracy. All thirty automated tests passed.”

## 2:40–3:00 — limitations and close

Show: Sidebar spending limit, then return attention to the app home screen.

Say: “The app includes cached responses and a local spending cap. Its sources are historical documents, and English-to-Hindi retrieval still has a known miss. The repository includes setup instructions, architecture, trained weights, and evaluation reports. Thank you.”

## Recording rule

If either written-answer example fails during rehearsal, resolve the error before recording that sequence. If you choose to demonstrate search with Write an answer disabled, change the narration to “retrieved evidence” and do not call it a generated answer. Do not claim that fine-tuning caused the entire improvement or that twelve successful retrieval checks establish perfect answer accuracy.
