# Design and training

The brief asks for 2–3 different construction documents, answers with citations, and a training component. A passage reranker fits the existing search pipeline and can be trained locally on a small reviewed dataset.

## Documents

| Document | Why include it | Extraction |
|---|---|---|
| IS 13920:1993, 24 pages | Engineering clauses, numbers, formulas and qualifiers | Selectable text; cached vision transcription on pages 8–9 |
| CPWD GCC 2020, 112 pages | Long contract clauses that continue across pages | Selectable text; a configured cleanup rule removes decorative repeated digits |
| Rajasthan PWD tender 68/2024–25, 6 pages | Hindi scans, deadlines and a cost table | Cached vision transcription |

The project covers 142 pages. Rates and court judgments were left outside this focused experiment. Document titles, file names, OCR policy, clause-heading patterns and example questions are in `config.toml`. A reviewed formula correction is in `data/extraction-corrections.json`; the original transcription remains in the OCR cache. The correction fixes the square-root scope in the minimum reinforcement formula rather than inventing a missing value.

## Question to answer

1. PyMuPDF extracts text or renders a page for vision transcription. Each page retains its document ID, PDF page number and source hash.
2. Passages stay inside one PDF page and retain exact character offsets. The default is 180 words with 35-word overlap; Hindi uses 90 with 20-word overlap to reduce subword truncation.
3. Local multilingual E5 vectors find similar meaning. BM25 finds matching words and numbers. Reciprocal rank fusion combines their rankings.
4. The union of the top 15 results from each method gives at most 30 candidates. MiniLM reads each question and passage together and reorders that shortlist.
5. DeepSeek reads the top six distinct parent pages. Reading whole pages preserves table headings, amounts and clause exceptions that can be split between passages.
6. Each claim must include a supplied citation ID and an exact supporting quote. Validation rejects unknown citations, unmatched quotes and invalid refusals. It does not prove that a quote logically supports every claim.

Clause headings are detected using the configured pattern and carried across page breaks with their original heading and page. Answer and OCR prompts are separate files. They ask for source-bound reading; no test question has a special-case answer in the pipeline.

## Models

- **DeepSeek V4.1 Flash via OpenRouter:** transcribes scanned pages and writes cited answers. It is not fine-tuned.
- **`intfloat/multilingual-e5-small`:** creates query/passage vectors locally. Its revision is pinned in `config.toml`; it is not fine-tuned.
- **`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`:** ranks actual question–passage text pairs. Its multilingual pretraining fits the English/Hindi corpus and its size permits local adaptation.

The method follows the [Sentence Transformers training guide](https://www.sbert.net/docs/cross_encoder/training_overview.html), its [binary cross-entropy objective](https://www.sbert.net/docs/package_reference/cross_encoder/losses.html), and the [model card](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1). The base model was pretrained on multilingual retrieval data; our adaptation uses only reviewed construction pairs.

## Training

146 question–passage pairs: 68 useful and 78 misleading, from 48 training questions. Negatives are reviewed competing search results; unreviewed candidates are not automatically treated as negative. Training rejects labels from held-out source pages, duplicated pairs and overlapping question groups before downloading a model.

Each input combines the actual question with document title, clause metadata and paragraph text. Training and inference use the same formatting. The question is preserved while the passage is truncated to fit the 512-token total limit. Output scores are relevance logits, not probabilities of answer correctness.

Only the final two transformer blocks and classifier are adapted: 3,697,153 of 117,641,089 parameters. Embeddings and earlier blocks remain frozen. This is partial fine-tuning, not LoRA or DeepSeek training. The included Safetensors checkpoint contains the updated subset; all 36 selected tensors changed. Inference combines it with the pinned base model and verifies the checkpoint hash and parameter names.

The recorded run used AdamW, learning rate 2e-5, weight decay 0.01, batch size 4, three epochs, 10% warmup, gradient clipping at 1, float32 and seed 42. Positive loss weight is 78/68. These settings now live in `config.toml`. Eager attention avoids a Mac GPU dropout limitation. Mac GPU and CPU are supported; runs need not be bit-identical across hardware.

Eight development questions select the checkpoint by MRR, then hit@5, then earlier epoch. All three epochs tied ranking performance, so epoch one was selected. Losses were about 0.441, 0.406 and 0.397. Final evaluation cannot modify weights or choose another epoch; its guard can only approve or reject activation.

The original checks and earlier ten questions are regression sets because they were already examined. Twelve new questions across six source groups were reserved before neural training. Their file hash, the frozen checkpoint metadata and the adaptation hash are preserved in `evaluation/protocol.json`. Seven earlier negative labels from those reserved pages were removed before fitting. No unused Hindi source group was available for a new independent Hindi hold-out.

## Costs and follow-up

Local embedding, reranking and training use no paid APIs. OpenRouter calls have provider price caps, output limits, response caching and a shared SQLite spending ledger. Before each request the app reserves an estimated cost; unresolved outcomes keep their reservations. OCR is already cached for the supplied corpus.

Streamlit source watching is disabled. During a fresh browser query its watcher inspected Transformers' lazy modules, attempted unrelated vision imports, and exhausted the process. The supported `fileWatcherType = "none"` setting avoids that scan; code edits need a server restart.

The next useful work is better English-to-Hindi recall and more independently reviewed training questions. The existing pretrained model explains most of the measured gain; the small adaptation set does not establish broad improvement from fine-tuning. See [EVALUATION.md](EVALUATION.md) before describing the results in the discussion.
