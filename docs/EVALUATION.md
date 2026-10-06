# Evaluation

The fine-tuned text model now ranks the reviewed evidence first for **19/20 original questions**, versus **14/20** with original fusion. It ranks evidence first for **12/12 questions newly reserved before neural training**, versus **11/12** for original fusion. The retrieval shortlist is unchanged. The new model passes the deployment guard and is active.

## Training and comparison

The implementation follows pretrained multilingual cross-encoder adaptation with BCE on source-reviewed query/passage labels. [DESIGN.md](DESIGN.md) records sources, model revision, input formatting, loss and hyperparameters. We adapted 3,697,153 of 117,641,089 parameters on 146 pairs (68 positive, 78 negative), using the last two transformer blocks and classifier. All 36 selected tensors changed. Epoch one was chosen on eight development questions after three training epochs tied ranking performance. No final test scores were used to fit parameters or select the epoch.

102 questions: 48 training, eight development, 24 original tests (20 answerable and four missing-information), ten earlier tests, twelve latest tests. The latest twelve cover six groups in the standard and contract. No unused Hindi tender group was available for an independent new Hindi test; original Hindi regressions remain included. Positive and negative fitting passages exclude every held-out source page. Seven negative labels were removed before neural training because they came from newly reserved pages. Dataset/checkpoint hashes were frozen before final scoring.

## Retrieval results

| Check / method | Evidence first | Evidence hit@5 | MRR | Candidate recall |
|---|---:|---:|---:|---:|
| Development / baseline | 6/8 | 87.5% | 0.792 | 87.5% |
| Development / pretrained | 7/8 | 87.5% | 0.875 | 87.5% |
| Development / trained | 7/8 | 87.5% | 0.875 | 87.5% |
| Original checks / baseline | 14/20 | 95.0% | 0.793 | 95.0% |
| Original checks / pretrained | 19/20 | 95.0% | 0.950 | 95.0% |
| Original checks / trained | 19/20 | 95.0% | 0.950 | 95.0% |
| Earlier ten checks / baseline | 6/10 | 80.0% | 0.679 | 100.0% |
| Earlier ten checks / pretrained | 8/10 | 100.0% | 0.883 | 100.0% |
| Earlier ten checks / trained | 8/10 | 100.0% | 0.900 | 100.0% |
| New twelve checks / baseline | 11/12 | 100.0% | 0.958 | 100.0% |
| New twelve checks / pretrained | 12/12 | 100.0% | 1.000 | 100.0% |
| New twelve checks / trained | 12/12 | 100.0% | 1.000 | 100.0% |

**Attribution matters:** the pretrained text model already achieves most of these improvements. Fine-tuning raises MRR on the earlier ten checks from 0.883 to 0.900, but ties the pretrained model on development, original checks and the newly reserved twelve. We do not claim a broad adaptation benefit based on that small, previously examined set. The new architecture is a stronger baseline; the real fine-tuning component remains measurable and inspectable.

Evidence first means the top passage contains the reviewed anchor in the correct document. Hit@5 checks the first five; MRR averages the reciprocal rank of the first supported passage, with zero for a miss. Some equivalent support can lack the exact anchor; this is reproducible but not exhaustive relevance annotation. Candidate recall checks the unchanged union of top 15 semantic and top 15 lexical matches. Unanswerable questions are excluded from retrieval metrics.

The English monsoon question still misses its Hindi source in the candidate pool. A reranker cannot fix that. Original tests and earlier ten questions have already been used in development. The latest twelve were reserved before neural training and only scored after freezing the selected checkpoint. All are authored by the same developer from the same small corpus, and paraphrases are correlated. They are not an external benchmark; future tuning would require new checks.

## Written answers and costs

The preserved **baseline** live run passed 24/24 citation/quote checks, answered 19/20 supported questions correctly on manual review, and declined all four missing-information questions. It also passed four extra table/formula/filter smoke checks. These scores do not measure the newly active text ranking. Neural training, retrieval evaluation and the code cleanup made no new paid answer calls. The cleaned answer prompt removes a source-specific reading hint; paid answer accuracy has not been reevaluated with that prompt. Quote matching still detects invented sources/quotes without proving semantic support; prior manual review and OCR formula provenance remain included.

This revision costs **$0 in additional API fees**. Prior spend remains $0.02412782 across 135 calls, no unresolved reservations, with a $0.50 configured app limit. Both base models download once for local inference/training. The roughly 15 MB trained subset and cached evaluated scores ship in the archive; private credentials, ledgers and base-model download caches do not. Reports keep source-page references and exact claim quotes rather than repeating entire pages for every question.

## Engineering checks

30 automated tests passed. They cover prior budget/citation/source/extraction checks plus held-out isolation including the latest source pages, real adapted tensors and frozen checkpoint hashes, invalid checkpoint hashes/paths, text score-cache invalidation by question/passage/checkpoint, and the unchanged deployment guard. Evaluation requires all four check sets for a neural model. Cache keys include exact text and checkpoint identity; the Streamlit resource cache includes model metadata hash. Query-vector cache keys also include the embedding token limit. A regression test compares every saved retrieval result, and ingestion reproduces every shipped page. An uncached model prediction checks that the reorganized checkpoint loader produces the saved scores.

Browser checks cover default trained search, comparison tables with pretrained results, original source viewing, document downloads, explicit original-search comparison and a 390px mobile view. These use document search rather than new paid answers. Source inspection and the generated report remain separate from the earlier baseline answer checks.

## Reproduce

```bash
uv sync --python 3.12 --extra dev --locked
uv run coso-rag train
uv run coso-rag evaluate
uv run pytest -q
uv run ruff check app.py src tests
uv run ruff format --check app.py src tests
```

Training uses a Mac GPU when available, otherwise CPU, and needs the pinned base model. CPU timings vary. Future training need not be bit-identical across hardware/dropout kernels; a fixed seed, protocol and frozen checkpoint are included. Evaluation cannot change adapted tensors or checkpoint epoch. The shipped index/OCR cache requires no paid extraction. `evaluate-live` explicitly invokes paid answer checks and writes `evaluation/reports/answers-latest.json`, preserving `answers-baseline.json`. It must be reviewed separately.
