# Site Notes

Ask questions about three construction documents in English or Hindi. The app finds useful passages, ranks them, and uses DeepSeek to write an answer with page citations and supporting quotes. You can open the original PDF page to check it.

This is the COSO take-home: cited RAG over an engineering standard, a contract and a Hindi tender, with a locally fine-tuned passage reranker.

[Watch the demo (2 min 41 sec)](https://youtu.be/t8HEJq1FGHc)

## Run

Use Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/). From this folder:

```bash
uv sync --python 3.12 --extra dev --locked
uv run streamlit run app.py
```

Open [localhost:8501](http://localhost:8501). Without an API key, you can search and read evidence. To generate written answers:

```bash
cp .env.example .env
```

Add your OpenRouter key to `.env`, then restart the app. Paid answers also require credits in your OpenRouter account. The default local spending limit is $0.50; it is a spending cap, not prepaid credit. No key is included in the submission.

On Windows, use PowerShell and enable UTF-8 before running Python commands so the Hindi text and configuration load correctly:

```powershell
$env:PYTHONUTF8 = '1'
uv sync --python 3.12 --extra dev --locked
# Only create .env if you have not already configured it:
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
.\run.ps1
```

The Windows launcher stores downloaded models in `.local/huggingface`. After both models have downloaded, use `.\run.ps1 -Offline` to skip online model checks. Answer generation still uses OpenRouter. For terminal commands that should reuse these model downloads, set `$env:HF_HOME = Join-Path (Get-Location) '.local\huggingface'` in the same PowerShell session.

Restart Streamlit after editing code or configuration. Automatic source watching is disabled because it probes Transformers' lazy imports and can load unrelated models.

The PDFs, extraction cache, index and adapted weights are included. You do not need to ingest or train before trying the app. For uncached questions, two pinned Hugging Face models download once: E5 and the MiniLM reranker, about 470 MB each. They run locally; DeepSeek answers use your OpenRouter account.

## Ask from the terminal

```bash
uv run coso-rag ask "What is the cap on total compensation for delay under CPWD clause 2?" --retrieve-only
uv run coso-rag ask "क्या कार्य पूरा करने की अवधि में मानसून शामिल है?"
uv run coso-rag ask "What is the online bid submission deadline in the Rajasthan tender?"
```

`--retrieve-only` makes no paid answer call. Use `--baseline` to compare original search, `--trained` to use the reranker explicitly, `--document contract` to restrict the corpus, and `--json` for structured output. Citations use PDF page numbers, including covers, rather than printed page numbers.

## Where to look

```text
app.py                  Streamlit entry point
config.toml             Models, limits, training settings and document registry
prompts/                Answer and OCR instructions
src/coso_rag/
  ingest.py             Extract text and preserve source pages
  retrieval.py          Keyword and semantic search
  reranking.py          Score question–paragraph pairs
  answering.py          Generate and validate cited answers
  training.py           Fine-tune the reranker
  datasets.py           Reviewed labels and split checks
  evaluation.py         Compare original/pretrained/trained retrieval
  live_evaluation.py    Explicit paid answer checks
  openrouter.py         Cached API calls and shared spending ledger
  ui.py / cli.py        Interfaces
artifacts/
  index/                Extracted pages, passages and embedding vectors
  model/                Adapted weights and model metadata
  cache/                OCR, query vectors and model scores
data/documents/        Original PDFs
evaluation/
  datasets/             Questions and reviewed training labels
  reports/              Retrieval results and earlier answer review
  protocol.json         Training and hold-out record
  frozen-model.json     Model metadata saved before final evaluation
tests/                 Checks grouped by responsibility
docs/                  Design choices and evaluation
```

See [the design](docs/DESIGN.md) for the pipeline and training choices, and [the evaluation](docs/EVALUATION.md) for results and limits. The retired feature-model experiments and development logs are kept outside the submission in a separate research archive.

## What was trained

The reranker is `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`. It reads your question alongside a candidate paragraph and returns a relevance score. We adapted its last two transformer blocks and classifier on 146 reviewed examples, updating 3,697,153 parameters. DeepSeek and E5 were not fine-tuned.

On the original checks, the correct paragraph ranked first for 14/20 questions with original search and 19/20 with either the pretrained or adapted reranker. Both rerankers scored 12/12 on questions reserved before this training run. Most improvement came from the pretrained model; adaptation added a small gain only on the earlier regression checks. These scores measure retrieval, not final answer correctness.

## Check or rebuild

```bash
uv run pytest -q
uv run ruff check app.py src tests
uv run ruff format --check app.py src tests
uv run coso-rag evaluate
```

To rebuild from the included extraction cache and train again:

```bash
uv run coso-rag ingest
uv run coso-rag train
uv run coso-rag evaluate
```

`ingest` makes no paid calls unless `--vision` is explicitly supplied. A missing OCR cache needs either the shipped cache or an authorized vision call. Training and retrieval evaluation run locally. Changing source text or models requires rebuilding the index and training again; the old model is not activated against a different index.

`uv run coso-rag evaluate-live` explicitly runs paid answer checks and writes `evaluation/reports/answers-latest.json`. It preserves the earlier baseline review. Check the spending ledger with `uv run coso-rag cost`.

## Limits

The documents are historical editions and a tender notice, not current regulations. OCR and formulas can be imperfect; one reviewed correction retains its original extraction. The English monsoon question still misses its Hindi evidence in the search shortlist. A reranker cannot repair that miss.

Provider price caps, bounded output and SQLite reservations constrain app spending. Unknown request costs retain their reservations; there are no automatic paid retries. The local budget is not an account-wide OpenRouter limit. Private keys and usage ledgers stay outside the submission. See the evaluation report for the earlier answer checks and their limitations.
