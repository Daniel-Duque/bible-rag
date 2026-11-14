# Bible RAG: KJV / Reina-Valera 1960

Ask questions about the Bible in English or Spanish from your terminal. Answers are grounded in,
and cite, passages from two translations: the **King James Version** (`src/data/kjv.pdf`) and the
**Reina-Valera 1960** (`src/data/biblia-reina-valera-1960.pdf`).

The Python service in `bible_rag/` is a refactor of the prototype notebook `RAG0.ipynb`. It uses
the same stack (multilingual MiniLM embeddings + FAISS + Qwen2.5-1.5B-Instruct), turned into an
installable package with a CLI, an HTTP service and tests.

> The React/Express demo in `src/` + `server.ts` (AI Studio app, `npm run dev`) is a separate
> front-end with a small hand-written corpus. This service doesn't change or depend on it.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 for you if needed).

```bash
uv sync                      # create .venv with runtime + dev dependencies
uv run bible-rag ingest      # parse both PDFs and build the indexes (one time, a few minutes)
uv run bible-rag ask "What does the Bible say about forgiveness?"
uv run bible-rag ask "¿Qué dice la Biblia sobre el perdón?"
```

The first run downloads the models from Hugging Face: the embedder (~0.5 GB) and
Qwen2.5-1.5B-Instruct (~3 GB). After that you can set `HF_HUB_OFFLINE=1`. It runs on CPU
(float32, ~7 GB RAM); CUDA or Apple-Silicon GPUs are used automatically when available.

## Using it from the terminal

| Command | What it does |
|---|---|
| `bible-rag ask "question"` | One question → streamed answer + cited sources |
| `bible-rag chat` | Interactive session; models load once, then each question is fast |
| `bible-rag search "text"` | Semantic search only, no LLM (fast, shows passage text) |
| `bible-rag serve` | HTTP service that keeps the models in memory |
| `bible-rag ingest [--force]` | (Re)build the indexes |

Useful options for `ask`, `search` and `chat`:

```bash
bible-rag ask -k 5 "..."                      # passages retrieved per version (default 3)
bible-rag ask --versions rvr1960 "..."        # only one translation (kjv | rvr1960 | kjv,rvr1960)
bible-rag ask --show-context "..."            # also print the retrieved passages
bible-rag ask --json "..." | jq .sources      # machine-readable output
echo "¿Quién fue Moisés?" | bible-rag ask     # the question can come from stdin
```

Inside `chat`: `/search <text>`, `/k <n>`, `/sources on|off`, `/help`, `/quit`.

Output looks like this:

```
<answer, streamed as it is generated, citing e.g. (John 3:15-17, KJV)>

Sources:
  [1] John 3:15-17 (KJV)  score=0.…  p.2417
  [2] Juan 3:14-18 (RVR1960)  score=0.…  p.1447
  ...
```

`p.` is the page in the PDF, so you can check any citation in the source document.

### As a service

Loading torch and the models takes seconds to minutes, so for repeated questions keep them warm:

```bash
uv run bible-rag serve                                   # http://127.0.0.1:8000
# from any other terminal:
uv run bible-rag ask --server http://127.0.0.1:8000 "What is faith?"
export BIBLE_RAG_SERVER_URL=http://127.0.0.1:8000        # or make it the default
uv run bible-rag chat
curl -s localhost:8000/ask -H 'content-type: application/json' \
     -d '{"question": "¿Qué es la fe?", "k": 3, "versions": ["kjv", "rvr1960"]}'
```

Endpoints: `GET /health`, `POST /search`, `POST /ask` (body: `question`, optional `k`,
`versions`). Interactive docs are at `/docs`. Use `serve --no-llm` for a retrieval-only
service. The service binds to localhost by default. Use `--host 0.0.0.0` only on a trusted
network: it has no authentication.

## Configuration

Settings come from `BIBLE_RAG_*` environment variables or a `.env` file. All are listed with
their defaults in [`.env.example`](.env.example). The most useful ones:

| Variable | Default | Notes |
|---|---|---|
| `BIBLE_RAG_LLM_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | Any HF chat model, e.g. `Qwen/Qwen2.5-3B-Instruct` for better answers |
| `BIBLE_RAG_EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Changing it requires `ingest --force` (checked at startup) |
| `BIBLE_RAG_DEVICE` | `auto` | `cpu`, `cuda`, `mps` |
| `BIBLE_RAG_TOP_K` | `3` | Passages **per version** |
| `BIBLE_RAG_TEMPERATURE` | `0.2` | `0` = deterministic |
| `BIBLE_RAG_INCLUDE_APOCRYPHA` | `false` | The KJV PDF includes the Apocrypha; RVR1960 has none |

## How it works

```
PDF ──pypdfium2──► page text ──version parser──► verses/lines with book:chapter:verse
    ──chunker (≤450 chars, same chapter, overlap)──► passages
    ──multilingual MiniLM──► FAISS index per version          (bible-rag ingest)

question ──embed──► top-k per version ──► prompt with cited passages ──► Qwen ──► answer + sources
```

Changes from the notebook, and why:

* **Verse-level citations.** The notebook cited only the file name. Each PDF now has its own
  parser. KJV references are exact. The RVR1960 PDF prints verse numbers in the margin, so its
  references are accurate to the printed line (a passage may start with the tail of the
  previous verse). A regression test checks the parsers against the canonical structure:
  66 books, 1,189 chapters, 31,102 KJV verses, with RVR1960 matching chapter by chapter.
* **Text cleanup.** Ligatures (`dĳo` → `dijo`), line-break hyphenation (`Jeho-vá` →
  `Jehová`, while real compounds like `Bet-el` are kept), drop caps, running headers/footers,
  table of contents and indexes are handled.
* **One FAISS index per version.** Multilingual embeddings still favour same-language text,
  so a single index would answer Spanish questions almost only from RVR1960. Querying both
  guarantees both translations appear in the context.
* **Chunk size 450 instead of 700.** The embedding model truncates at 128 tokens, so most of a
  700-character chunk was never embedded.
* **No LangChain.** Its FAISS wrapper lives in `langchain-community` (being sunset) and
  persists with pickle. FAISS + sentence-transformers are used directly, and passages are
  stored as JSON lines.
* **Runs anywhere.** Device/dtype are auto-detected (the notebook assumed a Colab T4), tokens
  are streamed to the terminal, and the system prompt is a Bible-study assistant that answers
  in the question's language and cites `(Book c:v, VERSION)`.

## Development

```bash
uv sync                               # also installs pytest, ruff, ipykernel (dev group)
make test        # or: uv run pytest                        (~1 min, everything)
make test-fast   # or: uv run pytest -m "not pdf and not hf"  (~15 s)
make lint        # ruff check + format check
make format
```

Test layers:

* `tests/test_corpus.py`: parser/chunker unit tests on synthetic pages that mimic each PDF.
* `tests/test_corpus_pdfs.py` (`pdf` marker): parses the real PDFs and checks the canonical
  book/chapter/verse counts.
* `tests/test_rag.py`, `tests/test_service_cli.py`: pipeline, HTTP service and CLI, with a
  fake embedder/LLM.
* `tests/test_hf_integration.py` (`hf` marker): builds *tiny random* Qwen2/BERT models on
  disk and runs the real transformers / sentence-transformers code through the CLI, with no
  downloads.

To use the notebook with this environment: `uv run python -m ipykernel install --user --name bible-rag`.

Project layout:

```
bible_rag/
  corpus.py   PDF extraction, KJV/RVR1960 parsers, cleanup, chunking
  index.py    embedder + per-version FAISS index (build/save/load)
  ingest.py   builds the indexes
  llm.py      local chat model (device/dtype, streaming)
  rag.py      retrieval + prompt + answer (BibleRAG)
  server.py   FastAPI app
  cli.py      `bible-rag` command
  config.py   settings from env/.env
tests/        pytest suite
storage/      generated indexes (git-ignored)
RAG0.ipynb    original prototype (SECOP contracts example)
```

## Notes

* Known text differences the tests account for: 3 John has 15 verses in RVR1960 (14 in KJV),
  and this RVR1960 PDF prints a stray `10` in Psalm 47.
* A 1.5B model is small. It follows the grounding instructions reasonably well but can still
  misquote. The cited sources are printed separately so answers can be checked.
  `BIBLE_RAG_LLM_MODEL=Qwen/Qwen2.5-3B-Instruct` (or 7B on a GPU) gives better answers.
* The RVR1960 PDF states "Permission for personal use only" (Bibles.org.uk). The Reina-Valera
  1960 text is copyrighted by the Sociedades Bíblicas en América Latina. Check the licensing
  before any public or commercial deployment. The KJV is public domain in most countries.
