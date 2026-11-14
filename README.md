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
uv sync                      # .venv with runtime + dev deps + offline fallback models (~130 MB)
uv run bible-rag models      # which models are usable on this machine
uv run bible-rag ingest      # parse both PDFs and build the indexes (one time, a few minutes)
uv run bible-rag ask "What does the Bible say about forgiveness?"
uv run bible-rag ask "¿Qué dice la Biblia sobre el perdón?"
```

The first run downloads the preferred models from Hugging Face: the embedder (~0.5 GB) and
Qwen2.5-1.5B-Instruct (~3 GB). They run on CPU (float32, ~7 GB RAM); CUDA or Apple-Silicon
GPUs are used automatically when available. **If Hugging Face can't be reached, the service
still works:** it switches automatically to models that ship with the install (see
[Model fallback](#model-fallback)). Run `uv run bible-rag models` to see what's usable on your
machine.

## Using it from the terminal

| Command | What it does |
|---|---|
| `bible-rag models [--test]` | Which embedders / LLMs are usable here, and which will be used |
| `bible-rag eval` | Measure retrieval and answer quality (see [Measuring usefulness](#measuring-usefulness)) |
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

Endpoints: `GET /health` (active models + fallback reasons), `GET /models`,
`POST /search`, `POST /ask` (body: `question`, optional `k`, `versions`; the response
includes `model`, `notes` and `grounding`). Interactive docs are at `/docs`. Use `serve --no-llm` for a retrieval-only
service. The service binds to localhost by default. Use `--host 0.0.0.0` only on a trusted
network: it has no authentication.

## Model fallback

Each model is picked from an ordered chain. The first entry that loads is used, and the reason
each earlier entry was skipped is reported. An LLM that fails *while* answering, before it
produced any text, hands the question to the next entry.

| | Chain (default order) | Works offline | Notes |
|---|---|---|---|
| **Embeddings** | `hf:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | after first download | preferred |
| | `usem3` | ✓ bundled | Google Universal Sentence Encoder multilingual v3 (pure numpy, from the `fast-universal-sentence-encoder` wheel); cross-lingual EN↔ES |
| **LLM** | `hf:Qwen/Qwen2.5-1.5B-Instruct` | after first download | preferred |
| | `ollama[:model]` | ✓ local server | free; uses the first installed model unless `BIBLE_RAG_OLLAMA_MODEL` is set |
| | `openai[:model]` | ✓ local server | any OpenAI-compatible server: LM Studio, llama.cpp `llama-server`, vLLM, LocalAI (`BIBLE_RAG_OPENAI_BASE_URL`) |
| | `smollm2` | ✓ bundled | SmolLM2-135M-Instruct, installed by `uv sync` (see below); **English only, weak** |
| | `extractive` | ✓ always | no LLM: quotes the best passages verbatim, in the question's language |

```bash
uv run bible-rag models          # what is usable here, and what will be used
uv run bible-rag models --test   # also load each available LLM and generate a reply
```

Example output in a sandbox where Hugging Face is blocked and no local server runs:

```
Embedding models (tried in this order):
  ✗ hf:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
      not cached and Hugging Face unreachable (ConnectError: [Errno -2] Name or service not known)
  ✓ usem3  <- will be used
Language models (tried in this order):
  ✗ hf:Qwen/Qwen2.5-1.5B-Instruct
      not cached and Hugging Face unreachable (...)
  ✗ ollama
      no server at http://127.0.0.1:11434/api/tags (ConnectError: [Errno 111] Connection refused)
  ✗ openai
      not configured (set BIBLE_RAG_OPENAI_BASE_URL)
  ✓ smollm2  <- will be used
  ✓ extractive
```

Details:

* **Fast failure.** An uncached Hugging Face model is checked with one short request
  (`BIBLE_RAG_CONNECT_TIMEOUT`, 3 s). Without that check, `huggingface_hub` retries with
  backoff for over a minute before giving up. Cached models load with `local_files_only`, so
  they work with no network.
* **Indexes follow the embedder.** Indexes are stored per embedder
  (`storage/index/<embedder>/<version>`). Queries always use the embedder that built the
  index, and the first embedder in the chain that has an index is selected. `ingest` builds
  with the first embedder that loads. Run it again once the preferred model is reachable
  (`bible-rag ingest --embedder hf:<model>`) and that index takes precedence.
* **Language limits.** SmolLM2 only handles English. A Spanish question is answered
  extractively rather than with broken Spanish. Small models get a shorter prompt and only
  the passages in their language.
* **Every answer is checked** (`bible_rag/grounding.py`). References such as `John 3:16`,
  `Juan 3:16` or `1 Cor. 13:4` must point into a retrieved passage, and quoted text must
  appear in one. Anything that doesn't is printed as
  `⚠ Not found in the retrieved passages, verify before trusting: …`, followed by the
  passages themselves. The HTTP API returns this as `grounding`, along with `model` and
  `notes`.
* **How the bundled LLM is shipped.** It comes from the `llm-smollm2` wheel on PyPI (by Simon
  Willison, Apache-2.0, hash-pinned in `uv.lock`), in the default `fallback-llm` dependency
  group. Only its `.gguf` file is used, loaded with transformers. The package's own runtime
  (`llm` + `llama-cpp-python`, which needs a C++ build) is disabled with
  `override-dependencies`. On first use the GGUF is converted once into
  `storage/models/` (~45 s), and later loads take about 1 s.

Set your own order with `BIBLE_RAG_LLM_CHAIN` / `BIBLE_RAG_EMBEDDING_CHAIN`, e.g.
`BIBLE_RAG_LLM_CHAIN=ollama:qwen2.5:7b,smollm2,extractive`.

## Measuring usefulness

`bible-rag eval` asks 16 questions about well-known passages (creation, the ark, Jonah, the
Ten Commandments, David and Goliath, the Beatitudes, the prodigal son, …), each in English
and in Spanish, and scores:

* **retrieval hit rate**: whether a passage from a correct chapter is among the top-k, per
  question language × translation;
* **answers**: which model answered, whether the answer cites a correct passage that the
  grounding check verified, and the share of answers with unverified references or quotes.

```bash
uv run bible-rag eval --retrieval-only       # fast, no LLM
uv run bible-rag eval --output report.json   # full run, per-question details in the JSON
```

Compare configurations by changing the chains (for example `BIBLE_RAG_LLM_CHAIN=extractive`
or `ollama:qwen2.5:7b`) and re-running.

### Results without Hugging Face

Measured in a sandbox where only PyPI was reachable, so the bundled `usem3` + `smollm2` were
used, with k = 3 passages per translation:

| Retrieval (a correct chapter among the retrieved passages) | English questions | Spanish questions |
|---|---|---|
| embeddings only (`BIBLE_RAG_RETRIEVAL=dense`) | 62.5 % | 56.3 % |
| **hybrid** embeddings + BM25 (default) | **75.0 %** | **75.0 %** |

| Answers (hybrid retrieval) | cites a correct passage | has unverified references/quotes |
|---|---|---|
| `smollm2` (English questions) | 4 / 16 | 9 / 16 |
| `extractive` (English questions) | 11 / 16 | 0 / 16 |
| `extractive` (Spanish questions) | 11 / 16 | 0 / 16 |

Retrieval is useful offline: most questions surface the right passage in both translations,
including across languages. The 135M bundled LLM, however, **lowers** answer quality compared
with quoting the passages directly. With *"Make thee an ark of gopher wood"* in its context, it
said the ark was made of cedar. It's kept in the chain because it is the model that is
guaranteed to load, and its answers are always checked, with the evidence printed. If you
prefer verbatim passages over a weak summary when no better model is available, drop it:
`BIBLE_RAG_LLM_CHAIN=hf:Qwen/Qwen2.5-1.5B-Instruct,ollama,openai,extractive`. Misses are
mostly stories known by a name the text doesn't use (Ten Commandments, the prodigal son, the
good Samaritan).

## Configuration

Settings come from `BIBLE_RAG_*` environment variables or a `.env` file. All are listed with
their defaults in [`.env.example`](.env.example). The most useful ones:

| Variable | Default | Notes |
|---|---|---|
| `BIBLE_RAG_LLM_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | Preferred model (first in the chain). Any HF chat model, e.g. `Qwen/Qwen2.5-3B-Instruct` |
| `BIBLE_RAG_LLM_CHAIN` | see [Model fallback](#model-fallback) | Comma-separated order of LLMs to try |
| `BIBLE_RAG_OLLAMA_URL` / `_MODEL` | `http://127.0.0.1:11434` / first installed | Local Ollama server |
| `BIBLE_RAG_OPENAI_BASE_URL` / `_MODEL` / `_API_KEY` | unset | LM Studio (`http://127.0.0.1:1234/v1`), llama.cpp server, vLLM… |
| `BIBLE_RAG_EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Preferred embedder; needs its own `ingest` |
| `BIBLE_RAG_EMBEDDING_CHAIN` | `hf:<embedding model>,usem3` | Order of embedders to try |
| `BIBLE_RAG_DEVICE` | `auto` | `cpu`, `cuda`, `mps` |
| `BIBLE_RAG_TOP_K` | `3` | Passages **per version** |
| `BIBLE_RAG_RETRIEVAL` | `hybrid` | `hybrid` (embeddings + BM25 keywords) or `dense` |
| `BIBLE_RAG_TEMPERATURE` | `0.2` | `0` = deterministic |
| `BIBLE_RAG_INCLUDE_APOCRYPHA` | `false` | The KJV PDF includes the Apocrypha; RVR1960 has none |

## How it works

```
PDF ──pypdfium2──► page text ──version parser──► verses/lines with book:chapter:verse
    ──chunker (≤450 chars, same chapter, overlap)──► passages
    ──multilingual MiniLM──► FAISS index per version          (bible-rag ingest)

question ──embed + BM25 keywords (fused)──► top-k per version ──► prompt with cited passages
         ──► first available LLM (fallback chain) ──► answer ──► grounding check ──► answer + sources
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
* **Hybrid retrieval.** Embedding search is fused with a small BM25 keyword index using
  reciprocal-rank fusion. Names such as *Goliath* or *Samaritan* are blurred by sentence
  embeddings but matched exactly by keywords. With the offline embedder, this raised the
  retrieval hit rate from 62.5 % to 75 % for both English and Spanish questions
  (`bible-rag eval --retrieval-only`, `BIBLE_RAG_RETRIEVAL=dense|hybrid`).
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
  hub.py      fast Hugging Face availability checks
  llm.py      chat models (transformers, Ollama/OpenAI-compatible, bundled SmolLM2), ModelChain
  rag.py      retrieval + prompt + answer with fallback (BibleRAG)
  grounding.py  verifies references and quotes of every answer
  probe.py    `bible-rag models`
  evaluation.py `bible-rag eval` gold questions and metrics
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
  misquote. The grounding check flags that, and the sources are always printed.
  `BIBLE_RAG_LLM_MODEL=Qwen/Qwen2.5-3B-Instruct`, or a 7B model through Ollama, gives better
  answers.
* The bundled SmolLM2-135M is a **safety net, not a quality model**. In tests it answered
  "Who built the ark?" correctly ("Noah"), but it also invented quotes and references on
  harder questions. Its answers are therefore always checked, and Spanish questions fall back
  to extractive answers.
* The RVR1960 PDF states "Permission for personal use only" (Bibles.org.uk). The Reina-Valera
  1960 text is copyrighted by the Sociedades Bíblicas en América Latina. Check the licensing
  before any public or commercial deployment. The KJV is public domain in most countries.
