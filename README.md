# Bible RAG: KJV / Reina-Valera 1960

Ask questions about the Bible in English or Spanish, from the terminal or a web chat. Answers
cite verses from the **King James Version** and the **Reina-Valera 1960** (`src/data/*.pdf`),
and every citation is checked against the retrieved passages. Refactored from `RAG0.ipynb`:
multilingual MiniLM embeddings + FAISS + Qwen2.5-1.5B-Instruct, with automatic fallbacks.

> `src/` (except `src/data`), `server.ts` and the npm files belong to a separate AI Studio demo;
> this project does not use them.

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                         # environment, incl. offline fallback models (~130 MB)
uv run bible-rag models         # which models are usable here
uv run bible-rag ingest         # build the search index (once, a few minutes)
uv run bible-rag ask "What does the Bible say about forgiveness?"
uv run bible-rag chat           # terminal chat, models load once
uv run streamlit run app.py     # web chat on http://localhost:8501
```

Other commands: `bible-rag search "text"` (retrieval only), `bible-rag eval` (quality metrics).
`ask`/`chat` options: `-k` passages per translation, `--versions kjv|rvr1960|kjv,rvr1960`,
`--show-context`.

## Serving the web chat online

```bash
uv run bible-rag ingest                                   # or let the first start build it
uv run streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Put it behind a reverse proxy with HTTPS (nginx, Caddy...). `.streamlit/config.toml` runs it
headless with usage statistics off. Models are loaded once and shared by all visitors, and one
answer is generated at a time per model, so concurrent users queue. The app has **no
authentication**, and generation is CPU-heavy. Qwen2.5-1.5B needs ~7 GB RAM on CPU. On a
smaller host, point the chain at a model server (`BIBLE_RAG_LLM_CHAIN=openai,extractive` with
`BIBLE_RAG_OPENAI_BASE_URL`) or use `extractive`.

## Models and fallback

Each model comes from an ordered chain, and the first entry that works is used. Unreachable
Hugging Face models are detected in ~3 s, and cached ones load offline. If an LLM fails before
its first token, the next entry answers. Every answer reports which model produced it and why
earlier ones were skipped.

| Chain entry | Notes |
|---|---|
| `hf:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | preferred embedder |
| `usem3` | Universal Sentence Encoder multilingual, bundled, offline, cross-lingual |
| `hf:Qwen/Qwen2.5-1.5B-Instruct` | preferred LLM |
| `ollama[:model]` | local [Ollama](https://ollama.com) server, first installed model by default |
| `openai[:model]` | any OpenAI-compatible server: LM Studio, llama.cpp, vLLM (`BIBLE_RAG_OPENAI_BASE_URL`) |
| `smollm2` | SmolLM2-135M-Instruct, bundled via the `llm-smollm2` wheel; English only, **weak** |
| `extractive` | no LLM: quotes the best passages; always works |

* Indexes live in `storage/index/<embedder>/<version>/`, and queries use the embedder that
  built them. Build one for the preferred model once it's reachable:
  `bible-rag ingest --embedder hf:<model>`.
* SmolLM2 gets a short prompt and English passages only. Spanish questions are answered
  extractively.
* Answers whose references or quotes aren't in the retrieved passages get a warning, and the
  passages are shown.

## Quality (`bible-rag eval`)

16 questions about well-known passages, each asked in English and in Spanish, scored against
the chapters that answer them. Measured with only the bundled models available (k = 3):

| | English | Spanish |
|---|---|---|
| correct chapter retrieved, hybrid search (default) | 75 % | 75 % |
| same, embeddings only (`BIBLE_RAG_RETRIEVAL=dense`) | 62.5 % | 56.3 % |
| answer cites a correct passage, `smollm2` | 4 / 16 (9 with unverified claims) | (extractive) |
| answer cites a correct passage, `extractive` | 11 / 16 (0 unverified) | 11 / 16 |

Retrieval works offline and across languages. The 135M fallback LLM is worse than quoting the
passages: with "Make thee an ark of gopher wood" in front of it, it answered "cedar". To prefer
passages over a weak summary, drop it from the chain:
`BIBLE_RAG_LLM_CHAIN=hf:Qwen/Qwen2.5-1.5B-Instruct,ollama,openai,extractive`. Re-run
`bible-rag eval --output report.json` to compare configurations.

## How it works

```
PDF -> verses/lines with book:chapter:verse -> passages of <= 450 chars (one chapter each)
    -> per translation: FAISS embedding index + BM25 keyword index        (bible-rag ingest)
question -> both indexes of each translation, fused -> prompt with cited passages
         -> first working LLM of the chain -> answer + grounding check
```

* **Verse-accurate citations.** The KJV parser gives exact verses. RVR1960 prints verse
  numbers in the margin, so its references are accurate to the printed line. Tests check the
  canonical structure (66 books, 1,189 chapters, 31,102 KJV verses).
* **Text cleanup.** Ligatures, line-break hyphenation (keeping names like *Bet-el*), drop
  caps, headers, table of contents and indexes.
* **One index per translation**, so every answer sees both languages. **Hybrid search**,
  because keywords find names (*Goliath*) that embeddings blur.
* **Chunks of 450 characters**, matching the MiniLM embedder's 128-token limit. The Apocrypha
  (KJV only) is excluded unless `BIBLE_RAG_INCLUDE_APOCRYPHA=true`.

## Configuration

`BIBLE_RAG_<SETTING>` environment variables or a `.env` file. See `.env.example` and
`bible_rag/config.py`. Device (CPU/CUDA/MPS) is detected automatically.

## Development

```bash
uv run pytest                       # everything (~2 min)
uv run pytest -m "not pdf and not hf"   # fast unit tests (~10 s)
uv run ruff check . && uv run ruff format .
```

Test markers: `pdf` parses the real PDFs; `hf` runs the real transformers / sentence-transformers
code on tiny locally built models and the bundled SmolLM2, with no downloads.

```
app.py              Streamlit web chat
bible_rag/
  config.py         settings
  corpus.py         PDF parsing, cleanup, chunking
  models.py         embedders, chat models, fallback chains, availability probe
  index.py          FAISS + BM25 index per translation, ingestion
  rag.py            retrieval, prompts, answers with failover
  grounding.py      reference / quote verification
  evaluation.py     gold questions and metrics
  cli.py            `bible-rag` command
tests/              pytest suite
RAG0.ipynb          original prototype
```

## Notes

* The RVR1960 PDF states "Permission for personal use only" (Bibles.org.uk), and the text is
  copyrighted by the Sociedades Bíblicas en América Latina. Check the licence before serving
  it publicly. The KJV is public domain in most countries.
* A 1.5B model can still misquote. The grounding check flags this, and the sources are
  always shown. Larger models (via Ollama or `BIBLE_RAG_LLM_MODEL`) answer better.
