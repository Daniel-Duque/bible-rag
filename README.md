# Bible RAG

A Bible study assistant for the terminal and the browser. Ask a question in English or Spanish
(Portuguese, French, German and Italian work too). The answer quotes the passages it relies on
from several translations and from the Hebrew and Greek originals, and every reference and
quote the answer makes is checked against those passages.

The assistant reads each question and picks one of three ways to answer it:

| You ask | The assistant |
|---|---|
| *Where was Jesus born?* | searches directly and shows what **each translation** says at the passages found |
| *Are there any paradoxes in the Bible according to alcohol?* | searches for the **affirming and the opposing statement** ("wine is good…" / "wine is a mocker…"), sorts the passages into the two sides, shows both in every translation and weighs whether they really contradict each other |
| *How do the translations render Psalm 16:10?* | puts the same verses **side by side**, with the Hebrew or Greek, and shows where the wording changes the meaning (KJV "hell", RV1909 "sepulcro", WEB "Sheol": all the Hebrew שְׁאוֹל *she'ôl*) |

The choice is made by meaning, not by keywords. When a question could be either kind, the
assistant answers with a direct search, and either of the other two tasks is one click (or one
option) away: see [Asking for a task](#asking-for-a-task). [How it works](docs/how-it-works.md)
explains the method and its limits.

It runs on an ordinary computer: no GPU is needed, and once installed it also works offline,
with the bundled models. The search indexes ship with the code, so the first question is
answered within seconds.

## Quick start

You need [uv](https://docs.astral.sh/uv/), which installs Python and every dependency into a
local `.venv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh        # macOS / Linux
# Windows: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then, from this folder:

```bash
uv sync                                                   # once: a few GB (PyTorch)
uv run bible-rag ask "Where was Jesus born?"
uv run bible-rag ask "Are there any paradoxes in the Bible according to alcohol?"
uv run bible-rag ask "How do the translations render Psalm 16:10?"
uv run bible-rag ask "¿Hay contradicciones en la Biblia sobre el divorcio?"
```

Or start the web chat and open <http://localhost:8501>:

```bash
uv run streamlit run app.py
```

## What an answer contains

1. **How the question was understood:** the task, the topic and why it was chosen, e.g.
   `[contradiction search (both statements) · alcohol (wine, strong drink) · closest to
   contradiction questions (0.61 vs general 0.46)]`.
2. **A synthesis** written by the language model. If no capable model is available, a short
   lead sentence takes its place.
3. **The passages**, quoted from every loaded translation and aligned verse for verse. For a
   contradiction they are grouped into the two sides. Each row carries notes on the wording:
   where the translations use different words, which Hebrew or Greek word is behind them
   (Strong's numbers), and known traps such as RV1909 "alcohol", which means eye paint.
4. **Sources** with their scores, the statements that were searched for, the model that
   answered and any warnings. Anything the model cites that is not in the passages is flagged
   as unverified.

## Using it

### Terminal

| Command | What it does |
|---|---|
| `bible-rag ask "question"` | one answer (the question can also come from stdin); `--task contradiction\|translation\|general` asks for a task |
| `bible-rag chat` | a conversation; models load once. Commands: `/contradiction <question>`, `/translation <question>`, `/direct <question>`, `/search <text>`, `/k <n>`, `/sources on\|off`, `/quit` |
| `bible-rag search "text"` | retrieval only: the best passages of each translation |
| `bible-rag versions` | the Bible texts, their licenses and whether they are indexed |
| `bible-rag models` | which embedders, NLI models and language models are usable here |
| `bible-rag ingest` | updates the local indexes if a text or the code reading it changed |
| `bible-rag serve` | the read-only HTTP API (see [GPU server](#running-the-models-on-a-gpu-machine)) |
| `bible-rag eval` | measures retrieval, answer and assistant quality |

Prefix each command with `uv run`. The `ask`, `chat` and `search` commands accept
`-k <n>` (passages per translation, or per side for contradictions), `--versions kjv,rv1909`
and `--show-context` (print the passages' full text). `make` offers shortcuts: `make app`,
`make chat`, `make test`…

### Web chat

`uv run streamlit run app.py` serves the chat on <http://localhost:8501> (another port:
`--server.port 8502`). The sidebar selects the translations and the number of passages, and
lists the texts' licenses. Each answer shows how the question was understood, with its sources
in an expander. Models are loaded once and shared by every browser tab.

### Asking for a task

The assistant chooses the task, but you can always ask for one:

| | Look for contradictions | Compare translations | Direct search |
|---|---|---|---|
| Web chat (buttons under the last answer) | **⚖️ Look for contradictions** | **🌐 Compare translations** | **🔎 Direct search** |
| `bible-rag ask` | `--task contradiction` | `--task translation` | `--task general` |
| `bible-rag chat` | `/contradiction <question>` | `/translation <question>` | `/direct <question>` |

The buttons ask the last question again with the chosen task; they show the two tasks that
weren't used. When the assistant was unsure, the terminal also says how to ask for the other
tasks. References can name verses ("John 3:16", "Juan 3,16", "Psalm 16:10-11") or a whole
chapter ("Compare the translations of Psalm 23").

## Configuration

Settings are environment variables, `BIBLE_RAG_<SETTING>`, or lines in a `.env` file in this
folder. All of them are listed with their defaults in [`bible_rag/config.py`](bible_rag/config.py).
The ones you are most likely to change:

| Variable | Default | Meaning |
|---|---|---|
| `BIBLE_RAG_VERSIONS` | `kjv,rv1909,wlc,sblgnt` | texts to use (`all` = the eight public ones, see [below](#bible-texts-and-licenses)) |
| `BIBLE_RAG_TOP_K` | `3` | passages per translation |
| `BIBLE_RAG_LLM_CHAIN` | `hf:Qwen/Qwen2.5-1.5B-Instruct,ollama,openai,smollm2,extractive` | language models to try, in order |
| `BIBLE_RAG_LLM_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | the Hugging Face model at the head of the default chain |
| `BIBLE_RAG_OPENAI_BASE_URL` | | an OpenAI-compatible server (LM Studio, llama.cpp, vLLM) |
| `BIBLE_RAG_NLI_MODEL` | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` | checks opposition between passages; empty = off |
| `BIBLE_RAG_EMBEDDING_MODEL` | | a sentence-transformers model to use instead of the bundled embedder |
| `BIBLE_RAG_DEVICE` | `auto` | `cpu`, `cuda` or `mps` |
| `BIBLE_RAG_SERVER_URL`, `BIBLE_RAG_API_KEY` | | use a [GPU server](#running-the-models-on-a-gpu-machine) |

Examples:

```bash
BIBLE_RAG_VERSIONS=all uv run streamlit run app.py           # all eight texts
BIBLE_RAG_LLM_CHAIN=ollama uv run bible-rag chat             # answers by a local Ollama model
BIBLE_RAG_LLM_CHAIN=extractive uv run streamlit run app.py   # no language model: fast
```

On Windows PowerShell set a variable first: `$env:BIBLE_RAG_VERSIONS="all"`.

## Models

Each kind of model comes from a chain, tried in order. The first entry that works is used,
unreachable models are detected in about three seconds, and each answer says which model
wrote it. `bible-rag models` shows what is usable on your machine.

| Role | Default | Fallbacks |
|---|---|---|
| Search (embeddings) | `usem3`: Universal Sentence Encoder multilingual, bundled, offline | any sentence-transformers model via `BIBLE_RAG_EMBEDDING_MODEL` (its indexes are built once with `bible-rag ingest`) |
| Answers (LLM) | `hf:Qwen/Qwen2.5-1.5B-Instruct` (downloaded once, ~3 GB) | `ollama[:model]`, `openai[:model]`, `smollm2` (bundled SmolLM2-135M, English only), `extractive` (no model: the passages alone) |
| Opposition check (NLI) | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` (downloaded on first use) | meaning similarity from the embeddings |

If the language model fails before its first word, the next one in the chain answers. The
bundled SmolLM2 is small and weak. It only writes answers to direct questions in English and
isn't loaded for anything else. On a fresh install it is converted once (about a minute). To
go without it, end the chain with `extractive`:
`BIBLE_RAG_LLM_CHAIN=hf:Qwen/Qwen2.5-1.5B-Instruct,ollama,openai,extractive`.

Qwen2.5-1.5B needs about 7 GB of RAM on a CPU. A larger model served by Ollama or on a
[GPU machine](#running-the-models-on-a-gpu-machine) writes better syntheses.

## Bible texts and licenses

| id | Text | Language | License | Default |
|---|---|---|---|---|
| `kjv` | King James Version (1769), with Strong's numbers | English | Public Domain¹ | ✓ |
| `rv1909` | Reina-Valera 1909 | Spanish | Public Domain | ✓ |
| `wlc` | Westminster Leningrad Codex: the Hebrew Bible, with the Biblical Aramaic (OT) | Hebrew, Aramaic | text unrestricted; word data CC BY 4.0² | ✓ |
| `sblgnt` | SBL Greek New Testament, with MorphGNT lemmas (NT) | Greek | CC BY 4.0 / CC BY-SA³ | ✓ |
| `web` | World English Bible | English | Public Domain⁴ | |
| `ylt` | Young's Literal Translation (1898) | English | Public Domain | |
| `leeser` | Leeser, The Twenty-Four Books of the Holy Scriptures (1853, OT, from the Hebrew) | English | Public Domain | |
| `etheridge` | Etheridge, the New Testament from the Syriac Peshitta (1849) | English | Public Domain | |
| `rvr1960` | Reina-Valera 1960, read from your own PDF | Spanish | © Sociedades Bíblicas; **personal use only** | refused⁵ |

¹ A Crown patent restricts *printing* the KJV in the UK; it is free elsewhere.
² Hebrew text UXLC 1.8 (Tanach.us: "may be viewed or copied without restriction"); words,
Strong's numbers and the verse alignment © Clear Bible, Inc. (Macula Hebrew,
`bible-alignments`), CC BY 4.0.
³ SBLGNT © 2010 Society of Biblical Literature and Logos Bible Software, CC BY 4.0 as stated
in Clear Bible's catalog (check sblgnt.com before serving it publicly); lemmas by MorphGNT
(James Tauber), CC BY-SA.
⁴ "World English Bible" is a trademark of eBible.org; the text is used unchanged, as required.
⁵ Used only with `BIBLE_RAG_ALLOW_RESTRICTED=true` and the PDF at
`data/biblia-reina-valera-1960.pdf`, for private use. The web app warns when it is loaded, and
the API refuses to serve it to other machines.

**Where the texts come from.** The public-domain translations are the eBible.org SWORD
modules packaged in Debian's main archive, which only accepts freely licensed works. The
Strong's Hebrew and Greek dictionaries come from Debian too. The Hebrew and Greek texts and
the other translations come from packages on PyPI. Every file is pinned by its SHA-256 in
[`bible_rag/sources.py`](bible_rag/sources.py) and refused if it doesn't match. Only the data
files are extracted: no code from those packages is installed or run.

**Verse numbering.** All texts use the English (KJV) verse numbers. The Hebrew Bible numbers
some verses differently (Psalm titles are verses, Malachi 4 is Malachi 3:19-24), so each Hebrew
word is moved to the English verse given by Clear Bible's manual WLC–YLT word alignment.

## Search indexes

The indexes for all eight public texts are in `storage/prebuilt/` (62 MB, built with the
bundled embedder). Everything else under `storage/` is created locally and ignored by git:
downloaded texts in `sword/` and `sources/`, indexes you build in `index/`, converted models
in `models/`.

Each index records a fingerprint of what it was built from: the source file, the code that
reads it, the passage size and the selection of books. If any of these changes, the index still
works, but answers carry a note until you run `bible-rag ingest`. That command rebuilds only the
affected texts into `storage/index/`, which takes precedence over the shipped copy. Passages
whose text did not change keep their embeddings, so the rebuild takes a minute or two, not the
10-15 minutes of embedding everything from scratch.

Maintainers regenerate the shipped copy with `make prebuild` (or
`BIBLE_RAG_VERSIONS=all uv run bible-rag prebuild`) and commit `storage/prebuilt/`. The files
are byte-for-byte reproducible, so git only shows real changes, and the test
`test_shipped_prebuild_is_current` fails if the code changed without a new prebuild. See
[How it works](docs/how-it-works.md#search-indexes) for the file format.

With another embedder (`BIBLE_RAG_EMBEDDING_MODEL`), `bible-rag ingest` builds its indexes once,
with a progress bar per text. `bible-rag ingest --sample` builds about a quarter of the Bible
(enough for every example in this README) to try an embedder quickly.

## Running the models on a GPU machine

Larger models need a GPU to answer in seconds. Run the assistant there as a **read-only API**
and point the terminal and the web chat at it. When the API can't answer, they fall back to
their local models and say so.

On the GPU machine:

```bash
uv sync
export BIBLE_RAG_VERSIONS=all BIBLE_RAG_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
export BIBLE_RAG_API_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
uv run bible-rag models                        # check that the models are found
uv run bible-rag serve                         # http://127.0.0.1:8000, this machine only
```

On your computer, through an SSH tunnel:

```bash
ssh -N -L 8000:127.0.0.1:8000 you@gpu-machine  # leave it running
export BIBLE_RAG_SERVER_URL=http://127.0.0.1:8000 BIBLE_RAG_API_KEY=<the key>
uv run streamlit run app.py                    # answered by the GPU; "Answered by: … @ <server>"
```

The API only answers questions: it cannot build indexes, download or switch models or touch
files, and it refuses to listen beyond 127.0.0.1 without an API key.
[The API](docs/api.md) describes the endpoints, the security rules and how to expose it over
HTTPS.

## Serving the web chat online

```bash
uv run streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Put it behind a reverse proxy with HTTPS (Caddy, nginx). `.streamlit/config.toml` runs it
headless with usage statistics off. The app has **no authentication**. Models are shared by
all visitors and generate one answer at a time, so concurrent users queue. On a small host,
point the chain at a model server (`BIBLE_RAG_LLM_CHAIN=openai,extractive` with
`BIBLE_RAG_OPENAI_BASE_URL`) or use `extractive`. Don't load RVR1960 on a public server.

## Troubleshooting

| Symptom | What to do |
|---|---|
| "The index of … is out of date" | a text or the code that reads it changed: `uv run bible-rag ingest` (it still answers meanwhile) |
| "No usable index" | `storage/prebuilt/` is missing or the embedder has no index: `uv run bible-rag ingest` |
| the first English direct question takes about a minute | the SmolLM2 fallback is being converted, once (see [Models](#models)) |
| "Fallback LLM … in use" | the preferred model couldn't be loaded (e.g. no internet for the download); answers still work |
| "Sides estimated from meaning similarity" | the NLI model isn't available; contradiction sides are less reliable |
| slow answers or out of memory | `BIBLE_RAG_LLM_CHAIN=extractive`, or a model server (Ollama) |
| "Another process is building the indexes: waiting" | only one build runs at a time; the second waits for the first |
| `Address already in use` | `uv run streamlit run app.py --server.port 8502` |

## Development

```bash
uv run pytest                                          # everything (~4 min)
uv run pytest -m "not pdf and not hf and not sword"    # unit tests, offline (~1 min)
uv run ruff check . && uv run ruff format --check .
```

Test markers: `sword` reads the real Bible modules (downloaded once from Debian), `pdf` parses
the RVR1960 PDF (skipped without it), `hf` runs the real transformers and sentence-transformers
code on tiny local models and the bundled SmolLM2. Routing, topics and opposition are tested
with the real bundled embedder, since they depend on meaning. `bible-rag eval` measures
quality; the latest figures are in [How it works](docs/how-it-works.md#quality).

```
app.py              the web chat (Streamlit)
bible_rag/
  cli.py            the `bible-rag` command
  config.py         settings (BIBLE_RAG_* variables)
  rag.py            answering: retrieval, the model chain, grounding
  agent.py          the assistant: routing and the three pipelines, prompts, answer layout
  semantic.py       decisions by meaning: routing, topics, subject words, opposition axis
  align.py          parallel passages across translations, key clauses, wording differences
  lexicon.py        topics: modern words -> each translation's words, statements, notes
  grounding.py      reference and quote checking
  index.py          per-text search index (embeddings + BM25), fingerprints, prebuild
  corpus.py         verses, passages, chunking, the RVR1960 PDF parser
  sources.py        the texts, their licenses, downloads, SWORD / WLC / SBLGNT readers
  models.py         embedders, NLI model, chat models, fallback chains
  evaluation.py     benchmark questions and metrics
  server.py         `bible-rag serve`: the read-only API
  remote.py         the API client, with fallback to the local models
  wire.py           the API's JSON format
data/               your own files (the RVR1960 PDF)
docs/               how it works, the API
storage/prebuilt/   the shipped search indexes
tests/              pytest suite
```

## Limitations

- The side a passage takes in a contradiction is estimated, not understood. The grouping is
  labelled as automatic, and the language model is asked to point out misplaced passages.
- Topics outside the built-in lexicon (16 topics, from alcohol to lying) are much harder without
  a capable language model and the NLI model.
- The bundled embedder doesn't know Hebrew or Greek, so the originals are shown next to what is
  found rather than searched. An embedder that knows them (e.g. LaBSE) searches them too.
- Small language models can still misquote. The citation check flags it, and the passages are
  always shown.
- Answers only weigh the passages shown; they are not a theological judgement.

## License

The code is under the [MIT License](LICENSE). The Bible texts keep their own licenses (see
[above](#bible-texts-and-licenses)).
