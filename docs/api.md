# The API (`bible-rag serve`)

Larger models need a GPU to answer in seconds. `bible-rag serve` runs the assistant on such a
machine as a **read-only HTTP API**. The terminal and the web chat on other computers use it
when `BIBLE_RAG_SERVER_URL` is set, and fall back to their own local models whenever it can't
answer.

## Setting it up

On the GPU machine (Linux; `uv sync` installs PyTorch with CUDA):

```bash
uv sync
export BIBLE_RAG_VERSIONS=all                       # the texts it will serve
export BIBLE_RAG_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct  # optional: a larger model
export BIBLE_RAG_API_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
echo "$BIBLE_RAG_API_KEY"                            # give it to the clients, keep it secret
uv run bible-rag models                              # check that the models are found
uv run bible-rag serve                               # http://127.0.0.1:8000 (this machine only)
```

The server loads every model at start-up, so no request ever triggers a download. It uses the
shipped indexes; with another embedder, build its indexes first with `bible-rag ingest`.

On a client, the safest link is an SSH tunnel, which needs nothing else opened on the server:

```bash
ssh -N -L 8000:127.0.0.1:8000 you@gpu-machine      # leave it running
export BIBLE_RAG_SERVER_URL=http://127.0.0.1:8000 BIBLE_RAG_API_KEY=<the key>
uv run bible-rag ask "Are there any paradoxes in the Bible according to alcohol?"
uv run streamlit run app.py
```

Answers then end with "Answered by: … @ 127.0.0.1". To open the API to a network instead,
serve it behind an HTTPS reverse proxy (Caddy, nginx) with
`uv run bible-rag serve --host 0.0.0.0 --port 8000` and set
`BIBLE_RAG_SERVER_URL=https://…` on the clients, or serve HTTPS directly with
`--ssl-certfile` and `--ssl-keyfile`.

Models worth trying on a GPU (not measured here; check with `bible-rag models` and
`bible-rag eval --agent`):

```bash
export BIBLE_RAG_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct             # better syntheses
export BIBLE_RAG_NLI_MODEL=MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
export BIBLE_RAG_EMBEDDING_MODEL=sentence-transformers/LaBSE    # knows Hebrew and Greek,
uv run bible-rag ingest                                         # so the originals are searched
```

## Endpoints

All bodies are JSON. Every endpoint except `/v1/health` needs the header
`Authorization: Bearer <key>` when the server has a key.

| Endpoint | Body | Reply |
|---|---|---|
| `GET /v1/health` | | `{"status": "ok", "protocol": 4}` |
| `GET /v1/info` | | the texts served, the embedder, the LLM, the NLI model, `top_k` |
| `POST /v1/search` | `{"question": "...", "k": 3, "versions": ["kjv"]}` | `{"sources": [passage, ...]}` |
| `POST /v1/ask` | `{"question": "...", "k": 3, "versions": ["kjv", "rv1909"], "task": "translation"}` | the answer, streamed as JSON lines |

`k`, `versions` and `task` are optional. Without `task` the assistant chooses the pipeline, as
in the terminal and the web chat; `"general"`, `"contradiction"`, `"translation"` or `"related"`
(the concepts related to a subject the Bible never names) asks for one. `/v1/ask` streams one
JSON object per line:

```
{"type": "plan", "plan": {"intent": "contradiction", "topic_name": "...", "reason": "...", "queries": [...], ...}}
{"type": "token", "text": "..."}                ... one per generated piece
{"type": "done", "answer": {"text": "...", "model": "...", "notes": [...], "grounding": {...}, "sources": [...]}}
```

or `{"type": "error", "detail": "the answer failed"}` if it fails midway. `protocol` changes
when the format changes incompatibly; a client refuses a server speaking another version.

## Security

- **Read-only by design.** No endpoint builds indexes, downloads or switches models, or reads
  or writes files. It answers questions about the indexes it was started with.
- **Local unless told otherwise.** It binds to 127.0.0.1. On any other address it refuses to
  start without an API key of at least 16 characters (compared in constant time), and refuses
  texts whose license doesn't allow serving them to others (RVR1960).
- **Validated requests.** Questions up to 1,000 characters, `k` up to 10, known texts and tasks
  only, no unknown fields, bodies up to 16 KB with a declared length.
- **Limits.** Requests per client per minute (`BIBLE_RAG_SERVER_RATE`, default 30) and answers
  generated at once (`BIBLE_RAG_SERVER_CONCURRENCY`, default 2; beyond it: 503 "busy").
- **No internals in errors.** Details go to the server log. The interactive docs (`/docs`) are
  off unless started with `--docs`, for local use.
- **The client protects the key.** It only sends it over HTTPS, or plain HTTP to 127.0.0.1
  (a tunnel), and follows no redirects. `BIBLE_RAG_SERVER_INSECURE=true` overrides the HTTPS
  rule, e.g. on a trusted private network.

## Fallback

If the server is unreachable, refuses the key, speaks another protocol, is busy or fails before
the first word of an answer, the client answers with its local models and adds a note saying
why. An invalid question (422) is reported as an error instead.
