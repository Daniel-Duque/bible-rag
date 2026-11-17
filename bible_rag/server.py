"""`bible-rag serve`: the assistant as a read-only HTTP API, for a GPU machine.

Run it where the models are fast (a GPU computer); the CLI and the web app elsewhere use it
with BIBLE_RAG_SERVER_URL and fall back to their local models when it can't be reached.

What it does NOT do, by design: ingest or rebuild indexes, download or switch models, read or
write files on request, run anything a request asks. It answers questions about the indexes
it was started with, nothing else.

Safety:
  * binds to 127.0.0.1 unless told otherwise; on any other address it requires an API key
    (BIBLE_RAG_API_KEY, compared in constant time) and refuses restricted texts (RVR1960);
  * every request is validated: question <= 1000 characters, k <= 10, known versions and
    tasks, no unknown fields, bodies <= 16 KB;
  * per-client rate limit (BIBLE_RAG_SERVER_RATE per minute) and a cap on answers being
    generated at once (BIBLE_RAG_SERVER_CONCURRENCY): beyond it, 503 "busy";
  * errors are reported without internals (details go to the server log); no interactive docs
    unless --docs (local use).
Serve it behind HTTPS (a reverse proxy) or reach it through an SSH tunnel: the key must not
travel in clear text.
"""

import collections
import hmac
import ipaddress
import json
import logging
import threading
import time
from typing import Literal

from .sources import VERSIONS
from .wire import PROTOCOL, answer_to_json, hit_to_json, plan_to_json

log = logging.getLogger(__name__)

MAX_QUESTION, MAX_K, MAX_BODY = 1000, 10, 16_384


class ServerConfigError(RuntimeError):
    pass


def is_loopback(host: str) -> bool:
    if host in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def check_exposure(host: str, api_key: str, versions) -> None:
    """Refuse unsafe configurations before serving anything."""
    if is_loopback(host):
        return
    if not api_key:
        raise ServerConfigError(
            f"Serving on {host} needs an API key: set BIBLE_RAG_API_KEY (long and random, e.g. "
            '`python -c "import secrets; print(secrets.token_urlsafe(32))"`).'
        )
    if len(api_key) < 16:
        raise ServerConfigError("BIBLE_RAG_API_KEY is too short (16 characters at least).")
    restricted = [VERSIONS[v].label for v in versions if not VERSIONS[v].public]
    if restricted:
        raise ServerConfigError(
            f"{', '.join(restricted)} may not be served to others (license): remove it from "
            "BIBLE_RAG_VERSIONS, or serve on 127.0.0.1 only."
        )


class RateLimiter:
    """At most `per_minute` requests per client in any 60 s window."""

    def __init__(self, per_minute: int):
        self.per_minute, self._seen = per_minute, collections.defaultdict(collections.deque)
        self._lock = threading.Lock()

    def allow(self, client: str) -> bool:
        now = time.monotonic()
        with self._lock:
            seen = self._seen[client]
            while seen and now - seen[0] > 60:
                seen.popleft()
            if len(seen) >= self.per_minute:
                return False
            seen.append(now)
            return True


def create_app(rag, api_key: str = "", rate: int = 30, concurrency: int = 2, docs=False):
    """The FastAPI app over a loaded BibleRAG (models and indexes stay in memory)."""
    from fastapi import Depends, FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field, field_validator

    served = list(rag.indexes)
    limiter, slots = RateLimiter(rate), threading.BoundedSemaphore(concurrency)

    class Query(BaseModel):
        model_config = {"extra": "forbid"}  # unknown fields are an error, not ignored

        question: str = Field(min_length=1, max_length=MAX_QUESTION)
        k: int | None = Field(None, ge=1, le=MAX_K)
        versions: list[str] | None = Field(None, max_length=len(VERSIONS))
        # None: the assistant chooses; else the pipeline asked for
        task: Literal["general", "contradiction", "translation", "related"] | None = None

        @field_validator("versions")
        @classmethod
        def known_versions(cls, value):
            if value is not None and set(value) - set(served):
                raise ValueError(f"versions must be among {served}")
            return value

    app = FastAPI(
        title="Bible RAG",
        version=str(PROTOCOL),
        docs_url="/docs" if docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if docs else None,
    )

    @app.middleware("http")
    async def guard(request: Request, call_next):
        size = request.headers.get("content-length")
        if request.method == "POST" and size is None:  # no unbounded (chunked) uploads
            return JSONResponse({"detail": "Content-Length required"}, status_code=411)
        if size and (not size.isdigit() or int(size) > MAX_BODY):
            return JSONResponse({"detail": "request too large"}, status_code=413)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(Exception)
    async def hide_internals(request: Request, exc: Exception):
        log.exception("Request failed: %s", request.url.path)
        return JSONResponse({"detail": "internal error"}, status_code=500)

    def authorized(request: Request) -> str:
        """The client's identity for rate limiting, once its key is checked."""
        if api_key:
            sent = request.headers.get("authorization", "")
            token = sent[7:] if sent.lower().startswith("bearer ") else ""
            if not hmac.compare_digest(token.encode(), api_key.encode()):
                raise HTTPException(401, "invalid or missing API key")
        client = request.client.host if request.client else "unknown"
        if not limiter.allow(client):
            raise HTTPException(429, "too many requests, slow down")
        return client

    @app.get("/v1/health")
    def health():
        return {"status": "ok", "protocol": PROTOCOL}

    @app.get("/v1/info")
    def info(_: str = Depends(authorized)):
        chain = rag.chain.active if rag.chain else "none"
        return {
            "protocol": PROTOCOL,
            "versions": [
                {
                    "id": v,
                    "label": VERSIONS[v].label,
                    "name": VERSIONS[v].name,
                    "language": VERSIONS[v].language,
                    "license": VERSIONS[v].license,
                }
                for v in served
            ],
            "embedder": rag.embedder_spec,
            "llm": chain,
            "nli": getattr(getattr(rag, "_nli", None), "model_name", ""),  # never loads here
            "top_k": rag.settings.top_k,
            "books": rag.settings.books or "all",
        }

    @app.post("/v1/search")
    def search(query: Query, _: str = Depends(authorized)):
        hits = rag.search(query.question, query.k, query.versions)
        return {"sources": [hit_to_json(h) for h in hits]}

    @app.post("/v1/ask")
    def ask(query: Query, _: str = Depends(authorized)):
        if rag.chain is None:
            raise HTTPException(503, "this server has no language model loaded")
        if not slots.acquire(blocking=False):
            raise HTTPException(503, "busy: too many answers being generated, retry shortly")
        try:
            answer = rag.ask_stream(query.question, query.k, query.versions, query.task)
        except ValueError as e:  # e.g. an empty question after stripping
            slots.release()
            raise HTTPException(422, str(e)) from e
        except Exception:
            slots.release()
            raise

        def stream():
            try:
                yield json.dumps({"type": "plan", "plan": plan_to_json(answer.plan)}) + "\n"
                for token in answer:
                    yield json.dumps({"type": "token", "text": token}) + "\n"
                yield json.dumps({"type": "done", "answer": answer_to_json(answer)}) + "\n"
            except Exception:  # noqa: BLE001 - the client sees a clean error line
                log.exception("Answer failed")
                yield json.dumps({"type": "error", "detail": "the answer failed"}) + "\n"
            finally:
                slots.release()

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    return app


def serve(settings, host: str, port: int, docs: bool = False, ssl_certfile=None, ssl_keyfile=None):
    """Load the models and indexes (never builds them) and serve until interrupted."""
    import uvicorn

    from .rag import BibleRAG

    check_exposure(host, settings.api_key, settings.versions)
    # the indexes are the shipped ones, or built beforehand (`bible-rag ingest`) for another
    # embedder; the LLM is loaded now, not on a request
    rag = BibleRAG.from_settings(settings, eager=True)
    # everything is loaded now: no request ever triggers a download
    if rag.nli is None and settings.nli_model:
        log.warning("NLI model unavailable (%s): opposition from embeddings", rag.nli_skipped)
    rag.warm_up()  # examples embedded before the first question
    app = create_app(rag, settings.api_key, settings.server_rate, settings.server_concurrency, docs)
    log.warning(
        "Serving %s on %s:%d (embedder %s, LLM %s)%s",
        ",".join(rag.indexes),
        host,
        port,
        rag.embedder_spec,
        rag.chain.active if rag.chain else "none",
        "" if settings.api_key else " without an API key (local only)",
    )
    uvicorn.run(
        app,
        host=host,
        port=port,
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
        log_level="warning",
        server_header=False,
    )
