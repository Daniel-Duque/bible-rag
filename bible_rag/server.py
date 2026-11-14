"""HTTP service: keeps the embedding model, FAISS indexes and LLM loaded in memory.

Start it with ``bible-rag serve`` and query it from any terminal with
``bible-rag ask --server http://127.0.0.1:8000 "..."`` or plain curl.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from .rag import BibleRAG, hit_to_dict

log = logging.getLogger(__name__)


class Query(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    k: int | None = Field(None, ge=1, le=20, description="Passages per version")
    versions: list[str] | None = Field(None, description="Subset of: kjv, rvr1960")


def create_app(rag: BibleRAG) -> FastAPI:
    app = FastAPI(title="Bible RAG (KJV / RVR1960)", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", **rag.info()}

    @app.post("/search")
    def search(q: Query) -> dict:
        try:
            hits = rag.search(q.question, q.k, q.versions)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return {"question": q.question, "sources": [hit_to_dict(h) for h in hits]}

    @app.get("/models")
    def models() -> list[dict]:
        """Availability of every embedder / LLM in the fallback chains."""
        from .probe import as_dicts, probe_all

        return as_dicts(probe_all(rag.settings))

    @app.post("/ask")
    async def ask(q: Query) -> dict:
        if rag.chain is None:
            raise HTTPException(503, "Service started without an LLM (--no-llm); use /search")
        try:
            answer = await run_in_threadpool(rag.ask, q.question, q.k, q.versions)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return answer.to_dict()

    return app
