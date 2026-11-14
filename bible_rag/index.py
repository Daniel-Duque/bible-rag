"""Embeddings + one FAISS index per Bible version.

The notebook used LangChain's FAISS wrapper, which persists through ``pickle``
(``allow_dangerous_deserialization=True``) and lives in ``langchain-community``,
a package that is being sunset. Here the same FAISS engine and embedding model
are used directly; passages are stored as plain JSON lines next to the index.

Layout of ``<index_dir>/<version>/``::

    index.faiss      inner-product index over L2-normalised embeddings (= cosine)
    passages.jsonl   one Passage per line, same order as the index rows
    manifest.json    embedding model + chunking parameters used to build it

There is one index per version on purpose: a multilingual embedding still
prefers same-language text, so a single shared index would mostly return
Spanish passages for Spanish questions. Querying each version separately
guarantees both translations are represented in the answer context.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .corpus import Passage

log = logging.getLogger(__name__)

INDEX_FORMAT = 1


class Embedder(Protocol):
    model_name: str

    def embed(self, texts: list[str], show_progress: bool = False) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return np.asarray(vectors, dtype="float32")


@dataclass
class Hit:
    passage: Passage
    score: float  # cosine similarity, higher is better


class IndexNotFoundError(RuntimeError):
    pass


class VersionIndex:
    """FAISS index + passages for a single Bible version."""

    def __init__(self, version: str, index, passages: list[Passage], manifest: dict):
        self.version = version
        self.index = index
        self.passages = passages
        self.manifest = manifest

    def __len__(self) -> int:
        return len(self.passages)

    @classmethod
    def build(
        cls, version: str, passages: list[Passage], embedder: Embedder, params: dict
    ) -> VersionIndex:
        import faiss

        if not passages:
            raise ValueError(f"{version}: no passages to index")
        t0 = time.perf_counter()
        vectors = embedder.embed([p.embedding_text() for p in passages], show_progress=True)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        manifest = {
            "format": INDEX_FORMAT,
            "version": version,
            "embedding_model": embedder.model_name,
            "dimension": int(vectors.shape[1]),
            "passages": len(passages),
            "build_seconds": round(time.perf_counter() - t0, 1),
            **params,
        }
        return cls(version, index, passages, manifest)

    def save(self, directory: Path) -> None:
        import faiss

        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        with open(directory / "passages.jsonl", "w", encoding="utf-8") as fh:
            for p in self.passages:
                fh.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
        (directory / "manifest.json").write_text(json.dumps(self.manifest, indent=2) + "\n")

    @classmethod
    def load(cls, directory: Path, expected_model: str | None = None) -> VersionIndex:
        import faiss

        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            raise IndexNotFoundError(
                f"No index at {directory}. Build it first with: bible-rag ingest"
            )
        manifest = json.loads(manifest_path.read_text())
        if expected_model and manifest.get("embedding_model") != expected_model:
            raise IndexNotFoundError(
                f"Index at {directory} was built with {manifest.get('embedding_model')!r} but "
                f"the configured embedding model is {expected_model!r}. "
                "Rebuild it with: bible-rag ingest --force"
            )
        index = faiss.read_index(str(directory / "index.faiss"))
        with open(directory / "passages.jsonl", encoding="utf-8") as fh:
            passages = [Passage(**json.loads(line)) for line in fh]
        if index.ntotal != len(passages):
            raise IndexNotFoundError(f"Corrupt index at {directory}: rows != passages")
        return cls(manifest["version"], index, passages, manifest)

    def search(self, query_vector: np.ndarray, k: int) -> list[Hit]:
        k = min(k, len(self.passages))
        if k <= 0:
            return []
        scores, ids = self.index.search(query_vector.reshape(1, -1), k)
        return [
            Hit(self.passages[i], float(s))
            for s, i in zip(scores[0], ids[0], strict=True)
            if i >= 0
        ]
