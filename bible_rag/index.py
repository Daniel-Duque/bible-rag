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
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 64,
        local_files_only: bool = False,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(
            model_name, device=device, local_files_only=local_files_only
        )

    def embed(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return np.asarray(vectors, dtype="float32")


class USEEmbedder:
    """Google Universal Sentence Encoder multilingual v3, weights bundled in the wheel.

    Works fully offline (pure numpy), maps 16 languages incl. English and Spanish into one
    space, so it is the guaranteed fallback when the Hugging Face model cannot be loaded.
    """

    model_name = "usem3"

    def __init__(self, batch_size: int = 256):
        from usem3 import USE

        self.batch_size = batch_size
        self._model = USE()

    def embed(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        batches = range(0, len(texts), self.batch_size)
        if show_progress and len(texts) > self.batch_size:
            from tqdm import tqdm

            batches = tqdm(batches, desc="Embedding", unit="batch")
        out = [self._model.encode(texts[i : i + self.batch_size]) for i in batches]
        return np.asarray(np.vstack(out), dtype="float32")


def load_embedder(spec: str, device: str = "cpu", connect_timeout: float = 3.0) -> Embedder:
    """Build an embedder from a chain entry: ``hf:<model id or path>`` or ``usem3``."""
    from .hub import prepare_hf_model

    kind, _, arg = spec.partition(":")
    if kind == "hf" and arg:
        local_only = prepare_hf_model(arg, connect_timeout)  # fails fast when offline
        return SentenceTransformerEmbedder(arg, device=device, local_files_only=local_only)
    if kind == "usem3":
        return USEEmbedder()
    raise ValueError(f"Unknown embedding spec {spec!r} (expected 'hf:<model>' or 'usem3')")


def embedder_key(spec: str) -> str:
    """Directory name for the indexes built with this embedder."""
    kind, _, arg = spec.partition(":")
    name = arg or kind
    return "".join(c if c.isalnum() or c in "-_." else "__" for c in name)


def embedder_model_name(spec: str) -> str:
    kind, _, arg = spec.partition(":")
    return arg if kind == "hf" else kind


@dataclass
class Hit:
    passage: Passage
    score: float  # cosine similarity to the question, higher is better
    rank: float = 0.0  # fused rank score used for ordering (hybrid retrieval)


_STOPWORDS = frozenset(
    (  # noqa: SIM905 - readable as a sentence
        "a an and are as at be but by for from had has have he her his i if in into is it its "
        "me my not of on or our shall she so that the their them then there they this thou "
        "thy to unto us was we were what when which who whom why will with ye you your how "
        "did does do said say says about according "
        "al como con cual de del el ella ellos en era es esta este fue ha la las le les lo los "
        "mi no nos o para pero por que se si sin su sus te tu un una y ya yo quien donde dice "
        "segun sobre"
    ).split()
)


def lexical_tokens(text: str) -> list[str]:
    """Lowercased, accent-folded word tokens without stopwords."""
    import re
    import unicodedata

    folded = "".join(
        c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn"
    )
    return [t for t in re.findall(r"\w+", folded) if t not in _STOPWORDS and len(t) > 1]


class LexicalIndex:
    """Small in-memory BM25 index.

    Names such as "Goliath" or "Samaritan" are what sentence embeddings blur and what
    keyword search finds; fused with the embeddings it lifts retrieval (see `bible-rag eval`).
    """

    def __init__(self, texts: list[str], k1: float = 1.2, b: float = 0.75):
        import math
        from collections import Counter, defaultdict

        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths: list[int] = []
        for doc_id, text in enumerate(texts):
            tokens = lexical_tokens(text)
            self.lengths.append(len(tokens))
            for token, tf in Counter(tokens).items():
                self.postings[token].append((doc_id, tf))
        n = len(texts)
        self.avg_len = (sum(self.lengths) / n) if n else 0.0
        self.idf = {
            t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()
        }

    def search(self, query: str, n: int) -> list[tuple[int, float]]:
        from collections import defaultdict

        scores: dict[int, float] = defaultdict(float)
        for token in set(lexical_tokens(query)):
            idf = self.idf.get(token)
            if idf is None:
                continue
            for doc_id, tf in self.postings[token]:
                norm = self.k1 * (1 - self.b + self.b * self.lengths[doc_id] / self.avg_len)
                scores[doc_id] += idf * tf * (self.k1 + 1) / (tf + norm)
        return sorted(scores.items(), key=lambda x: -x[1])[:n]


class IndexNotFoundError(RuntimeError):
    pass


class VersionIndex:
    """FAISS index + passages for a single Bible version."""

    def __init__(self, version: str, index, passages: list[Passage], manifest: dict):
        self.version = version
        self.index = index
        self.passages = passages
        self.manifest = manifest
        self._lexical: LexicalIndex | None = None

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

    @property
    def lexical(self) -> LexicalIndex:
        if self._lexical is None:  # built lazily: ~1 s per version
            self._lexical = LexicalIndex([p.embedding_text() for p in self.passages])
        return self._lexical

    def search(
        self,
        query_vector: np.ndarray,
        k: int,
        query_text: str | None = None,
        candidates: int = 50,
        rrf_k: int = 60,
    ) -> list[Hit]:
        """Dense search, or hybrid dense + BM25 with reciprocal rank fusion if text is given."""
        k = min(k, len(self.passages))
        if k <= 0:
            return []
        query = query_vector.reshape(1, -1).astype("float32")
        n = min(max(candidates, k), len(self.passages))
        scores, ids = self.index.search(query, n)
        cosine = {int(i): float(s) for s, i in zip(scores[0], ids[0], strict=True) if i >= 0}
        dense = list(cosine)
        if not query_text:
            return [Hit(self.passages[i], cosine[i], cosine[i]) for i in dense[:k]]

        fused: dict[int, float] = {}
        for rank, i in enumerate(dense):
            fused[i] = fused.get(i, 0.0) + 1 / (rrf_k + rank + 1)
        for rank, (i, _) in enumerate(self.lexical.search(query_text, n)):
            fused[i] = fused.get(i, 0.0) + 1 / (rrf_k + rank + 1)
        best = sorted(fused, key=lambda i: -fused[i])[:k]
        for i in best:  # keyword-only hits: compute their cosine for display
            if i not in cosine:
                cosine[i] = float(self.index.reconstruct(i) @ query[0])
        return [Hit(self.passages[i], cosine[i], fused[i]) for i in best]
