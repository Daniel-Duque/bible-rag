"""Per-version search index: FAISS (cosine) + BM25 keywords, fused by reciprocal rank.

Stored in <index_dir>/<embedder>/<version>/ as index.faiss + passages.jsonl + manifest.json
(plain JSON, no pickle). One index per version so every answer sees both translations
(a multilingual embedder still prefers same-language text).
"""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .config import Settings
from .corpus import Passage, chunk_units
from .models import describe_error, embedder_key, load_embedder
from .sources import VERSIONS, check_licenses, load_units

log = logging.getLogger(__name__)


def _faiss():
    import faiss

    # single-threaded search takes ~20 ms on these indexes, and FAISS's OpenMP pool otherwise
    # competes with PyTorch's: LLM generation in the same process was up to ~2x slower
    faiss.omp_set_num_threads(1)
    return faiss


class IndexNotFoundError(RuntimeError):
    pass


def index_is_current(directory: Path, version: str) -> bool:
    """An index exists and was built from the version's current source."""
    manifest = directory / "manifest.json"
    return (
        manifest.exists()
        and json.loads(manifest.read_text()).get("source") == VERSIONS[version].source
    )


@dataclass
class Hit:
    passage: Passage
    score: float  # cosine similarity (shown to users)
    rank: float = 0.0  # fused score (used for ordering)


# fmt: off
_STOP = frozenset((
    "a an and are as at be but by for from had has have he her his i if in into is it its me my "
    "not of on or our shall she so that the their them then there they this thou thy to unto us "
    "was we were what when which who whom why will with ye you your how did does do said say "
    "says about according al como con cual de del el ella ellos en era es esta este fue ha la "
    "las le les lo los mi no nos o para pero por que se si sin su sus te tu un una y ya yo quien "
    "donde dice segun sobre"
).split())
# fmt: on


def tokens(text: str) -> list[str]:
    """Lowercase, accent-free keywords (æ -> ae: the KJV writes Cæsar, Judæa)."""
    text = text.replace("æ", "ae").replace("Æ", "Ae").replace("œ", "oe")
    folded = "".join(
        c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn"
    )
    return [t for t in re.findall(r"\w+", folded) if len(t) > 1 and t not in _STOP]


class BM25:
    """In-memory keyword index: finds names (Goliath, Samaritan) that embeddings blur."""

    def __init__(self, texts: list[str], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths = []
        for doc, text in enumerate(texts):
            words = tokens(text)
            self.lengths.append(len(words))
            for word, tf in Counter(words).items():
                self.postings[word].append((doc, tf))
        n = len(texts)
        self.avg = sum(self.lengths) / max(n, 1)
        self.idf = {
            w: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for w, p in self.postings.items()
        }

    def search(self, query: str, n: int) -> list[int]:
        scores: dict[int, float] = defaultdict(float)
        for word in set(tokens(query)) & self.idf.keys():
            for doc, tf in self.postings[word]:
                norm = self.k1 * (1 - self.b + self.b * self.lengths[doc] / self.avg)
                scores[doc] += self.idf[word] * tf * (self.k1 + 1) / (tf + norm)
        return sorted(scores, key=scores.get, reverse=True)[:n]


class VersionIndex:
    def __init__(self, index, passages: list[Passage], manifest: dict):
        self.index, self.passages, self.manifest = index, passages, manifest
        self._bm25: BM25 | None = None

    def __len__(self) -> int:
        return len(self.passages)

    @classmethod
    def build(cls, passages: list[Passage], embedder, manifest: dict) -> VersionIndex:
        faiss = _faiss()

        vectors = embedder.embed([p.embedding_text() for p in passages], show_progress=True)
        index = faiss.IndexFlatIP(vectors.shape[1])  # inner product of unit vectors = cosine
        index.add(vectors)
        return cls(
            index,
            passages,
            {**manifest, "embedding_model": embedder.model_name, "passages": len(passages)},
        )

    def save(self, directory: Path) -> None:
        faiss = _faiss()

        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        lines = (json.dumps(asdict(p), ensure_ascii=False) for p in self.passages)
        (directory / "passages.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (directory / "manifest.json").write_text(json.dumps(self.manifest, indent=2))

    @classmethod
    def load(cls, directory: Path, expected_model: str | None = None) -> VersionIndex:
        faiss = _faiss()

        if not (directory / "manifest.json").exists():
            raise IndexNotFoundError(f"No index at {directory}. Build it with: bible-rag ingest")
        manifest = json.loads((directory / "manifest.json").read_text())
        if expected_model and manifest["embedding_model"] != expected_model:
            raise IndexNotFoundError(
                f"{directory} was built with {manifest['embedding_model']}, "
                f"not {expected_model}: run bible-rag ingest --force"
            )
        with open(directory / "passages.jsonl", encoding="utf-8") as fh:
            passages = [Passage(**json.loads(line)) for line in fh]
        return cls(faiss.read_index(str(directory / "index.faiss")), passages, manifest)

    def search(
        self,
        vector: np.ndarray,
        k: int,
        text: str | None = None,
        candidates: int = 50,
        rrf: int = 60,
    ) -> list[Hit]:
        """Top-k by cosine, or fused with BM25 (reciprocal rank fusion) when text is given."""
        query = vector.reshape(1, -1).astype("float32")
        scores, ids = self.index.search(query, min(max(candidates, k), len(self)))
        cosine = {int(i): float(s) for s, i in zip(scores[0], ids[0], strict=True) if i >= 0}
        if not text:
            return [Hit(self.passages[i], s, s) for i, s in list(cosine.items())[:k]]
        fused: dict[int, float] = defaultdict(float)
        for ranking in (list(cosine), self.bm25.search(text, candidates)):
            for rank, i in enumerate(ranking):
                fused[i] += 1 / (rrf + rank + 1)
        best = sorted(fused, key=fused.get, reverse=True)[:k]
        for i in best:  # keyword-only hits: compute their cosine for display
            cosine.setdefault(i, float(self.index.reconstruct(i) @ query[0]))
        return [Hit(self.passages[i], cosine[i], fused[i]) for i in best]

    @property
    def bm25(self) -> BM25:
        if self._bm25 is None:  # built on first use, ~1 s
            self._bm25 = BM25([p.embedding_text() for p in self.passages])
        return self._bm25


def build_indexes(
    settings: Settings, force: bool = False, spec: str | None = None, embedder=None
) -> dict[str, VersionIndex]:
    """Index every version with the first embedder of the chain that loads (or `spec`)."""
    check_licenses(settings.versions, settings.allow_restricted)
    if embedder is None:
        errors = []
        for candidate in [spec] if spec else settings.embedding_specs:
            try:
                device = settings.resolve_device()
                embedder = load_embedder(candidate, device, settings.connect_timeout)
                spec = candidate
                break
            except Exception as e:  # noqa: BLE001
                log.warning("Embedder %s unavailable: %s", candidate, describe_error(e))
                errors.append(f"{candidate}: {describe_error(e)}")
        else:
            raise RuntimeError("No embedder could be loaded: " + "; ".join(errors))
    spec = spec or embedder.model_name
    root, built = settings.index_dir / embedder_key(spec), {}
    for version in settings.versions:
        if index_is_current(root / version, version) and not force:
            continue
        log.info("%s: parsing and embedding with %s...", version, spec)
        v = VERSIONS[version]
        units = load_units(version, settings)
        passages = list(
            chunk_units(units, v.label, v.language, settings.chunk_size, settings.chunk_overlap)
        )
        index = VersionIndex.build(
            passages, embedder, {"embedder": spec, "source": v.source, "license": v.license}
        )
        tmp = root / f".{version}.tmp"  # write aside, then swap: never a half-written index
        shutil.rmtree(tmp, ignore_errors=True)
        index.save(tmp)
        shutil.rmtree(root / version, ignore_errors=True)
        tmp.rename(root / version)
        built[version] = index
    return built


def open_indexes(settings: Settings):
    """(embedder, {version: index}, spec, skipped) for the first embedder that has an index
    for every version and loads."""
    check_licenses(settings.versions, settings.allow_restricted)
    skipped = []
    for spec in settings.embedding_specs:
        dirs = {v: settings.index_dir / embedder_key(spec) / v for v in settings.versions}
        if not all(index_is_current(d, v) for v, d in dirs.items()):
            skipped.append((spec, "no current index (run: bible-rag ingest)"))
            continue
        try:
            embedder = load_embedder(spec, settings.resolve_device(), settings.connect_timeout)
        except Exception as e:  # noqa: BLE001
            skipped.append((spec, describe_error(e)))
            continue
        indexes = {v: VersionIndex.load(d, embedder.model_name) for v, d in dirs.items()}
        return embedder, indexes, spec, skipped
    raise IndexNotFoundError(
        "No usable index ("
        + "; ".join(f"{s}: {r}" for s, r in skipped)
        + "). Build one with: bible-rag ingest"
    )
