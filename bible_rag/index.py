"""Per-version search index: embeddings (cosine) + BM25 keywords, fused by reciprocal rank.

One index per version (a multilingual embedder still prefers same-language text), in
<dir>/<embedder>/<version>/, built by `bible-rag ingest` into storage/index/, or shipped
prebuilt in storage/prebuilt/ (`bible-rag prebuild`), so that a fresh copy answers at once.

Files (schema 3, no pickle): manifest.json; passages.jsonl.gz; vectors.npz (int8 codes + one
scale per vector: 1/4 of float32, same rankings; plus a hash of each passage's text);
bm25.npz (the keyword index as arrays: loads instantly); tags.json.gz (original words);
vocabulary.npz (the version's words, embedded).

Updating: each index records a fingerprint of what it was built from: the source file's
pinned hash, the code that turns it into passages (its syntax tree: comments and docstrings
don't count), the chunking settings and the books. When it changes, `ingest` rebuilds that
version, and every passage whose text didn't change keeps its embedding: only new or changed
passages are embedded. An out-of-date index still opens, with a note.
"""

from __future__ import annotations

import ast
import contextlib
import functools
import gzip
import hashlib
import json
import logging
import os
import re
import shutil
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .config import Settings
from .corpus import Passage, chunk_units, embeddable
from .models import describe_error, embedder_key, load_embedder
from .sources import PACKAGES, PYPI_PACKAGES, TAGGED, VERSIONS, check_licenses, load_units

log = logging.getLogger(__name__)


class IndexNotFoundError(RuntimeError):
    pass


SCHEMA = 3  # 3: compact files (int8 vectors, gzip, stored BM25), fingerprints


# ---------------------------------------------------------------------------- fingerprints
# the code that decides what an index contains: module -> top-level names (None: all of it)
_CODE = {
    "corpus.py": None,
    "sources.py": None,
    "index.py": ("tokens", "_STOP", "_MARKS", "BM25", "SCHEMA", "text_hashes"),
}


@functools.cache
def code_fingerprint() -> str:
    """Hash of the syntax tree of the code that builds an index, docstrings left out: editing
    comments, docs or unrelated functions doesn't make indexes out of date."""
    digest = hashlib.sha256()
    for module, wanted in _CODE.items():
        source = (Path(__file__).parent / module).read_text(encoding="utf-8")
        digest.update(code_digest(source, wanted).encode())
    return digest.hexdigest()[:16]


def code_digest(source: str, wanted=None) -> str:
    """Digest of the top-level definitions named in `wanted` (None: all), without docstrings
    and comments (which the syntax tree doesn't keep)."""
    digest = hashlib.sha256()
    for node in ast.parse(source).body:
        if wanted is not None and not _defined(node) & set(wanted):
            continue
        for inner in ast.walk(node):
            body = getattr(inner, "body", None)
            if (
                isinstance(body, list)
                and body
                and isinstance(body[0], ast.Expr)
                and isinstance(getattr(body[0], "value", None), ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                inner.body = body[1:] or [ast.Pass()]
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue  # the module docstring
        digest.update(ast.dump(node).encode())
    return digest.hexdigest()


def _defined(node) -> set[str]:
    if isinstance(node, ast.FunctionDef | ast.ClassDef):
        return {node.name}
    if isinstance(node, ast.Assign):
        return {t.id for t in node.targets if isinstance(t, ast.Name)}
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return {node.target.id}
    return set()


def source_pin(version: str, settings: Settings) -> str:
    """The exact source file: its pinned hash (the PDF: hashed)."""
    kind, _, location = VERSIONS[version].source.partition(":")
    if kind == "sword":
        return PACKAGES[location][1]
    if kind == "pypi":
        pin = PYPI_PACKAGES[location][1]
        if location == "sblgnt":  # its Strong's numbers come from the Greek dictionary
            pin += PACKAGES["strongsgreek"][1]
        return pin
    pdf = settings.data_dir / location
    return hashlib.sha256(pdf.read_bytes()).hexdigest() if pdf.exists() else "missing"


def fingerprint(version: str, settings: Settings) -> str:
    parts = (
        str(SCHEMA),
        version,
        VERSIONS[version].source,
        source_pin(version, settings),
        code_fingerprint(),
        f"{settings.chunk_size}/{settings.chunk_overlap}",
        settings.books or "all",
    )
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def index_root(settings: Settings, spec: str, prebuilt: bool = False) -> Path:
    """Where the indexes built with an embedder live (the local build or the shipped one); a
    subset of books (BIBLE_RAG_BOOKS, e.g. the quick `sample`) gets its own folder."""
    base = settings.prebuilt_dir if prebuilt else settings.index_dir
    return base / (embedder_key(spec) + settings.corpus_suffix)


def read_manifest(directory: Path, version: str) -> dict | None:
    """The manifest of a usable index of this version (current format), else None."""
    path = directory / "manifest.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if data.get("schema") != SCHEMA or data.get("source") != VERSIONS[version].source:
        return None
    return data


def index_is_current(directory: Path, version: str, settings: Settings | None = None) -> bool:
    """A usable index of this version (and, with settings, built from what it is now)."""
    data = read_manifest(directory, version)
    if data is None:
        return False
    return settings is None or data.get("fingerprint") == fingerprint(version, settings)


def find_index(settings: Settings, spec: str, version: str) -> tuple[Path, bool] | None:
    """(directory, up to date) of the best index of a version: an up-to-date one (the local
    build first, then the shipped one), else any usable one, else None."""
    dirs = [index_root(settings, spec, p) / version for p in (False, True)]
    found = [d for d in dirs if read_manifest(d, version)]
    if not found:
        return None
    current = fingerprint(version, settings)
    for d in found:
        if read_manifest(d, version).get("fingerprint") == current:
            return d, True
    return found[0], False


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


# combining marks (accents; Hebrew points and cantillation; Arabic vowels), removed after NFD
# by one regex: a per-character check in Python made indexing ~10x slower
_MARKS = re.compile(
    "[\u0300-\u036f\u0483-\u0489\u0591-\u05bd\u05bf\u05c1\u05c2\u05c4\u05c5\u05c7"
    "\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06dc\u06df-\u06e4\u06e7\u06e8\u06ea-\u06ed]"
)


def tokens(text: str) -> list[str]:
    """Lowercase, accent-free keywords (æ -> ae: the KJV writes Cæsar, Judæa)."""
    text = text.replace("æ", "ae").replace("Æ", "Ae").replace("œ", "oe")
    folded = _MARKS.sub("", unicodedata.normalize("NFD", text.lower()))
    return [t for t in re.findall(r"\w+", folded) if len(t) > 1 and t not in _STOP]


def save_npz(path: Path, compress: bool = False, **arrays) -> None:
    """numpy's .npz, byte-for-byte reproducible (fixed timestamps): rebuilding unchanged data
    changes no shipped file."""
    import zipfile

    method = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w", compression=method) as z:
        for name, array in arrays.items():
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = method
            with z.open(info, "w") as fh:
                np.lib.format.write_array(fh, np.asanyarray(array), allow_pickle=False)


def write_gzip(path: Path, text: str) -> None:
    """gzip without a timestamp nor a file name in its header (reproducible)."""
    with (
        open(path, "wb") as raw,
        gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as fh,
    ):
        fh.write(text.encode("utf-8"))


def text_hashes(texts: list[str]) -> np.ndarray:
    """64-bit hash of each text: an embedding is reused when its text is unchanged."""
    return np.array(
        [
            int.from_bytes(hashlib.blake2b(t.encode(), digest_size=8).digest(), "little")
            for t in texts
        ],
        dtype=np.uint64,
    )


def quantize(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """int8 codes and one scale per vector (measured: same top result for every evaluation
    query, 99 % of the top 10 identical, cosine error < 0.002)."""
    scales = np.abs(vectors).max(axis=1) / 127
    scales[scales == 0] = 1
    return np.round(vectors / scales[:, None]).astype(np.int8), scales.astype(np.float32)


def dequantize(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    vectors = codes.astype(np.float32) * scales[:, None]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)  # unit again (zero rows stay zero)


class BM25:
    """Keyword index: finds names (Goliath, Samaritan) that embeddings blur. Postings are
    arrays (CSR): scoring is vectorised, and the index is saved and loaded as is."""

    def __init__(self, texts: list[str] | None = None, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        if texts is None:
            return
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths = []
        for doc, text in enumerate(texts):
            words = tokens(text)
            lengths.append(len(words))
            for word, tf in Counter(words).items():
                postings[word].append((doc, tf))
        self.terms = sorted(postings)
        sizes = [len(postings[t]) for t in self.terms]
        self.indptr = np.zeros(len(self.terms) + 1, np.int64)
        self.indptr[1:] = np.cumsum(sizes)
        flat = [pair for t in self.terms for pair in postings[t]]
        self.docs = np.array([d for d, _ in flat], np.int32)
        self.tfs = np.array([tf for _, tf in flat], np.uint16)
        self.lengths = np.array(lengths, np.float32)
        self._finish()

    def _finish(self) -> None:
        self.position = {t: i for i, t in enumerate(self.terms)}
        n, df = len(self.lengths), np.diff(self.indptr)
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
        self.avg = float(self.lengths.mean()) if n else 1.0

    def docs_of(self, word: str) -> np.ndarray:
        i = self.position.get(word)
        return self.docs[self.indptr[i] : self.indptr[i + 1]] if i is not None else self.docs[:0]

    def frequent(self, min_docs: int) -> list[str]:
        """Words in at least min_docs passages."""
        df = np.diff(self.indptr)
        return [t for t, n in zip(self.terms, df, strict=True) if n >= min_docs]

    def search(self, query: str, n: int) -> list[int]:
        scores = np.zeros(len(self.lengths))
        for word in set(tokens(query)):
            i = self.position.get(word)
            if i is None:
                continue
            span = slice(self.indptr[i], self.indptr[i + 1])
            docs, tf = self.docs[span], self.tfs[span].astype(np.float64)
            norm = self.k1 * (1 - self.b + self.b * self.lengths[docs] / self.avg)
            scores[docs] += self.idf[i] * tf * (self.k1 + 1) / (tf + norm)
        found = np.flatnonzero(scores)
        return found[np.argsort(-scores[found], kind="stable")][:n].tolist()

    def save(self, path: Path) -> None:
        terms = np.frombuffer("\n".join(self.terms).encode(), np.uint8)
        save_npz(
            path,
            True,
            terms=terms,
            indptr=self.indptr,
            docs=self.docs,
            tfs=self.tfs,
            lengths=self.lengths,
        )

    @classmethod
    def load(cls, path: Path) -> BM25:
        data, bm25 = np.load(path), cls()
        raw = data["terms"].tobytes().decode()
        bm25.terms = raw.split("\n") if raw else []
        bm25.indptr, bm25.docs, bm25.tfs = data["indptr"], data["docs"], data["tfs"]
        bm25.lengths = data["lengths"]
        bm25._finish()
        return bm25


class VersionIndex:
    def __init__(
        self,
        vectors: np.ndarray,
        passages: list[Passage],
        manifest: dict,
        hashes: np.ndarray | None = None,
        codes: tuple[np.ndarray, np.ndarray] | None = None,
    ):
        self._vectors, self.passages, self.manifest = vectors, passages, manifest
        # the stored int8 form (kept as is: reusing it never re-quantises, so rebuilding
        # unchanged data writes identical files)
        self._codes = codes
        self.hashes = (
            hashes if hashes is not None else text_hashes([p.embedding_text() for p in passages])
        )
        self._bm25: BM25 | None = None
        self._folded: list[str] | None = None
        self._places: dict | None = None
        self._vocabulary: tuple[list[str], np.ndarray] | None = None
        self._tags: dict | None = None
        self.directory: Path | None = None  # where it was loaded from / saved to
        self.stale = False  # built from an older source or code (still usable)
        self.writable = True  # False: the shipped prebuild (nothing is written into it)
        self.reused = 0  # embeddings taken from an earlier build (when built)

    def __len__(self) -> int:
        return len(self.passages)

    # -------------------------------------------------- helpers for comparing translations
    @property
    def folded(self) -> list[str]:
        """Passage texts, lowercase and accent-free (for lexicon term matching)."""
        if self._folded is None:
            from .lexicon import fold

            self._folded = [fold(p.text) for p in self.passages]
        return self._folded

    @property
    def places(self) -> dict[tuple[int, int], list[int]]:
        """(canonical book index, chapter) -> passage ids, whatever the book's name."""
        if self._places is None:
            from .grounding import book_index

            books, places = {}, defaultdict(list)
            for i, p in enumerate(self.passages):
                if p.book not in books:
                    books[p.book] = book_index(p.book)
                places[(books[p.book], p.chapter)].append(i)
            self._places = dict(places)
        return self._places

    def at(
        self, book: int, chapter: int, first: int, last: int, vector: np.ndarray | None = None
    ) -> int | None:
        """The passage id that best covers verses first-last (parallel-passage lookup). Chunks
        are cut differently in each version, so several may overlap: with `vector`, the one
        closest to it (about what matters), else the one sharing the most verses."""

        def shared(i):
            p = self.passages[i]
            return min(last, p.verse_end) - max(first, p.verse_start) + 1

        ids = [i for i in self.places.get((book, chapter), []) if shared(i) > 0]
        if not ids:
            return None
        if vector is not None:
            return max(ids, key=lambda i: (float(self.vectors[i] @ vector), shared(i)))
        return max(ids, key=lambda i: (shared(i), -self.passages[i].verse_end))

    @property
    def vectors(self) -> np.ndarray:
        """Unit vectors, one per passage (zero for a language the embedder doesn't know)."""
        return self._vectors

    @property
    def embedded(self) -> bool:
        return self.manifest.get("embedded", True)

    def vocabulary(self, embedder, reuse: dict | None = None) -> tuple[list[str], np.ndarray]:
        """The words of this translation (in >= 2 passages) and their embeddings: lets the
        assistant find the words a translation uses for a subject by meaning ("alcohol" ->
        KJV "wine", "drunkenness"; RV1909 "vino", "embriaguez"). Computed once (`reuse`:
        word -> vector from an earlier build) and kept next to the index."""
        if self._vocabulary is None:
            path = self.directory / "vocabulary.npz" if self.directory else None
            if path and path.exists():
                data = np.load(path)
                if str(data["model"]) == embedder.model_name:
                    words = data["words"].tobytes().decode().split("\n")
                    self._vocabulary = (words, dequantize(data["codes"], data["scales"]))
            if self._vocabulary is None:
                words = self.bm25.frequent(2)
                reuse = reuse or {}  # word -> (int8 codes, scale): reused as stored
                missing = [w for w in words if w not in reuse]
                fresh = {}
                if missing:
                    codes, scales = quantize(embedder.embed(missing))
                    fresh = dict(zip(missing, zip(codes, scales, strict=True), strict=True))
                rows = [fresh[w] if w in fresh else reuse[w] for w in words]
                dim = len(rows[0][0]) if rows else 1
                codes = np.array([c for c, _ in rows], np.int8).reshape(len(rows), dim)
                scales = np.array([sc for _, sc in rows], np.float32)
                self._vocabulary = (words, dequantize(codes, scales))
                if path and self.writable:
                    self._save_vocabulary(path, embedder.model_name, codes, scales)
        return self._vocabulary

    def _save_vocabulary(self, path: Path, model: str, codes, scales) -> None:
        words, _ = self._vocabulary
        try:
            save_npz(
                path,
                True,
                words=np.frombuffer("\n".join(words).encode(), np.uint8),
                codes=codes,
                scales=scales,
                model=np.array(model),
            )
        except OSError as e:  # read-only storage (the shipped prebuild): keep it in memory
            log.debug("Could not save %s: %s", path, e)

    @property
    def tags(self) -> dict[str, list]:
        """Original-language words per verse, "book:chapter:verse" -> [[words, [Strong's]]]
        (KJV: the Strong's numbers of its words; WLC / SBLGNT: of the original words)."""
        if self._tags is None:
            path = self.directory / "tags.json.gz" if self.directory else None
            self._tags = {}
            if path is not None and path.exists():
                with gzip.open(path, "rt", encoding="utf-8") as fh:
                    self._tags = json.load(fh)
        return self._tags

    def verse_tags(self, book: int, chapter: int, first: int, last: int) -> list:
        return [
            t for v in range(first, last + 1) for t in self.tags.get(f"{book}:{chapter}:{v}", [])
        ]

    def passages_with(self, word: str, limit: int = 60) -> list[int]:
        """Ids of passages containing a (folded) word."""
        return self.bm25.docs_of(word)[:limit].tolist()

    @classmethod
    def build(
        cls,
        passages: list[Passage],
        embedder,
        manifest: dict,
        progress=None,
        step: int = 512,
        embed: bool = True,
        reuse: dict | None = None,
    ) -> VersionIndex:
        """Embed the passages and index them. `reuse` (text hash -> vector, from an earlier
        build) spares re-embedding unchanged passages; `progress(done, total)` reports.
        embed=False (a language the embedder doesn't know): zero vectors, nothing computed;
        the passages are then found by keyword or as parallels of other versions."""
        texts = [p.embedding_text() for p in passages]
        hashes = text_hashes(texts)
        reuse = reuse or {}
        meta = {"embedding_model": embedder.model_name, "passages": len(passages), "schema": SCHEMA}
        if not embed:
            dim = next((len(c) for c, _ in reuse.values()), None) or embedder.embed(["x"]).shape[1]
            if progress:
                progress(len(texts), len(texts))
            manifest = {**manifest, **meta, "embedded": False}
            return cls(np.zeros((len(texts), dim), np.float32), passages, manifest, hashes)
        missing = [i for i, h in enumerate(hashes.tolist()) if h not in reuse]
        reused = len(texts) - len(missing)
        fresh: dict[int, tuple] = {}
        if progress:
            progress(reused, len(texts))
        for start in range(0, len(missing), step):
            batch = missing[start : start + step]
            codes, scales = quantize(embedder.embed([texts[i] for i in batch]))
            fresh.update(zip(batch, zip(codes, scales, strict=True), strict=True))
            if progress:
                progress(reused + min(start + step, len(missing)), len(texts))
        rows = [fresh[i] if i in fresh else reuse[h] for i, h in enumerate(hashes.tolist())]
        manifest = {**manifest, **meta}
        if not rows:
            return cls(np.zeros((0, 1), np.float32), passages, manifest, hashes)
        codes = np.array([c for c, _ in rows], np.int8)
        scales = np.array([sc for _, sc in rows], np.float32)
        index = cls(dequantize(codes, scales), passages, manifest, hashes, (codes, scales))
        index.reused = reused
        return index

    def save(self, directory: Path) -> None:
        """All files, the manifest last (a directory without one is not an index)."""
        directory.mkdir(parents=True, exist_ok=True)
        lines = "".join(json.dumps(asdict(p), ensure_ascii=False) + "\n" for p in self.passages)
        write_gzip(directory / "passages.jsonl.gz", lines)
        if self.embedded:
            codes, scales = self._codes or quantize(self.vectors)
            save_npz(directory / "vectors.npz", codes=codes, scales=scales, hashes=self.hashes)
        else:
            save_npz(
                directory / "vectors.npz", hashes=self.hashes, dim=np.array(self.vectors.shape[1])
            )
        self.bm25.save(directory / "bm25.npz")
        if self._tags:
            write_gzip(directory / "tags.json.gz", json.dumps(self._tags, ensure_ascii=False))
        (directory / "manifest.json").write_text(json.dumps(self.manifest, indent=2))

    @classmethod
    def load(cls, directory: Path, expected_model: str | None = None) -> VersionIndex:
        path = directory / "manifest.json"
        if not path.exists():
            raise IndexNotFoundError(f"No index at {directory}. Build it with: bible-rag ingest")
        manifest = json.loads(path.read_text())
        if manifest.get("schema") != SCHEMA:
            raise IndexNotFoundError(f"{directory} is an older index format: bible-rag ingest")
        if expected_model and manifest["embedding_model"] != expected_model:
            raise IndexNotFoundError(
                f"{directory} was built with {manifest['embedding_model']}, "
                f"not {expected_model}: run bible-rag ingest --force"
            )
        with gzip.open(directory / "passages.jsonl.gz", "rt", encoding="utf-8") as fh:
            passages = [Passage(**json.loads(line)) for line in fh]
        data = np.load(directory / "vectors.npz")
        codes = None
        if manifest.get("embedded", True):
            codes = (data["codes"], data["scales"])
            vectors = dequantize(*codes)
        else:
            vectors = np.zeros((len(passages), int(data["dim"])), np.float32)
        index = cls(vectors, passages, manifest, data["hashes"], codes)
        index.directory = directory
        return index

    def search(
        self,
        vector: np.ndarray,
        k: int,
        text: str | None = None,
        candidates: int = 50,
        rrf: int = 60,
    ) -> list[Hit]:
        """Top-k by cosine, or fused with BM25 (reciprocal rank fusion) when text is given.
        Exact search: a matrix product over <= 12k vectors takes ~2 ms."""
        similarity = self.vectors @ vector.astype(np.float32)
        n = min(max(candidates, k), len(self))
        top = np.argpartition(-similarity, n - 1)[:n] if n < len(self) else np.arange(n)
        ranked = top[np.argsort(-similarity[top], kind="stable")]
        cosine = {int(i): float(similarity[i]) for i in ranked}
        if not text:
            return [Hit(self.passages[i], s, s) for i, s in list(cosine.items())[:k]]
        fused: dict[int, float] = defaultdict(float)
        for ranking in (list(cosine), self.bm25.search(text, candidates)):
            for rank, i in enumerate(ranking):
                fused[i] += 1 / (rrf + rank + 1)
        best = sorted(fused, key=fused.get, reverse=True)[:k]
        return [Hit(self.passages[i], float(similarity[i]), fused[i]) for i in best]

    def keyword_search(self, text: str, k: int) -> list[Hit]:
        """Keywords only (a language the embedder doesn't know, e.g. Hebrew or Greek)."""
        ids = self.bm25.search(text, k)
        return [Hit(self.passages[i], 0.0, 1 / (61 + rank)) for rank, i in enumerate(ids)]

    @property
    def bm25(self) -> BM25:
        if self._bm25 is None:  # the stored one, else built (~1 s)
            path = self.directory / "bm25.npz" if self.directory else None
            if path is not None and path.exists():
                self._bm25 = BM25.load(path)
            else:
                self._bm25 = BM25([p.embedding_text() for p in self.passages])
        return self._bm25


def _previous(settings: Settings, spec: str, version: str, model: str):
    """Embeddings of an earlier build of this version (any source or code), by text hash, and
    of its vocabulary, by word: what a rebuild reuses."""
    vectors, words = {}, {}
    for prebuilt in (True, False):
        directory = index_root(settings, spec, prebuilt) / version
        manifest = read_manifest(directory, version)
        if not manifest or manifest.get("embedding_model") != model:
            continue
        data = np.load(directory / "vectors.npz")
        if manifest.get("embedded", True):
            pairs = zip(data["codes"], data["scales"], strict=True)
            vectors.update(zip(data["hashes"].tolist(), pairs, strict=True))
        vocab = directory / "vocabulary.npz"
        if vocab.exists():
            v = np.load(vocab)
            if str(v["model"]) == model:
                names = v["words"].tobytes().decode().split("\n")
                pairs = zip(v["codes"], v["scales"], strict=True)
                words.update(zip(names, pairs, strict=True))
    return vectors, words


def pick_embedder(settings: Settings, spec: str | None = None):
    """(spec, embedder): `spec`, or the first of the chain that loads."""
    errors = []
    for candidate in [spec] if spec else settings.embedding_specs:
        try:
            device = settings.resolve_device()
            return candidate, load_embedder(candidate, device, settings.connect_timeout)
        except Exception as e:  # noqa: BLE001
            log.warning("Embedder %s unavailable: %s", candidate, describe_error(e))
            errors.append(f"{candidate}: {describe_error(e)}")
    raise RuntimeError("No embedder could be loaded: " + "; ".join(errors))


def prebuild(settings: Settings, spec: str | None = None, force: bool = False, progress=None):
    """Everything a fresh copy needs to answer at once, into storage/prebuilt (to be shipped
    with the code): the indexes (with vocabularies), the embedded examples of semantic.py and
    the Strong's dictionary. Only what changed is rebuilt. Returns the versions rebuilt."""
    from .semantic import Semantics
    from .sources import strongs_lexicon

    spec, embedder = pick_embedder(settings, spec)
    built = build_indexes(settings, force, spec, embedder, progress, prebuilt=True)
    # the examples: embedded once, for this embedder
    name = f"semantics-{embedder_key(embedder.model_name)}.npz"
    semantics = Semantics(embedder, settings.prebuilt_dir / name)
    semantics.topic_threshold  # noqa: B018 - embeds and saves every fixed text
    semantics.axis  # noqa: B018
    # the dictionary naming the original words (downloaded once from Debian)
    lexicon = json.dumps(strongs_lexicon(settings.sword_dir), ensure_ascii=False, sort_keys=True)
    target = settings.prebuilt_dir / "strongs.json.gz"
    if not target.exists() or gzip.decompress(target.read_bytes()).decode() != lexicon:
        target.write_bytes(gzip.compress(lexicon.encode(), mtime=0))
    return built


def build_indexes(
    settings: Settings,
    force: bool = False,
    spec: str | None = None,
    embedder=None,
    progress=None,
    prebuilt: bool = False,
) -> dict[str, VersionIndex]:
    """Index every version with the first embedder of the chain that loads (or `spec`), into
    storage/index (prebuilt=True: storage/prebuilt, what is shipped). A version whose index is
    up to date (here or shipped) is skipped; a rebuilt one reuses the embeddings of unchanged
    passages. `progress(version, done, total)` reports the embedding. Only one process builds
    at a time (a lock file): another one waits, then finds the indexes built."""
    check_licenses(settings.versions, settings.allow_restricted)
    if embedder is None:
        spec, embedder = pick_embedder(settings, spec)
    spec = spec or embedder.model_name
    root, built = index_root(settings, spec, prebuilt), {}
    root.mkdir(parents=True, exist_ok=True)
    with _build_lock(root / ".ingest.lock"):
        for version in settings.versions:
            current = fingerprint(version, settings)
            if not force:
                found = find_index(settings, spec, version)
                here = read_manifest(root / version, version)
                if (here or {}).get("fingerprint") == current or (
                    not prebuilt and found and found[1]
                ):
                    continue
            t0 = time.perf_counter()
            v = VERSIONS[version]
            units = _select_books(load_units(version, settings), settings.book_codes)
            passages = list(
                chunk_units(units, v.label, v.language, settings.chunk_size, settings.chunk_overlap)
            )
            read = time.perf_counter() - t0
            embed = embeddable(v.language, embedder)
            old_vectors, old_words = _previous(settings, spec, version, embedder.model_name)
            manifest = {
                "embedder": spec,
                "source": v.source,
                "license": v.license,
                "books": settings.books or "all",
                "fingerprint": current,
            }
            report = (lambda d, t, _v=version: progress(_v, d, t)) if progress else None
            t1 = time.perf_counter()
            index = VersionIndex.build(
                passages, embedder, manifest, report, embed=embed, reuse=old_vectors
            )
            index._tags = _tags_of(units) if version in TAGGED else {}
            if index._tags:  # Strong's numbers: prepare the dictionary naming them (once)
                from .sources import strongs_lexicon

                strongs_lexicon(settings.sword_dir)
            embedded = time.perf_counter() - t1
            # written aside (unique per process), then swapped: never a half-written index
            tmp = root / f".{version}.tmp-{os.getpid()}"
            shutil.rmtree(tmp, ignore_errors=True)
            index.save(tmp)
            index.directory = tmp
            # the vocabulary (meaning-based term lookup) is used in the first version of each
            # language: computed now for those, on demand for the others
            first = (
                next(x for x in settings.versions if VERSIONS[x].language == v.language) == version
            )
            if embed and first:
                index.vocabulary(embedder, old_words)
            shutil.rmtree(root / version, ignore_errors=True)
            tmp.rename(root / version)
            index.directory = root / version
            reused = index.reused
            log.info(
                "%s: %d passages; read in %.0f s; %s; %.0f s in all",
                version,
                len(passages),
                read,
                f"{reused} embeddings reused, {len(passages) - reused} computed in {embedded:.0f} s"
                if embed
                else "not embedded (language unknown to the embedder)",
                time.perf_counter() - t0,
            )
            built[version] = index
    return built


def _select_books(units, codes):
    """Only the books of BIBLE_RAG_BOOKS (OSIS codes), if set."""
    if not codes:
        return units
    from .corpus import BOOKS

    names = {name for osis, en, es in BOOKS if osis in codes for name in (en, es)}
    return [u for u in units if u.book in names]


def _tags_of(units) -> dict:
    from .grounding import book_index

    books, tags = {}, {}
    for u in units:
        if u.tags:
            if u.book not in books:
                books[u.book] = book_index(u.book)
            key = f"{books[u.book]}:{u.chapter}:{u.verse_start}"
            tags[key] = tags.get(key, []) + u.tags
    return tags


@contextlib.contextmanager
def _build_lock(path: Path):
    """An exclusive lock across processes (the CLI and the web app may both try to build)."""
    with open(path, "a+") as fh:
        try:
            import fcntl

            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                log.warning("Another process is building the indexes: waiting for it...")
                fcntl.flock(fh, fcntl.LOCK_EX)
            yield
            fcntl.flock(fh, fcntl.LOCK_UN)
        except ImportError:  # Windows
            import msvcrt

            while True:
                try:
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    log.warning("Another process is building the indexes: waiting for it...")
                    time.sleep(2)
            yield
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)


def open_indexes(settings: Settings):
    """(embedder, {version: index}, spec, skipped) for the first embedder of the chain that has
    an index of every version (built here, else shipped) and loads; `skipped`: the embedders
    that have indexes but failed to load, and why. Out-of-date indexes are used, marked stale
    (the answers say how to update them)."""
    check_licenses(settings.versions, settings.allow_restricted)
    skipped, missing = [], []
    for spec in settings.embedding_specs:
        found = {v: find_index(settings, spec, v) for v in settings.versions}
        if not all(found.values()) and not settings.books:
            # only the quick sample was built (`ingest --sample`): use it, and say so
            settings.books = "sample"
            sample = {v: find_index(settings, spec, v) for v in settings.versions}
            if all(sample.values()):
                found = sample
            else:
                settings.books = ""
        if not all(found.values()):
            missing.append(spec)
            continue
        try:
            embedder = load_embedder(spec, settings.resolve_device(), settings.connect_timeout)
        except Exception as e:  # noqa: BLE001
            skipped.append((spec, describe_error(e)))
            continue
        indexes = {}
        for v, (directory, current) in found.items():
            indexes[v] = VersionIndex.load(directory, embedder.model_name)
            indexes[v].stale = not current
            indexes[v].writable = not directory.is_relative_to(settings.prebuilt_dir)
        return embedder, indexes, spec, skipped
    reasons = [f"{s}: no index" for s in missing] + [f"{s}: {r}" for s, r in skipped]
    raise IndexNotFoundError(
        f"No usable index ({'; '.join(reasons)}). Build one with: bible-rag ingest"
    )
