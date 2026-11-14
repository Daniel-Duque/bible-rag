"""Build the per-version FAISS indexes from the PDFs (notebook steps 1-3)."""

from __future__ import annotations

import logging
import shutil
import time

from .config import Settings
from .corpus import VERSIONS, load_version
from .index import Embedder, VersionIndex

log = logging.getLogger(__name__)


def ingest(
    settings: Settings,
    embedder: Embedder | None = None,
    force: bool = False,
    versions: list[str] | None = None,
) -> dict[str, VersionIndex]:
    versions = versions or list(settings.versions)
    for v in versions:
        if v not in VERSIONS:
            raise ValueError(f"Unknown version {v!r}. Known: {', '.join(VERSIONS)}")

    todo = [v for v in versions if force or not (settings.index_dir / v / "manifest.json").exists()]
    built: dict[str, VersionIndex] = {}
    if not todo:
        log.info("All indexes already exist in %s (use --force to rebuild)", settings.index_dir)
        return built

    if embedder is None:
        from .index import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder(
            settings.embedding_model, device=settings.resolve_device()
        )

    params = {
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "include_apocrypha": settings.include_apocrypha,
    }
    for version in todo:
        t0 = time.perf_counter()
        passages = load_version(
            version,
            settings.data_dir,
            settings.chunk_size,
            settings.chunk_overlap,
            settings.include_apocrypha,
        )
        log.info(
            "%s: %d passages parsed in %.1fs, embedding...",
            version,
            len(passages),
            time.perf_counter() - t0,
        )
        index = VersionIndex.build(version, passages, embedder, params)
        target = settings.index_dir / version
        tmp = settings.index_dir / f".{version}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        index.save(tmp)  # write next to the target, then swap: never leave a half index
        shutil.rmtree(target, ignore_errors=True)
        tmp.rename(target)
        log.info("%s: index written to %s (%.1fs total)", version, target, time.perf_counter() - t0)
        built[version] = index
    return built
