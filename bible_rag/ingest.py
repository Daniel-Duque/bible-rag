"""Build the per-version FAISS indexes from the PDFs (notebook steps 1-3).

Indexes live in ``<index_dir>/<embedder>/<version>``. The first embedder of the chain that
loads is used, so ingestion works offline (falling back to the bundled USE-multilingual
encoder) and can be re-run later with the preferred model to add a better index.
"""

from __future__ import annotations

import logging
import shutil
import time

from .config import Settings
from .corpus import VERSIONS, load_version
from .index import Embedder, VersionIndex, embedder_key, load_embedder
from .llm import describe_error

log = logging.getLogger(__name__)


def pick_embedder(settings: Settings, specs: list[str] | None = None) -> tuple[str, Embedder]:
    errors = []
    for spec in specs or settings.embedding_specs:
        try:
            return spec, load_embedder(spec, settings.resolve_device(), settings.connect_timeout)
        except Exception as e:  # noqa: BLE001 - try the next one
            log.warning("Embedder %s unavailable: %s", spec, describe_error(e))
            errors.append(f"{spec}: {describe_error(e)}")
    raise RuntimeError("No embedding model could be loaded: " + "; ".join(errors))


def ingest(
    settings: Settings,
    embedder: Embedder | None = None,
    force: bool = False,
    versions: list[str] | None = None,
    embedder_spec: str | None = None,
) -> dict[str, VersionIndex]:
    versions = versions or list(settings.versions)
    for v in versions:
        if v not in VERSIONS:
            raise ValueError(f"Unknown version {v!r}. Known: {', '.join(VERSIONS)}")

    if embedder is None:
        embedder_spec, embedder = pick_embedder(
            settings, [embedder_spec] if embedder_spec else None
        )
    spec = embedder_spec or embedder.model_name
    root = settings.index_dir / embedder_key(spec)
    log.info("Embedding with %s -> %s", spec, root)

    todo = [v for v in versions if force or not (root / v / "manifest.json").exists()]
    built: dict[str, VersionIndex] = {}
    if not todo:
        log.info("All indexes already exist in %s (use --force to rebuild)", root)
        return built

    params = {
        "embedder_spec": spec,
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
        target = root / version
        tmp = root / f".{version}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        index.save(tmp)  # write next to the target, then swap: never leave a half index
        shutil.rmtree(target, ignore_errors=True)
        tmp.rename(target)
        log.info("%s: index written to %s (%.1fs total)", version, target, time.perf_counter() - t0)
        built[version] = index
    return built
