"""Report which embedders and LLMs of the fallback chains are usable here (`bible-rag models`)."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from .config import Settings
from .hub import ModelUnavailable, check_hf_model
from .index import embedder_key
from .llm import (
    EXTRACTIVE,
    bundled_smollm2_gguf,
    connect_ollama,
    connect_openai,
    create_chat_model,
    describe_error,
)


@dataclass
class ProbeResult:
    kind: str  # "embedder" | "llm"
    spec: str
    available: bool
    detail: str
    index: bool | None = None  # embedders: is there an index built with it?
    test: str | None = None  # --test: result of a real load + inference


def _probe_hf(repo: str, timeout: float) -> tuple[bool, str]:
    return check_hf_model(repo, timeout)


def probe_embedders(settings: Settings) -> list[ProbeResult]:
    results = []
    for spec in settings.embedding_specs:
        kind, _, arg = spec.partition(":")
        has_index = all(
            (settings.index_dir / embedder_key(spec) / v / "manifest.json").exists()
            for v in settings.versions
        )
        if kind == "hf":
            ok, detail = _probe_hf(arg, settings.connect_timeout)
        elif kind == "usem3":
            try:
                import usem3  # noqa: F401

                ok, detail = True, "bundled with the install (offline, multilingual)"
            except ImportError as e:
                ok, detail = False, describe_error(e)
        else:
            ok, detail = False, "unknown spec"
        results.append(ProbeResult("embedder", spec, ok, detail, index=has_index))
    return results


def probe_llms(settings: Settings) -> list[ProbeResult]:
    results = []
    for spec in settings.llm_specs:
        kind, _, arg = spec.partition(":")
        try:
            if kind == "hf":
                ok, detail = _probe_hf(arg, settings.connect_timeout)
            elif kind == "ollama":
                model = connect_ollama(settings, arg)
                ok, detail = True, f"Ollama at {settings.ollama_url}, model {model.model_name}"
            elif kind == "openai":
                model = connect_openai(settings, arg)
                ok, detail = True, f"{settings.openai_base_url}, model {model.model_name}"
            elif kind == "smollm2":
                path = bundled_smollm2_gguf()
                converted = (settings.model_cache_dir / "SmolLM2-135M-Instruct").exists()
                ok, detail = (
                    True,
                    (
                        f"bundled ({path.stat().st_size / 1e6:.0f} MB GGUF, English only"
                        f"{', converted' if converted else ', converted on first use'})"
                    ),
                )
            elif kind == EXTRACTIVE:
                ok, detail = True, "no model: quotes the retrieved passages"
            else:
                ok, detail = False, "unknown spec"
        except ModelUnavailable as e:
            ok, detail = False, str(e)
        results.append(ProbeResult("llm", spec, ok, detail))
    return results


def smoke_test_llm(spec: str, settings: Settings) -> str:
    """Actually load the model and generate a short reply."""
    t0 = time.perf_counter()
    try:
        model = create_chat_model(spec, settings, settings.resolve_device())
        if model is None:
            return "ok (no model needed)"
        reply = model.generate([{"role": "user", "content": "Reply with one word: hello"}])
        return f"ok in {time.perf_counter() - t0:.0f}s, replied {reply[:40]!r}"
    except Exception as e:  # noqa: BLE001
        return f"FAILED: {describe_error(e)}"


def probe_all(settings: Settings, test: bool = False) -> list[ProbeResult]:
    results = probe_embedders(settings) + probe_llms(settings)
    if test:
        for r in results:
            if r.kind == "llm" and r.available:
                r.test = smoke_test_llm(r.spec, settings)
    return results


def as_dicts(results: list[ProbeResult]) -> list[dict]:
    return [asdict(r) for r in results]
