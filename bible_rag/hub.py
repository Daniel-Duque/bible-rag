"""Fast availability checks for Hugging Face models.

Without these, loading an uncached model while the hub is unreachable makes
huggingface_hub retry every file with exponential backoff (over a minute) before failing,
which would delay every fallback. Cached models are loaded with ``local_files_only`` so a
flaky or missing network does not matter once a model has been downloaded.
"""

from __future__ import annotations

from pathlib import Path


class ModelUnavailable(RuntimeError):
    """Raised by a chain entry that cannot be used in this environment."""


def is_cached(repo: str) -> bool:
    from huggingface_hub import try_to_load_from_cache

    return any(
        isinstance(try_to_load_from_cache(repo, name), str)
        for name in ("config.json", "modules.json")
    )


def hub_reachable(repo: str, timeout: float) -> tuple[bool, str]:
    from huggingface_hub import HfApi

    try:
        HfApi().model_info(repo, timeout=timeout)
        return True, "downloadable from Hugging Face (downloaded on first use)"
    except Exception as e:  # noqa: BLE001
        text = str(e).strip().splitlines()[0] if str(e).strip() else ""
        return False, f"not cached and Hugging Face unreachable ({e.__class__.__name__}: {text})"


def check_hf_model(repo: str, timeout: float) -> tuple[bool, str]:
    """(available, detail) without downloading anything."""
    if Path(repo).expanduser().is_dir():
        return True, "local directory"
    if is_cached(repo):
        return True, "in the local Hugging Face cache"
    return hub_reachable(repo, timeout)


def prepare_hf_model(repo: str, timeout: float) -> bool:
    """Return ``local_files_only`` for loading ``repo``; raise ModelUnavailable quickly."""
    if Path(repo).expanduser().is_dir() or is_cached(repo):
        return True
    ok, detail = hub_reachable(repo, timeout)
    if not ok:
        raise ModelUnavailable(detail)
    return False
