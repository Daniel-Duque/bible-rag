"""Runtime settings, read from environment variables (and an optional ``.env`` file).

Every field can be overridden with ``BIBLE_RAG_<FIELD_NAME>``; see ``.env.example``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PREFIX = "BIBLE_RAG_"


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # Data / index locations (relative paths are resolved against the project root)
    data_dir: Path = PROJECT_ROOT / "src" / "data"
    index_dir: Path = PROJECT_ROOT / "storage" / "index"
    versions: tuple[str, ...] = ("kjv", "rvr1960")
    include_apocrypha: bool = False  # the KJV PDF includes it, RVR1960 does not

    # Chunking. paraphrase-multilingual-MiniLM-L12-v2 truncates at 128 tokens
    # (~450-500 chars), so larger chunks would be silently cut when embedded.
    chunk_size: int = 450
    chunk_overlap: int = 80

    # Models (same as RAG0.ipynb)
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    device: str = "auto"  # auto | cpu | cuda | mps

    # Retrieval / generation
    top_k: int = 3  # passages retrieved *per version*
    max_new_tokens: int = 512
    temperature: float = 0.2  # 0 = greedy decoding
    top_p: float = 0.9

    # HTTP service
    host: str = "127.0.0.1"
    port: int = 8000
    server_url: str | None = None  # when set, `ask`/`chat` use the running service

    @classmethod
    def from_env(cls, env_file: Path | None = None, **overrides) -> Settings:
        """Build settings from defaults < .env < environment < explicit overrides."""
        try:
            from dotenv import load_dotenv

            load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)
        except ImportError:  # pragma: no cover - python-dotenv is a dependency
            pass

        values: dict = {}
        for f in fields(cls):
            raw = os.environ.get(ENV_PREFIX + f.name.upper())
            if raw is None or raw == "":
                continue
            values[f.name] = _coerce(f.name, raw, getattr(cls(), f.name))
        values.update({k: v for k, v in overrides.items() if v is not None})
        settings = cls(**values)
        settings.data_dir = _resolve(settings.data_dir)
        settings.index_dir = _resolve(settings.index_dir)
        return settings

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


def _resolve(path: Path | str) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _coerce(name: str, raw: str, default):
    if isinstance(default, bool):
        return _parse_bool(raw)
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    if isinstance(default, tuple):
        return tuple(v.strip() for v in raw.split(",") if v.strip())
    if isinstance(default, Path):
        return Path(raw)
    return raw
