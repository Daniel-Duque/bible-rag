"""Settings: defaults < .env < BIBLE_RAG_<FIELD> environment variables < explicit overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    data_dir: Path = ROOT / "src" / "data"  # the two PDFs
    index_dir: Path = ROOT / "storage" / "index"  # <embedder>/<version>/
    model_cache_dir: Path = ROOT / "storage" / "models"  # converted GGUF fallback
    versions: tuple[str, ...] = ("kjv", "rvr1960")
    include_apocrypha: bool = False  # only the KJV PDF has it

    # 450 chars ~ the 128-token limit of the MiniLM embedder
    chunk_size: int = 450
    chunk_overlap: int = 80

    # Preferred models (as in RAG0.ipynb); chains are tried in order, see models.py
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    embedding_chain: tuple[str, ...] = ()  # default: hf:<embedding_model>, usem3
    llm_chain: tuple[str, ...] = ()  # default: hf:<llm_model>, ollama, openai, smollm2, extractive
    device: str = "auto"  # auto | cpu | cuda | mps

    # Free local servers speaking the OpenAI API
    ollama_url: str = "http://127.0.0.1:11434/v1"
    openai_base_url: str = ""  # LM Studio, llama.cpp server, vLLM...
    openai_api_key: str = ""
    connect_timeout: float = 3.0

    retrieval: str = "hybrid"  # hybrid (embeddings + BM25) | dense
    top_k: int = 3  # passages per version
    max_new_tokens: int = 512
    temperature: float = 0.2  # 0 = greedy
    top_p: float = 0.9

    @classmethod
    def from_env(cls, **overrides) -> Settings:
        try:
            from dotenv import load_dotenv

            load_dotenv(ROOT / ".env")
        except ImportError:
            pass
        values = {}
        for f in fields(cls):
            raw = os.environ.get(f"BIBLE_RAG_{f.name.upper()}")
            if raw:
                values[f.name] = _coerce(raw, f.default)
        values.update({k: v for k, v in overrides.items() if v is not None})
        s = cls(**values)
        for name in ("data_dir", "index_dir", "model_cache_dir"):
            path = Path(getattr(s, name)).expanduser()
            setattr(s, name, path if path.is_absolute() else ROOT / path)
        return s

    @property
    def embedding_specs(self) -> list[str]:
        return list(self.embedding_chain) or [f"hf:{self.embedding_model}", "usem3"]

    @property
    def llm_specs(self) -> list[str]:
        return list(self.llm_chain) or [
            f"hf:{self.llm_model}",
            "ollama",
            "openai",
            "smollm2",
            "extractive",
        ]

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch

        if torch.cuda.is_available():
            return "cuda"
        return "mps" if torch.backends.mps.is_available() else "cpu"


def _coerce(raw: str, default):
    if isinstance(default, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, tuple):
        return tuple(v.strip() for v in raw.split(",") if v.strip())
    return type(default)(raw) if isinstance(default, int | float | Path) else raw
