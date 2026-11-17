"""Settings. Each field can be set as an environment variable BIBLE_RAG_<FIELD> (or in a .env
file next to pyproject.toml): BIBLE_RAG_VERSIONS=kjv,rv1909,web, BIBLE_RAG_TOP_K=5...
Precedence: defaults < .env < environment < explicit overrides."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# BIBLE_RAG_BOOKS=sample (`bible-rag ingest --sample`): ~1/4 of the Bible, enough for every
# example in the README
SAMPLE_BOOKS = ("Gen", "Ps", "Prov", "Eccl", "Matt", "John", "Rom", "1Cor", "Eph", "Jas")


@dataclass
class Settings:
    # Bible texts (see sources.VERSIONS or `bible-rag versions`); "all" = every public one. The
    # Hebrew and Greek originals are shown verse for verse next to what is found
    versions: tuple[str, ...] = ("kjv", "rv1909", "wlc", "sblgnt")
    allow_restricted: bool = False  # allow texts whose license forbids public use (RVR1960)

    # where things are kept (relative paths are relative to the project folder)
    data_dir: Path = ROOT / "data"  # your own files: the RVR1960 PDF
    sword_dir: Path = ROOT / "storage" / "sword"  # Debian SWORD modules, Strong's dictionary
    sources_dir: Path = ROOT / "storage" / "sources"  # data files of the pinned PyPI packages
    index_dir: Path = ROOT / "storage" / "index"  # indexes built here: <embedder>/<version>/
    prebuilt_dir: Path = ROOT / "storage" / "prebuilt"  # the indexes shipped with the code
    model_cache_dir: Path = ROOT / "storage" / "models"  # the converted SmolLM2 fallback

    # "" = the whole Bible; "sample" = ~1/4 of it, to try things quickly; or OSIS codes
    # ("Gen,Ps,John"). Passages of ~450 characters fit the 128-token window of MiniLM-style
    # embedders
    books: str = ""
    chunk_size: int = 450
    chunk_overlap: int = 80

    # Models: each chain is tried in order and the first entry that works is used (models.py)
    embedding_model: str = ""  # a sentence-transformers model to prefer to the bundled usem3
    llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    embedding_chain: tuple[str, ...] = ()  # default: [hf:<embedding_model>,] usem3
    llm_chain: tuple[str, ...] = ()  # default: hf:<llm_model>, ollama, openai, smollm2, extractive
    device: str = "auto"  # auto | cpu | cuda | mps
    # natural language inference model that checks opposition; "" = off (embeddings only)
    nli_model: str = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"
    # a capable chat model also settles close routing calls and writes the two opposite
    # statements for topics outside the lexicon
    llm_planning: bool = True

    # local model servers speaking the OpenAI API
    ollama_url: str = "http://127.0.0.1:11434/v1"
    openai_base_url: str = ""  # LM Studio, llama.cpp server, vLLM...
    openai_api_key: str = ""
    connect_timeout: float = 3.0

    # API (`bible-rag serve` on a GPU machine; the CLI and the web app as its clients)
    api_key: str = ""  # required to serve beyond 127.0.0.1; the client sends the same key
    server_url: str = ""  # client: use this server, e.g. https://gpu.example.org
    server_timeout: float = 600.0  # client: longest wait for an answer, seconds
    server_insecure: bool = False  # client: allow the key over plain http to another host
    server_rate: int = 30  # server: requests per client per minute
    server_concurrency: int = 2  # server: answers generated at once (beyond: 503 busy)

    # retrieval and generation
    retrieval: str = "hybrid"  # hybrid (embeddings + BM25 keywords) | dense
    top_k: int = 3  # passages per translation (per side for contradictions)
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
        if s.versions == ("all",):
            from .sources import VERSIONS

            s.versions = tuple(v for v, version in VERSIONS.items() if version.public)
        for name in (
            "data_dir",
            "sword_dir",
            "sources_dir",
            "index_dir",
            "prebuilt_dir",
            "model_cache_dir",
        ):
            path = Path(getattr(s, name)).expanduser()
            setattr(s, name, path if path.is_absolute() else ROOT / path)
        return s

    @property
    def book_codes(self) -> tuple[str, ...]:
        if self.books.strip().lower() == "sample":
            return SAMPLE_BOOKS
        return tuple(b.strip() for b in self.books.split(",") if b.strip())

    @property
    def corpus_suffix(self) -> str:
        """Index folder suffix for a subset of books ("" for the whole Bible)."""
        if not self.books.strip():
            return ""
        if self.books.strip().lower() == "sample":
            return "@sample"
        return "@" + hashlib.sha1(",".join(sorted(self.book_codes)).encode()).hexdigest()[:8]

    @property
    def embedding_specs(self) -> list[str]:
        if self.embedding_chain:
            return list(self.embedding_chain)
        return [f"hf:{self.embedding_model}", "usem3"] if self.embedding_model else ["usem3"]

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
