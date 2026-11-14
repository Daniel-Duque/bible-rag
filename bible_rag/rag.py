"""Retrieval-augmented QA over the KJV and the Reina-Valera 1960.

This is the refactored version of ``responder_consulta_rag`` in RAG0.ipynb:
retrieve -> build a grounded prompt with citations -> generate -> return answer
plus sources. Models are injected so the pipeline can be tested without
downloading anything.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from .config import Settings
from .corpus import VERSIONS
from .index import Embedder, Hit, VersionIndex
from .llm import ChatModel

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a careful Bible study assistant. You answer questions using ONLY the \
Scripture passages provided in the context, which come from two translations: the King James \
Version (KJV, English) and the Reina-Valera 1960 (RVR1960, Spanish).

Rules:
- Base every statement on the provided passages. If they do not answer the question, say so \
plainly instead of guessing.
- Cite the passages you use with their reference and version exactly as given, e.g. \
(John 3:16, KJV) or (Juan 3:16, RVR1960).
- When both translations are relevant, you may briefly quote or compare them.
- Do not invent verses, references or doctrines that are not in the context.
- Be concise, neutral and respectful."""

_LANGUAGE_NAMES = {"en": "English", "es": "Spanish"}
_ES_HINTS = re.compile(
    r"[¿¡ñáéíóú]|\b(qu[eé]|c[oó]mo|d[oó]nde|cu[aá]l|por|para|dios|se[nñ]or|"
    r"el|la|los|las|del|es|un|una|dice|biblia|sobre)\b",
    re.IGNORECASE,
)
_EN_HINTS = re.compile(
    r"\b(what|how|where|who|why|which|the|is|are|does|did|god|lord|bible|"
    r"about|of|and|say|says)\b",
    re.IGNORECASE,
)


def detect_language(text: str) -> str:
    """Tiny ES/EN detector; enough to tell the model which language to answer in."""
    es, en = len(_ES_HINTS.findall(text)), len(_EN_HINTS.findall(text))
    return "es" if es > en else "en"


@dataclass
class Answer:
    question: str
    answer: str
    language: str
    sources: list[Hit] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "language": self.language,
            "sources": [hit_to_dict(h) for h in self.sources],
        }


def hit_to_dict(hit: Hit) -> dict:
    p = hit.passage
    return {
        "reference": p.reference,
        "version": p.version,
        "score": round(hit.score, 4),
        "text": p.text,
        "book": p.book,
        "chapter": p.chapter,
        "verse_start": p.verse_start,
        "verse_end": p.verse_end,
        "page": p.page,
    }


class BibleRAG:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        indexes: dict[str, VersionIndex],
        llm: ChatModel | None = None,
    ):
        if not indexes:
            raise ValueError("At least one version index is required")
        self.settings = settings
        self.embedder = embedder
        self.indexes = indexes
        self.llm = llm

    # ---------------------------------------------------------------- construction
    @classmethod
    def from_settings(cls, settings: Settings, load_llm: bool = True) -> BibleRAG:
        from .index import SentenceTransformerEmbedder
        from .llm import LocalChatModel

        device = settings.resolve_device()
        indexes = {
            v: VersionIndex.load(settings.index_dir / v, expected_model=settings.embedding_model)
            for v in settings.versions
        }
        embedder = SentenceTransformerEmbedder(settings.embedding_model, device=device)
        llm = None
        if load_llm:
            llm = LocalChatModel(
                settings.llm_model,
                device=device,
                max_new_tokens=settings.max_new_tokens,
                temperature=settings.temperature,
                top_p=settings.top_p,
            )
        return cls(settings, embedder, indexes, llm)

    # ---------------------------------------------------------------- retrieval
    def _select(self, versions: list[str] | None) -> list[str]:
        if not versions:
            return list(self.indexes)
        unknown = [v for v in versions if v not in self.indexes]
        if unknown:
            raise ValueError(
                f"Unknown or not indexed version(s): {', '.join(unknown)}. "
                f"Available: {', '.join(self.indexes)}"
            )
        return versions

    def search(
        self, question: str, k: int | None = None, versions: list[str] | None = None
    ) -> list[Hit]:
        """Top-k passages *per version*, ordered by similarity."""
        question = question.strip()
        if not question:
            raise ValueError("The question is empty")
        k = k or self.settings.top_k
        query = self.embedder.embed([question])[0]
        hits: list[Hit] = []
        for version in self._select(versions):
            hits.extend(self.indexes[version].search(query, k))
        return sorted(hits, key=lambda h: h.score, reverse=True)

    # ---------------------------------------------------------------- generation
    def build_messages(self, question: str, hits: list[Hit], language: str) -> list[dict]:
        context = "\n\n".join(
            f"[{i}] {h.passage.citation}\n{h.passage.text}" for i, h in enumerate(hits, 1)
        )
        user = (
            f"Context passages:\n{context}\n\n"
            f"Question: {question.strip()}\n\n"
            f"Answer in {_LANGUAGE_NAMES.get(language, 'the language of the question')}, "
            "citing the passages you rely on."
        )
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]

    def _require_llm(self) -> ChatModel:
        if self.llm is None:
            raise RuntimeError("No LLM loaded (retrieval-only mode). Use `search` instead.")
        return self.llm

    def ask(self, question: str, k: int | None = None, versions: list[str] | None = None) -> Answer:
        llm = self._require_llm()
        hits = self.search(question, k, versions)
        language = detect_language(question)
        text = llm.generate(self.build_messages(question, hits, language))
        return Answer(question, text, language, hits)

    def ask_stream(
        self, question: str, k: int | None = None, versions: list[str] | None = None
    ) -> tuple[list[Hit], Iterator[str]]:
        """Like :meth:`ask` but returns the sources immediately and a token iterator."""
        llm = self._require_llm()
        hits = self.search(question, k, versions)
        language = detect_language(question)
        return hits, llm.stream(self.build_messages(question, hits, language))

    def info(self) -> dict:
        return {
            "versions": {
                v: {
                    "name": VERSIONS[v].name,
                    "passages": len(idx),
                    "embedding_model": idx.manifest.get("embedding_model"),
                }
                for v, idx in self.indexes.items()
            },
            "llm": getattr(self.llm, "model_name", None),
        }
