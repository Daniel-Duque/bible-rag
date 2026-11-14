"""Retrieval-augmented QA over the KJV and the Reina-Valera 1960.

Refactored from ``responder_consulta_rag`` in RAG0.ipynb: retrieve -> grounded prompt ->
generate -> answer + sources. On top of the notebook:

* the embedder and the LLM come from fallback chains (see :mod:`bible_rag.llm`), so a
  question is answered even when the preferred models cannot be downloaded;
* every answer is checked against the retrieved passages (:mod:`bible_rag.grounding`).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from .config import Settings
from .corpus import VERSIONS
from .grounding import GroundingReport, check_answer
from .index import (
    Embedder,
    Hit,
    IndexNotFoundError,
    VersionIndex,
    embedder_key,
    load_embedder,
)
from .llm import ChatModel, ModelChain, create_chat_model, describe_error

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

# Small models follow short, concrete instructions better than long rule lists.
COMPACT_SYSTEM_PROMPT = (
    "You answer questions about the Bible using only the passages given. Quote words exactly "
    "from the passages. Put the reference in parentheses. Keep it short."
)
COMPACT_MAX_PASSAGES = 3

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

_EXTRACTIVE_INTRO = {
    "en": {
        "none": "No language model is available, so here are the passages that best match "
        "your question:",
        "language": "The available language model ({model}) cannot answer in English, so here "
        "are the passages that best match your question:",
    },
    "es": {
        "none": "No hay un modelo de lenguaje disponible; estos son los pasajes que mejor "
        "responden a su pregunta:",
        "language": "El modelo de lenguaje disponible ({model}) no responde en español; estos "
        "son los pasajes que mejor responden a su pregunta:",
    },
}


def detect_language(text: str) -> str:
    """Tiny ES/EN detector; enough to tell the model which language to answer in."""
    es, en = len(_ES_HINTS.findall(text)), len(_EN_HINTS.findall(text))
    return "es" if es > en else "en"


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


@dataclass
class Answer:
    question: str
    answer: str
    language: str
    sources: list[Hit] = field(default_factory=list)
    model: str = ""  # chain spec that produced the answer ("extractive" = no LLM)
    notes: list[str] = field(default_factory=list)
    grounding: GroundingReport | None = None

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "language": self.language,
            "model": self.model,
            "notes": self.notes,
            "grounding": self.grounding.to_dict() if self.grounding else None,
            "sources": [hit_to_dict(h) for h in self.sources],
        }


class StreamingAnswer:
    """Iterate to receive the answer text; afterwards ``model``, ``notes``, ``grounding``.

    If the active model fails before producing any text, the chain fails over to the next
    model transparently.
    """

    def __init__(
        self, rag: BibleRAG, question: str, hits: list[Hit], language: str, stream: bool = True
    ):
        self.rag, self.question, self.sources, self.language = rag, question, hits, language
        self._stream = stream
        self.model = ""
        self.notes: list[str] = []
        self.text = ""
        self.grounding: GroundingReport | None = None
        self._language_note = ""

    def __iter__(self) -> Iterator[str]:
        chain = self.rag.chain
        while True:
            model = chain.model
            self.model = chain.active_spec
            if model is None or not self.rag.supports(model, self.language):
                self.text = self.rag.extractive_answer(self.sources, self.language, model)
                if model is not None:  # say who did *not* answer, and why
                    self._language_note = (
                        f"'{model.label}' does not support "
                        f"'{self.language}'; answered extractively."
                    )
                    self.model = "extractive"
                yield self.text
                break
            messages = self.rag.build_messages(self.question, self.sources, self.language, model)
            emitted = False
            try:
                tokens = model.stream(messages) if self._stream else [model.generate(messages)]
                for token in tokens:
                    emitted = True
                    self.text += token
                    yield token
                break
            except Exception as e:  # noqa: BLE001
                if emitted:
                    raise
                chain.failover(e)
        self.text = self.text.strip()
        self.grounding = check_answer(self.text, self.sources)
        self.notes = self.rag.fallback_notes(answered_by=self.model)
        if self._language_note:
            self.notes.append(self._language_note)

    def to_answer(self) -> Answer:
        return Answer(
            self.question,
            self.text,
            self.language,
            self.sources,
            self.model,
            self.notes,
            self.grounding,
        )


class BibleRAG:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        indexes: dict[str, VersionIndex],
        chain: ModelChain | None = None,
        embedder_spec: str = "",
        embedder_skipped: list[tuple[str, str]] | None = None,
    ):
        if not indexes:
            raise ValueError("At least one version index is required")
        self.settings = settings
        self.embedder = embedder
        self.indexes = indexes
        self.chain = chain
        self.embedder_spec = embedder_spec or embedder.model_name
        self.embedder_skipped = embedder_skipped or []

    # ---------------------------------------------------------------- construction
    @classmethod
    def from_settings(cls, settings: Settings, load_llm: bool = True) -> BibleRAG:
        device = settings.resolve_device()
        embedder, indexes, spec, skipped = select_embedder(settings, device)
        chain = None
        if load_llm:
            chain = ModelChain(settings.llm_specs, lambda s: create_chat_model(s, settings, device))
            chain.model  # noqa: B018 - load now so startup reports what is used
        return cls(settings, embedder, indexes, chain, spec, skipped)

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
        text = question if self.settings.retrieval == "hybrid" else None
        hits: list[Hit] = []
        for version in self._select(versions):
            hits.extend(self.indexes[version].search(query, k, query_text=text))
        return sorted(hits, key=lambda h: (h.rank, h.score), reverse=True)

    # ---------------------------------------------------------------- generation
    @staticmethod
    def supports(model: ChatModel, language: str) -> bool:
        return model.languages is None or language in model.languages

    def build_messages(
        self, question: str, hits: list[Hit], language: str, model: ChatModel | None = None
    ) -> list[dict]:
        if model and model.compact:
            # Small models: fewer passages, only in languages they understand, and the short
            # "Passages / Question" format (asking a 135M model to "cite the passages" makes it
            # answer with a bare reference).
            langs = model.languages
            hits = [h for h in hits if not langs or h.passage.language in langs]
            context = "\n\n".join(
                f"{h.passage.citation}: {h.passage.text}" for h in hits[:COMPACT_MAX_PASSAGES]
            )
            return [
                {"role": "system", "content": COMPACT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Passages:\n{context}\n\nQuestion: {question.strip()}",
                },
            ]
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

    def extractive_answer(
        self, hits: list[Hit], language: str, model: ChatModel | None = None
    ) -> str:
        """No-LLM answer: the best passages, quoted verbatim (grounded by construction)."""
        texts = _EXTRACTIVE_INTRO.get(language, _EXTRACTIVE_INTRO["en"])
        intro = texts["language"].format(model=model.label) if model else texts["none"]
        same_lang = [h for h in hits if h.passage.language == language]
        chosen = (same_lang[:2] + [h for h in hits if h not in same_lang][:2]) or hits[:4]
        lines = [f"- {h.passage.citation}: “{h.passage.text}”" for h in chosen]
        return intro + "\n" + "\n".join(lines)

    def fallback_notes(self, answered_by: str | None = None) -> list[str]:
        notes = []
        if self.embedder_skipped:
            notes.append(
                f"Search used the fallback embedder '{self.embedder_spec}' "
                f"({self.embedder_skipped[0][0]}: {self.embedder_skipped[0][1]})."
            )
        if self.chain and self.chain.is_fallback:
            first, reason = self.chain.skipped[0]
            active = self.chain.active_spec
            verb = "Answered by" if answered_by in (None, active) else "Active"
            notes.append(f"{verb} fallback model '{active}' ({first}: {reason}).")
        return notes

    def _require_chain(self) -> ModelChain:
        if self.chain is None:
            raise RuntimeError("No LLM loaded (retrieval-only mode). Use `search` instead.")
        return self.chain

    def ask_stream(
        self,
        question: str,
        k: int | None = None,
        versions: list[str] | None = None,
        stream: bool = True,
    ) -> StreamingAnswer:
        self._require_chain()
        hits = self.search(question, k, versions)
        return StreamingAnswer(self, question, hits, detect_language(question), stream)

    def ask(self, question: str, k: int | None = None, versions: list[str] | None = None) -> Answer:
        result = self.ask_stream(question, k, versions, stream=False)
        for _ in result:
            pass
        return result.to_answer()

    def info(self) -> dict:
        return {
            "embedder": {
                "active": self.embedder_spec,
                "skipped": [{"spec": s, "reason": r} for s, r in self.embedder_skipped],
            },
            "versions": {
                v: {
                    "name": VERSIONS[v].name,
                    "passages": len(idx),
                    "embedding_model": idx.manifest.get("embedding_model"),
                }
                for v, idx in self.indexes.items()
            },
            "llm": self.chain.status() if self.chain else None,
        }


def select_embedder(settings: Settings, device: str):
    """First embedder of the chain that has an index for every version *and* loads."""
    skipped: list[tuple[str, str]] = []
    for spec in settings.embedding_specs:
        dirs = {v: settings.index_dir / embedder_key(spec) / v for v in settings.versions}
        missing = [v for v, d in dirs.items() if not (d / "manifest.json").exists()]
        if missing:
            skipped.append((spec, f"no index for {', '.join(missing)} (run: bible-rag ingest)"))
            continue
        try:
            embedder = load_embedder(spec, device, settings.connect_timeout)
        except Exception as e:  # noqa: BLE001 - unavailable: try the next one
            log.warning("Embedder %s unavailable: %s", spec, describe_error(e))
            skipped.append((spec, describe_error(e)))
            continue
        indexes = {
            v: VersionIndex.load(d, expected_model=embedder.model_name) for v, d in dirs.items()
        }
        if skipped:
            log.warning("Using fallback embedder %s", spec)
        return embedder, indexes, spec, skipped
    details = "; ".join(f"{s}: {r}" for s, r in skipped)
    raise IndexNotFoundError(f"No usable index ({details}). Build one with: bible-rag ingest")
