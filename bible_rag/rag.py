"""Question answering (refactor of responder_consulta_rag in RAG0.ipynb):
retrieve per version -> grounded prompt -> first working LLM of the chain -> checked answer."""

from __future__ import annotations

import re
from collections.abc import Iterator

from .config import Settings
from .grounding import Grounding, check_answer
from .index import Hit, open_indexes
from .models import ModelChain, create_chat_model

SYSTEM_PROMPT = """You are a careful Bible study assistant. Answer using ONLY the passages in \
the context, from the King James Version (KJV, English) and the Reina-Valera 1960 (RVR1960, \
Spanish).
- If the passages do not answer the question, say so instead of guessing.
- Cite passages exactly as given, e.g. (John 3:16, KJV) or (Juan 3:16, RVR1960).
- You may briefly quote or compare both translations.
- Do not invent verses, references or doctrines. Be concise, neutral and respectful."""
# small models follow a short instruction and a plain "Passages / Question" layout better
COMPACT_PROMPT = (
    "You answer questions about the Bible using only the passages given. Quote "
    "words exactly from the passages. Put the reference in parentheses. Keep it short."
)

_ES = re.compile(
    r"[¿¡ñáéíóú]|\b(qu[eé]|c[oó]mo|d[oó]nde|cu[aá]l|por|para|dios|se[nñ]or|el|la|"
    r"los|las|del|es|un|una|dice|biblia|sobre)\b",
    re.I,
)
_EN = re.compile(
    r"\b(what|how|where|who|why|which|the|is|are|does|did|god|lord|bible|about|of|"
    r"and|say|says)\b",
    re.I,
)
_EXTRACTIVE = {
    "en": "Passages that best answer your question:",
    "es": "Pasajes que mejor responden a su pregunta:",
}


def detect_language(text: str) -> str:
    return "es" if len(_ES.findall(text)) > len(_EN.findall(text)) else "en"


class Answer:
    """Iterate to stream the text; afterwards .text, .model, .notes and .grounding are set.
    If the model fails before its first token, the chain's next model answers instead."""

    def __init__(self, rag: BibleRAG, question: str, sources: list[Hit], language: str):
        self.rag, self.question, self.sources, self.language = rag, question, sources, language
        self.text, self.model, self.notes = "", "", []
        self.grounding: Grounding | None = None

    def __iter__(self) -> Iterator[str]:
        chain = self.rag.chain
        while True:
            model, self.model = chain.model, chain.active
            if model is None or (model.languages and self.language not in model.languages):
                if model is not None:  # e.g. the English-only fallback and a Spanish question
                    self.notes.append(f"{model.label} does not answer in '{self.language}'.")
                    self.model = "extractive"
                self.text = self.rag.extractive_answer(self.sources, self.language)
                yield self.text
                break
            try:
                for token in model.stream(
                    self.rag.messages(self.question, self.sources, self.language, model.compact)
                ):
                    self.text += token
                    yield token
                break
            except Exception as e:  # noqa: BLE001
                if self.text:
                    raise
                chain.failover(e)
        self.text = self.text.strip()
        self.grounding = check_answer(self.text, self.sources)
        self.notes = self.rag.fallback_notes() + self.notes


class BibleRAG:
    def __init__(
        self,
        settings: Settings,
        embedder,
        indexes: dict,
        chain: ModelChain | None = None,
        embedder_spec: str = "",
        embedder_skipped: list | None = None,
    ):
        self.settings, self.embedder, self.indexes, self.chain = settings, embedder, indexes, chain
        self.embedder_spec = embedder_spec or embedder.model_name
        self.embedder_skipped = embedder_skipped or []

    @classmethod
    def from_settings(cls, settings: Settings, load_llm: bool = True) -> BibleRAG:
        embedder, indexes, spec, skipped = open_indexes(settings)
        chain = None
        if load_llm:
            device = settings.resolve_device()
            chain = ModelChain(settings.llm_specs, lambda s: create_chat_model(s, settings, device))
            chain.model  # noqa: B018 - load now, so startup reports what is used
        return cls(settings, embedder, indexes, chain, spec, skipped)

    def search(
        self, question: str, k: int | None = None, versions: list[str] | None = None
    ) -> list[Hit]:
        """Top-k passages per version, best first."""
        if not question.strip():
            raise ValueError("The question is empty")
        unknown = set(versions or []) - self.indexes.keys()
        if unknown:
            raise ValueError(f"Unknown version(s): {', '.join(unknown)}")
        vector = self.embedder.embed([question])[0]
        text = question if self.settings.retrieval == "hybrid" else None
        hits = [
            h
            for v in versions or self.indexes
            for h in self.indexes[v].search(vector, k or self.settings.top_k, text)
        ]
        return sorted(hits, key=lambda h: (h.rank, h.score), reverse=True)

    def messages(self, question: str, hits: list[Hit], language: str, compact: bool = False):
        if compact:  # small English-only fallback: 3 English passages, short prompt
            english = [h for h in hits if h.passage.language == "en"][:3]
            context = "\n\n".join(f"{h.passage.citation}: {h.passage.text}" for h in english)
            user = f"Passages:\n{context}\n\nQuestion: {question.strip()}"
            return [
                {"role": "system", "content": COMPACT_PROMPT},
                {"role": "user", "content": user},
            ]
        context = "\n\n".join(
            f"[{i}] {h.passage.citation}\n{h.passage.text}" for i, h in enumerate(hits, 1)
        )
        lang = {"en": "English", "es": "Spanish"}[language]
        user = (
            f"Context passages:\n{context}\n\nQuestion: {question.strip()}\n\n"
            f"Answer in {lang}, citing the passages you rely on."
        )
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]

    @staticmethod
    def extractive_answer(hits: list[Hit], language: str) -> str:
        """No-LLM answer: the best passages verbatim, question's language first."""
        same = [h for h in hits if h.passage.language == language]
        chosen = same[:2] + [h for h in hits if h not in same][:2]
        lines = [f"- {h.passage.citation}: “{h.passage.text}”" for h in chosen]
        return "\n".join([_EXTRACTIVE[language], *lines])

    def fallback_notes(self) -> list[str]:
        notes = []
        if self.embedder_skipped:
            notes.append(
                f"Search uses fallback embedder {self.embedder_spec} "
                f"({self.embedder_skipped[0][0]}: {self.embedder_skipped[0][1]})."
            )
        if self.chain and self.chain.skipped:
            spec, reason = self.chain.skipped[0]
            notes.append(f"Fallback LLM {self.chain.active} in use ({spec}: {reason}).")
        return notes

    def ask_stream(
        self, question: str, k: int | None = None, versions: list[str] | None = None
    ) -> Answer:
        if self.chain is None:
            raise RuntimeError("No LLM loaded (retrieval-only mode): use search.")
        return Answer(self, question, self.search(question, k, versions), detect_language(question))

    def ask(self, question: str, k: int | None = None, versions: list[str] | None = None) -> Answer:
        answer = self.ask_stream(question, k, versions)
        for _ in answer:
            pass
        return answer
