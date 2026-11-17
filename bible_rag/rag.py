"""Question answering: the question -> the planner (agent.py) routes it (direct question,
contradiction on a topic, translation comparison) and gathers the passages, aligned across
translations -> prompt for that task -> first working LLM of the chain -> synthesis + the
passages in every translation, with the model's citations checked (grounding.py)."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator

from .agent import Plan, Planner
from .config import Settings
from .corpus import embeddable
from .grounding import Grounding, check_answer
from .index import Hit, open_indexes
from .models import ModelChain, create_chat_model, embedder_key
from .semantic import Semantics, semantics_for
from .sources import VERSIONS

log = logging.getLogger(__name__)

# Language identification by function words (multilingual embeddings are built to ignore the
# language, so they can't tell it): the answer is written in the question's language
_PROFILES = {
    "en": "the of and to is are was what how where who why which does did about say says "
    "bible god lord any there i you my it this that with should can could would will",
    "es": "el la los las del que qué es son por para cómo dónde quién cuál sobre dice biblia "
    "dios hay una un se",
    "pt": "o os as do da dos das que é são para como onde quem sobre diz bíblia deus existem "
    "há uma um não",
    "fr": "le la les du des que est sont pour comment où qui sur dit bible dieu il y a une un se",
    "de": "der die das den dem des und ist sind wie wo wer über sagt bibel gott gibt es eine "
    "ein sich",
    "it": "il lo la gli le del della che è sono per come dove chi sulla dice bibbia dio ci una un",
}
_PROFILE_WORDS = {lang: set(words.split()) for lang, words in _PROFILES.items()}
_MARKS = {"es": "ñ¿¡", "pt": "ãõç", "fr": "èêàùç", "de": "äöüß", "it": "ìò"}
LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "pt": "Portuguese",
    "fr": "French",
    "de": "German",
    "it": "Italian",
}


def identify_language(text: str) -> str:
    """en, es, pt, fr, de or it (ties: en)."""
    words = re.findall(r"\w+", text.lower())
    scores = {
        lang: sum(w in vocab for w in words) + 2 * sum(c in _MARKS.get(lang, "") for c in text)
        for lang, vocab in _PROFILE_WORDS.items()
    }
    best = max(scores, key=lambda lang: (scores[lang], lang == "en"))
    return best if scores[best] else "en"


def detect_language(text: str) -> str:
    """The language of the assistant's own texts: English or Spanish (Portuguese -> Spanish,
    the closest)."""
    lang = identify_language(text)
    return lang if lang in ("en", "es") else "es" if lang == "pt" else "en"


class Answer:
    """Iterate to stream the text; afterwards .text, .model, .notes and .grounding are set.

    The text is the model's synthesis (or, without a capable model, a lead sentence) followed
    by the plan's passages in every translation. If the model fails before its first token,
    the chain's next model answers instead. Only the model's part needs checking: the rest
    is quoted from the passages."""

    def __init__(self, rag: BibleRAG, plan: Plan):
        self.rag, self.plan = rag, plan
        self.question, self.language, self.intent = plan.question, plan.language, plan.intent
        self.sources: list[Hit] = plan.sources
        self.text, self.synthesis, self.model, self.notes = "", "", "", []
        self.grounding: Grounding | None = None

    def __iter__(self) -> Iterator[str]:
        chain, written = self.rag.chain, False  # written: by a language model
        while True:
            # a model that can't write this answer (known without loading it) isn't loaded
            model, reason = chain.model_for(self.plan.unsupported_by)
            self.model = chain.current
            if model is not None and reason is None:
                reason = self.plan.unsupported_by(model)
            if model is None or reason:
                if reason:  # e.g. the English-only fallback and a Spanish question
                    self.notes.append(reason)
                    self.model = "extractive"
                self.synthesis = self.plan.intro()
                yield self.synthesis
                break
            try:
                for token in model.stream(self.plan.messages(model.compact)):
                    self.synthesis += token
                    yield token
                written = True
                break
            except Exception as e:  # noqa: BLE001
                if self.synthesis:
                    raise
                chain.failover(e)
        self.synthesis = self.synthesis.strip()
        passages = self.plan.render()
        if passages:
            yield "\n\n" + passages
        self.text = f"{self.synthesis}\n\n{passages}".strip()
        # the model's words are checked; without a model, the quoted passages (the notes are
        # the assistant's own, not claims about the passages)
        checked = self.synthesis if written else self.plan.render(notes=False)
        self.grounding = check_answer(checked, self.sources)
        self.notes = self.rag.fallback_notes() + self.plan.checks + self.notes


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
        self.notes: list[str] = []  # shown with every answer (e.g. "server unavailable")

    @classmethod
    def from_settings(
        cls, settings: Settings, load_llm: bool = True, eager: bool = False
    ) -> BibleRAG:
        """eager: load the LLM now (a server); else on the first answer that uses it."""
        embedder, indexes, spec, skipped = open_indexes(settings)
        chain = None
        if load_llm:
            device = settings.resolve_device()
            chain = ModelChain(settings.llm_specs, lambda s: create_chat_model(s, settings, device))
            if eager:
                chain.model  # noqa: B018
        return cls(settings, embedder, indexes, chain, spec, skipped)

    def search(
        self,
        question: str,
        k: int | None = None,
        versions: list[str] | None = None,
        keywords: str = "",
    ) -> list[Hit]:
        """Top-k passages per version, best first. `keywords`: the Bible's own words for what
        the question is about, added to the keyword side of the search."""
        if not question.strip():
            raise ValueError("The question is empty")
        unknown = set(versions or []) - self.indexes.keys()
        if unknown:
            raise ValueError(f"Unknown version(s): {', '.join(unknown)}")
        # by default the versions the embedder can read; an original asked for explicitly is
        # searched by keywords (its words: Hebrew / Greek)
        versions = versions or self.planner.searchable(list(self.indexes))
        vector = self.embedder.embed([question])[0]
        text = f"{question} {keywords}".strip() if self.settings.retrieval == "hybrid" else None
        k = k or self.settings.top_k
        hits = []
        for v in versions:
            index = self.indexes[v]
            if embeddable(index.passages[0].language if index.passages else "", self.embedder):
                hits += index.search(vector, k, text)
            else:
                hits += index.keyword_search(question, k)
        return sorted(hits, key=lambda h: (h.rank, h.score), reverse=True)

    def fallback_notes(self) -> list[str]:
        notes = list(self.notes)
        stale = [VERSIONS[v].label for v, index in self.indexes.items() if index.stale]
        if stale:
            notes.append(
                f"The index of {', '.join(stale)} is out of date (its source or the code "
                "changed): `bible-rag ingest` updates it, reusing the unchanged embeddings."
            )
        if self.settings.books:
            notes.append(
                f"Searching only part of the Bible (BIBLE_RAG_BOOKS={self.settings.books}): "
                "run `bible-rag ingest` for all of it."
            )
        if self.embedder_skipped:
            notes.append(
                f"Search uses fallback embedder {self.embedder_spec} "
                f"({self.embedder_skipped[0][0]}: {self.embedder_skipped[0][1]})."
            )
        if self.chain and self.chain.skipped and self.chain.loaded:
            spec, reason = self.chain.skipped[0]
            notes.append(f"Fallback LLM {self.chain.current} in use ({spec}: {reason}).")
        return notes

    def warm_up(self) -> None:
        """Do the one-time work now rather than on the first question: the example texts'
        embeddings and the topic threshold (semantic.py), the NLI model if configured."""
        self.semantics.warm()
        for index in self.indexes.values():  # keyword index, folded text, words' tags
            index.bm25, index.folded, index.tags  # noqa: B018
        if self.chain is not None and self.settings.nli_model:
            self.nli  # noqa: B018

    @property
    def planner(self) -> Planner:
        return Planner(self)

    @property
    def nli(self):
        """The NLI model checking opposition, loaded on first use (None if unavailable; the
        reason is in .nli_skipped)."""
        if not hasattr(self, "_nli"):
            from .models import describe_error, load_nli

            self._nli, self.nli_skipped = None, ""
            if self.settings.nli_model:
                try:
                    self._nli = load_nli(self.settings, self.settings.resolve_device())
                except Exception as e:  # noqa: BLE001 - optional: embeddings are the fallback
                    self.nli_skipped = describe_error(e)
                    log.debug("NLI model unavailable: %s", self.nli_skipped)
        return self._nli

    @nli.setter
    def nli(self, model) -> None:
        self._nli = model

    @property
    def strongs(self) -> dict:
        """The Strong's dictionary prepared by `ingest` (empty if absent: nothing is downloaded
        while answering)."""
        if not hasattr(self, "_strongs"):
            self._strongs = load_strongs(self.settings)
        return self._strongs

    @strongs.setter
    def strongs(self, value: dict) -> None:
        self._strongs = value

    @property
    def semantics(self) -> Semantics:
        """Routing / topic / opposition by meaning, for this embedder (built once)."""
        name = f"semantics-{embedder_key(self.embedder.model_name)}.npz"
        return semantics_for(
            self.embedder, self.settings.index_dir / name, self.settings.prebuilt_dir / name
        )

    def plan(
        self,
        question: str,
        k: int | None = None,
        versions: list[str] | None = None,
        task: str | None = None,
    ) -> Plan:
        """Route the question (general / contradiction / translation; `task` chooses one
        explicitly) and gather the passages, without generating (see agent.py)."""
        plan = self.planner.plan(question, detect_language(question), k, versions, task)
        plan.reply_language = identify_language(question)
        return plan

    def ask_stream(
        self,
        question: str,
        k: int | None = None,
        versions: list[str] | None = None,
        task: str | None = None,
    ) -> Answer:
        if self.chain is None:
            raise RuntimeError("No LLM loaded (retrieval-only mode): use search.")
        return Answer(self, self.plan(question, k, versions, task))

    def ask(
        self,
        question: str,
        k: int | None = None,
        versions: list[str] | None = None,
        task: str | None = None,
    ) -> Answer:
        answer = self.ask_stream(question, k, versions, task)
        for _ in answer:
            pass
        return answer


def load_strongs(settings) -> dict:
    """The Strong's dictionary: prepared by `ingest` (storage/sword/strongs.json), else the
    shipped copy (storage/prebuilt/strongs.json.gz), else empty. Never downloads."""
    import gzip

    local = settings.sword_dir / "strongs.json"
    if local.exists():
        return json.loads(local.read_text(encoding="utf-8"))
    shipped = settings.prebuilt_dir / "strongs.json.gz"
    if shipped.exists():
        with gzip.open(shipped, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    return {}
