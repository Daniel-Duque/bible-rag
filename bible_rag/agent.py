"""The chat assistant's planner: what kind of question is this, and which pipeline answers it.

general        "Where was Jesus born?"
               direct semantic search; the answer is a synthesis plus what each translation
               says at the passages found (parallel passages aligned by verse).
contradiction  "Are there any paradoxes in the Bible about alcohol?"
               the topic is searched twice, for the affirming and for the opposing statement
               ("wine is good ..." / "wine is a mocker ..."); passages are split into the two
               sides by which statement they are closer to in meaning, and compared across
               translations.
translation    "How do the translations render Psalm 16:10?" / "Do the versions differ on hell?"
               the same passages side by side with the places where their wording differs.
related        "Should I use a cellphone?"
               a subject the Bible never names: the answer says so (and which texts were
               checked for its words), then shows the biblical themes related to it
               (communication, self-control...), each with how it relates, so that applying
               them stays visibly an inference.

Decisions are made by meaning (semantic.py), not keywords: the question is routed by its
closeness to example questions of each kind (or, for a subject closer to the things the Bible
never names than to those it does, to the related themes), its topic by closeness to the
lexicon's topics, and a passage's side by where it falls between the two statements and on
the opposition axis; an NLI model, if available, checks the opposition directly. When a
capable LLM is loaded it settles close routing calls and close calls on whether the Bible names
a subject, and writes the two statements for topics outside the lexicon.
Without any LLM everything still works; the LLM (if any) writes the synthesis (rag.Answer).
"""

from __future__ import annotations

import collections
import json
import logging
import re
from dataclasses import dataclass, field

import numpy as np

from .align import (
    Row,
    choose_keys,
    clauses,
    concept_note,
    excerpt,
    fill_parallels,
    group,
    key_verses,
    original_keys,
    renderings,
    word_changes,
)
from .corpus import embeddable
from .grounding import book_index, chapters, references, without_references
from .index import Hit, tokens
from .lexicon import (
    GENERIC_CLAIMS,
    GENERIC_SIDES,
    TOPICS,
    Topic,
    adhoc_topic,
    compile_terms,
    fold,
)
from .semantic import CONTRADICTION, GENERAL, INTENTS, RELATED, ROUTES, TRANSLATION, unit
from .sources import VERSIONS
from .themes import Theme

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- routing
@dataclass
class Route:
    intent: str
    reason: str  # why (shown to the user)
    scores: dict = field(default_factory=dict)  # closeness to each kind of question


# -------------------------------------------------------------------------------- plan
@dataclass
class Section:
    title: str
    rows: list[Row]
    note: str = ""  # shown under the title (e.g. how a related theme relates to the subject)
    key: str = ""  # what the section is (related: the theme's key)


@dataclass
class Absence:
    """Whether the Bible names what a question is about, and the evidence."""

    subject: str  # as the user wrote it ("cellphone"); "" when no subject word was found
    words: list[str]
    score: float  # > 0: not named (see Planner.absence)
    absent: bool
    by: str = "meaning"  # who decided: "meaning", or the LLM that settled a close call
    checked: list[str] = field(default_factory=list)  # texts searched for the words
    found: dict = field(default_factory=dict)  # text label -> (passages, first reference)
    views: tuple = ()  # the subject's vector and its words' (semantic.theme_scores)
    nearest: str = ""  # the Bible's closest word to the least covered subject word

    @property
    def vector(self) -> np.ndarray:
        """The subject with its words, for ranking passages."""
        return unit(np.sum(self.views, axis=0))

    @property
    def reason(self) -> str:
        how = f"{self.by} decided" if self.by != "meaning" else f"by meaning, {self.score:+.2f}"
        if self.nearest:
            how += f"; nearest Bible word “{self.nearest}”"
        side = "not named in the Bible" if self.absent else "named in the Bible"
        return f"{side} ({how})"


@dataclass
class Plan:
    """What the assistant found, ready to be written up by an LLM or shown as is."""

    intent: str
    question: str
    language: str
    versions: list[str]
    sections: list[Section]
    topic: Topic | None = None
    queries: list[str] = field(default_factory=list)  # what was searched, shown to the user
    notes: list[str] = field(default_factory=list)  # translation notes for the whole answer
    remarks: list[str] = field(default_factory=list)  # about the search itself
    reason: str = ""
    extra: list[Hit] = field(default_factory=list)  # passages the notes cite
    checks: list[str] = field(default_factory=list)  # how the findings were checked
    reply_language: str = ""  # the question's language, for the LLM (default: `language`)
    absence: Absence | None = None  # related: whether the Bible names the subject
    themes: list[str] = field(default_factory=list)  # related: the themes, as named in the lead

    @property
    def sources(self) -> list[Hit]:
        seen, out = set(), []
        hits = [h for s in self.sections for row in s.rows for h in row.hits.values()]
        for h in hits + self.extra:
            if id(h.passage) not in seen:
                seen.add(id(h.passage))
                out.append(h)
        return out

    @property
    def topic_name(self) -> str:
        if self.topic:
            return self.topic.name[self.language]
        return self.absence.subject if self.absence else ""

    # ----------------------------------------------------------------- LLM messages
    def messages(self, compact: bool = False) -> list[dict]:
        from .rag import LANGUAGE_NAMES

        lang = LANGUAGE_NAMES[self.reply_language or self.language]
        if compact:  # small English-only fallback: 3 English passages, short prompt
            english = [h for h in self.sources if h.passage.language == "en"][:3]
            context = "\n\n".join(f"{h.passage.citation}: {h.passage.text}" for h in english)
            user = f"Passages:\n{context}\n\nQuestion: {self.question.strip()}"
            return [
                {"role": "system", "content": COMPACT_PROMPT},
                {"role": "user", "content": user},
            ]
        labels = ", ".join(f"{VERSIONS[v].label} ({VERSIONS[v].name})" for v in self.versions)
        blocks, n = [], 0
        for section in self.sections:
            lines = [f"## {section.title}"] if len(self.sections) > 1 else []
            if section.note:  # under its heading
                lines = [f"{lines[0]}\n({section.note})" if lines else f"({section.note})"]
            for row in section.rows:
                for h in row.hits.values():
                    n += 1
                    lines.append(f"[{n}] {h.passage.citation}\n{h.passage.text}")
            if not section.rows:
                lines.append("(no passages found)")
            blocks.append("\n\n".join(lines))
        context = "\n\n".join(blocks)
        notes = [*self.notes, *(note for s in self.sections for r in s.rows for note in r.notes)]
        if notes:
            context += "\n\nNotes on the translations:\n" + "\n".join(f"- {n}" for n in notes)
        if self.intent == RELATED:  # what was checked: the model must not claim otherwise
            found = [self.intro().split(". ")[0].rstrip(".") + ".", *self.remarks]
            context += "\n\nWhat the search found:\n" + "\n".join(f"- {f}" for f in found)
        topic = self.topic_name or _TEXT["en"]["your_subject"]
        task = TASKS[self.intent].format(lang=lang, topic=topic)
        user = f"Context passages:\n{context}\n\nQuestion: {self.question.strip()}\n\n{task}"
        system = PROMPTS[self.intent].format(versions=labels)
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def unsupported_by(self, model) -> str | None:
        """Why this model should not write the answer (the plan is shown as is instead)."""
        language = self.reply_language or self.language
        if model.languages and language not in model.languages:
            return f"{model.label} does not answer in '{language}'."
        if model.compact and self.intent != GENERAL:
            return f"{model.label} is too small to compare passages; they are shown side by side."
        return None

    # ------------------------------------------------------------- deterministic text
    def intro(self) -> str:
        """The lead sentence of an answer written without an LLM."""
        t = _TEXT[self.language]
        if self.intent == CONTRADICTION and len(self.sections) == 2:
            a, b = (bool(s.rows) for s in self.sections)
            key = "both" if a and b else "one" if a or b else "none"
            return t[f"contradiction_{key}"].format(topic=self.topic_name)
        if self.intent == TRANSLATION:
            differs = any(r.notes for s in self.sections for r in s.rows)
            return t["translation_differs" if differs else "translation_same"]
        if self.intent == RELATED and self.absence is not None:
            a = self.absence
            subject = t["quote"].format(a.subject) if a.subject else t["your_subject"]
            if not a.absent:
                key = "related_named"
            else:
                key = "related_as_such" if a.found else "related_absent"
            themes = _join(self.themes, t["and"]) if self.themes else t["no_themes"]
            return _upper_first(t[key].format(subject=subject, themes=themes))
        return t["general"]

    def render(self, notes: bool = True) -> str:
        """Markdown appended to every answer: the passages, per translation, with notes
        (notes=False: only what is quoted from the passages)."""
        t = _TEXT[self.language]
        out = [f"_{remark}_\n" for remark in self.remarks] if notes else []
        for section in self.sections:
            out.append(f"**{section.title}**")
            if section.note and notes:
                out.append(f"_{section.note}_")
            if not section.rows:
                out.append(f"_{t['none_found']}_")
            for i, row in enumerate(section.rows, 1):
                out.append(f"{i}. {row.title}")
                for v, h in row.hits.items():
                    text = row.key.get(v) or h.passage.text
                    out.append(f"   - {VERSIONS[v].label}: “{excerpt(text, self.excerpt)}”")
                if notes:
                    out.extend(f"   - ↳ {note}" for note in row.notes)
            out.append("")
        if self.notes and notes:
            out.append(f"**{t['notes']}**")
            out.extend(f"- {note}" for note in self.notes)
        return "\n".join(out).strip()

    @property
    def excerpt(self) -> int:
        return 500 if self.intent == TRANSLATION else 240


# ----------------------------------------------------------------------------- prompts
COMPACT_PROMPT = (
    "You answer questions about the Bible using only the passages given. Quote "
    "words exactly from the passages. Put the reference in parentheses. Keep it short."
)
ROUTER_PROMPT = """Classify the user's question about the Bible. Reply with one word:
general - a direct question (what the Bible says, who, where, why, a verse's meaning)
contradiction - whether the Bible contradicts itself, or has passages for and against something
translation - how translations or versions word a passage, or whether their meaning differs"""
CLAIMS_PROMPT = """The user asks whether the Bible contradicts itself on a topic. Reply with \
JSON only, no other text:
{"topic_en": "<the topic, 1-4 words, English>", "topic_es": "<the same in Spanish>",
 "for_en": "<one sentence affirming or approving it, in biblical style>",
 "against_en": "<one sentence saying the opposite>",
 "for_es": "<for_en in Spanish>", "against_es": "<against_en in Spanish>"}"""
ABSENCE_PROMPT = """Does the Bible mention the subject of the user's question by name, or \
speak of it directly (in any words)? Things that did not exist in biblical times are not \
mentioned. Reply with one word: yes or no."""
PROMPTS = {
    GENERAL: """You are a careful Bible study assistant. Answer using ONLY the passages in \
the context, which come from these translations: {versions}.
- Start with a direct answer in one to three sentences.
- Cite passages exactly as given, e.g. (John 3:16, KJV) or (Juan 3:16, RV1909).
- If the translations word the answer differently, say so briefly.
- If the passages do not answer the question, say so instead of guessing.
- Do not invent verses, references or doctrines. Be concise, neutral and respectful.""",
    CONTRADICTION: """You are a careful, neutral Bible study assistant. The user asks whether \
the Bible is paradoxical or contradictory on a topic. The context has two groups of passages, \
found by searching for two opposite statements; each passage is given in several translations: \
{versions}.
- Say whether the passages are in real tension, or whether their context explains it (who \
speaks and to whom, law or narrative or poetry, moderation or excess, literal or figurative).
- Cite passages of both groups exactly as given, e.g. (Proverbs 20:1, KJV). The grouping is \
automatic: if a passage does not belong to its group, say so.
- If a translation words a passage so that its sense changes, point it out. Passages in Hebrew
or Greek are the originals, and the notes name the original word behind a translation's word:
use them to tell whether both sides use the same original word or a translation choice.
- Use ONLY the passages given. Do not invent verses. Do not take a denominational side.""",
    TRANSLATION: """You are a careful Bible translation assistant. The context shows the same \
passages in several translations: {versions}.
- Compare how the translations render them: point out words or phrases whose meaning differs, \
and say whether the difference changes the sense or is only style or language.
- Quote the exact words of each translation with its citation, e.g. “hell” (Psalms 16:10, KJV) \
and “sepulcro” (Salmos 16:10, RV1909).
- Passages in Hebrew or Greek are the originals, and the notes name the original word (Strong's \
number) behind a translation's word: say which rendering is closer to it.
- If the translations agree, say so. Use ONLY the passages and notes given; do not invent \
verses or claims about the original languages.""",
    RELATED: """You are a careful Bible study assistant. The user asks about something \
the Bible may not mention by name (see "What the search found"). The context has passages on \
related biblical themes, grouped under headings that say how each theme relates to the \
subject; each passage is given in several translations: {versions}.
- Start by saying plainly whether the Bible mentions the subject, as the search found.
- Then, for each group, explain in one or two sentences how the theme relates to the subject \
and what principle its passages give. Cite passages exactly as given, e.g. (2 John 1:12, KJV).
- Make clear that applying these passages to the subject is an inference, not something the \
text says. If a group does not really apply, say so.
- Use ONLY the passages given. Do not invent verses and do not claim that the Bible speaks of \
the subject. Be concise, neutral and respectful; do not take a denominational side.""",
}
TASKS = {
    GENERAL: "Answer in {lang}, citing the passages you rely on.",
    CONTRADICTION: "Answer in {lang}: are these passages about {topic} contradictory, or does "
    "their context reconcile them? Cite passages from both groups.",
    TRANSLATION: "Answer in {lang}: how do the translations differ here, and does it change "
    "the meaning? Cite the passages.",
    RELATED: "Answer in {lang}: what does the Bible say, and not say, about {topic}? Explain how "
    "each group of passages relates to it, citing them.",
}

_TEXT = {
    "en": {
        "general": "Passages that best answer your question:",
        "contradiction_both": "I searched for two opposite statements about {topic} and found "
        "passages supporting each, so the texts are in tension on this topic. Whether that is a "
        "contradiction depends on context (who speaks, to whom, when and why): compare them "
        "below.",
        "contradiction_one": "I searched for two opposite statements about {topic}, but the "
        "passages found only support one of them: on this topic the texts point the same way.",
        "contradiction_none": "I found no passages about {topic}. Try other words for it.",
        "translation_differs": "The same passages in each translation, with the places where "
        "their wording differs:",
        "translation_same": "The same passages in each translation (no wording difference on "
        "the topic was detected; compare them below):",
        "by_version": "What each translation says",
        "side_by_side": "Side by side",
        "none_found": "No passages found.",
        "notes": "Translation notes",
        "no_topic": "Name a topic to compare, e.g. ‘Are there paradoxes in the Bible about "
        "alcohol?’. Meanwhile, here is a direct search.",
        "meaning_differs": "Meaning may differ: {a} and {b} contradict each other here "
        "(NLI, {p:.0%})",
        "originals": "Original words",
        "side_words": "Original words on each side",
        "same_words": "Both sides use {words}: the tension is in the original text, not a "
        "translation choice.",
        "different_words": "The sides use different original words: part of the difference "
        "may lie in the vocabulary of the originals.",
        "checked_nli": "Sides checked for opposition by the NLI model {model}.",
        "checked_embeddings": "Sides estimated from meaning similarity (no NLI model loaded: "
        "see `bible-rag models`).",
        "changes": "{a} → {b}",
        "style": "plus {n} minor or archaic-to-modern changes (“thou” → “you”…)",
        "very_different": "{a} and {b} word this very differently",
        "related_absent": "the Bible does not mention {subject}. It does speak of related "
        "things: {themes}. The passages below are about those, not about {subject}: applying "
        "them to it is an inference.",
        "related_as_such": "the Bible does not speak of {subject} as such (for where its "
        "words occur, see the note below). It does speak of related things: {themes}. The "
        "passages below are about those, not about {subject}: applying them to it is an "
        "inference.",
        "related_named": "the Bible does speak of {subject} directly: a direct search shows "
        "those passages. These related themes widen the picture: {themes}.",
        "your_subject": "what you ask about",
        "quote": "“{}”",
        "no_themes": "no related theme was found",
        "and": "and",
        "related_link": "How this relates to “{subject}”: {link}.",
        "related_link_frame": "How this relates: {link}.",
        "not_in_texts": "“{words}” does not occur in {versions}.",
        "in_texts": "“{words}” occurs in {found}: check whether it means the same there.",
        "passages": "{n} passages, e.g. {example}",
        "passage": "1 passage: {example}",
        "not_checked": "No loaded text is in the question's language: whether the Bible names "
        "it was judged by meaning only.",
    },
    "es": {
        "general": "Pasajes que mejor responden a su pregunta:",
        "contradiction_both": "Busqué dos afirmaciones opuestas sobre {topic} y encontré "
        "pasajes que apoyan cada una, así que los textos están en tensión en este tema. Si es "
        "una contradicción depende del contexto (quién habla, a quién, cuándo y por qué): "
        "compárelos abajo.",
        "contradiction_one": "Busqué dos afirmaciones opuestas sobre {topic}, pero los pasajes "
        "encontrados sólo apoyan una: en este tema los textos apuntan en la misma dirección.",
        "contradiction_none": "No encontré pasajes sobre {topic}. Pruebe con otras palabras.",
        "translation_differs": "Los mismos pasajes en cada traducción, con los lugares donde "
        "la redacción difiere:",
        "translation_same": "Los mismos pasajes en cada traducción (no se detectó diferencia "
        "de redacción sobre el tema; compárelos abajo):",
        "by_version": "Qué dice cada traducción",
        "side_by_side": "Lado a lado",
        "none_found": "No se encontraron pasajes.",
        "notes": "Notas de traducción",
        "no_topic": "Indique un tema para comparar, p. ej. ‘¿Hay paradojas en la Biblia sobre "
        "el alcohol?’. Mientras tanto, ésta es una búsqueda directa.",
        "meaning_differs": "El sentido puede diferir: {a} y {b} se contradicen aquí (NLI, {p:.0%})",
        "originals": "Palabras originales",
        "side_words": "Palabras originales de cada lado",
        "same_words": "Ambos lados usan {words}: la tensión está en el texto original, no en "
        "una elección de traducción.",
        "different_words": "Los lados usan palabras originales distintas: parte de la "
        "diferencia puede estar en el vocabulario de los originales.",
        "checked_nli": "Lados verificados por oposición con el modelo NLI {model}.",
        "checked_embeddings": "Lados estimados por similitud de sentido (sin modelo NLI: "
        "vea `bible-rag models`).",
        "changes": "{a} → {b}",
        "style": "y {n} cambios menores o de estilo arcaico a moderno («thou» → «you»…)",
        "very_different": "{a} y {b} lo redactan de forma muy distinta",
        "related_absent": "la Biblia no menciona {subject}. Sí habla de cosas relacionadas: "
        "{themes}. Los pasajes de abajo tratan de eso, no de {subject}: aplicarlos es una "
        "inferencia.",
        "related_as_such": "la Biblia no habla de {subject} como tal (dónde aparecen sus "
        "palabras, vea la nota abajo). Sí habla de cosas relacionadas: {themes}. Los pasajes "
        "de abajo tratan de eso, no de {subject}: aplicarlos es una inferencia.",
        "related_named": "la Biblia sí habla de {subject} directamente: una búsqueda directa "
        "muestra esos pasajes. Estos temas relacionados amplían el panorama: {themes}.",
        "your_subject": "lo que usted pregunta",
        "quote": "«{}»",
        "no_themes": "no se encontró ningún tema relacionado",
        "and": "y",
        "related_link": "Relación con «{subject}»: {link}.",
        "related_link_frame": "Relación: {link}.",
        "not_in_texts": "«{words}» no aparece en {versions}.",
        "in_texts": "«{words}» aparece en {found}: compruebe si significa lo mismo allí.",
        "passages": "{n} pasajes, p. ej. {example}",
        "passage": "1 pasaje: {example}",
        "not_checked": "Ningún texto cargado está en el idioma de la pregunta: si la Biblia lo "
        "nombra se juzgó sólo por el sentido.",
    },
}


def _upper_first(text: str) -> str:
    return text[:1].upper() + text[1:]


def _join(items: list[str], conjunction: str) -> str:
    """'a', 'a and b', 'a, b and c'."""
    if len(items) < 2:
        return "".join(items)
    return f"{', '.join(items[:-1])} {conjunction} {items[-1]}"


# --------------------------------------------------------------------------- pipelines
class Planner:
    """Runs the pipelines over a BibleRAG's indexes and embedder. The class attributes are the
    tuning constants (tuned with `bible-rag eval --agent` on the bundled embedder)."""

    route_margin = 0.03  # routing scores closer than this: a capable LLM settles it
    pool_cap = 150  # passages per version whose clauses are embedded (contradiction search)
    axis_weight = 0.5  # stance: weight of the opposition axis (see _score_stance)
    margin = 0.02  # |stance| below this: on neither side
    # side ranking: closeness to the side's claim + topic_weight x relevance to the topic +
    # stance_weight x stance
    topic_weight = 1.0
    stance_weight = 0.5
    shortlist = 24  # translation search: rows whose key clauses are compared
    # NLI model (if loaded): checks the opposition of the best nli_shortlist x k candidates
    # of each side against both statements; flags translations that contradict each other
    nli_shortlist, nli_weight, nli_conflict = 4, 1.0, 0.5
    # topics outside the lexicon: words per translation found by meaning (see _expand)
    expand_terms, expand_span, expand_context = 6, 0.15, 2.0
    use_lexicon = True  # False: every topic is handled by meaning only (for evaluation)
    # subjects the Bible doesn't name (see `absence`): score > absent_margin, a close call
    # (within absent_band) settled by a capable LLM; the vocabulary's part of the score is
    # vocabulary_weight x how far the least covered subject word is from vocabulary_match
    absent_margin, absent_band = 0.0, 0.05
    vocabulary_weight, vocabulary_match = 0.3, 0.87
    # related themes: at most related_themes sections (a frame included), themes scoring
    # within theme_span of the best; the frame when the question asks whether something is
    # right (frame_threshold); passages ranked by closeness to the theme + subject_weight x
    # closeness to the subject, anchor_bonus for a theme's anchor passages
    related_themes, theme_span, frame_threshold = 3, 0.12, 0.42
    subject_weight, anchor_bonus = 0.5, 0.2

    def __init__(self, rag):
        self.rag = rag
        # per question, computed once: what it is about (subject, words), whether it is named
        self._abouts: dict[str, tuple[np.ndarray, list[str]]] = {}
        self._absences: dict[str, Absence] = {}

    @property
    def embedder(self):
        return self.rag.embedder

    @property
    def semantics(self):
        return self.rag.semantics

    def plan(
        self,
        question: str,
        language: str,
        k: int | None = None,
        versions: list[str] | None = None,
        task: str | None = None,
    ) -> Plan:
        """Run the question's pipeline: the one `task` asks for (general, contradiction,
        translation, related), else the one routing chooses."""
        if task not in (None, *INTENTS):
            raise ValueError(f"Unknown task {task!r}; choose from {', '.join(INTENTS)}")
        if not question.strip():
            raise ValueError("The question is empty")
        versions = list(versions or self.rag.indexes)
        unknown = set(versions) - self.rag.indexes.keys()
        if unknown:
            raise ValueError(f"Unknown version(s): {', '.join(sorted(unknown))}")
        k = k or self.rag.settings.top_k
        qvec = self.embedder.embed([question])[0]
        route = Route(task, "as requested") if task else self.route(question, qvec)
        plan = self.pipeline(route.intent)(question, language, k, versions, qvec)
        plan.reason = route.reason
        return plan

    def pipeline(self, intent: str):
        """The pipeline of a kind of question: (question, language, k, versions, qvec) -> Plan."""
        return {
            GENERAL: self.general,
            CONTRADICTION: self.contradiction,
            TRANSLATION: self.translation,
            RELATED: self.related,
        }[intent]

    # ---------------------------------------------------------------------- routing
    def route(self, question: str, qvec: np.ndarray) -> Route:
        """By meaning: the kind of question whose examples are closest (semantic.py). A close
        call goes to the LLM when a capable one is loaded."""
        intent, margin, scores = self.semantics.route(qvec)
        # a subject the Bible never names: its related themes, however it is asked ("Are
        # there contradictions about video games?" has no passages for or against). Not for
        # a comparison of translations, nor a question about the verses it cites.
        if intent != TRANSLATION and not references(question):
            absence = self.absence(question, qvec)
            if absence.absent:
                return Route(RELATED, absence.reason, scores)
        ranked = sorted(scores, key=scores.get, reverse=True)
        reason = (
            f"closest to {intent} questions ({scores[intent]:.2f} vs {ranked[1]} "
            f"{scores[ranked[1]]:.2f})"
        )
        if margin < self.route_margin and GENERAL in ranked[:2]:  # a close call
            if (llm := self._llm()) is not None:
                decided = self._llm_intent(llm, question)
                if decided:
                    why = f"{llm.label} chose between {ranked[0]} and {ranked[1]}"
                    return Route(decided, why, scores)
            # no one to settle it: a direct search is the safe answer (it shows every
            # translation too); the other tasks can still be asked for explicitly
            return Route(GENERAL, f"unsure between {ranked[0]} and {ranked[1]}", scores)
        return Route(intent, reason, scores)

    def _llm(self):
        """The chat model, if one is loaded that can follow planning instructions."""
        chain = self.rag.chain
        if chain is None or not self.rag.settings.llm_planning:
            return None
        try:  # a small model can't plan: not even loaded for that
            model, _ = chain.model_for(lambda m: "compact" if m.compact else None)
        except RuntimeError:
            return None
        return model if model is not None and not model.compact else None

    def _ask(self, llm, system: str, user: str, tokens: int) -> str:
        try:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            return llm.generate(messages, max_new_tokens=tokens)
        except Exception as e:  # noqa: BLE001 - planning help is optional
            log.warning("LLM planning call failed: %s", e)
            return ""

    def _llm_intent(self, llm, question: str) -> str | None:
        reply = self._ask(llm, ROUTER_PROMPT, question, 8).lower()
        found = [i for i in ROUTES if i in reply]
        return found[0] if len(found) == 1 else None

    def _about(self, question: str, qvec: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """What the question is about: its subject (the question minus its task) and its
        subject words, by meaning (semantic.py); computed once per question."""
        if question not in self._abouts:
            sem = self.semantics
            words = sem.subject_words(without_references(question), qvec)
            self._abouts[question] = (sem.subject(qvec), words)
        return self._abouts[question]

    def _subject(self, question: str, qvec: np.ndarray):
        """(lexicon topic or None, subject vector, subject words when there is no topic)."""
        subject, words = self._about(question, qvec)
        topic = self.topic_of(question, qvec) if self.use_lexicon else None
        return topic, subject, [] if topic else words

    def topic_of(self, question: str, qvec: np.ndarray) -> Topic | None:
        """The lexicon topic a question is about (semantic.topic), kept when the question
        surely is about it (closer than any question about another subject comes) or when its
        subject words are the topic's own words: "¿Hay contradicciones sobre el sexo?" comes
        near slavery by meaning, but "sexo" is none of its words (esclavo, siervo...), while
        "¿Se contradice la Biblia sobre la avaricia?" is as near to money and says *avaricia*."""
        sem = self.semantics
        subject, words = self._about(question, qvec)
        topic, closeness = sem.topic(subject)
        if topic is None or closeness >= sem.topic_certainty or not words:
            return topic
        return topic if self._topic_match(topic, words) >= self.vocabulary_match else None

    def _topic_match(self, topic: Topic, words: list[str]) -> float:
        """How close the subject words come to the topic's words in the Bible's vocabulary
        (its terms; "esclav*" -> esclavo, esclavos...): 1 for a subject word that shares its
        stem with one of them, else the closest by meaning (any language)."""
        keys = [k for w in words for k in tokens(fold(w)) if len(k) > 2]  # "war" counts
        stems = [fold(t.rstrip("*?")) for t in topic.terms("en") + topic.terms("es")]
        own, vectors = [], []
        for ws, vs in self._vocabularies():
            ids = [i for i, w in enumerate(ws) if any(w.startswith(s) for s in stems)]
            own += [ws[i] for i in ids]
            vectors += [vs[i] for i in ids]
        if not keys or not own:
            return 0.0
        if any(k.startswith(w) or w.startswith(k) for k in keys for w in own):
            return 1.0
        return float((self.embedder.embed(keys) @ np.array(vectors).T).max())

    def _vocabularies(self) -> list[tuple[list[str], np.ndarray]]:
        """The Bible's vocabulary: the words of the first translation of each language,
        embedded with its index."""
        found = [
            self.rag.indexes[v].vocabulary(self.embedder)
            for v in self.primary(list(self.rag.indexes))
        ]
        return [(ws, vs) for ws, vs in found if ws]

    # ---------------------------------------------------------------------- helpers
    def _language(self, v: str) -> str:
        return VERSIONS[v].language

    def _unit(self, texts: list[str]) -> np.ndarray:
        v = self.embedder.embed(texts).mean(axis=0)
        return v / max(float(np.linalg.norm(v)), 1e-9)

    def _topic_notes(self, plan: Plan) -> Plan:
        """Add the topic's translation notes, and the passages they cite as sources (so they
        can be checked, and an LLM citing them is verified)."""
        if not plan.topic:
            return plan
        for note in plan.topic.notes:
            if note.versions and not set(note.versions) & set(plan.versions):
                continue
            text = getattr(note, plan.language)
            plan.notes.append(text)
            for book, chapter, first, last in references(text):
                for v in plan.versions:
                    index = self.rag.indexes[v]
                    if (i := index.at(book, chapter, first, last)) is not None:
                        plan.extra.append(Hit(index.passages[i], 0.0))
        return plan

    def searchable(self, versions) -> list[str]:
        """The versions searched by meaning: those in a language the embedder knows (the
        bundled one doesn't know Hebrew or Greek: the originals are then shown as parallels
        of what is found, verse for verse)."""
        return [v for v in versions if embeddable(self._language(v), self.embedder)]

    def primary(self, versions) -> list[str]:
        """One searchable version per language (the first given): candidates are gathered and
        scored in these; the other translations join as parallels (same passages, aligned),
        which costs nothing extra per version."""
        chosen, languages = [], set()
        for v in self.searchable(versions):
            if self._language(v) not in languages:
                languages.add(self._language(v))
                chosen.append(v)
        return chosen

    def _originals(self, versions) -> set[str]:
        return {v for v in versions if VERSIONS[v].original}

    def _complete(self, rows, versions, vector, terms=None) -> None:
        """Key clauses of the passages found (which fix the row's key verses), then every other
        version's parallel passage for those verses and its matching clause; an original's
        key is exactly those verses."""
        originals = self._originals(versions)
        kw = dict(terms=terms, language_of=self._language, exclude=originals)
        choose_keys(rows, self.embedder, vector, **kw)
        fill_parallels(rows, self.rag.indexes, versions, vector, self.embedder, plain=originals)
        choose_keys(rows, self.embedder, vector, **kw)
        original_keys(rows, originals)

    def _original_words(self, rows, topic, language: str, every_topic: bool = False) -> list:
        """Name the original word behind the topic's words in each row: the KJV's Strong's
        numbers for its words in the key verses, the Strong's dictionary entry, and the form
        in the Hebrew / Greek text when loaded ("“hell” = H7585 לִשְׁאוֹל she'ôl"). Returns
        the codes found per row (for the sides of a contradiction)."""
        kjv, lexicon = self.rag.indexes.get("kjv"), self.rag.strongs
        if kjv is None or not lexicon or (topic is None and not every_topic):
            return []
        topics = TOPICS if every_topic or topic is None else [topic]
        found = []
        for row in rows:
            codes = {}
            hit = row.hits.get("kjv")
            if hit is not None and "kjv" in row.key:
                first, last = row.verses or key_verses(row, "kjv")
                tags = kjv.verse_tags(row.book, row.chapter, first, last)
                words = set()
                for t in topics:
                    rx = compile_terms(t.terms("en"))
                    if rx is not None:
                        words |= {m.lower() for m in rx.findall(fold(row.key["kjv"]))}
                for text, tag_codes in tags:
                    if words & set(tokens(text)) or any(w in fold(text) for w in words):
                        for code in tag_codes:
                            if code in lexicon:
                                codes.setdefault(code, text)
            if codes:
                parts = [self._word_note(code, text, row) for code, text in codes.items()]
                row.notes.append(f"{_TEXT[language]['originals']}: " + "; ".join(parts))
            found.append(list(codes))
        return found

    def _word_note(self, code: str, word: str, row: Row) -> str:
        entry = self.rag.strongs[code]
        original = entry["word"]
        source = self.rag.indexes.get("wlc" if code[0] == "H" else "sblgnt")
        if source is not None and row.verses:  # the word as written in this verse
            for text, codes in source.verse_tags(row.book, row.chapter, *row.verses):
                if code in codes:
                    original = text.strip(",.;:·")
                    break
        others = [r for r in entry["kjv"].split(", ") if r and r.lower() not in word.lower()]
        also = f" (KJV elsewhere: {', '.join(others[:4])})" if others else ""
        return f"“{word}” = {code} {original} {entry['translit']}{also}"

    def _side_words(self, found: list, sides, language: str) -> list[str]:
        """Which original words each side of a contradiction uses, and which both use."""
        t, lexicon = _TEXT[language], self.rag.strongs
        counts = [collections.Counter(c for row in side for c in row) for side in found]
        if not any(counts):
            return []
        name = lambda c: f"{c} {lexicon[c]['translit']}"  # noqa: E731
        parts = [
            f"{title}: " + (", ".join(f"{name(c)} ×{n}" for c, n in count.most_common(4)) or "—")
            for title, count in zip(sides, counts, strict=True)
        ]
        shared = sorted(set(counts[0]) & set(counts[1]))
        verdict = (
            t["same_words"].format(words=", ".join(name(c) for c in shared))
            if shared
            else t["different_words"]
        )
        return [f"{t['side_words']}: " + "; ".join(parts) + f". {verdict}"]

    def _concept_notes(self, rows, topic: Topic | None, language: str) -> None:
        if not topic or topic.key == "adhoc":
            return
        for row in rows:
            if len(row.hits) > 1 and (
                note := concept_note(
                    row, topic, self._language, lambda v: VERSIONS[v].label, language
                )
            ):
                row.notes.append(note)

    # ---------------------------------------------------------------------- general
    def general(
        self, question, language, k, versions, qvec, topic=None, remarks=(), concept_notes=True
    ) -> Plan:
        """Direct search, then every translation's parallel of the best passages."""
        topic = topic or self.topic_of(question, qvec)
        searched = self.searchable(versions)
        hits = self.rag.search(question, k, searched, self._keywords(question, qvec, topic))
        label_to_id = {VERSIONS[v].label: v for v in versions}
        rows = group([(label_to_id[h.passage.version], h) for h in hits])[:k]
        self._complete(rows, versions, qvec)
        if concept_notes:
            self._concept_notes(rows, topic, language)
            self._meaning_notes(rows, language)
            self._original_words(rows, topic, language)
        title = _TEXT[language]["by_version"]
        return Plan(
            GENERAL,
            question,
            language,
            versions,
            [Section(title, rows)],
            topic,
            queries=[question],
            remarks=list(remarks),
        )

    def _keywords(self, question: str, qvec: np.ndarray, topic: Topic | None) -> str:
        """The texts' own words for what is asked, added to the keyword search, so that
        "¿Es pecado emborracharse?" finds the passages on drunkenness rather than every
        passage on sin: for a lexicon topic, the terms of its concept closest to the subject
        (drunkenness, not wine in general), prefixes expanded to each text's words ("embriag*"
        -> embriaguez, embriagarse...; homographs such as "vino" = came left out); else the
        Bible's word for each subject word it words otherwise ("glotonería" -> glotón)."""
        subject, words = self._about(question, qvec)
        if topic is None:
            found = self._bible_words(words)
            near = (
                b for w, similarity, b in found if b != w and similarity >= self.vocabulary_match
            )
            return " ".join(dict.fromkeys(near))
        texts = [
            f"{c.gloss}: {', '.join(t.rstrip('*?') for t in c.en + c.es)}" for c in topic.concepts
        ]
        concept = topic.concepts[int(np.argmax(self.embedder.embed(texts) @ subject))]
        terms, versions = [], self.searchable(list(self.rag.indexes))
        for term in (t for t in concept.en + concept.es if not t.endswith("?")):
            if not term.endswith("*"):
                terms.append(term)
                continue
            for v in versions:
                terms += [
                    w for w in self.rag.indexes[v].bm25.terms if w.startswith(fold(term[:-1]))
                ]
        return " ".join(dict.fromkeys(terms))

    # ---------------------------------------------------------------- contradiction
    def contradiction(self, question, language, k, versions, qvec) -> Plan:
        sem = self.semantics
        topic, subject, words = self._subject(question, qvec)
        claims = None
        if topic is None:
            if not words:  # "are there contradictions in the Bible?": no topic to compare
                remarks = [_TEXT[language]["no_topic"]]
                return self.general(question, language, k, versions, qvec, remarks=remarks)
            expansions = self._expand(words, subject, self.primary(versions))
            topic = adhoc_topic(words, language, expansions)
            claims = self._llm_claims(question, topic)
        claims = claims or self._claims(topic)
        vec_a = self._unit([claims["en"][0], claims["es"][0]])
        vec_b = self._unit([claims["en"][1], claims["es"][1]])
        vec_t = self._unit([self._topic_text(topic, "en"), self._topic_text(topic, "es")])
        if topic.key == "adhoc":  # the question's own subject, and its two poles by meaning
            vec_t = unit(vec_t + subject)
            pole_a, pole_b = sem.polarize(vec_t)
            vec_a, vec_b = unit(vec_a + pole_a), unit(vec_b + pole_b)
        if sem.opposed(vec_a, vec_b) < 0 and not topic.claims:  # generic: A must approve
            vec_a, vec_b = vec_b, vec_a

        rows, searched = self._topic_rows(
            topic, self.primary(versions), vec_t, vec_a + vec_b, claims, k
        )
        self._score_stance(rows, topic, vec_a, vec_b, vec_t, searched)
        checked = self._nli_stance(rows, claims, k)  # opposition checked by an NLI model
        pool = checked or rows
        side_a = self._pick([r for r in pool if r.stance > self.margin], k, self._side_rank(0))
        side_b = self._pick([r for r in pool if r.stance < -self.margin], k, self._side_rank(1))
        chosen = side_a + side_b
        self._complete(chosen, versions, vec_t)  # keys are set: parallels follow them
        sides = (topic.sides or GENERIC_SIDES)[language]
        self._concept_notes(chosen, topic, language)
        self._meaning_notes(chosen, language)
        found = [self._original_words(s, topic, language) for s in (side_a, side_b)]
        t = _TEXT[language]
        how = (
            t["checked_nli"].format(model=self.rag.nli.model_name)
            if checked
            else t["checked_embeddings"]
        )
        return self._topic_notes(
            Plan(
                CONTRADICTION,
                question,
                language,
                versions,
                [Section(sides[0], side_a), Section(sides[1], side_b)],
                topic,
                queries=list(claims[language]),
                checks=[how],
                notes=self._side_words(found, sides, language),
            )
        )

    def _expand(self, words: list[str], subject: np.ndarray, versions) -> dict[str, list[str]]:
        """The words each translation uses for a subject outside the lexicon, by meaning: its
        vocabulary's nearest words to the subject (within `expand_span` of the best),
        ranked by how close the passages they occur in are to the subject (which discards
        look-alikes: "pork" -> "swine", not "porch"). {language: words}."""
        # the subject by meaning (the question minus its task) and by its words: each guards
        # against the other's mistakes (a task word left among the words, or vice versa)
        aim = unit(subject + self.embedder.embed([" ".join(words)])[0]) if words else subject
        found: dict[str, dict[str, float]] = {}
        for v in versions:
            index = self.rag.indexes[v]
            vocab, vectors = index.vocabulary(self.embedder)
            if not vocab:
                continue
            sims = vectors @ aim
            top = np.argsort(-sims)[:40]
            ranked = found.setdefault(self._language(v), {})
            for i in top:
                if sims[i] < sims[top[0]] - self.expand_span:
                    break
                docs = index.passages_with(vocab[i])
                context = float(np.mean(index.vectors[docs] @ subject)) if docs else 0.0
                score = float(sims[i]) + self.expand_context * context
                ranked[vocab[i]] = max(score, ranked.get(vocab[i], score))
        return {
            lang: sorted(scores, key=scores.get, reverse=True)[: self.expand_terms]
            for lang, scores in found.items()
        }

    def _llm_claims(self, question: str, topic: Topic) -> dict | None:
        """The two opposite statements for a topic outside the lexicon, written by the LLM,
        checked to be opposite in meaning (semantic opposition axis)."""
        llm = self._llm()
        if llm is None:
            return None
        reply = self._ask(llm, CLAIMS_PROMPT, question, 300)
        try:
            data = json.loads(re.search(r"\{.*\}", reply, re.S)[0])
            claims = {
                lang: (str(data[f"for_{lang}"]), str(data[f"against_{lang}"]))
                for lang in ("en", "es")
            }
            names = {lang: str(data[f"topic_{lang}"]).strip() for lang in ("en", "es")}
        except (TypeError, KeyError, ValueError):
            return None
        vectors = self.embedder.embed([claims["en"][0], claims["en"][1]])
        if float(vectors[0] @ vectors[1]) > 0.97:  # not opposite statements
            return None
        if self.rag.nli is not None:  # the NLI model must see them as contradicting
            both = self.rag.nli.scores(list(claims["en"]), list(claims["en"][::-1]))
            if float(both[:, 1].mean()) < self.nli_conflict:
                return None
        if self.semantics.opposed(vectors[0], vectors[1]) < 0:  # written the other way round
            claims = {lang: pair[::-1] for lang, pair in claims.items()}
        if all(names.values()):
            topic.name.update(names)
        return claims

    def _claims(self, topic: Topic) -> dict:
        if topic.claims:
            return topic.claims
        return {
            lang: tuple(_upper_first(c.format(t=topic.name[lang])) for c in pair)
            for lang, pair in GENERIC_CLAIMS.items()
        }

    @staticmethod
    def _topic_text(topic: Topic, language: str) -> str:
        glosses = ", ".join(c.gloss for c in topic.concepts)
        words = ", ".join(t.rstrip("*?") for t in topic.terms(language)[:8])
        return f"{topic.name[language]}: {glosses}; {words}".strip("; ")

    def _topic_rows(self, topic, versions, vec_t, claims_vec, claims, k):
        """Rows of passages naming the topic (its terms, per language), best `pool_cap` per
        version by closeness to the claims and the topic (`vec_t`, also their shown score),
        plus a semantic search for each claim. Returns the rows and the ids of the hits
        found by the claim searches."""
        items = []  # (priority, version, hit, confirmed)
        languages = {self._language(v) for v in versions}
        for v in versions:
            index, lang = self.rag.indexes[v], self._language(v)
            every, strong = (
                compile_terms(topic.terms(lang)),
                compile_terms(topic.terms(lang, False)),
            )
            if every is None:
                continue
            context = compile_terms(topic.context_terms(lang))
            ids = [
                i
                for i, f in enumerate(index.folded)
                if every.search(f) and (context is None or context.search(f))
            ]
            if not ids:
                continue
            vectors = index.vectors[ids]
            scores, relevance = vectors @ (claims_vec + vec_t), vectors @ vec_t
            for j in np.argsort(-scores)[: self.pool_cap]:
                i = ids[j]
                is_strong = strong is not None and bool(strong.search(index.folded[i]))
                hit = Hit(index.passages[i], float(relevance[j]))
                items.append((float(scores[j]), v, hit, is_strong))
        # semantic search for each of the two statements: finds what the terms miss ("every
        # creature of God is good"). Generic statements ("X is good") only drift off topic,
        # so with those this is only done when few passages name the topic
        searched = set()
        if topic.claims or len(items) < 4 * k:
            for v in versions:
                lang = self._language(v)
                every, strong = (
                    compile_terms(topic.terms(lang)),
                    compile_terms(topic.terms(lang, False)),
                )
                for claim in claims[lang]:
                    vector = self.embedder.embed([claim])[0]
                    hits = self.rag.indexes[v].search(vector, 2 * k, claim)
                    best = max((h.score for h in hits), default=0.0)
                    for h in hits:
                        if h.score < 0.5 * best:  # a keyword-only match, far from the claim
                            continue
                        folded = fold(h.passage.text)
                        # found by meaning, unless its only link to the topic is a homograph
                        confirmed = (strong is not None and bool(strong.search(folded))) or not (
                            every is not None and every.search(folded)
                        )
                        searched.add(id(h))
                        items.append((h.score, v, h, confirmed))
        items.sort(key=lambda x: (x[3], x[0]), reverse=True)  # confirmed matches first
        strong_of = {id(h): s for _, _, h, s in items}
        rows = group([(v, h) for _, v, h, _ in items])
        if len(languages) > 1:  # a homograph match ("vino" = came) needs a confirming parallel
            rows = [r for r in rows if any(strong_of[id(h)] for h in r.hits.values())]
        return rows, searched

    def _score_stance(self, rows, topic, vec_a, vec_b, vec_t, searched=frozenset()):
        """Per passage: its key clause (among the clauses naming the topic; any clause for
        passages found by a claim search), the one closest to either statement and to the
        topic. Its stance combines, averaged over translations:
        - which statement it is closer to in meaning (a - b);
        - where it falls on the opposition axis (approving vs condemning, semantic.py),
          relative to the other candidates, weighted by how much the two statements
          themselves differ along that axis (wine good / wine a mocker: much; faith / works:
          little)."""
        sem = self.semantics
        jobs = []  # (row, version, candidate clauses)
        for row in rows:
            for v, h in row.hits.items():
                rx = compile_terms(topic.terms(self._language(v)))
                cs = clauses(h.passage.text)
                if rx is not None and id(h) not in searched:
                    cs = [c for c in cs if rx.search(fold(c))] or cs
                jobs.append((row, v, cs))
        texts = [c for _, _, cs in jobs for c in cs]
        if not texts:
            return
        vectors, start = self.embedder.embed(texts), 0
        per_row: dict[int, list] = {}
        for row, v, cs in jobs:
            vs = vectors[start : start + len(cs)]
            start += len(cs)
            a, b, t = vs @ vec_a, vs @ vec_b, vs @ vec_t
            best = int(np.argmax(np.maximum(a, b) + t))
            row.key[v] = cs[best]
            approval = float(sem.approval(vs[best]))
            per_row.setdefault(id(row), []).append((a[best], b[best], t[best], approval))
        evaluative = max(0.0, float(unit(vec_a - vec_b) @ sem.axis))
        scored = [r for r in rows if id(r) in per_row]
        means = {id(r): np.mean(per_row[id(r)], axis=0) for r in scored}
        baseline = float(np.mean([m[3] for m in means.values()])) if means else 0.0
        for row in scored:
            a, b, t, approval = (float(x) for x in means[id(row)])
            row.fit, row.score = (a, b), t
            row.features = {
                "a": a,
                "b": b,
                "topic": t,
                "approval": approval - baseline,
                "evaluative": evaluative,
            }
            row.stance = self._stance(row.features)

    def _stance(self, f: dict) -> float:
        return (
            f["a"]
            - f["b"]
            + self.axis_weight * f["evaluative"] * f["approval"]
            + self.nli_weight * f.get("nli", 0.0)
        )

    def _nli_stance(self, rows: list[Row], claims: dict, k: int) -> list[Row]:
        """With an NLI model: the best candidates of each side are checked against both
        statements (entailed / contradicted), which settles their side. Returns them (empty
        without a model)."""
        nli = self.rag.nli
        if nli is None:
            return []
        shortlist = []
        for side, sign in ((0, 1), (1, -1)):
            ranked = [r for r in rows if sign * r.stance > self.margin]
            ranked.sort(key=self._side_rank(side), reverse=True)
            shortlist += ranked[: self.nli_shortlist * k]
        premises, hypotheses = [], []
        for row in shortlist:
            v = next(v for v in row.hits if v in row.key)
            premises += [row.key[v]] * 2
            hypotheses += list(claims[self._language(v)])
        scores = nli.scores(premises, hypotheses)
        for i, row in enumerate(shortlist):
            (entail_a, contra_a), (entail_b, contra_b) = scores[2 * i], scores[2 * i + 1]
            row.features["nli"] = float((entail_a - contra_a) - (entail_b - contra_b))
            row.stance = self._stance(row.features)
        return shortlist

    def _meaning_notes(self, rows: list[Row], language: str) -> None:
        """With an NLI model: flag a passage whose translations contradict each other (the
        same verses meaning different things), checked both ways."""
        nli = self.rag.nli
        keyed = lambda row: [v for v in row.key if v in row.hits and not VERSIONS[v].original]  # noqa: E731
        pairs = [(row, keyed(row)[0], b) for row in rows for b in keyed(row)[1:]]
        if nli is None or not pairs:
            return
        forward = nli.scores([r.key[a] for r, a, b in pairs], [r.key[b] for r, a, b in pairs])
        backward = nli.scores([r.key[b] for r, a, b in pairs], [r.key[a] for r, a, b in pairs])
        for (row, a, b), f, g in zip(pairs, forward, backward, strict=True):
            contradiction = (float(f[1]) + float(g[1])) / 2
            if contradiction >= self.nli_conflict:
                row.notes.append(
                    _TEXT[language]["meaning_differs"].format(
                        a=VERSIONS[a].label, b=VERSIONS[b].label, p=contradiction
                    )
                )

    def _side_rank(self, side: int):
        """A side's passages are ranked by closeness to its own claim (the semantic search
        for that statement), plus relevance to the topic and the passage's stance."""
        sign = 1 if side == 0 else -1
        return lambda r: (
            r.fit[side] + self.topic_weight * r.score + self.stance_weight * sign * r.stance
        )

    @staticmethod
    def _pick(rows: list[Row], k: int, rank=None) -> list[Row]:
        """Best k rows (by `rank`, default: score), one per chapter."""
        ranked = sorted(rows, key=rank or (lambda r: r.score), reverse=True)
        chosen, chapters = [], set()
        for r in ranked:
            if (r.book, r.chapter) not in chapters:
                chosen.append(r)
                chapters.add((r.book, r.chapter))
            if len(chosen) == k:
                break
        return chosen

    # ------------------------------------------------------------------ translation
    def translation(self, question, language, k, versions, qvec) -> Plan:
        t = _TEXT[language]
        refs = references(question) or self._chapter_verses(question, versions, k)
        topic, _, words = self._subject(question, qvec)
        rows: list[Row] = []
        queries = []
        if refs:  # the verses asked about, in every version
            for book, chapter, first, last in refs[: max(k, 1)]:
                row = Row(book, chapter, verses=(first, last))
                for v in versions:
                    index = self.rag.indexes[v]
                    if (i := index.at(book, chapter, first, last)) is not None:
                        row.hits[v] = Hit(index.passages[i], 1.0)
                if row.hits:
                    row.key = {v: h.passage.verse_text(first, last) for v, h in row.hits.items()}
                    rows.append(row)
            queries = [r.title for r in rows]
        elif topic is not None:  # where the translations word the topic differently
            vec_t = self._unit([self._topic_text(topic, "en"), self._topic_text(topic, "es")])
            rows = self._divergent_rows(topic, versions, vec_t, k)
            queries = [topic.name[language]]
        if not rows:  # no reference, no known topic: direct search on what it is about
            query = " ".join(words) or question
            vec = self.embedder.embed([query])[0]
            plan = self.general(query, language, k, versions, vec, topic, concept_notes=False)
            rows, queries = plan.sections[0].rows, [query]
        for row in rows:
            self._wording_notes(row, topic, language)
        self._meaning_notes(rows, language)
        self._original_words(rows, topic, language, every_topic=topic is None)
        return self._topic_notes(
            Plan(
                TRANSLATION,
                question,
                language,
                versions,
                [Section(t["side_by_side"], rows)],
                topic,
                queries=queries,
            )
        )

    def _chapter_verses(self, question: str, versions, k: int) -> list[tuple]:
        """A whole chapter asked about ("Psalm 23", "Juan 3"): its first k passages, as
        (book, chapter, first verse, last verse), cut as in the first translation loaded."""
        translations = [v for v in versions if not VERSIONS[v].original] or list(versions)
        index = self.rag.indexes[translations[0]]
        found = []
        for book, chapter in chapters(question):
            last = 0
            for i in sorted(index.places.get((book, chapter), [])):
                p = index.passages[i]
                if p.verse_start > last and len(found) < k:  # chunks overlap: skip repeats
                    found.append((book, chapter, p.verse_start, p.verse_end))
                    last = p.verse_end
        return found

    def _divergent_rows(self, topic: Topic, versions, vec_t, k) -> list[Row]:
        """Passages where the translations word the topic differently. The pool is the
        passages naming the topic's core concept (e.g. KJV "hell", RV1909 "infierno") in any
        version; the best `shortlist` rows get their key clauses aligned, and rows whose
        versions express the topic with more distinct concepts come first."""
        core = topic.concepts[0]
        items = []
        for v in versions:
            index = self.rag.indexes[v]
            rx = compile_terms(t for t in core.terms(self._language(v)) if not t.endswith("?"))
            if rx is None:
                continue
            ids = [i for i, f in enumerate(index.folded) if rx.search(f)]
            if ids:
                scores = index.vectors[ids] @ vec_t
                for j in np.argsort(-scores)[: self.pool_cap]:
                    items.append(
                        (float(scores[j]), v, Hit(index.passages[ids[j]], float(scores[j])))
                    )
        items.sort(key=lambda x: x[0], reverse=True)
        rows = group([(v, h) for _, v, h in items])[: self.shortlist]
        for row in rows:
            row.score = max(
                float(s) for s, _, h in items if id(h) in {id(x) for x in row.hits.values()}
            )

        def names_topic(v, clause):
            rx = compile_terms(core.terms(self._language(v)))
            return rx is not None and bool(rx.search(fold(clause)))

        self._complete(rows, versions, vec_t, terms=names_topic)

        def distinct(row: Row) -> int:  # ways the versions word the topic here
            found = renderings(row, topic, self._language)
            return len({frozenset(f) for f in found.values() if f})

        rows.sort(key=lambda r: (distinct(r), r.score), reverse=True)
        return self._pick(rows, k)

    def _wording_notes(self, row: Row, topic: Topic | None, language: str) -> None:
        t = _TEXT[language]
        # the topic asked about, or, for a passage cited without one, every lexicon topic:
        # the passage's own differences ("hell" / "sepulcro" in Psalm 16:10)
        for candidate in [topic] if topic and topic.key != "adhoc" else TOPICS:
            self._concept_notes([row], candidate, language)
        versions = [v for v in row.hits if not VERSIONS[v].original]
        for i, a in enumerate(versions):  # each language's first version vs the others
            if any(self._language(x) == self._language(a) for x in versions[:i]):
                continue
            for b in versions[i + 1 :]:
                if self._language(a) != self._language(b):
                    continue
                la, lb = VERSIONS[a].label, VERSIONS[b].label
                result = word_changes(row.key.get(a, ""), row.key.get(b, ""))
                if result is None:
                    row.notes.append(t["very_different"].format(a=la, b=lb))
                    continue
                changes, style = result
                shown = "; ".join(
                    t["changes"].format(a=f"“{x}”" if x else "—", b=f"“{y}”" if y else "—")
                    for x, y in changes
                )
                if style:
                    shown += ("; " if shown else "") + t["style"].format(n=style)
                if shown:
                    row.notes.append(f"{la} → {lb}: {shown}")

    # ---------------------------------------------------------------------- related
    def absence(self, question: str, qvec: np.ndarray) -> Absence:
        """Whether the Bible names what the question is about, from two kinds of meaning:
        - is the subject (and its words) closer to things the Bible never names than to
          things it does (semantic.absence)?
        - how close is the Bible's own vocabulary to it: every subject word needs a near
          equivalent in a translation ("Egypt", "España", "astrologers" have one; "Argentina",
          "celular", "ChatGPT" don't; `_coverage`; a subject that surely is a lexicon
          topic has its words: "emborracharse" is the alcohol topic's *embriagarse*)?
        The two add up to one score, absent above `absent_margin`; without subject words
        ("¿Está bien jugar videojuegos?") only the meaning speaks, and it must be clear (beyond
        `absent_band`: "Are there contradictions in the Bible?" has no subject to lack). A
        close call goes to a capable LLM if one is loaded. Where the words occur in the texts
        is gathered as evidence for the user, not to decide: KJV "smoking" is a furnace."""
        if question in self._absences:
            return self._absences[question]
        sem = self.semantics
        subject, words = self._about(question, qvec)
        wvec = self.embedder.embed([" ".join(words)])[0] if words else None
        views = (subject,) if wvec is None else (subject, wvec)
        score = float(np.mean(sem.absence_views(subject, wvec)))
        coverage, nearest = self._coverage(words)
        if coverage is not None:
            topic, closeness = sem.topic(subject)
            if topic is not None and closeness >= sem.topic_certainty:
                coverage, nearest = 1.0, ""  # surely a lexicon topic: its words are the texts'
            score += self.vocabulary_weight * (self.vocabulary_match - coverage)
        threshold = self.absent_margin + (0.0 if words else self.absent_band)
        absent, by = score > threshold, "meaning"
        if abs(score - self.absent_margin) < self.absent_band and (llm := self._llm()):
            decided = self._llm_absent(llm, question)
            if decided is not None:
                absent, by = decided, llm.label
        checked, found = self._occurrences(question, words)
        asked = {w.lower(): w for w in re.findall(r"\w+", question)}  # as the user wrote it
        shown = " ".join(asked.get(w, w) for w in words)
        result = Absence(shown, words, score, absent, by, checked, found, views, nearest)
        self._absences[question] = result
        return result

    def _bible_words(self, words: list[str]) -> list[tuple[str, float, str]]:
        """Each subject word (accents folded, stop words out) with the Bible's own word for
        it: (word, 1, word) if a translation has it (also in singular or plural), else its
        nearest word by meaning in the first translation of each language (their vocabularies
        are embedded with the index), as (word, similarity, nearest)."""
        keys = list(dict.fromkeys(t for w in words for t in tokens(fold(w))))
        vocabularies = self._vocabularies()
        if not keys or not vocabularies:
            return []
        known = {w for ws, _ in vocabularies for w in ws}
        found = []
        for key, vec in zip(keys, self.embedder.embed(keys), strict=True):
            forms = {key, f"{key}s", f"{key}es"}  # and its singular: -s, -es removed
            forms |= {key[: -len(end)] for end in ("s", "es") if key.endswith(end)}
            if known & {f for f in forms if len(f) > 2}:
                found.append((key, 1.0, key))
                continue
            best = (-1.0, "")
            for ws, vs in vocabularies:
                sims = vs @ vec
                i = int(np.argmax(sims))
                best = max(best, (float(sims[i]), ws[i]))
            found.append((key, *best))
        return found

    def _coverage(self, words: list[str]) -> tuple[float | None, str]:
        """How well the Bible's vocabulary covers the subject words: the least covered one,
        with its nearest Bible word. (None, "") without subject words."""
        found = self._bible_words(words)
        if not found:
            return None, ""
        _, similarity, nearest = min(found, key=lambda f: f[1])
        return similarity, "" if similarity >= 1.0 else nearest

    def _llm_absent(self, llm, question: str) -> bool | None:
        """The LLM's call on a close one: True when the Bible doesn't mention the subject."""
        reply = self._ask(llm, ABSENCE_PROMPT, question, 4).strip().lower()
        yes, no = reply.startswith(("yes", "sí", "si")), reply.startswith("no")
        return no if yes != no else None

    def _occurrences(self, question: str, words: list[str]) -> tuple[list[str], dict]:
        """The loaded translations in the question's language (their labels) and, for those
        containing the subject's words as a phrase (the last word also as singular or
        plural), how many passages and the first one: {label: (passages, reference)}."""
        from .rag import identify_language

        if not words:
            return [], {}
        language = identify_language(question)
        last = fold(words[-1])
        forms = {last, f"{last}s", f"{last}es", last[:-1] if last.endswith("s") else last}
        forms |= {last[:-2]} if last.endswith("es") else set()
        forms = {f for f in forms if len(f) >= 3}
        head = "".join(rf"{re.escape(fold(w))}\s+" for w in words[:-1])
        alternatives = "|".join(re.escape(f) for f in sorted(forms))
        rx = re.compile(rf"\b{head}(?:{alternatives})\b")
        # each word's keyword-index entry narrows the passages to look at (stop words have
        # none: they narrow nothing)
        keys = [[t for f in sorted(forms) for t in tokens(f)]]
        keys += [tokens(w) for w in words[:-1]]
        keys = [k for k in keys if k]
        checked, found = [], {}
        for v, index in self.rag.indexes.items():
            if self._language(v) != language or VERSIONS[v].original:
                continue
            label = VERSIONS[v].label
            checked.append(label)
            candidates = range(len(index.passages))
            for key in keys:
                docs = set().union(*(index.bm25.docs_of(t).tolist() for t in key))
                candidates = [i for i in candidates if i in docs]
            ids = [i for i in sorted(candidates) if rx.search(index.folded[i])]
            if ids:
                found[label] = (len(ids), index.passages[ids[0]].reference)
        return checked, found

    def _absence_remarks(self, absence: Absence, language: str) -> list[str]:
        """What the texts were checked for, shown above the passages."""
        t = _TEXT[language]
        if not absence.words:
            return []
        if not absence.checked:
            return [t["not_checked"]]
        words = absence.subject
        if absence.found:
            found = ", ".join(
                f"{label} ({t['passages' if n > 1 else 'passage'].format(n=n, example=example)})"
                for label, (n, example) in absence.found.items()
            )
            return [t["in_texts"].format(words=words, found=found)]
        return [t["not_in_texts"].format(words=words, versions=_join(absence.checked, t["and"]))]

    def related(self, question, language, k, versions, qvec) -> Plan:
        """A subject the Bible doesn't name: say so, then the biblical themes closest to it
        (themes.py), each a section with how it relates, its anchor passages and what a
        search for it finds, in every translation. A frame (principles for deciding) is
        added when the question asks whether something is right."""
        t = _TEXT[language]
        absence = self.absence(question, qvec)
        frames = [th for th, score in self.semantics.frames(qvec) if score >= self.frame_threshold]
        themes = self._themes(absence, max(1, self.related_themes - len(frames))) + frames
        per, primary = max(1, k - 1), self.primary(versions)
        sections, used = [], []
        for theme in themes:
            en, es = self.embedder.embed([theme.query["en"], theme.query["es"]])
            rows = self._theme_rows(theme, primary, {"en": en, "es": es}, absence.vector, per, used)
            self._complete(rows, versions, unit(en + es))
            self._meaning_notes(rows, language)
            used += rows
            how = "related_link_frame" if theme.frame or not absence.subject else "related_link"
            note = t[how].format(subject=absence.subject, link=theme.link[language])
            sections.append(Section(theme.title(language), rows, note, theme.key))
        return Plan(
            RELATED,
            question,
            language,
            versions,
            sections,
            queries=[th.query[language] for th in themes],
            remarks=self._absence_remarks(absence, language),
            absence=absence,
            themes=[f"{th.name[language]} ({th.scope[language]})" for th in themes],
        )

    def _themes(self, absence: Absence, room: int) -> list[Theme]:
        """The catalog themes closest to the subject: the best, and the next ones within
        `theme_span` of it (up to `room`)."""
        ranked = self.semantics.theme_scores(*absence.views)
        best = ranked[0][1]
        return [th for th, score in ranked[:room] if score >= best - self.theme_span]

    def _theme_rows(self, theme: Theme, versions, aims: dict, subject, per: int, used):
        """A theme's passages: its anchors, and the passages closest to its statement
        (`aims`: embedded, per language), in each version; ranked by closeness to the
        statement + `subject_weight` x closeness to the subject (anchors: + `anchor_bonus`),
        one per chapter, none already shown."""
        items = []  # (rank, version, hit, anchor verses)
        for v in versions:
            index = self.rag.indexes[v]
            aim = aims.get(self._language(v), aims["en"])
            closeness = index.vectors @ aim
            rank = closeness + self.subject_weight * (index.vectors @ subject)
            for ref in theme.anchors:
                for book, chapter, first, last in references(ref):
                    if (i := index.at(book, chapter, first, last, aim)) is not None:
                        hit = Hit(index.passages[i], float(closeness[i]))
                        items.append((float(rank[i]) + self.anchor_bonus, v, hit, (first, last)))
            for i in np.argsort(-rank)[: 2 * per]:
                items.append((float(rank[i]), v, Hit(index.passages[i], float(closeness[i])), None))
        items.sort(key=lambda x: x[0], reverse=True)
        rows = group([(v, h) for _, v, h, _ in items])
        of = {id(h): (r, verses) for r, _, h, verses in items}
        for row in rows:
            row.score = max(of[id(h)][0] for h in row.hits.values())
            verses = next((of[id(h)][1] for h in row.hits.values() if of[id(h)][1]), None)
            if verses:  # an anchor: its key is exactly those verses
                row.verses = verses
                row.key = {v: h.passage.verse_text(*verses) for v, h in row.hits.items()}
        shown = [(book_index(h.passage.book), h.passage) for r in used for h in r.hits.values()]
        fresh = [
            r
            for r in rows
            if not any(b == r.book and p.chapter == r.chapter and r.overlaps(p) for b, p in shown)
        ]
        return self._pick(fresh, per)
