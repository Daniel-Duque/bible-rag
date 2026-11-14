"""Measure whether the system is useful: `bible-rag eval`.

Each question (asked in English and in Spanish) has the chapters that answer it. We report

* **retrieval hit rate**: share of questions where at least one of the top-k passages of a
  translation lies in a gold chapter, per question language x translation;
* **answers** (unless ``--retrieval-only``): which model answered, whether the answer cites a
  gold passage that the grounding check verified, and how many references/quotes could
  *not* be verified in the retrieved passages (likely hallucinations).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from .corpus import BOOKS
from .grounding import book_index
from .rag import BibleRAG

_IDX = {en: i for i, (en, _) in enumerate(BOOKS)}

# (english question, spanish question, gold chapters as (KJV book name, chapter))
GOLD: list[tuple[str, str, list[tuple[str, int]]]] = [
    (
        "Who built the ark and why?",
        "¿Quién construyó el arca y por qué?",
        [("Genesis", 6), ("Genesis", 7)],
    ),
    (
        "What happened to Jonah inside the great fish?",
        "¿Qué le pasó a Jonás dentro del gran pez?",
        [("Jonah", 1), ("Jonah", 2)],
    ),
    (
        "Who is my shepherd according to the psalm?",
        "¿Quién es mi pastor según el salmo?",
        [("Psalm", 23), ("John", 10)],
    ),
    (
        "Why did God give his only begotten Son?",
        "¿Por qué dio Dios a su Hijo unigénito?",
        [("John", 3), ("1 John", 4)],
    ),
    (
        "What are the ten commandments?",
        "¿Cuáles son los diez mandamientos?",
        [("Exodus", 20), ("Deuteronomy", 5)],
    ),
    ("How did David defeat Goliath?", "¿Cómo venció David a Goliat?", [("1 Samuel", 17)]),
    (
        "What does Paul say about charity that suffers long and is kind?",
        "¿Qué dice Pablo sobre el amor que es sufrido y benigno?",
        [("1 Corinthians", 13)],
    ),
    (
        "What is faith, the substance of things hoped for?",
        "¿Qué es la fe, la certeza de lo que se espera?",
        [("Hebrews", 11)],
    ),
    (
        "Who are blessed according to the Sermon on the Mount?",
        "¿Quiénes son bienaventurados según el Sermón del Monte?",
        [("Matthew", 5), ("Luke", 6)],
    ),
    (
        "How did God create the world in the beginning?",
        "¿Cómo creó Dios el mundo en el principio?",
        [("Genesis", 1), ("Genesis", 2), ("John", 1)],
    ),
    (
        "What happened to the prodigal son who returned to his father?",
        "¿Qué le pasó al hijo pródigo que volvió a su padre?",
        [("Luke", 15)],
    ),
    (
        "How many times should I forgive my brother?",
        "¿Cuántas veces debo perdonar a mi hermano?",
        [("Matthew", 18), ("Luke", 17)],
    ),
    (
        "How should we pray? Teach us the Lord's prayer.",
        "¿Cómo debemos orar? El Padre nuestro.",
        [("Matthew", 6), ("Luke", 11)],
    ),
    (
        "How did Moses and Israel cross the Red Sea?",
        "¿Cómo cruzaron Moisés e Israel el mar Rojo?",
        [("Exodus", 14), ("Exodus", 15)],
    ),
    (
        "How did Jesus rise from the dead on the third day?",
        "¿Cómo resucitó Jesús de los muertos al tercer día?",
        [("Matthew", 28), ("Mark", 16), ("Luke", 24), ("John", 20), ("1 Corinthians", 15)],
    ),
    (
        "Who is my neighbour? The parable of the good Samaritan.",
        "¿Quién es mi prójimo? La parábola del buen samaritano.",
        [("Luke", 10)],
    ),
]


def gold_set(gold: list[tuple[str, int]]) -> set[tuple[int, int]]:
    return {(_IDX[book], chapter) for book, chapter in gold}


@dataclass
class QuestionResult:
    question: str
    language: str
    retrieval_hit: dict[str, bool]  # version label -> hit
    top_sources: list[str]
    model: str | None = None
    answer: str | None = None
    cites_gold: bool | None = None
    unsupported: list[str] = field(default_factory=list)
    seconds: float | None = None


def evaluate(
    rag: BibleRAG,
    k: int | None = None,
    answers: bool = True,
    languages: tuple[str, ...] = ("en", "es"),
    progress=None,
) -> dict:
    results: list[QuestionResult] = []
    for en_q, es_q, gold in GOLD:
        gold_chapters = gold_set(gold)
        for lang, question in (("en", en_q), ("es", es_q)):
            if lang not in languages:
                continue
            hits = rag.search(question, k)
            per_version: dict[str, bool] = {}
            for h in hits:
                in_gold = (book_index(h.passage.book), h.passage.chapter) in gold_chapters
                version = h.passage.version
                per_version[version] = per_version.get(version, False) or in_gold
            r = QuestionResult(question, lang, per_version, [h.passage.citation for h in hits[:4]])
            if answers:
                t0 = time.perf_counter()
                a = rag.ask(question, k)
                r.seconds = round(time.perf_counter() - t0, 1)
                r.model, r.answer = a.model, a.answer
                r.unsupported = [c.text for c in a.grounding.unsupported]
                supported = [c for c in a.grounding.references + a.grounding.quotes if c.supported]
                r.cites_gold = any(c.location in gold_chapters for c in supported)
            results.append(r)
            if progress:
                progress(r)
    return {"summary": summarize(results), "questions": [asdict(r) for r in results]}


def summarize(results: list[QuestionResult]) -> dict:
    summary: dict = {"questions": len(results), "retrieval_hit_rate": {}}
    for lang in sorted({r.language for r in results}):
        rows = [r for r in results if r.language == lang]
        for version in sorted({v for r in rows for v in r.retrieval_hit}):
            hits = [r.retrieval_hit.get(version, False) for r in rows]
            summary["retrieval_hit_rate"][f"{lang}->{version}"] = round(sum(hits) / len(hits), 3)
        any_hit = [any(r.retrieval_hit.values()) for r in rows]
        summary["retrieval_hit_rate"][f"{lang}->any"] = round(sum(any_hit) / len(any_hit), 3)
    answered = [r for r in results if r.model is not None]
    if answered:
        summary["answers"] = {
            "models": {
                m: sum(r.model == m for r in answered) for m in sorted({r.model for r in answered})
            },
            "cites_gold_passage": round(
                sum(bool(r.cites_gold) for r in answered) / len(answered), 3
            ),
            "with_unverified_claims": round(
                sum(bool(r.unsupported) for r in answered) / len(answered), 3
            ),
            "avg_seconds": round(sum(r.seconds or 0 for r in answered) / len(answered), 1),
        }
    return summary
