"""`bible-rag eval`: 16 questions (in English and Spanish) with the chapters that answer them.
Reports the retrieval hit rate per question language x translation and, for answers, how often
a correct passage is cited (verified) and how often something unverifiable is claimed."""

from __future__ import annotations

import time

from .corpus import BOOKS
from .grounding import book_index

_IDX = {en: i for i, (en, _) in enumerate(BOOKS)}

# (English question, Spanish question, gold chapters as (KJV book, chapter))
GOLD = [
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


def evaluate(rag, k: int | None = None, answers: bool = True, progress=None) -> dict:
    rows = []
    for en, es, gold in GOLD:
        chapters = {(_IDX[book], ch) for book, ch in gold}
        for lang, question in (("en", en), ("es", es)):
            hits = rag.search(question, k)
            row = {
                "question": question,
                "language": lang,
                "hit": {},
                "sources": [h.passage.citation for h in hits[:4]],
            }
            for h in hits:
                in_gold = (book_index(h.passage.book), h.passage.chapter) in chapters
                row["hit"][h.passage.version] = row["hit"].get(h.passage.version, False) or in_gold
            if answers:
                t0 = time.perf_counter()
                a = rag.ask(question, k)
                verified = [c for c in a.grounding.references + a.grounding.quotes if c.supported]
                row |= {
                    "model": a.model,
                    "answer": a.text,
                    "seconds": round(time.perf_counter() - t0, 1),
                    "cites_gold": any(c.location in chapters for c in verified),
                    "unverified": [c.text for c in a.grounding.unsupported],
                }
            rows.append(row)
            if progress:
                progress(row)
    return {"summary": summarize(rows), "questions": rows}


def summarize(rows: list[dict]) -> dict:
    def rate(flags):
        flags = list(flags)
        return round(sum(flags) / len(flags), 3) if flags else None

    hit_rate = {}
    for lang in sorted({r["language"] for r in rows}):
        rs = [r for r in rows if r["language"] == lang]
        for version in sorted({v for r in rs for v in r["hit"]}):
            hit_rate[f"{lang}->{version}"] = rate(r["hit"].get(version, False) for r in rs)
        hit_rate[f"{lang}->any"] = rate(any(r["hit"].values()) for r in rs)
    summary = {"questions": len(rows), "retrieval_hit_rate": hit_rate}
    if answered := [r for r in rows if "model" in r]:
        summary["answers"] = {
            "models": {
                m: sum(r["model"] == m for r in answered) for m in {r["model"] for r in answered}
            },
            "cites_gold_passage": rate(r["cites_gold"] for r in answered),
            "with_unverified_claims": rate(bool(r["unverified"]) for r in answered),
            "avg_seconds": round(sum(r["seconds"] for r in answered) / len(answered), 1),
        }
    return summary
