"""`bible-rag eval`: 16 questions (in English and Spanish) with the chapters that answer them.
Reports the retrieval hit rate per question language x translation and, for answers, how often
a correct passage is cited (verified) and how often something unverifiable is claimed.

`bible-rag eval --agent`: how the assistant routes questions (general / contradiction /
translation), and, for contradiction questions on six topics, how many of the passages it
puts on each side are well-known passages for that side (precision of the top k)."""

from __future__ import annotations

import time

from .corpus import BOOKS
from .grounding import book_index
from .semantic import CONTRADICTION, GENERAL

_IDX = {en: i for i, (_, en, _) in enumerate(BOOKS)}

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
        [("Psalms", 23), ("John", 10)],
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


def run_pipeline(planner, intent: str, question: str, language: str, k: int | None):
    """One pipeline's plan for a question, whatever the routing would choose (to measure the
    pipeline itself)."""
    qvec = planner.embedder.embed([question])[0]
    k = k or planner.rag.settings.top_k
    return planner.pipeline(intent)(question, language, k, list(planner.rag.indexes), qvec)


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
            # what the answer shows: the best passages and their parallel in every version
            plan = run_pipeline(rag.planner, GENERAL, question, lang, k)
            row["aligned"] = {
                h.passage.version: any(
                    (book_index(x.passage.book), x.passage.chapter) in chapters
                    for x in plan.sources
                    if x.passage.version == h.passage.version
                )
                for h in plan.sources
            }
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
    aligned = {
        f"{lang}->{version}": rate(
            r["aligned"].get(version, False) for r in rows if r["language"] == lang
        )
        for lang in sorted({r["language"] for r in rows})
        for version in sorted({v for r in rows for v in r.get("aligned", {})})
    }
    summary = {"questions": len(rows), "retrieval_hit_rate": hit_rate}
    if aligned:  # the passages an answer shows, with their parallels in every translation
        summary["aligned_hit_rate"] = aligned
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


# ------------------------------------------------------------------------------ agent
# Questions the routing examples (semantic.INTENT_EXAMPLES) don't contain, in six languages
ROUTING_GOLD = [
    ("Where was Jesus born?", "general"),
    ("¿Dónde nació Jesús?", "general"),
    ("Onde nasceu Jesus?", "general"),
    ("Où est né Jésus ?", "general"),
    ("Wo wurde Jesus geboren?", "general"),
    ("Chi ha costruito l'arca?", "general"),
    ("What does the Bible say about forgiveness?", "general"),
    ("¿Qué dice la Biblia sobre el perdón?", "general"),
    ("Who built the ark and why?", "general"),
    ("What does the Bible say about wine?", "general"),
    ("How should we resolve conflict with a brother?", "general"),
    ("Is it a sin to drink alcohol?", "general"),
    ("are there any paradoxes in the bible according to alcohol?", "contradiction"),
    ("Does the Bible contradict itself about faith and works?", "contradiction"),
    ("Is divorce allowed or forbidden in the Bible?", "contradiction"),
    ("Can anyone see God? Isn't that a contradiction?", "contradiction"),
    ("Is there a tension between James and Paul on works?", "contradiction"),
    ("Some passages praise wine, others condemn it. Why?", "contradiction"),
    ("¿Hay contradicciones en la Biblia sobre el alcohol?", "contradiction"),
    ("¿Se contradice la Biblia sobre la guerra?", "contradiction"),
    ("¿Hay paradojas en la Biblia sobre el dinero?", "contradiction"),
    ("Existem paradoxos na Bíblia sobre o álcool?", "contradiction"),
    ("La Bible se contredit-elle sur la guerre ?", "contradiction"),
    ("Widerspricht sich die Bibel beim Thema Scheidung?", "contradiction"),
    ("Ci sono contraddizioni nella Bibbia sul denaro?", "contradiction"),
    ("How do the translations render Psalm 16:10?", "translation"),
    ("Do the translations differ on hell?", "translation"),
    ("Compare the KJV and the RV1909 on John 3:16", "translation"),
    ("Does 'charity' in 1 Corinthians 13 mean the same in every version?", "translation"),
    ("Is 'hell' translated the same way in the WEB?", "translation"),
    ("¿Cómo traducen las versiones Juan 1:1?", "translation"),
    ("¿Difieren las traducciones sobre el infierno?", "translation"),
    ("¿La KJV y la Reina-Valera dicen lo mismo en Génesis 22:1?", "translation"),
    ("Como as versões traduzem o Salmo 23?", "translation"),
    ("Comment les versions traduisent-elles Jean 3:16 ?", "translation"),
    ("Wie übersetzen die Bibelausgaben Psalm 16,10?", "translation"),
    ("Le traduzioni dicono la stessa cosa in Genesi 22:1?", "translation"),
]

# question -> lexicon topic (None: a subject outside the lexicon), in six languages
TOPIC_GOLD = [
    ("are there any paradoxes in the bible according to alcohol?", "alcohol"),
    ("Existem paradoxos na Bíblia sobre o álcool?", "alcohol"),
    ("Some passages praise wine, others condemn it. Why?", "alcohol"),
    ("Does the Bible contradict itself about drinking beer?", "alcohol"),
    ("La Bible se contredit-elle sur la guerre ?", "war"),
    ("Is killing in self-defence allowed or forbidden?", "war"),
    ("Widerspricht sich die Bibel beim Thema Scheidung?", "divorce"),
    ("¿Es permitido volver a casarse tras separarse?", "divorce"),
    ("Ci sono contraddizioni nella Bibbia sul denaro?", "money"),
    ("Widerspricht sich die Bibel über Reichtum?", "money"),
    ("Does the Bible contradict itself about faith and works?", "faith_works"),
    ("Can anyone see God? Isn't that a contradiction?", "seeing_god"),
    ("Can we see the face of the Lord? Moses did", "seeing_god"),
    ("Are there contradictions in the Bible about eating pork?", "food"),
    ("Can Christians eat shellfish and pork?", "food"),
    ("Does God tempt people or not?", "temptation"),
    ("Is the devil or God the one who tests us?", "temptation"),
    ("Does God change his mind? The Bible seems inconsistent", "repentance"),
    ("Can Christians swear oaths? Some verses say no", "oaths"),
    ("Is anger a sin or not? Jesus got angry", "anger"),
    ("Does the Bible approve of slavery?", "slavery"),
    ("Were servants property in the Old Testament but free in the New?", "slavery"),
    ("Must we keep the sabbath? Paul says otherwise", "sabbath"),
    ("Is lying ever allowed in the Bible?", "lying"),
    ("Do the translations differ on hell?", "hell"),
    ("Does God hate anyone? God is love", "love"),
    ("Does the Bible contradict itself on dancing?", None),
    ("Are there paradoxes about tattoos?", None),
    ("¿Se contradice la Biblia sobre la oración?", None),
    ("Are the resurrection accounts contradictory?", None),
    ("Existem contradições sobre o batismo?", None),
    ("Does the Bible contradict itself about jewelry?", None),
    ("Are there paradoxes about astrology?", None),
    ("¿Se contradice la Biblia sobre la cremación?", None),
    ("Contradictions about the age of the earth?", None),
    ("Are there paradoxes about sleep in the Bible?", None),
    ("Existem contradições sobre os reis e o governo?", None),
    ("Y a-t-il des paradoxes sur la vieillesse ?", None),
    ("Is gluttony condemned or allowed?", None),
    ("Paradoxes about pride and humility?", None),
]

# question -> well-known chapters for the first and for the second claim (KJV book names)
# fmt: off
CONTRADICTION_GOLD = {
    "Are there any paradoxes in the Bible according to alcohol?": (
        "Psalms 104|Ecclesiastes 9|John 2|Proverbs 31|Judges 9|Deuteronomy 14|1 Timothy 5|"
        "Ecclesiastes 10|Zechariah 10|Zechariah 9|Amos 9|Isaiah 55|Genesis 27",
        "Proverbs 20|Proverbs 23|Ephesians 5|Isaiah 5|Isaiah 28|Habakkuk 2|Galatians 5|"
        "1 Corinthians 6|Leviticus 10|Numbers 6|Joel 1|Genesis 9|Proverbs 31|1 Timothy 3|"
        "Judges 13|Jeremiah 35|Luke 21|Romans 13|1 Samuel 1|Titus 2|1 Peter 4|Luke 1|Isaiah 24|"
        "Hosea 4",
    ),
    "Are there contradictions in the Bible about war?": (
        "Deuteronomy 20|Deuteronomy 2|Deuteronomy 7|Joshua 8|Joshua 6|Joshua 10|1 Samuel 15|"
        "Numbers 31|Numbers 21|Ecclesiastes 3|Joel 3|Exodus 17|Judges 1|2 Samuel 5|Psalms 144|"
        "Jeremiah 48|1 Chronicles 5|Exodus 15|Psalms 18",
        "Matthew 5|Exodus 20|Deuteronomy 5|Romans 12|Romans 13|Isaiah 2|Micah 4|Matthew 26|"
        "Psalms 34|Hebrews 12|James 4|Luke 6|1 Peter 3|Psalms 120|Psalms 46|Isaiah 9|"
        "Zechariah 9|Psalms 68",
    ),
    "Are there contradictions in the Bible about eating pork?": (
        "1 Timothy 4|Mark 7|Acts 10|Acts 11|Romans 14|Titus 1|Colossians 2|Genesis 9|"
        "1 Corinthians 10|1 Corinthians 8|Genesis 1",
        "Leviticus 11|Deuteronomy 14|Isaiah 65|Isaiah 66|Acts 10|Daniel 1|Leviticus 20|"
        "Leviticus 7|Ezekiel 4|Acts 15|Judges 13",
    ),
    "Does the Bible contradict itself about faith and works?": (
        "Romans 3|Romans 4|Romans 5|Galatians 2|Galatians 3|Ephesians 2|Titus 3|Romans 11|"
        "Philippians 3|John 3|Acts 16|Romans 9|Romans 10|Acts 13|2 Timothy 1",
        "James 2|Matthew 7|Matthew 16|Revelation 20|Revelation 22|Romans 2|Philippians 2|"
        "Matthew 25|Revelation 2|2 Corinthians 5|1 Peter 1|Titus 2|Ephesians 2",
    ),
    "Is divorce allowed or forbidden in the Bible?": (
        "Deuteronomy 24|Mark 10|Matthew 19|Matthew 5|Ezra 10|1 Corinthians 7|Jeremiah 3",
        "Malachi 2|Matthew 19|Mark 10|Luke 16|Matthew 5|1 Corinthians 7|Romans 7",
    ),
    "Can anyone see God? Isn't that a contradiction?": (
        "Genesis 32|Exodus 24|Exodus 33|Numbers 12|Isaiah 6|Job 42|Matthew 5|Judges 13|"
        "Deuteronomy 34|Judges 6|Genesis 16|Deuteronomy 5|Amos 9|1 Kings 22|Job 19|"
        "Revelation 22|1 John 3|Hebrews 12|Psalms 17",
        "John 1|1 John 4|Exodus 33|1 Timothy 6|John 6|Colossians 1|3 John 1|1 Timothy 1|John 5|"
        "Judges 13|Deuteronomy 4",
    ),
}
# fmt: on


def evaluate_agent(rag, k: int | None = None, progress=None) -> dict:
    """Routing and topic accuracy; side precision of the contradiction search, with the
    lexicon and by meaning only (no LLM needed)."""
    from .agent import Planner

    planner = Planner(rag)
    say = progress or (lambda line: None)
    sem = rag.semantics
    routing, topics, sides = [], [], []
    for question, expected in ROUTING_GOLD:
        got = planner.route(question, rag.embedder.embed([question])[0]).intent
        routing.append({"question": question, "expected": expected, "got": got})
        say(f"route  {'✓' if got == expected else '✗'} {got:13s} {question}")
    for question, expected in TOPIC_GOLD:
        topic, score = sem.topic(sem.subject(rag.embedder.embed([question])[0]))
        got = topic.key if topic else None
        topics.append({"question": question, "expected": expected, "got": got})
        say(f"topic  {'✓' if got == expected else '✗'} {got or '(own words)':13s} {question}")
    for use_lexicon in (True, False):
        planner.use_lexicon = use_lexicon
        for question, gold in CONTRADICTION_GOLD.items():
            plan = run_pipeline(planner, CONTRADICTION, question, "en", k)
            for section, chapters in zip(plan.sections, gold, strict=True):
                wanted = set(chapters.split("|"))
                found = [f"{BOOKS[row.book][1]} {row.chapter}" for row in section.rows]
                precision = sum(c in wanted for c in found) / len(found) if found else 0.0
                sides.append(
                    {
                        "question": question,
                        "lexicon": use_lexicon,
                        "side": section.title,
                        "found": found,
                        "precision": round(precision, 3),
                    }
                )
                how = "lexicon" if use_lexicon else "meaning"
                say(f"side   {precision:.2f} [{how}] {question[:38]:38s} {section.title}: {found}")

    def rate(rows):
        return round(sum(r["expected"] == r["got"] for r in rows) / len(rows), 3)

    def precision(lexicon):
        values = [s["precision"] for s in sides if s["lexicon"] == lexicon]
        return round(sum(values) / len(values), 3)

    summary = {
        "routing_accuracy": rate(routing),
        "topic_accuracy": rate(topics),
        "contradiction_side_precision": precision(True),
        "contradiction_side_precision_meaning_only": precision(False),
    }
    return {"summary": summary, "routing": routing, "topics": topics, "contradictions": sides}
