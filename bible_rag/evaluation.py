"""`bible-rag eval`: 16 questions (in English and Spanish) with the chapters that answer them.
Reports the retrieval hit rate per question language x translation and, for answers, how often
a correct passage is cited (verified) and how often something unverifiable is claimed.

`bible-rag eval --agent`: how the assistant routes questions (general / contradiction /
translation / related), and, for contradiction questions on six topics, how many of the passages
it puts on each side are well-known passages for that side (precision of the top k); for
subjects the Bible never names, how often that is recognised (and never claimed of a subject it
does name) and whether the related themes chosen fit."""

from __future__ import annotations

import time

from .corpus import BOOKS
from .grounding import book_index
from .semantic import CONTRADICTION, GENERAL, RELATED
from .themes import THEME_BY_KEY

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
    # subjects the Bible never names, whatever the question's form (none of them is among
    # semantic.UNNAMED_SUBJECTS)
    ("Is it wrong to wear a smartwatch all day?", "related"),
    ("Can Christians buy Bitcoin?", "related"),
    ("Does the Bible contradict itself about Bitcoin?", "related"),
    ("¿Es malo usar Telegram?", "related"),
    ("É pecado usar o Instagram?", "related"),
    ("Est-ce un péché d'utiliser Instagram ?", "related"),
    ("Ist Vaping eine Sünde?", "related"),
    ("Cosa dice la Bibbia sui robot?", "related"),
    ("¿Debería usar un celular?", "related"),
    ("¿Es pecado usar inteligencia artificial?", "related"),
    ("¿Qué dice la Biblia sobre Argentina?", "related"),  # a country it never names
    ("What does the Bible say about Germany?", "related"),
    ("¿Puedo tomar café?", "related"),
    ("What does the Bible say about India?", "general"),  # Esther 1:1: named
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
    # near a lexicon topic by meaning, but none of its words: not that topic
    ("hay contradicciones sobre el sexo en la biblia", None),  # not slavery
    ("Is polygamy allowed or forbidden in the Bible?", None),  # not divorce
    ("¿Se contradice la Biblia sobre el suicidio?", None),  # not hell
    ("¿Se contradice la Biblia sobre la salvación de los niños?", None),  # not faith / works
    ("¿Se contradice la Biblia sobre la muerte?", "hell"),  # the grave, "the dead know nothing"
    ("¿Se contradice la Biblia sobre la avaricia?", "money"),  # as near, and its own word
    ("Does the Bible contradict itself about killing?", "war"),
]

# Subjects the Bible never names (none among semantic.UNNAMED_SUBJECTS) -> the related themes
# (themes.py) that fit them. Written before measuring; "Should I take out a mortgage?" was
# dropped from it: the KJV does say "We have mortgaged our lands" (Nehemiah 5:3).
RELATED_GOLD = [
    ("Is it wrong to wear a smartwatch all day?", "time|self_control|skill"),
    ("¿Es malo usar Telegram?", "communication|speech|self_control|time|falsehood"),
    ("Is vaping a sin?", "substances|body|self_control"),
    ("What does the Bible say about robots?", "skill|work|idols"),
    ("Is Netflix a sin?", "eyes|play|time|self_control"),
    ("Can Christians buy Bitcoin?", "stewardship|chance|debt"),
    ("Is it OK to fly drones?", "skill|travel|heavens|weapons"),
    ("Should I use Tinder to find a wife?", "marriage|purity"),
    ("Are energy drinks a sin?", "body|substances|self_control"),
    ("What does the Bible say about antidepressants?", "healing|substances|body"),
    ("Are e-sports a waste of time?", "play|time|self_control"),
    ("What does the Bible say about Instagram influencers?", "pride|appearance|speech|eyes"),
    ("Is genetic engineering playing God?", "skill|life|idols|creation_care|wisdom"),
    ("Can I accept a blood transfusion?", "healing|body"),
    ("Est-ce un péché d'utiliser Instagram ?", "pride|communication|eyes|self_control|time"),
    ("Ist Vaping eine Sünde?", "substances|body|self_control"),
    ("Cosa dice la Bibbia sui robot?", "skill|work|idols"),
    ("É pecado usar o TikTok?", "time|self_control|eyes|play|pride"),
    ("Is it wrong to ride in a self-driving car?", "travel|skill"),
    ("What does the Bible say about nuclear power plants?", "skill|creation_care"),
    ("Is fantasy football gambling?", "chance|play"),
    ("¿Qué dice la Biblia sobre China?", "nations|government"),
    ("Is Russia mentioned in the Bible?", "nations|government"),
    ("¿Qué dice la Biblia sobre el comunismo?", "government|stewardship|work"),
    ("¿Es pecado donar órganos?", "healing|body|life"),
]
# Subjects the Bible does name, often in other words, and questions without a subject: never
# "not named in the Bible" (with the direct and contradiction questions of ROUTING_GOLD)
NAMED_GOLD = [
    "What does the Bible say about horses?",
    "Is jewelry allowed?",
    "What does the Bible say about astrology?",
    "What does the Bible say about taxes?",
    "What does the Bible say about fishing?",
    "Who was king David?",
    "What does the Bible say about tattoos?",
    "¿Qué dice la Biblia sobre el ayuno?",
    "What does the Bible say about sleep?",
    "Que dit la Bible sur le pardon ?",
    "Was sagt die Bibel über Geld?",
    "O que a Bíblia diz sobre o casamento?",
    "Cosa dice la Bibbia sulla preghiera?",
    "Are there contradictions in the Bible?",
    "¿Hay contradicciones en la Biblia?",
    "Is it allowed?",
    "Why?",
    "Is Italy in the Bible?",  # Acts 27:1
    "¿Aparece España en la Biblia?",  # Romanos 15:24
    "¿Qué dice la Biblia sobre Asiria?",
    "What does the Bible say about anxiety?",  # not a KJV word; Matthew 6:25-34
    "¿Existen errores en la Biblia?",
]

# Direct questions on whether something is a sin, which must find the passages on that sin
# rather than on sin in general: (English, Spanish, chapters that answer them). Written before
# measuring.
# fmt: off
SIN_GOLD = [
    ("Is it a sin to get drunk?", "¿Es pecado emborracharse?",
     "Ephesians 5|Proverbs 20|Proverbs 23|Galatians 5|1 Corinthians 6|1 Corinthians 5|Isaiah 5|"
     "Isaiah 28|Romans 13|1 Thessalonians 5|1 Peter 4|Luke 21|Habakkuk 2|Genesis 9|Joel 1|"
     "Hosea 4|Proverbs 31|Titus 2|1 Timothy 3|Deuteronomy 21"),
    ("Is it a sin to swear an oath?", "¿Es pecado jurar?",
     "Matthew 5|James 5|Leviticus 19|Leviticus 5|Numbers 30|Deuteronomy 23|Ecclesiastes 5|"
     "Deuteronomy 6|Deuteronomy 10|Hebrews 6|Matthew 23|Matthew 26|Exodus 20|Zechariah 8"),
    ("Is it a sin to lie?", "¿Es pecado mentir?",
     "Exodus 20|Leviticus 19|Proverbs 6|Proverbs 12|Proverbs 19|Proverbs 14|Colossians 3|"
     "Ephesians 4|Revelation 21|Revelation 22|John 8|Psalms 101|Acts 5|Deuteronomy 5|"
     "Zechariah 8|Proverbs 26"),
    ("Is it a sin to be angry?", "¿Está mal enfadarse?",
     "Ephesians 4|Matthew 5|James 1|Proverbs 14|Proverbs 15|Proverbs 16|Proverbs 19|"
     "Proverbs 29|Proverbs 22|Psalms 37|Ecclesiastes 7|Colossians 3|Galatians 5"),
    ("Is stealing wrong?", "¿Está mal robar?",
     "Exodus 20|Ephesians 4|Leviticus 19|Deuteronomy 5|Proverbs 30|Romans 13|1 Corinthians 6|"
     "Matthew 19|Exodus 22|Zechariah 5|Proverbs 29|Romans 2"),
    ("Is adultery a sin?", "¿Es pecado el adulterio?",
     "Exodus 20|Matthew 5|Hebrews 13|Proverbs 6|Proverbs 5|Leviticus 20|1 Corinthians 6|John 8|"
     "Deuteronomy 5|Matthew 19|Galatians 5|Mark 10|Romans 13|James 2"),
    ("Is laziness a sin?", "¿Es malo ser perezoso?",
     "Proverbs 6|Proverbs 10|Proverbs 12|Proverbs 13|Proverbs 19|Proverbs 20|Proverbs 21|"
     "Proverbs 24|Proverbs 26|2 Thessalonians 3|Ecclesiastes 10|Matthew 25|Proverbs 15|"
     "Proverbs 18"),
    ("Is envy a sin?", "¿Es pecado la envidia?",
     "Proverbs 14|Proverbs 23|Proverbs 24|Proverbs 27|James 3|James 4|Galatians 5|"
     "1 Corinthians 13|1 Corinthians 3|Psalms 37|Psalms 73|Exodus 20|Romans 1|Romans 13|Titus 3|"
     "1 Peter 2|Job 5"),
    ("Is gluttony a sin?", "¿Es pecado la glotonería?",
     "Proverbs 23|Proverbs 28|Deuteronomy 21|Philippians 3|Titus 1|Matthew 11|Luke 7|Luke 21|"
     "1 Corinthians 6|1 Corinthians 10|Proverbs 25"),
    ("Is revenge wrong?", "¿Está mal vengarse?",
     "Romans 12|Leviticus 19|Deuteronomy 32|Proverbs 20|Proverbs 24|Matthew 5|"
     "1 Thessalonians 5|1 Peter 3|Hebrews 10|Proverbs 25"),
]
# fmt: on

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
    routing, topics, sides = [], [], []
    for question, expected in ROUTING_GOLD:
        got = planner.route(question, rag.embedder.embed([question])[0]).intent
        routing.append({"question": question, "expected": expected, "got": got})
        say(f"route  {'✓' if got == expected else '✗'} {got:13s} {question}")
    for question, expected in TOPIC_GOLD:
        topic = planner.topic_of(question, rag.embedder.embed([question])[0])
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

    planner.use_lexicon = True
    related = evaluate_related(planner, say)
    sins = []  # the direct search on "is it a sin to…?": passages on that sin, in the top k
    for en, es, chapters in SIN_GOLD:
        wanted = set(chapters.split("|"))
        for language, question in (("en", en), ("es", es)):
            plan = run_pipeline(planner, GENERAL, question, language, k)
            found = [f"{BOOKS[row.book][1]} {row.chapter}" for row in plan.sections[0].rows]
            sins.append({"question": question, "found": found, "hit": bool(wanted & set(found))})
            say(f"sin    {'✓' if sins[-1]['hit'] else '✗'} {question}: {found}")

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
        "sin_question_hit": round(sum(s["hit"] for s in sins) / len(sins), 3),
        **related["summary"],
    }
    return {
        "summary": summary,
        "routing": routing,
        "topics": topics,
        "contradictions": sides,
        "related": related["questions"],
        "sins": sins,
    }


def evaluate_related(planner, say=lambda line: None) -> dict:
    """Subjects the Bible never names: how many are routed to the related themes, how many
    named subjects (and subject-less questions) are wrongly said to be absent, and whether
    the themes chosen fit (at least one of them: hit; the share of them: precision)."""
    embed = planner.embedder.embed
    rows, named = [], []
    for question, fitting in RELATED_GOLD:
        qvec = embed([question])[0]
        route = planner.route(question, qvec).intent
        plan = planner.related(question, "en", 1, list(planner.rag.indexes), qvec)
        chosen = [s.key for s in plan.sections if not THEME_BY_KEY[s.key].frame]
        good = set(fitting.split("|"))
        rows.append(
            {
                "question": question,
                "routed": route,
                "score": round(plan.absence.score, 3),
                "themes": chosen,
                "hit": any(t in good for t in chosen),
                "precision": round(sum(t in good for t in chosen) / max(len(chosen), 1), 3),
            }
        )
        mark = "✓" if route == RELATED else "✗"
        say(f"related {mark} {plan.absence.score:+.2f} {chosen} {question}")
    questions = NAMED_GOLD + [q for q, intent in ROUTING_GOLD if intent in (GENERAL, CONTRADICTION)]
    for question in questions:
        absent = planner.absence(question, embed([question])[0]).absent
        named.append({"question": question, "absent": absent})
        if absent:
            say(f"named   ✗ said to be absent: {question}")

    def share(flags):
        flags = list(flags)
        return round(sum(flags) / len(flags), 3)

    summary = {
        "related_recognised": share(r["routed"] == RELATED for r in rows),
        "named_wrongly_absent": share(n["absent"] for n in named),
        "related_theme_hit": share(r["hit"] for r in rows),
        "related_theme_precision": share(r["precision"] for r in rows),
    }
    return {"summary": summary, "questions": rows + named}
