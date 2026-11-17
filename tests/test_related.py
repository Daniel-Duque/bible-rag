"""Subjects the Bible never names ("Should I use a cellphone?"): recognised as such, answered
with the biblical themes related to them, each shown with how it relates, and never with a
claim that the Bible speaks of the subject itself (agent.Planner.related, themes.py).

The corpus is the real-text one of test_agent.py plus the verses the cellphone question needs:
letters and news from afar (2 John 1:12, Proverbs 25:25), not being mastered (1 Corinthians
6:12, 2 Peter 2:19), deciding on what the Bible doesn't name (1 Corinthians 10:23, Romans
14:23), the use of time, and Genesis 15:17, where the KJV's "smoking" is a furnace.
"""

import gzip
import json

import pytest

from bible_rag import cli, evaluation
from bible_rag.agent import PROMPTS, Planner
from bible_rag.config import ROOT, Settings
from bible_rag.grounding import book_index, references
from bible_rag.index import VersionIndex
from bible_rag.rag import BibleRAG
from bible_rag.semantic import CONTRADICTION, GENERAL, RELATED
from bible_rag.themes import THEME_BY_KEY, THEMES

from .conftest import FAKE_EMBEDDER, FakeLLM, make_chain, passage, with_bible_vocabularies
from .test_agent import CORPUS

# fmt: off
VERSES = {  # (book, chapter, verse, KJV, RV1909 book, RV1909, WEB)
    "letters": ("2 John", 1, 12,
        "Having many things to write unto you, I would not write with paper and ink: but I "
        "trust to come unto you, and speak face to face, that our joy may be full.",
        "2 Juan", "Aunque tengo muchas cosas que escribiros, no he querido comunicarlas por "
        "medio de papel y tinta; mas espero ir á vosotros, y hablar boca á boca, para que "
        "nuestro gozo sea cumplido.",
        "Having many things to write to you, I don’t want to do so with paper and ink, but I "
        "hope to come to you and to speak face to face, that our joy may be made full."),
    "news": ("Proverbs", 25, 25,
        "As cold waters to a thirsty soul, so is good news from a far country.",
        "Proverbios", "Como el agua fría al alma sedienta, así son las buenas nuevas de "
        "lejanas tierras.",
        "Like cold water to a thirsty soul, so is good news from a far country."),
    "mastered": ("1 Corinthians", 6, 12,
        "All things are lawful unto me, but all things are not expedient: all things are "
        "lawful for me, but I will not be brought under the power of any.",
        "1 Corintios", "Todas las cosas me son lícitas, mas no todas convienen: todas las "
        "cosas me son lícitas, mas yo no me meteré debajo de potestad de nada.",
        "“All things are lawful for me,” but not all things are expedient. “All things are "
        "lawful for me,” but I will not be brought under the power of anything."),
    "bondage": ("2 Peter", 2, 19,
        "While they promise them liberty, they themselves are the servants of corruption: for "
        "of whom a man is overcome, of the same is he brought in bondage.",
        "2 Pedro", "Prometiéndoles libertad, siendo ellos mismos siervos de corrupción. Porque "
        "el que es de alguno vencido, es sujeto á la servidumbre del que lo venció.",
        "promising them liberty, while they themselves are bondservants of corruption; for a "
        "man is brought into bondage by whoever overcomes him."),
    "edify": ("1 Corinthians", 10, 23,
        "All things are lawful for me, but all things are not expedient: all things are "
        "lawful for me, but all things edify not.",
        "1 Corintios", "Todo me es lícito, mas no todo conviene: todo me es lícito, mas no "
        "todo edifica.",
        "“All things are lawful for me,” but not all things are profitable. “All things are "
        "lawful for me,” but not all things build up."),
    "faith": ("Romans", 14, 23,
        "And he that doubteth is damned if he eat, because he eateth not of faith: for "
        "whatsoever is not of faith is sin.",
        "Romanos", "Mas el que hace diferencia, si comiere, es condenado, porque no comió por "
        "fe: y todo lo que no es de fe, es pecado.",
        "But he who doubts is condemned if he eats, because it isn’t of faith; and whatever "
        "is not of faith is sin."),
    "time": ("Ephesians", 5, 16,
        "Redeeming the time, because the days are evil.",
        "Efesios", "Redimiendo el tiempo, porque los días son malos.",
        "redeeming the time, because the days are evil."),
    "smoking": ("Genesis", 15, 17,
        "And it came to pass, that, when the sun went down, and it was dark, behold a smoking "
        "furnace, and a burning lamp that passed between those pieces.",
        "Génesis", "Y sucedió que puesto el sol, y ya obscurecido, dejóse ver un horno "
        "humeando, y una antorcha de fuego que pasó por entre los animales divididos.",
        "It came to pass that, when the sun went down, and it was dark, behold, a smoking "
        "furnace and a flaming torch passed between these pieces."),
    # two passages that a question "is it a sin to…?" must not be drawn to just for its "sin"
    "sin": ("Romans", 7, 13,
        "Was then that which is good made death unto me? God forbid. But sin, that it might "
        "appear sin, working death in me by that which is good; that sin by the commandment "
        "might become exceeding sinful.",
        "Romanos", "¿Luego lo que es bueno, á mí me es hecho muerte? No; sino que el pecado, "
        "para mostrarse pecado, por lo bueno me obró la muerte, haciéndose pecado sobremanera "
        "pecante por el mandamiento.",
        "Did then that which is good become death to me? May it never be! But sin, that it "
        "might be shown to be sin, was producing death in me through that which is good; that "
        "through the commandment sin might become exceedingly sinful."),
    "oath": ("Leviticus", 5, 4,
        "Or if a soul swear, pronouncing with his lips to do evil, or to do good, whatsoever "
        "it be that a man shall pronounce with an oath, and it be hid from him; when he "
        "knoweth of it, then he shall be guilty in one of these.",
        "Levítico", "También la persona que jurare, pronunciando con sus labios hacer mal ó "
        "bien, en cualesquiera cosas que el hombre profiere con juramento, y él no lo "
        "conociere; si después lo entiende, será culpado en una de estas cosas.",
        "“‘Or if anyone swears rashly with his lips to do evil or to do good—whatever it is "
        "that a man might utter rashly with an oath, and it is hidden from him—when he knows "
        "of it, then he will be guilty of one of these."),
}
# fmt: on


@pytest.fixture(scope="module")
def related_indexes(meaning, bible_vocabularies):
    corpus = {v: list(ps) for v, ps in CORPUS.items()}
    for book, chapter, verse, kjv, book_es, rv1909, web in VERSES.values():
        corpus["kjv"].append(passage("kjv", book, chapter, verse, verse, kjv))
        corpus["rv1909"].append(passage("rv1909", book_es, chapter, verse, verse, rv1909))
        corpus["web"].append(passage("web", book, chapter, verse, verse, web))
    indexes = {
        v: VersionIndex.build(ps, meaning, {"embedder": FAKE_EMBEDDER}) for v, ps in corpus.items()
    }
    return with_bible_vocabularies(indexes, bible_vocabularies)


@pytest.fixture
def related_rag(tmp_path, meaning, related_indexes):
    settings = Settings(
        index_dir=tmp_path / "index",
        prebuilt_dir=tmp_path / "prebuilt",
        versions=("kjv", "rv1909", "web"),
        top_k=3,
        embedding_chain=(FAKE_EMBEDDER,),
        nli_model="",
    )
    rag = BibleRAG(settings, meaning, dict(related_indexes), make_chain(FakeLLM()), FAKE_EMBEDDER)
    rag.nli = None
    return rag


def section(plan, key):
    return next(s for s in plan.sections if s.key == key)


def titles(s):
    return [row.title for row in s.rows]


# ------------------------------------------------------------------------------ catalog
def test_catalog_is_complete():
    assert len(THEME_BY_KEY) == len(THEMES)  # unique keys
    assert [t.key for t in THEMES if t.frame] == ["conscience"]
    for theme in THEMES:
        assert theme.examples and theme.anchors, theme.key
        for lang in ("en", "es"):
            assert theme.name[lang] and theme.scope[lang] and theme.query[lang], theme.key
            assert theme.link[lang] and not theme.link[lang].endswith("."), theme.key
        for anchor in theme.anchors:  # each one reference, parsed as written
            assert len(references(anchor)) == 1, anchor


def test_anchors_are_in_the_shipped_texts():
    """Every anchor verse exists in the KJV and the RV1909 as shipped (each was also read
    there: it must speak to its theme in its own context)."""
    root = ROOT / "storage" / "prebuilt" / "usem3"
    if not (root / "kjv").exists():
        pytest.skip("no prebuild in this copy")
    for version in ("kjv", "rv1909"):
        with gzip.open(root / version / "passages.jsonl.gz", "rt", encoding="utf-8") as fh:
            passages = [json.loads(line) for line in fh]
        verses = set()
        for p in passages:
            book = book_index(p["book"])
            verses |= {(book, p["chapter"], v) for v in range(p["verse_start"], p["verse_end"] + 1)}
        for theme in THEMES:
            for anchor in theme.anchors:
                ((book, chapter, first, last),) = references(anchor)
                missing = [v for v in range(first, last + 1) if (book, chapter, v) not in verses]
                assert not missing, f"{version}: {anchor} ({theme.key})"


# -------------------------------------------------------------------------- recognition
@pytest.mark.parametrize("question", evaluation.NAMED_GOLD)
def test_named_subjects_and_questions_without_one_are_never_absent(related_rag, question):
    """ "Astrology" (the astrologers of Babylon), "tattoos" (Leviticus 19:28), "taxes": named,
    though often in other words; "Are there contradictions in the Bible?" has no subject."""
    planner = Planner(related_rag)
    absence = planner.absence(question, related_rag.embedder.embed([question])[0])
    assert not absence.absent, absence.score


def test_the_users_question_is_recognised_whatever_its_form(related_rag):
    for question in (
        "Should I use a cellphone?",
        "Are there contradictions in the Bible about cellphones?",  # none: it never names them
        "¿Está bien usar el celular?",
    ):
        plan = related_rag.plan(question)
        assert plan.intent == RELATED, question
        assert plan.reason.startswith("not named in the Bible (by meaning, +")
    assert related_rag.plan("Is it a sin to drink alcohol?").intent == GENERAL  # wine: named


def test_a_close_call_goes_to_the_llm(related_rag, monkeypatch):
    monkeypatch.setattr(Planner, "absent_band", 1.0)  # every call is a close one
    question = "Should I use a cellphone?"
    related_rag.chain = make_chain(FakeLLM(plan_reply="No."))
    plan = related_rag.plan(question)
    assert plan.intent == RELATED and plan.reason.startswith("not named in the Bible (fake decided")
    related_rag.chain = make_chain(FakeLLM(plan_reply="Yes, it does."))
    assert related_rag.plan(question).intent == GENERAL
    related_rag.settings.llm_planning = False  # no LLM: the score decides
    assert related_rag.plan(question).intent == RELATED


@pytest.mark.parametrize(
    "question,nearest",
    [
        ("What does the Bible say about Argentina?", "italia"),  # a country it never names
        ("¿Qué dice la Biblia sobre la inteligencia artificial?", "artifice"),
        ("¿Debería usar un celular?", "telem"),
        ("Should I use ChatGPT?", "helchat"),
    ],
)
def test_no_bible_word_for_the_subject(related_rag, question, nearest):
    """Nothing lists countries or inventions: the Bible's own vocabulary has no word for
    them (its nearest word is shown), while "Egypt", "España" or "astrologers" are its own."""
    plan = related_rag.plan(question)
    assert plan.intent == RELATED
    assert plan.reason.endswith(f"nearest Bible word “{nearest}”)")


@pytest.mark.parametrize(
    "question",
    [
        "What does the Bible say about Egypt?",
        "¿Aparece España en la Biblia?",  # Romanos 15:24
        "What does the Bible say about India?",  # Esther 1:1
        "¿Qué dice la Biblia sobre Roma?",
        "What does the Bible say about anxiety?",  # not a KJV word; Matthew 6 is about it
    ],
)
def test_places_and_things_the_bible_names(related_rag, question):
    assert related_rag.plan(question).intent != RELATED


def test_a_sin_question_searches_for_that_sin(related_rag):
    """ "¿Es pecado emborracharse?" is a direct question about the alcohol topic: the words the
    texts use for its drunkenness concept join the keywords (not "pecado" alone), and a verb
    the RV1909 doesn't use (it says embriagarse, borracho) doesn't make it "not in the Bible"."""
    planner = Planner(related_rag)
    for question, expected in (
        ("¿Es pecado emborracharse?", {"embriagueis", "drunken"}),
        ("¿Puedo emborracharme?", {"embriagueis", "drunken"}),
        ("¿Es pecado jurar?", {"swear", "oath", "juramento", "jurare"}),
    ):
        plan = related_rag.plan(question)
        assert plan.intent == GENERAL, question
        qvec = related_rag.embedder.embed([question])[0]
        words = set(planner._keywords(question, qvec, plan.topic).split())
        assert expected <= words and "pecado" not in words, words


@pytest.fixture(scope="module")
def bible():
    """The real Bible: the shipped KJV and RV1909 indexes, with the bundled embedder (which
    words are rare, and which passages are about what, only shows on the whole text)."""
    root = ROOT / "storage" / "prebuilt"
    if not (root / "usem3" / "kjv").exists():
        pytest.skip("no prebuild in this copy")
    settings = Settings(
        versions=("kjv", "rv1909"),
        prebuilt_dir=root,
        index_dir=ROOT / "nonexistent",
        llm_chain=("extractive",),
        nli_model="",
    )
    return BibleRAG.from_settings(settings, load_llm=False)


DRUNK = ("drunk", "wine", "strong drink", "embriag", "borrach", "vino")


@pytest.mark.parametrize(
    "question",
    [
        "¿Es pecado emborracharse?",
        "¿Está mal emborracharse?",
        "¿Puedo emborracharme?",
        "¿Qué dice la Biblia sobre emborracharse?",
        "Is it a sin to get drunk?",
    ],
)
def test_in_the_bible_getting_drunk_finds_drunkenness(bible, question):
    """Not the same passages on sin for every "¿Es pecado…?" (Romans 7, and Leviticus 5 on
    oaths: "juramento"), but passages on drunkenness: the first one and most of them."""
    plan = bible.plan(question)
    assert plan.intent == GENERAL and plan.topic.key == "alcohol"
    on_it = [any(w in r.first.passage.text.lower() for w in DRUNK) for r in plan.sections[0].rows]
    assert on_it[0] and sum(on_it) >= 2, [r.title for r in plan.sections[0].rows]


def test_in_the_bible_a_contradiction_about_sex_is_about_sex(bible):
    """It came near slavery by meaning, and was answered about slavery ("esclavitud")."""
    plan = bible.plan("hay contradicciones sobre el sexo en la biblia")
    assert plan.intent == CONTRADICTION and plan.topic.key == "adhoc"
    assert plan.topic_name == "sexo" and "esclav" not in plan.render().lower()
    rows = [r for s in plan.sections for r in s.rows]
    words = ("fornicat", "fornica", "virgin", "adulter", "lie with", "lust")
    on_it = [any(w in r.first.passage.text.lower() for w in words) for r in rows]
    assert sum(on_it) >= len(rows) - 1, [r.title for r in rows]


def test_in_the_bible_each_sin_gets_its_own_passages(bible):
    drunk, oath = (
        bible.plan(q).sections[0].rows for q in ("¿Es pecado emborracharse?", "¿Es pecado jurar?")
    )
    assert {r.title for r in drunk}.isdisjoint({r.title for r in oath})
    assert all(
        any(w in r.first.passage.text.lower() for w in ("swear", "sware", "sworn", "oath", "jura"))
        for r in oath
    )


# ----------------------------------------------------------------------------- pipeline
def test_cellphone_related_themes_with_how_they_relate(related_rag):
    plan = related_rag.plan("Should I use a cellphone?")
    assert plan.topic_name == "cellphone"
    assert [s.key for s in plan.sections] == ["self_control", "communication", "conscience"]
    letters = section(plan, "communication")
    assert letters.title == "Communication: letters, messengers and news from afar"
    assert letters.note == (
        "How this relates to “cellphone”: in so far as it is a way of sending words and news "
        "to others."
    )
    # the theme's anchors, in every translation; an anchor's key is exactly its verse
    assert titles(letters) == [
        "2 John 1:12 · 2 Juan 1:12",
        "Proverbs 25:25 · Proverbios 25:25",
    ]
    assert list(letters.rows[0].hits) == ["kjv", "rv1909", "web"]
    assert letters.rows[0].key["rv1909"].startswith("Aunque tengo muchas cosas que escribiros")
    assert titles(section(plan, "self_control"))[0] == "1 Corinthians 6:12 · 1 Corintios 6:12"
    frame = section(plan, "conscience")  # "should I...?": the principles for deciding
    assert frame.note == "How this relates: the Bible's own principles for choices it doesn't name."
    assert set(titles(frame)) == {
        "1 Corinthians 10:23 · 1 Corintios 10:23",
        "Romans 14:23 · Romanos 14:23",
    }
    shown = [t for s in plan.sections for t in titles(s)]
    assert len(shown) == len(set(shown))  # no passage in two sections
    assert plan.queries == [THEME_BY_KEY[s.key].query["en"] for s in plan.sections]


def test_the_answer_says_what_it_shows_and_what_it_does_not(related_rag):
    related_rag.chain = make_chain(None)  # extractive: the assistant's own words
    answer = related_rag.ask("Should I use a cellphone?")
    assert answer.text.startswith(
        "The Bible does not mention “cellphone”. It does speak of related things: self-control "
        "(not being mastered by anything), communication (letters, messengers and news from "
        "afar) and deciding on what the Bible doesn't name (freedom, conscience and the good "
        "of others). The passages below are about those, not about “cellphone”: applying them "
        "to it is an inference."
    )
    assert "_“cellphone” does not occur in KJV and WEB._" in answer.text  # what was checked
    assert "_How this relates to “cellphone”: in so far as it can become a habit" in answer.text
    assert answer.grounding.grounded and not answer.grounding.unsupported


def test_a_word_in_the_texts_is_shown_not_hidden(related_rag):
    """KJV "smoking" is a furnace (Genesis 15:17): the answer says where the word occurs and
    that the Bible doesn't speak of smoking as such."""
    plan = related_rag.plan("Is smoking a sin?")
    assert plan.intent == RELATED
    assert plan.remarks == [
        "“smoking” occurs in KJV (1 passage: Genesis 15:17), WEB (1 passage: Genesis 15:17): "
        "check whether it means the same there."
    ]
    assert plan.intro().startswith("The Bible does not speak of “smoking” as such")


def test_in_spanish(related_rag):
    related_rag.chain = make_chain(None)
    answer = related_rag.ask("¿Está bien usar el celular?")
    assert answer.intent == RELATED and answer.language == "es"
    assert answer.text.startswith("La Biblia no menciona «")
    assert "**El dominio propio: no dejarse dominar por nada**" in answer.text
    assert "**Decidir sobre lo que la Biblia no nombra: libertad, conciencia" in answer.text
    assert "_Relación con «" in answer.text and "no aparece en RV1909._" in answer.text


def test_asked_for_a_subject_the_bible_names(related_rag):
    plan = related_rag.plan("What does the Bible say about wine?", task=RELATED)
    assert plan.reason == "as requested" and not plan.absence.absent
    assert plan.intro().startswith(
        "The Bible does speak of “wine” directly: a direct search shows those passages."
    )
    assert plan.remarks[0].startswith("“wine” occurs in KJV (")


def test_the_llm_is_told_what_was_checked(related_rag):
    llm = FakeLLM(reply="The Bible does not mention cellphones; see (2 John 1:12, KJV).")
    related_rag.chain = make_chain(llm)
    related_rag.settings.llm_planning = False  # the catalog chooses the themes
    answer = related_rag.ask("Should I use a cellphone?")
    system, user = llm.calls[-1]
    assert system["content"] == PROMPTS[RELATED].format(
        versions="KJV (King James Version (1769)), RV1909 (Reina-Valera 1909), "
        "WEB (World English Bible)"
    )
    content = user["content"]
    assert "## Communication: letters, messengers and news from afar\n(How this relates" in content
    assert (
        "What the search found:\n- The Bible does not mention “cellphone”.\n"
        "- “cellphone” does not occur in KJV and WEB." in content
    )
    assert content.endswith(
        "what does the Bible say, and not say, about cellphone? Explain how each group of "
        "passages relates to it, citing them."
    )
    assert answer.grounding.grounded


def test_small_model_does_not_write_related_answers(related_rag):
    small = FakeLLM()
    small.languages, small.label, small.compact = {"en"}, "smollm2", True
    related_rag.chain = make_chain(small)
    answer = related_rag.ask("Should I use a cellphone?")
    assert not small.calls and answer.model == "extractive"
    assert answer.text.startswith("The Bible does not mention “cellphone”.")


# --------------------------------------------------------------------------------- CLI
def test_cli_routes_and_asks_for_related(related_rag, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load", lambda settings, llm=True: related_rag)
    assert cli.main(["ask", "Should I use a cellphone?"]) == 0
    captured = capsys.readouterr()
    assert captured.err.startswith(
        "[related concepts (not named in the Bible) · cellphone · not named in the Bible"
    )
    assert "Searched for: “all things are lawful unto me" in captured.out
    assert cli.main(["ask", "--task", "related", "What does the Bible say about wine?"]) == 0
    assert "[related concepts (not named in the Bible) · wine · as requested]" in (
        capsys.readouterr().err
    )
    inputs = iter(["/related Is wine good?", "/help", "/quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    assert cli.main(["chat"]) == 0
    captured = capsys.readouterr()
    assert "[related concepts (not named in the Bible)" in captured.err
    assert "/related <question>" in captured.out


def test_evaluate_related(related_rag, monkeypatch):
    monkeypatch.setattr(
        evaluation, "RELATED_GOLD", [("Should I use a cellphone?", "communication|speech")]
    )
    monkeypatch.setattr(evaluation, "NAMED_GOLD", ["Who was king David?"])
    lines = []
    report = evaluation.evaluate_related(Planner(related_rag), lines.append)
    assert report["summary"] == {
        "related_recognised": 1.0,
        "named_wrongly_absent": 0.0,
        "related_theme_hit": 1.0,
        "related_theme_precision": 0.5,  # self-control and communication; the frame isn't counted
    }
    assert lines[0].startswith("related ✓")
