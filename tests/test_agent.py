"""The assistant's pipelines (agent.py, align.py, lexicon.py) on a small real-text corpus.

The corpus holds real KJV / RV1909 / WEB passages chosen for the cases the assistant must get
right: wine praised (Psalm 104:15) and warned against (Proverbs 20:1, Ephesians 5:18), the
RV1909 homographs "vino" (= came) and "alcohol" (= kohl, 2 Kings 9:30), and a word the versions
translate differently (Psalm 16:10: hell / sepulcro / Sheol).
"""

import json

import numpy as np
import pytest

from bible_rag import cli, evaluation
from bible_rag.agent import (
    CONTRADICTION,
    GENERAL,
    PROMPTS,
    TRANSLATION,
    Planner,
)
from bible_rag.align import (
    Row,
    clauses,
    concept_note,
    excerpt,
    group,
    renderings,
    word_changes,
)
from bible_rag.config import Settings
from bible_rag.grounding import chapters, references, without_references
from bible_rag.index import Hit, VersionIndex
from bible_rag.lexicon import TOPICS, adhoc_topic, compile_terms, fold
from bible_rag.rag import BibleRAG
from bible_rag.semantic import INTENT_EXAMPLES, semantics_for, unit
from bible_rag.sources import VERSIONS

from .conftest import (
    FAKE_EMBEDDER,
    FakeEmbedder,
    FakeLLM,
    make_chain,
    passage,
    with_bible_vocabularies,
)

TOPIC = {t.key: t for t in TOPICS}


# fmt: off
CORPUS = {
    "kjv": [
        passage("kjv", "Psalms", 104, 14, 15, "He causeth the grass to grow for the cattle, and "
                "herb for the service of man: that he may bring forth food out of the earth; And "
                "wine that maketh glad the heart of man, and oil to make his face to shine, and "
                "bread which strengtheneth man’s heart."),
        passage("kjv", "Proverbs", 20, 1, 2, "Wine is a mocker, strong drink is raging: and "
                "whosoever is deceived thereby is not wise. The fear of a king is as the roaring "
                "of a lion: whoso provoketh him to anger sinneth against his own soul."),
        passage("kjv", "Ephesians", 5, 17, 18, "Wherefore be ye not unwise, but understanding what "
                "the will of the Lord is. And be not drunk with wine, wherein is excess; but be "
                "filled with the Spirit;"),
        passage("kjv", "2 Kings", 9, 30, 31, "And when Jehu was come to Jezreel, Jezebel heard of "
                "it; and she painted her face, and tired her head, and looked out at a window. And "
                "as Jehu entered in at the gate, she said, Had Zimri peace, who slew his master?"),
        passage("kjv", "Psalms", 16, 10, 11, "For thou wilt not leave my soul in hell; neither "
                "wilt thou suffer thine Holy One to see corruption. Thou wilt shew me the path of "
                "life: in thy presence is fulness of joy."),
        passage("kjv", "Matthew", 2, 1, 2, "Now when Jesus was born in Bethlehem of Judaea in the "
                "days of Herod the king, behold, there came wise men from the east to Jerusalem, "
                "Saying, Where is he that is born King of the Jews?"),
    ],
    "rv1909": [
        passage("rv1909", "Salmos", 104, 14, 15, "El que hace producir el heno para las bestias, y "
                "la hierba para el servicio del hombre; sacando el pan de la tierra. Y el vino que "
                "alegra el corazón del hombre, y el aceite que hace lucir el rostro, y el pan que "
                "sustenta el corazón del hombre."),
        passage("rv1909", "Proverbios", 20, 1, 2, "EL vino es escarnecedor, la cerveza "
                "alborotadora; y cualquiera que por ello errare, no será sabio. Como bramido de "
                "cachorro de león es el terror del rey: el que lo hace enfurecerse, peca contra "
                "su alma."),
        passage("rv1909", "Efesios", 5, 17, 18, "Por tanto, no seáis imprudentes, sino entendidos "
                "de cuál sea la voluntad del Señor. Y no os embriaguéis de vino, en lo cual hay "
                "disolución; mas sed llenos de Espíritu;"),
        passage("rv1909", "2 Reyes", 9, 30, 31, "Vino después Jehú á Jezreel: y como Jezabel lo "
                "oyó, adornó sus ojos con alcohol, y atavió su cabeza, y asomóse á una ventana. Y "
                "como entraba Jehú por la puerta, ella dijo: ¿Sucedió bien á Zimri, que mató á su "
                "señor?"),
        passage("rv1909", "Salmos", 16, 10, 11, "Porque no dejarás mi alma en el sepulcro; ni "
                "permitirás que tu santo vea corrupción. Me mostrarás la senda de la vida: hartura "
                "de alegrías hay con tu rostro."),
        passage("rv1909", "Mateo", 2, 1, 2, "Y COMO fué nacido Jesús en Bethlehem de Judea en días "
                "del rey Herodes, he aquí unos magos vinieron del oriente á Jerusalem, Diciendo: "
                "¿Dónde está el Rey de los Judíos, que ha nacido?"),
    ],
    "web": [
        passage("web", "Psalms", 104, 15, 15, "wine that makes the heart of man glad, oil to make "
                "his face to shine, and bread that strengthens man’s heart."),
        passage("web", "Proverbs", 20, 1, 2, "Wine is a mocker and beer is a brawler. Whoever is "
                "led astray by them is not wise. The terror of a king is like the roaring of a "
                "lion. He who provokes him to anger forfeits his own life."),
        passage("web", "Ephesians", 5, 17, 18, "Therefore, don’t be foolish, but understand what "
                "the will of the Lord is. Don’t be drunken with wine, in which is dissipation, but "
                "be filled with the Spirit,"),
        passage("web", "2 Kings", 9, 30, 31, "When Jehu had come to Jezreel, Jezebel heard of it; "
                "and she painted her eyes, and adorned her head, and looked out at the window. As "
                "Jehu entered in at the gate, she said, “Do you come in peace, Zimri, you murderer "
                "of your master?”"),
        passage("web", "Psalms", 16, 10, 11, "For you will not leave my soul in Sheol, neither "
                "will you allow your holy one to see corruption. You will show me the path of "
                "life. In your presence is fullness of joy."),
        passage("web", "Matthew", 2, 1, 2, "Now when Jesus was born in Bethlehem of Judea in the "
                "days of King Herod, behold, wise men from the east came to Jerusalem, saying, "
                "“Where is he who is born King of the Jews?"),
    ],
}
# fmt: on


@pytest.fixture
def corpus_rag(tmp_path, meaning, bible_vocabularies):
    settings = Settings(
        index_dir=tmp_path / "index",
        prebuilt_dir=tmp_path / "prebuilt",
        versions=("kjv", "rv1909", "web"),
        top_k=2,
        embedding_chain=(FAKE_EMBEDDER,),
        nli_model="",
    )
    indexes = {
        v: VersionIndex.build(ps, meaning, {"embedder": FAKE_EMBEDDER}) for v, ps in CORPUS.items()
    }
    with_bible_vocabularies(indexes, bible_vocabularies)
    rag = BibleRAG(settings, meaning, indexes, make_chain(FakeLLM()), FAKE_EMBEDDER)
    rag.nli = None
    return rag


class FakeNLI:
    """NLI stand-in: `rule(premise, hypothesis)` -> (P(entailment), P(contradiction))."""

    model_name = "fake-nli"

    def __init__(self, rule):
        self.rule, self.calls = rule, []

    def scores(self, premises, hypotheses):
        self.calls.append(list(zip(premises, hypotheses, strict=True)))
        return np.array([self.rule(p, h) for p, h in zip(premises, hypotheses, strict=True)])


def refs(section):
    return [row.title for row in section.rows]


# ------------------------------------------------------------------ routing by meaning
@pytest.fixture
def planner(corpus_rag):
    return Planner(corpus_rag)


def route(planner, question):
    return planner.route(question, planner.embedder.embed([question])[0])


@pytest.mark.parametrize("question,intent", evaluation.ROUTING_GOLD)
def test_route_by_meaning_in_six_languages(planner, question, intent):
    """No keywords: the questions are not among the examples, and four of the languages
    (pt, fr, de, it) are in no example at all."""
    assert route(planner, question).intent == intent


def test_route_close_calls_and_llm(planner, corpus_rag):
    r = route(planner, "¿Jehová es mi pastor?")  # short yes/no: close to several kinds
    assert r.intent == GENERAL and r.scores
    planner.route_margin = 1.0  # every call is close: the LLM settles it
    corpus_rag.chain = make_chain(FakeLLM(plan_reply="Contradiction."))
    r = route(planner, "Where was Jesus born?")
    assert r.intent == CONTRADICTION and r.reason.startswith("fake chose between")
    corpus_rag.chain = make_chain(FakeLLM(plan_reply="I am not sure"))  # unusable reply
    assert route(planner, "Where was Jesus born?").intent == GENERAL
    corpus_rag.settings.llm_planning = False
    corpus_rag.chain = make_chain(FakeLLM(plan_reply="contradiction"))
    assert route(planner, "Where was Jesus born?").intent == GENERAL


def test_references():
    assert references("Compare John 3:16 and Juan 3,16-17, and 1 Cor. 13:4") == [
        (42, 3, 16, 16),
        (42, 3, 16, 17),  # "3,16": the usual Spanish / German form
        (45, 13, 4, 4),
    ]
    assert chapters("Differences in Psalm 23 and Juan 3? Not John 3:16") == [(18, 23), (42, 3)]
    assert chapters("the 10 commandments in 3 days") == []
    assert without_references("How do they render Psalm 23 and John 3:16?") == (
        "How do they render   and  ?"
    )


def test_each_task_can_be_asked_for(corpus_rag):
    """Routing picks the task, but any of the three can be asked for explicitly (the web
    chat's buttons, `ask --task`, the chat's /contradiction and /translation)."""
    question = "Where was Jesus born?"
    assert corpus_rag.plan(question).intent == GENERAL
    for task in (CONTRADICTION, TRANSLATION, GENERAL):
        plan = corpus_rag.plan(question, task=task)
        assert plan.intent == task and plan.reason == "as requested"
    wine = corpus_rag.plan("Is wine allowed?", task=CONTRADICTION)
    assert [s.title for s in wine.sections] == [
        "Passages that speak favourably of wine",
        "Passages that warn against it",
    ]
    with pytest.raises(ValueError, match="Unknown task"):
        corpus_rag.plan(question, task="debate")


# ------------------------------------------------------------------- topics by meaning
@pytest.mark.parametrize("question,key", evaluation.TOPIC_GOLD)
def test_topic_by_meaning(corpus_rag, question, key):
    """Paraphrases without the lexicon's words ("shellfish", "self-defence") find their
    topic; subjects outside the lexicon (jewelry, astrology...) are recognised as such, also
    when they come near a topic by meaning without being it ("sexo" is not slavery)."""
    topic = Planner(corpus_rag).topic_of(question, corpus_rag.embedder.embed([question])[0])
    assert (topic.key if topic else None) == key


@pytest.mark.parametrize(
    "question,words",
    [
        ("Are there any paradoxes in the Bible according to alcohol?", ["alcohol"]),
        ("Does the Bible contradict itself about faith and works?", ["faith", "works"]),
        ("Existem contradições sobre o batismo?", ["batismo"]),
        ("Y a-t-il des paradoxes sur la vieillesse ?", ["vieillesse"]),
        ("¿Se contradice la Biblia sobre la oración?", ["oración"]),
        ("Contradictions about long hair for men?", ["long", "hair", "men"]),
    ],
)
def test_subject_words_without_word_lists(corpus_rag, question, words):
    sem = corpus_rag.semantics
    assert sem.subject_words(question, corpus_rag.embedder.embed([question])[0]) == words


def test_opposition_axis(corpus_rag):
    sem = corpus_rag.semantics
    v = corpus_rag.embedder.embed(
        [
            "And wine that maketh glad the heart of man",
            "Woe unto them that are mighty to drink wine",
            "Y el vino que alegra el corazón del hombre",
            "¡Ay de los que son valientes para beber vino!",
        ]
    )
    glad, woe, alegra, ay = sem.approval(v)
    assert glad > woe and alegra > ay  # approving vs condemning, in both languages
    assert sem.opposed(v[0], v[1]) > 0 > sem.opposed(v[1], v[0])
    good, bad = sem.polarize(unit(corpus_rag.embedder.embed(["dancing"])[0]))
    assert sem.approval(good) > 0 > sem.approval(bad) and good @ bad > 0.5  # same topic


def test_semantics_are_built_once_per_embedder(corpus_rag):
    assert corpus_rag.semantics is semantics_for(corpus_rag.embedder)
    assert set(corpus_rag.semantics.intents) == set(INTENT_EXAMPLES)
    assert 0 < corpus_rag.semantics.topic_threshold < 1


# ----------------------------------------------------------------------------- lexicon
def test_fold_keeps_offsets():
    text = "Arrepintióse Jehová, á Dios; Cæsar"
    assert fold(text) == "arrepintiose jehova, a dios; cæsar" and len(fold(text)) == len(text)


def test_term_syntax_prefix_phrase_and_homograph():
    rx = compile_terms(["drunk*", "strong drink", "vino?"])
    assert rx.search("be not drunken") and rx.search("strong  drink is raging")
    assert rx.search("vino despues") and not rx.search("vinos")  # "vino?" is a whole word
    alcohol = TOPIC["alcohol"]
    assert "vino?" in alcohol.terms("es") and "vino?" not in alcohol.terms("es", False)


def test_adhoc_topic_matches_other_forms_and_expansions():
    topic = adhoc_topic(["dancing"], "en", {"en": ["dances"], "es": ["danzas"]})
    rx = compile_terms(topic.terms("en"))
    assert all(rx.search(w) for w in ("dance", "danced", "dancing")) and topic.key == "adhoc"
    assert topic.terms("es") == ["danzas"] and topic.examples == ("dancing",)


def test_lexicon_is_complete():
    for topic in TOPICS:
        assert topic.concepts and topic.examples
        for lang in ("en", "es"):
            assert topic.name[lang] and topic.terms(lang), (topic.key, lang)
            if topic.claims:
                assert len(topic.claims[lang]) == 2 and len(topic.sides[lang]) == 2
        for note in topic.notes:
            assert note.en and note.es and set(note.versions) <= VERSIONS.keys()


# ---------------------------------------------------------------------------- aligning
def test_parallel_lookup_by_verse_and_by_meaning():
    kjv = passage("kjv", "Joel", 1, 1, 4, "The word of the LORD that came to Joel.")
    a = passage("kjv", "Joel", 1, 5, 8, "Awake, ye drunkards, and weep for the new wine.")
    idx = VersionIndex.build([kjv, a], FakeEmbedder(), {})
    assert idx.at(28, 1, 4, 5) == 0  # tie on shared verses: the earlier chunk
    wine = FakeEmbedder().embed(["Awake, ye drunkards, and weep for the new wine."])[0]
    assert idx.at(28, 1, 4, 5, wine) == 1  # with a vector: the chunk about it
    assert idx.at(28, 2, 1, 1) is None and idx.places[(28, 1)] == [0, 1]


def test_group_merges_versions_and_drops_overlapping_chunks():
    kjv = [passage("kjv", "John", 3, 16, 17, "a"), passage("kjv", "John", 3, 17, 18, "b")]
    es = passage("rv1909", "Juan", 3, 15, 16, "c")
    rows = group([("kjv", Hit(kjv[0], 1)), ("rv1909", Hit(es, 1)), ("kjv", Hit(kjv[1], 1))])
    assert len(rows) == 1 and list(rows[0].hits) == ["kjv", "rv1909"]
    assert rows[0].title == "John 3:16-17 · Juan 3:15-16"


def test_clauses_and_excerpt():
    assert clauses("Wine is a mocker, strong drink is raging: and whosoever is deceived.") == [
        "Wine is a mocker, strong drink is raging: and whosoever is deceived."
    ]
    first, second = "a" * 30 + ", " + "b" * 40 + ";", "c" * 70 + "."
    assert clauses(f"{first} {second}") == [first, second]  # a clause of >= 60 chars stands
    assert excerpt("one two three four", 9) == "one two…"


def test_word_changes_separate_meaning_from_style():
    kjv = (
        "Charity suffereth long, and is kind; charity envieth not; is not easily provoked, "
        "thinketh no evil; For thou wilt not leave my soul in hell;"
    )
    web = (
        "Love is patient and is kind. Love doesn’t envy. is not provoked, takes no account of "
        "evil; For you will not leave my soul in Sheol,"
    )
    changes, minor = word_changes(kjv, web)
    assert ("Charity suffereth long", "Love is patient") in changes
    assert ("hell", "Sheol") in changes and ("easily", "") in changes
    assert ("thou wilt", "you will") not in changes and minor >= 1  # archaic -> modern
    assert word_changes("In the beginning God created", "Jesus wept bitterly then") is None


def test_concept_note_needs_two_different_renderings():
    hell = TOPIC["hell"]
    row = Row(18, 16, {v: Hit(CORPUS[v][4], 1) for v in CORPUS})
    row.key = {v: h.passage.text for v, h in row.hits.items()}
    note = concept_note(row, hell, lambda v: VERSIONS[v].language, lambda v: v.upper(), "en")
    assert note == (
        "Wording differs: KJV “hell” (hell) · RV1909 “sepulcro” (the grave) · WEB “Sheol” (Sheol)"
    )
    row.key["web"] = "For you will not leave my soul"  # no term: proves nothing, not flagged
    row.key["rv1909"] = "no dejarás mi alma en el infierno"
    assert concept_note(row, hell, lambda v: VERSIONS[v].language, str, "en") is None


# --------------------------------------------------------------------------- pipelines
def test_general_shows_every_translation_of_the_best_passages(corpus_rag):
    plan = corpus_rag.plan("Where was Jesus born in Bethlehem?", k=1)
    (section,) = plan.sections
    assert plan.intent == GENERAL and refs(section) == ["Matthew 2:1-2 · Mateo 2:1-2"]
    assert list(section.rows[0].hits) == ["kjv", "rv1909", "web"]  # parallels filled in
    text = plan.render()
    assert "**What each translation says**" in text and "   - RV1909: “Y COMO fué nacido" in text


def test_contradiction_searches_both_statements(corpus_rag):
    plan = corpus_rag.plan("are there any paradoxes in the bible according to alcohol?")
    favourable, warning = plan.sections
    assert plan.intent == CONTRADICTION and plan.topic.key == "alcohol"
    assert plan.queries == list(plan.topic.claims["en"])  # the two opposite statements
    assert favourable.title == "Passages that speak favourably of wine"
    assert refs(favourable) == ["Psalms 104:14-15 · Salmos 104:14-15 · Psalms 104:15"]
    assert set(refs(warning)) == {
        "Proverbs 20:1-2 · Proverbios 20:1-2",
        "Ephesians 5:17-18 · Efesios 5:17-18",
    }
    # each passage's key clause is the one about the topic, in every translation
    psalm = favourable.rows[0]
    assert "And wine that maketh glad the heart of man" in psalm.key["kjv"]
    assert "el vino que alegra el corazón" in psalm.key["rv1909"]
    # 2 Reyes 9:30 only matches "vino" (= came): dropped, its KJV parallel has no wine
    titles = refs(favourable) + refs(warning)
    assert not any("9:30" in t for t in titles)
    # the notes cite 2 Reyes 9:30 / Ezequiel 23:40: those passages become checkable sources
    assert any("kohl" in n for n in plan.notes) and any("vino" in n for n in plan.notes)
    assert "2 Reyes 9:30-31 (RV1909)" in [h.passage.citation for h in plan.sources]


def test_contradiction_answer_in_spanish_without_llm(corpus_rag):
    corpus_rag.chain = make_chain(None)  # extractive
    answer = corpus_rag.ask("¿Hay contradicciones en la Biblia sobre el alcohol?")
    assert answer.intent == CONTRADICTION and answer.language == "es"
    assert answer.text.startswith("Busqué dos afirmaciones opuestas sobre el alcohol")
    assert "**Pasajes favorables al vino**" in answer.text
    assert "**Notas de traducción**" in answer.text and answer.grounding.grounded


def test_contradiction_without_topic_falls_back_to_a_direct_search(corpus_rag):
    plan = corpus_rag.plan("Are there contradictions in the Bible?")
    assert plan.intent == GENERAL and "Name a topic" in plan.render()


def test_translation_of_a_reference(corpus_rag):
    plan = corpus_rag.plan("How do the translations render Psalm 16:10?")
    (row,) = plan.sections[0].rows
    assert plan.intent == TRANSLATION and list(row.hits) == ["kjv", "rv1909", "web"]
    # no topic named: the passage's own differences (every lexicon topic is checked), and
    # the same-language pair diffed word by word
    assert row.notes == [
        "Wording differs: KJV “hell” (hell) · RV1909 “sepulcro” (the grave) · WEB “Sheol” (Sheol)",
        "KJV → WEB: “hell” → “Sheol”; “wilt thou suffer thine” → “will you allow your”; plus 4 "
        "minor or archaic-to-modern changes (“thou” → “you”…)",
    ]
    assert plan.intro().startswith("The same passages in each translation, with the places")
    plan = corpus_rag.plan("Is hell translated the same in Psalm 16:10?")  # a topic too
    assert plan.sections[0].rows[0].notes[0] == (
        "Wording differs: KJV “hell” (hell) · RV1909 “sepulcro” (the grave) · WEB “Sheol” (Sheol)"
    )


def test_translation_of_a_whole_chapter_and_comma_references(corpus_rag):
    plan = corpus_rag.plan("Compare the translations of Psalm 16", task=TRANSLATION)
    assert [r.title for r in plan.sections[0].rows] == ["Psalms 16:10-11 · Salmos 16:10-11"]
    assert plan.sections[0].rows[0].notes[0].startswith("Wording differs: KJV “hell”")
    plan = corpus_rag.plan("¿Cómo traducen las versiones Salmos 16,10?")
    assert plan.intent == TRANSLATION
    assert plan.sections[0].rows[0].title == "Psalms 16:10-11 · Salmos 16:10-11"


def test_translation_of_a_topic_finds_where_versions_differ(corpus_rag):
    plan = corpus_rag.plan("Do the translations differ on hell?", k=1)
    assert refs(plan.sections[0]) == ["Psalms 16:10-11 · Salmos 16:10-11"]
    assert any("Sheol" in n for n in plan.notes)  # the topic's note


def test_answer_prompts_the_llm_per_intent(corpus_rag):
    llm = FakeLLM(reply="Psalm 104:15 praises wine (Psalms 104:14-15, KJV); Proverbs 20:1 warns.")
    corpus_rag.chain = make_chain(llm)
    answer = corpus_rag.ask("Is wine good or bad? Any contradiction?")
    system, user = llm.calls[-1]
    labels = (
        "KJV (King James Version (1769)), RV1909 (Reina-Valera 1909), WEB (World English Bible)"
    )
    assert system["content"] == PROMPTS[CONTRADICTION].format(versions=labels)
    assert "## Passages that speak favourably of wine" in user["content"]
    assert "## Passages that warn against it" in user["content"]
    assert "Notes on the translations:" in user["content"]
    # the answer: the model's synthesis, then the passages side by side; only the synthesis
    # (the model's words) is checked
    assert answer.text.startswith(llm.reply) and "**Passages that warn against it**" in answer.text
    assert answer.grounding.grounded and answer.synthesis == llm.reply


def test_small_model_does_not_write_comparisons(corpus_rag):
    small = FakeLLM()
    small.languages, small.label, small.compact = {"en"}, "smollm2", True
    corpus_rag.chain = make_chain(small)
    answer = corpus_rag.ask("How do the translations render Psalm 16:10?")
    assert not small.calls and answer.model == "extractive"
    assert "smollm2 is too small to compare passages" in answer.notes[-1]
    corpus_rag.ask("Where was Jesus born?")  # a direct question: the small model answers
    assert small.calls


def test_planner_validates(corpus_rag):
    with pytest.raises(ValueError, match="empty"):
        corpus_rag.plan("  ")
    with pytest.raises(ValueError, match="Unknown version"):
        corpus_rag.plan("wine", versions=["niv"])
    plan = corpus_rag.plan("are there paradoxes about wine?", versions=["kjv"])
    assert all(list(r.hits) == ["kjv"] for s in plan.sections for r in s.rows)
    assert not any("kohl" in n for n in plan.notes)  # RV1909 notes need the RV1909


def test_side_ranking_prefers_the_claim_closest_passage():
    rows = [Row(0, c) for c in (1, 2, 3)]
    # (closeness to the side's claim, relevance to the topic, stance)
    values = [(0.9, 0.3, 0.1), (0.1, 0.9, 0.3), (0.6, 0.5, 0.2)]
    for row, (a, t, s) in zip(rows, values, strict=True):
        row.fit, row.score, row.stance = (a, 0.0), t, s
    rank = Planner(None)._side_rank(0)
    assert [r.chapter for r in Planner._pick(rows, 2, rank)] == [1, 3]
    assert np.isclose(rank(rows[0]), 0.9 + 1.0 * 0.3 + 0.5 * 0.1)


# ------------------------------------------------------------------------ CLI and eval
def test_cli_ask_reports_the_route_and_searched_statements(corpus_rag, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load", lambda settings, llm=True: corpus_rag)
    assert cli.main(["ask", "Are there any paradoxes in the Bible according to alcohol?"]) == 0
    captured = capsys.readouterr()
    assert (
        "[contradiction search (both statements) · alcohol (wine, strong drink) · closest to "
        "contradiction questions" in captured.err
    )
    assert "Searched for: “wine is good" in captured.out
    assert cli.main(["ask", "--task", "translation", "Where was Jesus born?"]) == 0
    assert "[translation comparison · as requested]" in capsys.readouterr().err
    inputs = iter(
        [
            "How do the translations render Psalm 16:10?",
            "/contradiction Is wine allowed?",
            "/translation Psalm 104:15",
            "/help",
            "/quit",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    assert cli.main(["chat"]) == 0
    captured = capsys.readouterr()
    routes = [line for line in captured.err.splitlines() if line.startswith("[")]
    assert routes[0].startswith("[translation comparison · closest to translation questions")
    assert routes[1].startswith("[contradiction search (both statements) · alcohol")
    assert routes[1].endswith("as requested]") and routes[2].endswith("as requested]")
    assert "Commands: /contradiction <question>, /translation <question>" in captured.out


def test_cli_suggests_the_other_tasks_when_unsure(corpus_rag, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load", lambda settings, llm=True: corpus_rag)
    corpus_rag.settings.llm_planning = False
    monkeypatch.setattr(Planner, "route_margin", 1.0)  # every call is a close one
    assert cli.main(["ask", "Is drinking wine allowed or forbidden?"]) == 0
    out = capsys.readouterr().out
    assert "Unsure which task this is" in out and "--task contradiction|translation" in out


def test_evaluate_agent(corpus_rag, monkeypatch):
    monkeypatch.setattr(
        evaluation,
        "CONTRADICTION_GOLD",
        {"Is wine good or evil?": ("Psalms 104", "Proverbs 20|Isaiah 5")},
    )
    lines = []
    report = evaluation.evaluate_agent(corpus_rag, 2, lines.append)
    summary = report["summary"]
    assert summary["routing_accuracy"] == 1.0 and summary["topic_accuracy"] == 1.0
    with_lexicon = [s["precision"] for s in report["contradictions"] if s["lexicon"]]
    assert with_lexicon == [1.0, 0.5] and summary["contradiction_side_precision"] == 0.75
    assert "contradiction_side_precision_meaning_only" in summary and lines


def test_homographs_count_only_when_another_version_confirms_them():
    alcohol = TOPIC["alcohol"]
    kjv = passage(
        "kjv", "John", 2, 10, 10, "when men have well drunk, thou hast kept the good wine"
    )
    es = passage("rv1909", "Juan", 2, 10, 10, "cuando están satisfechos, has guardado el buen vino")
    row = Row(42, 2, {"kjv": Hit(kjv, 1), "rv1909": Hit(es, 1)})
    found = renderings(row, alcohol, lambda v: VERSIONS[v].language)
    # "vino" is confirmed as wine by the KJV; "drunk" (past of drink) is confirmed by nothing
    assert found == {"kjv": {"wine": ["wine"]}, "rv1909": {"wine": ["vino"]}}
    assert concept_note(row, alcohol, lambda v: VERSIONS[v].language, str, "en") is None


# ------------------------------------------------- topics outside the lexicon, by meaning
def test_expansion_finds_each_translations_words_by_meaning(corpus_rag):
    for index in corpus_rag.indexes.values():  # the corpus's own words: they are ranked by
        index._vocabulary = None  # the passages they occur in, which must be the same texts
    planner = Planner(corpus_rag)
    question = "Existem paradoxos na Bíblia sobre o álcool?"  # Portuguese: no such Bible here
    qvec = corpus_rag.embedder.embed([question])[0]
    subject = corpus_rag.semantics.subject(qvec)
    found = planner._expand(["álcool"], subject, ["kjv", "rv1909"])
    assert "wine" in found["en"] and "vino" in found["es"]


def test_contradiction_on_a_subject_outside_the_lexicon(corpus_rag):
    planner = Planner(corpus_rag)
    planner.use_lexicon = False  # as if "alcohol" were unknown: meaning only
    plan = planner.plan("Are there paradoxes in the Bible about drunkenness?", "en")
    assert plan.intent == CONTRADICTION and plan.topic.key == "adhoc"
    assert plan.queries == [
        "Drunkenness is good and blessed by God",
        "Drunkenness is evil and forbidden by God",
    ]
    warning = [r.title for r in plan.sections[1].rows]
    assert "Proverbs 20:1-2 · Proverbios 20:1-2" in warning  # found and placed by meaning
    assert plan.checks == [
        "Sides estimated from meaning similarity (no NLI model loaded: see `bible-rag models`)."
    ]


CLAIMS = {
    "topic_en": "dancing",
    "topic_es": "el baile",
    "for_en": "Praise him with the timbrel and dance.",
    "against_en": "Dancing is a sin that God condemns.",
    "for_es": "Alabadle con pandero y danza.",
    "against_es": "El baile es un pecado que Dios condena.",
}


def test_llm_writes_the_two_statements_checked_for_opposition(corpus_rag):
    planner = Planner(corpus_rag)
    planner.use_lexicon = False
    corpus_rag.chain = make_chain(FakeLLM(plan_reply=json.dumps(CLAIMS)))
    plan = planner.plan("Does the Bible contradict itself about dancing?", "en")
    assert plan.queries == [CLAIMS["for_en"], CLAIMS["against_en"]]
    assert plan.topic.name == {"en": "dancing", "es": "el baile"}
    # written the wrong way round: the opposition axis puts them back in order
    swapped = {**CLAIMS, "for_en": CLAIMS["against_en"], "against_en": CLAIMS["for_en"]}
    corpus_rag.chain = make_chain(FakeLLM(plan_reply=json.dumps(swapped)))
    plan = planner.plan("Does the Bible contradict itself about dancing?", "en")
    assert plan.queries == [CLAIMS["for_en"], CLAIMS["against_en"]]
    # not opposite, or not JSON: the generic statements are used
    same = {**CLAIMS, "against_en": CLAIMS["for_en"]}
    for reply in (json.dumps(same), "Sure! Dancing is fine."):
        corpus_rag.chain = make_chain(FakeLLM(plan_reply=reply))
        plan = planner.plan("Does the Bible contradict itself about dancing?", "en")
        assert plan.queries[0] == "Dancing is good and blessed by God"


# ------------------------------------------------------------------ NLI: opposition check
def wine_nli(premise, hypothesis):
    """Warnings contradict 'wine is good' and entail 'wine is a mocker', and vice versa."""
    warns = any(w in premise.lower() for w in ("mocker", "drunk", "embriag", "escarnecedor"))
    good = "good" in hypothesis or "bueno" in hypothesis
    return (0.1, 0.8) if warns == good else (0.8, 0.1)


def test_nli_settles_the_sides(corpus_rag):
    corpus_rag.nli = FakeNLI(wine_nli)
    plan = corpus_rag.plan("are there any paradoxes in the bible according to alcohol?")
    favourable, warning = plan.sections
    assert [r.title for r in favourable.rows][0].startswith("Psalms 104")
    assert {r.title.split(" ·")[0] for r in warning.rows} == {
        "Proverbs 20:1-2",
        "Ephesians 5:17-18",
    }
    assert all(r.features["nli"] > 0 for r in favourable.rows)
    assert all(r.features["nli"] < 0 for r in warning.rows)
    assert plan.checks == ["Sides checked for opposition by the NLI model fake-nli."]
    premise, hypothesis = corpus_rag.nli.calls[0][0]
    assert hypothesis in plan.topic.claims["en"] + plan.topic.claims["es"]


def test_nli_flags_translations_that_contradict_each_other(corpus_rag):
    def rule(premise, hypothesis):  # "Sheol" and "hell" treated as contradicting
        pair = {("sheol" in premise.lower()), ("sheol" in hypothesis.lower())}
        return (0.1, 0.9) if pair == {True, False} else (0.9, 0.05)

    corpus_rag.nli = FakeNLI(rule)
    plan = corpus_rag.plan("How do the translations render Psalm 16:10?")
    notes = plan.sections[0].rows[0].notes
    assert "Meaning may differ: KJV and WEB contradict each other here (NLI, 90%)" in notes
    assert not any("KJV and RV1909" in n for n in notes)


# ----------------------------------------------------------------- the original languages
ORIGINALS = {
    "wlc": [
        passage("wlc", "Psalms", 104, 15, 15, "וְיַיִן יְשַׂמַּח לְבַב אֱנוֹשׁ לְהַצְהִיל פָּנִים מִשָּׁמֶן"),
        passage("wlc", "Proverbs", 20, 1, 1, "לֵץ הַיַּיִן הֹמֶה שֵׁכָר וְכָל שֹׁגֶה בּוֹ לֹא יֶחְכָּם"),
        passage("wlc", "Psalms", 16, 10, 10, "כִּי לֹא תַעֲזֹב נַפְשִׁי לִשְׁאוֹל"),
    ],
    "sblgnt": [
        passage("sblgnt", "Ephesians", 5, 18, 18, "καὶ μὴ μεθύσκεσθε οἴνῳ, ἐν ᾧ ἐστιν ἀσωτία"),
        passage("sblgnt", "Matthew", 2, 1, 1, "Τοῦ δὲ Ἰησοῦ γεννηθέντος ἐν Βηθλέεμ"),
    ],
}
TAGS = {  # "book:chapter:verse" -> [[words, [Strong's]]]
    "kjv": {
        "18:104:15": [["wine", ["H3196"]]],
        "19:20:1": [["Wine", ["H3196"]], ["drink", ["H7941"]]],
        "18:16:10": [["hell", ["H7585"]]],
        "48:5:18": [["drunk", ["G3182"]], ["wine", ["G3631"]]],
    },
    "wlc": {
        "18:104:15": [["וְיַיִן", ["H3196"]]],
        "19:20:1": [["הַיַּיִן", ["H3196"]]],
        "18:16:10": [["לִשְׁאוֹל", ["H7585"]]],
    },
    "sblgnt": {"48:5:18": [["οἴνῳ", ["G3631"]]]},
}
LEXICON = {
    "H3196": {"word": "יין", "translit": "yayin", "kjv": "banqueting, wine", "def": "wine"},
    "H7941": {"word": "שׁכר", "translit": "shêkâr", "kjv": "strong drink", "def": "liquor"},
    "H7585": {"word": "שׁאל", "translit": "she'ôl", "kjv": "grave, hell, pit", "def": "hades"},
    "G3182": {"word": "μεθύσκω", "translit": "methýskō", "kjv": "be drunk", "def": "drink"},
    "G3631": {"word": "οἶνος", "translit": "oînos", "kjv": "wine", "def": "wine"},
}


@pytest.fixture
def with_originals(corpus_rag, meaning):
    """The corpus plus the Hebrew and Greek texts, the words' Strong's numbers and the
    dictionary (as `ingest` prepares them)."""
    for v, ps in ORIGINALS.items():
        corpus_rag.indexes[v] = VersionIndex.build(ps, meaning, {"embedder": FAKE_EMBEDDER})
    for v, tags in TAGS.items():
        corpus_rag.indexes[v]._tags = tags
    corpus_rag.settings.versions = ("kjv", "rv1909", "web", "wlc", "sblgnt")
    corpus_rag.strongs = LEXICON
    return corpus_rag


def test_originals_are_compared_not_searched(with_originals):
    planner = Planner(with_originals)
    versions = list(with_originals.settings.versions)
    assert planner.searchable(versions) == ["kjv", "rv1909", "web"]  # no Hebrew / Greek here
    assert planner.primary(versions) == ["kjv", "rv1909"]  # one per language gathers
    plan = with_originals.plan("Where was Jesus born in Bethlehem?", k=1)
    (row,) = plan.sections[0].rows
    assert list(row.hits) == ["kjv", "rv1909", "web", "sblgnt"]  # Matthew: Greek, no Hebrew
    assert row.key["sblgnt"] == "Τοῦ δὲ Ἰησοῦ γεννηθέντος ἐν Βηθλέεμ"


def test_original_word_behind_a_translation(with_originals):
    plan = with_originals.plan("How do the translations render Psalm 16:10?")
    (row,) = plan.sections[0].rows
    assert row.key["wlc"] == "כִּי לֹא תַעֲזֹב נַפְשִׁי לִשְׁאוֹל"  # exactly the verse asked
    assert "Original words: “hell” = H7585 לִשְׁאוֹל she'ôl (KJV elsewhere: grave, pit)" in row.notes
    assert "   - WLC: “כִּי לֹא תַעֲזֹב נַפְשִׁי לִשְׁאוֹל”" in plan.render()


def test_sides_of_a_contradiction_in_the_original_words(with_originals):
    plan = with_originals.plan("are there any paradoxes in the bible according to alcohol?")
    favourable, warning = plan.sections
    assert favourable.rows[0].key["wlc"].startswith("וְיַיִן יְשַׂמַּח")  # Psalm 104:15
    words = next(n for n in plan.notes if n.startswith("Original words on each side"))
    assert "Passages that speak favourably of wine: H3196 yayin ×1" in words
    assert "Both sides use H3196 yayin: the tension is in the original text" in words
    context = next(m["content"] for m in plan.messages() if m["role"] == "user")
    assert "Ephesians 5:18 (SBLGNT)" in context and "Original words" in context
