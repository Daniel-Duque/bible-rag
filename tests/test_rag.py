"""Index, hybrid search, fallback chains, answers, grounding and evaluation (all with fakes)."""

import hashlib

import numpy as np
import pytest

from bible_rag import evaluation
from bible_rag.corpus import Unit
from bible_rag.grounding import book_index, check_answer
from bible_rag.index import (
    BM25,
    Hit,
    IndexNotFoundError,
    VersionIndex,
    build_indexes,
    open_indexes,
    tokens,
)
from bible_rag.models import ModelUnavailable, embedder_key
from bible_rag.rag import COMPACT_PROMPT, SYSTEM_PROMPT, detect_language

from .conftest import FAKE_EMBEDDER, PASSAGES, FakeEmbedder, FakeLLM, make_chain, passage

HITS = [Hit(p, 0.5) for ps in PASSAGES.values() for p in ps]


# ------------------------------------------------------------------------------- index
def test_index_roundtrip_and_errors(settings, indexes):
    root = settings.index_dir / embedder_key(FAKE_EMBEDDER)
    loaded = VersionIndex.load(root / "kjv", FAKE_EMBEDDER)
    assert loaded.passages == PASSAGES["kjv"] and loaded.manifest["passages"] == 3
    with pytest.raises(IndexNotFoundError, match="bible-rag ingest"):
        VersionIndex.load(root / "missing")
    with pytest.raises(IndexNotFoundError, match="--force"):
        VersionIndex.load(root / "kjv", "another-model")


def test_bm25_and_hybrid_search_find_names_embeddings_miss():
    assert tokens("¿Cómo venció David a Goliat?") == ["vencio", "david", "goliat"]

    class ArbitraryEmbedder:  # ranks passages at random: only keywords can find the target
        model_name = "arbitrary"

        def embed(self, texts, show_progress=False):
            seeds = [int(hashlib.md5(t.encode()).hexdigest(), 16) % 2**32 for t in texts]
            v = np.array([np.random.default_rng(s).normal(size=16) for s in seeds], "float32")
            return v / np.linalg.norm(v, axis=1, keepdims=True)

    emb = ArbitraryEmbedder()
    filler = [passage("kjv", "Psalms", n, 1, 1, f"praise the lord, song {n}") for n in range(1, 40)]
    target = passage("kjv", "1 Samuel", 17, 49, 50, "the stone sank into Goliath forehead")
    idx = VersionIndex.build([*filler, target], emb, {})
    assert BM25([p.text for p in idx.passages]).search("Goliath", 3) == [39]
    question = "How did David defeat Goliath?"
    vector = emb.embed([question])[0]
    assert target not in [h.passage for h in idx.search(vector, 3)]
    hybrid = idx.search(vector, 3, question)
    assert hybrid[0].passage == target and all(-1 <= h.score <= 1 for h in hybrid)


def test_search_per_version_and_validation(rag):
    hits = rag.search("the Lord is my shepherd", k=1)
    assert [h.passage.version for h in hits] in (["KJV", "RV1909"], ["RV1909", "KJV"])
    assert hits[0].passage.citation == "Psalms 23:1-2 (KJV)"
    assert {h.passage.version for h in rag.search("Dios", versions=["rv1909"])} == {"RV1909"}
    with pytest.raises(ValueError, match="Unknown"):
        rag.search("x", versions=["niv"])
    with pytest.raises(ValueError, match="empty"):
        rag.search(" ")


def test_build_indexes_falls_back_and_open_indexes_follows(settings, monkeypatch):
    def load(spec, *a):
        if spec.startswith("hf:"):
            raise ModelUnavailable("Hugging Face unreachable")
        return FakeEmbedder()

    monkeypatch.setattr("bible_rag.index.load_embedder", load)
    units = {
        v: [Unit(p.book, p.chapter, p.verse_start, p.verse_end, p.text, 0, "NT") for p in ps]
        for v, ps in PASSAGES.items()
    }
    monkeypatch.setattr("bible_rag.index.load_units", lambda v, s: units[v])
    settings.embedding_chain = ("hf:some/model", "usem3")
    built = build_indexes(settings)
    assert built["kjv"].manifest["embedder"] == "usem3"
    assert build_indexes(settings) == {}  # already built
    embedder, indexes, spec, skipped = open_indexes(settings)
    assert spec == "usem3" and set(indexes) == {"kjv", "rv1909"}
    assert skipped == [("hf:some/model", "no current index (run: bible-rag ingest)")]
    # an index built from another source (e.g. the old KJV PDF) is stale: rebuilt
    manifest = settings.index_dir / "usem3" / "kjv" / "manifest.json"
    manifest.write_text(manifest.read_text().replace("sword:engKJV2006eb", "pdf:kjv.pdf"))
    assert set(build_indexes(settings)) == {"kjv"}
    settings.embedding_chain = ("hf:some/model",)
    with pytest.raises(IndexNotFoundError, match="bible-rag ingest"):
        open_indexes(settings)


# ------------------------------------------------------------------------ LLM chain
def test_chain_skips_models_that_do_not_load(fake_llm):
    chain = make_chain(ModelUnavailable("offline"), OSError("no memory"), fake_llm)
    assert chain.model is fake_llm and chain.active == "m2"
    assert [s for s, _ in chain.skipped] == ["m0", "m1"]
    with pytest.raises(RuntimeError, match="m0 -> ModelUnavailable: a"):
        make_chain(ModelUnavailable("a")).model  # noqa: B018


def test_failover_when_a_model_breaks_while_answering(rag, fake_llm):
    rag.chain = make_chain(FakeLLM(fail=True), fake_llm)
    answer = rag.ask("In the beginning", k=1)
    assert answer.text.startswith("According to") and answer.model == "m1"
    assert "model server went away" in answer.notes[-1]


def test_prompt_and_grounded_answer(rag, fake_llm):
    answer = rag.ask("¿Cómo amó Dios al mundo?", k=1)
    system, user = fake_llm.calls[-1]
    assert system["content"] == SYSTEM_PROMPT and "Answer in Spanish" in user["content"]
    assert "(KJV)" in user["content"] and "(RV1909)" in user["content"]
    assert answer.grounding.grounded and answer.notes == []


def test_extractive_and_english_only_models(rag):
    rag.chain = make_chain(None)  # extractive
    es = rag.ask("¿Jehová es mi pastor?", k=1)
    assert es.text.startswith("Pasajes que mejor responden")
    assert "Salmos 23:1-2 (RV1909): “Jehová es mi pastor" in es.text and es.grounding.grounded
    small = FakeLLM()
    small.languages, small.label, small.compact = {"en"}, "smollm2", True
    rag.chain = make_chain(small)
    es = rag.ask("¿Quién es mi pastor?", k=1)
    assert (
        es.model == "extractive"
        and "smollm2 does not answer in 'es'." in es.notes
        and not small.calls
    )
    rag.ask("Who is my shepherd?", k=2)
    system, user = small.calls[-1]
    assert system["content"] == COMPACT_PROMPT and "(RV1909)" not in user["content"]


@pytest.mark.parametrize(
    "text,lang",
    [
        ("¿Qué dice la Biblia sobre el amor?", "es"),
        ("Que dice Juan sobre la fe", "es"),
        ("What does the Bible say about love?", "en"),
    ],
)
def test_detect_language(text, lang):
    assert detect_language(text) == lang


# --------------------------------------------------------------------------- grounding
@pytest.mark.parametrize(
    "name,index",
    [
        ("John", 42),
        ("Juan", 42),
        ("1 Cor.", 45),
        ("Gen", 0),
        ("Psalms", 18),
        ("Cantar de los Cantares", 21),
        ("Chapter", None),
    ],
)
def test_book_index(name, index):
    assert book_index(name) == index


def test_grounding_checks_references_and_quotes():
    g = check_answer(
        'As John 3:16 says (Juan 3:17); "charity suffereth long" (1 Cor. 13:4). '
        'The word "rea" and “JEHOVA ES MI PASTOR; nada me faltara”, "John 3:16-17".',
        HITS,
    )
    assert [(c.text, c.supported) for c in g.references] == [
        ("John 3:16", True),
        ("Juan 3:17", True),
        ("1 Cor. 13:4", False),
        ("John 3:16-17", True),  # a quoted reference counts as a reference
    ]
    assert [(c.text, c.supported) for c in g.quotes] == [
        ("charity suffereth long", False),
        ("JEHOVA ES MI PASTOR; nada me faltara", True),
    ]
    assert not g.grounded and len(g.unsupported) == 2
    assert not check_answer("God is love.", HITS).grounded  # nothing to verify


# -------------------------------------------------------------------------- evaluation
def test_evaluate(rag, monkeypatch):
    monkeypatch.setattr(
        evaluation,
        "GOLD",
        [
            ("The Lord is my shepherd", "Jehová es mi pastor", [("Psalms", 23)]),
            ("In the beginning God created", "En el principio creó Dios", [("Exodus", 20)]),
        ],
    )
    report = evaluation.evaluate(rag, k=1)
    s = report["summary"]
    assert s["questions"] == 4 and s["retrieval_hit_rate"]["en->KJV"] == 0.5
    assert s["answers"]["models"] == {"m0": 4}
    assert report["questions"][0]["cites_gold"] is True
    for _, _, gold in evaluation.GOLD:  # gold book names exist
        assert all(book in evaluation._IDX for book, _ in gold)


def test_use_embedder_is_cross_lingual():
    from bible_rag.models import USEEmbedder

    v = USEEmbedder().embed(
        [
            "For God so loved the world, that he gave his only begotten Son",
            "Porque de tal manera amó Dios al mundo, que ha dado a su Hijo",
            "The Lord is my shepherd; I shall not want.",
        ]
    )
    assert v.shape == (3, 512) and v[0] @ v[1] > 0.5 > v[0] @ v[2]
