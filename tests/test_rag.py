"""Index, hybrid search, fallback chains, answers, grounding and evaluation (all with fakes)."""

import hashlib

import numpy as np
import pytest

from bible_rag import evaluation
from bible_rag.agent import COMPACT_PROMPT, GENERAL, PROMPTS
from bible_rag.corpus import Unit
from bible_rag.index import (
    BM25,
    IndexNotFoundError,
    VersionIndex,
    build_indexes,
    open_indexes,
    tokens,
)
from bible_rag.models import ModelUnavailable, embedder_key
from bible_rag.rag import detect_language

from .conftest import FAKE_EMBEDDER, PASSAGES, FakeEmbedder, FakeLLM, make_chain, passage


# ------------------------------------------------------------------------------- index
@pytest.mark.usefixtures("indexes")
def test_index_roundtrip_and_errors(settings):
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

        def embed(self, texts):
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
    assert skipped == []  # not a failure: that embedder simply has no index (no note)
    # an index built from another source is stale: rebuilt
    manifest = settings.index_dir / "usem3" / "kjv" / "manifest.json"
    manifest.write_text(manifest.read_text().replace("sword:engKJV2006eb", "sword:engKJV1769"))
    assert set(build_indexes(settings)) == {"kjv"}
    settings.embedding_chain = ("hf:some/model",)
    with pytest.raises(IndexNotFoundError, match="hf:some/model: no index.*bible-rag ingest"):
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
    assert system["content"] == PROMPTS[GENERAL].format(
        versions="KJV (King James Version (1769)), RV1909 (Reina-Valera 1909)"
    )
    assert "Answer in Spanish" in user["content"]
    assert "(KJV)" in user["content"] and "(RV1909)" in user["content"]
    assert answer.grounding.grounded and answer.notes == [] and answer.intent == "general"
    # the synthesis, then what each translation says at the passages found
    assert (
        answer.text.startswith(answer.synthesis) and "**Qué dice cada traducción**" in answer.text
    )


def test_extractive_and_english_only_models(rag):
    rag.chain = make_chain(None)  # extractive
    es = rag.ask("¿Jehová es mi pastor?", k=1)
    assert es.text.startswith("Pasajes que mejor responden")
    assert "1. Psalms 23:1-2 · Salmos 23:1-2" in es.text
    assert "   - RV1909: “Jehová es mi pastor" in es.text and es.grounding.grounded
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


@pytest.mark.parametrize(
    "text,lang,ui",
    [
        ("Existem paradoxos na Bíblia sobre o álcool?", "pt", "es"),
        ("La Bible se contredit-elle sur la guerre ?", "fr", "en"),
        ("Widerspricht sich die Bibel über Reichtum?", "de", "en"),
        ("Ci sono contraddizioni nella Bibbia sul denaro?", "it", "en"),
        ("¿Dónde nació Jesús?", "es", "es"),
        ("Where was Jesus born?", "en", "en"),
        ("Should I use a cellphone?", "en", "en"),  # not French for its "a"
        ("Can I use it on Sunday?", "en", "en"),
    ],
)
def test_identify_language_for_the_llm(text, lang, ui):
    from bible_rag.rag import identify_language

    assert identify_language(text) == lang and detect_language(text) == ui


def test_llm_answers_in_the_questions_language(rag, fake_llm):
    answer = rag.ask("Onde nasceu Jesus?", k=1)
    assert answer.plan.reply_language == "pt" and answer.language == "es"
    assert "Answer in Portuguese" in fake_llm.calls[-1][1]["content"]


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
    assert s["aligned_hit_rate"]["en->RV1909"] == 0.5  # Salmos 23 shown next to Psalms 23
    assert report["questions"][0]["cites_gold"] is True
    for _, _, gold in evaluation.GOLD:  # gold book names exist
        assert all(book in evaluation._IDX for book, _ in gold)


def test_summary_of_a_retrieval_only_run():
    rows = [{"language": "en", "hit": {"KJV": True, "RV1909": False}, "question": "q"}]
    assert evaluation.summarize(rows) == {
        "questions": 1,
        "retrieval_hit_rate": {"en->KJV": 1.0, "en->RV1909": 0.0, "en->any": 1.0},
    }


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


# --------------------------------------------------- ingestion: sample, lock, tags, progress
def test_sample_corpus_lock_progress_and_tags(settings, monkeypatch, meaning):
    import threading

    from bible_rag.index import SCHEMA, index_is_current, index_root

    units = {
        v: [Unit(p.book, p.chapter, p.verse_start, p.verse_end, p.text, 0, "NT") for p in ps]
        for v, ps in PASSAGES.items()
    }
    units["kjv"][2].tags = [["shepherd", ["H7462"]]]  # Psalms 23:1
    monkeypatch.setattr("bible_rag.index.load_units", lambda v, s: units[v])
    monkeypatch.setattr("bible_rag.sources.strongs_lexicon", lambda d: {})
    settings.books = "sample"  # Gen, Ps, ..., John: all three passages are in it
    assert settings.book_codes[:2] == ("Gen", "Ps") and settings.corpus_suffix == "@sample"
    root = index_root(settings, FAKE_EMBEDDER)
    assert root.name == "fake@sample"
    seen, built = [], []

    def build():
        built.append(
            build_indexes(
                settings,
                embedder=meaning,
                spec=FAKE_EMBEDDER,
                progress=lambda v, d, t: seen.append((v, d, t)),
            )
        )

    threads = [threading.Thread(target=build) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(len(b) for b in built) == [0, 2]  # the second waited, then found them built
    assert ("kjv", 3, 3) in seen and ("rv1909", 3, 3) in seen
    assert index_is_current(root / "kjv", "kjv", settings)
    loaded = VersionIndex.load(root / "kjv", FAKE_EMBEDDER)
    loaded.directory = root / "kjv"
    assert loaded.verse_tags(18, 23, 1, 2) == [["shepherd", ["H7462"]]]
    assert loaded.manifest["schema"] == SCHEMA and loaded.passages[0].starts == [[16, 0]]
    settings.books = "John"  # only John: the other books are left out
    assert [
        p.book for p in build_indexes(settings, embedder=meaning, spec="x")["kjv"].passages
    ] == ["John"]
    settings.books = ""  # no whole-Bible index yet: the sample is opened, and said so
    monkeypatch.setattr("bible_rag.index.load_embedder", lambda *a: meaning)
    _, indexes, _, _ = open_indexes(settings)
    assert settings.books == "sample" and set(indexes) == {"kjv", "rv1909"}


def test_all_versions_setting(monkeypatch):
    from bible_rag.config import Settings
    from bible_rag.sources import VERSIONS

    monkeypatch.setenv("BIBLE_RAG_VERSIONS", "all")
    versions = Settings.from_env().versions
    assert "wlc" in versions and "sblgnt" in versions and "rvr1960" not in versions
    assert set(versions) == {v for v, x in VERSIONS.items() if x.public}


# ------------------------------------------------------------ prebuild, updates, reuse, format
class CountingEmbedder:
    """The test embedder, counting what it embeds."""

    model_name, languages = FAKE_EMBEDDER, None

    def __init__(self, inner):
        self.inner, self.texts = inner, []

    def embed(self, texts):
        self.texts += list(texts)
        return self.inner.embed(texts)


@pytest.fixture
def units(monkeypatch):
    found = {
        v: [Unit(p.book, p.chapter, p.verse_start, p.verse_end, p.text, 0, "NT") for p in ps]
        for v, ps in PASSAGES.items()
    }
    monkeypatch.setattr("bible_rag.index.load_units", lambda v, s: found[v])
    return found


def test_prebuild_is_used_and_updated_incrementally(settings, units, meaning, monkeypatch):
    from bible_rag import index as index_module
    from bible_rag.index import find_index, index_root

    counting = CountingEmbedder(meaning)
    built = build_indexes(settings, embedder=counting, spec=FAKE_EMBEDDER, prebuilt=True)
    assert set(built) == {"kjv", "rv1909"} and len(counting.texts) > 6  # passages + vocabulary
    # a fresh copy: nothing built locally, the shipped prebuild answers
    monkeypatch.setattr("bible_rag.index.load_embedder", lambda *a: meaning)
    _, indexes, _, _ = index_module.open_indexes(settings)
    assert indexes["kjv"].directory == index_root(settings, FAKE_EMBEDDER, True) / "kjv"
    assert not indexes["kjv"].stale
    # ingest: the prebuild is up to date, nothing to do
    assert build_indexes(settings, embedder=counting, spec=FAKE_EMBEDDER) == {}
    # the code that makes passages changes: the indexes are out of date, still usable...
    index_module.code_fingerprint.cache_clear()
    monkeypatch.setattr("bible_rag.index.code_fingerprint", lambda: "changed")
    _, indexes, _, _ = index_module.open_indexes(settings)
    assert indexes["kjv"].stale and find_index(settings, FAKE_EMBEDDER, "kjv")[1] is False
    # ...and an update recomputes only what changed: one KJV verse
    units["kjv"][0].text = "For God so loved the world, he gave his Son."
    counting.texts.clear()
    updated = build_indexes(settings, embedder=counting, spec=FAKE_EMBEDDER)
    assert set(updated) == {"kjv", "rv1909"}
    assert updated["kjv"].reused == 2
    assert updated["rv1909"].reused == 3
    changed = [t for t in counting.texts if "he gave his Son" in t]
    assert len(changed) == 1 and len(counting.texts) <= 1 + 3  # + new vocabulary words only
    _, indexes, _, _ = index_module.open_indexes(settings)  # the local update wins
    assert indexes["kjv"].directory == index_root(settings, FAKE_EMBEDDER) / "kjv"
    assert not indexes["kjv"].stale


def test_compact_format_round_trip(settings, indexes):
    from bible_rag.index import BM25, dequantize, quantize

    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(50, 32)).astype("float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    codes, scales = quantize(vectors)
    assert codes.dtype == np.int8 and np.abs(dequantize(codes, scales) - vectors).max() < 0.01
    root = settings.index_dir / embedder_key(FAKE_EMBEDDER) / "kjv"
    loaded = VersionIndex.load(root, FAKE_EMBEDDER)
    original = indexes["kjv"]
    assert loaded.passages == original.passages and (loaded.hashes == original.hashes).all()
    assert np.abs(loaded.vectors - original.vectors).max() < 0.01
    stored = BM25.load(root / "bm25.npz")
    rebuilt = BM25([p.embedding_text() for p in original.passages])
    for query in ("the Lord is my shepherd", "Dios amó al mundo", "beginning earth", "nothing"):
        assert stored.search(query, 3) == rebuilt.search(query, 3)
    assert stored.docs_of("shepherd").tolist() == [2] and stored.frequent(2)


def test_semantics_cache_makes_a_new_process_embed_nothing(tmp_path, meaning):
    from bible_rag.semantic import Semantics

    first = Semantics(
        CountingEmbedder(meaning.inner if hasattr(meaning, "inner") else meaning),
        tmp_path / "s.npz",
    )
    first.topic_threshold, first.axis  # noqa: B018
    assert first.embedder.texts  # computed once
    second = Semantics(CountingEmbedder(meaning), tmp_path / "s.npz")
    assert second.topic_threshold == pytest.approx(first.topic_threshold, abs=1e-3)
    assert second.embedder.texts == []  # all from the cache
    shipped = Semantics(CountingEmbedder(meaning), None, tmp_path / "s.npz")
    shipped.axis  # noqa: B018
    assert shipped.embedder.texts == []  # read-only shipped copy works too


def test_small_model_is_not_even_loaded_when_it_would_not_answer(rag):
    from bible_rag.models import ModelChain

    loads = []

    def factory(spec):  # a server that isn't there, then the bundled small model
        if spec == "ollama":
            raise ModelUnavailable("no server")
        loads.append(spec)
        return FakeLLM()

    rag.chain = ModelChain(["ollama", "smollm2", "extractive"], factory)
    answer = rag.ask("How do the translations render John 3:16?")  # a comparison
    assert answer.model == "extractive" and loads == []
    assert "smollm2 is too small to compare passages" in answer.notes[-1]
    rag.ask("¿Quién es mi pastor?", k=1)  # Spanish: English-only model, not loaded either
    assert loads == []
    assert "loaded when first needed" in rag.chain.describe()
    rag.ask("Who is my shepherd?", k=1)  # English, direct: now it is used
    assert loads == ["smollm2"] and rag.chain.describe() == "smollm2"


def test_fingerprint_ignores_docs_and_comments_not_code():
    from bible_rag.index import code_digest

    base = 'def f(x):\n    """Docs."""\n    return x + 1  # one\n\ndef g():\n    return 2\n'
    docs = (
        '"""Module."""\ndef f(x):\n    """Other docs."""\n    # a comment\n'
        "    return x + 1\n\ndef g():\n    return 2\n"
    )
    code = base.replace("x + 1", "x + 2")
    assert code_digest(base) == code_digest(docs) != code_digest(code)
    assert code_digest(base, ["g"]) == code_digest(code, ["g"])  # only g is watched


def test_written_files_are_reproducible(tmp_path):
    from bible_rag.index import save_npz, write_gzip

    for n in (1, 2):
        save_npz(tmp_path / f"{n}.npz", True, a=np.arange(5, dtype=np.int8), b=np.ones(3))
        write_gzip(tmp_path / f"{n}.gz", "some text")
    assert (tmp_path / "1.npz").read_bytes() == (tmp_path / "2.npz").read_bytes()
    assert (tmp_path / "1.gz").read_bytes() == (tmp_path / "2.gz").read_bytes()
    assert np.load(tmp_path / "1.npz")["a"].tolist() == [0, 1, 2, 3, 4]


def test_out_of_date_index_is_used_and_said(rag):
    rag.indexes["kjv"].stale = True
    notes = rag.ask("In the beginning", k=1).notes
    assert any("The index of KJV is out of date" in n and "bible-rag ingest" in n for n in notes)


def test_shipped_prebuild_is_current(monkeypatch):
    """storage/prebuilt must match the code: after changing how texts become passages, run
    `BIBLE_RAG_VERSIONS=all uv run bible-rag prebuild` (it only redoes what changed)."""
    from bible_rag.config import ROOT, Settings
    from bible_rag.index import find_index
    from bible_rag.sources import VERSIONS

    monkeypatch.delenv("BIBLE_RAG_PREBUILT_DIR")
    settings = Settings(index_dir=ROOT / "nonexistent")  # the shipped prebuild only
    shipped = [v for v, x in VERSIONS.items() if x.public]
    if not (settings.prebuilt_dir / "usem3").exists():
        pytest.skip("no prebuild in this copy")
    stale = [v for v in shipped if not (find_index(settings, "usem3", v) or (0, False))[1]]
    assert not stale, (
        f"out of date: {stale}; run `BIBLE_RAG_VERSIONS=all uv run bible-rag prebuild`"
    )
    assert (settings.prebuilt_dir / "strongs.json.gz").exists()
    shipped = settings.prebuilt_dir / "semantics-usem3.npz"
    assert shipped.exists()

    class Unavailable:  # every fixed text must come from the shipped file
        model_name = str(np.load(shipped)["model"])

        def embed(self, texts):
            raise AssertionError(f"not in {shipped.name} (run `make prebuild`): {texts[:3]}")

    from bible_rag.semantic import Semantics

    Semantics(Unavailable(), shipped=shipped).warm()
