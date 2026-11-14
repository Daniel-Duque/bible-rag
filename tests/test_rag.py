import json

import pytest

from bible_rag.index import IndexNotFoundError, VersionIndex, embedder_key
from bible_rag.ingest import ingest
from bible_rag.rag import SYSTEM_PROMPT, detect_language, select_embedder

from .conftest import FAKE_EMBEDDER, PASSAGES, FakeEmbedder


def test_index_roundtrip(settings, indexes, embedder):
    root = settings.index_dir / embedder_key(FAKE_EMBEDDER)
    loaded = VersionIndex.load(root / "kjv", expected_model=embedder.model_name)
    assert len(loaded) == 3 and loaded.passages == PASSAGES["kjv"]
    manifest = json.loads((root / "kjv" / "manifest.json").read_text())
    assert manifest["embedding_model"] == embedder.model_name and manifest["passages"] == 3


def test_index_load_errors(settings, indexes):
    with pytest.raises(IndexNotFoundError, match="bible-rag ingest"):
        VersionIndex.load(settings.index_dir / "missing")
    with pytest.raises(IndexNotFoundError, match="--force"):
        VersionIndex.load(
            settings.index_dir / embedder_key(FAKE_EMBEDDER) / "kjv", expected_model="another/model"
        )


def test_search_returns_k_per_version_sorted(rag):
    hits = rag.search("the Lord is my shepherd", k=1)
    assert {h.passage.version for h in hits} == {"KJV", "RVR1960"}
    assert hits[0].passage.citation == "Psalm 23:1-2 (KJV)"
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_search_version_filter_and_validation(rag):
    assert {h.passage.version for h in rag.search("Dios", versions=["rvr1960"])} == {"RVR1960"}
    with pytest.raises(ValueError, match="Unknown"):
        rag.search("x", versions=["niv"])
    with pytest.raises(ValueError, match="empty"):
        rag.search("   ")


@pytest.mark.parametrize(
    "text,lang",
    [
        ("¿Qué dice la Biblia sobre el amor?", "es"),
        ("Que dice Juan sobre la fe", "es"),
        ("What does the Bible say about love?", "en"),
        ("Who was Moses", "en"),
    ],
)
def test_detect_language(text, lang):
    assert detect_language(text) == lang


def test_ask_builds_grounded_prompt(rag, fake_llm):
    answer = rag.ask("¿Cómo amó Dios al mundo?", k=1)
    system, user = fake_llm.calls[-1]
    assert system == {"role": "system", "content": SYSTEM_PROMPT}
    assert "Answer in Spanish" in user["content"]
    assert "Juan 3:16-17 (RVR1960)" in user["content"] and "(KJV)" in user["content"]
    assert answer.language == "es" and len(answer.sources) == 2
    assert answer.to_dict()["sources"][0]["reference"] in answer.answer
    assert answer.model == "m0" and answer.notes == []
    assert answer.grounding.grounded  # it cites the first retrieved passage


def test_ask_stream_and_retrieval_only(rag):
    result = rag.ask_stream("In the beginning", k=1)
    assert "".join(result).startswith("According to")
    assert result.model == "m0" and result.grounding is not None
    rag.chain = None
    with pytest.raises(RuntimeError, match="retrieval-only"):
        rag.ask("x")


def test_ingest_builds_and_skips(settings, embedder, monkeypatch):
    monkeypatch.setattr(
        "bible_rag.ingest.load_version", lambda version, *a, **kw: PASSAGES[version]
    )
    built = ingest(settings, embedder=embedder)
    root = settings.index_dir / embedder_key(FAKE_EMBEDDER)
    assert set(built) == {"kjv", "rvr1960"}
    assert (root / "rvr1960" / "index.faiss").exists()
    assert built["kjv"].manifest["embedder_spec"] == FAKE_EMBEDDER
    assert not list(root.glob(".*.tmp"))
    assert ingest(settings, embedder=embedder) == {}  # already built
    assert set(ingest(settings, embedder=embedder, force=True, versions=["kjv"])) == {"kjv"}


def test_ingest_falls_back_to_the_next_embedder(settings, monkeypatch):
    from bible_rag.hub import ModelUnavailable

    def load(spec, *a, **kw):
        if spec == "hf:unreachable/model":
            raise ModelUnavailable("Hugging Face unreachable")
        return FakeEmbedder()

    monkeypatch.setattr("bible_rag.ingest.load_embedder", load)
    monkeypatch.setattr("bible_rag.ingest.load_version", lambda v, *a, **kw: PASSAGES[v])
    settings.embedding_chain = ("hf:unreachable/model", "usem3")
    built = ingest(settings)
    assert built["kjv"].manifest["embedder_spec"] == "usem3"
    assert (settings.index_dir / "usem3" / "kjv" / "manifest.json").exists()


def test_select_embedder_uses_first_spec_with_index_that_loads(settings, indexes, monkeypatch):
    from bible_rag.hub import ModelUnavailable

    calls = []

    def load(spec, *a, **kw):
        calls.append(spec)
        if spec == "broken":
            raise ModelUnavailable("cannot load")
        return FakeEmbedder()

    monkeypatch.setattr("bible_rag.rag.load_embedder", load)
    # "no-index" has no index -> skipped without loading; "broken" has one but fails to load
    broken_root = settings.index_dir / embedder_key("broken")
    for version, idx in indexes.items():
        idx.save(broken_root / version)
    settings.embedding_chain = ("no-index", "broken", FAKE_EMBEDDER)
    emb, idx, spec, skipped = select_embedder(settings, "cpu")
    assert spec == FAKE_EMBEDDER and set(idx) == {"kjv", "rvr1960"}
    assert calls == ["broken", FAKE_EMBEDDER]
    assert [s for s, _ in skipped] == ["no-index", "broken"]
    assert "bible-rag ingest" in skipped[0][1]

    settings.embedding_chain = ("no-index",)
    with pytest.raises(IndexNotFoundError, match="bible-rag ingest"):
        select_embedder(settings, "cpu")


def test_use_multilingual_embedder_is_cross_lingual():
    """The bundled offline fallback embedder really aligns English and Spanish."""
    from bible_rag.index import USEEmbedder

    v = USEEmbedder().embed(
        [
            "For God so loved the world, that he gave his only begotten Son",
            "Porque de tal manera amó Dios al mundo, que ha dado a su Hijo unigénito",
            "The Lord is my shepherd; I shall not want.",
        ]
    )
    assert v.shape == (3, 512)
    assert v[0] @ v[1] > 0.5 > v[0] @ v[2]


def test_lexical_index_bm25_ranks_rare_terms_first():
    from bible_rag.index import LexicalIndex, lexical_tokens

    assert lexical_tokens("¿Cómo venció David a Goliat?") == ["vencio", "david", "goliat"]
    idx = LexicalIndex(
        [
            "David the king reigned over Israel",
            "David went out against Goliath the Philistine",
            "And David said unto Saul",
        ]
    )
    assert [i for i, _ in idx.search("How did David defeat Goliath?", 3)][0] == 1
    assert idx.search("unknown words only", 3) == []


def test_hybrid_search_recovers_keyword_matches():
    """A passage the embedder misses is found through an exact keyword (a name)."""
    import hashlib

    import numpy as np

    from bible_rag.index import VersionIndex

    from .conftest import make_passage

    class ArbitraryEmbedder:  # semantically useless: ranking by embedding is arbitrary
        model_name = "arbitrary"

        def embed(self, texts, show_progress=False):
            vecs = [
                np.random.default_rng(int(hashlib.md5(t.encode()).hexdigest(), 16) % 2**32).normal(
                    size=16
                )
                for t in texts
            ]
            v = np.asarray(vecs, dtype="float32")
            return v / np.linalg.norm(v, axis=1, keepdims=True)

    emb = ArbitraryEmbedder()
    filler = [
        make_passage("kjv", "Psalm", n, 1, 1, f"praise the lord, song number {n}")
        for n in range(1, 40)
    ]
    target = make_passage("kjv", "1 Samuel", 17, 49, 50, "the stone sank into Goliath forehead")
    idx = VersionIndex.build("kjv", [*filler, target], emb, {})
    query = emb.embed(["How did David defeat Goliath?"])[0]
    dense = [h.passage for h in idx.search(query, 3)]
    hybrid = idx.search(query, 3, query_text="How did David defeat Goliath?")
    assert target not in dense
    assert hybrid[0].passage == target
    assert all(-1.0 <= h.score <= 1.0 for h in hybrid)  # cosine kept for display


def test_retrieval_mode_setting(rag):
    rag.settings.retrieval = "dense"
    dense = rag.search("The Lord is my shepherd", k=1)
    rag.settings.retrieval = "hybrid"
    hybrid = rag.search("The Lord is my shepherd", k=1)
    assert dense[0].passage == hybrid[0].passage  # same best hit, both modes work
