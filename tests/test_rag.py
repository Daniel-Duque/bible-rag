import json

import pytest

from bible_rag.index import IndexNotFoundError, VersionIndex
from bible_rag.ingest import ingest
from bible_rag.rag import SYSTEM_PROMPT, detect_language

from .conftest import PASSAGES


def test_index_roundtrip(settings, indexes, embedder):
    loaded = VersionIndex.load(settings.index_dir / "kjv", expected_model=embedder.model_name)
    assert len(loaded) == 3 and loaded.passages == PASSAGES["kjv"]
    manifest = json.loads((settings.index_dir / "kjv" / "manifest.json").read_text())
    assert manifest["embedding_model"] == embedder.model_name and manifest["passages"] == 3


def test_index_load_errors(settings, indexes):
    with pytest.raises(IndexNotFoundError, match="bible-rag ingest"):
        VersionIndex.load(settings.index_dir / "missing")
    with pytest.raises(IndexNotFoundError, match="--force"):
        VersionIndex.load(settings.index_dir / "kjv", expected_model="another/model")


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


def test_ask_builds_grounded_prompt(rag):
    answer = rag.ask("¿Cómo amó Dios al mundo?", k=1)
    system, user = rag.llm.calls[-1]
    assert system == {"role": "system", "content": SYSTEM_PROMPT}
    assert "Answer in Spanish" in user["content"]
    assert "Juan 3:16-17 (RVR1960)" in user["content"] and "(KJV)" in user["content"]
    assert answer.language == "es" and len(answer.sources) == 2
    assert answer.to_dict()["sources"][0]["reference"] in answer.answer


def test_ask_stream_and_retrieval_only(rag):
    hits, tokens = rag.ask_stream("In the beginning", k=1)
    assert "".join(tokens).startswith("According to")
    rag.llm = None
    with pytest.raises(RuntimeError, match="retrieval-only"):
        rag.ask("x")


def test_ingest_builds_and_skips(settings, embedder, monkeypatch):
    monkeypatch.setattr(
        "bible_rag.ingest.load_version", lambda version, *a, **kw: PASSAGES[version]
    )
    built = ingest(settings, embedder=embedder)
    assert set(built) == {"kjv", "rvr1960"}
    assert (settings.index_dir / "rvr1960" / "index.faiss").exists()
    assert not list(settings.index_dir.glob(".*.tmp"))
    assert ingest(settings, embedder=embedder) == {}  # already built
    assert set(ingest(settings, embedder=embedder, force=True, versions=["kjv"])) == {"kjv"}
