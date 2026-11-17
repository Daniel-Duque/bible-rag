"""Fakes and fixtures: nothing here downloads a model."""

import hashlib
import re

import numpy as np
import pytest

from bible_rag.config import Settings
from bible_rag.corpus import Passage
from bible_rag.index import VersionIndex
from bible_rag.models import ModelChain, embedder_key
from bible_rag.rag import BibleRAG
from bible_rag.sources import VERSIONS

FAKE_EMBEDDER = "fake"


class FakeEmbedder:
    """Bag-of-words hashing: lexical overlap = similarity."""

    model_name, dims = FAKE_EMBEDDER, 64

    def embed(self, texts):
        out = np.zeros((len(texts), self.dims), dtype="float32")
        for row, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                out[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dims] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


class FakeLLM:
    """Replies 'According to <first passage>, ...'; fail=True raises like a dead server."""

    model_name, label, languages, compact = "fake-llm", "fake", None, False

    def __init__(self, fail=False, reply=None, plan_reply="general"):
        self.fail, self.reply, self.calls = fail, reply, []
        self.plan_reply = plan_reply  # answer to planning calls (routing, statements)

    def stream(self, messages):
        if self.fail:
            raise ConnectionError("model server went away")
        self.calls.append(messages)
        first = re.search(r"\[1\] (.+)|Passages:\n(.+?\)):", messages[-1]["content"])
        if first is None:  # a planning call (routing, statements): no passages to cite
            text = self.plan_reply
        else:
            text = self.reply or f"According to {first[1] or first[2]}, God loves the world."
        yield from re.findall(r"\S+\s*", text)

    def generate(self, messages, max_new_tokens=None):
        return "".join(self.stream(messages))


def passage(version, book, chapter, v1, v2, text):
    v = VERSIONS[version]
    return Passage(v.label, v.language, book, chapter, v1, v2, text)


# fmt: off
PASSAGES = {
    "kjv": [
        passage("kjv", "John", 3, 16, 17, "For God so loved the world, that he gave his only "
                "begotten Son, that whosoever believeth in him should not perish."),
        passage("kjv", "Genesis", 1, 1, 2, "In the beginning God created the heaven and the "
                "earth. And the earth was without form, and void."),
        passage("kjv", "Psalms", 23, 1, 2, "The Lord is my shepherd; I shall not want. He maketh "
                "me to lie down in green pastures."),
    ],
    "rv1909": [
        passage("rv1909", "Juan", 3, 16, 17, "Porque de tal manera amó Dios al mundo, que ha dado "
                "á su Hijo unigénito, para que todo aquel que en él cree, no se pierda."),
        passage("rv1909", "Génesis", 1, 1, 2, "En el principio crió Dios los cielos y la tierra. "
                "Y la tierra estaba desordenada y vacía."),
        passage("rv1909", "Salmos", 23, 1, 2, "Jehová es mi pastor; nada me faltará. En lugares "
                "de delicados pastos me hará yacer."),
    ],
}
# fmt: on


def make_chain(*models):
    """Chain over prepared objects; an Exception instance means 'fails to load'."""

    def factory(spec):
        model = models[int(spec[1:])]
        if isinstance(model, Exception):
            raise model
        return model

    return ModelChain([f"m{i}" for i in range(len(models))], factory)


class MeaningEmbedder:
    """The real bundled multilingual encoder (offline, deterministic) under the fake's name:
    routing, topics and opposition are decided by meaning, so tests need real embeddings."""

    model_name = FAKE_EMBEDDER

    def __init__(self, model):
        self._m = model
        self.languages = model.languages  # no Hebrew / Greek, like the real one

    def embed(self, texts):
        return self._m.embed(texts)


@pytest.fixture(scope="session")
def meaning():
    from bible_rag.models import USEEmbedder

    return MeaningEmbedder(USEEmbedder())


@pytest.fixture(autouse=True)
def no_shipped_prebuild(monkeypatch, tmp_path_factory):
    """Tests never see the real storage/prebuilt (unless they ask for it)."""
    monkeypatch.setenv("BIBLE_RAG_PREBUILT_DIR", str(tmp_path_factory.mktemp("prebuilt")))


@pytest.fixture
def settings(tmp_path):
    return Settings(
        versions=("kjv", "rv1909"),
        index_dir=tmp_path / "index",
        prebuilt_dir=tmp_path / "prebuilt",  # never the real shipped one
        model_cache_dir=tmp_path / "models",
        top_k=2,
        embedding_chain=(FAKE_EMBEDDER,),
        connect_timeout=0.5,
        nli_model="",  # tests never download
    )


@pytest.fixture
def indexes(settings, meaning):
    built = {}
    for version, passages in PASSAGES.items():
        manifest = {"embedder": FAKE_EMBEDDER, "source": VERSIONS[version].source}
        built[version] = VersionIndex.build(passages, meaning, manifest)
        built[version].save(settings.index_dir / embedder_key(FAKE_EMBEDDER) / version)
    return built


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def rag(settings, indexes, fake_llm, meaning):
    rag = BibleRAG(settings, meaning, indexes, make_chain(fake_llm), FAKE_EMBEDDER)
    rag.nli = None  # no NLI model in unit tests (see test_agent for a fake one)
    return rag
