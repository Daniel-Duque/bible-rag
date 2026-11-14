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

FAKE_EMBEDDER = "fake"


class FakeEmbedder:
    """Bag-of-words hashing: lexical overlap = similarity."""

    model_name = FAKE_EMBEDDER

    def embed(self, texts, show_progress=False):
        out = np.zeros((len(texts), 64), dtype="float32")
        for row, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                out[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


class FakeLLM:
    """Replies 'According to <first passage>, ...'; fail=True raises like a dead server."""

    model_name, label, languages, compact = "fake-llm", "fake", None, False

    def __init__(self, fail=False, reply=None):
        self.fail, self.reply, self.calls = fail, reply, []

    def stream(self, messages):
        if self.fail:
            raise ConnectionError("model server went away")
        self.calls.append(messages)
        first = re.search(r"\[1\] (.+)|Passages:\n(.+?\)):", messages[-1]["content"])
        text = self.reply or f"According to {first[1] or first[2]}, God loves the world."
        yield from re.findall(r"\S+\s*", text)

    def generate(self, messages):
        return "".join(self.stream(messages))


def passage(version, book, chapter, v1, v2, text):
    en = version == "kjv"
    return Passage(
        "KJV" if en else "RVR1960", "en" if en else "es", book, chapter, v1, v2, text, 1, "NT"
    )


PASSAGES = {
    "kjv": [
        passage(
            "kjv",
            "John",
            3,
            16,
            17,
            "For God so loved the world, that he gave his only "
            "begotten Son, that whosoever believeth in him should not perish.",
        ),
        passage(
            "kjv",
            "Genesis",
            1,
            1,
            2,
            "In the beginning God created the heaven and the "
            "earth. And the earth was without form, and void.",
        ),
        passage(
            "kjv",
            "Psalm",
            23,
            1,
            2,
            "The Lord is my shepherd; I shall not want. He maketh "
            "me to lie down in green pastures.",
        ),
    ],
    "rvr1960": [
        passage(
            "rvr1960",
            "Juan",
            3,
            16,
            17,
            "Porque de tal manera amó Dios al mundo, que ha "
            "dado a su Hijo unigénito, para que todo aquel que en él cree, no se pierda.",
        ),
        passage(
            "rvr1960",
            "Génesis",
            1,
            1,
            2,
            "En el principio creó Dios los cielos y la tierra. "
            "Y la tierra estaba desordenada y vacía.",
        ),
        passage(
            "rvr1960",
            "Salmos",
            23,
            1,
            2,
            "Jehová es mi pastor; nada me faltará. En lugares "
            "de delicados pastos me hará descansar.",
        ),
    ],
}


def make_chain(*models):
    """Chain over prepared objects; an Exception instance means 'fails to load'."""

    def factory(spec):
        model = models[int(spec[1:])]
        if isinstance(model, Exception):
            raise model
        return model

    return ModelChain([f"m{i}" for i in range(len(models))], factory)


@pytest.fixture
def settings(tmp_path):
    return Settings(
        index_dir=tmp_path / "index",
        model_cache_dir=tmp_path / "models",
        top_k=2,
        embedding_chain=(FAKE_EMBEDDER,),
        connect_timeout=0.5,
    )


@pytest.fixture
def indexes(settings):
    built = {}
    for version, passages in PASSAGES.items():
        built[version] = VersionIndex.build(passages, FakeEmbedder(), {"embedder": FAKE_EMBEDDER})
        built[version].save(settings.index_dir / embedder_key(FAKE_EMBEDDER) / version)
    return built


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def rag(settings, indexes, fake_llm):
    return BibleRAG(settings, FakeEmbedder(), indexes, make_chain(fake_llm), FAKE_EMBEDDER)
