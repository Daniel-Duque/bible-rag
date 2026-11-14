"""Shared fixtures. Nothing here downloads models: embeddings and the LLM are faked."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator

import numpy as np
import pytest

from bible_rag.config import Settings
from bible_rag.corpus import Passage
from bible_rag.index import VersionIndex, embedder_key
from bible_rag.llm import ModelChain
from bible_rag.rag import BibleRAG

DIM = 64
FAKE_EMBEDDER = "fake/hashing-embedder"


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder (lexical overlap => similarity)."""

    model_name = FAKE_EMBEDDER

    def embed(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), DIM), dtype="float32")
        for row, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                out[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


class FakeLLM:
    """Answers 'According to <first passage>, ...' citing the first context passage."""

    model_name = "fake/echo-llm"
    label = "fake"
    languages = None
    compact = False

    def __init__(self, fail: bool = False):
        self.calls: list[list[dict]] = []
        self.fail = fail

    def _reply(self, messages: list[dict]) -> str:
        if self.fail:
            raise ConnectionError("model server went away")
        self.calls.append(messages)
        content = messages[-1]["content"]
        first = re.search(r"\[1\] (.+)|Passages:\n(.+?\)):", content)
        first = first.group(1) or first.group(2)
        return f"According to {first}, God loves the world."

    def generate(self, messages: list[dict]) -> str:
        return self._reply(messages)

    def stream(self, messages: list[dict]) -> Iterator[str]:
        yield from re.findall(r"\S+\s*", self._reply(messages))  # tokens keep their spaces


def make_passage(version: str, book: str, chapter: int, v1: int, v2: int, text: str) -> Passage:
    return Passage(
        version=version.upper() if version == "kjv" else "RVR1960",
        language="en" if version == "kjv" else "es",
        book=book,
        chapter=chapter,
        verse_start=v1,
        verse_end=v2,
        text=text,
        page=1,
        section="NT",
    )


PASSAGES = {
    "kjv": [
        make_passage(
            "kjv",
            "John",
            3,
            16,
            17,
            "For God so loved the world, that he gave his "
            "only begotten Son, that whosoever believeth in him should not perish.",
        ),
        make_passage(
            "kjv",
            "Genesis",
            1,
            1,
            2,
            "In the beginning God created the heaven and "
            "the earth. And the earth was without form, and void.",
        ),
        make_passage(
            "kjv",
            "Psalm",
            23,
            1,
            2,
            "The Lord is my shepherd; I shall not want. He "
            "maketh me to lie down in green pastures.",
        ),
    ],
    "rvr1960": [
        make_passage(
            "rvr1960",
            "Juan",
            3,
            16,
            17,
            "Porque de tal manera amó Dios al mundo, que "
            "ha dado a su Hijo unigénito, para que todo aquel que en él cree, no se "
            "pierda.",
        ),
        make_passage(
            "rvr1960",
            "Génesis",
            1,
            1,
            2,
            "En el principio creó Dios los cielos y la "
            "tierra. Y la tierra estaba desordenada y vacía.",
        ),
        make_passage(
            "rvr1960",
            "Salmos",
            23,
            1,
            2,
            "Jehová es mi pastor; nada me faltará. En "
            "lugares de delicados pastos me hará descansar.",
        ),
    ],
}


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        index_dir=tmp_path / "index",
        data_dir=tmp_path / "data",
        top_k=2,
        model_cache_dir=tmp_path / "models",
        embedding_chain=(FAKE_EMBEDDER,),
        connect_timeout=0.5,
    )


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def indexes(settings, embedder) -> dict[str, VersionIndex]:
    built = {}
    for version, passages in PASSAGES.items():
        idx = VersionIndex.build(version, passages, embedder, {"chunk_size": 450})
        idx.save(settings.index_dir / embedder_key(FAKE_EMBEDDER) / version)
        built[version] = idx
    return built


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


def make_chain(*models) -> ModelChain:
    """Chain over prepared objects; an Exception instance means 'fails to load'."""
    table = {f"m{i}": m for i, m in enumerate(models)}

    def factory(spec):
        model = table[spec]
        if isinstance(model, Exception):
            raise model
        return model

    return ModelChain(list(table), factory)


@pytest.fixture
def rag(settings, embedder, indexes, fake_llm) -> BibleRAG:
    return BibleRAG(settings, embedder, indexes, make_chain(fake_llm), FAKE_EMBEDDER)
