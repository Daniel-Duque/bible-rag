"""Regression test on the real PDFs: the parsers must recover the canonical structure.

Takes ~20 s (PDF text extraction). Skip with: pytest -m "not pdf"
"""

from collections import defaultdict

import pytest

from bible_rag.config import Settings
from bible_rag.corpus import VERSIONS, extract_pages, load_version, parse_kjv, parse_rvr

DATA = Settings().data_dir
TITLE_PHRASES = (
    "libro primero",
    "libro segundo",
    "libro tercero",
    "libro cuarto",
    "libro quinto",
    "primer libro",
    "segundo libro",
    "santo evangelio",
    "epístola del apóstol",
    "epístola universal",
)
pytestmark = [
    pytest.mark.pdf,
    pytest.mark.skipif(not (DATA / VERSIONS["kjv"].filename).exists(), reason="PDFs not found"),
]


def structure(units):
    chapters = defaultdict(int)  # (book, chapter) -> last verse
    for u in units:
        if u.section != "Apocrypha":
            chapters[(u.book, u.chapter)] = max(chapters[(u.book, u.chapter)], u.verse_end)
    books = list(dict.fromkeys(b for b, _ in chapters))
    return books, chapters


@pytest.fixture(scope="module")
def kjv():
    return structure(parse_kjv(extract_pages(DATA / VERSIONS["kjv"].filename)))


@pytest.fixture(scope="module")
def rvr():
    return structure(parse_rvr(extract_pages(DATA / VERSIONS["rvr1960"].filename)))


def test_kjv_matches_the_protestant_canon(kjv):
    books, chapters = kjv
    assert len(books) == 66 and books[0] == "Genesis" and books[-1] == "Revelation"
    assert len(chapters) == 1189
    assert sum(chapters.values()) == 31102  # verses in the KJV
    assert chapters[("Psalm", 119)] == 176 and chapters[("John", 3)] == 36


def test_rvr_matches_the_kjv_versification(kjv, rvr):
    kjv_books, kjv_chapters = kjv
    rvr_books, rvr_chapters = rvr
    assert len(rvr_books) == 66 and rvr_books[0] == "Génesis" and rvr_books[-1] == "Apocalipsis"
    assert len(rvr_chapters) == 1189
    to_kjv = dict(zip(rvr_books, kjv_books, strict=True))
    diffs = {
        (b, c): (v, kjv_chapters[(to_kjv[b], c)])
        for (b, c), v in rvr_chapters.items()
        if kjv_chapters[(to_kjv[b], c)] != v
    }
    # Genuine differences: 3 John has 15 verses in RVR1960; the PDF prints a "10" in Psalm 47.
    assert diffs == {("3 Juan", 1): (15, 14), ("Salmos", 47): (10, 9)}


def test_passages_are_clean_and_sized():
    for version in VERSIONS:
        passages = load_version(version, DATA, chunk_size=450, chunk_overlap=80)
        assert len(passages) > 9000
        for p in passages:
            assert len(p.text) <= 600, p.citation  # a single very long verse may exceed 450
            assert not any(ch in p.text for ch in "\ufffe\u00ad\u0133\ufb01"), p.citation
            assert "Jeho-vá" not in p.text, p.citation
            # book titles of opening pages (chapter 1) must not leak into the text
            if p.chapter == 1 and p.verse_start == 1:
                head = p.text[:80].lower()
                assert not any(t in head for t in TITLE_PHRASES), p.citation
        john = next(
            p
            for p in passages
            if p.chapter == 3 and p.book in ("John", "Juan") and p.verse_start <= 16 <= p.verse_end
        )
        assert ("God so loved" in john.text) or ("de tal manera amó Dios" in john.text)
