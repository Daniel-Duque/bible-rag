"""Parsers: synthetic pages reproducing each PDF layout, then the real PDFs (marker `pdf`)."""

from collections import defaultdict

import pytest

from bible_rag.config import Settings
from bible_rag.corpus import (
    BOOKS,
    Passage,
    Unit,
    _strip_opening,
    chunk_units,
    extract_pages,
    join_units_text,
    normalize,
    parse_rvr,
    resolve_line_hyphens,
)
from bible_rag.sources import load_units

# ----------------------------------------------------------------------------- helpers


def kjv_page(body: str, footer: str, page: int = 1) -> str:
    # pdfium repeats the running title: "<title>\n<page>\n<title>"
    return f"{body}\n{footer}\n{page}\n{footer}"


RVR_TOC = """Libros del Antiguo Testamento
Página Página
{ot}
Libros del Nuevo Testamento
Página Página
{nt}
"""


def rvr_pages(body_pages: dict[int, str], books: list[tuple[str, int]], nt_from: int) -> list[str]:
    """Build a fake RVR PDF: TOC on page index 2, printed page p at index p + 3."""
    lines = [f"{name} . . . . . {page}" for name, page in books]
    toc = RVR_TOC.format(ot="\n".join(lines[:nt_from]), nt="\n".join(lines[nt_from:]))
    n = max(body_pages) + 1
    pages = [""] * n
    pages[2] = toc
    for idx, text in body_pages.items():
        pages[idx] = text
    return pages


def refs(units):
    return [(u.book, u.chapter, u.verse_start, u.verse_end) for u in units]


# ----------------------------------------------------------------------------- generic


def test_normalize_folds_ligatures():
    assert normalize("dĳo ediﬁcó") == "dijo edificó"


def test_reference_format():
    p = Passage("KJV", "en", "John", 3, 16, 16, "x", 1, "NT")
    assert (p.reference, p.citation) == ("John 3:16", "John 3:16 (KJV)")
    p.verse_end = 18
    assert p.reference == "John 3:16-18" and p.embedding_text() == "John 3:16-18: x"


def test_join_units_text_rejoins_hyphenated_words():
    units = [
        Unit("B", 1, 1, 1, "las tinieblas es", 1, "OT", join_next=True),
        Unit("B", 1, 1, 1, "taban sobre", 1, "OT"),
        Unit("B", 1, 1, 1, "Tubal-", 1, "OT", join_next=True),
        Unit("B", 1, 1, 1, "caín fue", 1, "OT"),
    ]
    assert join_units_text(units) == "las tinieblas estaban sobre Tubal-caín fue"


# ----------------------------------------------------------------------------- KJV


# ----------------------------------------------------------------------------- RVR1960


def test_rvr_toc_book_names_and_opening_page():
    pages = rvr_pages(
        {
            4: "libro primero de moisés\ngénesis\nE\n"
            "n el principio creó Dios los cielos y la 1, 2\n"
            "tierra. Y dĳo Dios: Sea 3\nla luz.",
            1304: "el santo ev angelio según\nsan mateo\nL\nibro de la genealogía de Jesucristo 1",
        },
        [("GÉNESIS", 1), ("MATEO", 1301), ("I Samuel", 5000), ("El Apocalipsis", 6000)],
        nt_from=1,
    )
    units = list(parse_rvr(pages[:1305], expected_books=None))
    assert [(u.book, u.section) for u in units] == [("Génesis", "OT")] * 3 + [("Mateo", "NT")]
    assert units[0].text == "En el principio creó Dios los cielos y la"  # drop cap re-attached
    assert refs(units)[:3] == [("Génesis", 1, 1, 2), ("Génesis", 1, 2, 3), ("Génesis", 1, 3, 3)]
    assert units[1].text == "tierra. Y dijo Dios: Sea"  # ligature + margin number removed


def test_rvr_chapter_marker_on_last_line_of_page_uses_next_page():
    # Chapter 2 has two verses; the "3" closing page 1 is ambiguous (verse 3 of chapter 2,
    # or chapter 3). The "2" at the top of the next page says it is a chapter.
    pages = rvr_pages(
        {
            4: "libro\ngénesis\nE\nn el principio creó 1\nDios los cielos. Y 2\nla tierra. 3\n"
            "Fueron, pues, acabados 2\nlos cielos. Y reposó 2\nLa serpiente era astuta. 3",
            5: "3. 2–3 Génesis 2\ny dijo a la mujer 2\ncomeréis del fruto. 3",
        },
        [("GÉNESIS", 1), ("MATEO", 9000)],
        nt_from=1,
    )
    units = list(parse_rvr(pages, expected_books=None))
    assert refs(units) == [
        ("Génesis", 1, 1, 1),
        ("Génesis", 1, 1, 2),
        ("Génesis", 1, 2, 3),
        ("Génesis", 2, 1, 1),
        ("Génesis", 2, 1, 2),
        ("Génesis", 3, 1, 1),  # not "2:3"
        ("Génesis", 3, 1, 2),
        ("Génesis", 3, 2, 3),
    ]


def test_rvr_soft_hyphen_with_hidden_margin_number():
    pages = rvr_pages(
        {
            4: "libro\nsalmos\nB\nienaventurado el varón 1\n",
            5: "1. 2–3 Salmos 2\nsino que en la ley 2\n"
            "de Jehová está su delicia. Publicad en\ufffe3 tre los pueblos sus obras.",
        },
        [("Salmos", 1), ("Mateo", 9000)],
        nt_from=1,
    )
    units = list(parse_rvr(pages, expected_books=None))
    assert join_units_text(units[-2:]) == (
        "de Jehová está su delicia. Publicad entre los pueblos sus obras."
    )
    assert refs(units)[-1] == ("Salmos", 1, 2, 3)  # the hidden "3" was recovered


# ----------------------------------------------------------------------------- chunking


def test_chunks_respect_size_overlap_and_chapters():
    units = [Unit("John", 3, v, v, f"verse {v} " + "x" * 40, 10, "NT") for v in range(1, 11)]
    units += [Unit("John", 4, 1, 1, "new chapter", 12, "NT")]
    chunks = list(chunk_units(units, "KJV", "en", 150, 60))
    assert all(len(c.text) <= 150 for c in chunks)
    assert all(c.chapter == 3 for c in chunks[:-1]) and chunks[-1].reference == "John 4:1"
    # consecutive chunks overlap by one verse
    assert chunks[0].verse_end == chunks[1].verse_start
    assert chunks[0].citation == "John 3:1-3 (KJV)"
    assert chunks[0].embedding_text().startswith("John 3:1-3: verse 1")


def test_resolve_line_hyphens_uses_the_corpus():
    units = [
        Unit("B", 1, 1, 1, "Y habló Jeho-", 1, "OT", join_next=True),
        Unit("B", 1, 1, 1, "vá a Moisés. Jehová dijo", 1, "OT"),
        Unit("B", 1, 1, 1, "y llegó a Bet-", 1, "OT", join_next=True),
        Unit("B", 1, 1, 1, "el, y de Bet-el salió", 1, "OT"),
    ]
    text = join_units_text(resolve_line_hyphens(units))
    assert text == "Y habló Jehová a Moisés. Jehová dijo y llegó a Bet-el, y de Bet-el salió"


def test_strip_opening_layouts():
    # title + drop cap + lowercase continuation
    assert _strip_opening(["libro primero de moisés", "génesis", "E", "n el principio 1"]) == [
        "En el principio 1"
    ]
    # accent line + drop cap + margin verse number before the continuation
    assert _strip_opening(["deuteronomio", "\u0301", "E", "1 stas son las palabras"]) == [
        "1 Éstas son las palabras"
    ]
    assert _strip_opening(["joel", "P", "1, 2 alabra de Jehová", "esto"]) == [
        "1, 2 Palabra de Jehová",
        "esto",
    ]
    # capitalised title
    assert _strip_opening(["Rut", "A", "conteció en los días 1"]) == ["Aconteció en los días 1"]
    # no drop cap at all
    assert _strip_opening(["lamentaciones de", "jeremias", "¡Cómo ha quedado sola! 1"]) == [
        "¡Cómo ha quedado sola! 1"
    ]


# ------------------------------------------- real RVR1960 PDF vs the Debian KJV (~60 s)
DATA = Settings().data_dir
RVR = DATA / "biblia-reina-valera-1960.pdf"
TITLES = (
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


def structure(units):
    """{(book index, chapter): last verse}."""
    index = {name: i for i, (_, en, es) in enumerate(BOOKS) for name in (en, es)}
    chapters = defaultdict(int)
    for u in units:
        key = index[u.book], u.chapter
        chapters[key] = max(chapters[key], u.verse_end)
    return chapters


@pytest.mark.pdf
@pytest.mark.sword
@pytest.mark.skipif(not RVR.exists(), reason="RVR1960 PDF not found")
def test_rvr1960_pdf_matches_the_kjv_versification():
    units = resolve_line_hyphens(list(parse_rvr(extract_pages(RVR))))
    assert [b for b in dict.fromkeys(u.book for u in units)] == [es for _, _, es in BOOKS]
    rvr, kjv = structure(units), structure(load_units("kjv", Settings.from_env()))
    assert len(rvr) == len(kjv) == 1189 and sum(kjv.values()) == 31102
    diffs = {(BOOKS[b][2], c): (v, kjv[b, c]) for (b, c), v in rvr.items() if kjv[b, c] != v}
    # genuine: 3 John has 15 verses in RVR1960; this PDF prints a stray "10" in Psalm 47
    assert diffs == {("3 Juan", 1): (15, 14), ("Salmos", 47): (10, 9)}
    for p in chunk_units(units, "RVR1960", "es", 450, 80):
        assert len(p.text) <= 600 and "Jeho-vá" not in p.text, p.citation
        assert not any(c in p.text for c in "\ufffe\u00ad\u0133\ufb01"), p.citation
        if p.chapter == 1 and p.verse_start == 1:  # no book title leaked into the text
            assert not any(t in p.text[:80].lower() for t in TITLES), p.citation
