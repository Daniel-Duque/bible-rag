"""Parser tests on synthetic pages that reproduce the layouts of the two PDFs."""

from bible_rag.corpus import (
    VERSIONS,
    Unit,
    chunk_units,
    format_reference,
    join_units_text,
    normalize,
    parse_kjv,
    parse_rvr,
    resolve_line_hyphens,
)

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


def test_format_reference():
    assert format_reference("Juan", 3, 16, 16) == "Juan 3:16"
    assert format_reference("John", 3, 16, 18) == "John 3:16-18"


def test_join_units_text_rejoins_hyphenated_words():
    units = [
        Unit("B", 1, 1, 1, "las tinieblas es", 1, "OT", join_next=True),
        Unit("B", 1, 1, 1, "taban sobre", 1, "OT"),
        Unit("B", 1, 1, 1, "Tubal-", 1, "OT", join_next=True),
        Unit("B", 1, 1, 1, "caín fue", 1, "OT"),
    ]
    assert join_units_text(units) == "las tinieblas estaban sobre Tubal-caín fue"


# ----------------------------------------------------------------------------- KJV


def test_kjv_skips_front_matter_and_parses_verses():
    pages = [
        kjv_page("Contents\nGenesis 1 26", "xii"),  # table of contents
        kjv_page("THE TRANSLATORS TO THE READER\nblah", "The Translators to the Reader"),
        kjv_page("Old Testament", "Old Testament"),
        kjv_page("The First Book of Moses, called\nGenesis", "Genesis"),
        kjv_page(
            "Chapter 1\nIn the beginning God created the heaven and the earth.\n"
            "2And the earth was without form, and void; and darkness was upon\n"
            "the face of the deep.\n3\nAnd God said, Let there be light: and there was "
            "light.",
            "Genesis 1",
        ),
        kjv_page(
            "4And God saw the light, that it was good: and God divided the li\ufffeght.",
            "Genesis 1",
        ),
        kjv_page("Chapter 2\nThus the heavens and the earth were finished.", "Genesis 2"),
        kjv_page("Indexes", "Indexes"),
        kjv_page("Index of Scripture References\nGalatians 4:16", "Index of Scripture References"),
    ]
    units = list(parse_kjv(pages))
    assert refs(units) == [
        ("Genesis", 1, 1, 1),
        ("Genesis", 1, 2, 2),
        ("Genesis", 1, 3, 3),
        ("Genesis", 1, 4, 4),
        ("Genesis", 2, 1, 1),
    ]
    assert units[1].text == (
        "And the earth was without form, and void; and darkness was upon the face of the deep."
    )
    assert units[2].text.startswith("And God said")  # verse number on its own line
    assert units[3].text.endswith("divided the light.")  # pdfium soft hyphen removed
    assert {u.section for u in units} == {"OT"}


def test_kjv_resets_between_single_chapter_books_and_labels_apocrypha():
    pages = [
        kjv_page("Old Testament", "Old Testament"),
        kjv_page("Obadiah", "Obadiah"),
        kjv_page(
            "Chapter 1\nThe vision of Obadiah.\n2Behold, I have made thee small.", "Obadiah 1"
        ),
        kjv_page("Jonah", "Jonah"),
        kjv_page("Chapter 1\nNow the word of the Lord came unto Jonah.", "Jonah 1"),
        kjv_page("Apocrypha", "Apocryphal or Deuterocanonical Books"),
        kjv_page("Susanna", "Susanna"),
        kjv_page("Chapter 1\nThere dwelt a man in Babylon.\n2And he took a wife.", "Chapter 1"),
    ]
    units = list(parse_kjv(pages))
    assert refs(units) == [
        ("Obadiah", 1, 1, 1),
        ("Obadiah", 1, 2, 2),
        ("Jonah", 1, 1, 1),
        ("Susanna", 1, 1, 1),
        ("Susanna", 1, 2, 2),
    ]
    assert units[-1].section == "Apocrypha"


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
    chunks = list(chunk_units(units, VERSIONS["kjv"], chunk_size=150, chunk_overlap=60))
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
