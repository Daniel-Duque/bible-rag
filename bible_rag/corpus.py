"""Turn the two Bible PDFs into verse-referenced passages ready for embedding.

The notebook (RAG0.ipynb) loaded PDFs page by page and split them into fixed-size
character chunks, so the only citation available was "file name". For a Bible the
useful citation is *Book chapter:verse*, so each version gets a small parser that
understands its PDF layout:

* **KJV** (``kjv.pdf``, XEP-rendered): one chapter per page, the page footer is
  ``<page number>`` + ``<Book chapter>``, and every verse starts on its own line
  with its number glued to the text (``21It devoureth...``). Verse references are exact.
* **RVR1960** (``biblia-reina-valera-1960.pdf``, pdfTeX): running header with the
  verse range of the page (``4. 15–5. 9 Génesis 6``) and verse numbers in the
  margin, i.e. a number marks the *line* where a verse starts. References are
  therefore line-accurate (a passage may include the tail of the previous verse).

Both parsers emit :class:`Unit` objects (a verse for KJV, a printed line for
RVR1960) that :func:`chunk_units` groups into passages of ~``chunk_size`` chars
without crossing chapter boundaries.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

SOFT_HYPHEN = "\u00ad"  # sentinel used for "word continues on next line"
PDFIUM_SOFT_HYPHEN = "\ufffe"  # how pdfium reports TeX's discretionary hyphens


# --------------------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class VersionSpec:
    id: str  # short id used in the index path / CLI ("kjv", "rvr1960")
    label: str  # printed in citations ("KJV", "RVR1960")
    name: str
    language: str  # ISO 639-1
    filename: str  # PDF file name inside the data directory


VERSIONS: dict[str, VersionSpec] = {
    "kjv": VersionSpec("kjv", "KJV", "King James Version", "en", "kjv.pdf"),
    "rvr1960": VersionSpec(
        "rvr1960", "RVR1960", "Reina-Valera 1960", "es", "biblia-reina-valera-1960.pdf"
    ),
}


# Canonical order; names exactly as the parsers produce them (KJV, RVR1960).
BOOKS: list[tuple[str, str]] = [
    ("Genesis", "Génesis"),
    ("Exodus", "Éxodo"),
    ("Leviticus", "Levítico"),
    ("Numbers", "Números"),
    ("Deuteronomy", "Deuteronomio"),
    ("Joshua", "Josué"),
    ("Judges", "Jueces"),
    ("Ruth", "Rut"),
    ("1 Samuel", "1 Samuel"),
    ("2 Samuel", "2 Samuel"),
    ("1 Kings", "1 Reyes"),
    ("2 Kings", "2 Reyes"),
    ("1 Chronicles", "1 Crónicas"),
    ("2 Chronicles", "2 Crónicas"),
    ("Ezra", "Esdras"),
    ("Nehemiah", "Nehemías"),
    ("Esther", "Ester"),
    ("Job", "Job"),
    ("Psalm", "Salmos"),
    ("Proverbs", "Proverbios"),
    ("Ecclesiastes", "Eclesiastés"),
    ("Song of Solomon", "Cantar de los Cantares"),
    ("Isaiah", "Isaías"),
    ("Jeremiah", "Jeremías"),
    ("Lamentations", "Lamentaciones"),
    ("Ezekiel", "Ezequiel"),
    ("Daniel", "Daniel"),
    ("Hosea", "Oseas"),
    ("Joel", "Joel"),
    ("Amos", "Amós"),
    ("Obadiah", "Abdías"),
    ("Jonah", "Jonás"),
    ("Micah", "Miqueas"),
    ("Nahum", "Nahum"),
    ("Habakkuk", "Habacuc"),
    ("Zephaniah", "Sofonías"),
    ("Haggai", "Hageo"),
    ("Zechariah", "Zacarías"),
    ("Malachi", "Malaquías"),
    ("Matthew", "Mateo"),
    ("Mark", "Marcos"),
    ("Luke", "Lucas"),
    ("John", "Juan"),
    ("Acts", "Hechos"),
    ("Romans", "Romanos"),
    ("1 Corinthians", "1 Corintios"),
    ("2 Corinthians", "2 Corintios"),
    ("Galatians", "Gálatas"),
    ("Ephesians", "Efesios"),
    ("Philippians", "Filipenses"),
    ("Colossians", "Colosenses"),
    ("1 Thessalonians", "1 Tesalonicenses"),
    ("2 Thessalonians", "2 Tesalonicenses"),
    ("1 Timothy", "1 Timoteo"),
    ("2 Timothy", "2 Timoteo"),
    ("Titus", "Tito"),
    ("Philemon", "Filemón"),
    ("Hebrews", "Hebreos"),
    ("James", "Santiago"),
    ("1 Peter", "1 Pedro"),
    ("2 Peter", "2 Pedro"),
    ("1 John", "1 Juan"),
    ("2 John", "2 Juan"),
    ("3 John", "3 Juan"),
    ("Jude", "Judas"),
    ("Revelation", "Apocalipsis"),
]


@dataclass
class Passage:
    """A retrievable chunk: a few consecutive verses of one chapter in one version."""

    version: str  # VersionSpec.label
    language: str
    book: str
    chapter: int
    verse_start: int
    verse_end: int
    text: str
    page: int
    section: str

    @property
    def reference(self) -> str:
        return format_reference(self.book, self.chapter, self.verse_start, self.verse_end)

    @property
    def citation(self) -> str:
        return f"{self.reference} ({self.version})"

    def embedding_text(self) -> str:
        # The reference is part of the embedded text so "Juan 3" / "Psalm 23" style
        # queries get some signal too.
        return f"{self.reference}: {self.text}"


@dataclass
class Unit:
    """Smallest piece of text with a known position (a verse or a printed line)."""

    book: str
    chapter: int
    verse_start: int
    verse_end: int
    text: str
    page: int  # 1-based PDF page index (what a PDF viewer shows)
    section: str  # "OT", "NT" or "Apocrypha"
    join_next: bool = False  # text ends mid-word: join with the next unit without a space


# --------------------------------------------------------------------------------------
# PDF extraction + generic text helpers
# --------------------------------------------------------------------------------------


def extract_pages(pdf_path: Path) -> list[str]:
    """Return the raw text of every page. pypdfium2 is ~10x faster than pypdf here."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        pages = []
        for page in doc:
            textpage = page.get_textpage()
            pages.append(textpage.get_text_range().replace("\r\n", "\n").replace("\r", "\n"))
            textpage.close()
            page.close()
        return pages
    finally:
        doc.close()


def normalize(text: str) -> str:
    """NFKC folds the typographic ligatures in the PDFs (``ĳ`` -> ``ij``, ``ﬁ`` -> ``fi``)."""
    return unicodedata.normalize("NFKC", text).replace("\u00a0", " ")


def join_units_text(units: Iterable[Unit]) -> str:
    out: list[str] = []
    glue = ""
    for u in units:
        out.append(glue + u.text)
        glue = "" if u.join_next else " "
    return re.sub(r"\s+", " ", "".join(out)).strip()


def _lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.split("\n") if ln.strip()]


# --------------------------------------------------------------------------------------
# KJV
# --------------------------------------------------------------------------------------

KJV_SECTION_MARKERS = {
    "Old Testament": "OT",
    "Apocryphal or Deuterocanonical Books": "Apocrypha",
    "New Testament": "NT",
}
_KJV_FOOTER_CHAPTER = re.compile(r"^(.+?) (\d+)$")
_KJV_CHAPTER_LINE = re.compile(r"^Chapter (\d+)$")
_KJV_VERSE_LINE = re.compile(r"^(\d{1,3})(?![\d,:])(.*)$")  # "21It devoureth" or a bare "5"


def _kjv_footer(lines: list[str]) -> tuple[str, list[str]]:
    """Split off the running footer. pdfium yields ``[title,] page-number, title``."""
    if not lines:
        return "", lines
    title = lines[-1]
    body = lines[:-1]
    while body and (body[-1] == title or body[-1].isdigit()):
        body.pop()
    return title, body


def parse_kjv(pages: list[str]) -> Iterator[Unit]:
    section: str | None = None  # None until the "Old Testament" divider page
    book: str | None = None
    current_book: str | None = None
    chapter = 0
    verse = 0
    current: Unit | None = None

    for page_idx, raw in enumerate(pages):
        # pdfium keeps hyphenated words on one line, marked with U+FFFE: "be\ufffecame".
        footer, body = _kjv_footer(_lines(normalize(raw).replace(PDFIUM_SOFT_HYPHEN, "")))
        if footer in KJV_SECTION_MARKERS:
            section = KJV_SECTION_MARKERS[footer]
            continue
        if footer == "Indexes":
            break
        if section is None or not footer:
            continue  # front matter, table of contents, translators' preface

        m = _KJV_FOOTER_CHAPTER.match(footer)
        if not m:
            book = footer  # book title page ("The First Book of Moses, called Genesis")
            continue
        if m.group(1) != "Chapter":  # Susanna / Bel use "Chapter N" as footer
            book = m.group(1)
        if book is None:
            continue

        page_chapter = int(m.group(2))
        if (book, page_chapter) != (current_book, chapter):  # e.g. Obadiah 1 -> Jonah 1
            if current:
                yield current
                current = None
            current_book, chapter, verse = book, page_chapter, 0

        for line in body:
            if _KJV_CHAPTER_LINE.match(line) or line == book:
                continue
            vm = _KJV_VERSE_LINE.match(line)
            if vm and verse < int(vm.group(1)) <= verse + 3:  # tolerate a missed number
                if current:
                    yield current
                verse = int(vm.group(1))
                current = Unit(
                    book, chapter, verse, verse, vm.group(2).strip(), page_idx + 1, section
                )
            elif current is None and verse == 0:
                verse = 1  # first verse of a chapter has no number
                current = Unit(book, chapter, 1, 1, line, page_idx + 1, section)
            elif current is not None:
                current.text = f"{current.text} {line}" if current.text else line
            else:
                log.debug("KJV: dropped line on page %s: %r", page_idx + 1, line)
    if current:
        yield current


# --------------------------------------------------------------------------------------
# RVR1960
# --------------------------------------------------------------------------------------

_RVR_TOC_LINE = re.compile(r"^(?P<name>[^.\d][^.]*?)\s*(?:\.\s*)+(?P<page>\d+)$")
_RANGE = r"(?P<c1>\d+)\.\s*(?P<v1>\d+)(?:–(?:(?P<c2>\d+)\.\s*)?(?P<v2>\d+))?"
_RVR_HEADERS = [
    re.compile(rf"^{_RANGE}\s+(?P<book>\D+?)\s+\d+$"),  # "4. 15–5. 9 Génesis 6"
    re.compile(rf"^\d+\s+(?P<book>\D+?)\s+{_RANGE}$"),  # "3 Génesis 2. 6–25"
    re.compile(r"^\d+\s+(?P<book>\D+)$"),  # "1557 Romanos" (last, short page)
    re.compile(r"^(?P<book>\D+?)\s+\d+$"),  # "Colosenses 1622"
]
_MARKERS = r"\d{1,3}(?:,\s?\d{1,3})*"
_LEFT_MARKERS = re.compile(rf"^({_MARKERS})\s+(?=\S)")
_RIGHT_MARKERS = re.compile(rf"(?<=\S)\s?({_MARKERS})$")
_ROMAN_PREFIX = {"I": "1", "II": "2", "III": "3"}
# Printed page 1 (Génesis) is 0-based PDF page index 4 (1-based page 5).
RVR_PAGE_OFFSET = 3


def _rvr_book_name(toc_name: str) -> str:
    name = toc_name.strip()
    if name.isupper():
        name = name.capitalize()  # "GÉNESIS" -> "Génesis"
    first, _, rest = name.partition(" ")
    if first in _ROMAN_PREFIX and rest:
        name = f"{_ROMAN_PREFIX[first]} {rest}"
    return name.removeprefix("El ")  # "El Apocalipsis" -> "Apocalipsis"


def parse_rvr_toc(pages: list[str], expected_books: int | None = 66) -> list[tuple[int, str, str]]:
    """Return ``[(first_page_index, book, section)]`` from the printed table of contents."""
    books: list[tuple[int, str, str]] = []
    section = "OT"
    for raw in pages[:4]:
        for line in _lines(normalize(raw)):
            if "Nuevo Testamento" in line:
                section = "NT"
            m = _RVR_TOC_LINE.match(line)
            if m:
                page_index = int(m.group("page")) + RVR_PAGE_OFFSET
                books.append((page_index, _rvr_book_name(m.group("name")), section))
    books.sort()
    if expected_books is not None and len(books) != expected_books:
        raise ValueError(
            f"RVR1960 table of contents: expected {expected_books} books, parsed {len(books)}"
        )
    return books


def _split_markers(line: str) -> tuple[list[int], str]:
    markers: list[int] = []
    m = _LEFT_MARKERS.match(line)
    if m:
        markers += [int(x) for x in m.group(1).split(",")]
        line = line[m.end() :]
    m = _RIGHT_MARKERS.search(line)
    if m:
        markers += [int(x) for x in m.group(1).split(",")]
        line = line[: m.start()].rstrip()
    return markers, line


def _strip_opening(lines: list[str]) -> list[str]:
    """Drop the book title of an opening page and re-attach the drop cap.

    Layouts seen in the PDF (title lines are small caps, i.e. lowercase text)::

        libro primero de moisés / génesis / E / n el principio...    -> En el principio...
        deuteronomio / ́ / E / 1 stas son las palabras...            -> 1 Éstas son...
        joel / P / 1, 2 alabra de Jehová...                           -> 1, 2 Palabra...
        Rut / A / conteció en los días...                             -> Aconteció en...
        lamentaciones de / jeremias / ¡Cómo ha quedado sola...        (no drop cap)

    A margin verse number between the drop cap and the word is moved in front of the
    rebuilt word, where :func:`_split_markers` picks it up.
    """
    for j in range(min(len(lines) - 1, 8)):  # the drop cap: everything before it is title
        if not re.fullmatch(r"[A-ZÁÉÍÓÚÑ]", lines[j]):
            continue
        m = re.match(rf"({_MARKERS}\s+)?([a-záéíóúüñ].*)$", lines[j + 1])
        if m:
            cap = lines[j]
            if j > 0 and lines[j - 1] in ("´", "\u0301", "'"):  # "´" + "E" -> "É"
                cap = unicodedata.normalize("NFC", cap + "\u0301")
            return [f"{m.group(1) or ''}{cap}{m.group(2)}", *lines[j + 2 :]]
    i = 0  # no drop cap (Lamentaciones): drop the small-caps title lines
    while i < len(lines) and re.fullmatch(r"[a-záéíóúüñ ]+", lines[i]):
        i += 1
    return lines[i:]


def parse_rvr(pages: list[str], expected_books: int | None = 66) -> Iterator[Unit]:
    toc = parse_rvr_toc(pages, expected_books)
    starts = {page: (book, section) for page, book, section in toc}

    # Pass 1: per page -> (page_idx, opening book/section or None, header, [(markers, text)])
    layout = []
    for page_idx in range(toc[0][0], len(pages)):
        text = normalize(pages[page_idx])
        # A discretionary hyphen joins two printed lines (and may hide the next line's
        # margin number: "en\ufffe12 tre"); split them back and remember to re-join the word.
        text = text.replace(PDFIUM_SOFT_HYPHEN, SOFT_HYPHEN + "\n")
        lines = _lines(text)
        header = None
        if page_idx in starts:
            lines = _strip_opening(lines)
        else:
            header = next(
                (m for rx in _RVR_HEADERS if (m := rx.match(lines[0] if lines else ""))), None
            )
            if header is None:
                continue  # e.g. the "EL NUEVO TESTAMENTO" title page
            lines = lines[1:]
        parsed = [_split_markers(ln) for ln in lines]
        layout.append((page_idx, starts.get(page_idx), header, parsed))

    # Flat marker stream so the chapter/verse disambiguation can look across page breaks.
    stream = [m for *_, parsed in layout for markers, _ in parsed for m in markers]
    pos = 0
    book = section = None
    chapter = verse = 0

    # Pass 2: walk the margin numbers. A number is the next verse (verse + 1) or, when it
    # equals chapter + 1 and is followed by a "2", the start of a new chapter.
    for page_idx, opening, header, parsed in layout:
        if opening:
            (book, section), chapter, verse = opening, 1, 0
        elif header.groupdict().get("c1"):
            # Header = first verse that *starts* on this page: resync if we drifted.
            c1, v1 = int(header["c1"]), int(header["v1"])
            if (chapter, verse + 1) != (c1, v1) and v1 > 1:
                log.debug(
                    "RVR resync p%s: state %s:%s header %s:%s", page_idx + 1, chapter, verse, c1, v1
                )
                chapter, verse = c1, v1 - 1

        for markers, body in parsed:
            start_v = max(verse, 1)
            new_chapter = False
            for m in markers:
                nxt = stream[pos + 1] if pos + 1 < len(stream) else None
                pos += 1
                is_verse = m == verse + 1
                if m == chapter + 1 and (nxt == 2 or not is_verse):
                    chapter, verse, new_chapter = m, 1, True
                elif is_verse or verse < m <= verse + 3:  # tolerate a missed marker
                    verse = m
            if new_chapter:  # chapters always start at the beginning of a printed line
                start_v = 1
            join = body.endswith(SOFT_HYPHEN)
            body = body.rstrip(SOFT_HYPHEN)
            if not body:
                continue
            yield Unit(
                book,
                chapter,
                start_v,
                max(verse, start_v),
                body,
                page_idx + 1,
                section,
                join_next=join or body.endswith("-"),
            )


def resolve_line_hyphens(units: list[Unit]) -> list[Unit]:
    """Decide, for every line ending in a hard "-", whether it is a hyphenation or a compound.

    pdfium reports most TeX hyphenations as U+FFFE (handled in :func:`parse_rvr`) but some
    come through as a plain "-" (``Jeho-`` / ``vá``). Proper names are genuinely hyphenated
    with a lowercase second part (``Bet-el``, ``Tubal-caín``), so the corpus decides: keep
    the hyphen only if the hyphenated form is more frequent elsewhere than the joined word.
    """
    words: Counter[str] = Counter()
    compounds: Counter[str] = Counter()
    for u in units:
        words.update(w.lower() for w in re.findall(r"\w+", u.text))
        compounds.update(c.lower() for c in re.findall(r"\w+-\w+", u.text))
    for unit, nxt in zip(units, units[1:]):  # noqa: B905 - pairs of consecutive units
        left = re.search(r"(\w+)-$", unit.text)
        right = re.match(r"\w+", nxt.text)
        if not (left and right):
            continue
        joined, hyphenated = left.group(1) + right.group(0), f"{left.group(1)}-{right.group(0)}"
        if compounds[hyphenated.lower()] > words[joined.lower()]:
            continue  # a real compound such as "Bet-el"
        unit.text = unit.text[:-1]
        unit.join_next = True
    return units


# --------------------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------------------


def format_reference(book: str, chapter: int, v1: int, v2: int) -> str:
    return f"{book} {chapter}:{v1}" if v1 == v2 else f"{book} {chapter}:{v1}-{v2}"


def _make_passage(units: list[Unit], version: VersionSpec) -> Passage:
    first = units[0]
    return Passage(
        version=version.label,
        language=version.language,
        book=first.book,
        chapter=first.chapter,
        verse_start=min(u.verse_start for u in units),
        verse_end=max(u.verse_end for u in units),
        text=join_units_text(units),
        page=first.page,
        section=first.section,
    )


def chunk_units(
    units: Iterable[Unit], version: VersionSpec, chunk_size: int, chunk_overlap: int
) -> Iterator[Passage]:
    """Greedily pack consecutive units of the same chapter into ~chunk_size passages."""
    buf: list[Unit] = []
    size = 0
    for unit in units:
        same_chapter = bool(buf) and (buf[0].book, buf[0].chapter) == (unit.book, unit.chapter)
        if buf and (not same_chapter or size + len(unit.text) > chunk_size):
            yield _make_passage(buf, version)
            if same_chapter and chunk_overlap > 0:  # carry the tail over for context
                tail: list[Unit] = []
                tail_size = 0
                for u in reversed(buf):
                    if tail_size + len(u.text) > chunk_overlap:
                        break
                    tail.insert(0, u)
                    tail_size += len(u.text) + 1
                buf, size = tail, tail_size
            else:
                buf, size = [], 0
        buf.append(unit)
        size += len(unit.text) + 1
    if buf:
        yield _make_passage(buf, version)


PARSERS: dict[str, Callable[[list[str]], Iterator[Unit]]] = {
    "kjv": parse_kjv,
    "rvr1960": parse_rvr,
}


def load_version(
    version_id: str,
    data_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    include_apocrypha: bool = False,
) -> list[Passage]:
    version = VERSIONS[version_id]
    pdf_path = data_dir / version.filename
    if not pdf_path.exists():
        raise FileNotFoundError(f"{version.name}: PDF not found at {pdf_path}")
    pages = extract_pages(pdf_path)
    units = resolve_line_hyphens(list(PARSERS[version_id](pages)))
    if not include_apocrypha:
        units = [u for u in units if u.section != "Apocrypha"]
    return list(chunk_units(units, version, chunk_size, chunk_overlap))
