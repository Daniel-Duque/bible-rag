"""Parse the two Bible PDFs into verse-referenced passages.

KJV (XEP): one chapter per page, footer "<page> / <Book chapter>", each verse starts a line
with its number glued on ("21It devoureth") -> exact verse references.
RVR1960 (pdfTeX): header with the page's verse range ("4. 15–5. 9 Génesis 6"), verse numbers
in the margin mark the *line* where a verse starts -> line-accurate references.
Both parsers yield Units (a verse / a printed line) that chunk_units() packs into Passages.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

SOFT_HYPHEN = "\u00ad"  # our marker: word continues on the next line
PDFIUM_SOFT_HYPHEN = "\ufffe"  # how pdfium reports TeX discretionary hyphens


class Version(NamedTuple):
    label: str  # shown in citations
    name: str
    language: str
    filename: str


VERSIONS = {
    "kjv": Version("KJV", "King James Version", "en", "kjv.pdf"),
    "rvr1960": Version("RVR1960", "Reina-Valera 1960", "es", "biblia-reina-valera-1960.pdf"),
}

# Canonical order, names as the parsers produce them (KJV, RVR1960).
# fmt: off
BOOKS = [
    ("Genesis", "Génesis"), ("Exodus", "Éxodo"), ("Leviticus", "Levítico"), ("Numbers", "Números"),
    ("Deuteronomy", "Deuteronomio"), ("Joshua", "Josué"), ("Judges", "Jueces"), ("Ruth", "Rut"),
    ("1 Samuel", "1 Samuel"), ("2 Samuel", "2 Samuel"), ("1 Kings", "1 Reyes"),
    ("2 Kings", "2 Reyes"), ("1 Chronicles", "1 Crónicas"), ("2 Chronicles", "2 Crónicas"),
    ("Ezra", "Esdras"), ("Nehemiah", "Nehemías"), ("Esther", "Ester"), ("Job", "Job"),
    ("Psalm", "Salmos"), ("Proverbs", "Proverbios"), ("Ecclesiastes", "Eclesiastés"),
    ("Song of Solomon", "Cantar de los Cantares"), ("Isaiah", "Isaías"), ("Jeremiah", "Jeremías"),
    ("Lamentations", "Lamentaciones"), ("Ezekiel", "Ezequiel"), ("Daniel", "Daniel"),
    ("Hosea", "Oseas"), ("Joel", "Joel"), ("Amos", "Amós"), ("Obadiah", "Abdías"),
    ("Jonah", "Jonás"), ("Micah", "Miqueas"), ("Nahum", "Nahum"), ("Habakkuk", "Habacuc"),
    ("Zephaniah", "Sofonías"), ("Haggai", "Hageo"), ("Zechariah", "Zacarías"),
    ("Malachi", "Malaquías"), ("Matthew", "Mateo"), ("Mark", "Marcos"), ("Luke", "Lucas"),
    ("John", "Juan"), ("Acts", "Hechos"), ("Romans", "Romanos"), ("1 Corinthians", "1 Corintios"),
    ("2 Corinthians", "2 Corintios"), ("Galatians", "Gálatas"), ("Ephesians", "Efesios"),
    ("Philippians", "Filipenses"), ("Colossians", "Colosenses"),
    ("1 Thessalonians", "1 Tesalonicenses"), ("2 Thessalonians", "2 Tesalonicenses"),
    ("1 Timothy", "1 Timoteo"), ("2 Timothy", "2 Timoteo"), ("Titus", "Tito"),
    ("Philemon", "Filemón"), ("Hebrews", "Hebreos"), ("James", "Santiago"), ("1 Peter", "1 Pedro"),
    ("2 Peter", "2 Pedro"), ("1 John", "1 Juan"), ("2 John", "2 Juan"), ("3 John", "3 Juan"),
    ("Jude", "Judas"), ("Revelation", "Apocalipsis"),
]
# fmt: on


@dataclass
class Passage:
    """A retrievable chunk: consecutive verses of one chapter in one version."""

    version: str
    language: str
    book: str
    chapter: int
    verse_start: int
    verse_end: int
    text: str
    page: int  # 1-based PDF page
    section: str  # OT | NT | Apocrypha

    @property
    def reference(self) -> str:
        v = f"{self.verse_start}" + (
            f"-{self.verse_end}" if self.verse_end != self.verse_start else ""
        )
        return f"{self.book} {self.chapter}:{v}"

    @property
    def citation(self) -> str:
        return f"{self.reference} ({self.version})"

    def embedding_text(self) -> str:
        return f"{self.reference}: {self.text}"  # lets "Juan 3" style queries match too


@dataclass
class Unit:
    """A verse (KJV) or printed line (RVR1960) with its position."""

    book: str
    chapter: int
    verse_start: int
    verse_end: int
    text: str
    page: int
    section: str
    join_next: bool = False  # ends mid-word: join the next unit without a space


def extract_pages(pdf: Path) -> list[str]:
    import pypdfium2 as pdfium  # ~10x faster than pypdf on these files

    doc = pdfium.PdfDocument(str(pdf))
    try:
        return [p.get_textpage().get_text_range().replace("\r\n", "\n") for p in doc]
    finally:
        doc.close()


def normalize(text: str) -> str:
    """NFKC folds the PDFs' ligatures (ĳ -> ij, ﬁ -> fi)."""
    return unicodedata.normalize("NFKC", text).replace("\u00a0", " ")


def _lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.split("\n") if ln.strip()]


def join_units_text(units: Iterable[Unit]) -> str:
    out, glue = [], ""
    for u in units:
        out.append(glue + u.text)
        glue = "" if u.join_next else " "
    return re.sub(r"\s+", " ", "".join(out)).strip()


# ----------------------------------------------------------------------------------- KJV
_KJV_SECTIONS = {
    "Old Testament": "OT",
    "Apocryphal or Deuterocanonical Books": "Apocrypha",
    "New Testament": "NT",
}
_KJV_FOOTER = re.compile(r"^(.+?) (\d+)$")
_KJV_VERSE = re.compile(r"^(\d{1,3})(?![\d,:])(.*)$")  # "21It devoureth" or a bare "5"


def _kjv_footer(lines: list[str]) -> tuple[str, list[str]]:
    """Split off the running footer; pdfium yields [title,] page number, title."""
    if not lines:
        return "", lines
    title, body = lines[-1], lines[:-1]
    while body and (body[-1] == title or body[-1].isdigit()):
        body.pop()
    return title, body


def parse_kjv(pages: list[str]) -> Iterator[Unit]:
    section = book = current_book = None
    chapter = verse = 0
    current: Unit | None = None
    for idx, raw in enumerate(pages):
        # pdfium keeps hyphenated words on one line with U+FFFE: "be\ufffecame"
        footer, body = _kjv_footer(_lines(normalize(raw).replace(PDFIUM_SOFT_HYPHEN, "")))
        if footer in _KJV_SECTIONS:
            section = _KJV_SECTIONS[footer]
            continue
        if footer == "Indexes":
            break
        if section is None or not footer:
            continue  # front matter, table of contents, preface
        m = _KJV_FOOTER.match(footer)
        if not m:
            book = footer  # book title page
            continue
        if m[1] != "Chapter":  # Susanna / Bel use "Chapter N" as footer
            book = m[1]
        if book is None:
            continue
        if (book, int(m[2])) != (current_book, chapter):  # new chapter (also Obadiah 1 -> Jonah 1)
            if current:
                yield current
                current = None
            current_book, chapter, verse = book, int(m[2]), 0
        for line in body:
            if line == book or re.match(r"^Chapter \d+$", line):
                continue
            vm = _KJV_VERSE.match(line)
            if vm and verse < int(vm[1]) <= verse + 3:  # tolerate a missed number
                if current:
                    yield current
                verse = int(vm[1])
                current = Unit(book, chapter, verse, verse, vm[2].strip(), idx + 1, section)
            elif current is None and verse == 0:  # verse 1 has no number
                verse = 1
                current = Unit(book, chapter, 1, 1, line, idx + 1, section)
            elif current is not None:
                current.text = f"{current.text} {line}" if current.text else line
    if current:
        yield current


# ------------------------------------------------------------------------------- RVR1960
_TOC_LINE = re.compile(r"^(?P<name>[^.\d][^.]*?)\s*(?:\.\s*)+(?P<page>\d+)$")
_RANGE = r"(?P<c1>\d+)\.\s*(?P<v1>\d+)(?:–(?:(?P<c2>\d+)\.\s*)?(?P<v2>\d+))?"
_HEADERS = [
    re.compile(rf"^{_RANGE}\s+(?P<book>\D+?)\s+\d+$"),  # "4. 15–5. 9 Génesis 6"
    re.compile(rf"^\d+\s+(?P<book>\D+?)\s+{_RANGE}$"),  # "3 Génesis 2. 6–25"
    re.compile(r"^\d+\s+(?P<book>\D+)$"),  # "1557 Romanos"
    re.compile(r"^(?P<book>\D+?)\s+\d+$"),  # "Colosenses 1622"
]
_MARKERS = r"\d{1,3}(?:,\s?\d{1,3})*"
_LEFT = re.compile(rf"^({_MARKERS})\s+(?=\S)")
_RIGHT = re.compile(rf"(?<=\S)\s?({_MARKERS})$")
RVR_PAGE_OFFSET = 3  # printed page 1 (Génesis) = PDF page index 4


def parse_rvr_toc(pages: list[str], expected_books: int | None = 66) -> list[tuple[int, str, str]]:
    """[(first page index, book, OT|NT)] from the printed table of contents."""
    books, section = [], "OT"
    for raw in pages[:4]:
        for line in _lines(normalize(raw)):
            section = "NT" if "Nuevo Testamento" in line else section
            if m := _TOC_LINE.match(line):
                name = m["name"].strip()
                name = name.capitalize() if name.isupper() else name  # GÉNESIS -> Génesis
                first, _, rest = name.partition(" ")
                if first in ("I", "II", "III") and rest:
                    name = f"{len(first)} {rest}"  # II Samuel -> 2 Samuel
                books.append((int(m["page"]) + RVR_PAGE_OFFSET, name.removeprefix("El "), section))
    if expected_books is not None and len(books) != expected_books:
        raise ValueError(f"RVR1960 TOC: expected {expected_books} books, parsed {len(books)}")
    return sorted(books)


def _split_markers(line: str) -> tuple[list[int], str]:
    markers = []
    if m := _LEFT.match(line):
        markers += [int(x) for x in m[1].split(",")]
        line = line[m.end() :]
    if m := _RIGHT.search(line):
        markers += [int(x) for x in m[1].split(",")]
        line = line[: m.start()].rstrip()
    return markers, line


def _strip_opening(lines: list[str]) -> list[str]:
    """Drop a book's title and re-attach its drop cap. Layouts in the PDF:
    génesis / E / n el principio -> En el principio; deuteronomio / ́ / E / 1 stas -> 1 Éstas;
    Rut / A / conteció -> Aconteció; lamentaciones de / jeremias / ¡Cómo... (no drop cap).
    """
    for j in range(min(len(lines) - 1, 8)):
        m = re.match(rf"({_MARKERS}\s+)?([a-záéíóúüñ].*)$", lines[j + 1])
        if re.fullmatch(r"[A-ZÁÉÍÓÚÑ]", lines[j]) and m:
            cap = lines[j]
            if j and lines[j - 1] in ("´", "\u0301", "'"):  # accent printed above the drop cap
                cap = unicodedata.normalize("NFC", cap + "\u0301")
            return [f"{m[1] or ''}{cap}{m[2]}", *lines[j + 2 :]]
    i = 0
    while i < len(lines) and re.fullmatch(r"[a-záéíóúüñ ]+", lines[i]):
        i += 1  # small-caps title lines
    return lines[i:]


def parse_rvr(pages: list[str], expected_books: int | None = 66) -> Iterator[Unit]:
    toc = parse_rvr_toc(pages, expected_books)
    starts = {page: (book, section) for page, book, section in toc}
    layout = []  # pass 1: (page, opening, header, [(markers, text)])
    for idx in range(toc[0][0], len(pages)):
        # a discretionary hyphen merges two lines and can hide the next margin number
        # ("en\ufffe12 tre"): split them again and mark the word to be re-joined
        lines = _lines(normalize(pages[idx]).replace(PDFIUM_SOFT_HYPHEN, SOFT_HYPHEN + "\n"))
        header = None
        if idx in starts:
            lines = _strip_opening(lines)
        else:
            header = next(
                (m for rx in _HEADERS if (m := rx.match(lines[0] if lines else ""))), None
            )
            if header is None:
                continue  # title pages
            lines = lines[1:]
        layout.append((idx, starts.get(idx), header, [_split_markers(ln) for ln in lines]))

    # pass 2: walk the margin numbers. m == verse+1 is the next verse; m == chapter+1 followed
    # by a 2 starts a chapter (the lookahead crosses page breaks, hence the flat stream)
    stream = [m for *_, parsed in layout for markers, _ in parsed for m in markers]
    pos, book, section, chapter, verse = 0, None, None, 0, 0
    for idx, opening, header, parsed in layout:
        if opening:
            (book, section), chapter, verse = opening, 1, 0
        elif header.groupdict().get("c1"):
            c1, v1 = int(header["c1"]), int(header["v1"])  # first verse starting on this page
            if (chapter, verse + 1) != (c1, v1) and v1 > 1:
                chapter, verse = c1, v1 - 1  # resync
        for markers, body in parsed:
            start_v, new_chapter = max(verse, 1), False
            for m in markers:
                nxt = stream[pos + 1] if pos + 1 < len(stream) else None
                pos += 1
                if m == chapter + 1 and (nxt == 2 or m != verse + 1):
                    chapter, verse, new_chapter = m, 1, True
                elif verse < m <= verse + 3:  # next verse, tolerating a missed marker
                    verse = m
            start_v = 1 if new_chapter else start_v  # chapters start at a line start
            join = body.endswith(SOFT_HYPHEN)
            body = body.rstrip(SOFT_HYPHEN)
            if body:
                yield Unit(
                    book,
                    chapter,
                    start_v,
                    max(verse, start_v),
                    body,
                    idx + 1,
                    section,
                    join_next=join or body.endswith("-"),
                )


def resolve_line_hyphens(units: list[Unit]) -> list[Unit]:
    """A line ending in "-" is a hyphenation ("Jeho-/vá") unless the corpus shows a compound
    ("Bet-el", "Tubal-caín") more often than the joined word."""
    words = Counter(w.lower() for u in units for w in re.findall(r"\w+", u.text))
    compounds = Counter(c.lower() for u in units for c in re.findall(r"\w+-\w+", u.text))
    for unit, nxt in zip(units, units[1:]):  # noqa: B905
        left, right = re.search(r"(\w+)-$", unit.text), re.match(r"\w+", nxt.text)
        if left and right:
            joined, compound = left[1] + right[0], f"{left[1]}-{right[0]}"
            if compounds[compound.lower()] <= words[joined.lower()]:
                unit.text, unit.join_next = unit.text[:-1], True
    return units


# ------------------------------------------------------------------------------ chunking
def chunk_units(
    units: Iterable[Unit], version: Version, size: int, overlap: int
) -> Iterator[Passage]:
    """Pack consecutive units of one chapter into ~size-char passages, overlapping ~overlap."""

    def passage(buf: list[Unit]) -> Passage:
        u = buf[0]
        return Passage(
            version.label,
            version.language,
            u.book,
            u.chapter,
            min(x.verse_start for x in buf),
            max(x.verse_end for x in buf),
            join_units_text(buf),
            u.page,
            u.section,
        )

    buf, used = [], 0
    for unit in units:
        same = bool(buf) and (buf[0].book, buf[0].chapter) == (unit.book, unit.chapter)
        if buf and (not same or used + len(unit.text) > size):
            yield passage(buf)
            tail, used = [], 0
            if same:  # carry the last units over for context
                for u in reversed(buf):
                    if used + len(u.text) > overlap:
                        break
                    tail.insert(0, u)
                    used += len(u.text) + 1
            buf = tail
        buf.append(unit)
        used += len(unit.text) + 1
    if buf:
        yield passage(buf)


def load_version(
    version_id: str, data_dir: Path, size: int, overlap: int, include_apocrypha: bool = False
) -> list[Passage]:
    version = VERSIONS[version_id]
    pdf = data_dir / version.filename
    if not pdf.exists():
        raise FileNotFoundError(f"{version.name}: PDF not found at {pdf}")
    parser = parse_kjv if version_id == "kjv" else parse_rvr
    units = resolve_line_hyphens(list(parser(extract_pages(pdf))))
    units = [u for u in units if include_apocrypha or u.section != "Apocrypha"]
    return list(chunk_units(units, version, size, overlap))
