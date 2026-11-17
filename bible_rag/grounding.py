"""Check an answer against its retrieved passages: every reference ("John 3:16", "Juan 3:16",
"1 Cor. 13:4") must point into one, every quote must appear in one (ignoring case/accents)."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from .corpus import BOOKS


def _fold(text: str) -> str:
    """Lowercase, accents removed, same length (so match offsets stay valid)."""
    return "".join(unicodedata.normalize("NFD", c)[0] for c in text).lower()


def _words(text: str) -> str:
    return " ".join(re.findall(r"\w+", _fold(text)))


_NAMES = {_fold(name): i for i, (_, en, es) in enumerate(BOOKS) for name in (en, es)}
_NAMES |= {"psalm": 18, "salmo": 18, "song of songs": 21, "cantares": 21, "revelations": 65}
_BOOK = r"(?P<book>(?:[123]\s?)?[a-z][a-z]+\.?(?:\s(?:of|de|los)\s[a-z]+|\s[a-z]+)?)"
# "John 3:16", "Juan 3:16-18", and "Juan 3,16" (the comma is usual in Spanish and German)
REFERENCE = re.compile(
    _BOOK + r"\s+(?P<ch>\d{1,3})(?::\s?|,)(?P<v1>\d{1,3})(?:\s*[-–]\s*(?P<v2>\d{1,3}))?"
)
CHAPTER = re.compile(_BOOK + r"\s+(?P<ch>\d{1,3})\b(?![:,.]\d)")  # "Psalm 23", "Juan 3"
_QUOTE = re.compile(r'"([^"]*)"|“([^”]*)”|«([^»]*)»')  # paired left to right, any length


def resolve_book(name: str) -> tuple[int, int] | None:
    """(book index, leading words to drop) for a captured name: "as John" -> John,
    "1 Cor." -> 1 Corinthians (a unique prefix), any language."""
    words = _fold(name).replace(".", "").split()
    suffixes = [(start, " ".join(words[start:])) for start in range(len(words))]
    for start, cand in suffixes:
        if cand in _NAMES:
            return _NAMES[cand], start
    for start, cand in suffixes:
        hits = {i for n, i in _NAMES.items() if n.startswith(cand)}
        if len(cand.replace(" ", "")) >= 3 and len(hits) == 1:
            return hits.pop(), start
    return None


def book_index(name: str) -> int | None:
    """Canonical index (corpus.BOOKS) of a book name in English or Spanish."""
    return r[0] if (r := resolve_book(name)) else None


def references(text: str) -> list[tuple[int, int, int, int]]:
    """(book, chapter, first verse, last verse) of each reference in a text, any language."""
    refs = []
    for m in REFERENCE.finditer(_fold(text)):
        if resolved := resolve_book(m["book"]):
            v1 = int(m["v1"])
            refs.append((resolved[0], int(m["ch"]), v1, int(m["v2"] or v1)))
    return list(dict.fromkeys(refs))


def chapters(text: str) -> list[tuple[int, int]]:
    """(book, chapter) of each whole chapter named in a text ("Psalm 23", "Juan 3")."""
    found = []
    for m in CHAPTER.finditer(_fold(text)):
        if resolved := resolve_book(m["book"]):
            found.append((resolved[0], int(m["ch"])))
    return list(dict.fromkeys(found))


def without_references(text: str) -> str:
    """The text with its references ("John 3:16", "Psalm 23") blanked out."""
    folded, spans = _fold(text), []
    for pattern in (REFERENCE, CHAPTER):
        for m in pattern.finditer(folded):
            if resolved := resolve_book(m["book"]):
                dropped = m["book"].split(" ")[: resolved[1]]  # "as John": "as" is kept
                start = m.start("book") + sum(len(w) + 1 for w in dropped)
                spans.append((start, m.end()))
    for start, end in sorted(spans, reverse=True):
        text = text[:start] + " " + text[end:]
    return text


@dataclass
class Check:
    text: str
    supported: bool
    location: tuple[int, int] | None = None  # (book index, chapter) of the supporting passage


@dataclass
class Grounding:
    references: list[Check] = field(default_factory=list)
    quotes: list[Check] = field(default_factory=list)

    @property
    def unsupported(self) -> list[Check]:
        return [c for c in self.references + self.quotes if not c.supported]

    @property
    def grounded(self) -> bool:  # something verified, nothing unverified
        return bool(self.references or self.quotes) and not self.unsupported


def check_answer(answer: str, hits) -> Grounding:
    sources = [(book_index(h.passage.book), h.passage) for h in hits]
    report, seen = Grounding(), set()
    for m in REFERENCE.finditer(_fold(answer)):
        if not (resolved := resolve_book(m["book"])):
            continue
        book, drop = resolved
        ch, v1 = int(m["ch"]), int(m["v1"])
        v2 = int(m["v2"] or v1)
        if (book, ch, v1, v2) in seen:
            continue
        seen.add((book, ch, v1, v2))
        text = answer[m.start("book") : m.end()].strip().split(" ", drop)[-1]
        ok = any(
            b == book and p.chapter == ch and v1 <= p.verse_end and v2 >= p.verse_start
            for b, p in sources
        )
        report.references.append(Check(text, ok, (book, ch) if ok else None))
    texts = [(_words(p.text), b, p.chapter) for b, p in sources]
    for m in _QUOTE.finditer(answer):
        quoted = next(g for g in m.groups() if g is not None).strip()
        if len(quoted) < 15 or REFERENCE.fullmatch(_fold(quoted).strip(" .,;()")):
            continue  # too short, or a quoted reference (checked above)
        parts = [w for part in re.split(r"\.\.\.|…", quoted) if len(w := _words(part)) >= 10]
        found = [next(((b, c) for t, b, c in texts if part in t), None) for part in parts]
        if parts:
            ok = all(found)
            report.quotes.append(Check(quoted, ok, found[0] if ok else None))
    return report
