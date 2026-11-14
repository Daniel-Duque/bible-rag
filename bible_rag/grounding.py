"""Check an answer against the passages it was generated from.

Small models (and sometimes large ones) cite verses that were not retrieved or "quote"
text that is not in the Bible. Every answer is checked, whatever model produced it:

* **references** such as ``John 3:16``, ``Juan 3:16-17`` or ``1 Cor. 13:4`` must point
  into a retrieved passage (same book in either language, same chapter, overlapping verses);
* **quotes** (text between quotation marks) must appear verbatim in a retrieved passage,
  ignoring case, accents and punctuation.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from .corpus import BOOKS
from .index import Hit


def _fold(text: str) -> str:
    """Lowercase + strip accents, keeping the length (so match offsets stay valid)."""
    return "".join(unicodedata.normalize("NFD", ch)[0] for ch in text).lower()


def _words(text: str) -> str:
    return " ".join(re.findall(r"\w+", _fold(text)))


_NAMES: dict[str, int] = {}
for _i, _pair in enumerate(BOOKS):
    for _name in _pair:
        _NAMES[_fold(_name)] = _i
_NAMES.update(
    {
        "psalms": 18,
        "salmo": 18,
        "song of songs": 21,
        "cantares": 21,
        "revelations": 65,
        "apocalypse": 65,
    }
)

_REF = re.compile(
    r"(?P<book>(?:[123]\s?)?[a-z][a-z]+\.?(?:\s(?:of|de|los)\s[a-z]+|\s[a-z]+)?)"
    r"\s+(?P<ch>\d{1,3}):(?P<v1>\d{1,3})(?:\s*[-–]\s*(?P<v2>\d{1,3}))?"
)
# Pairs are matched left to right for each quote style *before* filtering by length, so a
# short quote ("word") can't make the text between two quotes look like a quotation.
_QUOTE = re.compile(r'"([^"]*)"|“([^”]*)”|«([^»]*)»')
_MIN_QUOTE = 15


def _resolve(name: str) -> tuple[int, int] | None:
    """(canonical index, number of leading words to drop) for a captured book name."""
    words = re.sub(r"\s+", " ", _fold(name).replace(".", "")).strip().split(" ")
    # the regex may capture a leading word ("as John"): try the longest known suffix first
    for start in range(len(words)):
        candidate = " ".join(words[start:])
        if candidate in _NAMES:
            return _NAMES[candidate], start
    # abbreviations: "1 cor" -> "1 corinthians"/"1 corintios", "gen" -> "genesis"
    for start in range(len(words)):
        candidate = " ".join(words[start:])
        if len(candidate.replace(" ", "")) < 3:
            continue
        matches = {i for n, i in _NAMES.items() if n.startswith(candidate)}
        if len(matches) == 1:
            return matches.pop(), start
    return None


def book_index(name: str) -> int | None:
    """Resolve a (possibly abbreviated, either-language) book name to its canonical index."""
    resolved = _resolve(name)
    return resolved[0] if resolved else None


@dataclass
class Check:
    text: str
    supported: bool
    passage: str | None = None  # citation of the retrieved passage that supports it
    location: tuple[int, int] | None = field(default=None, repr=False)  # (book index, chapter)


@dataclass
class GroundingReport:
    references: list[Check] = field(default_factory=list)
    quotes: list[Check] = field(default_factory=list)

    @property
    def unsupported(self) -> list[Check]:
        return [c for c in self.references + self.quotes if not c.supported]

    @property
    def grounded(self) -> bool:
        """At least one verified reference/quote and nothing unverified."""
        checks = self.references + self.quotes
        return bool(checks) and not self.unsupported

    def to_dict(self) -> dict:
        return {
            "grounded": self.grounded,
            "references": [_check_dict(c) for c in self.references],
            "quotes": [_check_dict(c) for c in self.quotes],
        }


def _check_dict(c: Check) -> dict:
    return {"text": c.text, "supported": c.supported, "passage": c.passage}


def check_answer(answer: str, hits: list[Hit]) -> GroundingReport:
    folded = _fold(answer)
    sources = [(book_index(h.passage.book), h.passage) for h in hits]
    report = GroundingReport()
    seen: set[str] = set()
    for m in _REF.finditer(folded):
        resolved = _resolve(m["book"])
        if resolved is None:
            continue  # "chapter 3:16" etc.: not a book reference
        idx, drop = resolved
        ch, v1 = int(m["ch"]), int(m["v1"])
        v2 = int(m["v2"]) if m["v2"] else v1
        original = answer[m.start("book") : m.end()].strip()
        original = original.split(" ", drop)[drop] if drop else original
        key = f"{idx}:{ch}:{v1}-{v2}"
        if key in seen:
            continue
        seen.add(key)
        match = next(
            (
                p
                for b, p in sources
                if b == idx and p.chapter == ch and v1 <= p.verse_end and v2 >= p.verse_start
            ),
            None,
        )
        report.references.append(_check(original, match, idx))

    corpus = [
        (_words(h.passage.text), b, h.passage) for (b, _), h in zip(sources, hits, strict=True)
    ]
    for m in _QUOTE.finditer(answer):
        quoted = next(g for g in m.groups() if g is not None)
        if len(quoted) < _MIN_QUOTE:
            continue
        if _REF.fullmatch(_fold(quoted).strip(" .,;()")):
            continue  # a quoted reference ("Genesis 7:17-20") is checked as a reference
        parts = [_words(part) for part in re.split(r"\.\.\.|…", quoted)]
        parts = [p for p in parts if len(p) >= 10]
        if not parts:
            continue
        # every fragment must be found; the passage of the first one is reported
        found = [next(((b, p) for text, b, p in corpus if part in text), None) for part in parts]
        ok = all(found)
        b, p = found[0] if ok else (None, None)
        report.quotes.append(_check(quoted.strip(), p, b))
    return report


def _check(text: str, passage, book: int | None) -> Check:
    if passage is None:
        return Check(text, False)
    return Check(text, True, passage.citation, (book, passage.chapter))
