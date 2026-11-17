"""Parallel passages across translations, their key clauses and how their wording differs.

A `Row` is one place in the Bible (book, chapter, verses) with its passage in each loaded
version. Passages are cut independently per version (KJV Matthew 2:1-3, RV1909 Mateo 2:1-2),
so rows group passages whose verse ranges overlap.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

import numpy as np

from .corpus import Passage
from .grounding import book_index
from .index import Hit
from .lexicon import Topic, compile_terms, fold


@dataclass
class Row:
    book: int  # canonical book index (corpus.BOOKS)
    chapter: int
    hits: dict[str, Hit] = field(default_factory=dict)  # version id -> passage found there
    key: dict[str, str] = field(default_factory=dict)  # version id -> clause that matters
    score: float = 0.0  # ordering within its section
    stance: float = 0.0  # contradiction search: > 0 first claim, < 0 second claim
    fit: tuple[float, float] = (0.0, 0.0)  # closeness to claim A and to claim B
    verses: tuple[int, int] | None = None  # the verses its key clauses are in
    features: dict = field(default_factory=dict)  # what the stance was computed from
    notes: list[str] = field(default_factory=list)

    @property
    def title(self) -> str:
        """'Matthew 2:1-3 · Mateo 2:1-2' (each distinct reference once)."""
        refs = dict.fromkeys(h.passage.reference for h in self.hits.values())
        return " · ".join(refs)

    def overlaps(self, p: Passage) -> bool:
        return any(
            p.verse_start <= h.passage.verse_end and p.verse_end >= h.passage.verse_start
            for h in self.hits.values()
        )

    @property
    def first(self) -> Hit:
        return next(iter(self.hits.values()))


def group(hits: list[tuple[str, Hit]], rows: list[Row] | None = None) -> list[Row]:
    """Rows from (version id, hit) pairs given best first: a hit joins the first row of its
    chapter it overlaps; if that row already has its version (overlapping chunks), it's a
    duplicate and is dropped."""
    rows = rows if rows is not None else []
    for version, hit in hits:
        p = hit.passage
        book = book_index(p.book)
        row = next(
            (r for r in rows if (r.book, r.chapter) == (book, p.chapter) and r.overlaps(p)), None
        )
        if row is None:
            rows.append(Row(book, p.chapter, {version: hit}))
        elif version not in row.hits:
            row.hits[version] = hit
    return rows


def fill_parallels(
    rows: list[Row],
    indexes: dict,
    versions: list[str],
    vector=None,
    embedder=None,
    plain=frozenset(),
) -> None:
    """Add the parallel passage of every missing version, for the row's key verses (else the
    first passage's). Chunks differ per version, so several may overlap: the one closest to
    the row's key clause is taken (with `embedder` and keys chosen), else to `vector`; for
    `plain` versions (the embedder doesn't know their language) the one sharing most verses."""
    aims = {}
    if embedder is not None:
        keyed = [r for r in rows if r.key and len(r.hits) < len(versions)]
        if keyed:
            vs = embedder.embed([next(iter(r.key.values())) for r in keyed])
            aims = {id(r): v for r, v in zip(keyed, vs, strict=True)}
    for row in rows:
        anchor = row.first.passage
        first, last = row.verses or (anchor.verse_start, anchor.verse_end)
        for v in versions:
            if v in row.hits:
                continue
            aim = None if v in plain else aims.get(id(row), vector)
            i = indexes[v].at(row.book, row.chapter, first, last, aim)
            if i is not None:
                known = vector is not None and v not in plain
                score = float(indexes[v].vectors[i] @ vector) if known else 0.0
                row.hits[v] = Hit(indexes[v].passages[i], score, 0.0)
        row.hits = {v: row.hits[v] for v in versions if v in row.hits}  # stable version order


def clauses(text: str, short: int = 60) -> list[str]:
    """Sentences / clauses (KJV style: split at . ; : ? !), short ones joined to the next."""
    parts = [c.strip() for c in re.split(r"(?<=[.;:?!])\s+", text) if c.strip()]
    out: list[str] = []
    for part in parts:
        if out and len(out[-1]) < short:
            out[-1] = f"{out[-1]} {part}"
        else:
            out.append(part)
    return out


def key_verses(row: Row, version: str) -> tuple[int, int]:
    """The verses a version's key clause is in (its whole passage if not found)."""
    p = row.hits[version].passage
    key = row.key.get(version, "")
    start = p.text.find(key) if key else -1
    if start < 0:
        return p.verse_start, p.verse_end
    return p.verses_in(start, start + len(key))


def in_verses(p: Passage, clause: str, verses: tuple[int, int] | None) -> bool:
    """Is the clause within the verses (True when that can't be told)?"""
    if verses is None or not p.starts:
        return True
    start = p.text.find(clause)
    if start < 0:
        return True
    first, last = p.verses_in(start, start + len(clause))
    return first <= verses[1] and last >= verses[0]


def excerpt(clause: str, limit: int = 220) -> str:
    """Trim at a word boundary; the '…' stays outside what is quoted."""
    if len(clause) <= limit:
        return clause
    return clause[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "…"


def choose_keys(
    rows: list[Row],
    embedder,
    target: np.ndarray | None,
    terms=None,
    language_of=None,
    exclude=frozenset(),
) -> None:
    """Give every passage of a row its key clause. The anchor is a key already chosen, or
    else the clause of the row's first version closest to `target` (among clauses naming
    the topic, when `terms(version, clause)` says so); every other version gets its clause
    closest to the anchor, or pair of adjacent clauses (embeddings are cross-lingual; a
    version in the anchor's language is matched by its words, with `language_of`), among its
    clauses in the row's key verses when known. `exclude`: versions left alone (originals:
    see original_keys). Sets the row's key verses."""
    jobs = []  # (row, version, candidate clauses)
    for row in rows:
        versions = [v for v in row.hits if v not in exclude]
        if not versions:
            continue
        anchor = next((v for v in versions if v in row.key), None)
        order = sorted(versions, key=lambda v: v != (anchor or versions[0]))
        for v in order:
            if v in row.key and v != anchor:
                continue
            p = row.hits[v].passage
            cs = [row.key[v]] if v == anchor else clauses(p.text)
            if v != anchor and row.verses:  # a translation's clauses in the same verses
                cs = [c for c in cs if in_verses(p, c, row.verses)] or cs
            if v != anchor:  # a sentence may span two clauses, and translations punctuate
                # differently: adjacent pairs are candidates too
                cs = cs + [f"{a} {b}" for a, b in zip(cs, cs[1:], strict=False)]
            if cs:
                jobs.append((row, v, cs))
    texts = [c for _, _, cs in jobs for c in cs]
    if not texts:
        return
    vectors, start = embedder.embed(texts), 0
    anchors: dict[int, tuple[str, str, np.ndarray]] = {}  # row -> (version, clause, vector)
    for row, v, cs in jobs:
        vs = vectors[start : start + len(cs)]
        start += len(cs)
        if id(row) not in anchors:  # first job of a row: its anchor
            candidates = list(range(len(cs)))
            if terms is not None and (on := [i for i, c in enumerate(cs) if terms(v, c)]):
                candidates = on
            aim = target if target is not None else vs.mean(axis=0)
            best = max(candidates, key=lambda i: float(vs[i] @ aim))
            anchors[id(row)] = (v, cs[best], vs[best])
            row.key[v] = cs[best]
            row.verses = row.verses or key_verses(row, v)
            continue
        v0, clause0, vector0 = anchors[id(row)]
        if language_of is not None and language_of(v) == language_of(v0):
            words0 = [_norm(w) for w in clause0.split()]
            best = max(
                range(len(cs)),
                key=lambda i: difflib.SequenceMatcher(
                    a=words0, b=[_norm(w) for w in cs[i].split()]
                ).ratio(),
            )
        else:
            best = int(np.argmax(vs @ vector0))
        row.key[v] = cs[best]


def original_keys(rows: list[Row], originals) -> None:
    """The key of an original-language passage: exactly the row's key verses (the words can't
    be matched to a translation's clause, the verses can)."""
    for row in rows:
        for v in originals:
            if v in row.hits and row.verses:
                row.key[v] = row.hits[v].passage.verse_text(*row.verses)


# ------------------------------------------------------------------- wording differences
def concept_words(
    topic: Topic, language: str, text: str, ambiguous: bool = True
) -> dict[str, list[str]]:
    """{concept gloss: [words as written]} for the topic's concepts found in text
    (ambiguous=False: homographs such as "vino?" / "drunk?" don't count)."""
    folded, found = fold(text), {}
    for concept in topic.concepts:
        terms = [t for t in concept.terms(language) if ambiguous or not t.endswith("?")]
        rx = compile_terms(terms)
        if rx:
            words = [text[m.start() : m.end()] for m in rx.finditer(folded)]
            if words:
                found[concept.gloss] = list(dict.fromkeys(words))
    return found


def renderings(row: Row, topic: Topic, language_of, texts=None) -> dict[str, dict]:
    """How each version of a row words the topic: {version: {concept: [words]}}. A homograph
    ("vino" = wine or came, "drunk" = drunken or past of drink) counts only for a concept that
    another version confirms with an unambiguous word."""
    texts = texts or {v: row.key.get(v) or h.passage.text for v, h in row.hits.items()}
    sure = {v: concept_words(topic, language_of(v), t, False) for v, t in texts.items()}
    found = {}
    for v, t in texts.items():
        confirmed = {c for w, f in sure.items() if w != v for c in f}
        every = concept_words(topic, language_of(v), t)
        found[v] = {c: ws for c, ws in every.items() if c in sure[v] or c in confirmed}
    return found


def concept_note(row: Row, topic: Topic, language_of, labels, lang: str) -> str | None:
    """'Wording differs: KJV “hell” · RV1909 “sepulcro” (the grave) · WEB “Sheol”' when the
    versions express the topic with different concepts in the same place."""
    # only versions that name the topic are compared: the lexicon can't list every phrasing,
    # so a version where no term matched proves nothing
    named = {v: f for v, f in renderings(row, topic, language_of).items() if f}
    if len({frozenset(f) for f in named.values()}) < 2:
        return None
    parts = []
    for v, f in named.items():
        words = ", ".join(f"“{w}”" for ws in f.values() for w in ws[:2])
        parts.append(f"{labels(v)} {words} ({', '.join(f)})")
    return f"{_DIFFERS[lang]}: " + " · ".join(parts)


# archaic KJV forms and their modern equivalents: a change between them is style, not sense
_MODERN = {
    "thou": "you",
    "thee": "you",
    "ye": "you",
    "thy": "your",
    "thine": "your",
    "unto": "to",
    "hath": "has",
    "hast": "have",
    "doth": "does",
    "dost": "do",
    "shalt": "shall",
    "wilt": "will",
    "art": "are",
    "saith": "says",
    "spake": "spoke",
    "shew": "show",
    "shewed": "showed",
    "yea": "yes",
    "nay": "no",
    "wherefore": "why",
    "whither": "where",
    "thence": "there",
    "hither": "here",
    "wast": "were",
    "didst": "did",
    "canst": "can",
    "shouldest": "should",
}


_FUNCTION = frozenset(
    "a an and are as at be but by for from had has have he her him his i if in into is it its "
    "me my not of on or our shall she so that the their them then there they this to unto us "
    "was we were when which who will with ye you your thee thou thy also even now".split()
)


def _norm(word: str) -> str:
    return re.sub(r"\W", "", fold(word))


def _modern(words: list[str]) -> str:
    out = []
    for w in words:
        w = _MODERN.get(w, w)
        w = re.sub(r"(?<=\w{3})(eth|est)$", "", w)  # loveth, lovest -> lov
        out.append(re.sub(r"(e|es|s|ed)$", "", w))  # love, loves, loved -> lov
    return " ".join(out)


def word_changes(a: str, b: str, limit: int = 6) -> tuple[list[tuple[str, str]], int] | None:
    """Phrases that changed from a to b (same-language versions). Returns (changes in wording,
    number of other changes: archaic-vs-modern style, spelling, long additions such as a verse
    one passage has and the other lacks), or None when the texts share too little to compare.
    """
    wa, wb = a.split(), b.split()
    na, nb = [_norm(w) for w in wa], [_norm(w) for w in wb]
    sm = difflib.SequenceMatcher(a=na, b=nb, autojunk=False)
    shared = sum(block.size for block in sm.get_matching_blocks())
    if not wa or not wb or shared < 0.3 * min(len(wa), len(wb)):
        return None
    changes, minor = [], 0
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        x, y = na[i1:i2], nb[j1:j2]
        style = _modern(x) == _modern(y) or (
            x and y and difflib.SequenceMatcher(a=" ".join(x), b=" ".join(y)).ratio() > 0.85
        )  # "thou wilt" -> "you will", "fulness" -> "fullness"
        if op == "replace":
            shown = not style and max(len(x), len(y)) <= 6
        else:  # a few words added or dropped ("not easily provoked" -> "not provoked")
            shown = len(x + y) <= 3 and any(w not in _FUNCTION for w in x + y)
        pair = tuple(" ".join(ws).strip(_PUNCT) for ws in (wa[i1:i2], wb[j1:j2]))
        if shown and pair not in changes:
            changes.append(pair)
        elif not shown:
            minor += 1
    return changes[:limit], minor + max(len(changes) - limit, 0)


_PUNCT = ",;:.!?“”‘’\"'()"
_DIFFERS = {"en": "Wording differs", "es": "La redacción difiere"}
