"""Bible versions, their licenses, and where the text comes from.

Public-domain translations are SWORD modules packaged by Debian (main archive = license
reviewed by Debian), downloaded by pinned file + SHA-256 and read with pysword. The hashes
below come from the trixie package index, whose signature was verified with the Debian
archive key. Debian also packages Strong's Hebrew and Greek dictionaries (public domain),
used to name the original words.

The original-language texts and more translations come from Python packages on PyPI,
likewise pinned by file + SHA-256 (as published on PyPI): only their data files are
extracted and parsed; no package code is installed, imported or run.
  wlc        Hebrew Bible (Westminster Leningrad Codex, incl. the Biblical Aramaic of Daniel and
             Ezra), one Strong's number per word; its Hebrew versification is converted to the
             English one with a manual word alignment (both from `bible-alignments`, Clear Bible)
  sblgnt     Greek New Testament (SBLGNT with MorphGNT lemmas, from `py-sblgnt`)
  etheridge  the New Testament translated from the Aramaic (Syriac) Peshitta, 1849
  ylt, leeser literal translations from the Hebrew and Greek (1898; Leeser: Hebrew OT, 1853)
RVR1960 is read from your own PDF (data/biblia-reina-valera-1960.pdf); its license is "personal
use only", so it is refused unless allow_restricted is set.
"""

from __future__ import annotations

import ast
import collections
import csv
import functools
import hashlib
import html
import io
import json
import lzma
import re
import struct
import tarfile
import unicodedata
import zipfile
import zlib
from pathlib import Path
from typing import NamedTuple

from .corpus import BOOKS, Unit


class Version(NamedTuple):
    label: str  # shown in citations
    name: str
    language: str
    license: str
    public: bool  # may be redistributed / served publicly
    source: str  # sword:<module> | pdf:<file in data_dir> | pypi:<reader>
    conditions: str = ""

    @property
    def original(self) -> bool:
        """Hebrew / Aramaic or Greek source text (not a translation)."""
        return self.language in ("hbo", "grc")


VERSIONS = {
    "kjv": Version(
        "KJV",
        "King James Version (1769)",
        "en",
        "Public Domain",
        True,
        "sword:engKJV2006eb",
        "UK only: printing needs permission (Crown letters patent); free elsewhere.",
    ),
    "rv1909": Version(
        "RV1909", "Reina-Valera 1909", "es", "Public Domain", True, "sword:spaRV1909eb"
    ),
    "web": Version(
        "WEB",
        "World English Bible",
        "en",
        "Public Domain",
        True,
        "sword:engWEB2015eb",
        "'World English Bible' is a trademark of eBible.org: only for the unchanged text.",
    ),
    "rvr1960": Version(
        "RVR1960",
        "Reina-Valera 1960",
        "es",
        "© Sociedades Bíblicas en América Latina; PDF: personal use only",
        False,
        "pdf:biblia-reina-valera-1960.pdf",
        "Not for public serving or redistribution.",
    ),
    "wlc": Version(
        "WLC",
        "Westminster Leningrad Codex: Hebrew Bible, with the Biblical Aramaic (OT)",
        "hbo",
        "Hebrew text unrestricted (UXLC, tanach.us); word data CC BY 4.0 (Clear Bible)",
        True,
        "pypi:wlc",
        "Credit: UXLC 1.8, Tanach.us; Macula Hebrew data and alignment © Clear Bible, Inc., "
        "CC BY 4.0. Verses renumbered to the English versification.",
    ),
    "sblgnt": Version(
        "SBLGNT",
        "SBL Greek New Testament (NT)",
        "grc",
        "CC BY 4.0 (SBL and Logos Bible Software); MorphGNT lemmas CC BY-SA",
        True,
        "pypi:sblgnt",
        "Credit: SBLGNT © 2010 Society of Biblical Literature and Logos Bible Software "
        "(license as stated in Clear Bible's catalog; check sblgnt.com before serving it); "
        "lemmas by MorphGNT (James Tauber).",
    ),
    "etheridge": Version(
        "Etheridge",
        "Etheridge's translation of the Syriac (Aramaic) Peshitta (1849, NT)",
        "en",
        "Public Domain",
        True,
        "pypi:etheridge",
        "An English witness of the Aramaic New Testament (the Peshitta).",
    ),
    "ylt": Version(
        "YLT", "Young's Literal Translation (1898)", "en", "Public Domain", True, "pypi:ylt"
    ),
    "leeser": Version(
        "Leeser",
        "Leeser's translation of the Hebrew Scriptures (1853, OT)",
        "en",
        "Public Domain",
        True,
        "pypi:leeser",
        "A Jewish translation from the Hebrew Masoretic text.",
    ),
}
# whose per-word Strong's numbers are kept: the KJV's name the original word behind an English
# word, the originals' give its form in the verse (the other modules' tags aren't used)
TAGGED = ("kjv", "wlc", "sblgnt")

DEBIAN = "https://deb.debian.org/debian/"
# module -> (Debian package file, SHA-256)
PACKAGES = {
    "engKJV2006eb": (
        "pool/main/s/sword-text-kjv/sword-text-kjv_22.29-1_all.deb",
        "abaed17486a0e3d2895a932c9cd7fff2a3a2bd5dc84d37518aa272b2afbfa433",
    ),
    "spaRV1909eb": (
        "pool/main/s/sword-text-sparv/sword-text-sparv_2.87-1_all.deb",
        "c39b44396b6898a05b3753ac6c55c88f1a5a41f1fd6a0bc5dac7fe96aca5141b",
    ),
    "engWEB2015eb": (
        "pool/main/s/sword-text-web/sword-text-web_522.0-1_all.deb",
        "b2029a802c44f1d6103de45e9ef5c64a9bd66466776e593e386b5ea84bfc110f",
    ),
    # Strong's dictionaries (public domain): the original word behind a KJV word
    "strongshebrew": (
        "pool/main/s/sword-dict-strongs-hebrew/sword-dict-strongs-hebrew_3.0-2_all.deb",
        "58c887317b8c8019dca98a74e83af50a096050fb383e5a1aa5bf46b7cda41ac4",
    ),
    "strongsgreek": (
        "pool/main/s/sword-dict-strongs-greek/sword-dict-strongs-greek_3.0-4_all.deb",
        "2fa839ec14bbafedb38a4ad6ca9ead1c63ac11a7323e560ccb053c184e443095",
    ),
}

PYPI = "https://files.pythonhosted.org/packages/"
# reader -> (file, SHA-256 as published on PyPI, data files used: exact paths or dir/ prefixes)
PYPI_PACKAGES = {
    "wlc": (
        "6c/4e/2cc7c90f1341a2062b6db749500922ec2cf3008cb4a0a627c12bab78d559/"
        "bible_alignments-0.3.5-py3-none-any.whl",
        "72f43650950cb455f5078da9206277d97c91823dc9363e569c0b236df7eda0c1",
        ("data/sources/WLC-LEB.tsv", "data/alignments/eng/YLT/WLC-YLT-manual.json"),
    ),
    "sblgnt": (
        "94/8f/1d06b1ed8631e5feea8afb4a0d013e783225941c977a7ad31439f21e83db/py-sblgnt-0.10.tar.gz",
        "f0eb2940e1c94fd576a0d492fd5e863f5218d56c09c324a1f8a1556364242edc",
        ("py-sblgnt-0.10/pysblgnt/sblgnt/",),
    ),
    "etheridge": (
        "4c/64/08f492e49c122e320a57f8e8336cfc1c555cfec8f85dc11818c34eddea1a/"
        "pythonbible_etheridge-0.0.2-py3-none-any.whl",
        "9c7d7b75f31a3a1528ad4ffba6ffd7008827126f912c39f7f451b0f76b732233",
        ("pythonbible_etheridge/plain_text_bible.py",),
    ),
    "ylt": (
        "30/77/fe18dc3ac39b1299f55f14402b3044bac7f2e136423eb8a2126c9de74bb4/"
        "pythonbible_ylt-0.0.2-py3-none-any.whl",
        "dfff6ff29788ab270852d38bdd041b65d37fc6945a63a12370bc34503a327517",
        ("pythonbible_ylt/plain_text_bible.py",),
    ),
    "leeser": (
        "e5/72/f9c8e88a7599a048f5b77e65faf3a6c85ae507a1a23497e81b9871850adc/"
        "pythonbible_leeser-0.0.2-py3-none-any.whl",
        "1e6933e9eb2a1bbbd8b5978a19a9c15f4e4025f79f411d0821f5b668a7674e7b",
        ("pythonbible_leeser/plain_text_bible.py",),
    ),
}


class LicenseError(ValueError):
    pass


def check_licenses(version_ids, allow_restricted: bool) -> None:
    """Refuse versions whose license does not allow public use, unless explicitly allowed."""
    for vid in version_ids:
        if vid not in VERSIONS:
            raise ValueError(f"Unknown version {vid!r}; known: {', '.join(VERSIONS)}")
        v = VERSIONS[vid]
        if not v.public and not allow_restricted:
            raise LicenseError(
                f"{v.label} is '{v.license}'. To use it privately set "
                "BIBLE_RAG_ALLOW_RESTRICTED=true; do not serve it publicly."
            )


# ------------------------------------------------------------------------------ Debian
def _deb_members(data: bytes) -> dict[str, bytes]:
    """Members of a .deb (an `ar` archive), no dpkg needed."""
    if not data.startswith(b"!<arch>\n"):
        raise ValueError("not a .deb file")
    members, pos = {}, 8
    while pos + 60 <= len(data):
        name, size = (
            data[pos : pos + 16].decode().strip().rstrip("/"),
            int(data[pos + 48 : pos + 58]),
        )
        members[name] = data[pos + 60 : pos + 60 + size]
        pos += 60 + size + size % 2
    return members


def fetch_module(module: str, sword_dir: Path, timeout: float = 60) -> Path:
    """Download, verify and unpack a Debian SWORD package into sword_dir (once)."""
    conf = sword_dir / "mods.d" / f"{module}.conf"
    if conf.exists():
        return conf
    path, sha256 = PACKAGES[module]
    data = _download(DEBIAN + path, sha256, timeout)
    tar_xz = _deb_members(data)["data.tar.xz"]
    with tarfile.open(fileobj=io.BytesIO(lzma.decompress(tar_xz))) as tar:
        for m in tar.getmembers():
            rel = m.name.split("usr/share/sword/", 1)
            if m.isfile() and len(rel) == 2 and ".." not in Path(rel[1]).parts:  # module files only
                target = sword_dir / rel[1]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(tar.extractfile(m).read())
    return conf


def _download(url: str, sha256: str, timeout: float) -> bytes:
    """The file at url, refused unless its SHA-256 is the pinned one."""
    import httpx

    data = httpx.get(url, timeout=timeout, follow_redirects=True).raise_for_status().content
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError(f"{url}: SHA-256 mismatch, refusing to use it")
    return data


def _safe(root: Path, member: str) -> Path | None:
    rel = Path(member)
    if rel.is_absolute() or ".." in rel.parts:
        return None
    return root / rel


def fetch_pypi(reader: str, sources_dir: Path, timeout: float = 120) -> Path:
    """Download a pinned PyPI file, verify it, and extract only its data files (once)."""
    path, sha256, wanted = PYPI_PACKAGES[reader]
    root = sources_dir / reader
    done = root / ".complete"
    if done.exists():
        return root
    data = _download(PYPI + path, sha256, timeout)

    def keep(name: str) -> bool:
        return any(name == w or (w.endswith("/") and name.startswith(w)) for w in wanted)

    if path.endswith(".whl"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            members = [(n, z.read(n)) for n in z.namelist() if keep(n) and not n.endswith("/")]
    else:
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            members = [
                (m.name, tar.extractfile(m).read())
                for m in tar.getmembers()
                if m.isfile() and keep(m.name)
            ]
    if not members:
        raise ValueError(f"{path}: none of {wanted} found")
    for name, content in members:
        if (target := _safe(root, name)) is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    done.write_text(sha256)
    return root


# ------------------------------------------------------------------------------- SWORD
_GLUE = "\x00"  # inline tag between two word characters: same word or two words?


def osis_to_text(raw: str) -> str:
    """OSIS markup -> plain verse text, words unchanged. Ambiguous boundaries keep a _GLUE
    marker for resolve_glue() ("díjo<tc>le</tc>" -> díjole, "LORD<tc>is" -> LORD is)."""
    s = re.sub(r"<note\b.*?</note>", " ", raw, flags=re.S)  # footnotes, cross-references
    s = re.sub(
        r'<div\b(?=[^>]*type="(?:glossary|introduction)")[^>]*sID="([^"]+)".*?eID="\1"[^>]*>',
        " ",
        s,
        flags=re.S,
    )  # WEB glossary at the end of Revelation
    s = re.sub(r'<title(?![^>]*canonical="true")[^>]*>.*?</title>', " ", s, flags=re.S)  # headings
    s = re.sub(r"</?(?:l|lg|lb|div|p|chapter|milestone|title|list|item)\b[^>]*>", " ", s)
    # inline markup never implies a space ("where</w><w>in"), except that the source drops the
    # space around added words / divine names ("LORD</divineName><transChange>is"): mark those
    s = re.sub(
        r"(?:<[^>]+>)+",
        lambda m: _GLUE if re.search(r"transChange|divineName|<seg", m[0]) else "",
        s,
    )
    s = html.unescape(s).replace("¶", " ")
    s = re.sub(rf"{_GLUE}+", _GLUE, s)
    s = re.sub(rf"(?<!\w){_GLUE}|{_GLUE}(?!\w)", "", s)  # keep markers only between letters
    s = re.sub(r"\s+", " ", s)
    return re.sub(r" ([,.;:!?’”)\]])", r"\1", s).strip()  # space left by a removed note


# fmt: off
_ENCLITICS = {a + b for a in ("", "me", "te", "se", "nos", "os")
              for b in ("", "me", "te", "se", "nos", "os", "le", "les", "lo", "los", "la", "las")}
_NOT_VERBS = set("éste ésta éstos éstas ése ésa ésos ésas aquél aquélla aquéllos aquéllas cuánto "
                 "cuánta cuántos cuántas también sólo después árbol ángel".split())
# fmt: on


def resolve_glue(texts: list[str], language: str) -> list[str]:
    """Decide each marked boundary. Spanish: join enclitic pronouns to verb forms (díjo+le,
    sáca+lo, pagar+la, dad+me, júrame+lo, dígan+lo); ambiguous monosyllables ("haz lo que")
    stay apart. English: join only if the joined word occurs elsewhere in the text."""
    vocab = {
        w.lower()
        for t in texts
        for tok in t.split()
        if _GLUE not in tok
        for w in re.findall(r"\w+", tok)
    }

    def joins(left: str, right: str) -> bool:
        left, right = left.lower(), right.lower()
        if language != "es":
            return left + right in vocab
        return (
            right in _ENCLITICS - {""}
            and left not in _NOT_VERBS
            and (
                left not in vocab  # "sáca", "díjo": accent shifted by the enclitic
                or re.search(r"(ar|er|ir|ír|ndo|d|me|te|se|nos)$", left) is not None
                # stressed earlier syllable + verb ending: "dígan", "envía" (not "ejército")
                or re.search(r"[áéíóú]\w*(a|an|e|en)$", left) is not None
            )
        )

    def fix(m: re.Match) -> str:
        out, joined = m[1], False
        for part in m[2].split(_GLUE)[1:]:
            left, right = re.search(r"\w+$", out)[0], re.match(r"\w+", part)[0]
            # a word that just took an enclitic takes no other ("Díganlo los", not "Díganlolos")
            joined = not joined and joins(left, right)
            out += ("" if joined else " ") + part
        return out

    return [re.sub(rf"(\S*?)((?:{_GLUE}\S+)+)", fix, t) if _GLUE in t else t for t in texts]


_W = re.compile(r'<w\b[^>]*?lemma="([^"]*)"[^>]*>(.*?)</w>', re.S)


def strongs_tags(raw: str) -> list:
    """[[words, [Strong's codes]], ...] of an OSIS verse with Strong's numbers (KJV)."""
    tags = []
    for lemma, inner in _W.findall(raw):
        codes = [_code(c) for c in re.findall(r"strong:([HG]\d+)", lemma)]
        words = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", inner))).strip()
        if codes and words:
            tags.append([words, codes])
    return tags


def _code(code: str) -> str:
    """H07585 / 07585 (Hebrew) -> H7585; letter suffixes (0871a) dropped."""
    m = re.match(r"([HG]?)0*(\d+)", code)
    return f"{m[1] or 'H'}{m[2]}"


def read_sword(module: str, sword_dir: Path, language: str) -> list[Unit]:
    """One Unit per verse of the 66-book canon (with its Strong's numbers, if tagged)."""
    from pysword.modules import SwordModules

    sword = SwordModules(str(sword_dir))
    sword.parse_modules()
    bible = sword.get_bible_from_module(module)
    # pysword decompresses a whole block per verse; caching it makes reading ~500x faster
    bible._decompressed_text = functools.lru_cache(maxsize=4)(bible._decompressed_text)
    names = {osis: (en, es) for osis, en, es in BOOKS}
    units = []
    for testament, books in bible.get_structure().get_books().items():
        for book in books:
            if book.osis_name not in names:
                continue  # deuterocanonical books (WEB)
            name = names[book.osis_name][language == "es"]
            refs = [(c, v) for c, n in enumerate(book.chapter_lengths, 1) for v in range(1, n + 1)]
            raw = bible.get_iter(books=[book.osis_name], clean=False)
            for (chapter, verse), osis in zip(refs, raw, strict=True):
                if text := osis_to_text(osis):
                    unit = Unit(name, chapter, verse, verse, text, 0, testament.upper())
                    unit.tags = strongs_tags(osis) if "strong:" in osis else []
                    units.append(unit)
    for unit, text in zip(units, resolve_glue([u.text for u in units], language), strict=True):
        unit.text = text
    return units


def load_units(version_id: str, settings) -> list[Unit]:
    version = VERSIONS[version_id]
    kind, _, location = version.source.partition(":")
    if kind == "sword":
        fetch_module(location, settings.sword_dir)
        return read_sword(location, settings.sword_dir, version.language)
    if kind == "pypi":
        root = fetch_pypi(location, settings.sources_dir)
        if location == "wlc":
            return read_wlc(root)
        if location == "sblgnt":
            return read_sblgnt(root, strongs_lexicon(settings.sword_dir))
        return read_pythonbible(next(root.glob("*/plain_text_bible.py")))
    from .corpus import extract_pages, parse_rvr, resolve_line_hyphens

    pdf = settings.data_dir / location
    if not pdf.exists():
        raise FileNotFoundError(f"{version.name}: PDF not found at {pdf}")
    return resolve_line_hyphens(list(parse_rvr(extract_pages(pdf))))


# --------------------------------------------------------------------- PyPI readers
def _unit(book: int, chapter: int, verse: int, text: str, tags=None) -> Unit:
    """A verse of book index `book` (corpus.BOOKS), with the English book name."""
    unit = Unit(BOOKS[book][1], chapter, verse, verse, text, 0, "OT" if book < 39 else "NT")
    unit.tags = tags or []
    return unit


# Windows-1252 punctuation left as C1 control characters in some texts
_C1 = str.maketrans(
    {"\x91": "‘", "\x92": "’", "\x93": "“", "\x94": "”", "\x96": "–", "\x97": "—", "\x85": "…"}
)


def read_pythonbible(path: Path) -> list[Unit]:
    """A `pythonbible` text module, read as data: the file is parsed (ast), never executed.
    It holds the whole text and the start / end offset of every verse (id BBCCCVVV)."""
    call = ast.parse(path.read_text(encoding="utf-8")).body[-1].value
    text = call.args[1].value
    offsets = [
        {ast.literal_eval(k): ast.literal_eval(v) for k, v in zip(d.keys, d.values, strict=True)}
        for d in call.args[2:4]
    ]
    units = []
    for vid in sorted(offsets[0]):
        book, chapter, verse = vid // 1_000_000, vid // 1000 % 1000, vid % 1000
        if not 1 <= book <= 66:
            continue
        raw = text[offsets[0][vid] : offsets[1][vid]].translate(_C1)
        verse_text = re.sub(r"\s+", " ", re.sub(r"^\s*\d+\.\s*", "", raw)).strip()
        if verse_text:
            units.append(_unit(book - 1, chapter, verse, verse_text))
    return units


def read_wlc(root: Path) -> list[Unit]:
    """The Hebrew Bible, word by word with Strong's numbers, renumbered to the English
    versification: each Hebrew word goes to the English verse its aligned YLT word is in
    (Hebrew Psalm 51:3 -> Psalm 51:1, Malachi 3:19 -> 4:1, Joel 3:1 -> 2:28); an unaligned
    word follows the previous word of its verse."""
    align = json.loads((root / "data/alignments/eng/YLT/WLC-YLT-manual.json").read_text())

    def flat(ids):
        for i in ids:
            yield from (flat(i) if isinstance(i, list) else [i])

    english = {}  # Hebrew token id -> English verse (BBCCCVVV)
    for a in align:
        targets = sorted({t[:8] for t in flat(a["target_ids"])})
        for source in flat(a["source_ids"]):
            if targets:
                english[source] = targets[0]
    verses: dict[str, list] = collections.defaultdict(list)  # English verse -> [(word id, ...)]
    with open(root / "data/sources/WLC-LEB.tsv", encoding="utf-8") as fh:
        rows = csv.reader(fh, delimiter="\t")
        next(rows)
        last: dict[str, str] = {}  # Hebrew verse -> English verse of its previous word
        pending: dict[str, list] = collections.defaultdict(list)  # words before any aligned
        for row in rows:
            token, text, strongs = row[0], row[2], row[3]
            hebrew_verse = token[:8]
            target = english.get(token) or last.get(hebrew_verse)
            item = (token[:11], token[11:], text, strongs)
            if target is None:
                pending[hebrew_verse].append(item)
                continue
            if hebrew_verse in pending:
                verses[target].extend(pending.pop(hebrew_verse))
            last[hebrew_verse] = target
            verses[target].append(item)
    units = []
    for vid in sorted(verses):
        book, chapter, verse = int(vid[:2]) - 1, int(vid[2:5]), max(int(vid[5:8]), 1)
        words: dict[str, list] = {}  # word id -> parts (prefixes, stem, suffixes)
        for word_id, _, text, strongs in verses[vid]:
            words.setdefault(word_id, []).append((text, strongs))
        text = " ".join("".join(t for t, _ in parts) for parts in words.values())
        tags = [
            ["".join(t for t, _ in parts), [_code(c) for _, c in parts if c.strip(" 0")]]
            for parts in words.values()
        ]
        if units and (units[-1].book, units[-1].chapter, units[-1].verse_start) == (
            BOOKS[book][1],
            chapter,
            verse,
        ):  # a psalm title ("verse 0") joins verse 1
            units[-1].text += " " + text
            units[-1].tags += tags
        else:
            units.append(_unit(book, chapter, verse, text, tags))
    return units


_SIGLA = re.compile("[\u2e00-\u2e0f]")  # ⸀ ⸂ ⸃: variant markers of the SBLGNT
_GREEK_BOOKS = {f"{n:02d}": 38 + n for n in range(1, 28)}  # MorphGNT 01 = Matthew (index 39)


def read_sblgnt(root: Path, lexicon: dict) -> list[Unit]:
    """The Greek New Testament (SBLGNT), one word per line with its lemma (MorphGNT);
    lemmas are linked to Strong's numbers through the Strong's Greek dictionary."""
    greek = {}
    for code, entry in lexicon.items():
        if code.startswith("G"):
            greek.setdefault(_plain(entry["word"]), code)
    verses: dict[tuple, list] = collections.defaultdict(list)
    for path in sorted((root / "py-sblgnt-0.10/pysblgnt/sblgnt").glob("*-morphgnt.txt")):
        for line in path.read_text(encoding="utf-8").splitlines():
            bcv, _, _, _, text, _, _, lemma = line.split()
            verses[(_GREEK_BOOKS[bcv[:2]], int(bcv[2:4]), int(bcv[4:6]))].append((text, lemma))
    units = []
    for (book, chapter, verse), words in sorted(verses.items()):
        tags = [[w, [greek[_plain(lemma)]] if _plain(lemma) in greek else []] for w, lemma in words]
        text = _SIGLA.sub("", " ".join(w for w, _ in words))  # critical-apparatus marks
        tags = [[_SIGLA.sub("", w), codes] for w, codes in tags]
        units.append(_unit(book, chapter, verse, text, tags))
    return units


def _plain(word: str) -> str:
    decomposed = unicodedata.normalize("NFD", word)
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn").lower()


# ------------------------------------------------------------------- Strong's dictionary
def strongs_lexicon(sword_dir: Path) -> dict[str, dict]:
    """{"H7585": {"word": "שׁאל", "translit": "sh'ôl", "kjv": "grave, hell, pit", "def": ...}}
    from the public-domain Strong's dictionaries (Debian), cached as strongs.json."""
    cache = sword_dir / "strongs.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    lexicon = {}
    for module, prefix in (("strongshebrew", "H"), ("strongsgreek", "G")):
        fetch_module(module, sword_dir)
        base = sword_dir / "modules" / "lexdict" / "zld" / module / "dict"
        for key, entry in read_zld(base).items():
            if key.isdigit() and (parsed := _strongs_entry(entry)):
                lexicon[f"{prefix}{int(key)}"] = parsed
    cache.write_text(json.dumps(lexicon, ensure_ascii=False), encoding="utf-8")
    return lexicon


def _strongs_entry(entry: str) -> dict | None:
    """An entry's word, transliteration, meaning and KJV renderings. Hebrew entries end with
    "- {grave} {hell} pit.", Greek ones with ":--grave, hell."."""

    def plain(x: str) -> str:
        return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", x))).strip()

    words = re.findall(r"<orth>([^<]+)</orth>", entry)
    translit = re.search(r'<orth[^>]*type="trans"[^>]*>(.*?)</orth>', entry, re.S)
    definition = re.search(r"<def>(.*?)</def>", entry, re.S)
    if not words or not definition:
        return None
    text = plain(definition[1])
    split = re.search(r":?--|\s-\s", text)
    meaning, kjv = (text[: split.start()], text[split.end() :]) if split else (text, "")
    kjv = re.sub(r"\}\s*(?=\S)", ", ", kjv).replace("{", "").replace("}", "")
    return {
        "word": words[0].split()[0].strip(),
        "translit": plain(translit[1]).split()[0] if translit else "",
        "kjv": re.sub(r"\s+", " ", kjv).strip(" .,;"),
        "def": re.sub(r"[{}]", "", meaning).strip(" :;")[:300],
    }


def read_zld(base: Path) -> dict[str, str]:
    """Entries of a SWORD compressed dictionary (zLD: .idx/.dat keys, .zdx/.zdt zlib blocks)."""
    idx, dat = Path(f"{base}.idx").read_bytes(), Path(f"{base}.dat").read_bytes()
    zdx, zdt = Path(f"{base}.zdx").read_bytes(), Path(f"{base}.zdt").read_bytes()
    blocks: dict[int, list[bytes]] = {}

    def block(n: int) -> list[bytes]:
        if n not in blocks:
            offset, size = struct.unpack_from("<II", zdx, n * 8)
            raw = zlib.decompress(zdt[offset : offset + size])
            count = struct.unpack_from("<I", raw, 0)[0]
            table = [struct.unpack_from("<II", raw, 4 + i * 8) for i in range(count)]
            blocks[n] = [raw[o : o + s] for o, s in table]
        return blocks[n]

    entries = {}
    for i in range(len(idx) // 8):
        offset, size = struct.unpack_from("<II", idx, i * 8)
        key, _, rest = dat[offset : offset + size].partition(b"\n")
        number, entry = struct.unpack_from("<II", rest, 0)
        text = block(number)[entry].rstrip(b"\0").decode("utf-8", "replace")
        entries[key.decode("utf-8", "replace").strip()] = text
    return entries
