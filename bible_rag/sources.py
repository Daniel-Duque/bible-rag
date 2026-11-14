"""Bible versions, their licenses, and where the text comes from.

Public-domain texts are SWORD modules packaged by Debian (main archive = license reviewed by
Debian), downloaded by pinned file + SHA-256 and read with pysword. The hashes below come from
the trixie package index, whose signature was verified with the Debian archive key.
RVR1960 (PDF) is "personal use only": it is refused unless allow_restricted is set.
"""

from __future__ import annotations

import functools
import hashlib
import html
import io
import lzma
import re
import tarfile
from pathlib import Path
from typing import NamedTuple

from .corpus import BOOKS, Unit


class Version(NamedTuple):
    label: str  # shown in citations
    name: str
    language: str
    license: str
    public: bool  # may be redistributed / served publicly
    source: str  # sword:<module> | pdf:<file in data_dir>
    conditions: str = ""


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
}

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
    import httpx

    path, sha256 = PACKAGES[module]
    data = (
        httpx.get(DEBIAN + path, timeout=timeout, follow_redirects=True).raise_for_status().content
    )
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError(f"{path}: SHA-256 mismatch, refusing to use it")
    tar_xz = _deb_members(data)["data.tar.xz"]
    with tarfile.open(fileobj=io.BytesIO(lzma.decompress(tar_xz))) as tar:
        for m in tar.getmembers():
            rel = m.name.split("usr/share/sword/", 1)
            if m.isfile() and len(rel) == 2 and ".." not in Path(rel[1]).parts:  # module files only
                target = sword_dir / rel[1]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(tar.extractfile(m).read())
    return conf


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


def read_sword(module: str, sword_dir: Path, language: str) -> list[Unit]:
    """One Unit per verse of the 66-book canon."""
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
            for (chapter, verse), text in zip(refs, raw, strict=True):
                if text := osis_to_text(text):
                    units.append(Unit(name, chapter, verse, verse, text, 0, testament.upper()))
    for unit, text in zip(units, resolve_glue([u.text for u in units], language), strict=True):
        unit.text = text
    return units


def load_units(version_id: str, settings) -> list[Unit]:
    version = VERSIONS[version_id]
    kind, _, location = version.source.partition(":")
    if kind == "sword":
        fetch_module(location, settings.sword_dir)
        return read_sword(location, settings.sword_dir, version.language)
    from .corpus import extract_pages, parse_rvr, resolve_line_hyphens

    pdf = settings.data_dir / location
    if not pdf.exists():
        raise FileNotFoundError(f"{version.name}: PDF not found at {pdf}")
    return resolve_line_hyphens(list(parse_rvr(extract_pages(pdf))))
