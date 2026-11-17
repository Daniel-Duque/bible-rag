"""Licensed sources: OSIS cleaning, word-boundary repair, license gate, Debian packages, and
(marker `sword`) the real public-domain modules."""

import hashlib
import io
import lzma
import tarfile

import httpx
import pytest

from bible_rag import sources
from bible_rag.config import Settings
from bible_rag.corpus import strip_marks
from bible_rag.sources import (
    _GLUE,
    VERSIONS,
    LicenseError,
    check_licenses,
    fetch_module,
    load_units,
    osis_to_text,
    resolve_glue,
)


def test_osis_to_text_keeps_words_and_drops_apparatus():
    raw = (
        '<title type="section">Heading</title><title canonical="true">A Psalm of David.</title>'
        '<w lemma="x">only</w> born</q><note placement="foot">3:16 The phrase…</note><q>Son'
        '</q>, <w>where</w><w>in</w> ¶ it &amp; <div sID="g1" type="glossary"/>Zion is a '
        'name…<div eID="g1" type="glossary"/>'
    )
    assert osis_to_text(raw) == "A Psalm of David. only born Son, wherein it &"
    # added words and divine names lose their spaces in the source: marked for resolve_glue
    assert osis_to_text("The <divineName>LORD</divineName><transChange>is</transChange> my") == (
        f"The LORD{_GLUE}is my"
    )


def test_resolve_glue_spanish_enclitics_and_english_vocabulary():
    g = _GLUE
    es = [
        f"y díjo{g}le{g}Dios:",
        f"allí{g}también",
        f"Dígan{g}lo{g}los redimidos",
        f"Júrame{g}lo",
        f"éste{g}le dijo",
        f"haz{g}lo que",
        f"pagar{g}la",
        "Dijo y díjole; haz bien",  # vocabulary: "haz" also stands alone
    ]
    assert resolve_glue(es, "es")[:7] == [
        "y díjole Dios:",
        "allí también",
        "Díganlo los redimidos",
        "Júramelo",
        "éste le dijo",
        "haz lo que",
        "pagarla",
    ]
    en = [f"The LORD{g}is my", f"with{g}out", "without measure"]
    assert resolve_glue(en, "en")[:2] == ["The LORD is my", "without"]


def test_license_gate():
    check_licenses(["kjv", "rv1909", "web"], allow_restricted=False)
    with pytest.raises(LicenseError, match="personal use only"):
        check_licenses(["kjv", "rvr1960"], allow_restricted=False)
    check_licenses(["rvr1960"], allow_restricted=True)
    with pytest.raises(ValueError, match="Unknown version"):
        check_licenses(["niv"], allow_restricted=True)
    for v in VERSIONS.values():  # public: public domain, or an attribution license + credit
        if v.public and v.license != "Public Domain":
            assert "CC BY" in v.license and v.conditions.startswith("Credit:"), v.label


def fake_deb(files: dict[str, bytes]) -> bytes:
    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    out = b"!<arch>\n"
    for name, data in (
        ("debian-binary", b"2.0\n"),
        ("data.tar.xz", lzma.compress(tar_buf.getvalue())),
    ):
        out += f"{name:<16}{0:<12}{0:<6}{0:<6}{100644:<8}{len(data):<10}`\n".encode() + data
        out += b"\n" if len(data) % 2 else b""
    return out


def test_fetch_module_verifies_and_unpacks(tmp_path, monkeypatch):
    deb = fake_deb(
        {
            "./usr/share/sword/mods.d/Test.conf": b"[Test]",
            "./usr/share/sword/modules/texts/ztext/test/nt.bzz": b"data",
            "./usr/share/sword/../../evil": b"x",
            "./usr/share/doc/x/copyright": b"pd",
        }
    )
    monkeypatch.setitem(
        sources.PACKAGES, "Test", ("pool/test.deb", hashlib.sha256(deb).hexdigest())
    )
    monkeypatch.setattr(
        httpx,
        "get",
        lambda *a, **kw: httpx.Response(
            200, content=deb, request=httpx.Request("GET", "https://x")
        ),
    )
    conf = fetch_module("Test", tmp_path)
    assert conf.read_bytes() == b"[Test]"
    assert (tmp_path / "modules/texts/ztext/test/nt.bzz").exists()
    assert sorted(p.name for p in tmp_path.rglob("*") if p.is_file()) == ["Test.conf", "nt.bzz"]
    monkeypatch.setitem(sources.PACKAGES, "Bad", ("pool/test.deb", "0" * 64))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        fetch_module("Bad", tmp_path / "other")


# ------------------------------------------------------------- real modules (downloads once)
@pytest.fixture(scope="module")
def texts():
    settings = Settings.from_env()
    return {
        v: {(u.book, u.chapter, u.verse_start): u.text for u in load_units(v, settings)}
        for v in ("kjv", "rv1909", "web")
    }


@pytest.mark.sword
def test_real_modules(texts):
    kjv, rv, web = texts["kjv"], texts["rv1909"], texts["web"]
    assert len(kjv) == 31102 and len({(b, c) for b, c, _ in kjv}) == 1189
    assert len({(b, c) for b, c, _ in rv}) == len({(b, c) for b, c, _ in web}) == 1189
    assert kjv["John", 3, 16].startswith(
        "For God so loved the world, that he gave his only begotten Son"
    )
    assert kjv["Psalms", 23, 1] == "A Psalm of David. The LORD is my shepherd; I shall not want."
    assert rv["1 Reyes", 3, 5].startswith(
        "Y aparecióse Jehová á Salomón en Gabaón una noche en sueños, y díjole Dios:"
    )
    assert rv["Salmos", 107, 2].startswith("Díganlo los redimidos de Jehová")
    assert "only born Son" in web["John", 3, 16]
    assert (
        web["Revelation", 22, 21]
        == "The grace of the Lord Jesus Christ be with all the saints. Amen."
    )
    for text in (t for v in texts.values() for t in v.values()):
        assert not any(c in text for c in "<>¶\x00"), text


# ---------------------------------------------------- original-language texts and Strong's
def test_strongs_tags_from_osis():
    osis = (
        'For thou wilt not <w lemma="strong:H5800">leave</w> my soul in '
        '<w lemma="strong:H7585">hell</w>; <w lemma="strong:G3588 strong:G2316">the God</w>'
        ' <w lemma="lemma.TR:x">no strong</w>'
    )
    assert sources.strongs_tags(osis) == [
        ["leave", ["H5800"]],
        ["hell", ["H7585"]],
        ["the God", ["G3588", "G2316"]],
    ]
    assert [sources._code(c) for c in ("H07585", "0871a", "G0080")] == ["H7585", "H871", "G80"]


def test_strongs_dictionary_entries_in_both_formats():
    hebrew = (
        '<entryFree n="7585"><orth>שׁאל    שׁאול</orth><lb/><orth rend="bold" type="trans">'
        "sh<hi>e</hi>'ôl  sh'ôl</orth><def>From <ref>H7592</ref>; <hi>hades</hi> or the world "
        "of the dead: - {grave} {hell} pit.</def></entryFree>"
    )
    greek = (
        '<entryFree n="00086"><orth>ᾅδης</orth><orth rend="bold" type="trans">háidēs</orth>'
        '<def><orig>from G1</orig> properly, unseen, i.e. "Hades":--grave, hell.</def>'
    )
    assert sources._strongs_entry(hebrew) == {
        "word": "שׁאל",
        "translit": "she'ôl",
        "kjv": "grave, hell, pit",
        "def": "From H7592; hades or the world of the dead",
    }
    entry = sources._strongs_entry(greek)
    assert entry["word"] == "ᾅδης" and entry["kjv"] == "grave, hell"
    assert sources._strongs_entry("<entryFree>no definition</entryFree>") is None


def test_read_zld(tmp_path):
    """A compressed SWORD dictionary: keys -> (block, entry), zlib blocks of entries."""
    import struct
    import zlib

    entries = [b"first entry", b"second entry"]
    block = struct.pack("<I", 2)
    body, table = b"", b""
    offset = 4 + 8 * len(entries)
    for e in entries:
        table += struct.pack("<II", offset + len(body), len(e))
        body += e
    compressed = zlib.compress(block + table + body)
    (tmp_path / "d.zdt").write_bytes(compressed)
    (tmp_path / "d.zdx").write_bytes(struct.pack("<II", 0, len(compressed)))
    dat, idx = b"", b""
    for n, key in enumerate((b"00001", b"00002")):
        record = key + b"\n" + struct.pack("<II", 0, n)
        idx += struct.pack("<II", len(dat), len(record))
        dat += record
    (tmp_path / "d.dat").write_bytes(dat)
    (tmp_path / "d.idx").write_bytes(idx)
    assert sources.read_zld(tmp_path / "d") == {"00001": "first entry", "00002": "second entry"}


def test_pythonbible_file_is_read_not_run(tmp_path):
    marker = tmp_path / "pwned"
    code = (
        "import os\n"
        f"open({str(marker)!r}, 'w')\n"  # would run if the file were executed
        "bible = Bible(Version.X, ' 1. In the beginning  God. 1. Amen\\x92s.',"
        " {1001001: 0, 66022021: 27}, {1001001: 27, 66022021: 41}, {}, {}, {}, True)\n"
    )
    path = tmp_path / "plain_text_bible.py"
    path.write_text(code, encoding="utf-8")
    units = sources.read_pythonbible(path)
    assert [(u.book, u.chapter, u.verse_start, u.text) for u in units] == [
        ("Genesis", 1, 1, "In the beginning God."),
        ("Revelation", 22, 21, "Amen’s."),
    ]
    assert not marker.exists()


def test_wlc_is_renumbered_to_the_english_verses(tmp_path):
    """Hebrew Malachi 3:19 is English 4:1; a psalm title (Hebrew verse 1, English "verse 0")
    joins verse 1; a word without alignment follows its neighbours."""
    import json

    root = tmp_path / "data"
    (root / "sources").mkdir(parents=True)
    (root / "alignments/eng/YLT").mkdir(parents=True)
    rows = [
        ("identifier", "alt", "text", "strongs", "gloss"),
        ("390030190011", "", "כִּי", "3588", "for"),
        ("390030190021", "", "הִנֵּה", "2009", "behold"),
        ("190030010011", "", "מִזְמוֹר", "4210", "psalm"),
        ("190030020011", "", "יְהוָה", "3068", "LORD"),
    ]
    (root / "sources/WLC-LEB.tsv").write_text(
        "\n".join("\t".join(r) for r in rows), encoding="utf-8"
    )
    align = [
        {"source_ids": ["390030190011"], "target_ids": ["39004001001"]},
        {"source_ids": [["190030010011"]], "target_ids": ["19003000001"]},
        {"source_ids": ["190030020011"], "target_ids": ["19003001002"]},
    ]
    (root / "alignments/eng/YLT/WLC-YLT-manual.json").write_text(json.dumps(align))
    units = sources.read_wlc(tmp_path)
    assert [(u.book, u.chapter, u.verse_start, u.text) for u in units] == [
        ("Psalms", 3, 1, "מִזְמוֹר יְהוָה"),
        ("Malachi", 4, 1, "כִּי הִנֵּה"),
    ]
    assert units[1].tags == [["כִּי", ["H3588"]], ["הִנֵּה", ["H2009"]]]


def test_fetch_pypi_checks_the_hash_and_extracts_only_data(tmp_path, monkeypatch):
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("pkg/plain_text_bible.py", "bible = 1")
        z.writestr("pkg/__init__.py", "import os")
        z.writestr("../evil.py", "x")
    wheel = buffer.getvalue()
    monkeypatch.setitem(
        sources.PYPI_PACKAGES,
        "t",
        ("a/b/t.whl", hashlib.sha256(wheel).hexdigest(), ("pkg/plain_text_bible.py", "../")),
    )
    monkeypatch.setattr(
        httpx,
        "get",
        lambda *a, **kw: httpx.Response(200, content=wheel, request=httpx.Request("GET", "x")),
    )
    root = sources.fetch_pypi("t", tmp_path)
    files = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file())
    assert files == ["t/.complete", "t/pkg/plain_text_bible.py"]  # no code, nothing outside
    assert sources.fetch_pypi("t", tmp_path) == root  # once
    monkeypatch.setitem(sources.PYPI_PACKAGES, "bad", ("a/b/t.whl", "0" * 64, ("x",)))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        sources.fetch_pypi("bad", tmp_path)


@pytest.fixture(scope="module")
def originals():
    settings = Settings.from_env()
    out = {}
    for v in ("kjv", "wlc", "sblgnt", "etheridge", "ylt", "leeser"):
        out[v] = {(u.book, u.chapter, u.verse_start): u for u in load_units(v, settings)}
    return out, sources.strongs_lexicon(settings.sword_dir)


@pytest.mark.sword
def test_real_originals_and_more_translations(originals):
    texts, lexicon = originals
    assert len(texts["wlc"]) == 23144 and len(texts["sblgnt"]) == 7927
    assert len(texts["ylt"]) == 31102 and len(texts["leeser"]) > 23000
    assert len(texts["etheridge"]) > 7900
    # Hebrew versification -> English: Hebrew Malachi 3:19 = 4:1, Joel 3:1 = 2:28
    plain = lambda v, *ref: strip_marks(texts[v][ref].text)  # noqa: E731 - mark order varies
    assert plain("wlc", "Malachi", 4, 1).startswith("כי הנה היום בא")
    assert "אשפוך את רוחי" in plain("wlc", "Joel", 2, 28)
    assert "εν αρχη ην ο λογος" in plain("sblgnt", "John", 1, 1).lower()
    assert "⸀" not in texts["sblgnt"]["Matthew", 5, 22].text  # apparatus marks removed
    assert "shiul" in texts["etheridge"]["Revelation", 20, 14].text  # Aramaic: Sheol
    assert "to the grave" in texts["leeser"]["Psalms", 16, 10].text
    # one English word, three original words: the KJV's "hell"
    kjv = texts["kjv"]
    assert ["hell", ["H7585"]] in kjv["Psalms", 16, 10].tags
    assert ["hell", ["G86"]] in kjv["Revelation", 20, 14].tags
    assert any(codes == ["G1067"] for _, codes in texts["sblgnt"]["Matthew", 5, 22].tags)
    assert lexicon["H7585"]["kjv"] == "grave, hell, pit" and lexicon["G1067"]["kjv"] == "hell"
    assert lexicon["H5254"]["def"].startswith("A primitive root; to test")  # Genesis 22:1
