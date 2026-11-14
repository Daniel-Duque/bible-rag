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
    assert all(v.license == "Public Domain" for k, v in VERSIONS.items() if v.public)


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
