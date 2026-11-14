"""Terminal commands and the Streamlit chat, driven with the fake pipeline."""

import io
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from bible_rag import cli
from bible_rag.rag import BibleRAG

APP = str(Path(__file__).parent.parent / "app.py")


@pytest.fixture
def run(rag, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load", lambda settings, llm=True: rag)

    def run(*argv):
        return cli.main(list(argv)), capsys.readouterr().out

    return run


def test_ask_prints_answer_sources_and_model(run):
    code, out = run("ask", "-k", "1", "In", "the", "beginning")
    assert code == 0 and out.startswith("According to ")
    assert "Sources:" in out and "(KJV)" in out and "Answered by: m0" in out and "⚠" not in out


def test_unverified_answer_shows_warning_and_evidence(run, fake_llm):
    fake_llm.reply = 'See 1 Cor. 13:4: "charity suffereth long and is kind".'
    code, out = run("ask", "-k", "1", "what is love")
    assert "⚠ Not found in the retrieved passages" in out and "1 Cor. 13:4" in out
    assert sum(line.startswith("      ") for line in out.splitlines()) == 2  # passage texts


def test_search_stdin_versions_and_chat(run, monkeypatch):
    code, out = run("search", "--versions", "kjv", "good", "shepherd")
    assert "Psalms 23:1-2 (KJV)" in out and "RV1909" not in out
    monkeypatch.setattr("sys.stdin", io.StringIO("In the beginning\n"))
    assert "Genesis 1:1-2" in run("search")[1]
    inputs = iter(["/k 1", "Who is my shepherd?", "/search pastor", "/nope", "/quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    code, out = run("chat")
    assert code == 0 and "According to" in out and "Commands:" in out
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["ask", "--versions", "niv", "x"])


def test_errors_are_reported_cleanly(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BIBLE_RAG_INDEX_DIR", str(tmp_path))
    assert cli.main(["search", "love"]) == 1
    assert "bible-rag ingest" in capsys.readouterr().err


@pytest.fixture
def app(rag, monkeypatch):
    st.cache_resource.clear()
    monkeypatch.setattr(BibleRAG, "from_settings", classmethod(lambda cls, s, load_llm=True: rag))
    return AppTest.from_file(APP, default_timeout=30).run()


def test_app_chat(app, fake_llm):
    assert not app.exception and app.title[0].value == "📖 Bible RAG"
    assert "Answers: `m0`" in app.sidebar.caption[0].value
    app.chat_input[0].set_value("Who is my shepherd?").run()
    user, bot = app.chat_message
    assert user.markdown[0].value == "Who is my shepherd?"
    assert bot.markdown[0].value.startswith("According to")
    assert "answered by m0" in bot.expander[0].label and len(bot.expander[0].markdown) == 4
    fake_llm.reply = 'It says "the sheep never sleep at all".'
    app.chat_input[0].set_value("Do sheep sleep?").run()
    assert len(app.chat_message) == 4  # history kept
    assert "the sheep never sleep" in app.chat_message[-1].warning[0].value


def test_app_builds_missing_index_on_first_start(rag, monkeypatch):
    from bible_rag.index import IndexNotFoundError

    st.cache_resource.clear()
    calls = []

    def from_settings(cls, settings, load_llm=True):
        if not calls:
            calls.append("missing")
            raise IndexNotFoundError("no index")
        return rag

    monkeypatch.setattr(BibleRAG, "from_settings", classmethod(from_settings))
    monkeypatch.setattr("bible_rag.index.build_indexes", lambda s: calls.append("built"))
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception and calls == ["missing", "built"]


def test_versions_report_and_license_refusal(monkeypatch, capsys, tmp_path):
    assert cli.main(["versions"]) == 0
    out = capsys.readouterr().out
    assert "✓ rv1909" in out and "✗ rvr1960" in out and "NOT for public use" in out
    monkeypatch.setenv("BIBLE_RAG_VERSIONS", "kjv,rvr1960")
    monkeypatch.setenv("BIBLE_RAG_INDEX_DIR", str(tmp_path))
    assert cli.main(["ingest"]) == 1
    assert "BIBLE_RAG_ALLOW_RESTRICTED" in capsys.readouterr().err
