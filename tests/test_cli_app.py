"""Terminal commands and the Streamlit chat, driven with the fake pipeline."""

import io
import shutil
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
    assert "Answers: `m0 (loaded when first needed)`" in app.sidebar.caption[0].value
    app.chat_input[0].set_value("Who is my shepherd?").run()
    user, bot = app.chat_message
    assert user.markdown[0].value == "Who is my shepherd?"
    assert bot.markdown[0].value.startswith("According to")
    assert "answered by m0" in bot.expander[0].label and len(bot.expander[0].markdown) == 4
    fake_llm.reply = 'It says "the sheep never sleep at all".'
    app.chat_input[0].set_value("Do sheep sleep?").run()
    assert len(app.chat_message) == 4  # history kept
    assert "the sheep never sleep" in app.chat_message[-1].warning[0].value


def test_app_rebuilds_the_assistant_when_its_code_changes(rag, monkeypatch, tmp_path):
    """A running app must not keep answering with the code it started with: after an
    update the assistant is rebuilt (Streamlit reloads the changed modules themselves)."""
    st.cache_resource.clear()
    shutil.copy(APP, tmp_path / "app.py")  # its code_version() reads tmp_path/bible_rag
    (tmp_path / "bible_rag").mkdir()
    code = tmp_path / "bible_rag" / "agent.py"
    code.write_text("# the assistant, version 1\n")
    built = []

    def from_settings(cls, settings, load_llm=True):
        built.append(settings)
        return rag

    monkeypatch.setattr(BibleRAG, "from_settings", classmethod(from_settings))
    for _ in range(2):  # same code: loaded once, shared by every run and session
        assert not AppTest.from_file(str(tmp_path / "app.py"), default_timeout=30).run().exception
    assert len(built) == 1
    code.write_text("# the assistant, version 2: a new task\n")
    assert not AppTest.from_file(str(tmp_path / "app.py"), default_timeout=30).run().exception
    assert len(built) == 2


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

    def build(settings, progress=None):
        progress("kjv", 512, 1024)  # the page shows the embedding's progress
        calls.append("built")

    monkeypatch.setattr("bible_rag.index.build_indexes", build)
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


def test_app_routes_each_question_itself(app):
    assert not app.sidebar.radio  # no mode to choose: the assistant decides
    app.chat_input[0].set_value("Are there contradictions in the Bible about love?").run()
    bot = app.chat_message[-1]
    assert not app.exception
    assert bot.caption[0].value.startswith(
        "⚖️ Contradiction search: the two opposite statements were searched · love and hate · "
        "_closest to contradiction questions ("
    )  # why it was routed there: closeness to example questions of each kind
    assert bot.expander[0].caption[0].value.startswith("Searched for: “God is love")
    app.chat_input[0].set_value("How do the translations render John 3:16?").run()
    assert app.chat_message[-1].caption[0].value.startswith("🌐 Translation comparison · _closest")
    assert "John 3:16-17 · Juan 3:16-17" in app.chat_message[-1].markdown[0].value


def test_app_buttons_ask_the_same_question_with_another_task(app):
    app.chat_input[0].set_value("Who is my shepherd?").run()
    assert app.chat_message[-1].caption[0].value.startswith("🔎 Direct search")
    # under the last answer: the other tasks
    assert [b.label for b in app.main.button] == [
        "⚖️ Look for contradictions",
        "🌐 Compare translations",
        "🧭 Related concepts",
    ]
    app.main.button[1].click().run()
    user, bot = app.chat_message[-2:]
    assert user.caption[0].value == "Asked as: 🌐 Compare translations"
    assert user.markdown[0].value == "Who is my shepherd?"
    assert bot.caption[0].value == "🌐 Translation comparison · _as requested_"
    assert [b.label for b in app.main.button] == [
        "⚖️ Look for contradictions",
        "🧭 Related concepts",
        "🔎 Direct search",
    ]
    app.main.button[0].click().run()
    assert app.chat_message[-1].caption[0].value.startswith("⚖️ Contradiction search")
    app.main.button[1].click().run()
    assert app.chat_message[-1].caption[0].value.startswith("🧭 Related concepts")
    assert len(app.chat_message) == 8 and not app.exception


def test_app_recognises_a_subject_the_bible_never_names(app):
    app.chat_input[0].set_value("Should I use a cellphone?").run()
    bot = app.chat_message[-1]
    assert not app.exception
    assert bot.caption[0].value.startswith(
        "🧭 Related concepts: the Bible doesn't name this, so its related themes are shown · "
        "cellphone · _not named in the Bible (by meaning"
    )
    assert bot.expander[0].caption[0].value.startswith("Searched for: “")
