import json

import pytest
from fastapi.testclient import TestClient

from bible_rag import cli
from bible_rag.server import create_app


@pytest.fixture
def client(rag):
    return TestClient(create_app(rag))


def test_health(client):
    data = client.get("/health").json()
    assert data["status"] == "ok" and data["versions"]["kjv"]["passages"] == 3


def test_search_and_ask_endpoints(client):
    r = client.post("/search", json={"question": "Jehová es mi pastor", "k": 1})
    assert r.status_code == 200 and len(r.json()["sources"]) == 2
    r = client.post("/ask", json={"question": "Who created the earth?", "versions": ["kjv"]})
    body = r.json()
    assert r.status_code == 200 and body["language"] == "en"
    assert [s["version"] for s in body["sources"]] == ["KJV", "KJV"]
    assert body["model"] == "m0" and body["grounding"]["grounded"] is True


def test_endpoint_errors(client, rag):
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/search", json={"question": "x", "versions": ["niv"]}).status_code == 400
    rag.chain = None
    assert client.post("/ask", json={"question": "x"}).status_code == 503


# ------------------------------------------------------------------------------ CLI
class FakeBackendFactory:
    """Patch make_backend so the CLI runs against the fake in-memory RAG."""

    def __init__(self, rag):
        self.rag = rag

    def __call__(self, args, settings, load_llm):
        backend = cli.LocalBackend.__new__(cli.LocalBackend)
        backend.rag = self.rag
        return backend


@pytest.fixture
def run_cli(rag, monkeypatch, capsys):
    monkeypatch.setattr(cli, "make_backend", FakeBackendFactory(rag))

    def run(*argv):
        code = cli.main(list(argv))
        return code, capsys.readouterr()

    return run


def test_cli_ask_streams_answer_and_sources(run_cli):
    code, out = run_cli("ask", "What", "is", "the", "beginning?", "-k", "1")
    assert code == 0
    assert out.out.startswith("According to ")
    assert "Sources:" in out.out and "(KJV)" in out.out and "(RVR1960)" in out.out
    assert "Answered by: m0" in out.out and "⚠" not in out.out


def test_cli_flags_unverified_claims_and_shows_evidence(run_cli, fake_llm, monkeypatch):
    monkeypatch.setattr(
        fake_llm, "_reply", lambda m: 'See 1 Cor. 13:4: "charity suffereth long and is kind".'
    )
    code, out = run_cli("ask", "-k", "1", "what is love")
    assert code == 0
    assert "⚠ Not found in the retrieved passages" in out.out and "1 Cor. 13:4" in out.out
    # the retrieved passages are printed verbatim (lines indented under each source)
    passage_lines = [ln for ln in out.out.splitlines() if ln.startswith("      ")]
    assert len(passage_lines) == 2


def test_cli_json_and_search(run_cli):
    code, out = run_cli("ask", "--json", "Dios amó al mundo")
    data = json.loads(out.out)
    assert code == 0 and data["answer"].startswith("According to") and len(data["sources"]) == 4
    assert data["model"] == "m0" and data["grounding"]["grounded"] is True
    code, out = run_cli("search", "--versions", "kjv", "shepherd")
    assert code == 0 and "Psalm 23:1-2 (KJV)" in out.out and "RVR1960" not in out.out


def test_cli_reads_question_from_stdin(run_cli, monkeypatch):
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO("In the beginning\n"))
    code, out = run_cli("search")
    assert code == 0 and "In the beginning" in out.out


def test_cli_chat_session(run_cli, monkeypatch):
    inputs = iter(["/k 1", "Who is my shepherd?", "/search pastor", "/nope", "/quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    code, out = run_cli("chat")
    assert code == 0
    assert "k = 1" in out.out and "According to" in out.out and "Unknown command" in out.out


def test_cli_remote_backend_against_service(rag, capsys):
    backend = cli.RemoteBackend("http://testserver", client=TestClient(create_app(rag)))
    args = cli.build_parser().parse_args(["ask", "x", "-k", "1"])
    cli.run_ask(backend, "Who created the heaven?", args)
    out = capsys.readouterr().out
    assert "According to" in out and "Sources:" in out and "Answered by: m0" in out


def test_cli_reports_missing_index(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BIBLE_RAG_INDEX_DIR", str(tmp_path / "nothing"))
    monkeypatch.setattr("bible_rag.rag.load_embedder", None)  # never reached: no index
    assert cli.main(["search", "love"]) == 1
    assert "bible-rag ingest" in capsys.readouterr().err


def test_cli_versions_option_parsing():
    parser = cli.build_parser()
    args = parser.parse_args(["search", "--versions", "kjv", "good", "shepherd"])
    assert args.versions == ["kjv"] and args.question == ["good", "shepherd"]
    assert parser.parse_args(["ask", "--versions", "KJV,rvr1960", "x"]).versions == [
        "kjv",
        "rvr1960",
    ]
    with pytest.raises(SystemExit):
        parser.parse_args(["ask", "--versions", "niv", "x"])


def test_cli_reports_model_download_errors(monkeypatch, capsys):
    def boom(*a, **kw):
        raise OSError("We couldn't connect to 'https://huggingface.co'")

    monkeypatch.setattr("bible_rag.ingest.ingest", boom)
    assert cli.main(["ingest"]) == 1
    err = capsys.readouterr().err
    assert "couldn't connect" in err and "BIBLE_RAG_*_MODEL" in err and "Traceback" not in err


def test_cli_answer_without_citations_shows_the_evidence(run_cli, fake_llm, monkeypatch):
    monkeypatch.setattr(fake_llm, "_reply", lambda m: "God made everything.")
    code, out = run_cli("ask", "-k", "1", "In the beginning")
    assert code == 0 and "cites no verse that could be checked" in out.out
    assert len([ln for ln in out.out.splitlines() if ln.startswith("      ")]) == 2
