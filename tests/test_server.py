"""The read-only API (`bible-rag serve`) and its client, with the fake pipeline: safety checks,
validation, limits, the streamed answer, and the fallback to the local models."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from bible_rag import cli
from bible_rag.remote import RemoteAnswer, RemoteRAG, RemoteUnavailable, check_url, open_backend
from bible_rag.server import MAX_QUESTION, ServerConfigError, check_exposure, create_app
from bible_rag.wire import PROTOCOL

KEY = "k" * 32


@pytest.fixture
def app(rag):
    return create_app(rag, api_key=KEY, rate=1000, concurrency=2)


@pytest.fixture
def http(app):
    return TestClient(app, headers={"Authorization": f"Bearer {KEY}"})


def remote(settings, http, local=None):
    settings.server_url, settings.api_key = "https://testserver", KEY
    return RemoteRAG(settings, client=http, local=local)


# ------------------------------------------------------------------------------ server
def test_exposure_rules():
    check_exposure("127.0.0.1", "", ["kjv"])  # local: no key needed
    check_exposure("localhost", "", ["kjv", "rvr1960"])  # restricted text, privately: fine
    with pytest.raises(ServerConfigError, match="needs an API key"):
        check_exposure("0.0.0.0", "", ["kjv"])
    with pytest.raises(ServerConfigError, match="too short"):
        check_exposure("0.0.0.0", "secret", ["kjv"])
    with pytest.raises(ServerConfigError, match="may not be served"):
        check_exposure("0.0.0.0", KEY, ["kjv", "rvr1960"])
    check_exposure("0.0.0.0", KEY, ["kjv", "wlc", "sblgnt"])


def test_auth_and_no_dangerous_routes(app, http):
    anonymous = TestClient(app)
    assert anonymous.get("/v1/health").json()["status"] == "ok"  # no secrets in it
    assert anonymous.get("/v1/info").status_code == 401
    wrong = TestClient(app, headers={"Authorization": "Bearer nope"})
    assert wrong.post("/v1/ask", json={"question": "x"}).status_code == 401
    info = http.get("/v1/info").json()
    assert [v["id"] for v in info["versions"]] == ["kjv", "rv1909"] and info["protocol"] == PROTOCOL
    for path in ("/v1/ingest", "/ingest", "/v1/models", "/docs", "/openapi.json", "/v1/files"):
        assert http.post(path, json={}).status_code in (404, 405)
        assert http.get(path).status_code in (404, 405)
    r = http.get("/v1/health")
    assert r.headers["x-content-type-options"] == "nosniff" and "server" not in r.headers


def test_validation_and_limits(rag, http):
    bad = [
        {"question": ""},
        {"question": "x" * (MAX_QUESTION + 1)},
        {"question": "x", "k": 99},
        {"question": "x", "versions": ["niv"]},
        {"question": "x", "task": "debate"},
        {"question": "x", "path": "/etc/passwd"},  # unknown fields are refused
    ]
    for body in bad:
        assert http.post("/v1/ask", json=body).status_code == 422, body
    big = json.dumps({"question": "x" * 20_000})
    assert (
        http.post("/v1/ask", content=big, headers={"content-type": "application/json"}).status_code
        == 413
    )
    chunked = http.post(
        "/v1/search",
        content=iter([b'{"question": "x"}']),
        headers={"content-type": "application/json"},
    )
    assert chunked.status_code == 411  # no unbounded uploads
    limited = TestClient(
        create_app(rag, api_key=KEY, rate=2), headers={"Authorization": f"Bearer {KEY}"}
    )
    codes = [limited.get("/v1/info").status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_busy_server_says_so(rag):
    busy = TestClient(
        create_app(rag, api_key=KEY, concurrency=1), headers={"Authorization": f"Bearer {KEY}"}
    )
    with busy.stream("POST", "/v1/ask", json={"question": "In the beginning"}) as first:
        assert first.status_code == 200
        second = busy.post("/v1/ask", json={"question": "In the beginning"})
        assert second.status_code == 503 and "busy" in second.json()["detail"]
        list(first.iter_lines())
    assert busy.post("/v1/ask", json={"question": "In the beginning"}).status_code == 200


def test_ask_streams_plan_tokens_and_answer(http):
    lines = [
        json.loads(x)
        for x in http.post("/v1/ask", json={"question": "In the beginning"}).iter_lines()
        if x
    ]
    assert [m["type"] for m in lines[:2]] == ["plan", "token"] and lines[-1]["type"] == "done"
    assert lines[0]["plan"]["intent"] == "general"
    answer = lines[-1]["answer"]
    asked = http.post("/v1/ask", json={"question": "In the beginning", "task": "translation"})
    plan = json.loads(next(asked.iter_lines()))["plan"]
    assert plan["intent"] == "translation" and plan["reason"] == "as requested"
    asked = http.post("/v1/ask", json={"question": "Should I use a cellphone?", "task": "related"})
    plan = json.loads(next(asked.iter_lines()))["plan"]
    assert plan["intent"] == "related" and plan["topic_name"] == "cellphone"
    assert answer["text"].startswith("According to") and answer["grounding"]["references"]
    assert "".join(m["text"] for m in lines if m["type"] == "token").strip() == answer["text"]


def test_search_endpoint(http):
    sources = http.post("/v1/search", json={"question": "my shepherd", "k": 1}).json()["sources"]
    assert {s["version"] for s in sources} == {"KJV", "RV1909"}


# ------------------------------------------------------------------------------ client
def test_client_answers_like_the_local_pipeline(settings, http, rag):
    client = remote(settings, http)
    assert list(client.indexes) == ["kjv", "rv1909"] and "@ testserver" in client.chain.active
    answer = client.ask_stream("In the beginning", 1)
    assert isinstance(answer, RemoteAnswer) and answer.plan.intent == "general"
    streamed = "".join(answer)
    local = rag.ask("In the beginning", 1)
    assert streamed.strip() == answer.text == local.text
    assert [h.passage.citation for h in answer.sources] == [
        h.passage.citation for h in local.sources
    ]
    assert answer.grounding.grounded and answer.model == "m0 @ testserver"
    assert any("Answered by the server testserver" in n for n in answer.notes)
    assert [h.passage.citation for h in client.search("my shepherd", 1)][0].endswith("(KJV)")
    asked = client.ask_stream("In the beginning", 1, task="translation")
    assert asked.plan.intent == "translation" and asked.plan.reason == "as requested"
    "".join(asked)
    asked = client.ask_stream("Should I use a cellphone?", 1)  # routed by the server
    assert asked.plan.intent == "related" and asked.plan.topic_name == "cellphone"
    "".join(asked)
    with pytest.raises(ValueError):
        client.ask_stream("x" * (MAX_QUESTION + 1))


def test_client_falls_back_to_local_models(settings, rag, app):
    def down(request):
        raise httpx.ConnectError("connection refused")

    # unreachable at start: the local models, with a note
    settings.server_url, settings.api_key = "http://127.0.0.1:9", KEY
    opened = []
    monkey = pytest.MonkeyPatch()
    monkey.setattr(
        "bible_rag.rag.BibleRAG.from_settings",
        classmethod(lambda cls, s, load_llm=True: opened.append(1) or rag),
    )
    try:
        backend = open_backend(settings)
        assert backend is rag and "unavailable" in rag.notes[-1] and opened
        # the key refused: same
        wrong = TestClient(app, headers={"Authorization": "Bearer wrong"})
        with pytest.raises(RemoteUnavailable, match="refused the API key"):
            RemoteRAG(settings, client=wrong)
    finally:
        monkey.undo()

    # reachable at start, then failing on a question: that question is answered locally
    http = TestClient(app, headers={"Authorization": f"Bearer {KEY}"})
    client = remote(settings, http, local=lambda: rag)
    client.client = httpx.Client(base_url="http://x", transport=httpx.MockTransport(down))
    answer = client.ask_stream("In the beginning", 1)
    text = "".join(answer)
    assert text.startswith("According to") and "answered locally" in answer.notes[-1]
    busy = httpx.MockTransport(lambda r: httpx.Response(503, json={"detail": "busy"}))
    client.client = httpx.Client(base_url="http://x", transport=busy)
    answer = client.ask_stream("In the beginning", 1)
    assert "".join(answer) and "HTTP 503: busy" in answer.notes[-1]
    cut = httpx.MockTransport(
        lambda r: httpx.Response(
            200, content=b'{"type": "plan", "plan": {"intent": "general", "language": "en"}}\n'
        )
    )
    client.client = httpx.Client(base_url="http://x", transport=cut)
    answer = client.ask_stream("In the beginning", 1)
    assert "".join(answer).startswith("According to")  # cut before the first word: local
    assert answer.text.startswith("According to") and "cut off" in answer.notes[-1]


def test_client_never_leaks_the_key():
    assert check_url("https://gpu.example.org/", KEY) == "https://gpu.example.org"
    assert check_url("http://127.0.0.1:8000", KEY)  # an SSH tunnel
    with pytest.raises(RemoteUnavailable, match="clear text"):
        check_url("http://gpu.example.org:8000", KEY)
    assert check_url("http://gpu.example.org:8000", KEY, insecure=True)
    with pytest.raises(RemoteUnavailable, match="not an http"):
        check_url("file:///etc/passwd", KEY)


def test_cli_uses_the_server(settings, http, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load", lambda s, llm=True: remote(settings, http))
    assert cli.main(["ask", "-k", "1", "In", "the", "beginning"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("According to") and "Answered by: m0 @ testserver" in out


def test_serve_command_refuses_unsafe_setups(monkeypatch):
    monkeypatch.delenv("BIBLE_RAG_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="needs an API key"):
        cli.main(["serve", "--host", "0.0.0.0"])
