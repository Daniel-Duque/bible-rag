"""Automatic model fallback: chain order, runtime failover, language limits, connectors."""

import json

import httpx
import pytest

from bible_rag.hub import ModelUnavailable
from bible_rag.llm import (
    ModelChain,
    bundled_smollm2_gguf,
    connect_ollama,
    connect_openai,
    create_chat_model,
)

from .conftest import FakeLLM, make_chain


def test_chain_uses_first_model_that_loads(fake_llm):
    chain = make_chain(ModelUnavailable("Hugging Face unreachable"), OSError("no memory"), fake_llm)
    assert chain.model is fake_llm
    assert chain.active_spec == "m2" and chain.is_fallback
    assert [s for s, _ in chain.skipped] == ["m0", "m1"]
    assert "Hugging Face unreachable" in chain.skipped[0][1]
    assert chain.status()["active"] == "m2"


def test_chain_exhausted_raises_with_reasons():
    chain = make_chain(ModelUnavailable("a"), ModelUnavailable("b"))
    with pytest.raises(RuntimeError, match="m0 -> ModelUnavailable: a; m1 -> ModelUnavailable: b"):
        chain.model  # noqa: B018


def test_runtime_failure_fails_over_before_first_token(rag, fake_llm):
    broken = FakeLLM(fail=True)
    rag.chain = make_chain(broken, fake_llm)
    answer = rag.ask("In the beginning", k=1)
    assert answer.answer.startswith("According to") and answer.model == "m1"
    assert any("m1" in n and "model server went away" in n for n in answer.notes)
    # streaming takes the same path
    rag.chain = make_chain(FakeLLM(fail=True), fake_llm)
    result = rag.ask_stream("In the beginning", k=1)
    assert "".join(result).startswith("According to") and result.model == "m1"


def test_extractive_is_the_last_resort(rag):
    rag.chain = make_chain(ModelUnavailable("offline"), None)  # None = extractive
    answer = rag.ask("¿Jehová es mi pastor?", k=1)
    assert answer.model == "m1"
    assert answer.answer.startswith("No hay un modelo de lenguaje disponible")
    assert "Salmos 23:1-2 (RVR1960): “Jehová es mi pastor" in answer.answer
    assert answer.grounding.grounded  # verbatim quotes + real references


def test_english_only_model_gets_extractive_answer_in_spanish(rag):
    small = FakeLLM()
    small.languages, small.label, small.compact = frozenset({"en"}), "smollm2", True
    rag.chain = make_chain(small)
    es = rag.ask("¿Quién es mi pastor?", k=1)
    assert "smollm2" in es.answer and "no responde en español" in es.answer
    assert es.model == "extractive" and "'smollm2' does not support 'es'" in es.notes[-1]
    assert not any(n.startswith("Answered by") for n in es.notes)
    assert small.calls == []
    en = rag.ask("Who is my shepherd?", k=2)
    assert en.answer.startswith("According to")
    # compact prompt, only English passages are given to the small model
    system, user = small.calls[-1]
    assert "Keep it short" in system["content"]
    assert user["content"].startswith("Passages:\nPsalm 23:1-2 (KJV): The Lord is my shepherd")
    assert "(RVR1960)" not in user["content"]


# ------------------------------------------------------------------------------ connectors
def ollama_transport(models=("qwen2.5:3b",), fail_chat=False):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": m} for m in models]})
        if request.url.path == "/v1/chat/completions":
            if fail_chat:
                return httpx.Response(500, json={"error": "boom"})
            body = json.loads(request.content)
            assert body["model"] == models[0] and body["messages"][0]["role"] == "user"
            if body["stream"]:
                chunks = [{"choices": [{"delta": {"content": t}}]} for t in ("Hel", "lo")]
                sse = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
                return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})
            return httpx.Response(200, json={"choices": [{"message": {"content": " Hello "}}]})
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_ollama_connector_generate_and_stream(settings):
    model = connect_ollama(settings, transport=ollama_transport())
    assert model.label == "ollama:qwen2.5:3b" and model.languages is None
    messages = [{"role": "user", "content": "hi"}]
    assert model.generate(messages) == "Hello"
    assert "".join(model.stream(messages)) == "Hello"


def test_ollama_connector_errors(settings):
    with pytest.raises(ModelUnavailable, match="no models"):
        connect_ollama(settings, transport=ollama_transport(models=()))
    with pytest.raises(ModelUnavailable, match="no model 'llama3'"):
        connect_ollama(settings, "llama3", transport=ollama_transport())
    refused = httpx.MockTransport(lambda r: (_ for _ in ()).throw(httpx.ConnectError("refused")))
    with pytest.raises(ModelUnavailable, match="no server"):
        connect_ollama(settings, transport=refused)
    model = connect_ollama(settings, transport=ollama_transport(fail_chat=True))
    with pytest.raises(httpx.HTTPStatusError):  # -> ModelChain.failover at runtime
        model.generate([{"role": "user", "content": "x"}])


def test_openai_compatible_connector(settings):
    with pytest.raises(ModelUnavailable, match="not configured"):
        connect_openai(settings)

    def handler(request):
        assert request.headers["authorization"] == "Bearer secret"
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "local-model"}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    settings.openai_base_url, settings.openai_api_key = "http://lmstudio:1234/v1", "secret"
    model = connect_openai(settings, transport=httpx.MockTransport(handler))
    assert model.label == "openai:local-model"
    assert model.generate([{"role": "user", "content": "x"}]) == "ok"


def test_create_chat_model_specs(settings):
    assert create_chat_model("extractive", settings, "cpu") is None
    with pytest.raises(ValueError, match="Unknown LLM spec"):
        create_chat_model("gpt-9", settings, "cpu")


def test_default_chains(settings):
    settings.embedding_chain = ()
    assert settings.embedding_specs == [f"hf:{settings.embedding_model}", "usem3"]
    assert settings.llm_specs == [
        f"hf:{settings.llm_model}",
        "ollama",
        "openai",
        "smollm2",
        "extractive",
    ]


def test_bundled_smollm2_is_installed():
    """The offline fallback LLM ships with `uv sync` (fallback-llm dependency group)."""
    path = bundled_smollm2_gguf()
    assert path.name == "SmolLM2-135M-Instruct.Q4_1.gguf" and path.stat().st_size > 90e6


def test_hf_model_fails_fast_when_offline(monkeypatch):
    from bible_rag import hub

    monkeypatch.setattr(hub, "is_cached", lambda repo: False)
    monkeypatch.setattr(hub, "hub_reachable", lambda repo, t: (False, "unreachable"))
    with pytest.raises(ModelUnavailable, match="unreachable"):
        hub.prepare_hf_model("org/model", 1.0)
    monkeypatch.setattr(hub, "is_cached", lambda repo: True)
    assert hub.prepare_hf_model("org/model", 1.0) is True  # cached -> local_files_only


def test_chain_is_used_by_from_settings(settings, indexes, monkeypatch, fake_llm):
    from bible_rag.rag import BibleRAG

    monkeypatch.setattr(
        "bible_rag.rag.load_embedder",
        lambda *a, **kw: __import__("tests.conftest", fromlist=["FakeEmbedder"]).FakeEmbedder(),
    )

    def create(spec, settings, device):
        if spec.startswith("hf:"):
            raise ModelUnavailable("Hugging Face unreachable")
        return fake_llm

    monkeypatch.setattr("bible_rag.rag.create_chat_model", create)
    settings.llm_chain = ("hf:Qwen/Qwen2.5-1.5B-Instruct", "smollm2")
    rag = BibleRAG.from_settings(settings)
    assert isinstance(rag.chain, ModelChain) and rag.chain.active_spec == "smollm2"
    answer = rag.ask("In the beginning God created", k=1)
    assert answer.model == "smollm2"
    assert "Answered by fallback model 'smollm2'" in answer.notes[0]
