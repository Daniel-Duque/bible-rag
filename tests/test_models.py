"""Model loading: local servers (mocked HTTP), offline detection, probe, and (marker `hf`)
the real transformers / sentence-transformers code on tiny local models + bundled SmolLM2."""

import json

import httpx
import pytest

from bible_rag import models
from bible_rag.models import ModelUnavailable, OpenAIChatModel, create_chat_model, probe


def server(models_listed=("qwen2.5:3b",), fail_chat=False):
    """Mock of an OpenAI-compatible server (what Ollama exposes under /v1)."""

    def handle(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": m} for m in models_listed]})
        if fail_chat:
            return httpx.Response(500)
        body = json.loads(request.content)
        if body["stream"]:
            sse = (
                "".join(
                    f"data: {json.dumps({'choices': [{'delta': {'content': t}}]})}\n\n"
                    for t in ("Hel", "lo")
                )
                + "data: [DONE]\n\n"
            )
            return httpx.Response(200, text=sse)
        return httpx.Response(200, json={"choices": [{"message": {"content": " Hello "}}]})

    return httpx.MockTransport(handle)


def test_openai_compatible_server(settings):
    m = OpenAIChatModel("http://x/v1", settings, kind="ollama", transport=server())
    assert m.label == "ollama:qwen2.5:3b"  # first listed model by default
    msgs = [{"role": "user", "content": "hi"}]
    assert m.generate(msgs) == "Hello" and "".join(m.stream(msgs)) == "Hello"
    with pytest.raises(ModelUnavailable, match="has no models"):
        OpenAIChatModel("http://x/v1", settings, transport=server(()))
    with pytest.raises(ModelUnavailable, match="not on the server"):
        OpenAIChatModel("http://x/v1", settings, "llama3", transport=server())
    broken = OpenAIChatModel("http://x/v1", settings, transport=server(fail_chat=True))
    with pytest.raises(httpx.HTTPStatusError):  # -> ModelChain.failover
        broken.generate(msgs)


def test_unreachable_models_fail_fast(settings, monkeypatch):
    settings.ollama_url = "http://127.0.0.1:9/v1"  # nothing listens there
    with pytest.raises(ModelUnavailable, match="no server"):
        create_chat_model("ollama", settings, "cpu")
    with pytest.raises(ModelUnavailable, match="not configured"):
        create_chat_model("openai", settings, "cpu")
    monkeypatch.setattr(models, "hf_status", lambda repo, t: (False, False, "unreachable"))
    with pytest.raises(ModelUnavailable, match="unreachable"):
        create_chat_model("hf:org/model", settings, "cpu")
    assert create_chat_model("extractive", settings, "cpu") is None
    with pytest.raises(ValueError, match="Unknown LLM"):
        create_chat_model("gpt-9", settings, "cpu")


def test_probe_and_bundled_fallbacks(settings, monkeypatch):
    monkeypatch.setattr(models, "hf_status", lambda repo, t: (False, False, "unreachable"))
    settings.embedding_chain = ("hf:some/model", "usem3")
    settings.llm_chain = ("hf:Qwen/Qwen2.5-1.5B-Instruct", "smollm2", "extractive")
    rows = {r["spec"]: r for r in probe(settings)}
    assert not rows["hf:some/model"]["ok"] and rows["usem3"]["ok"]
    assert rows["smollm2"]["ok"] and rows["extractive"]["ok"]
    assert models.smollm2_gguf().stat().st_size > 90e6  # shipped by `uv sync`


# ---------------------------------------------------- real model code, no downloads (hf)
WORDS = ["god", "love", "world", "beginning", "dios", "amor", "mundo", "principio", "the"]
TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
    "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


@pytest.fixture(scope="session")
def tiny(tmp_path_factory):
    """Random-weight Qwen2 chat model and BERT sentence embedder, saved like real ones."""
    import torch
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.modules import Pooling, Transformer
    from tokenizers import Tokenizer, pre_tokenizers
    from tokenizers import models as tm
    from transformers import (
        BertConfig,
        BertModel,
        BertTokenizerFast,
        PreTrainedTokenizerFast,
        Qwen2Config,
        Qwen2ForCausalLM,
    )

    torch.manual_seed(0)
    root = tmp_path_factory.mktemp("tiny")
    vocab = {w: i for i, w in enumerate(["[UNK]", "[PAD]", "<|im_start|>", "<|im_end|>", *WORDS])}
    tok = Tokenizer(tm.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="<|im_end|>",
        additional_special_tokens=["<|im_start|>"],
    )
    tok.chat_template = TEMPLATE
    Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=len(vocab),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
    ).save_pretrained(root / "llm")
    tok.save_pretrained(root / "llm")
    (root / "bert").mkdir()
    (root / "bert" / "vocab.txt").write_text(
        "\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS])
    )
    BertTokenizerFast(str(root / "bert" / "vocab.txt")).save_pretrained(root / "bert")
    BertModel(
        BertConfig(
            vocab_size=len(WORDS) + 5,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
        )
    ).save_pretrained(root / "bert")
    t = Transformer(str(root / "bert"))
    SentenceTransformer(modules=[t, Pooling(t.get_embedding_dimension())]).save(str(root / "st"))
    return {"llm": str(root / "llm"), "embedder": str(root / "st")}


@pytest.mark.hf
def test_transformers_chat_and_embedder(tiny, settings):
    settings.max_new_tokens, settings.temperature = 5, 0
    llm = create_chat_model(f"hf:{tiny['llm']}", settings, "cpu")
    msgs = [{"role": "user", "content": "god love world"}]
    assert "<|im_start|>" not in llm.generate(msgs)
    assert "".join(llm.stream(msgs)).strip() == llm.generate(msgs)  # greedy: stable
    vectors = models.load_embedder(f"hf:{tiny['embedder']}").embed(["god love", "dios amor"])
    assert vectors.shape == (2, 32) and abs(float(vectors[0] @ vectors[0]) - 1) < 1e-5


@pytest.mark.hf
def test_bundled_smollm2_runs(settings):
    """Plumbing only: quality is measured by `bible-rag eval` (it is weak)."""
    settings.max_new_tokens = 16
    m = create_chat_model("smollm2", settings, "cpu")
    assert m.label == "smollm2" and m.languages == {"en"} and m.compact
    reply = m.generate([{"role": "user", "content": "Who built the ark?"}])
    assert len(reply.split()) >= 3 and "<|im_end|>" not in reply
    assert (settings.model_cache_dir / "SmolLM2-135M-Instruct" / "model.safetensors").exists()
