"""Exercise the real transformers / sentence-transformers code paths without downloads.

Tiny, randomly initialised models (a Qwen2 causal LM and a BERT encoder) are created
on disk and loaded exactly like the real ones would be, via their directory path.
Answers are gibberish; what is tested is the plumbing: loading, chat templating,
generation, streaming, embedding, indexing and the CLI end to end.
"""

from __future__ import annotations

import pytest

pytestmark = [
    pytest.mark.hf,
    # fixture code uses APIs that exist across sentence-transformers 3.x - 6.x
    pytest.mark.filterwarnings("ignore:Importing from 'sentence_transformers.models'"),
    pytest.mark.filterwarnings("ignore:The `get_word_embedding_dimension`"),
]

WORDS = [
    *("god", "love", "world", "beginning", "heaven", "earth", "lord", "shepherd"),
    *("dios", "amor", "mundo", "principio", "cielos", "tierra", "pastor", "jehová"),
    *("answer", "the", "is", "my", "in", "of", "and", "a"),
]
CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
    "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


@pytest.fixture(scope="session")
def tiny_models(tmp_path_factory):
    import torch
    from sentence_transformers import SentenceTransformer, models
    from tokenizers import Tokenizer, pre_tokenizers
    from tokenizers import models as tok_models
    from transformers import (
        BertConfig,
        BertModel,
        BertTokenizerFast,
        PreTrainedTokenizerFast,
        Qwen2Config,
        Qwen2ForCausalLM,
    )

    torch.manual_seed(0)
    root = tmp_path_factory.mktemp("tiny-models")

    # --- chat LLM -------------------------------------------------------------------
    specials = ["[UNK]", "[PAD]", "<|im_start|>", "<|im_end|>"]
    vocab = {w: i for i, w in enumerate(specials + WORDS)}
    tok = Tokenizer(tok_models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    llm_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="<|im_end|>",
        additional_special_tokens=["<|im_start|>"],
    )
    llm_tok.chat_template = CHAT_TEMPLATE
    cfg = Qwen2Config(
        vocab_size=len(vocab),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        eos_token_id=vocab["<|im_end|>"],
        pad_token_id=vocab["[PAD]"],
    )
    llm_dir = root / "tiny-qwen2"
    Qwen2ForCausalLM(cfg).save_pretrained(llm_dir)
    llm_tok.save_pretrained(llm_dir)

    # --- sentence embedder ----------------------------------------------------------------
    bert_dir = root / "tiny-bert"
    bert_dir.mkdir()
    (bert_dir / "vocab.txt").write_text(
        "\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]) + "\n"
    )
    BertTokenizerFast(str(bert_dir / "vocab.txt")).save_pretrained(bert_dir)
    BertModel(
        BertConfig(
            vocab_size=len(WORDS) + 5,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
        )
    ).save_pretrained(bert_dir)
    transformer = models.Transformer(str(bert_dir), max_seq_length=128)
    pooling = models.Pooling(transformer.get_word_embedding_dimension())
    st_dir = root / "tiny-st"
    SentenceTransformer(modules=[transformer, pooling]).save(str(st_dir))
    return {"llm": str(llm_dir), "embedder": str(st_dir)}


def test_local_chat_model_generate_and_stream(tiny_models):
    from bible_rag.llm import LocalChatModel

    llm = LocalChatModel(tiny_models["llm"], device="cpu", max_new_tokens=6, temperature=0)
    messages = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "god love world"},
    ]
    text = llm.generate(messages)
    assert isinstance(text, str) and "<|im_start|>" not in text
    assert "".join(llm.stream(messages)).strip() == text  # greedy => identical


def test_sentence_transformer_embedder(tiny_models):
    import numpy as np

    from bible_rag.index import SentenceTransformerEmbedder

    emb = SentenceTransformerEmbedder(tiny_models["embedder"], device="cpu")
    vectors = emb.embed(["god love world", "dios amor mundo"])
    assert vectors.shape == (2, 32) and vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_cli_end_to_end_with_real_model_code(tiny_models, tmp_path, monkeypatch, capsys):
    from bible_rag import cli

    from .conftest import PASSAGES

    monkeypatch.setenv("BIBLE_RAG_INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("BIBLE_RAG_EMBEDDING_MODEL", tiny_models["embedder"])
    monkeypatch.setenv("BIBLE_RAG_LLM_MODEL", tiny_models["llm"])
    monkeypatch.setenv("BIBLE_RAG_MAX_NEW_TOKENS", "5")
    monkeypatch.setenv("BIBLE_RAG_DEVICE", "cpu")
    monkeypatch.setattr("bible_rag.ingest.load_version", lambda v, *a, **kw: PASSAGES[v])

    assert cli.main(["ingest"]) == 0
    assert "kjv: 3 passages, embedder hf:" in capsys.readouterr().out

    assert cli.main(["ask", "-k", "1", "god love world"]) == 0
    out = capsys.readouterr().out
    assert "Sources:" in out and "(KJV)" in out and "(RVR1960)" in out
    assert f"Answered by: hf:{tiny_models['llm']}" in out

    assert cli.main(["search", "--json", "--versions", "rvr1960", "pastor"]) == 0
    assert '"version": "RVR1960"' in capsys.readouterr().out


@pytest.fixture(scope="session")
def smollm2_settings(tmp_path_factory):
    from bible_rag.config import Settings

    return Settings(model_cache_dir=tmp_path_factory.mktemp("models"), max_new_tokens=24)


def test_bundled_smollm2_converts_loads_and_answers(smollm2_settings):
    """The real offline fallback LLM: GGUF from the wheel -> transformers -> generated text.

    Only the plumbing is asserted. Answer *quality* is measured by `bible-rag eval` (a 135M
    model gets simple questions wrong, which is why every answer is grounding-checked).
    """
    import time

    from bible_rag.llm import create_chat_model

    model = create_chat_model("smollm2", smollm2_settings, "cpu")
    assert model.label == "smollm2" and model.languages == {"en"} and model.compact
    messages = [
        {"role": "system", "content": "Answer using only the passage. Keep it short."},
        {
            "role": "user",
            "content": "Passage: Genesis 6:14 (KJV): Make thee an ark of gopher "
            "wood.\n\nQuestion: What was the ark made of?",
        },
    ]
    reply = model.generate(messages)
    assert isinstance(reply, str) and len(reply.split()) >= 3
    assert "<|im_start|>" not in reply and "<|im_end|>" not in reply
    assert "".join(model.stream(messages)).strip() == reply  # greedy: stream == generate
    # converted once; the second load reuses the cache
    assert (
        smollm2_settings.model_cache_dir / "SmolLM2-135M-Instruct" / "model.safetensors"
    ).exists()
    t0 = time.perf_counter()
    create_chat_model("smollm2", smollm2_settings, "cpu")
    assert time.perf_counter() - t0 < 20
