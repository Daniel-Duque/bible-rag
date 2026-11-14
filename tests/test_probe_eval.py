import json

from bible_rag import cli, evaluation
from bible_rag.evaluation import QuestionResult, evaluate, summarize
from bible_rag.probe import probe_all

from .conftest import FAKE_EMBEDDER


def test_probe_reports_every_chain_entry(settings, indexes, monkeypatch):
    monkeypatch.setattr(
        "bible_rag.probe.check_hf_model", lambda repo, t: (False, "Hugging Face unreachable")
    )
    settings.embedding_chain = ("hf:some/model", "usem3", FAKE_EMBEDDER)
    settings.llm_chain = (
        "hf:Qwen/Qwen2.5-1.5B-Instruct",
        "ollama",
        "openai",
        "smollm2",
        "extractive",
    )
    settings.ollama_url = "http://127.0.0.1:9"  # nothing listens here
    results = {(r.kind, r.spec): r for r in probe_all(settings)}
    assert not results["embedder", "hf:some/model"].available
    assert results["embedder", "usem3"].available
    assert results["embedder", FAKE_EMBEDDER].index is True
    assert not results["llm", "hf:Qwen/Qwen2.5-1.5B-Instruct"].available
    assert "no server" in results["llm", "ollama"].detail
    assert "not configured" in results["llm", "openai"].detail
    assert results["llm", "smollm2"].available and results["llm", "extractive"].available


def test_models_command(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("BIBLE_RAG_INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("BIBLE_RAG_OLLAMA_URL", "http://127.0.0.1:9")
    monkeypatch.setattr(
        "bible_rag.probe.check_hf_model", lambda repo, t: (False, "Hugging Face unreachable")
    )
    assert cli.main(["models"]) == 0
    out = capsys.readouterr().out
    assert "✓ usem3  <- will be used" in out and "✓ smollm2  <- will be used" in out
    assert "✗ hf:Qwen/Qwen2.5-1.5B-Instruct" in out
    assert cli.main(["models", "--json"]) == 0
    assert {r["spec"] for r in json.loads(capsys.readouterr().out)} >= {"usem3", "smollm2"}


def test_evaluate_scores_retrieval_and_answers(rag, monkeypatch):
    monkeypatch.setattr(
        evaluation,
        "GOLD",
        [
            ("The Lord is my shepherd", "Jehová es mi pastor", [("Psalm", 23)]),
            ("In the beginning God created", "En el principio creó Dios", [("Exodus", 20)]),
        ],
    )
    report = evaluate(rag, k=1)
    s = report["summary"]
    assert s["questions"] == 4
    assert s["retrieval_hit_rate"]["en->KJV"] == 0.5  # psalm found, creation has wrong gold
    assert s["answers"]["models"] == {"m0": 4}
    first = report["questions"][0]
    assert first["retrieval_hit"]["KJV"] is True and first["cites_gold"] is True


def test_summarize_handles_retrieval_only():
    r = QuestionResult("q", "en", {"KJV": True, "RVR1960": False}, [])
    s = summarize([r])
    assert s["retrieval_hit_rate"] == {"en->KJV": 1.0, "en->RVR1960": 0.0, "en->any": 1.0}
    assert "answers" not in s


def test_gold_set_names_are_canonical():
    for _, _, gold in evaluation.GOLD:
        assert evaluation.gold_set(gold)  # KeyError if a book name is misspelled
