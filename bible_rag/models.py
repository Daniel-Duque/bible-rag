"""Embedders, chat models and the fallback chains.

Chains are tried in order; the first entry that loads is used (`bible-rag models` shows which).
  embeddings: hf:<model>      sentence-transformers model (cached or downloadable)
              usem3           Universal Sentence Encoder multilingual, bundled, offline
  LLM:        hf:<model>      transformers chat model
              ollama[:model]  local Ollama server (OpenAI API), first installed model by default
              openai[:model]  any OpenAI-compatible server (BIBLE_RAG_OPENAI_BASE_URL)
              smollm2         SmolLM2-135M-Instruct, bundled (llm-smollm2 wheel), English only
              extractive      no LLM: quote the retrieved passages
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import shutil
import threading
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class ModelUnavailable(RuntimeError):
    """This chain entry can't be used here."""


def describe_error(e: BaseException) -> str:
    first = str(e).strip().splitlines()[0] if str(e).strip() else ""
    return f"{type(e).__name__}: {first}"[:200]


# ------------------------------------------------------------------------- Hugging Face
def hf_status(repo: str, timeout: float) -> tuple[bool, bool, str]:
    """(available, local_only, detail) without downloading. Checking first avoids the minute
    of download retries huggingface_hub does when offline; cached models load offline."""
    from huggingface_hub import HfApi, try_to_load_from_cache

    if Path(repo).expanduser().is_dir():
        return True, True, "local directory"
    if any(
        isinstance(try_to_load_from_cache(repo, f), str) for f in ("config.json", "modules.json")
    ):
        return True, True, "in the local Hugging Face cache"
    try:
        HfApi().model_info(repo, timeout=timeout)
        return True, False, "downloadable from Hugging Face"
    except Exception as e:  # noqa: BLE001
        return False, False, f"not cached and Hugging Face unreachable ({describe_error(e)})"


def _hf_local_only(repo: str, timeout: float) -> bool:
    ok, local_only, detail = hf_status(repo, timeout)
    if not ok:
        raise ModelUnavailable(detail)
    return local_only


# ---------------------------------------------------------------------------- embedders
class STEmbedder:
    def __init__(self, name: str, device: str, local_only: bool):
        from sentence_transformers import SentenceTransformer

        self.model_name = name
        self._m = SentenceTransformer(name, device=device, local_files_only=local_only)

    def embed(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        return self._m.encode(
            texts, batch_size=64, normalize_embeddings=True, show_progress_bar=show_progress
        ).astype("float32")


class USEEmbedder:
    """Google USE-multilingual v3 (pure numpy, weights in the wheel): EN/ES share one space."""

    model_name = "usem3"

    def __init__(self):
        from usem3 import USE

        self._m = USE()

    def embed(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        from tqdm import tqdm

        starts = tqdm(range(0, len(texts), 256), disable=not show_progress, desc="Embedding")
        return np.vstack([self._m.encode(texts[i : i + 256]) for i in starts]).astype("float32")


def load_embedder(spec: str, device: str = "cpu", timeout: float = 3.0):
    kind, _, arg = spec.partition(":")
    if kind == "hf" and arg:
        return STEmbedder(arg, device, _hf_local_only(arg, timeout))
    if kind == "usem3":
        return USEEmbedder()
    raise ValueError(f"Unknown embedder {spec!r} (use hf:<model> or usem3)")


def embedder_key(spec: str) -> str:
    """Directory name for the indexes built with an embedder."""
    name = spec.partition(":")[2] or spec
    return "".join(c if c.isalnum() or c in "-_." else "__" for c in name)


# -------------------------------------------------------------------------- chat models
class LocalChatModel:
    """transformers chat model on CPU / CUDA / MPS, with token streaming."""

    def __init__(
        self,
        name: str,
        device: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        label: str,
        local_only: bool = False,
        languages=None,
        compact: bool = False,
        repetition_penalty: float = 1.0,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name, self.label, self.device = name, label, device
        self.languages, self.compact = languages, compact  # compact: short prompt (small model)
        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=local_only)
        self.model = (
            AutoModelForCausalLM.from_pretrained(name, dtype=dtype, local_files_only=local_only)
            .to(device)
            .eval()
        )
        self.gen = {
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            self.gen.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            self.gen.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        self._lock = threading.Lock()  # one generation at a time (shared by app sessions)

    def _inputs(self, messages):
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.device)

    def generate(self, messages: list[dict]) -> str:
        return "".join(self.stream(messages)).strip()

    def stream(self, messages: list[dict]) -> Iterator[str]:
        import torch
        from transformers import TextIteratorStreamer

        inputs = self._inputs(messages)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        def run():
            with torch.inference_mode():
                self.model.generate(**inputs, **self.gen, streamer=streamer)

        with self._lock:
            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            yield from streamer
            thread.join()


class OpenAIChatModel:
    """Any server speaking the OpenAI chat API: Ollama, LM Studio, llama.cpp, vLLM..."""

    languages, compact = None, False

    def __init__(
        self,
        base_url: str,
        settings,
        model: str = "",
        api_key: str = "",
        kind: str = "openai",
        transport=None,
    ):
        import httpx

        self.client = httpx.Client(
            base_url=base_url.rstrip("/"),
            transport=transport,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            timeout=httpx.Timeout(600, connect=settings.connect_timeout),
        )
        try:
            listed = [m["id"] for m in self.client.get("/models").raise_for_status().json()["data"]]
        except httpx.HTTPError as e:
            raise ModelUnavailable(f"no server at {base_url} ({describe_error(e)})") from e
        if model and model not in listed and f"{model}:latest" not in listed:
            raise ModelUnavailable(f"model {model!r} not on the server (has: {listed})")
        if not (model or listed):
            raise ModelUnavailable(f"{base_url} has no models (e.g. `ollama pull qwen2.5:3b`)")
        self.model_name = model or listed[0]
        self.label = f"{kind}:{self.model_name}"
        self.params = {
            "max_tokens": settings.max_new_tokens,
            "temperature": settings.temperature,
            "top_p": settings.top_p,
        }

    def _payload(self, messages, stream):
        return {"model": self.model_name, "messages": messages, "stream": stream, **self.params}

    def generate(self, messages: list[dict]) -> str:
        r = self.client.post("/chat/completions", json=self._payload(messages, False))
        return (r.raise_for_status().json()["choices"][0]["message"]["content"] or "").strip()

    def stream(self, messages: list[dict]) -> Iterator[str]:
        with self.client.stream(
            "POST", "/chat/completions", json=self._payload(messages, True)
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():  # server-sent events: "data: {...}"
                data = line.removeprefix("data:").strip()
                if not line.startswith("data:") or data == "[DONE]":
                    continue
                if delta := json.loads(data)["choices"][0].get("delta", {}).get("content"):
                    yield delta


def smollm2_gguf() -> Path:
    """The GGUF inside the llm-smollm2 wheel (found without importing the package, whose own
    runtime deps are disabled in pyproject.toml)."""
    try:
        dist = importlib.metadata.distribution("llm-smollm2")
    except importlib.metadata.PackageNotFoundError as e:
        raise ModelUnavailable("llm-smollm2 not installed (run `uv sync`)") from e
    path = Path(dist.locate_file("llm_smollm2/SmolLM2-135M-Instruct.Q4_1.gguf"))
    if not path.exists():
        raise ModelUnavailable(f"{path} missing")
    return path


def convert_gguf(gguf: Path, target: Path) -> Path:
    """Dequantise a GGUF once into a transformers folder (later loads take ~1 s)."""
    import torch
    from safetensors.torch import save_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    stamp = f"{gguf.name}:{gguf.stat().st_size}"
    if (target / "source.txt").exists() and (target / "source.txt").read_text() == stamp:
        return target
    log.info("Converting %s (one time)...", gguf.name)
    tmp = target.with_name(target.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    tok = AutoTokenizer.from_pretrained(gguf.parent, gguf_file=gguf.name)
    model = AutoModelForCausalLM.from_pretrained(
        gguf.parent, gguf_file=gguf.name, dtype=torch.float32
    )
    config = model.config.to_dict()
    config.pop("quantization_config", None)  # plain float32 weights after loading
    (tmp / "config.json").write_text(json.dumps(config))
    model.generation_config.save_pretrained(tmp)
    save_model(model, str(tmp / "model.safetensors"))
    tok.save_pretrained(tmp)
    (tmp / "source.txt").write_text(stamp)
    shutil.rmtree(target, ignore_errors=True)
    tmp.rename(target)
    return target


def create_chat_model(spec: str, settings, device: str):
    """One chain entry -> chat model (None for extractive)."""
    kind, _, arg = spec.partition(":")
    common = dict(
        device=device,
        max_new_tokens=settings.max_new_tokens,
        temperature=settings.temperature,
        top_p=settings.top_p,
    )
    if kind == "extractive":
        return None
    if kind == "hf" and arg:
        return LocalChatModel(
            arg, label=spec, local_only=_hf_local_only(arg, settings.connect_timeout), **common
        )
    if kind == "ollama":
        return OpenAIChatModel(settings.ollama_url, settings, arg, kind="ollama")
    if kind == "openai":
        if not settings.openai_base_url:
            raise ModelUnavailable("not configured (set BIBLE_RAG_OPENAI_BASE_URL)")
        return OpenAIChatModel(settings.openai_base_url, settings, arg, settings.openai_api_key)
    if kind == "smollm2":
        path = convert_gguf(smollm2_gguf(), settings.model_cache_dir / "SmolLM2-135M-Instruct")
        return LocalChatModel(
            str(path),
            label="smollm2",
            local_only=True,
            languages={"en"},
            compact=True,
            repetition_penalty=1.1,
            **{
                **common,
                "temperature": 0,  # greedy suits a 135M model
                "max_new_tokens": min(settings.max_new_tokens, 256),
            },
        )
    raise ValueError(f"Unknown LLM spec {spec!r}")


class ModelChain:
    """Use the first spec that loads; on an error while answering, move to the next one."""

    def __init__(self, specs: list[str], factory: Callable):
        self.specs, self.factory = list(specs), factory
        self.position, self.skipped = 0, []  # skipped: [(spec, reason)]
        self._model, self._loaded = None, False

    @property
    def model(self):
        while not self._loaded:
            if self.position >= len(self.specs):
                raise RuntimeError(
                    "No language model could be loaded: "
                    + "; ".join(f"{s} -> {r}" for s, r in self.skipped)
                )
            try:
                self._model, self._loaded = self.factory(self.specs[self.position]), True
            except Exception as e:  # noqa: BLE001 - any failure: try the next entry
                log.warning("LLM %s unavailable: %s", self.specs[self.position], describe_error(e))
                self.skipped.append((self.specs[self.position], describe_error(e)))
                self.position += 1
        return self._model

    @property
    def active(self) -> str:
        self.model  # noqa: B018
        return self.specs[self.position]

    def failover(self, error: BaseException):
        self.skipped.append((self.specs[self.position], f"failed: {describe_error(error)}"))
        self.position, self._loaded = self.position + 1, False
        return self.model


# -------------------------------------------------------------------------------- probe
def probe(settings, test: bool = False) -> list[dict]:
    """Availability of every chain entry (cheap checks; test=True also loads and generates)."""
    rows = []
    for spec in settings.embedding_specs:
        kind, _, arg = spec.partition(":")
        ok, _, detail = (
            hf_status(arg, settings.connect_timeout)
            if kind == "hf"
            else (kind == "usem3", True, "bundled, offline, multilingual")
        )
        has_index = all(
            (settings.index_dir / embedder_key(spec) / v / "manifest.json").exists()
            for v in settings.versions
        )
        rows.append(
            {"kind": "embedder", "spec": spec, "ok": ok, "detail": detail, "index": has_index}
        )
    for spec in settings.llm_specs:
        kind, _, arg = spec.partition(":")
        try:
            if kind == "hf":
                ok, _, detail = hf_status(arg, settings.connect_timeout)
            elif kind == "smollm2":
                ok, detail = True, f"bundled ({smollm2_gguf().stat().st_size // 10**6} MB, English)"
            elif kind == "extractive":
                ok, detail = True, "no model: quotes the retrieved passages"
            else:
                model = create_chat_model(spec, settings, "cpu")
                ok, detail = True, model.label
        except ModelUnavailable as e:
            ok, detail = False, str(e)
        row = {"kind": "llm", "spec": spec, "ok": ok, "detail": detail}
        if test and ok:
            try:
                m = create_chat_model(spec, settings, settings.resolve_device())
                row["test"] = (
                    repr(m.generate([{"role": "user", "content": "Say hello."}])[:40])
                    if m
                    else "ok"
                )
            except Exception as e:  # noqa: BLE001
                row["test"] = f"FAILED {describe_error(e)}"
        rows.append(row)
    return rows
