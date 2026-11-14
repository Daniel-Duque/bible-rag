"""Chat models and the automatic fallback chain.

The preferred model is the notebook's Qwen2.5-1.5B-Instruct, downloaded from Hugging Face.
When it cannot be loaded (no network, blocked hub, not enough memory...) or fails while
answering, :class:`ModelChain` moves on to the next entry of ``Settings.llm_specs``:

=================  ==========================================================================
``hf:<model>``     Any Hugging Face chat model (or a local directory), run with transformers.
``ollama[:name]``  A local Ollama server (free). Default model: the first one installed.
``openai[:name]``  Any OpenAI-compatible local server: LM Studio, llama.cpp server, vLLM...
``smollm2``        SmolLM2-135M-Instruct, bundled with the install (``llm-smollm2`` wheel),
                   so it works offline. Small and English-only; its answers are checked.
``extractive``     No LLM: the answer is built from the retrieved passages. Always works.
=================  ==========================================================================
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import shutil
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Protocol

from .hub import ModelUnavailable, prepare_hf_model

log = logging.getLogger(__name__)

EXTRACTIVE = "extractive"


class ChatModel(Protocol):
    model_name: str
    label: str  # chain spec that produced it, e.g. "hf:Qwen/Qwen2.5-1.5B-Instruct"
    languages: frozenset[str] | None  # None = multilingual
    compact: bool  # small model: shorter prompt, fewer passages

    def generate(self, messages: list[dict]) -> str: ...

    def stream(self, messages: list[dict]) -> Iterator[str]: ...


# ------------------------------------------------------------------------------ transformers
class LocalChatModel:
    """A chat model run in-process with transformers (CPU, CUDA or Apple MPS)."""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        label: str | None = None,
        languages: frozenset[str] | None = None,
        compact: bool = False,
        local_files_only: bool = False,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.label = label or f"hf:{model_name}"
        self.languages = languages
        self.compact = compact
        self.device = device
        # float16 is slow or unsupported for CPU matmuls
        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        log.info("Loading %s on %s (%s)", model_name, device, dtype)
        kw = {"local_files_only": local_files_only}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kw)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, **kw)
        self.model.to(device)
        self.model.eval()
        self.gen_kwargs: dict = {"max_new_tokens": max_new_tokens}
        if repetition_penalty != 1.0:
            self.gen_kwargs["repetition_penalty"] = repetition_penalty
        if temperature > 0:
            self.gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            self.gen_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        # One generation at a time: the HTTP service may call us from several threads.
        self._lock = threading.Lock()

    def _inputs(self, messages: list[dict]):
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.device)

    def generate(self, messages: list[dict]) -> str:
        import torch

        inputs = self._inputs(messages)
        with self._lock, torch.inference_mode():
            output = self.model.generate(
                **inputs, **self.gen_kwargs, pad_token_id=self.tokenizer.eos_token_id
            )
        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def stream(self, messages: list[dict]) -> Iterator[str]:
        import torch
        from transformers import TextIteratorStreamer

        inputs = self._inputs(messages)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        def run():
            with torch.inference_mode():
                self.model.generate(
                    **inputs,
                    **self.gen_kwargs,
                    streamer=streamer,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

        with self._lock:
            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            yield from streamer
            thread.join()


# ------------------------------------------------------------------------------ local servers
class OpenAICompatibleChatModel:
    """Chat completions over HTTP: Ollama, LM Studio, llama.cpp server, vLLM, LocalAI..."""

    languages = None
    compact = False

    def __init__(
        self,
        base_url: str,
        model: str,
        label: str,
        api_key: str = "",
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        connect_timeout: float = 3.0,
        read_timeout: float = 600.0,
        transport=None,
    ):
        import httpx

        self.model_name = model
        self.label = label
        self.params = {"max_tokens": max_new_tokens, "temperature": temperature, "top_p": top_p}
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            transport=transport,
            timeout=httpx.Timeout(read_timeout, connect=connect_timeout),
        )

    def _payload(self, messages: list[dict], stream: bool) -> dict:
        return {"model": self.model_name, "messages": messages, "stream": stream, **self.params}

    def generate(self, messages: list[dict]) -> str:
        r = self.client.post("/chat/completions", json=self._payload(messages, False))
        r.raise_for_status()
        return (r.json()["choices"][0]["message"]["content"] or "").strip()

    def stream(self, messages: list[dict]) -> Iterator[str]:
        with self.client.stream(
            "POST", "/chat/completions", json=self._payload(messages, True)
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                delta = json.loads(data)["choices"][0].get("delta", {}).get("content")
                if delta:
                    yield delta


def _http_get_json(url: str, timeout: float, transport=None, api_key: str = "") -> dict:
    import httpx

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        with httpx.Client(timeout=timeout, transport=transport, headers=headers) as c:
            r = c.get(url)
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        raise ModelUnavailable(f"no server at {url} ({e.__class__.__name__}: {e})") from e


def connect_ollama(settings, model: str = "", transport=None) -> OpenAICompatibleChatModel:
    base = settings.ollama_url.rstrip("/")
    tags = _http_get_json(f"{base}/api/tags", settings.connect_timeout, transport)
    installed = [m["name"] for m in tags.get("models", [])]
    wanted = model or settings.ollama_model
    if not installed:
        raise ModelUnavailable("Ollama is running but has no models (try: ollama pull qwen2.5:3b)")
    if wanted and wanted not in installed and f"{wanted}:latest" not in installed:
        raise ModelUnavailable(f"Ollama has no model {wanted!r} (installed: {installed})")
    name = wanted or installed[0]
    return OpenAICompatibleChatModel(
        f"{base}/v1",
        name,
        label=f"ollama:{name}",
        max_new_tokens=settings.max_new_tokens,
        temperature=settings.temperature,
        top_p=settings.top_p,
        connect_timeout=settings.connect_timeout,
        transport=transport,
    )


def connect_openai(settings, model: str = "", transport=None) -> OpenAICompatibleChatModel:
    if not settings.openai_base_url:
        raise ModelUnavailable("not configured (set BIBLE_RAG_OPENAI_BASE_URL)")
    base = settings.openai_base_url.rstrip("/")
    listing = _http_get_json(
        f"{base}/models", settings.connect_timeout, transport, settings.openai_api_key
    )
    available = [m["id"] for m in listing.get("data", [])]
    name = model or settings.openai_model or (available[0] if available else "")
    if not name:
        raise ModelUnavailable(f"{base} lists no models; set BIBLE_RAG_OPENAI_MODEL")
    return OpenAICompatibleChatModel(
        base,
        name,
        label=f"openai:{name}",
        api_key=settings.openai_api_key,
        max_new_tokens=settings.max_new_tokens,
        temperature=settings.temperature,
        top_p=settings.top_p,
        connect_timeout=settings.connect_timeout,
        transport=transport,
    )


# ------------------------------------------------------------------------------ bundled model
SMOLLM2_DIST = "llm-smollm2"
SMOLLM2_FILE = "llm_smollm2/SmolLM2-135M-Instruct.Q4_1.gguf"


def bundled_smollm2_gguf() -> Path:
    """Path of the GGUF shipped in the ``llm-smollm2`` wheel (no import of the package)."""
    try:
        path = Path(importlib.metadata.distribution(SMOLLM2_DIST).locate_file(SMOLLM2_FILE))
    except importlib.metadata.PackageNotFoundError as e:
        raise ModelUnavailable(
            f"package {SMOLLM2_DIST!r} not installed (run `uv sync`; it is in the default "
            "`fallback-llm` dependency group)"
        ) from e
    if not path.exists():
        raise ModelUnavailable(f"{path} missing from the {SMOLLM2_DIST} install")
    return path


def convert_gguf(gguf: Path, target: Path) -> Path:
    """Dequantise a GGUF once into a regular transformers folder (loads in ~1 s afterwards)."""
    import torch
    from safetensors.torch import save_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    stamp = f"{gguf.name}:{gguf.stat().st_size}"
    marker = target / "converted-from.txt"
    if marker.exists() and marker.read_text() == stamp:
        return target
    log.info("Converting %s to %s (one time)...", gguf.name, target)
    tmp = target.with_name(target.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    tokenizer = AutoTokenizer.from_pretrained(gguf.parent, gguf_file=gguf.name)
    model = AutoModelForCausalLM.from_pretrained(
        gguf.parent, gguf_file=gguf.name, dtype=torch.float32
    )
    config = model.config.to_dict()
    config.pop("quantization_config", None)  # weights are plain float32 after loading
    (tmp / "config.json").write_text(json.dumps(config, indent=2))
    model.generation_config.save_pretrained(tmp)
    save_model(model, str(tmp / "model.safetensors"))
    tokenizer.save_pretrained(tmp)
    (tmp / "converted-from.txt").write_text(stamp)
    shutil.rmtree(target, ignore_errors=True)
    tmp.rename(target)
    return target


def load_smollm2(settings, device: str) -> LocalChatModel:
    path = convert_gguf(bundled_smollm2_gguf(), settings.model_cache_dir / "SmolLM2-135M-Instruct")
    return LocalChatModel(
        str(path),
        device=device,
        max_new_tokens=min(settings.max_new_tokens, 256),
        temperature=0,
        repetition_penalty=1.1,  # greedy decoding suits a 135M model best
        label="smollm2",
        languages=frozenset({"en"}),
        compact=True,
    )


# ------------------------------------------------------------------------------ the chain
def create_chat_model(spec: str, settings, device: str) -> ChatModel | None:
    """Instantiate one chain entry. Returns None for ``extractive`` (no LLM)."""
    kind, _, arg = spec.partition(":")
    if kind == EXTRACTIVE:
        return None
    if kind == "hf" and arg:
        local_only = prepare_hf_model(arg, settings.connect_timeout)  # fails fast if offline
        return LocalChatModel(
            arg,
            device=device,
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            top_p=settings.top_p,
            label=spec,
            local_files_only=local_only,
        )
    if kind == "ollama":
        return connect_ollama(settings, arg)
    if kind == "openai":
        return connect_openai(settings, arg)
    if kind == "smollm2":
        return load_smollm2(settings, device)
    raise ValueError(
        f"Unknown LLM spec {spec!r}: use hf:<model>, ollama[:model], openai[:model], "
        "smollm2 or extractive"
    )


def describe_error(error: BaseException) -> str:
    text = str(error).strip().splitlines()[0] if str(error).strip() else ""
    return f"{error.__class__.__name__}: {text}"[:200]


class ModelChain:
    """Ordered list of model specs; the first one that loads is used.

    If the active model raises while answering, :meth:`failover` records the error and
    switches to the next entry, so a question is always answered by *something*.
    """

    def __init__(self, specs: list[str], factory: Callable[[str], ChatModel | None]):
        if not specs:
            raise ValueError("The LLM chain is empty")
        self.specs = list(specs)
        self.factory = factory
        self.position = 0
        self.skipped: list[tuple[str, str]] = []  # (spec, reason) for every failed entry
        self._model: ChatModel | None = None
        self._loaded = False

    @property
    def model(self) -> ChatModel | None:
        """The active chat model (None = extractive answers)."""
        while not self._loaded:
            if self.position >= len(self.specs):
                tried = "; ".join(f"{s} -> {r}" for s, r in self.skipped)
                raise RuntimeError(f"No language model could be loaded ({tried})")
            spec = self.specs[self.position]
            try:
                self._model = self.factory(spec)
                self._loaded = True
                if self.skipped:
                    log.warning("Using %s (fallback)", spec)
            except Exception as e:  # noqa: BLE001 - any failure means "try the next one"
                log.warning("LLM %s unavailable: %s", spec, describe_error(e))
                log.debug("load failure", exc_info=True)
                self.skipped.append((spec, describe_error(e)))
                self.position += 1
        return self._model

    @property
    def active_spec(self) -> str:
        self.model  # noqa: B018 - make sure something is loaded
        return self.specs[self.position]

    @property
    def is_fallback(self) -> bool:
        return self.active_spec != self.specs[0]

    def failover(self, error: BaseException) -> ChatModel | None:
        spec = self.specs[self.position]
        log.warning(
            "LLM %s failed while answering: %s; switching to the next one",
            spec,
            describe_error(error),
        )
        self.skipped.append((spec, f"failed while answering: {describe_error(error)}"))
        self.position += 1
        self._loaded = False
        self._model = None
        return self.model

    def status(self) -> dict:
        return {
            "active": self.active_spec,
            "fallback": self.is_fallback,
            "skipped": [{"spec": s, "reason": r} for s, r in self.skipped],
            "chain": self.specs,
        }
