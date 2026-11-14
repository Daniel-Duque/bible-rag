"""Local chat LLM (Qwen2.5-1.5B-Instruct by default), as in RAG0.ipynb.

Differences from the notebook:

* device and dtype are picked automatically (the notebook assumed a CUDA T4):
  float16 on CUDA/MPS, float32 on CPU (float16 matmuls are slow or unsupported on CPU);
* only the newly generated tokens are decoded, instead of slicing the prompt
  string off the pipeline output;
* generation parameters are passed per call (this removes the notebook's
  "Both max_new_tokens and max_length seem to have been set" warning);
* tokens can be streamed, so answers appear progressively in the terminal.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from typing import Protocol

log = logging.getLogger(__name__)


class ChatModel(Protocol):
    model_name: str

    def generate(self, messages: list[dict]) -> str: ...

    def stream(self, messages: list[dict]) -> Iterator[str]: ...


class LocalChatModel:
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.device = device
        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        log.info("Loading %s on %s (%s)", model_name, device, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        self.model.to(device)
        self.model.eval()
        self.gen_kwargs: dict = {"max_new_tokens": max_new_tokens}
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
