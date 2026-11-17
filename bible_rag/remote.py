"""Use a `bible-rag serve` API (e.g. on a GPU machine) instead of the local models, and fall
back to the local models whenever it can't answer.

`open_backend(settings)` returns what the CLI and the web app talk to:
  * BIBLE_RAG_SERVER_URL unset: the local BibleRAG;
  * set and the server answers /v1/info with the key: a RemoteRAG (same methods);
  * set but unreachable / refusing the key / speaking another protocol: the local BibleRAG,
    with a note saying why.
A RemoteRAG that fails on a question before its first word (network, busy, error) answers
that question with the local models too (loaded then, once), with a note.
"""

from __future__ import annotations

import json
import logging
from urllib.parse import urlsplit

from .wire import PROTOCOL, RemotePlan, grounding_from_json, hit_from_json

log = logging.getLogger(__name__)


class RemoteUnavailable(RuntimeError):
    """The server can't be used (unreachable, wrong key, incompatible, unsafe URL)."""


def check_url(url: str, api_key: str, insecure: bool = False) -> str:
    """The server URL, if it is safe to send the key to it."""
    from .server import is_loopback

    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise RemoteUnavailable(f"not an http(s) URL: {url!r}")
    if parts.scheme == "http" and api_key and not is_loopback(parts.hostname) and not insecure:
        raise RemoteUnavailable(
            f"refusing to send the API key in clear text to {parts.hostname}: use https://, or "
            "an SSH tunnel to http://127.0.0.1 (BIBLE_RAG_SERVER_INSECURE=true to override)"
        )
    return url.rstrip("/")


class RemoteChain:
    """What the CLI and the web app show of the server's language model."""

    def __init__(self, active: str):
        self.active, self.skipped = active, []

    def describe(self) -> str:
        return self.active


class RemoteRAG:
    """The API client, shaped like BibleRAG for the CLI and the web app."""

    def __init__(self, settings, client=None, local=None):
        import httpx

        self.settings = settings
        self.url = check_url(settings.server_url, settings.api_key, settings.server_insecure)
        self.host = urlsplit(self.url).hostname
        headers = {"Authorization": f"Bearer {settings.api_key}"} if settings.api_key else {}
        self.client = client or httpx.Client(
            base_url=self.url,
            headers=headers,
            timeout=httpx.Timeout(settings.server_timeout, connect=settings.connect_timeout),
            follow_redirects=False,  # the key never goes to another host
        )
        self._local_factory, self._local = local, None
        info = self._info()
        self.indexes = {str(v["id"]): None for v in info["versions"]}
        self.embedder_spec = f"{info['embedder']} @ {self.host}"
        self.chain = RemoteChain(f"{info['llm']} @ {self.host}")
        self.settings.top_k = int(info.get("top_k", settings.top_k))

    def _info(self) -> dict:
        import httpx

        try:
            r = self.client.get("/v1/info")
        except httpx.HTTPError as e:
            raise RemoteUnavailable(f"no answer from {self.url} ({type(e).__name__})") from e
        if r.status_code == 401:
            raise RemoteUnavailable("the server refused the API key (BIBLE_RAG_API_KEY)")
        if r.status_code != 200:
            raise RemoteUnavailable(f"{self.url} answered HTTP {r.status_code}")
        info = r.json()
        if info.get("protocol") != PROTOCOL:
            raise RemoteUnavailable(
                f"incompatible server (protocol {info.get('protocol')}, this client "
                f"{PROTOCOL}): update bible-rag on one side"
            )
        return info

    def fallback_notes(self) -> list[str]:
        return [f"Answered by the server {self.host} ({self.embedder_spec}, {self.chain.active})."]

    # ------------------------------------------------------------------ local fallback
    def local(self):
        """The local models, loaded on first need (when the server fails)."""
        if self._local is None:
            if self._local_factory is None:
                raise RemoteUnavailable("the server failed and no local models are configured")
            self._local = self._local_factory()
        return self._local

    def _fall_back(self, why: str):
        log.warning("Server %s failed (%s): using the local models", self.url, why)
        return self.local(), f"The server {self.host} failed ({why}): answered locally."

    # ------------------------------------------------------------------------- queries
    @staticmethod
    def _body(question, k, versions, task=None) -> dict:
        return {"question": question, "k": k, "versions": versions, "task": task}

    def search(self, question: str, k: int | None = None, versions=None):
        import httpx

        try:
            r = self.client.post("/v1/search", json=self._body(question, k, versions))
            if r.status_code == 422:
                raise ValueError(_detail(r))
            r.raise_for_status()
            return [hit_from_json(h) for h in r.json()["sources"]]
        except httpx.HTTPError as e:
            local, _ = self._fall_back(type(e).__name__)
            return local.search(question, k, versions)

    def ask_stream(self, question, k=None, versions=None, task=None):
        """The answer, streaming; the plan is read right away (the CLI shows it first)."""
        import httpx

        request = self.client.build_request(
            "POST", "/v1/ask", json=self._body(question, k, versions, task)
        )
        try:
            response = self.client.send(request, stream=True)
        except httpx.HTTPError as e:
            return self._local_answer(question, k, versions, task, type(e).__name__)
        if response.status_code != 200:
            response.read()
            response.close()
            if response.status_code == 422:  # the question itself is invalid: say so
                raise ValueError(_detail(response))
            why = f"HTTP {response.status_code}: {_detail(response)}"
            return self._local_answer(question, k, versions, task, why)
        lines = response.iter_lines()
        try:
            first = json.loads(next(lines))
            if first.get("type") != "plan":
                raise ValueError(first.get("detail", "unexpected reply"))
        except (StopIteration, ValueError, httpx.HTTPError) as e:
            response.close()
            return self._local_answer(question, k, versions, task, str(e) or type(e).__name__)
        return RemoteAnswer(self, response, lines, first["plan"], (question, k, versions, task))

    def _local_answer(self, question, k, versions, task, why):
        local, note = self._fall_back(why)
        answer = local.ask_stream(question, k, versions, task)
        answer.notes.append(note)
        return answer


def _detail(response) -> str:
    try:
        detail = response.json().get("detail", "")
    except ValueError:
        detail = ""
    return detail if isinstance(detail, str) else json.dumps(detail)[:200]


class RemoteAnswer:
    """A server's streamed answer, with the attributes of rag.Answer once iterated. If the
    stream breaks before the first word, the local models answer instead."""

    def __init__(self, rag: RemoteRAG, response, lines, plan: dict, query: tuple):
        self.rag, self._response, self._lines, self._query = rag, response, lines, query
        self.plan = RemotePlan.from_json(plan)
        self.question, self.intent = query[0], self.plan.intent
        self.language = self.plan.language
        self.text, self.synthesis, self.model = "", "", ""
        self.notes: list[str] = []
        self.sources, self.grounding = [], None

    def __iter__(self):
        import httpx

        started = False
        try:
            for line in self._lines:
                message = json.loads(line)
                kind = message.get("type")
                if kind == "token":
                    started = True
                    yield str(message["text"])
                elif kind == "done":
                    self._finish(message["answer"])
                    return
                elif kind == "error":
                    raise RuntimeError(f"server: {message.get('detail', 'error')}")
            raise RuntimeError("server: the answer was cut off")
        except (RuntimeError, httpx.HTTPError, ValueError) as e:
            if started:
                raise
            local = self.rag._local_answer(*self._query, str(e))
            yield from local
            self._adopt(local)
        finally:
            self._response.close()

    def _finish(self, data: dict) -> None:
        self.text, self.synthesis = str(data["text"]), str(data["synthesis"])
        self.model = f"{data['model']} @ {self.rag.host}"
        self.notes = [str(n) for n in data.get("notes", [])] + self.rag.fallback_notes()
        self.sources = [hit_from_json(h) for h in data.get("sources", [])]
        self.grounding = grounding_from_json(data.get("grounding", {}))

    def _adopt(self, local) -> None:
        self.plan, self.intent, self.language = local.plan, local.intent, local.language
        self.text, self.synthesis, self.model = local.text, local.synthesis, local.model
        self.notes, self.sources, self.grounding = local.notes, local.sources, local.grounding


def open_backend(settings, load_llm: bool = True):
    """The server if configured and usable, else the local models (see the module doc)."""
    from .rag import BibleRAG

    def local():
        return BibleRAG.from_settings(settings, load_llm=load_llm)

    if settings.server_url:
        try:
            return RemoteRAG(settings, local=local)
        except RemoteUnavailable as e:
            log.warning("Server unavailable: %s", e)
            rag = local()
            rag.notes.append(f"Server {settings.server_url} unavailable ({e}): local models used.")
            return rag
    return local()
