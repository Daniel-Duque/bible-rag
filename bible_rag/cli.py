"""Terminal interface.

    bible-rag models                       which embedders / LLMs are usable here
    bible-rag ingest                       build the indexes from the PDFs (once)
    bible-rag search "amor al prójimo"     retrieval only, no LLM (fast)
    bible-rag ask "What is faith?"         one question, answer + sources
    bible-rag chat                         interactive session (models loaded once)
    bible-rag serve                        HTTP service keeping models in memory
    bible-rag eval                         measure retrieval and answer quality

Models are picked from fallback chains: if the preferred model cannot be loaded the next
one is used automatically (see `bible-rag models`). ask, search and chat load the models
in this process; with --server URL (or BIBLE_RAG_SERVER_URL) they use a running
`bible-rag serve` instead, which keeps the models in memory between questions.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

from .config import Settings
from .corpus import VERSIONS

log = logging.getLogger("bible_rag")


# ------------------------------------------------------------------------------ backends
@dataclass
class AnswerView:
    """Backend-independent answer: stream ``tokens``, then call ``finish()`` for metadata."""

    sources: list[dict]
    tokens: Iterator[str]
    finish: Callable[[], dict]  # -> {"model", "notes", "grounding"}


class LocalBackend:
    """Runs the RAG pipeline in this process."""

    def __init__(self, settings: Settings, load_llm: bool):
        from .rag import BibleRAG

        t0 = time.perf_counter()
        print("Loading models and indexes...", file=sys.stderr, flush=True)
        self.rag = BibleRAG.from_settings(settings, load_llm=load_llm)
        info = self.rag.info()
        parts = [f"embedder {_status(info['embedder']['active'], self.rag.embedder_skipped)}"]
        if self.rag.chain:
            parts.append(f"LLM {_status(self.rag.chain.active_spec, self.rag.chain.skipped)}")
        print(
            f"Ready in {time.perf_counter() - t0:.0f}s: {', '.join(parts)}.",
            file=sys.stderr,
            flush=True,
        )

    def search(self, question: str, k: int | None, versions: list[str] | None) -> list[dict]:
        from .rag import hit_to_dict

        return [hit_to_dict(h) for h in self.rag.search(question, k, versions)]

    def ask(self, question: str, k: int | None, versions: list[str] | None) -> AnswerView:
        from .rag import hit_to_dict

        result = self.rag.ask_stream(question, k, versions)

        def finish() -> dict:
            return {
                "model": result.model,
                "notes": result.notes,
                "grounding": result.grounding.to_dict() if result.grounding else None,
            }

        return AnswerView([hit_to_dict(h) for h in result.sources], iter(result), finish)


def _status(active: str, skipped: list[tuple[str, str]]) -> str:
    return f"{active} (fallback; {skipped[0][0]} unavailable)" if skipped else active


class RemoteBackend:
    """Talks to a running `bible-rag serve`."""

    def __init__(self, url: str, timeout: float = 600.0, client=None):
        import httpx

        self.client = client or httpx.Client(base_url=url.rstrip("/"), timeout=timeout)
        try:
            self.client.get("/health").raise_for_status()
        except httpx.HTTPError as e:
            raise SystemExit(f"Cannot reach the Bible RAG service at {url}: {e}") from e

    def _post(self, path: str, question: str, k: int | None, versions: list[str] | None):
        r = self.client.post(path, json={"question": question, "k": k, "versions": versions})
        if r.status_code >= 400:
            detail = r.json().get("detail", r.text) if r.content else r.reason_phrase
            raise ValueError(f"Service error {r.status_code}: {detail}")
        return r.json()

    def search(self, question: str, k: int | None, versions: list[str] | None) -> list[dict]:
        return self._post("/search", question, k, versions)["sources"]

    def ask(self, question: str, k: int | None, versions: list[str] | None) -> AnswerView:
        data = self._post("/ask", question, k, versions)
        meta = {key: data.get(key) for key in ("model", "notes", "grounding")}
        return AnswerView(data["sources"], iter([data["answer"]]), lambda: meta)


def make_backend(args: argparse.Namespace, settings: Settings, load_llm: bool):
    url = getattr(args, "server", None) or settings.server_url
    return RemoteBackend(url) if url else LocalBackend(settings, load_llm=load_llm)


# ------------------------------------------------------------------------------ output
def print_sources(sources: Iterable[dict], full_text: bool = False, out=None) -> None:
    out = out or sys.stdout  # resolved at call time, so redirection works
    print("\nSources:", file=out)
    for i, s in enumerate(sources, 1):
        print(
            f"  [{i}] {s['reference']} ({s['version']})  score={s['score']:.3f}  p.{s['page']}",
            file=out,
        )
        if full_text:
            print(f"      {s['text']}", file=out)


def print_checks(meta: dict, sources: list[dict], show_context: bool) -> None:
    grounding = meta.get("grounding") or {}
    unsupported = [
        c["text"]
        for c in grounding.get("references", []) + grounding.get("quotes", [])
        if not c["supported"]
    ]
    model = meta.get("model")
    uncited = not grounding.get("grounded") and not unsupported and model != "extractive"
    if unsupported:
        print(
            "\n⚠ Not found in the retrieved passages, verify before trusting: "
            + "; ".join(f"“{u}”" for u in unsupported)
        )
    elif uncited:
        print("\nℹ The answer cites no verse that could be checked; rely on the passages below.")
    # With a weak or unverifiable answer, show the evidence itself
    print_sources(sources, full_text=show_context or bool(unsupported) or uncited)
    if model:
        print(f"\nAnswered by: {model}")
    for note in meta.get("notes") or []:
        print(f"Note: {note}")


def run_ask(backend, question: str, args: argparse.Namespace) -> None:
    view = backend.ask(question, args.k, args.versions)
    if args.json:
        text = "".join(view.tokens).strip()
        print(
            json.dumps(
                {"question": question, "answer": text, **view.finish(), "sources": view.sources},
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    for token in view.tokens:
        print(token, end="", flush=True)
    print()
    print_checks(view.finish(), view.sources, args.show_context)


def run_search(backend, question: str, args: argparse.Namespace) -> None:
    sources = backend.search(question, args.k, args.versions)
    if args.json:
        print(json.dumps({"question": question, "sources": sources}, ensure_ascii=False, indent=2))
        return
    print(f"Results for: {question!r}")
    print_sources(sources, full_text=True)


def read_question(args: argparse.Namespace) -> str:
    question = " ".join(args.question).strip() if args.question else ""
    if not question and not sys.stdin.isatty():
        question = sys.stdin.read().strip()  # echo "..." | bible-rag ask
    if not question:
        raise SystemExit('Please provide a question, e.g.: bible-rag ask "What is love?"')
    return question


# ------------------------------------------------------------------------------ commands
def cmd_models(args: argparse.Namespace, settings: Settings) -> int:
    from .probe import as_dicts, probe_all

    results = probe_all(settings, test=args.test)
    if args.json:
        print(json.dumps(as_dicts(results), indent=2))
        return 0
    first_ok = {
        kind: next((r.spec for r in results if r.kind == kind and r.available), None)
        for kind in ("embedder", "llm")
    }
    for kind, title in (("embedder", "Embedding models"), ("llm", "Language models")):
        print(f"{title} (tried in this order):")
        for r in (r for r in results if r.kind == kind):
            mark = "✓" if r.available else "✗"
            arrow = "  <- will be used" if r.spec == first_ok[kind] else ""
            print(f"  {mark} {r.spec}{arrow}\n      {r.detail}")
            if r.index is not None:
                print(f"      index: {'built' if r.index else 'not built (bible-rag ingest)'}")
            if r.test:
                print(f"      test: {r.test}")
        print()
    return 0


def cmd_ingest(args: argparse.Namespace, settings: Settings) -> int:
    from .ingest import ingest

    built = ingest(settings, force=args.force, versions=args.versions, embedder_spec=args.embedder)
    for version, index in built.items():
        print(f"{version}: {len(index)} passages, embedder {index.manifest['embedder_spec']}")
    if not built:
        print("Indexes already present (use --force to rebuild).")
    return 0


def cmd_search(args: argparse.Namespace, settings: Settings) -> int:
    question = read_question(args)
    run_search(make_backend(args, settings, load_llm=False), question, args)
    return 0


def cmd_ask(args: argparse.Namespace, settings: Settings) -> int:
    question = read_question(args)
    run_ask(make_backend(args, settings, load_llm=True), question, args)
    return 0


CHAT_HELP = """Type a question in English or Spanish and press Enter.
Commands: /search <text>  retrieval only | /sources on|off  show passage text
          /k <n>  passages per version  | /help | /quit"""


def cmd_chat(args: argparse.Namespace, settings: Settings) -> int:
    backend = make_backend(args, settings, load_llm=True)
    print("Bible RAG - KJV / Reina-Valera 1960\n" + CHAT_HELP)
    while True:
        try:
            line = input("\n? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        try:
            if line in ("/quit", "/exit", "/q", "exit", "quit", "salir"):
                return 0
            if line == "/help":
                print(CHAT_HELP)
            elif line.startswith("/k "):
                args.k = int(line.split()[1])
                print(f"k = {args.k} passages per version")
            elif line.startswith("/sources "):
                args.show_context = line.split()[1] == "on"
            elif line.startswith("/search "):
                run_search(backend, line[len("/search ") :], args)
            elif line.startswith("/"):
                print("Unknown command. " + CHAT_HELP)
            else:
                run_ask(backend, line, args)
        except KeyboardInterrupt:
            print("\n(interrupted)")
        except ValueError as e:
            print(f"Error: {e}")
    return 0


def cmd_serve(args: argparse.Namespace, settings: Settings) -> int:
    import uvicorn

    from .rag import BibleRAG
    from .server import create_app

    rag = BibleRAG.from_settings(settings, load_llm=not args.no_llm)
    uvicorn.run(
        create_app(rag),
        host=args.host or settings.host,
        port=args.port or settings.port,
        log_level="info",
    )
    return 0


def cmd_eval(args: argparse.Namespace, settings: Settings) -> int:
    from .evaluation import evaluate
    from .rag import BibleRAG

    rag = BibleRAG.from_settings(settings, load_llm=not args.retrieval_only)

    def progress(r) -> None:
        hit = " ".join(f"{v}:{'✓' if ok else '✗'}" for v, ok in sorted(r.retrieval_hit.items()))
        line = f"[{r.language}] {r.question[:58]:58s} retrieval {hit}"
        if r.model is not None:
            verdict = "cites gold ✓" if r.cites_gold else "cites gold ✗"
            flag = f", {len(r.unsupported)} unverified" if r.unsupported else ""
            line += f" | {r.model}: {verdict}{flag} ({r.seconds}s)"
        print(line, file=sys.stderr, flush=True)

    report = evaluate(
        rag,
        k=args.k,
        answers=not args.retrieval_only,
        languages=tuple(args.lang),
        progress=progress,
    )
    report["config"] = {
        "embedder": rag.embedder_spec,
        "k": args.k or settings.top_k,
        "llm": rag.chain.status() if rag.chain else None,
    }
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
    print(json.dumps(report if args.json else report["summary"], ensure_ascii=False, indent=2))
    return 0


# ------------------------------------------------------------------------------ parser
def parse_versions(value: str) -> list[str]:
    versions = [v.strip().lower() for v in value.split(",") if v.strip()]
    unknown = [v for v in versions if v not in VERSIONS]
    if unknown or not versions:
        raise argparse.ArgumentTypeError(
            f"invalid version(s) {value!r}; choose from: {', '.join(VERSIONS)}"
        )
    return versions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bible-rag", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    versions_kw = dict(
        type=parse_versions,
        metavar="LIST",
        help=f"comma-separated subset of: {','.join(VERSIONS)} (default: all)",
    )

    p = sub.add_parser("models", help="show which embedders / LLMs are usable here")
    p.add_argument(
        "--test",
        action="store_true",
        help="also load every available LLM and generate a short reply",
    )
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("ingest", help="parse the PDFs and build the vector indexes")
    p.add_argument("--force", action="store_true", help="rebuild existing indexes")
    p.add_argument(
        "--include-apocrypha",
        action="store_true",
        default=None,
        help="also index the KJV Apocrypha",
    )
    p.add_argument(
        "--embedder",
        metavar="SPEC",
        help="use this embedder instead of the first available (e.g. usem3)",
    )
    p.add_argument("--versions", **versions_kw)
    p.set_defaults(func=cmd_ingest)

    def query_parser(name: str, help_: str, func, with_question: bool = True):
        p = sub.add_parser(name, help=help_)
        if with_question:
            p.add_argument("question", nargs="*", help="question (or read from stdin)")
        p.add_argument("-k", type=int, default=None, help="passages per version")
        p.add_argument("--versions", **versions_kw)
        p.add_argument(
            "--server",
            metavar="URL",
            help="use a running `bible-rag serve` (env: BIBLE_RAG_SERVER_URL)",
        )
        p.add_argument("--json", action="store_true", help="machine-readable output")
        p.add_argument(
            "--show-context", action="store_true", help="print the text of every retrieved passage"
        )
        p.set_defaults(func=func)
        return p

    query_parser("search", "semantic search only (no LLM)", cmd_search)
    query_parser("ask", "answer one question", cmd_ask)
    query_parser("chat", "interactive question/answer session", cmd_chat, with_question=False)

    p = sub.add_parser("serve", help="start the HTTP service")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--no-llm", action="store_true", help="retrieval-only service (/search)")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("eval", help="measure retrieval and answer quality on gold questions")
    p.add_argument("-k", type=int, default=None, help="passages per version")
    p.add_argument("--retrieval-only", action="store_true", help="do not generate answers")
    p.add_argument("--lang", nargs="+", choices=["en", "es"], default=["en", "es"])
    p.add_argument("--output", metavar="FILE", help="write the full report as JSON")
    p.add_argument("--json", action="store_true", help="print the full report")
    p.set_defaults(func=cmd_eval)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    if not args.verbose:  # keep the terminal readable
        for noisy in ("httpx", "sentence_transformers", "transformers", "faiss", "huggingface_hub"):
            logging.getLogger(noisy).setLevel(logging.ERROR)
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    settings = Settings.from_env(include_apocrypha=getattr(args, "include_apocrypha", None))
    try:
        return args.func(args, settings)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        log.debug("fatal", exc_info=True)
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except OSError as e:  # transformers / huggingface_hub: download or model-name problems
        log.debug("fatal", exc_info=True)
        print(
            f"Error: {e}\nModels are downloaded from Hugging Face on first use: check the "
            "connection and the BIBLE_RAG_*_MODEL names (run with -v for details).",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
