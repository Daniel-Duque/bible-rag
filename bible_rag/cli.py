"""`bible-rag` command line.

ask "question"    one answer with checked citations. The assistant reads the question and
                  picks its task: a direct search ("Where was Jesus born?"), both sides of a
                  contradiction ("Are there any paradoxes in the Bible about alcohol?"), a
                  translation comparison ("How do the versions render Psalm 16:10?") or, for
                  a subject the Bible never names, the related concepts it does speak of
                  ("Should I use a cellphone?");
                  --task contradiction|translation|related|general asks for one explicitly
chat              the same, as a conversation (models load once; /contradiction <question>,
                  /translation <question>, /related <question> ask for a task)
search "text"     retrieval only: the best passages of each translation
versions          the Bible texts, their licenses and sources
models [--test]   which embedders, NLI models and LLMs are usable here
ingest            update the local indexes if a text or the code reading it changed
prebuild          regenerate the indexes shipped in storage/prebuilt (maintainers)
serve             read-only HTTP API for other machines (e.g. on a GPU computer)
eval              measure retrieval, answer and assistant quality
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

from .config import Settings
from .sources import VERSIONS


def load(settings: Settings, llm: bool = True):
    """The server (BIBLE_RAG_SERVER_URL) if usable, else the local models and indexes."""
    from .remote import open_backend

    t0 = time.perf_counter()
    where = f" (server {settings.server_url})" if settings.server_url else ""
    print(f"Loading models and indexes{where}...", file=sys.stderr, flush=True)
    rag = open_backend(settings, load_llm=llm)
    used = f"embedder {rag.embedder_spec}" + (f", LLM {rag.chain.describe()}" if rag.chain else "")
    print(f"Ready in {time.perf_counter() - t0:.0f}s ({used}).", file=sys.stderr, flush=True)
    for note in getattr(rag, "notes", []):
        print(f"Note: {note}", file=sys.stderr, flush=True)
    return rag


def print_sources(hits, full: bool, cited=()) -> None:
    """`cited`: hits that only back a translation note (no score of their own)."""
    print("\nSources:")
    cited = {id(h) for h in cited}
    for i, h in enumerate(hits, 1):
        page = f"  p.{h.passage.page}" if h.passage.page else ""
        score = "cited by a note" if id(h) in cited else f"score={h.score:.3f}"
        print(f"  [{i}] {h.passage.citation}  {score}{page}")
        if full:
            print(f"      {h.passage.text}")


_ROUTE_LABELS = {
    "general": "direct search",
    "contradiction": "contradiction search (both statements)",
    "translation": "translation comparison",
    "related": "related concepts (not named in the Bible)",
}


def describe_plan(plan) -> str:
    """'[contradiction search (both statements) · alcohol ...]' shown before an answer."""
    parts = [_ROUTE_LABELS[plan.intent]]
    if plan.topic_name:
        parts.append(plan.topic_name)
    if plan.reason:
        parts.append(plan.reason)  # why: closeness to each kind of question, or the LLM
    return f"[{' · '.join(parts)}]"


TASKS = ("general", "contradiction", "translation", "related")
TASK_HINT = (
    "Unsure which task this is: to compare translations, look for contradictions or see the "
    "related concepts instead, use {how}."
)


def print_answer(rag, question: str, args, task: str | None = None, hint: str = "") -> None:
    answer = rag.ask_stream(question, args.k, args.versions, task or getattr(args, "task", None))
    print(describe_plan(answer.plan), file=sys.stderr, flush=True)
    for token in answer:
        print(token, end="", flush=True)
    print()
    g = answer.grounding
    if g.unsupported:
        print(
            "\n⚠ Not found in the retrieved passages, verify before trusting: "
            + "; ".join(f"“{c.text}”" for c in g.unsupported)
        )
    elif not g.grounded and answer.model != "extractive":
        print("\nℹ The answer cites no verse that could be checked; see the passages below.")
    # with an unverifiable answer, show the evidence itself
    print_sources(answer.sources, args.show_context or not g.grounded, answer.plan.extra)
    if answer.intent != "general":
        print("Searched for: " + " | ".join(f"“{q}”" for q in answer.plan.queries))
    print(f"\nAnswered by: {answer.model}")
    for note in answer.notes:
        print(f"Note: {note}")
    if hint and answer.plan.reason.startswith("unsure"):
        print(TASK_HINT.format(how=hint))


def question_of(args) -> str:
    q = " ".join(args.question).strip() or ("" if sys.stdin.isatty() else sys.stdin.read().strip())
    if not q:
        raise SystemExit('Please ask a question, e.g. bible-rag ask "What is love?"')
    return q


def cmd_versions(_args, settings):
    from .index import index_is_current

    for vid, v in VERSIONS.items():
        state = "selected" if vid in settings.versions else "available"
        if any(index_is_current(d, vid) for d in settings.index_dir.glob(f"*/{vid}")):
            state += ", indexed"
        elif any(index_is_current(d, vid) for d in settings.prebuilt_dir.glob(f"*/{vid}")):
            state += ", prebuilt"
        print(
            f"{'✓' if v.public else '✗'} {vid:8s} {v.name} [{state}]\n"
            f"    license: {v.license}{'' if v.public else '  (NOT for public use)'}\n"
            f"    source:  {v.source}" + (f"\n    note:    {v.conditions}" if v.conditions else "")
        )
    allowed = "allowed" if settings.allow_restricted else "refused"
    print(
        f"\nSelected: {','.join(settings.versions)} (BIBLE_RAG_VERSIONS). "
        f"Restricted versions: {allowed} (BIBLE_RAG_ALLOW_RESTRICTED)."
    )


def cmd_models(args, settings):
    from .models import probe

    for kind, title in (
        ("embedder", "Embedding models"),
        ("nli", "Opposition check (NLI) models"),
        ("llm", "Language models"),
    ):
        rows = [r for r in probe(settings, args.test) if r["kind"] == kind]
        first = next((r["spec"] for r in rows if r["ok"]), None)
        print(f"{title} (tried in order):")
        for r in rows:
            print(
                f"  {'✓' if r['ok'] else '✗'} {r['spec']}"
                f"{'  <- will be used' if r['spec'] == first else ''}\n      {r['detail']}"
            )
            if "index" in r:
                print(f"      index: {'built' if r['index'] else 'not built (bible-rag ingest)'}")
            if "test" in r:
                print(f"      test: {r['test']}")


def embedding_progress():
    """A progress bar per version while embedding, drawn in ASCII: on a terminal that isn't
    UTF-8 tqdm's block characters print as 'âââ' and every update leaves a new line."""
    from tqdm import tqdm

    bars = {}

    def progress(version: str, done: int, total: int) -> None:
        if version not in bars:
            bars[version] = tqdm(desc=f"{version}: embedding", total=total, ascii=True)
        bars[version].update(done - bars[version].n)
        if done >= total:
            bars.pop(version).close()

    return progress


def built_line(version: str, index) -> str:
    if not index.embedded:
        return f"{version}: {len(index)} passages (not embedded: a language the embedder lacks)"
    return f"{version}: {len(index)} passages, {index.reused} embeddings reused"


def cmd_ingest(args, settings):
    from .index import build_indexes

    if args.sample:
        settings.books = "sample"
    built = build_indexes(
        settings, force=args.force, spec=args.embedder, progress=embedding_progress()
    )
    for version, index in built.items():
        print(built_line(version, index))
    if not built:
        print(
            "Everything is up to date (the shipped prebuild or your local indexes); "
            "--force rebuilds, reusing unchanged embeddings."
        )


def cmd_prebuild(args, settings):
    from .index import prebuild

    built = prebuild(settings, args.embedder, args.force, embedding_progress())
    for version, index in built.items():
        print(built_line(version, index))
    if not built:
        print("The prebuild is up to date.")
    files = [f for f in settings.prebuilt_dir.rglob("*") if f.is_file() and "lock" not in f.name]
    print(f"{settings.prebuilt_dir}: {sum(f.stat().st_size for f in files) / 1e6:.1f} MB")


def cmd_search(args, settings):
    question = question_of(args)
    print_sources(load(settings, llm=False).search(question, args.k, args.versions), True)


def cmd_ask(args, settings):
    question = question_of(args)
    print_answer(load(settings), question, args, hint="--task contradiction|translation|related")


CHAT_HELP = (
    "Commands: /contradiction <question>, /translation <question>, /related <question>, "
    "/direct <question> (ask for a task), /search <text>, /k <n>, /sources on|off, /quit"
)
CHAT_TASKS = {
    "/contradiction": "contradiction",
    "/translation": "translation",
    "/related": "related",
    "/direct": "general",
}


def cmd_chat(args, settings):
    rag = load(settings)
    print(
        "Bible RAG. Ask in English or Spanish: a direct question, a contradiction "
        "(“Are there paradoxes in the Bible about alcohol?”), a translation comparison "
        "(“How do the versions render Psalm 16:10?”) or about something the Bible never "
        "names (“Should I use a cellphone?”).\n" + CHAT_HELP
    )
    while True:
        try:
            line = input("\n? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if line in ("/quit", "/exit", "quit", "exit", "salir"):
            return
        command, _, rest = line.partition(" ")
        if command in CHAT_TASKS and rest.strip():
            print_answer(rag, rest.strip(), args, CHAT_TASKS[command])
        elif line.startswith("/k "):
            args.k = int(line.split()[1])
        elif line.startswith("/sources "):
            args.show_context = line.endswith("on")
        elif line.startswith("/search "):
            print_sources(rag.search(line[8:], args.k, args.versions), True)
        elif line.startswith("/"):
            print(CHAT_HELP)
        elif line:
            print_answer(
                rag,
                line,
                args,
                hint="/contradiction, /translation or /related <question>",
            )


def cmd_serve(args, settings):
    from .server import ServerConfigError, serve

    try:
        serve(settings, args.host, args.port, args.docs, args.ssl_certfile, args.ssl_keyfile)
    except ServerConfigError as e:
        raise SystemExit(f"Error: {e}") from e


def cmd_eval(args, settings):
    from .evaluation import evaluate, evaluate_agent

    settings.server_url = ""  # measures this machine's models (and the planner's internals)

    def progress(r):
        hits = " ".join(f"{v}:{'✓' if ok else '✗'}" for v, ok in sorted(r["hit"].items()))
        answer = (
            f" | {r['model']}: gold {'✓' if r['cites_gold'] else '✗'}"
            f"{', unverified ' + str(len(r['unverified'])) if r['unverified'] else ''}"
            if "model" in r
            else ""
        )
        print(f"[{r['language']}] {r['question'][:55]:55s} {hits}{answer}", file=sys.stderr)

    if args.agent:  # routing, topics and contradiction sides: retrieval only
        rag = load(settings, llm=False)
        report = evaluate_agent(rag, args.k, lambda line: print(line, file=sys.stderr))
    else:
        rag = load(settings, llm=not args.retrieval_only)
        report = evaluate(rag, args.k, answers=not args.retrieval_only, progress=progress)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
    print(json.dumps(report["summary"], indent=2))


def versions_arg(value: str) -> list[str]:
    versions = [v.strip().lower() for v in value.split(",") if v.strip()]
    if not versions or set(versions) - VERSIONS.keys():
        raise argparse.ArgumentTypeError(f"choose from: {','.join(VERSIONS)}")
    return versions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bible-rag", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func, help_text in (
        ("ask", cmd_ask, "answer a question"),
        ("chat", cmd_chat, "interactive session"),
        ("search", cmd_search, "retrieval only"),
    ):
        p = sub.add_parser(name, help=help_text)
        if name != "chat":
            p.add_argument("question", nargs="*", help="the question (or read from stdin)")
        p.add_argument("-k", type=int, help="passages per translation (per side: contradictions)")
        p.add_argument("--versions", type=versions_arg, help="e.g. kjv or kjv,rv1909")
        p.add_argument("--show-context", action="store_true", help="print the passages' text")
        if name == "ask":
            p.add_argument(
                "--task",
                choices=TASKS,
                help="ask for a task instead of letting the assistant choose",
            )
        p.set_defaults(func=func)
    sub.add_parser("versions", help="Bible texts and licenses").set_defaults(func=cmd_versions)
    p = sub.add_parser("models", help="usable embedders, NLI models and LLMs")
    p.add_argument("--test", action="store_true", help="also load each LLM and generate")
    p.set_defaults(func=cmd_models)
    p = sub.add_parser("ingest", help="update the local indexes (only what changed)")
    p.add_argument("--force", action="store_true", help="rebuild even what is up to date")
    p.add_argument("--embedder", help="embedder spec instead of the chain's, e.g. hf:<model>")
    p.add_argument(
        "--sample",
        action="store_true",
        help="only ~1/4 of the Bible (a quick test; same as BIBLE_RAG_BOOKS=sample)",
    )
    p.set_defaults(func=cmd_ingest)
    p = sub.add_parser("prebuild", help="regenerate storage/prebuilt (shipped with the code)")
    p.add_argument("--force", action="store_true", help="rebuild even what is up to date")
    p.add_argument("--embedder", help="embedder spec instead of the chain's")
    p.set_defaults(func=cmd_prebuild)
    p = sub.add_parser("serve", help="read-only HTTP API (e.g. on a GPU machine)")
    p.add_argument("--host", default="127.0.0.1", help="127.0.0.1 (default) or 0.0.0.0 + key")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--docs", action="store_true", help="interactive API docs (local use)")
    p.add_argument("--ssl-certfile", help="serve HTTPS directly (else use a reverse proxy)")
    p.add_argument("--ssl-keyfile")
    p.set_defaults(func=cmd_serve)
    p = sub.add_parser("eval", help="measure quality")
    p.add_argument("-k", type=int, help="passages per translation")
    p.add_argument("--retrieval-only", action="store_true", help="no answers (no LLM)")
    p.add_argument("--agent", action="store_true", help="the assistant: routing, topics, sides")
    p.add_argument("--output", help="write the full report (JSON)")
    p.set_defaults(func=cmd_eval)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if not args.verbose:
        for noisy in ("httpx", "sentence_transformers", "transformers", "huggingface_hub"):
            logging.getLogger(noisy).setLevel(logging.ERROR)
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    try:
        args.func(args, Settings.from_env())
        return 0
    except (OSError, RuntimeError, ValueError) as e:
        logging.getLogger(__name__).debug("fatal", exc_info=True)
        print(f"Error: {e}", file=sys.stderr)
        return 1
