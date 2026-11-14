"""`bible-rag` command line.

versions          Bible texts, their licenses and sources
models [--test]   which embedders / LLMs are usable here (and which will be used)
ingest [--force]  download the texts and build the indexes (once)
search "text"     retrieval only
ask "question"    one answer with checked citations
chat              interactive session (models load once)
eval              measure retrieval and answer quality
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
    from .rag import BibleRAG

    t0 = time.perf_counter()
    print("Loading models and indexes...", file=sys.stderr, flush=True)
    rag = BibleRAG.from_settings(settings, load_llm=llm)
    used = f"embedder {rag.embedder_spec}" + (f", LLM {rag.chain.active}" if rag.chain else "")
    print(f"Ready in {time.perf_counter() - t0:.0f}s ({used}).", file=sys.stderr, flush=True)
    return rag


def print_sources(hits, full: bool) -> None:
    print("\nSources:")
    for i, h in enumerate(hits, 1):
        page = f"  p.{h.passage.page}" if h.passage.page else ""
        print(f"  [{i}] {h.passage.citation}  score={h.score:.3f}{page}")
        if full:
            print(f"      {h.passage.text}")


def print_answer(rag, question: str, args) -> None:
    answer = rag.ask_stream(question, args.k, args.versions)
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
    print_sources(answer.sources, args.show_context or not g.grounded)
    print(f"\nAnswered by: {answer.model}")
    for note in answer.notes:
        print(f"Note: {note}")


def question_of(args) -> str:
    q = " ".join(args.question).strip() or ("" if sys.stdin.isatty() else sys.stdin.read().strip())
    if not q:
        raise SystemExit('Please ask a question, e.g. bible-rag ask "What is love?"')
    return q


def cmd_versions(args, settings):
    from .index import index_is_current

    for vid, v in VERSIONS.items():
        state = "selected" if vid in settings.versions else "available"
        if any(index_is_current(d, vid) for d in settings.index_dir.glob(f"*/{vid}")):
            state += ", indexed"
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

    for kind, title in (("embedder", "Embedding models"), ("llm", "Language models")):
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


def cmd_ingest(args, settings):
    from .index import build_indexes

    built = build_indexes(settings, force=args.force, spec=args.embedder)
    for version, index in built.items():
        print(f"{version}: {len(index)} passages ({index.manifest['embedder']})")
    if not built:
        print("Indexes already built (use --force to rebuild).")


def cmd_search(args, settings):
    question = question_of(args)
    print_sources(load(settings, llm=False).search(question, args.k, args.versions), True)


def cmd_ask(args, settings):
    question = question_of(args)
    print_answer(load(settings), question, args)


def cmd_chat(args, settings):
    rag = load(settings)
    print(
        "Bible RAG (KJV / RVR1960). Ask in English or Spanish. "
        "Commands: /search <text>, /k <n>, /sources on|off, /quit"
    )
    while True:
        try:
            line = input("\n? ").strip()
        except (EOFError, KeyboardInterrupt):
            return print()
        if line in ("/quit", "/exit", "quit", "exit", "salir"):
            return
        if line.startswith("/k "):
            args.k = int(line.split()[1])
        elif line.startswith("/sources "):
            args.show_context = line.endswith("on")
        elif line.startswith("/search "):
            print_sources(rag.search(line[8:], args.k, args.versions), True)
        elif line.startswith("/"):
            print("Commands: /search <text>, /k <n>, /sources on|off, /quit")
        elif line:
            print_answer(rag, line, args)


def cmd_eval(args, settings):
    from .evaluation import evaluate

    rag = load(settings, llm=not args.retrieval_only)

    def progress(r):
        hits = " ".join(f"{v}:{'✓' if ok else '✗'}" for v, ok in sorted(r["hit"].items()))
        answer = (
            f" | {r['model']}: gold {'✓' if r['cites_gold'] else '✗'}"
            f"{', unverified ' + str(len(r['unverified'])) if r['unverified'] else ''}"
            if "model" in r
            else ""
        )
        print(f"[{r['language']}] {r['question'][:55]:55s} {hits}{answer}", file=sys.stderr)

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
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("versions", help="Bible texts and licenses").set_defaults(func=cmd_versions)
    p = sub.add_parser("models", help="usable embedders / LLMs")
    p.add_argument("--test", action="store_true", help="also load each LLM and generate")
    p.set_defaults(func=cmd_models)
    p = sub.add_parser("ingest", help="build the indexes")
    p.add_argument("--force", action="store_true", help="rebuild existing indexes")
    p.add_argument("--embedder", help="embedder spec instead of the first available, e.g. usem3")
    p.set_defaults(func=cmd_ingest)
    for name, func, has_q in (
        ("search", cmd_search, True),
        ("ask", cmd_ask, True),
        ("chat", cmd_chat, False),
    ):
        p = sub.add_parser(name, help=func.__name__[4:])
        if has_q:
            p.add_argument("question", nargs="*", help="question (or stdin)")
        p.add_argument("-k", type=int, help="passages per version")
        p.add_argument("--versions", type=versions_arg, help="e.g. kjv or kjv,rv1909")
        p.add_argument("--show-context", action="store_true", help="print passage texts")
        p.set_defaults(func=func)
    p = sub.add_parser("eval", help="measure quality")
    p.add_argument("-k", type=int)
    p.add_argument("--retrieval-only", action="store_true")
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
        for noisy in ("httpx", "sentence_transformers", "transformers", "huggingface_hub", "faiss"):
            logging.getLogger(noisy).setLevel(logging.ERROR)
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    try:
        args.func(args, Settings.from_env())
        return 0
    except (OSError, RuntimeError, ValueError) as e:
        logging.getLogger(__name__).debug("fatal", exc_info=True)
        print(f"Error: {e}", file=sys.stderr)
        return 1
