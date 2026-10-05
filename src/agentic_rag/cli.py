"""Command line: ``agentic-rag eval run|compare|check`` and ``agentic-rag serve``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from agentic_rag._version import __version__
from agentic_rag.adapters.lexical_bm25 import Bm25Index
from agentic_rag.evaluation import check, compare_runs, load_dataset, load_run, run_retrieval, save_run
from agentic_rag.evaluation.dataset import Dataset
from agentic_rag.ports import Chunk

__all__ = ["main"]


def _bm25_system(dataset: Dataset, k1: float, b: float) -> tuple[str, object]:
    index = Bm25Index(k1=k1, b=b)
    index.add(
        [
            Chunk(id=doc_id, document_id=doc_id, index=0, text=text, document_name=doc_id)
            for doc_id, text in dataset.corpus.items()
        ]
    )

    def search(query: str, k: int) -> list[str]:
        return [hit.chunk.id for hit in index.search(query, top_k=k)]

    return f"bm25(k1={k1},b={b})", search


def _eval_run(args: argparse.Namespace) -> int:
    try:
        dataset = load_dataset(Path(args.dataset), split=args.split, limit=args.limit)
    except (OSError, ValueError) as exc:
        print(f"cannot read the dataset: {exc}", file=sys.stderr)
        return 2
    if args.system != "bm25":
        print(f"unknown system {args.system!r}; available: bm25", file=sys.stderr)
        return 2
    name, search = _bm25_system(dataset, args.bm25_k1, args.bm25_b)
    run = run_retrieval(dataset, search, system=name, k=args.k, seed=args.seed)  # type: ignore[arg-type]
    if args.out:
        save_run(run, Path(args.out))
    print(json.dumps({key: run[key] for key in ("dataset", "system", "n_queries", "n_documents", "summary")}, indent=2))
    return 0


def _eval_compare(args: argparse.Namespace) -> int:
    cmp = compare_runs(load_run(Path(args.a)), load_run(Path(args.b)), args.metric, seed=args.seed)
    print(json.dumps(cmp.to_dict(), indent=2))
    return 0


def _eval_check(args: argparse.Namespace) -> int:
    result = check(
        load_run(Path(args.baseline)),
        load_run(Path(args.candidate)),
        args.metric,
        alpha=args.alpha,
        epsilon=args.epsilon,
        floor=args.floor,
        seed=args.seed,
    )
    print(json.dumps({"passed": result.passed, "reasons": result.reasons, **result.comparison.to_dict()}, indent=2))
    return 0 if result.passed else 1


def _serve(args: argparse.Namespace) -> int:
    from agentic_rag.server import serve

    serve(args.host, args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentic-rag")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the HTTP server (configuration from AGENTIC_RAG_* variables)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=_serve)

    ev = sub.add_parser("eval", help="measure retrieval quality").add_subparsers(dest="eval_command", required=True)
    run = ev.add_parser("run", help="score a retrieval system on a BEIR-format dataset")
    run.add_argument("--dataset", required=True, help="directory with corpus.jsonl, queries.jsonl, qrels/<split>.tsv")
    run.add_argument("--system", default="bm25")
    run.add_argument("--split", default="test")
    run.add_argument("--k", type=int, default=10)
    run.add_argument("--limit", type=int)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--bm25-k1", type=float, default=1.2)
    run.add_argument("--bm25-b", type=float, default=0.75)
    run.add_argument("--out")
    run.set_defaults(func=_eval_run)

    cmp = ev.add_parser("compare", help="paired comparison of two run files (B against A)")
    cmp.add_argument("a")
    cmp.add_argument("b")
    cmp.add_argument("--metric", default="ndcg@10")
    cmp.add_argument("--seed", type=int, default=0)
    cmp.set_defaults(func=_eval_compare)

    chk = ev.add_parser("check", help="regression gate: exit 1 if the candidate is worse than the baseline")
    chk.add_argument("--baseline", required=True)
    chk.add_argument("--candidate", required=True)
    chk.add_argument("--metric", default="ndcg@10")
    chk.add_argument("--alpha", type=float, default=0.05)
    chk.add_argument("--epsilon", type=float, default=0.005)
    chk.add_argument("--floor", type=float)
    chk.add_argument("--seed", type=int, default=0)
    chk.set_defaults(func=_eval_check)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
