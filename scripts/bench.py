#!/usr/bin/env python3
"""Benchmark harness: realistic synthetic corpus, fixed seed, warm and cold paths, several client counts.

Measures the *engine* (service + Milvus + BM25 + fusion): embeddings and the chat model are the in-memory fakes, so
no network or paid API is involved and model latency is deliberately excluded. Milvus is whatever
AGENTIC_RAG_MILVUS_URI points at (a standalone server) or a Milvus Lite file (embedded, single-process).

    python scripts/bench.py [--docs 200] [--queries 300] [--seed 1] [--clients 1 4 16] [--profile] [--out FILE]
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import json
import os
import platform
import pstats
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag.config import load_settings  # noqa: E402
from agentic_rag.container import build_container, default_registries  # noqa: E402
from agentic_rag.contracts import QueryRequest, SearchRequest  # noqa: E402
from agentic_rag.testing import FakeChatModel, FakeEmbedder  # noqa: E402


def corpus(n_docs: int, seed: int) -> tuple[list[tuple[str, bytes]], list[str]]:
    """Zipf-distributed vocabulary, 40 paragraphs of ~60 words per document (about 15 KB each)."""
    rng = random.Random(seed)  # noqa: S311 - reproducible benchmark data, not security
    vocab = [f"w{i:04d}{''.join(rng.choices('abcdefghij', k=rng.randint(2, 6)))}" for i in range(5000)]
    weights = [1.0 / (rank + 1) for rank in range(len(vocab))]
    docs = []
    for d in range(n_docs):
        paragraphs = [" ".join(rng.choices(vocab, weights, k=60)) + "." for _ in range(40)]
        docs.append((f"doc{d:04d}.txt", "\n\n".join(paragraphs).encode()))
    queries = [" ".join(rng.choices(vocab, weights, k=4)) for _ in range(2000)]
    return docs, queries


def pct(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(p / 100 * len(ordered)))]


async def timed(coro_factory, n: int, clients: int) -> dict[str, float]:  # type: ignore[no-untyped-def]
    latencies: list[float] = []
    sem = asyncio.Semaphore(clients)

    async def one(i: int) -> None:
        async with sem:
            t = time.perf_counter()
            await coro_factory(i)
            latencies.append((time.perf_counter() - t) * 1000)

    start = time.perf_counter()
    await asyncio.gather(*(one(i) for i in range(n)))
    wall = time.perf_counter() - start
    return {
        "clients": clients,
        "n": n,
        "throughput_per_s": round(n / wall, 1),
        "p50_ms": round(pct(latencies, 50), 2),
        "p95_ms": round(pct(latencies, 95), 2),
        "p99_ms": round(pct(latencies, 99), 2),
        "mean_ms": round(statistics.fmean(latencies), 2),
    }


async def run(args: argparse.Namespace) -> dict[str, object]:
    workdir = tempfile.mkdtemp(prefix="bench_")
    uri = os.environ.get("AGENTIC_RAG_MILVUS_URI") or os.path.join(workdir, "bench.db")
    env = {
        "AGENTIC_RAG_API_KEYS": "k" * 20,
        "AGENTIC_RAG_MILVUS_URI": uri,
        "AGENTIC_RAG_MILVUS_COLLECTION": "bench",
        "AGENTIC_RAG_EMBEDDER": "fake",
        "AGENTIC_RAG_CHAT_MODEL": "fake",
        "AGENTIC_RAG_EMBEDDING_DIMENSION": "64",
        "AGENTIC_RAG_MAX_CHUNKS_PER_DOCUMENT": "5000",
        "AGENTIC_RAG_MAX_CONCURRENT_INGESTS": str(args.ingest_clients),
    }
    regs = default_registries()
    regs.embedder.register("fake", lambda s, ctx: FakeEmbedder(s.embedding_dimension))
    chat = FakeChatModel(['{"answer": "ok", "citations": []}'])
    regs.chat_model.register("fake", lambda s, ctx: chat)
    container = await build_container(load_settings(env), regs)
    svc = container.service
    docs, queries = corpus(args.docs, args.seed)
    result: dict[str, object] = {
        "machine": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpus": os.cpu_count(),
            "milvus": "standalone" if uri.startswith("http") else "milvus-lite (embedded, single process)",
        },
        "corpus": {"docs": len(docs), "bytes": sum(len(b) for _, b in docs), "seed": args.seed},
    }
    try:
        while not (await svc.ready()).ready:  # wait for the lexical index hydration
            await asyncio.sleep(0.05)
        profiler = cProfile.Profile() if args.profile else None
        if profiler:
            profiler.enable()
        t = time.perf_counter()
        results = []
        sem = asyncio.Semaphore(args.ingest_clients)

        async def ingest(item: tuple[str, bytes]) -> None:
            async with sem:
                results.append(await svc.ingest_document(*item))

        await asyncio.gather(*(ingest(d) for d in docs))
        wall = time.perf_counter() - t
        chunks = sum(r.chunks_indexed for r in results)
        result["ingest_cold"] = {
            "clients": args.ingest_clients,
            "docs_per_s": round(len(docs) / wall, 2),
            "chunks_per_s": round(chunks / wall, 1),
            "chunks": chunks,
            "wall_s": round(wall, 2),
        }
        if profiler:
            profiler.disable()
            stats = pstats.Stats(profiler)
            stats.sort_stats("cumulative")
            result["ingest_profile_top"] = _top(stats)
        # warm paths
        await timed(lambda i: svc.search(SearchRequest(query=queries[i], top_k=8)), 50, 4)  # warm-up
        result["search_warm"] = [
            await timed(lambda i: svc.search(SearchRequest(query=queries[i % len(queries)], top_k=8)), args.queries, c)
            for c in args.clients
        ]
        result["query_warm"] = [
            await timed(lambda i: svc.query(QueryRequest(question=queries[i % len(queries)])), args.queries, c)
            for c in args.clients
        ]
        if args.profile:
            profiler = cProfile.Profile()
            profiler.enable()
            await timed(lambda i: svc.search(SearchRequest(query=queries[i], top_k=8)), args.queries, 4)
            profiler.disable()
            stats = pstats.Stats(profiler)
            stats.sort_stats("cumulative")
            result["search_profile_top"] = _top(stats)
    finally:
        await container.aclose()
    return result


def _top(stats: pstats.Stats, n: int = 12) -> list[str]:
    rows = []
    for (filename, line, name), (_, ncalls, tottime, cumtime, _) in sorted(
        stats.stats.items(),
        key=lambda kv: -kv[1][3],  # type: ignore[attr-defined]
    )[:60]:
        if "agentic_rag" in filename or "pymilvus" in filename or "milvus_lite" in filename:
            rows.append(
                f"{cumtime:7.2f}s cum {tottime:6.2f}s self {ncalls:7d} calls  {Path(filename).name}:{line}({name})"
            )
        if len(rows) >= n:
            break
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", type=int, default=200)
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--clients", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--ingest-clients", type=int, default=4)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    result = asyncio.run(run(args))
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
