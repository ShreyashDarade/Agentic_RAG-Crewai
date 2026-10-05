#!/usr/bin/env python3
"""Render docs/benchmark.md's tables from the JSON files in docs/bench/ (so the document cannot drift from the data)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    ("baseline-run1", "baseline (run 1)"),
    ("baseline-run2", "baseline (run 2)"),
    ("after-A-cache-verify", "A: cache the index verification"),
    ("after-B-two-phase-commit", "A+B: two-phase commit marker, 500-row batches"),
    ("before-review-round-1-run1", "commit before review round 1, measured in the same session (run 1)"),
    ("before-review-round-1-run2", "commit before review round 1, measured in the same session (run 2)"),
    ("after-review-round-1-eight-workers-run1", "after review round 1, 8 store workers on Lite (run 1)"),
    ("after-review-round-1-eight-workers-run2", "after review round 1, 8 store workers on Lite (run 2)"),
    (
        "after-review-round-1-one-worker-for-lite-run1",
        "after review round 1, 1 store worker on Lite: **current** (run 1)",
    ),
    (
        "after-review-round-1-one-worker-for-lite-run2",
        "after review round 1, 1 store worker on Lite: **current** (run 2)",
    ),
    ("stages-current", "after review round 1, 1 store worker on Lite: **current** (run 3, with stage timing)"),
    ("experiment-store-on-default-executor-run1", "experiment: store calls on the default executor (run 1)"),
    ("experiment-store-on-default-executor-run2", "experiment: store calls on the default executor (run 2)"),
    ("experiment-lexical-on-default-executor-run1", "experiment: lexical search on the default executor (run 1)"),
    ("experiment-lexical-on-default-executor-run2", "experiment: lexical search on the default executor (run 2)"),
]


def stages() -> str:
    lines = [
        "| Run | Indexed chunks | Query embedding (fake) | Dense search incl. embedding | BM25 | `service.search` end to end |",
        "|---|---|---|---|---|---|",
    ]
    for key, label in RUNS:
        d = json.loads((ROOT / "docs" / "bench" / f"{key}.json").read_text())
        if "stages_ms" in d:
            s = d["stages_ms"]
            lines.append(
                f"| {label} | {s['indexed_chunks']} | {s['embed_query_p50']} ms | {s['dense_search_incl_embed_p50']} ms | "
                f"{s['bm25_p50']} ms | {s['service_search_p50']} ms |"
            )
    return "\n".join(lines) + "\n"


def render() -> str:
    data = {key: json.loads((ROOT / "docs" / "bench" / f"{key}.json").read_text()) for key, _ in RUNS}
    first = data[RUNS[0][0]]
    lines = [
        f"Machine: {first['machine']['cpus']} vCPU, Python {first['machine']['python']}, {first['machine']['platform']}; "
        f"Milvus: {first['machine']['milvus']}. Corpus: {first['corpus']['docs']} synthetic documents, "
        f"{first['corpus']['bytes']:,} bytes, seed {first['corpus']['seed']}.",
        "",
        "| Run | Ingest docs/s | Ingest chunks/s | Search c=1 p50 / p95 ms | c=4 p50 / p95 ms | c=16 p50 / p95 ms | Search req/s (c=1 / 4 / 16) |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, label in RUNS:
        d = data[key]
        s = {r["clients"]: r for r in d["search_warm"]}
        lines.append(
            f"| {label} | {d['ingest_cold']['docs_per_s']} | {d['ingest_cold']['chunks_per_s']} | "
            f"{s[1]['p50_ms']} / {s[1]['p95_ms']} | {s[4]['p50_ms']} / {s[4]['p95_ms']} | {s[16]['p50_ms']} / {s[16]['p95_ms']} | "
            f"{s[1]['throughput_per_s']} / {s[4]['throughput_per_s']} / {s[16]['throughput_per_s']} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.stdout.write(render())
    sys.stdout.write("\nStages (p50, one caller):\n\n" + stages())
