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
]


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
