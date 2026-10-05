"""Run a retrieval system over a dataset, compare two run files, and gate a candidate against a baseline."""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_rag.evaluation.dataset import Dataset
from agentic_rag.evaluation.metrics import METRIC_DEFINITIONS, evaluate_query
from agentic_rag.evaluation.stats import Comparison, bootstrap_ci, compare_paired

__all__ = ["GateResult", "check", "compare_runs", "load_run", "run_retrieval", "save_run"]

Search = Callable[[str, int], list[str]]


def run_retrieval(dataset: Dataset, search: Search, *, system: str, k: int = 10, seed: int = 0) -> dict[str, Any]:
    """Score ``search`` (query text, k) -> ranked doc ids on every judged query."""
    per_query: dict[str, dict[str, float]] = {}
    for qid, text in dataset.queries.items():
        ranking = list(dict.fromkeys(search(text, k)))  # a repeated id is a system bug; keep first occurrence
        per_query[qid] = evaluate_query(ranking, dataset.qrels[qid], k)
    metrics = sorted(next(iter(per_query.values())))
    summary = {}
    for metric in metrics:
        values = [q[metric] for q in per_query.values()]
        lo, hi = bootstrap_ci(values, seed=seed)
        summary[metric] = {"mean": statistics.fmean(values), "ci95": [lo, hi]}
    return {
        "dataset": dataset.name,
        "system": system,
        "k": k,
        "seed": seed,
        "n_queries": len(per_query),
        "n_documents": len(dataset.corpus),
        "metric_definitions": METRIC_DEFINITIONS,
        "summary": summary,
        "per_query": per_query,
    }


def save_run(run: Mapping[str, Any], path: Path) -> None:
    path.write_text(json.dumps(run, indent=1, sort_keys=True) + "\n")


def load_run(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def _series(run: Mapping[str, Any], metric: str) -> dict[str, float]:
    return {qid: scores[metric] for qid, scores in run["per_query"].items()}


def compare_runs(a: Mapping[str, Any], b: Mapping[str, Any], metric: str, *, seed: int = 0) -> Comparison:
    if a["dataset"] != b["dataset"]:
        raise ValueError("the runs are for different datasets")
    return compare_paired(metric, _series(a, metric), _series(b, metric), seed=seed)


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reasons: list[str]
    comparison: Comparison


def check(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    metric: str,
    *,
    alpha: float = 0.05,
    epsilon: float = 0.005,
    floor: float | None = None,
    seed: int = 0,
) -> GateResult:
    """Regression gate: fail on a significant loss, on a confidently-negative interval, or on a broken absolute floor.

    The thresholds are proposals to calibrate after the first runs; they are not published rules.
    """
    cmp = compare_runs(baseline, candidate, metric, seed=seed)
    reasons = []
    if cmp.p_value < alpha and cmp.mean_delta < 0:
        reasons.append(f"significant regression: delta {cmp.mean_delta:+.4f}, p={cmp.p_value:.4f}")
    if cmp.ci95[1] < -epsilon:
        reasons.append(f"the whole 95% interval of the change is below -{epsilon}: {cmp.ci95}")
    if floor is not None and cmp.mean_b < floor:
        reasons.append(f"{metric} {cmp.mean_b:.4f} is below the floor {floor}")
    return GateResult(not reasons, reasons, cmp)
