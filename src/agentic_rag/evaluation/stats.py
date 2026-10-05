"""Paired statistics for comparing two systems on the same queries (Smucker 2007, Sakai 2014; research section 3).

Seeded and deterministic. Standard library only.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

__all__ = ["Comparison", "bootstrap_ci", "compare_paired", "holm", "paired_randomization_p"]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def bootstrap_ci(
    values: Sequence[float], *, resamples: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap interval of the mean, resampling queries."""
    if not values:
        raise ValueError("no values")
    rng = random.Random(seed)  # noqa: S311 - reproducible resampling, not security
    n = len(values)
    means = sorted(_mean([values[rng.randrange(n)] for _ in range(n)]) for _ in range(resamples))
    lo = means[int((alpha / 2) * resamples)]
    hi = means[min(resamples - 1, int((1 - alpha / 2) * resamples))]
    return lo, hi


def paired_randomization_p(deltas: Sequence[float], *, permutations: int = 10_000, seed: int = 0) -> float:
    """Two-sided sign-flip randomisation test on per-query differences; the statistic is the mean difference."""
    if not deltas:
        raise ValueError("no differences")
    rng = random.Random(seed)  # noqa: S311 - reproducible permutations, not security
    observed = abs(_mean(deltas))
    n = len(deltas)
    extreme = 0
    for _ in range(permutations):
        total = sum(d if rng.random() < 0.5 else -d for d in deltas)
        if abs(total / n) >= observed - 1e-15:
            extreme += 1
    return (extreme + 1) / (permutations + 1)


@dataclass(frozen=True, slots=True)
class Comparison:
    metric: str
    n_queries: int
    mean_a: float
    mean_b: float
    mean_delta: float  # b - a
    ci95: tuple[float, float]
    effect_size: float  # Sakai: |mean d| / sqrt(unbiased variance of d)
    p_value: float

    def to_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "n_queries": self.n_queries,
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "mean_delta": self.mean_delta,
            "ci95": list(self.ci95),
            "effect_size": self.effect_size,
            "p_value": self.p_value,
        }


def compare_paired(
    metric: str,
    a: Mapping[str, float],
    b: Mapping[str, float],
    *,
    resamples: int = 10_000,
    permutations: int = 10_000,
    seed: int = 0,
) -> Comparison:
    """``b`` against ``a`` over the queries both scored. Refuses to compare runs over different query sets."""
    if set(a) != set(b):
        raise ValueError("the two runs scored different queries; a paired comparison needs identical query sets")
    ids = sorted(a)
    deltas = [b[q] - a[q] for q in ids]
    n = len(deltas)
    mean_d = _mean(deltas)
    var = sum((d - mean_d) ** 2 for d in deltas) / (n - 1) if n > 1 else 0.0
    effect = abs(mean_d) / math.sqrt(var) if var > 0 else (0.0 if mean_d == 0 else math.inf)
    return Comparison(
        metric=metric,
        n_queries=n,
        mean_a=_mean([a[q] for q in ids]),
        mean_b=_mean([b[q] for q in ids]),
        mean_delta=mean_d,
        ci95=bootstrap_ci(deltas, resamples=resamples, seed=seed),
        effect_size=effect,
        p_value=paired_randomization_p(deltas, permutations=permutations, seed=seed),
    )


def holm(p_values: Mapping[str, float], *, alpha: float = 0.05) -> dict[str, bool]:
    """Holm-Bonferroni: which hypotheses are rejected when several configurations were compared."""
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    rejected: dict[str, bool] = {}
    stop = False
    for i, (name, p) in enumerate(ordered):
        stop = stop or p > alpha / (m - i)
        rejected[name] = not stop
    return rejected
