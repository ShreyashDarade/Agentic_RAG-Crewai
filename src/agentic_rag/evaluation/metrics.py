"""Retrieval metrics with their definitions pinned (research section 3: BEIR/TREC practice).

* relevance is a non-negative integer; relevant means grade >= 1; unjudged documents are not relevant;
* nDCG@k uses linear gain (the grade) and a ``log2(rank + 1)`` discount; the ideal ranking sorts all judged grades;
* recall@k = relevant retrieved in the top k / relevant judged; precision@k = relevant in the top k / k;
* MRR@k = 1 / rank of the first relevant document within k (0 if none); hit@k = 1 if one is in the top k.
Standard library only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

__all__ = ["METRIC_DEFINITIONS", "evaluate_query", "hit_at_k", "mrr_at_k", "ndcg_at_k", "precision_at_k", "recall_at_k"]

METRIC_DEFINITIONS = {
    "ndcg": "linear gain, log2(rank+1) discount, ideal = all judged grades sorted",
    "recall": "relevant in top k / relevant judged",
    "precision": "relevant in top k / k",
    "mrr": "1 / rank of first relevant in top k",
    "hit": "any relevant in top k",
    "relevant": "grade >= 1; unjudged = not relevant",
}


def _relevant(qrels: Mapping[str, int]) -> set[str]:
    return {doc for doc, grade in qrels.items() if grade >= 1}


def ndcg_at_k(ranking: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    gains = [max(qrels.get(doc, 0), 0) for doc in ranking[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted((g for g in qrels.values() if g > 0), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranking: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    relevant = _relevant(qrels)
    return len(relevant & set(ranking[:k])) / len(relevant) if relevant else 0.0


def precision_at_k(ranking: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    relevant = _relevant(qrels)
    return sum(1 for doc in ranking[:k] if doc in relevant) / k


def mrr_at_k(ranking: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    relevant = _relevant(qrels)
    for rank, doc in enumerate(ranking[:k], start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def hit_at_k(ranking: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    return 1.0 if _relevant(qrels) & set(ranking[:k]) else 0.0


def evaluate_query(ranking: Sequence[str], qrels: Mapping[str, int], k: int) -> dict[str, float]:
    """All metrics for one query; a ranking must not repeat a document (a repeat is a bug in the system under test)."""
    if len(set(ranking)) != len(ranking):
        raise ValueError("a ranking contains the same document twice")
    return {
        f"ndcg@{k}": ndcg_at_k(ranking, qrels, k),
        f"recall@{k}": recall_at_k(ranking, qrels, k),
        f"precision@{k}": precision_at_k(ranking, qrels, k),
        f"mrr@{k}": mrr_at_k(ranking, qrels, k),
        f"hit@{k}": hit_at_k(ranking, qrels, k),
    }
