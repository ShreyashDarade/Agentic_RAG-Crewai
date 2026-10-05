"""Reciprocal Rank Fusion (Cormack et al., SIGIR 2009; research section 3)."""

from __future__ import annotations

from collections.abc import Sequence

from agentic_rag.ports import ScoredChunk

__all__ = ["reciprocal_rank_fusion"]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[ScoredChunk]], *, k: int = 60, top_k: int | None = None
) -> list[ScoredChunk]:
    """Fuse ranked lists on chunk id using 1-based ranks. Ties break by earliest best rank, then id."""
    if k <= 0:
        raise ValueError("k must be positive")
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    chosen: dict[str, ScoredChunk] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            cid = hit.chunk.id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in best_rank or rank < best_rank[cid]:
                best_rank[cid] = rank
                chosen[cid] = hit
    ordered = sorted(scores, key=lambda cid: (-scores[cid], best_rank[cid], cid))
    fused = [ScoredChunk(chosen[cid].chunk, scores[cid]) for cid in ordered]
    return fused[:top_k] if top_k is not None else fused
