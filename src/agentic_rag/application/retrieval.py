"""Hybrid retrieval: dense + lexical, fused with RRF, optionally reranked."""

from __future__ import annotations

import asyncio

from agentic_rag.application.fusion import reciprocal_rank_fusion
from agentic_rag.application.limits import RetrievalConfig
from agentic_rag.blocking import BlockingPool
from agentic_rag.errors import Overloaded
from agentic_rag.ports import (
    ChunkFilter,
    Embedder,
    LexicalIndex,
    Reranker,
    ScoredChunk,
    VectorSearcher,
)

__all__ = ["HybridRetriever"]


class HybridRetriever:
    """Implements the ``Retriever`` port from narrower ports."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        searcher: VectorSearcher,
        lexical: LexicalIndex | None,
        reranker: Reranker | None,
        config: RetrievalConfig,
    ) -> None:
        self._embedder = embedder
        self._searcher = searcher
        self._lexical = lexical if config.use_lexical else None
        self._reranker = reranker if config.use_reranker else None
        self._config = config
        self._pool = BlockingPool("lexical-search", workers=2, backlog=256, saturated=Overloaded)

    def close(self) -> None:
        self._pool.close()

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filter: ChunkFilter | None = None,
    ) -> list[ScoredChunk]:
        pool = max(self._config.candidate_pool, top_k)
        dense_task = self._dense(query, pool, filter)
        if self._lexical is None:
            dense = await dense_task
            candidates = dense
        else:
            lexical_task = self._pool.run(self._lexical.search, query, top_k=pool, filter=filter)
            dense, lexical = await asyncio.gather(dense_task, lexical_task)
            candidates = reciprocal_rank_fusion([dense, lexical], k=self._config.rrf_k, top_k=pool)
        if self._reranker is not None and candidates:
            return await self._reranker.rerank(query, candidates, top_k=top_k)
        return candidates[:top_k]

    async def _dense(self, query: str, pool: int, filter: ChunkFilter | None) -> list[ScoredChunk]:
        vector = await self._embedder.embed_query(query)
        return await self._searcher.search(vector, top_k=pool, filter=filter)
