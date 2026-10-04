"""In-memory BM25 over chunks. Single-process by design: rebuilt from the vector store at start-up."""

from __future__ import annotations

import math
import re
import threading
from collections import Counter, defaultdict
from collections.abc import Sequence

from agentic_rag.ports import Chunk, ChunkFilter, ScoredChunk

__all__ = ["Bm25Index"]

_TOKEN = re.compile(r"\w+", re.UNICODE)


class Bm25Index:
    """Okapi BM25 with an inverted index. Thread-safe: reads and writes share one lock."""

    def __init__(self, *, k1: float = 1.2, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._lock = threading.RLock()
        self._chunks: dict[str, Chunk] = {}
        self._lengths: dict[str, int] = {}
        self._postings: dict[str, dict[str, int]] = defaultdict(dict)
        self._by_document: dict[str, set[str]] = defaultdict(set)
        self._total_length = 0

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return _TOKEN.findall(text.lower())

    def add(self, chunks: Sequence[Chunk]) -> None:
        with self._lock:
            for chunk in chunks:
                if chunk.id in self._chunks:
                    self._remove_chunk(chunk.id)
                tokens = self._tokens(chunk.text)
                self._chunks[chunk.id] = chunk
                self._lengths[chunk.id] = len(tokens)
                self._total_length += len(tokens)
                self._by_document[chunk.document_id].add(chunk.id)
                for term, tf in Counter(tokens).items():
                    self._postings[term][chunk.id] = tf

    def remove_document(self, document_id: str) -> None:
        with self._lock:
            for chunk_id in list(self._by_document.get(document_id, ())):
                self._remove_chunk(chunk_id)

    def _remove_chunk(self, chunk_id: str) -> None:
        chunk = self._chunks.pop(chunk_id)
        self._total_length -= self._lengths.pop(chunk_id)
        self._by_document[chunk.document_id].discard(chunk_id)
        if not self._by_document[chunk.document_id]:
            del self._by_document[chunk.document_id]
        for term in set(self._tokens(chunk.text)):
            postings = self._postings.get(term)
            if postings is not None:
                postings.pop(chunk_id, None)
                if not postings:
                    del self._postings[term]

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filter: ChunkFilter | None = None,
    ) -> list[ScoredChunk]:
        allowed = set(filter.document_ids) if filter and filter.document_ids else None
        with self._lock:
            n = len(self._chunks)
            if n == 0:
                return []
            avg = self._total_length / n or 1.0
            scores: dict[str, float] = defaultdict(float)
            for term in set(self._tokens(query)):
                postings = self._postings.get(term)
                if not postings:
                    continue
                idf = math.log(1.0 + (n - len(postings) + 0.5) / (len(postings) + 0.5))
                for chunk_id, tf in postings.items():
                    if allowed is not None and self._chunks[chunk_id].document_id not in allowed:
                        continue
                    norm = tf + self._k1 * (1 - self._b + self._b * self._lengths[chunk_id] / avg)
                    scores[chunk_id] += idf * tf * (self._k1 + 1) / norm
            ranked = sorted(scores, key=lambda cid: (-scores[cid], cid))[:top_k]
            return [ScoredChunk(self._chunks[cid], scores[cid]) for cid in ranked]
