"""Milvus vector store: writer, searcher, catalog, scanner and health check in one adapter.

``pymilvus`` is imported only in this package. Blocking client calls run in worker threads.
Milvus errors are mapped to typed errors; anything that is not a Milvus error is a bug and propagates.
"""

from agentic_rag.adapters.milvus.store import MilvusSettings, MilvusStore

__all__ = ["MilvusSettings", "MilvusStore"]
