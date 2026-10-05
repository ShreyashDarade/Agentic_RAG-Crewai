"""What a plug-in author needs: the ports, the value types and the registries (stable tier).

A plug-in is a module with ``register(registries)``; name it in ``AGENTIC_RAG_PLUGINS`` or expose an entry point
in the group ``agentic_rag.<kind>``. Prove it with ``agentic_rag.testing``.
"""

from agentic_rag.ports import (
    AnswerPipeline,
    ChatMessage,
    ChatModel,
    Chunk,
    Chunker,
    ChunkFilter,
    ChunkScanner,
    Completion,
    DocumentCatalog,
    DocumentParser,
    DocumentRecord,
    Embedder,
    Healthcheck,
    LexicalIndex,
    ParsedDocument,
    PipelineAnswer,
    Reranker,
    Retriever,
    ScoredChunk,
    VectorSearcher,
    VectorWriter,
)
from agentic_rag.registry import KINDS, Registries, Registry

__all__ = [
    "KINDS",
    "AnswerPipeline",
    "ChatMessage",
    "ChatModel",
    "Chunk",
    "ChunkFilter",
    "ChunkScanner",
    "Chunker",
    "Completion",
    "DocumentCatalog",
    "DocumentParser",
    "DocumentRecord",
    "Embedder",
    "Healthcheck",
    "LexicalIndex",
    "ParsedDocument",
    "PipelineAnswer",
    "Registries",
    "Registry",
    "Reranker",
    "Retriever",
    "ScoredChunk",
    "VectorSearcher",
    "VectorWriter",
]
