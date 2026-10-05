"""Plain value types exchanged across ports. Standard library only; all frozen."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

__all__ = [
    "ChatMessage",
    "Chunk",
    "ChunkFilter",
    "Completion",
    "DocumentRecord",
    "ParsedDocument",
    "PipelineAnswer",
    "ScoredChunk",
]

Scalar = str | int | float | bool


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """Text extracted from a file. ``pages`` (when present) are the per-page texts of ``text``."""

    text: str
    pages: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Chunk:
    """A unit of retrieval. ``id`` is content-addressed (ADR-0007)."""

    id: str
    document_id: str
    index: int
    text: str
    document_name: str
    page: int | None = None
    metadata: Mapping[str, Scalar] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True, slots=True)
class ChunkFilter:
    """Typed restriction. Adapters translate it; callers never build expression strings."""

    document_ids: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    id: str
    name: str
    content_sha256: str
    content_type: str
    chunk_count: int
    embedding_model: str
    index_version: str = ""


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass(frozen=True, slots=True)
class Completion:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class PipelineAnswer:
    """What an answer pipeline returns: text, the chunk ids it cites, and what it retrieved."""

    text: str
    cited_chunk_ids: tuple[str, ...]
    retrieved: tuple[ScoredChunk, ...]
