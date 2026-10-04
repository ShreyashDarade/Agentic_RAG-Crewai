"""A plug-in that adds a component with zero edits under src/ (open-closed proof)."""

from __future__ import annotations

from agentic_rag.adapters.chunker_recursive import RecursiveChunker
from agentic_rag.registry import Registries


class TinyChunker(RecursiveChunker):
    version = "tiny-1"

    def __init__(self) -> None:
        super().__init__(max_chars=100, overlap_chars=10)


def register(registries: Registries) -> None:
    registries.chunker.register("tiny", TinyChunker)
