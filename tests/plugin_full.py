"""A complete plug-in: new vector store, embedder, chat model, chunker and pipeline with zero edits under src/."""

from __future__ import annotations

from typing import Any

from agentic_rag.registry import Registries
from agentic_rag.testing import FakeChatModel, FakeEmbedder, FakeVectorStore

SHARED_STORE = FakeVectorStore()


class ShoutingPipeline:
    name = "shouting"

    async def answer(self, question: str, *, retriever: Any, top_k: int, filter: Any = None) -> Any:
        from agentic_rag.ports import PipelineAnswer

        hits = await retriever.retrieve(question, top_k=top_k, filter=filter)
        text = question.upper()
        return PipelineAnswer(text=text, cited_chunk_ids=tuple(h.chunk.id for h in hits[:1]), retrieved=tuple(hits))


def register(registries: Registries) -> None:
    registries.vector_store.register("memory", lambda s, ctx: SHARED_STORE)
    registries.embedder.register("hashing", lambda s, ctx: FakeEmbedder(s.embedding_dimension))
    registries.chat_model.register("scripted", lambda s, ctx: FakeChatModel(["{}"]))
    registries.answer_pipeline.register("shouting", lambda s, ctx: ShoutingPipeline())
