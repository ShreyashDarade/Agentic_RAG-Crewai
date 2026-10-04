"""Mutation proof for G9: each conformance suite must FAIL on a deliberately broken implementation.

A suite that cannot reject a broken component proves nothing. Each case names the check expected to catch it.
"""

from __future__ import annotations

import inspect
import uuid
from collections.abc import Sequence
from typing import Any

import pytest

from agentic_rag.adapters.chunker_recursive import RecursiveChunker
from agentic_rag.adapters.lexical_bm25 import Bm25Index
from agentic_rag.adapters.parser_text import TextParser
from agentic_rag.adapters.pipeline_direct import DirectPipeline
from agentic_rag.ports import ChatMessage, Chunk, ChunkFilter, Completion, ParsedDocument, ScoredChunk
from agentic_rag.testing import (
    AnswerPipelineContract,
    ChatModelContract,
    ChunkerContract,
    EmbedderContract,
    FakeChatModel,
    FakeEmbedder,
    FakeReranker,
    FakeVectorStore,
    LexicalIndexContract,
    ParserContract,
    RerankerContract,
    VectorStoreContract,
)


async def failures(suite: type) -> set[str]:
    """Names of the suite's checks that fail (skips do not count as failures)."""
    instance = suite()
    failed: set[str] = set()
    for name in dir(instance):
        if not name.startswith("test_"):
            continue
        try:
            result = getattr(instance, name)()
            if inspect.isawaitable(result):
                await result
        except BaseException as exc:  # noqa: BLE001 - pytest.skip raises a BaseException subclass; classified below
            if type(exc).__name__ not in {"Skipped"}:
                failed.add(name)
    return failed


# -- vector store -------------------------------------------------------------------------------------


class WorstFirst(FakeVectorStore):
    async def search(
        self, vector: Sequence[float], *, top_k: int, filter: ChunkFilter | None = None
    ) -> list[ScoredChunk]:
        return list(reversed(await super().search(vector, top_k=top_k, filter=filter)))


class NoUpsertIdempotence(FakeVectorStore):
    async def upsert(
        self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]], *, embedding_model: str
    ) -> None:
        await super().upsert(chunks, vectors, embedding_model=embedding_model)
        for i, (chunk, vec) in enumerate(zip(chunks, vectors, strict=True)):
            self._rows[f"{chunk.id}#dup{len(self._rows)}{i}"] = (chunk, list(vec), embedding_model)


class IgnoresFilter(FakeVectorStore):
    async def search(
        self, vector: Sequence[float], *, top_k: int, filter: ChunkFilter | None = None
    ) -> list[ScoredChunk]:
        return await super().search(vector, top_k=top_k, filter=None)


class InjectableFilter(FakeVectorStore):
    """Treats the filter value as an expression: anything containing ' or ' matches everything."""

    async def search(
        self, vector: Sequence[float], *, top_k: int, filter: ChunkFilter | None = None
    ) -> list[ScoredChunk]:
        if (
            filter
            and filter.document_ids
            and any(" or " in d or d.endswith("\\") for d in filter.document_ids)
        ):
            filter = None
        return await super().search(vector, top_k=top_k, filter=filter)


class SwallowsOutage(FakeVectorStore):
    async def search(
        self, vector: Sequence[float], *, top_k: int, filter: ChunkFilter | None = None
    ) -> list[ScoredChunk]:
        try:
            return await super().search(vector, top_k=top_k, filter=filter)
        except Exception:  # noqa: BLE001 - the defect under test
            return []


class AcceptsAnyDimension(FakeVectorStore):
    async def ensure_ready(self, *, dimension: int, embedding_model: str) -> None:
        return None


class DeleteIgnoresKeep(FakeVectorStore):
    async def delete_document(self, document_id: str, *, keep_chunk_ids: Any = ()) -> int:
        return await super().delete_document(document_id)


class CatalogWithoutMarker(FakeVectorStore):
    async def get_document(self, document_id: str) -> Any:
        for chunk, _, model in self._rows.values():
            if chunk.document_id == document_id:
                from agentic_rag.testing.fakes import _record

                return _record(chunk, model)
        return None


class ScanDropsTheLastBatch(FakeVectorStore):
    async def scan(self, *, batch_size: int) -> Any:
        batches = [b async for b in super().scan(batch_size=batch_size)]
        for batch in batches[:-1]:
            yield batch


STORE_CASES = [
    (WorstFirst, "test_search_returns_nearest_first_and_respects_top_k"),
    (NoUpsertIdempotence, "test_upsert_is_idempotent"),
    (IgnoresFilter, "test_filter_restricts_to_documents"),
    (InjectableFilter, "test_hostile_filter_value_never_matches_other_documents"),
    (SwallowsOutage, "test_unavailable_backend_raises_a_typed_error"),
    (AcceptsAnyDimension, "test_ensure_ready_rejects_a_different_dimension_or_model"),
    (DeleteIgnoresKeep, "test_delete_document_sweeps_all_but_kept"),
    (CatalogWithoutMarker, "test_catalog_requires_the_commit_marker_chunk"),
    (ScanDropsTheLastBatch, "test_scan_yields_every_chunk"),
]


@pytest.mark.parametrize(("broken", "expected"), STORE_CASES, ids=[c[0].__name__ for c in STORE_CASES])
async def test_vector_store_suite_rejects(broken: type[FakeVectorStore], expected: str) -> None:
    class Suite(VectorStoreContract):
        async def create(self) -> Any:
            return broken()

        async def create_failing(self) -> Any:
            return broken(unavailable=True)

    assert expected in await failures(Suite)


# -- embedder ------------------------------------------------------------------------------------------


class WrongDimension(FakeEmbedder):
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [v[:-1] for v in await super().embed_documents(texts)]


class ReversesOrder(FakeEmbedder):
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return list(reversed(await super().embed_documents(texts)))


class SwallowsProviderFailure(FakeEmbedder):
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            return await super().embed_documents(texts)
        except Exception:  # noqa: BLE001 - the defect under test
            return [[0.0] * self.dimension for _ in texts]


EMBEDDER_CASES = [
    (WrongDimension, "test_one_vector_per_input_in_order_with_declared_dimension"),
    (ReversesOrder, "test_one_vector_per_input_in_order_with_declared_dimension"),
    (SwallowsProviderFailure, "test_failure_is_typed"),
]


@pytest.mark.parametrize(("broken", "expected"), EMBEDDER_CASES, ids=[c[0].__name__ for c in EMBEDDER_CASES])
async def test_embedder_suite_rejects(broken: type[FakeEmbedder], expected: str) -> None:
    class Suite(EmbedderContract):
        async def create(self) -> Any:
            return broken(16)

        async def create_failing(self) -> Any:
            return broken(16, fail=True)

    assert expected in await failures(Suite)


# -- lexical index ------------------------------------------------------------------------------------


class NeverRemoves(Bm25Index):
    def remove_document(self, document_id: str) -> None:
        return None


class LexicalIgnoresFilter(Bm25Index):
    def search(self, query: str, *, top_k: int, filter: ChunkFilter | None = None) -> list[ScoredChunk]:
        return super().search(query, top_k=top_k, filter=None)


class WorstFirstLexical(Bm25Index):
    def search(self, query: str, *, top_k: int, filter: ChunkFilter | None = None) -> list[ScoredChunk]:
        return list(reversed(super().search(query, top_k=top_k, filter=filter)))


LEXICAL_CASES = [
    (NeverRemoves, "test_remove_document"),
    (LexicalIgnoresFilter, "test_filter_and_top_k"),
    (WorstFirstLexical, "test_ranks_by_term_match"),
]


@pytest.mark.parametrize(("broken", "expected"), LEXICAL_CASES, ids=[c[0].__name__ for c in LEXICAL_CASES])
async def test_lexical_suite_rejects(broken: type[Bm25Index], expected: str) -> None:
    class Suite(LexicalIndexContract):
        def create(self) -> Any:
            return broken()

    assert expected in await failures(Suite)


# -- reranker, chat model ----------------------------------------------------------------------------------


class InventsCandidates(FakeReranker):
    async def rerank(self, query: str, candidates: Sequence[ScoredChunk], *, top_k: int) -> list[ScoredChunk]:
        from agentic_rag.testing import make_chunks

        return [ScoredChunk(c, 1.0) for c in make_chunks("ghost", ["boo"])] if candidates else []


class IgnoresTopK(FakeReranker):
    async def rerank(self, query: str, candidates: Sequence[ScoredChunk], *, top_k: int) -> list[ScoredChunk]:
        return list(candidates)


@pytest.mark.parametrize("broken", [InventsCandidates, IgnoresTopK])
async def test_reranker_suite_rejects(broken: type[FakeReranker]) -> None:
    class Suite(RerankerContract):
        async def create(self) -> Any:
            return broken()

    assert "test_returns_a_subset_of_candidates_at_most_top_k" in await failures(Suite)


class SilentChat(FakeChatModel):
    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> Completion:
        try:
            return await super().complete(messages, max_tokens=max_tokens)
        except Exception:  # noqa: BLE001 - the defect under test
            return Completion(text="")


async def test_chat_suite_rejects_a_model_that_hides_failures() -> None:
    class Suite(ChatModelContract):
        async def create(self) -> Any:
            return SilentChat(["ok"])

        async def create_failing(self) -> Any:
            return SilentChat(fail=True)

    assert "test_failure_is_typed" in await failures(Suite)


# -- parser, chunker -----------------------------------------------------------------------------------------


class AcceptsGarbage(TextParser):
    def parse(self, data: bytes, *, name: str) -> ParsedDocument:
        return ParsedDocument(text=data.decode("utf-8", errors="replace") or "x")


class UpperCaseExtensions(TextParser):
    extensions = frozenset({".TXT"})


@pytest.mark.parametrize(
    ("broken", "expected"),
    [
        (AcceptsGarbage, "test_corrupt_input_raises_a_typed_error"),
        (UpperCaseExtensions, "test_extensions_are_lowercase_with_dot"),
    ],
)
async def test_parser_suite_rejects(broken: type[TextParser], expected: str) -> None:
    class Suite(ParserContract):
        def create(self) -> Any:
            return broken()

        def sample(self) -> bytes:
            return b"hello world"

        def corrupt(self) -> bytes:
            return b"\xff\xfe\x00 binary"

    assert expected in await failures(Suite)


class RandomIds(RecursiveChunker):
    def chunk(self, doc: ParsedDocument, *, document_id: str, document_name: str) -> list[Chunk]:
        import dataclasses

        return [
            dataclasses.replace(c, id="ch_" + uuid.uuid4().hex)
            for c in super().chunk(doc, document_id=document_id, document_name=document_name)
        ]


class DropsTheTail(RecursiveChunker):
    def chunk(self, doc: ParsedDocument, *, document_id: str, document_name: str) -> list[Chunk]:
        return super().chunk(doc, document_id=document_id, document_name=document_name)[:-1]


class ConstantIds(RecursiveChunker):
    def chunk(self, doc: ParsedDocument, *, document_id: str, document_name: str) -> list[Chunk]:
        import dataclasses

        return [
            dataclasses.replace(c, id="ch_same")
            for c in super().chunk(doc, document_id=document_id, document_name=document_name)
        ]


@pytest.mark.parametrize(
    ("broken", "expected"),
    [
        (RandomIds, "test_ids_are_deterministic_and_unique"),
        (DropsTheTail, "test_all_words_are_covered"),
        (ConstantIds, "test_ids_are_deterministic_and_unique"),
    ],
)
async def test_chunker_suite_rejects(broken: type[RecursiveChunker], expected: str) -> None:
    class Suite(ChunkerContract):
        def create(self) -> Any:
            return broken(max_chars=400, overlap_chars=60)

    assert expected in await failures(Suite)


# -- answer pipeline --------------------------------------------------------------------------------------------


class CitesAnything(DirectPipeline):
    async def answer(self, question: str, *, retriever: Any, top_k: int, filter: Any = None) -> Any:
        from agentic_rag.ports import PipelineAnswer

        real = await super().answer(question, retriever=retriever, top_k=top_k, filter=filter)
        return PipelineAnswer(real.text, (*real.cited_chunk_ids, "ch_invented"), real.retrieved)


class CallsModelWithoutContext(DirectPipeline):
    async def answer(self, question: str, *, retriever: Any, top_k: int, filter: Any = None) -> Any:
        from agentic_rag.ports import PipelineAnswer

        hits = await retriever.retrieve(question, top_k=top_k, filter=filter)
        out = await self._chat.complete([ChatMessage("user", question)], max_tokens=10)
        return PipelineAnswer(out.text or "answer", (), tuple(hits))


class AcceptsGarbageOutput(DirectPipeline):
    async def answer(self, question: str, *, retriever: Any, top_k: int, filter: Any = None) -> Any:
        from agentic_rag.errors import ModelOutputInvalid
        from agentic_rag.ports import PipelineAnswer

        try:
            return await super().answer(question, retriever=retriever, top_k=top_k, filter=filter)
        except ModelOutputInvalid:
            return PipelineAnswer("unknown", (), ())


@pytest.mark.parametrize(
    ("broken", "expected"),
    [
        (CitesAnything, "test_cites_only_what_it_retrieved"),
        (CallsModelWithoutContext, "test_nothing_retrieved_means_no_citations_and_no_model_call"),
        (AcceptsGarbageOutput, "test_unparseable_model_output_is_a_typed_error"),
    ],
)
async def test_pipeline_suite_rejects(broken: type[DirectPipeline], expected: str) -> None:
    class Suite(AnswerPipelineContract):
        def create(self, chat: Any) -> Any:
            return broken(chat)

    assert expected in await failures(Suite)
