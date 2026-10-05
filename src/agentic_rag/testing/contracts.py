"""Conformance suites: one per port, run against every implementation (framework section 7, rule L).

Subclass a suite, implement ``create`` (and, to exercise failure behaviour, ``create_failing``),
and let pytest collect the subclass::

    class TestMyStore(VectorStoreContract):
        async def create(self):
            return MyStore(...)

A suite that cannot be satisfied by a broken implementation is worthless, so each check here is
exercised against deliberately broken implementations in ``tests/conformance`` (mutation proof).
"""

from __future__ import annotations

import contextlib
import hashlib
import math
from collections.abc import Awaitable, Callable
from typing import Any, NoReturn

from agentic_rag.errors import (
    DocumentEmpty,
    DocumentParseFailed,
    EmbeddingFailed,
    IndexIncompatible,
    ModelFailed,
    ModelOutputInvalid,
    RagError,
    UpstreamRateLimited,
    ValidationFailed,
    VectorStoreError,
    VectorStoreUnavailable,
)
from agentic_rag.ports import (
    ChatMessage,
    Chunk,
    ChunkFilter,
    ParsedDocument,
    ScoredChunk,
)

__all__ = [
    "AnswerPipelineContract",
    "ChatModelContract",
    "ChunkerContract",
    "EmbedderContract",
    "LexicalIndexContract",
    "ParserContract",
    "RerankerContract",
    "VectorStoreContract",
    "make_chunks",
]

DIM = 8


def _skip(reason: str) -> NoReturn:
    import pytest

    pytest.skip(reason)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _vec(*hot: int, dim: int = DIM) -> list[float]:
    v = [0.0] * dim
    for i in hot:
        v[i] = 1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def make_chunks(document_id: str, texts: list[str], *, name: str = "doc.txt") -> list[Chunk]:
    """Chunks whose chunk 0 carries the document record metadata."""
    sha = hashlib.sha256(document_id.encode()).hexdigest()
    return [
        Chunk(
            id=f"ch_{document_id}_{i}",
            document_id=document_id,
            index=i,
            text=text,
            document_name=name,
            metadata={"content_sha256": sha, "content_type": ".txt", "chunk_count": len(texts)},
        )
        for i, text in enumerate(texts)
    ]


class VectorStoreContract:
    """A store implementing ``VectorWriter``, ``VectorSearcher``, ``DocumentCatalog`` and ``ChunkScanner``."""

    model = "model-a"

    async def create(self) -> Any:
        raise NotImplementedError

    async def create_failing(self) -> Any | None:
        """A store whose backend is unreachable. Return ``None`` if that cannot be simulated."""
        return None

    async def _loaded(self) -> Any:
        store = await self.create()
        await store.ensure_ready(dimension=DIM, embedding_model=self.model)
        a = make_chunks("docA", ["alpha one", "alpha two", "alpha three"])
        b = make_chunks("docB", ["beta one", "beta two"], name="b.txt")
        vecs_a = [_vec(0), _vec(0, 1), _vec(0, 2)]
        vecs_b = [_vec(5), _vec(5, 6)]
        await store.upsert(a[1:] + a[:1], vecs_a[1:] + vecs_a[:1], embedding_model=self.model)
        await store.upsert(b[1:] + b[:1], vecs_b[1:] + vecs_b[:1], embedding_model=self.model)
        return store

    async def test_search_returns_nearest_first_and_respects_top_k(self) -> None:
        store = await self._loaded()
        hits = await store.search(_vec(0), top_k=2)
        assert hits[0].chunk.id == "ch_docA_0"
        assert len(hits) == 2
        assert all(isinstance(h, ScoredChunk) for h in hits)
        assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)

    async def test_upsert_is_idempotent(self) -> None:
        store = await self._loaded()
        before = len(await store.search(_vec(0), top_k=50))
        a = make_chunks("docA", ["alpha one", "alpha two", "alpha three"])
        await store.upsert(a, [_vec(0), _vec(0, 1), _vec(0, 2)], embedding_model=self.model)
        assert len(await store.search(_vec(0), top_k=50)) == before == 5

    async def test_filter_restricts_to_documents(self) -> None:
        store = await self._loaded()
        hits = await store.search(_vec(5), top_k=10, filter=ChunkFilter(document_ids=("docB",)))
        assert hits and {h.chunk.document_id for h in hits} == {"docB"}

    async def test_hostile_filter_value_never_matches_other_documents(self) -> None:
        store = await self._loaded()
        payloads = ['docA" or document_id != "', "docA' or 1==1 or '", 'x"] or id != ["', "docA\\"]
        for payload in payloads:
            try:
                hits = await store.search(_vec(0), top_k=10, filter=ChunkFilter(document_ids=(payload,)))
            except ValidationFailed:
                continue
            assert hits == [], f"filter value {payload!r} matched chunks"

    async def test_filter_accepts_several_documents(self) -> None:
        store = await self._loaded()
        hits = await store.search(_vec(0, 5), top_k=10, filter=ChunkFilter(document_ids=("docA", "docB")))
        assert {h.chunk.document_id for h in hits} == {"docA", "docB"}
        assert len(hits) == 5
        reversed_order = await store.search(_vec(0, 5), top_k=10, filter=ChunkFilter(document_ids=("docB", "docA")))
        assert {h.chunk.document_id for h in reversed_order} == {"docA", "docB"}

    async def test_a_filtered_search_still_returns_top_k(self) -> None:
        store = await self._loaded()
        only_a = ChunkFilter(document_ids=("docA",))
        assert len(await store.search(_vec(0), top_k=3, filter=only_a)) == 3  # docA has exactly three chunks
        assert len(await store.search(_vec(0), top_k=2, filter=only_a)) == 2
        assert len(await store.search(_vec(0), top_k=50)) == 5

    async def test_upsert_replaces_an_existing_chunk(self) -> None:
        store = await self._loaded()
        changed = Chunk(id="ch_docA_1", document_id="docA", index=1, text="alpha TWO REPLACED", document_name="doc.txt")
        await store.upsert([changed], [_vec(7)], embedding_model=self.model)
        hits = await store.search(_vec(7), top_k=1)
        assert hits[0].chunk.id == "ch_docA_1"
        assert hits[0].chunk.text == "alpha TWO REPLACED"  # the old text and vector are gone, not kept beside it
        assert len(await store.search(_vec(0), top_k=50)) == 5

    async def test_every_field_survives_a_round_trip(self) -> None:
        store = await self.create()
        await store.ensure_ready(dimension=DIM, embedding_model=self.model)
        meta: dict[str, str | int | float | bool] = {
            "content_sha256": "ab" * 32,
            "content_type": ".pdf",
            "chunk_count": 1,
            "s": "x",
            "n": 3,
            "f": 0.5,
            "b": True,
        }
        chunk = Chunk(
            id="ch_docP_0",
            document_id="docP",
            index=0,
            text="pages matter",
            document_name="paper.pdf",
            page=7,
            metadata=meta,
        )
        plain = Chunk(id="ch_docQ_0", document_id="docQ", index=0, text="no page here", document_name="q.txt")
        await store.upsert([chunk, plain], [_vec(3), _vec(4)], embedding_model=self.model)
        found = (await store.search(_vec(3), top_k=1))[0].chunk
        assert (found.id, found.document_id, found.index, found.text) == ("ch_docP_0", "docP", 0, "pages matter")
        assert (found.document_name, found.page) == ("paper.pdf", 7)
        assert dict(found.metadata) == meta
        assert (await store.search(_vec(4), top_k=1))[0].chunk.page is None
        record = await store.get_document("docP")
        assert record is not None
        assert (record.name, record.content_type, record.chunk_count, record.content_sha256) == (
            "paper.pdf",
            ".pdf",
            1,
            "ab" * 32,
        )
        scanned = {c.id: c async for batch in store.scan(batch_size=10) for c in batch}
        assert scanned["ch_docP_0"].page == 7 and scanned["ch_docP_0"].document_name == "paper.pdf"
        assert dict(scanned["ch_docP_0"].metadata) == meta

    async def test_hostile_ids_are_data_for_get_and_delete_too(self) -> None:
        store = await self._loaded()
        for payload in ['docA" or document_id != "', "docA' or 1==1 or '", 'x"] or id != ["', "docA\\", "*", ""]:
            with contextlib.suppress(ValidationFailed):  # refusing a hostile id outright is as good as finding nothing
                assert await store.get_document(payload) is None, f"get_document({payload!r}) found something"
            with contextlib.suppress(ValidationFailed):
                assert await store.delete_document(payload) == 0, f"delete_document({payload!r}) deleted something"
        assert len(await store.search(_vec(0), top_k=50)) == 5  # nothing was deleted by any of them

    async def test_delete_document_sweeps_all_but_kept(self) -> None:
        store = await self._loaded()
        deleted = await store.delete_document("docA", keep_chunk_ids={"ch_docA_0"})
        assert deleted == 2
        remaining = {h.chunk.id for h in await store.search(_vec(0), top_k=50)}
        assert "ch_docA_0" in remaining and "ch_docA_1" not in remaining
        assert await store.delete_document("docA") == 1
        assert await store.get_document("docA") is None

    async def test_catalog_requires_the_commit_marker_chunk(self) -> None:
        store = await self.create()
        await store.ensure_ready(dimension=DIM, embedding_model=self.model)
        chunks = make_chunks("docC", ["c0", "c1"])
        await store.upsert(chunks[1:], [_vec(1)], embedding_model=self.model)
        assert await store.get_document("docC") is None
        await store.upsert(chunks[:1], [_vec(2)], embedding_model=self.model)
        record = await store.get_document("docC")
        assert record is not None and record.chunk_count == 2 and record.embedding_model == self.model

    async def test_list_documents_is_stable_and_pages(self) -> None:
        store = await self._loaded()
        first = await store.list_documents(offset=0, limit=1)
        rest = await store.list_documents(offset=1, limit=5)
        assert [d.id for d in first + rest] == ["docA", "docB"]

    async def test_scan_yields_every_chunk(self) -> None:
        store = await self._loaded()
        seen: list[str] = []
        async for batch in store.scan(batch_size=2):
            assert len(batch) <= 2
            seen.extend(c.id for c in batch)
        assert sorted(seen) == sorted(["ch_docA_0", "ch_docA_1", "ch_docA_2", "ch_docB_0", "ch_docB_1"])

    async def test_ensure_ready_rejects_a_different_dimension_or_model(self) -> None:
        store = await self._loaded()
        await store.ensure_ready(dimension=DIM, embedding_model=self.model)  # idempotent
        for kwargs in (
            {"dimension": DIM + 1, "embedding_model": self.model},
            {"dimension": DIM, "embedding_model": "model-b"},
        ):
            try:
                await store.ensure_ready(**kwargs)
            except IndexIncompatible:
                continue
            raise AssertionError(f"ensure_ready accepted {kwargs}")

    async def test_unavailable_backend_raises_a_typed_error(self) -> None:
        store = await self.create_failing()
        if store is None:
            _skip("this implementation cannot simulate an unavailable backend")

        async def drain() -> None:
            async for _ in store.scan(batch_size=1):
                pass

        calls: list[Callable[[], Awaitable[Any]]] = [
            lambda: store.search(_vec(0), top_k=1),
            lambda: store.search(_vec(0), top_k=1, filter=ChunkFilter(document_ids=("docA",))),
            lambda: store.get_document("docA"),
            lambda: store.list_documents(offset=0, limit=5),
            lambda: store.upsert(make_chunks("x", ["t"]), [_vec(0)], embedding_model=self.model),
            lambda: store.delete_document("docA"),
            lambda: store.ensure_ready(dimension=DIM, embedding_model=self.model),
            lambda: store.check(),
            drain,
        ]
        for call in calls:
            try:
                await call()
            except (VectorStoreUnavailable, VectorStoreError):
                continue
            raise AssertionError("an unavailable backend did not raise a typed error")


class EmbedderContract:
    async def create(self) -> Any:
        raise NotImplementedError

    async def create_failing(self) -> Any | None:
        return None

    async def test_one_vector_per_input_in_order_with_declared_dimension(self) -> None:
        emb = await self.create()
        texts = ["red apple", "blue sky is wide", "x"]
        vectors = await emb.embed_documents(texts)
        assert len(vectors) == 3
        assert all(len(v) == emb.dimension for v in vectors)
        for text, vector in zip(texts, vectors, strict=True):  # order preserved, independent of batching
            alone = await emb.embed_query(text)
            assert _cosine(vector, alone) > 0.999, f"vector for {text!r} is not the one embedded alone"

    async def test_empty_input_gives_empty_output(self) -> None:
        assert await (await self.create()).embed_documents([]) == []

    async def test_query_vector_has_declared_dimension(self) -> None:
        emb = await self.create()
        assert len(await emb.embed_query("hello")) == emb.dimension
        assert isinstance(emb.model_id, str) and emb.model_id

    async def test_failure_is_typed(self) -> None:
        emb = await self.create_failing()
        if emb is None:
            _skip("cannot simulate a failing provider")
        try:
            await emb.embed_documents(["x"])
        except (EmbeddingFailed, UpstreamRateLimited):
            return
        raise AssertionError("a failing embedder did not raise EmbeddingFailed/UpstreamRateLimited")


class LexicalIndexContract:
    def create(self) -> Any:
        raise NotImplementedError

    def _loaded(self) -> Any:
        index = self.create()
        index.add(make_chunks("d1", ["the quick brown fox", "lazy dogs sleep"]))
        index.add(make_chunks("d2", ["quick quick quick fox jumps"], name="two.txt"))
        return index

    def test_ranks_by_term_match(self) -> None:
        hits = self._loaded().search("quick fox", top_k=5)
        assert hits and hits[0].chunk.document_id == "d2"
        assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)

    def test_no_match_is_empty(self) -> None:
        assert self._loaded().search("zzzz", top_k=5) == []

    def test_filter_and_top_k(self) -> None:
        index = self._loaded()
        best_overall = index.search("quick fox", top_k=1)[0].chunk.document_id
        assert best_overall == "d2"  # so a filter that is ignored would return d2 below
        hits = index.search("quick fox", top_k=1, filter=ChunkFilter(document_ids=("d1",)))
        assert len(hits) == 1
        assert hits[0].chunk.document_id == "d1"
        assert index.search("quick fox", top_k=5, filter=ChunkFilter(document_ids=("nobody",))) == []

    def test_remove_document(self) -> None:
        index = self._loaded()
        index.remove_document("d2")
        assert all(h.chunk.document_id != "d2" for h in index.search("quick", top_k=5))
        index.remove_document("never-existed")  # no error


class RerankerContract:
    async def create(self) -> Any:
        raise NotImplementedError

    async def test_returns_a_subset_of_candidates_at_most_top_k(self) -> None:
        reranker = await self.create()
        chunks = make_chunks("d", ["red apple pie", "blue whale", "apple tart recipe"])
        cands = [ScoredChunk(c, 1.0 - i * 0.1) for i, c in enumerate(chunks)]
        out = await reranker.rerank("apple recipe", cands, top_k=2)
        assert len(out) <= 2
        assert {h.chunk.id for h in out} <= {c.id for c in chunks}
        assert len({h.chunk.id for h in out}) == len(out)

    async def test_empty_candidates(self) -> None:
        assert await (await self.create()).rerank("q", [], top_k=3) == []


class ChatModelContract:
    async def create(self) -> Any:
        raise NotImplementedError

    async def create_failing(self) -> Any | None:
        return None

    async def test_returns_text(self) -> None:
        model = await self.create()
        out = await model.complete([ChatMessage("user", "Say ok.")], max_tokens=16)
        assert isinstance(out.text, str)

    async def test_failure_is_typed(self) -> None:
        model = await self.create_failing()
        if model is None:
            _skip("cannot simulate a failing provider")
        try:
            await model.complete([ChatMessage("user", "x")], max_tokens=4)
        except (ModelFailed, UpstreamRateLimited):
            return
        raise AssertionError("a failing model did not raise ModelFailed/UpstreamRateLimited")


class ParserContract:
    """Provide ``create``, a valid ``sample`` and an unparseable ``corrupt`` payload."""

    sample_name = "sample.txt"
    sample_expect = "hello"

    def create(self) -> Any:
        raise NotImplementedError

    def sample(self) -> bytes:
        raise NotImplementedError

    def corrupt(self) -> bytes:
        raise NotImplementedError

    def empty(self) -> bytes | None:
        """A valid file with no text, or ``None`` if the format cannot express that."""
        return None

    def test_extensions_are_lowercase_with_dot(self) -> None:
        exts = self.create().extensions
        assert exts and all(e.startswith(".") and e == e.lower() for e in exts)

    def test_parses_a_valid_sample(self) -> None:
        doc = self.create().parse(self.sample(), name=self.sample_name)
        assert isinstance(doc, ParsedDocument) and self.sample_expect in doc.text

    def test_corrupt_input_raises_a_typed_error(self) -> None:
        try:
            self.create().parse(self.corrupt(), name=self.sample_name)
        except (DocumentParseFailed, DocumentEmpty):
            return
        except RagError as exc:
            raise AssertionError(f"wrong error class {type(exc).__name__}") from exc
        raise AssertionError("corrupt input was accepted")

    def test_no_text_raises_document_empty(self) -> None:
        data = self.empty()
        if data is None:
            _skip("format cannot be empty")
        try:
            self.create().parse(data, name=self.sample_name)
        except DocumentEmpty:
            return
        raise AssertionError("a document without text was accepted")


class ChunkerContract:
    long_text = " ".join(f"word{i}" for i in range(2000))

    def create(self) -> Any:
        raise NotImplementedError

    def _chunks(self, text: str | None = None, doc: str = "doc_x") -> list[Chunk]:
        parsed = ParsedDocument(text=text or self.long_text)
        return list(self.create().chunk(parsed, document_id=doc, document_name="n.txt"))

    def test_ids_are_deterministic_and_unique(self) -> None:
        a, b = self._chunks(), self._chunks()
        assert [c.id for c in a] == [c.id for c in b]
        assert len({c.id for c in a}) == len(a) > 1

    def test_ids_depend_on_the_document(self) -> None:
        assert {c.id for c in self._chunks(doc="d1")}.isdisjoint({c.id for c in self._chunks(doc="d2")})

    def test_indexes_are_sequential_and_text_nonempty(self) -> None:
        chunks = self._chunks()
        assert [c.index for c in chunks] == list(range(len(chunks)))
        assert all(c.text.strip() for c in chunks)
        assert all(c.document_id == "doc_x" and c.document_name == "n.txt" for c in chunks)

    def test_every_word_is_covered(self) -> None:
        covered = " ".join(c.text for c in self._chunks()).split()
        missing = {f"word{i}" for i in range(2000)} - set(covered)
        assert not missing, f"{len(missing)} words are in no chunk, for example {sorted(missing)[:3]}"

    def test_no_chunk_is_longer_than_the_declared_maximum(self) -> None:
        chunker = self.create()
        assert isinstance(chunker.max_chars, int) and chunker.max_chars > 0
        longest = max(len(c.text) for c in self._chunks())
        assert longest <= chunker.max_chars, f"a chunk has {longest} characters, the maximum is {chunker.max_chars}"
        one_giant_word = "x" * (chunker.max_chars * 3)
        assert all(len(c.text) <= chunker.max_chars for c in self._chunks(one_giant_word))

    def test_version_is_declared(self) -> None:
        assert isinstance(self.create().version, str) and self.create().version


class AnswerPipelineContract:
    """``create(chat)`` returns an ``AnswerPipeline`` that talks to the model only through ``chat``."""

    def create(self, chat: Any) -> Any:
        raise NotImplementedError

    @staticmethod
    def _hits() -> list[ScoredChunk]:
        return [
            ScoredChunk(c, 1.0 - i / 10) for i, c in enumerate(make_chunks("d", ["red apples are fruit", "blue sky"]))
        ]

    @staticmethod
    def _reply(answer: str, citations: list[str]) -> str:
        import json

        return json.dumps({"answer": answer, "citations": citations})

    async def test_cites_only_what_it_retrieved(self) -> None:
        from agentic_rag.testing.fakes import FakeChatModel, FakeRetriever

        chat = FakeChatModel([self._reply("Apples are fruit.", ["ch_d_0"])])
        out = await self.create(chat).answer("what are apples", retriever=FakeRetriever(self._hits()), top_k=5)
        assert out.text and set(out.cited_chunk_ids) <= {h.chunk.id for h in out.retrieved}
        assert out.cited_chunk_ids == ("ch_d_0",)

    async def test_nothing_retrieved_means_no_citations_and_no_model_call(self) -> None:
        from agentic_rag.testing.fakes import FakeChatModel, FakeRetriever

        chat = FakeChatModel([self._reply("made up", ["ch_x"])])
        out = await self.create(chat).answer("anything", retriever=FakeRetriever([]), top_k=5)
        assert out.text and out.cited_chunk_ids == () and out.retrieved == ()
        assert chat.calls == []

    async def test_unparseable_model_output_is_a_typed_error(self) -> None:
        from agentic_rag.testing.fakes import FakeChatModel, FakeRetriever

        chat = FakeChatModel(["definitely not json"])
        try:
            await self.create(chat).answer("q", retriever=FakeRetriever(self._hits()), top_k=5)
        except ModelOutputInvalid:
            return
        raise AssertionError("garbage model output was accepted")

    async def test_model_failure_propagates_typed(self) -> None:
        from agentic_rag.testing.fakes import FakeChatModel, FakeRetriever

        try:
            await self.create(FakeChatModel(fail=True)).answer("q", retriever=FakeRetriever(self._hits()), top_k=5)
        except ModelFailed:
            return
        raise AssertionError("a failing model did not raise ModelFailed")
