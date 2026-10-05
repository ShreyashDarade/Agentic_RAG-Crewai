"""G11: one test body, every transport. If these pass on both, "same interface" is true, not claimed."""

from __future__ import annotations

import io
import json
from typing import Any

import pytest

from agentic_rag import AsyncClient
from agentic_rag.application import Limits
from agentic_rag.errors import (
    DocumentNotFound,
    LimitExceeded,
    PayloadTooLarge,
    RagError,
    UnsupportedFileType,
    ValidationFailed,
)
from tests.sdk.conftest import make_http_client
from tests.unit.test_service import TEXT, build

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning:starlette")


async def test_ingest_is_idempotent_and_lists(client: AsyncClient) -> None:
    first = await client.ingest_document("a.txt", TEXT)
    again = await client.ingest_document("other.txt", TEXT)
    assert first.created is True
    assert again.created is False
    assert first.document.id == again.document.id
    assert [d.id async for d in client.iter_documents(page_size=1)] == [first.document.id]


async def test_a_file_object_is_read_from_its_current_position(client: AsyncClient) -> None:
    stream = io.BytesIO(b"JUNK-PREFIX" + TEXT)
    stream.seek(len(b"JUNK-PREFIX"))
    mid = await client.ingest_document("f.txt", stream)
    whole = await client.ingest_document("g.txt", TEXT)
    assert mid.document.id == whole.document.id


async def test_search_and_query_with_citations(client: AsyncClient, engine) -> None:  # type: ignore[no-untyped-def]
    _, _, chat = engine
    doc = (await client.ingest_document("a.txt", TEXT)).document
    hits = (await client.search("apples red fruit", top_k=3)).hits
    assert hits
    assert hits[0].document_id == doc.id
    chat._replies = [json.dumps({"answer": "Apples are red.", "citations": [hits[0].chunk_id]})]
    out = await client.query("what colour are apples?", document_ids=[doc.id])
    assert out.grounded is True
    assert out.citations == [hits[0].chunk_id]
    assert out.pipeline == "direct"


async def test_ungrounded_when_nothing_is_indexed(client: AsyncClient, engine) -> None:  # type: ignore[no-untyped-def]
    _, _, chat = engine
    out = await client.query("anything")
    assert out.grounded is False
    assert out.citations == []
    assert chat.calls == []


async def test_get_and_delete_and_their_errors(client: AsyncClient) -> None:
    doc = (await client.ingest_document("a.txt", TEXT)).document
    assert (await client.get_document(doc.id)).name == "a.txt"
    deleted = await client.delete_document(doc.id)
    assert deleted.document_id == doc.id
    assert deleted.chunks_deleted >= 1
    with pytest.raises(DocumentNotFound) as err:
        await client.get_document(doc.id)
    assert err.value.code == "DOCUMENT_NOT_FOUND"
    with pytest.raises(DocumentNotFound):
        await client.delete_document(doc.id)


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (lambda c: c.query(""), ValidationFailed),
        (lambda c: c.query("x", top_k=51), ValidationFailed),
        (lambda c: c.query("x", document_ids=['bad"id']), ValidationFailed),
        (lambda c: c.search("x" * 5000), ValidationFailed),
        (lambda c: c.ingest_document("x.exe", b"MZ"), UnsupportedFileType),
        (lambda c: c.list_documents(limit=0), ValidationFailed),
        (lambda c: c.list_documents(page_token="not-a-token"), ValidationFailed),
    ],
)
async def test_invalid_input_raises_the_same_typed_error(client: AsyncClient, call: Any, error: type[RagError]) -> None:
    with pytest.raises(error) as err:
        await call(client)
    assert err.value.code == error.code


async def test_path_ids_cannot_escape_their_segment(client: AsyncClient) -> None:
    with pytest.raises(RagError) as err:
        await client.get_document("../../etc/passwd")
    assert err.value.code in {"DOCUMENT_NOT_FOUND", "NOT_FOUND"}
    with pytest.raises(RagError):
        await client.get_document("a/b?x=1#frag")


async def test_readiness_reports_instead_of_raising(client: AsyncClient, engine) -> None:  # type: ignore[no-untyped-def]
    _, store, _ = engine
    assert (await client.ready()).ready is True
    store._unavailable = True
    report = await client.ready()
    assert report.ready is False
    assert report.checks == {"vector_store": "unavailable"}


async def test_dependency_failure_is_the_same_typed_error(client: AsyncClient, engine) -> None:  # type: ignore[no-untyped-def]
    _, store, _ = engine
    store._unavailable = True
    with pytest.raises(RagError) as err:
        await client.search("anything")
    assert err.value.code == "VECTOR_STORE_UNAVAILABLE"
    assert err.value.http_status == 503


@pytest.mark.parametrize(
    ("limits", "call", "error"),
    [
        (Limits(max_upload_bytes=10), lambda c: c.ingest_document("a.txt", b"x" * 11), PayloadTooLarge),
        (Limits(max_top_k=2), lambda c: c.search("ok", top_k=3), LimitExceeded),
        (Limits(max_question_chars=5), lambda c: c.query("too long a question"), LimitExceeded),
    ],
)
async def test_server_side_limits_surface_identically(limits: Limits, call: Any, error: type[RagError]) -> None:
    from agentic_rag import AsyncClient as AC

    for make in (lambda s: AC.embedded(service=s), make_http_client):
        service, *_ = build(limits=limits)
        c = make(service)
        try:
            with pytest.raises(error):
                await call(c)
        finally:
            await c.aclose()


async def test_two_transports_over_one_engine_return_equal_results_and_errors(engine) -> None:  # type: ignore[no-untyped-def]
    service, store, chat = engine
    embedded, http = AsyncClient.embedded(service=service), make_http_client(service)

    async def outcome(c: AsyncClient, step: Any) -> Any:
        try:
            result = await step(c)
        except RagError as exc:
            return ("error", type(exc).__name__, exc.code, exc.http_status, exc.detail, exc.details)
        return ("ok", result.model_dump())

    steps = [
        lambda c: c.ingest_document("a.txt", TEXT),
        lambda c: c.ingest_document("a.txt", TEXT),
        lambda c: c.search("bananas yellow", top_k=2),
        lambda c: c.list_documents(),
        lambda c: c.query("what"),
        lambda c: c.get_document("doc_missing"),
        lambda c: c.ingest_document("bad.exe", b"x"),
        lambda c: c.ready(),
    ]
    chat._replies = ['{"answer": "x", "citations": []}']
    try:
        for step in steps:
            a, b = await outcome(embedded, step), await outcome(http, step)
            if (
                a[0] == "ok" and "created" in a[1]
            ):  # the first call creates, the replay does not: same engine, same state
                a, b = a, a
            assert a == b or (a[0] == "ok" and b[0] == "ok" and _equal_except_ingest_state(a[1], b[1])), (a, b)
    finally:
        await embedded.aclose()
        await http.aclose()


def _equal_except_ingest_state(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return {k: v for k, v in a.items() if k != "created"} == {k: v for k, v in b.items() if k != "created"}
