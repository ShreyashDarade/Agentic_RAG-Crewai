"""The retry matrix (framework section 8, ADR-0009) against a fault-injecting transport. No test sleeps."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx
import pytest

from agentic_rag.errors import (
    ClientTimeout,
    ConnectionFailed,
    DocumentNotFound,
    InvalidResponse,
    Overloaded,
    RagStatusError,
)
from agentic_rag.sdk._http import HttpBackend
from agentic_rag.sdk._retry import RetryPolicy, TokenBucket, parse_retry_after

DOC = {
    "id": "doc_x",
    "name": "a.txt",
    "content_sha256": "0" * 64,
    "content_type": ".txt",
    "chunk_count": 1,
    "embedding_model": "m",
}
READY = {"ready": False, "checks": {"vector_store": "unavailable"}}


def problem(code: str, status: int, **extra: Any) -> httpx.Response:
    return httpx.Response(
        status,
        json={"type": "urn:x", "title": "t", "status": status, "code": code, **extra},
        headers={
            "content-type": "application/problem+json",
            **({"retry-after": extra.pop("ra")} if "ra" in extra else {}),
        },
    )


class Harness:
    """Scripted responses/exceptions, a fake clock, recorded sleeps."""

    def __init__(self, script: list[Any], **policy: Any) -> None:
        self.script = list(script)
        self.requests: list[httpx.Request] = []
        self.sleeps: list[float] = []
        self.draws: list[tuple[float, float]] = []
        self.now = 1000.0

        async def sleep(seconds: float) -> None:
            self.sleeps.append(seconds)
            self.now += seconds

        def uniform(a: float, b: float) -> float:
            self.draws.append((a, b))
            return b  # worst case: the longest wait the jitter allows

        defaults = {"max_retries": 3, "sleep": sleep, "uniform": uniform, "monotonic": lambda: self.now}
        self.policy = RetryPolicy(**{**defaults, **policy})
        transport = httpx.MockTransport(self._handle)
        self.backend = HttpBackend(
            client=httpx.AsyncClient(transport=transport, base_url="http://localhost"),
            owns_client=True,
            api_key="k" * 20,
            timeout=60.0,
            connect_timeout=5.0,
            retry=self.policy,
        )

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, Exception):
            raise item
        return item(request) if callable(item) else item

    def send(self, **kwargs: Any) -> Any:
        return self.backend._send("GET", "/v1/documents/doc_x", **kwargs)


ok = httpx.Response(200, json=DOC)


@pytest.mark.parametrize("idempotent", [True, False])
async def test_never_sent_failures_retry_for_every_call(idempotent: bool) -> None:
    h = Harness([httpx.ConnectError("refused"), httpx.ConnectTimeout("slow"), ok])
    response = await h.send(idempotent=idempotent)
    assert response.status_code == 200
    assert len(h.requests) == 3


@pytest.mark.parametrize("status", [429, 503])
@pytest.mark.parametrize("idempotent", [True, False])
async def test_refused_before_work_retries_for_every_call_and_honours_retry_after(
    status: int, idempotent: bool
) -> None:
    refused = httpx.Response(status, json={"code": "OVERLOADED", "status": status}, headers={"retry-after": "2"})
    h = Harness([refused, ok])
    assert (await h.send(idempotent=idempotent)).status_code == 200
    assert h.sleeps == [2.0]
    assert h.draws == []  # the server's delay replaced our own jitter


async def test_retry_after_beyond_the_cap_is_not_waited_for() -> None:
    refused = httpx.Response(503, json={"code": "OVERLOADED", "status": 503}, headers={"retry-after": "120"})
    h = Harness([refused, ok], retry_after_cap=60.0)
    with pytest.raises(Overloaded):
        await h.send(idempotent=True)
    assert h.sleeps == []
    assert len(h.requests) == 1


@pytest.mark.parametrize(
    "ambiguous", [httpx.RemoteProtocolError("reset"), httpx.ReadError("eof"), httpx.WriteError("broken pipe")]
)
async def test_ambiguous_failures_retry_only_idempotent_calls(ambiguous: Exception) -> None:
    h = Harness([ambiguous, ok])
    assert (await h.send(idempotent=True)).status_code == 200
    h = Harness([ambiguous, ok])
    with pytest.raises(ConnectionFailed):
        await h.send(idempotent=False)
    assert len(h.requests) == 1  # a non-idempotent call is never re-sent after an ambiguous failure


@pytest.mark.parametrize("status", [408, 502, 504])
async def test_ambiguous_statuses_retry_only_idempotent_calls(status: int) -> None:
    h = Harness([httpx.Response(status, json={"code": "DEADLINE_EXCEEDED", "status": status}), ok])
    assert (await h.send(idempotent=True)).status_code == 200
    h = Harness([httpx.Response(status, json={"code": "DEADLINE_EXCEEDED", "status": status}), ok])
    with pytest.raises(Exception) as err:
        await h.send(idempotent=False)
    assert getattr(err.value, "code", None) == "DEADLINE_EXCEEDED"
    assert len(h.requests) == 1


async def test_a_read_timeout_is_never_retried() -> None:
    h = Harness([httpx.ReadTimeout("slow"), ok])
    with pytest.raises(ClientTimeout):
        await h.send(idempotent=True)
    assert len(h.requests) == 1


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (400, "VALIDATION_FAILED"),
        (401, "AUTHENTICATION_FAILED"),
        (404, "NOT_FOUND"),
        (422, "VALIDATION_FAILED"),
        (500, "INTERNAL_ERROR"),
    ],
)
async def test_other_statuses_are_not_retried(status: int, code: str) -> None:
    h = Harness([httpx.Response(status, json={"code": code, "status": status}), ok])
    with pytest.raises(Exception) as err:
        await h.send(idempotent=True)
    assert err.value.code == code  # type: ignore[attr-defined]
    assert len(h.requests) == 1


async def test_exhausted_retries_raise_the_servers_typed_error() -> None:
    h = Harness([httpx.Response(503, json={"code": "OVERLOADED", "status": 503})], max_retries=2)
    with pytest.raises(Overloaded):
        await h.send(idempotent=True)
    assert len(h.requests) == 3


async def test_backoff_is_full_jitter_within_a_capped_exponential_window() -> None:
    h = Harness(
        [httpx.Response(502, json={"code": "DEPENDENCY_ERROR", "status": 502})],
        max_retries=4,
        base_delay=0.5,
        max_delay=3.0,
        bucket_capacity=100,
    )
    with pytest.raises(Exception):
        await h.send(idempotent=True)
    assert h.draws == [(0.0, 0.5), (0.0, 1.0), (0.0, 2.0), (0.0, 3.0)]


async def test_the_retry_budget_stops_a_retry_storm() -> None:
    h = Harness(
        [httpx.Response(503, json={"code": "OVERLOADED", "status": 503})],
        max_retries=10,
        bucket_capacity=2,
        bucket_refill_per_second=0.0,
    )
    with pytest.raises(Overloaded):
        await h.send(idempotent=True)
    assert len(h.requests) == 3  # the first try plus exactly two budgeted retries


async def test_the_total_deadline_spans_all_attempts() -> None:
    h = Harness(
        [httpx.Response(503, json={"code": "OVERLOADED", "status": 503}, headers={"retry-after": "30"})],
        max_retries=10,
        bucket_capacity=100,
    )
    h.backend._timeout = 50.0
    with pytest.raises(Overloaded):
        await h.send(idempotent=True)  # 30 s is affordable once; a second 30 s would pass the 50 s deadline
    assert h.sleeps == [30.0]


async def test_one_request_id_per_logical_call_and_errors_carry_it() -> None:
    h = Harness([httpx.ConnectError("x"), httpx.Response(503, json={"code": "OVERLOADED", "status": 503}), ok])
    await h.send(idempotent=True)
    ids = {r.headers["x-request-id"] for r in h.requests}
    assert len(ids) == 1
    assert next(iter(ids)) != ""
    h = Harness([httpx.Response(404, json={"code": "DOCUMENT_NOT_FOUND", "status": 404})])
    with pytest.raises(DocumentNotFound) as err:
        await h.send(idempotent=True)
    assert err.value.request_id == h.requests[0].headers["x-request-id"]


async def test_a_retried_delete_that_finds_nothing_means_the_first_attempt_worked() -> None:
    gone = httpx.Response(404, json={"code": "DOCUMENT_NOT_FOUND", "status": 404})
    h = Harness([httpx.ReadError("reset after sending"), gone])
    result = await h.backend.delete_document("doc_x")
    assert (result.document_id, result.chunks_deleted) == ("doc_x", 0)
    h = Harness([gone])
    with pytest.raises(DocumentNotFound):  # a first-attempt 404 is a real 404
        await h.backend.delete_document("doc_x")


async def test_path_ids_are_percent_encoded() -> None:
    h = Harness([httpx.Response(404, json={"code": "DOCUMENT_NOT_FOUND", "status": 404})])
    with pytest.raises(DocumentNotFound):
        await h.backend.get_document("a/b?c#d%20")
    assert h.requests[0].url.raw_path == b"/v1/documents/a%2Fb%3Fc%23d%2520"


async def test_unknown_codes_and_non_json_errors_stay_usable() -> None:
    h = Harness([httpx.Response(418, json={"code": "BREWING_COFFEE", "status": 418, "request_id": "abc12345"})])
    with pytest.raises(RagStatusError) as err:
        await h.send(idempotent=True)
    assert (err.value.code, err.value.http_status, err.value.request_id) == ("BREWING_COFFEE", 418, "abc12345")
    h = Harness(
        [httpx.Response(502, text="<html>bad gateway</html>"), httpx.Response(502, text="<html>bad gateway</html>")],
        max_retries=1,
    )
    with pytest.raises(RagStatusError) as err2:
        await h.send(idempotent=True)
    assert err2.value.code == "HTTP_502"
    assert len(h.requests) == 2


async def test_responses_ignore_unknown_fields_but_reject_malformed_bodies() -> None:
    h = Harness([httpx.Response(200, json={**DOC, "added_in_a_newer_server": [1, 2]})])
    assert (await h.backend.get_document("doc_x")).id == "doc_x"
    h = Harness([httpx.Response(200, json={"id": 5})])
    with pytest.raises(InvalidResponse):
        await h.backend.get_document("doc_x")
    h = Harness([httpx.Response(200, text="not json")])
    with pytest.raises(InvalidResponse):
        await h.backend.get_document("doc_x")


async def test_ready_returns_the_report_on_503_instead_of_raising() -> None:
    h = Harness([httpx.Response(503, json=READY)])
    report = await h.backend.ready()
    assert report.ready is False
    assert len(h.requests) == 1  # 503 from /readyz is data, not a reason to retry


async def test_requests_carry_the_api_key_and_json_bodies_omit_unset_fields() -> None:
    h = Harness([lambda r: httpx.Response(200, json={"answer": "a", "grounded": False, "pipeline": "direct"})])
    from agentic_rag.contracts import QueryRequest

    await h.backend.query(QueryRequest(question="q"))
    assert h.requests[0].headers["authorization"] == "Bearer " + "k" * 20
    assert json.loads(h.requests[0].content) == {"question": "q"}


def test_retry_after_parsing_and_policy_validation() -> None:
    assert parse_retry_after("7") == 7.0
    assert parse_retry_after(None) is None
    assert parse_retry_after("soon") is None
    assert parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT", now=lambda: 1445412480.0 - 5) == 5.0
    assert parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT", now=lambda: 1445412480.0 + 5) == 0.0
    with pytest.raises(ValueError):
        RetryPolicy(max_retries=-1)
    ticks: Iterator[float] = iter([0.0, 0.0, 0.0, 0.0, 10.0])
    bucket = TokenBucket(2, 0.1, lambda: next(ticks))
    assert [bucket.try_take(), bucket.try_take(), bucket.try_take(), bucket.try_take()] == [True, True, False, True]
