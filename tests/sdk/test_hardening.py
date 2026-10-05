"""Review round 1 (SDK): a real total deadline, typed failures only, no leaks, the same ids on both transports."""

from __future__ import annotations

import asyncio
import gc
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import httpx
import pytest

from agentic_rag import AsyncClient, Client
from agentic_rag.errors import (
    ClientTimeout,
    DocumentNotFound,
    RagError,
    RagStatusError,
    UsageError,
    error_from_problem,
)
from agentic_rag.sdk._retry import parse_retry_after
from tests.sdk.test_retries import DOC, Harness, ok, problem

ROOT = Path(__file__).resolve().parents[2]


# -- the timeout is a total deadline (review B5) -----------------------------------------------------------------


async def test_a_server_that_drips_bytes_cannot_stretch_the_deadline() -> None:
    async def dripping_body() -> object:
        for _ in range(100):
            await asyncio.sleep(0.2)
            yield b" "

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/json"}, content=dripping_body())  # type: ignore[arg-type]

    client = AsyncClient.http(
        "http://localhost",
        timeout=1.0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://localhost"),
    )
    started = time.monotonic()
    with pytest.raises(ClientTimeout):
        await client.search("apples")
    assert time.monotonic() - started < 3.0  # per-read timeouts alone would have kept it going for ~20 s
    await client.aclose()


# -- failures are RagError, whatever the server sends (review B8, A16) -------------------------------------------


@pytest.mark.parametrize("value", ["²", "٣", "1e3", "nan", "inf", "-5", "", "9" * 500])
def test_retry_after_never_raises_and_never_returns_nonsense(value: str) -> None:
    seconds = parse_retry_after(value)
    assert seconds is None or (seconds >= 0 and seconds == seconds)


@pytest.mark.parametrize(
    "body",
    [
        {"code": "VALIDATION_FAILED", "status": "abc"},
        {"code": "VALIDATION_FAILED", "status": None},
        {"code": "SOMETHING_NEW", "status": [1]},
        {"code": "OVERLOADED", "status": 503, "retry_after": float("nan")},
        {"code": "OVERLOADED", "status": 503, "retry_after": float("inf")},
        {"code": "OVERLOADED", "status": 503, "retry_after": -3},
        {"code": 7, "status": 500},
        {"code": "X", "status": 10**40},
    ],
)
def test_error_from_problem_is_total(body: dict[str, object]) -> None:
    error = error_from_problem(body, status=500)
    assert isinstance(error, RagError)
    assert error.retry_after is None or (error.retry_after >= 0 and error.retry_after == error.retry_after)
    error.to_problem()  # and it can be serialised again


async def test_a_hostile_error_body_surfaces_as_a_typed_error_from_the_client() -> None:
    h = Harness([httpx.Response(500, json={"code": "X", "status": "abc"})])
    with pytest.raises(RagStatusError):
        await h.backend.get_document("doc_x")
    h = Harness([httpx.Response(503, content=b"", headers=[(b"retry-after", b"\xb2")])], max_retries=1)
    with pytest.raises(RagError):
        await h.backend.get_document("doc_x")


# -- DELETE retry rule (review B8) -------------------------------------------------------------------------------


GONE = httpx.Response(404, json={"code": "DOCUMENT_NOT_FOUND", "status": 404})


@pytest.mark.parametrize(
    "first", [httpx.ConnectError("never sent"), problem("OVERLOADED", 503), problem("RATE_LIMITED", 429)]
)
async def test_a_404_after_a_retry_is_only_success_if_an_earlier_attempt_may_have_reached_the_server(
    first: object,
) -> None:
    h = Harness([first, GONE])
    with pytest.raises(DocumentNotFound):  # the first attempt provably did nothing, so this is a real 404
        await h.backend.delete_document("doc_x")
    assert len(h.requests) == 2


@pytest.mark.parametrize("first", [httpx.ReadError("reset"), problem("DEADLINE_EXCEEDED", 504)])
async def test_a_404_after_an_ambiguous_attempt_means_the_delete_worked(first: object) -> None:
    h = Harness([first, GONE])
    result = await h.backend.delete_document("doc_x")
    assert result.chunks_deleted == 0


# -- ids are the same on both transports (review B8) -------------------------------------------------------------


@pytest.mark.parametrize("hostile", ["", ".", ".."])
async def test_ids_that_urls_would_rewrite_are_not_found_without_a_request(hostile: str) -> None:
    h = Harness([ok])
    with pytest.raises(DocumentNotFound):
        await h.backend.get_document(hostile)
    with pytest.raises(DocumentNotFound):
        await h.backend.delete_document(hostile)
    assert h.requests == []  # nothing for httpx to normalise into a different route


# -- closed clients (review B7) ----------------------------------------------------------------------------------


async def test_an_async_http_client_that_is_closed_raises_a_typed_error() -> None:
    client = AsyncClient.http(
        "http://localhost",
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=DOC)), base_url="http://localhost"
        ),
    )
    await client.aclose()
    with pytest.raises(UsageError):
        await client.get_document("doc_x")


async def test_an_embedded_client_that_is_closed_raises_a_typed_error() -> None:
    from tests.unit.test_service import build

    service, *_ = build()
    client = AsyncClient.embedded(service=service)
    assert (await client.ready()).ready
    await client.aclose()
    with pytest.raises(UsageError):
        await client.ready()


def test_an_api_key_is_judged_by_the_url_the_client_will_really_use() -> None:
    inner = httpx.AsyncClient(base_url="http://example.com")  # plain http to a non-local host
    with pytest.raises(UsageError, match="plain http"):
        AsyncClient.http("https://decoy.invalid", api_key="k" * 20, http_client=inner)


# -- the blocking bridge (review B6) ----------------------------------------------------------------------------


def test_dropped_clients_do_not_leak_threads() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ready": True, "checks": {}})

    before = threading.active_count()
    for _ in range(40):
        c = Client.http(
            "http://localhost",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://localhost"),
        )
        assert c.ready().ready
        del c
    gc.collect()
    for _ in range(100):
        if threading.active_count() <= before + 2:
            break
        time.sleep(0.05)
        gc.collect()
    assert threading.active_count() <= before + 2


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_a_client_used_in_a_forked_child_fails_fast_instead_of_hanging() -> None:
    script = textwrap.dedent(
        """
        import os, sys, httpx
        from agentic_rag import Client
        from agentic_rag.errors import UsageError

        def handler(request):
            return httpx.Response(200, json={"ready": True, "checks": {}})

        c = Client.http("http://localhost", timeout=5, http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://localhost"))
        assert c.ready().ready
        pid = os.fork()
        if pid == 0:
            try:
                c.ready()
                print("child: call succeeded")
            except UsageError:
                print("child: UsageError")
            except BaseException as exc:
                print("child:", type(exc).__name__)
            sys.stdout.flush()
            os._exit(0)
        os.waitpid(pid, 0)
        c.close()
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert out.returncode == 0, out.stderr
    assert "child: UsageError" in out.stdout, out.stdout
