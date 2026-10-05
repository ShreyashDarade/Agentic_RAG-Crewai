"""Review round 1 (HTTP layer): authenticate before any work, shed only real work, count errors once."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from agentic_rag.api import ApiConfig, create_app
from tests.conftest import API_KEY
from tests.unit.test_service import build

H = {"Authorization": f"Bearer {API_KEY}"}


def _metric(text: str, name: str) -> float:
    for line in text.splitlines():
        if line.startswith((name + " ", name + "{")):
            return float(line.rsplit(" ", 1)[1])
    raise AssertionError(f"{name} not exposed:\n{text}")


async def _raw(
    app: Any, *, method: str, path: str, headers: list[tuple[bytes, bytes]], root_path: str = ""
) -> tuple[int, int]:
    """Drive the ASGI app directly. Returns (status, number of times the app asked for the request body)."""
    body_reads = 0
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        nonlocal body_reads
        body_reads += 1
        return {"type": "http.request", "body": b"x" * 10, "more_body": True}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "root_path": root_path,
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 1),
        "server": ("testserver", 80),
    }
    await app(scope, receive, send)
    start = next(m for m in sent if m["type"] == "http.response.start")
    return int(start["status"]), body_reads


async def test_unauthenticated_requests_are_rejected_before_the_body_is_read() -> None:
    service, *_ = build()
    app = create_app(config=ApiConfig(api_keys=(API_KEY,)), service=service)
    for path in ("/v1/search", "/v1/documents", "/v1/query"):
        status, reads = await _raw(
            app,
            method="POST",
            path=path,
            headers=[(b"content-type", b"multipart/form-data; boundary=x"), (b"content-length", b"1000")],
        )
        assert status == 401, path
        assert reads == 0, f"{path}: the body was read before the key was checked"


async def test_a_malformed_body_without_a_key_is_401_not_422() -> None:
    service, *_ = build()
    app = create_app(config=ApiConfig(api_keys=(API_KEY,)), service=service)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as client:
        r = await client.post("/v1/search", content=b"{not json", headers={"Content-Type": "application/json"})
        assert r.status_code == 401
        assert r.headers["www-authenticate"] == "Bearer"
        assert r.json()["code"] == "AUTHENTICATION_FAILED"
        assert r.headers["x-request-id"]
        assert (await client.get("/metrics")).status_code == 401
        wrong = await client.post("/v1/search", json={"query": "x"}, headers={"Authorization": "Bearer nope"})
        assert wrong.status_code == 401


async def test_unauthenticated_requests_do_not_consume_in_flight_slots() -> None:
    service, *_ = build()
    gate = asyncio.Event()
    real_search = service.search

    async def slow_search(request):  # type: ignore[no-untyped-def]
        await gate.wait()
        return await real_search(request)

    service.search = slow_search  # type: ignore[method-assign]
    app = create_app(config=ApiConfig(api_keys=(API_KEY,), max_inflight=2), service=service)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as client:
        held = [asyncio.create_task(client.post("/v1/search", headers=H, json={"query": "a"})) for _ in range(2)]
        await asyncio.sleep(0.1)
        anonymous = [await client.post("/v1/search", json={"query": "a"}) for _ in range(5)]
        assert {r.status_code for r in anonymous} == {401}  # not 503: they were never counted
        overflow = await client.post("/v1/search", headers=H, json={"query": "a"})
        assert overflow.status_code == 503  # the bound still applies to authenticated work
        gate.set()
        assert {r.status_code for r in await asyncio.gather(*held)} == {200}


@pytest.mark.parametrize("root_path", ["", "/rag"])
@pytest.mark.parametrize("path_includes_root", [False, True])
async def test_load_shedding_works_behind_a_root_path(root_path: str, path_includes_root: bool) -> None:
    # uvicorn --root-path puts the prefix into scope["path"]; the ASGI spec says it should not. Both must shed.
    service, *_ = build()
    gate = asyncio.Event()
    real_search = service.search

    async def slow_search(request):  # type: ignore[no-untyped-def]
        await gate.wait()
        return await real_search(request)

    service.search = slow_search  # type: ignore[method-assign]
    app = create_app(config=ApiConfig(api_keys=(API_KEY,), max_inflight=2), service=service)
    path = (root_path if path_includes_root else "") + "/v1/search"
    headers = [
        (b"authorization", f"Bearer {API_KEY}".encode()),
        (b"content-type", b"application/json"),
    ]

    async def call() -> int:
        body = b'{"query": "apples"}'
        sent: list[dict[str, Any]] = []
        chunks = iter([{"type": "http.request", "body": body, "more_body": False}])

        async def receive() -> dict[str, Any]:
            try:
                return next(chunks)
            except StopIteration:
                await asyncio.sleep(3600)
                return {"type": "http.disconnect"}

        async def send(message: dict[str, Any]) -> None:
            sent.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "root_path": root_path,
            "query_string": b"",
            "headers": [*headers, (b"content-length", str(len(body)).encode())],
            "client": ("127.0.0.1", 1),
            "server": ("testserver", 80),
        }
        await app(scope, receive, send)
        return int(next(m for m in sent if m["type"] == "http.response.start")["status"])

    tasks = [asyncio.create_task(call()) for _ in range(6)]
    await asyncio.sleep(0.3)
    gate.set()
    codes = sorted(await asyncio.gather(*tasks))
    assert codes == [200, 200, 503, 503, 503, 503]


async def test_an_oversized_body_is_counted_once_as_payload_too_large() -> None:
    service, *_ = build()
    app = create_app(config=ApiConfig(api_keys=(API_KEY,), max_upload_bytes=1000), service=service)
    big = b"a" * 200_000
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as client:
        declared = await client.post("/v1/documents", headers=H, files={"file": ("a.txt", big)})
        assert declared.status_code == 413
        assert declared.json()["code"] == "PAYLOAD_TOO_LARGE"
        metrics = (await client.get("/metrics", headers=H)).text
    assert _metric(metrics, 'agentic_rag_errors_total{code="PAYLOAD_TOO_LARGE"}') == 1.0
    assert 'code="VALIDATION_FAILED"' not in metrics
