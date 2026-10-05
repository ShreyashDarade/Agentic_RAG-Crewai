"""G-ops: killing a dependency makes /readyz, /metrics and the logs say the same true thing; overload is shed."""

from __future__ import annotations

import asyncio
import json
import logging

import httpx
import pytest
from fastapi.testclient import TestClient

from agentic_rag.api import ApiConfig, create_app
from agentic_rag.api.logging import JsonFormatter
from tests.conftest import API_KEY
from tests.unit.test_service import TEXT, build

H = {"Authorization": f"Bearer {API_KEY}"}


def _metric(text: str, name: str) -> float:
    for line in text.splitlines():
        if line.startswith((name + " ", name + "{")):
            return float(line.rsplit(" ", 1)[1])
    raise AssertionError(f"{name} not exposed:\n{text}")


def test_a_dead_dependency_is_reported_identically_by_readyz_metrics_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    service, store, _ = build()
    app = create_app(config=ApiConfig(api_keys=(API_KEY,)), service=service)
    with TestClient(app) as client, caplog.at_level(logging.INFO):
        assert client.get("/readyz").status_code == 200
        up = client.get("/metrics", headers=H).text
        assert 'agentic_rag_dependency_up{dependency="vector_store"} 1.0' in up
        assert _metric(up, "agentic_rag_ready") == 1.0

        store._unavailable = True  # the dependency dies
        r = client.get("/readyz", headers={"X-Request-ID": "outage-0001-abcd"})
        assert r.status_code == 503
        assert r.json() == {"ready": False, "checks": {"vector_store": "unavailable"}}
        down = client.get("/metrics", headers=H).text
        assert 'agentic_rag_dependency_up{dependency="vector_store"} 0.0' in down
        assert _metric(down, "agentic_rag_ready") == 0.0
        warnings = [rec for rec in caplog.records if getattr(rec, "event", "") == "not_ready"]
        assert warnings
        assert warnings[-1].failing == ["vector_store"]  # type: ignore[attr-defined]
        access = [rec for rec in caplog.records if getattr(rec, "request_id", "") == "outage-0001-abcd"]
        assert access
        assert access[-1].status == 503  # type: ignore[attr-defined]
        assert client.get("/healthz").status_code == 200  # liveness is independent of dependencies


def test_metrics_need_the_key_and_use_route_templates_not_raw_paths() -> None:
    service, *_ = build()
    app = create_app(config=ApiConfig(api_keys=(API_KEY,)), service=service)
    with TestClient(app) as client:
        assert client.get("/metrics").status_code == 401
        client.get("/v1/documents/doc_secret-id-1234", headers=H)
        client.post("/v1/documents", headers=H, files={"file": ("a.txt", TEXT, "text/plain")})
        text = client.get("/metrics", headers=H).text
    assert 'route="/v1/documents/{document_id}"' in text
    assert "doc_secret-id-1234" not in text  # an id in the path must never become a label value
    assert 'agentic_rag_errors_total{code="DOCUMENT_NOT_FOUND"} 1.0' in text
    assert 'status_class="2xx"' in text
    assert 'status_class="4xx"' in text


async def test_overload_is_shed_with_503_and_retry_after_while_health_stays_up() -> None:
    service, *_ = build()
    gate = asyncio.Event()
    real_search = service.search

    async def slow_search(request):  # type: ignore[no-untyped-def]
        await gate.wait()
        return await real_search(request)

    service.search = slow_search  # type: ignore[method-assign]
    app = create_app(config=ApiConfig(api_keys=(API_KEY,), max_inflight=2), service=service)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        tasks = [asyncio.create_task(client.post("/v1/search", headers=H, json={"query": "apples"})) for _ in range(5)]
        await asyncio.sleep(0.2)
        assert (await client.get("/healthz")).status_code == 200  # exempt from shedding
        gate.set()
        responses = await asyncio.gather(*tasks)
    codes = sorted(r.status_code for r in responses)
    assert codes == [200, 200, 503, 503, 503]
    shed = next(r for r in responses if r.status_code == 503)
    assert shed.json()["code"] == "OVERLOADED"
    assert 1 <= int(shed.headers["retry-after"]) <= 3
    assert shed.headers["x-request-id"]
    assert _metric(app.state.metrics.render().decode(), "agentic_rag_load_shed_total") == 3.0


def test_draining_server_reports_not_ready_but_alive() -> None:
    service, *_ = build()
    app = create_app(config=ApiConfig(api_keys=(API_KEY,)), service=service)
    with TestClient(app) as client:
        assert client.get("/readyz").status_code == 200
        app.state.draining = True
        r = client.get("/readyz")
        assert r.status_code == 503
        assert r.json()["checks"] == {"server": "draining"}
        assert client.get("/healthz").status_code == 200


def test_json_logs_carry_the_request_id_and_never_raw_paths(caplog: pytest.LogCaptureFixture) -> None:
    service, *_ = build()
    app = create_app(config=ApiConfig(api_keys=(API_KEY,)), service=service)
    formatter = JsonFormatter()
    with TestClient(app) as client, caplog.at_level(logging.INFO, logger="agentic_rag.access"):
        client.get("/v1/documents/doc_abc", headers={**H, "X-Request-ID": "req-abcdefgh12"})
    line = json.loads(formatter.format(next(r for r in caplog.records if r.name == "agentic_rag.access")))
    assert line["request_id"] == "req-abcdefgh12"
    assert line["route"] == "/v1/documents/{document_id}"
    assert line["status"] == 404
    assert line["event"] == "request"
    assert line["duration_ms"] >= 0
