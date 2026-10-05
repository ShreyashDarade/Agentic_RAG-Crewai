from __future__ import annotations

import json
import re

import pytest
from fastapi.testclient import TestClient

from agentic_rag.api import ApiConfig, create_app
from agentic_rag.errors import ConfigurationError, Overloaded
from tests.conftest import API_KEY
from tests.unit.test_service import TEXT

PROBLEM = "application/problem+json"


def _upload(client: TestClient, headers: dict[str, str], name: str = "a.txt", data: bytes = TEXT):
    return client.post("/v1/documents", headers=headers, files={"file": (name, data, "text/plain")})


def test_every_v1_route_requires_the_api_key(client: TestClient) -> None:
    for method, path in [
        ("post", "/v1/query"),
        ("post", "/v1/search"),
        ("get", "/v1/documents"),
        ("get", "/v1/documents/x"),
        ("delete", "/v1/documents/x"),
        ("post", "/v1/documents"),
    ]:
        r = getattr(client, method)(path)
        assert r.status_code == 401, path
        assert r.headers["content-type"].startswith(PROBLEM)
        assert r.json()["code"] == "AUTHENTICATION_FAILED"
        assert r.headers["www-authenticate"] == "Bearer"
    wrong = client.get("/v1/documents", headers={"Authorization": "Bearer nope-nope-nope-nope"})
    assert wrong.status_code == 401


def test_health_and_readiness_need_no_key(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}
    r = client.get("/readyz")
    assert r.status_code == 200 and r.json() == {"ready": True, "checks": {"vector_store": "ok"}}


def test_readiness_goes_503_honestly_when_the_store_is_down(client: TestClient, parts) -> None:  # type: ignore[no-untyped-def]
    _, store, _ = parts
    store._unavailable = True
    r = client.get("/readyz")
    assert r.status_code == 503
    assert r.json() == {"ready": False, "checks": {"vector_store": "unavailable"}}
    assert client.get("/healthz").status_code == 200  # liveness never checks dependencies


def test_request_id_on_every_response_and_inbound_ids_are_validated(client: TestClient) -> None:
    assert len(client.get("/healthz").headers["x-request-id"]) == 32
    ok = client.get("/healthz", headers={"X-Request-ID": "trace-12345678"})
    assert ok.headers["x-request-id"] == "trace-12345678"
    for hostile in ("has spaces and <script>", "x" * 200, "short", "a/b/../c-12345678"):
        echoed = client.get("/healthz", headers={"X-Request-ID": hostile}).headers["x-request-id"]
        assert echoed != hostile
        assert re.fullmatch(r"[A-Za-z0-9_-]{8,64}", echoed)
    err = client.get("/v1/documents")
    assert err.json()["request_id"] == err.headers["x-request-id"]


def test_ingest_search_list_get_delete_flow(client: TestClient, api_headers: dict[str, str]) -> None:
    created = _upload(client, api_headers)
    assert created.status_code == 201
    doc = created.json()["document"]
    again = _upload(client, api_headers, name="other-name.txt")
    assert again.status_code == 200 and again.json()["created"] is False
    hits = client.post("/v1/search", headers=api_headers, json={"query": "apples red", "top_k": 3}).json()["hits"]
    assert hits and hits[0]["document_id"] == doc["id"]
    listed = client.get("/v1/documents?limit=1", headers=api_headers).json()
    assert [d["id"] for d in listed["items"]] == [doc["id"]] and listed["next_page_token"] is None
    assert client.get(f"/v1/documents/{doc['id']}", headers=api_headers).json()["name"] == "a.txt"
    assert client.delete(f"/v1/documents/{doc['id']}", headers=api_headers).json()["document_id"] == doc["id"]
    assert client.get(f"/v1/documents/{doc['id']}", headers=api_headers).json()["code"] == "DOCUMENT_NOT_FOUND"


def test_query_answers_with_citations(client: TestClient, api_headers: dict[str, str], parts) -> None:  # type: ignore[no-untyped-def]
    _upload(client, api_headers)
    hits = client.post("/v1/search", headers=api_headers, json={"query": "apples"}).json()["hits"]
    _, _, chat = parts
    chat._replies = [json.dumps({"answer": "Apples are red.", "citations": [hits[0]["chunk_id"]]})]
    r = client.post("/v1/query", headers=api_headers, json={"question": "what colour are apples?"})
    body = r.json()
    assert r.status_code == 200 and body["grounded"] is True and body["citations"] == [hits[0]["chunk_id"]]
    assert body["pipeline"] == "direct"


def test_validation_errors_are_problem_json_and_never_echo_the_input(
    client: TestClient, api_headers: dict[str, str]
) -> None:
    r = client.post("/v1/query", headers=api_headers, json={"question": "", "top_k": "SECRET-INPUT", "extra": 1})
    assert r.status_code == 422 and r.headers["content-type"].startswith(PROBLEM)
    assert r.json()["code"] == "VALIDATION_FAILED"
    assert "SECRET-INPUT" not in r.text
    assert client.post("/v1/query", headers=api_headers, json={"question": "x", "top_k": 51}).status_code == 422
    assert client.get("/v1/documents?limit=1000", headers=api_headers).status_code == 422


def test_unknown_routes_and_methods_are_problems(client: TestClient, api_headers: dict[str, str]) -> None:
    assert client.get("/nope").json()["code"] == "NOT_FOUND"
    r = client.put("/v1/query", headers=api_headers)
    assert r.status_code == 405 and r.json()["code"] == "METHOD_NOT_ALLOWED"


def test_oversized_upload_is_rejected_before_it_is_read(client: TestClient, api_headers: dict[str, str]) -> None:
    r = _upload(client, api_headers, data=b"a" * 300_000)
    assert r.status_code == 413 and r.json()["code"] == "PAYLOAD_TOO_LARGE"
    assert r.headers["x-request-id"]


def test_chunked_oversized_body_is_cut_off(client: TestClient, api_headers: dict[str, str]) -> None:
    def gen():  # type: ignore[no-untyped-def]
        for _ in range(100):
            yield b"x" * 10_000

    r = client.post(
        "/v1/documents",
        headers={**api_headers, "Content-Type": "multipart/form-data; boundary=b"},
        content=gen(),
    )
    assert r.status_code == 413


def test_unsupported_type_and_hostile_names(client: TestClient, api_headers: dict[str, str]) -> None:
    assert _upload(client, api_headers, name="x.exe", data=b"MZ").json()["code"] == "UNSUPPORTED_FILE_TYPE"
    r = _upload(client, api_headers, name="../../etc/passwd.txt", data=TEXT)
    assert r.status_code == 201 and r.json()["document"]["name"] == "passwd.txt"


def test_unexpected_exceptions_become_a_generic_500_with_a_request_id(parts) -> None:  # type: ignore[no-untyped-def]
    service, *_ = parts

    async def boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("secret-connection-string://user:pw@host")

    service.list_documents = boom  # type: ignore[method-assign]
    app = create_app(config=ApiConfig(api_keys=(API_KEY,)), service=service)
    with TestClient(app) as c:
        r = c.get("/v1/documents", headers={"Authorization": f"Bearer {API_KEY}"})
    assert r.status_code == 500 and r.json()["code"] == "INTERNAL_ERROR"
    assert "secret-connection-string" not in r.text
    assert r.json()["request_id"] == r.headers["x-request-id"]


def test_retry_after_is_sent_for_overload(parts) -> None:  # type: ignore[no-untyped-def]
    service, *_ = parts

    async def busy(*_a: object, **_k: object) -> None:
        raise Overloaded(retry_after=3)

    service.list_documents = busy  # type: ignore[method-assign]
    app = create_app(config=ApiConfig(api_keys=(API_KEY,)), service=service)
    with TestClient(app) as c:
        r = c.get("/v1/documents", headers={"Authorization": f"Bearer {API_KEY}"})
    assert r.status_code == 503 and r.headers["retry-after"] == "3" and r.json()["code"] == "OVERLOADED"


def test_the_app_refuses_to_start_without_keys_unless_explicitly_open() -> None:
    with pytest.raises(ConfigurationError):
        create_app(config=ApiConfig())
    create_app(config=ApiConfig(allow_unauthenticated=True))


def test_cors_is_off_by_default(client: TestClient) -> None:
    r = client.options("/v1/query", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in r.headers
