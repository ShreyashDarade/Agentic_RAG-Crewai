from __future__ import annotations

import socket
import threading
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest
import uvicorn

from agentic_rag import AsyncClient
from agentic_rag.api import ApiConfig, create_app
from tests.conftest import API_KEY
from tests.unit.test_service import build

TRANSPORTS = ["embedded", "http"]


def make_http_client(service: Any, *, key: str | None = API_KEY) -> AsyncClient:
    """HTTP through the real FastAPI app (ASGI transport: no socket, every middleware runs)."""
    app = create_app(config=ApiConfig(api_keys=(API_KEY,), max_upload_bytes=200_000), service=service)
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost")
    return AsyncClient.http("http://localhost", api_key=key, http_client=http)


@pytest.fixture
def engine():  # type: ignore[no-untyped-def]
    """(service, store, chat) over fakes: pure logic, no network."""
    return build()


@pytest.fixture(params=TRANSPORTS)
async def client(request: pytest.FixtureRequest, engine) -> AsyncIterator[AsyncClient]:  # type: ignore[no-untyped-def]
    service, *_ = engine
    c = AsyncClient.embedded(service=service) if request.param == "embedded" else make_http_client(service)
    yield c
    await c.aclose()


class LiveServer:
    def __init__(self, service: Any) -> None:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        app = create_app(config=ApiConfig(api_keys=(API_KEY,), max_upload_bytes=200_000), service=service)
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        self.thread.start()
        for _ in range(100):
            if self.server.started:
                return
            time.sleep(0.05)
        raise RuntimeError("test server did not start")

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


@pytest.fixture
def live_server(engine) -> Iterator[LiveServer]:  # type: ignore[no-untyped-def]
    service, *_ = engine
    server = LiveServer(service)
    server.start()
    yield server
    server.stop()
