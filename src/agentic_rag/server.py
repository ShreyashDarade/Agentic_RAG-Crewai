"""Server entry points: ``agentic-rag serve`` (drains on SIGTERM) or ``uvicorn --factory agentic_rag.server:create``.

An entry module may import the container, the API and the settings; nothing imports it.
"""

from __future__ import annotations

import asyncio
import logging
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from agentic_rag.api import ApiConfig, create_app
from agentic_rag.api.logging import configure_logging
from agentic_rag.config import Settings, load_settings
from agentic_rag.container import build_container

__all__ = ["create", "serve"]

logger = logging.getLogger("agentic_rag.server")

_GRACEFUL_SHUTDOWN_SECONDS = 30


def create(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()  # raises ConfigurationError before any port is bound
    configure_logging(settings.log_level, json_format=settings.log_json)
    if settings.allow_unauthenticated:
        logger.warning("AGENTIC_RAG_ALLOW_UNAUTHENTICATED is set: every route is open")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        container = await build_container(settings)
        app.state.service = container.service
        try:
            yield
        finally:
            app.state.draining = True
            await container.aclose()

    config = ApiConfig(
        api_keys=tuple(k.get_secret_value() for k in settings.api_keys),
        allow_unauthenticated=settings.allow_unauthenticated,
        cors_allow_origins=tuple(settings.cors_allow_origins),
        max_upload_bytes=settings.max_upload_bytes,
        max_inflight=settings.max_inflight_requests,
    )
    return create_app(config=config, lifespan=lifespan)


class _DrainingServer(uvicorn.Server):
    """uvicorn closes its listening socket the moment it is told to stop, which is before the application's lifespan
    shutdown runs, so an application cannot drain from its own lifespan. This hook runs first: it flips ``/readyz`` to
    503 while the listener is still open and in-flight requests keep going, waits, and only then lets uvicorn stop."""

    def __init__(self, config: uvicorn.Config, *, app: FastAPI, drain_seconds: float) -> None:
        super().__init__(config)
        self._app = app
        self._drain_seconds = drain_seconds

    async def shutdown(self, sockets: list[socket.socket] | None = None) -> None:
        self._app.state.draining = True
        logger.info("draining", extra={"event": "draining", "seconds": self._drain_seconds})
        await asyncio.sleep(self._drain_seconds)
        await super().shutdown(sockets=sockets)


def serve(host: str, port: int) -> None:
    settings = load_settings()
    app = create(settings)
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        timeout_graceful_shutdown=_GRACEFUL_SHUTDOWN_SECONDS,
        log_config=None,  # uvicorn must not install its own plain-text handlers: ours print JSON
        access_log=False,  # ours is the access log: route templates, never raw paths or query strings
    )
    server = _DrainingServer(config, app=app, drain_seconds=settings.shutdown_drain_seconds)
    server.run()
