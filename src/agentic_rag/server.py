"""Server entry point: ``uvicorn --factory agentic_rag.server:create``.

An entry module may import the container, the API and the settings; nothing imports it.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from agentic_rag.api import ApiConfig, create_app
from agentic_rag.config import load_settings
from agentic_rag.container import build_container

__all__ = ["create"]

logger = logging.getLogger("agentic_rag.server")


def create() -> FastAPI:
    settings = load_settings()  # raises ConfigurationError before any port is bound
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if settings.allow_unauthenticated:
        logger.warning("AGENTIC_RAG_ALLOW_UNAUTHENTICATED is set: every route is open")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        container = await build_container(settings)
        app.state.service = container.service
        try:
            yield
        finally:
            await container.aclose()

    config = ApiConfig(
        api_keys=tuple(k.get_secret_value() for k in settings.api_keys),
        allow_unauthenticated=settings.allow_unauthenticated,
        cors_allow_origins=tuple(settings.cors_allow_origins),
        max_upload_bytes=settings.max_upload_bytes,
    )
    return create_app(config=config, lifespan=lifespan)
