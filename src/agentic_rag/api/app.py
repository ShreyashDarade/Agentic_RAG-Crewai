from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agentic_rag._version import __version__
from agentic_rag.api.config import ApiConfig
from agentic_rag.api.errors import install_error_handlers
from agentic_rag.api.metrics import Metrics
from agentic_rag.api.middleware import (
    AuthMiddleware,
    BodyLimitMiddleware,
    LoadSheddingMiddleware,
    RequestContextMiddleware,
)
from agentic_rag.api.routes import health_router, ops_router, v1_router
from agentic_rag.application import Service
from agentic_rag.errors import ConfigurationError

__all__ = ["create_app"]


def create_app(
    *,
    config: ApiConfig | None = None,
    service: Service | None = None,
    lifespan: Callable[[FastAPI], AbstractAsyncContextManager[None]] | None = None,
) -> FastAPI:
    """Build the HTTP application.

    Pass ``service`` for tests and export; in production a ``lifespan`` builds it and sets
    ``app.state.service``. Without API keys the app refuses to start unless unauthenticated mode
    is explicitly enabled.
    """
    cfg = config or ApiConfig()
    if not cfg.api_keys and not cfg.allow_unauthenticated:
        raise ConfigurationError("AGENTIC_RAG_API_KEYS is required (or set AGENTIC_RAG_ALLOW_UNAUTHENTICATED)")
    if cfg.api_keys and cfg.allow_unauthenticated:
        raise ConfigurationError("API keys and ALLOW_UNAUTHENTICATED are both set; choose one")
    app = FastAPI(
        title="Agentic RAG",
        version=__version__,
        description="Retrieval-augmented question answering over your documents.",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
    )
    app.state.config = cfg
    app.state.service = service
    app.state.metrics = metrics = Metrics()
    app.state.draining = False
    install_error_handlers(app)
    app.include_router(v1_router)
    app.include_router(health_router)
    app.include_router(ops_router)
    # Last added is outermost. Order, outside in: request context (every response carries an id) -> authentication
    # (nothing is read or counted for an anonymous caller) -> load shedding -> body limit -> CORS -> routes.
    if cfg.cors_allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cfg.cors_allow_origins),
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
            allow_credentials=False,
        )
    app.add_middleware(BodyLimitMiddleware, max_bytes=cfg.max_upload_bytes, metrics=metrics)
    app.add_middleware(LoadSheddingMiddleware, max_inflight=cfg.max_inflight, metrics=metrics)
    app.add_middleware(AuthMiddleware, config=cfg, metrics=metrics)
    app.add_middleware(RequestContextMiddleware, metrics=metrics)
    return app
