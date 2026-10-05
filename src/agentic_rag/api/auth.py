from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from agentic_rag.api.config import ApiConfig
from agentic_rag.errors import AuthenticationFailed

__all__ = ["bearer_token", "is_authorized", "require_api_key"]

_bearer = HTTPBearer(auto_error=False, description="Static API key (ADR-0006)")


def bearer_token(authorization: str | None) -> str | None:
    """The credentials of an ``Authorization: Bearer <token>`` header, or ``None`` if it is not one."""
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    token = token.strip()
    return token if scheme.lower() == "bearer" and token else None


def is_authorized(config: ApiConfig, token: str | None) -> bool:
    """The single place a key is checked: the early middleware and the route dependency both call it."""
    if config.allow_unauthenticated:
        return True
    if token is None:
        return False
    presented = token.encode()
    # Compare against every key without early exit so timing does not reveal which key matched.
    matches = [secrets.compare_digest(presented, key.encode()) for key in config.api_keys]
    return any(matches)


def require_api_key(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> None:
    """Defence in depth (and the OpenAPI security scheme): the middleware has normally rejected the request already."""
    config: ApiConfig = request.app.state.config
    if not is_authorized(config, credentials.credentials if credentials else None):
        raise AuthenticationFailed()
