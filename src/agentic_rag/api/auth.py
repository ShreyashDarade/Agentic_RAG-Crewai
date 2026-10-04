from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from agentic_rag.api.config import ApiConfig
from agentic_rag.errors import AuthenticationFailed

__all__ = ["require_api_key"]

_bearer = HTTPBearer(auto_error=False, description="Static API key (ADR-0006)")


def require_api_key(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> None:
    config: ApiConfig = request.app.state.config
    if config.allow_unauthenticated:
        return
    presented = credentials.credentials.encode() if credentials else b""
    # Compare against every key without early exit so timing does not reveal which key matched.
    matches = [secrets.compare_digest(presented, key.encode()) for key in config.api_keys]
    if not (credentials and any(matches)):
        raise AuthenticationFailed()
