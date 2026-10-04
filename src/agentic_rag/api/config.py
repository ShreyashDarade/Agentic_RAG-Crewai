from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ApiConfig"]


@dataclass(frozen=True, slots=True)
class ApiConfig:
    """Everything the HTTP layer needs to know, as plain values (the API never reads settings itself)."""

    api_keys: tuple[str, ...] = ()
    allow_unauthenticated: bool = False
    cors_allow_origins: tuple[str, ...] = ()
    max_upload_bytes: int = 25 * 1024 * 1024
