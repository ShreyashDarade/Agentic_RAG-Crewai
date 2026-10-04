"""The only place a Milvus filter expression is built (ADR-0007, framework section 11).

Values come from a typed :class:`~agentic_rag.ports.ChunkFilter` and must match a closed alphabet
that cannot contain a quote, backslash, bracket or whitespace, so a rendered literal cannot break
out of its string. (Milvus Lite does not support ``filter_params`` templating, so templating cannot be
the single code path; validation by construction is.)
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from agentic_rag.errors import ValidationFailed
from agentic_rag.ports import ChunkFilter

__all__ = ["quote_ids", "render"]

_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def quote_ids(ids: Iterable[str]) -> str:
    """``["a", "b"]`` as a Milvus list literal, after validating every id."""
    values = list(ids)
    for value in values:
        if not _ID.fullmatch(value):
            raise ValidationFailed("an identifier contains characters that are not allowed")
    return "[" + ", ".join(f'"{v}"' for v in values) + "]"


def render(flt: ChunkFilter | None) -> str:
    """Filter expression for ``flt`` (empty string means no restriction)."""
    if flt is None or not flt.document_ids:
        return ""
    return f"document_id in {quote_ids(flt.document_ids)}"
