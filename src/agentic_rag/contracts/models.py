"""Contract models. Hard upper bounds live here; settings may lower them, never raise them."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

__all__ = [
    "MAX_PAGE_SIZE",
    "MAX_QUESTION_CHARS",
    "MAX_TOP_K",
    "DeleteResult",
    "DocumentInfo",
    "DocumentList",
    "IngestResult",
    "ProblemDetails",
    "QueryRequest",
    "QueryResponse",
    "ReadyResponse",
    "SearchHit",
    "SearchRequest",
    "SearchResponse",
    "Source",
]

MAX_QUESTION_CHARS = 4000
MAX_TOP_K = 50
MAX_PAGE_SIZE = 200
MAX_IDS_PER_FILTER = 100

#: Identifiers are restricted to an alphabet that cannot break out of a filter expression or a path.
Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9._-]{1,128}$")]


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _Response(BaseModel):
    model_config = ConfigDict(extra="allow")


class QueryRequest(_Request):
    """Ask a question over the indexed documents."""

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    top_k: int | None = Field(default=None, ge=1, le=MAX_TOP_K)
    document_ids: list[Identifier] | None = Field(default=None, max_length=MAX_IDS_PER_FILTER)


class SearchRequest(_Request):
    """Retrieve ranked chunks without generating an answer."""

    query: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    top_k: int | None = Field(default=None, ge=1, le=MAX_TOP_K)
    document_ids: list[Identifier] | None = Field(default=None, max_length=MAX_IDS_PER_FILTER)


class Source(_Response):
    """A retrieved chunk that supports an answer."""

    chunk_id: str
    document_id: str
    document_name: str | None = None
    snippet: str
    score: float
    page: int | None = None


class QueryResponse(_Response):
    """An answer with the chunks it cites.

    ``grounded`` is ``False`` when nothing relevant was retrieved: ``answer`` then states that, no
    model was asked, and ``citations`` is empty.
    """

    answer: str
    grounded: bool
    citations: list[str] = Field(default_factory=list, description="chunk ids, all present in sources")
    sources: list[Source] = Field(default_factory=list)
    pipeline: str


class SearchHit(_Response):
    chunk_id: str
    document_id: str
    document_name: str | None = None
    text: str
    score: float
    page: int | None = None


class SearchResponse(_Response):
    hits: list[SearchHit]


class DocumentInfo(_Response):
    id: str
    name: str
    content_sha256: str
    content_type: str
    chunk_count: int
    embedding_model: str


class DocumentList(_Response):
    items: list[DocumentInfo]
    next_page_token: str | None = None


class IngestResult(_Response):
    document: DocumentInfo
    created: bool = Field(description="False when identical content was already indexed")
    chunks_indexed: int


class DeleteResult(_Response):
    document_id: str
    chunks_deleted: int


class ReadyResponse(_Response):
    ready: bool
    checks: dict[str, str] = Field(default_factory=dict)


class ProblemDetails(_Response):
    """RFC 9457 problem details plus ``code`` and ``request_id``."""

    type: str
    title: str
    status: int
    code: str
    detail: str | None = None
    details: dict[str, Any] | None = None
    request_id: str | None = None
    retry_after: float | None = None
