"""Wire models: what the HTTP API and both SDK transports exchange (ADR-0002, framework section 6).

Only the standard library and pydantic are imported here. Request models forbid unknown fields so
typos fail loudly; response models keep unknown fields so a newer server never breaks an older client.
"""

from agentic_rag.contracts.models import (
    DeleteResult,
    DocumentInfo,
    DocumentList,
    IngestResult,
    ProblemDetails,
    QueryRequest,
    QueryResponse,
    ReadyResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
    Source,
)

__all__ = [
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
