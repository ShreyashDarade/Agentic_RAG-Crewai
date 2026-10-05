# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/). Versioning: [SemVer](https://semver.org/); `0.y.z` until the wire contract and SDK
surface are frozen (ADR-0008).

## [0.1.0] - unreleased

First release of the reconstructed system. The earlier prototype (versioned "2.0.0" in its source) could not answer a query and is not a
compatibility baseline (`docs/baseline.md`); nothing from it is supported.

### Added
- HTTP API under `/v1` (documents, search, query), `/healthz`, `/readyz`, `/metrics`; static bearer-key authentication; problem+json errors
  with stable codes and request ids; load shedding; JSON logs.
- `Client` / `AsyncClient` over HTTP or in process, with a tested retry matrix and a thin install that imports no engine library.
- Engine: content-addressed ingestion (text, Markdown, HTML, PDF, DOCX), recursive chunking, dense + BM25 retrieval with reciprocal rank
  fusion, grounded answers with checked citations.
- Components behind registries: vector stores Milvus, Chroma, Qdrant; embedders and chat models OpenAI and any CrewAI provider; answer
  pipelines `direct` and `crewai` (a four-agent crew).
- Evaluation harness (`agentic-rag eval`), validated on SciFact; benchmark harness; mutation-proof log for every governance rule.

### Changed during the independent review (before the first release)
Three independent reviewers read and attacked the code; every reproduced finding was fixed test-first or documented as a limit
(`docs/review-round-1.md`). Behaviour that differs from the first build: authentication runs before any body is read and before load
shedding counts a request; every use case has the request deadline; a document id is derived from the file type and the bytes, ids of any
other shape are `DOCUMENT_NOT_FOUND`, and an empty `document_ids` filter is a validation error; a re-upload after a chunking, parsing or
embedding change re-indexes; `/readyz` has an `index` check and the read path refuses an index built by another embedding model;
`AGENTIC_RAG_ALLOW_UNAUTHENTICATED` together with API keys is a start-up error; `agentic-rag serve` drains on SIGTERM and prints JSON
only; secrets can be given as `*_FILE`; the CrewAI crew stores nothing on disk and its search tool is bounded; the DOCX parser keeps
document order and has tighter size bounds; the chunker no longer drops a final chunk equal to the previous one.

### Known limitations
See README "Not provided" and `docs/operations.md` "Known limits".
