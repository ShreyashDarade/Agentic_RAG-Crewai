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

### Known limitations
See README "Not provided" and `docs/operations.md` "Known limits".
