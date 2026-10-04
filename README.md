# agentic-rag

Retrieval-augmented question answering over your documents: an HTTP service, a Python SDK and an in-process engine.

> **Status: 0.1.0, under reconstruction.** What exists today is listed below; what does not is listed after it.
> See `docs/baseline.md` (where this started), `docs/framework.md` (the rules) and `docs/adr/` (the decisions).

## What works now (verified by the test suite)

* HTTP API under `/v1`: `POST /documents` (upload), `GET /documents`, `GET|DELETE /documents/{id}`,
  `POST /search`, `POST /query`; `GET /healthz` (liveness), `GET /readyz` (dependency readiness).
* Every route except `/healthz` and `/readyz` needs `Authorization: Bearer <key>` (`AGENTIC_RAG_API_KEYS`).
* Errors are `application/problem+json` with a stable `code` and a `request_id` (also in the `X-Request-ID` header).
* Ingestion of `.txt .md .rst .csv .html .htm .pdf .docx`; chunk and document ids are content-addressed, so re-uploading
  the same bytes is a no-op and re-ingesting replaces the old chunks.
* Retrieval: dense (Milvus HNSW, cosine) + BM25, fused with reciprocal rank fusion.
* Answers are JSON-validated and every citation must be a chunk that was retrieved; nothing is generated when nothing
  was retrieved.

## What does **not** exist yet

SDK (`Client`), OCR for scanned PDFs, `.xlsx`/`.pptx`, cross-encoder reranking, the CrewAI pipeline and the other
providers (ADR-0005, ADR-0010), metrics, load shedding, benchmarks, evaluation. Defaults (chunk size, `top_k`, fusion
constants) are **unmeasured**.

## Develop

```bash
uv sync --extra engine --extra server --extra parsers --group dev
scripts/dev-services.sh up && eval "$(scripts/dev-services.sh env)"   # Milvus (standalone via docker, else Milvus Lite)
uv run pytest
```

Run the server (needs `AGENTIC_RAG_API_KEYS`, `AGENTIC_RAG_MILVUS_URI`, `AGENTIC_RAG_OPENAI_API_KEY`):

```bash
uv run uvicorn --factory agentic_rag.server:create
```

Configuration is environment-only (`AGENTIC_RAG_*`); unknown or invalid values stop the process before it binds a port.
`.env.example` lists the secrets. Nothing secret is committed.
