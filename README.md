# agentic-rag

Retrieval-augmented question answering over your documents. One Python package, three ways to use it: an HTTP service, a client
SDK for that service, and the same engine running inside your process. The same methods, models and errors on both transports.

> **Version 0.1.0.** The wire contract and the SDK surface are not frozen yet (`0.y.z`). What exists is listed here; what does not is
> listed under *Not provided*. Defaults are mostly **unmeasured** (`docs/defaults.toml`); provider support is stated per provider in
> `docs/providers.md`.

## Install

| You want | Command | Pulls in |
|---|---|---|
| Call a server (thin client) | `pip install agentic-rag` | `httpx`, `pydantic` only. No engine library is imported (a test blocks them all and still passes). |
| Run the engine in your process | `pip install "agentic-rag[engine]"` | `pymilvus`, `openai`, `pydantic-settings`, `numpy` |
| Serve the HTTP API | `pip install "agentic-rag[server]"` | engine + `fastapi`, `uvicorn`, `python-multipart`, `prometheus-client` |
| Read PDF, DOCX, HTML | add `[parsers]` | `pymupdf`, `python-docx`, `beautifulsoup4`, `lxml`, `openpyxl`, `python-pptx` (`.xlsx`/`.pptx` have no parser yet) |
| Multi-agent answers and any CrewAI-routed model or embedder | add `[crewai]` | `crewai` |
| Chroma or Qdrant instead of Milvus | add `[chroma]` or `[qdrant]` | `chromadb` / `qdrant-client` |

Python 3.11, 3.12 and 3.13 are tested. Releases are not published to PyPI yet: install from a checkout (`pip install -e ".[server,parsers]"`).

## Quick start: HTTP

```bash
export AGENTIC_RAG_API_KEYS=$(python -c "import secrets; print(secrets.token_urlsafe(24))")
export AGENTIC_RAG_OPENAI_API_KEY=...            # and a Milvus: AGENTIC_RAG_MILVUS_URI=http://localhost:19530
agentic-rag serve                                # or: uvicorn --factory agentic_rag.server:create
```

```python
from agentic_rag import Client

with Client.http("http://127.0.0.1:8000", api_key="<your key>") as rag:
    rag.ingest_document("handbook.pdf", open("handbook.pdf", "rb"))
    answer = rag.query("How many vacation days do new hires get?")
    print(answer.answer, answer.citations)        # every citation is a chunk that was retrieved
```

## Quick start: in process (same API)

```python
from agentic_rag import Client                    # needs agentic-rag[engine]; settings come from AGENTIC_RAG_* variables

with Client.embedded() as rag:
    rag.ingest_document("notes.txt", b"Apples are red.")
    print(rag.search("apple colour").hits[0].text)
```

Use `AsyncClient` (same methods, `await`) inside an event loop; a blocking `Client` call inside a running loop raises `UsageError`.

## HTTP API (`docs/openapi.json` is the contract)

`POST /v1/documents` (multipart upload), `GET /v1/documents`, `GET /v1/documents/{document_id}`, `DELETE /v1/documents/{document_id}`,
`POST /v1/search`, `POST /v1/query`; `GET /healthz` (liveness), `GET /readyz` (dependency readiness), `GET /metrics` (Prometheus).
Everything except `/healthz` and `/readyz` needs `Authorization: Bearer <key>`.

## What is guaranteed

* **Errors** are `application/problem+json` with a stable `code` and a `request_id`; the SDK raises the same `RagError` subclass over both
  transports. Codes are append-only (`docs/error_codes.json`). Provider and database text never reaches a response.
* **Retries (SDK)**: never-sent failures and `429`/`503` are retried for every call (honouring `Retry-After`, capped); ambiguous failures
  (connection reset, `408`/`502`/`504`) only for idempotent calls; a read timeout is never retried; backoff is full-jitter with a retry
  budget; the total deadline (60 s) is above the server's (55 s) so the server's typed `DEADLINE_EXCEEDED` arrives first.
* **Ingest is idempotent**: ids derive from the bytes, so re-uploading is a no-op and re-ingesting replaces the document's chunks.
* **Answers are checked**: output must be valid JSON and every cited chunk must have been retrieved; nothing is generated when nothing was
  retrieved (`grounded: false`).
* **Safety**: bearer keys (two may be active), bounded uploads/questions/`top_k`/concurrency, filters from a closed alphabet, no files
  written by name, no URL fetching, load shedding with `503` + `Retry-After`. See `docs/security.md` for the threat model and its gaps.
* **Versioning**: SemVer for the SDK and `/v1`; additive-only within a major; machine-checked by snapshots of the OpenAPI document, the
  public API and the error codes. "Two minor releases of notice before removal" is a review rule, not machine-enforced.

## Not provided

No user accounts, tenants or per-document access control (every key holder sees every document). No streaming. No OCR (scanned PDFs are
rejected as empty), no `.xlsx`/`.pptx`, no reranker, no conversation memory. The BM25 index is per process (see `docs/operations.md`).
Hosted providers (OpenAI, Anthropic, Cohere, …) and standalone Milvus were **not** exercised against the real services here; Milvus Lite,
Chroma, Qdrant and local OpenAI-compatible stubs were. No LLM-judged answer-quality evaluation exists.

## Configuration

Environment only, prefix `AGENTIC_RAG_`; an unknown or invalid variable stops the process before it binds a port, and the error names the
variable, never its value. `.env.example` lists the secrets; nothing secret is committed. Components are chosen by name
(`AGENTIC_RAG_VECTOR_STORE`, `_EMBEDDER`, `_CHAT_MODEL`, `_ANSWER_PIPELINE`) and extended with plug-ins (`AGENTIC_RAG_PLUGINS`,
`agentic_rag.extend`, `agentic_rag.testing`).

## Develop

```bash
uv sync --extra engine --extra server --extra parsers --extra crewai --extra chroma --extra qdrant --group dev
scripts/dev-services.sh up && eval "$(scripts/dev-services.sh env)"      # Milvus: standalone via docker if reachable, else Lite
uv run pytest                      # unit, conformance, API, SDK parity, integration on real local engines
uv run lint-imports && uv run mypy && uv run ruff check src tests scripts && uv run python scripts/snapshots.py check
uv run python scripts/prove_rules.py   # break every governance rule on purpose; each check must fail (docs/mutation-proofs.md)
```

Where to read next: `docs/framework.md` (the rules and the checks that enforce them), `docs/adr/` (why), `docs/baseline.md` (where this
started), `docs/operations.md`, `docs/benchmark.md`, `docs/evaluation.md`, `docs/providers.md`, `CHANGELOG.md`.
