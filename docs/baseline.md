# Baseline (Step 1)

Status of the repository at commit `c3ef91c` before any change. Every number below names the command that produced it
(run in a scratch venv, Python 3.11.15, `uv pip install -r requirements.txt`, 4 vCPU / 15 GB sandbox).
Findings marked **[verified]** were reproduced by running code; **[read]** were established by reading the source
only; **[unverified]** are candidates that were not executed.

## 1. What it is

A FastAPI service (`api/`) that answers questions over ingested documents:
`Supervisor -> Retriever -> Generator -> Feedback` "agents" (`agents/`), driven by `orchestrator/crew_manager.py`;
documents are loaded/OCR'd/chunked (`data_pipeline/`), embedded with OpenAI (`embeddings/`), stored in Milvus
(HNSW) and searched with dense + BM25, RRF fusion, cross-encoder rerank and MMR (`retriever/`).

* ~14,000 lines of Python, 0 tests, no CI other than a bot that commits a log line daily
  (`.github/workflows/daily_commit.yml`; `git add .` then push — it would commit anything in the tree).
* Entry points: `python run.py` (uvicorn), `api.main:app`, `docker-compose.yml`. No CLI, no SDK, no library API.

## 2. Promises made to callers (README / OpenAPI) vs reality

| Promise | Reality |
|---|---|
| `POST /api/v1/agent_query` answers a question | **Cannot initialise.** `llm/__init__.py` imports `LLMRole`, which `llm/base_llm.py` does not define. `CrewManager.initialize()` raises, the app boots "degraded", every query returns 500. **[verified]** (`TestClient`, `OPENAI_API_KEY=sk-test`, no Milvus). |
| `/health` reports component health | Returns HTTP 200 with `status: degraded` and every component `not_initialized`; the "healthy" branch only checks that an object exists, never that OpenAI/Milvus is reachable. Docker `HEALTHCHECK` (`curl -f /health`) passes while the service is unusable. **[verified]** |
| Ingest endpoints (`/ingest`, `/ingest/file`, `/ingest/upload`, `/ingest/files`, `/ingest/reset`, `/ingest/discover`) | Routes call `IngestionPipeline.process_file`, `.state`, `.discover_files`, `config.raw_documents_dir`, `ingest_directory(force=, batch_size=)`, `reset(clear_vector_store=)`; none exist. **[verified]** (`hasattr`/`inspect.signature`). Only `/ingest/status` matches the implementation. |
| `/search` direct search | Calls `RetrieverAgent.retrieve(filters=...)`; no such parameter, and treats the returned dataclass as a dict. **[verified]** signature, **[read]** the dict use. |
| Hybrid dense+BM25 / RRF / rerank / MMR | Present in `AdvancedRetriever`. Never measured. Rerank and MMR failures silently degrade to unranked results; BM25 index is built from the first 10,000 rows only and is never refreshed on ingest. **[read]** |
| "CrewAI multi-agent orchestration" | `crewai.Agent` objects are built lazily via a property that nothing reads; no `Crew`, `Task` or `kickoff` call exists anywhere (`grep -rnE "Crew\(|Task\(|kickoff"` -> no hits). The "agents" are plain classes calling the LLM through `run_async_task`. CrewAI is effectively unused at runtime but is a hard dependency (`crewai>=0.28`, resolved 1.14.1). **[verified]** |
| Web search | `RetrieverAgent` checks `hasattr(retriever, "search_web")`; `AdvancedRetriever` has no such method, so web search is silently skipped. `HybridRetriever` calls an `async` method without awaiting. **[read]** |
| Config via `config/config.yaml`, `crew_config.yaml` | No Python code loads either file (`grep -rnE "yaml\.|safe_load|config\.yaml"` -> nothing). Behaviour is set by dataclass defaults and 30 scattered `os.getenv` calls. **[verified]** |
| Table/image cross-reference linking | `IngestionPipeline` reads `loaded_doc.metadata["tables"/"images"]`; the loader stores them on `.tables/.images`, so both are always empty. **[read]** |

## 3. Data stores, external calls, concurrency

* **Milvus** (`embeddings/milvus_store.py`): one collection `documents`, HNSW, COSINE, `consistency_level="Strong"`;
  `MILVUS_URI`/`MILVUS_TOKEN` (Zilliz). Retries every exception 3x with blocking `time.sleep`.
  Filters are built by f-string interpolation (`content_type`, `parent_id`, `id in [...]`). **[read]**
* **OpenAI**: chat via `AsyncOpenAI` in `llm/openai_client.py` (retries *every* exception, including auth and
  bad-request, up to `retry_attempts`; non-rate-limit errors are all re-raised as `InvalidRequestError`);
  embeddings via a second path in `embeddings/openai_embedder.py` with its own sync wrapper. **[read]**
* **Local disk**: `data/memory/memory.json` (rewritten in full on every query), `data/traces/<trace_id>.json`,
  uploaded files in `data/raw/`. Process-local global singletons in `api/main.py`, `api/routes/*.py`.
* **Concurrency**: async routes call synchronous code (OCR, embedding, Milvus, LLM via `run_async_task`) directly on
  the event loop; `embed_sync` inside a running loop raises `RuntimeError: This event loop is already running`
  **[verified]**. `run_async_task` spawns a thread + new loop per call and closes it; the shared `AsyncOpenAI` client is
  then reused across dead loops. In-memory dedup/memory/trace/BM25 state is per process, so `WORKERS>1` gives each
  worker different state. **[read]**
* **Resource bounds**: upload read fully into memory (no size cap); `max_file_size_mb` is 50 in the pipeline and 100 in the
  loader; no rate limiter although `RATE_LIMIT_*` env vars are documented (nothing reads them); no request timeout.

## 4. Known defects that matter most (ordered by blast radius)

1. **App cannot answer a query** (`LLMRole` import) — [verified].
2. **Ingestion API is non-functional** (missing methods) — [verified].
3. **Security**: unauthenticated `DELETE /ingest/reset` (drops the collection); server-side arbitrary directory/file
   ingestion from request body; upload filename joined into a path without sanitising
   (`raw_dir / file.filename`; `Path("/app/data/raw") / "/etc/passwd" == "/etc/passwd"` **[verified]**), no size or extension
   limit; CORS `*` with credentials; filter-expression injection into Milvus; `config/.env` is tracked in git
   (`git ls-files` lists it) — **its values were deliberately not inspected**; if any are real they must be treated as
   leaked and rotated; there is no `.gitignore`.
4. **Silent failures** (violates "no silent fallbacks"): 115 `except` sites, 12 bare `except:`; `CrewManager` returns an
   ungrounded plain-LLM answer with `success=True` when the pipeline raises; ingest returns `success=True` for duplicates
   and for files whose OCR failed; `CrossEncoderReranker`/MMR/BM25 failures degrade quietly; `embeddings.ChunkTagger` is always
   `None` because `ChunkTagConfig` is missing **[verified]**; `HybridRetriever` import errors are swallowed.
5. **Dead code**: `agents/tools/*` use `from ...llm` imports that fail (`ImportError: attempted relative import beyond
   top-level package` **[verified]**) and nothing imports them; `metadata_filter`, `chunk_tags`, `hybrid_retriever` are unused.
6. **Packaging**: repo root contains `__init__.py` (mypy refuses: "not a valid Python package name"), no `pyproject.toml`,
   no lock file, 20+ unpinned lower-bound dependencies; install is 6.9 GB (`du -sh venv`, torch + CUDA wheels, EasyOCR, crewai).
   `requirements.txt` asks for `camelot-py[cv]` (extra does not exist in the resolved version: uv warning).

## 5. Measurements

| Check | Command | Result |
|---|---|---|
| Tests | `pytest --collect-only -q` | `no tests collected` (0 test files) |
| Coverage | — | not measurable (no tests) |
| Lint | `ruff check . --statistics` (ruff 0.16.10 default rule set) | 991 findings, 733 auto-fixable |
| Types | `mypy .` | refuses to run: root `__init__.py` makes the checkout "not a valid Python package name"; not retried with per-package config yet |
| Import of app | `python -c "import api.main"` (no services) | succeeds, 0.38 s (all heavy imports are lazy) |
| App with no services | `TestClient(app)`: `/health`, `/api/v1/agent_query`, `/api/v1/ingest/status` | 200 (degraded), 500, 500 |
| Real Milvus | `docker pull milvusdb/milvus:v2.6.4` | **blocked**: registry blob download returns 502 through the sandbox proxy (network policy); Docker daemon itself starts (`dockerd`) |
| Embedded Milvus | `pip install milvus-lite` + insert/search | works (engine is real Milvus, but embedded: not the gRPC server nor the HNSW index path) |

Runtime of the existing suite: n/a.

## 6. Where it breaks first

1. Process start -> first request: `llm/__init__.py:15` (`LLMRole`), then `CrewManager.initialize()`.
2. First ingest request: `api/routes/ingest.py:64` (`ingest_directory(force=...)` TypeError) and `:142/:208` (`process_file`).
3. First real load: sync work on the event loop (`embed_sync`, `run_async_task`, Milvus `time.sleep`) blocks all requests;
   BM25 rebuilt only at start-up; per-query full rewrite of `memory.json`.
4. First hostile request: path traversal on upload, arbitrary-path ingest, collection drop.

## 7. Untested areas

All of it. Specifically unknown: retrieval quality of every stage (no labelled data, no metrics), HNSW parameter
suitability, OCR/loader behaviour on real documents, behaviour under concurrency, memory growth, and whether the
chunker's offsets/cross-references are correct (the audit suggests offsets are double-counted in `_split_recursive`,
**[unverified]**).

## 8. What this baseline did not do

* Did not run against real Milvus standalone or real OpenAI (no credentials used; the committed `.env` was not read) —
  only start-up/route behaviour with fake keys was exercised.
* Several module-level claims (chunker offsets, OCR `paragraph`/`detail` handling, trace path handling in
  `/trace/{trace_id}`) come from a read-only delegate audit plus my own reading; only those marked **[verified]** were executed.
