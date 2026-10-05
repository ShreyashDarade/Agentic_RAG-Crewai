# Operations runbook

What an on-call engineer can see, and what to do about the top failure modes. Everything here is exercised by tests unless marked
*unverified*.

## Probes and signals

| Signal | Meaning |
|---|---|
| `GET /healthz` | The process answers. Checks **no** dependency, so an outage never makes an orchestrator restart healthy pods. |
| `GET /readyz` | `200` only if every dependency check passes and the server is not draining; otherwise `503` with `{"ready": false, "checks": {...}}`. Checks: `vector_store` (the store answers), `index` (the collection exists or can be created and was built by **this** embedding model and vector size), `lexical_index` (in-memory BM25 rebuilt from the store), `server` (`draining` during shutdown). Values are `ok`, `unavailable`, `timeout` or, for `index`, `incompatible`; never raw dependency text. Note that `/readyz` creates the collection if it does not exist yet. |
| `GET /metrics` (needs the API key) | Prometheus exposition. Re-evaluates readiness first, so `agentic_rag_ready` and `agentic_rag_dependency_up{dependency=...}` always agree with `/readyz`. |
| Logs | Under `agentic-rag serve` (the image's command), one JSON object per line on stdout and nothing else: `ts, level, logger, message, request_id` plus `event, method, route, status, duration_ms` on access lines and `event="not_ready", failing=[...]` when readiness fails. `route` is the route template, never the raw path or a query string. Started as `uvicorn --factory agentic_rag.server:create` instead, uvicorn prints its own plain-text start-up and access lines (which include query strings), so use `serve`. |
| `X-Request-ID` | On every response, error body (`request_id`) and log line. Quote it in a bug report. A client-supplied id is used only if it matches `[A-Za-z0-9_-]{8,64}`. |

Metrics: `agentic_rag_http_request_duration_seconds{method,route,status_class}` (histogram, buckets 5 ms to 60 s),
`agentic_rag_http_requests_in_flight`, `agentic_rag_errors_total{code}` (stable error codes), `agentic_rag_load_shed_total`,
`agentic_rag_dependency_up{dependency}`, `agentic_rag_ready`. Not provided because the feature does not exist: cache hit rate,
queue depth.

## Failure modes

| Symptom | Likely cause | Look at | Action |
|---|---|---|---|
| `/readyz` 503, `vector_store: unavailable`; searches fail with `VECTOR_STORE_UNAVAILABLE` (503) | Milvus down or unreachable | `agentic_rag_dependency_up{dependency="vector_store"}`, log `not_ready` | Restore Milvus. Clients retry 503s with backoff; no restart of this service is needed, readiness recovers by itself. |
| `/readyz` 503, `lexical_index: unavailable` after a restart | The in-memory BM25 index is still being rebuilt, or the rebuild keeps failing | log `lexical index rebuild failed; retrying` with a traceback (every 5 s, for any exception) | Wait for the store; with a very large collection the rebuild takes time (it scans every chunk). Documents deleted or re-ingested while the rebuild runs are not resurrected by it. |
| `/readyz` 503, `index: incompatible`; searches fail with `409 INDEX_INCOMPATIBLE` | The configured embedding model or vector size is not the one the collection was built with (for example a rolled-out config change) | the check, and the error `details` | Roll the config back, or point at a new collection and re-ingest. Serving a mismatched index would return meaningless rankings, so it is refused on reads as well as writes. |
| Searches fail fast with `VECTOR_STORE_UNAVAILABLE` although the store looks up; `/readyz` `vector_store: timeout` | A store that accepts connections and never answers has used up the store's eight worker threads (one for Milvus Lite) and its queue | `agentic_rag_dependency_up{dependency="vector_store"}`, `/readyz` | Restart the store; the service recovers by itself once calls return. Milvus and Qdrant calls time out after `AGENTIC_RAG_STORE_TIMEOUT_SECONDS` (20); **remote Chroma has no client timeout** (its client library offers none), so only the bounded pool protects the process. |
| Ingest returns `503`/`504` and later `DELETE` says `DOCUMENT_NOT_FOUND`, but the text is still found by search | A new ingest failed after writing some chunks and the best-effort cleanup also failed; the chunks have no commit marker | search hits whose document is not listed | `DELETE /v1/documents/{id}` removes such leftovers (it answers 404 only if there is nothing at all); re-uploading the same bytes also finishes the job. |
| `503 OVERLOADED` with `Retry-After`; `agentic_rag_load_shed_total` rising | More than `AGENTIC_RAG_MAX_INFLIGHT_REQUESTS` concurrent requests | `agentic_rag_http_requests_in_flight`, latency histogram | Add replicas or raise the limit if the datastore has headroom (the default is unmeasured). |
| `502 MODEL_FAILED`, `502 EMBEDDING_FAILED`, `429 UPSTREAM_RATE_LIMITED` | OpenAI errors or rate limits | `agentic_rag_errors_total{code=...}` | Check provider status and quota. A `CONFIGURATION_ERROR` from the provider path means the configured key is rejected. |
| `409 INDEX_INCOMPATIBLE` on ingest | The collection was built with another embedding model or vector size | the error `details` | Use a new collection name, or re-ingest everything with the new model. Mixing models in one collection is refused on purpose. |
| `502 MODEL_OUTPUT_INVALID` | The model returned non-JSON or cited a chunk that was not retrieved | `agentic_rag_errors_total{code="MODEL_OUTPUT_INVALID"}` | Occasional: retry. Frequent: the model or prompt changed; the rate is the signal to evaluate (Step 18). |
| `504 DEADLINE_EXCEEDED` | A request ran past `AGENTIC_RAG_REQUEST_DEADLINE_SECONDS` (55) | latency histogram, `route` | Keep the client's total timeout above this (the SDK default is 60 s) so the typed error arrives first. |
| `401` everywhere | Wrong or rotated API key | `AUTHENTICATION_FAILED` count | Two keys can be active at once: deploy the new key next to the old one, switch clients, then remove the old one. |
| Process refuses to start | Invalid or unknown `AGENTIC_RAG_*` variable, or no API keys | start-up error names the variable, never its value | Fix the variable. `AGENTIC_RAG_ALLOW_UNAUTHENTICATED=true` exists for local development only. |

## Shutdown

Run the server with `agentic-rag serve` (the image does). On SIGTERM it flips to *draining* (`/readyz` returns 503 with
`server: draining`) **while the listening socket is still open**, waits `AGENTIC_RAG_SHUTDOWN_DRAIN_SECONDS` (default 3) so a load
balancer notices, then lets uvicorn stop; `--timeout-graceful-shutdown` (30 s, fixed in `serve`) bounds in-flight requests, and the
clients are closed last. This is tested against a real process (`tests/api/test_live_server.py`). Started with
`uvicorn --factory agentic_rag.server:create` instead, uvicorn closes the listener first, so there is no drain: use `serve`. Set the
orchestrator's termination grace period above drain + graceful timeout + the request deadline (about 90 s). The Kubernetes timing advice
is *unverified* here; the endpoint-removal race it addresses is documented by Kubernetes. The image's `HEALTHCHECK` polls `/healthz`
(liveness), not `/readyz`, so a dependency outage or the BM25 rebuild never makes a container runtime restart a healthy container.

## Behind a prefix or a proxy

`--root-path`/`root_path` is supported: authentication and load shedding look at the path below the prefix, whether the server puts the
prefix into the path (uvicorn does) or not (`tests/api/test_hardening.py`). A reverse proxy should enforce its own read timeouts for
request bodies: an authenticated client that announces a body and never sends it holds one of the `MAX_INFLIGHT_REQUESTS` slots until
it disconnects (anonymous clients are rejected before any body is read and cost no slot).

## CrewAI runtime notes

* The crew runs on its own threads, one per concurrent crew (`AGENTIC_RAG_CREWAI_MAX_CONCURRENT`, default 4); provider calls run on a
  separate pool, so no pool waits on itself. More crews than that (plus an equal queue) get `503 OVERLOADED` at once.
* `AGENTIC_RAG_CREWAI_MAX_SECONDS` must be below `AGENTIC_RAG_REQUEST_DEADLINE_SECONDS`; the server refuses to start otherwise. When the
  request deadline fires, the crew is stopped from making further provider calls (a call already in flight finishes in the background,
  bounded by `AGENTIC_RAG_CREWAI_PROVIDER_TIMEOUT_SECONDS`).
* CrewAI normally writes the question, retrieved document text and answers of every run to a SQLite file in the user's data
  directory; this service replaces that store with a no-op. Importing `crewai` still creates an (empty) data directory under
  `XDG_DATA_HOME`: on a read-only filesystem point it at a writable path, or start-up fails with a `CONFIGURATION_ERROR`.

## Known limits

* The BM25 index is per process and lives in memory: every replica rebuilds it at start-up and sees only its own writes until restarted.
  For several replicas behind one collection, a document ingested through replica A is not found lexically by replica B until B
  restarts. Dense retrieval is unaffected. (Moving lexical search into Milvus is the intended fix; it is not done.)
* Per-document locking is per process. Two replicas ingesting the same bytes at the same moment both succeed and converge on the same
  content-addressed ids (idempotent), but a delete racing an ingest of the same document on another replica is not coordinated.
* Listing documents reads at most 16,000 documents and refuses beyond that (`LIMIT_EXCEEDED`).
* Deleting a document on one replica does not remove it from another replica's in-memory BM25 index until that replica restarts, so a
  deleted document can still be returned by lexical search there (its dense hits are gone). A document ingested or re-uploaded after a
  chunking or parser change is re-indexed; the old chunks are swept.
* Document ids are `doc_` plus 32 hex characters derived from the file *type and bytes*: the same bytes uploaded as `.txt` and as `.html` are
  two documents. Any other id is `DOCUMENT_NOT_FOUND` without touching the store.
* A parser or the chunker can still be slow on a pathological file (HTML parses at about 2.4 s per MB, so a 25 MiB upload can occupy a
  worker thread for about a minute). The service bounds *how many* such threads exist (`AGENTIC_RAG_MAX_CONCURRENT_INGESTS`, held until the
  thread really returns, even after the request was cancelled) and refuses text that cannot fit the chunk limit before chunking it; it
  cannot interrupt one that is running. Lower `AGENTIC_RAG_MAX_UPLOAD_BYTES` if that matters to you.
