# Operations runbook

What an on-call engineer can see, and what to do about the top failure modes. Everything here is exercised by tests unless marked
*unverified*.

## Probes and signals

| Signal | Meaning |
|---|---|
| `GET /healthz` | The process answers. Checks **no** dependency, so an outage never makes an orchestrator restart healthy pods. |
| `GET /readyz` | `200` only if every dependency check passes and the server is not draining; otherwise `503` with `{"ready": false, "checks": {...}}`. Checks: `vector_store` (Milvus reachable), `lexical_index` (in-memory BM25 rebuilt from the store), `server` (`draining` during shutdown). Values are `ok`, `unavailable` or `timeout`; never raw dependency text. |
| `GET /metrics` (needs the API key) | Prometheus exposition. Re-evaluates readiness first, so `agentic_rag_ready` and `agentic_rag_dependency_up{dependency=...}` always agree with `/readyz`. |
| Logs | One JSON object per line on stdout: `ts, level, logger, message, request_id` plus `event, method, route, status, duration_ms` on access lines and `event="not_ready", failing=[...]` when readiness fails. `route` is the route template, never the raw path. |
| `X-Request-ID` | On every response, error body (`request_id`) and log line. Quote it in a bug report. A client-supplied id is used only if it matches `[A-Za-z0-9_-]{8,64}`. |

Metrics: `agentic_rag_http_request_duration_seconds{method,route,status_class}` (histogram, buckets 5 ms to 60 s),
`agentic_rag_http_requests_in_flight`, `agentic_rag_errors_total{code}` (stable error codes), `agentic_rag_load_shed_total`,
`agentic_rag_dependency_up{dependency}`, `agentic_rag_ready`. Not provided because the feature does not exist: cache hit rate,
queue depth.

## Failure modes

| Symptom | Likely cause | Look at | Action |
|---|---|---|---|
| `/readyz` 503, `vector_store: unavailable`; searches fail with `VECTOR_STORE_UNAVAILABLE` (503) | Milvus down or unreachable | `agentic_rag_dependency_up{dependency="vector_store"}`, log `not_ready` | Restore Milvus. Clients retry 503s with backoff; no restart of this service is needed, readiness recovers by itself. |
| `/readyz` 503, `lexical_index: unavailable` after a restart | The in-memory BM25 index is still being rebuilt, or the rebuild keeps failing because Milvus is down | log `lexical index rebuild failed (CODE); retrying` (every 5 s) | Wait for Milvus; with a very large collection the rebuild takes time (it scans every chunk). |
| `503 OVERLOADED` with `Retry-After`; `agentic_rag_load_shed_total` rising | More than `AGENTIC_RAG_MAX_INFLIGHT_REQUESTS` concurrent requests | `agentic_rag_http_requests_in_flight`, latency histogram | Add replicas or raise the limit if the datastore has headroom (the default is unmeasured). |
| `502 MODEL_FAILED`, `502 EMBEDDING_FAILED`, `429 UPSTREAM_RATE_LIMITED` | OpenAI errors or rate limits | `agentic_rag_errors_total{code=...}` | Check provider status and quota. A `CONFIGURATION_ERROR` from the provider path means the configured key is rejected. |
| `409 INDEX_INCOMPATIBLE` on ingest | The collection was built with another embedding model or vector size | the error `details` | Use a new collection name, or re-ingest everything with the new model. Mixing models in one collection is refused on purpose. |
| `502 MODEL_OUTPUT_INVALID` | The model returned non-JSON or cited a chunk that was not retrieved | `agentic_rag_errors_total{code="MODEL_OUTPUT_INVALID"}` | Occasional: retry. Frequent: the model or prompt changed; the rate is the signal to evaluate (Step 18). |
| `504 DEADLINE_EXCEEDED` | A request ran past `AGENTIC_RAG_REQUEST_DEADLINE_SECONDS` (55) | latency histogram, `route` | Keep the client's total timeout above this (the SDK default is 60 s) so the typed error arrives first. |
| `401` everywhere | Wrong or rotated API key | `AUTHENTICATION_FAILED` count | Two keys can be active at once: deploy the new key next to the old one, switch clients, then remove the old one. |
| Process refuses to start | Invalid or unknown `AGENTIC_RAG_*` variable, or no API keys | start-up error names the variable, never its value | Fix the variable. `AGENTIC_RAG_ALLOW_UNAUTHENTICATED=true` exists for local development only. |

## Shutdown

On SIGTERM the server flips to *draining* (`/readyz` returns 503 with `server: draining`), waits `AGENTIC_RAG_SHUTDOWN_DRAIN_SECONDS`
(default 3) so a load balancer notices, then closes its clients; uvicorn's `--timeout-graceful-shutdown` bounds in-flight requests (the
container image sets 30 s). Set the orchestrator's termination grace period above drain + graceful timeout + the request deadline
(about 90 s). The Kubernetes timing advice is *unverified* here; the endpoint-removal race it addresses is documented by Kubernetes.

## Known limits

* The BM25 index is per process and lives in memory: every replica rebuilds it at start-up and sees only its own writes until restarted.
  For several replicas behind one collection, a document ingested through replica A is not found lexically by replica B until B
  restarts. Dense retrieval is unaffected. (Moving lexical search into Milvus is the intended fix; it is not done.)
* Per-document locking is per process. Two replicas ingesting the same bytes at the same moment both succeed and converge on the same
  content-addressed ids (idempotent), but a delete racing an ingest of the same document on another replica is not coordinated.
* Listing documents reads at most 16,000 documents and refuses beyond that (`LIMIT_EXCEEDED`).
