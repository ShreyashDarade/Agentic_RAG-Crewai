# Security: threat model and controls

Written from the code as it stands. "Test" names the check that proves a control; "review" means no tool checks it.

## Who sends what

| Actor | Sends | Worst realistic input |
|---|---|---|
| Authenticated API client (holds a static key) | documents, questions, ids | huge or hostile files, zip bombs, prompt-injection text inside documents, malformed ids, floods of requests |
| Anyone who can reach the port | requests without a key | the same, unauthenticated; probing for `/metrics`, `/docs` |
| Document authors (indirect) | text that is later shown to the model | "ignore previous instructions" and exfiltration instructions inside indexed content |
| Upstream providers (OpenAI, Milvus) | responses | wrong shapes, rate limits, outages, error text containing secrets |
| Operator | environment variables | typos, weak keys |

Out of scope by decision (ADR-0006): users, roles, tenants, per-document access control. **Every holder of an API key can read every
indexed document and delete any of them.** The service is meant to sit behind a trusted gateway or be used by one trust domain.

## STRIDE summary

| Threat | Where | Control | Evidence |
|---|---|---|---|
| Spoofing | any `/v1` route and `/metrics` | Static bearer key(s) from the environment, constant-time comparison against every configured key; 401 + `WWW-Authenticate`; the app will not start without keys unless unauthenticated mode is explicitly enabled, and unauthenticated mode together with keys is a start-up error. The key is checked by a middleware **before any request body is read or any in-flight slot is taken** (an anonymous 25 MB upload costs the server nothing); the route dependency checks again | `test_every_v1_route_requires_the_api_key`, `tests/api/test_hardening.py`, M23, M45, M66 |
| Tampering (index) | ingest | Content-addressed ids; model/size mismatch refused; filters built from a closed alphabet | conformance hostile-filter checks, M22 |
| Tampering (prompt) | retrieved text | Chunks are HTML-escaped and wrapped as untrusted data; the model must return JSON; every citation must be a retrieved chunk id | `test_a_citation_that_was_not_retrieved_is_rejected`, pipeline conformance |
| Repudiation | all | Request id on every request, response, error body and log line; JSON access log | `test_request_id_on_every_response...` |
| Information disclosure | errors | Fixed public messages; dependency text never serialised; validation errors never echo the input; unexpected exceptions become a generic 500; every adapter's error mapping is exercised with secret-bearing provider and store errors | `test_dependency_text_never_reaches_the_wire`, `tests/conformance/test_error_text.py`, `test_unexpected_exceptions_become_a_generic_500...`, M25, M95-M97 |
| Information disclosure | secrets | `SecretStr`, never logged; settings errors name variables, not values (and a secret given both directly and as a `_FILE` is an error); no credential-shaped string is tracked in git | `test_settings_validate_at_start_and_never_echo_values`, `tests/architecture/test_no_secrets_committed.py`, M65, M108 |
| Information disclosure | CrewAI persistence | CrewAI stores each run's task outputs (question, retrieved document text, answers) in a plaintext SQLite file under the user's data directory; this service replaces that store with a no-op, so nothing is written | `tests/unit/test_crew_hardening.py::test_a_request_writes_no_files`, M81 |
| Denial of service | upload | Content-Length and streamed byte cap before the body is spooled (after authentication); exact cap in the service; extension allow-list; chunk-count cap (also checked **before** chunking, from the text length); DOCX member/size cap (8 MiB per part, 32 MiB total declared); PDF page cap | `test_oversized_upload...`, `test_chunked_oversized_body_is_cut_off`, `tests/unit/test_limits.py`, `tests/unit/test_parsers_hardening.py`, M59, M104 |
| Denial of service | load | Bounded in-flight requests (authenticated only) with 503 + Retry-After; a deadline on **every** use case; bounded embedding concurrency and batch size; every dependency call on a bounded pool so a hung dependency fails fast instead of starving the process | `test_overload_is_shed...`, `tests/unit/test_service_hardening.py`, `tests/unit/test_blocking.py`, `tests/integration/test_hung_store.py`, M47, M58, M69 |
| Denial of service | slow body | An **authenticated** client that announces a body and never sends it holds one in-flight slot until it disconnects; there is no body-read timeout in the app (uvicorn has none). Put a reverse proxy with read timeouts in front | none (documented gap) |
| Denial of service | CPU in parsers and chunker | **Not bounded by time**: a pathological HTML/PDF/DOCX can still burn a worker thread (HTML parses at about 2.4 s/MB). Bounded instead by *count*: ingest slots are held until the thread really returns, text that cannot fit the chunk limit is refused before chunking, the chunker is linear, and DOCX parts are capped. The real fix, a subprocess with a timeout, is not built | `tests/unit/test_service_hardening.py::test_timed_out_ingests_cannot_pile_up_parser_threads`, M57 |
| Elevation of privilege | uploads | The display name is reduced to a base name; **no file is written to disk by name** (bytes go straight to the parser); a document id is only ever `doc_` + 32 hex characters and any other string is `DOCUMENT_NOT_FOUND` without reaching the store | `test_unsupported_type_and_hostile_names`, `tests/unit/test_service_hardening.py`, M53 |
| Elevation of privilege | filters | Closed alphabet `[A-Za-z0-9._-]{1,128}`; Milvus Lite lacks `filter_params`, so templating is not the single path | M22 |
| SSRF | n/a | The service fetches no URLs supplied by users. (Provider base URLs come from settings only.) | review |
| CORS | browsers | Off by default; if enabled, an explicit origin list, no credentials | `test_cors_is_off_by_default` |

## Other controls and honest gaps

* Secrets only from the environment; the one prototype `.env` that was committed is out of the index but **still in git history**:
  rotate anything that was ever in it.
* Dependencies are audited with `pip-audit --strict` against the lock (clean at the time of writing; CI runs it per PR). A weekly
  scheduled audit is not configured.
* `/docs` and `/openapi.json` are served without the key (they describe the API, not data); so are `/healthz` and `/readyz`. Disable the
  former behind the gateway if that matters to you.
* Prompt injection cannot be eliminated (OWASP LLM01 says so). The controls above limit the blast radius: with the `direct` pipeline
  the model has no tools and no write path; with the `crewai` pipeline the only tool is read-only retrieval bound to the request's
  filter, limited by a search budget and by the same context budget as the first prompt. Output is validated and citations must name
  chunks the model was *shown*. A hostile document can still make an answer wrong; and in the crew, one agent's rewrite of a chunk
  reaches the next agent as ordinary prose, so the "untrusted data" marking does not follow it (residual risk, unmitigated).
* No rate limit per key; shedding is global (authenticated requests only).
* A leftover `AGENTIC_RAG_ALLOW_UNAUTHENTICATED=true` next to configured keys is a start-up error, not a silently open server.
* The embedded SDK runs the engine with the caller's privileges and no authentication by design.
