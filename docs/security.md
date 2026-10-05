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
| Spoofing | any `/v1` route | Static bearer key(s) from the environment, constant-time comparison against every configured key; 401 + `WWW-Authenticate`; the app will not start without keys unless unauthenticated mode is explicitly enabled | `test_every_v1_route_requires_the_api_key`, M23 |
| Tampering (index) | ingest | Content-addressed ids; model/size mismatch refused; filters built from a closed alphabet | conformance hostile-filter checks, M22 |
| Tampering (prompt) | retrieved text | Chunks are HTML-escaped and wrapped as untrusted data; the model must return JSON; every citation must be a retrieved chunk id | `test_a_citation_that_was_not_retrieved_is_rejected`, pipeline conformance |
| Repudiation | all | Request id on every request, response, error body and log line; JSON access log | `test_request_id_on_every_response...` |
| Information disclosure | errors | Fixed public messages; dependency text never serialised; validation errors never echo the input; unexpected exceptions become a generic 500 | `test_dependency_text_never_reaches_the_wire`, `test_unexpected_exceptions_become_a_generic_500...`, M25 |
| Information disclosure | secrets | `SecretStr`, never logged; settings errors name variables, not values; `.env` not committed | `test_settings_validate_at_start_and_never_echo_values` |
| Denial of service | upload | Content-Length and streamed byte cap before the body is spooled; exact cap in the service; extension allow-list; chunk-count cap; DOCX member/size cap; PDF page cap | `test_oversized_upload...`, `test_chunked_oversized_body_is_cut_off`, limit tests |
| Denial of service | load | Bounded in-flight requests with 503 + Retry-After; per-request deadline; bounded embedding concurrency and batch size | `test_overload_is_shed...`, `test_deadline_is_a_typed_error` |
| Denial of service | CPU in parsers | **Not bounded by time**: a pathological PDF/DOCX can still burn a worker thread for a long time (review; the fix is a subprocess with a timeout) | none |
| Elevation of privilege | uploads | The display name is reduced to a base name; **no file is written to disk by name** (bytes go straight to the parser) | `test_unsupported_type_and_hostile_names` |
| Elevation of privilege | filters | Closed alphabet `[A-Za-z0-9._-]{1,128}`; Milvus Lite lacks `filter_params`, so templating is not the single path | M22 |
| SSRF | n/a | The service fetches no URLs supplied by users. (Provider base URLs come from settings only.) | review |
| CORS | browsers | Off by default; if enabled, an explicit origin list, no credentials | `test_cors_is_off_by_default` |

## Other controls and honest gaps

* Secrets only from the environment; the one prototype `.env` that was committed is out of the index but **still in git history**:
  rotate anything that was ever in it.
* Dependencies are audited with `pip-audit --strict` against the lock (clean at the time of writing; CI runs it per PR). A weekly
  scheduled audit is not configured.
* `/docs` and `/openapi.json` are served without the key (they describe the API, not data). Disable them behind the gateway if that
  matters to you.
* Prompt injection cannot be eliminated (OWASP LLM01 says so). The controls above limit the blast radius: the model has no tools and no
  write path, its output is validated, and citations are checked. A hostile document can still make an answer wrong.
* No rate limit per key; shedding is global.
* The embedded SDK runs the engine with the caller's privileges and no authentication by design.
