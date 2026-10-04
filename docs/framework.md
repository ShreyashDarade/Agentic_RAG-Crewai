# Framework

How this system may be built and may change. Written before the rewrite; the code is made to fit it.
Status: **draft for owner review (Checkpoint 1)**. Nothing below is implemented yet; "enforced by" names the tool that
*will* enforce it and the step that will add it. A rule with no tool is labelled **review rule**.
Evidence for choices: [baseline](baseline.md) (what exists), [research](research.md) (prior art, cited by row),
[ADRs](adr/) (the decisions that make an item "fixed").

## 1. Scope and non-goals

**Product.** A retrieval-augmented question-answering service: ingest documents (PDF, DOCX, images/OCR, text, HTML,
spreadsheets, slides), index them in Milvus, answer questions with cited sources. Consumers: Python programs that call
it over HTTP or embed it in-process, and anything that can speak the checked-in OpenAPI contract.

**Owner decisions (recorded from the scoping questions).**

| Decision | Choice | Consequence |
|---|---|---|
| CrewAI | "Make it real": the four stages run as an actual CrewAI crew | Shipped as the `crewai` pipeline adapter behind the `AnswerPipeline` port. A second, plain `direct` pipeline is also registered. The *default* is the one the evaluation (Step 18) shows is not worse on quality, latency or cost; if the crew loses, `direct` is the default and the result is documented. CrewAI is an extra, never a base dependency. (ADR-0005) |
| SDK | Python only | One distribution; thin HTTP client by default, engine as an extra. No generated clients; other languages use `docs/openapi.json`. (ADR-0003, 0004) |
| Providers and CrewAI RAG scope (added after Checkpoint-1 draft) | Use CrewAI's retrieval/embedding/chunking/storage/memory capabilities, with multiple LLM, embedding and vector-DB providers | Supersedes the "OpenAI + Milvus only" default. All CrewAI RAG features live behind ports in `adapters.crewai`; Milvus, Chroma and Qdrant are all registered vector stores; provider support is declared per provider as verified/unverified. Bounded by the disposition table in research §4. (ADR-0010) |
| Auth | Built-in static API key | `Authorization: Bearer <key>` on every route except liveness/readiness; keys come from the environment; two keys may be active at once for rotation. No users, roles or tenants. (ADR-0006) |

**Defaults chosen by me (not owner decisions; change by ADR).**
Python 3.11 – 3.13 supported (3.14 added when every engine dependency publishes wheels; not verified today).
Primary vector store: Milvus (HNSW, COSINE); additional registered stores: Chroma and Qdrant via CrewAI. Primary LLM/embeddings: OpenAI; others via CrewAI (ADR-0010).
Package manager `uv` with a committed lock file. Version starts at `0.1.0` (the current `2.0.0` label describes a build
that never answered a query, see baseline §2). Public HTTP prefix `/v1`.

**Non-goals.** Using CrewAI features outside the RAG parts (A2A, MCP client, skills, scaffolding CLI); User accounts, tenants, per-document ACLs; admin UI; a hosted offering; training or fine-tuning models;
non-Python SDKs; multi-collection routing by request (one configured collection per deployment); streaming resumption
(a query stream is not resumable, research §2 A6).

## 2. Layers and allowed dependency directions

Import package `agentic_rag` (distribution `agentic-rag`). Arrows mean "may import".

```
 transports:   api (FastAPI)   cli   sdk (client, embedded backend)          composition root: container
                   │            │        │                                         │ (only module that imports adapters)
                   ▼            ▼        ▼                                         ▼
 application:   service  (one method per use case; takes ports, returns contract models)
                   │
                   ▼
 ports:         Protocols + plain value types           adapters.<technology>  ──► ports
                   │                                     (one third-party library per package)
                   ▼
 foundation:    contracts (wire models)   errors   registry   — stdlib (+ pydantic for contracts) only
```

| Layer | Package | May import | Must not import |
|---|---|---|---|
| foundation | `errors`, `registry` | stdlib | everything else |
| foundation | `contracts` | stdlib, pydantic, `errors` | fastapi, httpx, any adapter library |
| ports | `ports` | foundation, stdlib | any third-party library |
| application | `application` | ports, foundation | fastapi, httpx, pymilvus, openai, crewai, any adapter |
| adapters | `adapters.<name>` | ports, foundation, **its own** third-party library | other adapters, application, transports |
| transports | `api`, `cli` | application, foundation | ports directly, adapters |
| SDK | `sdk`, `client`, `models`, `extend`, `testing` | foundation, stdlib, httpx, pydantic | fastapi, pymilvus, openai, crewai, torch, adapters, application (the embedded backend imports `container` lazily and fails with an `ImportError` naming the `engine` extra) |
| composition root | `container` | everything | — (nothing may import it except `api` startup, `cli`, and the embedded SDK backend) |

Third-party confinement (each library lives in exactly one package): `pymilvus` → `adapters.milvus`; `openai` →
`adapters.openai`; `crewai`, `chromadb`, `qdrant_client`, `docling`, `litellm` → `adapters.crewai` (all CrewAI-backed adapters: `pipeline`, `chat_model`, `embedder`, `vector_store`, `knowledge`, `memory`, `tools`, `chunker`, `parser`); `fastapi`/`starlette`/`uvicorn` → `api`; `httpx` → `sdk`;
`sentence_transformers`/`torch` → `adapters.rerank_cross_encoder`; `easyocr`/`cv2` → `adapters.ocr_easyocr`;
`fitz` → `adapters.parser_pdf`; `docx` → `adapters.parser_docx`; … one row per library in `tests/architecture/confinement.toml`.

## 3. FIXED and FREE

| FIXED (change needs an ADR + the compatibility process in §9) | FREE (change by ordinary PR) |
|---|---|
| Public SDK surface: names in `agentic_rag.__all__`, `client`, `errors`, `models`, `extend`, `testing` (snapshot `docs/public_api.txt`) | Everything under `application`, `adapters.*`, `api` internals, `_`-prefixed names |
| Wire contract: routes under `/v1`, request/response models, headers, status codes (snapshot `docs/openapi.json`) | Prompts, chunk sizes, retrieval defaults (but defaults must stay evidence-backed, §11) |
| Error taxonomy: class → code → HTTP status, public messages (`docs/error_codes.json`, append-only) | Log wording, metric *help* strings, internal module layout inside a layer |
| Ports (method names, types, failure contract) and the extension contracts of §7 | Choice of algorithm inside an adapter |
| Layer rules and third-party confinement (§2) | Test layout |
| Retry matrix and idempotency semantics (§8) | Concrete backoff constants *within* the documented bounds |
| Stored data identity: chunk id = content-addressed hash (ADR-0007) | Index parameters (`M`, `efConstruction`, `ef`) — changing them is a *reindex*, recorded in collection metadata |

## 4. Stability tiers

`stable` — covered by §9 compatibility rules. `experimental` — may change in a minor release, marked with
`@experimental` and listed under "Experimental" in `docs/public_api.txt`; never used by a stable name. `internal` — a
leading underscore or anywhere outside `__all__`; no promise. A name is public **iff** it is in an `__all__`
(research §1 "API-surface (public definition)"). *Enforced by:* `griffe check` + the `public_api.txt` snapshot test (Step 11); the tier
column is checked by a test that every `__all__` name has exactly one tier.

## 5. Error model (ADR-0002)

* Root `RagError(Exception)`. Each subclass defines **its own** class attributes `code` (UPPER_SNAKE, regex
  `[A-Z][A-Z0-9_]+[A-Z0-9]`, ≤ 63 chars), `http_status`, `public_message` (fixed template; dynamic values go in
  `details`, never in the message). A subclass that does not define `code` is rejected at class-creation time
  (`__init_subclass__`), so a child cannot inherit and silently reuse a parent's code.
* Dependency failures are wrapped: `raise VectorStoreUnavailable(...) from exc`. Raw dependency text goes to logs only and is
  never serialised (`__cause__` is not part of any body). "Return empty/default and log" is a defect.
* Wire form: `application/problem+json` (RFC 9457) with extension members `code`, `request_id`, `details`,
  optional `errors[]` for validation; `status` equals the HTTP status.
* Catalog `code → class` is built from the class hierarchy; the HTTP client rebuilds the same exception class from a body;
  the embedded backend raises the same class directly. An unknown code from a newer server becomes `RagStatusError`
  keeping `code`, `status`, `request_id`, `details`.
* *Enforced by:* round-trip test over every code; a test that injects a secret-bearing exception into each adapter and
  asserts the string appears in no response body/header; `grep`-style architecture test that forbids `except Exception`
  that does not re-raise or wrap, and bare `except:`, outside a short, reviewed allow-list (Step 9/11).
  Codes are append-only: `docs/error_codes.json` snapshot test.

## 6. Wire contract rules

1. Routes are thin: parse → call `application.Service` → map result/errors. A route is ≤ ~15 lines and imports no port or
   adapter. *Enforced by:* import-linter (`api` may import only `application`, foundation) + a size/AST check.
2. `docs/openapi.json` is generated by `create_app(fake_container).openapi()` (sorted keys, no services needed), checked in,
   and regenerated only by `scripts/export_openapi.py`. *Enforced by:* CI regenerate-and-diff; `oasdiff breaking
   --fail-on ERR` against the last release tag for incompatibility.
3. Additive-only inside `/v1`: new optional request fields and new response fields are allowed; renaming, removing,
   retyping, or tightening is a new major (`/v2`). Request models forbid unknown fields; **response models in the SDK ignore
   them** (research §2 A5). Enums that servers may extend are documented as open.
4. Every response (including 4xx/5xx/429/503) carries `X-Request-ID`; an inbound id is accepted only if it matches
   `^[A-Za-z0-9_-]{8,64}$`, else regenerated (research §1 "Request IDs", §2 D6).
5. Mutating routes accept `Idempotency-Key`; same key + different body → 422 `IDEMPOTENCY_KEY_REUSED`; in flight → 409.
6. Overload: bounded in-flight work; beyond it, immediate `503` with integer `Retry-After` and code `OVERLOADED`
   (research §2 B8). Health and readiness are exempt.
7. Destructive and server-filesystem surfaces (`DELETE` collection, ingest-by-server-path) are **off** unless the settings
   enable them, and are never reachable without the API key even then.

## 7. SOLID as rules (each has a check that fails on violation)

| | Rule | Check |
|---|---|---|
| **S** | Parsers parse, chunkers chunk, routes translate, services orchestrate; no module imports across its role (layers in §2). | import-linter layers + forbidden contracts; `exhaustive = true` so a new top-level module must be classified. |
| **O** | Behaviour is added by registering a component in a name→factory registry (entry-point group `agentic_rag.<kind>`, or a module named in config exposing `register(registries)`); never by editing a switch. Unknown name → `UnknownComponent` listing valid names. | `tests/architecture/test_open_closed.py`: a fake plug-in module in the test suite registers a new chunker and a new vector store and the service uses them with **zero edits under `src/`**. |
| **L** | Every implementation of a port passes the same conformance suite *including failure behaviour* (timeout, bad input, unavailable). | `agentic_rag.testing.contracts` abstract suites run against every built-in; a deliberately broken implementation per suite must be rejected (mutation proof). |
| **I** | Ports are small, one capability each (e.g. `VectorWriter`, `VectorSearcher`, `Embedder`, `ChatModel`, `Reranker`, `DocumentParser`, `Chunker`, `AnswerPipeline`, `JobQueue`, `Clock`); consumers depend only on what they call. | AST test: a port Protocol has ≤ 6 methods; a service constructor parameter's type is the narrowest port it uses (review rule for "narrowest"). |
| **D** | High-level code depends on ports; **one** composition root chooses concrete classes. | import-linter: ports/application import no framework or adapter; only `container` imports `adapters.*`. |

Extension points (fixed contracts): `vector_store`, `embedder`, `chat_model`, `reranker`, `parser` (per file type),
`chunker`, `answer_pipeline`, `job_queue`, `cache`. Each registry lives in `registry`; each has a `Capabilities`
declaration (e.g. `supports_filter`, `max_batch`) so conformance tests skip by capability rather than by name.

## 8. Retry, timeout and idempotency (ADR-0009, research §2 B1–B8)

| Failure | Retry? |
|---|---|
| Never sent (connect refused/DNS) | yes, every call |
| Refused before work: `429`, `503` | yes, every call; honour `Retry-After`, capped (default 60 s) and bounded by remaining deadline |
| Ambiguous: connection reset after send, `408`, `502`, `504` | only idempotent calls (reads, `DELETE`, and mutations carrying an `Idempotency-Key`) |
| Read timeout | **never** auto-retried |
| Non-idempotent call without key (e.g. appending a chat turn) | never re-sent |
| `DELETE` retry that sees `404` | treated as success of the first attempt |

Client total deadline (default 60 s) > server request deadline (default 55 s) > sum of upstream budgets, each computed
from the remaining budget, so the server's typed `DEADLINE_EXCEEDED` (504) arrives before the client gives up.
Retries happen at one layer only (the outermost); server-internal dependency calls retry at most once.
Full-jitter backoff, per-client retry token bucket. *Enforced by:* SDK matrix tests against a fault-injecting server;
a test asserting `client_deadline > server_deadline` from the default settings. Exact constants are proposals (Step 12).

## 9. SDK rules (ADR-0004)

1. One facade, public methods written once, over a narrow `Backend` protocol; `HttpBackend` and `EmbeddedBackend` only
   move requests/results. Same models, errors, method names in both. A method that cannot work in a mode does not exist there.
2. Async is the implementation; the blocking client is a bridge over **one** background event loop. A blocking call made
   inside a running loop raises `UsageError`, never stalls. `close()` cancels in-flight calls and is idempotent; a failed
   constructor leaks no thread, loop or socket.
3. Thin install: base dependencies are `httpx` and `pydantic` only. Engine names are absent from `__all__` on a thin install
   and raise `ImportError` naming the extra. Extras: `engine`, `server`, `worker`, `crewai`, `ocr`, `rerank`, `dev`.
4. Safety: percent-encode every path id; strict response validation but unknown fields ignored; spec-correct SSE
   (split only on `\n`, `\r\n`, `\r`; incremental UTF-8; server escapes U+2028/2029/0085); conflicting configuration is an
   error; an upload stream resumed from mid-file sends identical bytes on both transports.
5. *Enforced by:* `tests/sdk/test_parity.py` (one test body over {embedded, HTTP-through-the-real-app}); a clean-venv test that
   imports the thin client and asserts `sys.modules` contains none of `fastapi, pymilvus, openai, crewai, torch, …`;
   `mypy --strict` + `py.typed` wheel check.

## 10. Versioning and deprecation (ADR-0008)

SemVer for the SDK surface and for the wire contract (`/v1` carries only the major). `0.y.z` until the contract is frozen;
then `1.0.0`. A deprecation = `deprecated(since, remove_in, alternative, escalate_in)` from `agentic_rag._compat`:
validates metadata (removal only in a later **major**), emits `AgenticRagDeprecationWarning` with correct `stacklevel`,
escalates to a visible warning in the last minor. The repo's own test suite turns the library's deprecation warnings into errors.

| Statement | Status |
|---|---|
| Removing/renaming/moving a public name or parameter, adding a required parameter, changing a default, removing a base class | **machine-enforced** (`griffe check --against <last tag>`) |
| Removing/retyping a wire field, removing a route, new required request field | **machine-enforced** (`oasdiff breaking`) |
| Deprecation metadata well-formed; removal in a later major; warning category/stacklevel | **machine-enforced** (decorator validation + tests) |
| "Two minor releases and ≥ 6 months of notice before removal" | **review rule — not machine-enforced** (version numbers cannot prove it) |
| Behaviour change with the same signature; error-message or retry-default changes; changelog entry | **review rule** (PR template checklist) |

## 11. Configuration, security, quality rules

* One typed `Settings` (pydantic-settings), built once at startup (not at import); invalid or unknown values are errors;
  the process exits before binding a port. Secrets come from the environment (or `*_FILE` mounts), are `SecretStr`, never
  logged, never in error text. `config/.env` leaves version control; `.env.example` has no values. *Enforced by:* settings
  tests; secret-redaction test; secret scanner in CI (Step 15/20).
* Every attacker- or accident-controlled quantity is bounded by a setting with a default: upload bytes, pages, pixels,
  archive members/uncompressed bytes, history length, query length, batch size, `top_k`, concurrency, queue depth, timeouts.
  *Enforced by:* one test per limit that exceeds it and sees its typed rejection.
* Filters sent to Milvus are built only in one module with templated parameters and allow-listed field names; ids are
  regex-validated; user text never reaches an expression string. *Enforced by:* injection-payload tests; forbidden-import
  contract on `pymilvus` filter helpers outside that module.
* Uploads: extension allow-list **and** magic bytes, server-generated names, streamed size cap, path resolved and checked
  `is_relative_to(root)`. Retrieved text is delimited as untrusted data in prompts; the crew's tools are read-only retrieval.
* Dependencies audited (`pip-audit --strict` on the lock) per PR and weekly.
* A default (retrieval depth, fusion, rerank, MMR, multi-query, `ef`, crew vs direct) is **backed by a measurement or
  marked "unmeasured"** in `docs/evaluation.md`; a feature that hurts defaults OFF. *Enforced by:* a test that every
  default in `Settings` has an entry in `docs/defaults.toml` with `evidence = "<run id>" | "unmeasured"`.
* Observability contract: request id on every request/response/log line/error body; `/healthz` checks no dependency;
  `/readyz` fails honestly when Milvus is down or models are not loaded; metrics use bounded labels (route template, not path).

## 12. Governance table (rule → check → where it runs)

"Seen fail" is recorded in the Step-11 mutation log: break the rule on purpose, watch the check fail, revert.

| # | Rule | Check | Runs in | Seen fail |
|---|---|---|---|---|
| G1 | Layer directions (§2) | import-linter `layers`, `exhaustive=true` | CI `arch`, pre-commit | Step 11 |
| G2 | Third-party confinement; thin client imports stdlib+httpx+pydantic only | import-linter `forbidden` (`include_external_packages`) + `tests/architecture/test_confinement.py` + clean-venv `sys.modules` test | CI `arch`, `packaging` | Step 11 |
| G3 | Every package classified; every heavy library confined | exhaustive test over the package tree and `confinement.toml` | CI `arch` | Step 11 |
| G4 | Routes only call the service | import-linter + AST size/call check | CI `arch` | Step 11 |
| G5 | Public API unchanged unless intended | `public_api.txt` snapshot + `griffe check --against <tag>` | CI `contract` | Step 11 |
| G6 | Wire contract unchanged unless intended | `openapi.json` regenerate-and-diff + `oasdiff breaking` | CI `contract` | Step 11 |
| G7 | Error codes append-only, unique, round-trip | `error_codes.json` snapshot + catalog tests | CI `unit` | Step 9/11 |
| G8 | Strict typing on the public package; `py.typed` shipped | `mypy --strict`; wheel content test | CI `lint`, `packaging` | Step 11 |
| G9 | Every port implementation conforms | `agentic_rag.testing.contracts` suites over all built-ins and the fake plug-in | CI `conformance` | Step 11 |
| G10 | Open-closed: add a component with zero core edits | fake plug-in test | CI `arch` | Step 11 |
| G11 | Transports behave identically | parity suite, plus two-transports-one-engine comparison | CI `parity` | Step 13 |
| G12 | No silent fallbacks | AST test for swallowed broad excepts; fault-injection tests expect typed errors | CI `unit` | Step 9/11 |
| G13 | Limits enforced | one exceed-the-limit test per limit | CI `unit` | Step 15 |
| G14 | Defaults evidence-backed | `defaults.toml` test + `eval check` regression gate | CI `eval` | Step 18 |
| G15 | Docs match code (claims in README/framework) | docs-vs-code grep test for routes, settings names, extras, codes | CI `docs` | Step 20 |
| G16 | Dependencies free of known advisories | `pip-audit --strict` | CI `security` (PR + weekly) | Step 15 |
| G17 | Lint/format | ruff check + format | CI `lint` | n/a (tool defaults) |
| G18 | Deprecation metadata valid; own warnings are errors | decorator tests; pytest `filterwarnings=error` for the package's own category | CI `unit` | Step 14 |
| G19 | "Two minors of notice" before removal | **review rule — not machine-enforced** | PR checklist | — |
| G20 | A change to a FIXED item has an ADR | **review rule — not machine-enforced**; mitigated: G5/G6/G7 snapshots fail, forcing the PR to touch a file that the PR template ties to an ADR link | PR template | — |
| G21 | "Narrowest port" for each consumer | **review rule — not machine-enforced** (size of Protocol *is* checked) | PR checklist | — |

## 13. Open items for the owner at this checkpoint

0. **Scope amendment received.** After the draft was written you asked for CrewAI's RAG features to be used as fully as possible with multiple providers. That is recorded as ADR-0010 and research §4. Most providers cannot be verified here (no credentials; Docker registry blocked), so `docs/providers.md` will report each as verified or unverified. Please confirm the bounded reading of "100%" (RAG parts only; network/DB/write tools off by default).

1. **Real Milvus for tests.** This sandbox cannot pull the Milvus standalone image (registry blob download returns 502
   through the network policy). Plan: local/dev tests run on Milvus Lite (a real Milvus engine, embedded; accepts HNSW
   collection creation — verified), CI runs a Milvus standalone service container. Differences between the two are
   not yet characterised; the standalone path is therefore **unverified from here**. If you want tests against
   standalone from this environment, the environment's network policy needs the Docker registry hosts allowed.
2. **Real OpenAI.** No key was used or read. Evaluation (Step 18) and any real-model integration test need a key supplied
   through the environment; without one, the harness runs with recorded/cached embeddings and the LLM-judged tier is skipped.
3. **`config/.env` is tracked in git.** Its contents were not inspected. If it ever held real credentials they must be
   rotated; removing it from the index is the first implementation commit.
4. Judgment calls to confirm: version reset to `0.1.0`; default pipeline decided by measurement (crew vs direct);
   two simultaneous API keys for rotation; `/v1` prefix and `/healthz`, `/readyz` replacing `/health`.
