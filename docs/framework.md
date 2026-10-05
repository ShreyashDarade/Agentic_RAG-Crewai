# Framework

How this system may be built and may change. Written before the rewrite; the code is made to fit it.
Status: **implemented** (0.1.0). Sections 1-11 state the rules; section 12 lists, for each rule, the check that enforces it, where that check
runs, and the mutation case in [mutation-proofs.md](mutation-proofs.md) that shows it failing. A rule with no tool is labelled **review rule**.
Where the build departs from what was first drafted, the text below says so (for example the extras and extension points that were
planned and not built are listed in section 13).
Evidence for choices: [baseline](baseline.md) (what exists), [research](research.md) (prior art, cited by row),
[ADRs](adr/) (the decisions that make an item "fixed").

## 1. Scope and non-goals

**Product.** A retrieval-augmented question-answering service: ingest documents (text/Markdown/CSV, HTML, PDF, DOCX; no OCR, spreadsheets
or slides), index them (Milvus by default; Chroma and Qdrant are registered too), answer questions with cited sources. Consumers: Python programs that call
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
| foundation | `errors`, `registry`, `blocking` (bounded worker pools) | stdlib | everything else |
| foundation | `contracts` | stdlib, pydantic, `errors` | fastapi, httpx, any adapter library |
| ports | `ports` | foundation, stdlib | any third-party library |
| application | `application` | ports, foundation | fastapi, httpx, pymilvus, openai, crewai, any adapter |
| adapters | `adapters.<name>` | ports, foundation, **its own** third-party library | other adapters, application, transports |
| transports | `api` | application, foundation | ports directly, adapters |
| entry points | `server`, `embedded`, `cli` | container, api, config; `cli` also `server` | — |
| SDK | `sdk`, `client`, `models`, `extend`, `testing` | foundation, stdlib, httpx, pydantic | fastapi, pymilvus, openai, crewai, torch, adapters, application (the embedded backend imports `container` lazily and fails with an `ImportError` naming the `engine` extra) |
| composition root | `container` | everything | — (nothing may import it except `server`, `cli`, and the embedded SDK backend) |

Third-party confinement (each library lives in exactly one package): `pymilvus` → `adapters.milvus`; `openai` →
`adapters.openai`; `crewai` → `adapters.crewai` (`pipeline`, `chat_model`, `embedder`); `chromadb` → `adapters.chroma`; `qdrant_client` → `adapters.qdrant`
(ADR-0012: these two use their own client libraries, not CrewAI's); `fastapi`/`starlette` → `api` (and `server` for the app object);
`uvicorn` → `server`; `httpx` → `sdk`; `fitz` → `adapters.parser_pdf`; `docx` → `adapters.parser_docx`; `bs4` → `adapters.parser_html`;
… one row per library in `tests/architecture/confinement.toml`.

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
5. Ingest is idempotent **by content addressing** (document and chunk ids derive from the bytes; re-uploading is a no-op, re-ingesting replaces).
   An `Idempotency-Key` header for other mutations (same key + different body → 422 `IDEMPOTENCY_KEY_REUSED`; in flight → 409) is **not
   implemented yet** (Step 16); the error classes exist but nothing raises them.
6. Overload: bounded in-flight work; beyond it, immediate `503` with integer `Retry-After` and code `OVERLOADED`
   (research §2 B8). Health and readiness are exempt.
7. There is no collection-drop endpoint and no ingest-from-a-server-path endpoint (the prototype's two most dangerous routes were not
   re-created). Document deletion exists and needs the API key. Unauthenticated mode needs an explicit setting and logs a warning.

## 7. SOLID as rules (each has a check that fails on violation)

| | Rule | Check |
|---|---|---|
| **S** | Parsers parse, chunkers chunk, routes translate, services orchestrate; no module imports across its role (layers in §2). | import-linter layers + forbidden contracts; `exhaustive = true` so a new top-level module must be classified. |
| **O** | Behaviour is added by registering a component in a name→factory registry (entry-point group `agentic_rag.<kind>`, or a module named in config exposing `register(registries)`); never by editing a switch. Unknown name → `UnknownComponent` listing valid names. | `tests/architecture/test_open_closed.py`: a fake plug-in module in the test suite registers a new chunker and a new vector store and the service uses them with **zero edits under `src/`**. |
| **L** | Every implementation of a port passes the same conformance suite *including failure behaviour* (timeout, bad input, unavailable). | `agentic_rag.testing.contracts` abstract suites run against every built-in; a deliberately broken implementation per suite must be rejected (mutation proof). |
| **I** | Ports are small, one capability each (e.g. `VectorWriter`, `VectorSearcher`, `Embedder`, `ChatModel`, `Reranker`, `DocumentParser`, `Chunker`, `AnswerPipeline`, `LexicalIndex`); consumers depend only on what they call. | AST test: a port Protocol has ≤ 6 methods; a service constructor parameter's type is the narrowest port it uses (review rule for "narrowest"). |
| **D** | High-level code depends on ports; **one** composition root chooses concrete classes. | import-linter: ports/application import no framework or adapter; only `container` imports `adapters.*`. |

Extension points (fixed contracts), exactly the registry kinds in `agentic_rag.registry.KINDS`: `vector_store`, `lexical_index`, `embedder`,
`chat_model`, `reranker`, `parser` (per file type), `chunker`, `answer_pipeline`. A queue, a cache and a clock were considered and not
built. There is no capability declaration: a conformance suite skips only what an implementation cannot simulate (for example an
unreachable backend).

## 8. Retry, timeout and idempotency (ADR-0009, research §2 B1–B8)

| Failure | Retry? |
|---|---|
| Never sent (connect refused/DNS) | yes, every call |
| Refused before work: `429`, `503` | yes, every call; honour `Retry-After`, capped (default 60 s) and bounded by remaining deadline |
| Ambiguous: connection reset after send, `408`, `502`, `504` | only idempotent calls (reads, `DELETE`, and mutations carrying an `Idempotency-Key`) |
| Read timeout | **never** auto-retried |
| Non-idempotent call without key (e.g. appending a chat turn) | never re-sent |
| `DELETE` retry that sees `404` | treated as success of an earlier attempt **only if that attempt failed ambiguously** (it may have reached the server); after a never-sent or refused attempt a 404 is a real 404 |

Client total deadline (default 60 s, enforced on the whole attempt, not per socket operation) > server request deadline (default 55 s,
applied to **every** use case) > the crew's own limit (`crewai_max_seconds`, must be lower; start-up enforces it), so the server's typed
`DEADLINE_EXCEEDED` (504) arrives before the client gives up. Every call to a dependency also has its own bound: store timeouts, provider
timeouts, and bounded worker pools that fail fast when a dependency stops answering.
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
   and raise `ImportError` naming the extra. Extras: `engine`, `parsers`, `server`, `crewai`, `chroma`, `qdrant` (a `worker`, `ocr` and `rerank` extra were planned and not built).
4. Safety: percent-encode every path id (and answer ids URL normalisation would rewrite, `""`, `.`, `..`, with `DOCUMENT_NOT_FOUND`
   locally, as the server does); strict response validation but unknown fields ignored; failures are always `RagError` whatever the server
   sends (a hostile status or `Retry-After` cannot raise a raw exception); a closed client raises `UsageError`; a blocking client used
   after `fork()` fails fast; a dropped blocking client stops its thread; conflicting configuration is an error; an upload stream resumed
   from mid-file sends identical bytes on both transports. (There is no SSE: streaming is not provided.)
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
  the process exits before binding a port. Secrets come from the environment (or `*_FILE` mounts, implemented), are `SecretStr`, never
  logged, never in error text. `config/.env` left version control (it is **still in git history**); `.env.example` has no values.
  *Enforced by:* settings tests; secret-redaction tests; `tests/architecture/test_no_secrets_committed.py` (a pattern scan of tracked files;
  not a substitute for a dedicated scanner or for rotating a leaked key).
* Every attacker- or accident-controlled quantity is bounded by a setting with a default: upload bytes, pages, pixels,
  archive members/uncompressed bytes, history length, query length, batch size, `top_k`, concurrency, queue depth, timeouts.
  *Enforced by:* `tests/unit/test_limits.py` (one test per limit that exceeds it and sees its typed rejection), mutations M99-M105. Not tested
  and said so: the crew's `max_iter` (CrewAI enforces it) and `max_concurrent` beyond the refusal test.
* Filters sent to Milvus are built only in `adapters.milvus.filters`: field names are fixed, every value must match the closed alphabet
  `[A-Za-z0-9._-]{1,128}` (no quote, backslash, bracket or space) before it is rendered, and user text never reaches an expression
  string. Templating (`filter_params`) is not the single path because Milvus Lite does not support it (verified). *Enforced by:*
  conformance injection payloads on every store; mutation M22; `pymilvus` confined to its adapter (import-linter).
* Uploads: extension allow-list (no magic-byte sniffing; a parser rejects what it cannot read), no file is written by name, a streamed
  size cap. Retrieved text is escaped and delimited as untrusted data in prompts; the crew's only tool is read-only retrieval with a search
  budget and the same context budget.
* Dependencies audited (`pip-audit --strict` on the lock) per PR and weekly.
* A default (retrieval depth, fusion, rerank, MMR, multi-query, `ef`, crew vs direct) is **backed by a measurement or
  marked "unmeasured"** in `docs/evaluation.md`; a feature that hurts defaults OFF. *Enforced by:* a test that every
  setting is either given an entry in `docs/defaults.toml` with `evidence = "<run file>" | "unmeasured"` or classified operational in the test.
* Observability contract: request id on every request/response/log line/error body; `/healthz` checks no dependency (the image's
  healthcheck uses it); `/readyz` fails honestly when the store is down, the index does not match the embedder, or the lexical index is
  still rebuilding; metrics use bounded labels (route template, not path). Under `agentic-rag serve` stdout is JSON lines only.

## 12. Governance table (rule → check → where it runs)

"Seen fail" cites cases in [mutation-proofs.md](mutation-proofs.md) (regenerate: `python scripts/prove_rules.py`): the rule is broken on
purpose in a copy of the repository, the check must fail on the copy and pass on a clean one. CI jobs, by name: `static` (ruff, mypy,
import-linter, snapshots, griffe once a release tag exists), `tests` (the whole pytest suite on Python 3.11-3.13, minimal and full extras,
against Milvus standalone), `eval-gate`, `packaging`, `security` (pip-audit), `mutation-proofs`. **None of the CI jobs has been run**: this
environment has no GitHub Actions; every command in them was run locally.

| # | Rule | Check | Runs in | Seen fail |
|---|---|---|---|---|
| G1 | Layer directions (§2) | import-linter `layers`, `exhaustive=true` | CI `static` | M01, M02 |
| G2 | Third-party confinement; thin client imports stdlib+httpx+pydantic only | import-linter `forbidden` (`include_external_packages`) + `tests/architecture/test_confinement.py` + the thin-import test + `scripts/check_wheel.py` (installs only the wheel in a clean venv) | CI `static`, `tests`, `packaging` | M03, M04, M05, M29, M30, M98 |
| G3 | Every package classified; every heavy library confined | exhaustive test over the package tree and `confinement.toml` | CI `static` | M06, M07 |
| G4 | Routes only call the service | import-linter + AST size/call check | CI `static` | M08, M09 |
| G5 | Public API unchanged unless intended | `public_api.txt` snapshot (functions, classmethods, properties, fields) + `griffe check --against <tag>` | CI `static` (griffe only once a tag exists; locally exercised against `HEAD` by M11) | M10, M11, M80 |
| G6 | Wire contract unchanged unless intended | `openapi.json` regenerate-and-diff (`oasdiff breaking` is **not** run: not installed here) | CI `static` | M12 |
| G7 | Error codes append-only, unique, round-trip | `error_codes.json` snapshot + catalog tests | CI `tests` | M13, M14 |
| G8 | Strict typing on the public package; `py.typed` shipped | `mypy --strict`; the wheel check asserts `py.typed` is in the wheel | CI `static`, `packaging` | M15 |
| G9 | Every port implementation conforms, and the suites reject broken ones | `agentic_rag.testing.contracts` suites over all built-ins and the fake plug-in; `tests/conformance/test_suites_reject_broken.py` runs every suite against deliberately broken implementations (about 40 mutants) | CI `tests` | M16, M17 + the mutants |
| G10 | Open-closed: add a component with zero core edits | fake plug-in test | CI `tests` | M18 |
| G11 | Transports behave identically | parity suite (one body over embedded and HTTP-through-the-real-app), plus two-transports-one-engine comparison | CI `tests` | M26, M27, M28, M73, M74 |
| G12 | No silent fallbacks; dependency text never reaches a caller | AST test for swallowed broad excepts; fault-injection tests expect typed errors; `tests/conformance/test_error_text.py` drives each adapter's error mapping with secret-bearing errors | CI `tests` | M19, M20, M62, M83, M95-M97 |
| G13 | Limits enforced | `tests/unit/test_limits.py`: one exceed-the-limit test per limit (not covered: the crew's `max_iter`) | CI `tests` | M21, M99-M105 |
| G14 | Defaults evidence-backed; retrieval regression gate | `defaults.toml` test (every setting classified or documented, values match code) + `eval check` on the committed golden set (synthetic; BM25 only) | CI `tests`, `eval-gate` | M39, M94, M106 |
| G15 | Docs match code | docs-vs-code tests: routes, settings and `_FILE` names, extras, registered components, every test a providers row cites exists. **Not checked:** error codes in prose, the contents of the benchmark/evaluation tables beyond `bench_report.py` and the defaults test | CI `tests` | M40, M93 |
| G16 | Dependencies free of known advisories | `pip-audit --strict` | CI `security` (PR; weekly schedule not added) | shown ad hoc, see mutation-proofs.md |
| G17 | Lint/format | ruff check + format | CI `static` | n/a (tool defaults) |
| G18 | Deprecation metadata valid; own warnings are errors | decorator tests; pytest `filterwarnings=error` for the package's own category | CI `tests` | M33 |
| G19 | "Two minors of notice" before removal | **review rule — not machine-enforced** | PR checklist | — |
| G20 | A change to a FIXED item has an ADR | **review rule — not machine-enforced**; mitigated: G5/G6/G7 snapshots fail, forcing the PR to touch a file that the PR template ties to an ADR link | PR template | — |
| G21 | "Narrowest port" for each consumer | **review rule — not machine-enforced** (size of Protocol *is* checked) | PR checklist | — |
| G22 | Security controls behave (authentication before work, filters, error text, key transport, request id, operability) | the API, SDK, conformance and live-process tests named in [security.md](security.md) and [operations.md](operations.md) | CI `tests` | M22-M25, M34-M37, M43-M45, M47, M53-M58, M61, M63-M72, M75-M79, M81-M92, M107-M109 |

Not machine-enforced at all, and said so: G19-G21; that retrieval **defaults** are good (only that each is documented as measured or
unmeasured); that a benchmark number in prose still equals the JSON it came from (the tables are rendered from the JSON, the sentences
around them are not checked); ADR text matching the code.

## 13. Open items for the owner

1. **Real Milvus for tests.** This sandbox cannot pull the Milvus standalone image (registry blob download returns 502 through the
   network policy). Local/dev tests run on Milvus Lite (a real Milvus engine, embedded); CI starts a Milvus standalone service container.
   Differences between the two are not characterised; the standalone path is **unverified from here**, including the concurrent first
   creation of a collection by several replicas.
2. **Real providers.** No OpenAI, Anthropic or other credentials were used or read. Every hosted provider is unverified against the real
   service (`docs/providers.md`); embedding- and LLM-dependent quality (dense retrieval, fusion, chunking, `direct` against `crewai`, answer
   faithfulness) is unmeasured (`docs/defaults.toml`).
3. **`config/.env` was tracked in git.** Its contents were never inspected. It is out of the index but remains in history: rotate anything
   that was ever in it.
4. **CI has not run.** The workflow, the Dockerfile and docker-compose were not executed here; each command in them was run locally.
5. Planned and not built: an LLM-judged evaluation tier; a reranker; OCR; spreadsheet and slide parsers; per-key rate limits; a subprocess
   sandbox with a timeout for parsers; moving lexical search into the store so replicas agree; `oasdiff`; a weekly dependency audit.
6. Judgment calls to confirm: version reset to `0.1.0`; the default pipeline stays `direct` until measured; two simultaneous API keys for
   rotation; `/v1` prefix and `/healthz`, `/readyz` replacing `/health`; the document id covers the file type as well as the bytes.
