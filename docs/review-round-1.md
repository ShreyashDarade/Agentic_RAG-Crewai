# Independent review, round 1

Three reviewers with no stake in the build each read one area (A engine and adapters, B HTTP transport and SDK, C CrewAI, evaluation and
the documents' honesty), reproduced their findings with scripts, and reported. This is every finding and what became of it. "Fixed" means
a test that failed before the change now passes and a mutation case in `mutation-proofs.md` shows the test failing without the fix.
"Documented" means a limit that remains, now stated where a reader will meet it. The loop stopped when a further round was not worth
the cost, not because nothing remained: see "Not done" below.

## Area A: engine and adapters

| # | Finding | Disposition |
|---|---|---|
| A1 | No dependency call had a timeout; a hung store starved the shared executor and so the whole process | **Fixed**: bounded daemon-thread pools per store (`blocking.py`, ADR-0013), Milvus and Qdrant client timeouts, no lock held across remote Chroma/Qdrant calls, one worker for embedded Milvus. **Documented**: remote Chroma has no client timeout (its library has none); only the pool bounds it |
| A2 | The deadline did not bound the work; ingest slots were released while threads ran on; chunker quadratic; DOCX bomb | **Fixed**: a slot is held until the thread returns; the chunker is linear; text that cannot fit the chunk limit is refused before chunking; DOCX parts capped at 8 MiB (32 MiB total). **Documented**: a running parser still cannot be interrupted (HTML about 2.4 s/MB) |
| A3 | A partial ingest left chunks searchable but unlisted and undeletable | **Fixed**: failed new ingests are discarded (best effort); leftovers can be deleted |
| A4 | The commit marker was written before the sweep and the lexical update | **Fixed**: write, sweep, lexical, commit |
| A5 | Embedding model and size checked only on ingest | **Fixed**: checked on search/query and reported by `/readyz` (`index`) |
| A6 | The start-up lexical rebuild could resurrect deleted documents | **Fixed** on one replica. **Documented**: other replicas keep a deleted document in their BM25 index until restart |
| A7 | Conformance suites accepted broken stores and chunkers | **Fixed**: ten new store mutants, two new chunker mutants and a new pipeline mutant are rejected; the stronger suites also found that the chunker dropped an identical final chunk (**fixed**) |
| A8 | `get`, `list`, `delete` had no deadline | **Fixed** |
| A9 | The per-document lock table was never pruned | **Fixed** (reference-counted; unknown ids never reach it) |
| A10 | Embedding batches kept calling the provider after the request failed | **Fixed** |
| A11 | DOCX text order and merged cells | **Fixed** |
| A12 | "Re-ingest replaces" was unreachable | **Fixed**: chunker settings and parser version are part of the index version |
| A13 | The same bytes under another extension returned the earlier parse | **Fixed**: the document id covers the file type |
| A14 | `document_ids=[]` searched everything | **Fixed**: 422 on every transport |
| A15 | `MilvusStore._exists` cached true forever | **Fixed** (reset on any store error) |
| A16 | `error_from_problem` raised raw exceptions on odd bodies | **Fixed** |
| - | Reasoned only: Milvus VARCHAR length may count bytes on standalone; concurrent first creation of a collection | **Not verified** (no standalone Milvus here) |

## Area B: HTTP transport and SDK

| # | Finding | Disposition |
|---|---|---|
| B1 | Work done before authentication; anonymous requests counted by load shedding | **Fixed** (ADR-0014). **Documented**: an authenticated client can hold a slot with a body it never sends |
| B2 | Load shedding silently off under `--root-path` | **Fixed** |
| B3 | No server deadline on get, list, delete | **Fixed** |
| B4 | The documented drain could not happen under uvicorn | **Fixed** under `agentic-rag serve` (tested with a real process and SIGTERM). **Documented**: `uvicorn --factory` does not drain |
| B5 | The client `timeout` was not a total deadline | **Fixed** |
| B6 | Blocking client: hang after fork, thread leak when dropped | **Fixed** |
| B7 | Partial container build leaked; lexical rebuild died silently; use after close | **Fixed** |
| B8 | DELETE retry rule too broad; dot-segment ids differ between transports; raw exceptions from `Retry-After` and error bodies | **Fixed**; the new parity test also found ids containing `/` behaving differently (**fixed** with a path converter) |
| B9 | `AGENTIC_RAG_API_KEY` rejected by the server; `*_FILE` secrets claimed but missing; `use_reranker` without a reranker a silent no-op; `ALLOW_UNAUTHENTICATED` overrode keys; the plain-http check ignored a supplied client's base URL | **Fixed** (all five) |
| B10 | Docs vs behaviour: JSON-only logs, `HEALTHCHECK` on `/readyz`, 413 counted as `VALIDATION_FAILED` | **Fixed** |

## Area C: CrewAI, evaluation, documents

| # | Finding | Disposition |
|---|---|---|
| C1 | Crew threads outlived timeouts, starved the executor, deadlocked with a large `max_concurrent` | **Fixed**: separate pools, outer cancellation stops provider calls, provider timeout, `crewai_max_seconds` must be below the request deadline. **Documented**: CrewAI embedder providers take their timeout from provider-specific options only |
| C2 | CrewAI wrote user data to a plaintext SQLite file | **Fixed** (no-op store). **Documented**: importing `crewai` still creates an empty directory; on a read-only filesystem set `XDG_DATA_HOME` (start-up reports it as a typed error) |
| C3 | The search tool bypassed the context bound | **Fixed** |
| C4 | A failing search let the crew answer from less (silent fallback) | **Fixed** |
| C5 | Hostile `Retry-After` broke error serialisation | **Fixed** (at the source, in `RagError`) |
| C6 | Citations weaker for the crew than for `direct`; second-order injection | **Fixed**: only shown chunks can be cited. **Documented**: the "untrusted" marking does not survive an agent rewriting a chunk |
| C8 | Dataset loader: zero-grade queries, header-less qrels, blank lines | **Fixed**; the SciFact result is unchanged |
| C10 | Public API snapshot missed classmethods; mutation log out of date; CI would fail | **Fixed**: classmethods and properties are snapshotted, the log is regenerated, the CI job no longer diffs it |
| C11 | `evaluation.md` claimed a CI gate that did not exist; framework rows named jobs that do not exist | **Fixed**: a real golden-set gate (tests and a CI job), a real `packaging` job, the table uses the real job names |
| C12 | Framework sections stale in both directions | **Fixed** |
| C13 | Rows no check enforced: limits, secrets through adapters, defaults, error-code docs, providers status | **Fixed** except error codes in prose, which the table now says is not checked |
| C14 | Smaller false statements (README open routes, `.env.example`, undocumented variables, benchmark figures, `grounded`, ADR claims) | **Fixed**; the variables now have a generated reference (`configuration.md`) and the benchmark stage timings come from `bench.py --stages` |
| C7, C9 | Embedder error leakage, per-request leaks, metrics and statistics | **Nothing found** |

## How the proofs were produced
`docs/mutation-proofs.md` has one row per case (110). All rows come from one full run (`scripts/prove_rules.py --jobs 3`) except M89, M92
and M108, whose cases were corrected after that run showed them to be wrong (a wall-clock bound that quadratic code still met; a mutation
in the wrong block; a check that needs a git repository) and were re-run with `--merge`. The run also exposed two weak tests (M53 never saw
the store being reached; M67 and M68 ran the installed checkout instead of the mutated copy) and a mutation that hung instead of failing
(M75); all were fixed before the log was written. An earlier version of this log had 25 rows for 44 cases.

## What the review did not find, and what it could not
* Hosted providers and standalone Milvus were not reachable; those paths were reviewed by reading, not by running.
* No reviewer ran the CI workflow, the Dockerfile or the compose file.
* A re-review of the fixes was **not** done by a fresh reviewer; the fixes were checked by their own tests, the mutation proofs and the
  existing suite only.

## Not done
Subprocess sandbox with a timeout for parsers; a body-read timeout in the app; per-key rate limits; lexical search inside the store;
re-measurement against standalone Milvus (the search-latency cost of the pools, a few percent at one client, was measured and not
explained: `docs/benchmark.md`).
