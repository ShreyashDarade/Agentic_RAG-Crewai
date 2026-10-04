# Mutation proofs

Each rule in the governance table (framework section 12) was broken on purpose in a copy of the repository and its check was run;
a rule counts as *enforced* only if the check passes on the clean copy and fails on the mutated one.
Regenerate with `python scripts/prove_rules.py` (about two minutes). Commands run from the repository root with the project's virtualenv.

| # | Rule | Mutation | Check | Clean copy | Mutated copy | Evidence (first matching output line) |
|---|---|---|---|---|---|---|
| M01 | G1 layers | application imports the HTTP layer | `lint-imports` | pass | FAILS as required | `G1 layers: higher layers may import lower ones, never the reverse (exhaustive) BROKEN` |
| M02 | G1 adapter independence | pipeline_direct imports the milvus adapter | `lint-imports` | pass | FAILS as required | `G1 adapters are independent of each other BROKEN` |
| M03 | G2 domain purity | application imports pymilvus | `lint-imports` | pass | FAILS as required | `G2 domain code imports no framework or adapter library BROKEN` |
| M04 | G2 confinement | the milvus adapter imports openai | `lint-imports` | pass | FAILS as required | `G2 openai is confined to adapters.openai BROKEN (4 ignored imports)` |
| M05 | G2 thin foundation | ports import httpx | `lint-imports` | pass | FAILS as required | `G2 domain code imports no framework or adapter library BROKEN` |
| M06 | G2/G3 unclassified library | a new third-party import without a classification | `python -m pytest -q -x tests/architecture/test_confinement.py` | pass | FAILS as required | `unclassified = sorted({lib for _, lib in _third_party_imports() if lib not in allowed})` |
| M07 | G3 exhaustive classification | a new top-level module nobody classified | `lint-imports` | pass | FAILS as required | `- agentic_rag.extras` |
| M08 | G4 routes via the service | a route module imports a port | `lint-imports` | pass | FAILS as required | `G4 routes reach the system only through the application service BROKEN` |
| M09 | G4 thin routes | a health route grows 20 lines of logic | `python -m pytest -q -x tests/architecture/test_routes_and_errors.py` | pass | FAILS as required | `>           assert body_lines <= MAX_ROUTE_LINES, f"{fn.name} has {body_lines} body lines"` |
| M10 | G5 public API snapshot | a name disappears from contracts.__all__ | `python -m pytest -q -x tests/contract` | pass | FAILS as required | `__________________ test_snapshot_matches_code[public_api.txt] __________________` |
| M11 | G5 API compatibility vs last ref (griffe) | a public keyword parameter is removed | `griffe check agentic_rag -s src --against HEAD -f oneline` | pass | FAILS as required | `src/agentic_rag/errors.py:354: error_from_problem(status): Parameter was removed` |
| M12 | G6 wire contract snapshot | a response field changes type | `python scripts/snapshots.py check` | pass | FAILS as required | `--- openapi.json (checked in)` |
| M13 | G7 error codes append-only | an error code is renamed | `python -m pytest -q -x tests/contract` | pass | FAILS as required | `_________________ test_snapshot_matches_code[error_codes.json] _________________` |
| M14 | G7 error status stable | NotFound changes status | `python -m pytest -q -x tests/contract` | pass | FAILS as required | `_________________ test_snapshot_matches_code[error_codes.json] _________________` |
| M15 | G8 strict typing | an untyped function in the public package | `mypy` | pass | FAILS as required | `src/agentic_rag/registry.py:118: error: Function is missing a type annotation  [no-untyped-def]` |
| M16 | G9 conformance on a real adapter | Milvus delete ignores keep_chunk_ids | `python -m pytest -q -x tests/conformance/test_stores.py -k Milvus and sweeps` | pass | FAILS as required | `FAILED tests/conformance/test_stores.py::TestMilvusStore::test_delete_document_sweeps_all_but_kept` |
| M17 | G9/G12 conformance catches a silent fallback | the OpenAI embedder returns [] on provider errors | `python -m pytest -q -x tests/conformance/test_openai.py` | pass | FAILS as required | `FAILED tests/conformance/test_openai.py::TestOpenAIEmbedder::test_failure_is_typed` |
| M18 | G10 open-closed | the registry ignores registered factories | `python -m pytest -q -x tests/architecture/test_open_closed.py` | pass | FAILS as required | `FAILED tests/architecture/test_open_closed.py::test_a_plugin_adds_store_embedder_model_and_pipeline_without_touching_the_core` |
| M19 | G12 no silent fallbacks (lint) | a parser failure is swallowed into an empty document | `ruff check src` | pass | FAILS as required | `BLE001 Do not catch blind exception: `Exception`` |
| M20 | G12 no silent fallbacks (AST) | a broad except returns a default | `python -m pytest -q -x tests/architecture/test_routes_and_errors.py` | pass | FAILS as required | `>                   assert raises or logs, f"{path.name}:{node.lineno} swallows an exception"` |
| M21 | G13 limits | the upload size check is removed | `python -m pytest -q -x tests/unit/test_service.py` | pass | FAILS as required | `FAILED tests/unit/test_service.py::test_limits_are_enforced_with_typed_rejections` |
| M22 | Security: filter alphabet | id validation in the Milvus filter builder is removed | `python -m pytest -q -x tests/conformance/test_stores.py -k Milvus and hostile` | pass | FAILS as required | `FAILED tests/conformance/test_stores.py::TestMilvusStore::test_hostile_filter_value_never_matches_other_documents` |
| M23 | Security: authentication | the API-key check is disabled | `python -m pytest -q -x tests/api/test_api.py` | pass | FAILS as required | `FAILED tests/api/test_api.py::test_every_v1_route_requires_the_api_key - Asse...` |
| M24 | Security: request-id validation | any inbound X-Request-ID is trusted | `python -m pytest -q -x tests/api/test_api.py` | pass | FAILS as required | `FAILED tests/api/test_api.py::test_request_id_on_every_response_and_inbound_ids_are_validated` |
| M25 | Security: error text leak | an unexpected exception message is returned to the client | `python -m pytest -q -x tests/api/test_api.py` | pass | FAILS as required | `FAILED tests/api/test_api.py::test_unexpected_exceptions_become_a_generic_500_with_a_request_id` |

## Checked outside the automated harness

| Rule | Check | Evidence |
|---|---|---|
| G16 dependency advisories | `uv export --frozen --no-hashes --no-dev --extra engine --extra server --extra parsers --no-emit-project \| pip-audit -r - --no-deps --disable-pip --strict` | Clean on the locked set (155 pins): `No known vulnerabilities found`. Failure shown on a deliberately vulnerable pin: `pip-audit -r bad.txt` with `urllib3==1.25.0` exits 1 with `Found 11 known vulnerabilities in 1 package`. The audit queries the network, so it is a CI job, not part of this offline harness. |
| G17 lint/format | `ruff check`, `ruff format --check` | Tool defaults; not mutation-tested. |
| G11 parity, G14 evaluation gate, G15 docs-vs-code, G18 deprecation | not implemented yet | Steps 13, 18, 20 and 14 respectively. |

## What the proofs found while they were being written

* The conformance check for the lexical index's filter could not tell a filter that is ignored from one that is applied (the unfiltered best hit
  happened to be the filtered one). Case-writing exposed it; the check now asserts its own premise.
* The request-id test used a CRLF payload that the HTTP client rejects before the server sees it, so trusting any inbound id passed (M24 first
  reported `NOT CAUGHT`). The test now uses ids with spaces, markup, excess length and path characters.
* The confinement analysis found `agentic_rag.container` importing `openai` directly; the client construction moved into `adapters.openai`.
