#!/usr/bin/env python3
"""Mutation proof for the governance table (framework section 12).

For each rule: copy the repository, break the rule on purpose, run the rule's check, and require that it FAILS
(while the same check PASSES on an unmutated copy). The result is written to docs/mutation-proofs.md.

    scripts/prove_rules.py            run every case and rewrite docs/mutation-proofs.md
    scripts/prove_rules.py --jobs 3   the same, three cases at a time (each in its own copy)
    scripts/prove_rules.py M03 M10    run only these cases (does not rewrite the log)
    scripts/prove_rules.py --merge M03 M10   the same, then replace just those rows in docs/mutation-proofs.md
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv" / "bin"
IGNORE = shutil.ignore_patterns(
    ".git",
    ".venv",
    ".dev",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".coverage",
    "htmlcov",
)

LINT = [str(VENV / "lint-imports")]
PYTEST = [str(VENV / "python"), "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"]
MYPY = [str(VENV / "mypy")]
RUFF = [str(VENV / "ruff"), "check", "src"]
SNAP = [str(VENV / "python"), "scripts/snapshots.py", "check"]
GRIFFE = [str(VENV / "griffe"), "check", "agentic_rag", "-s", "src", "--against", "HEAD", "-f", "oneline"]


@dataclass
class Edit:
    path: str
    old: str | None = None  # None with new => append; "" => create file
    new: str = ""


@dataclass
class Case:
    id: str
    rule: str
    what: str
    edits: list[Edit]
    command: list[str]
    expect: str
    needs_git: bool = False
    milvus: bool = False
    result: dict[str, str] = field(default_factory=dict)


def E(path: str, old: str | None, new: str = "") -> Edit:
    return Edit(path, old, new)


S = "src/agentic_rag/"
CASES: list[Case] = [
    Case(
        "M01",
        "G1 layers",
        "application imports the HTTP layer",
        [E(S + "application/limits.py", None, "\nfrom agentic_rag.api import ApiConfig  # noqa\n")],
        LINT,
        "G1 layers",
    ),
    Case(
        "M02",
        "G1 adapter independence",
        "pipeline_direct imports the milvus adapter",
        [
            E(
                S + "adapters/pipeline_direct/__init__.py",
                None,
                "\nfrom agentic_rag.adapters.milvus import MilvusStore  # noqa\n",
            )
        ],
        LINT,
        "G1 adapters are independent",
    ),
    Case(
        "M03",
        "G2 domain purity",
        "application imports pymilvus",
        [E(S + "application/service.py", None, "\nimport pymilvus  # noqa\n")],
        LINT,
        "G2 domain code",
    ),
    Case(
        "M04",
        "G2 confinement",
        "the milvus adapter imports openai",
        [E(S + "adapters/milvus/filters.py", None, "\nimport openai  # noqa\n")],
        LINT,
        "G2 openai is confined",
    ),
    Case(
        "M05",
        "G2 thin foundation",
        "ports import httpx",
        [E(S + "ports/types.py", None, "\nimport httpx  # noqa\n")],
        LINT,
        "G2 domain code",
    ),
    Case(
        "M06",
        "G2/G3 unclassified library",
        "a new third-party import without a classification",
        [E(S + "adapters/lexical_bm25/__init__.py", None, "\nimport numpy  # noqa\n")],
        PYTEST + ["tests/architecture/test_confinement.py"],
        "unclassified",
    ),
    Case(
        "M07",
        "G3 exhaustive classification",
        "a new top-level module nobody classified",
        [E(S + "extras.py", "", '"""unclassified"""\n')],
        LINT,
        "extras",
    ),
    Case(
        "M08",
        "G4 routes via the service",
        "a route module imports a port",
        [E(S + "api/routes.py", None, "\nfrom agentic_rag.ports import Embedder  # noqa\n")],
        LINT,
        "G4 routes",
    ),
    Case(
        "M09",
        "G4 thin routes",
        "a health route grows 20 lines of logic",
        [
            E(
                S + "api/routes.py",
                '    return {"status": "ok"}',
                "\n".join(f"    value_{i} = {i}" for i in range(20)) + '\n    return {"status": "ok"}',
            )
        ],
        PYTEST + ["tests/architecture/test_routes_and_errors.py"],
        "body lines",
    ),
    Case(
        "M10",
        "G5 public API snapshot",
        "a name disappears from contracts.__all__",
        [E(S + "contracts/__init__.py", '    "DeleteResult",\n', "")],
        PYTEST + ["tests/contract"],
        "public_api.txt",
    ),
    Case(
        "M11",
        "G5 API compatibility vs last ref (griffe)",
        "a public keyword parameter is removed",
        [
            E(
                S + "errors.py",
                "def error_from_problem(body: Mapping[str, Any], *, status: int | None = None)",
                "def error_from_problem(body: Mapping[str, Any])",
            ),
            E(
                S + "errors.py",
                'http_status = _status_of(body.get("status"), fallback=status or 0)',
                'http_status = _status_of(body.get("status"), fallback=0)',
            ),
        ],
        GRIFFE,
        "error_from_problem",
        needs_git=True,
    ),
    Case(
        "M12",
        "G6 wire contract snapshot",
        "a response field changes type",
        [
            E(
                S + "contracts/models.py",
                "    text: str\n    score: float\n    page: int | None = None\n\n\nclass SearchResponse",
                "    text: str\n    score: str\n    page: int | None = None\n\n\nclass SearchResponse",
            )
        ],
        SNAP,
        "openapi.json",
    ),
    Case(
        "M13",
        "G7 error codes append-only",
        "an error code is renamed",
        [E(S + "errors.py", 'code = "DOCUMENT_NOT_FOUND"', 'code = "DOC_NOT_FOUND"')],
        PYTEST + ["tests/contract"],
        "error_codes.json",
    ),
    Case(
        "M14",
        "G7 error status stable",
        "NotFound changes status",
        [
            E(
                S + "errors.py",
                '    code = "NOT_FOUND"\n    http_status = 404',
                '    code = "NOT_FOUND"\n    http_status = 410',
            )
        ],
        PYTEST + ["tests/contract"],
        "error_codes.json",
    ),
    Case(
        "M15",
        "G8 strict typing",
        "an untyped function in the public package",
        [E(S + "registry.py", None, "\n\ndef helper(x):\n    return x\n")],
        MYPY,
        "no-untyped-def",
    ),
    Case(
        "M16",
        "G9 conformance on a real adapter",
        "Milvus delete ignores keep_chunk_ids",
        [
            E(
                S + "adapters/milvus/store.py",
                '            expr += f" and id not in {quote_ids(sorted(keep_chunk_ids))}"',
                "            pass",
            )
        ],
        PYTEST + ["tests/conformance/test_stores.py", "-k", "Milvus and sweeps"],
        "FAILED",
        milvus=True,
    ),
    Case(
        "M17",
        "G9/G12 conformance catches a silent fallback",
        "the OpenAI embedder returns [] on provider errors",
        [
            E(
                S + "adapters/openai/embedder.py",
                "            raise map_error(exc, EmbeddingFailed) from exc",
                "            return []",
            )
        ],
        PYTEST + ["tests/conformance/test_openai.py"],
        "FAILED",
    ),
    Case(
        "M18",
        "G10 open-closed",
        "the registry ignores registered factories",
        [
            E(
                S + "registry.py",
                "        factory = self._factories.get(name)\n        if factory is None:\n            factory = self._load_entry_point(name)",
                "        factory = None\n        if factory is None:\n            factory = self._load_entry_point(name)",
            )
        ],
        PYTEST + ["tests/architecture/test_open_closed.py"],
        "FAILED",
    ),
    Case(
        "M19",
        "G12 no silent fallbacks (lint)",
        "a parser failure is swallowed into an empty document",
        [
            E(
                S + "application/service.py",
                "            raise DocumentParseFailed() from exc",
                "            return ParsedDocument(text='')",
            )
        ],
        RUFF,
        "BLE001",
    ),
    Case(
        "M20",
        "G12 no silent fallbacks (AST)",
        "a broad except returns a default",
        [
            E(
                S + "application/service.py",
                "            raise DocumentParseFailed() from exc",
                "            return ParsedDocument(text='')",
            )
        ],
        PYTEST + ["tests/architecture/test_routes_and_errors.py"],
        "swallows an exception",
    ),
    Case(
        "M21",
        "G13 limits",
        "the upload size check is removed",
        [E(S + "application/service.py", "if len(data) > self._limits.max_upload_bytes:", "if False:")],
        PYTEST + ["tests/unit/test_service.py"],
        "FAILED",
    ),
    Case(
        "M22",
        "Security: filter alphabet",
        "id validation in the Milvus filter builder is removed",
        [E(S + "adapters/milvus/filters.py", "        if not _ID.fullmatch(value):", "        if False:")],
        PYTEST + ["tests/conformance/test_stores.py", "-k", "Milvus and hostile"],
        "FAILED",
        milvus=True,
    ),
    Case(
        "M23",
        "Security: authentication",
        "the API-key check is disabled",
        [E(S + "api/auth.py", "    return any(matches)", "    return True")],
        PYTEST + ["tests/api/test_api.py"],
        "FAILED",
    ),
    Case(
        "M24",
        "Security: request-id validation",
        "any inbound X-Request-ID is trusted",
        [
            E(
                S + "api/middleware.py",
                "request_id = inbound if inbound and _VALID_ID.fullmatch(inbound) else uuid.uuid4().hex",
                "request_id = inbound or uuid.uuid4().hex",
            )
        ],
        PYTEST + ["tests/api/test_api.py"],
        "FAILED",
    ),
    Case(
        "M25",
        "Security: error text leak",
        "an unexpected exception message is returned to the client",
        [
            E(
                S + "api/middleware.py",
                "_send_problem(send_with_id, RagError(request_id=request_id))",
                "_send_problem(send_with_id, RagError(str(sys.exc_info()[1]), request_id=request_id))",
            ),
            E(S + "api/middleware.py", "import re\n", "import re\nimport sys\n"),
        ],
        PYTEST + ["tests/api/test_api.py"],
        "FAILED",
    ),
    Case(
        "M26",
        "G11 parity (routes)",
        "the HTTP backend lists documents on the wrong path",
        [
            E(
                S + "sdk/_http.py",
                '"GET", "/v1/documents", idempotent=True, params=params',
                '"GET", "/v1/docs", idempotent=True, params=params',
            )
        ],
        PYTEST + ["tests/sdk/test_parity.py"],
        "FAILED",
    ),
    Case(
        "M27",
        "G11 parity (errors)",
        "the HTTP backend stops rebuilding typed errors from problem bodies",
        [
            E(
                S + "sdk/_http.py",
                "        error = error_from_problem(body, status=response.status_code)\n",
                "        error = RagStatusError(server_code=str(body['code']), status=response.status_code)\n",
            )
        ],
        PYTEST + ["tests/sdk/test_parity.py"],
        "FAILED",
    ),
    Case(
        "M28",
        "G11 parity (readiness)",
        "the embedded backend always reports ready",
        [
            E(
                S + "embedded.py",
                "return _same_types_as_http(ReadyResponse, await (await self._svc()).ready())",
                "return ReadyResponse(ready=True, checks={})",
            )
        ],
        PYTEST + ["tests/sdk/test_parity.py"],
        "FAILED",
    ),
    Case(
        "M29",
        "G2 thin client",
        "the retry module imports openai",
        [E(S + "sdk/_retry.py", None, "\nimport openai  # noqa\n")],
        LINT,
        "thin client",
    ),
    Case(
        "M30",
        "G2 thin client (runtime proof)",
        "the client module imports pymilvus",
        [E(S + "client.py", None, "\nimport pymilvus  # noqa\n")],
        PYTEST + ["tests/sdk/test_client.py", "-k", "thin"],
        "FAILED",
    ),
    Case(
        "M31",
        "Retry matrix",
        "ambiguous failures are retried for non-idempotent calls",
        [
            E(
                S + "sdk/_http.py",
                "failure, retry_ok = ConnectionFailed(request_id=request_id), idempotent",
                "failure, retry_ok = ConnectionFailed(request_id=request_id), True",
            )
        ],
        PYTEST + ["tests/sdk/test_retries.py"],
        "FAILED",
    ),
    Case(
        "M32",
        "Retry matrix",
        "a read timeout is retried",
        [
            E(
                S + "sdk/_http.py",
                "raise ClientTimeout(request_id=request_id) from exc  # never retried: the work may be running",
                "failure, retry_ok = ClientTimeout(request_id=request_id), True",
            )
        ],
        PYTEST + ["tests/sdk/test_retries.py"],
        "FAILED",
    ),
    Case(
        "M33",
        "G18 deprecation policy",
        "removal in the same major is allowed",
        [E(S + "_compat.py", "if remove_v[0] <= since_v[0]:", "if False:")],
        PYTEST + ["tests/unit/test_compat.py"],
        "FAILED",
    ),
    Case(
        "M34",
        "Security: key over plain http",
        "the client sends the API key over http to any host",
        [
            E(
                S + "client.py",
                'if key and parts.scheme == "http" and parts.hostname not in _LOCAL_HOSTS and not allow_insecure:',
                "if False:",
            )
        ],
        PYTEST + ["tests/sdk/test_client.py"],
        "FAILED",
    ),
    Case(
        "M35",
        "Operability: load shedding",
        "the in-flight bound is ignored",
        [E(S + "api/middleware.py", "if self._inflight >= self.max_inflight:", "if False:")],
        PYTEST + ["tests/api/test_operability.py"],
        "FAILED",
    ),
    Case(
        "M36",
        "Operability: one truth",
        "/metrics stops re-evaluating readiness",
        [
            E(
                S + "api/routes.py",
                "    await check_readiness(request, service)\n    return Response(request.app.state.metrics.render()",
                "    return Response(request.app.state.metrics.render()",
            )
        ],
        PYTEST + ["tests/api/test_operability.py"],
        "FAILED",
    ),
    Case(
        "M37",
        "Operability: bounded metric labels",
        "the raw path becomes a metric label",
        [
            E(
                S + "api/middleware.py",
                'route = getattr(matched, "path_format", None) or getattr(matched, "path", None) or "unmatched"',
                'route = scope["path"]',
            )
        ],
        PYTEST + ["tests/api/test_operability.py"],
        "FAILED",
    ),
    Case(
        "M38",
        "Evaluation metrics",
        "the nDCG discount is off by one rank",
        [
            E(
                S + "evaluation/metrics.py",
                "dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))",
                "dcg = sum(g / math.log2(i + 1.5) for i, g in enumerate(gains))",
            )
        ],
        PYTEST + ["tests/unit/test_evaluation.py"],
        "FAILED",
    ),
    Case(
        "M39",
        "G14 defaults need evidence",
        "a quality default changes without updating docs/defaults.toml",
        [E(S + "config.py", "default_top_k: int = Field(default=8,", "default_top_k: int = Field(default=9,")],
        PYTEST + ["tests/docs/test_docs_match_code.py"],
        "FAILED",
    ),
    Case(
        "M40",
        "G15 docs match code",
        "a document names an environment variable that does not exist",
        [E("docs/operations.md", None, "\nSet AGENTIC_RAG_NONEXISTENT_KNOB to fix it.\n")],
        PYTEST + ["tests/docs/test_docs_match_code.py"],
        "FAILED",
    ),
    Case(
        "M41",
        "Retries at one layer (crew)",
        "the crew LLM stops failing fast after the first provider error",
        [
            E(
                S + "adapters/crewai/_llm.py",
                "        if self._error is not None:\n            raise self._error\n        raw =",
                "        raw =",
            )
        ],
        PYTEST + ["tests/unit/test_crew_pipeline.py"],
        "FAILED",
    ),
    Case(
        "M42",
        "Retries at one layer (CrewAI chat)",
        "the CrewAI chat adapter keeps the provider SDK's default retries",
        [E(S + "adapters/crewai/chat.py", '                "max_retries": 1,\n', "")],
        PYTEST + ["tests/conformance/test_providers_over_http.py", "-k", "at_most_twice"],
        "FAILED",
    ),
    Case(
        "M43",
        "Security: Chroma filter",
        "the Chroma adapter ignores the document filter",
        [
            E(
                S + "adapters/chroma/__init__.py",
                '    return {"document_id": {"$in": list(flt.document_ids)}}',
                "    return None",
            )
        ],
        PYTEST + ["tests/conformance/test_stores.py", "-k", "Chroma and filter"],
        "FAILED",
    ),
    Case(
        "M44",
        "Security: Qdrant filter",
        "the Qdrant adapter ignores the document filter",
        [
            E(
                S + "adapters/qdrant/__init__.py",
                '    return models.Filter(must=[_match("document_id", list(flt.document_ids))])',
                "    return None",
            )
        ],
        PYTEST + ["tests/conformance/test_stores.py", "-k", "Qdrant and filter"],
        "FAILED",
    ),
    # -- review round 1 ---------------------------------------------------------------------------------------------
    Case(
        "M45",
        "Security: authenticate before any work",
        "the early authentication middleware is removed",
        [E(S + "api/app.py", "    app.add_middleware(AuthMiddleware, config=cfg, metrics=metrics)\n", "")],
        PYTEST + ["tests/api/test_hardening.py", "-k", "before_the_body or 401_not_422 or in_flight_slots"],
        "FAILED",
    ),
    Case(
        "M46",
        "Load shedding under --root-path",
        "the route path keeps the root-path prefix",
        [E(S + "api/middleware.py", '    if root and (path == root or path.startswith(root + "/")):', "    if False:")],
        PYTEST + ["tests/api/test_hardening.py", "-k", "root_path"],
        "FAILED",
    ),
    Case(
        "M47",
        "Every use case has a deadline",
        "get_document runs without the request deadline",
        [
            E(
                S + "application/service.py",
                "        async with self._deadline():\n            record = await self._catalog.get_document(document_id)\n        if record is None:",
                "        record = await self._catalog.get_document(document_id)\n        if record is None:",
            )
        ],
        PYTEST + ["tests/unit/test_service_hardening.py", "-k", "obey_the_request_deadline"],
        "FAILED",
    ),
    Case(
        "M48",
        "Ingest commits last (ADR-0007)",
        "the commit marker is written together with the body, before the sweep",
        [
            E(
                S + "application/service.py",
                "body = [i for i, chunk in enumerate(chunks) if chunk.index != 0]",
                "body = list(range(len(chunks)))",
            )
        ],
        PYTEST + ["tests/unit/test_service_hardening.py", "-k", "commit_marker_is_written_last"],
        "FAILED",
    ),
    Case(
        "M49",
        "Ingest leaves nothing half-done",
        "a failed new ingest keeps the chunks it wrote",
        [
            E(
                S + "application/service.py",
                "                if existing is None:\n                    await self._discard_uncommitted(document_id)",
                "                if False:\n                    pass",
            )
        ],
        PYTEST + ["tests/unit/test_service_hardening.py", "-k", "failed_commit_leaves_nothing"],
        "FAILED",
    ),
    Case(
        "M50",
        "Orphans are deletable",
        "delete refuses chunks that have no commit marker",
        [
            E(
                S + "application/service.py",
                "            if record is None and deleted == 0:",
                "            if record is None:",
            )
        ],
        PYTEST + ["tests/unit/test_service_hardening.py", "-k", "without_a_catalog_entry"],
        "FAILED",
    ),
    Case(
        "M51",
        "Re-upload replaces",
        "a re-upload after a chunking change is echoed instead of re-indexed",
        [E(S + "application/service.py", " and existing.index_version == index_version:", ":")],
        PYTEST + ["tests/unit/test_service_hardening.py", "-k", "chunking_recipe"],
        "FAILED",
    ),
    Case(
        "M52",
        "Identity covers what the parser sees",
        "the document id ignores the file type",
        [E(S + "application/service.py", 'hashlib.sha256(extension.encode() + b"\\0" + data)', "hashlib.sha256(data)")],
        PYTEST + ["tests/unit/test_service_hardening.py", "-k", "another_file_type"],
        "FAILED",
    ),
    Case(
        "M53",
        "Hostile ids cost nothing",
        "ids that cannot exist reach the store and the lock table",
        [E(S + "application/service.py", "    if not _DOCUMENT_ID.fullmatch(document_id):", "    if False:")],
        PYTEST + ["tests/unit/test_service_hardening.py", "-k", "cannot_exist"],
        "FAILED",
    ),
    Case(
        "M54",
        "Bounded in-process state",
        "the per-document lock table is never pruned",
        [
            E(
                S + "application/service.py",
                "            if entry.users == 0:\n                del self._doc_locks[document_id]",
                "            if False:\n                pass",
            )
        ],
        PYTEST + ["tests/unit/test_service_hardening.py", "-k", "lock_table"],
        "FAILED",
    ),
    Case(
        "M55",
        "The index must match the embedder on reads",
        "search and query skip the index compatibility check",
        [
            E(
                S + "application/service.py",
                "        if self._index_checked and not force:\n            return\n",
                "        return\n",
            )
        ],
        PYTEST + ["tests/unit/test_service_hardening.py", "-k", "same_size_is_refused"],
        "FAILED",
    ),
    Case(
        "M56",
        "One failure stops the rest",
        "a failed embedding batch lets the other batches finish",
        [E(S + "application/service.py", "return_when=asyncio.FIRST_EXCEPTION", "return_when=asyncio.ALL_COMPLETED")],
        PYTEST + ["tests/unit/test_service_hardening.py", "-k", "cancels_the_others"],
        "FAILED",
    ),
    Case(
        "M57",
        "Bounded blocking work",
        "ingest parsing runs on the shared default executor",
        [
            E(
                S + "application/service.py",
                "chunks = await self._cpu.run(self._parse_and_chunk, parser, data, safe_name, document_id)",
                "chunks = await asyncio.to_thread(self._parse_and_chunk, parser, data, safe_name, document_id)",
            )
        ],
        PYTEST + ["tests/unit/test_service_hardening.py", "-k", "parser_threads"],
        "FAILED",
    ),
    Case(
        "M58",
        "Bounded blocking work (pool)",
        "a pool frees its slot when the awaiting task is cancelled, not when the thread returns",
        [
            E(S + "blocking.py", "        future.add_done_callback(self._release)\n", ""),
            E(
                S + "blocking.py",
                "        return await asyncio.wrap_future(future)",
                "        try:\n            return await asyncio.wrap_future(future)\n        finally:\n            self._release(future)",
            ),
        ],
        PYTEST + ["tests/unit/test_blocking.py"],
        "FAILED",
    ),
    Case(
        "M59",
        "Limits are checked before the work",
        "an oversized text is chunked before the chunk limit is applied",
        [
            E(
                S + "application/service.py",
                "        if non_blank > self._limits.max_chunks_per_document * self._chunker.max_chars:",
                "        if False:",
            )
        ],
        PYTEST + ["tests/unit/test_service_hardening.py", "-k", "refused_before_chunking"],
        "FAILED",
    ),
    Case(
        "M60",
        "Fail closed on an empty filter",
        "an empty document filter means search everything",
        [
            E(
                S + "contracts/models.py",
                "Field(default=None, min_length=1, max_length=MAX_IDS_PER_FILTER)",
                "Field(default=None, max_length=MAX_IDS_PER_FILTER)",
            )
        ],
        PYTEST + ["tests/unit/test_service_hardening.py", "-k", "empty_document_filter"],
        "FAILED",
    ),
    Case(
        "M61",
        "A rebuild cannot undo a delete",
        "the lexical rebuild adds documents the service touched meanwhile",
        [
            E(
                S + "container.py",
                "            fresh = [c for c in chunks if c.document_id not in touched]",
                "            fresh = list(chunks)",
            )
        ],
        PYTEST + ["tests/unit/test_container_hardening.py", "-k", "during_the_scan"],
        "FAILED",
    ),
    Case(
        "M62",
        "No silent death of a background task",
        "the lexical rebuild gives up on any error that is not a RagError",
        [
            E(
                S + "container.py",
                "            except Exception:  # any failure, not only RagError",
                "            except ConfigurationError:  # any failure, not only RagError",
            )
        ],
        PYTEST + ["tests/unit/test_container_hardening.py", "-k", "survives_any_exception"],
        "FAILED",
    ),
    Case(
        "M63",
        "A failed start-up leaks nothing",
        "build_container does not close what it opened before a failure",
        [
            E(
                S + "container.py",
                '        for closer in reversed(ctx.closers):\n            try:\n                await closer()\n            except Exception:\n                logger.exception("error while closing a resource after a failed start-up")',
                "        for closer in []:\n            pass",
            )
        ],
        PYTEST + ["tests/unit/test_container_hardening.py", "-k", "failed_build or reranker_setting"],
        "FAILED",
    ),
    Case(
        "M64",
        "No silent no-op configuration",
        "use_reranker and reranker may disagree",
        [E(S + "container.py", "    if settings.use_reranker != (reranker is not None):", "    if False:")],
        PYTEST + ["tests/unit/test_container_hardening.py", "-k", "reranker_setting"],
        "FAILED",
    ),
    Case(
        "M65",
        "Secrets never given twice",
        "a secret may be set directly and by file",
        [
            E(
                S + "config.py",
                '        if field in values:\n            raise ConfigurationError(\n                f"set {ENV_PREFIX + field.upper()} or {variable}, not both"',
                '        if False:\n            raise ConfigurationError(\n                f"set {ENV_PREFIX + field.upper()} or {variable}, not both"',
            )
        ],
        PYTEST + ["tests/unit/test_container_hardening.py", "-k", "twice_or_from_a_missing"],
        "FAILED",
    ),
    Case(
        "M66",
        "A leftover dev flag cannot open a protected server",
        "unauthenticated mode is allowed together with API keys",
        [E(S + "config.py", "    if settings.allow_unauthenticated and settings.api_keys:", "    if False:")],
        PYTEST + ["tests/unit/test_container_hardening.py", "-k", "cannot_be_combined"],
        "FAILED",
    ),
    Case(
        "M67",
        "The server drains before it stops listening",
        "SIGTERM does not flip readiness to draining",
        [E(S + "server.py", "        self._app.state.draining = True\n        logger.info(", "        logger.info(")],
        PYTEST + ["tests/api/test_live_server.py", "-k", "sigterm"],
        "FAILED",
    ),
    Case(
        "M68",
        "Operational output is JSON only",
        "uvicorn installs its own plain-text log handlers again",
        [
            E(
                S + "server.py",
                "        log_config=None,  # uvicorn must not install its own plain-text handlers: ours print JSON\n",
                "",
            )
        ],
        PYTEST + ["tests/api/test_live_server.py", "-k", "json"],
        "FAILED",
    ),
    Case(
        "M69",
        "A hung store costs one typed error",
        "the Qdrant client has no timeout",
        [E(S + "adapters/qdrant/__init__.py", "timeout=max(1, round(self._s.timeout_seconds))", "timeout=None")],
        PYTEST + ["tests/integration/test_hung_store.py", "-k", "qdrant"],
        "FAILED",
    ),
    Case(
        "M70",
        "SDK timeout is a total deadline",
        "the client timeout is applied per socket operation only",
        [E(S + "sdk/_http.py", "async with asyncio.timeout(remaining):", "async with asyncio.timeout(None):")],
        PYTEST + ["tests/sdk/test_hardening.py", "-k", "drips"],
        "FAILED",
    ),
    Case(
        "M71",
        "Retried DELETE semantics",
        "a 404 after a never-sent attempt counts as a successful delete",
        [
            E(
                S + "sdk/_http.py",
                "may_have_reached_server and isinstance(failure, DocumentNotFound)",
                "attempt > 0 and isinstance(failure, DocumentNotFound)",
            )
        ],
        PYTEST + ["tests/sdk/test_hardening.py", "-k", "reached_the_server"],
        "FAILED",
    ),
    Case(
        "M72",
        "Failures are RagError",
        "error_from_problem trusts the server's status field",
        [
            E(
                S + "errors.py",
                'http_status = _status_of(body.get("status"), fallback=status or 0)',
                'http_status = int(body.get("status", status or 0))',
            )
        ],
        PYTEST + ["tests/sdk/test_hardening.py", "-k", "error_from_problem_is_total"],
        "FAILED",
    ),
    Case(
        "M73",
        "Same ids on every transport",
        "the HTTP backend sends ids that URL normalisation rewrites",
        [E(S + "sdk/_http.py", '    if document_id in {"", ".", ".."}:', "    if False:")],
        PYTEST + ["tests/sdk/test_hardening.py", "-k", "urls_would_rewrite"],
        "FAILED",
    ),
    Case(
        "M74",
        "Parity: ids with slashes",
        "the routes do not accept slashes in a document id",
        [
            E(
                S + "api/routes.py",
                '"/documents/{document_id:path}", operation_id="get_document"',
                '"/documents/{document_id}", operation_id="get_document"',
            )
        ],
        PYTEST + ["tests/sdk/test_parity.py", "-k", "cannot_exist"],
        "FAILED",
    ),
    Case(
        "M75",
        "No leaked threads",
        "a dropped blocking client keeps its thread",
        [
            E(
                S + "sdk/_sync.py",
                "weakref.finalize(self, _stop, self._loop)",
                "weakref.finalize(self, lambda loop: None, self._loop)",
            )
        ],
        PYTEST + ["tests/sdk/test_hardening.py", "-k", "leak_threads"],
        "FAILED",
    ),
    Case(
        "M76",
        "No hang after fork",
        "a client used in a forked child waits for a thread that does not exist",
        [
            E(
                S + "sdk/_sync.py",
                "        if os.getpid() != self._pid:\n            coro.close()",
                "        if False:\n            coro.close()",
            )
        ],
        PYTEST + ["tests/sdk/test_hardening.py", "-k", "forked_child"],
        "FAILED",
    ),
    Case(
        "M77",
        "Closed means closed (HTTP)",
        "a closed AsyncClient keeps sending",
        [
            E(
                S + "sdk/_http.py",
                '        if self._closed:\n            raise UsageError("the client is closed")\n        policy = self._retry',
                "        policy = self._retry",
            )
        ],
        PYTEST + ["tests/sdk/test_hardening.py", "-k", "async_http_client_that_is_closed"],
        "FAILED",
    ),
    Case(
        "M78",
        "Closed means closed (embedded)",
        "a closed embedded client keeps using the engine",
        [
            E(
                S + "embedded.py",
                '        if self._closed:\n            raise UsageError("the client is closed")\n        if self._service is not None:',
                "        if self._service is not None:",
            )
        ],
        PYTEST + ["tests/sdk/test_hardening.py", "-k", "embedded_client_that_is_closed"],
        "FAILED",
    ),
    Case(
        "M79",
        "No key over plain http",
        "the plain-http check ignores a supplied client's real base URL",
        [
            E(
                S + "client.py",
                "parts = urlsplit(str(http_client.base_url) if http_client is not None else base_url)",
                "parts = urlsplit(base_url)",
            )
        ],
        PYTEST + ["tests/sdk/test_hardening.py", "-k", "judged_by_the_url"],
        "FAILED",
    ),
    Case(
        "M80",
        "Public API snapshot covers classmethods",
        "Client.http is renamed",
        [
            E(
                S + "client.py",
                "    def http(\n        cls,\n        base_url: str,\n        *,",
                "    def connect(\n        cls,\n        base_url: str,\n        *,",
            )
        ],
        SNAP,
        "public_api.txt",
    ),
    Case(
        "M81",
        "CrewAI keeps no task outputs",
        "the crew stores its task outputs in CrewAI's SQLite file again",
        [E(S + "adapters/crewai/pipeline.py", "        crew = _StatelessCrew(", "        crew = Crew(")],
        PYTEST + ["tests/unit/test_crew_hardening.py", "-k", "writes_no_files"],
        "FAILED",
    ),
    Case(
        "M82",
        "CrewAI prompts stay bounded",
        "the search tool returns every hit regardless of the context budget",
        [
            E(
                S + "adapters/crewai/pipeline.py",
                "        shown = _within_budget(hits, self.max_context_chars)\n        for hit in shown:",
                "        shown = hits\n        for hit in shown:",
            )
        ],
        PYTEST + ["tests/unit/test_crew_hardening.py", "-k", "context_budget or only_chunks_the_model"],
        "FAILED",
    ),
    Case(
        "M83",
        "CrewAI has no silent fallback",
        "a failed search is not recorded as the request's failure",
        [
            E(
                S + "adapters/crewai/pipeline.py",
                "        self.llm.cancel(error)\n        return error",
                "        return error",
            )
        ],
        PYTEST + ["tests/unit/test_crew_hardening.py", "-k", "store_failure_inside_the_tool"],
        "FAILED",
    ),
    Case(
        "M84",
        "A deadline stops the crew",
        "an outer cancellation does not stop the crew's provider calls",
        [
            E(
                S + "adapters/crewai/pipeline.py",
                '            llm.cancel(DeadlineExceeded("the request was cancelled"))  # the thread cannot be killed, only starved\n            raise',
                "            raise",
            )
        ],
        PYTEST + ["tests/unit/test_crew_hardening.py", "-k", "outer_deadline"],
        "FAILED",
    ),
    Case(
        "M85",
        "Error serialisation is total",
        "RagError keeps a NaN retry_after",
        [
            E(
                S + "errors.py",
                "        self.retry_after = _seconds_of(retry_after)",
                "        self.retry_after = retry_after",
            )
        ],
        PYTEST + ["tests/unit/test_crew_hardening.py", "-k", "error_model_itself"],
        "FAILED",
    ),
    Case(
        "M86",
        "DOCX keeps document order",
        "tables are read after all paragraphs",
        [
            E(
                S + "adapters/parser_docx/__init__.py",
                "for block in document.iter_inner_content():",
                "for block in [*document.paragraphs, *document.tables]:",
            )
        ],
        PYTEST + ["tests/unit/test_parsers_hardening.py", "-k", "stays_between"],
        "FAILED",
    ),
    Case(
        "M87",
        "DOCX merged cells",
        "a merged cell is read once per column it spans",
        [E(S + "adapters/parser_docx/__init__.py", "            if id(cell._tc) in seen:", "            if False:")],
        PYTEST + ["tests/unit/test_parsers_hardening.py", "-k", "merged_cell"],
        "FAILED",
    ),
    Case(
        "M88",
        "Chunking keeps all the text",
        "the chunker drops a final chunk equal to the previous one",
        [
            E(
                S + "adapters/chunker_recursive/__init__.py",
                '            pieces.append("\\n".join(current))',
                '            last = "\\n".join(current)\n            if not pieces or last != pieces[-1]:\n                pieces.append(last)',
            )
        ],
        PYTEST + ["tests/unit/test_parsers_hardening.py", "-k", "periodic"],
        "FAILED",
    ),
    Case(
        "M89",
        "Chunking is linear",
        "the hard split re-slices the remainder on every cut",
        [
            E(
                S + "adapters/chunker_recursive/__init__.py",
                "            start = cut\n            while start < end and text[start].isspace():\n                start += 1",
                "            text = text[cut:].lstrip()\n            start, end = 0, len(text)",
            )
        ],
        PYTEST + ["tests/unit/test_parsers_hardening.py", "-k", "scales_linearly"],
        "FAILED",
    ),
    Case(
        "M90",
        "Dataset loader reports bad lines",
        "a malformed qrels line is not reported",
        [E(S + "evaluation/dataset.py", "            if len(fields) != 3:", "            if False:")],
        PYTEST + ["tests/unit/test_evaluation.py", "-k", "blank_lines_and_reports"],
        "FAILED",
    ),
    Case(
        "M91",
        "Evaluation excludes unanswerable queries",
        "queries with only grade-0 judgments are scored",
        [E(S + "evaluation/dataset.py", "if any(g > 0 for g in j.values())", "if True")],
        PYTEST + ["tests/unit/test_evaluation.py", "-k", "excluded_as_in_trec_eval"],
        "FAILED",
    ),
    Case(
        "M92",
        "Evaluation keeps the first judgment",
        "the first qrels row is always treated as a header",
        [
            E(
                S + "evaluation/dataset.py",
                '            fields = line.rstrip("\\r\\n").split("\\t")\n',
                "            if number == 1:\n                continue  # always treat the first line as a header\n"
                '            fields = line.rstrip("\\r\\n").split("\\t")\n',
            )
        ],
        PYTEST + ["tests/unit/test_evaluation.py", "-k", "first_judgment"],
        "FAILED",
    ),
    Case(
        "M93",
        "Docs: cited tests exist",
        "an unverified provider row cites a test that does not exist",
        [
            E(
                "docs/providers.md",
                "`tests/conformance/test_providers_over_http.py::TestCrewAIEmbedder`",
                "`tests/conformance/test_providers_over_http.py::TestNoSuchThing`",
            )
        ],
        PYTEST + ["tests/docs/test_docs_match_code.py"],
        "FAILED",
    ),
    Case(
        "M94",
        "Defaults: a new setting must be classified",
        "a setting is added with neither a defaults.toml entry nor an operational classification",
        [
            E(
                S + "config.py",
                "    pipeline_max_tokens: int = Field(default=700, ge=16, le=8192)\n",
                "    pipeline_max_tokens: int = Field(default=700, ge=16, le=8192)\n    a_new_knob: int = 1\n",
            )
        ],
        PYTEST + ["tests/docs/test_docs_match_code.py", "-k", "classified"],
        "FAILED",
    ),
    Case(
        "M95",
        "Dependency text never reaches a caller (OpenAI)",
        "the OpenAI adapter copies the SDK's error text into its public error",
        [E(S + "adapters/openai/_errors.py", "    return failure()", "    return failure(str(exc))")],
        PYTEST + ["tests/conformance/test_error_text.py"],
        "FAILED",
    ),
    Case(
        "M96",
        "Dependency text never reaches a caller (CrewAI)",
        "the CrewAI adapter copies the provider's error text into its public error",
        [E(S + "adapters/crewai/_errors.py", "    return failure()", "    return failure(str(exc))")],
        PYTEST + ["tests/conformance/test_error_text.py"],
        "FAILED",
    ),
    Case(
        "M97",
        "Dependency text never reaches a caller (Milvus)",
        "the Milvus adapter copies the client's error text into its public error",
        [
            E(
                S + "adapters/milvus/store.py",
                "                raise VectorStoreError() from exc",
                "                raise VectorStoreError(str(exc)) from exc",
            )
        ],
        PYTEST + ["tests/conformance/test_error_text.py"],
        "FAILED",
    ),
    Case(
        "M98",
        "G2/G8 packaging: the thin wheel",
        "models imports an engine library, so the installed wheel cannot import without it",
        [E(S + "models.py", None, "\nimport pydantic_settings  # noqa\n")],
        [str(VENV / "python"), "scripts/check_wheel.py"],
        "pydantic_settings",
    ),
    Case(
        "M99",
        "G13 limits: chunk count",
        "the per-document chunk limit is not enforced after chunking",
        [
            E(
                S + "application/service.py",
                "        if len(chunks) > self._limits.max_chunks_per_document:",
                "        if False:",
            )
        ],
        PYTEST + ["tests/unit/test_limits.py", "-k", "max_chunks_per_document"],
        "FAILED",
    ),
    Case(
        "M100",
        "G13 limits: page size",
        "list_documents accepts any page size",
        [
            E(
                S + "application/service.py",
                "        if not 1 <= limit <= self._limits.max_page_size:",
                "        if False:",
            )
        ],
        PYTEST + ["tests/unit/test_limits.py", "-k", "max_page_size"],
        "FAILED",
    ),
    Case(
        "M101",
        "G13 limits: embedding batches",
        "all chunks are sent to the embedder in one call",
        [E(S + "application/service.py", "        batch = self._limits.embed_batch_size\n", "        batch = 10**9\n")],
        PYTEST + ["tests/unit/test_limits.py", "-k", "embed_batch_size"],
        "FAILED",
    ),
    Case(
        "M102",
        "G13 limits: concurrent embedding batches",
        "the embedding concurrency bound is ignored",
        [
            E(
                S + "application/service.py",
                "        slots = asyncio.Semaphore(self._limits.max_concurrent_embed_batches)",
                "        slots = asyncio.Semaphore(10**6)",
            )
        ],
        PYTEST + ["tests/unit/test_limits.py", "-k", "max_concurrent_embed_batches"],
        "FAILED",
    ),
    Case(
        "M103",
        "G13 limits: concurrent ingests",
        "the ingest concurrency bound is ignored",
        [
            E(
                S + "application/service.py",
                "        self._ingest_slots = asyncio.Semaphore(limits.max_concurrent_ingests)",
                "        self._ingest_slots = asyncio.Semaphore(10**6)",
            ),
            E(
                S + "application/service.py",
                "            workers=limits.max_concurrent_ingests,\n            backlog=limits.max_concurrent_ingests,",
                "            workers=64,\n            backlog=64,",
            ),
        ],
        PYTEST + ["tests/unit/test_limits.py", "-k", "max_concurrent_ingests"],
        "FAILED",
    ),
    Case(
        "M104",
        "G13 limits: PDF pages",
        "the PDF page limit is not enforced",
        [
            E(
                S + "adapters/parser_pdf/__init__.py",
                "            if doc.page_count > self._max_pages:",
                "            if False:",
            )
        ],
        PYTEST + ["tests/unit/test_limits.py", "-k", "pdf_max_pages"],
        "FAILED",
    ),
    Case(
        "M105",
        "G13 limits: CrewAI search budget",
        "the search tool ignores its budget",
        [E(S + "adapters/crewai/pipeline.py", "        if self._used >= self.budget:", "        if False:")],
        PYTEST + ["tests/unit/test_crew_hardening.py", "-k", "search_budget"],
        "FAILED",
    ),
    Case(
        "M106",
        "G14 evaluation gate",
        "BM25 ranks worst first",
        [
            E(
                S + "adapters/lexical_bm25/__init__.py",
                "ranked = sorted(scores, key=lambda cid: (-scores[cid], cid))[:top_k]",
                "ranked = sorted(scores, key=lambda cid: (scores[cid], cid))[:top_k]",
            )
        ],
        PYTEST + ["tests/unit/test_golden_gate.py", "-k", "current_retrieval"],
        "FAILED",
    ),
    Case(
        "M107",
        "Telemetry stays off",
        "the container no longer disables CrewAI's usage telemetry",
        [E(S + "container.py", 'os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")', "pass")],
        PYTEST + ["tests/unit/test_telemetry_off.py", "-k", "container"],
        "FAILED",
    ),
    Case(
        "M108",
        "Security: no credential committed",
        "an API-key-shaped string is committed in the README",
        [E("README.md", None, '\nexample: api_key = "' + "q" * 40 + '"\n')],
        PYTEST + ["tests/architecture/test_no_secrets_committed.py", "-k", "credential_shaped"],
        "FAILED",
        needs_git=True,
    ),
    Case(
        "M109",
        "Prompt injection hygiene (contract)",
        "a pipeline sends retrieved text to the model unescaped",
        [
            E(
                S + "ports/answers.py",
                'f"{html.escape(hit.chunk.text, quote=False)}</chunk>"',
                'f"{hit.chunk.text}</chunk>"',
            )
        ],
        PYTEST + ["tests/conformance/test_components.py", "-k", "delimiter"],
        "FAILED",
    ),
    Case(
        "M110",
        "A start-up failure is a typed error",
        "an unwritable CrewAI data directory escapes as a raw OSError",
        [
            E(
                S + "container.py",
                "    except OSError as exc:  # importing crewai creates a data directory under XDG_DATA_HOME\n        raise _crewai_data_dir() from exc",
                "    except ZeroDivisionError:\n        raise",
            ),
            E(
                S + "container.py",
                "    except OSError as exc:  # importing crewai creates a data directory under XDG_DATA_HOME\n        raise _crewai_data_dir() from exc",
                "    except ZeroDivisionError:\n        raise",
            ),
            E(
                S + "container.py",
                "    except OSError as exc:  # importing crewai creates a data directory under XDG_DATA_HOME\n        raise _crewai_data_dir() from exc",
                "    except ZeroDivisionError:\n        raise",
            ),
        ],
        PYTEST + ["tests/unit/test_container_hardening.py", "-k", "data_directory"],
        "FAILED",
    ),
]


def apply(tree: Path, edits: list[Edit]) -> None:
    for edit in edits:
        path = tree / edit.path
        if edit.old is None:
            path.write_text(path.read_text() + edit.new)
        elif edit.old == "":
            path.write_text(edit.new)
        else:
            text = path.read_text()
            if edit.old not in text:
                raise SystemExit(f"mutation does not apply (text not found): {edit.path}: {edit.old[:60]!r}")
            path.write_text(text.replace(edit.old, edit.new, 1))


def run(tree: Path, command: list[str], milvus: bool) -> tuple[int, str]:
    env = {**os.environ, "PYTHONPATH": str(tree / "src"), "AGENTIC_RAG_MILVUS_URI": ""}
    env.pop("AGENTIC_RAG_MILVUS_URI")
    try:
        proc = subprocess.run(command, cwd=tree, env=env, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        # A check that hangs has not *failed* in the way the proof needs ("FAILED" and a message): report it as such.
        return 124, "TIMEOUT: the check did not finish in 300 s"
    return proc.returncode, proc.stdout + proc.stderr


def fresh(needs_git: bool) -> Path:
    tree = Path(tempfile.mkdtemp(prefix="prove_"))
    shutil.copytree(ROOT, tree, ignore=IGNORE, dirs_exist_ok=True)
    if needs_git:
        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
        }
        for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "baseline"]):
            subprocess.run(["git", *args], cwd=tree, env=env, check=True, capture_output=True)
    return tree


def evidence(output: str, expect: str) -> str:
    for line in output.splitlines():
        if expect in line:
            return line.strip()[:150]
    return output.strip().splitlines()[-1][:150] if output.strip() else ""


def prove(case: Case) -> bool:
    control_tree = fresh(case.needs_git)
    code, out = run(control_tree, case.command, case.milvus)
    shutil.rmtree(control_tree, ignore_errors=True)
    control_ok = code == 0
    tree = fresh(case.needs_git)
    apply(tree, case.edits)
    mcode, mout = run(tree, case.command, case.milvus)
    shutil.rmtree(tree, ignore_errors=True)
    failed = mcode != 0 and case.expect in mout
    case.result = {
        "control": "pass" if control_ok else f"FAIL(exit {code})",
        "mutated": "FAILS as required" if failed else f"NOT CAUGHT (exit {mcode})",
        "evidence": evidence(mout, case.expect) if failed else "",
    }
    print(
        f"{case.id} {case.rule}: control={case.result['control']} mutated={case.result['mutated']}",
        flush=True,
    )
    return control_ok and failed


def _row(c: Case) -> str:
    cmd = " ".join(Path(p).name if p.startswith("/") else p for p in c.command).replace("-p no:cacheprovider ", "")
    ev = c.result["evidence"].replace("|", "\\|")
    return f"| {c.id} | {c.rule} | {c.what} | `{cmd}` | {c.result['control']} | {c.result['mutated']} | `{ev}` |"


def write_log(cases: list[Case]) -> None:
    lines = [
        "# Mutation proofs",
        "",
        "Each rule in the governance table (framework section 12) was broken on purpose in a copy of the repository and its check was run;",
        "a rule counts as *enforced* only if the check passes on the clean copy and fails on the mutated one.",
        "Regenerate with `python scripts/prove_rules.py` (about an hour with `--jobs 3`: every case runs its check twice). Commands run from the repository root with the project's virtualenv.",
        "",
        "| # | Rule | Mutation | Check | Clean copy | Mutated copy | Evidence (first matching output line) |",
        "|---|---|---|---|---|---|---|",
    ]
    lines.extend(_row(c) for c in cases)
    (ROOT / "docs" / "mutation-proofs.md").write_text("\n".join(lines) + "\n")


def merge_log(rerun: list[Case]) -> None:
    """Replace the rows of the cases just re-run, keep every other row of the existing log."""
    path = ROOT / "docs" / "mutation-proofs.md"
    new_rows = {c.id: _row(c) for c in rerun}
    lines = []
    for line in path.read_text().splitlines():
        case_id = line.split("|")[1].strip() if line.startswith("| M") else None
        lines.append(new_rows.pop(case_id, line) if case_id else line)
    lines.extend(new_rows.values())  # a case that was not in the log yet
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str]) -> int:
    jobs, merge = 1, False
    while argv[:1] and argv[0] in {"--jobs", "--merge"}:
        if argv[0] == "--jobs":
            jobs, argv = int(argv[1]), argv[2:]
        else:
            merge, argv = True, argv[1:]
    selected = [c for c in CASES if not argv or c.id in argv]
    with ThreadPoolExecutor(max_workers=jobs) as pool:  # each case works in its own copy of the repository
        ok = list(pool.map(prove, selected))
    if not argv:
        write_log(selected)
    elif merge:
        merge_log(selected)
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
