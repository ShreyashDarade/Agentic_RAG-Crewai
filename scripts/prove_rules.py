#!/usr/bin/env python3
"""Mutation proof for the governance table (framework section 12).

For each rule: copy the repository, break the rule on purpose, run the rule's check, and require that it FAILS
(while the same check PASSES on an unmutated copy). The result is written to docs/mutation-proofs.md.

    scripts/prove_rules.py            run every case and rewrite docs/mutation-proofs.md
    scripts/prove_rules.py M03 M10    run only these cases (does not rewrite the log)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv" / "bin"
IGNORE = shutil.ignore_patterns(
    ".git", ".venv", ".dev", "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache", ".coverage", "htmlcov"
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
    Case("M01", "G1 layers", "application imports the HTTP layer",
         [E(S + "application/limits.py", None, "\nfrom agentic_rag.api import ApiConfig  # noqa\n")], LINT, "G1 layers"),
    Case("M02", "G1 adapter independence", "pipeline_direct imports the milvus adapter",
         [E(S + "adapters/pipeline_direct/__init__.py", None, "\nfrom agentic_rag.adapters.milvus import MilvusStore  # noqa\n")],
         LINT, "G1 adapters are independent"),
    Case("M03", "G2 domain purity", "application imports pymilvus",
         [E(S + "application/service.py", None, "\nimport pymilvus  # noqa\n")], LINT, "G2 domain code"),
    Case("M04", "G2 confinement", "the milvus adapter imports openai",
         [E(S + "adapters/milvus/filters.py", None, "\nimport openai  # noqa\n")], LINT, "G2 openai is confined"),
    Case("M05", "G2 thin foundation", "ports import httpx",
         [E(S + "ports/types.py", None, "\nimport httpx  # noqa\n")], LINT, "G2 domain code"),
    Case("M06", "G2/G3 unclassified library", "a new third-party import without a classification",
         [E(S + "adapters/lexical_bm25/__init__.py", None, "\nimport numpy  # noqa\n")],
         PYTEST + ["tests/architecture/test_confinement.py"], "unclassified"),
    Case("M07", "G3 exhaustive classification", "a new top-level module nobody classified",
         [E(S + "extras.py", "", '"""unclassified"""\n')], LINT, "extras"),
    Case("M08", "G4 routes via the service", "a route module imports a port",
         [E(S + "api/routes.py", None, "\nfrom agentic_rag.ports import Embedder  # noqa\n")], LINT, "G4 routes"),
    Case("M09", "G4 thin routes", "a health route grows 20 lines of logic",
         [E(S + "api/routes.py", '    return {"status": "ok"}',
            "\n".join(f"    value_{i} = {i}" for i in range(20)) + '\n    return {"status": "ok"}')],
         PYTEST + ["tests/architecture/test_routes_and_errors.py"], "body lines"),
    Case("M10", "G5 public API snapshot", "a name disappears from contracts.__all__",
         [E(S + "contracts/__init__.py", '    "DeleteResult",\n', "")], PYTEST + ["tests/contract"], "public_api.txt"),
    Case("M11", "G5 API compatibility vs last ref (griffe)", "a public keyword parameter is removed",
         [E(S + "errors.py", "def error_from_problem(body: Mapping[str, Any], *, status: int | None = None)",
            "def error_from_problem(body: Mapping[str, Any])"),
          E(S + "errors.py", "http_status = int(body.get(\"status\", status or 0))", "http_status = int(body.get(\"status\", 0))")],
         GRIFFE, "error_from_problem", needs_git=True),
    Case("M12", "G6 wire contract snapshot", "a response field changes type",
         [E(S + "contracts/models.py", "    text: str\n    score: float\n    page: int | None = None\n\n\nclass SearchResponse",
            "    text: str\n    score: str\n    page: int | None = None\n\n\nclass SearchResponse")],
         SNAP, "openapi.json"),
    Case("M13", "G7 error codes append-only", "an error code is renamed",
         [E(S + "errors.py", 'code = "DOCUMENT_NOT_FOUND"', 'code = "DOC_NOT_FOUND"')],
         PYTEST + ["tests/contract"], "error_codes.json"),
    Case("M14", "G7 error status stable", "NotFound changes status",
         [E(S + "errors.py", "    code = \"NOT_FOUND\"\n    http_status = 404", "    code = \"NOT_FOUND\"\n    http_status = 410")],
         PYTEST + ["tests/contract"], "error_codes.json"),
    Case("M15", "G8 strict typing", "an untyped function in the public package",
         [E(S + "registry.py", None, "\n\ndef helper(x):\n    return x\n")], MYPY, "no-untyped-def"),
    Case("M16", "G9 conformance on a real adapter", "Milvus delete ignores keep_chunk_ids",
         [E(S + "adapters/milvus/store.py", '            expr += f" and id not in {quote_ids(sorted(keep_chunk_ids))}"', "            pass")],
         PYTEST + ["tests/conformance/test_stores.py", "-k", "Milvus and sweeps"], "FAILED", milvus=True),
    Case("M17", "G9/G12 conformance catches a silent fallback", "the OpenAI embedder returns [] on provider errors",
         [E(S + "adapters/openai/embedder.py", "            raise map_error(exc, EmbeddingFailed) from exc", "            return []")],
         PYTEST + ["tests/conformance/test_openai.py"], "FAILED"),
    Case("M18", "G10 open-closed", "the registry ignores registered factories",
         [E(S + "registry.py", "        factory = self._factories.get(name)\n        if factory is None:\n            factory = self._load_entry_point(name)",
            "        factory = None\n        if factory is None:\n            factory = self._load_entry_point(name)")],
         PYTEST + ["tests/architecture/test_open_closed.py"], "FAILED"),
    Case("M19", "G12 no silent fallbacks (lint)", "a parser failure is swallowed into an empty document",
         [E(S + "application/service.py", "            raise DocumentParseFailed() from exc", "            return ParsedDocument(text='')")],
         RUFF, "BLE001"),
    Case("M20", "G12 no silent fallbacks (AST)", "a broad except returns a default",
         [E(S + "application/service.py", "            raise DocumentParseFailed() from exc", "            return ParsedDocument(text='')")],
         PYTEST + ["tests/architecture/test_routes_and_errors.py"], "swallows an exception"),
    Case("M21", "G13 limits", "the upload size check is removed",
         [E(S + "application/service.py", "if len(data) > self._limits.max_upload_bytes:", "if False:")],
         PYTEST + ["tests/unit/test_service.py"], "FAILED"),
    Case("M22", "Security: filter alphabet", "id validation in the Milvus filter builder is removed",
         [E(S + "adapters/milvus/filters.py", "        if not _ID.fullmatch(value):", "        if False:")],
         PYTEST + ["tests/conformance/test_stores.py", "-k", "Milvus and hostile"], "FAILED", milvus=True),
    Case("M23", "Security: authentication", "the API-key check is disabled",
         [E(S + "api/auth.py", "    if not (credentials and any(matches)):", "    if False:")],
         PYTEST + ["tests/api/test_api.py"], "FAILED"),
    Case("M24", "Security: request-id validation", "any inbound X-Request-ID is trusted",
         [E(S + "api/middleware.py", 'request_id = inbound if inbound and _VALID_ID.fullmatch(inbound) else uuid.uuid4().hex',
            "request_id = inbound or uuid.uuid4().hex")],
         PYTEST + ["tests/api/test_api.py"], "FAILED"),
    Case("M25", "Security: error text leak", "an unexpected exception message is returned to the client",
         [E(S + "api/middleware.py", "_send_problem(send_with_id, RagError(request_id=request_id))",
            "_send_problem(send_with_id, RagError(str(sys.exc_info()[1]), request_id=request_id))"),
          E(S + "api/middleware.py", "import re\n", "import re\nimport sys\n")],
         PYTEST + ["tests/api/test_api.py"], "FAILED"),
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
    proc = subprocess.run(command, cwd=tree, env=env, capture_output=True, text=True, timeout=600)
    return proc.returncode, proc.stdout + proc.stderr


def fresh(needs_git: bool) -> Path:
    tree = Path(tempfile.mkdtemp(prefix="prove_"))
    shutil.copytree(ROOT, tree, ignore=IGNORE, dirs_exist_ok=True)
    if needs_git:
        env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
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
    print(f"{case.id} {case.rule}: control={case.result['control']} mutated={case.result['mutated']}", flush=True)
    return control_ok and failed


def write_log(cases: list[Case]) -> None:
    lines = [
        "# Mutation proofs",
        "",
        "Each rule in the governance table (framework section 12) was broken on purpose in a copy of the repository and its check was run;",
        "a rule counts as *enforced* only if the check passes on the clean copy and fails on the mutated one.",
        "Regenerate with `python scripts/prove_rules.py` (about two minutes). Commands run from the repository root with the project's virtualenv.",
        "",
        "| # | Rule | Mutation | Check | Clean copy | Mutated copy | Evidence (first matching output line) |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in cases:
        cmd = " ".join(Path(p).name if p.startswith("/") else p for p in c.command).replace("-p no:cacheprovider ", "")
        ev = c.result["evidence"].replace("|", "\\|")
        lines.append(f"| {c.id} | {c.rule} | {c.what} | `{cmd}` | {c.result['control']} | {c.result['mutated']} | `{ev}` |")
    (ROOT / "docs" / "mutation-proofs.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str]) -> int:
    selected = [c for c in CASES if not argv or c.id in argv]
    ok = [prove(c) for c in selected]
    if not argv:
        write_log(selected)
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
