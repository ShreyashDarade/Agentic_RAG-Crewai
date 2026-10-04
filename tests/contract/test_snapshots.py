"""G5/G6/G7: the fixed surfaces equal their checked-in snapshots. Regenerate only with scripts/snapshots.py."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("snapshots", ROOT / "scripts" / "snapshots.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


snapshots = _load()
HINT = "If this change is intended: write/update the ADR, then run `python scripts/snapshots.py regenerate`."


@pytest.mark.parametrize("name", sorted(snapshots.SNAPSHOTS))
def test_snapshot_matches_code(name: str) -> None:
    difference = snapshots.diff(name)
    assert not difference, f"docs/{name} is out of date:\n{difference}\n{HINT}"


def test_error_codes_are_append_only() -> None:
    assert snapshots.breaking_error_changes() == []


def test_error_code_snapshot_is_well_formed() -> None:
    entries = json.loads((ROOT / "docs" / "error_codes.json").read_text())
    codes = [e["code"] for e in entries]
    assert len(codes) == len(set(codes))
    assert all(re.fullmatch(r"[A-Z][A-Z0-9_]+[A-Z0-9]", c) and len(c) <= 63 for c in codes)


def test_openapi_documents_security_problem_responses_and_unique_operation_ids() -> None:
    spec = json.loads((ROOT / "docs" / "openapi.json").read_text())
    assert spec["openapi"].startswith("3.1")
    assert "HTTPBearer" in spec["components"]["securitySchemes"]
    ids = [op["operationId"] for path in spec["paths"].values() for op in path.values()]
    assert len(ids) == len(set(ids))
    for path, ops in spec["paths"].items():
        for method, op in ops.items():
            if path.startswith("/v1"):
                assert op.get("security"), f"{method} {path} is not marked as requiring the API key"
                assert "401" in op["responses"]
