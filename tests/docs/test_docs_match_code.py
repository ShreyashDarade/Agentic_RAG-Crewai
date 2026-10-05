"""G14/G15: documents claim only what the code and the evidence show."""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

import pytest

from agentic_rag.config import Settings

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"

REQUIRED_DEFAULTS = {
    "default_top_k": "default_top_k",
    "candidate_pool": "candidate_pool",
    "rrf_k": "rrf_k",
    "use_lexical": "use_lexical",
    "use_reranker": "use_reranker",
    "chunk_max_chars": "chunk_max_chars",
    "chunk_overlap_chars": "chunk_overlap_chars",
    "answer_pipeline": "answer_pipeline",
    "hnsw_m": "hnsw_m",
    "hnsw_ef_construction": "hnsw_ef_construction",
    "search_ef": "search_ef",
    "pipeline_max_tokens": "pipeline_max_tokens",
}


def test_every_quality_default_has_evidence_or_says_unmeasured_and_matches_settings() -> None:
    entries = tomllib.loads((DOCS / "defaults.toml").read_text())
    defaults = {name: field.default for name, field in Settings.model_fields.items()}
    for name, setting in REQUIRED_DEFAULTS.items():
        assert name in entries, f"docs/defaults.toml has no entry for {name}"
        assert entries[name]["value"] == defaults[setting], (
            f"{name}: documented {entries[name]['value']!r}, code {defaults[setting]!r}"
        )
    for name, entry in entries.items():
        evidence = entry["evidence"]
        assert evidence == "unmeasured" or (ROOT / evidence).is_file(), f"{name}: evidence {evidence!r} does not exist"


def _table_rows(text: str) -> list[list[str]]:
    rows = []
    for line in text.splitlines():
        if line.startswith("|") and not re.match(r"^\|[\s:-]+\|", line):
            rows.append([c.strip() for c in line.strip().strip("|").split("|")])
    return rows


def test_every_verified_provider_row_cites_a_test_that_exists() -> None:
    text = (DOCS / "providers.md").read_text()
    checked = 0
    for row in _table_rows(text):
        status = " ".join(row).lower()
        if "unverified" in status or "verified" not in status:
            continue
        evidence = row[-1]
        for ref in re.findall(r"`(tests/[^`:]+\.py)(?:::([A-Za-z0-9_]+))?", evidence):
            path, symbol = ref
            assert (ROOT / path).is_file(), f"{path} does not exist"
            if symbol:
                assert symbol in (ROOT / path).read_text(), f"{symbol} not found in {path}"
            checked += 1
    assert checked >= 6


def _registered_names() -> dict[str, set[str]]:
    tree = ast.parse((ROOT / "src" / "agentic_rag" / "container.py").read_text())
    found: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "register"
            and isinstance(node.func.value, ast.Attribute)
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            found.setdefault(node.func.value.attr, set()).add(str(node.args[0].value))
    return found


def test_every_registered_component_is_documented_in_providers_md() -> None:
    text = (DOCS / "providers.md").read_text()
    for kind, names in _registered_names().items():
        if kind in {"parser", "chunker", "lexical_index"}:
            continue
        for name in names:
            assert f"`{name}`" in text, f"component {kind}/{name} is registered but not in docs/providers.md"


def test_the_readme_install_matrix_matches_the_extras() -> None:
    extras = set(tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"])
    readme = (ROOT / "README.md").read_text()
    for extra in extras - {"worker"}:
        assert f"[{extra}]" in readme or f"`{extra}`" in readme, f"extra {extra} is not documented in README.md"


def test_documented_routes_exist_in_the_openapi_snapshot() -> None:
    import json

    spec = json.loads((DOCS / "openapi.json").read_text())
    readme = (ROOT / "README.md").read_text()
    paths = set(spec["paths"])
    for route in re.findall(r"`(?:GET|POST|DELETE) (/[\w/{}-]+)`", readme):
        assert route in paths, f"README documents {route}, which is not in docs/openapi.json"


def test_environment_variables_named_in_docs_exist_in_settings() -> None:
    known = {"AGENTIC_RAG_" + n.upper() for n in Settings.model_fields} | {"AGENTIC_RAG_API_KEY"}
    for doc in [*DOCS.glob("*.md"), ROOT / "README.md", ROOT / ".env.example"]:
        for var in set(re.findall(r"AGENTIC_RAG_[A-Z_]+", doc.read_text())):
            if var.endswith("_") or (var in {"AGENTIC_RAG_PLUGINS"} and "plugins" in Settings.model_fields):
                continue
            assert var in known, f"{doc.name} mentions {var}, which is not a setting"


@pytest.mark.parametrize("doc", ["framework.md", "operations.md", "security.md", "benchmark.md", "evaluation.md"])
def test_docs_exist(doc: str) -> None:
    assert (DOCS / doc).is_file()
