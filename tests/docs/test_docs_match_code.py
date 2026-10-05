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

#: Settings that are not quality defaults: component identity, credentials, security, limits, timeouts, logging.
#: Every other setting must have an entry (with evidence or "unmeasured") in docs/defaults.toml, so adding a setting
#: forces a decision: classify it here or document its evidence.
OPERATIONAL = {
    "api_keys", "cors_allow_origins", "allow_unauthenticated",
    "vector_store", "lexical_index", "embedder", "chat_model", "reranker", "chunker", "parsers", "plugins",
    "milvus_uri", "milvus_token", "milvus_collection", "consistency_level",
    "chroma_path", "chroma_url", "chroma_collection",
    "qdrant_location", "qdrant_api_key", "qdrant_collection",
    "crewai_llm_model", "crewai_llm_api_key", "crewai_llm_base_url",
    "crewai_embedder_provider", "crewai_embedder_model", "crewai_embedder_api_key", "crewai_embedder_base_url",
    "crewai_embedder_options", "crewai_max_concurrent", "crewai_provider_timeout_seconds",
    "openai_api_key", "openai_base_url", "openai_timeout_seconds", "embedding_model", "embedding_dimension",
    "openai_chat_model",
    "pdf_max_pages", "max_upload_bytes", "max_question_chars", "max_top_k", "max_page_size",
    "max_chunks_per_document", "embed_batch_size", "max_concurrent_embed_batches", "max_concurrent_ingests",
    "request_deadline_seconds", "health_check_timeout_seconds", "store_timeout_seconds", "max_inflight_requests",
    "shutdown_drain_seconds", "log_json", "log_level",
}  # fmt: skip


def test_every_setting_is_classified_and_every_quality_default_has_evidence_or_says_unmeasured() -> None:
    entries = tomllib.loads((DOCS / "defaults.toml").read_text())
    defaults = {name: field.default for name, field in Settings.model_fields.items()}
    unclassified = sorted(set(defaults) - OPERATIONAL - set(entries))
    assert not unclassified, (
        f"settings with neither a docs/defaults.toml entry nor an OPERATIONAL classification: {unclassified}"
    )
    assert not OPERATIONAL - set(defaults), (
        f"OPERATIONAL names a setting that does not exist: {sorted(OPERATIONAL - set(defaults))}"
    )
    assert not OPERATIONAL & set(entries), "a setting is both operational and a documented quality default"
    for name, entry in entries.items():
        if name in defaults:
            assert entry["value"] == defaults[name], f"{name}: documented {entry['value']!r}, code {defaults[name]!r}"
        else:
            assert name.startswith("bm25_"), f"docs/defaults.toml has an entry for {name}, which is not a setting"
        evidence = entry["evidence"]
        assert evidence == "unmeasured" or (ROOT / evidence).is_file(), f"{name}: evidence {evidence!r} does not exist"


def _table_rows(text: str) -> list[list[str]]:
    rows = []
    for line in text.splitlines():
        if line.startswith("|") and not re.match(r"^\|[\s:-]+\|", line):
            rows.append([c.strip() for c in line.strip().strip("|").split("|")])
    return rows


def _tests_text() -> str:
    return "\n".join(p.read_text() for p in (ROOT / "tests").rglob("*.py"))


def test_every_test_a_provider_row_cites_exists_and_every_verified_row_cites_one() -> None:
    text = (DOCS / "providers.md").read_text()
    sources = _tests_text()
    verified_rows = 0
    for row in _table_rows(text):
        evidence = row[-1]
        refs: list[str] = []
        for token in re.findall(r"`([^`]+)`", evidence):
            if token.startswith("tests/"):
                path, _, symbol = token.partition("::")
                assert (ROOT / path).is_file(), f"{path} does not exist"
                if symbol:
                    assert symbol in (ROOT / path).read_text(), f"{symbol} not found in {path}"
                refs.append(token)
            elif re.fullmatch(r"Test\w+(::\w+)?", token) or re.fullmatch(r"test_\w+", token):
                for part in token.split("::"):
                    assert re.search(rf"\b{part}\b", sources), f"{part} is cited in providers.md but is in no test file"
                refs.append(token)
        if re.search(r"(?<!un)verified", " ".join(row[:-1]).lower()):
            verified_rows += 1
            assert refs, f"a row claims 'verified' but cites no test: {row[0]}"
    assert verified_rows >= 6


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
    secrets = [n for n, f in Settings.model_fields.items() if "SecretStr" in str(f.annotation)]
    known = (
        {"AGENTIC_RAG_" + n.upper() for n in Settings.model_fields}
        | {"AGENTIC_RAG_" + n.upper() + "_FILE" for n in secrets}  # mounted secrets
        | {"AGENTIC_RAG_API_KEY"}  # the SDK's variable, accepted and ignored by the server
    )
    for doc in [*DOCS.glob("*.md"), ROOT / "README.md", ROOT / ".env.example"]:
        for var in set(re.findall(r"AGENTIC_RAG_[A-Z_]+", doc.read_text())):
            if var.endswith("_") or (var in {"AGENTIC_RAG_PLUGINS"} and "plugins" in Settings.model_fields):
                continue
            assert var in known, f"{doc.name} mentions {var}, which is not a setting"


@pytest.mark.parametrize("doc", ["framework.md", "operations.md", "security.md", "benchmark.md", "evaluation.md"])
def test_docs_exist(doc: str) -> None:
    assert (DOCS / doc).is_file()
