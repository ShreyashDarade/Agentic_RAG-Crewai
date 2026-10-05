from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session")
def milvus_uri(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A real Milvus: ``AGENTIC_RAG_MILVUS_URI`` (standalone in CI) or a Milvus Lite file."""
    explicit = os.environ.get("AGENTIC_RAG_MILVUS_URI")
    if explicit:
        return explicit
    pytest.importorskip("milvus_lite")
    return str(tmp_path_factory.mktemp("milvus") / "lite.db")


@pytest.fixture
def collection_name(request: pytest.FixtureRequest) -> str:
    return "t_" + "".join(ch if ch.isalnum() else "_" for ch in request.node.name)[-40:] + "_" + os.urandom(3).hex()


API_KEY = "test-key-0123456789abcdef"


@pytest.fixture
def api_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {API_KEY}"}
