from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from agentic_rag.api import ApiConfig, create_app
from tests.conftest import API_KEY
from tests.unit.test_service import build


@pytest.fixture
def parts():  # type: ignore[no-untyped-def]
    return build()


@pytest.fixture
def client(parts) -> Iterator[TestClient]:  # type: ignore[no-untyped-def]
    service, *_ = parts
    app = create_app(config=ApiConfig(api_keys=(API_KEY,), max_upload_bytes=200_000), service=service)
    with TestClient(app) as c:
        yield c
