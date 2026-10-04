from __future__ import annotations

import json
import re

import pytest

from agentic_rag import errors
from agentic_rag.errors import RagError, RagStatusError, catalog, error_from_problem

WIRE = {code: cls for code, cls in catalog().items() if cls.http_status != 0}


@pytest.mark.parametrize("code", sorted(WIRE))
def test_every_wire_code_round_trips_to_the_same_class(code: str) -> None:
    cls = WIRE[code]
    original = cls("safe detail", details={"k": 1}, request_id="req_12345678", retry_after=2.0)
    rebuilt = error_from_problem(json.loads(json.dumps(original.to_problem())))
    assert type(rebuilt) is cls
    assert rebuilt.code == code
    assert rebuilt.http_status == cls.http_status
    assert rebuilt.detail == "safe detail"
    assert rebuilt.details == {"k": 1}
    assert rebuilt.request_id == "req_12345678"
    assert rebuilt.retry_after == 2.0


def test_unknown_code_from_a_newer_server_keeps_everything() -> None:
    body = {
        "type": "urn:x",
        "title": "New",
        "status": 418,
        "code": "BREWING_COFFEE",
        "request_id": "abcdefgh1",
        "details": {"pot": "tea"},
    }
    err = error_from_problem(body)
    assert isinstance(err, RagStatusError)
    assert (err.code, err.http_status, err.request_id, err.details) == (
        "BREWING_COFFEE",
        418,
        "abcdefgh1",
        {"pot": "tea"},
    )


def test_codes_are_unique_well_formed_and_not_inherited() -> None:
    pattern = re.compile(r"^[A-Z][A-Z0-9_]+[A-Z0-9]$")
    seen: dict[str, str] = {}
    for code, cls in catalog().items():
        assert pattern.match(code) and len(code) <= 63
        assert cls.__dict__.get("code", "INTERNAL_ERROR") == code
        assert code not in seen
        seen[code] = cls.__name__


def test_a_subclass_cannot_reuse_its_parents_identity() -> None:
    with pytest.raises(TypeError, match="own"):

        class Bad(errors.NotFound):
            pass


def test_a_duplicate_code_is_rejected() -> None:
    with pytest.raises(TypeError, match="already belongs"):

        class Dup(RagError):
            code = "NOT_FOUND"
            http_status = 404
            public_message = "dup"


def test_a_malformed_code_is_rejected() -> None:
    with pytest.raises(TypeError, match="must match"):

        class Lower(RagError):
            code = "not_upper"
            http_status = 400
            public_message = "x"


def test_dependency_text_never_reaches_the_wire() -> None:
    try:
        try:
            raise RuntimeError("password=hunter2 host=10.0.0.5")
        except RuntimeError as exc:
            raise errors.VectorStoreError() from exc
    except errors.VectorStoreError as err:
        body = json.dumps(err.to_problem())
    assert "hunter2" not in body
    assert "10.0.0.5" not in body


def test_client_side_errors_are_not_on_the_wire() -> None:
    assert errors.UsageError.http_status == 0
    assert isinstance(error_from_problem({"code": "USAGE_ERROR", "status": 400}), RagStatusError)
