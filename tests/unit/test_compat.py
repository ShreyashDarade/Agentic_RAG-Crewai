"""G18: deprecation metadata is validated, warnings point at the caller, escalation follows the version."""

from __future__ import annotations

import warnings

import pytest

from agentic_rag import _compat
from agentic_rag._compat import (
    AgenticRagDeprecationWarning,
    AgenticRagFutureWarning,
    deprecated,
    experimental,
    tier_of,
)


@pytest.fixture
def version(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    def set_version(value: str) -> None:
        monkeypatch.setattr(_compat, "__version__", value)

    set_version("1.4.0")
    return set_version


@pytest.mark.parametrize(
    "kwargs",
    [
        {"since": "1.0.0", "remove_in": "1.9.0"},  # removal in the same major
        {"since": "1.0.0", "remove_in": "0.9.0"},  # removal in an earlier major
        {"since": "2.0.0", "remove_in": "3.0.0"},  # since is newer than the package
        {"since": "1.0", "remove_in": "2.0.0"},  # malformed
        {"since": "1.0.0", "remove_in": "2.0.0", "escalate_in": "2.0.0"},  # escalation at/after removal
        {"since": "1.2.0", "remove_in": "2.0.0", "escalate_in": "1.1.0"},  # escalation before the deprecation
    ],
)
def test_bad_policies_fail_when_the_decorator_is_applied(version, kwargs: dict[str, str]) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        deprecated(**kwargs)


def test_function_warns_at_the_callers_line_with_since_removal_and_alternative(version) -> None:  # type: ignore[no-untyped-def]
    @deprecated(since="1.2.0", remove_in="2.0.0", alternative="new_thing()")
    def old(x: int) -> int:
        """Doc."""
        return x + 1

    with pytest.warns(AgenticRagDeprecationWarning) as record:
        assert old(1) == 2  # <- the warning must be attributed to this line
    warning = record[0]
    assert warning.filename == __file__
    text = str(warning.message)
    assert "since 1.2.0" in text
    assert "2.0.0" in text
    assert "new_thing()" in text
    assert ".. deprecated:: 1.2.0" in (old.__doc__ or "")
    assert old.__name__ == "old"


async def test_async_function_and_class_are_covered(version) -> None:  # type: ignore[no-untyped-def]
    @deprecated(since="1.2.0", remove_in="2.0.0")
    async def old() -> str:
        return "ok"

    @deprecated(since="1.3.0", remove_in="3.0.0", alternative="New")
    class Old:
        def __init__(self, value: int) -> None:
            self.value = value

    with pytest.warns(AgenticRagDeprecationWarning):
        assert await old() == "ok"
    with pytest.warns(AgenticRagDeprecationWarning, match="Old is deprecated"):
        assert Old(3).value == 3


def test_the_warning_escalates_to_a_visible_one_from_the_escalation_release(version) -> None:  # type: ignore[no-untyped-def]
    @deprecated(since="1.2.0", remove_in="2.0.0", escalate_in="1.6.0")
    def old() -> None:
        return None

    with pytest.warns(AgenticRagDeprecationWarning):
        old()
    version("1.6.0")
    with pytest.warns(AgenticRagFutureWarning):
        old()
    assert not issubclass(AgenticRagFutureWarning, AgenticRagDeprecationWarning)
    assert issubclass(AgenticRagFutureWarning, FutureWarning)
    assert issubclass(AgenticRagDeprecationWarning, DeprecationWarning)


def test_the_suites_own_policy_turns_our_warnings_into_errors(version) -> None:  # type: ignore[no-untyped-def]
    @deprecated(since="1.2.0", remove_in="2.0.0")
    def old() -> None:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("error", AgenticRagDeprecationWarning)
        with pytest.raises(AgenticRagDeprecationWarning):
            old()


def test_experimental_marks_the_tier() -> None:
    @experimental
    def shiny() -> None:
        """Doc."""

    def plain() -> None:
        return None

    assert tier_of(shiny) == "experimental"
    assert tier_of(plain) == "stable"
    assert "experimental" in (shiny.__doc__ or "")
