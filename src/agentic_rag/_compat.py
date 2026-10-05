"""Deprecation and stability-tier helpers (framework section 10, ADR-0008). Standard library only.

``deprecated(since, remove_in, alternative, escalate_in)`` validates its metadata when the decorator is
applied (so a bad policy fails at import, not in production):

* versions are ``MAJOR.MINOR.PATCH``;
* ``since`` is not in the future, and removal happens only in a later MAJOR than ``since``;
* ``escalate_in`` (optional) lies in ``[since, remove_in)``.

Until ``escalate_in`` the warning is an :class:`AgenticRagDeprecationWarning` (a ``DeprecationWarning``: shown
to developers, ignored by default for libraries); from ``escalate_in`` on it is an
:class:`AgenticRagFutureWarning` (a ``FutureWarning``: visible by default). ``stacklevel`` points at the caller.

Not machine-enforced: "two minor releases and six months of notice" (version numbers cannot prove it).
"""

from __future__ import annotations

import functools
import inspect
import re
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

from agentic_rag._version import __version__

__all__ = [
    "AgenticRagDeprecationWarning",
    "AgenticRagFutureWarning",
    "deprecated",
    "experimental",
    "tier_of",
]

F = TypeVar("F", bound=Callable[..., Any])
_VERSION = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
TIER_ATTRIBUTE = "__agentic_rag_tier__"


class AgenticRagDeprecationWarning(DeprecationWarning):
    """A deprecated name was used. In your tests: ``-W error::agentic_rag.AgenticRagDeprecationWarning``."""


class AgenticRagFutureWarning(FutureWarning):
    """The deprecated name is in its last minor release before removal; shown by default."""


def _parse(version: str, what: str) -> tuple[int, int, int]:
    match = _VERSION.match(version)
    if match is None:
        raise ValueError(f"{what} must look like MAJOR.MINOR.PATCH, got {version!r}")
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch)


def _current() -> tuple[int, int, int]:
    return _parse(__version__, "the package version")


def deprecated(
    *, since: str, remove_in: str, alternative: str | None = None, escalate_in: str | None = None
) -> Callable[[F], F]:
    since_v = _parse(since, "since")
    remove_v = _parse(remove_in, "remove_in")
    escalate_v = _parse(escalate_in, "escalate_in") if escalate_in is not None else None
    if since_v > _current():
        raise ValueError(f"since={since} is newer than the current version {__version__}")
    if remove_v[0] <= since_v[0]:
        raise ValueError(f"remove_in={remove_in} must be a later MAJOR version than since={since}")
    if escalate_v is not None and not since_v <= escalate_v < remove_v:
        raise ValueError(f"escalate_in={escalate_in} must lie in [{since}, {remove_in})")

    def message(name: str) -> str:
        text = f"{name} is deprecated since {since} and will be removed in {remove_in}"
        return text + (f"; use {alternative} instead." if alternative else ".")

    def warn(name: str) -> None:
        category: type[Warning] = AgenticRagDeprecationWarning
        if escalate_v is not None and _current() >= escalate_v:
            category = AgenticRagFutureWarning
        warnings.warn(message(name), category, stacklevel=3)  # warn() <- wrapper <- the caller

    def decorate(target: F) -> F:
        name = getattr(target, "__qualname__", repr(target))
        note = f"\n\n.. deprecated:: {since}\n   {message(name)}"
        if inspect.isclass(target):
            original_init = target.__init__

            @functools.wraps(original_init)
            def init(self: Any, *args: Any, **kwargs: Any) -> None:
                warn(name)
                original_init(self, *args, **kwargs)

            target.__init__ = init  # type: ignore[method-assign]
            target.__doc__ = (target.__doc__ or "") + note
            return target
        if inspect.iscoroutinefunction(target):

            @functools.wraps(target)
            async def awrapper(*args: Any, **kwargs: Any) -> Any:
                warn(name)
                return await target(*args, **kwargs)

            awrapper.__doc__ = (target.__doc__ or "") + note
            return awrapper  # type: ignore[return-value]

        @functools.wraps(target)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warn(name)
            return target(*args, **kwargs)

        wrapper.__doc__ = (target.__doc__ or "") + note
        return wrapper  # type: ignore[return-value]

    return decorate


def experimental(target: F) -> F:
    """Mark a public name as experimental: it may change in a minor release (framework section 4)."""
    setattr(target, TIER_ATTRIBUTE, "experimental")
    target.__doc__ = (target.__doc__ or "") + "\n\n.. warning:: experimental; may change in a minor release."
    return target


def tier_of(obj: object) -> str:
    return str(getattr(obj, TIER_ATTRIBUTE, "stable"))
