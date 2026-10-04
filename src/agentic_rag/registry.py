"""Name -> factory registries, one per extension point (framework section 7).

Behaviour is added by registering a component, never by editing a switch. A component can be
registered in code, through the entry-point group ``agentic_rag.<kind>``, or by a module named
in settings that exposes ``register(registries)``. Unknown names fail closed with a typed
error that lists the valid names.

Standard library only.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from importlib import metadata
from typing import Any, Generic, TypeVar

from agentic_rag.errors import ConfigurationError, UnknownComponent

__all__ = ["KINDS", "Registries", "Registry", "load_plugins"]

T = TypeVar("T")

#: Extension points. Each has a contract in ``agentic_rag.ports`` and a conformance suite.
KINDS: tuple[str, ...] = (
    "vector_store",
    "lexical_index",
    "embedder",
    "chat_model",
    "reranker",
    "parser",
    "chunker",
    "answer_pipeline",
)


class Registry(Generic[T]):
    """Maps a component name to a factory that builds it."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._factories: dict[str, Callable[..., T]] = {}

    def register(self, name: str, factory: Callable[..., T], *, replace: bool = False) -> None:
        if not name or not name.replace("_", "").replace("-", "").isalnum():
            raise ConfigurationError(f"invalid {self.kind} name {name!r}")
        if name in self._factories and not replace:
            raise ConfigurationError(f"{self.kind} {name!r} is already registered")
        self._factories[name] = factory

    def names(self) -> list[str]:
        return sorted(set(self._factories) | set(self._entry_point_names()))

    def get(self, name: str) -> Callable[..., T]:
        factory = self._factories.get(name)
        if factory is None:
            factory = self._load_entry_point(name)
        if factory is None:
            raise UnknownComponent(
                f"unknown {self.kind} {name!r}",
                details={"kind": self.kind, "name": name, "valid": self.names()},
            )
        return factory

    def create(self, name: str, *args: Any, **kwargs: Any) -> T:
        return self.get(name)(*args, **kwargs)

    def _group(self) -> str:
        return f"agentic_rag.{self.kind}"

    def _entry_point_names(self) -> list[str]:
        return [ep.name for ep in metadata.entry_points(group=self._group())]

    def _load_entry_point(self, name: str) -> Callable[..., T] | None:
        for ep in metadata.entry_points(group=self._group(), name=name):
            loaded = ep.load()
            if not callable(loaded):
                raise ConfigurationError(f"entry point {ep.value!r} is not callable")
            self._factories[name] = loaded
            return loaded  # type: ignore[no-any-return]
        return None


class Registries:
    """One :class:`Registry` per extension point."""

    def __init__(self) -> None:
        self._by_kind: dict[str, Registry[Any]] = {kind: Registry(kind) for kind in KINDS}

    def __getitem__(self, kind: str) -> Registry[Any]:
        try:
            return self._by_kind[kind]
        except KeyError:
            raise UnknownComponent(
                f"unknown extension point {kind!r}",
                details={"kind": kind, "valid": list(KINDS)},
            ) from None

    def __getattr__(self, kind: str) -> Registry[Any]:
        if kind.startswith("_"):
            raise AttributeError(kind)
        return self[kind]


def load_plugins(registries: Registries, modules: list[str]) -> None:
    """Import each named module and call its ``register(registries)``."""
    for name in modules:
        try:
            module = importlib.import_module(name)
        except ImportError as exc:
            raise ConfigurationError(f"plugin module {name!r} could not be imported") from exc
        register = getattr(module, "register", None)
        if not callable(register):
            raise ConfigurationError(f"plugin module {name!r} has no register(registries)")
        register(registries)
