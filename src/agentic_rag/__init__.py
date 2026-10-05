"""Agentic RAG: retrieval-augmented question answering.

``Client`` and ``AsyncClient`` need only ``httpx`` and ``pydantic``. Engine names (``Settings``,
``load_settings``, ``build_container``) exist only when ``agentic-rag[engine]`` is installed; on a thin
install they are absent from ``__all__`` and accessing them raises an ``ImportError`` naming the extra.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

from agentic_rag._compat import AgenticRagDeprecationWarning, AgenticRagFutureWarning
from agentic_rag._version import __version__
from agentic_rag.client import AsyncClient, Client

__all__ = [
    "AgenticRagDeprecationWarning",
    "AgenticRagFutureWarning",
    "AsyncClient",
    "Client",
    "__version__",
]

_ENGINE_NAMES: dict[str, str] = {
    "Settings": "agentic_rag.config",
    "load_settings": "agentic_rag.config",
    "build_container": "agentic_rag.container",
}


def _engine_installed() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("pydantic_settings", "pymilvus", "openai"))


if _engine_installed():
    __all__ += sorted(_ENGINE_NAMES)


def __getattr__(name: str) -> Any:
    module = _ENGINE_NAMES.get(name)
    if module is None:
        raise AttributeError(f"module 'agentic_rag' has no attribute {name!r}")
    try:
        return getattr(importlib.import_module(module), name)
    except ImportError as exc:
        raise ImportError(f"agentic_rag.{name} needs the engine: pip install 'agentic-rag[engine]'") from exc
