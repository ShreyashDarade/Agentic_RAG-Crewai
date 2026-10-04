"""G4 (routes stay thin) and G12 (no silent fallbacks) beyond what the linters already enforce."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "agentic_rag"
MAX_ROUTE_LINES = 15


def test_route_handlers_are_thin() -> None:
    tree = ast.parse((SRC / "api" / "routes.py").read_text())
    routes = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef)
        and any(
            isinstance(d, ast.Call)
            and isinstance(d.func, ast.Attribute)
            and d.func.attr in {"get", "post", "delete", "put"}
            for d in n.decorator_list
        )
    ]
    assert len(routes) >= 8
    for fn in routes:
        body_lines = (fn.end_lineno or fn.lineno) - fn.body[0].lineno + 1
        assert body_lines <= MAX_ROUTE_LINES, f"{fn.name} has {body_lines} body lines"
        assert not any(isinstance(n, ast.Try | ast.For | ast.While) for n in ast.walk(fn)), (
            f"{fn.name} has control flow"
        )


def _is_broad(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    names = [n.id for n in ast.walk(handler.type) if isinstance(n, ast.Name)]
    return any(n in {"Exception", "BaseException"} for n in names)


def test_broad_excepts_wrap_or_log_and_never_swallow() -> None:
    """A broad `except Exception` must re-raise (usually `raise Typed from exc`) or log with a traceback.

    Returning a default or passing silently is the defect this rule exists to prevent (no silent fallbacks).
    """
    broad = 0
    for path in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ExceptHandler) and _is_broad(node):
                broad += 1
                raises = any(isinstance(n, ast.Raise) for n in ast.walk(node))
                logs = any(
                    isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "exception"
                    for n in ast.walk(node)
                )
                assert raises or logs, f"{path.name}:{node.lineno} swallows an exception"
    assert broad >= 3  # the rule is being exercised, not vacuous
