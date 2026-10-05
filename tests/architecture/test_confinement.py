"""G2/G3: every third-party import is classified and confined; every top-level package has a layer."""

from __future__ import annotations

import ast
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "agentic_rag"
STDLIB = set(sys.stdlib_module_names)


def _module_name(path: Path) -> str:
    parts = path.relative_to(ROOT / "src").with_suffix("").parts
    return ".".join(parts).removesuffix(".__init__")


def _third_party_imports() -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for path in sorted(SRC.rglob("*.py")):
        module = _module_name(path)
        for node in ast.walk(ast.parse(path.read_text())):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for name in names:
                top = name.split(".")[0]
                if top not in STDLIB and top != "agentic_rag":
                    found.append((module, top))
    return found


def _allowed() -> dict[str, list[str]]:
    return tomllib.loads((Path(__file__).parent / "confinement.toml").read_text())["allowed"]


def _matches(module: str, pattern: str) -> bool:
    if pattern.endswith(".*"):
        base = pattern[:-2]
        return module == base or module.startswith(base + ".")
    return module == pattern


def test_every_third_party_library_is_classified() -> None:
    allowed = _allowed()
    unclassified = sorted({lib for _, lib in _third_party_imports() if lib not in allowed})
    assert not unclassified, f"add these to tests/architecture/confinement.toml (or remove the import): {unclassified}"


def test_every_library_is_imported_only_by_its_own_package() -> None:
    allowed = _allowed()
    violations = [
        f"{module} imports {lib}"
        for module, lib in _third_party_imports()
        if lib in allowed and not any(_matches(module, p) for p in allowed[lib])
    ]
    assert not violations, "\n".join(violations)


def test_the_allow_list_has_no_stale_entries() -> None:
    used = {(lib, module) for module, lib in _third_party_imports()}
    for lib, patterns in _allowed().items():
        for pattern in patterns:
            assert any(_matches(m, pattern) for name, m in used if name == lib), f"{lib}: {pattern} is never used"


def test_every_top_level_package_is_assigned_to_a_layer() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    contract = next(c for c in config["tool"]["importlinter"]["contracts"] if c["type"] == "layers")
    assigned = {name.strip() for layer in contract["layers"] for name in layer.split("|")}
    top_level = {
        p.stem if p.is_file() else p.name
        for p in SRC.iterdir()
        if not p.name.startswith("__") and p.suffix in {"", ".py"}
    }
    assert top_level <= assigned, f"unclassified: {sorted(top_level - assigned)}"
    assert contract["exhaustive"] is True
