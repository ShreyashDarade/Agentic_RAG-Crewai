#!/usr/bin/env python3
"""Snapshots of the fixed surfaces (framework section 3). They change only by an explicit command.

    scripts/snapshots.py check                     compare the code with the checked-in snapshots
    scripts/snapshots.py regenerate                rewrite snapshots (error codes: append-only)
    scripts/snapshots.py regenerate --allow-breaking   also allow removing/changing error codes

Snapshots: docs/openapi.json (wire contract), docs/public_api.txt (public SDK surface),
docs/error_codes.json (error taxonomy), docs/configuration.md (every setting, generated).
"""

from __future__ import annotations

import argparse
import difflib
import importlib
import inspect
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

#: Modules whose ``__all__`` is the public surface, with their stability tier.
PUBLIC_MODULES: dict[str, str] = {
    "agentic_rag": "stable",
    "agentic_rag.errors": "stable",
    "agentic_rag.contracts": "stable",
    "agentic_rag.ports": "stable",
    "agentic_rag.registry": "stable",
    "agentic_rag.testing": "stable",
    "agentic_rag.client": "stable",
    "agentic_rag.models": "stable",
    "agentic_rag.extend": "stable",
}


def openapi_text() -> str:
    from agentic_rag.api import ApiConfig, create_app

    app = create_app(config=ApiConfig(allow_unauthenticated=True))
    return json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n"


def _describe(obj: Any) -> str:
    if inspect.isclass(obj):
        bases = ", ".join(b.__name__ for b in obj.__bases__ if b is not object)
        lines = [f"class {obj.__name__}({bases})"]
        if hasattr(obj, "model_fields"):
            for name, field in sorted(obj.model_fields.items()):
                lines.append(f"    field {name}: {field.annotation}{' (required)' if field.is_required() else ''}")
        elif hasattr(obj, "__dataclass_fields__"):
            for name, field in obj.__dataclass_fields__.items():
                lines.append(f"    field {name}: {field.type}")
        for name, member in sorted(vars(obj).items()):
            if name.startswith("_") and name != "__init__":
                continue
            if inspect.isfunction(member):
                lines.append(f"    def {name}{inspect.signature(member)}")
            elif isinstance(member, classmethod | staticmethod):  # Client.http and Client.embedded are classmethods
                lines.append(f"    {type(member).__name__} {name}{inspect.signature(member.__func__)}")
            elif isinstance(member, property):
                lines.append(f"    property {name}")
            elif isinstance(member, (str, int, float, bool)) and name in {
                "code",
                "http_status",
                "public_message",
            }:
                lines.append(f"    {name} = {member!r}")
        return "\n".join(lines)
    if callable(obj):
        return f"def {obj.__name__}{inspect.signature(obj)}"
    return f"value {obj!r}" if isinstance(obj, (str, int, float, bool)) else f"object {type(obj).__name__}"


def _scrub(text: str) -> str:
    """Remove memory addresses so the snapshot is deterministic."""
    return re.sub(r" at 0x[0-9a-fA-F]+", "", text)


def public_api_text() -> str:
    out = ["# Public API snapshot. Regenerate with: scripts/snapshots.py regenerate", ""]
    for module_name, tier in PUBLIC_MODULES.items():
        module = importlib.import_module(module_name)
        out.append(f"## {module_name}  [{tier}]")
        for name in sorted(module.__all__):
            obj = getattr(module, name)
            tier_note = "  [experimental]" if getattr(obj, "__agentic_rag_tier__", "stable") == "experimental" else ""
            out.append(f"{name}{tier_note}: " + _scrub(_describe(obj)).replace("\n", "\n    "))
        out.append("")
    return "\n".join(out)


def configuration_text() -> str:
    """Every setting, generated from the code, so the reference cannot lag behind it."""
    from agentic_rag.config import ENV_PREFIX, FILE_SUFFIX, Settings

    lines = [
        "# Configuration reference",
        "",
        "Generated from `agentic_rag.config.Settings` by `scripts/snapshots.py regenerate`; do not edit. Every variable has the prefix",
        f"`{ENV_PREFIX}`. A secret may instead be given as the path of a file in the variable with the suffix `{FILE_SUFFIX}`",
        "(for mounted secrets), never both. An unknown variable in this namespace stops the process before it binds a port; the error names",
        "the variable and never its value. Meaning and operating advice: `docs/operations.md`, `docs/security.md`, `docs/providers.md`.",
        "",
        "| Variable | Default | Allowed | Secret |",
        "|---|---|---|---|",
    ]
    for name, field in Settings.model_fields.items():
        default = field.default if field.default_factory is None else field.default_factory()  # type: ignore[call-arg]
        secret = "SecretStr" in str(field.annotation)
        bounds = []
        for meta in field.metadata:
            for attr in ("ge", "gt", "le", "lt", "min_length", "max_length", "pattern"):
                if getattr(meta, attr, None) is not None:
                    bounds.append(f"{attr} {getattr(meta, attr)}")
        allowed = ", ".join(bounds)
        annotation = str(field.annotation).replace("typing.", "").replace("<class '", "").replace("'>", "")
        if "Literal" in annotation:
            allowed = annotation[annotation.index("[") + 1 : annotation.rindex("]")].replace("'", "")
        shown = "(unset)" if default in (None, [], "") else f"`{default}`"
        lines.append(
            f"| `{ENV_PREFIX}{name.upper()}` | {shown} | {allowed.replace('|', '/')} | {'yes' if secret else ''} |"
        )
    return "\n".join(lines) + "\n"


def error_codes() -> list[dict[str, Any]]:
    from agentic_rag.errors import catalog

    return [
        {
            "code": code,
            "class": cls.__name__,
            "http_status": cls.http_status,
            "public_message": cls.public_message,
        }
        for code, cls in sorted(catalog().items())
    ]


def error_codes_text() -> str:
    return json.dumps(error_codes(), indent=2) + "\n"


SNAPSHOTS = {
    "openapi.json": openapi_text,
    "public_api.txt": public_api_text,
    "error_codes.json": error_codes_text,
    "configuration.md": configuration_text,
}


def diff(name: str) -> str:
    path = DOCS / name
    expected = path.read_text() if path.exists() else ""
    actual = SNAPSHOTS[name]()
    return "".join(
        difflib.unified_diff(
            expected.splitlines(True), actual.splitlines(True), f"{name} (checked in)", f"{name} (code)"
        )
    )


def breaking_error_changes() -> list[str]:
    """Error codes may be added, never removed or altered."""
    path = DOCS / "error_codes.json"
    if not path.exists():
        return []
    old = {e["code"]: e for e in json.loads(path.read_text())}
    new = {e["code"]: e for e in error_codes()}
    problems = [f"removed: {c}" for c in old if c not in new]
    problems += [f"changed: {c}" for c in old if c in new and old[c] != new[c]]
    return problems


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["check", "regenerate"])
    parser.add_argument("--allow-breaking", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "check":
        bad = [name for name in SNAPSHOTS if diff(name)]
        for name in bad:
            sys.stdout.write(diff(name))
        return 1 if bad else 0
    problems = breaking_error_changes()
    if problems and not args.allow_breaking:
        print(
            "refusing to regenerate: error codes are append-only:\n  " + "\n  ".join(problems),
            file=sys.stderr,
        )
        return 2
    DOCS.mkdir(exist_ok=True)
    for name, make in SNAPSHOTS.items():
        (DOCS / name).write_text(make())
        print(f"wrote docs/{name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
