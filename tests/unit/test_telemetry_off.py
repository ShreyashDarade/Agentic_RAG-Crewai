"""ADR-0005/0010: CrewAI's anonymous usage telemetry is switched off before CrewAI can read the setting."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _env_after(statement: str) -> str:
    env = {k: v for k, v in os.environ.items() if k != "CREWAI_DISABLE_TELEMETRY"}
    env["PYTHONPATH"] = str(ROOT / "src")
    code = f"import os; assert 'CREWAI_DISABLE_TELEMETRY' not in os.environ; {statement}; print(os.environ.get('CREWAI_DISABLE_TELEMETRY'))"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120, check=False
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_importing_the_container_disables_crewai_telemetry() -> None:
    pytest.importorskip("pymilvus")
    assert _env_after("import agentic_rag.container") == "true"


def test_importing_the_crewai_adapters_directly_disables_it_too() -> None:
    pytest.importorskip("crewai")
    for module in ("pipeline", "chat", "embedder"):
        assert _env_after(f"import agentic_rag.adapters.crewai.{module}") == "true", module


def test_an_operator_who_set_the_variable_keeps_their_value() -> None:
    pytest.importorskip("crewai")
    env = {**os.environ, "CREWAI_DISABLE_TELEMETRY": "1", "PYTHONPATH": str(ROOT / "src")}
    code = "import os, agentic_rag.adapters.crewai; print(os.environ['CREWAI_DISABLE_TELEMETRY'])"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120, check=False
    )
    assert out.stdout.strip() == "1"
