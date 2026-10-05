"""G14: the evaluation regression gate runs on a committed golden set, so a retrieval regression fails a build.

The golden set is small and synthetic (24 documents, 24 queries): it catches broken tokenising, scoring or ranking, not
subtle quality loss on real data. Real-data quality is measured by ``agentic-rag eval run`` on a BEIR dataset (docs/evaluation.md).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_rag.adapters.lexical_bm25 import Bm25Index
from agentic_rag.cli import main

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests" / "eval" / "golden"
BASELINE = ROOT / "docs" / "eval" / "golden-bm25-baseline.json"


def _run(tmp_path: Path, name: str) -> Path:
    out = tmp_path / name
    assert main(["eval", "run", "--dataset", str(GOLDEN), "--out", str(out)]) == 0
    return out


def test_the_current_retrieval_passes_the_gate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    candidate = _run(tmp_path, "candidate.json")
    assert main(["eval", "check", "--baseline", str(BASELINE), "--candidate", str(candidate)]) == 0
    assert '"passed": true' in capsys.readouterr().out


def test_a_retrieval_regression_fails_the_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real = Bm25Index.search

    def worst_first(self: Bm25Index, query: str, *, top_k: int, filter: object = None) -> list:  # type: ignore[type-arg]
        return list(reversed(real(self, query, top_k=top_k, filter=filter)))  # type: ignore[arg-type]

    monkeypatch.setattr(Bm25Index, "search", worst_first)
    candidate = _run(tmp_path, "regressed.json")
    assert main(["eval", "check", "--baseline", str(BASELINE), "--candidate", str(candidate)]) == 1


def test_the_golden_set_is_not_trivially_perfect() -> None:
    import json

    run = json.loads(BASELINE.read_text())
    assert run["n_queries"] >= 24
    assert 0.9 < run["summary"]["ndcg@10"]["mean"] < 1.0  # at least one query is not answered at rank 1
