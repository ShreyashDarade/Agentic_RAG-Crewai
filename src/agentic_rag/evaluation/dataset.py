"""BEIR-format datasets (``corpus.jsonl``, ``queries.jsonl``, ``qrels/<split>.tsv``); use it for a golden set too."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Dataset", "load_dataset"]


@dataclass(frozen=True)
class Dataset:
    name: str
    corpus: dict[str, str]  # doc id -> "title text"
    queries: dict[str, str]  # query id -> text (only queries that have judgments)
    qrels: dict[str, dict[str, int]]  # query id -> doc id -> grade


def _jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(f"{path.name} line {number}: not valid JSON") from None
            if not isinstance(row, dict):
                raise ValueError(f"{path.name} line {number}: expected an object")
            rows.append(row)
    return rows


def _read_qrels(path: Path) -> dict[str, dict[str, int]]:
    """``query-id<TAB>corpus-id<TAB>score``. A header row is recognised by its non-numeric score, not assumed."""
    qrels: dict[str, dict[str, int]] = {}
    seen_row = False
    with path.open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 3:
                raise ValueError(f"qrels line {number}: expected three tab-separated fields")
            try:
                grade = int(fields[2])
            except ValueError:
                if not seen_row:
                    seen_row = True  # the header row
                    continue
                raise ValueError(f"qrels line {number}: the score is not an integer") from None
            seen_row = True
            qrels.setdefault(fields[0], {})[fields[1]] = grade
    return qrels


def load_dataset(path: Path, *, split: str = "test", limit: int | None = None) -> Dataset:
    """Queries without any relevant document (grade > 0) are left out: they cannot be answered, and trec_eval skips
    them too. Counting them would pull every system's mean down by the same arbitrary amount."""
    corpus = {
        str(row["_id"]): f"{row.get('title', '')} {row.get('text', '')}".strip()
        for row in _jsonl(path / "corpus.jsonl")
    }
    all_queries = {str(row["_id"]): str(row["text"]) for row in _jsonl(path / "queries.jsonl")}
    qrels = {q: j for q, j in _read_qrels(path / "qrels" / f"{split}.tsv").items() if any(g > 0 for g in j.values())}
    unknown = {d for judged in qrels.values() for d in judged} - set(corpus)
    if unknown:
        raise ValueError(f"{len(unknown)} judged documents are missing from the corpus")
    ids = sorted(q for q in qrels if q in all_queries)
    if limit is not None:
        ids = ids[:limit]
    return Dataset(
        name=path.name,
        corpus=corpus,
        queries={q: all_queries[q] for q in ids},
        qrels={q: qrels[q] for q in ids},
    )
