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


def load_dataset(path: Path, *, split: str = "test", limit: int | None = None) -> Dataset:
    corpus: dict[str, str] = {}
    with (path / "corpus.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            corpus[str(row["_id"])] = f"{row.get('title', '')} {row.get('text', '')}".strip()
    all_queries: dict[str, str] = {}
    with (path / "queries.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            all_queries[str(row["_id"])] = row["text"]
    qrels: dict[str, dict[str, int]] = {}
    with (path / "qrels" / f"{split}.tsv").open(encoding="utf-8") as fh:
        next(fh)  # header: query-id corpus-id score
        for line in fh:
            qid, did, score = line.rstrip("\n").split("\t")
            qrels.setdefault(qid, {})[did] = int(score)
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
