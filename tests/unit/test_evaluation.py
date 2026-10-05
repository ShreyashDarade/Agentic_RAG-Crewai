"""Metrics against hand-computed values; statistics against their defining properties; the gate against known regressions."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest

from agentic_rag.evaluation import (
    bootstrap_ci,
    check,
    compare_paired,
    evaluate_query,
    hit_at_k,
    holm,
    load_dataset,
    mrr_at_k,
    ndcg_at_k,
    paired_randomization_p,
    precision_at_k,
    recall_at_k,
    run_retrieval,
)

QRELS = {"a": 1, "b": 2, "c": 0, "z": 1}  # relevant: a, b (grade 2), z


def test_metrics_match_hand_computation() -> None:
    ranking = ["x", "b", "a", "y", "c"]
    dcg = 2 / math.log2(3) + 1 / math.log2(4)  # b at rank 2 (gain 2), a at rank 3 (gain 1)
    ideal = 2 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)  # b, then a and z
    assert ndcg_at_k(ranking, QRELS, 5) == pytest.approx(dcg / ideal)
    assert recall_at_k(ranking, QRELS, 5) == pytest.approx(2 / 3)
    assert precision_at_k(ranking, QRELS, 5) == pytest.approx(2 / 5)
    assert mrr_at_k(ranking, QRELS, 5) == pytest.approx(1 / 2)
    assert hit_at_k(ranking, QRELS, 1) == 0.0
    assert hit_at_k(ranking, QRELS, 2) == 1.0
    assert mrr_at_k(ranking, QRELS, 1) == 0.0


def test_perfect_empty_and_degenerate_cases() -> None:
    assert ndcg_at_k(["b", "a", "z"], QRELS, 10) == pytest.approx(1.0)
    assert ndcg_at_k([], QRELS, 10) == 0.0
    assert ndcg_at_k(["a"], {"c": 0}, 10) == 0.0  # nothing relevant judged: defined as 0, not a division error
    assert recall_at_k(["a"], {}, 10) == 0.0
    assert evaluate_query(["a"], QRELS, 3).keys() == {"ndcg@3", "recall@3", "precision@3", "mrr@3", "hit@3"}
    with pytest.raises(ValueError, match="twice"):
        evaluate_query(["a", "a"], QRELS, 3)


def test_unjudged_documents_count_as_not_relevant() -> None:
    assert precision_at_k(["unjudged1", "unjudged2"], QRELS, 2) == 0.0


def test_bootstrap_interval_is_seeded_contains_the_mean_and_narrows_with_more_data() -> None:
    rng = random.Random(1)
    small = [rng.random() for _ in range(30)]
    large = [rng.random() for _ in range(600)]
    lo, hi = bootstrap_ci(small, seed=3)
    assert (lo, hi) == bootstrap_ci(small, seed=3)
    assert lo <= sum(small) / len(small) <= hi
    lo2, hi2 = bootstrap_ci(large, seed=3)
    assert (hi2 - lo2) < (hi - lo)


def test_randomization_test_detects_a_real_shift_and_not_noise() -> None:
    rng = random.Random(2)
    noise = [rng.gauss(0, 1) for _ in range(200)]
    shifted = [d + 0.6 for d in noise]
    assert paired_randomization_p(noise, seed=1) > 0.05
    assert paired_randomization_p(shifted, seed=1) < 0.001
    assert paired_randomization_p([0.0] * 20, seed=1) == 1.0


def test_paired_comparison_reports_delta_interval_effect_size_and_refuses_unequal_query_sets() -> None:
    a = {f"q{i}": 0.5 for i in range(50)}
    b = {f"q{i}": 0.5 + (0.1 if i % 2 else 0.2) for i in range(50)}
    cmp = compare_paired("ndcg@10", a, b)
    assert cmp.mean_delta == pytest.approx(0.15)
    assert cmp.ci95[0] > 0
    assert cmp.p_value < 0.001
    assert cmp.effect_size == pytest.approx(0.15 / math.sqrt(sum((d - 0.15) ** 2 for d in [0.1, 0.2] * 25) / 49))
    with pytest.raises(ValueError, match="different queries"):
        compare_paired("m", a, {"other": 1.0})


def test_holm_controls_the_family() -> None:
    assert holm({"a": 0.001, "b": 0.04, "c": 0.5}) == {"a": True, "b": False, "c": False}  # 0.04 > 0.05/2
    assert holm({"a": 0.01, "b": 0.02}) == {"a": True, "b": True}


def _tiny_dataset(tmp_path: Path) -> Path:
    (tmp_path / "qrels").mkdir()
    docs = {f"d{i}": f"topic{i} common words about subject{i % 5}" for i in range(40)}
    (tmp_path / "corpus.jsonl").write_text(
        "\n".join(json.dumps({"_id": k, "title": "", "text": v}) for k, v in docs.items())
    )
    queries = {f"q{i}": f"topic{i} subject{i % 5}" for i in range(30)}
    (tmp_path / "queries.jsonl").write_text("\n".join(json.dumps({"_id": k, "text": v}) for k, v in queries.items()))
    (tmp_path / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\n" + "".join(f"q{i}\td{i}\t1\n" for i in range(30))
    )
    return tmp_path


def test_dataset_loader_and_runner_and_gate(tmp_path: Path) -> None:
    ds = load_dataset(_tiny_dataset(tmp_path))
    assert len(ds.queries) == 30
    assert len(ds.corpus) == 40

    def good(query: str, k: int) -> list[str]:
        topic = query.split()[0].removeprefix("topic")
        return [f"d{topic}"] + [f"d{i}" for i in range(40) if f"d{i}" != f"d{topic}"][: k - 1]

    def bad(query: str, k: int) -> list[str]:
        return [f"d{i}" for i in range(39, 39 - k, -1)]

    base = run_retrieval(ds, good, system="good", k=10)
    worse = run_retrieval(ds, bad, system="bad", k=10)
    assert base["summary"]["ndcg@10"]["mean"] == pytest.approx(1.0)
    same = check(base, base, "ndcg@10")
    assert same.passed
    regression = check(base, worse, "ndcg@10")
    assert not regression.passed
    assert any("significant regression" in r for r in regression.reasons)
    assert not check(base, base, "ndcg@10", floor=1.5).passed  # an absolute floor can fail on its own
    assert check(worse, base, "ndcg@10").passed  # an improvement passes


def test_dataset_loader_rejects_judgments_for_missing_documents(tmp_path: Path) -> None:
    root = _tiny_dataset(tmp_path)
    (root / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\tmissing\t1\n")
    with pytest.raises(ValueError, match="missing from the corpus"):
        load_dataset(root)


def test_dataset_loader_keeps_the_first_judgment_when_there_is_no_header(tmp_path: Path) -> None:
    root = _tiny_dataset(tmp_path)
    (root / "qrels" / "test.tsv").write_text("q1\td1\t1\nq2\td2\t1\n")  # no "query-id corpus-id score" row
    assert sorted(load_dataset(root).queries) == ["q1", "q2"]  # the first query used to vanish silently


def test_dataset_loader_tolerates_blank_lines_and_reports_malformed_ones(tmp_path: Path) -> None:
    root = _tiny_dataset(tmp_path)
    (root / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\n\n")
    assert sorted(load_dataset(root).queries) == ["q1"]
    (root / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\n")
    with pytest.raises(ValueError, match="qrels line 2"):
        load_dataset(root)
    (root / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\tmaybe\n")
    with pytest.raises(ValueError, match="qrels line 2"):
        load_dataset(root)


def test_queries_with_no_relevant_document_are_excluded_as_in_trec_eval(tmp_path: Path) -> None:
    root = _tiny_dataset(tmp_path)
    (root / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td2\t0\nq3\td3\t0\nq3\td4\t1\n")
    ds = load_dataset(root)
    assert sorted(ds.queries) == [
        "q1",
        "q3",
    ]  # q2 has only a grade-0 judgment: it cannot be answered, so it cannot count
    run = run_retrieval(ds, lambda q, k: [f"d{q.split()[0].removeprefix('topic')}"], system="oracle-ish", k=10)
    assert run["n_queries"] == 2
