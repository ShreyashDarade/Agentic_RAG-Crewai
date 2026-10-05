"""Measure retrieval quality: metrics, paired statistics, runs, comparison and a regression gate."""

from agentic_rag.evaluation.dataset import Dataset, load_dataset
from agentic_rag.evaluation.metrics import evaluate_query, hit_at_k, mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k
from agentic_rag.evaluation.runner import GateResult, check, compare_runs, load_run, run_retrieval, save_run
from agentic_rag.evaluation.stats import Comparison, bootstrap_ci, compare_paired, holm, paired_randomization_p

__all__ = [
    "Comparison",
    "Dataset",
    "GateResult",
    "bootstrap_ci",
    "check",
    "compare_paired",
    "compare_runs",
    "evaluate_query",
    "hit_at_k",
    "holm",
    "load_dataset",
    "load_run",
    "mrr_at_k",
    "ndcg_at_k",
    "paired_randomization_p",
    "precision_at_k",
    "recall_at_k",
    "run_retrieval",
    "save_run",
]
