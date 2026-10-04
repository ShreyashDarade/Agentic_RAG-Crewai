"""Bounds on everything attacker- or accident-controlled (framework section 11)."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Limits", "RetrievalConfig"]


@dataclass(frozen=True, slots=True)
class Limits:
    max_upload_bytes: int = 25 * 1024 * 1024
    max_question_chars: int = 4000
    max_top_k: int = 50
    max_page_size: int = 100
    max_chunks_per_document: int = 5000
    embed_batch_size: int = 64
    max_concurrent_embed_batches: int = 4
    max_concurrent_ingests: int = 4
    request_deadline_seconds: float = 55.0
    health_check_timeout_seconds: float = 2.0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if getattr(self, name) <= 0:
                raise ValueError(f"limit {name} must be positive")


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """Retrieval defaults. Each is **unmeasured** until the evaluation (Step 18) records evidence."""

    default_top_k: int = 8
    candidate_pool: int = 40
    rrf_k: int = 60
    use_lexical: bool = True
    use_reranker: bool = False
    pipeline_max_tokens: int = 700
