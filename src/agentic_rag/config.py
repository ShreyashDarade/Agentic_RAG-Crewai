"""One typed settings object, validated at start-up (framework section 11).

Secrets are ``SecretStr`` and come from the environment. Invalid *or unknown* ``AGENTIC_RAG_*`` values
are errors, reported without echoing any value.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, NoDecode, PydanticBaseSettingsSource, SettingsConfigDict

from agentic_rag.application import Limits, RetrievalConfig
from agentic_rag.errors import ConfigurationError

__all__ = ["ENV_PREFIX", "Settings", "load_settings"]

ENV_PREFIX = "AGENTIC_RAG_"


def _split(value: object) -> object:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return value


CsvList = Annotated[list[str], NoDecode]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix=ENV_PREFIX, extra="forbid", frozen=True)

    # -- security --------------------------------------------------------------------------------
    api_keys: Annotated[list[SecretStr], NoDecode] = Field(default_factory=list, max_length=2)
    cors_allow_origins: CsvList = Field(default_factory=list)
    allow_unauthenticated: bool = False  # local development only; the server logs a warning

    # -- components (names in the registries) ------------------------------------------------------
    vector_store: str = "milvus"
    lexical_index: str = "bm25"
    embedder: str = "openai"
    chat_model: str = "openai"
    reranker: str | None = None
    chunker: str = "recursive"
    answer_pipeline: str = "direct"
    parsers: CsvList = Field(default_factory=lambda: ["text", "html", "pdf", "docx"])
    plugins: CsvList = Field(default_factory=list)

    # -- Milvus ----------------------------------------------------------------------------------
    milvus_uri: str | None = None
    milvus_token: SecretStr | None = None
    milvus_collection: str = Field(default="documents", pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,254}$")
    hnsw_m: int = Field(default=32, ge=4, le=64)
    hnsw_ef_construction: int = Field(default=200, ge=8, le=1024)
    search_ef: int = Field(default=64, ge=1, le=4096)
    consistency_level: Literal["Strong", "Session", "Bounded", "Eventually"] = "Strong"

    # -- other vector stores (components: vector_store = chroma | qdrant) -----------------------------------------
    chroma_path: str | None = None
    chroma_url: str | None = None
    chroma_collection: str = Field(default="documents", pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{1,510}[A-Za-z0-9]$")
    qdrant_location: str | None = None  # ":memory:", a directory, or http(s)://host:port
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = Field(default="documents", pattern=r"^[A-Za-z0-9_-]{1,255}$")

    # -- CrewAI providers (components: chat_model/embedder = crewai, answer_pipeline = crewai) ----------------------
    crewai_llm_model: str | None = (
        None  # provider-prefixed, e.g. "openai/gpt-4o-mini", "anthropic/claude-...", "ollama/llama3"
    )
    crewai_llm_api_key: SecretStr | None = None
    crewai_llm_base_url: str | None = None
    crewai_embedder_provider: str | None = None  # a CrewAI embedding provider name, e.g. "openai", "cohere", "ollama"
    crewai_embedder_model: str | None = None
    crewai_embedder_api_key: SecretStr | None = None
    crewai_embedder_base_url: str | None = None
    crewai_embedder_options: str = "{}"  # extra provider options as JSON (non-secret)
    crewai_max_iter: int = Field(default=3, ge=1, le=10)
    crewai_max_seconds: int = Field(default=40, ge=1, le=300)
    crewai_max_concurrent: int = Field(default=4, ge=1, le=64)

    # -- OpenAI ----------------------------------------------------------------------------------
    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None
    openai_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = Field(default=1536, ge=8, le=4096)
    openai_chat_model: str = "gpt-4o-mini"

    # -- chunking ----------------------------------------------------------------------------------
    chunk_max_chars: int = Field(default=1500, ge=100, le=20000)
    chunk_overlap_chars: int = Field(default=200, ge=0, le=5000)
    pdf_max_pages: int = Field(default=500, ge=1, le=10000)

    # -- limits (bounds on everything attacker- or accident-controlled) ----------------------------
    max_upload_bytes: int = Field(default=25 * 1024 * 1024, ge=1024, le=512 * 1024 * 1024)
    max_question_chars: int = Field(default=4000, ge=1, le=4000)
    max_top_k: int = Field(default=50, ge=1, le=50)
    max_page_size: int = Field(default=100, ge=1, le=200)
    max_chunks_per_document: int = Field(default=5000, ge=1, le=100_000)
    embed_batch_size: int = Field(default=64, ge=1, le=2048)
    max_concurrent_embed_batches: int = Field(default=4, ge=1, le=64)
    max_concurrent_ingests: int = Field(default=4, ge=1, le=64)
    request_deadline_seconds: float = Field(default=55.0, gt=0, le=600)
    health_check_timeout_seconds: float = Field(default=2.0, gt=0, le=30)
    max_inflight_requests: int = Field(default=64, ge=1, le=10_000)
    shutdown_drain_seconds: float = Field(default=3.0, ge=0, le=60)
    log_json: bool = True

    # -- retrieval defaults (unmeasured until the evaluation records evidence) ---------------------
    default_top_k: int = Field(default=8, ge=1, le=50)
    candidate_pool: int = Field(default=40, ge=1, le=500)
    rrf_k: int = Field(default=60, ge=1, le=1000)
    use_lexical: bool = True
    use_reranker: bool = False
    pipeline_max_tokens: int = Field(default=700, ge=16, le=8192)

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @field_validator("api_keys", mode="before")
    @classmethod
    def _keys(cls, value: object) -> object:
        return _split(value)

    @field_validator("api_keys")
    @classmethod
    def _key_strength(cls, keys: list[SecretStr]) -> list[SecretStr]:
        if any(len(k.get_secret_value()) < 16 for k in keys):
            raise ValueError("each API key must be at least 16 characters")
        return keys

    @field_validator("cors_allow_origins", "parsers", "plugins", mode="before")
    @classmethod
    def _csv(cls, value: object) -> object:
        return _split(value)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Values are read by load_settings() from an explicit mapping, never implicitly from os.environ.
        return (init_settings,)

    def to_limits(self) -> Limits:
        return Limits(
            max_upload_bytes=self.max_upload_bytes,
            max_question_chars=self.max_question_chars,
            max_top_k=self.max_top_k,
            max_page_size=self.max_page_size,
            max_chunks_per_document=self.max_chunks_per_document,
            embed_batch_size=self.embed_batch_size,
            max_concurrent_embed_batches=self.max_concurrent_embed_batches,
            max_concurrent_ingests=self.max_concurrent_ingests,
            request_deadline_seconds=self.request_deadline_seconds,
            health_check_timeout_seconds=self.health_check_timeout_seconds,
        )

    def to_retrieval(self) -> RetrievalConfig:
        return RetrievalConfig(
            default_top_k=self.default_top_k,
            candidate_pool=self.candidate_pool,
            rrf_k=self.rrf_k,
            use_lexical=self.use_lexical,
            use_reranker=self.use_reranker,
            pipeline_max_tokens=self.pipeline_max_tokens,
        )


def load_settings(environ: Mapping[str, str] | None = None) -> Settings:
    """Build Settings from ``environ`` (default ``os.environ``); failures are typed and value-free."""
    env = os.environ if environ is None else environ
    names = {ENV_PREFIX + name.upper(): name for name in Settings.model_fields}
    unknown = sorted(k for k in env if k.upper().startswith(ENV_PREFIX) and k.upper() not in names)
    if unknown:
        raise ConfigurationError("unknown configuration variables: " + ", ".join(unknown), details={"unknown": unknown})
    values: dict[str, object] = {names[k.upper()]: v for k, v in env.items() if k.upper() in names}
    try:
        settings = Settings(**values)  # type: ignore[arg-type]
    except ValidationError as exc:
        problems = [
            {"variable": ENV_PREFIX + ".".join(str(p) for p in e["loc"]).upper(), "problem": e["msg"]}
            for e in exc.errors(include_input=False, include_url=False)
        ]
        raise ConfigurationError(
            "invalid configuration: " + "; ".join(f"{p['variable']}: {p['problem']}" for p in problems),
            details={"errors": problems},
        ) from None
    if settings.chunk_overlap_chars >= settings.chunk_max_chars // 2:
        raise ConfigurationError("AGENTIC_RAG_CHUNK_OVERLAP_CHARS must be less than half of the chunk size")
    return settings
