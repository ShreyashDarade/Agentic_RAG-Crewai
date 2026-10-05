# Configuration reference

Generated from `agentic_rag.config.Settings` by `scripts/snapshots.py regenerate`; do not edit. Every variable has the prefix
`AGENTIC_RAG_`. A secret may instead be given as the path of a file in the variable with the suffix `_FILE`
(for mounted secrets), never both. An unknown variable in this namespace stops the process before it binds a port; the error names
the variable and never its value. Meaning and operating advice: `docs/operations.md`, `docs/security.md`, `docs/providers.md`.

| Variable | Default | Allowed | Secret |
|---|---|---|---|
| `AGENTIC_RAG_API_KEYS` | (unset) | max_length 2 | yes |
| `AGENTIC_RAG_CORS_ALLOW_ORIGINS` | (unset) |  |  |
| `AGENTIC_RAG_ALLOW_UNAUTHENTICATED` | `False` |  |  |
| `AGENTIC_RAG_VECTOR_STORE` | `milvus` |  |  |
| `AGENTIC_RAG_LEXICAL_INDEX` | `bm25` |  |  |
| `AGENTIC_RAG_EMBEDDER` | `openai` |  |  |
| `AGENTIC_RAG_CHAT_MODEL` | `openai` |  |  |
| `AGENTIC_RAG_RERANKER` | (unset) |  |  |
| `AGENTIC_RAG_CHUNKER` | `recursive` |  |  |
| `AGENTIC_RAG_ANSWER_PIPELINE` | `direct` |  |  |
| `AGENTIC_RAG_PARSERS` | `['text', 'html', 'pdf', 'docx']` |  |  |
| `AGENTIC_RAG_PLUGINS` | (unset) |  |  |
| `AGENTIC_RAG_MILVUS_URI` | (unset) |  |  |
| `AGENTIC_RAG_MILVUS_TOKEN` | (unset) |  | yes |
| `AGENTIC_RAG_MILVUS_COLLECTION` | `documents` | pattern ^[A-Za-z_][A-Za-z0-9_]{0,254}$ |  |
| `AGENTIC_RAG_HNSW_M` | `32` | ge 4, le 64 |  |
| `AGENTIC_RAG_HNSW_EF_CONSTRUCTION` | `200` | ge 8, le 1024 |  |
| `AGENTIC_RAG_SEARCH_EF` | `64` | ge 1, le 4096 |  |
| `AGENTIC_RAG_CONSISTENCY_LEVEL` | `Strong` | Strong, Session, Bounded, Eventually |  |
| `AGENTIC_RAG_CHROMA_PATH` | (unset) |  |  |
| `AGENTIC_RAG_CHROMA_URL` | (unset) |  |  |
| `AGENTIC_RAG_CHROMA_COLLECTION` | `documents` | pattern ^[A-Za-z0-9][A-Za-z0-9._-]{1,510}[A-Za-z0-9]$ |  |
| `AGENTIC_RAG_QDRANT_LOCATION` | (unset) |  |  |
| `AGENTIC_RAG_QDRANT_API_KEY` | (unset) |  | yes |
| `AGENTIC_RAG_QDRANT_COLLECTION` | `documents` | pattern ^[A-Za-z0-9_-]{1,255}$ |  |
| `AGENTIC_RAG_CREWAI_LLM_MODEL` | (unset) |  |  |
| `AGENTIC_RAG_CREWAI_LLM_API_KEY` | (unset) |  | yes |
| `AGENTIC_RAG_CREWAI_LLM_BASE_URL` | (unset) |  |  |
| `AGENTIC_RAG_CREWAI_EMBEDDER_PROVIDER` | (unset) |  |  |
| `AGENTIC_RAG_CREWAI_EMBEDDER_MODEL` | (unset) |  |  |
| `AGENTIC_RAG_CREWAI_EMBEDDER_API_KEY` | (unset) |  | yes |
| `AGENTIC_RAG_CREWAI_EMBEDDER_BASE_URL` | (unset) |  |  |
| `AGENTIC_RAG_CREWAI_EMBEDDER_OPTIONS` | `{}` |  |  |
| `AGENTIC_RAG_CREWAI_MAX_ITER` | `3` | ge 1, le 10 |  |
| `AGENTIC_RAG_CREWAI_MAX_SECONDS` | `40` | ge 1, le 300 |  |
| `AGENTIC_RAG_CREWAI_MAX_CONCURRENT` | `4` | ge 1, le 64 |  |
| `AGENTIC_RAG_CREWAI_PROVIDER_TIMEOUT_SECONDS` | `30.0` | gt 0, le 300 |  |
| `AGENTIC_RAG_CREWAI_MAX_CONTEXT_CHARS` | `12000` | ge 1000, le 200000 |  |
| `AGENTIC_RAG_CREWAI_SEARCH_BUDGET` | `4` | ge 1, le 20 |  |
| `AGENTIC_RAG_OPENAI_API_KEY` | (unset) |  | yes |
| `AGENTIC_RAG_OPENAI_BASE_URL` | (unset) |  |  |
| `AGENTIC_RAG_OPENAI_TIMEOUT_SECONDS` | `30.0` | gt 0, le 300 |  |
| `AGENTIC_RAG_EMBEDDING_MODEL` | `text-embedding-3-small` |  |  |
| `AGENTIC_RAG_EMBEDDING_DIMENSION` | `1536` | ge 8, le 4096 |  |
| `AGENTIC_RAG_OPENAI_CHAT_MODEL` | `gpt-4o-mini` |  |  |
| `AGENTIC_RAG_CHUNK_MAX_CHARS` | `1500` | ge 100, le 20000 |  |
| `AGENTIC_RAG_CHUNK_OVERLAP_CHARS` | `200` | ge 0, le 5000 |  |
| `AGENTIC_RAG_PDF_MAX_PAGES` | `500` | ge 1, le 10000 |  |
| `AGENTIC_RAG_MAX_UPLOAD_BYTES` | `26214400` | ge 1024, le 536870912 |  |
| `AGENTIC_RAG_MAX_QUESTION_CHARS` | `4000` | ge 1, le 4000 |  |
| `AGENTIC_RAG_MAX_TOP_K` | `50` | ge 1, le 50 |  |
| `AGENTIC_RAG_MAX_PAGE_SIZE` | `100` | ge 1, le 200 |  |
| `AGENTIC_RAG_MAX_CHUNKS_PER_DOCUMENT` | `5000` | ge 1, le 100000 |  |
| `AGENTIC_RAG_EMBED_BATCH_SIZE` | `64` | ge 1, le 2048 |  |
| `AGENTIC_RAG_MAX_CONCURRENT_EMBED_BATCHES` | `4` | ge 1, le 64 |  |
| `AGENTIC_RAG_MAX_CONCURRENT_INGESTS` | `4` | ge 1, le 64 |  |
| `AGENTIC_RAG_REQUEST_DEADLINE_SECONDS` | `55.0` | gt 0, le 600 |  |
| `AGENTIC_RAG_HEALTH_CHECK_TIMEOUT_SECONDS` | `2.0` | gt 0, le 30 |  |
| `AGENTIC_RAG_STORE_TIMEOUT_SECONDS` | `20.0` | gt 0, le 120 |  |
| `AGENTIC_RAG_MAX_INFLIGHT_REQUESTS` | `64` | ge 1, le 10000 |  |
| `AGENTIC_RAG_SHUTDOWN_DRAIN_SECONDS` | `3.0` | ge 0, le 60 |  |
| `AGENTIC_RAG_LOG_JSON` | `True` |  |  |
| `AGENTIC_RAG_DEFAULT_TOP_K` | `8` | ge 1, le 50 |  |
| `AGENTIC_RAG_CANDIDATE_POOL` | `40` | ge 1, le 500 |  |
| `AGENTIC_RAG_RRF_K` | `60` | ge 1, le 1000 |  |
| `AGENTIC_RAG_USE_LEXICAL` | `True` |  |  |
| `AGENTIC_RAG_USE_RERANKER` | `False` |  |  |
| `AGENTIC_RAG_PIPELINE_MAX_TOKENS` | `700` | ge 16, le 8192 |  |
| `AGENTIC_RAG_LOG_LEVEL` | `INFO` | DEBUG, INFO, WARNING, ERROR |  |
