# Providers: what is verified and what is not

"Verified" means a conformance suite or integration test **passed in this repository's environment** against the thing named in
*Against*. Nothing here claims behaviour of a paid service that was not called: no OpenAI, Anthropic, Cohere, … credentials were
available, so every hosted provider is **unverified against the real service** and says so. The status test
(`tests/docs/test_docs_match_code.py`) requires every `verified` row to cite a test that exists.

Select a component with `AGENTIC_RAG_VECTOR_STORE`, `AGENTIC_RAG_EMBEDDER`, `AGENTIC_RAG_CHAT_MODEL`, `AGENTIC_RAG_ANSWER_PIPELINE`
(`docs/framework.md` section 7 explains the registries; plug-ins add more). Every variable, with its default and bounds, is in the
generated [configuration reference](configuration.md); the ones each component reads:

| Component | Variables |
|---|---|
| `milvus` | `AGENTIC_RAG_MILVUS_URI` (a `.db` path is Milvus Lite), `_MILVUS_TOKEN`, `_MILVUS_COLLECTION`, `_HNSW_M`, `_HNSW_EF_CONSTRUCTION`, `_SEARCH_EF`, `_CONSISTENCY_LEVEL`, `_STORE_TIMEOUT_SECONDS` |
| `chroma` | exactly one of `AGENTIC_RAG_CHROMA_PATH`, `_CHROMA_URL`; `_CHROMA_COLLECTION` (no client timeout exists for a remote server) |
| `qdrant` | `AGENTIC_RAG_QDRANT_LOCATION` (`:memory:`, a directory, or `http(s)://host:port`), `_QDRANT_API_KEY`, `_QDRANT_COLLECTION`, `_STORE_TIMEOUT_SECONDS` |
| `openai` (embedder, chat) | `AGENTIC_RAG_OPENAI_API_KEY`, `_OPENAI_BASE_URL`, `_OPENAI_TIMEOUT_SECONDS`, `_EMBEDDING_MODEL`, `_EMBEDDING_DIMENSION`, `_OPENAI_CHAT_MODEL` |
| `crewai` (chat) | `AGENTIC_RAG_CREWAI_LLM_MODEL` (CrewAI's provider-prefixed name), `_CREWAI_LLM_API_KEY`, `_CREWAI_LLM_BASE_URL`, `_CREWAI_PROVIDER_TIMEOUT_SECONDS` |
| `crewai` (embedder) | `AGENTIC_RAG_CREWAI_EMBEDDER_PROVIDER`, `_CREWAI_EMBEDDER_MODEL`, `_CREWAI_EMBEDDER_API_KEY`, `_CREWAI_EMBEDDER_BASE_URL`, `_CREWAI_EMBEDDER_OPTIONS` (JSON), `_EMBEDDING_DIMENSION` |
| `crewai` (pipeline) | `AGENTIC_RAG_CREWAI_MAX_ITER`, `_CREWAI_MAX_SECONDS` (below the request deadline), `_CREWAI_MAX_CONCURRENT`, `_CREWAI_MAX_CONTEXT_CHARS`, `_CREWAI_SEARCH_BUDGET` |

## Vector stores (`vector_store`)

| Name | Extra | Status | Against | Evidence |
|---|---|---|---|---|
| `milvus` (HNSW, cosine) | `engine` | verified | Milvus Lite (embedded) | `tests/conformance/test_stores.py::TestMilvusStore`, `tests/integration/test_stores_through_container.py` |
| `milvus` | `engine` | unverified | Milvus standalone/Zilliz: the image could not be pulled here; the CI job starts it | CI `tests` job (not run here) |
| `chroma` (local directory) | `chroma` | verified | Chroma persistent client | `tests/conformance/test_stores.py::TestChromaStore` |
| `chroma` (server URL) | `chroma` | unverified | only "unreachable server gives `VECTOR_STORE_UNAVAILABLE`" was exercised | `TestChromaStore::test_unavailable_backend_raises_a_typed_error` |
| `qdrant` (local path / memory) | `qdrant` | verified | Qdrant local mode | `tests/conformance/test_stores.py::TestQdrantStore` |
| `qdrant` (server URL) | `qdrant` | unverified | only the unreachable-server error was exercised | `TestQdrantStore::test_unavailable_backend_raises_a_typed_error` |
| Weaviate, MongoDB Atlas, Couchbase, SingleStore, Snowflake, Databricks (CrewAI tools exist for these) | — | **not implemented** | no adapter; none of those services were available | — |

Chroma and Qdrant use their own client libraries rather than CrewAI's RAG clients, because CrewAI's clients embed query *text*
themselves while this system's `VectorSearcher` takes vectors computed by the configured embedder (ADR-0012).

## Embedders (`embedder`)

| Name | Extra | Status | Against | Evidence |
|---|---|---|---|---|
| `openai` | `engine` | unverified against OpenAI; verified against a local stub and a mock transport | stub HTTP server speaking the documented JSON | `tests/conformance/test_providers_over_http.py::TestOpenAIEmbedderOverHttp`, `tests/conformance/test_openai.py` |
| `crewai` + provider `openai` | `crewai` | unverified against OpenAI; verified through CrewAI against the same stub | stub | `tests/conformance/test_providers_over_http.py::TestCrewAIEmbedder` |
| `crewai` + any of: `azure`, `amazon-bedrock`, `cohere`, `google-generativeai`, `google-vertex`, `huggingface`, `instructor`, `jina`, `ollama`, `onnx`, `openclip`, `roboflow`, `sentence-transformer`, `text2vec`, `voyageai`, `watsonx`, `custom` | `crewai` (+ the provider's own SDK) | **unverified** | not run: no credentials, no local models, no network downloads | — |

The vector size is declared (`AGENTIC_RAG_EMBEDDING_DIMENSION`) and checked on every call; the model id (`provider:model:size`) is stored per
row and a different one is refused on write (`INDEX_INCOMPATIBLE`). CrewAI's embedding providers do not accept a retry limit, so each
keeps its provider SDK's default retries (one more retry layer than the framework's "at most once").

## Chat models (`chat_model`)

| Name | Extra | Status | Against | Evidence |
|---|---|---|---|---|
| `openai` | `engine` | unverified against OpenAI; verified against stub/mock | stub | `TestOpenAIChatOverHttp`, `tests/conformance/test_openai.py` |
| `crewai` + `openai/...` | `crewai` | unverified against OpenAI; verified through CrewAI against the stub, including 429/401/500 mapping and option forwarding | stub | `tests/conformance/test_providers_over_http.py::TestCrewAIChatModel` |
| `crewai` + `anthropic/...`, `azure/...`, `gemini/...`, `bedrock/...`, `ollama/...`, OpenAI-compatible servers (OpenRouter, DeepSeek, vLLM, Cerebras, DashScope), anything LiteLLM routes | `crewai` (+ provider SDK) | **unverified** | not run | — |

## Answer pipelines (`answer_pipeline`)

| Name | Status | Evidence |
|---|---|---|
| `direct`: retrieve, one grounded JSON generation call | verified (contract, service tests, end-to-end) | `TestDirectPipeline`, `tests/unit/test_service.py` |
| `crewai`: a real four-agent CrewAI crew (planner, retriever with a read-only search tool, writer, verifier), sequential, bounded iterations/time/searches/context, fail-fast on provider and search errors, no per-run files | verified with a scripted model (contract; tool use; escaping; deadline; no retry multiplication; no starvation under many slow crews; nothing written to disk; bounded prompts; a failing search fails the request) and end to end through CrewAI's own LLM and embedder against the stub | `TestCrewPipeline`, `tests/unit/test_crew_pipeline.py`, `tests/unit/test_crew_hardening.py`, `tests/integration/test_stores_through_container.py::test_crewai_providers_and_pipeline_through_the_container` |

Which pipeline should be the default is an evaluation question (ADR-0005): the harness can compare them, but **no run with a real model
exists**, so the default stays `direct` and is marked "unmeasured" in `docs/defaults.toml`. The crew costs about five model calls per
question against one for `direct`.

## CrewAI capabilities reviewed and not implemented (and why)

| CrewAI offers | Decision |
|---|---|
| Knowledge sources (string, text, PDF, CSV, Excel, JSON, Docling) and `Knowledge` storage | Not used: this service already ingests, chunks, embeds and stores documents behind its own ports; a second ingestion path would duplicate and diverge. Docling (layout-aware parsing and hierarchical chunking) is the one worth adding as a `parser`/`chunker` adapter; not done. |
| Memory (`Memory`, scopes, LanceDB/Qdrant-edge storage) | Not used, on purpose: memory is per-caller conversational state, and this service has no users or tenants (ADR-0006), so shared memory would leak one caller's questions into another's answers. The crew runs with `memory=False`. |
| crewai-tools loaders and search tools (web pages, GitHub, YouTube, MySQL/Postgres, scrapers, search APIs) | Not enabled: each fetches from arbitrary hosts or databases (SSRF, data exfiltration, OWASP LLM06). The crew's only tool is read-only retrieval over the request's own filter. |
| Hierarchical process, planning, manager LLM | Not used: the four-step flow is fixed and bounded; an LLM-chosen flow adds cost and variance with no measurement showing a benefit. |
| Flows, A2A, MCP, skills | Out of scope for the RAG parts. |
| Telemetry | Disabled (`CREWAI_DISABLE_TELEMETRY`, `tests/unit/test_telemetry_off.py`) before CrewAI is imported, unless the operator set the variable themselves. CrewAI still prints provider error text to its own console/log output on failures: that text stays in server logs and is never put in API responses. |
| Task-output persistence (every kickoff writes the question, retrieved text and answers to a SQLite file for replay) | Replaced by a no-op (`tests/unit/test_crew_hardening.py::test_a_request_writes_no_files`): a service that never replays should not keep plaintext copies of documents. |
