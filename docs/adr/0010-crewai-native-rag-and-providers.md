# ADR-0010: Use CrewAI's RAG capabilities through ports; multiple model/embedding/vector providers

Status: proposed · Date: 2026-10-04 · Supersedes part of ADR-0005 and the "OpenAI + Milvus only" default (framework §1) · Owner request

## Context
The owner asked to use CrewAI's RAG offerings as fully as possible, with multiple model providers across embedding, chunking and
vector databases. Inventory (research §4): 18 embedding providers; native LLM routing for OpenAI/Anthropic/Azure/Gemini/Bedrock and
OpenAI-compatible providers, LiteLLM fallback; vector clients for Chroma and Qdrant only (no Milvus); pluggable knowledge-storage and
memory-storage protocols; knowledge sources and chunkers; crewai-tools loaders and search tools.

## Decision
1. Every CrewAI capability that is a retrieval, embedding, chunking, parsing, memory or storage feature is registered **behind the
   existing ports** as an adapter inside the single package `adapters.crewai` (so `crewai` stays confined). Nothing outside `adapters.crewai` imports it.
2. Providers are selected by settings, not code: `chat_model`, `embedder`, `vector_store`, `chunker`, `parser`, `tool` registries.
   Shipped sets: LLM = OpenAI direct + any CrewAI/LiteLLM provider string; embedder = OpenAI direct + CrewAI's providers;
   vector store = Milvus (pymilvus) + CrewAI Chroma + CrewAI Qdrant.
3. Milvus is connected to CrewAI through its pluggable seams (`BaseKnowledgeStorage`, memory `StorageBackend`), because CrewAI's
   RAG config only supports Chroma and Qdrant.
4. "100%" is bounded by the disposition table in research §4: out-of-scope items are listed, and tools with network or write
   capability are off by default.
5. Each provider has a status in `docs/providers.md`: `verified` (conformance suite passed here), `unverified` (no credentials/service),
   with the command that produced the status. We never describe an unverified provider as supported.
6. Defaults are chosen by measurement (framework §11): with many providers the *default* of each registry is whatever the
   evaluation supports, and mixing embedding models in one collection is a typed error (dimension and model id stored in collection metadata).

## Consequences
Large surface; most of it unverifiable without paid credentials or extra services. Conformance suites per port keep it honest.
Install size grows only for users who pick extras (`crewai`, `docling`, per-provider extras). More dependency-audit surface.
Memory and knowledge written by CrewAI flows must still use content-addressed ids (ADR-0007) where the protocol lets us.

## Enforced by
Confinement contract (`crewai`, `chromadb`, `qdrant_client`, `docling`, `litellm` only in `adapters.crewai`); conformance suites
over every registered adapter; `providers.md` status test (a provider marked `verified` must have a recorded run); telemetry-off test.
