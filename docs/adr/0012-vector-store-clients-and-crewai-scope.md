# ADR-0012: Chroma and Qdrant through their own clients; the CrewAI scope that was actually built

Status: accepted (implemented) · Date: 2026-10-05 · Supersedes parts of ADR-0010

## Context
ADR-0010 planned Chroma and Qdrant "through CrewAI's RAG clients", Milvus plugged into CrewAI's `BaseKnowledgeStorage` and memory
`StorageBackend`, and "100 %" of CrewAI's RAG features behind ports. Reading CrewAI 1.15's source and building it showed:
its RAG clients take query *text* and embed it themselves, so they cannot sit behind a `VectorSearcher` that receives vectors from the
configured embedder; memory is per-caller state that this tenant-less service must not share; knowledge sources duplicate ingestion.

## Decision
1. `chroma` and `qdrant` vector stores use `chromadb` and `qdrant-client` directly, each confined to its own adapter package and run
   through the same conformance suite as Milvus. Filters are typed objects.
2. CrewAI is used where it fits the ports: `crewai` **answer pipeline** (a real crew), `crewai` **chat model** (`crewai.LLM`, any provider
   it routes to) and `crewai` **embedder** (its embedding providers). Each is confined to `adapters.crewai`.
3. CrewAI knowledge, memory, loaders and search tools are **not** used; `docs/providers.md` lists each with the reason.
4. Every provider has a status in `docs/providers.md`; "verified" requires a passing test in this repository, and hosted providers are
   "unverified" because no credentials were available.

## Consequences
The multi-provider surface is real for embedding, chat and storage; the "all CrewAI features" ambition is narrower than ADR-0010 hoped,
and the owner should know that. A deployment wanting CrewAI memory or knowledge needs a tenant model first (ADR-0006 non-goal).

## Enforced by
Import-linter confinement of `crewai`, `chromadb`, `qdrant_client`; conformance suites; `tests/docs/test_docs_match_code.py`.
