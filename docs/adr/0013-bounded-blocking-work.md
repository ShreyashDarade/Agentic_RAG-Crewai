# ADR-0013: Blocking calls run on bounded pools that are released when the thread returns

Status: accepted · Date: 2026-10-05 · Origin: independent review round 1

## Context
Every store client, the lexical index, the parsers and the CrewAI crew are blocking code, run through `asyncio.to_thread`. That shares
one default executor (about eight threads on four CPUs) between all of them, and cancelling the awaiting task does not stop its thread.
Reproduced: a store that accepts the connection and never answers used up the executor; afterwards `asyncio.to_thread(lambda: 1)` did not
return, so parsing, BM25 search, health checks and every other store call stalled behind it. The same happened with many slow crews, which
also waited on the executor for the provider calls they needed (a circular wait). A semaphore around ingest was released on timeout while
its parser thread kept running.

## Decision
`agentic_rag.blocking.BlockingPool`: a small set of daemon threads per user of blocking code (each vector store, the lexical index, the
ingest parse/chunk work, the crew, the provider chat and embedding calls). A call counts as running until **its thread returns**, not until
the awaiting task is cancelled; when every worker and every queue slot is taken by calls that are not finishing, the next call raises the
caller's typed error at once (`VECTOR_STORE_UNAVAILABLE`, `OVERLOADED`) instead of queueing for a deadline. Pools that wait on each other are
different pools. Stores also pass a client timeout (Milvus, Qdrant); Chroma's client offers none, so the pool is its only bound. An
embedded engine that runs one call at a time (Milvus Lite) gets one worker (measured: eight threads contending for its lock gave a 45 %
worse p95 at 16 clients, `docs/benchmark.md`). `crewai_max_seconds` must be below the request deadline; every use case has the deadline.

## Consequences
A hung dependency strands at most its pool's workers and then fails fast; the rest of the process keeps working. A thread stuck in a call
is abandoned, not killed (it cannot be); daemon threads do not block interpreter exit. Cost: a few percent on warm search latency that was
measured and not fully explained. Parsers and the chunker can still run long on a pathological file: their number is bounded, not their time.

## Enforced by
`tests/unit/test_blocking.py`, `tests/integration/test_hung_store.py`, `tests/unit/test_service_hardening.py`,
`tests/unit/test_crew_hardening.py`; mutations M57, M58, M69, M81-M84.
