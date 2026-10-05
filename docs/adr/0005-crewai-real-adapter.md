# ADR-0005: CrewAI as a real, optional pipeline adapter; default decided by measurement

Status: accepted, amended by review round 1 (below) · Date: 2026-10-04 · Owner decision: "make it real"

## Context
Baseline §2: CrewAI is a hard dependency but no `Crew`, `Task` or `kickoff` exists; the "agents" are plain classes calling the LLM.
The owner chose to make the multi-agent crew real rather than remove or demote it.

## Decision
`AnswerPipeline` is a port. Two adapters are registered: `crewai` (planner → retriever → writer → verifier as CrewAI agents/tasks in a
sequential `Crew`) and `direct` (retrieve, then one grounded generation call). The crew's tools are read-only retrieval bound to the
request's filter; iteration, token and wall-clock budgets are mandatory. Both pass the same `AnswerPipeline` conformance suite
(grounded citations, empty-retrieval refusal, typed errors for bad output and a failing model, retrieved text that cannot close its own
delimiter). The wall-clock bound is enforced by the service's request deadline for both and by the crew's own limit, tested in
`tests/unit/test_crew_pipeline.py` and `tests/unit/test_crew_hardening.py`; there is no shared "injection corpus", only the escaping check. The default is whichever the evaluation shows is not
worse in quality and acceptable in latency/cost; the other stays available and the result is written to `docs/evaluation.md`.

## Consequences
CrewAI is an extra (`agentic-rag[crewai]`); its large install no longer burdens other users. Honest risk: a crew adds LLM calls,
latency and nondeterminism; the owner's decision is honoured by shipping it, and measurement decides what is default.

## Enforced by
Conformance suite over both adapters; `defaults.toml` evidence test; confinement of `crewai` to `adapters.crewai` (Steps 8, 18).

## Amendment: review round 1
The crew now runs on its own bounded thread pool, with provider calls on a separate pool, so no pool waits on itself and a hung provider
cannot starve the process (ADR-0013). Its search tool is bounded by the same context budget as the first prompt and by a search budget;
a failing search fails the request rather than letting the crew answer from less; only chunks the model was shown may be cited; an outer
cancellation stops further provider calls; CrewAI's per-run SQLite store of task outputs is replaced by a no-op. The crew's limit
(`crewai_max_seconds`) must be below the request deadline.
