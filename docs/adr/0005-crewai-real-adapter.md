# ADR-0005: CrewAI as a real, optional pipeline adapter; default decided by measurement

Status: proposed · Date: 2026-10-04 · Owner decision: "make it real"

## Context
Baseline §2: CrewAI is a hard dependency but no `Crew`, `Task` or `kickoff` exists; the "agents" are plain classes calling the LLM.
The owner chose to make the multi-agent crew real rather than remove or demote it.

## Decision
`AnswerPipeline` is a port. Two adapters are registered: `crewai` (Supervisor → Retriever → Generator → Feedback as CrewAI
agents/tasks in a `Crew`) and `direct` (the same four stages as plain calls). The crew's tools are read-only retrieval bound to the
request's filter; iteration, token and wall-clock budgets are mandatory. Both pass the same `AnswerPipeline` conformance suite
(grounded citations, empty-retrieval refusal, deadline, injection corpus). The default is whichever the evaluation shows is not
worse in quality and acceptable in latency/cost; the other stays available and the result is written to `docs/evaluation.md`.

## Consequences
CrewAI is an extra (`agentic-rag[crewai]`); its large install no longer burdens other users. Honest risk: a crew adds LLM calls,
latency and nondeterminism; the owner's decision is honoured by shipping it, and measurement decides what is default.

## Enforced by
Conformance suite over both adapters; `defaults.toml` evidence test; confinement of `crewai` to `adapters.crewai` (Steps 8, 18).
