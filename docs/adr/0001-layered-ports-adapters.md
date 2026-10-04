# ADR-0001: Layered architecture, ports and adapters, one composition root

Status: proposed (Checkpoint 1) · Date: 2026-10-04

## Context
The current code constructs concrete dependencies everywhere (`CrewManager._initialize_llm`, `IngestionPipeline._initialize_vector_store`,
route modules with global singletons), so nothing can be swapped or tested without live services and the layers cannot be
checked (baseline §3, §4). Research: dependency rule, hexagonal ports, composition root, import-linter layers/forbidden/independence
(research §1 "Layering", "Ports and adapters", "Composition root", "Enforcing layers").

## Decision
Layers and directions as in framework §2: foundation (`errors`, `registry`, `contracts`) ← `ports` ← `application` ← transports
(`api`, `cli`, `sdk`); adapters depend on ports only; `container` is the single composition root and the only importer of
`adapters.*`. Ports are `typing.Protocol`s with plain value types; one third-party library per adapter package.
Dependencies are explicit constructor parameters; no module-level singletons.

## Consequences
Swapping an adapter is a settings change. Fakes ship in `agentic_rag.testing` and back unit tests. Adding a transport needs no
business logic. A rewrite of imports is required across the whole tree. Rejected: a DI-container library (manual wiring is enough
at this size); per-call ports (research §1: Cockburn warns against many fine-grained ports).

## Enforced by
import-linter `layers` (`exhaustive = true`), `forbidden` (with external packages), `independence` between adapters;
`tests/architecture/test_confinement.py` (Step 11).
