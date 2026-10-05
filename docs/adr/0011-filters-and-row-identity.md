# ADR-0011: Closed-alphabet filter values; embedding identity stored per row

Status: accepted (implemented) · Date: 2026-10-04 · Supersedes parts of ADR-0007

## Context
ADR-0007 chose parameter templating (`filter_params`) for every Milvus filter and recorded index parameters in collection metadata.
Both were tested against Milvus Lite (the real engine this environment can run) and neither works there: `filter_params` is rejected
by its expression parser (`unexpected character '{'`), and `describe_collection` returns no description, so a metadata blob is lost.

## Decision
1. Filters are rendered by one module from a typed `ChunkFilter`. Field names are fixed; every value must match `[A-Za-z0-9._-]{1,128}`
   (no quote, backslash, bracket, comma or whitespace) before it is placed in a double-quoted literal, so a value cannot leave its
   string. Anything else raises `ValidationFailed`. This is one code path for Lite and standalone.
2. Each row stores the embedding model id (`embedding_model`); the vector dimension is read from the collection schema. `ensure_ready`
   compares both and raises `IndexIncompatible`. Index parameters are settings; changing them is a documented reindex, not detected
   automatically.

## Consequences
Injection safety rests on the alphabet check plus the conformance payload tests rather than on server-side binding; if templating becomes
available everywhere it can be added behind the same function. A larger identifier alphabet (for example spaces in external ids) is
impossible without revisiting this decision.

## Enforced by
`tests/conformance` hostile-filter checks (Milvus and fake), mutation M22, import-linter confinement of `pymilvus`.
