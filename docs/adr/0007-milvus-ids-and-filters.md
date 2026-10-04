# ADR-0007: Content-addressed chunk ids; templated filters only

Status: proposed · Date: 2026-10-04

## Context
Current ids are `<file stem>_<chunk id>` (stem comes from an upload filename) and filters are f-strings
(`content_type == "{x}"`, `id in {list}`): ingestion is not idempotent and values can break out of expressions (baseline §3, §4.3).
Research §3: Milvus upsert needs `auto_id=False` for idempotence; filter templating; Spring AI CVE-2026-41705 as a real instance;
OWASP injection prevention.

## Decision
Chunk primary key = hash of (document content hash, chunk index, chunker version), `auto_id=False`. Re-ingest = delete by document id then
upsert, so shrinking documents leave no orphans. One module builds every filter using `filter_params` templating and allow-listed field
names; ids are regex-validated. Index parameters are settings and are recorded in collection metadata; changing them is a reindex.

## Consequences
Replays are safe (supports `Idempotency-Key`, ADR-0009). Atomicity across documents is not guaranteed by Milvus; ingest is ordered
write → sweep → commit per document. Whether `filter_params` alone prevents injection is unverified in the docs; the validation layer stays.

## Enforced by
Conformance tests for `VectorStore`; injection-payload tests; forbidden-import contract (Steps 8, 15, 16).
