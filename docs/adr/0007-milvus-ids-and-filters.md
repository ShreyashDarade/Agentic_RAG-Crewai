# ADR-0007: Content-addressed chunk ids; templated filters only

Status: accepted, superseded in part by ADR-0011 (filters) and amended by review round 1 (below) · Date: 2026-10-04

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

## Amendment: review round 1
* Ingest order is **write the body chunks → sweep stale chunks of the document → update the lexical index → write chunk 0 (the commit
  marker)**. Until chunk 0 lands the document is not in the catalog, so a retry (same ids) repeats the whole job instead of being told it
  already exists; a failed first ingest is discarded (best effort) and leftovers without a marker can be deleted.
* The document id is `doc_` + 32 hex characters of sha256(file extension, a NUL byte, the bytes): the same bytes parsed as another file
  type are another document. The chunk id includes the chunker version, which now includes its settings, and a document records the
  chunker and parser versions it was indexed with, so a re-upload after any of them changed is re-indexed.
* Filters are not `filter_params` templated (Milvus Lite lacks it); they are built from a closed alphabet (ADR-0011).
