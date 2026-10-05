# ADR-0009: Retry matrix, deadlines, idempotency keys, load shedding

Status: proposed · Date: 2026-10-04

## Context
Current client-side behaviour is none; server-side the LLM client retries every exception including bad requests and auth errors
(baseline §3). Research §2 B1–B8: RFC 9110 idempotency, Retry-After, AWS/Google SRE guidance, Stripe idempotency keys, OpenAI/Anthropic
retry matrices.

## Decision
Retry matrix, deadline ordering and idempotency semantics as in framework §8. Mutations accept `Idempotency-Key`; the SDK generates one
per logical call and reuses it across retries; the server stores (key, body hash) → response for 24 h and does not cache 5xx for ingest
(writes are upserts by content hash, ADR-0007). Overload sheds with `503` + `Retry-After` and code `OVERLOADED`.

## Consequences
Needs a small shared store for idempotency records (Redis or a Milvus side collection: decided at Step 16 by measurement). Stricter than
OpenAI/Anthropic (they retry all ≥ 500 and 409); chosen because a 500 on ingest can follow a partial write.

## Enforced by
SDK retry-matrix tests against a fault-injecting server; deadline-ordering test; load-shedding test (Steps 12, 16).
