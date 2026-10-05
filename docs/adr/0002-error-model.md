# ADR-0002: One error model; stable codes; problem+json on the wire

Status: proposed · Date: 2026-10-04

## Context
Today failures surface as `HTTPException(500, detail=str(e))` (raw dependency text to clients), 115 broad `except` sites, and
"success with a fallback answer" (baseline §4.4). Research: RFC 9457, AIP-193, Stripe/Anthropic/OpenAI SDK error hierarchies,
OWASP error handling (research §1 "Error body shape", "Stable machine codes", "Error taxonomy", "Safe public messages").

## Decision
`RagError` root; each subclass declares its own `code`, `http_status`, `public_message` (enforced in `__init_subclass__`).
Wire body is `application/problem+json` plus `code`, `request_id`, `details`. A catalog maps code → class so HTTP and embedded
transports raise the same exception; unknown codes from newer servers become `RagStatusError` retaining the code. Dependency
failures are wrapped with `from`; raw text is logged, never serialised. Codes are append-only.

## Consequences
Callers can program against failures; every adapter needs a failure mapping and a conformance case. The existing `success: false`
envelope on 200/500 responses is removed (breaking, but the service never worked: baseline §2).

## Enforced by
Catalog round-trip test; secret-leak injection test; AST test for swallowed broad excepts; `error_codes.json` snapshot (Steps 9, 11).
