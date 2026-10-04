# ADR-0004: SDK = one facade over a narrow Backend; HTTP and embedded backends

Status: proposed · Date: 2026-10-04

## Context
There is no SDK today. Requirement: call the system remotely or run it in-process with identical behaviour. Research: httpx's
transport injection and `ASGITransport` (research §2 A2); anyio blocking portal and CPython's "cannot be called from a running
loop" (§2 A7, A8); spec-correct SSE (§2 A6).

## Decision
Public methods are written once on a facade; a `Backend` protocol has an `HttpBackend` (httpx) and an `EmbeddedBackend`
(calls `application.Service` directly, passing results and errors through the same contract models and error catalog). Async is the
implementation; the blocking client uses one background loop; sync-in-running-loop raises `UsageError`.

## Consequences / rejected alternative
Rejected: embedded transport implemented as `ASGITransport(app)` (research §2 A2 recommends it because parity is then structural).
It would force HTTP serialisation and ASGI lifespan management on in-process users. The compensating control is the parity suite
(one test body over both backends, plus a two-backends-one-engine comparison) with a mutation proof. Revisit if parity bugs recur.

## Enforced by
`tests/sdk/test_parity.py`; thin-import test; `mypy --strict` (Steps 12, 13).
