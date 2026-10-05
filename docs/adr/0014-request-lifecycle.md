# ADR-0014: Authenticate before any work; deadline on every use case; drain before the listener closes

Status: accepted · Date: 2026-10-05 · Origin: independent review round 1

## Context
Reproduced findings: (1) a malformed body returned 422 instead of 401, a 5 MB upload was spooled before the 401, and four anonymous
requests that announced a body and never sent it made authenticated requests get 503 for as long as they stayed open; (2) load shedding
looked at `scope["path"]`, which uvicorn's `--root-path` prefixes, so shedding silently turned off; (3) `get`, `list` and `delete` had no
deadline although the documents said every request has one; (4) the documented drain could not happen under uvicorn, which closes its
listener before the application's lifespan shutdown runs.

## Decision
Outside in: request context → **authentication** → load shedding → body limit → CORS → routes. The path is judged below the root path,
whichever way the server reports it. Every service method that touches a dependency runs under the request deadline. `agentic-rag serve`
runs a `uvicorn.Server` subclass that flips `/readyz` to 503 in `shutdown()` **before** calling uvicorn's own, waits the drain period, and
only then lets uvicorn stop; it turns uvicorn's own log handlers and access log off so that stdout is JSON only. The image runs `serve`
and its health check uses `/healthz`.

## Consequences
Anonymous traffic costs the server one header comparison. `uvicorn --factory agentic_rag.server:create` still works but does not drain and
prints uvicorn's plain-text logs; the docs say to use `serve`. An authenticated slow-body client can still hold a slot (documented).

## Enforced by
`tests/api/test_hardening.py`, `tests/api/test_live_server.py` (a real process and a real SIGTERM), `tests/unit/test_service_hardening.py`;
mutations M45, M46, M47, M67, M68.
