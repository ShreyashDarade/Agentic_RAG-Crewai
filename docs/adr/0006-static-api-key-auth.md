# ADR-0006: Static bearer API key; destructive surface off by default

Status: accepted, amended by review round 1 (below) · Date: 2026-10-04 · Owner decision: built-in static API key

## Context
Baseline §4.3: no auth, CORS `*` with credentials, unauthenticated `DELETE /ingest/reset`, arbitrary server-path ingestion.
Research: OWASP API Top 10 (API2, API4, API5, API8), File Upload and Secrets cheat sheets (research §3).

## Decision
`Authorization: Bearer <key>` required on all routes except `/healthz` and `/readyz`; keys from `AGENTIC_RAG_API_KEYS`
(comma-separated, at most two, for rotation), compared in constant time, never logged. Missing/invalid → `401 AUTHENTICATION_FAILED`.
CORS is an explicit allow-list (default: none). Collection deletion and ingest-by-server-path are disabled unless enabled in
settings. The SDK takes `api_key` (or env) and sends it only over the configured base URL.

## Consequences
No users, roles or tenants (non-goal). A single shared secret is coarse: anyone with a key can do everything the enabled
surface allows. Rotation works by deploying two keys, switching clients, then dropping the old one.

## Enforced by
Auth tests over every route (generated from the OpenAPI document so a new route without auth fails); settings validation (Step 15).

## Amendment: review round 1
Authentication is a middleware that runs **before** the body is read and before load shedding counts the request, because FastAPI parses a
body before a route-level dependency runs: an anonymous caller could otherwise make the server read and parse a 25 MB body and could fill
every in-flight slot. The route dependency stays as a second check and as the OpenAPI security scheme; both use one function
(`is_authorized`). `/metrics` needs the key as well as `/v1`. Setting `AGENTIC_RAG_ALLOW_UNAUTHENTICATED` together with keys is a start-up
error (it used to override the keys). Keys may be given as files (`AGENTIC_RAG_API_KEYS_FILE`). Known gap: an authenticated client can hold
a slot by announcing a body and not sending it; a reverse proxy with read timeouts is the mitigation.
