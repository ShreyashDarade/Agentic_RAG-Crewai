# ADR-0006: Static bearer API key; destructive surface off by default

Status: proposed · Date: 2026-10-04 · Owner decision: built-in static API key

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
