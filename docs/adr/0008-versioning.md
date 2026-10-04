# ADR-0008: SemVer for SDK and wire; `/v1`; start at 0.1.0

Status: proposed · Date: 2026-10-04

## Context
Code claims `2.0.0` but no endpoint worked (baseline §2). Research §2 C1–C5: SemVer text, PEP 702, Python warning audiences,
Django/NumPy/pandas deprecation policies, AIP-180/185 (major only in the URL).

## Decision
Package version `0.1.0`; `0.y.z` until the wire contract and SDK surface are frozen, then `1.0.0`. Routes under `/v1`; additive-only
within a major. Deprecation via `deprecated(since, remove_in, alternative, escalate_in)`; removal only in a later major; notice target
≥ 2 minor releases and ≥ 6 months (a **review rule**, not machine-enforced). Deprecations raise a library-specific
`DeprecationWarning` subclass for developers; operator-visible behaviour changes also log at startup.

## Consequences
The version number goes down relative to the current label; tags/releases start fresh. Compatibility is machine-checked only for
signature and schema changes (griffe, oasdiff); semantic changes rely on review.

## Enforced by
`griffe check`, `oasdiff breaking`, decorator tests, pytest warning filters (Steps 11, 14).
