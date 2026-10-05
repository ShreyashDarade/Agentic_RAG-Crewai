# ADR-0003: One distribution, thin by default, engine as extras

Status: proposed · Date: 2026-10-04

## Context
`requirements.txt` installs ~6.9 GB (torch with CUDA wheels, EasyOCR, CrewAI, …) for every use, including callers who only need
HTTP (baseline §4.6). Research: openai/anthropic/stripe/httpx keep core dependencies minimal and use extras (research §2 A3).

## Decision
Distribution `agentic-rag`, import package `agentic_rag`, `src/` layout, `pyproject.toml` + committed `uv.lock`. Base dependencies:
`httpx`, `pydantic`. Extras: `engine` (in-process use), `server` (FastAPI/uvicorn), `worker`, `crewai`, `ocr`, `rerank`, `dev`.
Engine-only names are absent from `__all__` on a thin install and raise an `ImportError` naming the extra.

## Consequences
Install matrix in README must be proven from clean virtualenvs (Step 20). The owner's choice of "Python only" removes any
second-language release pipeline. Rejected: separate `-client` distribution (two release trains; the owner asked for one SDK
with two transports).

## Enforced by
Clean-venv test asserting `pip list` and `sys.modules` for the thin install; wheel-content test (`py.typed`) (Steps 12, 20).
