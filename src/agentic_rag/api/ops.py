"""Operational helpers kept out of the route functions."""

from __future__ import annotations

import logging

from fastapi import Request

from agentic_rag.application import Service
from agentic_rag.contracts import ReadyResponse

__all__ = ["check_readiness"]

logger = logging.getLogger("agentic_rag.ops")


async def check_readiness(request: Request, service: Service) -> ReadyResponse:
    """The one definition of "ready": used by /readyz and by /metrics so they can never disagree."""
    if request.app.state.draining:
        report = ReadyResponse(ready=False, checks={"server": "draining"})
    else:
        report = await service.ready()
    request.app.state.metrics.set_readiness(report)
    if not report.ready:
        failing = sorted(name for name, state in report.checks.items() if state != "ok")
        logger.warning("not ready", extra={"event": "not_ready", "failing": failing})
    return report
