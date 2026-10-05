"""Prometheus metrics with bounded label sets (route *templates*, status classes, error codes)."""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from agentic_rag.contracts import ReadyResponse

__all__ = ["CONTENT_TYPE_LATEST", "Metrics"]

# Buckets reach well above the 10 s default because answering is model-bound (framework section 11).
_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self._duration = Histogram(
            "agentic_rag_http_request_duration_seconds",
            "HTTP request latency",
            ["method", "route", "status_class"],
            buckets=_BUCKETS,
            registry=self.registry,
        )
        self._in_flight = Gauge("agentic_rag_http_requests_in_flight", "Requests being served", registry=self.registry)
        self._errors = Counter(
            "agentic_rag_errors_total", "Errors returned, by stable error code", ["code"], registry=self.registry
        )
        self._shed = Counter(
            "agentic_rag_load_shed_total", "Requests rejected because the server was full", registry=self.registry
        )
        self._dependency_up = Gauge(
            "agentic_rag_dependency_up",
            "1 if the dependency check passed at the last readiness evaluation",
            ["dependency"],
            registry=self.registry,
        )
        self._ready = Gauge("agentic_rag_ready", "1 if every dependency is ready", registry=self.registry)

    def request_started(self) -> None:
        self._in_flight.inc()

    def request_finished(self, method: str, route: str, status: int, seconds: float) -> None:
        self._in_flight.dec()
        self._duration.labels(method, route, f"{status // 100}xx").observe(seconds)

    def error(self, code: str) -> None:
        self._errors.labels(code).inc()

    def shed(self) -> None:
        self._shed.inc()

    def set_readiness(self, report: ReadyResponse) -> None:
        self._ready.set(1 if report.ready else 0)
        for name, state in report.checks.items():
            self._dependency_up.labels(name).set(1 if state == "ok" else 0)

    def render(self) -> bytes:
        return generate_latest(self.registry)
