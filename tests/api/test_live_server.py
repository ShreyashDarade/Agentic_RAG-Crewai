"""Behaviour that only exists in a real process: the SIGTERM drain and what the server prints (review B4, B10)."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from tests.conftest import API_KEY

ROOT = Path(__file__).resolve().parents[2]
ENV = {
    "AGENTIC_RAG_PLUGINS": "tests.plugin_full",
    "AGENTIC_RAG_API_KEYS": API_KEY,
    "AGENTIC_RAG_VECTOR_STORE": "memory",
    "AGENTIC_RAG_EMBEDDER": "hashing",
    "AGENTIC_RAG_CHAT_MODEL": "scripted",
    "AGENTIC_RAG_ANSWER_PIPELINE": "shouting",
    "AGENTIC_RAG_EMBEDDING_DIMENSION": "32",
    "AGENTIC_RAG_SHUTDOWN_DRAIN_SECONDS": "3",
}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def server() -> Iterator[tuple[subprocess.Popen[str], str]]:
    port = _free_port()
    # src first: the process must run the code of the tree this test belongs to (a mutated copy in the proofs), not an
    # editable install of another checkout; the repository root second so that ``tests.plugin_full`` imports.
    env = {**os.environ, **ENV, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT)]), "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "agentic_rag.cli", "serve", "--port", str(port)],
        env=env,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                if httpx.get(base + "/readyz", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.1)
        else:
            raise AssertionError("the server never became ready")
        yield proc, base
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)


def test_sigterm_flips_readyz_to_draining_while_the_listener_is_still_open(
    server: tuple[subprocess.Popen[str], str],
) -> None:
    proc, base = server
    started = time.monotonic()
    proc.send_signal(signal.SIGTERM)
    seen_draining = False
    while time.monotonic() - started < 2.5:
        try:
            r = httpx.get(base + "/readyz", timeout=1)
        except httpx.HTTPError:
            break
        if r.status_code == 503 and r.json()["checks"] == {"server": "draining"}:
            seen_draining = True
            assert httpx.get(base + "/healthz", timeout=1).status_code == 200  # alive while draining
            break
        time.sleep(0.05)
    assert seen_draining, "/readyz never answered 503 'draining' after SIGTERM: the listener closed first"
    assert proc.wait(timeout=30) in (0, -signal.SIGTERM)  # uvicorn re-raises the signal once it has stopped cleanly
    assert time.monotonic() - started >= 3  # it really waited for the drain period


def test_everything_the_server_prints_is_json_and_never_contains_a_query_string(
    server: tuple[subprocess.Popen[str], str],
) -> None:
    proc, base = server
    httpx.get(base + "/v1/documents?page_token=SECRET-TOKEN-VALUE", headers={"Authorization": f"Bearer {API_KEY}"})
    proc.send_signal(signal.SIGTERM)
    out, _ = proc.communicate(timeout=30)
    lines = [line for line in out.splitlines() if line.strip()]
    assert lines
    for line in lines:
        json.loads(line)  # uvicorn's own start-up and access lines used to be plain text
    assert "SECRET-TOKEN-VALUE" not in out
