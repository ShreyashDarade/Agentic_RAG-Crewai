#!/usr/bin/env python3
"""Packaging check (G2 packaging half, G8 wheel half): build the wheel, install ONLY it, and prove the thin client works.

    scripts/check_wheel.py

* the wheel ships ``py.typed`` and every top-level module the layers contract names;
* in a clean virtual environment with only the base dependencies, ``import agentic_rag`` and the SDK, models, extension and
  testing modules work and none of the engine libraries (fastapi, pymilvus, openai, crewai, ...) end up in ``sys.modules``;
* the engine names are absent from ``agentic_rag.__all__`` and fail with an ImportError that names the extra;
* the client can talk to a server (a stub served with the standard library) from that environment.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROBE = r"""
import json, sys, threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import agentic_rag
from agentic_rag import Client
import agentic_rag.client, agentic_rag.models, agentic_rag.extend, agentic_rag.testing

FORBIDDEN = ["fastapi", "starlette", "uvicorn", "pymilvus", "openai", "crewai", "chromadb", "qdrant_client",
             "pydantic_settings", "numpy", "torch", "bs4", "lxml", "fitz", "pymupdf", "docx", "prometheus_client"]
loaded = sorted(m for m in FORBIDDEN if m in sys.modules)
assert not loaded, f"the thin client imported engine libraries: {loaded}"
ENGINE = ("Settings", "load_settings", "build_container")
assert not set(ENGINE) & set(agentic_rag.__all__), agentic_rag.__all__
for name in ENGINE:
    try:
        getattr(agentic_rag, name)
    except ImportError as exc:
        assert "agentic-rag[engine]" in str(exc), exc
    else:
        raise AssertionError(f"{name} resolved on a thin install")

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        body = json.dumps({"ready": True, "checks": {}}).encode()
        self.send_response(200); self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body))); self.end_headers(); self.wfile.write(body)

server = HTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=server.serve_forever, daemon=True).start()
with Client.http(f"http://127.0.0.1:{server.server_address[1]}") as client:
    assert client.ready().ready is True
loaded = sorted(m for m in FORBIDDEN if m in sys.modules)
assert not loaded, f"making a call imported engine libraries: {loaded}"
print("thin client ok:", agentic_rag.__version__)
"""


def run(*args: str, cwd: Path = ROOT) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="check_wheel_") as tmp:
        tmp_path = Path(tmp)
        dist = tmp_path / "dist"
        run("uv", "build", "--wheel", "--out-dir", str(dist))
        wheel = next(dist.glob("agentic_rag-*.whl"))
        with zipfile.ZipFile(wheel) as archive:
            names = set(archive.namelist())
        assert "agentic_rag/py.typed" in names, "the wheel does not ship py.typed"
        for module in ("client.py", "errors.py", "models.py", "extend.py", "blocking.py", "container.py", "server.py"):
            assert f"agentic_rag/{module}" in names, f"agentic_rag/{module} is missing from the wheel"
        venv = tmp_path / "venv"
        run("uv", "venv", "--python", f"{sys.version_info.major}.{sys.version_info.minor}", str(venv))
        python = venv / "bin" / "python"
        run("uv", "pip", "install", "--python", str(python), str(wheel))
        run(str(python), "-c", PROBE, cwd=tmp_path)  # from outside the repository: the installed wheel, not src/
    print("wheel ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
