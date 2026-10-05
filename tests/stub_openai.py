"""A tiny local OpenAI-compatible HTTP server (chat completions + embeddings) for failure-injection tests.

It is a stand-in: it speaks the documented JSON shapes, so tests prove our adapters over a real socket, but they
say nothing about the real OpenAI service (no key was available).
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

DIM = 12


def _vector(text: str) -> list[float]:
    return [1.0 + len(text) % 7] + [float(len(text) % 5)] * (DIM - 1)


class StubOpenAI:
    def __init__(self) -> None:
        self.mode = "ok"  # ok | 500 | 429 | 401
        self.chat_reply = "ok"
        self.requests: list[dict[str, Any]] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                return None

            def _send(self, status: int, body: dict[str, Any], headers: dict[str, str] | None = None) -> None:
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self) -> None:
                length = int(self.headers.get("content-length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                stub.requests.append({"path": self.path, "body": body, "auth": self.headers.get("authorization")})
                if stub.mode != "ok":
                    status = {"500": 500, "429": 429, "401": 401}[stub.mode]
                    extra = {"retry-after": "7"} if status == 429 else {}
                    self._send(status, {"error": {"message": "SECRET-DETAIL", "type": "x"}}, extra)
                    return
                if self.path.endswith("/embeddings"):
                    inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
                    data = [{"object": "embedding", "index": i, "embedding": _vector(t)} for i, t in enumerate(inputs)]
                    self._send(
                        200,
                        {
                            "object": "list",
                            "data": data,
                            "model": body["model"],
                            "usage": {"prompt_tokens": 1, "total_tokens": 1},
                        },
                    )
                else:
                    self._send(
                        200,
                        {
                            "id": "c",
                            "object": "chat.completion",
                            "created": 0,
                            "model": body["model"],
                            "choices": [
                                {
                                    "index": 0,
                                    "finish_reason": "stop",
                                    "message": {"role": "assistant", "content": stub.chat_reply},
                                }
                            ],
                            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                        },
                    )

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def start(self) -> StubOpenAI:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
