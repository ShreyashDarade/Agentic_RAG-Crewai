"""Pure helpers every answer pipeline shares: how retrieved text is shown to a model and how a model's JSON is read."""

from __future__ import annotations

import html
import json
from collections.abc import Sequence
from typing import Any

from agentic_rag.errors import ModelOutputInvalid
from agentic_rag.ports.types import ScoredChunk

__all__ = ["NO_INFORMATION", "parse_answer", "render_chunks"]

NO_INFORMATION = "I could not find information about that in the indexed documents."


def render_chunks(hits: Sequence[ScoredChunk]) -> str:
    """Retrieved text as escaped, delimited, untrusted data (OWASP LLM01): a chunk cannot close its own element."""
    return "\n".join(
        f'<chunk id="{hit.chunk.id}" source="{html.escape(hit.chunk.document_name, quote=True)}">'
        f"{html.escape(hit.chunk.text, quote=False)}</chunk>"
        for hit in hits
    )


def parse_answer(raw: str) -> tuple[str, tuple[str, ...]]:
    """``{"answer": str, "citations": [str]}`` from a model, possibly wrapped in a Markdown code fence."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[4:] if text.lower().startswith("json") else text
    try:
        data: Any = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise ModelOutputInvalid("the model did not return JSON") from exc
    if not isinstance(data, dict):
        raise ModelOutputInvalid("the model returned JSON that is not an object")
    answer, citations = data.get("answer"), data.get("citations", [])
    if not isinstance(answer, str) or not answer.strip():
        raise ModelOutputInvalid("the model returned no answer text")
    if not isinstance(citations, list) or not all(isinstance(c, str) for c in citations):
        raise ModelOutputInvalid("the model returned malformed citations")
    return answer.strip(), tuple(citations)
