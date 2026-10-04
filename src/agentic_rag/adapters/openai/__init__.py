"""OpenAI embedder and chat model. The ``openai`` package is imported only in this package."""

from agentic_rag.adapters.openai.chat import OpenAIChatModel
from agentic_rag.adapters.openai.client import create_client
from agentic_rag.adapters.openai.embedder import OpenAIEmbedder

__all__ = ["OpenAIChatModel", "OpenAIEmbedder", "create_client"]
