"""CrewAI-backed components. ``crewai`` is imported only in this package (ADR-0005, ADR-0010)."""

from agentic_rag.adapters.crewai.chat import CrewAIChatModel
from agentic_rag.adapters.crewai.embedder import CrewAIEmbedder
from agentic_rag.adapters.crewai.pipeline import CrewPipeline

__all__ = ["CrewAIChatModel", "CrewAIEmbedder", "CrewPipeline"]
