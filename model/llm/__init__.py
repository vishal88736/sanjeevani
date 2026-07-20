"""Answer generation via Groq, plus a placeholder medical-knowledge hook."""

from .groq_client import GroqAnswerer
from .knowledge_base import retrieve_context

__all__ = ["GroqAnswerer", "retrieve_context"]
