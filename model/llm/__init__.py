"""
Gemma-4-driven reasoning (extraction/planning + clinical triage), WHO
grounding, function-calling actions, and session memory.
"""

from .ollama_client import OllamaAnswerer
from .reasoning import GemmaReasoner, ExtractionResult, TriageResult, ReasoningParseError
from .knowledge_base import retrieve_context, match_topics
from .functions import dispatch as dispatch_function
from .memory import ConversationStore

__all__ = [
    "OllamaAnswerer",
    "GemmaReasoner",
    "ExtractionResult",
    "TriageResult",
    "ReasoningParseError",
    "retrieve_context",
    "match_topics",
    "dispatch_function",
    "ConversationStore",
]
