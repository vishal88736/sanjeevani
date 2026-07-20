"""
PLACEHOLDER medical knowledge source.

Sanjeevani's answers should eventually be grounded in a verified
medical reference (a curated triage protocol, WHO/ICMR guidelines, a
vetted clinical dataset — whatever source you provide). That source
is not wired up yet, so this module is an intentional stub: it always
reports `is_grounded=False` and an empty context, which the Groq
prompt (see groq_client.py) uses to answer with visibly stronger
caveats and to keep the "DRAFT" badge on the frontend honest.

To connect a real source later, replace the body of `retrieve_context`
with an actual lookup (a vector search over your dataset, a SQL query,
an API call to a verified clinical knowledge base, etc.) and set
`is_grounded=True` with the `sources` you retrieved from.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class KnowledgeBaseResult:
    context: str
    sources: list[str] = field(default_factory=list)
    is_grounded: bool = False


def retrieve_context(query: str) -> KnowledgeBaseResult:
    """PLACEHOLDER: returns no grounding context. Replace with a real
    lookup against your verified medical data source."""
    # TODO(sanjeevani): plug in the real medical data source here.
    return KnowledgeBaseResult(context="", sources=[], is_grounded=False)
