"""
Shared Ollama Cloud client plumbing, plus a simple single-call
fallback answerer.

The primary reasoning path is model/llm/reasoning.py's GemmaReasoner
(multi-stage: extract & plan -> retrieve -> clinical reasoning). This
module's `OllamaAnswerer` is the fallback used when that multi-stage
pipeline fails end-to-end (e.g. Gemma's JSON output couldn't be parsed
even after a retry) — see model/pipeline.py.

Note on provider choice: Groq does not currently host Gemma 4 (their
Gemma offering was deprecated in 2025 and hasn't been reinstated as of
this writing) — Gemma 4 is served either directly by Google's Gemini
API or, as used here, through Ollama Cloud, which requires its own
account and API key at ollama.com/settings/keys. That key is read
from OLLAMA_API_KEY and is never hardcoded.
"""

from __future__ import annotations

import logging
from typing import Optional

from model.config import Settings, settings as default_settings
from model.llm.knowledge_base import KnowledgeBaseResult

logger = logging.getLogger(__name__)


def get_client(cfg: Settings = default_settings):
    """Builds an authenticated Ollama Cloud client. Raises a clear
    RuntimeError if no API key is configured."""
    if not cfg.ollama_api_key:
        logger.error("OLLAMA_API_KEY is not set — cannot reach Gemma 4.")
        raise RuntimeError(
            "OLLAMA_API_KEY is not set. Copy .env.example to .env, create a key at "
            "ollama.com/settings/keys, and add it there."
        )
    from ollama import Client

    return Client(
        host=cfg.ollama_host,
        headers={"Authorization": f"Bearer {cfg.ollama_api_key}"},
    )


SYSTEM_PROMPT = """You are Sanjeevani, a preliminary health-information assistant for \
rural and semi-urban India. A person has described a symptom or health question, which \
has already been transcribed and translated into English for you. This may be part of \
an ongoing conversation — use the earlier turns for context.

Rules you must always follow:
- You are NOT a doctor and you do NOT diagnose. Give general, cautious health information only.
- Always recommend the person confirm with a qualified doctor, nurse, or local health \
worker (e.g. an ASHA worker or the nearest Primary Health Centre), especially before \
taking any medication.
- If the described symptoms could indicate a medical emergency, say so clearly and tell \
them to call 108 (India's emergency ambulance number) or go to the nearest hospital \
immediately.
- Keep answers short, plain, and calm. Avoid jargon; explain any medical term you must use.
- Never invent statistics, drug dosages, or diagnoses you are not certain of.
"""

UNGROUNDED_NOTICE = (
    "\n\nNote: no matching WHO reference was found for this question. Answer only with "
    "well-established, general health information, keep it especially cautious, and lean "
    "more heavily on recommending in-person care."
)


class OllamaAnswerer:
    """Simple, single-call fallback: one Gemma 4 request, no structured
    extraction/triage. Used when GemmaReasoner's multi-stage pipeline
    fails end-to-end so the person still gets an answer."""

    def __init__(self, cfg: Settings = default_settings):
        self.cfg = cfg
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            self._client = get_client(self.cfg)
        return self._client

    def answer(
        self,
        english_query: str,
        kb_result: Optional[KnowledgeBaseResult] = None,
        history: Optional[list[dict]] = None,
    ) -> str:
        client = self._ensure_client()

        system_prompt = SYSTEM_PROMPT
        if kb_result and kb_result.is_grounded and kb_result.context:
            system_prompt += f"\n\nUse this WHO reference material:\n\n{kb_result.context}"
        else:
            system_prompt += UNGROUNDED_NOTICE

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history or [])
        messages.append({"role": "user", "content": english_query})

        try:
            response = client.chat(
                model=self.cfg.ollama_model,
                messages=messages,
                options={"temperature": 0.3},
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Ollama fallback request failed")
            raise RuntimeError(f"Answer generation failed: {exc}") from exc

        return response["message"]["content"].strip()
