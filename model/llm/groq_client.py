"""
Answer generation via Groq's OpenAI-compatible chat completions API.

The API key is never hardcoded — it's read from the GROQ_API_KEY
environment variable (see model/config.py and .env.example) and is
expected to be supplied by whoever deploys this.
"""

from __future__ import annotations

import logging

from model.config import Settings, settings as default_settings
from model.llm.knowledge_base import KnowledgeBaseResult

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are Sanjeevani, a preliminary health-information assistant for \
rural and semi-urban India. A person has described a symptom or health question, which \
has already been transcribed and translated into English for you.

Rules you must always follow:
- You are NOT a doctor and you do NOT diagnose. Give general, cautious health information only.
- Always recommend the person confirm with a qualified doctor, nurse, or local health \
worker (e.g. an ASHA worker or the nearest Primary Health Centre), especially before \
taking any medication.
- If the described symptoms could indicate a medical emergency (e.g. chest pain, \
difficulty breathing, severe bleeding, loss of consciousness, suspected stroke, a child \
with a very high fever or seizures), say so clearly and tell them to call 108 (India's \
emergency ambulance number) or go to the nearest hospital immediately.
- Keep answers short, plain, and calm — written for someone who may be reading this in \
their second or third language. Avoid jargon; explain any medical term you must use.
- Never invent statistics, drug dosages, or diagnoses you are not certain of.
"""

UNGROUNDED_NOTICE = (
    "\n\nNote: no verified clinical reference source is connected yet for this "
    "deployment (this is a placeholder integration). Answer only with well-established, "
    "general health information, keep it especially cautious, and lean more heavily on "
    "recommending in-person care."
)


class GroqAnswerer:
    """Lazy-loading wrapper around the Groq chat completions client."""

    def __init__(self, cfg: Settings = default_settings):
        self.cfg = cfg
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if not self.cfg.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        from groq import Groq

        self._client = Groq(api_key=self.cfg.groq_api_key)
        return self._client

    def answer(self, english_query: str, kb_result: KnowledgeBaseResult) -> str:
        client = self._ensure_client()

        system_prompt = SYSTEM_PROMPT
        if kb_result.is_grounded and kb_result.context:
            system_prompt += (
                "\n\nUse the following verified reference material as your primary "
                f"source. If it doesn't cover the question, say so rather than "
                f"guessing:\n\n{kb_result.context}"
            )
        else:
            system_prompt += UNGROUNDED_NOTICE

        try:
            completion = client.chat.completions.create(
                model=self.cfg.groq_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": english_query},
                ],
                temperature=0.3,
                max_tokens=500,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Groq request failed")
            raise RuntimeError(f"Answer generation failed: {exc}") from exc

        return completion.choices[0].message.content.strip()
