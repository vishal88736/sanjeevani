"""
Wrapper around AI4Bharat's IndicTrans2, both directions:
  - indic-en: translating what the person said/typed into English
  - en-indic: translating Gemma 4's English answer back into the
    person's language, so it can be spoken aloud by the TTS module

Follows the official HF usage pattern from the IndicTrans2 repo:
tokenizer + AutoModelForSeq2SeqLM + IndicTransToolkit's IndicProcessor
for the FLORES-style pre/post-processing IndicTrans2 expects. Each
direction is a separate checkpoint and is lazy-loaded independently,
so a deployment that only ever uses one direction doesn't pay for both.

Requires: pip install IndicTransToolkit (see requirements.txt)
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from model.config import LANGUAGE_BY_CODE, Settings, settings as default_settings

logger = logging.getLogger(__name__)


class _Direction:
    """Holds the lazily-loaded model/tokenizer for one translation direction."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None


class IndicTranslator:
    def __init__(self, cfg: Settings = default_settings):
        self.cfg = cfg
        self._indic_en = _Direction(cfg.translation_indic_en_model_id)
        self._en_indic = _Direction(cfg.translation_en_indic_model_id)
        self._processor = None
        self._device = None

    def _ensure_processor(self) -> None:
        if self._processor is not None:
            return
        from IndicTransToolkit.processor import IndicProcessor

        self._device = "cuda" if (self.cfg.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
        self._processor = IndicProcessor(inference=True)

    def _ensure_direction_loaded(self, direction: _Direction) -> None:
        self._ensure_processor()
        if direction.model is not None:
            return
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        logger.info("Loading IndicTrans2 '%s' onto %s (first use only)...",
                    direction.model_id, self._device)
        direction.tokenizer = AutoTokenizer.from_pretrained(direction.model_id, trust_remote_code=True)
        direction.model = AutoModelForSeq2SeqLM.from_pretrained(
            direction.model_id, trust_remote_code=True
        ).to(self._device)
        direction.model.eval()
        logger.info("IndicTrans2 '%s' loaded.", direction.model_id)

    def _translate(self, direction: _Direction, text: str, src_lang: str, tgt_lang: str) -> str:
        self._ensure_direction_loaded(direction)

        batch = self._processor.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)
        inputs = direction.tokenizer(
            batch, truncation=True, padding="longest", return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            generated_tokens = direction.model.generate(
                **inputs, num_beams=5, max_length=256, use_cache=True,
            )

        decoded = direction.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return self._processor.postprocess_batch(decoded, lang=tgt_lang)[0].strip()

    def translate_to_english(self, text: str, source_language_code: str) -> str:
        """Translate `text` (in the language identified by our short UI
        code, e.g. 'hi', 'ta') into English."""
        if not text.strip():
            return ""
        lang_info = LANGUAGE_BY_CODE.get(source_language_code)
        if lang_info is None:
            raise ValueError(f"Unsupported language code '{source_language_code}'")
        if source_language_code == "en":
            return text.strip()

        return self._translate(self._indic_en, text, lang_info.flores_code, "eng_Latn")

    def translate_from_english(self, text: str, target_language_code: str) -> str:
        """Translate English `text` into the language identified by
        `target_language_code`. Used to prepare the LLM's answer for
        text-to-speech in the person's own language."""
        if not text.strip():
            return ""
        lang_info = LANGUAGE_BY_CODE.get(target_language_code)
        if lang_info is None:
            raise ValueError(f"Unsupported language code '{target_language_code}'")
        if target_language_code == "en":
            return text.strip()

        return self._translate(self._en_indic, text, "eng_Latn", lang_info.flores_code)
