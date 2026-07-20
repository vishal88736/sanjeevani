"""
Wrapper around AI4Bharat's IndicTrans2 (indic-en direction) for
translating transcribed/typed text into English.

Follows the official HF usage pattern from the IndicTrans2 repo:
tokenizer + AutoModelForSeq2SeqLM + IndicTransToolkit's IndicProcessor
for the FLORES-style pre/post-processing IndicTrans2 expects.

Requires: pip install IndicTransToolkit (see requirements.txt)
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from model.config import LANGUAGE_BY_CODE, Settings, settings as default_settings

logger = logging.getLogger(__name__)


class IndicTranslator:
    """Lazy-loading wrapper around IndicTrans2 indic->en."""

    def __init__(self, cfg: Settings = default_settings):
        self.cfg = cfg
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._device = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from IndicTransToolkit.processor import IndicProcessor

        self._device = "cuda" if (self.cfg.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
        logger.info("Loading IndicTrans2 '%s' onto %s (first use only)...",
                    self.cfg.translation_model_id, self._device)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.translation_model_id, trust_remote_code=True
        )
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.cfg.translation_model_id, trust_remote_code=True
        ).to(self._device)
        self._model.eval()
        self._processor = IndicProcessor(inference=True)
        logger.info("IndicTrans2 loaded.")

    def translate_to_english(self, text: str, source_language_code: str) -> str:
        """Translate `text` (in the language identified by our short UI
        code, e.g. 'hi', 'ta') into English."""
        if not text.strip():
            return ""

        lang_info = LANGUAGE_BY_CODE.get(source_language_code)
        if lang_info is None:
            raise ValueError(f"Unsupported language code '{source_language_code}'")

        if source_language_code == "en":
            return text.strip()  # already English, skip the model entirely

        self._ensure_loaded()
        src_lang, tgt_lang = lang_info.flores_code, "eng_Latn"

        batch = self._processor.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)
        inputs = self._tokenizer(
            batch, truncation=True, padding="longest", return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            generated_tokens = self._model.generate(
                **inputs,
                num_beams=5,
                max_length=256,
                use_cache=True,
            )

        decoded = self._tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        translations = self._processor.postprocess_batch(decoded, lang=tgt_lang)
        return translations[0].strip()
