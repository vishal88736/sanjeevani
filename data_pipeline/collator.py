"""
Data collator that dynamically pads variable-length audio waveforms and
tokenized text within a batch, and produces attention/length masks
needed by the Conformer encoder and CTC/attention losses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from .indicvoices_dataset import ASRExample


@dataclass
class ASRBatch:
    audio_signal: torch.Tensor          # (B, T_max) float32
    audio_signal_lengths: torch.Tensor  # (B,) int64, samples per example
    text_tokens: Optional[torch.Tensor] = None       # (B, L_max) int64, padded with pad_id
    text_lengths: Optional[torch.Tensor] = None      # (B,) int64
    raw_texts: Optional[list[str]] = None
    languages: Optional[list[str]] = None


class ASRDataCollator:
    """Callable collator for use as `collate_fn` in a PyTorch DataLoader.

    Text tokenization is optional at the collator level: NeMo's Conformer
    model tokenizes internally from raw text during training, so by
    default this collator only pads audio and passes raw text through.
    Set `tokenizer` to enable pre-tokenized batches (used by the
    Whisper fallback path).
    """

    def __init__(self, tokenizer: Optional[Any] = None, pad_id: int = 0):
        self.tokenizer = tokenizer
        self.pad_id = pad_id

    def __call__(self, examples: list[ASRExample]) -> ASRBatch:
        waveforms = [torch.from_numpy(ex.waveform).float() for ex in examples]
        lengths = torch.tensor([w.shape[0] for w in waveforms], dtype=torch.long)
        max_len = int(lengths.max().item())

        padded_audio = torch.zeros(len(waveforms), max_len, dtype=torch.float32)
        for i, w in enumerate(waveforms):
            padded_audio[i, : w.shape[0]] = w

        raw_texts = [ex.text for ex in examples]
        languages = [ex.language for ex in examples]

        text_tokens, text_lengths = None, None
        if self.tokenizer is not None:
            token_ids = [
                torch.tensor(self.tokenizer.encode(t), dtype=torch.long) for t in raw_texts
            ]
            text_lengths = torch.tensor([len(t) for t in token_ids], dtype=torch.long)
            max_text_len = int(text_lengths.max().item())
            text_tokens = torch.full(
                (len(token_ids), max_text_len), self.pad_id, dtype=torch.long
            )
            for i, t in enumerate(token_ids):
                text_tokens[i, : t.shape[0]] = t

        return ASRBatch(
            audio_signal=padded_audio,
            audio_signal_lengths=lengths,
            text_tokens=text_tokens,
            text_lengths=text_lengths,
            raw_texts=raw_texts,
            languages=languages,
        )
