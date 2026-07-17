"""
Streaming / microphone inference interface.

Full low-latency streaming ASR (incremental encoder states, endpoint
detection, partial-hypothesis emission) is out of scope for this
module and is left as a placeholder. NeMo's `EncDecRNNTBPEModel` /
cache-aware streaming Conformer variants support this natively; wiring
that up is a natural next step once the offline pipeline is validated.

This class defines the interface a real implementation would expose,
and provides a *chunked* (not truly streaming) reference implementation
that buffers fixed-size audio chunks and re-transcribes the growing
buffer — useful for demos, not for production low-latency use.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np

from models.base_asr_model import BaseASRModel, TranscriptionResult
from utils.audio_utils import AudioProcessingConfig, normalize_volume, resample

logger = logging.getLogger(__name__)


class StreamingTranscriber:
    """Reference (non-production) chunked streaming implementation.

    Usage:
        stream = StreamingTranscriber(model, on_partial=print)
        for chunk in microphone_chunks():   # e.g. 200ms float32 PCM chunks
            stream.push_chunk(chunk, sample_rate=16000)
        final = stream.finalize()

    NOTE: This re-runs full-context inference on the growing buffer on
    every chunk, which is O(n^2) over a session and NOT suitable for
    long-running production streaming. See module docstring.
    """

    def __init__(
        self,
        model: BaseASRModel,
        on_partial: Optional[Callable[[str], None]] = None,
        audio_config: Optional[AudioProcessingConfig] = None,
        min_buffer_sec: float = 1.0,
    ):
        self.model = model
        self.on_partial = on_partial
        self.audio_config = audio_config or AudioProcessingConfig()
        self.min_buffer_sec = min_buffer_sec
        self._buffer = np.zeros(0, dtype=np.float32)

    def push_chunk(self, chunk: np.ndarray, sample_rate: int) -> Optional[str]:
        """Append a raw audio chunk to the buffer and emit a partial
        transcript once enough audio has accumulated."""
        if sample_rate != self.audio_config.target_sample_rate:
            chunk = resample(chunk, sample_rate, self.audio_config.target_sample_rate)
        if self.audio_config.normalize:
            chunk = normalize_volume(chunk)

        self._buffer = np.concatenate([self._buffer, chunk.astype(np.float32)])

        duration = len(self._buffer) / self.audio_config.target_sample_rate
        if duration < self.min_buffer_sec:
            return None

        result = self._transcribe_buffer()
        if self.on_partial:
            self.on_partial(result.text)
        return result.text

    def finalize(self) -> TranscriptionResult:
        """Run a final transcription pass over the full buffer and reset state."""
        result = self._transcribe_buffer()
        self._buffer = np.zeros(0, dtype=np.float32)
        return result

    def _transcribe_buffer(self) -> TranscriptionResult:
        import torch

        audio_tensor = torch.from_numpy(self._buffer).unsqueeze(0)
        lengths = torch.tensor([self._buffer.shape[0]], dtype=torch.long)
        results = self.model.transcribe_batch(audio_tensor, lengths, decoding_strategy="greedy")
        return results[0]
