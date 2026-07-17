"""WER/CER metrics and full-dataset evaluation utilities."""

from .metrics import compute_wer, compute_cer
from .evaluator import Evaluator

__all__ = ["compute_wer", "compute_cer", "Evaluator"]
