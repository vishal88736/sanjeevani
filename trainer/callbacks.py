"""Training callbacks. Currently: early stopping on a monitored metric."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stops training once `monitor` fails to improve by `min_delta` for
    `patience` consecutive evaluations."""

    def __init__(
        self,
        monitor: str = "val_wer",
        mode: str = "min",
        patience: int = 5,
        min_delta: float = 0.001,
    ):
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self._best: float | None = None
        self._num_bad_evals = 0
        self.should_stop = False

    def step(self, metrics: dict[str, float]) -> bool:
        """Update state with the latest eval metrics. Returns True if training should stop."""
        if self.monitor not in metrics:
            logger.warning(
                "EarlyStopping monitor '%s' not found in metrics %s; skipping check.",
                self.monitor, list(metrics.keys()),
            )
            return False

        value = metrics[self.monitor]
        if self._best is None:
            self._best = value
            return False

        improved = (
            value < self._best - self.min_delta
            if self.mode == "min"
            else value > self._best + self.min_delta
        )

        if improved:
            self._best = value
            self._num_bad_evals = 0
        else:
            self._num_bad_evals += 1
            logger.info(
                "No improvement in %s for %d/%d evaluations.",
                self.monitor, self._num_bad_evals, self.patience,
            )

        if self._num_bad_evals >= self.patience:
            logger.info("Early stopping triggered (monitor=%s, best=%.4f).", self.monitor, self._best)
            self.should_stop = True

        return self.should_stop
