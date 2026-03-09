from __future__ import annotations

import numpy as np

from final_metric_refactor.distribution.direction import DirectionMetric
from final_metric_refactor.distribution._shared import LocalKNNContext


class DirectionSignalRunner:
    """Wrapper for local direction anomaly signal."""

    def __init__(self) -> None:
        self.metric = DirectionMetric()

    def compute(
        self,
        unit_delta: np.ndarray,
        knn_context: LocalKNNContext,
        weights: np.ndarray | None,
        tau: np.ndarray,
    ) -> np.ndarray:
        return self.metric.compute(unit_delta, knn_context, sample_weights=weights, tau=tau)
