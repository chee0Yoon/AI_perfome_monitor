from __future__ import annotations

import numpy as np

from final_metric_refactor.distribution.length import LengthMetric
from final_metric_refactor.distribution._shared import LocalKNNContext


class LengthSignalRunner:
    """Wrapper for local length anomaly signal."""

    def __init__(self) -> None:
        self.metric = LengthMetric()

    def compute(
        self,
        tau: np.ndarray,
        knn_context: LocalKNNContext,
        weights: np.ndarray | None,
    ) -> np.ndarray:
        return self.metric.compute(tau, knn_context, sample_weights=weights)
