from __future__ import annotations

import numpy as np

from final_metric_refactor.distribution.sim_conflict import SimilarInputConflictMetric


class SimilarInputConflictSignalRunner:
    """Wrapper for similar-input conflict anomaly signal."""

    def __init__(self, similarity_threshold: float, similarity_k: int) -> None:
        self.metric = SimilarInputConflictMetric(
            similarity_threshold=similarity_threshold,
            similarity_k=similarity_k,
        )

    def compute(self, x_norm: np.ndarray, unit_delta: np.ndarray, tau: np.ndarray) -> np.ndarray:
        return self.metric.compute(x_norm, unit_delta, tau)
