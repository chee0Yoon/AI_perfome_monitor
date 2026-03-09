from __future__ import annotations

import numpy as np

from final_metric_refactor.distribution.output_density import OutputDensityMetric


class OutputSignalRunner:
    """Wrapper for output density anomaly signal."""

    def __init__(self, min_k: int, max_k: int) -> None:
        self.metric = OutputDensityMetric(min_k=min_k, max_k=max_k)

    def compute(
        self,
        y_norm: np.ndarray,
        ref_mask: np.ndarray,
        weights: np.ndarray | None,
    ) -> tuple[np.ndarray, list[int]]:
        return self.metric.compute(y_norm, ref_mask, ref_weights=weights)
