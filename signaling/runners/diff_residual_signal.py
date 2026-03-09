from __future__ import annotations

import numpy as np

from final_metric_refactor.distribution.diff_residual import DiffResidualMetric
from final_metric_refactor.distribution._shared import LocalKNNContext


class DiffResidualSignalRunner:
    """Wrapper for diff-residual anomaly signal."""

    def __init__(
        self,
        cov_shrinkage: float,
        aux_enabled: bool = False,
        aux_lambda: float = 0.7,
        aux_model: str = "linear",
        row_chunk_workers: int = 0,
    ) -> None:
        self.metric = DiffResidualMetric(
            cov_shrinkage=cov_shrinkage,
            aux_enabled=aux_enabled,
            aux_lambda=aux_lambda,
            aux_model=aux_model,
            max_workers=row_chunk_workers,
        )

    def compute(
        self,
        d_raw: np.ndarray,
        knn_context: LocalKNNContext,
        weights: np.ndarray | None,
        input_embs: np.ndarray,
        direction_signal: np.ndarray | None = None,
        length_signal: np.ndarray | None = None,
        ref_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        signal = self.metric.compute(
            d_raw,
            knn_context,
            sample_weights=weights,
            X=input_embs,
            direction_signal=direction_signal,
            length_signal=length_signal,
            ref_mask=ref_mask,
        )
        meta = dict(getattr(self.metric, "last_meta", {}) or {})
        return signal, meta
