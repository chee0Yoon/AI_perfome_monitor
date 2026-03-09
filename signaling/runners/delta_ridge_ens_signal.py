from __future__ import annotations

import numpy as np

from final_metric_refactor.distribution.delta_ridge_ensemble import DeltaRidgeEnsembleMetric
from final_metric_refactor.distribution._shared import LocalKNNContext


class DeltaRidgeEnsSignalRunner:
    """Wrapper for ensemble residual anomaly signal."""

    def __init__(
        self,
        *,
        rp_dims: int,
        alpha: float,
        cv_mode: str,
        kfolds: int,
        split_train_ratio: float,
        random_state: int,
        residual: str,
        fit_intercept: bool,
        members_nystrom: int,
        members_lowrank: int,
        row_subsample: float,
        ranks: list[int] | tuple[int, ...],
        landmark_policy: str,
        landmark_cap: int,
        fusion: str,
        debug_members: bool,
    ) -> None:
        self.metric = DeltaRidgeEnsembleMetric(
            rp_dims=rp_dims,
            alpha=alpha,
            cv_mode=cv_mode,
            kfolds=kfolds,
            split_train_ratio=split_train_ratio,
            random_state=random_state,
            residual=residual,
            fit_intercept=fit_intercept,
            members_nystrom=members_nystrom,
            members_lowrank=members_lowrank,
            row_subsample=row_subsample,
            ranks=ranks,
            landmark_policy=landmark_policy,
            landmark_cap=landmark_cap,
            fusion=fusion,
            debug_members=debug_members,
        )

    def compute(
        self,
        d_raw: np.ndarray,
        knn_context: LocalKNNContext,
        weights: np.ndarray | None,
        input_embs: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, object]]:
        signal = self.metric.compute(
            d_raw,
            knn_context,
            sample_weights=weights,
            X=input_embs,
        )
        meta = dict(getattr(self.metric, "last_meta", {}) or {})
        return signal, meta
