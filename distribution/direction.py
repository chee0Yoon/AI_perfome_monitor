"""Direction Metric - Angular deviation from local neighborhood mean direction."""

import numpy as np

from final_metric_refactor.distribution._shared import LocalKNNContext, weighted_quantile


class DirectionMetric:
    """Detects outputs with unusual direction relative to local neighborhood."""

    def compute(
        self,
        U: np.ndarray,
        context: LocalKNNContext,
        sample_weights: np.ndarray | None = None,
        tau: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute direction scores.

        Args:
            U: Unit direction vectors of shape (N, D).
            context: LocalKNNContext with pre-computed k-NN results.
            sample_weights: Optional per-row weights.
            tau: Optional movement magnitude per row. When provided, applies
                tau-based scaling to angle/spread score:
                scale = sqrt(clip(tau / median(tau), 0.25, 4.0)).
        """
        n = len(U)
        direction_signal = np.zeros(n)
        weights = None
        if sample_weights is not None:
            weights = np.asarray(sample_weights, dtype=float)
            if len(weights) != n:
                raise ValueError("sample_weights length mismatch in DirectionMetric.compute")

        tau_scale = np.ones(n, dtype=float)
        if tau is not None:
            tau_arr = np.asarray(tau, dtype=float).ravel()
            if len(tau_arr) != n:
                raise ValueError("tau length mismatch in DirectionMetric.compute")
            tau_arr = np.nan_to_num(tau_arr, nan=0.0, posinf=0.0, neginf=0.0)
            tau_arr = np.clip(tau_arr, 0.0, np.inf)
            tau_med = float(np.median(tau_arr))
            if not np.isfinite(tau_med) or tau_med <= 0.0:
                tau_med = 1.0
            tau_rel = tau_arr / (tau_med + 1e-9)
            tau_scale = np.sqrt(np.clip(tau_rel, 0.25, 4.0))

        for i in range(n):
            k_i = context.used_k[i]
            if k_i == 0:
                direction_signal[i] = 0.0
                continue

            neigh_idx = context.knn_idx[i, :k_i]
            local_u = U[neigh_idx]

            if weights is None:
                mu_u = local_u.mean(axis=0)
                local_w = None
            else:
                local_w = np.where(np.isfinite(weights[neigh_idx]) & (weights[neigh_idx] > 0), weights[neigh_idx], 0.0)
                sw = float(local_w.sum())
                if sw <= 0:
                    mu_u = local_u.mean(axis=0)
                    local_w = None
                else:
                    mu_u = (local_u * local_w[:, None]).sum(axis=0) / sw

            mu_u_norm = np.linalg.norm(mu_u)
            if mu_u_norm > 0:
                mu_u = mu_u / mu_u_norm
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    point_sim = float(U[i] @ mu_u)
                    spread_sim = local_u @ mu_u
                point_sim = float(np.nan_to_num(point_sim, nan=0.0, posinf=1.0, neginf=-1.0))
                spread_sim = np.nan_to_num(spread_sim, nan=0.0, posinf=1.0, neginf=-1.0)
                angle = np.arccos(np.clip(point_sim, -1.0, 1.0))
                spreads = np.arccos(np.clip(spread_sim, -1.0, 1.0))
                spread = (
                    weighted_quantile(spreads, local_w, 0.5)
                    if local_w is not None
                    else float(np.median(spreads))
                )
                if not np.isfinite(spread):
                    spread = float(np.median(spreads)) if len(spreads) > 0 else 0.0
                base = angle / (spread + 1e-6)
                direction_signal[i] = base * tau_scale[i]
            else:
                direction_signal[i] = 1.0 * tau_scale[i]

        return direction_signal
