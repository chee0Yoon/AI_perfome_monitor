"""Length Metric - MAD-normalized deviation of displacement magnitude."""

import numpy as np

from final_metric_refactor.distribution._shared import LocalKNNContext, weighted_median_mad


class LengthMetric:
    """Detects outputs with unusual displacement magnitude relative to neighbors."""

    def compute(
        self,
        tau: np.ndarray,
        context: LocalKNNContext,
        sample_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute length scores."""
        n = len(tau)
        length_signal = np.zeros(n)
        weights = None
        if sample_weights is not None:
            weights = np.asarray(sample_weights, dtype=float)
            if len(weights) != n:
                raise ValueError("sample_weights length mismatch in LengthMetric.compute")

        for i in range(n):
            k_i = context.used_k[i]
            if k_i == 0:
                length_signal[i] = 0.0
                continue

            neigh_idx = context.knn_idx[i, :k_i]
            local_tau = tau[neigh_idx]
            local_w = None
            if weights is not None:
                local_w = np.where(np.isfinite(weights[neigh_idx]) & (weights[neigh_idx] > 0), weights[neigh_idx], 0.0)
                if float(local_w.sum()) <= 0:
                    local_w = None

            tau_med, tau_mad = weighted_median_mad(local_tau, local_w)
            if not np.isfinite(tau_med):
                tau_med = float(np.median(local_tau)) if len(local_tau) > 0 else 0.0
            if not np.isfinite(tau_mad):
                tau_mad = float(np.median(np.abs(local_tau - tau_med))) if len(local_tau) > 0 else 0.0

            length_signal[i] = abs(float(tau[i]) - tau_med) / (1.4826 * tau_mad + 1e-6)

        return length_signal
