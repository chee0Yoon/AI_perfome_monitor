"""
Output Density Metric - k-NN ensemble scoring for output space anomalies.

Detects outputs that are outliers in the embedding space by computing
an ensemble of k-NN density scores at multiple k values.
"""

import numpy as np

from final_metric_refactor.shared.geometry import knn_self, robust_z
from final_metric_refactor.distribution._shared import choose_k_candidates, weighted_robust_z


class OutputDensityMetric:
    """Detects output embeddings that are anomalous in density."""

    def __init__(self, min_k: int = 3, max_k: int = 50):
        self.min_k = min_k
        self.max_k = max_k
        self.used_ks: list[int] = []

    def compute(
        self,
        Y: np.ndarray,
        ref_mask: np.ndarray,
        ref_weights: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[int]]:
        """Compute output density scores using k-NN ensemble."""
        n = len(Y)
        ks, _ = choose_k_candidates(n, min_k=self.min_k, max_k=self.max_k)
        self.used_ks = ks

        z_scores = []
        for k in ks:
            dists, _ = knn_self(Y, n_neighbors=k, metric="euclidean")
            if dists.shape[1] > 0:
                avg_dist = dists.mean(axis=1)
            else:
                avg_dist = np.zeros(n, dtype=float)
            if ref_weights is None:
                z = robust_z(avg_dist, ref_mask=ref_mask)
            else:
                z = weighted_robust_z(avg_dist, ref_mask=ref_mask, ref_weights=ref_weights)
            z_scores.append(z)

        score = np.median(np.vstack(z_scores), axis=0)
        return score, ks
