"""Similar Input Conflict Metric - Detects conflicting outputs for similar inputs."""

import numpy as np

from final_metric_refactor.shared.geometry import knn_self


class SimilarInputConflictMetric:
    """Detects outputs that conflict with similar inputs (high-similarity neighbors)."""

    def __init__(self, similarity_threshold: float = 0.9, similarity_k: int = 50):
        """Initialize the similar input conflict metric.

        Args:
            similarity_threshold: Cosine similarity cutoff for "similar" inputs.
            similarity_k: Number of neighbors to search for similarity.
        """
        self.similarity_threshold = similarity_threshold
        self.similarity_k = similarity_k

    def compute(
        self,
        x_norm: np.ndarray,
        u: np.ndarray,
        tau: np.ndarray,
    ) -> np.ndarray:
        """Compute similar input conflict scores.

        Args:
            x_norm: L2-normalized input embeddings of shape (N, D_x).
            u: Unit direction vectors of shape (N, D).
            tau: Displacement magnitudes of shape (N,).

        Returns:
            Conflict scores of shape (N,).
        """
        dists, idxs = knn_self(x_norm, n_neighbors=self.similarity_k, metric="cosine")

        scores = np.zeros(len(x_norm))
        for i in range(len(x_norm)):
            sim = 1.0 - dists[i]
            neigh = idxs[i][sim >= self.similarity_threshold]
            if len(neigh) < 3:
                scores[i] = 0.0
                continue

            local_u = u[neigh]
            mu_u = local_u.mean(axis=0)
            mu_u_norm = np.linalg.norm(mu_u)
            if mu_u_norm > 0:
                mu_u = mu_u / mu_u_norm
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    point_sim = float(u[i] @ mu_u)
                    spread_sim = local_u @ mu_u
                point_sim = float(np.nan_to_num(point_sim, nan=0.0, posinf=1.0, neginf=-1.0))
                spread_sim = np.nan_to_num(spread_sim, nan=0.0, posinf=1.0, neginf=-1.0)
                angle = np.arccos(np.clip(point_sim, -1.0, 1.0))
                spreads = np.arccos(np.clip(spread_sim, -1.0, 1.0))
                spread = float(np.median(spreads))
                dir_part = angle / (spread + 1e-6)
            else:
                dir_part = 1.0

            local_tau = tau[neigh]
            tau_med = float(np.median(local_tau))
            tau_mad = float(np.median(np.abs(local_tau - tau_med)))
            len_part = abs(float(tau[i]) - tau_med) / (1.4826 * tau_mad + 1e-6)

            scores[i] = dir_part + len_part

        return scores
