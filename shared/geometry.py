"""
Geometry and math utilities for embedding-based anomaly detection.

Provides core linear algebra functions for PCA, k-NN, normalization, and
robust statistical computations used across metrics.
"""

import numpy as np

try:
    from sklearn.decomposition import PCA as _SklearnPCA
except Exception:
    _SklearnPCA = None

EPS = 1e-9


def normalize_rows(X: np.ndarray) -> np.ndarray:
    """L2-normalize rows of a matrix (in-place modification).

    Args:
        X: Array of shape (N, D).

    Returns:
        L2-normalized array, same shape. Rows with near-zero norm are left as 1.0.
    """
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.nan_to_num(norms, nan=1.0, posinf=1.0, neginf=1.0)
    norms = np.where(norms <= 1e-12, 1.0, norms)
    return X / norms


def sanitize_matrix(X: np.ndarray) -> np.ndarray:
    """Convert to float and replace non-finite values with 0.

    Args:
        X: Input array.

    Returns:
        Float array with all non-finite values replaced by 0.
    """
    X = np.asarray(X)
    if not np.issubdtype(X.dtype, np.floating):
        X = X.astype(np.float64)
    if not np.all(np.isfinite(X)):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def pca_fit_transform(X: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    """PCA dimensionality reduction.

    Args:
        X: Input array of shape (N, D).
        n_components: Number of PCA components to extract.

    Returns:
        (Xt, evr) where:
        - Xt: transformed array of shape (N, min(n_components, D, N-1)) as float32
        - evr: explained variance ratio array
    """
    X = sanitize_matrix(X)
    if X.shape[0] <= 1:
        Xt = np.zeros((X.shape[0], 1), dtype=np.float32)
        return Xt, np.zeros(1, dtype=float)

    n_components = max(1, min(n_components, X.shape[1], max(1, X.shape[0] - 1)))

    if _SklearnPCA is not None:
        pca = _SklearnPCA(n_components=n_components)
        Xt = pca.fit_transform(X).astype(np.float32, copy=False)
        evr = np.asarray(pca.explained_variance_ratio_, dtype=float)
        return Xt, evr

    # Fallback: numpy-only SVD-based PCA
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    r = min(n_components, U.shape[1])
    Xt = (U[:, :r] * S[:r]).astype(np.float32, copy=False)
    if X.shape[0] <= 1:
        return Xt, np.zeros(r, dtype=float)
    total_var = float(np.sum(S**2) / (X.shape[0] - 1) + EPS)
    explained = (S[:r] ** 2) / (X.shape[0] - 1)
    evr = np.asarray(explained / total_var, dtype=float)
    return Xt, evr


def ensure_2d_coords(X: np.ndarray) -> np.ndarray:
    """Ensure output is exactly (N, 2) for 2D scatter plotting.

    Args:
        X: Input array of any shape.

    Returns:
        Array of shape (N, 2) as float.
    """
    X = sanitize_matrix(X)
    n = X.shape[0]
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[1] >= 2:
        return X[:, :2]
    if X.shape[1] == 1:
        return np.hstack([X, np.zeros((n, 1), dtype=X.dtype)])
    return np.zeros((n, 2), dtype=float)


def knn_self(
    X: np.ndarray,
    n_neighbors: int,
    metric: str = "euclidean",
) -> tuple[np.ndarray, np.ndarray]:
    """Find k-nearest neighbors in a point cloud (self-query, excluding self).

    Args:
        X: Input array of shape (N, D).
        n_neighbors: Number of neighbors to find per point.
        metric: Distance metric ("euclidean" or "cosine").

    Returns:
        (dist_out, idx_out) where:
        - dist_out: shape (N, k), distances to neighbors (float32)
        - idx_out: shape (N, k), indices of neighbors (int64)

    Raises:
        ValueError: If metric is not supported.
    """
    X = sanitize_matrix(X).astype(np.float64, copy=False)
    n = X.shape[0]
    if n <= 1:
        return np.empty((n, 0), dtype=np.float32), np.empty((n, 0), dtype=np.int64)

    k = min(max(1, int(n_neighbors)), n - 1)
    chunk = max(128, min(1024, 32768 // max(1, X.shape[1])))

    idx_out = np.empty((n, k), dtype=np.int64)
    dist_out = np.empty((n, k), dtype=np.float32)

    if metric == "euclidean":
        x_norm = np.sum(X * X, axis=1)
    elif metric == "cosine":
        x_norm = None
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        Q = X[start:end]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if metric == "euclidean":
                q_norm = np.sum(Q * Q, axis=1, keepdims=True)
                d2 = q_norm + x_norm[None, :] - 2.0 * (Q @ X.T)
                np.maximum(d2, 0.0, out=d2)
                D = np.sqrt(d2, out=d2)
            else:
                sim = Q @ X.T
                D = 1.0 - sim
        D = np.nan_to_num(D, nan=np.inf, posinf=np.inf, neginf=np.inf)

        row_idx = np.arange(start, end)
        D[np.arange(end - start), row_idx] = np.inf

        part = np.argpartition(D, kth=k - 1, axis=1)[:, :k]
        part_dist = np.take_along_axis(D, part, axis=1)
        order = np.argsort(part_dist, axis=1)
        idx_sorted = np.take_along_axis(part, order, axis=1)
        dist_sorted = np.take_along_axis(part_dist, order, axis=1)

        idx_out[start:end] = idx_sorted
        dist_out[start:end] = dist_sorted.astype(np.float32, copy=False)

    return dist_out, idx_out


def robust_z(values: np.ndarray, ref_mask: np.ndarray) -> np.ndarray:
    """Robust z-score using median and MAD (median absolute deviation).

    Args:
        values: Input array of shape (N,).
        ref_mask: Boolean mask of shape (N,) selecting reference population.
                 If all False, uses all values.

    Returns:
        Z-score array of same shape as values.
    """
    ref = values[ref_mask] if ref_mask.any() else values
    med = np.median(ref)
    mad = np.median(np.abs(ref - med))
    scale = 1.4826 * mad + EPS
    return (values - med) / scale


def robust_z_from_reference(values: np.ndarray, ref_values: np.ndarray) -> np.ndarray:
    """Robust z-score computed from a separate reference population.

    Args:
        values: Values to z-score.
        ref_values: Reference population for computing median/MAD.

    Returns:
        Z-score array of same shape as values.
    """
    med = float(np.median(ref_values))
    mad = float(np.median(np.abs(ref_values - med)))
    scale = 1.4826 * mad + EPS
    return (values - med) / scale


def safe_quantile(values: np.ndarray, ref_mask: np.ndarray, q: float) -> float:
    """Compute quantile over a reference subset.

    Args:
        values: Input array.
        ref_mask: Boolean mask selecting reference population.
        q: Quantile level in [0, 1].

    Returns:
        Scalar quantile value.
    """
    ref = values[ref_mask] if ref_mask.any() else values
    return float(np.quantile(ref, q))
