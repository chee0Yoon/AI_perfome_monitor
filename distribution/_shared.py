"""
Shared k-NN context and helper functions for distribution metrics.

Handles pre-computation of k-NN results used by direction, length, and
DiffResidual metrics, plus weighted robust-stat utilities for UDF refinement.
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from final_metric_refactor.shared.geometry import knn_self, pca_fit_transform, sanitize_matrix

try:
    from sklearn.covariance import LedoitWolf as _LedoitWolf
except Exception:
    _LedoitWolf = None

EPS = 1e-9


@dataclass
class LocalKNNContext:
    """Pre-computed k-NN context for local conditional metrics."""

    knn_dist: np.ndarray  # (N, max_k) euclidean distances to neighbors
    knn_idx: np.ndarray  # (N, max_k) neighbor indices
    used_k: np.ndarray  # (N,) adaptive k per point
    x_red: np.ndarray  # (N, D) input coordinates


def choose_k_candidates(n: int, min_k: int, max_k: int) -> tuple[list[int], int]:
    """Generate candidate k values for ensemble scoring."""
    n_eff = max(2, n)
    base = int(round(math.sqrt(n_eff)))
    base = min(max_k, max(min_k, base))
    cands = {
        min(max_k, max(min_k, int(round(base * 0.5)))),
        base,
        min(max_k, max(min_k, int(round(base * 1.5)))),
    }
    cands = sorted(k for k in cands if k < n_eff)
    if not cands:
        cands = [max(1, n_eff - 1)]
    return cands, base


def reduce_for_scoring(
    X: np.ndarray,
    var_target: float,
    min_dims: int,
    max_dims: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Legacy PCA reduction helper (kept for backward compatibility)."""
    n, d = X.shape
    max_allowed = min(max_dims, d, max(1, n - 1))
    if max_allowed <= min_dims:
        return X.copy(), {"used_dims": d, "target_var_reached": None, "pca_first2_var": None}

    Xt_full, evr = pca_fit_transform(X, n_components=max_allowed)
    cum = np.cumsum(evr)
    k = int(np.searchsorted(cum, var_target) + 1)
    k = min(max_allowed, max(min_dims, k))
    Xt = Xt_full[:, :k]
    first2 = float(np.sum(evr[:2])) if len(evr) >= 2 else None
    reached = float(cum[k - 1]) if len(cum) >= k else None
    meta = {"used_dims": int(k), "target_var_reached": reached, "pca_first2_var": first2}
    return Xt, meta


def weighted_quantile(values: np.ndarray, weights: np.ndarray | None, q: float) -> float:
    """Compute a weighted quantile with finite-value filtering."""
    vals = np.asarray(values, dtype=float).ravel()
    if vals.size == 0:
        return float("nan")
    if weights is None:
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return float("nan")
        return float(np.quantile(vals, float(q)))

    w = np.asarray(weights, dtype=float).ravel()
    if w.size != vals.size:
        raise ValueError("weights length mismatch in weighted_quantile")

    mask = np.isfinite(vals) & np.isfinite(w) & (w > 0)
    vals = vals[mask]
    w = w[mask]
    if vals.size == 0:
        return float("nan")

    order = np.argsort(vals)
    vals = vals[order]
    w = w[order]
    cw = np.cumsum(w)
    total = float(cw[-1])
    if total <= 0:
        return float(np.quantile(vals, float(q)))

    # Midpoint CDF (mass centered on each sample) gives stable weighted
    # quantiles and avoids pulling medians too far toward lower bins.
    cdf = (cw - 0.5 * w) / total
    p = float(np.clip(q, 0.0, 1.0))
    if p <= cdf[0]:
        return float(vals[0])
    if p >= cdf[-1]:
        return float(vals[-1])
    return float(np.interp(p, cdf, vals))


def weighted_median_mad(values: np.ndarray, weights: np.ndarray | None) -> tuple[float, float]:
    """Return weighted median and weighted MAD (or unweighted fallback)."""
    vals = np.asarray(values, dtype=float).ravel()
    if vals.size == 0:
        return float("nan"), float("nan")

    med = weighted_quantile(vals, weights, 0.5)
    if not np.isfinite(med):
        return float("nan"), float("nan")

    abs_dev = np.abs(vals - med)
    mad = weighted_quantile(abs_dev, weights, 0.5)
    return float(med), float(mad)


def weighted_robust_z(
    values: np.ndarray,
    ref_mask: np.ndarray,
    ref_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Robust z using median/MAD from a weighted reference subset."""
    vals = np.asarray(values, dtype=float).ravel()
    mask = np.asarray(ref_mask, dtype=bool).ravel()
    if vals.size != mask.size:
        raise ValueError("ref_mask length mismatch in weighted_robust_z")

    if ref_weights is not None:
        w = np.asarray(ref_weights, dtype=float).ravel()
        if w.size != vals.size:
            raise ValueError("ref_weights length mismatch in weighted_robust_z")
    else:
        w = None

    if not mask.any():
        mask = np.ones(vals.size, dtype=bool)

    ref_vals = vals[mask]
    ref_w = w[mask] if w is not None else None
    med, mad = weighted_median_mad(ref_vals, ref_w)
    if not np.isfinite(med):
        med = float(np.nanmedian(vals[np.isfinite(vals)])) if np.isfinite(vals).any() else 0.0
    if not np.isfinite(mad):
        mad = 0.0

    scale = 1.4826 * mad + EPS
    z = (vals - med) / scale
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def _pick_local_k(
    dvec: np.ndarray,
    base_k: int,
    min_k: int,
    gap_ratio: float,
) -> int:
    """Adaptively select k by detecting natural gaps in sorted distances."""
    if len(dvec) == 0:
        return 0
    max_k = len(dvec)
    k = min(max_k, max(min_k, base_k))
    start = max(1, min_k - 1)
    for j in range(start, max_k - 1):
        left = dvec[j]
        right = dvec[j + 1]
        ratio = right / (left + EPS)
        if ratio >= gap_ratio:
            k = j + 1
            break
    return min(max_k, max(1, k))


def _fallback_shrunk_cov(centered: np.ndarray, alpha: float = 0.20) -> np.ndarray:
    """Compute shrunk covariance with adaptive shrinkage for high-dim cases."""
    n, p = centered.shape
    if n <= 1:
        return np.eye(p, dtype=np.float64)

    cov = np.cov(centered, rowvar=False, bias=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]], dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)

    trace = float(np.trace(cov))
    target_scale = trace / max(p, 1)
    target = np.eye(p, dtype=np.float64) * target_scale

    alpha_adapt = max(alpha, min(0.95, p / max(n - 1, 1)))
    shrunk = (1.0 - alpha_adapt) * cov + alpha_adapt * target
    return shrunk


def _weighted_mean(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float).ravel()
    s = float(np.sum(w))
    if s <= 0:
        return np.mean(X, axis=0)
    return (X * w[:, None]).sum(axis=0) / s


def _weighted_shrunk_cov(centered: np.ndarray, weights: np.ndarray, alpha: float = 0.20) -> np.ndarray:
    """Weighted covariance with diagonal shrinkage target."""
    n, p = centered.shape
    if n <= 1:
        return np.eye(p, dtype=np.float64)

    centered = np.nan_to_num(np.asarray(centered, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    w = np.asarray(weights, dtype=float).ravel()
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    sw = float(w.sum())
    if sw <= 0:
        return _fallback_shrunk_cov(centered, alpha=alpha)

    wn = w / sw
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        cov = centered.T @ (centered * wn[:, None])
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    cov = np.asarray(cov, dtype=np.float64)

    trace = float(np.trace(cov))
    target_scale = trace / max(p, 1)
    target = np.eye(p, dtype=np.float64) * target_scale

    # Effective sample size adjustment.
    ess = 1.0 / max(np.sum(wn * wn), EPS)
    alpha_adapt = max(alpha, min(0.95, p / max(ess - 1.0, 1.0)))
    shrunk = (1.0 - alpha_adapt) * cov + alpha_adapt * target
    return shrunk


def local_mahalanobis_score(
    local_d: np.ndarray,
    point_d: np.ndarray,
    cov_shrinkage: float = 0.20,
    local_weights: np.ndarray | None = None,
) -> float:
    """Mahalanobis distance of a point from a (possibly weighted) neighborhood."""
    local_d = sanitize_matrix(local_d).astype(np.float64, copy=False)
    point_d = sanitize_matrix(point_d).astype(np.float64, copy=False)

    if local_d.ndim != 2 or local_d.shape[0] < 3:
        mu = np.mean(local_d, axis=0) if local_d.size else np.zeros_like(point_d)
        sd = np.std(local_d, axis=0) + 1e-6 if local_d.size else np.ones_like(point_d)
        return float(np.sqrt(np.mean(((point_d - mu) / sd) ** 2)))

    use_weighted = local_weights is not None
    if use_weighted:
        lw = np.asarray(local_weights, dtype=float).ravel()
        if lw.size != local_d.shape[0]:
            raise ValueError("local_weights length mismatch in local_mahalanobis_score")
        lw = np.where(np.isfinite(lw) & (lw > 0), lw, 0.0)
        if float(lw.sum()) <= 0:
            use_weighted = False

    if use_weighted:
        mu = _weighted_mean(local_d, lw)
        centered = local_d - mu
        resid = point_d - mu
        cov = _weighted_shrunk_cov(centered, lw, alpha=cov_shrinkage)
    else:
        mu = local_d.mean(axis=0)
        centered = local_d - mu
        resid = point_d - mu
        cov = None
        if _LedoitWolf is not None:
            try:
                lw_model = _LedoitWolf(assume_centered=True)
                lw_model.fit(centered)
                cov = np.asarray(lw_model.covariance_, dtype=np.float64)
            except Exception:
                cov = None
        if cov is None:
            cov = _fallback_shrunk_cov(centered, alpha=cov_shrinkage)

    p = cov.shape[0]
    cov = cov + np.eye(p, dtype=np.float64) * 1e-6
    try:
        chol = np.linalg.cholesky(cov)
        y = np.linalg.solve(chol, resid)
        return float(np.sqrt(np.dot(y, y)))
    except Exception:
        pinv = np.linalg.pinv(cov, hermitian=True)
        val = float(resid @ pinv @ resid)
        return float(np.sqrt(max(val, 0.0)))


def build_local_knn_context(
    x_red: np.ndarray,
    max_k: int,
    min_k: int,
    gap_ratio: float,
) -> LocalKNNContext:
    """Build k-NN context for local conditional metrics."""
    n = x_red.shape[0]
    dists, idx = knn_self(x_red, n_neighbors=max_k, metric="euclidean")

    base_k = int(round(math.sqrt(max(2, n))))
    base_k = min(max_k, max(min_k, base_k))

    used_k = np.zeros(n, dtype=int)
    for i in range(n):
        k_i = _pick_local_k(dists[i], base_k, min_k, gap_ratio)
        used_k[i] = k_i

    return LocalKNNContext(
        knn_dist=dists,
        knn_idx=idx,
        used_k=used_k,
        x_red=x_red,
    )
