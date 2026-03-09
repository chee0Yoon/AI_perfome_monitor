"""Delta-Ridge ensemble metric.

Ensemble composition:
- Nyström kernel ridge learners
- Low-rank delta linear learners

This metric computes out-of-fold residual scores to avoid leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from final_metric_refactor.distribution._shared import LocalKNNContext
from final_metric_refactor.shared.geometry import sanitize_matrix

EPS = 1e-9
COEF_ABS_MAX = 1e6


def _fit_weighted_ridge(
    X: np.ndarray,
    Y: np.ndarray,
    alpha: float,
    weights: np.ndarray | None,
    fit_intercept: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form weighted ridge with optional intercept."""
    X = sanitize_matrix(X).astype(np.float64, copy=False)
    Y = sanitize_matrix(Y).astype(np.float64, copy=False)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
        raise ValueError("X/Y shape mismatch in _fit_weighted_ridge")

    n_samples, n_features = X.shape
    out_dim = Y.shape[1]
    if n_samples == 0:
        return np.zeros((n_features, out_dim), dtype=np.float64), np.zeros(out_dim, dtype=np.float64)

    w = None
    if weights is not None:
        w = np.asarray(weights, dtype=float).ravel()
        if w.size != n_samples:
            raise ValueError("weights length mismatch in _fit_weighted_ridge")
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
        if float(np.sum(w)) <= 0:
            w = None

    if fit_intercept:
        if w is None:
            x_mean = np.mean(X, axis=0)
            y_mean = np.mean(Y, axis=0)
        else:
            sw = float(np.sum(w))
            x_mean = (X * w[:, None]).sum(axis=0) / max(sw, EPS)
            y_mean = (Y * w[:, None]).sum(axis=0) / max(sw, EPS)
        Xc = X - x_mean
        Yc = Y - y_mean
    else:
        x_mean = np.zeros(n_features, dtype=np.float64)
        y_mean = np.zeros(out_dim, dtype=np.float64)
        Xc = X
        Yc = Y

    if w is None:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            xtx = Xc.T @ Xc
            xty = Xc.T @ Yc
    else:
        sw = np.sqrt(w)
        Xw = Xc * sw[:, None]
        Yw = Yc * sw[:, None]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            xtx = Xw.T @ Xw
            xty = Xw.T @ Yw
    xtx = np.nan_to_num(xtx, nan=0.0, posinf=0.0, neginf=0.0)
    xty = np.nan_to_num(xty, nan=0.0, posinf=0.0, neginf=0.0)

    reg = np.eye(n_features, dtype=np.float64) * float(max(alpha, 1e-12))
    system = xtx + reg
    try:
        coef = np.linalg.solve(system, xty)
    except Exception:
        coef = np.linalg.pinv(system, hermitian=True) @ xty
    coef = np.asarray(coef, dtype=np.float64)
    coef = np.nan_to_num(coef, nan=0.0, posinf=0.0, neginf=0.0)
    coef = np.clip(coef, -COEF_ABS_MAX, COEF_ABS_MAX)

    pred_mean = _safe_predict(np.asarray(x_mean, dtype=np.float64)[None, :], coef, None)[0]
    bias = np.asarray(y_mean - pred_mean, dtype=np.float64)
    bias = np.nan_to_num(bias, nan=0.0, posinf=0.0, neginf=0.0)
    if not fit_intercept:
        bias = np.zeros(out_dim, dtype=np.float64)
    return coef, bias


def _safe_predict(X: np.ndarray, coef: np.ndarray, intercept: np.ndarray | None = None) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        pred = X @ coef
    if intercept is not None:
        pred = pred + intercept[None, :]
    return np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)


def _cosine_residual(Y: np.ndarray, Y_hat: np.ndarray) -> np.ndarray:
    Y = sanitize_matrix(Y).astype(np.float64, copy=False)
    Y_hat = sanitize_matrix(Y_hat).astype(np.float64, copy=False)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        dot = np.sum(Y * Y_hat, axis=1)
    yn = np.linalg.norm(Y, axis=1)
    yhn = np.linalg.norm(Y_hat, axis=1)
    denom = yn * yhn + EPS
    cos = np.clip(dot / denom, -1.0, 1.0)
    resid = 1.0 - cos
    y_zero = yn <= 1e-12
    yh_zero = yhn <= 1e-12
    resid[y_zero & yh_zero] = 0.0
    resid[y_zero ^ yh_zero] = 1.0
    return np.asarray(resid, dtype=float)


def _sample_projection(dim: int, dims: int, rng: np.random.Generator) -> np.ndarray | None:
    d = int(dim)
    k = int(max(1, min(int(dims), d)))
    if k >= d:
        return None
    proj = rng.standard_normal((d, k), dtype=np.float64)
    norm = np.linalg.norm(proj, axis=0, keepdims=True)
    norm = np.where(norm <= 1e-12, 1.0, norm)
    return (proj / norm) / np.sqrt(float(k))


def _apply_projection(X: np.ndarray, proj: np.ndarray | None) -> np.ndarray:
    if proj is None:
        return np.asarray(X, dtype=np.float64, copy=False)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        xr = X @ proj
    xr = np.nan_to_num(xr, nan=0.0, posinf=0.0, neginf=0.0)
    return xr.astype(np.float64, copy=False)


def _rbf_kernel(A: np.ndarray, B: np.ndarray, gamma: float) -> np.ndarray:
    A = sanitize_matrix(A).astype(np.float64, copy=False)
    B = sanitize_matrix(B).astype(np.float64, copy=False)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        an = np.sum(A * A, axis=1, keepdims=True)
        bn = np.sum(B * B, axis=1, keepdims=True).T
        d2 = an + bn - 2.0 * (A @ B.T)
    d2 = np.maximum(np.nan_to_num(d2, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        K = np.exp(-float(max(gamma, 1e-12)) * d2)
    return np.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)


def _median_mad(values: np.ndarray) -> tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    return med, max(mad, 1e-12)


@dataclass
class _MemberResult:
    test_score: np.ndarray
    train_med: float
    train_mad: float
    reliability: float
    member_key: str


class DeltaRidgeEnsembleMetric:
    """Nyström + low-rank delta residual ensemble."""

    def __init__(
        self,
        rp_dims: int = 96,
        alpha: float = 0.1,
        cv_mode: str = "auto",
        kfolds: int = 5,
        split_train_ratio: float = 0.8,
        random_state: int = 42,
        residual: str = "l2",
        fit_intercept: bool = True,
        members_nystrom: int = 4,
        members_lowrank: int = 4,
        row_subsample: float = 0.85,
        ranks: list[int] | tuple[int, ...] = (16, 32),
        landmark_policy: str = "sqrt",
        landmark_cap: int = 512,
        fusion: str = "robust_z_weighted",
        debug_members: bool = True,
    ):
        allowed_cv = {"auto", "loo", "kfold", "split"}
        cv_mode_norm = str(cv_mode).strip().lower()
        if cv_mode_norm not in allowed_cv:
            raise ValueError(f"Unsupported delta_ens cv_mode: {cv_mode}")

        allowed_residual = {"l2", "cosine"}
        residual_norm = str(residual).strip().lower()
        if residual_norm not in allowed_residual:
            raise ValueError(f"Unsupported delta_ens residual: {residual}")

        allowed_landmark_policy = {"sqrt", "fixed"}
        landmark_policy_norm = str(landmark_policy).strip().lower()
        if landmark_policy_norm not in allowed_landmark_policy:
            raise ValueError(f"Unsupported delta_ens landmark policy: {landmark_policy}")

        allowed_fusion = {"robust_z_weighted", "mean", "max"}
        fusion_norm = str(fusion).strip().lower()
        if fusion_norm not in allowed_fusion:
            raise ValueError(f"Unsupported delta_ens fusion: {fusion}")

        parsed_ranks: list[int] = []
        for r in ranks:
            try:
                rr = int(r)
            except Exception:
                continue
            if rr > 0:
                parsed_ranks.append(rr)
        if not parsed_ranks:
            parsed_ranks = [16, 32]

        self.rp_dims = int(max(1, rp_dims))
        self.alpha = float(max(1e-12, alpha))
        self.cv_mode = cv_mode_norm
        self.kfolds = int(max(2, kfolds))
        self.split_train_ratio = float(np.clip(split_train_ratio, 0.05, 0.95))
        self.random_state = int(random_state)
        self.residual = residual_norm
        self.fit_intercept = bool(fit_intercept)
        self.members_nystrom = int(max(0, members_nystrom))
        self.members_lowrank = int(max(0, members_lowrank))
        self.row_subsample = float(np.clip(row_subsample, 0.2, 1.0))
        self.ranks = parsed_ranks
        self.landmark_policy = landmark_policy_norm
        self.landmark_cap = int(max(8, landmark_cap))
        self.fusion = fusion_norm
        self.debug_members = bool(debug_members)
        self.last_meta: dict[str, Any] = {}

    def _resolve_cv_mode(self, n_samples: int) -> str:
        if self.cv_mode != "auto":
            return self.cv_mode
        if n_samples <= 80:
            return "loo"
        if n_samples <= 5000:
            return "kfold"
        return "split"

    def _estimate_gamma(self, X: np.ndarray, rng: np.random.Generator) -> float:
        n = X.shape[0]
        if n <= 1:
            return 1.0 / max(X.shape[1], 1)
        s = min(256, n)
        idx = rng.choice(n, size=s, replace=False) if s < n else np.arange(n, dtype=int)
        xs = X[idx]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            x2 = np.sum(xs * xs, axis=1, keepdims=True)
            d2 = x2 + x2.T - 2.0 * (xs @ xs.T)
        d2 = np.maximum(np.nan_to_num(d2, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        tri = d2[np.triu_indices_from(d2, k=1)]
        tri = tri[np.isfinite(tri)]
        med = float(np.median(tri)) if tri.size > 0 else 0.0
        if med <= 1e-12:
            return 1.0 / max(X.shape[1], 1)
        return float(1.0 / (2.0 * med + EPS))

    def _choose_landmarks(self, n_rows: int, rng: np.random.Generator) -> np.ndarray:
        if self.landmark_policy == "fixed":
            m = min(self.landmark_cap, n_rows)
        else:
            m = min(self.landmark_cap, int(np.ceil(np.sqrt(max(n_rows, 1)))))
        m = int(max(2, min(m, n_rows)))
        if m >= n_rows:
            return np.arange(n_rows, dtype=int)
        return np.asarray(rng.choice(n_rows, size=m, replace=False), dtype=int)

    def _member_subsample(
        self,
        n_rows: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if n_rows <= 2 or self.row_subsample >= 0.999:
            return np.arange(n_rows, dtype=int)
        m = int(round(float(n_rows) * self.row_subsample))
        m = int(max(2, min(m, n_rows)))
        return np.asarray(rng.choice(n_rows, size=m, replace=False), dtype=int)

    def _score_residual(self, Y_true: np.ndarray, Y_hat: np.ndarray) -> np.ndarray:
        if self.residual == "cosine":
            return _cosine_residual(Y_true, Y_hat)
        return np.linalg.norm(Y_true - Y_hat, axis=1)

    def _fit_nystrom_member(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        train_weights: np.ndarray | None,
        rng: np.random.Generator,
        member_key: str,
    ) -> _MemberResult:
        landmark_idx = self._choose_landmarks(X_train.shape[0], rng)
        landmarks = X_train[landmark_idx]
        gamma = self._estimate_gamma(X_train, rng=rng)

        k_mm = _rbf_kernel(landmarks, landmarks, gamma=gamma)
        k_mm = k_mm + np.eye(k_mm.shape[0], dtype=np.float64) * 1e-6
        try:
            eigvals, eigvecs = np.linalg.eigh(k_mm)
            eigvals = np.clip(np.asarray(eigvals, dtype=np.float64), 1e-9, None)
            inv_sqrt = (eigvecs * (1.0 / np.sqrt(eigvals))[None, :]) @ eigvecs.T
        except Exception:
            inv_sqrt = np.linalg.pinv(k_mm, hermitian=True)
            inv_sqrt = np.nan_to_num(inv_sqrt, nan=0.0, posinf=0.0, neginf=0.0)

        phi_train = _rbf_kernel(X_train, landmarks, gamma=gamma) @ inv_sqrt
        phi_test = _rbf_kernel(X_test, landmarks, gamma=gamma) @ inv_sqrt

        coef, bias = _fit_weighted_ridge(
            phi_train,
            Y_train,
            alpha=self.alpha,
            weights=train_weights,
            fit_intercept=self.fit_intercept,
        )
        y_hat_test = _safe_predict(phi_test, coef, bias)
        test_score = self._score_residual(Y_test, y_hat_test)

        y_hat_train = _safe_predict(phi_train, coef, bias)
        train_score = self._score_residual(Y_train, y_hat_train)
        med, mad = _median_mad(train_score)
        reliability = 1.0 / (mad + EPS)
        return _MemberResult(
            test_score=np.asarray(test_score, dtype=float),
            train_med=float(med),
            train_mad=float(mad),
            reliability=float(reliability),
            member_key=member_key,
        )

    def _fit_lowrank_member(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        train_weights: np.ndarray | None,
        rank: int,
        member_key: str,
    ) -> _MemberResult:
        coef, bias = _fit_weighted_ridge(
            X_train,
            Y_train,
            alpha=self.alpha,
            weights=train_weights,
            fit_intercept=self.fit_intercept,
        )
        coef = np.nan_to_num(np.asarray(coef, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        coef = np.clip(coef, -COEF_ABS_MAX, COEF_ABS_MAX)
        max_rank = int(min(coef.shape[0], coef.shape[1]))
        use_rank = int(max(1, min(rank, max_rank)))
        if use_rank < max_rank:
            try:
                u, s, vt = np.linalg.svd(coef, full_matrices=False)
                s = np.nan_to_num(np.asarray(s, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
                s = np.clip(s, 0.0, COEF_ABS_MAX)
                with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
                    coef_lr = (u[:, :use_rank] * s[:use_rank][None, :]) @ vt[:use_rank, :]
                coef_lr = np.nan_to_num(coef_lr, nan=0.0, posinf=0.0, neginf=0.0)
            except Exception:
                coef_lr = coef
        else:
            coef_lr = coef

        y_hat_test = _safe_predict(X_test, coef_lr, bias)
        test_score = self._score_residual(Y_test, y_hat_test)

        y_hat_train = _safe_predict(X_train, coef_lr, bias)
        train_score = self._score_residual(Y_train, y_hat_train)
        med, mad = _median_mad(train_score)
        reliability = 1.0 / (mad + EPS)
        return _MemberResult(
            test_score=np.asarray(test_score, dtype=float),
            train_med=float(med),
            train_mad=float(mad),
            reliability=float(reliability),
            member_key=member_key,
        )

    def _fuse_members(self, members: list[_MemberResult]) -> tuple[np.ndarray, list[dict[str, Any]]]:
        if not members:
            return np.zeros(0, dtype=float), []
        n = members[0].test_score.shape[0]
        score_mat = np.zeros((len(members), n), dtype=np.float64)
        weights = np.zeros(len(members), dtype=np.float64)
        member_meta: list[dict[str, Any]] = []
        for i, m in enumerate(members):
            z = (m.test_score - m.train_med) / (1.4826 * m.train_mad + EPS)
            z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
            z = np.maximum(z, 0.0)
            score_mat[i] = z
            weights[i] = max(m.reliability, 1e-6)
            member_meta.append(
                {
                    "member_key": m.member_key,
                    "train_med": float(m.train_med),
                    "train_mad": float(m.train_mad),
                    "reliability": float(m.reliability),
                }
            )

        if self.fusion == "max":
            fused = np.max(score_mat, axis=0)
            return fused.astype(float), member_meta
        if self.fusion == "mean":
            fused = np.mean(score_mat, axis=0)
            return fused.astype(float), member_meta

        sw = float(np.sum(weights))
        if sw <= 0:
            fused = np.mean(score_mat, axis=0)
        else:
            fused = (weights[:, None] * score_mat).sum(axis=0) / sw
        return fused.astype(float), member_meta

    def _fit_predict_fold(
        self,
        X: np.ndarray,
        D: np.ndarray,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        sample_weights: np.ndarray | None,
        fold_seed: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
        x_train_full = X[train_idx]
        d_train_full = D[train_idx]
        x_test_full = X[test_idx]
        d_test_full = D[test_idx]
        w_train_full = sample_weights[train_idx] if sample_weights is not None else None

        nys_members: list[_MemberResult] = []
        low_members: list[_MemberResult] = []
        debug_scores: dict[str, np.ndarray] = {}
        rng_base = np.random.default_rng(int(fold_seed))

        for m in range(self.members_nystrom):
            mrng = np.random.default_rng(int(rng_base.integers(1, 10_000_000)))
            sub_idx = self._member_subsample(len(train_idx), mrng)
            x_train = x_train_full[sub_idx]
            d_train = d_train_full[sub_idx]
            w_train = w_train_full[sub_idx] if w_train_full is not None else None

            x_rng = np.random.default_rng(int(rng_base.integers(1, 10_000_000)))
            d_rng = np.random.default_rng(int(rng_base.integers(1, 10_000_000)))
            x_proj = _sample_projection(x_train.shape[1], self.rp_dims, x_rng)
            d_proj = _sample_projection(d_train.shape[1], self.rp_dims, d_rng)
            x_train_rp = _apply_projection(x_train, x_proj)
            x_test_rp = _apply_projection(x_test_full, x_proj)
            d_train_rp = _apply_projection(d_train, d_proj)
            d_test_rp = _apply_projection(d_test_full, d_proj)

            member_key = f"nystrom_{m}"
            res = self._fit_nystrom_member(
                X_train=x_train_rp,
                Y_train=d_train_rp,
                X_test=x_test_rp,
                Y_test=d_test_rp,
                train_weights=w_train,
                rng=mrng,
                member_key=member_key,
            )
            nys_members.append(res)
            if self.debug_members:
                debug_scores[member_key] = res.test_score.copy()

        for m in range(self.members_lowrank):
            mrng = np.random.default_rng(int(rng_base.integers(1, 10_000_000)))
            sub_idx = self._member_subsample(len(train_idx), mrng)
            x_train = x_train_full[sub_idx]
            d_train = d_train_full[sub_idx]
            w_train = w_train_full[sub_idx] if w_train_full is not None else None

            x_rng = np.random.default_rng(int(rng_base.integers(1, 10_000_000)))
            d_rng = np.random.default_rng(int(rng_base.integers(1, 10_000_000)))
            x_proj = _sample_projection(x_train.shape[1], self.rp_dims, x_rng)
            d_proj = _sample_projection(d_train.shape[1], self.rp_dims, d_rng)
            x_train_rp = _apply_projection(x_train, x_proj)
            x_test_rp = _apply_projection(x_test_full, x_proj)
            d_train_rp = _apply_projection(d_train, d_proj)
            d_test_rp = _apply_projection(d_test_full, d_proj)

            rank = int(self.ranks[m % len(self.ranks)])
            member_key = f"lowrank_{m}_r{rank}"
            res = self._fit_lowrank_member(
                X_train=x_train_rp,
                Y_train=d_train_rp,
                X_test=x_test_rp,
                Y_test=d_test_rp,
                train_weights=w_train,
                rank=rank,
                member_key=member_key,
            )
            low_members.append(res)
            if self.debug_members:
                debug_scores[member_key] = res.test_score.copy()

        nys_score, nys_meta = self._fuse_members(nys_members)
        low_score, low_meta = self._fuse_members(low_members)

        if nys_score.size == 0 and low_score.size == 0:
            final = np.zeros(len(test_idx), dtype=float)
        elif nys_score.size == 0:
            final = low_score
        elif low_score.size == 0:
            final = nys_score
        else:
            final = 0.5 * nys_score + 0.5 * low_score

        fold_meta = {
            "nystrom_member_meta": nys_meta,
            "lowrank_member_meta": low_meta,
            "nystrom_count": int(len(nys_members)),
            "lowrank_count": int(len(low_members)),
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
        }
        return np.asarray(final, dtype=float), np.asarray(nys_score, dtype=float), np.asarray(low_score, dtype=float), debug_scores, fold_meta

    def _compute_oof(
        self,
        X: np.ndarray,
        D: np.ndarray,
        sample_weights: np.ndarray | None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        X = sanitize_matrix(X).astype(np.float64, copy=False)
        D = sanitize_matrix(D).astype(np.float64, copy=False)
        n = X.shape[0]

        if n < 3:
            score = np.linalg.norm(D, axis=1)
            meta = {
                "method": "delta_ridge_ens",
                "family": "nystrom_lowrank",
                "cv_mode": "insufficient_data",
                "rp_dims": int(min(self.rp_dims, X.shape[1])),
                "target_dims": int(D.shape[1]),
                "alpha": float(self.alpha),
                "residual_norm": self.residual,
                "fit_intercept": bool(self.fit_intercept),
                "members_nystrom": int(self.members_nystrom),
                "members_lowrank": int(self.members_lowrank),
                "members_total": int(self.members_nystrom + self.members_lowrank),
                "row_subsample": float(self.row_subsample),
                "ranks": ",".join(str(x) for x in self.ranks),
                "landmark_policy": self.landmark_policy,
                "landmark_cap": int(self.landmark_cap),
                "fusion": self.fusion,
                "debug_members": bool(self.debug_members),
            }
            return np.asarray(score, dtype=float), meta

        cv_mode = self._resolve_cv_mode(n)
        score = np.zeros(n, dtype=float)
        nys_score_all = np.zeros(n, dtype=float)
        low_score_all = np.zeros(n, dtype=float)
        fold_metas: list[dict[str, Any]] = []
        debug_member_all: dict[str, np.ndarray] = {}

        rng = np.random.default_rng(self.random_state)
        if cv_mode == "loo":
            folds = [(np.array([i], dtype=int), i) for i in range(n)]
        elif cv_mode == "kfold":
            k = int(max(2, min(self.kfolds, n)))
            idx = np.arange(n, dtype=int)
            rng.shuffle(idx)
            fold_idx = [fold for fold in np.array_split(idx, k) if len(fold) > 0]
            folds = [(test_idx.astype(int), i) for i, test_idx in enumerate(fold_idx)]
        else:
            perm = rng.permutation(n)
            n_train = int(round(n * self.split_train_ratio))
            n_train = int(min(max(1, n_train), n - 1))
            train_idx = perm[:n_train]
            val_idx = perm[n_train:]
            folds = [(val_idx.astype(int), 0), (train_idx.astype(int), 1)]

        for test_idx, fold_id in folds:
            train_mask = np.ones(n, dtype=bool)
            train_mask[test_idx] = False
            if int(train_mask.sum()) < 2:
                train_mask[:] = True
                train_mask[test_idx] = False
            train_idx = np.where(train_mask)[0]
            if len(train_idx) < 2:
                score[test_idx] = np.linalg.norm(D[test_idx], axis=1)
                continue

            fold_seed = int(self.random_state + 97 * (fold_id + 1))
            fold_score, fold_nys, fold_low, fold_debug, fold_meta = self._fit_predict_fold(
                X=X,
                D=D,
                train_idx=train_idx,
                test_idx=test_idx,
                sample_weights=sample_weights,
                fold_seed=fold_seed,
            )
            score[test_idx] = fold_score
            if fold_nys.size == len(test_idx):
                nys_score_all[test_idx] = fold_nys
            if fold_low.size == len(test_idx):
                low_score_all[test_idx] = fold_low
            fold_metas.append(fold_meta)

            if self.debug_members and fold_debug:
                for mk, mv in fold_debug.items():
                    if mk not in debug_member_all:
                        debug_member_all[mk] = np.full(n, np.nan, dtype=float)
                    debug_member_all[mk][test_idx] = mv

        meta: dict[str, Any] = {
            "method": "delta_ridge_ens",
            "family": "nystrom_lowrank",
            "cv_mode": cv_mode,
            "rp_dims": int(min(self.rp_dims, X.shape[1])),
            "target_dims": int(D.shape[1]),
            "alpha": float(self.alpha),
            "residual_norm": self.residual,
            "fit_intercept": bool(self.fit_intercept),
            "members_nystrom": int(self.members_nystrom),
            "members_lowrank": int(self.members_lowrank),
            "members_total": int(self.members_nystrom + self.members_lowrank),
            "row_subsample": float(self.row_subsample),
            "ranks": ",".join(str(x) for x in self.ranks),
            "landmark_policy": self.landmark_policy,
            "landmark_cap": int(self.landmark_cap),
            "fusion": self.fusion,
            "debug_members": bool(self.debug_members),
            "folds": int(len(folds)),
            "family_nystrom_mean": float(np.nanmean(nys_score_all)) if np.isfinite(nys_score_all).any() else np.nan,
            "family_lowrank_mean": float(np.nanmean(low_score_all)) if np.isfinite(low_score_all).any() else np.nan,
        }
        if self.debug_members:
            for mk, mv in sorted(debug_member_all.items()):
                meta[f"member_signal_{mk}"] = mv.tolist()
        if fold_metas:
            meta["fold_meta"] = fold_metas
        return np.asarray(score, dtype=float), meta

    def compute(
        self,
        D: np.ndarray,
        context: LocalKNNContext,
        sample_weights: np.ndarray | None = None,
        X: np.ndarray | None = None,
    ) -> np.ndarray:
        D = sanitize_matrix(D).astype(np.float64, copy=False)
        n = len(D)
        weights = None
        if sample_weights is not None:
            weights = np.asarray(sample_weights, dtype=float).ravel()
            if len(weights) != n:
                raise ValueError("sample_weights length mismatch in DeltaRidgeEnsembleMetric.compute")
            weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)

        X_use = context.x_red if X is None else X
        X_use = sanitize_matrix(X_use).astype(np.float64, copy=False)
        if X_use.ndim != 2 or D.ndim != 2 or X_use.shape[0] != D.shape[0]:
            raise ValueError("X/D shape mismatch in DeltaRidgeEnsembleMetric.compute")

        score, meta = self._compute_oof(X_use, D, sample_weights=weights)
        self.last_meta = meta
        return score
