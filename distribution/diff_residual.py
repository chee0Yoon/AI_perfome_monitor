"""Diff residual metric (local Mahalanobis + optional aux residual boost)."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np

from final_metric_refactor.distribution._shared import LocalKNNContext, local_mahalanobis_score
from final_metric_refactor.shared.geometry import sanitize_matrix


class DiffResidualMetric:
    """Detect unusual embedding deltas with local Mahalanobis distance."""

    def __init__(
        self,
        cov_shrinkage: float = 0.20,
        aux_enabled: bool = False,
        aux_lambda: float = 0.7,
        aux_model: str = "linear",
        row_chunk_size: int = 256,
        max_workers: int = 0,
        parallel_min_rows: int = 256,
    ):
        self.cov_shrinkage = float(cov_shrinkage)
        self.aux_enabled = bool(aux_enabled)
        self.aux_lambda = float(max(0.0, aux_lambda))
        self.aux_model = str(aux_model).strip().lower()
        if self.aux_model not in {"linear", "poly2"}:
            raise ValueError(f"Unsupported diff residual aux_model: {aux_model}")
        self.row_chunk_size = int(max(1, row_chunk_size))
        self.max_workers = int(max(0, max_workers))
        self.parallel_min_rows = int(max(1, parallel_min_rows))
        self.last_meta: dict[str, Any] = {}

    def _resolve_workers(self, n_rows: int) -> int:
        if self.max_workers > 0:
            return min(self.max_workers, max(1, n_rows))
        cpu = os.cpu_count() or 1
        return min(8, max(1, cpu), max(1, n_rows))

    def _compute_local_mahal_range(
        self,
        D: np.ndarray,
        context: LocalKNNContext,
        weights: np.ndarray | None,
        start: int,
        end: int,
    ) -> tuple[int, np.ndarray]:
        scores = np.zeros(end - start, dtype=float)
        for offset, i in enumerate(range(start, end)):
            k_i = int(context.used_k[i])
            if k_i <= 0:
                scores[offset] = 0.0
                continue
            neigh_idx = context.knn_idx[i, :k_i]
            local_d = D[neigh_idx]
            local_w = None if weights is None else weights[neigh_idx]
            scores[offset] = local_mahalanobis_score(
                local_d=local_d,
                point_d=D[i],
                cov_shrinkage=self.cov_shrinkage,
                local_weights=local_w,
            )
        return start, scores

    def _build_aux_features(self, direction_signal: np.ndarray, length_signal: np.ndarray) -> np.ndarray:
        g = np.asarray(direction_signal, dtype=np.float64).ravel()
        l = np.asarray(length_signal, dtype=np.float64).ravel()
        if self.aux_model == "linear":
            return np.column_stack([np.ones(len(g), dtype=np.float64), g, l])
        return np.column_stack(
            [
                np.ones(len(g), dtype=np.float64),
                g,
                l,
                g * l,
                g * g,
                l * l,
            ]
        )

    def compute(
        self,
        D: np.ndarray,
        context: LocalKNNContext,
        sample_weights: np.ndarray | None = None,
        X: np.ndarray | None = None,
        direction_signal: np.ndarray | None = None,
        length_signal: np.ndarray | None = None,
        ref_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        _ = X
        D = sanitize_matrix(D).astype(np.float64, copy=False)
        n = len(D)
        weights = None
        if sample_weights is not None:
            weights = np.asarray(sample_weights, dtype=float).ravel()
            if len(weights) != n:
                raise ValueError("sample_weights length mismatch in DiffResidualMetric.compute")
            weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)

        workers = self._resolve_workers(n)
        chunk_size = int(max(1, self.row_chunk_size))
        parallel_enabled = bool(n >= self.parallel_min_rows and workers > 1)

        diff_residual_signal = np.zeros(n, dtype=float)
        if parallel_enabled:
            futures = []
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for start in range(0, n, chunk_size):
                    end = min(n, start + chunk_size)
                    futures.append(
                        executor.submit(
                            self._compute_local_mahal_range,
                            D,
                            context,
                            weights,
                            start,
                            end,
                        )
                    )
                for fut in as_completed(futures):
                    start, scores = fut.result()
                    diff_residual_signal[start : start + len(scores)] = scores
        else:
            _, scores = self._compute_local_mahal_range(
                D=D,
                context=context,
                weights=weights,
                start=0,
                end=n,
            )
            diff_residual_signal[:] = scores

        signal_out = diff_residual_signal.copy()
        aux_applied = False
        aux_reason = ""
        aux_ref_size = 0
        aux_mad = float("nan")

        if self.aux_enabled:
            if direction_signal is None or length_signal is None:
                aux_reason = "missing_direction_or_length"
            else:
                g = np.asarray(direction_signal, dtype=np.float64).ravel()
                l = np.asarray(length_signal, dtype=np.float64).ravel()
                if len(g) != n or len(l) != n:
                    aux_reason = "direction_or_length_length_mismatch"
                else:
                    fit_mask = np.ones(n, dtype=bool) if ref_mask is None else np.asarray(ref_mask, dtype=bool).ravel()
                    if len(fit_mask) != n:
                        fit_mask = np.ones(n, dtype=bool)
                    fit_mask &= np.isfinite(diff_residual_signal) & np.isfinite(g) & np.isfinite(l)
                    if weights is not None:
                        fit_mask &= weights > 0

                    aux_ref_size = int(fit_mask.sum())
                    min_ref = 20 if self.aux_model == "linear" else 30
                    if aux_ref_size < min_ref:
                        aux_reason = f"insufficient_reference({aux_ref_size})"
                    else:
                        feats = self._build_aux_features(g, l)
                        y_ref = diff_residual_signal[fit_mask]
                        x_ref = feats[fit_mask]
                        if weights is not None:
                            w_ref = np.sqrt(np.clip(weights[fit_mask], 1e-12, np.inf))
                            x_fit = x_ref * w_ref[:, None]
                            y_fit = y_ref * w_ref
                        else:
                            x_fit = x_ref
                            y_fit = y_ref

                        try:
                            coef, *_ = np.linalg.lstsq(x_fit, y_fit, rcond=None)
                            pred = feats @ coef
                            resid = diff_residual_signal - pred
                            ref_resid = resid[fit_mask]
                            ref_med = float(np.median(ref_resid))
                            aux_mad = float(np.median(np.abs(ref_resid - ref_med)))
                            if np.isfinite(aux_mad) and aux_mad > 0.0:
                                z_resid = (resid - ref_med) / (1.4826 * aux_mad + 1e-6)
                                z_resid = np.nan_to_num(z_resid, nan=0.0, posinf=0.0, neginf=0.0)
                                bonus = np.maximum(z_resid, 0.0)
                                signal_out = diff_residual_signal + self.aux_lambda * bonus
                                aux_applied = True
                            else:
                                aux_reason = "zero_or_nan_residual_mad"
                        except Exception as exc:
                            aux_reason = f"lstsq_failed:{type(exc).__name__}"

        method = "local_mahalanobis"
        if self.aux_enabled and aux_applied:
            method = f"local_mahalanobis_aux_{self.aux_model}"

        self.last_meta = {
            "method": method,
            "reduced_dims": int(D.shape[1]) if D.ndim == 2 else 0,
            "cov_shrinkage": float(self.cov_shrinkage),
            "aux_enabled": bool(self.aux_enabled),
            "aux_lambda": float(self.aux_lambda),
            "aux_model": str(self.aux_model),
            "aux_applied": bool(aux_applied),
            "aux_ref_size": int(aux_ref_size),
            "aux_residual_mad": float(aux_mad) if np.isfinite(aux_mad) else np.nan,
            "aux_reason": str(aux_reason),
            "parallel_enabled": bool(parallel_enabled),
            "parallel_workers": int(workers if parallel_enabled else 1),
            "parallel_chunk_size": int(chunk_size),
            "parallel_min_rows": int(self.parallel_min_rows),
        }
        return signal_out
