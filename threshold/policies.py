#!/usr/bin/env python3
"""Threshold policy utilities for final metric runtime.

This module contains the minimal, deployment-oriented subset of threshold
selection logic used by ``run_final_metric.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from final_metric_refactor.config import RUNTIME_RULE_AVAILABLE_COL_NOMASK, RUNTIME_RULE_SIGNAL_COL_NOMASK

SIGNAL_COL = dict(RUNTIME_RULE_SIGNAL_COL_NOMASK)
AVAILABLE_COL = dict(RUNTIME_RULE_AVAILABLE_COL_NOMASK)

# Backward-compatible legacy row-results columns.
LEGACY_SCORE_COL = {
    "output": "output_score_nomask",
    "direction": "direction_score_nomask",
    "length": "length_score_nomask",
    "diff_residual": "diff_residual_score_nomask",
    "delta_ridge_ens": "delta_ridge_ens_score_nomask",
    "similar_input_conflict": "similar_input_conflict_score_nomask",
    "discourse_instability": "discourse_instability_score_nomask",
    "contradiction": "contradiction_score_nomask",
    "self_contradiction": "self_contradiction_score_nomask",
}

EPS = 1e-12


def resolve_signal_col(rule: str, row_df: pd.DataFrame) -> str:
    preferred = SIGNAL_COL.get(rule, "")
    if preferred and preferred in row_df.columns:
        return preferred
    legacy = LEGACY_SCORE_COL.get(rule, "")
    if legacy and legacy in row_df.columns:
        return legacy
    return preferred


def normalize_id(v: Any) -> str:
    s = str(v).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in {'"', "'"}:
        s = s[1:-1]
    return s.strip()


def compute_labels_bad(
    source_df: pd.DataFrame,
    row_df: pd.DataFrame,
    source_id_col: str,
    results_id_col: str,
    label_col: str,
    bad_label: str,
) -> tuple[np.ndarray, np.ndarray, bool]:
    n = len(row_df)
    y = np.zeros(n, dtype=bool)
    known = np.zeros(n, dtype=bool)

    target = str(bad_label).strip().lower()

    if source_id_col in source_df.columns and label_col in source_df.columns and results_id_col in row_df.columns:
        sid = source_df[source_id_col].map(normalize_id)
        sval = source_df[label_col].fillna("").astype(str).str.strip().str.lower()
        lut = dict(zip(sid.tolist(), sval.tolist(), strict=False))

        rid = row_df[results_id_col].map(normalize_id)
        mapped = rid.map(lambda x: lut.get(x, ""))
        m_known = mapped.ne("").to_numpy(dtype=bool)
        y[m_known] = mapped.eq(target).to_numpy(dtype=bool)[m_known]
        known[m_known] = True

    if "label_raw" in row_df.columns:
        raw = row_df["label_raw"].fillna("").astype(str).str.strip().str.lower()
        m_raw = raw.ne("").to_numpy(dtype=bool)
        fill = (~known) & m_raw
        if np.any(fill):
            y[fill] = raw.eq(target).to_numpy(dtype=bool)[fill]
            known[fill] = True

    return y, known, bool(np.any(known))


def binary_metrics(
    y_true_bad: np.ndarray,
    y_pred_bad: np.ndarray,
    known_mask: np.ndarray | None,
) -> dict[str, float]:
    yp = np.asarray(y_pred_bad, dtype=bool)
    pred_bad_rate = float(np.mean(yp)) if len(yp) else 0.0

    if known_mask is None:
        return {
            "tp": float("nan"),
            "fp": float("nan"),
            "tn": float("nan"),
            "fn": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "fpr": float("nan"),
            "pred_bad_rate": pred_bad_rate,
        }

    km = np.asarray(known_mask, dtype=bool)
    if len(km) != len(yp) or np.sum(km) == 0:
        return {
            "tp": float("nan"),
            "fp": float("nan"),
            "tn": float("nan"),
            "fn": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "fpr": float("nan"),
            "pred_bad_rate": pred_bad_rate,
        }

    yt = np.asarray(y_true_bad, dtype=bool)[km]
    yp = yp[km]
    tp = int(np.sum(yt & yp))
    fp = int(np.sum((~yt) & yp))
    tn = int(np.sum((~yt) & (~yp)))
    fn = int(np.sum(yt & (~yp)))
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float((2.0 * precision * recall) / (precision + recall)) if (precision + recall) else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
    return {
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "pred_bad_rate": pred_bad_rate,
    }


def minmax_scale(values: np.ndarray, nan_fill: float) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    out = np.full(len(x), float(nan_fill), dtype=float)
    finite = np.isfinite(x)
    if not np.any(finite):
        return out
    lo = float(np.min(x[finite]))
    hi = float(np.max(x[finite]))
    if hi - lo <= 1e-12:
        out[finite] = 0.5
        return out
    out[finite] = (x[finite] - lo) / (hi - lo)
    return out


def cluster_compactness(output_x: np.ndarray, output_y: np.ndarray, pred_bad: np.ndarray) -> float:
    ox = np.asarray(output_x, dtype=float)
    oy = np.asarray(output_y, dtype=float)
    pb = np.asarray(pred_bad, dtype=bool)

    use = pb & np.isfinite(ox) & np.isfinite(oy)
    if np.sum(use) < 3:
        return float("nan")

    x = ox[use]
    y = oy[use]
    cx = float(np.mean(x))
    cy = float(np.mean(y))
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return float(np.median(d))


def stability_metrics_for_pred(
    pred_bad: np.ndarray,
    direction_signal: np.ndarray,
    residual_signal: np.ndarray,
    ensemble_std: np.ndarray,
    output_x: np.ndarray,
    output_y: np.ndarray,
) -> dict[str, float]:
    pb = np.asarray(pred_bad, dtype=bool)
    direction = np.asarray(direction_signal, dtype=float)[pb]
    residual = np.asarray(residual_signal, dtype=float)[pb]
    ens = np.asarray(ensemble_std, dtype=float)[pb]

    direction_var = float(np.nanvar(direction)) if np.any(np.isfinite(direction)) else float("nan")
    residual_var = float(np.nanvar(residual)) if np.any(np.isfinite(residual)) else float("nan")
    ensemble_var = float(np.nanvar(ens)) if np.any(np.isfinite(ens)) else float("nan")
    compactness = cluster_compactness(output_x=output_x, output_y=output_y, pred_bad=pb)

    return {
        "direction_var": direction_var,
        "residual_var": residual_var,
        "ensemble_var": ensemble_var,
        "cluster_compactness": compactness,
    }


def compute_policy_features(row_df: pd.DataFrame) -> dict[str, np.ndarray]:
    n = len(row_df)

    direction = (
        pd.to_numeric(row_df["direction_signal_nomask"], errors="coerce").to_numpy(dtype=float)
        if "direction_signal_nomask" in row_df.columns
        else np.full(n, np.nan, dtype=float)
    )
    residual = (
        pd.to_numeric(row_df["diff_residual_signal_nomask"], errors="coerce").to_numpy(dtype=float)
        if "diff_residual_signal_nomask" in row_df.columns
        else np.full(n, np.nan, dtype=float)
    )

    member_cols = [
        c
        for c in row_df.columns
        if c.startswith("delta_ridge_ens_member_") and (c.endswith("_signal_nomask") or c.endswith("_score_nomask"))
    ]
    if member_cols:
        mat = row_df[member_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        # Avoid RuntimeWarning from np.nanstd on all-NaN rows by explicit row-wise moments.
        valid = np.isfinite(mat)
        valid_count = valid.sum(axis=1)
        mat_filled = np.where(valid, mat, 0.0)
        row_mean = np.divide(
            mat_filled.sum(axis=1),
            valid_count,
            out=np.zeros(n, dtype=float),
            where=valid_count > 0,
        )
        sq = np.where(valid, (mat_filled - row_mean[:, None]) ** 2, 0.0)
        row_var = np.divide(
            sq.sum(axis=1),
            valid_count,
            out=np.full(n, np.nan, dtype=float),
            where=valid_count > 0,
        )
        ensemble_std = np.sqrt(row_var)
    elif "delta_ridge_ens_signal_nomask" in row_df.columns:
        ensemble_std = pd.to_numeric(row_df["delta_ridge_ens_signal_nomask"], errors="coerce").to_numpy(dtype=float)
    else:
        ensemble_std = np.full(n, np.nan, dtype=float)

    output_x = (
        pd.to_numeric(row_df["output_pca_x_nomask"], errors="coerce").to_numpy(dtype=float)
        if "output_pca_x_nomask" in row_df.columns
        else np.full(n, np.nan, dtype=float)
    )
    output_y = (
        pd.to_numeric(row_df["output_pca_y_nomask"], errors="coerce").to_numpy(dtype=float)
        if "output_pca_y_nomask" in row_df.columns
        else np.full(n, np.nan, dtype=float)
    )

    return {
        "direction_signal": direction,
        "residual_signal": residual,
        "ensemble_std": ensemble_std,
        "output_x": output_x,
        "output_y": output_y,
    }


def _quantile_thresholds(values: np.ndarray, tail_q: float, tail_direction: str) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan")

    q = float(np.clip(tail_q, 1e-6, 0.499999 if tail_direction == "two_sided" else 0.999999))
    if tail_direction == "lower":
        thr = float(np.quantile(v, q))
        return thr, float("nan")
    if tail_direction == "two_sided":
        lo = float(np.quantile(v, q))
        hi = float(np.quantile(v, 1.0 - q))
        return lo, hi

    thr = float(np.quantile(v, 1.0 - q))
    return thr, float("nan")


def _build_fail_by_threshold(score: np.ndarray, base: np.ndarray, tail_direction: str, low: float, high: float) -> np.ndarray:
    fail = np.zeros(len(score), dtype=bool)
    if tail_direction == "lower":
        fail[base] = score[base] <= float(low)
    elif tail_direction == "two_sided":
        fail[base] = (score[base] <= float(low)) | (score[base] >= float(high))
    else:
        fail[base] = score[base] >= float(low)
    return fail


def _adjust_zero_threshold_for_dist(
    values: np.ndarray,
    tail_direction: str,
    low: float,
    high: float,
) -> tuple[float, float, str]:
    """Guardrail for dist policy: if threshold is exactly 0, move to next value."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float(low), float(high), "none"

    lo = float(low)
    hi = float(high)
    tags: list[str] = []

    def _is_zero(x: float) -> bool:
        return bool(np.isfinite(x) and abs(float(x)) <= float(EPS))

    if tail_direction == "upper":
        if _is_zero(lo):
            nxt = v[v > lo]
            if len(nxt) > 0:
                lo = float(np.min(nxt))
                tags.append("zero_to_next_upper")
    elif tail_direction == "lower":
        if _is_zero(lo):
            prv = v[v < lo]
            if len(prv) > 0:
                lo = float(np.max(prv))
                tags.append("zero_to_prev_lower")
    elif tail_direction == "two_sided":
        if _is_zero(lo):
            prv = v[v < lo]
            if len(prv) > 0:
                lo = float(np.max(prv))
                tags.append("zero_to_prev_lower")
            else:
                lo = float(np.nextafter(0.0, float("-inf")))
                tags.append("zero_to_eps_lower")
        if _is_zero(hi):
            nxt = v[v > hi]
            if len(nxt) > 0:
                hi = float(np.min(nxt))
                tags.append("zero_to_next_upper")
            else:
                hi = float(np.nextafter(0.0, float("inf")))
                tags.append("zero_to_eps_upper")

    tag = "|".join(tags) if tags else "none"
    return lo, hi, tag


def trigger_mask(
    score: np.ndarray,
    base: np.ndarray,
    tail_direction: str,
    threshold: float,
    threshold_low: float,
    threshold_high: float,
) -> np.ndarray:
    """Build a boolean trigger mask from threshold fields."""
    if tail_direction == "two_sided":
        low = float(threshold_low)
        high = float(threshold_high)
        if not np.isfinite(low) and np.isfinite(threshold):
            low = float(threshold)
        if not np.isfinite(high) and np.isfinite(threshold):
            high = float(threshold)
        if not np.isfinite(low) or not np.isfinite(high):
            return np.zeros(len(score), dtype=bool)
        return _build_fail_by_threshold(score=score, base=base, tail_direction=tail_direction, low=low, high=high)

    low = float(threshold_low)
    if not np.isfinite(low):
        low = float(threshold)
    if not np.isfinite(low):
        return np.zeros(len(score), dtype=bool)
    return _build_fail_by_threshold(score=score, base=base, tail_direction=tail_direction, low=low, high=float("nan"))


def _mad_sigma(values: np.ndarray, mad_eps: float) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan")
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    sigma = float(1.4826 * mad)
    if np.isfinite(sigma) and sigma > float(mad_eps):
        return sigma
    return float("nan")


def _core_threshold_1d(
    x: np.ndarray,
    x_tail_start: float,
    *,
    core_kappa: float,
    core_quantile: float,
    core_min_count: int,
    mad_eps: float,
) -> tuple[float, str, int, float, float]:
    vals = np.asarray(x, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0 or not np.isfinite(x_tail_start):
        return float("nan"), "core_missing", 0, float("nan"), float("nan")

    core = vals[vals <= float(x_tail_start)]
    core_n = int(len(core))
    med_c = float(np.median(core)) if core_n > 0 else float("nan")
    sigma_c = _mad_sigma(core, mad_eps=float(mad_eps)) if core_n > 0 else float("nan")

    if core_n >= int(max(3, core_min_count)) and np.isfinite(sigma_c) and sigma_c > float(mad_eps):
        t_core = float(med_c + (float(core_kappa) * sigma_c))
        source = "core_median_mad"
    else:
        fallback_src = "core_quantile_fallback"
        if core_n < int(max(3, core_min_count)):
            fallback_src += "_small_core"
        elif not np.isfinite(sigma_c) or sigma_c <= float(mad_eps):
            fallback_src += "_mad_zero"
        ref = core if core_n > 0 else vals
        if len(ref) == 0:
            return float("nan"), "core_missing", core_n, med_c, sigma_c
        q = float(np.clip(core_quantile, 1e-6, 0.999999))
        t_core = float(np.quantile(ref, q))
        source = fallback_src

    if np.isfinite(t_core) and t_core >= float(x_tail_start):
        t_core = float(np.nextafter(float(x_tail_start), float("-inf")))
        source += "|clip_lt_tail_start"
    return t_core, source, core_n, med_c, sigma_c


def _estimate_exceptional_out_1d(
    x: np.ndarray,
    x_tail_start: float,
    *,
    mad_eps: float,
    d1_lambda: float,
    d2_lambda: float,
    consecutive: int,
    grid_points: int,
    min_tail_points: int,
    fallback_quantile: float,
    min_delta_ratio: float,
) -> tuple[float, str]:
    vals = np.asarray(x, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0 or not np.isfinite(x_tail_start):
        return float("nan"), "exceptional_missing"

    tail = np.sort(vals[vals >= float(x_tail_start)])
    if len(tail) == 0:
        return float("nan"), "exceptional_tail_empty"

    # Strong fallback for sparse tails.
    if len(tail) < int(max(8, min_tail_points)):
        q_fb = float(np.clip(fallback_quantile, 0.5, 0.999999))
        t_fb = float(np.quantile(tail, q_fb))
        span = max(float(EPS), float(tail[-1] - x_tail_start))
        min_delta = max(float(EPS), float(min_delta_ratio) * span)
        t_hard = max(float(x_tail_start) + min_delta, t_fb)
        if t_hard > float(tail[-1]):
            t_hard = float(tail[-1])
        if not np.isfinite(t_hard) or t_hard <= float(x_tail_start) + float(EPS):
            return float("nan"), "exceptional_quantile_fallback_small_tail_degenerate"
        return t_hard, f"exceptional_quantile_fallback_small_tail|q={q_fb:.3g}"

    hi = float(np.quantile(tail, 0.999))
    if not np.isfinite(hi) or hi <= float(x_tail_start) + float(EPS):
        hi = float(tail[-1])
    if not np.isfinite(hi) or hi <= float(x_tail_start) + float(EPS):
        return float("nan"), "exceptional_tail_span_missing"

    points = max(64, int(grid_points))
    grid = np.linspace(float(x_tail_start), hi, points)
    ccdf = np.array([float(np.mean(tail >= g)) for g in grid], dtype=float)
    log_ccdf = np.log(np.maximum(ccdf, 1e-12))

    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=float)
    kernel = kernel / np.sum(kernel)
    smooth = np.convolve(log_ccdf, kernel, mode="same")
    d1 = np.gradient(smooth, grid)
    d2 = np.gradient(d1, grid)

    valid = np.isfinite(d1) & np.isfinite(d2) & (ccdf > (1.0 / max(1.0, float(len(tail))))) & (ccdf < 0.8)
    if np.sum(valid) < 8:
        valid = np.isfinite(d1) & np.isfinite(d2)
    if np.sum(valid) < 8:
        return float("nan"), "exceptional_derivative_support_missing"

    sigma_d1 = _mad_sigma(d1[valid], mad_eps=float(mad_eps))
    sigma_d2 = _mad_sigma(d2[valid], mad_eps=float(mad_eps))
    if not np.isfinite(sigma_d1) or sigma_d1 <= float(mad_eps):
        sigma_d1 = float(np.nanstd(d1[valid]))
    if not np.isfinite(sigma_d2) or sigma_d2 <= float(mad_eps):
        sigma_d2 = float(np.nanstd(d2[valid]))
    if not np.isfinite(sigma_d1) or sigma_d1 <= float(mad_eps):
        return float("nan"), "exceptional_d1_sigma_missing"
    if not np.isfinite(sigma_d2) or sigma_d2 <= float(mad_eps):
        return float("nan"), "exceptional_d2_sigma_missing"

    span = max(float(EPS), hi - float(x_tail_start))
    min_delta = max(float(EPS), float(min_delta_ratio) * span)
    thr1 = float(max(float(mad_eps), float(d1_lambda) * sigma_d1))
    thr2 = float(max(float(mad_eps), float(d2_lambda) * sigma_d2))

    cond = (
        np.isfinite(d1)
        & np.isfinite(d2)
        & (np.abs(d1) <= thr1)
        & (np.abs(d2) <= thr2)
        & (grid >= (float(x_tail_start) + min_delta))
    )
    run = 0
    for i in range(len(cond)):
        if bool(cond[i]):
            run += 1
        else:
            run = 0
        if run >= int(max(2, consecutive)):
            j = int(i - int(max(2, consecutive)) + 1)
            t_hard = float(grid[j])
            if np.isfinite(t_hard) and t_hard > float(x_tail_start) + float(EPS):
                return (
                    t_hard,
                    f"exceptional_d2_floor|d1_l={float(d1_lambda):.3g}|d2_l={float(d2_lambda):.3g}|L={int(max(2, consecutive))}",
                )

    q_fb = float(np.clip(fallback_quantile, 0.5, 0.999999))
    t_fb = float(np.quantile(tail, q_fb))
    t_hard = max(float(x_tail_start) + min_delta, t_fb)
    if t_hard > hi:
        t_hard = hi
    if not np.isfinite(t_hard) or t_hard <= float(x_tail_start) + float(EPS):
        return float("nan"), "exceptional_quantile_fallback_degenerate"
    return t_hard, f"exceptional_quantile_fallback|q={q_fb:.3g}"


def derive_tristate_thresholds_from_fail(
    values: np.ndarray,
    tail_direction: str,
    fail_threshold: float,
    fail_low: float,
    fail_high: float,
    warn_ratio: float = 0.8,
    hard_ratio: float = 1.2,
    core_kappa: float = 1.0,
    core_quantile: float = 0.85,
    core_min_count: int = 12,
    mad_eps: float = 1e-9,
    exceptional_d1_lambda: float = 1.0,
    exceptional_d2_lambda: float = 1.0,
    exceptional_consecutive: int = 4,
    exceptional_grid_points: int = 140,
    exceptional_min_tail_points: int = 24,
    exceptional_fallback_quantile: float = 0.90,
    exceptional_min_delta_ratio: float = 0.05,
) -> dict[str, float]:
    """Derive core/tail_start/exceptional_out thresholds (warn/fail/hard aliases)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    out = {
        "warn_threshold": float("nan"),
        "warn_threshold_low": float("nan"),
        "warn_threshold_high": float("nan"),
        "hard_fail_threshold": float("nan"),
        "hard_fail_threshold_low": float("nan"),
        "hard_fail_threshold_high": float("nan"),
        "core_threshold": float("nan"),
        "core_threshold_low": float("nan"),
        "core_threshold_high": float("nan"),
        "tail_start_threshold": float("nan"),
        "tail_start_threshold_low": float("nan"),
        "tail_start_threshold_high": float("nan"),
        "exceptional_out_threshold": float("nan"),
        "exceptional_out_threshold_low": float("nan"),
        "exceptional_out_threshold_high": float("nan"),
        "core_source": "missing",
        "exceptional_out_source": "missing",
        "core_support_rows": 0.0,
    }

    if len(v) == 0:
        return out

    med = float(np.median(v))
    h_ratio = float(max(hard_ratio, 1.0))

    if tail_direction == "two_sided":
        lo = float(fail_low) if np.isfinite(fail_low) else float("nan")
        hi = float(fail_high) if np.isfinite(fail_high) else float("nan")
        if (not np.isfinite(lo) or not np.isfinite(hi)) and np.isfinite(fail_threshold):
            d0 = abs(float(fail_threshold) - med)
            lo = med - d0
            hi = med + d0
        if not np.isfinite(lo) or not np.isfinite(hi):
            return out

        d_fail = float(max(abs(med - lo), abs(hi - med)))
        if not np.isfinite(d_fail) or d_fail <= float(EPS):
            return out

        dist = np.abs(v - med)
        core_d, core_src, core_n, _, _ = _core_threshold_1d(
            x=dist,
            x_tail_start=d_fail,
            core_kappa=float(core_kappa),
            core_quantile=float(core_quantile),
            core_min_count=int(core_min_count),
            mad_eps=float(mad_eps),
        )
        if not np.isfinite(core_d):
            core_d = float(max(float(EPS), d_fail * float(np.clip(warn_ratio, 0.0, 1.0))))
            core_src = "core_legacy_ratio_fallback"
        core_d = min(core_d, float(np.nextafter(d_fail, float("-inf"))))

        hard_d, hard_src = _estimate_exceptional_out_1d(
            x=dist,
            x_tail_start=d_fail,
            mad_eps=float(mad_eps),
            d1_lambda=float(exceptional_d1_lambda),
            d2_lambda=float(exceptional_d2_lambda),
            consecutive=int(exceptional_consecutive),
            grid_points=int(exceptional_grid_points),
            min_tail_points=int(exceptional_min_tail_points),
            fallback_quantile=float(exceptional_fallback_quantile),
            min_delta_ratio=float(exceptional_min_delta_ratio),
        )
        if not np.isfinite(hard_d):
            hard_d = float(max(d_fail * h_ratio, d_fail + float(EPS)))
            hard_src = "exceptional_legacy_ratio_fallback"
        hard_d = max(hard_d, d_fail + float(EPS))

        warn_lo = float(med - core_d)
        warn_hi = float(med + core_d)
        hard_lo = float(med - hard_d)
        hard_hi = float(med + hard_d)

        out.update(
            {
                "warn_threshold": float(warn_hi),
                "warn_threshold_low": float(warn_lo),
                "warn_threshold_high": float(warn_hi),
                "hard_fail_threshold": float(hard_hi),
                "hard_fail_threshold_low": float(hard_lo),
                "hard_fail_threshold_high": float(hard_hi),
                "core_threshold": float(warn_hi),
                "core_threshold_low": float(warn_lo),
                "core_threshold_high": float(warn_hi),
                "tail_start_threshold": float(hi),
                "tail_start_threshold_low": float(lo),
                "tail_start_threshold_high": float(hi),
                "exceptional_out_threshold": float(hard_hi),
                "exceptional_out_threshold_low": float(hard_lo),
                "exceptional_out_threshold_high": float(hard_hi),
                "core_source": core_src,
                "exceptional_out_source": hard_src,
                "core_support_rows": float(core_n),
            }
        )
        return out

    fail_t = float(fail_low) if np.isfinite(fail_low) else float(fail_threshold)
    if not np.isfinite(fail_t):
        return out

    if tail_direction == "lower":
        x = -v
        x_tail = -float(fail_t)
    else:
        x = v
        x_tail = float(fail_t)

    core_x, core_src, core_n, core_med_x, core_sigma_x = _core_threshold_1d(
        x=x,
        x_tail_start=x_tail,
        core_kappa=float(core_kappa),
        core_quantile=float(core_quantile),
        core_min_count=int(core_min_count),
        mad_eps=float(mad_eps),
    )
    if not np.isfinite(core_x):
        d0 = max(float(EPS), abs(float(fail_t) - med))
        w_ratio = float(np.clip(warn_ratio, 0.0, 1.0))
        core_x = float(x_tail - (w_ratio * d0)) if tail_direction == "lower" else float(x_tail - ((1.0 - w_ratio) * d0))
        core_src = "core_legacy_ratio_fallback"

    k_abs = float(abs(core_kappa))
    if (
        np.isfinite(core_med_x)
        and np.isfinite(core_sigma_x)
        and core_sigma_x > float(mad_eps)
        and k_abs > 0.0
    ):
        core_lo_x = float(core_med_x - (k_abs * core_sigma_x))
        core_hi_x = float(core_med_x + (k_abs * core_sigma_x))
    else:
        # Fallback: keep explicit range fields finite while preserving core_x as upper bound.
        core_hi_x = float(core_x)
        core_lo_x = float("nan")
        if np.isfinite(core_hi_x):
            core_vals = x[np.isfinite(x) & (x <= float(x_tail))]
            if len(core_vals) > 0:
                core_lo_x = float(np.nanmin(core_vals))

    if np.isfinite(core_hi_x) and core_hi_x >= float(x_tail):
        core_hi_x = float(np.nextafter(float(x_tail), float("-inf")))
    if np.isfinite(core_lo_x) and np.isfinite(core_hi_x) and core_lo_x >= core_hi_x:
        core_lo_x = float(np.nextafter(float(core_hi_x), float("-inf")))

    hard_x, hard_src = _estimate_exceptional_out_1d(
        x=x,
        x_tail_start=x_tail,
        mad_eps=float(mad_eps),
        d1_lambda=float(exceptional_d1_lambda),
        d2_lambda=float(exceptional_d2_lambda),
        consecutive=int(exceptional_consecutive),
        grid_points=int(exceptional_grid_points),
        min_tail_points=int(exceptional_min_tail_points),
        fallback_quantile=float(exceptional_fallback_quantile),
        min_delta_ratio=float(exceptional_min_delta_ratio),
    )
    if not np.isfinite(hard_x):
        d = max(float(EPS), abs(float(fail_t) - med))
        if tail_direction == "lower":
            hard_t_legacy = med - (h_ratio * d)
            hard_x = -float(hard_t_legacy)
        else:
            hard_t_legacy = med + (h_ratio * d)
            hard_x = float(hard_t_legacy)
        hard_src = "exceptional_legacy_ratio_fallback"

    if tail_direction == "lower":
        warn_t = float(-core_x)
        hard_t = float(-hard_x)
        core_low_t = float(-core_hi_x) if np.isfinite(core_hi_x) else float("nan")
        core_high_t = float(-core_lo_x) if np.isfinite(core_lo_x) else float("nan")
        if warn_t <= float(fail_t):
            warn_t = float(np.nextafter(float(fail_t), float("inf")))
            core_src += "|clip_gt_tail_start"
        if hard_t >= float(fail_t):
            hard_t = float(np.nextafter(float(fail_t), float("-inf")))
            hard_src += "|clip_lt_tail_start"
    else:
        warn_t = float(core_x)
        hard_t = float(hard_x)
        core_low_t = float(core_lo_x) if np.isfinite(core_lo_x) else float("nan")
        core_high_t = float(core_hi_x) if np.isfinite(core_hi_x) else float("nan")
        if warn_t >= float(fail_t):
            warn_t = float(np.nextafter(float(fail_t), float("-inf")))
            core_src += "|clip_lt_tail_start"
        if hard_t <= float(fail_t):
            hard_t = float(np.nextafter(float(fail_t), float("inf")))
            hard_src += "|clip_gt_tail_start"

    if not np.isfinite(core_low_t):
        core_low_t = float(warn_t)
    if not np.isfinite(core_high_t):
        core_high_t = float(warn_t)
    if core_high_t < core_low_t:
        core_low_t, core_high_t = core_high_t, core_low_t

    out.update(
        {
            "warn_threshold": float(warn_t),
            "warn_threshold_low": float(core_low_t),
            "warn_threshold_high": float(core_high_t),
            "hard_fail_threshold": float(hard_t),
            "hard_fail_threshold_low": float(hard_t),
            "hard_fail_threshold_high": float("nan"),
            "core_threshold": float(warn_t),
            "core_threshold_low": float(core_low_t),
            "core_threshold_high": float(core_high_t),
            "tail_start_threshold": float(fail_t),
            "tail_start_threshold_low": float(fail_t),
            "tail_start_threshold_high": float("nan"),
            "exceptional_out_threshold": float(hard_t),
            "exceptional_out_threshold_low": float(hard_t),
            "exceptional_out_threshold_high": float("nan"),
            "core_source": core_src,
            "exceptional_out_source": hard_src,
            "core_support_rows": float(core_n),
        }
    )
    return out


def robust_scale(values: np.ndarray, mad_eps: float) -> tuple[float, str]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), "missing"

    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    iqr = float(np.quantile(v, 0.75) - np.quantile(v, 0.25))
    q90_10 = float(np.quantile(v, 0.90) - np.quantile(v, 0.10))

    mad_scale = float(1.4826 * mad)
    iqr_scale = float(iqr / 1.349) if np.isfinite(iqr) else float("nan")
    q9010_scale = float(0.7413 * q90_10) if np.isfinite(q90_10) else float("nan")

    if np.isfinite(mad_scale) and mad_scale > float(mad_eps):
        return mad_scale, "mad"

    v_hi = v[v > med]
    if len(v_hi) >= 20:
        med_hi = float(np.median(v_hi))
        mad_hi = float(np.median(np.abs(v_hi - med_hi)))
        mad_hi_scale = float(1.4826 * mad_hi)
        iqr_hi = float(np.quantile(v_hi, 0.75) - np.quantile(v_hi, 0.25))
        iqr_hi_scale = float(iqr_hi / 1.349) if np.isfinite(iqr_hi) else float("nan")

        if np.isfinite(mad_hi_scale) and mad_hi_scale > float(mad_eps):
            return mad_hi_scale, "mad_upper"
        if np.isfinite(iqr_hi_scale) and iqr_hi_scale > float(mad_eps):
            return iqr_hi_scale, "iqr_upper"

    if np.isfinite(iqr_scale) and iqr_scale > float(mad_eps):
        return iqr_scale, "iqr"
    if np.isfinite(q9010_scale) and q9010_scale > float(mad_eps):
        return q9010_scale, "q90_10"

    for s in [mad_scale, iqr_scale, q9010_scale]:
        if np.isfinite(s):
            return float(s), "flat"
    return float("nan"), "missing"


def estimate_tail_start_k(
    z_for_tail: np.ndarray,
    min_survival: float,
    max_survival: float,
    grid_points: int,
) -> float:
    z = np.asarray(z_for_tail, dtype=float)
    z = z[np.isfinite(z)]
    z = z[z >= 0.0]
    if len(z) < 40:
        return float("nan")

    hi = float(np.quantile(z, 0.995))
    if not np.isfinite(hi) or hi <= 0.0:
        return float("nan")

    points = max(64, int(grid_points))
    grid = np.linspace(0.0, hi, points)
    ccdf = np.array([float(np.mean(z >= g)) for g in grid], dtype=float)
    log_ccdf = np.log(np.maximum(ccdf, 1e-12))

    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=float)
    kernel = kernel / np.sum(kernel)
    smooth = np.convolve(log_ccdf, kernel, mode="same")
    d1 = np.gradient(smooth, grid)
    d2 = np.gradient(d1, grid)

    valid = (
        (ccdf >= float(min_survival))
        & (ccdf <= float(max_survival))
        & (grid >= 0.5)
        & np.isfinite(d2)
    )
    if np.sum(valid) < 5:
        valid = (
            (ccdf >= max(1e-5, float(min_survival) * 0.5))
            & (ccdf <= min(0.95, max(float(max_survival), 0.5)))
            & (grid >= 0.5)
            & np.isfinite(d2)
        )
    if np.sum(valid) < 5:
        return float("nan")

    idxs = np.where(valid)[0]
    i = int(idxs[np.argmin(d2[idxs])])
    return float(grid[i])


def robust_sigma_1d(values: np.ndarray, mad_eps: float) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan")

    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    sigma = float(1.4826 * mad)
    if np.isfinite(sigma) and sigma > float(mad_eps):
        return sigma

    iqr = float(np.quantile(v, 0.75) - np.quantile(v, 0.25))
    sigma_iqr = float(iqr / 1.349) if np.isfinite(iqr) else float("nan")
    if np.isfinite(sigma_iqr) and sigma_iqr > float(mad_eps):
        return sigma_iqr

    q90_10 = float(np.quantile(v, 0.90) - np.quantile(v, 0.10))
    sigma_q = float(0.7413 * q90_10) if np.isfinite(q90_10) else float("nan")
    if np.isfinite(sigma_q) and sigma_q > float(mad_eps):
        return sigma_q
    return float("nan")


def estimate_gap_jump_k_info(
    z_for_tail: np.ndarray,
    min_survival: float,
    max_survival: float,
    mad_eps: float,
) -> tuple[float, float, str]:
    z = np.asarray(z_for_tail, dtype=float)
    z = z[np.isfinite(z)]
    z = z[z >= 0.0]
    if len(z) < 40:
        return float("nan"), float("nan"), "none"

    z = np.sort(z)
    dz = np.diff(z)
    if len(dz) < 8:
        return float("nan"), float("nan"), "none"

    n = len(z)
    surv = (n - (np.arange(len(dz)) + 1)) / max(float(n), 1.0)
    pos_mass = float(np.mean(z > 0.0))
    max_surv_eff = min(0.95, max(float(max_survival), pos_mass + 0.05))

    valid = (dz > 0.0) & (surv >= float(min_survival)) & (surv <= max_surv_eff)
    if np.sum(valid) < 5:
        valid = (dz > 0.0) & (surv >= max(1e-5, float(min_survival) * 0.5)) & (surv <= min(0.98, max_surv_eff + 0.1))
    if np.sum(valid) < 5:
        return float("nan"), float("nan"), "none"

    ref = dz[valid]
    sigma = robust_sigma_1d(ref, mad_eps=float(mad_eps))
    if not np.isfinite(sigma) or sigma <= float(mad_eps):
        return float("nan"), float("nan"), "none"

    med = float(np.median(ref))
    z_gap = (dz - med) / max(sigma, float(mad_eps))

    strong = valid & np.isfinite(z_gap) & (z_gap >= 4.0)
    if np.any(strong):
        idx = int(np.where(strong)[0][0])
        return float(z[idx + 1]), float(z_gap[idx]), "strong_first"

    idxs = np.where(valid & np.isfinite(z_gap))[0]
    if len(idxs) == 0:
        return float("nan"), float("nan"), "none"
    best = int(idxs[np.argmax(z_gap[idxs])])
    best_z = float(z_gap[best])
    if best_z < 2.5:
        return float("nan"), best_z, "none"
    return float(z[best + 1]), best_z, "weak_max"


def estimate_gap_jump_k(
    z_for_tail: np.ndarray,
    min_survival: float,
    max_survival: float,
    mad_eps: float,
) -> float:
    k, _, _ = estimate_gap_jump_k_info(
        z_for_tail=z_for_tail,
        min_survival=min_survival,
        max_survival=max_survival,
        mad_eps=mad_eps,
    )
    return float(k)


def robust_z_fail_mask(
    score: np.ndarray,
    base: np.ndarray,
    k: float,
    tail_direction: str,
    mad_eps: float,
) -> tuple[np.ndarray, dict[str, float]]:
    n = len(score)
    fail = np.zeros(n, dtype=bool)
    vals = np.asarray(score, dtype=float)[base]
    vals = vals[np.isfinite(vals)]

    meta = {
        "threshold_applied": float("nan"),
        "threshold_low": float("nan"),
        "threshold_high": float("nan"),
        "median": float("nan"),
        "mad": float("nan"),
        "sigma": float("nan"),
        "source": "robust_z",
    }

    if len(vals) == 0:
        meta["source"] = "missing"
        return fail, meta

    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    sigma, scale_source = robust_scale(vals, mad_eps=float(mad_eps))

    meta["median"] = med
    meta["mad"] = mad
    meta["sigma"] = sigma

    if not np.isfinite(sigma) or sigma <= float(mad_eps):
        meta["source"] = f"robust_z_{scale_source}"
        if tail_direction == "upper":
            meta["threshold_applied"] = float("inf")
            meta["threshold_low"] = float("inf")
        elif tail_direction == "lower":
            meta["threshold_applied"] = float("-inf")
            meta["threshold_low"] = float("-inf")
        else:
            meta["threshold_applied"] = float("inf")
            meta["threshold_low"] = float("-inf")
            meta["threshold_high"] = float("inf")
        return fail, meta

    z = (score - med) / max(sigma, float(mad_eps))
    if tail_direction == "lower":
        fail[base] = z[base] <= -float(k)
        low = med - float(k) * sigma
        meta["threshold_applied"] = low
        meta["threshold_low"] = low
    elif tail_direction == "two_sided":
        fail[base] = np.abs(z[base]) >= float(k)
        lo = med - float(k) * sigma
        hi = med + float(k) * sigma
        meta["threshold_low"] = lo
        meta["threshold_high"] = hi
        meta["threshold_applied"] = hi
    else:
        fail[base] = z[base] >= float(k)
        high = med + float(k) * sigma
        meta["threshold_applied"] = high
        meta["threshold_low"] = high

    meta["source"] = f"robust_z_{scale_source}"
    return fail, meta


def robust_z_tail_start_fail_mask(
    score: np.ndarray,
    base: np.ndarray,
    robust_z_k: float,
    floor_k: float,
    max_k: float,
    tail_direction: str,
    mad_eps: float,
    tail_start_min_survival: float,
    tail_start_max_survival: float,
    tail_start_grid_points: int,
) -> tuple[np.ndarray, dict[str, float]]:
    n = len(score)
    fail = np.zeros(n, dtype=bool)
    vals = np.asarray(score, dtype=float)[base]
    vals = vals[np.isfinite(vals)]

    meta = {
        "threshold_applied": float("nan"),
        "threshold_low": float("nan"),
        "threshold_high": float("nan"),
        "median": float("nan"),
        "mad": float("nan"),
        "sigma": float("nan"),
        "source": "tail_start",
    }

    if len(vals) == 0:
        meta["source"] = "missing"
        return fail, meta

    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    sigma, scale_source = robust_scale(vals, mad_eps=float(mad_eps))

    meta["median"] = med
    meta["mad"] = mad
    meta["sigma"] = sigma

    if not np.isfinite(sigma) or sigma <= float(mad_eps):
        meta["source"] = f"tail_start_{scale_source}"
        if tail_direction == "upper":
            meta["threshold_applied"] = float("inf")
            meta["threshold_low"] = float("inf")
        elif tail_direction == "lower":
            meta["threshold_applied"] = float("-inf")
            meta["threshold_low"] = float("-inf")
        else:
            meta["threshold_applied"] = float("inf")
            meta["threshold_low"] = float("-inf")
            meta["threshold_high"] = float("inf")
        return fail, meta

    z = (score - med) / max(sigma, float(mad_eps))
    if tail_direction == "lower":
        z_tail = -z[base]
    elif tail_direction == "two_sided":
        z_tail = np.abs(z[base])
    else:
        z_tail = z[base]

    pos_mass = float(np.mean(z_tail > 0.0)) if len(z_tail) else 0.0
    max_surv_eff = min(0.95, max(float(tail_start_max_survival), pos_mass + 0.05))

    k_tail_curv = estimate_tail_start_k(
        z_for_tail=z_tail,
        min_survival=float(tail_start_min_survival),
        max_survival=max_surv_eff,
        grid_points=int(tail_start_grid_points),
    )
    k_tail_gap, gap_z, gap_kind = estimate_gap_jump_k_info(
        z_for_tail=z_tail,
        min_survival=float(tail_start_min_survival),
        max_survival=max_surv_eff,
        mad_eps=float(mad_eps),
    )

    # Continuity guardrail: when a strong discontinuity exists, keep tail-start
    # at the continuity-end (gap boundary), not at a later curvature point.
    continuity_guard = bool(np.isfinite(k_tail_gap) and np.isfinite(gap_z) and float(gap_z) >= 4.0)
    if continuity_guard:
        k_tail = float(k_tail_gap)
        tail_pick = "gap_continuity_guard"
    else:
        if np.isfinite(k_tail_curv) and np.isfinite(k_tail_gap):
            if float(k_tail_curv) >= float(k_tail_gap):
                k_tail = float(k_tail_curv)
                tail_pick = "curv_larger"
            else:
                k_tail = float(k_tail_gap)
                tail_pick = "gap_larger"
        elif np.isfinite(k_tail_curv):
            k_tail = float(k_tail_curv)
            tail_pick = "curv_only"
        elif np.isfinite(k_tail_gap):
            k_tail = float(k_tail_gap)
            tail_pick = "gap_only"
        else:
            k_tail = float("nan")
            tail_pick = "none"

    k0 = float(robust_z_k)
    if np.isfinite(floor_k):
        k_floor = float(floor_k)
        floor_source = "floor_user"
    else:
        k_floor = 1.0 if tail_direction == "two_sided" else 0.5
        floor_source = "floor_adaptive"

    k_upper = float(max_k) if np.isfinite(max_k) else float("inf")
    if np.isfinite(k_tail):
        k_eff = min(k_upper, max(k_floor, float(k_tail)))
        source_tail = (
            f"tailstart_gap={k_tail_gap:.4g}|tailstart_curv={k_tail_curv:.4g}|"
            f"gap_z={gap_z:.4g}|gap_kind={gap_kind}|continuity_guard={int(continuity_guard)}|"
            f"tail_pick={tail_pick}|tailstart_pick={k_tail:.4g}|posmass={pos_mass:.4g}|maxsurv={max_surv_eff:.4g}"
        )
    else:
        k_eff = min(k_upper, max(k_floor, float(k0)))
        source_tail = (
            f"tailstart_missing|gap_z={gap_z:.4g}|gap_kind={gap_kind}|continuity_guard={int(continuity_guard)}|"
            f"tail_pick={tail_pick}|posmass={pos_mass:.4g}|maxsurv={max_surv_eff:.4g}"
        )

    if tail_direction == "lower":
        fail[base] = z[base] <= -k_eff
        low = med - k_eff * sigma
        meta["threshold_applied"] = low
        meta["threshold_low"] = low
    elif tail_direction == "two_sided":
        fail[base] = np.abs(z[base]) >= k_eff
        lo = med - k_eff * sigma
        hi = med + k_eff * sigma
        meta["threshold_low"] = lo
        meta["threshold_high"] = hi
        meta["threshold_applied"] = hi
    else:
        fail[base] = z[base] >= k_eff
        high = med + k_eff * sigma
        meta["threshold_applied"] = high
        meta["threshold_low"] = high

    meta["source"] = f"tail_start_{scale_source}|{source_tail}|{floor_source}|k_eff={k_eff:.4g}"
    return fail, meta


def quantile_tail_fail_mask(
    score: np.ndarray,
    base: np.ndarray,
    q: float,
    tail_direction: str,
) -> tuple[np.ndarray, dict[str, float]]:
    vals = np.asarray(score, dtype=float)[base]
    vals = vals[np.isfinite(vals)]
    low, high = _quantile_thresholds(vals, q, tail_direction)
    fail = _build_fail_by_threshold(score=score, base=base, tail_direction=tail_direction, low=low, high=high)

    return fail, {
        "threshold_applied": float(high if tail_direction == "two_sided" else low),
        "threshold_low": float(low),
        "threshold_high": float(high),
        "median": float("nan"),
        "mad": float("nan"),
        "sigma": float("nan"),
        "source": "quantile_tail",
    }


def dist_stability_jump_fail_mask(
    score: np.ndarray,
    base: np.ndarray,
    tail_direction: str,
    q_min: int,
    q_max: int,
    q_step: int,
    features: dict[str, np.ndarray],
    fallback_quantile: float,
) -> tuple[np.ndarray, dict[str, float]]:
    n = len(score)
    vals = np.asarray(score, dtype=float)[base]
    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        return np.zeros(n, dtype=bool), {
            "threshold_applied": float("nan"),
            "threshold_low": float("nan"),
            "threshold_high": float("nan"),
            "median": float("nan"),
            "mad": float("nan"),
            "sigma": float("nan"),
            "source": "missing",
        }

    q_values = np.arange(int(q_min), int(q_max) + 1, max(1, int(q_step)), dtype=int)
    rows: list[dict[str, float]] = []
    residual_signal = np.asarray(
        features.get("residual_signal", features.get("residual_score", np.full(n, np.nan, dtype=float))),
        dtype=float,
    )

    for q_percent in q_values.tolist():
        tail_q = float(q_percent) / 100.0
        low, high = _quantile_thresholds(vals, tail_q, tail_direction)
        low, high, zero_guard_tag = _adjust_zero_threshold_for_dist(
            values=vals,
            tail_direction=tail_direction,
            low=low,
            high=high,
        )
        pred = _build_fail_by_threshold(score=score, base=base, tail_direction=tail_direction, low=low, high=high)

        st = stability_metrics_for_pred(
            pred_bad=pred,
            direction_signal=features["direction_signal"],
            residual_signal=residual_signal,
            ensemble_std=features["ensemble_std"],
            output_x=features["output_x"],
            output_y=features["output_y"],
        )
        rows.append(
            {
                "q_percent": float(q_percent),
                "threshold_low": float(low),
                "threshold_high": float(high),
                "zero_guard_tag": str(zero_guard_tag),
                **st,
            }
        )

    q_df = pd.DataFrame(rows)
    if q_df.empty:
        fail, meta = quantile_tail_fail_mask(
            score=score,
            base=base,
            q=(1.0 - fallback_quantile),
            tail_direction=tail_direction,
        )
        low = float(meta.get("threshold_low", np.nan))
        high = float(meta.get("threshold_high", np.nan))
        low, high, zero_guard_tag = _adjust_zero_threshold_for_dist(
            values=vals,
            tail_direction=tail_direction,
            low=low,
            high=high,
        )
        if zero_guard_tag != "none":
            fail = _build_fail_by_threshold(
                score=score,
                base=base,
                tail_direction=tail_direction,
                low=low,
                high=high,
            )
            meta["threshold_low"] = float(low)
            meta["threshold_high"] = float(high)
            meta["threshold_applied"] = float(high if tail_direction == "two_sided" else low)
            meta["source"] = f"{meta.get('source', 'quantile_tail')}|{zero_guard_tag}"
        return fail, meta

    q_df["direction_var_n"] = minmax_scale(q_df["direction_var"].to_numpy(dtype=float), nan_fill=1.0)
    q_df["residual_var_n"] = minmax_scale(q_df["residual_var"].to_numpy(dtype=float), nan_fill=1.0)
    q_df["ensemble_var_n"] = minmax_scale(q_df["ensemble_var"].to_numpy(dtype=float), nan_fill=1.0)
    q_df["compactness_n"] = minmax_scale(q_df["cluster_compactness"].to_numpy(dtype=float), nan_fill=1.0)

    q_df["instability_index"] = (
        q_df["direction_var_n"]
        + q_df["residual_var_n"]
        + q_df["ensemble_var_n"]
        + q_df["compactness_n"]
    ) / 4.0

    smooth = q_df["instability_index"].rolling(window=3, min_periods=1, center=True).mean()
    delta = smooth.diff().fillna(0.0)
    if len(delta) > 1:
        jump_pos = int(np.argmax(delta.to_numpy(dtype=float)))
        chosen_pos = max(0, jump_pos - 1)
    else:
        chosen_pos = 0

    chosen = q_df.iloc[int(chosen_pos)]
    low = float(chosen["threshold_low"])
    high = float(chosen["threshold_high"])
    fail = _build_fail_by_threshold(score=score, base=base, tail_direction=tail_direction, low=low, high=high)
    zero_guard_tag = str(chosen.get("zero_guard_tag", "none"))
    if zero_guard_tag == "nan":
        zero_guard_tag = "none"
    source = "dist_stability_jump" if zero_guard_tag == "none" else f"dist_stability_jump|{zero_guard_tag}"

    return fail, {
        "threshold_applied": float(high if tail_direction == "two_sided" else low),
        "threshold_low": low,
        "threshold_high": high,
        "median": float("nan"),
        "mad": float("nan"),
        "sigma": float("nan"),
        "source": source,
    }


def choose_rule_threshold_and_fail(
    rule: str,
    score: np.ndarray,
    base: np.ndarray,
    policy: str,
    y_bad: np.ndarray,
    label_known: np.ndarray,
    threshold_points: int,
    robust_z_k: float,
    tail_start_floor_k: float,
    tail_start_max_k: float,
    tail_start_min_survival: float,
    tail_start_max_survival: float,
    tail_start_grid_points: int,
    tail_direction: str,
    mad_eps: float,
    fallback_quantile: float,
    quantile_tail_q: float,
    dist_q_min: int,
    dist_q_max: int,
    dist_q_step: int,
    features: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, float]]:
    _ = (y_bad, label_known, threshold_points)

    tail_start_max_k_eff = float(tail_start_max_k)
    if (
        policy == "robust_z_tail_start"
        and str(rule).strip().lower() == "output"
        and np.isfinite(tail_start_max_k_eff)
        and tail_start_max_k_eff <= 8.0
    ):
        tail_start_max_k_eff = float("nan")

    if policy == "robust_z":
        return robust_z_fail_mask(
            score=score,
            base=base,
            k=float(robust_z_k),
            tail_direction=tail_direction,
            mad_eps=float(mad_eps),
        )

    if policy == "robust_z_tail_start":
        return robust_z_tail_start_fail_mask(
            score=score,
            base=base,
            robust_z_k=float(robust_z_k),
            floor_k=float(tail_start_floor_k),
            max_k=float(tail_start_max_k_eff),
            tail_direction=tail_direction,
            mad_eps=float(mad_eps),
            tail_start_min_survival=float(tail_start_min_survival),
            tail_start_max_survival=float(tail_start_max_survival),
            tail_start_grid_points=int(tail_start_grid_points),
        )

    if policy == "quantile_tail":
        return quantile_tail_fail_mask(
            score=score,
            base=base,
            q=float(quantile_tail_q),
            tail_direction=tail_direction,
        )

    if policy == "dist_stability_jump":
        return dist_stability_jump_fail_mask(
            score=score,
            base=base,
            tail_direction=tail_direction,
            q_min=int(dist_q_min),
            q_max=int(dist_q_max),
            q_step=int(dist_q_step),
            features=features,
            fallback_quantile=float(fallback_quantile),
        )

    # Guard fallback.
    fail, meta = quantile_tail_fail_mask(
        score=score,
        base=base,
        q=(1.0 - float(np.clip(fallback_quantile, 1e-6, 0.999999))),
        tail_direction=tail_direction,
    )
    meta["source"] = "policy_fallback_quantile"
    return fail, meta


__all__ = [
    "AVAILABLE_COL",
    "SIGNAL_COL",
    "resolve_signal_col",
    "binary_metrics",
    "choose_rule_threshold_and_fail",
    "compute_labels_bad",
    "compute_policy_features",
    "derive_tristate_thresholds_from_fail",
    "trigger_mask",
]

# Backward compatibility exports.
SCORE_COL = SIGNAL_COL


def main() -> None:
    """CLI entrypoint for threshold policies module."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Threshold policy utilities for final metric runtime.",
        epilog="This module contains the minimal, deployment-oriented subset of threshold selection logic.",
    )
    parser.add_argument("--version", action="version", version="1.0.0")
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show module information",
    )

    args = parser.parse_args()

    if args.info:
        print("Threshold Policies Module")
        print("=" * 50)
        print("\nAvailable functions:")
        for name in __all__:
            print(f"  - {name}")
        print("\nKey utility functions:")
        print("  - choose_rule_threshold_and_fail(): Select threshold for a rule")
        print("  - derive_tristate_thresholds_from_fail(): Compute core/tail_start/exceptional")
        print("  - compute_policy_features(): Extract distribution features from signals")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
