#!/usr/bin/env python3
"""Test-only threshold policy experiment helpers.

This module is intentionally scoped for final_metric/test experiments.
It does not mutate production threshold modules.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
FINAL_DIR = THIS_DIR.parent
if str(FINAL_DIR) not in sys.path:
    sys.path.insert(0, str(FINAL_DIR))

try:
    from final_metric_refactor.threshold import choose_rule_threshold_and_fail
except Exception:  # pragma: no cover
    choose_rule_threshold_and_fail = None

EPS = 1e-12


def _fmt(v: float) -> str:
    if np.isfinite(v):
        return f"{float(v):.4g}"
    return "nan"


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


def estimate_tail_start_k_logccdf(
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


def estimate_gap_jump_k(
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
    max_surv_eff = min(0.95, max(float(min_survival) + 1e-6, float(max_survival)))

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

    idxs = np.where(valid & np.isfinite(z_gap))[0]
    if len(idxs) == 0:
        return float("nan"), float("nan"), "none"
    best = int(idxs[np.argmax(z_gap[idxs])])
    best_z = float(z_gap[best])
    if best_z < 2.5:
        return float("nan"), best_z, "none"
    k_gap = float(z[best + 1])
    if best_z >= 4.0:
        return k_gap, best_z, "strong"
    return k_gap, best_z, "weak"


def estimate_gap_jump_k_first_strong(
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
    max_surv_eff = min(0.95, max(float(min_survival) + 1e-6, float(max_survival)))

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
        return float(z[idx + 1]), float(z_gap[idx]), "strong"

    idxs = np.where(valid & np.isfinite(z_gap))[0]
    if len(idxs) == 0:
        return float("nan"), float("nan"), "none"
    best = int(idxs[np.argmax(z_gap[idxs])])
    return float("nan"), float(z_gap[best]), "none"


def estimate_extreme_hard_k(
    z_for_tail: np.ndarray,
    k_eff: float,
    mad_eps: float,
    strong_z: float,
    min_tail_points: int,
    min_keep_points: int,
    min_tail_quantile: float,
    min_delta_k: float,
) -> tuple[float, float, str]:
    z = np.asarray(z_for_tail, dtype=float)
    z = z[np.isfinite(z)]
    z = z[z >= 0.0]
    if len(z) < max(40, int(min_tail_points)):
        return float("nan"), float("nan"), "none"
    if not np.isfinite(k_eff):
        return float("nan"), float("nan"), "none"

    tail = np.sort(z[z >= float(k_eff)])
    if len(tail) < int(min_tail_points):
        return float("nan"), float("nan"), "none"

    dz = np.diff(tail)
    if len(dz) < 8:
        return float("nan"), float("nan"), "none"

    start_i = int(np.floor(max(0.0, min(0.95, float(min_tail_quantile))) * max(len(dz) - 1, 1)))
    valid = np.zeros(len(dz), dtype=bool)
    valid[start_i:] = True
    valid &= dz > 0.0
    if np.sum(valid) < 5:
        return float("nan"), float("nan"), "none"

    ref = dz[valid]
    sigma = robust_sigma_1d(ref, mad_eps=float(mad_eps))
    if not np.isfinite(sigma) or sigma <= float(mad_eps):
        return float("nan"), float("nan"), "none"

    med = float(np.median(ref))
    z_gap = (dz - med) / max(sigma, float(mad_eps))
    strong = np.where(valid & np.isfinite(z_gap) & (z_gap >= float(strong_z)))[0]

    for idx in strong:
        k_cand = float(tail[idx + 1])
        if (k_cand - float(k_eff)) < float(min_delta_k):
            continue
        keep = int(np.sum(tail >= k_cand))
        if keep < int(min_keep_points):
            continue
        return k_cand, float(z_gap[idx]), "internal_strong"

    idxs = np.where(valid & np.isfinite(z_gap))[0]
    if len(idxs) == 0:
        return float("nan"), float("nan"), "none"
    best = int(idxs[np.argmax(z_gap[idxs])])
    return float("nan"), float(z_gap[best]), "none"


def pick_tail_k(
    *,
    k_curv: float,
    k_gap: float,
    gap_z: float,
    gap_strong_z: float,
    gap_weak_z: float,
) -> tuple[float, str]:
    if np.isfinite(k_gap) and np.isfinite(gap_z) and float(gap_z) >= float(gap_strong_z):
        return float(k_gap), "gap_strong"
    if np.isfinite(k_curv):
        return float(k_curv), "curv"
    if np.isfinite(k_gap) and np.isfinite(gap_z) and float(gap_z) >= float(gap_weak_z):
        return float(k_gap), "gap_weak"
    return float("nan"), "none"


def robust_z_tail_start_test(
    *,
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
    gap_strong_z: float,
    gap_weak_z: float,
    soft_k_floor: float,
) -> tuple[np.ndarray, dict[str, float]]:
    n = len(score)
    fail = np.zeros(n, dtype=bool)
    vals = np.asarray(score, dtype=float)[base]
    vals = vals[np.isfinite(vals)]

    meta: dict[str, float] = {
        "threshold_applied": float("nan"),
        "threshold_low": float("nan"),
        "threshold_high": float("nan"),
        "median": float("nan"),
        "mad": float("nan"),
        "sigma": float("nan"),
        "k_tail_curv": float("nan"),
        "k_tail_gap": float("nan"),
        "k_tail_gap_first": float("nan"),
        "k_eff": float("nan"),
        "k_hard": float("nan"),
        "k_soft": float("nan"),
        "gap_z": float("nan"),
        "gap_first_z": float("nan"),
        "hard_gap_z": float("nan"),
        "source": "robust_z_tail_start_test",
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
        meta["source"] = f"robust_z_tail_start_test_{scale_source}"
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
    max_surv_eff = min(
        0.95,
        max(float(tail_start_min_survival) + 1e-6, float(tail_start_max_survival)),
    )

    k_tail_curv = estimate_tail_start_k_logccdf(
        z_for_tail=z_tail,
        min_survival=float(tail_start_min_survival),
        max_survival=max_surv_eff,
        grid_points=int(tail_start_grid_points),
    )
    k_tail_gap, gap_z, gap_kind = estimate_gap_jump_k(
        z_for_tail=z_tail,
        min_survival=float(tail_start_min_survival),
        max_survival=max_surv_eff,
        mad_eps=float(mad_eps),
    )
    k_tail_gap_first, gap_first_z, gap_first_kind = estimate_gap_jump_k_first_strong(
        z_for_tail=z_tail,
        min_survival=float(tail_start_min_survival),
        max_survival=max_surv_eff,
        mad_eps=float(mad_eps),
    )

    k_tail, tail_pick = pick_tail_k(
        k_curv=k_tail_curv,
        k_gap=k_tail_gap,
        gap_z=gap_z,
        gap_strong_z=float(gap_strong_z),
        gap_weak_z=float(gap_weak_z),
    )
    meta["k_tail_curv"] = float(k_tail_curv) if np.isfinite(k_tail_curv) else float("nan")
    meta["k_tail_gap"] = float(k_tail_gap) if np.isfinite(k_tail_gap) else float("nan")
    meta["k_tail_gap_first"] = float(k_tail_gap_first) if np.isfinite(k_tail_gap_first) else float("nan")
    meta["gap_z"] = float(gap_z) if np.isfinite(gap_z) else float("nan")
    meta["gap_first_z"] = float(gap_first_z) if np.isfinite(gap_first_z) else float("nan")

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
            f"tailstart_gap={_fmt(k_tail_gap)}|tailstart_curv={_fmt(k_tail_curv)}|"
            f"gap_z={_fmt(gap_z)}|gap_kind={gap_kind}|tail_pick={tail_pick}|"
            f"posmass={pos_mass:.4g}|maxsurv={max_surv_eff:.4g}"
        )
    else:
        k_eff = min(k_upper, max(k_floor, float(k0)))
        source_tail = (
            f"tailstart_missing|tailstart_gap={_fmt(k_tail_gap)}|tailstart_curv={_fmt(k_tail_curv)}|"
            f"gap_z={_fmt(gap_z)}|gap_kind={gap_kind}|tail_pick={tail_pick}|"
            f"posmass={pos_mass:.4g}|maxsurv={max_surv_eff:.4g}"
        )
    meta["k_eff"] = float(k_eff) if np.isfinite(k_eff) else float("nan")

    # Hard candidate for extreme cases: re-split inside the already detected tail.
    hard_z = max(4.0, float(gap_strong_z))
    min_tail_points = max(32, int(0.04 * len(z_tail)))
    min_keep_points = max(12, int(0.01 * len(z_tail)))
    k_hard_ext, hard_gap_z, hard_pick = estimate_extreme_hard_k(
        z_for_tail=z_tail,
        k_eff=float(k_eff),
        mad_eps=float(mad_eps),
        strong_z=float(hard_z),
        min_tail_points=int(min_tail_points),
        min_keep_points=int(min_keep_points),
        min_tail_quantile=0.60,
        min_delta_k=0.15,
    )
    if np.isfinite(k_hard_ext):
        # Hard is allowed to move beyond max_k cap when a clear internal split exists.
        k_hard = max(float(k_eff), float(k_hard_ext))
    else:
        k_hard = float(k_eff)
    meta["k_hard"] = float(k_hard) if np.isfinite(k_hard) else float("nan")
    meta["hard_gap_z"] = float(hard_gap_z) if np.isfinite(hard_gap_z) else float("nan")

    # Soft candidate for fail/warn: first onset gap -> curvature -> weak max-gap -> k0.
    # Use gap_weak_z for onset so steady long-tail onset is not dropped by a strict strong-z gate.
    if np.isfinite(k_tail_gap_first) and np.isfinite(gap_first_z) and float(gap_first_z) >= float(gap_weak_z):
        k_soft_raw = float(k_tail_gap_first)
        soft_pick = "gap_first_onset"
    elif np.isfinite(k_tail_curv):
        k_soft_raw = float(k_tail_curv)
        soft_pick = "curv"
    elif np.isfinite(k_tail_gap) and np.isfinite(gap_z) and float(gap_z) >= float(gap_weak_z):
        k_soft_raw = float(k_tail_gap)
        soft_pick = "gap_max_weak"
    else:
        k_soft_raw = float(k0)
        soft_pick = "k0"

    k_soft_floor_eff = max(float(k_floor), float(soft_k_floor))
    k_soft = min(k_upper, max(k_soft_floor_eff, k_soft_raw))
    if np.isfinite(k_eff) and k_soft > k_eff:
        k_soft = float(k_eff)
    meta["k_soft"] = float(k_soft) if np.isfinite(k_soft) else float("nan")

    if tail_direction == "lower":
        fail[base] = z[base] < -k_eff
        low = med - k_eff * sigma
        meta["threshold_applied"] = low
        meta["threshold_low"] = low
    elif tail_direction == "two_sided":
        fail[base] = np.abs(z[base]) > k_eff
        lo = med - k_eff * sigma
        hi = med + k_eff * sigma
        meta["threshold_low"] = lo
        meta["threshold_high"] = hi
        meta["threshold_applied"] = hi
    else:
        fail[base] = z[base] > k_eff
        high = med + k_eff * sigma
        meta["threshold_applied"] = high
        meta["threshold_low"] = high

    meta["source"] = (
        f"robust_z_tail_start_test_{scale_source}|{source_tail}|{floor_source}|"
        f"k_eff={k_eff:.4g}|soft_pick={soft_pick}|soft_floor={k_soft_floor_eff:.4g}|k_soft={k_soft:.4g}|"
        f"gap_first_kind={gap_first_kind}|hard_pick={hard_pick}|k_hard={k_hard:.4g}|hard_gap_z={_fmt(hard_gap_z)}"
    )
    return fail, meta


def _run_core_policy(
    *,
    policy: str,
    rule: str,
    score: np.ndarray,
    base: np.ndarray,
    tail_direction: str,
    mad_eps: float,
    threshold_points: int,
    robust_z_k: float,
    tail_start_floor_k: float,
    tail_start_max_k: float,
    tail_start_min_survival: float,
    tail_start_max_survival: float,
    tail_start_grid_points: int,
    fallback_quantile: float,
    dist_q_min: int,
    dist_q_max: int,
    dist_q_step: int,
    features: dict[str, np.ndarray] | None,
) -> tuple[np.ndarray, dict[str, float]]:
    n = len(score)
    if choose_rule_threshold_and_fail is None:
        return np.zeros(n, dtype=bool), {
            "threshold_applied": float("nan"),
            "threshold_low": float("nan"),
            "threshold_high": float("nan"),
            "median": float("nan"),
            "mad": float("nan"),
            "sigma": float("nan"),
            "source": "policy_module_missing",
        }

    if features is None:
        return np.zeros(n, dtype=bool), {
            "threshold_applied": float("nan"),
            "threshold_low": float("nan"),
            "threshold_high": float("nan"),
            "median": float("nan"),
            "mad": float("nan"),
            "sigma": float("nan"),
            "source": "policy_features_missing",
        }

    y_bad = np.zeros(n, dtype=bool)
    label_known = np.zeros(n, dtype=bool)
    fail, meta = choose_rule_threshold_and_fail(
        rule=rule,
        score=score,
        base=base,
        policy=policy,
        y_bad=y_bad,
        label_known=label_known,
        threshold_points=int(threshold_points),
        robust_z_k=float(robust_z_k),
        tail_start_floor_k=float(tail_start_floor_k),
        tail_start_max_k=float(tail_start_max_k),
        tail_start_min_survival=float(tail_start_min_survival),
        tail_start_max_survival=float(tail_start_max_survival),
        tail_start_grid_points=int(tail_start_grid_points),
        tail_direction=tail_direction,
        mad_eps=float(mad_eps),
        fallback_quantile=float(fallback_quantile),
        quantile_tail_q=float(1.0 - np.clip(float(fallback_quantile), 1e-6, 0.999999)),
        dist_q_min=int(dist_q_min),
        dist_q_max=int(dist_q_max),
        dist_q_step=int(dist_q_step),
        features=features,
    )
    return fail, meta


def dist_stability_jump_test(
    *,
    rule: str,
    score: np.ndarray,
    base: np.ndarray,
    tail_direction: str,
    mad_eps: float,
    threshold_points: int,
    robust_z_k: float,
    tail_start_floor_k: float,
    tail_start_max_k: float,
    tail_start_min_survival: float,
    tail_start_max_survival: float,
    tail_start_grid_points: int,
    fallback_quantile: float,
    dist_q_min: int,
    dist_q_max: int,
    dist_q_step: int,
    features: dict[str, np.ndarray] | None,
) -> tuple[np.ndarray, dict[str, float]]:
    fail, meta = _run_core_policy(
        policy="dist_stability_jump",
        rule=rule,
        score=score,
        base=base,
        tail_direction=tail_direction,
        mad_eps=mad_eps,
        threshold_points=threshold_points,
        robust_z_k=robust_z_k,
        tail_start_floor_k=tail_start_floor_k,
        tail_start_max_k=tail_start_max_k,
        tail_start_min_survival=tail_start_min_survival,
        tail_start_max_survival=tail_start_max_survival,
        tail_start_grid_points=tail_start_grid_points,
        fallback_quantile=fallback_quantile,
        dist_q_min=dist_q_min,
        dist_q_max=dist_q_max,
        dist_q_step=dist_q_step,
        features=features,
    )
    return fail, meta


def quantile_tail_guard_test(
    *,
    rule: str,
    score: np.ndarray,
    base: np.ndarray,
    tail_direction: str,
    mad_eps: float,
    threshold_points: int,
    robust_z_k: float,
    tail_start_floor_k: float,
    tail_start_max_k: float,
    tail_start_min_survival: float,
    tail_start_max_survival: float,
    tail_start_grid_points: int,
    fallback_quantile: float,
    dist_q_min: int,
    dist_q_max: int,
    dist_q_step: int,
    features: dict[str, np.ndarray] | None,
) -> tuple[np.ndarray, dict[str, float]]:
    fail, meta = _run_core_policy(
        policy="quantile_tail",
        rule=rule,
        score=score,
        base=base,
        tail_direction=tail_direction,
        mad_eps=mad_eps,
        threshold_points=threshold_points,
        robust_z_k=robust_z_k,
        tail_start_floor_k=tail_start_floor_k,
        tail_start_max_k=tail_start_max_k,
        tail_start_min_survival=tail_start_min_survival,
        tail_start_max_survival=tail_start_max_survival,
        tail_start_grid_points=tail_start_grid_points,
        fallback_quantile=fallback_quantile,
        dist_q_min=dist_q_min,
        dist_q_max=dist_q_max,
        dist_q_step=dist_q_step,
        features=features,
    )
    source = str(meta.get("source", ""))
    meta["source"] = f"quantile_tail_guard_test|{source}" if source else "quantile_tail_guard_test"
    return fail, meta


__all__ = [
    "dist_stability_jump_test",
    "estimate_gap_jump_k",
    "estimate_tail_start_k_logccdf",
    "pick_tail_k",
    "quantile_tail_guard_test",
    "robust_z_tail_start_test",
]
