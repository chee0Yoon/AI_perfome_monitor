#!/usr/bin/env python3
"""Build test-only multi-tier thresholds from existing row_results.

This script is intentionally test-scoped:
- input: existing row_results CSV only
- output: threshold tables (+ optional diagnostics html)
- no mutation of production runtime logic
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from threshold_policy_experiment import (
    dist_stability_jump_test,
    quantile_tail_guard_test,
    robust_z_tail_start_test,
)

try:
    import plotly.graph_objects as go
    import plotly.io as pio
except Exception:  # pragma: no cover
    go = None
    pio = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from final_metric_refactor.config import (
    RUNTIME_RULE_AVAILABLE_COL_NOMASK,
    RUNTIME_RULE_ORDER,
    RUNTIME_RULE_SIGNAL_COL_NOMASK,
    normalize_runtime_rule_key,
)


RULE_ORDER = list(RUNTIME_RULE_ORDER)
RULE_SCORE_COL_NOMASK = dict(RUNTIME_RULE_SIGNAL_COL_NOMASK)
RULE_AVAILABLE_COL_NOMASK = dict(RUNTIME_RULE_AVAILABLE_COL_NOMASK)

RULE_LABEL = {
    "output": "Output",
    "direction": "Direction",
    "length": "Length",
    "diff_residual": "Diff Residual",
    "delta_ridge_ens": "Delta Ridge Ensemble",
    "similar_input_conflict": "Similar Input Conflict",
    "discourse_instability": "Discourse Instability",
    "contradiction": "Contradiction",
    "self_contradiction": "Self Contradiction",
}

EPS = 1e-12


def parse_rules(raw: str) -> list[str]:
    out: list[str] = []
    for tok in str(raw).split(","):
        t = normalize_runtime_rule_key(tok)
        if not t:
            continue
        if t in RULE_ORDER and t not in out:
            out.append(t)
    return out or list(RULE_ORDER)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test-only threshold builder from row_results")
    p.add_argument("--row-results-csv", required=True)
    p.add_argument("--output-dir", default="", help="Default: <row_results_dir>/threshold_test_report")
    p.add_argument("--report-dir-name", default="report")
    p.add_argument("--tag", default="threshold_test")
    p.add_argument("--mode", default="nomask", choices=["nomask", "mask"])
    p.add_argument("--rules", default=",".join(RULE_ORDER))
    p.add_argument("--tail-direction", default="two_sided", choices=["upper", "lower", "two_sided"])
    p.add_argument("--grid-points", type=int, default=160)
    p.add_argument("--min-support-rows", type=int, default=16)
    p.add_argument("--min-survival", type=float, default=0.005)
    p.add_argument("--max-survival", type=float, default=0.30)
    p.add_argument("--lambda1", type=float, default=0.8)
    p.add_argument("--lambda2", type=float, default=0.8)
    p.add_argument("--hend-l", type=int, default=3)
    p.add_argument("--mad-eps", type=float, default=1e-9)
    p.add_argument("--fallback-mode", default="quantile_tail", choices=["none", "quantile_tail"])
    p.add_argument("--fallback-warn-quantile", type=float, default=0.99)
    p.add_argument("--fallback-fail-quantile", type=float, default=0.995)
    p.add_argument("--fallback-hard-quantile", type=float, default=0.998)
    p.add_argument("--main-threshold-policy", default="hybrid", choices=["hybrid", "derivative"])
    p.add_argument("--exp-policy", default="delta_gap_finitefb_v1", choices=["delta_gap_finitefb_v1"])
    # Recommended test profile from target experiments (closest to user goals)
    p.add_argument("--gap-strong-z", type=float, default=6.0)
    p.add_argument("--gap-weak-z", type=float, default=3.5)
    p.add_argument("--fallback-trigger", default="finite_only", choices=["finite_only"])
    p.add_argument("--threshold-points", type=int, default=260)
    p.add_argument("--robust-z-k", type=float, default=3.5)
    p.add_argument("--tail-start-floor-k", type=float, default=float("nan"))
    p.add_argument("--tail-start-max-k", type=float, default=8.0)
    p.add_argument("--tail-start-min-survival", type=float, default=0.005)
    p.add_argument("--tail-start-max-survival", type=float, default=0.30)
    p.add_argument("--tail-start-grid-points", type=int, default=160)
    p.add_argument("--soft-k-floor", type=float, default=1.5)
    p.add_argument("--fallback-quantile", type=float, default=0.995)
    p.add_argument("--dist-q-min", type=int, default=1)
    p.add_argument("--dist-q-max", type=int, default=30)
    p.add_argument("--dist-q-step", type=int, default=1)
    p.add_argument("--emit-plot", action="store_true", default=False)
    p.add_argument("--hist-bins", type=int, default=60)
    return p.parse_args()


def safe_bool_series(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.fillna(False).to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).to_numpy(dtype=float) != 0.0
    mapped = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False})
        .fillna(False)
    )
    return mapped.to_numpy(dtype=bool)


def mad_sigma(values: np.ndarray, mad_eps: float) -> tuple[float, str]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), "missing"

    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    sigma_mad = float(1.4826 * mad)
    if np.isfinite(sigma_mad) and sigma_mad > float(mad_eps):
        return sigma_mad, "mad"

    iqr = float(np.quantile(v, 0.75) - np.quantile(v, 0.25))
    sigma_iqr = float(iqr / 1.349) if np.isfinite(iqr) else float("nan")
    if np.isfinite(sigma_iqr) and sigma_iqr > float(mad_eps):
        return sigma_iqr, "iqr"

    q90_10 = float(np.quantile(v, 0.90) - np.quantile(v, 0.10))
    sigma_q = float(0.7413 * q90_10) if np.isfinite(q90_10) else float("nan")
    if np.isfinite(sigma_q) and sigma_q > float(mad_eps):
        return sigma_q, "q90_10"
    return float("nan"), "flat"


def first_true_run(mask: np.ndarray, run_len: int, start: int = 0) -> int | None:
    m = np.asarray(mask, dtype=bool)
    n = len(m)
    if run_len <= 1:
        idx = np.where(m[start:])[0]
        return int(start + idx[0]) if len(idx) else None
    if n < run_len:
        return None
    i = max(0, int(start))
    while i + run_len <= n:
        if np.all(m[i : i + run_len]):
            return int(i)
        i += 1
    return None


def score_col_for_mode(rule: str, mode: str) -> str:
    base = RULE_SCORE_COL_NOMASK[rule]
    return base if mode == "nomask" else base.replace("_nomask", "_mask")


def avail_col_for_mode(rule: str, mode: str) -> str | None:
    base = RULE_AVAILABLE_COL_NOMASK.get(rule)
    if base is None:
        return None
    return base if mode == "nomask" else base.replace("_nomask", "_mask")


def build_policy_features_for_mode(row_df: pd.DataFrame, mode: str) -> dict[str, np.ndarray]:
    n = len(row_df)

    d_col = score_col_for_mode("direction", mode)
    r_col = score_col_for_mode("diff_residual", mode)
    direction = (
        pd.to_numeric(row_df[d_col], errors="coerce").to_numpy(dtype=float)
        if d_col in row_df.columns
        else np.full(n, np.nan, dtype=float)
    )
    residual = (
        pd.to_numeric(row_df[r_col], errors="coerce").to_numpy(dtype=float)
        if r_col in row_df.columns
        else np.full(n, np.nan, dtype=float)
    )

    member_signal_suffix = f"_signal_{mode}"
    member_score_suffix = f"_score_{mode}"
    member_cols = [
        c
        for c in row_df.columns
        if c.startswith("delta_ridge_ens_member_")
        and (c.endswith(member_signal_suffix) or c.endswith(member_score_suffix))
    ]
    if member_cols:
        mats = np.column_stack([pd.to_numeric(row_df[c], errors="coerce").to_numpy(dtype=float) for c in member_cols])
        valid = np.isfinite(mats)
        valid_count = valid.sum(axis=1)
        mats_filled = np.where(valid, mats, 0.0)
        row_mean = np.divide(
            mats_filled.sum(axis=1),
            valid_count,
            out=np.zeros(n, dtype=float),
            where=valid_count > 0,
        )
        sq = np.where(valid, (mats_filled - row_mean[:, None]) ** 2, 0.0)
        row_var = np.divide(
            sq.sum(axis=1),
            valid_count,
            out=np.full(n, np.nan, dtype=float),
            where=valid_count > 0,
        )
        ensemble_std = np.sqrt(row_var)
    else:
        ensemble_std = np.full(n, np.nan, dtype=float)

    ox_col = f"output_pca_x_{mode}"
    oy_col = f"output_pca_y_{mode}"
    output_x = (
        pd.to_numeric(row_df[ox_col], errors="coerce").to_numpy(dtype=float)
        if ox_col in row_df.columns
        else np.full(n, np.nan, dtype=float)
    )
    output_y = (
        pd.to_numeric(row_df[oy_col], errors="coerce").to_numpy(dtype=float)
        if oy_col in row_df.columns
        else np.full(n, np.nan, dtype=float)
    )

    return {
        "direction_signal": direction,
        "residual_signal": residual,
        "residual_score": residual,  # backward compatibility for older policy helpers
        "ensemble_std": ensemble_std,
        "output_x": output_x,
        "output_y": output_y,
    }


def k_to_threshold(median: float, sigma: float, k: float, tail_direction: str) -> tuple[float, float, float]:
    if not np.isfinite(k) or not np.isfinite(median) or not np.isfinite(sigma):
        return float("nan"), float("nan"), float("nan")
    if tail_direction == "lower":
        t = float(median - k * sigma)
        return t, t, float("nan")
    if tail_direction == "two_sided":
        lo = float(median - k * sigma)
        hi = float(median + k * sigma)
        return hi, lo, hi
    t = float(median + k * sigma)
    return t, t, float("nan")


def quantile_to_threshold(values: np.ndarray, q: float, tail_direction: str) -> tuple[float, float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")

    q_eff = float(np.clip(q, 1e-6, 0.999999))
    if tail_direction == "lower":
        t = float(np.quantile(v, 1.0 - q_eff))
        return t, t, float("nan")
    if tail_direction == "two_sided":
        lo = float(np.quantile(v, (1.0 - q_eff) * 0.5))
        hi = float(np.quantile(v, 1.0 - (1.0 - q_eff) * 0.5))
        return hi, lo, hi
    t = float(np.quantile(v, q_eff))
    return t, t, float("nan")


@dataclass
class RuleThresholds:
    rule: str
    mode: str
    support_rows: int
    median: float
    sigma: float
    sigma_source: str
    k_warn: float
    k_fail: float
    k_hard_fail: float
    warn_threshold: float
    warn_threshold_low: float
    warn_threshold_high: float
    fail_threshold: float
    fail_threshold_low: float
    fail_threshold_high: float
    hard_fail_threshold: float
    hard_fail_threshold_low: float
    hard_fail_threshold_high: float
    status: str
    reason: str
    selected_method: str = ""


def is_strict_abnormal_test(
    meta: dict[str, Any],
    support_rows: int,
    min_support_rows: int,
    fallback_trigger: str,
) -> bool:
    thr = float(meta.get("threshold_applied", float("nan")))
    if not np.isfinite(thr):
        return True
    if support_rows < int(min_support_rows):
        return True
    if str(fallback_trigger) == "finite_only":
        return False
    source = str(meta.get("source", "") or "").strip().lower()
    if "tailstart_missing" in source or source in {"", "missing"}:
        return True
    return False


def tristate_from_fail_threshold(
    *,
    values: np.ndarray,
    tail_direction: str,
    fail_threshold: float,
    fail_low: float,
    fail_high: float,
) -> tuple[float, float, float, float, float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return (float("nan"),) * 6

    med = float(np.median(v))

    if tail_direction == "two_sided":
        lo = float(fail_low) if np.isfinite(fail_low) else float("nan")
        hi = float(fail_high) if np.isfinite(fail_high) else float("nan")
        if not np.isfinite(lo) or not np.isfinite(hi):
            if not np.isfinite(fail_threshold):
                return (float("nan"),) * 6
            d = abs(float(fail_threshold) - med)
            lo = med - d
            hi = med + d

        dlo = max(EPS, abs(med - lo))
        dhi = max(EPS, abs(hi - med))

        warn_lo = med - 0.8 * dlo
        warn_hi = med + 0.8 * dhi
        hard_lo = med - 1.2 * dlo
        hard_hi = med + 1.2 * dhi
        return float(warn_hi), float(warn_lo), float(warn_hi), float(hard_hi), float(hard_lo), float(hard_hi)

    if tail_direction == "lower":
        fail_t = float(fail_low if np.isfinite(fail_low) else fail_threshold)
        if not np.isfinite(fail_t):
            return (float("nan"),) * 6
        d = max(EPS, med - fail_t)
        warn_t = med - 0.8 * d
        hard_t = med - 1.2 * d
        return float(warn_t), float(warn_t), float("nan"), float(hard_t), float(hard_t), float("nan")

    fail_t = float(fail_low if np.isfinite(fail_low) else fail_threshold)
    if not np.isfinite(fail_t):
        return (float("nan"),) * 6
    d = max(EPS, fail_t - med)
    warn_t = med + 0.8 * d
    hard_t = med + 1.2 * d
    return float(warn_t), float(warn_t), float("nan"), float(hard_t), float(hard_t), float("nan")


def compute_rule_thresholds_hybrid(
    score: np.ndarray,
    base: np.ndarray,
    *,
    rule: str,
    mode: str,
    tail_direction: str,
    min_support_rows: int,
    mad_eps: float,
    threshold_points: int,
    robust_z_k: float,
    tail_start_floor_k: float,
    tail_start_max_k: float,
    tail_start_min_survival: float,
    tail_start_max_survival: float,
    tail_start_grid_points: int,
    soft_k_floor: float,
    fallback_quantile: float,
    dist_q_min: int,
    dist_q_max: int,
    dist_q_step: int,
    exp_policy: str,
    gap_strong_z: float,
    gap_weak_z: float,
    fallback_trigger: str,
    features: dict[str, np.ndarray] | None,
) -> RuleThresholds:
    support = int(np.sum(base))
    vals = np.asarray(score, dtype=float)[base]
    vals = vals[np.isfinite(vals)]
    med = float(np.median(vals)) if len(vals) else float("nan")
    sigma, sigma_source = mad_sigma(vals, mad_eps=mad_eps) if len(vals) else (float("nan"), "missing")

    default = RuleThresholds(
        rule=rule,
        mode=mode,
        support_rows=support,
        median=med,
        sigma=sigma,
        sigma_source=sigma_source,
        k_warn=float("nan"),
        k_fail=float("nan"),
        k_hard_fail=float("nan"),
        warn_threshold=float("nan"),
        warn_threshold_low=float("nan"),
        warn_threshold_high=float("nan"),
        fail_threshold=float("nan"),
        fail_threshold_low=float("nan"),
        fail_threshold_high=float("nan"),
        hard_fail_threshold=float("nan"),
        hard_fail_threshold_low=float("nan"),
        hard_fail_threshold_high=float("nan"),
        status="na",
        reason="unknown",
    )

    if support < int(min_support_rows):
        default.reason = "insufficient_support"
        return default
    if len(vals) < int(min_support_rows):
        default.reason = "insufficient_finite_support"
        return default
    if str(exp_policy) != "delta_gap_finitefb_v1":
        default.reason = f"unsupported_exp_policy:{exp_policy}"
        return default
    if features is None:
        default.reason = "hybrid_features_missing"
        return default

    tail_start_max_k_eff = float(tail_start_max_k)
    if (
        str(rule).strip().lower() == "output"
        and np.isfinite(tail_start_max_k_eff)
        and tail_start_max_k_eff <= 8.0
    ):
        tail_start_max_k_eff = float("nan")

    fail_1, meta_1 = robust_z_tail_start_test(
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
        gap_strong_z=float(gap_strong_z),
        gap_weak_z=float(gap_weak_z),
        soft_k_floor=float(soft_k_floor),
    )

    abnormal_1 = is_strict_abnormal_test(
        meta_1,
        support_rows=support,
        min_support_rows=min_support_rows,
        fallback_trigger=fallback_trigger,
    )
    if not abnormal_1:
        selected_method = "robust_z_tail_start_test"
        fail = fail_1
        meta = meta_1
    else:
        fail_2, meta_2 = dist_stability_jump_test(
            rule=rule,
            score=score,
            base=base,
            tail_direction=tail_direction,
            mad_eps=float(mad_eps),
            threshold_points=int(threshold_points),
            robust_z_k=float(robust_z_k),
            tail_start_floor_k=float(tail_start_floor_k),
            tail_start_max_k=float(tail_start_max_k),
            tail_start_min_survival=float(tail_start_min_survival),
            tail_start_max_survival=float(tail_start_max_survival),
            tail_start_grid_points=int(tail_start_grid_points),
            fallback_quantile=float(fallback_quantile),
            dist_q_min=int(dist_q_min),
            dist_q_max=int(dist_q_max),
            dist_q_step=int(dist_q_step),
            features=features,
        )
        abnormal_2 = is_strict_abnormal_test(
            meta_2,
            support_rows=support,
            min_support_rows=min_support_rows,
            fallback_trigger=fallback_trigger,
        )
        if not abnormal_2:
            selected_method = "dist_stability_jump_test"
            fail = fail_2
            meta = meta_2
        else:
            fail_3, meta_3 = quantile_tail_guard_test(
                rule=rule,
                score=score,
                base=base,
                tail_direction=tail_direction,
                mad_eps=float(mad_eps),
                threshold_points=int(threshold_points),
                robust_z_k=float(robust_z_k),
                tail_start_floor_k=float(tail_start_floor_k),
                tail_start_max_k=float(tail_start_max_k),
                tail_start_min_survival=float(tail_start_min_survival),
                tail_start_max_survival=float(tail_start_max_survival),
                tail_start_grid_points=int(tail_start_grid_points),
                fallback_quantile=float(fallback_quantile),
                dist_q_min=int(dist_q_min),
                dist_q_max=int(dist_q_max),
                dist_q_step=int(dist_q_step),
                features=features,
            )
            selected_method = "quantile_tail_guard_test"
            fail = fail_3
            meta = meta_3

    _ = fail
    fail_t = float(meta.get("threshold_applied", float("nan")))
    fail_lo = float(meta.get("threshold_low", float("nan")))
    fail_hi = float(meta.get("threshold_high", float("nan")))

    warn_t, warn_lo, warn_hi, hard_t, hard_lo, hard_hi = tristate_from_fail_threshold(
        values=vals,
        tail_direction=tail_direction,
        fail_threshold=fail_t,
        fail_low=fail_lo,
        fail_high=fail_hi,
    )

    k_warn = float("nan")
    k_fail = float("nan")
    k_hard = float("nan")
    soft_policy_tag = ""

    # Unified (all-rules) soft/hard mapping for robust-z tailstart method.
    # fail := k_soft, warn := 0.8*k_soft, hard := k_eff
    if selected_method == "robust_z_tail_start_test":
        med_meta = float(meta.get("median", np.nan))
        sigma_meta = float(meta.get("sigma", np.nan))
        k_soft_meta = float(meta.get("k_soft", np.nan))
        k_hard_meta = float(meta.get("k_hard", meta.get("k_eff", np.nan)))
        if (
            np.isfinite(med_meta)
            and np.isfinite(sigma_meta)
            and sigma_meta > float(mad_eps)
            and np.isfinite(k_soft_meta)
            and np.isfinite(k_hard_meta)
        ):
            k_fail = max(0.0, float(k_soft_meta))
            k_warn = max(0.0, 0.8 * k_fail)
            k_hard = max(k_fail, float(k_hard_meta))

            warn_t, warn_lo, warn_hi = k_to_threshold(
                median=med_meta,
                sigma=sigma_meta,
                k=float(k_warn),
                tail_direction=tail_direction,
            )
            fail_t, fail_lo, fail_hi = k_to_threshold(
                median=med_meta,
                sigma=sigma_meta,
                k=float(k_fail),
                tail_direction=tail_direction,
            )
            hard_t, hard_lo, hard_hi = k_to_threshold(
                median=med_meta,
                sigma=sigma_meta,
                k=float(k_hard),
                tail_direction=tail_direction,
            )
            soft_policy_tag = "soft_global_from_ksoft"
    elif np.isfinite(med) and np.isfinite(sigma) and sigma > float(mad_eps):
        # For fallback methods, expose approximate k values from fail threshold distance.
        if tail_direction == "two_sided":
            if np.isfinite(fail_lo) and np.isfinite(fail_hi):
                dist = max(abs(med - fail_lo), abs(fail_hi - med))
            elif np.isfinite(fail_t):
                dist = abs(fail_t - med)
            else:
                dist = float("nan")
        elif tail_direction == "lower":
            fail_eff = float(fail_lo) if np.isfinite(fail_lo) else float(fail_t)
            dist = med - fail_eff if np.isfinite(fail_eff) else float("nan")
        else:
            fail_eff = float(fail_lo) if np.isfinite(fail_lo) else float(fail_t)
            dist = fail_eff - med if np.isfinite(fail_eff) else float("nan")

        if np.isfinite(dist) and dist >= 0.0:
            k_fail = float(dist / max(sigma, float(mad_eps)))
            k_warn = float(0.8 * k_fail)
            k_hard = float(max(k_fail, 1.2 * k_fail))

    status = "ok" if np.isfinite(fail_t) else "na"
    reason = f"{selected_method}|{str(meta.get('source', ''))}"
    if soft_policy_tag:
        reason = f"{reason}|{soft_policy_tag}"
    if not np.isfinite(fail_t):
        reason = f"{reason}|non_finite_fail_threshold"

    return RuleThresholds(
        rule=rule,
        mode=mode,
        support_rows=support,
        median=med,
        sigma=sigma,
        sigma_source=sigma_source,
        k_warn=k_warn,
        k_fail=k_fail,
        k_hard_fail=k_hard,
        warn_threshold=warn_t,
        warn_threshold_low=warn_lo,
        warn_threshold_high=warn_hi,
        fail_threshold=fail_t,
        fail_threshold_low=fail_lo,
        fail_threshold_high=fail_hi,
        hard_fail_threshold=hard_t,
        hard_fail_threshold_low=hard_lo,
        hard_fail_threshold_high=hard_hi,
        status=status,
        reason=reason,
        selected_method=selected_method,
    )


def compute_rule_thresholds(
    score: np.ndarray,
    base: np.ndarray,
    *,
    tail_direction: str,
    grid_points: int,
    min_support_rows: int,
    min_survival: float,
    max_survival: float,
    lambda1: float,
    lambda2: float,
    hend_l: int,
    mad_eps: float,
    fallback_mode: str,
    fallback_warn_quantile: float,
    fallback_fail_quantile: float,
    fallback_hard_quantile: float,
    rule: str,
    mode: str,
) -> RuleThresholds:
    default = RuleThresholds(
        rule=rule,
        mode=mode,
        support_rows=int(np.sum(base)),
        median=float("nan"),
        sigma=float("nan"),
        sigma_source="missing",
        k_warn=float("nan"),
        k_fail=float("nan"),
        k_hard_fail=float("nan"),
        warn_threshold=float("nan"),
        warn_threshold_low=float("nan"),
        warn_threshold_high=float("nan"),
        fail_threshold=float("nan"),
        fail_threshold_low=float("nan"),
        fail_threshold_high=float("nan"),
        hard_fail_threshold=float("nan"),
        hard_fail_threshold_low=float("nan"),
        hard_fail_threshold_high=float("nan"),
        status="na",
        reason="unknown",
    )

    def fallback_result(reason: str, med_for_fallback: float, sigma_for_fallback: float, sigma_src: str) -> RuleThresholds:
        if str(fallback_mode) != "quantile_tail":
            default.reason = reason
            return default

        vals_fb = np.asarray(score, dtype=float)[base]
        vals_fb = vals_fb[np.isfinite(vals_fb)]
        if len(vals_fb) == 0:
            default.reason = f"{reason}|fallback_no_values"
            return default

        w_t, w_lo, w_hi = quantile_to_threshold(vals_fb, q=float(fallback_warn_quantile), tail_direction=tail_direction)
        f_t, f_lo, f_hi = quantile_to_threshold(vals_fb, q=float(fallback_fail_quantile), tail_direction=tail_direction)
        h_t, h_lo, h_hi = quantile_to_threshold(vals_fb, q=float(fallback_hard_quantile), tail_direction=tail_direction)

        # Keep monotonic order for one-sided applied thresholds.
        if tail_direction != "two_sided":
            wv = w_t
            fv = f_t
            hv = h_t
            if tail_direction == "upper":
                if np.isfinite(wv) and np.isfinite(fv) and wv > fv:
                    wv = fv
                if np.isfinite(fv) and np.isfinite(hv) and hv < fv:
                    hv = fv
                if np.isfinite(wv) and np.isfinite(hv) and wv > hv:
                    wv = hv
            else:
                if np.isfinite(wv) and np.isfinite(fv) and wv < fv:
                    wv = fv
                if np.isfinite(fv) and np.isfinite(hv) and hv > fv:
                    hv = fv
                if np.isfinite(wv) and np.isfinite(hv) and wv < hv:
                    wv = hv
            w_t, f_t, h_t = float(wv), float(fv), float(hv)
            w_lo, f_lo, h_lo = w_t, f_t, h_t
            w_hi, f_hi, h_hi = float("nan"), float("nan"), float("nan")

        return RuleThresholds(
            rule=rule,
            mode=mode,
            support_rows=int(np.sum(base)),
            median=float(med_for_fallback),
            sigma=float(sigma_for_fallback),
            sigma_source=str(sigma_src),
            k_warn=float("nan"),
            k_fail=float("nan"),
            k_hard_fail=float("nan"),
            warn_threshold=float(w_t),
            warn_threshold_low=float(w_lo),
            warn_threshold_high=float(w_hi),
            fail_threshold=float(f_t),
            fail_threshold_low=float(f_lo),
            fail_threshold_high=float(f_hi),
            hard_fail_threshold=float(h_t),
            hard_fail_threshold_low=float(h_lo),
            hard_fail_threshold_high=float(h_hi),
            status="fallback",
            reason=f"{reason}|fallback_quantile_tail",
            selected_method="derivative_fallback_quantile_tail",
        )

    support = int(np.sum(base))
    if support < int(min_support_rows):
        return fallback_result("insufficient_support", float("nan"), float("nan"), "missing")

    vals = np.asarray(score, dtype=float)[base]
    vals = vals[np.isfinite(vals)]
    if len(vals) < int(min_support_rows):
        return fallback_result("insufficient_finite_support", float("nan"), float("nan"), "missing")

    med = float(np.median(vals))
    sigma, sigma_source = mad_sigma(vals, mad_eps=mad_eps)
    default.median = med
    default.sigma = sigma
    default.sigma_source = sigma_source
    if not np.isfinite(sigma) or sigma <= float(mad_eps):
        return fallback_result("invalid_sigma", med, sigma, sigma_source)

    z = (np.asarray(score, dtype=float) - med) / max(sigma, float(mad_eps))
    z_tail = np.asarray(z[base], dtype=float)
    if tail_direction == "lower":
        z_tail = -z_tail
    elif tail_direction == "two_sided":
        z_tail = np.abs(z_tail)

    z_tail = z_tail[np.isfinite(z_tail)]
    if len(z_tail) < int(min_support_rows):
        return fallback_result("insufficient_tail_support", med, sigma, sigma_source)

    hi = float(np.quantile(z_tail, 0.995))
    if (not np.isfinite(hi)) or hi <= 0.0:
        return fallback_result("no_positive_tail", med, sigma, sigma_source)

    g = max(48, int(grid_points))
    k_grid = np.linspace(0.0, hi, g)
    s = np.array([float(np.mean(z_tail >= kv)) for kv in k_grid], dtype=float)
    if len(s) < 6:
        return fallback_result("short_grid", med, sigma, sigma_source)

    dk = np.diff(k_grid)
    dk = np.where(np.abs(dk) <= EPS, EPS, dk)
    delta = np.diff(s) / dk
    if len(delta) < 4:
        return fallback_result("short_delta", med, sigma, sigma_source)

    gamma = np.diff(delta) / dk[1:]
    if len(gamma) < max(4, int(hend_l)):
        return fallback_result("short_gamma", med, sigma, sigma_source)

    # Alignment on gamma domain.
    # delta_h/gamma_h/s_h share the same index length.
    delta_h = delta[1:]
    gamma_h = gamma
    s_h = s[2:]
    k_h = k_grid[2:]

    valid_h = (
        (s_h >= float(min_survival))
        & (s_h <= float(max_survival))
        & np.isfinite(delta_h)
        & np.isfinite(gamma_h)
    )
    if np.sum(valid_h) < max(6, int(hend_l)):
        return fallback_result("insufficient_valid_window", med, sigma, sigma_source)

    sigma_delta, _ = mad_sigma(delta_h[valid_h], mad_eps=mad_eps)
    sigma_gamma, _ = mad_sigma(gamma_h[valid_h], mad_eps=mad_eps)
    if (not np.isfinite(sigma_delta)) or (not np.isfinite(sigma_gamma)):
        return fallback_result("invalid_derivative_sigma", med, sigma, sigma_source)

    cond = valid_h & (
        (np.abs(delta_h) <= float(lambda1) * sigma_delta)
        & (np.abs(gamma_h) <= float(lambda2) * sigma_gamma)
    )

    fail_idx = first_true_run(cond, run_len=int(hend_l), start=0)
    if fail_idx is None:
        return fallback_result("hend_fail_not_found", med, sigma, sigma_source)

    valid_idxs = np.where(valid_h)[0]
    warn_idx = int(valid_idxs[np.argmin(gamma_h[valid_idxs])]) if len(valid_idxs) else None
    if warn_idx is None:
        return fallback_result("warn_knee_not_found", med, sigma, sigma_source)

    surv_fail = float(s_h[fail_idx]) if np.isfinite(s_h[fail_idx]) else float(min_survival)
    deep_tail = cond & (s_h <= max(float(min_survival), 0.5 * surv_fail))
    hard_idx = first_true_run(deep_tail, run_len=int(hend_l), start=int(fail_idx + hend_l))

    k_warn = float(k_h[warn_idx]) if np.isfinite(k_h[warn_idx]) else float("nan")
    k_fail = float(k_h[fail_idx]) if np.isfinite(k_h[fail_idx]) else float("nan")
    k_hard = float(k_h[hard_idx]) if hard_idx is not None and np.isfinite(k_h[hard_idx]) else float("nan")

    # Monotonic order enforcement among finite values.
    if np.isfinite(k_warn) and np.isfinite(k_fail) and k_warn > k_fail:
        k_warn = k_fail
    if np.isfinite(k_fail) and np.isfinite(k_hard) and k_hard < k_fail:
        k_hard = k_fail
    if np.isfinite(k_warn) and np.isfinite(k_hard) and k_warn > k_hard:
        k_warn = k_hard

    warn_t, warn_lo, warn_hi = k_to_threshold(median=med, sigma=sigma, k=k_warn, tail_direction=tail_direction)
    fail_t, fail_lo, fail_hi = k_to_threshold(median=med, sigma=sigma, k=k_fail, tail_direction=tail_direction)
    hard_t, hard_lo, hard_hi = k_to_threshold(median=med, sigma=sigma, k=k_hard, tail_direction=tail_direction)

    status = "ok" if np.isfinite(fail_t) else "partial"
    reason = "ok"
    if not np.isfinite(k_hard):
        reason = "hard_fail_not_found"
        status = "partial"

    return RuleThresholds(
        rule=rule,
        mode=mode,
        support_rows=support,
        median=med,
        sigma=sigma,
        sigma_source=sigma_source,
        k_warn=k_warn,
        k_fail=k_fail,
        k_hard_fail=k_hard,
        warn_threshold=warn_t,
        warn_threshold_low=warn_lo,
        warn_threshold_high=warn_hi,
        fail_threshold=fail_t,
        fail_threshold_low=fail_lo,
        fail_threshold_high=fail_hi,
        hard_fail_threshold=hard_t,
        hard_fail_threshold_low=hard_lo,
        hard_fail_threshold_high=hard_hi,
        status=status,
        reason=reason,
        selected_method="derivative",
    )


def write_threshold_plot_html(
    *,
    out_html: Path,
    row_df: pd.DataFrame,
    mode: str,
    tail_direction: str,
    thresholds_df: pd.DataFrame,
    bins: int,
) -> None:
    if go is None or pio is None:
        return

    html_parts: list[str] = []
    html_parts.append(
        "<html><head><meta charset='utf-8'><title>Threshold Test Plot</title>"
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:24px;}"
        "h1{margin-bottom:8px;}h2{margin-top:30px;margin-bottom:8px;}.chart{margin-bottom:18px;}</style>"
        "</head><body>"
    )
    html_parts.append(f"<h1>Threshold Test Plot ({mode})</h1>")

    include_js: str | bool = "cdn"
    for _, tr in thresholds_df.iterrows():
        rule = str(tr["rule"])
        sc = score_col_for_mode(rule, mode)
        if sc not in row_df.columns:
            continue

        score = pd.to_numeric(row_df[sc], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(score)
        hard_gate = (
            safe_bool_series(row_df["hard_gate_pass"])
            if "hard_gate_pass" in row_df.columns
            else np.ones(len(row_df), dtype=bool)
        )
        available = np.ones(len(row_df), dtype=bool)
        av_col = avail_col_for_mode(rule, mode)
        if av_col and av_col in row_df.columns:
            available = safe_bool_series(row_df[av_col])
        base = hard_gate & available & finite
        vals = score[base]
        if len(vals) == 0:
            continue

        lo = float(np.min(vals))
        hi = float(np.max(vals))
        if lo == hi:
            lo -= 0.5
            hi += 0.5
        edges = np.linspace(lo, hi, max(10, int(bins)) + 1)
        centers = (edges[:-1] + edges[1:]) * 0.5

        fig = go.Figure()

        known = np.zeros(len(score), dtype=bool)
        good = np.zeros(len(score), dtype=bool)
        if "label_is_correct" in row_df.columns:
            s = row_df["label_is_correct"]
            if s.dtype == bool:
                good = s.fillna(False).to_numpy(dtype=bool)
                known = (~s.isna()).to_numpy(dtype=bool) if s.hasnans else np.ones(len(s), dtype=bool)
            elif pd.api.types.is_numeric_dtype(s):
                v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
                known = np.isfinite(v)
                good = np.zeros(len(v), dtype=bool)
                good[known] = v[known] >= 0.5
            else:
                m = (
                    s.astype(str)
                    .str.strip()
                    .str.lower()
                    .map(
                        {
                            "true": True,
                            "false": False,
                            "1": True,
                            "0": False,
                            "yes": True,
                            "no": False,
                            "correct": True,
                            "incorrect": False,
                            "pass": True,
                            "fail": False,
                            "good": True,
                            "bad": False,
                        }
                    )
                )
                known = m.notna().to_numpy(dtype=bool)
                good = m.fillna(False).to_numpy(dtype=bool)

        if (not np.any(base & known)) and ("label_raw" in row_df.columns):
            lr = row_df["label_raw"].fillna("").astype(str).str.strip().str.lower()
            g = lr.isin({"correct", "true", "1", "pass", "good"}).to_numpy(dtype=bool)
            b = lr.isin({"incorrect", "false", "0", "fail", "bad"}).to_numpy(dtype=bool)
            known = g | b
            good = g

        good_mask = base & known & good
        bad_mask = base & known & (~good)

        cnt_good = np.histogram(score[good_mask], bins=edges)[0].astype(float)
        cnt_bad = np.histogram(score[bad_mask], bins=edges)[0].astype(float)
        denom = max(float(np.sum(base)), 1.0)
        cnt_good /= denom
        cnt_bad /= denom

        fig.add_trace(go.Bar(x=centers, y=cnt_good, name="good ratio", marker_color="#22c55e", opacity=0.58))
        fig.add_trace(go.Bar(x=centers, y=cnt_bad, name="bad ratio", marker_color="#ef4444", opacity=0.62))

        w = float(tr.get("warn_threshold", np.nan))
        f = float(tr.get("fail_threshold", np.nan))
        h = float(tr.get("hard_fail_threshold", np.nan))
        if np.isfinite(w):
            fig.add_vline(x=w, line_color="#f59e0b", line_width=2, line_dash="dot")
        if np.isfinite(f):
            fig.add_vline(x=f, line_color="#2563eb", line_width=2, line_dash="dash")
        if np.isfinite(h):
            fig.add_vline(x=h, line_color="#dc2626", line_width=2, line_dash="solid")

        fig.update_layout(
            template="plotly_white",
            title=f"{RULE_LABEL.get(rule, rule)} | good/bad histogram + warn/fail/hard",
            xaxis_title="score",
            yaxis_title="ratio",
            barmode="overlay",
            height=320,
            margin=dict(l=40, r=20, t=50, b=40),
            legend=dict(orientation="h"),
        )

        html_parts.append(f"<h2>{RULE_LABEL.get(rule, rule)}</h2>")
        html_parts.append(f"<div class='chart'>{pio.to_html(fig, full_html=False, include_plotlyjs=include_js)}</div>")
        include_js = False

    html_parts.append("</body></html>")
    out_html.write_text("\n".join(html_parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    row_csv = Path(args.row_results_csv).resolve()
    if not row_csv.exists():
        raise FileNotFoundError(row_csv)

    if args.output_dir:
        out_root = Path(args.output_dir).resolve()
    else:
        out_root = row_csv.parent / "threshold_test_report"
    out_root.mkdir(parents=True, exist_ok=True)
    report_dir = out_root / str(args.report_dir_name)
    report_dir.mkdir(parents=True, exist_ok=True)

    row_df = pd.read_csv(row_csv)
    rules = parse_rules(args.rules)
    mode = str(args.mode)
    main_threshold_policy = str(args.main_threshold_policy)

    if "hard_gate_pass" in row_df.columns:
        hard_gate = safe_bool_series(row_df["hard_gate_pass"])
    else:
        hard_gate = np.ones(len(row_df), dtype=bool)

    features: dict[str, np.ndarray] | None = None
    if main_threshold_policy == "hybrid":
        features = build_policy_features_for_mode(row_df=row_df, mode=mode)

    rows: list[dict[str, Any]] = []
    for rule in rules:
        sc = score_col_for_mode(rule, mode)
        av_col = avail_col_for_mode(rule, mode)

        if sc not in row_df.columns:
            rt = RuleThresholds(
                rule=rule,
                mode=mode,
                support_rows=0,
                median=float("nan"),
                sigma=float("nan"),
                sigma_source="missing",
                k_warn=float("nan"),
                k_fail=float("nan"),
                k_hard_fail=float("nan"),
                warn_threshold=float("nan"),
                warn_threshold_low=float("nan"),
                warn_threshold_high=float("nan"),
                fail_threshold=float("nan"),
                fail_threshold_low=float("nan"),
                fail_threshold_high=float("nan"),
                hard_fail_threshold=float("nan"),
                hard_fail_threshold_low=float("nan"),
                hard_fail_threshold_high=float("nan"),
                status="na",
                reason=f"missing_score_col:{sc}",
            )
            rows.append(rt.__dict__)
            continue

        score = pd.to_numeric(row_df[sc], errors="coerce").to_numpy(dtype=float)
        available = np.ones(len(row_df), dtype=bool)
        if av_col and av_col in row_df.columns:
            available = safe_bool_series(row_df[av_col])

        base = hard_gate & available & np.isfinite(score)
        if main_threshold_policy == "hybrid":
            rt = compute_rule_thresholds_hybrid(
                score=score,
                base=base,
                rule=rule,
                mode=mode,
                tail_direction=str(args.tail_direction),
                min_support_rows=int(args.min_support_rows),
                mad_eps=float(args.mad_eps),
                threshold_points=int(args.threshold_points),
                robust_z_k=float(args.robust_z_k),
                tail_start_floor_k=float(args.tail_start_floor_k),
                tail_start_max_k=float(args.tail_start_max_k),
                tail_start_min_survival=float(args.tail_start_min_survival),
                tail_start_max_survival=float(args.tail_start_max_survival),
                tail_start_grid_points=int(args.tail_start_grid_points),
                soft_k_floor=float(args.soft_k_floor),
                fallback_quantile=float(args.fallback_quantile),
                dist_q_min=int(args.dist_q_min),
                dist_q_max=int(args.dist_q_max),
                dist_q_step=int(args.dist_q_step),
                exp_policy=str(args.exp_policy),
                gap_strong_z=float(args.gap_strong_z),
                gap_weak_z=float(args.gap_weak_z),
                fallback_trigger=str(args.fallback_trigger),
                features=features,
            )
        else:
            rt = compute_rule_thresholds(
                score=score,
                base=base,
                tail_direction=str(args.tail_direction),
                grid_points=int(args.grid_points),
                min_support_rows=int(args.min_support_rows),
                min_survival=float(args.min_survival),
                max_survival=float(args.max_survival),
                lambda1=float(args.lambda1),
                lambda2=float(args.lambda2),
                hend_l=int(args.hend_l),
                mad_eps=float(args.mad_eps),
                fallback_mode=str(args.fallback_mode),
                fallback_warn_quantile=float(args.fallback_warn_quantile),
                fallback_fail_quantile=float(args.fallback_fail_quantile),
                fallback_hard_quantile=float(args.fallback_hard_quantile),
                rule=rule,
                mode=mode,
            )
        rows.append(rt.__dict__)

    thresholds_df = pd.DataFrame(rows)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stem = row_csv.stem
    base_name = f"{args.tag}_{stem}_{mode}_{ts}"

    thr_csv = report_dir / f"{base_name}_rule_thresholds.csv"
    thresholds_df.to_csv(thr_csv, index=False)

    compact_cols = [
        "rule",
        "mode",
        "selected_method",
        "status",
        "reason",
        "support_rows",
        "warn_threshold",
        "fail_threshold",
        "hard_fail_threshold",
    ]
    compact_csv = report_dir / f"{base_name}_rule_thresholds_compact.csv"
    thresholds_df[compact_cols].to_csv(compact_csv, index=False)

    cfg = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_results_csv": str(row_csv),
        "rules": rules,
        "mode": mode,
        "main_threshold_policy": main_threshold_policy,
        "exp_policy": str(args.exp_policy),
        "gap_strong_z": float(args.gap_strong_z),
        "gap_weak_z": float(args.gap_weak_z),
        "fallback_trigger": str(args.fallback_trigger),
        "tail_direction": args.tail_direction,
        "grid_points": int(args.grid_points),
        "min_support_rows": int(args.min_support_rows),
        "min_survival": float(args.min_survival),
        "max_survival": float(args.max_survival),
        "lambda1": float(args.lambda1),
        "lambda2": float(args.lambda2),
        "hend_l": int(args.hend_l),
        "mad_eps": float(args.mad_eps),
        "fallback_mode": str(args.fallback_mode),
        "fallback_warn_quantile": float(args.fallback_warn_quantile),
        "fallback_fail_quantile": float(args.fallback_fail_quantile),
        "fallback_hard_quantile": float(args.fallback_hard_quantile),
        "threshold_points": int(args.threshold_points),
        "robust_z_k": float(args.robust_z_k),
        "tail_start_floor_k": float(args.tail_start_floor_k),
        "tail_start_max_k": float(args.tail_start_max_k),
        "tail_start_min_survival": float(args.tail_start_min_survival),
        "tail_start_max_survival": float(args.tail_start_max_survival),
        "tail_start_grid_points": int(args.tail_start_grid_points),
        "soft_k_floor": float(args.soft_k_floor),
        "fallback_quantile": float(args.fallback_quantile),
        "dist_q_min": int(args.dist_q_min),
        "dist_q_max": int(args.dist_q_max),
        "dist_q_step": int(args.dist_q_step),
        "output_rule_thresholds_csv": str(thr_csv),
        "output_compact_csv": str(compact_csv),
    }
    cfg_json = report_dir / f"{base_name}_run_config.json"
    cfg_json.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    html_out = None
    if bool(args.emit_plot):
        html_out = report_dir / f"{base_name}_threshold_diagnostics.html"
        write_threshold_plot_html(
            out_html=html_out,
            row_df=row_df,
            mode=mode,
            tail_direction=str(args.tail_direction),
            thresholds_df=thresholds_df,
            bins=int(args.hist_bins),
        )

    print(f"[DONE] rule_thresholds_csv: {thr_csv}")
    print(f"[DONE] compact_csv: {compact_csv}")
    print(f"[DONE] run_config: {cfg_json}")
    if html_out is not None:
        print(f"[DONE] diagnostics_html: {html_out}")
    print("[SUMMARY]")
    print(
        thresholds_df[
            [
                "rule",
                "selected_method",
                "status",
                "reason",
                "support_rows",
                "warn_threshold",
                "fail_threshold",
                "hard_fail_threshold",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
