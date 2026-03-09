#!/usr/bin/env python3
"""Test-only experiment: threshold-free scoring vs current New_score.

Core idea:
- Score uses only continuous risk from signal geometry/statistics.
- Thresholds are used only for state labels and explain variables.
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
from pandas.errors import EmptyDataError

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots
except Exception:  # pragma: no cover
    go = None
    pio = None
    make_subplots = None

THIS_DIR = Path(__file__).resolve().parent
FINAL_DIR = THIS_DIR.parent
REPO_ROOT = FINAL_DIR.parent
if str(FINAL_DIR) not in sys.path:
    sys.path.insert(0, str(FINAL_DIR))

from final_metric_refactor.config import RUNTIME_RULE_AVAILABLE_COL_NOMASK, RUNTIME_RULE_SIGNAL_COL_NOMASK, SCORE_RUNTIME  # noqa: E402
from score.bundle_score import (  # noqa: E402
    EPS,
    _build_rule_stat,
    _conf_components,
    _quality_from_risk,
    _rate_on_base,
    _safe_rate,
    _threshold_map,
    classify_diag_quadrants,
    compute_bundle_scores,
    mad,
    state_array,
    to_bool_array,
)


@dataclass(frozen=True)
class CaseSpec:
    name: str
    row_results_csv: Path
    thresholds_csv: Path


@dataclass(frozen=True)
class ContinuousScoreConfig:
    z_fail_ref: float = 2.0
    z_hard_ref: float = 3.5
    tau_rate: float = 0.60
    tau_mass: float = 0.80
    core_quantile: float = 0.70
    gamma: float = 1.0
    rate_lambda_out: float = 3.0
    rate_lambda_other: float = 4.0
    w_core: float = 0.15
    w_rate: float = 0.55
    w_mass: float = 0.30
    hard_tail_penalty_alpha: float = 0.25
    na_neutral_score: float = 3.0
    sem_na_neutral_score: float = 3.0
    min_support_rows: int = 8


def default_case_paths() -> tuple[CaseSpec, CaseSpec]:
    good = CaseSpec(
        name="n100_good",
        row_results_csv=REPO_ROOT
        / "final_metric/results/new_score_n100_good_noguard_plot_dev_20260303/report/new_score_n100_good_noguard_plot_dev_n100_good_performance_source_row_results.csv",
        thresholds_csv=REPO_ROOT
        / "final_metric/results/new_score_n100_good_noguard_plot_dev_20260303/report/new_score_n100_good_noguard_plot_dev_n100_good_performance_source_thresholds_summary.csv",
    )
    bad = CaseSpec(
        name="n100_bad_1pct",
        row_results_csv=REPO_ROOT
        / "final_metric/results/new_score_n100_bad_noguard_plot_dev_20260303/report/new_score_n100_bad_noguard_plot_dev_n100_errors_about_1pct_source_row_results.csv",
        thresholds_csv=REPO_ROOT
        / "final_metric/results/new_score_n100_bad_noguard_plot_dev_20260303/report/new_score_n100_bad_noguard_plot_dev_n100_errors_about_1pct_source_thresholds_summary.csv",
    )
    return good, bad


def parse_args() -> argparse.Namespace:
    good_default, bad_default = default_case_paths()

    p = argparse.ArgumentParser(description="Threshold-free score experiment (test-only)")
    p.add_argument(
        "--case",
        action="append",
        default=[],
        help="Custom case in format: name::row_results_csv::thresholds_summary_csv (repeatable)",
    )
    p.add_argument("--good-name", default=good_default.name)
    p.add_argument("--good-row-results-csv", default=str(good_default.row_results_csv))
    p.add_argument("--good-thresholds-csv", default=str(good_default.thresholds_csv))
    p.add_argument("--bad-name", default=bad_default.name)
    p.add_argument("--bad-row-results-csv", default=str(bad_default.row_results_csv))
    p.add_argument("--bad-thresholds-csv", default=str(bad_default.thresholds_csv))

    p.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "final_metric/results/threshold_free_compare"),
    )
    p.add_argument("--tag", default="threshold_free")
    p.add_argument("--emit-plot", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--z-fail-ref", type=float, default=2.0)
    p.add_argument("--z-hard-ref", type=float, default=3.5)
    p.add_argument("--tau-rate", type=float, default=0.60)
    p.add_argument("--tau-mass", type=float, default=0.80)
    p.add_argument("--core-quantile", type=float, default=0.70)
    return p.parse_args()


def parse_cases(args: argparse.Namespace) -> list[CaseSpec]:
    out: list[CaseSpec] = []
    if args.case:
        for raw in args.case:
            parts = str(raw).split("::")
            if len(parts) != 3:
                raise ValueError(f"Invalid --case format: {raw}")
            name, row_csv, thr_csv = parts
            out.append(CaseSpec(name=str(name).strip(), row_results_csv=Path(row_csv).resolve(), thresholds_csv=Path(thr_csv).resolve()))
    else:
        out = [
            CaseSpec(
                name=str(args.good_name).strip(),
                row_results_csv=Path(str(args.good_row_results_csv)).resolve(),
                thresholds_csv=Path(str(args.good_thresholds_csv)).resolve(),
            ),
            CaseSpec(
                name=str(args.bad_name).strip(),
                row_results_csv=Path(str(args.bad_row_results_csv)).resolve(),
                thresholds_csv=Path(str(args.bad_thresholds_csv)).resolve(),
            ),
        ]
    for c in out:
        if not c.name:
            raise ValueError("Case name cannot be empty")
        if not c.row_results_csv.exists():
            raise FileNotFoundError(f"row_results.csv not found: {c.row_results_csv}")
        if not c.thresholds_csv.exists():
            raise FileNotFoundError(f"thresholds_summary.csv not found: {c.thresholds_csv}")
    return out


def _clip01(x: float) -> float:
    if not np.isfinite(x):
        return float("nan")
    return float(np.clip(float(x), 0.0, 1.0))


def _softplus(x: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(x, dtype=float)
    out = np.log1p(np.exp(np.clip(arr, -60.0, 60.0)))
    if np.isscalar(x):
        return float(out.item())
    return out


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(x, dtype=float)
    z = np.clip(arr, -60.0, 60.0)
    out = 1.0 / (1.0 + np.exp(-z))
    if np.isscalar(x):
        return float(out.item())
    return out


def _risk_from_scale(raw: float, scale: float) -> float:
    if (not np.isfinite(raw)) or (not np.isfinite(scale)):
        return float("nan")
    r = float(max(0.0, raw))
    s = float(max(float(EPS), scale))
    return float(np.clip(r / (r + s), 0.0, 1.0))


def _bundle_risk_noisy_or(axis_risks: dict[str, float], axis_weights: dict[str, float]) -> float:
    pairs: list[tuple[float, float]] = []
    for k, risk in axis_risks.items():
        if (k not in axis_weights) or (not np.isfinite(risk)):
            continue
        w = float(max(0.0, axis_weights[k]))
        if w <= 0.0:
            continue
        pairs.append((w, float(np.clip(risk, 0.0, 1.0))))
    if not pairs:
        return float("nan")
    ws = np.asarray([w for w, _ in pairs], dtype=float)
    ws = ws / float(np.sum(ws))
    rs = np.asarray([r for _, r in pairs], dtype=float)
    terms = np.clip(1.0 - ws * rs, 0.0, 1.0)
    return float(1.0 - np.prod(terms))


def _apply_soft_hard_tail_penalty(base_risk: float, hard_rate: float, mass_risk: float, alpha: float) -> tuple[float, float]:
    if not np.isfinite(base_risk):
        return float("nan"), float("nan")
    br = float(np.clip(base_risk, 0.0, 1.0))
    hr = float(np.clip(hard_rate, 0.0, 1.0)) if np.isfinite(hard_rate) else 0.0
    mr = float(np.clip(mass_risk, 0.0, 1.0)) if np.isfinite(mass_risk) else 0.0
    penalty = float(max(0.0, alpha) * hr * (0.5 + 0.5 * mr))
    return float(np.clip(br + penalty, 0.0, 1.0)), float(penalty)


def _band_from_state(rate_fail: float, rate_hard: float) -> str:
    if np.isfinite(rate_hard) and float(rate_hard) > 0.0:
        return "exceptional"
    if np.isfinite(rate_fail) and float(rate_fail) > 0.0:
        return "tail"
    return "core"


def _diag_cause(q10: float, q01: float, q11: float) -> str:
    vals = {"q10": q10, "q01": q01, "q11": q11}
    finite = {k: v for k, v in vals.items() if np.isfinite(v)}
    if not finite:
        return "진단 불가(유효 샘플 부족)"
    top = max(finite.items(), key=lambda kv: kv[1])[0]
    if top == "q10":
        return "분포 형태 이탈(Residual) 중심"
    if top == "q01":
        return "입력조건 예측 이탈(Ridge) 중심"
    return "둘 다(형태+예측) 동시 붕괴"


def _mean_or_nan(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_jsonable(v) for v in obj)
    return obj


def _extract_state_rates(row_df: pd.DataFrame, rule: str, valid_mask: np.ndarray) -> dict[str, float]:
    state_col = f"{rule}_state_nomask"
    if state_col not in row_df.columns:
        return {
            "pass_rate": float("nan"),
            "warn_rate": float("nan"),
            "fail_rate": float("nan"),
            "hard_rate": float("nan"),
        }
    state = row_df[state_col].fillna("").astype(str).str.strip().str.lower().to_numpy()
    m = np.asarray(valid_mask, dtype=bool)
    n = int(np.sum(m))
    if n <= 0:
        return {
            "pass_rate": float("nan"),
            "warn_rate": float("nan"),
            "fail_rate": float("nan"),
            "hard_rate": float("nan"),
        }
    s = state[m]
    return {
        "pass_rate": float(np.mean(s == "pass")),
        "warn_rate": float(np.mean(s == "warn")),
        "fail_rate": float(np.mean(s == "fail")),
        "hard_rate": float(np.mean(s == "hard_fail")),
    }


def _continuous_rule_profile(
    *,
    stat: Any,
    cfg: ContinuousScoreConfig,
    s_core: float,
    s_mass: float,
    rate_lambda: float,
    eta_hard: float,
) -> dict[str, Any]:
    n = len(stat.z)
    valid = np.asarray(stat.valid_mask, dtype=bool) & np.isfinite(np.asarray(stat.z, dtype=float))
    z_full = np.full(n, np.nan, dtype=float)
    p_fail_full = np.full(n, np.nan, dtype=float)
    p_hard_full = np.full(n, np.nan, dtype=float)
    if not np.any(valid):
        return {
            "valid_mask": valid,
            "z_cont": z_full,
            "p_fail_soft": p_fail_full,
            "p_hard_soft": p_hard_full,
            "rate_fail_soft": float("nan"),
            "rate_hard_soft": float("nan"),
            "tail_mass_cont": float("nan"),
            "core_mad_cont": float("nan"),
            "r_rate": float("nan"),
            "r_mass": float("nan"),
            "r_core": float("nan"),
            "q_rate": float("nan"),
            "q_mass": float("nan"),
            "q_core": float("nan"),
        }

    z = np.maximum(0.0, np.asarray(stat.z, dtype=float)[valid])
    z_full[valid] = z

    fail_soft = _sigmoid((z - float(cfg.z_fail_ref)) / max(float(EPS), float(cfg.tau_rate)))
    hard_soft = _sigmoid((z - float(cfg.z_hard_ref)) / max(float(EPS), float(cfg.tau_rate)))
    p_fail_full[valid] = fail_soft
    p_hard_full[valid] = hard_soft

    rate_fail_soft = float(np.mean(fail_soft))
    rate_hard_soft = float(np.mean(hard_soft))
    r_rate = _clip01(rate_fail_soft + float(rate_lambda) * rate_hard_soft)

    fail_soft_excess = _softplus((z - float(cfg.z_fail_ref)) / max(float(EPS), float(cfg.tau_mass)))
    hard_soft_excess = _softplus((z - float(cfg.z_hard_ref)) / max(float(EPS), float(cfg.tau_mass)))
    tail_mass_cont = float(np.mean(fail_soft_excess) + float(eta_hard) * np.mean(hard_soft_excess))
    r_mass = _risk_from_scale(tail_mass_cont, float(s_mass))

    if z.size >= 3:
        q_cut = float(np.quantile(z, float(np.clip(cfg.core_quantile, 0.50, 0.95))))
        core_slice = z[z <= q_cut]
        if core_slice.size < 3:
            core_slice = z
    else:
        core_slice = z
    core_mad_cont = float(mad(core_slice))
    r_core = _risk_from_scale(core_mad_cont, float(s_core))

    q_rate = _quality_from_risk(float(r_rate), gamma=float(cfg.gamma))
    q_mass = _quality_from_risk(float(r_mass), gamma=float(cfg.gamma))
    q_core = _quality_from_risk(float(r_core), gamma=float(cfg.gamma))
    return {
        "valid_mask": valid,
        "z_cont": z_full,
        "p_fail_soft": p_fail_full,
        "p_hard_soft": p_hard_full,
        "rate_fail_soft": float(rate_fail_soft),
        "rate_hard_soft": float(rate_hard_soft),
        "tail_mass_cont": float(tail_mass_cont),
        "core_mad_cont": float(core_mad_cont),
        "r_rate": float(r_rate),
        "r_mass": float(r_mass),
        "r_core": float(r_core),
        "q_rate": float(q_rate),
        "q_mass": float(q_mass),
        "q_core": float(q_core),
    }


def _threshold_explainer(
    *,
    row_df: pd.DataFrame,
    rule: str,
    stat: Any,
) -> dict[str, Any]:
    valid = np.asarray(stat.valid_mask, dtype=bool) & np.isfinite(np.asarray(stat.z, dtype=float))
    z = np.maximum(0.0, np.asarray(stat.z, dtype=float))
    zv = z[valid]
    k_fail = float(stat.k_fail) if np.isfinite(stat.k_fail) else float("nan")
    k_hard = float(stat.k_hard) if np.isfinite(stat.k_hard) else float("nan")

    if zv.size == 0 or (not np.isfinite(k_fail)):
        rate_fail = float("nan")
        excess_fail = float("nan")
    else:
        rate_fail = float(np.mean(zv >= k_fail))
        excess_fail = float(np.mean(np.maximum(0.0, zv - k_fail)))

    if zv.size == 0 or (not np.isfinite(k_hard)):
        rate_hard = float("nan")
        excess_hard = float("nan")
    else:
        rate_hard = float(np.mean(zv >= k_hard))
        excess_hard = float(np.mean(np.maximum(0.0, zv - k_hard)))

    state_rates = _extract_state_rates(row_df=row_df, rule=rule, valid_mask=valid)
    return {
        "n_valid": int(np.sum(valid)),
        "k_fail": (float(k_fail) if np.isfinite(k_fail) else None),
        "k_hard": (float(k_hard) if np.isfinite(k_hard) else None),
        "rate_fail": (float(rate_fail) if np.isfinite(rate_fail) else None),
        "rate_hard": (float(rate_hard) if np.isfinite(rate_hard) else None),
        "excess_fail": (float(excess_fail) if np.isfinite(excess_fail) else None),
        "excess_hard": (float(excess_hard) if np.isfinite(excess_hard) else None),
        "state_pass_rate": (float(state_rates["pass_rate"]) if np.isfinite(state_rates["pass_rate"]) else None),
        "state_warn_rate": (float(state_rates["warn_rate"]) if np.isfinite(state_rates["warn_rate"]) else None),
        "state_fail_rate": (float(state_rates["fail_rate"]) if np.isfinite(state_rates["fail_rate"]) else None),
        "state_hard_rate": (float(state_rates["hard_rate"]) if np.isfinite(state_rates["hard_rate"]) else None),
    }


def _diag_state_quadrants(diff_stat: Any, ridge_stat: Any) -> dict[str, float]:
    valid = np.asarray(diff_stat.valid_mask, dtype=bool) & np.asarray(ridge_stat.valid_mask, dtype=bool)
    n = int(np.sum(valid))
    if n <= 0:
        return {"q00": float("nan"), "q10": float("nan"), "q01": float("nan"), "q11": float("nan")}
    diff_bad = np.asarray(diff_stat.fail_mask | diff_stat.hard_mask, dtype=bool) & valid
    ridge_bad = np.asarray(ridge_stat.fail_mask | ridge_stat.hard_mask, dtype=bool) & valid
    return {
        "q00": float(np.mean((~diff_bad) & (~ridge_bad) & valid) * len(valid) / n),
        "q10": float(np.mean(diff_bad & (~ridge_bad) & valid) * len(valid) / n),
        "q01": float(np.mean((~diff_bad) & ridge_bad & valid) * len(valid) / n),
        "q11": float(np.mean(diff_bad & ridge_bad & valid) * len(valid) / n),
    }


def _score_runtime_from_defaults(args: argparse.Namespace) -> ContinuousScoreConfig:
    return ContinuousScoreConfig(
        z_fail_ref=float(args.z_fail_ref),
        z_hard_ref=float(args.z_hard_ref),
        tau_rate=float(args.tau_rate),
        tau_mass=float(args.tau_mass),
        core_quantile=float(args.core_quantile),
        gamma=float(getattr(SCORE_RUNTIME, "new_score_gamma", 1.0)),
        rate_lambda_out=3.0,
        rate_lambda_other=4.0,
        w_core=float(getattr(SCORE_RUNTIME, "new_score_w_core", 0.15)),
        w_rate=float(getattr(SCORE_RUNTIME, "new_score_w_rate", 0.55)),
        w_mass=float(getattr(SCORE_RUNTIME, "new_score_w_mass", 0.30)),
        hard_tail_penalty_alpha=float(getattr(SCORE_RUNTIME, "new_score_hard_tail_penalty_alpha", 0.25)),
        na_neutral_score=float(getattr(SCORE_RUNTIME, "new_score_na_neutral_score", 3.0)),
        sem_na_neutral_score=float(getattr(SCORE_RUNTIME, "new_score_sem_na_neutral_score", 3.0)),
        min_support_rows=int(getattr(SCORE_RUNTIME, "min_support_rows", 8)),
    )


def evaluate_case(case: CaseSpec, cfg: ContinuousScoreConfig) -> dict[str, Any]:
    row_df = pd.read_csv(case.row_results_csv)
    try:
        threshold_df = pd.read_csv(case.thresholds_csv)
    except EmptyDataError:
        threshold_df = pd.DataFrame()

    runtime = SCORE_RUNTIME
    n = len(row_df)
    hard_gate = to_bool_array(row_df.get("hard_gate_pass", True), n)
    eta = float(getattr(runtime, "eta_hard", 3.0))
    z_mass_cap = float(getattr(runtime, "mass_excess_cap", getattr(runtime, "z_mass_cap", 12.0)))
    raw_mass_cap = float(getattr(runtime, "raw_mass_cap", 6.0))
    raw_mass_kfail_trigger = float(getattr(runtime, "raw_mass_kfail_trigger", 1e6))

    threshold_map = _threshold_map(threshold_df)
    required_rules = [
        "output",
        "diff_residual",
        "delta_ridge_ens",
        "discourse_instability",
        "contradiction",
    ]
    rules: dict[str, Any] = {
        r: _build_rule_stat(
            row_df=row_df,
            threshold_map=threshold_map,
            rule=r,
            hard_gate=hard_gate,
            eta=eta,
            z_mass_cap=z_mass_cap,
            raw_mass_cap=raw_mass_cap,
            raw_mass_kfail_trigger=raw_mass_kfail_trigger,
        )
        for r in required_rules
    }

    baseline = compute_bundle_scores(
        row_df=row_df,
        threshold_summary_df=threshold_df,
        score_runtime=runtime,
    )
    baseline_new = {k: float(v) for k, v in baseline.payload.get("New_score", {}).items()}

    axis_weights = {"core": float(cfg.w_core), "rate": float(cfg.w_rate), "mass": float(cfg.w_mass)}
    diag_weights = {
        "rate": float(cfg.w_rate / max(cfg.w_rate + cfg.w_mass, EPS)),
        "mass": float(cfg.w_mass / max(cfg.w_rate + cfg.w_mass, EPS)),
    }
    s_core = {
        "output": float(getattr(runtime, "new_score_s_core_output", 0.50)),
        "diff_residual": float(getattr(runtime, "new_score_s_core_diff_residual", 0.50)),
        "delta_ridge_ens": float(getattr(runtime, "new_score_s_core_delta_ridge_ens", 0.50)),
        "discourse_instability": float(getattr(runtime, "new_score_s_core_discourse_instability", 0.50)),
        "contradiction": float(getattr(runtime, "new_score_s_core_contradiction", 0.50)),
    }
    s_mass = {
        "output": float(getattr(runtime, "new_score_s_mass_output", 0.50)),
        "diff_residual": float(getattr(runtime, "new_score_s_mass_diff_residual", 0.50)),
        "delta_ridge_ens": float(getattr(runtime, "new_score_s_mass_delta_ridge_ens", 0.50)),
        "discourse_instability": float(getattr(runtime, "new_score_s_mass_discourse_instability", 0.50)),
        "contradiction": float(getattr(runtime, "new_score_s_mass_contradiction", 0.50)),
    }

    prof_output = _continuous_rule_profile(
        stat=rules["output"],
        cfg=cfg,
        s_core=s_core["output"],
        s_mass=s_mass["output"],
        rate_lambda=float(cfg.rate_lambda_out),
        eta_hard=eta,
    )
    prof_diff = _continuous_rule_profile(
        stat=rules["diff_residual"],
        cfg=cfg,
        s_core=s_core["diff_residual"],
        s_mass=s_mass["diff_residual"],
        rate_lambda=float(cfg.rate_lambda_other),
        eta_hard=eta,
    )
    prof_ridge = _continuous_rule_profile(
        stat=rules["delta_ridge_ens"],
        cfg=cfg,
        s_core=s_core["delta_ridge_ens"],
        s_mass=s_mass["delta_ridge_ens"],
        rate_lambda=float(cfg.rate_lambda_other),
        eta_hard=eta,
    )
    prof_disc = _continuous_rule_profile(
        stat=rules["discourse_instability"],
        cfg=cfg,
        s_core=s_core["discourse_instability"],
        s_mass=s_mass["discourse_instability"],
        rate_lambda=float(cfg.rate_lambda_other),
        eta_hard=eta,
    )
    prof_contr = _continuous_rule_profile(
        stat=rules["contradiction"],
        cfg=cfg,
        s_core=s_core["contradiction"],
        s_mass=s_mass["contradiction"],
        rate_lambda=float(cfg.rate_lambda_other),
        eta_hard=eta,
    )

    sem_zero_cut = float(getattr(runtime, "sem_zero_dominance_cutoff", 0.90))
    sem_zero_eps = float(getattr(runtime, "sem_zero_dominance_eps", 1e-12))

    def zero_dom_ratio(rule_stat: Any) -> float:
        m = np.asarray(rule_stat.valid_mask, dtype=bool) & np.isfinite(np.asarray(rule_stat.signal, dtype=float))
        if np.sum(m) <= 0:
            return float("nan")
        sig = np.asarray(rule_stat.signal, dtype=float)[m]
        return float(np.mean(np.abs(sig) <= float(max(0.0, sem_zero_eps))))

    sem_zero_disc = zero_dom_ratio(rules["discourse_instability"])
    sem_zero_contr = zero_dom_ratio(rules["contradiction"])
    sem_uninformative = bool(
        np.isfinite(sem_zero_disc)
        and np.isfinite(sem_zero_contr)
        and float(sem_zero_disc) >= float(sem_zero_cut)
        and float(sem_zero_contr) >= float(sem_zero_cut)
    )

    # COV
    cov_rules = ["output", "diff_residual", "delta_ridge_ens"]
    cov_finite_row = np.ones(n, dtype=bool)
    cov_available_row = np.ones(n, dtype=bool)
    for r in cov_rules:
        rs = rules[r]
        cov_finite_row &= (~rs.available_mask) | rs.finite_mask
        cov_available_row &= rs.available_mask
    coverage_mask = hard_gate & cov_finite_row & cov_available_row
    coverage = _safe_rate(coverage_mask)
    score_cov = float(np.clip(5.0 * coverage, 0.0, 5.0)) if np.isfinite(coverage) else float(cfg.na_neutral_score)

    # OUT
    out_base = _bundle_risk_noisy_or(
        {"core": float(prof_output["r_core"]), "rate": float(prof_output["r_rate"]), "mass": float(prof_output["r_mass"])},
        axis_weights,
    )
    out_risk, out_penalty = _apply_soft_hard_tail_penalty(
        base_risk=out_base,
        hard_rate=float(prof_output["rate_hard_soft"]),
        mass_risk=float(prof_output["r_mass"]),
        alpha=float(cfg.hard_tail_penalty_alpha),
    )
    score_out = float(_quality_from_risk(out_risk, gamma=float(cfg.gamma))) if np.isfinite(out_risk) else float(cfg.na_neutral_score)

    # RID
    rid_base = _bundle_risk_noisy_or(
        {"core": float(prof_diff["r_core"]), "rate": float(prof_diff["r_rate"]), "mass": float(prof_diff["r_mass"])},
        axis_weights,
    )
    rid_risk, rid_penalty = _apply_soft_hard_tail_penalty(
        base_risk=rid_base,
        hard_rate=float(prof_diff["rate_hard_soft"]),
        mass_risk=float(prof_diff["r_mass"]),
        alpha=float(cfg.hard_tail_penalty_alpha),
    )
    score_rid = float(_quality_from_risk(rid_risk, gamma=float(cfg.gamma))) if np.isfinite(rid_risk) else float(cfg.na_neutral_score)

    # DIAG (soft quadrants from continuous anomaly probability)
    p_a = np.asarray(prof_diff["p_fail_soft"], dtype=float)
    p_b = np.asarray(prof_ridge["p_fail_soft"], dtype=float)
    diag_valid = np.isfinite(p_a) & np.isfinite(p_b)
    if np.sum(diag_valid) <= 0:
        q00_soft = q10_soft = q01_soft = q11_soft = float("nan")
    else:
        aa = p_a[diag_valid]
        bb = p_b[diag_valid]
        q10_soft = float(np.mean(aa * (1.0 - bb)))
        q01_soft = float(np.mean((1.0 - aa) * bb))
        q11_soft = float(np.mean(aa * bb))
        q00_soft = float(np.mean((1.0 - aa) * (1.0 - bb)))
    diag_rate_risk = _clip01(
        (q10_soft + q01_soft + q11_soft)
        if np.isfinite(q10_soft) and np.isfinite(q01_soft) and np.isfinite(q11_soft)
        else float("nan")
    )
    diag_mass_risk = _clip01(
        (q11_soft + 0.5 * min(q10_soft, q01_soft))
        if np.isfinite(q11_soft) and np.isfinite(q10_soft) and np.isfinite(q01_soft)
        else float("nan")
    )
    diag_base = _bundle_risk_noisy_or(
        {"rate": float(diag_rate_risk), "mass": float(diag_mass_risk)},
        diag_weights,
    )
    diag_hard_rate = _mean_or_nan(np.asarray([float(prof_diff["rate_hard_soft"]), float(prof_ridge["rate_hard_soft"])], dtype=float))
    diag_risk, diag_penalty = _apply_soft_hard_tail_penalty(
        base_risk=diag_base,
        hard_rate=diag_hard_rate,
        mass_risk=float(diag_mass_risk),
        alpha=float(cfg.hard_tail_penalty_alpha),
    )
    score_diag = float(_quality_from_risk(diag_risk, gamma=float(cfg.gamma))) if np.isfinite(diag_risk) else float(cfg.na_neutral_score)

    # SEM
    sem_r_core = _mean_or_nan(np.asarray([float(prof_disc["r_core"]), float(prof_contr["r_core"])], dtype=float))
    sem_r_rate = _mean_or_nan(np.asarray([float(prof_disc["r_rate"]), float(prof_contr["r_rate"])], dtype=float))
    sem_r_mass = _mean_or_nan(np.asarray([float(prof_disc["r_mass"]), float(prof_contr["r_mass"])], dtype=float))
    sem_hard_rate = _mean_or_nan(np.asarray([float(prof_disc["rate_hard_soft"]), float(prof_contr["rate_hard_soft"])], dtype=float))

    sem_na = bool(sem_uninformative or (not np.isfinite(sem_r_core)) or (not np.isfinite(sem_r_rate)) or (not np.isfinite(sem_r_mass)))
    if sem_na:
        sem_base = sem_penalty = sem_risk = float("nan")
        score_sem = float(cfg.sem_na_neutral_score)
    else:
        sem_base = _bundle_risk_noisy_or({"core": sem_r_core, "rate": sem_r_rate, "mass": sem_r_mass}, axis_weights)
        sem_risk, sem_penalty = _apply_soft_hard_tail_penalty(
            base_risk=sem_base,
            hard_rate=sem_hard_rate,
            mass_risk=float(sem_r_mass),
            alpha=float(cfg.hard_tail_penalty_alpha),
        )
        score_sem = float(_quality_from_risk(sem_risk, gamma=float(cfg.gamma))) if np.isfinite(sem_risk) else float(cfg.sem_na_neutral_score)

    # Threshold explainers and threshold-state bands
    ex_output = _threshold_explainer(row_df=row_df, rule="output", stat=rules["output"])
    ex_diff = _threshold_explainer(row_df=row_df, rule="diff_residual", stat=rules["diff_residual"])
    ex_ridge = _threshold_explainer(row_df=row_df, rule="delta_ridge_ens", stat=rules["delta_ridge_ens"])
    ex_disc = _threshold_explainer(row_df=row_df, rule="discourse_instability", stat=rules["discourse_instability"])
    ex_contr = _threshold_explainer(row_df=row_df, rule="contradiction", stat=rules["contradiction"])

    diag_state = _diag_state_quadrants(rules["diff_residual"], rules["delta_ridge_ens"])
    diag_state_cause = _diag_cause(diag_state["q10"], diag_state["q01"], diag_state["q11"])
    diag_soft_cause = _diag_cause(q10_soft, q01_soft, q11_soft)

    # CONF (same structure: min data/calc/th * op)
    cov_conf = _conf_components(
        n_valid=int(np.sum(coverage_mask)),
        finite_rate=_rate_on_base(cov_finite_row, hard_gate),
        available_rate=_rate_on_base(cov_available_row, hard_gate),
        rules=[rules[r] for r in cov_rules],
        min_support_rows=int(cfg.min_support_rows),
        conf_op_norm=1.0,
    )
    out_conf = _conf_components(
        n_valid=int(rules["output"].n_valid),
        finite_rate=_rate_on_base(rules["output"].finite_mask, hard_gate),
        available_rate=_rate_on_base(rules["output"].available_mask, hard_gate),
        rules=[rules["output"]],
        min_support_rows=int(cfg.min_support_rows),
        conf_op_norm=1.0,
    )
    rid_conf = _conf_components(
        n_valid=int(rules["diff_residual"].n_valid),
        finite_rate=_rate_on_base(rules["diff_residual"].finite_mask, hard_gate),
        available_rate=_rate_on_base(rules["diff_residual"].available_mask, hard_gate),
        rules=[rules["diff_residual"]],
        min_support_rows=int(cfg.min_support_rows),
        conf_op_norm=1.0,
    )
    diag_conf = _conf_components(
        n_valid=int(np.sum(np.asarray(rules["diff_residual"].valid_mask, dtype=bool) & np.asarray(rules["delta_ridge_ens"].valid_mask, dtype=bool))),
        finite_rate=_rate_on_base(rules["diff_residual"].finite_mask & rules["delta_ridge_ens"].finite_mask, hard_gate),
        available_rate=_rate_on_base(rules["diff_residual"].available_mask & rules["delta_ridge_ens"].available_mask, hard_gate),
        rules=[rules["diff_residual"], rules["delta_ridge_ens"]],
        min_support_rows=int(cfg.min_support_rows),
        conf_op_norm=1.0,
    )
    conf_data_norm = float(min(cov_conf["Conf_data_norm"], out_conf["Conf_data_norm"], rid_conf["Conf_data_norm"], diag_conf["Conf_data_norm"]))
    conf_calc_norm = float(min(cov_conf["Conf_calc_norm"], out_conf["Conf_calc_norm"], rid_conf["Conf_calc_norm"], diag_conf["Conf_calc_norm"]))
    conf_th_norm = float(min(cov_conf["Conf_th_norm"], out_conf["Conf_th_norm"], rid_conf["Conf_th_norm"], diag_conf["Conf_th_norm"]))
    conf_norm = float(np.clip(min(conf_data_norm, conf_calc_norm, conf_th_norm), 0.0, 1.0))
    score_conf = float(np.clip(5.0 * conf_norm, 0.0, 5.0))

    tf_bundle_scores = {
        "COV": float(score_cov),
        "OUT": float(score_out),
        "RID": float(score_rid),
        "DIAG": float(score_diag),
        "SEM": float(score_sem),
        "CONF": float(score_conf),
    }
    tf_bundle_band = {
        "COV": ("core" if score_cov >= 4.0 else "tail" if score_cov >= 2.0 else "exceptional"),
        "OUT": _band_from_state(ex_output["state_fail_rate"] or 0.0, ex_output["state_hard_rate"] or 0.0),
        "RID": _band_from_state(ex_diff["state_fail_rate"] or 0.0, ex_diff["state_hard_rate"] or 0.0),
        "DIAG": ("exceptional" if (diag_state.get("q11", 0.0) or 0.0) > 0.0 else "tail" if (diag_state.get("q10", 0.0) or 0.0) + (diag_state.get("q01", 0.0) or 0.0) > 0.0 else "core"),
        "SEM": ("NA" if sem_na else _band_from_state(_mean_or_nan(np.asarray([ex_disc["state_fail_rate"], ex_contr["state_fail_rate"]], dtype=float)), _mean_or_nan(np.asarray([ex_disc["state_hard_rate"], ex_contr["state_hard_rate"]], dtype=float)))),
        "CONF": ("core" if score_conf >= 4.0 else "tail" if score_conf >= 2.0 else "exceptional"),
    }

    rule_rows: list[dict[str, Any]] = []
    for rule, stat, prof, ex in [
        ("output", rules["output"], prof_output, ex_output),
        ("diff_residual", rules["diff_residual"], prof_diff, ex_diff),
        ("delta_ridge_ens", rules["delta_ridge_ens"], prof_ridge, ex_ridge),
        ("discourse_instability", rules["discourse_instability"], prof_disc, ex_disc),
        ("contradiction", rules["contradiction"], prof_contr, ex_contr),
    ]:
        rule_rows.append(
            {
                "case": case.name,
                "rule": rule,
                "n_valid": int(ex["n_valid"]),
                "r_core": (float(prof["r_core"]) if np.isfinite(prof["r_core"]) else np.nan),
                "r_rate": (float(prof["r_rate"]) if np.isfinite(prof["r_rate"]) else np.nan),
                "r_mass": (float(prof["r_mass"]) if np.isfinite(prof["r_mass"]) else np.nan),
                "q_core": (float(prof["q_core"]) if np.isfinite(prof["q_core"]) else np.nan),
                "q_rate": (float(prof["q_rate"]) if np.isfinite(prof["q_rate"]) else np.nan),
                "q_mass": (float(prof["q_mass"]) if np.isfinite(prof["q_mass"]) else np.nan),
                "rate_fail_soft": (float(prof["rate_fail_soft"]) if np.isfinite(prof["rate_fail_soft"]) else np.nan),
                "rate_hard_soft": (float(prof["rate_hard_soft"]) if np.isfinite(prof["rate_hard_soft"]) else np.nan),
                "tail_mass_cont": (float(prof["tail_mass_cont"]) if np.isfinite(prof["tail_mass_cont"]) else np.nan),
                "core_mad_cont": (float(prof["core_mad_cont"]) if np.isfinite(prof["core_mad_cont"]) else np.nan),
                "k_fail": (float(ex["k_fail"]) if ex["k_fail"] is not None else np.nan),
                "k_hard": (float(ex["k_hard"]) if ex["k_hard"] is not None else np.nan),
                "rate_fail": (float(ex["rate_fail"]) if ex["rate_fail"] is not None else np.nan),
                "rate_hard": (float(ex["rate_hard"]) if ex["rate_hard"] is not None else np.nan),
                "excess_fail": (float(ex["excess_fail"]) if ex["excess_fail"] is not None else np.nan),
                "excess_hard": (float(ex["excess_hard"]) if ex["excess_hard"] is not None else np.nan),
                "state_pass_rate": (float(ex["state_pass_rate"]) if ex["state_pass_rate"] is not None else np.nan),
                "state_warn_rate": (float(ex["state_warn_rate"]) if ex["state_warn_rate"] is not None else np.nan),
                "state_fail_rate": (float(ex["state_fail_rate"]) if ex["state_fail_rate"] is not None else np.nan),
                "state_hard_rate": (float(ex["state_hard_rate"]) if ex["state_hard_rate"] is not None else np.nan),
            }
        )

    diag_row = {
        "case": case.name,
        "soft_q00": q00_soft,
        "soft_q10": q10_soft,
        "soft_q01": q01_soft,
        "soft_q11": q11_soft,
        "soft_cause": diag_soft_cause,
        "state_q00": diag_state["q00"],
        "state_q10": diag_state["q10"],
        "state_q01": diag_state["q01"],
        "state_q11": diag_state["q11"],
        "state_cause": diag_state_cause,
        "diag_rate_risk_soft": diag_rate_risk,
        "diag_mass_risk_soft": diag_mass_risk,
        "diag_penalty_soft": diag_penalty,
        "diag_risk_soft": diag_risk,
    }

    score_rows: list[dict[str, Any]] = []
    for bundle in ["COV", "OUT", "RID", "DIAG", "SEM", "CONF"]:
        current = float(baseline_new.get(bundle, np.nan))
        tf = float(tf_bundle_scores[bundle])
        score_rows.append(
            {
                "case": case.name,
                "bundle": bundle,
                "current_new_score": current,
                "threshold_free_score": tf,
                "delta_tf_minus_current": (float(tf - current) if np.isfinite(current) else np.nan),
                "band_threshold_state": str(tf_bundle_band.get(bundle, "")),
            }
        )

    dist_rows: list[dict[str, Any]] = []
    for rule in ["output", "diff_residual", "delta_ridge_ens"]:
        prof = {"output": prof_output, "diff_residual": prof_diff, "delta_ridge_ens": prof_ridge}[rule]
        z = np.asarray(prof["z_cont"], dtype=float)
        z = z[np.isfinite(z)]
        if z.size == 0:
            dist_rows.append(
                {
                    "case": case.name,
                    "rule": rule,
                    "n": 0,
                    "z_p50": np.nan,
                    "z_p90": np.nan,
                    "z_p95": np.nan,
                    "z_p99": np.nan,
                    "z_mean": np.nan,
                }
            )
            continue
        dist_rows.append(
            {
                "case": case.name,
                "rule": rule,
                "n": int(z.size),
                "z_p50": float(np.quantile(z, 0.50)),
                "z_p90": float(np.quantile(z, 0.90)),
                "z_p95": float(np.quantile(z, 0.95)),
                "z_p99": float(np.quantile(z, 0.99)),
                "z_mean": float(np.mean(z)),
            }
        )

    return {
        "case": case.name,
        "rows": int(n),
        "baseline_new_score": baseline_new,
        "threshold_free_score": tf_bundle_scores,
        "threshold_free_band": tf_bundle_band,
        "score_rows": score_rows,
        "rule_rows": rule_rows,
        "diag_row": diag_row,
        "dist_rows": dist_rows,
        "profiles": {
            "output": prof_output,
            "diff_residual": prof_diff,
            "delta_ridge_ens": prof_ridge,
        },
        "penalties": {
            "OUT": float(out_penalty) if np.isfinite(out_penalty) else None,
            "RID": float(rid_penalty) if np.isfinite(rid_penalty) else None,
            "DIAG": float(diag_penalty) if np.isfinite(diag_penalty) else None,
            "SEM": float(sem_penalty) if np.isfinite(sem_penalty) else None,
        },
        "sem": {
            "na": bool(sem_na),
            "zero_ratio_disc": float(sem_zero_disc) if np.isfinite(sem_zero_disc) else None,
            "zero_ratio_contr": float(sem_zero_contr) if np.isfinite(sem_zero_contr) else None,
            "zero_cutoff": float(sem_zero_cut),
        },
    }


def render_dashboard_html(out_html: Path, case_results: list[dict[str, Any]]) -> None:
    if go is None or pio is None or make_subplots is None:
        return
    bundle_order = ["COV", "OUT", "RID", "DIAG", "SEM", "CONF"]
    theta = bundle_order + [bundle_order[0]]

    fig_radar = make_subplots(
        rows=1,
        cols=max(1, len(case_results)),
        specs=[[{"type": "polar"} for _ in range(max(1, len(case_results)))]],
        subplot_titles=[str(x["case"]) for x in case_results],
    )
    for idx, res in enumerate(case_results, start=1):
        curr = [float(res["baseline_new_score"].get(b, np.nan)) for b in bundle_order]
        tf = [float(res["threshold_free_score"].get(b, np.nan)) for b in bundle_order]
        fig_radar.add_trace(
            go.Scatterpolar(
                r=curr + [curr[0]],
                theta=theta,
                mode="lines+markers",
                name=f"{res['case']} current",
                line=dict(width=2),
                opacity=0.75,
            ),
            row=1,
            col=idx,
        )
        fig_radar.add_trace(
            go.Scatterpolar(
                r=tf + [tf[0]],
                theta=theta,
                mode="lines+markers",
                name=f"{res['case']} threshold_free",
                line=dict(width=2),
                opacity=0.90,
            ),
            row=1,
            col=idx,
        )
        fig_radar.update_polars(radialaxis=dict(range=[0, 5]), row=1, col=idx)
    fig_radar.update_layout(title="Current New_score vs Threshold-free Score (Radar)", showlegend=True, template="plotly_white")

    bar_rows: list[dict[str, Any]] = []
    for res in case_results:
        for b in bundle_order:
            bar_rows.append({"case": res["case"], "bundle": b, "method": "current", "score": float(res["baseline_new_score"].get(b, np.nan))})
            bar_rows.append({"case": res["case"], "bundle": b, "method": "threshold_free", "score": float(res["threshold_free_score"].get(b, np.nan))})
    bar_df = pd.DataFrame(bar_rows)
    fig_bar = go.Figure()
    for (case, method), g in bar_df.groupby(["case", "method"], sort=False):
        fig_bar.add_trace(
            go.Bar(
                x=[f"{case}:{b}" for b in g["bundle"].tolist()],
                y=g["score"].tolist(),
                name=f"{case}:{method}",
            )
        )
    fig_bar.update_layout(
        title="Bundle Score Comparison",
        barmode="group",
        yaxis=dict(range=[0, 5], title="score"),
        xaxis_title="case:bundle",
        template="plotly_white",
    )

    fig_dist = make_subplots(rows=1, cols=2, subplot_titles=["diff_residual z_cont", "delta_ridge_ens z_cont"])
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    for i, res in enumerate(case_results):
        c = colors[i % len(colors)]
        z_diff = np.asarray(res["profiles"]["diff_residual"]["z_cont"], dtype=float)
        z_diff = z_diff[np.isfinite(z_diff)]
        z_ridge = np.asarray(res["profiles"]["delta_ridge_ens"]["z_cont"], dtype=float)
        z_ridge = z_ridge[np.isfinite(z_ridge)]
        fig_dist.add_trace(
            go.Histogram(x=z_diff, name=f"{res['case']} diff", opacity=0.55, marker_color=c, nbinsx=40),
            row=1,
            col=1,
        )
        fig_dist.add_trace(
            go.Histogram(x=z_ridge, name=f"{res['case']} ridge", opacity=0.55, marker_color=c, nbinsx=40, showlegend=False),
            row=1,
            col=2,
        )
    fig_dist.update_layout(
        barmode="overlay",
        title="RID/DIAG Distribution View (continuous z)",
        template="plotly_white",
    )

    html = "\n".join(
        [
            "<html><head><meta charset='utf-8'><title>threshold_free_compare</title></head><body>",
            "<h2>Threshold-free vs Current New_score</h2>",
            pio.to_html(fig_radar, include_plotlyjs="cdn", full_html=False),
            pio.to_html(fig_bar, include_plotlyjs=False, full_html=False),
            pio.to_html(fig_dist, include_plotlyjs=False, full_html=False),
            "</body></html>",
        ]
    )
    out_html.write_text(html, encoding="utf-8")


def write_markdown_report(
    *,
    out_md: Path,
    case_results: list[dict[str, Any]],
    score_df: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Threshold-free Score Experiment")
    lines.append("")
    lines.append("- score input: continuous risk only (threshold not used)")
    lines.append("- threshold role: state labels + explain vars (`rate_fail`, `rate_hard`, `excess_fail`, `excess_hard`)")
    lines.append("")
    lines.append("## Bundle Comparison (threshold_free - current)")
    lines.append("")
    piv = score_df.pivot_table(index=["case", "bundle"], values=["current_new_score", "threshold_free_score", "delta_tf_minus_current"], aggfunc="first").reset_index()
    lines.append(piv.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## DIAG Cause (state/soft)")
    lines.append("")
    diag_rows = []
    for res in case_results:
        d = dict(res["diag_row"])
        diag_rows.append(
            {
                "case": res["case"],
                "soft_cause": d["soft_cause"],
                "state_cause": d["state_cause"],
                "soft_q10": d["soft_q10"],
                "soft_q01": d["soft_q01"],
                "soft_q11": d["soft_q11"],
                "state_q10": d["state_q10"],
                "state_q01": d["state_q01"],
                "state_q11": d["state_q11"],
            }
        )
    lines.append(pd.DataFrame(diag_rows).to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = _score_runtime_from_defaults(args)
    cases = parse_cases(args)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(str(args.output_root)).resolve() / f"{args.tag}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    case_results: list[dict[str, Any]] = [evaluate_case(c, cfg) for c in cases]

    score_df = pd.DataFrame([r for res in case_results for r in res["score_rows"]])
    rule_df = pd.DataFrame([r for res in case_results for r in res["rule_rows"]])
    diag_df = pd.DataFrame([res["diag_row"] for res in case_results])
    dist_df = pd.DataFrame([r for res in case_results for r in res["dist_rows"]])

    score_df.to_csv(out_dir / "threshold_free_compare_scores.csv", index=False)
    rule_df.to_csv(out_dir / "threshold_free_rule_explainer.csv", index=False)
    diag_df.to_csv(out_dir / "threshold_free_diag_explainer.csv", index=False)
    dist_df.to_csv(out_dir / "threshold_free_distribution_summary.csv", index=False)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases": [dict(name=c.name, row_results_csv=str(c.row_results_csv), thresholds_csv=str(c.thresholds_csv)) for c in cases],
        "config": {
            "z_fail_ref": float(cfg.z_fail_ref),
            "z_hard_ref": float(cfg.z_hard_ref),
            "tau_rate": float(cfg.tau_rate),
            "tau_mass": float(cfg.tau_mass),
            "core_quantile": float(cfg.core_quantile),
            "gamma": float(cfg.gamma),
            "weights": {"core": float(cfg.w_core), "rate": float(cfg.w_rate), "mass": float(cfg.w_mass)},
            "hard_tail_penalty_alpha": float(cfg.hard_tail_penalty_alpha),
        },
        "results": case_results,
    }
    (out_dir / "threshold_free_compare_payload.json").write_text(
        json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_markdown_report(out_md=out_dir / "threshold_free_compare_report.md", case_results=case_results, score_df=score_df)
    if bool(args.emit_plot):
        render_dashboard_html(out_html=out_dir / "threshold_free_compare_dashboard.html", case_results=case_results)

    print(json.dumps({"output_dir": str(out_dir), "cases": [c.name for c in cases]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
