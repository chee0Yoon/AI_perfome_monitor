#!/usr/bin/env python3
"""Test-only threshold target experiments for warn/fail/hard quality.

Goals (user-tunable):
- hard: bad precision should be 1.0
- fail: bad precision should be >= 0.88
- warn: good ratio should be < 0.30

This script runs multiple threshold-building methods (test-only), then evaluates
tri-state bucket quality on the same row_results dataset.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
)

RULE_ORDER = list(RUNTIME_RULE_ORDER)
RULE_SIGNAL_COL_NOMASK = dict(RUNTIME_RULE_SIGNAL_COL_NOMASK)
# Backward compatibility alias.
RULE_SCORE_COL_NOMASK = RULE_SIGNAL_COL_NOMASK

RULE_AVAILABLE_COL_NOMASK = dict(RUNTIME_RULE_AVAILABLE_COL_NOMASK)


@dataclass
class MethodSpec:
    name: str
    main_threshold_policy: str
    soft_k_floor: float
    gap_strong_z: float
    gap_weak_z: float
    tail_start_max_k: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run test-only threshold target experiments")
    p.add_argument("--row-results-csv", required=True)
    p.add_argument(
        "--output-root",
        default="/Users/cyyoon/dev/llmops/final_metric/results/threshold_target_experiments",
    )
    p.add_argument("--mode", default="nomask", choices=["nomask"])
    p.add_argument("--tail-direction", default="two_sided", choices=["upper", "lower", "two_sided"])
    p.add_argument("--min-support-rows", type=int, default=16)
    p.add_argument("--target-hard-bad-precision", type=float, default=1.0)
    p.add_argument("--target-fail-bad-precision", type=float, default=0.88)
    p.add_argument("--target-warn-good-ratio", type=float, default=0.30)
    p.add_argument("--emit-plot", action="store_true", default=True)
    p.add_argument("--skip-builder", action="store_true", default=False)
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


def build_known_good_mask(row_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    known = np.zeros(len(row_df), dtype=bool)
    good = np.zeros(len(row_df), dtype=bool)

    if "label_is_correct" in row_df.columns:
        s = row_df["label_is_correct"]
        if s.dtype == bool:
            known = s.notna().to_numpy(dtype=bool)
            good = s.fillna(False).to_numpy(dtype=bool)
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

    if (not np.any(known)) and ("label_raw" in row_df.columns):
        lr = row_df["label_raw"].fillna("").astype(str).str.strip().str.lower()
        g = lr.isin({"correct", "true", "1", "pass", "good"}).to_numpy(dtype=bool)
        b = lr.isin({"incorrect", "false", "0", "fail", "bad"}).to_numpy(dtype=bool)
        known = g | b
        good = g

    return known, good


def trigger_mask(
    score: np.ndarray,
    *,
    tail_direction: str,
    threshold: float,
    threshold_low: float,
    threshold_high: float,
    median: float,
) -> np.ndarray:
    s = np.asarray(score, dtype=float)
    out = np.zeros(len(s), dtype=bool)
    finite = np.isfinite(s)
    if not np.any(finite):
        return out

    if tail_direction == "upper":
        t = threshold_low if np.isfinite(threshold_low) else threshold
        if np.isfinite(t):
            out[finite] = s[finite] >= float(t)
        return out

    if tail_direction == "lower":
        t = threshold_low if np.isfinite(threshold_low) else threshold
        if np.isfinite(t):
            out[finite] = s[finite] <= float(t)
        return out

    # two_sided
    if np.isfinite(threshold_low) and np.isfinite(threshold_high):
        out[finite] = (s[finite] <= float(threshold_low)) | (s[finite] >= float(threshold_high))
        return out
    if np.isfinite(threshold_high):
        out[finite] = s[finite] >= float(threshold_high)
        return out
    if np.isfinite(threshold_low):
        if np.isfinite(median):
            if threshold_low < median:
                out[finite] = s[finite] <= float(threshold_low)
            else:
                out[finite] = s[finite] >= float(threshold_low)
        else:
            out[finite] = s[finite] >= float(threshold_low)
        return out
    if np.isfinite(threshold):
        out[finite] = s[finite] >= float(threshold)
    return out


def get_method_specs() -> list[MethodSpec]:
    # 6+ methods as requested.
    return [
        MethodSpec("m01_hybrid_base", "hybrid", 0.0, 4.0, 2.5, 8.0),
        MethodSpec("m02_hybrid_soft15", "hybrid", 1.5, 4.0, 2.5, 8.0),
        MethodSpec("m03_hybrid_soft20", "hybrid", 2.0, 4.0, 2.5, 8.0),
        MethodSpec("m04_hybrid_gap50_soft10", "hybrid", 1.0, 5.0, 3.0, 8.0),
        MethodSpec("m05_hybrid_gap60_soft15", "hybrid", 1.5, 6.0, 3.5, 8.0),
        MethodSpec("m06_hybrid_maxk12", "hybrid", 1.0, 4.0, 2.5, 12.0),
        MethodSpec("m07_derivative_baseline", "derivative", 0.0, 4.0, 2.5, 8.0),
    ]


def run_builder(
    *,
    python_exec: str,
    build_script: Path,
    row_results_csv: Path,
    output_root: Path,
    mode: str,
    tail_direction: str,
    min_support_rows: int,
    spec: MethodSpec,
) -> None:
    cmd = [
        python_exec,
        str(build_script),
        "--row-results-csv",
        str(row_results_csv),
        "--output-dir",
        str(output_root),
        "--report-dir-name",
        "report",
        "--mode",
        mode,
        "--tail-direction",
        tail_direction,
        "--min-support-rows",
        str(min_support_rows),
        "--main-threshold-policy",
        spec.main_threshold_policy,
        "--exp-policy",
        "delta_gap_finitefb_v1",
        "--fallback-trigger",
        "finite_only",
        "--tag",
        f"targetexp_{spec.name}",
        "--gap-strong-z",
        str(spec.gap_strong_z),
        "--gap-weak-z",
        str(spec.gap_weak_z),
        "--soft-k-floor",
        str(spec.soft_k_floor),
        "--tail-start-max-k",
        str(spec.tail_start_max_k),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def find_latest_artifacts(
    report_dir: Path,
    *,
    tag: str,
    row_stem: str,
    mode: str,
) -> tuple[Path, Path]:
    cfg_glob = f"{tag}_{row_stem}_{mode}_*_run_config.json"
    cfgs = sorted(report_dir.glob(cfg_glob), key=lambda p: p.stat().st_mtime)
    if not cfgs:
        raise FileNotFoundError(f"run config not found: {cfg_glob}")
    cfg_path = cfgs[-1]
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    thr_csv = Path(cfg["output_rule_thresholds_csv"])
    return thr_csv, cfg_path


def evaluate_method(
    *,
    row_df: pd.DataFrame,
    thresholds_df: pd.DataFrame,
    mode: str,
    tail_direction: str,
    target_hard_bad_precision: float,
    target_fail_bad_precision: float,
    target_warn_good_ratio: float,
    method_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    known, good = build_known_good_mask(row_df=row_df)
    bad = ~good

    if "hard_gate_pass" in row_df.columns:
        hard_gate = safe_bool_series(row_df["hard_gate_pass"])
    else:
        hard_gate = np.ones(len(row_df), dtype=bool)

    rule_rows: list[dict[str, Any]] = []
    sum_hard_n = 0
    sum_hard_bad = 0
    sum_fail_n = 0
    sum_fail_bad = 0
    sum_warn_n = 0
    sum_warn_good = 0

    for _, tr in thresholds_df.iterrows():
        rule = str(tr["rule"])
        signal_col = RULE_SIGNAL_COL_NOMASK.get(rule)
        if not signal_col or signal_col not in row_df.columns:
            continue

        avail = np.ones(len(row_df), dtype=bool)
        av_col = RULE_AVAILABLE_COL_NOMASK.get(rule)
        if av_col and av_col in row_df.columns:
            avail = safe_bool_series(row_df[av_col])

        score = pd.to_numeric(row_df[signal_col], errors="coerce").to_numpy(dtype=float)
        base = hard_gate & avail & np.isfinite(score) & known
        if not np.any(base):
            continue

        median = float(pd.to_numeric(pd.Series([tr.get("median", np.nan)]), errors="coerce").iloc[0])
        warn_t = float(pd.to_numeric(pd.Series([tr.get("warn_threshold", np.nan)]), errors="coerce").iloc[0])
        warn_lo = float(pd.to_numeric(pd.Series([tr.get("warn_threshold_low", np.nan)]), errors="coerce").iloc[0])
        warn_hi = float(pd.to_numeric(pd.Series([tr.get("warn_threshold_high", np.nan)]), errors="coerce").iloc[0])
        fail_t = float(pd.to_numeric(pd.Series([tr.get("fail_threshold", np.nan)]), errors="coerce").iloc[0])
        fail_lo = float(pd.to_numeric(pd.Series([tr.get("fail_threshold_low", np.nan)]), errors="coerce").iloc[0])
        fail_hi = float(pd.to_numeric(pd.Series([tr.get("fail_threshold_high", np.nan)]), errors="coerce").iloc[0])
        hard_t = float(pd.to_numeric(pd.Series([tr.get("hard_fail_threshold", np.nan)]), errors="coerce").iloc[0])
        hard_lo = float(pd.to_numeric(pd.Series([tr.get("hard_fail_threshold_low", np.nan)]), errors="coerce").iloc[0])
        hard_hi = float(pd.to_numeric(pd.Series([tr.get("hard_fail_threshold_high", np.nan)]), errors="coerce").iloc[0])

        warn_mask = trigger_mask(
            score,
            tail_direction=tail_direction,
            threshold=warn_t,
            threshold_low=warn_lo,
            threshold_high=warn_hi,
            median=median,
        ) & base
        fail_mask = trigger_mask(
            score,
            tail_direction=tail_direction,
            threshold=fail_t,
            threshold_low=fail_lo,
            threshold_high=fail_hi,
            median=median,
        ) & base
        hard_mask = trigger_mask(
            score,
            tail_direction=tail_direction,
            threshold=hard_t,
            threshold_low=hard_lo,
            threshold_high=hard_hi,
            median=median,
        ) & base

        hard_only = hard_mask
        fail_only = fail_mask & (~hard_only)
        warn_only = warn_mask & (~fail_mask) & (~hard_only)

        hard_n = int(np.sum(hard_only))
        hard_bad = int(np.sum(bad & hard_only))
        fail_n = int(np.sum(fail_only))
        fail_bad = int(np.sum(bad & fail_only))
        warn_n = int(np.sum(warn_only))
        warn_good = int(np.sum(good & warn_only))

        hard_bad_precision = float(hard_bad / hard_n) if hard_n > 0 else float("nan")
        fail_bad_precision = float(fail_bad / fail_n) if fail_n > 0 else float("nan")
        warn_good_ratio = float(warn_good / warn_n) if warn_n > 0 else float("nan")

        hard_pass = bool(hard_n > 0 and np.isfinite(hard_bad_precision) and hard_bad_precision >= float(target_hard_bad_precision))
        fail_pass = bool(fail_n > 0 and np.isfinite(fail_bad_precision) and fail_bad_precision >= float(target_fail_bad_precision))
        warn_pass = bool(warn_n > 0 and np.isfinite(warn_good_ratio) and warn_good_ratio < float(target_warn_good_ratio))

        sum_hard_n += hard_n
        sum_hard_bad += hard_bad
        sum_fail_n += fail_n
        sum_fail_bad += fail_bad
        sum_warn_n += warn_n
        sum_warn_good += warn_good

        rule_rows.append(
            {
                "method": method_name,
                "rule": rule,
                "support_rows": int(np.sum(base)),
                "selected_method": str(tr.get("selected_method", "")),
                "hard_n": hard_n,
                "hard_bad_n": hard_bad,
                "hard_bad_precision": hard_bad_precision,
                "hard_pass_target": hard_pass,
                "fail_n": fail_n,
                "fail_bad_n": fail_bad,
                "fail_bad_precision": fail_bad_precision,
                "fail_pass_target": fail_pass,
                "warn_n": warn_n,
                "warn_good_n": warn_good,
                "warn_good_ratio": warn_good_ratio,
                "warn_pass_target": warn_pass,
                "warn_threshold": warn_t,
                "fail_threshold": fail_t,
                "hard_fail_threshold": hard_t,
            }
        )

    rule_df = pd.DataFrame(rule_rows)
    hard_bad_precision_agg = float(sum_hard_bad / sum_hard_n) if sum_hard_n > 0 else float("nan")
    fail_bad_precision_agg = float(sum_fail_bad / sum_fail_n) if sum_fail_n > 0 else float("nan")
    warn_good_ratio_agg = float(sum_warn_good / sum_warn_n) if sum_warn_n > 0 else float("nan")

    summary = {
        "method": method_name,
        "rules_evaluated": int(len(rule_df)),
        "hard_n_total": int(sum_hard_n),
        "hard_bad_precision_agg": hard_bad_precision_agg,
        "hard_target_met_agg": bool(
            sum_hard_n > 0
            and np.isfinite(hard_bad_precision_agg)
            and hard_bad_precision_agg >= float(target_hard_bad_precision)
        ),
        "hard_target_met_rules": int(rule_df["hard_pass_target"].sum()) if len(rule_df) else 0,
        "fail_n_total": int(sum_fail_n),
        "fail_bad_precision_agg": fail_bad_precision_agg,
        "fail_target_met_agg": bool(
            sum_fail_n > 0
            and np.isfinite(fail_bad_precision_agg)
            and fail_bad_precision_agg >= float(target_fail_bad_precision)
        ),
        "fail_target_met_rules": int(rule_df["fail_pass_target"].sum()) if len(rule_df) else 0,
        "warn_n_total": int(sum_warn_n),
        "warn_good_ratio_agg": warn_good_ratio_agg,
        "warn_target_met_agg": bool(
            sum_warn_n > 0
            and np.isfinite(warn_good_ratio_agg)
            and warn_good_ratio_agg < float(target_warn_good_ratio)
        ),
        "warn_target_met_rules": int(rule_df["warn_pass_target"].sum()) if len(rule_df) else 0,
    }
    summary["all_targets_met_agg"] = bool(
        summary["hard_target_met_agg"] and summary["fail_target_met_agg"] and summary["warn_target_met_agg"]
    )
    return rule_df, summary


def write_plot(
    *,
    out_html: Path,
    summary_df: pd.DataFrame,
    target_hard_bad_precision: float,
    target_fail_bad_precision: float,
    target_warn_good_ratio: float,
) -> None:
    if go is None or pio is None:
        return

    x = summary_df["method"].astype(str).tolist()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="hard_bad_precision_agg",
            x=x,
            y=summary_df["hard_bad_precision_agg"],
            marker_color="#dc2626",
        )
    )
    fig.add_trace(
        go.Bar(
            name="fail_bad_precision_agg",
            x=x,
            y=summary_df["fail_bad_precision_agg"],
            marker_color="#2563eb",
        )
    )
    fig.add_trace(
        go.Bar(
            name="warn_good_ratio_agg (lower is better)",
            x=x,
            y=summary_df["warn_good_ratio_agg"],
            marker_color="#f59e0b",
        )
    )
    fig.add_hline(y=float(target_hard_bad_precision), line_dash="dot", line_color="#991b1b")
    fig.add_hline(y=float(target_fail_bad_precision), line_dash="dash", line_color="#1d4ed8")
    fig.add_hline(y=float(target_warn_good_ratio), line_dash="dashdot", line_color="#b45309")
    fig.update_layout(
        barmode="group",
        title="Threshold Target Experiment Summary",
        xaxis_title="method",
        yaxis_title="ratio",
        template="plotly_white",
        legend_title_text="metric",
    )
    out_html.write_text(pio.to_html(fig, include_plotlyjs="cdn", full_html=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    row_csv = Path(args.row_results_csv).resolve()
    output_root = Path(args.output_root).resolve()
    report_dir = output_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    row_df = pd.read_csv(row_csv)
    row_stem = row_csv.stem
    build_script = Path(__file__).resolve().parent / "build_thresholds_from_row_results.py"
    methods = get_method_specs()

    all_rule_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []

    for spec in methods:
        tag = f"targetexp_{spec.name}"
        if not bool(args.skip_builder):
            run_builder(
                python_exec=sys.executable,
                build_script=build_script,
                row_results_csv=row_csv,
                output_root=output_root,
                mode=str(args.mode),
                tail_direction=str(args.tail_direction),
                min_support_rows=int(args.min_support_rows),
                spec=spec,
            )

        thr_csv, cfg_json = find_latest_artifacts(
            report_dir=report_dir,
            tag=tag,
            row_stem=row_stem,
            mode=str(args.mode),
        )
        cfg = json.loads(cfg_json.read_text(encoding="utf-8"))
        tail_direction = str(cfg.get("tail_direction", args.tail_direction))
        thr_df = pd.read_csv(thr_csv)

        rule_df, summary = evaluate_method(
            row_df=row_df,
            thresholds_df=thr_df,
            mode=str(args.mode),
            tail_direction=tail_direction,
            target_hard_bad_precision=float(args.target_hard_bad_precision),
            target_fail_bad_precision=float(args.target_fail_bad_precision),
            target_warn_good_ratio=float(args.target_warn_good_ratio),
            method_name=spec.name,
        )
        summary["threshold_csv"] = str(thr_csv)
        summary["run_config_json"] = str(cfg_json)
        summaries.append(summary)
        all_rule_frames.append(rule_df)

    summary_df = pd.DataFrame(summaries).sort_values(
        by=["all_targets_met_agg", "fail_bad_precision_agg", "hard_bad_precision_agg"],
        ascending=[False, False, False],
    )
    detail_df = pd.concat(all_rule_frames, axis=0, ignore_index=True) if all_rule_frames else pd.DataFrame()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    summary_csv = report_dir / f"targetexp_summary_{ts}.csv"
    detail_csv = report_dir / f"targetexp_rule_detail_{ts}.csv"
    summary_df.to_csv(summary_csv, index=False)
    detail_df.to_csv(detail_csv, index=False)

    cfg = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_results_csv": str(row_csv),
        "target_hard_bad_precision": float(args.target_hard_bad_precision),
        "target_fail_bad_precision": float(args.target_fail_bad_precision),
        "target_warn_good_ratio": float(args.target_warn_good_ratio),
        "methods": [m.__dict__ for m in methods],
        "output_summary_csv": str(summary_csv),
        "output_detail_csv": str(detail_csv),
    }
    cfg_json = report_dir / f"targetexp_run_config_{ts}.json"
    cfg_json.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_html = None
    if bool(args.emit_plot):
        plot_html = report_dir / f"targetexp_summary_{ts}.html"
        write_plot(
            out_html=plot_html,
            summary_df=summary_df,
            target_hard_bad_precision=float(args.target_hard_bad_precision),
            target_fail_bad_precision=float(args.target_fail_bad_precision),
            target_warn_good_ratio=float(args.target_warn_good_ratio),
        )

    print(f"[DONE] summary_csv: {summary_csv}")
    print(f"[DONE] detail_csv: {detail_csv}")
    print(f"[DONE] run_config: {cfg_json}")
    if plot_html is not None:
        print(f"[DONE] summary_html: {plot_html}")
    print("[SUMMARY]")
    show_cols = [
        "method",
        "rules_evaluated",
        "hard_n_total",
        "hard_bad_precision_agg",
        "hard_target_met_agg",
        "fail_n_total",
        "fail_bad_precision_agg",
        "fail_target_met_agg",
        "warn_n_total",
        "warn_good_ratio_agg",
        "warn_target_met_agg",
        "all_targets_met_agg",
    ]
    print(summary_df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
