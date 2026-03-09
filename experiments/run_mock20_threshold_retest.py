#!/usr/bin/env python3
"""Batch re-test current threshold policy on mock-suite row_results (test-only)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from threshold_target_experiments import (
    RULE_AVAILABLE_COL_NOMASK,
    RULE_SIGNAL_COL_NOMASK,
    build_known_good_mask,
    safe_bool_series,
    trigger_mask,
)

try:
    import plotly.graph_objects as go
    import plotly.io as pio
except Exception:  # pragma: no cover
    go = None
    pio = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re-test current threshold policy on mock 20 suite")
    p.add_argument(
        "--runs-root",
        default="/Users/cyyoon/dev/llmops/final_metric/results/mock_suite_20260225_batch1/runs",
    )
    p.add_argument(
        "--output-root",
        default="/Users/cyyoon/dev/llmops/final_metric/results/mock_suite_20260225_batch1/threshold_retest",
    )
    p.add_argument("--mode", default="nomask", choices=["nomask"])
    p.add_argument("--tail-direction", default="two_sided", choices=["upper", "lower", "two_sided"])
    p.add_argument("--target-hard-bad-precision", type=float, default=1.0)
    p.add_argument("--target-fail-bad-precision", type=float, default=0.88)
    p.add_argument("--target-warn-good-ratio", type=float, default=0.30)
    p.add_argument("--emit-plot", action="store_true", default=True)
    return p.parse_args()


def run_builder(
    *,
    python_exec: str,
    build_script: Path,
    row_results_csv: Path,
    output_root: Path,
    mode: str,
    tail_direction: str,
) -> tuple[Path, Path]:
    dataset = row_results_csv.parent.name
    tag = f"mock20_retest_{dataset}"
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
        "--main-threshold-policy",
        "hybrid",
        "--exp-policy",
        "delta_gap_finitefb_v1",
        "--fallback-trigger",
        "finite_only",
        "--tag",
        tag,
        "--emit-plot",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    report_dir = output_root / "report"
    cfgs = sorted(
        report_dir.glob(f"{tag}_{row_results_csv.stem}_{mode}_*_run_config.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not cfgs:
        raise FileNotFoundError(f"run config not found for {row_results_csv}")
    cfg = cfgs[-1]
    obj = json.loads(cfg.read_text(encoding="utf-8"))
    return Path(obj["output_rule_thresholds_csv"]), cfg


def evaluate_one(
    *,
    row_df: pd.DataFrame,
    thresholds_df: pd.DataFrame,
    tail_direction: str,
    target_hard_bad_precision: float,
    target_fail_bad_precision: float,
    target_warn_good_ratio: float,
    dataset: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    known, good = build_known_good_mask(row_df=row_df)
    bad = ~good
    hard_gate = safe_bool_series(row_df["hard_gate_pass"]) if "hard_gate_pass" in row_df.columns else np.ones(len(row_df), dtype=bool)

    detail_rows: list[dict[str, Any]] = []
    sum_hn = 0
    sum_hb = 0
    sum_fn = 0
    sum_fb = 0
    sum_wn = 0
    sum_wg = 0

    for _, tr in thresholds_df.iterrows():
        rule = str(tr["rule"])
        sc = RULE_SIGNAL_COL_NOMASK.get(rule)
        if not sc or sc not in row_df.columns:
            continue
        av = np.ones(len(row_df), dtype=bool)
        av_col = RULE_AVAILABLE_COL_NOMASK.get(rule)
        if av_col and av_col in row_df.columns:
            av = safe_bool_series(row_df[av_col])

        score = pd.to_numeric(row_df[sc], errors="coerce").to_numpy(dtype=float)
        base = hard_gate & av & np.isfinite(score) & known
        if not np.any(base):
            continue

        med = float(pd.to_numeric(pd.Series([tr.get("median", np.nan)]), errors="coerce").iloc[0])
        w = trigger_mask(
            score,
            tail_direction=tail_direction,
            threshold=float(tr.get("warn_threshold", np.nan)),
            threshold_low=float(tr.get("warn_threshold_low", np.nan)),
            threshold_high=float(tr.get("warn_threshold_high", np.nan)),
            median=med,
        ) & base
        f = trigger_mask(
            score,
            tail_direction=tail_direction,
            threshold=float(tr.get("fail_threshold", np.nan)),
            threshold_low=float(tr.get("fail_threshold_low", np.nan)),
            threshold_high=float(tr.get("fail_threshold_high", np.nan)),
            median=med,
        ) & base
        h = trigger_mask(
            score,
            tail_direction=tail_direction,
            threshold=float(tr.get("hard_fail_threshold", np.nan)),
            threshold_low=float(tr.get("hard_fail_threshold_low", np.nan)),
            threshold_high=float(tr.get("hard_fail_threshold_high", np.nan)),
            median=med,
        ) & base

        hard_only = h
        fail_only = f & (~hard_only)
        warn_only = w & (~f) & (~hard_only)

        hn = int(np.sum(hard_only))
        hb = int(np.sum(bad & hard_only))
        fn = int(np.sum(fail_only))
        fb = int(np.sum(bad & fail_only))
        wn = int(np.sum(warn_only))
        wg = int(np.sum(good & warn_only))

        hbp = float(hb / hn) if hn > 0 else float("nan")
        fbp = float(fb / fn) if fn > 0 else float("nan")
        wgr = float(wg / wn) if wn > 0 else float("nan")

        detail_rows.append(
            {
                "dataset": dataset,
                "rule": rule,
                "support_rows": int(np.sum(base)),
                "hard_n": hn,
                "hard_bad_precision": hbp,
                "hard_pass_target": bool(hn > 0 and np.isfinite(hbp) and hbp >= target_hard_bad_precision),
                "fail_n": fn,
                "fail_bad_precision": fbp,
                "fail_pass_target": bool(fn > 0 and np.isfinite(fbp) and fbp >= target_fail_bad_precision),
                "warn_n": wn,
                "warn_good_ratio": wgr,
                "warn_pass_target": bool(wn > 0 and np.isfinite(wgr) and wgr < target_warn_good_ratio),
            }
        )

        sum_hn += hn
        sum_hb += hb
        sum_fn += fn
        sum_fb += fb
        sum_wn += wn
        sum_wg += wg

    detail_df = pd.DataFrame(detail_rows)
    h_agg = float(sum_hb / sum_hn) if sum_hn > 0 else float("nan")
    f_agg = float(sum_fb / sum_fn) if sum_fn > 0 else float("nan")
    w_agg = float(sum_wg / sum_wn) if sum_wn > 0 else float("nan")
    summary = {
        "dataset": dataset,
        "rows": int(len(row_df)),
        "rules_evaluated": int(len(detail_df)),
        "hard_n_total": int(sum_hn),
        "hard_bad_precision_agg": h_agg,
        "hard_target_met_agg": bool(sum_hn > 0 and np.isfinite(h_agg) and h_agg >= target_hard_bad_precision),
        "hard_target_met_rules": int(detail_df["hard_pass_target"].sum()) if len(detail_df) else 0,
        "fail_n_total": int(sum_fn),
        "fail_bad_precision_agg": f_agg,
        "fail_target_met_agg": bool(sum_fn > 0 and np.isfinite(f_agg) and f_agg >= target_fail_bad_precision),
        "fail_target_met_rules": int(detail_df["fail_pass_target"].sum()) if len(detail_df) else 0,
        "warn_n_total": int(sum_wn),
        "warn_good_ratio_agg": w_agg,
        "warn_target_met_agg": bool(sum_wn > 0 and np.isfinite(w_agg) and w_agg < target_warn_good_ratio),
        "warn_target_met_rules": int(detail_df["warn_pass_target"].sum()) if len(detail_df) else 0,
    }
    summary["all_targets_met_agg"] = bool(
        summary["hard_target_met_agg"] and summary["fail_target_met_agg"] and summary["warn_target_met_agg"]
    )
    return detail_df, summary


def write_plot(out_html: Path, summary_df: pd.DataFrame) -> None:
    if go is None or pio is None:
        return
    x = summary_df["dataset"].astype(str).tolist()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=summary_df["hard_bad_precision_agg"], name="hard_bad_precision", marker_color="#dc2626"))
    fig.add_trace(go.Bar(x=x, y=summary_df["fail_bad_precision_agg"], name="fail_bad_precision", marker_color="#2563eb"))
    fig.add_trace(go.Bar(x=x, y=summary_df["warn_good_ratio_agg"], name="warn_good_ratio", marker_color="#f59e0b"))
    fig.update_layout(
        barmode="group",
        title="Mock20 Retest Summary",
        xaxis_title="dataset",
        yaxis_title="ratio",
        template="plotly_white",
    )
    out_html.write_text(pio.to_html(fig, include_plotlyjs="cdn", full_html=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root).resolve()
    output_root = Path(args.output_root).resolve()
    report_dir = output_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    row_csvs = sorted(runs_root.glob("mock_*/*_row_results.csv"))
    if not row_csvs:
        raise FileNotFoundError(f"no row_results found under {runs_root}")

    build_script = Path(__file__).resolve().parent / "build_thresholds_from_row_results.py"
    detail_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []

    for row_csv in row_csvs:
        dataset = row_csv.parent.name
        thr_csv, cfg_json = run_builder(
            python_exec=sys.executable,
            build_script=build_script,
            row_results_csv=row_csv,
            output_root=output_root,
            mode=str(args.mode),
            tail_direction=str(args.tail_direction),
        )
        cfg = json.loads(cfg_json.read_text(encoding="utf-8"))
        tail_direction = str(cfg.get("tail_direction", args.tail_direction))

        row_df = pd.read_csv(row_csv)
        thr_df = pd.read_csv(thr_csv)
        detail_df, summary = evaluate_one(
            row_df=row_df,
            thresholds_df=thr_df,
            tail_direction=tail_direction,
            target_hard_bad_precision=float(args.target_hard_bad_precision),
            target_fail_bad_precision=float(args.target_fail_bad_precision),
            target_warn_good_ratio=float(args.target_warn_good_ratio),
            dataset=dataset,
        )
        summary["row_results_csv"] = str(row_csv)
        summary["threshold_csv"] = str(thr_csv)
        summary["run_config_json"] = str(cfg_json)
        detail_frames.append(detail_df)
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries).sort_values("dataset")
    detail_df = pd.concat(detail_frames, axis=0, ignore_index=True) if detail_frames else pd.DataFrame()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    summary_csv = report_dir / f"mock20_retest_summary_{ts}.csv"
    detail_csv = report_dir / f"mock20_retest_rule_detail_{ts}.csv"
    summary_df.to_csv(summary_csv, index=False)
    detail_df.to_csv(detail_csv, index=False)

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs_root": str(runs_root),
        "datasets": [p.parent.name for p in row_csvs],
        "target_hard_bad_precision": float(args.target_hard_bad_precision),
        "target_fail_bad_precision": float(args.target_fail_bad_precision),
        "target_warn_good_ratio": float(args.target_warn_good_ratio),
        "summary_csv": str(summary_csv),
        "detail_csv": str(detail_csv),
    }
    meta_json = report_dir / f"mock20_retest_run_config_{ts}.json"
    meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_html = None
    if bool(args.emit_plot):
        plot_html = report_dir / f"mock20_retest_summary_{ts}.html"
        write_plot(plot_html, summary_df)

    # Global aggregate
    g_hn = int(summary_df["hard_n_total"].sum())
    g_fn = int(summary_df["fail_n_total"].sum())
    g_wn = int(summary_df["warn_n_total"].sum())
    g_h = (
        float((summary_df["hard_bad_precision_agg"] * summary_df["hard_n_total"]).sum() / g_hn)
        if g_hn > 0
        else float("nan")
    )
    g_f = (
        float((summary_df["fail_bad_precision_agg"] * summary_df["fail_n_total"]).sum() / g_fn)
        if g_fn > 0
        else float("nan")
    )
    g_w = (
        float((summary_df["warn_good_ratio_agg"] * summary_df["warn_n_total"]).sum() / g_wn)
        if g_wn > 0
        else float("nan")
    )

    print(f"[DONE] summary_csv: {summary_csv}")
    print(f"[DONE] detail_csv: {detail_csv}")
    print(f"[DONE] run_config: {meta_json}")
    if plot_html is not None:
        print(f"[DONE] summary_html: {plot_html}")
    print("[GLOBAL]")
    print(
        f"datasets={len(summary_df)} "
        f"hard_bad_precision_agg={g_h:.4f} fail_bad_precision_agg={g_f:.4f} warn_good_ratio_agg={g_w:.4f}"
    )
    cols = [
        "dataset",
        "rows",
        "hard_bad_precision_agg",
        "fail_bad_precision_agg",
        "warn_good_ratio_agg",
        "hard_target_met_agg",
        "fail_target_met_agg",
        "warn_target_met_agg",
        "all_targets_met_agg",
    ]
    print(summary_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
