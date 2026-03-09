#!/usr/bin/env python3
"""Benchmark fixed UDF runtime/quality.

UDF is fixed in distribution_outlier_pipeline (enabled, iterations=1).
This script benchmarks that fixed runtime and downstream quality.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "final_metric"
PIPELINE = FINAL / "distribution_outlier_pipeline.py"
FINAL_RUNNER = FINAL / "run_final_metric.py"

DEFAULT_SOURCE_CANDIDATES = [
    ROOT / "data" / "ambiguous_prompt_benchmark_v3_large.csv",
    FINAL / "data" / "ambiguous_prompt_benchmark_v3_large.csv",
]
DEFAULT_RULES = "output,direction,length,diff_residual,delta_ridge_ens,similar_input_conflict,discourse_instability,contradiction"


@dataclass
class Variant:
    name: str


def default_source_csv() -> Path:
    for path in DEFAULT_SOURCE_CANDIDATES:
        if path.exists():
            return path
    return DEFAULT_SOURCE_CANDIDATES[0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark UDF iterations")
    p.add_argument("--source-csv", default=str(default_source_csv()))
    p.add_argument("--output-root", default="")
    p.add_argument("--embedding-backend", default="hash", choices=["hash", "sentence-transformers", "auto"])
    p.add_argument("--embedding-model", default="google/embeddinggemma-300m")
    p.add_argument("--embedding-batch-size", type=int, default=64)
    p.add_argument("--rules", default=DEFAULT_RULES)
    p.add_argument("--tail-direction", default="two_sided", choices=["upper", "lower", "two_sided"])
    p.add_argument("--max-rows", type=int, default=0, help="0 means full dataset")
    return p.parse_args()


def run_cmd(cmd: list[str]) -> tuple[int, str, str, float]:
    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dur = time.perf_counter() - start
    return proc.returncode, proc.stdout, proc.stderr, dur


def safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


def binary_metrics(y_true_bad: np.ndarray, y_pred_bad: np.ndarray) -> dict[str, float]:
    yt = np.asarray(y_true_bad, dtype=bool)
    yp = np.asarray(y_pred_bad, dtype=bool)
    tp = int(np.sum(yt & yp))
    fp = int(np.sum((~yt) & yp))
    tn = int(np.sum((~yt) & (~yp)))
    fn = int(np.sum(yt & (~yp)))
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    fpr = safe_div(fp, fp + tn)
    return {
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "pred_bad_rate": float(np.mean(yp)) if len(yp) else 0.0,
    }


def get_source_labels(source_csv: Path) -> pd.DataFrame:
    src = pd.read_csv(source_csv)
    if "id" not in src.columns or "eval" not in src.columns:
        raise ValueError("source CSV must contain id and eval columns")
    lab = src[["id", "eval"]].copy()
    lab["id"] = lab["id"].astype(str)
    lab["is_bad"] = lab["eval"].astype(str).str.strip().str.lower().eq("incorrect")
    return lab[["id", "is_bad"]]


def main() -> None:
    args = parse_args()
    source_csv = Path(args.source_csv)
    if not source_csv.exists():
        raise FileNotFoundError(source_csv)

    if args.output_root:
        out_root = Path(args.output_root)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = FINAL / "results" / f"udf_iter_benchmark_{ts}"

    out_root.mkdir(parents=True, exist_ok=True)
    pipe_root = out_root / "pipeline_runs"
    final_root = out_root / "final_runs"
    report_root = out_root / "reports"
    pipe_root.mkdir(parents=True, exist_ok=True)
    final_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    variants = [Variant("udf_1_fixed")]

    label_df = get_source_labels(source_csv)

    rows: list[dict[str, Any]] = []
    long_rule_rows: list[pd.DataFrame] = []

    for v in variants:
        tag = f"udfbench_{v.name}"
        pipe_dir = pipe_root / v.name
        pipe_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(PIPELINE),
            "--csv-path",
            str(source_csv),
            "--id-col",
            "id",
            "--prompt-col",
            "Prompt",
            "--input-col",
            "input",
            "--output-col",
            "expectedOutput",
            "--output-dir",
            str(pipe_dir),
            "--tag",
            tag,
            "--embedding-backend",
            str(args.embedding_backend),
            "--embedding-model",
            str(args.embedding_model),
            "--embedding-batch-size",
            str(args.embedding_batch_size),
            "--no-save-plots",
            "--no-save-umap",
        ]
        if args.max_rows and args.max_rows > 0:
            cmd += ["--max-rows", str(args.max_rows)]

        rc, stdout, stderr, t_pipeline = run_cmd(cmd)
        if rc != 0:
            fail_path = report_root / f"{v.name}_pipeline_failed.log"
            fail_path.write_text(f"STDOUT\n{stdout}\n\nSTDERR\n{stderr}", encoding="utf-8")
            raise RuntimeError(f"pipeline failed for {v.name}: {fail_path}")

        pipe_report_dir = pipe_dir / "report"
        row_path = pipe_report_dir / f"{tag}_{source_csv.stem}_row_results.csv"
        summary_path = pipe_report_dir / f"{tag}_{source_csv.stem}_summary.csv"
        if not row_path.exists():
            raise FileNotFoundError(row_path)

        final_dir = final_root / v.name
        final_dir.mkdir(parents=True, exist_ok=True)
        final_tag = f"udfbench_final_{v.name}"
        cmd2 = [
            sys.executable,
            str(FINAL_RUNNER),
            "--source-csv",
            str(source_csv),
            "--row-results-csv",
            str(row_path),
            "--output-dir",
            str(final_dir),
            "--tag",
            final_tag,
            "--rules",
            str(args.rules),
            "--tail-direction",
            str(args.tail_direction),
            "--no-emit-plot",
        ]
        rc2, stdout2, stderr2, t_final = run_cmd(cmd2)
        if rc2 != 0:
            fail_path = report_root / f"{v.name}_final_failed.log"
            fail_path.write_text(f"STDOUT\n{stdout2}\n\nSTDERR\n{stderr2}", encoding="utf-8")
            raise RuntimeError(f"final_metric failed for {v.name}: {fail_path}")

        final_report_dir = final_dir / "report"
        thr_path = final_report_dir / f"{final_tag}_{source_csv.stem}_thresholds_summary.csv"
        final_row_path = final_report_dir / f"{final_tag}_{source_csv.stem}_row_results.csv"
        if not thr_path.exists() or not final_row_path.exists():
            raise FileNotFoundError(f"missing outputs for {v.name}")

        thr = pd.read_csv(thr_path)
        fr = pd.read_csv(final_row_path)

        rr = pd.read_csv(row_path)
        if "row_id" not in rr.columns:
            rr["row_id"] = fr.get("row_id", pd.Series(dtype=str)).astype(str)

        pred_bad = ~fr["final_pass_nomask"].fillna(False).astype(bool).to_numpy()

        joined = pd.DataFrame({"row_id": fr["row_id"].astype(str), "pred_bad": pred_bad}).merge(
            label_df,
            left_on="row_id",
            right_on="id",
            how="left",
        )
        known = joined["is_bad"].notna().to_numpy()
        y_true = joined["is_bad"].fillna(False).to_numpy(dtype=bool)
        y_pred = joined["pred_bad"].to_numpy(dtype=bool)

        all_mm = binary_metrics(y_true, y_pred)
        known_mm = binary_metrics(y_true[known], y_pred[known]) if np.any(known) else {k: float("nan") for k in ["tp","fp","tn","fn","precision","recall","f1","fpr","pred_bad_rate"]}

        hard_gate = rr["hard_gate_pass"].fillna(False).astype(bool).to_numpy() if "hard_gate_pass" in rr.columns else np.ones(len(rr), dtype=bool)
        hard_mask = known & hard_gate
        hard_mm = binary_metrics(y_true[hard_mask], y_pred[hard_mask]) if np.any(hard_mask) else {k: float("nan") for k in ["tp","fp","tn","fn","precision","recall","f1","fpr","pred_bad_rate"]}

        mean_f1 = float(pd.to_numeric(thr["f1"], errors="coerce").mean())
        mean_recall = float(pd.to_numeric(thr["recall"], errors="coerce").mean())
        mean_fpr = float(pd.to_numeric(thr["fpr"], errors="coerce").mean())
        finite_thr = bool(pd.to_numeric(thr["threshold_applied"], errors="coerce").notna().all())
        methods = ",".join(sorted(set(thr["selected_method"].astype(str).tolist())))

        udf_round_applied = np.nan
        udf_converged = np.nan
        if summary_path.exists():
            ps = pd.read_csv(summary_path)
            if not ps.empty:
                udf_round_applied = float(ps.iloc[0].get("udf_iterations_applied", np.nan))
                udf_converged = ps.iloc[0].get("udf_converged", np.nan)

        row = {
            "variant": v.name,
            "udf_enabled": True,
            "udf_iterations": 1,
            "pipeline_seconds": float(t_pipeline),
            "final_seconds": float(t_final),
            "total_seconds": float(t_pipeline + t_final),
            "hard_gate_pass_rate": float(np.mean(hard_gate)) if len(hard_gate) else np.nan,
            "udf_iterations_applied": udf_round_applied,
            "udf_converged": udf_converged,
            "rule_mean_f1": mean_f1,
            "rule_mean_recall": mean_recall,
            "rule_mean_fpr": mean_fpr,
            "rule_threshold_all_finite": finite_thr,
            "rule_selected_methods": methods,
            "final_all_precision": float(all_mm["precision"]),
            "final_all_recall": float(all_mm["recall"]),
            "final_all_f1": float(all_mm["f1"]),
            "final_all_fpr": float(all_mm["fpr"]),
            "final_known_precision": float(known_mm["precision"]),
            "final_known_recall": float(known_mm["recall"]),
            "final_known_f1": float(known_mm["f1"]),
            "final_known_fpr": float(known_mm["fpr"]),
            "final_hard_precision": float(hard_mm["precision"]),
            "final_hard_recall": float(hard_mm["recall"]),
            "final_hard_f1": float(hard_mm["f1"]),
            "final_hard_fpr": float(hard_mm["fpr"]),
            "rows": int(len(fr)),
            "rows_labeled": int(np.sum(known)),
            "rows_hard_labeled": int(np.sum(hard_mask)),
        }
        rows.append(row)

        thr_cp = thr.copy()
        thr_cp.insert(0, "variant", v.name)
        long_rule_rows.append(thr_cp)

        print(
            f"[DONE] {v.name}: total={row['total_seconds']:.2f}s "
            f"ruleF1={row['rule_mean_f1']:.4f} hardF1={row['final_hard_f1']:.4f}"
        )

    df = pd.DataFrame(rows).sort_values("total_seconds")

    base = df[df["variant"] == "udf_3"]
    if len(base) == 0 and len(df) > 0:
        base = df.iloc[[0]]
    if len(base) == 1:
        btime = float(base.iloc[0]["total_seconds"])
        bf1 = float(base.iloc[0]["final_hard_f1"])
        df["speedup_vs_udf3"] = btime / df["total_seconds"]
        df["delta_hard_f1_vs_udf3"] = df["final_hard_f1"] - bf1
    else:
        df["speedup_vs_udf3"] = np.nan
        df["delta_hard_f1_vs_udf3"] = np.nan

    out_csv = report_root / "udf_benchmark_summary.csv"
    df.to_csv(out_csv, index=False)

    long_df = pd.concat(long_rule_rows, ignore_index=True) if long_rule_rows else pd.DataFrame()
    long_csv = report_root / "udf_benchmark_rule_long.csv"
    long_df.to_csv(long_csv, index=False)

    overview = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(source_csv),
        "embedding_backend": str(args.embedding_backend),
        "embedding_model": str(args.embedding_model),
        "rows": int(df["rows"].iloc[0]) if len(df) else 0,
        "best_runtime_variant": str(df.iloc[0]["variant"]) if len(df) else "",
        "best_runtime_seconds": float(df["total_seconds"].min()) if len(df) else float("nan"),
        "best_hard_f1_variant": str(df.iloc[df["final_hard_f1"].idxmax()]["variant"]) if len(df) else "",
        "best_hard_f1": float(df["final_hard_f1"].max()) if len(df) else float("nan"),
    }
    (report_root / "udf_benchmark_overview.json").write_text(json.dumps(overview, indent=2), encoding="utf-8")

    md_lines = []
    md_lines.append("# UDF Iteration Benchmark")
    md_lines.append("")
    md_lines.append(f"- source: `{source_csv}`")
    md_lines.append(f"- embedding_backend: `{args.embedding_backend}`")
    md_lines.append(f"- embedding_model: `{args.embedding_model}`")
    md_lines.append("")
    display_cols = [
        "variant",
        "udf_enabled",
        "udf_iterations",
        "total_seconds",
        "speedup_vs_udf3",
        "rule_mean_f1",
        "rule_mean_recall",
        "rule_mean_fpr",
        "final_hard_f1",
        "final_hard_recall",
        "final_hard_fpr",
        "delta_hard_f1_vs_udf3",
    ]
    md_lines.append(df[display_cols].to_markdown(index=False))
    (report_root / "udf_benchmark_report.md").write_text("\n".join(md_lines), encoding="utf-8")

    print("[DONE] reports:", report_root)


if __name__ == "__main__":
    main()
