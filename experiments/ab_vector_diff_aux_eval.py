#!/usr/bin/env python3
"""Run A/B evaluation for diff-residual auxiliary vector-diff scoring.

A: baseline local Mahalanobis diff_residual
B: baseline + auxiliary residual boost from direction/length
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


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "distribution_outlier_pipeline.py"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    csv_path: Path
    split: str  # "big" or "mock20"


def parse_args() -> argparse.Namespace:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    p = argparse.ArgumentParser(description="A/B test for diff-residual auxiliary scoring")
    p.add_argument(
        "--big-source-csv",
        default=str(ROOT / "data" / "ambiguous_prompt_benchmark_v3_large.csv"),
    )
    p.add_argument(
        "--mock-datasets-root",
        default=str(ROOT / "results" / "mock_suite_20260225_batch1" / "datasets"),
        help="Directory containing mock_00..mock_19 subdirectories with *_source.csv",
    )
    p.add_argument(
        "--output-root",
        default=str(ROOT / "results" / f"ab_vector_diff_aux_{ts}"),
    )
    p.add_argument(
        "--embedding-backend",
        default="hash",
        choices=["auto", "sentence-transformers", "hash"],
    )
    p.add_argument("--embedding-model", default="google/embeddinggemma-300m")
    p.add_argument("--embedding-batch-size", type=int, default=64)
    p.add_argument("--hash-dim", type=int, default=768)
    p.add_argument("--diff-residual-aux-lambda", type=float, default=0.1)
    p.add_argument("--diff-residual-aux-model", default="linear", choices=["linear", "poly2"])
    p.add_argument(
        "--disable-hard-gates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable IFEval/schema/textlen to isolate distribution metric behavior.",
    )
    p.add_argument("--save-plots", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--save-umap", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--max-rows", type=int, default=0, help="0 means full rows")
    return p.parse_args()


def discover_datasets(args: argparse.Namespace) -> list[DatasetSpec]:
    out: list[DatasetSpec] = []
    big_csv = Path(args.big_source_csv).resolve()
    if not big_csv.exists():
        raise FileNotFoundError(f"big source csv not found: {big_csv}")
    out.append(DatasetSpec(name="big_main", csv_path=big_csv, split="big"))

    mock_root = Path(args.mock_datasets_root).resolve()
    if not mock_root.exists():
        raise FileNotFoundError(f"mock datasets root not found: {mock_root}")
    mock_csvs = sorted(mock_root.glob("mock_*/*_source.csv"))
    if len(mock_csvs) != 20:
        raise ValueError(f"expected 20 mock source csv files under {mock_root}, found {len(mock_csvs)}")
    for p in mock_csvs:
        out.append(DatasetSpec(name=p.parent.name, csv_path=p.resolve(), split="mock20"))
    return out


def build_cmd(
    *,
    args: argparse.Namespace,
    csv_path: Path,
    out_dir: Path,
    tag: str,
    aux_enabled: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        str(PIPELINE),
        "--csv-path",
        str(csv_path),
        "--output-dir",
        str(out_dir),
        "--report-dir-name",
        "report",
        "--tag",
        tag,
        "--embedding-backend",
        str(args.embedding_backend),
        "--embedding-model",
        str(args.embedding_model),
        "--embedding-batch-size",
        str(int(args.embedding_batch_size)),
        "--hash-dim",
        str(int(args.hash_dim)),
        "--diff-residual-aux-lambda",
        str(float(args.diff_residual_aux_lambda)),
        "--diff-residual-aux-model",
        str(args.diff_residual_aux_model),
    ]
    if bool(aux_enabled):
        cmd.append("--diff-residual-aux-enabled")
    else:
        cmd.append("--no-diff-residual-aux-enabled")
    if bool(args.disable_hard_gates):
        cmd.extend(["--disable-ifeval", "--disable-schema", "--disable-textlen"])
    if bool(args.save_plots):
        cmd.append("--save-plots")
    else:
        cmd.append("--no-save-plots")
    if bool(args.save_umap):
        cmd.append("--save-umap")
    else:
        cmd.append("--no-save-umap")
    if int(args.max_rows) > 0:
        cmd.extend(["--max-rows", str(int(args.max_rows))])
    return cmd


def run_pipeline(
    *,
    args: argparse.Namespace,
    spec: DatasetSpec,
    variant: str,
    aux_enabled: bool,
    run_root: Path,
) -> Path:
    variant_dir = run_root / variant / spec.name
    variant_dir.mkdir(parents=True, exist_ok=True)
    tag = f"ab_{variant}_{spec.name}"
    cmd = build_cmd(
        args=args,
        csv_path=spec.csv_path,
        out_dir=variant_dir,
        tag=tag,
        aux_enabled=aux_enabled,
    )
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Pipeline run failed\n"
            f"dataset={spec.name}, variant={variant}\n"
            f"cmd={' '.join(cmd)}\n"
            f"stdout={proc.stdout}\n"
            f"stderr={proc.stderr}"
        )
    row_candidates = sorted((variant_dir / "report").glob(f"{tag}_{spec.csv_path.stem}_row_results.csv"))
    if not row_candidates:
        row_candidates = sorted((variant_dir / "report").glob("*_row_results.csv"))
    if not row_candidates:
        raise FileNotFoundError(f"row_results not found for dataset={spec.name}, variant={variant}")
    return row_candidates[-1]


def _safe_bool(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).to_numpy(dtype=float) != 0.0
    mapped = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False})
        .fillna(False)
    )
    return mapped.to_numpy(dtype=bool)


def binary_metrics(y_bad: np.ndarray, pred_bad: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y_bad, dtype=bool)
    p = np.asarray(pred_bad, dtype=bool)
    tp = int(np.sum(y & p))
    fp = int(np.sum((~y) & p))
    tn = int(np.sum((~y) & (~p)))
    fn = int(np.sum(y & (~p)))
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    f1 = float(2 * precision * recall / (precision + recall)) if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0 else float("nan")
    accuracy = float((tp + tn) / max(len(y), 1))
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "n": int(len(y)),
        "bad_rate": float(np.mean(y)) if len(y) else float("nan"),
    }


def evaluate_row_results(
    *,
    row_csv: Path,
    spec: DatasetSpec,
    variant: str,
) -> list[dict[str, Any]]:
    df = pd.read_csv(row_csv)
    if "label_is_correct" not in df.columns:
        raise ValueError(f"label_is_correct missing in {row_csv}")
    y_bad_all = pd.to_numeric(df["label_is_correct"], errors="coerce")
    known = np.isfinite(y_bad_all.to_numpy(dtype=float))
    if not np.any(known):
        raise ValueError(f"no known labels in {row_csv}")

    out: list[dict[str, Any]] = []
    for mode in ("nomask", "mask"):
        eval_col = f"distribution_evaluated_{mode}"
        eval_mask = np.ones(len(df), dtype=bool)
        if eval_col in df.columns:
            eval_mask = _safe_bool(df[eval_col])
        base_mask = known & eval_mask
        if not np.any(base_mask):
            continue
        y_bad = (1.0 - y_bad_all.to_numpy(dtype=float)[base_mask]).astype(bool)

        pass_specs = [
            ("final_pass", f"final_pass_{mode}"),
            ("distribution_pass", f"distribution_pass_{mode}"),
            ("diff_residual_pass", f"diff_residual_pass_{mode}"),
        ]
        for metric_key, pass_col in pass_specs:
            if pass_col not in df.columns:
                continue
            pass_mask = _safe_bool(df[pass_col])[base_mask]
            pred_bad = ~pass_mask
            m = binary_metrics(y_bad, pred_bad)
            out.append(
                {
                    "dataset": spec.name,
                    "split": spec.split,
                    "variant": variant,
                    "mode": mode,
                    "metric": metric_key,
                    "row_results_csv": str(row_csv),
                    **m,
                }
            )
    return out


def build_report(
    *,
    metrics_df: pd.DataFrame,
    output_root: Path,
    args: argparse.Namespace,
    run_map: dict[str, dict[str, str]],
) -> tuple[Path, Path]:
    metrics_csv = output_root / "ab_metrics_long.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    piv = metrics_df.pivot_table(
        index=["dataset", "split", "mode", "metric"],
        columns="variant",
        values=["f1", "precision", "recall", "accuracy", "tp", "fp", "tn", "fn", "n", "bad_rate"],
        aggfunc="first",
    )
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.reset_index()
    for c in ["f1", "precision", "recall", "accuracy"]:
        if f"{c}_baseline" in piv.columns and f"{c}_aux" in piv.columns:
            piv[f"delta_{c}"] = piv[f"{c}_aux"] - piv[f"{c}_baseline"]
    if "tp_baseline" in piv.columns and "tp_aux" in piv.columns:
        piv["delta_tp"] = piv["tp_aux"] - piv["tp_baseline"]
    if "fp_baseline" in piv.columns and "fp_aux" in piv.columns:
        piv["delta_fp"] = piv["fp_aux"] - piv["fp_baseline"]

    cmp_csv = output_root / "ab_metrics_comparison.csv"
    piv.to_csv(cmp_csv, index=False)

    def _subset(split: str, mode: str, metric: str) -> pd.DataFrame:
        return piv[(piv["split"] == split) & (piv["mode"] == mode) & (piv["metric"] == metric)].copy()

    big_main = _subset("big", "nomask", "final_pass")
    mock_main = _subset("mock20", "nomask", "final_pass")
    mock_diff = _subset("mock20", "nomask", "diff_residual_pass")

    lines: list[str] = []
    lines.append("# AB Report: Diff Residual Aux Vector-Diff")
    lines.append("")
    lines.append(f"- Generated at (UTC): {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- Output root: {output_root}")
    lines.append(f"- Big dataset: {Path(args.big_source_csv).resolve()}")
    lines.append(f"- Mock datasets root: {Path(args.mock_datasets_root).resolve()}")
    lines.append(f"- Embedding backend: {args.embedding_backend}")
    lines.append(f"- Disable hard gates: {bool(args.disable_hard_gates)}")
    lines.append(f"- Aux lambda/model: {float(args.diff_residual_aux_lambda)} / {str(args.diff_residual_aux_model)}")
    lines.append("")
    lines.append("## Big Dataset (nomask, final_pass)")
    lines.append("")
    if len(big_main) == 1:
        row = big_main.iloc[0]
        lines.append(f"- Baseline F1: {row['f1_baseline']:.4f}")
        lines.append(f"- Aux F1: {row['f1_aux']:.4f}")
        lines.append(f"- Delta F1: {row['delta_f1']:+.4f}")
        lines.append(f"- Baseline Recall: {row['recall_baseline']:.4f}")
        lines.append(f"- Aux Recall: {row['recall_aux']:.4f}")
        lines.append(f"- Delta Recall: {row['delta_recall']:+.4f}")
        lines.append(f"- Baseline Precision: {row['precision_baseline']:.4f}")
        lines.append(f"- Aux Precision: {row['precision_aux']:.4f}")
        lines.append(f"- Delta Precision: {row['delta_precision']:+.4f}")
        lines.append(f"- Delta TP / FP: {int(row.get('delta_tp', 0)):+d} / {int(row.get('delta_fp', 0)):+d}")
    else:
        lines.append("- Big dataset result not found.")
    lines.append("")
    lines.append("## Mock20 Aggregate (nomask, final_pass)")
    lines.append("")
    if len(mock_main) > 0:
        wins = int((mock_main["delta_f1"] > 0).sum())
        ties = int((mock_main["delta_f1"] == 0).sum())
        losses = int((mock_main["delta_f1"] < 0).sum())
        lines.append(f"- Datasets: {len(mock_main)}")
        lines.append(f"- F1 delta mean / median: {mock_main['delta_f1'].mean():+.4f} / {mock_main['delta_f1'].median():+.4f}")
        lines.append(f"- Recall delta mean / median: {mock_main['delta_recall'].mean():+.4f} / {mock_main['delta_recall'].median():+.4f}")
        lines.append(f"- Precision delta mean / median: {mock_main['delta_precision'].mean():+.4f} / {mock_main['delta_precision'].median():+.4f}")
        lines.append(f"- Win / Tie / Loss (F1): {wins} / {ties} / {losses}")
        lines.append(f"- Delta TP mean / median: {mock_main['delta_tp'].mean():+.2f} / {mock_main['delta_tp'].median():+.2f}")
        lines.append(f"- Delta FP mean / median: {mock_main['delta_fp'].mean():+.2f} / {mock_main['delta_fp'].median():+.2f}")
    else:
        lines.append("- Mock20 final_pass result not found.")
    lines.append("")
    lines.append("## Mock20 Aggregate (nomask, diff_residual_pass)")
    lines.append("")
    if len(mock_diff) > 0:
        wins = int((mock_diff["delta_f1"] > 0).sum())
        ties = int((mock_diff["delta_f1"] == 0).sum())
        losses = int((mock_diff["delta_f1"] < 0).sum())
        lines.append(f"- Datasets: {len(mock_diff)}")
        lines.append(f"- F1 delta mean / median: {mock_diff['delta_f1'].mean():+.4f} / {mock_diff['delta_f1'].median():+.4f}")
        lines.append(f"- Recall delta mean / median: {mock_diff['delta_recall'].mean():+.4f} / {mock_diff['delta_recall'].median():+.4f}")
        lines.append(f"- Precision delta mean / median: {mock_diff['delta_precision'].mean():+.4f} / {mock_diff['delta_precision'].median():+.4f}")
        lines.append(f"- Win / Tie / Loss (F1): {wins} / {ties} / {losses}")
    else:
        lines.append("- Mock20 diff_residual_pass result not found.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Long metrics CSV: {metrics_csv}")
    lines.append(f"- Comparison CSV: {cmp_csv}")
    lines.append("- Run rows (baseline/aux):")
    for k in sorted(run_map.keys()):
        lines.append(f"  - {k}: baseline={run_map[k].get('baseline','')}, aux={run_map[k].get('aux','')}")

    report_path = output_root / "ab_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path, cmp_csv


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    datasets = discover_datasets(args)
    run_root = output_root / "runs"
    run_root.mkdir(parents=True, exist_ok=True)

    all_metrics: list[dict[str, Any]] = []
    run_map: dict[str, dict[str, str]] = {}

    for spec in datasets:
        run_map[spec.name] = {}
        for variant, aux_enabled in (("baseline", False), ("aux", True)):
            row_csv = run_pipeline(
                args=args,
                spec=spec,
                variant=variant,
                aux_enabled=aux_enabled,
                run_root=run_root,
            )
            run_map[spec.name][variant] = str(row_csv)
            all_metrics.extend(
                evaluate_row_results(
                    row_csv=row_csv,
                    spec=spec,
                    variant=variant,
                )
            )
            print(f"[DONE] dataset={spec.name} split={spec.split} variant={variant} row_results={row_csv}")

    metrics_df = pd.DataFrame(all_metrics)
    if metrics_df.empty:
        raise RuntimeError("No evaluation metrics produced.")

    report_path, cmp_csv = build_report(
        metrics_df=metrics_df,
        output_root=output_root,
        args=args,
        run_map=run_map,
    )

    config_path = output_root / "ab_run_config.json"
    config_path.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "args": vars(args),
                "datasets": [{"name": d.name, "csv_path": str(d.csv_path), "split": d.split} for d in datasets],
                "report_path": str(report_path),
                "comparison_csv": str(cmp_csv),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[DONE] report: {report_path}")
    print(f"[DONE] comparison_csv: {cmp_csv}")
    print(f"[DONE] run_config: {config_path}")


if __name__ == "__main__":
    main()
