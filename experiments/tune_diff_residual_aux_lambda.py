#!/usr/bin/env python3
"""Tune diff-residual auxiliary lambda for stability across datasets.

Runs:
- baseline (no aux) once per dataset
- aux runs for each lambda
Then compares deltas against baseline and ranks lambdas by stability.
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
GEN_COMPLEX = ROOT / "data" / "generate_complex_mock_benchmarks.py"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    csv_path: Path
    split: str  # big / extra / mock20


def parse_args() -> argparse.Namespace:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    p = argparse.ArgumentParser(description="Lambda sweep for diff-residual aux stability")
    p.add_argument("--output-root", default=str(ROOT / "results" / f"lambda_tune_diff_aux_{ts}"))
    p.add_argument("--big-source-csv", default=str(ROOT / "data" / "ambiguous_prompt_benchmark_v3_large.csv"))
    p.add_argument(
        "--mock-datasets-root",
        default=str(ROOT / "results" / "mock_suite_20260225_batch1" / "datasets"),
    )
    p.add_argument(
        "--lambdas",
        default="0.1,0.2,0.3,0.4,0.5,0.7,0.9,1.1",
        help="Comma-separated aux lambda values.",
    )
    p.add_argument("--aux-model", default="linear", choices=["linear", "poly2"])
    p.add_argument("--embedding-backend", default="hash", choices=["auto", "sentence-transformers", "hash"])
    p.add_argument("--embedding-model", default="google/embeddinggemma-300m")
    p.add_argument("--embedding-batch-size", type=int, default=64)
    p.add_argument("--hash-dim", type=int, default=768)
    p.add_argument("--disable-hard-gates", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-plots", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--save-umap", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument(
        "--generate-extra-complex",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate complex_large and complex_low_error datasets and include in tuning.",
    )
    p.add_argument("--complex-large-good", type=int, default=2600)
    p.add_argument("--complex-large-bad", type=int, default=1000)
    p.add_argument("--complex-small-good", type=int, default=1100)
    p.add_argument("--complex-small-bad", type=int, default=30)
    p.add_argument("--complex-seed", type=int, default=4242)
    return p.parse_args()


def parse_lambdas(raw: str) -> list[float]:
    vals: list[float] = []
    for t in str(raw).split(","):
        s = t.strip()
        if not s:
            continue
        vals.append(float(s))
    if not vals:
        raise ValueError("No lambda values provided.")
    uniq = sorted(set(vals))
    return uniq


def maybe_generate_complex(args: argparse.Namespace, output_root: Path) -> list[DatasetSpec]:
    if not bool(args.generate_extra_complex):
        return []
    extra_dir = output_root / "generated_data"
    extra_dir.mkdir(parents=True, exist_ok=True)
    large_csv = extra_dir / "complex_mock_benchmark_large.csv"
    small_csv = extra_dir / "complex_mock_benchmark_low_error.csv"

    cmd = [
        sys.executable,
        str(GEN_COMPLEX),
        "--large-output-csv",
        str(large_csv),
        "--small-output-csv",
        str(small_csv),
        "--n-large-good",
        str(int(args.complex_large_good)),
        "--n-large-bad",
        str(int(args.complex_large_bad)),
        "--n-small-good",
        str(int(args.complex_small_good)),
        "--n-small-bad",
        str(int(args.complex_small_bad)),
        "--seed",
        str(int(args.complex_seed)),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Failed to generate complex datasets.\n"
            f"cmd={' '.join(cmd)}\n"
            f"stdout={proc.stdout}\n"
            f"stderr={proc.stderr}"
        )
    out: list[DatasetSpec] = []
    for name, pth in (
        ("complex_large", large_csv),
        ("complex_low_error", small_csv),
    ):
        if not pth.exists():
            raise FileNotFoundError(f"Generated dataset not found: {pth}")
        out.append(DatasetSpec(name=name, csv_path=pth.resolve(), split="extra"))
    return out


def discover_datasets(args: argparse.Namespace, output_root: Path) -> list[DatasetSpec]:
    out: list[DatasetSpec] = []
    big_csv = Path(args.big_source_csv).resolve()
    if not big_csv.exists():
        raise FileNotFoundError(f"big source csv not found: {big_csv}")
    out.append(DatasetSpec(name="big_main", csv_path=big_csv, split="big"))

    out.extend(maybe_generate_complex(args, output_root))

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
    aux_lambda: float,
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
        str(float(aux_lambda)),
        "--diff-residual-aux-model",
        str(args.aux_model),
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
    variant_key: str,
    aux_enabled: bool,
    aux_lambda: float,
    run_root: Path,
) -> Path:
    variant_dir = run_root / variant_key / spec.name
    variant_dir.mkdir(parents=True, exist_ok=True)
    tag = f"lam_{variant_key}_{spec.name}"
    cmd = build_cmd(
        args=args,
        csv_path=spec.csv_path,
        out_dir=variant_dir,
        tag=tag,
        aux_enabled=aux_enabled,
        aux_lambda=aux_lambda,
    )
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Pipeline run failed\n"
            f"dataset={spec.name}, variant={variant_key}\n"
            f"cmd={' '.join(cmd)}\n"
            f"stdout={proc.stdout}\n"
            f"stderr={proc.stderr}"
        )
    row_candidates = sorted((variant_dir / "report").glob(f"{tag}_{spec.csv_path.stem}_row_results.csv"))
    if not row_candidates:
        row_candidates = sorted((variant_dir / "report").glob("*_row_results.csv"))
    if not row_candidates:
        raise FileNotFoundError(f"row_results not found for dataset={spec.name}, variant={variant_key}")
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
    f1 = (
        float(2 * precision * recall / (precision + recall))
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
        else float("nan")
    )
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


def evaluate_row_results(*, row_csv: Path, spec: DatasetSpec, variant_key: str) -> list[dict[str, Any]]:
    df = pd.read_csv(row_csv)
    if "label_is_correct" not in df.columns:
        raise ValueError(f"label_is_correct missing in {row_csv}")
    y_bad_all = pd.to_numeric(df["label_is_correct"], errors="coerce")
    known = np.isfinite(y_bad_all.to_numpy(dtype=float))
    if not np.any(known):
        raise ValueError(f"no known labels in {row_csv}")

    rows: list[dict[str, Any]] = []
    for mode in ("nomask", "mask"):
        eval_col = f"distribution_evaluated_{mode}"
        eval_mask = np.ones(len(df), dtype=bool)
        if eval_col in df.columns:
            eval_mask = _safe_bool(df[eval_col])
        base_mask = known & eval_mask
        if not np.any(base_mask):
            continue
        y_bad = (1.0 - y_bad_all.to_numpy(dtype=float)[base_mask]).astype(bool)

        for metric_key, pass_col in (
            ("final_pass", f"final_pass_{mode}"),
            ("distribution_pass", f"distribution_pass_{mode}"),
            ("diff_residual_pass", f"diff_residual_pass_{mode}"),
        ):
            if pass_col not in df.columns:
                continue
            pass_mask = _safe_bool(df[pass_col])[base_mask]
            pred_bad = ~pass_mask
            m = binary_metrics(y_bad, pred_bad)
            rows.append(
                {
                    "dataset": spec.name,
                    "split": spec.split,
                    "variant": variant_key,
                    "mode": mode,
                    "metric": metric_key,
                    "row_results_csv": str(row_csv),
                    **m,
                }
            )
    return rows


def summarize_lambda_stability(comp_df: pd.DataFrame) -> pd.DataFrame:
    target = comp_df[(comp_df["metric"] == "final_pass") & (comp_df["mode"] == "nomask")].copy()
    if target.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for lam, g in target.groupby("lambda", dropna=False):
        arr = pd.to_numeric(g["delta_f1"], errors="coerce").to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            continue
        rows.append(
            {
                "lambda": float(lam),
                "datasets": int(len(arr)),
                "mean_delta_f1": float(np.mean(arr)),
                "median_delta_f1": float(np.median(arr)),
                "std_delta_f1": float(np.std(arr)),
                "min_delta_f1": float(np.min(arr)),
                "p10_delta_f1": float(np.quantile(arr, 0.10)),
                "wins": int(np.sum(arr > 0)),
                "ties": int(np.sum(arr == 0)),
                "losses": int(np.sum(arr < 0)),
            }
        )
    out = pd.DataFrame(rows).sort_values("lambda").reset_index(drop=True)
    return out


def rank_lambdas(stability_df: pd.DataFrame) -> pd.DataFrame:
    if stability_df.empty:
        return stability_df
    ranked = stability_df.copy()
    ranked = ranked.sort_values(
        by=["losses", "min_delta_f1", "std_delta_f1", "mean_delta_f1", "lambda"],
        ascending=[True, False, True, False, True],
    ).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def build_report(
    *,
    output_root: Path,
    args: argparse.Namespace,
    lambdas: list[float],
    run_map: dict[str, dict[str, str]],
    comp_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    ranked_df: pd.DataFrame,
) -> Path:
    lines: list[str] = []
    lines.append("# Lambda Tuning Report: Diff Residual Aux Stability")
    lines.append("")
    lines.append(f"- Generated at (UTC): {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- Output root: {output_root}")
    lines.append(f"- Lambdas: {lambdas}")
    lines.append(f"- Aux model: {args.aux_model}")
    lines.append(f"- Disable hard gates: {bool(args.disable_hard_gates)}")
    lines.append("")

    lines.append("## Ranking Criterion")
    lines.append("")
    lines.append("- Primary target: `final_pass` + `nomask` across all datasets.")
    lines.append("- Sort order: `losses asc`, `min_delta_f1 desc`, `std_delta_f1 asc`, `mean_delta_f1 desc`.")
    lines.append("")

    lines.append("## Lambda Ranking")
    lines.append("")
    if ranked_df.empty:
        lines.append("- No stability rows available.")
    else:
        for _, r in ranked_df.iterrows():
            lam_val = float(r["lambda"])
            lines.append(
                f"- Rank {int(r['rank'])} | lambda={lam_val:.3f} | "
                f"losses={int(r['losses'])}, wins={int(r['wins'])}, ties={int(r['ties'])} | "
                f"mean={float(r['mean_delta_f1']):+.4f}, min={float(r['min_delta_f1']):+.4f}, std={float(r['std_delta_f1']):.4f}"
            )
    lines.append("")

    if not ranked_df.empty:
        best = ranked_df.iloc[0]
        best_lambda = float(best["lambda"])
        lines.append("## Recommended Lambda")
        lines.append("")
        lines.append(
            f"- Recommended: `{best_lambda:.3f}` "
            f"(losses={int(best['losses'])}, min_delta_f1={float(best['min_delta_f1']):+.4f}, "
            f"mean_delta_f1={float(best['mean_delta_f1']):+.4f})"
        )
        lines.append("")

        best_rows = comp_df[
            (comp_df["lambda"] == best_lambda)
            & (comp_df["metric"] == "final_pass")
            & (comp_df["mode"] == "nomask")
        ].copy()
        if not best_rows.empty:
            lines.append("### Best Lambda Per-Dataset (final_pass/nomask)")
            lines.append("")
            for split in ("big", "extra", "mock20"):
                sub = best_rows[best_rows["split"] == split]
                if sub.empty:
                    continue
                lines.append(f"- {split}: mean delta_f1={sub['delta_f1'].mean():+.4f}, "
                             f"min={sub['delta_f1'].min():+.4f}, max={sub['delta_f1'].max():+.4f}, "
                             f"wins/ties/losses={int((sub['delta_f1']>0).sum())}/"
                             f"{int((sub['delta_f1']==0).sum())}/"
                             f"{int((sub['delta_f1']<0).sum())}")
            lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Lambda comparison CSV: {output_root / 'lambda_metrics_comparison.csv'}")
    lines.append(f"- Stability CSV: {output_root / 'lambda_stability_final_nomask.csv'}")
    lines.append(f"- Ranking CSV: {output_root / 'lambda_ranking.csv'}")
    lines.append(f"- Long metrics CSV: {output_root / 'lambda_metrics_long.csv'}")
    lines.append("- Run map:")
    for ds in sorted(run_map.keys()):
        base = run_map[ds].get("baseline", "")
        lines.append(f"  - {ds}: baseline={base}")
        for lam in lambdas:
            key = f"lambda_{lam:.3f}"
            if key in run_map[ds]:
                lines.append(f"    - {key}: {run_map[ds][key]}")

    report_path = output_root / "lambda_tuning_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    lambdas = parse_lambdas(args.lambdas)

    datasets = discover_datasets(args, output_root)
    run_root = output_root / "runs"
    run_root.mkdir(parents=True, exist_ok=True)

    run_map: dict[str, dict[str, str]] = {}
    all_metrics: list[dict[str, Any]] = []

    # Baseline once.
    for spec in datasets:
        run_map[spec.name] = {}
        row_csv = run_pipeline(
            args=args,
            spec=spec,
            variant_key="baseline",
            aux_enabled=False,
            aux_lambda=0.0,
            run_root=run_root,
        )
        run_map[spec.name]["baseline"] = str(row_csv)
        all_metrics.extend(evaluate_row_results(row_csv=row_csv, spec=spec, variant_key="baseline"))
        print(f"[DONE] baseline dataset={spec.name} split={spec.split}")

    # Aux sweep.
    for lam in lambdas:
        lam_key = f"lambda_{lam:.3f}"
        for spec in datasets:
            row_csv = run_pipeline(
                args=args,
                spec=spec,
                variant_key=lam_key,
                aux_enabled=True,
                aux_lambda=float(lam),
                run_root=run_root,
            )
            run_map[spec.name][lam_key] = str(row_csv)
            all_metrics.extend(evaluate_row_results(row_csv=row_csv, spec=spec, variant_key=lam_key))
            print(f"[DONE] lambda={lam:.3f} dataset={spec.name} split={spec.split}")

    metrics_df = pd.DataFrame(all_metrics)
    if metrics_df.empty:
        raise RuntimeError("No metrics collected.")
    metrics_long_csv = output_root / "lambda_metrics_long.csv"
    metrics_df.to_csv(metrics_long_csv, index=False)

    base = metrics_df[metrics_df["variant"] == "baseline"].copy()
    base = base.rename(
        columns={
            "f1": "f1_baseline",
            "precision": "precision_baseline",
            "recall": "recall_baseline",
            "accuracy": "accuracy_baseline",
            "tp": "tp_baseline",
            "fp": "fp_baseline",
            "tn": "tn_baseline",
            "fn": "fn_baseline",
            "n": "n_baseline",
            "bad_rate": "bad_rate_baseline",
            "row_results_csv": "row_results_csv_baseline",
        }
    )

    aux = metrics_df[metrics_df["variant"] != "baseline"].copy()
    aux["lambda"] = aux["variant"].str.replace("lambda_", "", regex=False).astype(float)
    aux = aux.rename(
        columns={
            "f1": "f1_aux",
            "precision": "precision_aux",
            "recall": "recall_aux",
            "accuracy": "accuracy_aux",
            "tp": "tp_aux",
            "fp": "fp_aux",
            "tn": "tn_aux",
            "fn": "fn_aux",
            "n": "n_aux",
            "bad_rate": "bad_rate_aux",
            "row_results_csv": "row_results_csv_aux",
        }
    )

    key_cols = ["dataset", "split", "mode", "metric"]
    comp = aux.merge(
        base[
            key_cols
            + [
                "f1_baseline",
                "precision_baseline",
                "recall_baseline",
                "accuracy_baseline",
                "tp_baseline",
                "fp_baseline",
                "tn_baseline",
                "fn_baseline",
                "n_baseline",
                "bad_rate_baseline",
                "row_results_csv_baseline",
            ]
        ],
        on=key_cols,
        how="left",
    )

    for c in ("f1", "precision", "recall", "accuracy"):
        comp[f"delta_{c}"] = comp[f"{c}_aux"] - comp[f"{c}_baseline"]
    comp["delta_tp"] = comp["tp_aux"] - comp["tp_baseline"]
    comp["delta_fp"] = comp["fp_aux"] - comp["fp_baseline"]

    comp_csv = output_root / "lambda_metrics_comparison.csv"
    comp.to_csv(comp_csv, index=False)

    stability = summarize_lambda_stability(comp)
    stability_csv = output_root / "lambda_stability_final_nomask.csv"
    stability.to_csv(stability_csv, index=False)

    ranked = rank_lambdas(stability)
    ranking_csv = output_root / "lambda_ranking.csv"
    ranked.to_csv(ranking_csv, index=False)

    report_path = build_report(
        output_root=output_root,
        args=args,
        lambdas=lambdas,
        run_map=run_map,
        comp_df=comp,
        stability_df=stability,
        ranked_df=ranked,
    )

    config_path = output_root / "lambda_tune_run_config.json"
    config_path.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "args": vars(args),
                "datasets": [{"name": d.name, "csv_path": str(d.csv_path), "split": d.split} for d in datasets],
                "lambdas": lambdas,
                "report_path": str(report_path),
                "comparison_csv": str(comp_csv),
                "stability_csv": str(stability_csv),
                "ranking_csv": str(ranking_csv),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[DONE] report: {report_path}")
    print(f"[DONE] comparison_csv: {comp_csv}")
    print(f"[DONE] stability_csv: {stability_csv}")
    print(f"[DONE] ranking_csv: {ranking_csv}")
    print(f"[DONE] run_config: {config_path}")


if __name__ == "__main__":
    main()
