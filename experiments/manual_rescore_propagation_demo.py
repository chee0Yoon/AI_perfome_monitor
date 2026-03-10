#!/usr/bin/env python3
"""Propagation demo for one human seed_pass override."""

from __future__ import annotations

import argparse
import json
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

from final_metric_refactor.config import SCORE_RUNTIME
from final_metric_refactor.run import (
    _apply_manual_rescore_nomask,
    _mean_bundle_new_score,
    apply_hybrid_thresholds_nomask,
)
from final_metric_refactor.scoring import compute_bundle_scores


def build_demo_source_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [f"r{i}" for i in range(12)],
            "eval": [
                "correct",
                "correct",
                "incorrect",
                "correct",
                "incorrect",
                "correct",
                "correct",
                "incorrect",
                "correct",
                "correct",
                "incorrect",
                "correct",
            ],
        }
    )


def build_demo_row_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": [f"r{i}" for i in range(12)],
            "hard_gate_pass": [True] * 12,
            "distribution_group_id": ["g0000"] * 6 + ["g0001"] * 6,
            "distribution_group_col": ["Prompt"] * 12,
            "distribution_group_size": [6] * 12,
            "source_input": [f"in{i}" for i in range(12)],
            "source_output": [f"out{i}" for i in range(12)],
            "output_signal_nomask": [
                0.2,
                8.6045,
                6.2872,
                0.5899,
                6.5327,
                0.2523,
                0.3774,
                6.1894,
                0.5966,
                0.3865,
                5.1543,
                0.3750,
            ],
            "output_pca_x_nomask": np.linspace(-1.0, 1.0, 12),
            "output_pca_y_nomask": np.linspace(0.0, 2.0, 12),
        }
    )


def build_demo_input_norm() -> np.ndarray:
    raw = np.asarray(
        [
            [1.0, 0.0],
            [0.999, 0.01],
            [0.98, 0.20],
            [0.50, 0.866],
            [0.0, 1.0],
            [0.30, 0.95],
            [-1.0, 0.0],
            [-0.98, 0.20],
            [-0.95, -0.10],
            [-0.90, -0.20],
            [-0.85, -0.30],
            [-0.80, -0.40],
        ],
        dtype=float,
    )
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


def build_args(manual_override_csv: str = "") -> Namespace:
    return Namespace(
        source_id_col="id",
        results_id_col="row_id",
        label_col="eval",
        bad_label="incorrect",
        tail_direction="upper",
        manual_override_csv=manual_override_csv,
    )


def run_demo(output_dir: Path | None = None) -> dict[str, object]:
    source_df = build_demo_source_df()
    row_df = build_demo_row_df()
    input_norm = build_demo_input_norm()
    args = build_args()

    baseline_row_df, baseline_thr = apply_hybrid_thresholds_nomask(
        source_df=source_df,
        row_df=row_df,
        rules=["output"],
        args=args,
    )
    baseline_art = compute_bundle_scores(
        row_df=baseline_row_df,
        threshold_summary_df=baseline_thr,
        score_runtime=SCORE_RUNTIME,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        override_path = Path(tmpdir) / "manual_override.csv"
        pd.DataFrame([{"row_id": "r0", "review_label": "seed_pass"}]).to_csv(override_path, index=False)
        args.manual_override_csv = str(override_path)
        rescored_row_df, rescored_thr, stats = _apply_manual_rescore_nomask(
            source_df=source_df,
            row_df=baseline_row_df,
            threshold_summary_df=baseline_thr,
            args=args,
            rules=["output"],
            input_norm=input_norm,
            output_norm=None,
            embedding_meta=None,
        )

    rescored_art = compute_bundle_scores(
        row_df=rescored_row_df,
        threshold_summary_df=rescored_thr,
        score_runtime=SCORE_RUNTIME,
    )
    changed_rows = rescored_row_df.loc[
        rescored_row_df["manual_rescore_applied"].astype(bool),
        [
            "row_id",
            "manual_review_label",
            "manual_anchor_pass",
            "manual_propagated_pass",
            "distribution_state_nomask",
            "final_pass_nomask",
        ],
    ].copy()

    payload = {
        "summary": {
            "selected_k": int(stats["selected_k"]),
            "affected_groups": int(stats["affected_groups"]),
            "anchor_rows": int(stats["anchor_rows"]),
            "propagated_rows": int(stats["propagated_rows"]),
            "final_pass_rate_pre": float(stats["final_pass_rate_pre"]),
            "final_pass_rate_post": float(stats["final_pass_rate_post"]),
            "new_score_pre": float(stats["new_score_pre"]),
            "new_score_post": float(stats["new_score_post"]),
            "bundle_new_score_pre_recomputed": float(_mean_bundle_new_score(baseline_art.summary_df)),
            "bundle_new_score_post_recomputed": float(_mean_bundle_new_score(rescored_art.summary_df)),
        },
        "changed_rows": changed_rows.to_dict(orient="records"),
    }

    if output_dir is not None:
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        source_df.to_csv(out_dir / "demo_source.csv", index=False)
        pd.DataFrame([{"row_id": "r0", "review_label": "seed_pass"}]).to_csv(out_dir / "demo_manual_override.csv", index=False)
        baseline_row_df.to_csv(out_dir / "demo_baseline_row_results.csv", index=False)
        rescored_row_df.to_csv(out_dir / "demo_rescored_row_results.csv", index=False)
        rescored_thr.to_csv(out_dir / "demo_rescored_thresholds.csv", index=False)
        (out_dir / "demo_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run manual rescore propagation demo")
    parser.add_argument("--output-dir", default="", help="Optional directory to export demo CSV/JSON artifacts")
    args = parser.parse_args()

    payload = run_demo(output_dir=Path(args.output_dir) if args.output_dir else None)
    print("SUMMARY")
    for key, value in payload["summary"].items():
        print(f"{key}: {value}")
    print("\nCHANGED_ROWS")
    print(pd.DataFrame(payload["changed_rows"]).to_string(index=False))


if __name__ == "__main__":
    main()
