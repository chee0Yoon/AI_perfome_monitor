#!/usr/bin/env python3
"""Unit tests for manual human-pass rescore helpers."""

from __future__ import annotations

import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
FINAL_DIR = THIS_DIR.parent
ROOT_DIR = FINAL_DIR.parent
for p in (ROOT_DIR, FINAL_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from final_metric_refactor.manual_review import (  # noqa: E402
    apply_manual_final_overrides,
    build_group_local_propagation_mask,
    choose_manual_review_k,
    load_manual_override_csv,
)
from final_metric_refactor.run import (  # noqa: E402
    _apply_manual_rescore_nomask,
    build_raw_data_export,
    apply_hybrid_thresholds_nomask,
)
from final_metric_refactor.scoring import compute_bundle_scores  # noqa: E402
from final_metric_refactor.config import SCORE_RUNTIME  # noqa: E402


class ManualRescoreTest(unittest.TestCase):
    def _args(self, manual_override_csv: str = "") -> Namespace:
        return Namespace(
            source_id_col="id",
            results_id_col="row_id",
            label_col="eval",
            bad_label="incorrect",
            tail_direction="upper",
            manual_override_csv=manual_override_csv,
        )

    def _source_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "id": [f"r{i}" for i in range(10)],
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
                ],
            }
        )

    def _row_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "row_id": [f"r{i}" for i in range(10)],
                "hard_gate_pass": [True] * 10,
                "distribution_group_id": ["g0000"] * 6 + ["g0001"] * 4,
                "distribution_group_col": ["Prompt"] * 10,
                "distribution_group_size": [6] * 6 + [4] * 4,
                "source_input": [f"in{i}" for i in range(10)],
                "source_output": [f"out{i}" for i in range(10)],
                "output_signal_nomask": [0.20, 0.24, 5.00, 0.30, 4.80, 0.28, 0.35, 4.70, 0.32, 0.34],
                "output_pca_x_nomask": [0.0, 0.1, 2.0, 0.3, 1.8, 0.2, -0.1, -2.0, -0.2, -0.3],
                "output_pca_y_nomask": [0.0, 0.1, 2.0, 0.2, 1.7, 0.3, 0.1, 2.1, 0.2, 0.0],
            }
        )

    def _input_norm(self) -> np.ndarray:
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
            ],
            dtype=float,
        )
        return raw / np.linalg.norm(raw, axis=1, keepdims=True)

    def test_override_csv_rejects_unknown_row_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "override.csv"
            pd.DataFrame({"row_id": ["missing"], "review_label": ["seed_pass"]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "unknown row_id"):
                load_manual_override_csv(path, valid_row_ids=["r0", "r1"])

    def test_override_csv_rejects_invalid_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "override.csv"
            pd.DataFrame({"row_id": ["r0"], "review_label": ["maybe"]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "invalid review_label"):
                load_manual_override_csv(path, valid_row_ids=["r0", "r1"])

    def test_group_local_propagation_stays_inside_group(self) -> None:
        propagated = build_group_local_propagation_mask(
            input_norm=self._input_norm(),
            group_ids=np.asarray(["g0000"] * 6 + ["g0001"] * 4, dtype=object),
            anchor_mask=np.asarray([True, False, False, False, False, False, False, False, False, False], dtype=bool),
            k=1,
        )
        self.assertListEqual(propagated.tolist(), [False, True, False, False, False, False, False, False, False, False])

    def test_precision_first_selects_smallest_positive_k(self) -> None:
        chosen = choose_manual_review_k(
            [
                {"k": 0, "precision": 1.0, "new_score_post": 3.0},
                {"k": 1, "precision": 1.0, "new_score_post": 3.5},
                {"k": 3, "precision": 0.66, "new_score_post": 4.0},
            ]
        )
        self.assertEqual(int(chosen["k"]), 1)

    def test_final_overrides_win(self) -> None:
        row_df = pd.DataFrame(
            {
                "row_id": ["r0", "r1"],
                "distribution_pass_nomask": [False, True],
                "distribution_warn_nomask": [True, False],
                "distribution_fail_nomask": [False, False],
                "distribution_hard_fail_nomask": [False, False],
                "distribution_state_nomask": ["warn", "pass"],
                "final_pass_nomask": [False, True],
                "final_state_nomask": ["warn", "pass"],
            }
        )
        out = apply_manual_final_overrides(
            row_df,
            row_id_col="row_id",
            final_pass_ids={"r0"},
            final_fail_ids={"r1"},
        )
        self.assertTrue(bool(out.loc[out["row_id"] == "r0", "final_pass_nomask"].iloc[0]))
        self.assertEqual(str(out.loc[out["row_id"] == "r0", "final_state_nomask"].iloc[0]), "pass")
        self.assertFalse(bool(out.loc[out["row_id"] == "r1", "final_pass_nomask"].iloc[0]))
        self.assertEqual(str(out.loc[out["row_id"] == "r1", "final_state_nomask"].iloc[0]), "fail")

    def test_no_manual_override_keeps_threshold_output_identical(self) -> None:
        source_df = self._source_df()
        row_df = self._row_df()
        args = self._args()
        baseline_row_df, baseline_thr = apply_hybrid_thresholds_nomask(
            source_df=source_df,
            row_df=row_df,
            rules=["output"],
            args=args,
        )
        rescored_row_df, rescored_thr, stats = _apply_manual_rescore_nomask(
            source_df=source_df,
            row_df=baseline_row_df,
            threshold_summary_df=baseline_thr,
            args=args,
            rules=["output"],
            input_norm=self._input_norm(),
            output_norm=None,
            embedding_meta=None,
        )
        pd.testing.assert_frame_equal(baseline_row_df, rescored_row_df)
        pd.testing.assert_frame_equal(baseline_thr, rescored_thr)
        self.assertFalse(bool(stats["enabled"]))

    def test_manual_seed_pass_rescore_updates_only_affected_group(self) -> None:
        source_df = self._source_df()
        row_df = self._row_df()
        base_args = self._args()
        baseline_row_df, baseline_thr = apply_hybrid_thresholds_nomask(
            source_df=source_df,
            row_df=row_df,
            rules=["output"],
            args=base_args,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            override_path = Path(tmpdir) / "override.csv"
            pd.DataFrame(
                [
                    {"row_id": "r0", "review_label": "seed_pass"},
                    {"row_id": "r7", "review_label": "final_fail"},
                ]
            ).to_csv(override_path, index=False)
            args = self._args(str(override_path))
            rescored_row_df, rescored_thr, stats = _apply_manual_rescore_nomask(
                source_df=source_df,
                row_df=baseline_row_df,
                threshold_summary_df=baseline_thr,
                args=args,
                rules=["output"],
                input_norm=self._input_norm(),
                output_norm=None,
                embedding_meta=None,
            )

        self.assertEqual(int(stats["selected_k"]), 1)
        self.assertEqual(int(stats["affected_groups"]), 1)
        self.assertEqual(int(stats["propagated_rows"]), 1)
        self.assertTrue(bool(rescored_row_df.loc[rescored_row_df["row_id"] == "r1", "manual_propagated_pass"].iloc[0]))
        self.assertFalse(bool(rescored_row_df.loc[rescored_row_df["row_id"] == "r2", "final_pass_nomask"].iloc[0]))
        self.assertFalse(bool(rescored_row_df.loc[rescored_row_df["row_id"] == "r7", "final_pass_nomask"].iloc[0]))
        self.assertEqual(
            str(rescored_row_df.loc[rescored_row_df["row_id"] == "r7", "final_state_nomask"].iloc[0]),
            "fail",
        )

        unaffected_cols = [
            "distribution_pass_nomask",
            "distribution_state_nomask",
            "final_pass_nomask",
            "final_state_nomask",
        ]
        baseline_unaffected = baseline_row_df.loc[baseline_row_df["distribution_group_id"] == "g0001", unaffected_cols].reset_index(drop=True)
        rescored_unaffected = rescored_row_df.loc[rescored_row_df["distribution_group_id"] == "g0001", unaffected_cols].reset_index(drop=True)
        baseline_unaffected.loc[baseline_unaffected.index[1], "distribution_state_nomask"] = "fail"
        baseline_unaffected.loc[baseline_unaffected.index[1], "final_state_nomask"] = "fail"
        baseline_unaffected.loc[baseline_unaffected.index[1], "distribution_pass_nomask"] = False
        baseline_unaffected.loc[baseline_unaffected.index[1], "final_pass_nomask"] = False
        pd.testing.assert_frame_equal(baseline_unaffected, rescored_unaffected)

        self.assertGreaterEqual(float(stats["new_score_post"]), float(stats["new_score_pre"]))
        self.assertFalse(rescored_thr.empty)

    def test_raw_export_contains_review_model_final_and_score_columns(self) -> None:
        source_df = self._source_df()
        row_df = self._row_df()
        args = self._args()
        baseline_row_df, baseline_thr = apply_hybrid_thresholds_nomask(
            source_df=source_df,
            row_df=row_df,
            rules=["output"],
            args=args,
        )
        artifacts = compute_bundle_scores(
            row_df=baseline_row_df,
            threshold_summary_df=baseline_thr,
            score_runtime=SCORE_RUNTIME,
        )
        exported = build_raw_data_export(
            source_df=source_df,
            row_df=baseline_row_df,
            score_summary_df=artifacts.summary_df,
            score_detail_df=artifacts.detail_df,
            args=args,
            pre_manual_row_df=baseline_row_df,
        )
        for col in [
            "review_eval",
            "model_state",
            "model_pass",
            "final_state",
            "final_eval",
            "score_cov_new",
            "score_out_new",
            "score_out_rate_precise",
        ]:
            self.assertIn(col, exported.columns)
        self.assertEqual(str(exported.loc[0, "review_eval"]), "pass")
        self.assertIn(str(exported.loc[0, "model_state"]), {"pass", "warn", "fail", "hard_fail", "na"})


if __name__ == "__main__":
    unittest.main()
