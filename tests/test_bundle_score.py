#!/usr/bin/env python3
"""Unit tests for bundle score zero-threshold density fallback."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
FINAL_DIR = THIS_DIR.parent
ROOT_DIR = FINAL_DIR.parent
for p in (ROOT_DIR, FINAL_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from final_metric_refactor.config import SCORE_RUNTIME  # noqa: E402
from final_metric_refactor.scoring.bundle.orchestrator import compute_bundle_scores  # noqa: E402


class BundleScoreZeroThresholdFallbackTest(unittest.TestCase):
    def _base_row_df(self, n: int = 24) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "row_id": [f"r{i}" for i in range(n)],
                "hard_gate_pass": [1] * n,
                "ifeval_pass": [1] * n,
                "schema_pass": [1] * n,
                "textlen_pass": [1] * n,
                "source_input": [f"in{i}" for i in range(n)],
                "source_output": [f"out{i}" for i in range(n)],
                "output_signal_nomask": [0.2] * n,
                "direction_signal_nomask": [0.1] * n,
                "length_signal_nomask": [0.1] * n,
                "diff_residual_signal_nomask": [0.2] * n,
                "delta_ridge_ens_signal_nomask": [0.2] * n,
                "discourse_instability_signal_nomask": [0.2] * n,
                "contradiction_signal_nomask": [0.2] * n,
                "discourse_instability_available_nomask": [1] * n,
                "contradiction_available_nomask": [1] * n,
                "output_state_nomask": ["pass"] * n,
                "diff_residual_state_nomask": ["pass"] * n,
                "delta_ridge_ens_state_nomask": ["pass"] * n,
                "discourse_instability_state_nomask": ["pass"] * n,
                "contradiction_state_nomask": ["pass"] * n,
            }
        )

    def _threshold_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "rule": [
                    "output",
                    "direction",
                    "length",
                    "diff_residual",
                    "delta_ridge_ens",
                    "discourse_instability",
                    "contradiction",
                ],
                "selected_method": ["tail_start"] * 7,
                "threshold_source": ["tail_start"] * 7,
                "support_rows": [64] * 7,
                "tail_start_threshold": [1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0],
                "exceptional_out_threshold": [2.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
            }
        )

    def test_rid_dense_zero_hard_threshold_gives_perfect_score(self) -> None:
        row_df = self._base_row_df()
        row_df["diff_residual_signal_nomask"] = [0.0] * len(row_df)
        thr = self._threshold_df()
        thr.loc[thr["rule"] == "diff_residual", "tail_start_threshold"] = 0.0
        thr.loc[thr["rule"] == "diff_residual", "exceptional_out_threshold"] = 0.0

        artifacts = compute_bundle_scores(
            row_df=row_df,
            threshold_summary_df=thr,
            score_runtime=SCORE_RUNTIME,
            input_norm=None,
            output_norm=None,
            embedding_meta={"source": "unit_test_rid_zero_dense"},
        )

        meta = artifacts.payload["raw"]["threshold_explain"]["zero_threshold_density_adjustments"]["diff_residual"]
        self.assertTrue(bool(meta["applied"]))
        self.assertEqual(str(meta["reason"]), "hard_fail_threshold_zero_dense")
        self.assertAlmostEqual(float(meta["zero_threshold_density"]["density_score"]), 5.0, places=8)

        rid_triplet = artifacts.payload["raw"]["triplet_aggregation"]["RID"]
        self.assertAlmostEqual(float(rid_triplet["subscores_precise"][1]), 5.0, places=8)
        self.assertAlmostEqual(float(rid_triplet["subscores_precise"][2]), 5.0, places=8)

        rid_score = float(artifacts.summary_df.loc[artifacts.summary_df["bundle"] == "RID", "bundle_score"].iloc[0])
        self.assertAlmostEqual(rid_score, 5.0, places=8)

    def test_sem_zero_fail_threshold_uses_gap_based_four_to_five_score(self) -> None:
        row_df = self._base_row_df()
        row_df["discourse_instability_signal_nomask"] = [0.0] * len(row_df)
        thr = self._threshold_df()
        thr.loc[thr["rule"] == "discourse_instability", "tail_start_threshold"] = 0.0
        thr.loc[thr["rule"] == "discourse_instability", "exceptional_out_threshold"] = 3.0

        artifacts = compute_bundle_scores(
            row_df=row_df,
            threshold_summary_df=thr,
            score_runtime=SCORE_RUNTIME,
            input_norm=None,
            output_norm=None,
            embedding_meta={"source": "unit_test_sem_zero_fail_gap"},
        )

        meta = artifacts.payload["raw"]["threshold_explain"]["zero_threshold_density_adjustments"]["discourse_instability"]
        self.assertTrue(bool(meta["applied"]))
        self.assertEqual(str(meta["reason"]), "fail_threshold_zero_dense")
        self.assertGreater(float(meta["zero_threshold_density"]["density_score"]), 4.0)
        self.assertLess(float(meta["zero_threshold_density"]["density_score"]), 5.0)

        sem_score = float(artifacts.summary_df.loc[artifacts.summary_df["bundle"] == "SEM", "bundle_score"].iloc[0])
        self.assertGreater(sem_score, 4.0)
        self.assertLessEqual(sem_score, 5.0)

    def test_missing_thresholds_still_uses_continuous_distribution_score(self) -> None:
        row_df = self._base_row_df()
        row_df["diff_residual_signal_nomask"] = [0.1] * 18 + [3.5, 3.8, 4.0, 4.2, 4.5, 4.8]

        artifacts = compute_bundle_scores(
            row_df=row_df,
            threshold_summary_df=pd.DataFrame(),
            score_runtime=SCORE_RUNTIME,
            input_norm=None,
            output_norm=None,
            embedding_meta={"source": "unit_test_missing_thresholds_continuous_score"},
        )

        meta = artifacts.payload["raw"]["threshold_explain"]["zero_threshold_density_adjustments"]["diff_residual"]
        self.assertFalse(bool(meta["applied"]))

        rid_score = float(artifacts.summary_df.loc[artifacts.summary_df["bundle"] == "RID", "bundle_score"].iloc[0])
        self.assertNotAlmostEqual(rid_score, 3.0, places=8)

    def test_missing_distribution_signal_does_not_trigger_dense_zero_fallback(self) -> None:
        row_df = self._base_row_df()
        row_df["diff_residual_signal_nomask"] = [float("nan")] * len(row_df)

        artifacts = compute_bundle_scores(
            row_df=row_df,
            threshold_summary_df=pd.DataFrame(),
            score_runtime=SCORE_RUNTIME,
            input_norm=None,
            output_norm=None,
            embedding_meta={"source": "unit_test_missing_distribution_signal"},
        )

        meta = artifacts.payload["raw"]["threshold_explain"]["zero_threshold_density_adjustments"]["diff_residual"]
        self.assertFalse(bool(meta["applied"]))

        rid_score = float(artifacts.summary_df.loc[artifacts.summary_df["bundle"] == "RID", "bundle_score"].iloc[0])
        self.assertAlmostEqual(rid_score, 3.0, places=8)

    def test_sem_not_applicable_exports_null_bundle_score(self) -> None:
        row_df = self._base_row_df()
        row_df["discourse_instability_signal_nomask"] = [0.0] * len(row_df)
        row_df["contradiction_signal_nomask"] = [0.0] * len(row_df)
        row_df["discourse_instability_state_nomask"] = ["pass"] * len(row_df)
        row_df["contradiction_state_nomask"] = ["pass"] * len(row_df)

        artifacts = compute_bundle_scores(
            row_df=row_df,
            threshold_summary_df=self._threshold_df(),
            score_runtime=SCORE_RUNTIME,
            input_norm=None,
            output_norm=None,
            embedding_meta={"source": "unit_test_sem_null_when_not_applicable"},
        )

        sem_row = artifacts.summary_df.loc[artifacts.summary_df["bundle"] == "SEM"].iloc[0]
        self.assertTrue(pd.isna(sem_row["bundle_score"]))
        self.assertTrue(pd.isna(sem_row["New_score"]))
        self.assertTrue(pd.isna(sem_row["bundle_score_bucket"]))
        self.assertEqual(str(sem_row["key_risk_note"]), "SEM score unavailable")

        sem_detail = artifacts.detail_df.loc[artifacts.detail_df["bundle"] == "SEM"]
        self.assertTrue(sem_detail["subscore"].isna().all())
        self.assertTrue(sem_detail["subscore_precise"].isna().all())

        self.assertIsNone(artifacts.payload["bundle_scores"]["SEM"])
        self.assertIsNone(artifacts.payload["bundle_scores_bucket"]["SEM"])
        self.assertIsNone(artifacts.payload["new_scores"]["SEM"])


if __name__ == "__main__":
    unittest.main()
