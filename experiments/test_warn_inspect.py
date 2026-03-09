#!/usr/bin/env python3
"""Unit tests for warn_inspect module."""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
FINAL_DIR = THIS_DIR.parent
ROOT_DIR = FINAL_DIR.parent
for p in (ROOT_DIR, FINAL_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from final_metric_refactor.config import SCORE_RUNTIME  # noqa: E402
from final_metric_refactor.scoring.warn_inspect import compute_warn_inspect  # noqa: E402


class WarnInspectTest(unittest.TestCase):
    def _sample_row_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "row_id": ["a", "dup", "dup", "c", "d", "e"],
                "hard_gate_pass": [1, 1, 1, 1, 1, 1],
                "source_input": ["in_a", "in_b1", "in_b2", "in_c", "in_d", "in_e"],
                "source_output": ["out_a", "out_b1", "out_b2", "out_c", "out_d", "out_e"],
                "output_signal_nomask": [0.2, 0.7, 1.8, 0.2, 0.4, 1.1],
                "diff_residual_signal_nomask": [0.3, 0.8, 1.6, 0.2, 0.3, 0.7],
                "delta_ridge_ens_signal_nomask": [0.3, 0.7, 1.4, 0.2, 0.3, 0.8],
                "discourse_instability_signal_nomask": [0.2, 0.3, 1.1, 0.2, 0.2, 1.0],
                "contradiction_signal_nomask": [0.2, 0.3, 1.0, 0.2, 0.2, 1.2],
                "discourse_instability_available_nomask": [1, 1, 1, 1, 1, 1],
                "contradiction_available_nomask": [1, 1, 1, 1, 1, 1],
                "output_state_nomask": ["hard_fail", "fail", "hard_fail", "pass", "fail", "pass"],
                "diff_residual_state_nomask": ["pass", "fail", "fail", "pass", "fail", "pass"],
                "delta_ridge_ens_state_nomask": ["pass", "fail", "fail", "pass", "pass", "pass"],
                "discourse_instability_state_nomask": ["pass", "pass", "pass", "pass", "pass", "pass"],
                "contradiction_state_nomask": ["pass", "pass", "pass", "pass", "pass", "hard_fail"],
            }
        )

    def test_warn_inspect_candidate_and_tags(self) -> None:
        runtime = replace(
            SCORE_RUNTIME,
            new_score_risk_case_top_k=20,
            new_score_risk_case_multi_fail_cut=3,
            new_score_risk_case_mix_cont=0.70,
            new_score_risk_case_mix_state=0.30,
        )
        row_df = self._sample_row_df()
        artifacts = compute_warn_inspect(
            row_df=row_df,
            threshold_summary_df=pd.DataFrame(),
            score_runtime=runtime,
            bundle_payload={},
        )

        rows = artifacts.payload.get("rows", [])
        self.assertGreater(len(rows), 0)
        row_ids = [str(r["row_id"]) for r in rows]
        self.assertEqual(len(row_ids), len(set(row_ids)))  # dedup by row_id
        self.assertLessEqual(len(rows), 20)

        prev_score = None
        for r in rows:
            tags = list(r.get("tags", []))
            self.assertTrue(("HARD" in tags) or ("MULTI" in tags))
            if "HARD" in tags:
                self.assertGreater(int(r.get("hard_count", 0)), 0)
            if "MULTI" in tags:
                self.assertGreaterEqual(int(r.get("fail_count", 0)), 3)
            self.assertIn("detail_fail_any", r)
            self.assertIn("detail_fail_leaf_count", r)
            self.assertIn("detail_failed_leaf_paths", r)
            score = float(r.get("rank_score", 0.0))
            if prev_score is not None:
                self.assertLessEqual(score, prev_score + 1e-12)
            prev_score = score

        summary = artifacts.summary
        self.assertGreaterEqual(int(summary.get("n_candidates", 0)), len(rows))
        self.assertGreaterEqual(int(summary.get("n_hard", 0)), 1)
        self.assertGreaterEqual(int(summary.get("n_multi", 0)), 1)
        self.assertIn("n_detail_with_paths", artifacts.payload.get("summary", {}))
        self.assertIn("detail_leaf_summary", artifacts.payload)

    def test_warn_inspect_top_k_enforced(self) -> None:
        runtime = replace(
            SCORE_RUNTIME,
            new_score_risk_case_top_k=2,
            new_score_risk_case_multi_fail_cut=3,
        )
        row_df = self._sample_row_df()
        artifacts = compute_warn_inspect(
            row_df=row_df,
            threshold_summary_df=pd.DataFrame(),
            score_runtime=runtime,
            bundle_payload={},
        )
        rows = artifacts.payload.get("rows", [])
        self.assertEqual(len(rows), 2)
        self.assertEqual(int(artifacts.payload.get("selection", {}).get("top_k", 0)), 2)

    def test_warn_inspect_hard_weight_dominates_fail_count(self) -> None:
        runtime = replace(
            SCORE_RUNTIME,
            new_score_risk_case_top_k=20,
            new_score_risk_case_multi_fail_cut=1,
            new_score_risk_case_mix_cont=0.0,
            new_score_risk_case_mix_state=1.0,
            new_score_risk_case_state_hard_weight=8.0,
            new_score_risk_case_state_fail_weight=1.0,
        )
        row_df = pd.DataFrame(
            {
                "row_id": ["hard_one", "fail_many"],
                "hard_gate_pass": [1, 1],
                "source_input": ["in_hard", "in_fail"],
                "source_output": ["out_hard", "out_fail"],
                "output_signal_nomask": [0.1, 0.1],
                "diff_residual_signal_nomask": [0.1, 0.1],
                "delta_ridge_ens_signal_nomask": [0.1, 0.1],
                "discourse_instability_signal_nomask": [0.1, 0.1],
                "contradiction_signal_nomask": [0.1, 0.1],
                "output_state_nomask": ["hard_fail", "fail"],
                "diff_residual_state_nomask": ["pass", "fail"],
                "delta_ridge_ens_state_nomask": ["pass", "fail"],
                "discourse_instability_state_nomask": ["pass", "fail"],
                "contradiction_state_nomask": ["pass", "pass"],
            }
        )
        artifacts = compute_warn_inspect(
            row_df=row_df,
            threshold_summary_df=pd.DataFrame(),
            score_runtime=runtime,
            bundle_payload={},
        )
        rows = artifacts.payload.get("rows", [])
        by_id = {str(r["row_id"]): r for r in rows}
        self.assertIn("hard_one", by_id)
        self.assertIn("fail_many", by_id)
        hard_score = float(by_id["hard_one"]["state_severity"])
        fail_score = float(by_id["fail_many"]["state_severity"])
        self.assertGreater(hard_score, fail_score)


if __name__ == "__main__":
    unittest.main()
