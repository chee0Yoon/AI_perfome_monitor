#!/usr/bin/env python3
"""Tests for dashboard renderer payload safety and warn panel wiring."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from urllib.parse import quote

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
FINAL_DIR = THIS_DIR.parent
if str(FINAL_DIR) not in sys.path:
    sys.path.insert(0, str(FINAL_DIR))

from score.score_dashboard import render_bundle_score_dashboard  # noqa: E402


class ScoreDashboardRenderTest(unittest.TestCase):
    def test_warn_payload_script_safe_escape(self) -> None:
        summary_df = pd.DataFrame(
            [
                {
                    "bundle": "OUT",
                    "bundle_score": 3.5,
                    "bundle_score_bucket": 4,
                    "New_score": 3.5,
                    "bundle_confidence": 5,
                    "key_risk": 0.3,
                    "key_risk_raw": 0.3,
                    "key_risk_note": "test",
                }
            ]
        )
        detail_df = pd.DataFrame(
            [
                {
                    "bundle": "OUT",
                    "submetric": "OUT_rate",
                    "subscore": 3,
                    "subscore_precise": 3.0,
                    "order": 0,
                }
            ]
        )
        attack = "</script><script>alert(1)</script>"
        payload = {
            "warn_inspect": {
                "version": "v1",
                "selection": {"top_k": 20, "multi_fail_cut": 3, "preview_chars": 180},
                "summary": {"n_rows": 1, "n_candidates": 1, "n_hard": 1, "n_multi": 0, "n_selected": 1},
                "detail_leaf_summary": {
                    "enabled": True,
                    "n_detail_rows": 1,
                    "n_rows_with_paths": 1,
                    "top_leaves": [
                        {"leaf_path": "feedback", "affected_rows": 1, "affected_share": 1.0, "avg_fail_leaf_count": 2.0}
                    ],
                    "issue_summary": "상세 하락이 'feedback' leaf에 집중되어 있습니다.",
                },
                "rows": [
                    {
                        "rank": 1,
                        "row_index": 0,
                        "row_id": "x1",
                        "tags": ["HARD"],
                        "rank_score": 0.95,
                        "row_cont": 0.90,
                        "state_severity": 1.0,
                        "hard_count": 1,
                        "fail_count": 1,
                        "hard_rules": ["output"],
                        "fail_rules": ["output"],
                        "dominant_bundle": "OUT",
                        "dominant_rule": "output",
                        "bundle_risk_contrib": {"OUT": 0.9, "RID": 0.1, "DIAG": 0.1, "SEM": 0.0},
                        "detail_fail_any": True,
                        "detail_fail_leaf_count": 2,
                        "detail_eval_leaf_count": 3,
                        "detail_failed_leaf_paths": "feedback|answer",
                        "detail_primary_leaf": "feedback",
                        "source_input": "safe input",
                        "source_output": attack,
                    }
                ],
            }
        }

        with tempfile.TemporaryDirectory() as td:
            out_html = Path(td) / "dashboard.html"
            render_bundle_score_dashboard(
                output_html=out_html,
                summary_df=summary_df,
                detail_df=detail_df,
                payload=payload,
                diagnostics_html_path=None,
            )
            html_text = out_html.read_text(encoding="utf-8")

        self.assertIn("warn_inspect_table", html_text)
        self.assertIn("leaf_gate_tables", html_text)
        self.assertIn("leaf_issue_feedback", html_text)
        self.assertIn("문제 leaf path", html_text)
        self.assertNotIn(attack, html_text)
        self.assertIn("\\u003c/script\\u003e\\u003cscript\\u003ealert(1)\\u003c/script\\u003e", html_text)

    def test_dashboard_diagnostics_external_uri(self) -> None:
        summary_df = pd.DataFrame(
            [
                {
                    "bundle": "OUT",
                    "bundle_score": 3.5,
                    "bundle_score_bucket": 4,
                    "New_score": 3.5,
                    "bundle_confidence": 5,
                    "key_risk": 0.3,
                    "key_risk_raw": 0.3,
                    "key_risk_note": "test",
                }
            ]
        )
        detail_df = pd.DataFrame(
            [
                {
                    "bundle": "OUT",
                    "submetric": "OUT_rate",
                    "subscore": 3,
                    "subscore_precise": 3.0,
                    "order": 0,
                }
            ]
        )
        with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2:
            out_html = Path(td1) / "dashboard.html"
            diag_html = Path(td2) / "diag test.html"
            diag_html.write_text("<html></html>", encoding="utf-8")

            render_bundle_score_dashboard(
                output_html=out_html,
                summary_df=summary_df,
                detail_df=detail_df,
                payload={},
                diagnostics_html_path=diag_html,
            )
            html_text = out_html.read_text(encoding="utf-8")

            self.assertIn("file://", html_text)
            self.assertIn(quote("diag test.html"), html_text)


if __name__ == "__main__":
    unittest.main()
