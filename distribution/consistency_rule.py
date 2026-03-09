"""Consistency Rule Metric - Detects verdict-feedback consistency issues (observe-only)."""

from typing import Any

import numpy as np


class ConsistencyRuleMetric:
    """Detects outputs where verdict and feedback have conflicting sentiment (observe-only).

    This metric is typically used for observation only and does not affect the final pass/fail decision.
    """

    def compute(
        self,
        output_dicts: list[dict[str, Any] | None],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Compute consistency scores (observe-only).

        Args:
            output_dicts: Parsed JSON output dictionaries.

        Returns:
            (scores, available, meta) where:
            - scores: (N,) array of consistency scores (0.0 if consistent, >0 if inconsistent)
            - available: (N,) bool array indicating where both verdict and feedback were found
            - meta: dict with metadata about the computation
        """
        n = len(output_dicts)
        scores = np.zeros(n)
        available = np.zeros(n, dtype=bool)

        # Minimal placeholder implementation
        # Full implementation would parse verdict fields, analyze feedback polarity, etc.
        # For now, mark all as unavailable since this requires complex logic

        return scores, available, {
            "selected_verdict_key": None,
            "selected_evidence_key": None,
            "available_rows": 0,
            "conflict_rows": 0,
            "conflict_rate": 0.0,
        }
