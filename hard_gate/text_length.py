"""
Text Length Metric - Validates text length constraints for JSON outputs.

Per-JSON-leaf-path text length percentile-based gate that ensures output
text fields meet minimum length thresholds computed from reference distribution.
"""

from typing import Any

import numpy as np

from final_metric_refactor.shared.preprocessor import flatten_json_leaves, safe_json_load


class TextLengthMetric:
    """Text length metric for JSON outputs.

    Computes per-JSON-leaf-path minimum length thresholds based on the reference
    distribution, then validates that individual outputs meet these thresholds.
    """

    def __init__(
        self,
        min_ratio: float = 0.5,
        min_support_ratio: float = 0.1,
    ):
        """Initialize the text length metric.

        Args:
            min_ratio: Minimum acceptable text length as a ratio of reference median.
                      E.g., 0.5 = text must be at least 50% of reference median length.
            min_support_ratio: Minimum fraction of samples that must have a field
                             populated to include it in thresholds (default 0.1 = 10%).
        """
        self.min_ratio = min_ratio
        self.min_support_ratio = min_support_ratio
        self.thresholds: dict[str, float] = {}

    def _extract_leaves(self, value: Any) -> dict[str, str]:
        """Extract string leaves from a JSON value.

        Args:
            value: Raw value (JSON string, dict, or other).

        Returns:
            Dictionary mapping leaf paths to string values.
        """
        parsed = safe_json_load(value)
        if parsed is None:
            return {"_raw_text": str(value) if value is not None else ""}
        if isinstance(parsed, dict):
            leaves = flatten_json_leaves(parsed)
            return {k: v for k, v in leaves.items() if isinstance(v, str)} or {"_raw_text": str(value)}
        return {"_raw_text": str(value)}

    def compute_thresholds(self, outputs: list[Any]) -> dict[str, float]:
        """Compute per-path minimum length thresholds from reference outputs.

        Args:
            outputs: List of output values to compute thresholds from.

        Returns:
            Dictionary mapping leaf paths to minimum acceptable length thresholds.
        """
        n = len(outputs)
        rows = [self._extract_leaves(v) for v in outputs]
        all_paths = sorted({p for row in rows for p in row.keys()})

        min_support = max(3, int(np.ceil(n * self.min_support_ratio)))
        thresholds: dict[str, float] = {}

        for path in all_paths:
            vals = [row.get(path, "") for row in rows]
            supports = [len(v.strip()) > 0 for v in vals]
            if sum(supports) < min_support:
                continue

            lengths = np.array([len(v) for v in vals if len(v.strip()) > 0], dtype=float)
            if lengths.size == 0:
                continue

            thresholds[path] = float(np.median(lengths) * self.min_ratio)

        self.thresholds = thresholds
        return thresholds

    def check(self, outputs: list[Any]) -> tuple[list[bool], list[list[str]]]:
        """Check if outputs meet length thresholds.

        Args:
            outputs: List of outputs to validate.

        Returns:
            Tuple of (passed, failed_reasons) where:
            - passed: List of bools, True if output passes all checks
            - failed_reasons: List of error lists, each containing path/length failures
        """
        n = len(outputs)
        passed = []
        failed_reasons = []

        rows = [self._extract_leaves(v) for v in outputs]

        for row in rows:
            errors: list[str] = []
            for path, th in self.thresholds.items():
                val = row.get(path, "")
                if len(val.strip()) == 0:
                    continue
                l = len(val)
                if l <= th:
                    errors.append(f"{path}(len={l},th={th:.0f})")

            passed.append(len(errors) == 0)
            failed_reasons.append(errors)

        return passed, failed_reasons
