"""
Text Length Gate - Wrapper for text length validation of JSON outputs.

Provides a hard gate to filter outputs that don't meet minimum text length
requirements for JSON leaf paths.
"""

from dataclasses import dataclass

import pandas as pd

from final_metric_refactor.hard_gate.text_length import TextLengthMetric
from final_metric_refactor.shared.preprocessor import safe_bool_series


@dataclass
class TextLengthGateResult:
    """Result from the text length gate."""

    num_outputs: int
    num_passed: int
    pass_rate: float
    per_output_results: list[dict]


class TextLengthGate:
    """
    Hard gate for text length validation.

    Validates that JSON outputs have minimum text length for all leaf paths,
    based on percentiles computed from a reference distribution.

    Usage:
        gate = TextLengthGate(min_ratio=0.5, min_support_ratio=0.1)
        # Compute thresholds from reference outputs
        gate.compute_thresholds(reference_outputs)
        # Validate test outputs
        result = gate.validate(test_outputs)
    """

    def __init__(
        self,
        min_ratio: float = 0.5,
        min_support_ratio: float = 0.1,
    ):
        """Initialize the text length gate.

        Args:
            min_ratio: Minimum acceptable length as a ratio of reference median.
            min_support_ratio: Minimum fraction of samples that must have a field.
        """
        self.metric = TextLengthMetric(
            min_ratio=min_ratio,
            min_support_ratio=min_support_ratio,
        )

    def compute_thresholds(self, outputs: list) -> dict[str, float]:
        """Compute length thresholds from reference outputs.

        Args:
            outputs: List of reference outputs to compute thresholds from.

        Returns:
            Dictionary of per-path minimum length thresholds.
        """
        return self.metric.compute_thresholds(outputs)

    def validate(self, outputs: list) -> tuple[pd.Series, pd.Series]:
        """Validate outputs against computed length thresholds.

        Args:
            outputs: List of output strings to validate.

        Returns:
            Tuple of (passed: pd.Series[bool], failed_reasons: pd.Series[list[str]]).
            - passed: Boolean series, True if output passes all checks
            - failed_reasons: Series of error lists, each containing path/length failures
        """
        passed, failed_reasons = self.metric.check(outputs)

        passed_series = pd.Series(passed, dtype=bool)
        failed_series = pd.Series(failed_reasons, dtype=object)

        return passed_series, failed_series

    def validate_with_dataframe(
        self,
        df: pd.DataFrame,
        output_col: str,
    ) -> tuple[pd.Series, pd.Series]:
        """Validate outputs in a DataFrame column.

        Args:
            df: DataFrame containing outputs.
            output_col: Column name containing output values.

        Returns:
            Tuple of (passed, failed_reasons) series to be merged into df.
        """
        outputs = df[output_col].fillna("").astype(str).tolist()
        return self.validate(outputs)
