"""API-first entrypoint for final_metric_refactor."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from final_metric_refactor.config import FinalMetricConfig
from final_metric_refactor.config.data_paths import default_ambiguous_csv
from final_metric_refactor.run import RunArtifacts, run


def run_default() -> RunArtifacts:
    """Run final_metric with default configuration and current timestamp."""
    run_tag = datetime.now().strftime("runtime_%Y%m%d_%H%M%S")
    config = FinalMetricConfig(
        run_tag=run_tag,
        source_csv=Path(default_ambiguous_csv()),
    )
    return run(config)

