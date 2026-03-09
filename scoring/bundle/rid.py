from __future__ import annotations

from typing import Any

import pandas as pd

from final_metric_refactor.scoring.bundle.common import BundleView, slice_bundle


def build_rid_bundle_view(
    summary_df: pd.DataFrame,
    detail_df: pd.DataFrame,
    payload: dict[str, Any],
) -> BundleView:
    return slice_bundle(summary_df, detail_df, payload, "RID")
