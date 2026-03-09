from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class BundleView:
    bundle: str
    summary: pd.DataFrame
    detail: pd.DataFrame
    payload: dict[str, Any]


def slice_bundle(summary_df: pd.DataFrame, detail_df: pd.DataFrame, payload: dict[str, Any], bundle: str) -> BundleView:
    b = str(bundle)
    s = summary_df[summary_df.get("bundle", pd.Series([], dtype=object)).astype(str) == b].copy() if not summary_df.empty else pd.DataFrame()
    d = detail_df[detail_df.get("bundle", pd.Series([], dtype=object)).astype(str) == b].copy() if not detail_df.empty else pd.DataFrame()
    return BundleView(bundle=b, summary=s, detail=d, payload=dict(payload or {}))
