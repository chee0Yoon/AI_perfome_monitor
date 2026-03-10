"""Manual review helpers for human-pass rescore flow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from final_metric_refactor.shared.geometry import knn_self, sanitize_matrix

ALLOWED_MANUAL_REVIEW_LABELS = frozenset({"seed_pass", "final_pass", "final_fail"})
MANUAL_REVIEW_K_CANDIDATES = (0, 1, 3, 5)


@dataclass(frozen=True)
class ManualOverrideTable:
    table: pd.DataFrame
    seed_pass_ids: set[str]
    final_pass_ids: set[str]
    final_fail_ids: set[str]


def _normalize_row_id(value: Any) -> str:
    return str(value).strip()


def load_manual_override_csv(csv_path: Path | str, valid_row_ids: list[str] | np.ndarray | pd.Series) -> ManualOverrideTable:
    path = Path(csv_path).resolve()
    df = pd.read_csv(path)
    required = {"row_id", "review_label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"manual override CSV is missing required columns: {missing}")
    if df.empty:
        raise ValueError("manual override CSV is empty")

    out = df.copy()
    out["row_id"] = out["row_id"].map(_normalize_row_id)
    out["review_label"] = out["review_label"].fillna("").astype(str).str.strip().str.lower()

    bad_labels = sorted(set(out["review_label"]) - set(ALLOWED_MANUAL_REVIEW_LABELS))
    if bad_labels:
        raise ValueError(f"manual override CSV has invalid review_label values: {bad_labels}")

    dup_ids = out["row_id"][out["row_id"].duplicated()].unique().tolist()
    if dup_ids:
        raise ValueError(f"manual override CSV contains duplicated row_id values: {dup_ids[:10]}")

    valid = {str(v).strip() for v in list(valid_row_ids)}
    unknown_ids = sorted(set(out["row_id"]) - valid)
    if unknown_ids:
        raise ValueError(f"manual override CSV contains unknown row_id values: {unknown_ids[:10]}")

    for col in ("reviewer", "note"):
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str)

    return ManualOverrideTable(
        table=out[["row_id", "review_label", "reviewer", "note"]].copy(),
        seed_pass_ids=set(out.loc[out["review_label"].eq("seed_pass"), "row_id"].tolist()),
        final_pass_ids=set(out.loc[out["review_label"].eq("final_pass"), "row_id"].tolist()),
        final_fail_ids=set(out.loc[out["review_label"].eq("final_fail"), "row_id"].tolist()),
    )


def build_group_local_propagation_mask(
    *,
    input_norm: np.ndarray,
    group_ids: list[str] | np.ndarray | pd.Series,
    anchor_mask: np.ndarray,
    k: int,
) -> np.ndarray:
    x = sanitize_matrix(np.asarray(input_norm, dtype=float))
    gids = np.asarray(pd.Series(group_ids).fillna("g0000").astype(str).tolist(), dtype=object)
    anchors = np.asarray(anchor_mask, dtype=bool)
    if x.ndim != 2:
        raise ValueError("input_norm must be a 2D array")
    if len(x) != len(gids) or len(x) != len(anchors):
        raise ValueError("input_norm, group_ids, and anchor_mask must have the same length")
    if int(k) <= 0 or not np.any(anchors):
        return np.zeros(len(gids), dtype=bool)

    propagated = np.zeros(len(gids), dtype=bool)
    for gid in pd.unique(gids):
        idx = np.where(gids == gid)[0]
        if len(idx) <= 1:
            continue
        local_anchor_pos = np.where(anchors[idx])[0]
        if len(local_anchor_pos) == 0:
            continue
        local_x = x[idx]
        _, nbr = knn_self(local_x, n_neighbors=min(int(k), len(idx) - 1), metric="cosine")
        if nbr.size == 0:
            continue
        for anchor_pos in local_anchor_pos.tolist():
            neighbor_pos = nbr[int(anchor_pos)].tolist()
            for local_pos in neighbor_pos[: int(k)]:
                propagated[int(idx[int(local_pos)])] = True
    propagated &= ~anchors
    return propagated


def compute_manual_pass_precision(
    *,
    selected_mask: np.ndarray,
    y_bad: np.ndarray,
    label_known: np.ndarray,
) -> float:
    selected = np.asarray(selected_mask, dtype=bool)
    bad = np.asarray(y_bad, dtype=bool)
    known = np.asarray(label_known, dtype=bool)
    use = selected & known
    if not np.any(use):
        return float("nan")
    return float(np.mean(~bad[use]))


def choose_manual_review_k(candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidate_rows:
        raise ValueError("candidate_rows must not be empty")
    rows = sorted(candidate_rows, key=lambda rec: int(rec.get("k", 0)))
    baseline = next((rec for rec in rows if int(rec.get("k", 0)) == 0), rows[0])
    baseline_precision = float(baseline.get("precision", np.nan))
    if not np.isfinite(baseline_precision):
        return baseline

    qualified = [
        rec
        for rec in rows
        if int(rec.get("k", 0)) > 0
        and np.isfinite(float(rec.get("precision", np.nan)))
        and float(rec.get("precision", np.nan)) + 1e-12 >= baseline_precision
    ]
    return qualified[0] if qualified else baseline


def apply_manual_final_overrides(
    row_df: pd.DataFrame,
    *,
    row_id_col: str,
    final_pass_ids: set[str],
    final_fail_ids: set[str],
) -> pd.DataFrame:
    out = row_df.copy()
    row_ids = out[row_id_col].fillna("").astype(str).map(_normalize_row_id)
    pass_mask = row_ids.isin(final_pass_ids).to_numpy(dtype=bool)
    fail_mask = row_ids.isin(final_fail_ids).to_numpy(dtype=bool)

    if np.any(pass_mask):
        out.loc[pass_mask, "distribution_pass_nomask"] = True
        out.loc[pass_mask, "distribution_warn_nomask"] = False
        out.loc[pass_mask, "distribution_fail_nomask"] = False
        out.loc[pass_mask, "distribution_hard_fail_nomask"] = False
        out.loc[pass_mask, "distribution_state_nomask"] = "pass"
        out.loc[pass_mask, "final_pass_nomask"] = True
        out.loc[pass_mask, "final_state_nomask"] = "pass"

    if np.any(fail_mask):
        out.loc[fail_mask, "distribution_pass_nomask"] = False
        out.loc[fail_mask, "distribution_warn_nomask"] = False
        out.loc[fail_mask, "distribution_fail_nomask"] = True
        out.loc[fail_mask, "distribution_hard_fail_nomask"] = False
        out.loc[fail_mask, "distribution_state_nomask"] = "fail"
        out.loc[fail_mask, "final_pass_nomask"] = False
        out.loc[fail_mask, "final_state_nomask"] = "fail"

    return out
