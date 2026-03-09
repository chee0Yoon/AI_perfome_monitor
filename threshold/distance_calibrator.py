#!/usr/bin/env python3
"""
Distance-rule threshold calibration utilities.

This module calibrates quantile thresholds for distance rules using cross-validation:
- output
- direction
- length
- diff_resid
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

EPS = 1e-9

DISTANCE_RULES = ("output", "direction", "length", "diff_resid")
RULE_TO_SIGNAL_PREFIX = {
    "output": "output_signal",
    "direction": "direction_signal",
    "length": "length_signal",
    "diff_resid": "diff_residual_signal",
}


@dataclass(frozen=True)
class CandidateQuantiles:
    quantiles: dict[str, float]


def parse_quantile_range_spec(spec: str) -> list[float]:
    """
    Parse range style quantile spec, e.g. "0.70~0.99(step 0.01)".
    """
    text = str(spec).strip()
    pat = r"^\s*([0-9]*\.?[0-9]+)\s*~\s*([0-9]*\.?[0-9]+)\s*\(\s*step\s*([0-9]*\.?[0-9]+)\s*\)\s*$"
    m = re.match(pat, text)
    if not m:
        raise ValueError(f"Invalid quantile range spec: {spec}")
    start = float(m.group(1))
    end = float(m.group(2))
    step = float(m.group(3))
    if not (0.0 < start < 1.0 and 0.0 < end < 1.0):
        raise ValueError(f"Quantiles must be in (0,1): {spec}")
    if end < start:
        raise ValueError(f"Invalid range; end < start: {spec}")
    if step <= 0:
        raise ValueError(f"Step must be > 0: {spec}")

    vals: list[float] = []
    cur = start
    while cur <= end + 1e-12:
        vals.append(round(float(cur), 6))
        cur += step
    vals = sorted(set(vals))
    return [v for v in vals if 0.0 < v < 1.0]


def parse_quantile_csv_spec(spec: str) -> list[float]:
    vals: list[float] = []
    for tok in str(spec).split(","):
        t = tok.strip()
        if not t:
            continue
        q = float(t)
        if not (0.0 < q < 1.0):
            raise ValueError(f"Quantile must be in (0,1): {t}")
        vals.append(round(q, 6))
    vals = sorted(set(vals))
    if not vals:
        raise ValueError("No quantiles provided in CSV spec.")
    return vals


def parse_csv_tokens(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def validate_calibration_rules(rules: list[str]) -> list[str]:
    out: list[str] = []
    for r in rules:
        key = str(r).strip().lower()
        if key not in DISTANCE_RULES:
            raise ValueError(f"Calibration rules must be subset of {DISTANCE_RULES}. Got: {r}")
        if key not in out:
            out.append(key)
    if not out:
        raise ValueError("No calibration rules provided.")
    if "output" not in out:
        raise ValueError("Calibration rules must include 'output'.")
    return out


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum((~y_true) & y_pred))
    tn = int(np.sum((~y_true) & (~y_pred)))
    fn = int(np.sum(y_true & (~y_pred)))
    precision = float(tp / (tp + fp + EPS))
    recall = float(tp / (tp + fn + EPS))
    f1 = float((2.0 * precision * recall) / (precision + recall + EPS))
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _make_stratified_folds(y: np.ndarray, n_folds: int, seed: int = 42) -> np.ndarray:
    n = len(y)
    n_folds = max(2, min(int(n_folds), n))
    rng = np.random.default_rng(seed)
    fold_ids = np.zeros(n, dtype=int)

    y = np.asarray(y, dtype=bool)
    idx_pos = np.where(y)[0]
    idx_neg = np.where(~y)[0]
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)

    for i, idx in enumerate(idx_pos):
        fold_ids[idx] = i % n_folds
    for i, idx in enumerate(idx_neg):
        fold_ids[idx] = i % n_folds
    return fold_ids


def _safe_quantile(values: np.ndarray, q: float) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.quantile(vals, float(q)))


def fit_thresholds_for_rules(
    df: pd.DataFrame,
    mode: str,
    train_mask: np.ndarray,
    quantiles: dict[str, float],
    rules: list[str],
    group_col: str = "distribution_group_id",
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    train_mask = np.asarray(train_mask, dtype=bool)
    if len(train_mask) != len(df):
        raise ValueError("train_mask length mismatch.")
    if not train_mask.any():
        raise ValueError("train_mask has no True rows.")

    groups = df[group_col].astype(str).to_numpy(dtype=object)
    by_group: dict[str, dict[str, float]] = {}
    global_thresholds: dict[str, float] = {}

    for rule in rules:
        col = f"{RULE_TO_SIGNAL_PREFIX[rule]}_{mode}"
        global_thresholds[rule] = _safe_quantile(df.loc[train_mask, col].to_numpy(dtype=float), quantiles[rule])

    train_groups = pd.unique(groups[train_mask])
    for gid in train_groups:
        gmask = train_mask & (groups == gid)
        by_group[str(gid)] = {}
        for rule in rules:
            col = f"{RULE_TO_SIGNAL_PREFIX[rule]}_{mode}"
            th = _safe_quantile(df.loc[gmask, col].to_numpy(dtype=float), quantiles[rule])
            if not np.isfinite(th):
                th = global_thresholds[rule]
            by_group[str(gid)][rule] = float(th)

    return by_group, global_thresholds


def predict_distance_anomaly(
    df: pd.DataFrame,
    mode: str,
    target_mask: np.ndarray,
    rules: list[str],
    by_group_thresholds: dict[str, dict[str, float]],
    global_thresholds: dict[str, float],
    group_col: str = "distribution_group_id",
) -> np.ndarray:
    target_mask = np.asarray(target_mask, dtype=bool)
    groups = df[group_col].astype(str).to_numpy(dtype=object)
    pred = np.zeros(len(df), dtype=bool)
    target_idx = np.where(target_mask)[0]
    if len(target_idx) == 0:
        return pred

    signal_cache: dict[str, np.ndarray] = {
        rule: df[f"{RULE_TO_SIGNAL_PREFIX[rule]}_{mode}"].to_numpy(dtype=float) for rule in rules
    }
    for i in target_idx:
        gid = str(groups[i])
        th_map = by_group_thresholds.get(gid)
        failed = False
        for rule in rules:
            th = float(global_thresholds[rule]) if th_map is None else float(th_map.get(rule, global_thresholds[rule]))
            score = float(signal_cache[rule][i])
            if not np.isfinite(score):
                continue
            if score > th:
                failed = True
                break
        pred[i] = failed
    return pred


def build_quantile_candidates(
    rules: list[str],
    output_quantiles: list[float],
    other_quantiles: list[float],
) -> list[CandidateQuantiles]:
    rules = validate_calibration_rules(rules)
    others = [r for r in rules if r != "output"]
    out: list[CandidateQuantiles] = []
    if not others:
        for q_out in output_quantiles:
            out.append(CandidateQuantiles(quantiles={"output": float(q_out)}))
        return out

    for q_out in output_quantiles:
        for tpl in itertools.product(other_quantiles, repeat=len(others)):
            qmap = {"output": float(q_out)}
            for r, q in zip(others, tpl):
                qmap[r] = float(q)
            out.append(CandidateQuantiles(quantiles=qmap))
    return out


def select_best_candidate(candidate_df: pd.DataFrame, min_precision: float) -> tuple[pd.Series, bool]:
    if candidate_df.empty:
        raise ValueError("candidate_df is empty.")
    cdf = candidate_df.copy()
    filtered = cdf[cdf["mean_precision"] >= float(min_precision)]
    precision_constraint_met = not filtered.empty
    base = filtered if precision_constraint_met else cdf
    base = base.sort_values(
        by=["mean_f1", "mean_fp", "mean_recall", "mean_precision"],
        ascending=[False, True, False, False],
    ).reset_index(drop=True)
    return base.iloc[0], precision_constraint_met


def calibrate_mode_quantiles(
    df: pd.DataFrame,
    mode: str,
    y_is_bad: np.ndarray,
    eval_mask: np.ndarray,
    rules: list[str],
    output_quantiles: list[float],
    other_quantiles: list[float],
    cv_folds: int,
    min_precision: float,
    group_col: str = "distribution_group_id",
    seed: int = 42,
) -> dict[str, Any]:
    rules = validate_calibration_rules(rules)
    y_is_bad = np.asarray(y_is_bad, dtype=bool)
    eval_mask = np.asarray(eval_mask, dtype=bool)
    if len(y_is_bad) != len(df) or len(eval_mask) != len(df):
        raise ValueError("length mismatch in calibration inputs.")

    idx = np.where(eval_mask)[0]
    if len(idx) < 2:
        raise ValueError(f"Not enough evaluation rows for calibration: mode={mode}, rows={len(idx)}")

    y_eval = y_is_bad[idx]
    n_folds_eff = max(2, min(int(cv_folds), len(idx)))
    fold_ids = _make_stratified_folds(y_eval, n_folds=n_folds_eff, seed=seed)

    candidates = build_quantile_candidates(rules=rules, output_quantiles=output_quantiles, other_quantiles=other_quantiles)
    if not candidates:
        raise ValueError("No quantile candidates generated.")

    candidate_rows: list[dict[str, Any]] = []
    cv_rows: list[dict[str, Any]] = []
    for cid, cand in enumerate(candidates):
        fold_metrics: list[dict[str, float]] = []
        for fold in range(n_folds_eff):
            train_mask = np.zeros(len(df), dtype=bool)
            val_mask = np.zeros(len(df), dtype=bool)
            train_mask[idx[fold_ids != fold]] = True
            val_mask[idx[fold_ids == fold]] = True
            if not train_mask.any() or not val_mask.any():
                continue

            by_group, global_th = fit_thresholds_for_rules(
                df=df,
                mode=mode,
                train_mask=train_mask,
                quantiles=cand.quantiles,
                rules=rules,
                group_col=group_col,
            )
            pred_all = predict_distance_anomaly(
                df=df,
                mode=mode,
                target_mask=val_mask,
                rules=rules,
                by_group_thresholds=by_group,
                global_thresholds=global_th,
                group_col=group_col,
            )
            y_val = y_is_bad[val_mask]
            p_val = pred_all[val_mask]
            m = _binary_metrics(y_val, p_val)
            fold_metrics.append(m)
            cv_rows.append(
                {
                    "mode": mode,
                    "candidate_id": cid,
                    "fold": fold,
                    **{f"q_{k}": float(v) for k, v in cand.quantiles.items()},
                    **m,
                    "support": int(val_mask.sum()),
                }
            )

        if not fold_metrics:
            continue

        mean_precision = float(np.mean([m["precision"] for m in fold_metrics]))
        mean_recall = float(np.mean([m["recall"] for m in fold_metrics]))
        mean_f1 = float(np.mean([m["f1"] for m in fold_metrics]))
        mean_fp = float(np.mean([m["fp"] for m in fold_metrics]))
        mean_tp = float(np.mean([m["tp"] for m in fold_metrics]))
        mean_fn = float(np.mean([m["fn"] for m in fold_metrics]))
        mean_tn = float(np.mean([m["tn"] for m in fold_metrics]))

        candidate_rows.append(
            {
                "mode": mode,
                "candidate_id": cid,
                **{f"q_{k}": float(v) for k, v in cand.quantiles.items()},
                "mean_precision": mean_precision,
                "mean_recall": mean_recall,
                "mean_f1": mean_f1,
                "mean_fp": mean_fp,
                "mean_tp": mean_tp,
                "mean_fn": mean_fn,
                "mean_tn": mean_tn,
                "fold_count": len(fold_metrics),
            }
        )

    candidate_df = pd.DataFrame(candidate_rows)
    if candidate_df.empty:
        raise ValueError(f"No valid candidates evaluated for mode={mode}")

    selected_row, precision_constraint_met = select_best_candidate(candidate_df, min_precision=min_precision)
    selected_quantiles = {r: float(selected_row[f"q_{r}"]) for r in rules}

    full_by_group, full_global = fit_thresholds_for_rules(
        df=df,
        mode=mode,
        train_mask=eval_mask,
        quantiles=selected_quantiles,
        rules=rules,
        group_col=group_col,
    )

    candidate_df = candidate_df.sort_values(
        by=["mean_f1", "mean_fp", "mean_recall", "mean_precision"],
        ascending=[False, True, False, False],
    ).reset_index(drop=True)
    selected_key = int(selected_row["candidate_id"])
    candidate_df["selected"] = candidate_df["candidate_id"].astype(int).eq(selected_key)

    return {
        "mode": mode,
        "rules": list(rules),
        "selected_quantiles": selected_quantiles,
        "selected_metrics": {
            "mean_precision": float(selected_row["mean_precision"]),
            "mean_recall": float(selected_row["mean_recall"]),
            "mean_f1": float(selected_row["mean_f1"]),
            "mean_fp": float(selected_row["mean_fp"]),
            "fold_count": int(selected_row["fold_count"]),
            "precision_constraint_met": bool(precision_constraint_met),
        },
        "group_thresholds": full_by_group,
        "global_thresholds": full_global,
        "candidate_df": candidate_df,
        "cv_df": pd.DataFrame(cv_rows),
        "eval_support": int(eval_mask.sum()),
    }


def apply_mode_quantiles(
    df: pd.DataFrame,
    mode: str,
    rules: list[str],
    selected_quantiles: dict[str, float],
    eval_mask: np.ndarray,
    group_col: str = "distribution_group_id",
) -> dict[str, Any]:
    rules = validate_calibration_rules(rules)
    eval_mask = np.asarray(eval_mask, dtype=bool)
    if len(eval_mask) != len(df):
        raise ValueError("eval_mask length mismatch.")
    if not eval_mask.any():
        raise ValueError("No rows to apply quantiles.")

    quantiles = {r: float(selected_quantiles[r]) for r in rules}
    by_group, global_th = fit_thresholds_for_rules(
        df=df,
        mode=mode,
        train_mask=eval_mask,
        quantiles=quantiles,
        rules=rules,
        group_col=group_col,
    )

    pred = predict_distance_anomaly(
        df=df,
        mode=mode,
        target_mask=eval_mask,
        rules=rules,
        by_group_thresholds=by_group,
        global_thresholds=global_th,
        group_col=group_col,
    )
    pass_map: dict[str, np.ndarray] = {}
    for rule in rules:
        score = df[f"{RULE_TO_SIGNAL_PREFIX[rule]}_{mode}"].to_numpy(dtype=float)
        p = np.ones(len(df), dtype=bool)
        p[eval_mask] = score[eval_mask] <= float(global_th[rule])
        groups = df[group_col].astype(str).to_numpy(dtype=object)
        for gid, th_map in by_group.items():
            gmask = eval_mask & (groups == str(gid))
            if gmask.any():
                p[gmask] = score[gmask] <= float(th_map[rule])
        pass_map[rule] = p

    return {
        "mode": mode,
        "rules": list(rules),
        "selected_quantiles": quantiles,
        "group_thresholds": by_group,
        "global_thresholds": global_th,
        "pred_anomaly": pred,
        "pass_map": pass_map,
        "eval_support": int(eval_mask.sum()),
    }


def flatten_calibration_json(calibration: dict[str, Any]) -> dict[str, Any]:
    modes = calibration.get("modes", {})
    out_modes: dict[str, Any] = {}
    for mode, md in modes.items():
        out_modes[mode] = {
            "rules": list(md.get("rules", [])),
            "selected_quantiles": {k: float(v) for k, v in md.get("selected_quantiles", {}).items()},
            "selected_metrics": {
                k: (float(v) if isinstance(v, (float, int, np.number)) and not isinstance(v, bool) else v)
                for k, v in md.get("selected_metrics", {}).items()
            },
        }
    return {
        "rules": list(calibration.get("rules", [])),
        "min_precision": float(calibration.get("min_precision", 0.95)),
        "cv_folds": int(calibration.get("cv_folds", 5)),
        "modes": out_modes,
    }


def main() -> None:
    """CLI entrypoint for distance calibration module."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Distance-rule threshold calibration utilities.",
        epilog="Calibrates quantile thresholds for distance rules: output, direction, length, diff_resid",
    )
    parser.add_argument("--version", action="version", version="1.0.0")
    parser.add_argument("--info", action="store_true", help="Show module information")

    args = parser.parse_args()

    if args.info:
        print("Distance Calibration Module")
        print("=" * 50)
        print("\nRules:")
        for rule in DISTANCE_RULES:
            print(f"  - {rule}")
        print("\nKey functions:")
        print("  - calibrate_distance_thresholds(): Main calibration routine")
        print("  - parse_quantile_range_spec(): Parse quantile specs like '0.70~0.99(step 0.01)'")
        print("  - flatten_calibration_json(): Flatten calibration results to JSON-serializable format")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
