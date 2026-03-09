#!/usr/bin/env python3
"""
Independent tri-state (pass/warn/fail) calibration per metric rule.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

EPS = 1e-9


def _safe_quantile(values: np.ndarray, q: float) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.quantile(vals, float(q)))


def _normalize_quantile_pair(q_warn: float, q_fail: float) -> tuple[float, float]:
    qw = float(q_warn)
    qf = float(q_fail)
    if qf <= 0 or qf >= 1:
        raise ValueError(f"q_fail must be in (0,1): {qf}")
    if qw <= 0 or qw >= 1:
        raise ValueError(f"q_warn must be in (0,1): {qw}")
    if qw >= qf:
        qw = max(0.0001, qf - 0.01)
    return qw, qf


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


def fit_rule_thresholds(
    scores: np.ndarray,
    groups: np.ndarray,
    train_mask: np.ndarray,
    q_warn: float,
    q_fail: float,
    available_mask: np.ndarray | None = None,
) -> tuple[dict[str, tuple[float, float]], tuple[float, float]]:
    scores = np.asarray(scores, dtype=float)
    groups = np.asarray(groups, dtype=object)
    train_mask = np.asarray(train_mask, dtype=bool)
    avail = np.ones(len(scores), dtype=bool) if available_mask is None else np.asarray(available_mask, dtype=bool)
    qw, qf = _normalize_quantile_pair(q_warn, q_fail)

    base_mask = train_mask & avail
    if not base_mask.any():
        base_mask = train_mask
    gwarn = _safe_quantile(scores[base_mask], qw)
    gfail = _safe_quantile(scores[base_mask], qf)
    if not np.isfinite(gwarn):
        gwarn = _safe_quantile(scores[train_mask], qw)
    if not np.isfinite(gfail):
        gfail = _safe_quantile(scores[train_mask], qf)
    if not np.isfinite(gwarn):
        gwarn = float(np.nanmedian(scores))
    if not np.isfinite(gfail):
        gfail = float(np.nanmedian(scores))
    if gwarn > gfail:
        gwarn = gfail
    global_pair = (float(gwarn), float(gfail))

    by_group: dict[str, tuple[float, float]] = {}
    for gid in pd.unique(groups[train_mask]):
        gid_str = str(gid)
        gmask = train_mask & (groups == gid) & avail
        if not gmask.any():
            gmask = train_mask & (groups == gid)
        w = _safe_quantile(scores[gmask], qw) if gmask.any() else float("nan")
        f = _safe_quantile(scores[gmask], qf) if gmask.any() else float("nan")
        if not np.isfinite(w):
            w = global_pair[0]
        if not np.isfinite(f):
            f = global_pair[1]
        if w > f:
            w = f
        by_group[gid_str] = (float(w), float(f))
    return by_group, global_pair


def apply_rule_thresholds(
    scores: np.ndarray,
    groups: np.ndarray,
    target_mask: np.ndarray,
    by_group_thresholds: dict[str, tuple[float, float]],
    global_thresholds: tuple[float, float],
    available_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    scores = np.asarray(scores, dtype=float)
    groups = np.asarray(groups, dtype=object)
    target_mask = np.asarray(target_mask, dtype=bool)
    avail = np.ones(len(scores), dtype=bool) if available_mask is None else np.asarray(available_mask, dtype=bool)
    status = np.array(["na"] * len(scores), dtype=object)
    pass_mask = np.zeros(len(scores), dtype=bool)
    warn_mask = np.zeros(len(scores), dtype=bool)
    fail_mask = np.zeros(len(scores), dtype=bool)

    idx = np.where(target_mask)[0]
    for i in idx:
        if not avail[i]:
            status[i] = "na"
            continue
        pair = by_group_thresholds.get(str(groups[i]), global_thresholds)
        w, f = float(pair[0]), float(pair[1])
        sc = float(scores[i])
        if not np.isfinite(sc):
            status[i] = "na"
            continue
        if sc > f:
            status[i] = "fail"
            fail_mask[i] = True
        elif sc > w:
            status[i] = "warn"
            warn_mask[i] = True
        else:
            status[i] = "pass"
            pass_mask[i] = True

    return {
        "status": status,
        "pass": pass_mask,
        "warn": warn_mask,
        "fail": fail_mask,
    }


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


def evaluate_rule_states(
    y_bad: np.ndarray,
    states: dict[str, np.ndarray],
    eval_mask: np.ndarray,
) -> dict[str, float]:
    y_bad = np.asarray(y_bad, dtype=bool)
    eval_mask = np.asarray(eval_mask, dtype=bool)
    fail = np.asarray(states["fail"], dtype=bool) & eval_mask
    warn = np.asarray(states["warn"], dtype=bool) & eval_mask
    pas = np.asarray(states["pass"], dtype=bool) & eval_mask
    em = eval_mask & (fail | warn | pas)
    if not em.any():
        return {
            "fail_precision": 0.0,
            "fail_recall": 0.0,
            "fail_f1": 0.0,
            "fail_fp": 0.0,
            "warn_bad_rate": 0.0,
            "pass_bad_rate": 0.0,
            "warn_lift": 0.0,
            "warn_coverage": 0.0,
        }

    b = _binary_metrics(y_bad[em], fail[em])
    warn_bad_rate = float(y_bad[warn].mean()) if warn.any() else 0.0
    pass_bad_rate = float(y_bad[pas].mean()) if pas.any() else 0.0
    warn_lift = warn_bad_rate - pass_bad_rate
    warn_coverage = float(warn[em].mean())
    return {
        "fail_precision": float(b["precision"]),
        "fail_recall": float(b["recall"]),
        "fail_f1": float(b["f1"]),
        "fail_fp": float(b["fp"]),
        "warn_bad_rate": warn_bad_rate,
        "pass_bad_rate": pass_bad_rate,
        "warn_lift": float(warn_lift),
        "warn_coverage": float(warn_coverage),
    }


def select_best_candidate(candidate_df: pd.DataFrame, min_fail_precision: float) -> tuple[pd.Series, bool]:
    cdf = candidate_df.copy()
    filtered = cdf[cdf["mean_fail_precision"] >= float(min_fail_precision)]
    constrained = not filtered.empty
    base = filtered if constrained else cdf
    base = base.sort_values(
        by=["mean_fail_f1", "mean_warn_lift", "mean_fail_fp", "mean_fail_recall"],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)
    return base.iloc[0], constrained


def calibrate_rule_tristate(
    scores: np.ndarray,
    groups: np.ndarray,
    y_bad: np.ndarray,
    eval_mask: np.ndarray,
    available_mask: np.ndarray,
    warn_quantiles: list[float],
    fail_quantiles: list[float],
    cv_folds: int,
    min_fail_precision: float,
    seed: int = 42,
) -> dict[str, Any]:
    scores = np.asarray(scores, dtype=float)
    groups = np.asarray(groups, dtype=object)
    y_bad = np.asarray(y_bad, dtype=bool)
    eval_mask = np.asarray(eval_mask, dtype=bool)
    available_mask = np.asarray(available_mask, dtype=bool)

    eval_idx = np.where(eval_mask)[0]
    if len(eval_idx) < 2:
        raise ValueError(f"Not enough rows for tri-state calibration: {len(eval_idx)}")
    y_eval = y_bad[eval_idx]
    n_folds = max(2, min(int(cv_folds), len(eval_idx)))
    folds = _make_stratified_folds(y_eval, n_folds=n_folds, seed=seed)

    cand_rows: list[dict[str, Any]] = []
    cv_rows: list[dict[str, Any]] = []
    cid = 0
    for qf in sorted(set(float(x) for x in fail_quantiles)):
        for qw in sorted(set(float(x) for x in warn_quantiles)):
            if qw >= qf:
                continue
            fold_metrics: list[dict[str, float]] = []
            for f in range(n_folds):
                tr = np.zeros(len(scores), dtype=bool)
                va = np.zeros(len(scores), dtype=bool)
                tr[eval_idx[folds != f]] = True
                va[eval_idx[folds == f]] = True
                if not tr.any() or not va.any():
                    continue
                bg, gp = fit_rule_thresholds(
                    scores=scores,
                    groups=groups,
                    train_mask=tr,
                    q_warn=qw,
                    q_fail=qf,
                    available_mask=available_mask,
                )
                states = apply_rule_thresholds(
                    scores=scores,
                    groups=groups,
                    target_mask=va,
                    by_group_thresholds=bg,
                    global_thresholds=gp,
                    available_mask=available_mask,
                )
                met = evaluate_rule_states(y_bad=y_bad, states=states, eval_mask=va)
                fold_metrics.append(met)
                cv_rows.append(
                    {
                        "candidate_id": cid,
                        "fold": int(f),
                        "q_warn": float(qw),
                        "q_fail": float(qf),
                        **{k: float(v) for k, v in met.items()},
                        "support": int(va.sum()),
                    }
                )

            if not fold_metrics:
                cid += 1
                continue

            row = {
                "candidate_id": int(cid),
                "q_warn": float(qw),
                "q_fail": float(qf),
                "mean_fail_precision": float(np.mean([m["fail_precision"] for m in fold_metrics])),
                "mean_fail_recall": float(np.mean([m["fail_recall"] for m in fold_metrics])),
                "mean_fail_f1": float(np.mean([m["fail_f1"] for m in fold_metrics])),
                "mean_fail_fp": float(np.mean([m["fail_fp"] for m in fold_metrics])),
                "mean_warn_lift": float(np.mean([m["warn_lift"] for m in fold_metrics])),
                "mean_warn_bad_rate": float(np.mean([m["warn_bad_rate"] for m in fold_metrics])),
                "mean_pass_bad_rate": float(np.mean([m["pass_bad_rate"] for m in fold_metrics])),
                "mean_warn_coverage": float(np.mean([m["warn_coverage"] for m in fold_metrics])),
                "fold_count": int(len(fold_metrics)),
            }
            cand_rows.append(row)
            cid += 1

    cand_df = pd.DataFrame(cand_rows)
    if cand_df.empty:
        raise ValueError("No tri-state candidates evaluated.")
    selected_row, constrained = select_best_candidate(cand_df, min_fail_precision=min_fail_precision)
    q_warn = float(selected_row["q_warn"])
    q_fail = float(selected_row["q_fail"])

    bg, gp = fit_rule_thresholds(
        scores=scores,
        groups=groups,
        train_mask=eval_mask,
        q_warn=q_warn,
        q_fail=q_fail,
        available_mask=available_mask,
    )
    states = apply_rule_thresholds(
        scores=scores,
        groups=groups,
        target_mask=eval_mask,
        by_group_thresholds=bg,
        global_thresholds=gp,
        available_mask=available_mask,
    )
    cand_df = cand_df.sort_values(
        by=["mean_fail_f1", "mean_warn_lift", "mean_fail_fp", "mean_fail_recall"],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)
    cand_df["selected"] = cand_df["candidate_id"].astype(int).eq(int(selected_row["candidate_id"]))

    return {
        "selected_quantiles": {"warn": q_warn, "fail": q_fail},
        "selected_metrics": {
            "mean_fail_precision": float(selected_row["mean_fail_precision"]),
            "mean_fail_recall": float(selected_row["mean_fail_recall"]),
            "mean_fail_f1": float(selected_row["mean_fail_f1"]),
            "mean_fail_fp": float(selected_row["mean_fail_fp"]),
            "mean_warn_lift": float(selected_row["mean_warn_lift"]),
            "mean_warn_coverage": float(selected_row["mean_warn_coverage"]),
            "precision_constraint_met": bool(constrained),
            "fold_count": int(selected_row["fold_count"]),
        },
        "group_thresholds": bg,
        "global_thresholds": {"warn": float(gp[0]), "fail": float(gp[1])},
        "states": states,
        "candidate_df": cand_df,
        "cv_df": pd.DataFrame(cv_rows),
        "eval_support": int(eval_mask.sum()),
    }


def apply_rule_tristate(
    scores: np.ndarray,
    groups: np.ndarray,
    eval_mask: np.ndarray,
    available_mask: np.ndarray,
    q_warn: float,
    q_fail: float,
) -> dict[str, Any]:
    bg, gp = fit_rule_thresholds(
        scores=np.asarray(scores, dtype=float),
        groups=np.asarray(groups, dtype=object),
        train_mask=np.asarray(eval_mask, dtype=bool),
        q_warn=float(q_warn),
        q_fail=float(q_fail),
        available_mask=np.asarray(available_mask, dtype=bool),
    )
    states = apply_rule_thresholds(
        scores=np.asarray(scores, dtype=float),
        groups=np.asarray(groups, dtype=object),
        target_mask=np.asarray(eval_mask, dtype=bool),
        by_group_thresholds=bg,
        global_thresholds=gp,
        available_mask=np.asarray(available_mask, dtype=bool),
    )
    return {
        "selected_quantiles": {"warn": float(q_warn), "fail": float(q_fail)},
        "group_thresholds": bg,
        "global_thresholds": {"warn": float(gp[0]), "fail": float(gp[1])},
        "states": states,
        "eval_support": int(np.asarray(eval_mask, dtype=bool).sum()),
    }


def main() -> None:
    """CLI entrypoint for tristate calibration module."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Tri-state (warn/fail) threshold calibration utilities.",
        epilog="Calibrates warn and fail thresholds for rule-based anomaly detection using cross-validation.",
    )
    parser.add_argument("--version", action="version", version="1.0.0")
    parser.add_argument("--info", action="store_true", help="Show module information")

    args = parser.parse_args()

    if args.info:
        print("Tri-state Calibration Module")
        print("=" * 50)
        print("\nStates:")
        print("  - warn: Lower threshold for warning")
        print("  - fail: Higher threshold for failure")
        print("\nKey functions:")
        print("  - calibrate_rule_tristate(): Main tristate calibration routine")
        print("  - apply_rule_tristate(): Apply computed thresholds to signal data")
        print("  - evaluate_rule_states(): Evaluate state accuracy on reference labels")
        return

    parser.print_help()


if __name__ == "__main__":
    main()

