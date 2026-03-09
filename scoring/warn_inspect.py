"""Independent warn-inspect module for high-risk sample surfacing.

This module is intentionally separated from score aggregation:
- `score.bundle_score` remains score-only.
- `inspect.warn_inspect` extracts row-level risk candidates for operations review.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from final_metric_refactor.config import RUNTIME_RULE_AVAILABLE_COL_NOMASK, RUNTIME_RULE_SIGNAL_COL_NOMASK

EPS = 1e-9
TARGET_RULES = [
    "output",
    "diff_residual",
    "delta_ridge_ens",
    "discourse_instability",
    "contradiction",
]


@dataclass(frozen=True)
class WarnInspectArtifacts:
    rows_df: pd.DataFrame
    payload: dict[str, Any]
    summary: dict[str, Any]


def _to_bool_array(series: pd.Series | np.ndarray | list[Any] | bool, size: int) -> np.ndarray:
    if isinstance(series, bool):
        return np.full(size, bool(series), dtype=bool)
    arr = pd.Series(series) if not isinstance(series, pd.Series) else series
    if len(arr) != size:
        out = np.zeros(size, dtype=bool)
        n = min(size, len(arr))
        if n > 0:
            out[:n] = _to_bool_array(arr.iloc[:n], n)
        return out
    if arr.dtype == bool:
        return arr.fillna(False).to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(arr):
        return arr.fillna(0).to_numpy(dtype=float) != 0.0
    mapped = (
        arr.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False})
        .fillna(False)
    )
    return mapped.to_numpy(dtype=bool)


def _robust_center_scale(values: np.ndarray) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size <= 0:
        return float("nan"), float("nan")
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    sigma = float(1.4826 * mad)
    if (not np.isfinite(sigma)) or sigma <= float(EPS):
        sigma = float(EPS)
    return med, sigma


def _sigmoid_safe(x: np.ndarray) -> np.ndarray:
    z = np.clip(np.asarray(x, dtype=float), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def _softplus_safe(x: np.ndarray) -> np.ndarray:
    z = np.clip(np.asarray(x, dtype=float), -60.0, 60.0)
    return np.log1p(np.exp(z))


def _clip01(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(arr, dtype=float), 0.0, 1.0)


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        v = float(value)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _risk_from_scale(raw: np.ndarray, scale: float) -> np.ndarray:
    r = np.maximum(0.0, np.asarray(raw, dtype=float))
    s = max(float(EPS), float(scale))
    return _clip01(r / (r + s))


def _noisy_or_two_axis(rate: np.ndarray, mass: np.ndarray, w_rate: float, w_mass: float) -> np.ndarray:
    wr = max(0.0, float(w_rate))
    wm = max(0.0, float(w_mass))
    denom = wr + wm
    if denom <= float(EPS):
        wr_n = 0.5
        wm_n = 0.5
    else:
        wr_n = wr / denom
        wm_n = wm / denom
    return _clip01(1.0 - ((1.0 - wr_n * rate) * (1.0 - wm_n * mass)))


def _hard_tail_penalty(p_hard: np.ndarray, mass_risk: np.ndarray, alpha: float) -> np.ndarray:
    ph = _clip01(np.nan_to_num(np.asarray(p_hard, dtype=float), nan=0.0))
    mr = _clip01(np.nan_to_num(np.asarray(mass_risk, dtype=float), nan=0.0))
    a = max(0.0, float(alpha))
    return a * ph * (0.5 + 0.5 * mr)


def _state_arrays(row_df: pd.DataFrame, rule: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    col = f"{rule}_state_nomask"
    if col not in row_df.columns:
        return np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    s = row_df[col].fillna("").astype(str).str.strip().str.lower().to_numpy()
    hard = s == "hard_fail"
    fail = (s == "fail") | hard
    return hard, fail


def _string_col(row_df: pd.DataFrame, col: str, n: int, fallback_prefix: str = "") -> np.ndarray:
    if col in row_df.columns:
        return row_df[col].fillna("").astype(str).to_numpy()
    if fallback_prefix:
        return np.asarray([f"{fallback_prefix}{i}" for i in range(n)], dtype=object)
    return np.asarray([""] * n, dtype=object)


def _nonneg_int_col(row_df: pd.DataFrame, col: str, n: int, default: int = 0) -> np.ndarray:
    if col in row_df.columns:
        s = pd.to_numeric(row_df[col], errors="coerce")
    else:
        s = pd.Series([default] * n, dtype=float)
    return s.fillna(default).clip(lower=0).astype(int).to_numpy(dtype=int)


def _parse_detail_paths(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for tok in text.split("|"):
        path = str(tok).strip()
        if not path or path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def _build_detail_leaf_summary(
    *,
    rows_df: pd.DataFrame,
    n_detail_rows: int,
    max_items: int = 5,
) -> dict[str, Any]:
    if rows_df.empty or n_detail_rows <= 0 or "detail_failed_leaf_paths" not in rows_df.columns:
        return {
            "enabled": False,
            "n_detail_rows": int(max(0, n_detail_rows)),
            "n_rows_with_paths": 0,
            "top_leaves": [],
            "issue_summary": "상세 모드 leaf 하락 신호가 없습니다.",
        }

    leaf_counter: Counter[str] = Counter()
    leaf_fail_counts: dict[str, list[int]] = {}
    first_seen: dict[str, int] = {}
    rows_with_paths = 0
    for _, row in rows_df.iterrows():
        paths = _parse_detail_paths(row.get("detail_failed_leaf_paths", ""))
        if not paths:
            continue
        rows_with_paths += 1
        row_fail_count = int(_safe_float(row.get("detail_fail_leaf_count", 0.0), default=0.0))
        for p in paths:
            if p not in first_seen:
                first_seen[p] = len(first_seen)
            leaf_counter[p] += 1
            leaf_fail_counts.setdefault(p, []).append(max(0, row_fail_count))

    if not leaf_counter:
        return {
            "enabled": True,
            "n_detail_rows": int(max(0, n_detail_rows)),
            "n_rows_with_paths": int(rows_with_paths),
            "top_leaves": [],
            "issue_summary": "상세 fail 행이 있으나 leaf path 정보가 비어 있습니다.",
        }

    top_leaves: list[dict[str, Any]] = []
    max_items = int(max(1, max_items))
    ranked = sorted(leaf_counter.items(), key=lambda x: (-int(x[1]), int(first_seen.get(str(x[0]), 10**9))))
    for leaf_path, count in ranked[:max_items]:
        row_counts = leaf_fail_counts.get(leaf_path, [])
        top_leaves.append(
            {
                "leaf_path": str(leaf_path),
                "affected_rows": int(count),
                "affected_share": float(count / float(max(1, n_detail_rows))),
                "avg_fail_leaf_count": float(np.mean(row_counts)) if row_counts else 0.0,
            }
        )

    top = top_leaves[0]
    top_path = str(top.get("leaf_path", ""))
    top_count = int(top.get("affected_rows", 0))
    top_share = float(top.get("affected_share", 0.0))
    if top_share >= 0.60:
        issue_summary = (
            f"상세 하락이 '{top_path}' leaf에 집중되어 있습니다 "
            f"({top_count}/{int(n_detail_rows)} rows). 해당 leaf 규칙/템플릿 점검을 우선 권장합니다."
        )
    else:
        preview = ", ".join([str(x.get("leaf_path", "")) for x in top_leaves[:3] if str(x.get("leaf_path", ""))])
        issue_summary = (
            f"상세 하락이 여러 leaf에 분산되어 있습니다. 상위 leaf: {preview}. "
            "공통 prompt/포맷 규칙부터 점검하세요."
        )

    return {
        "enabled": True,
        "n_detail_rows": int(max(0, n_detail_rows)),
        "n_rows_with_paths": int(rows_with_paths),
        "top_leaves": top_leaves,
        "issue_summary": issue_summary,
    }


def _bundle_weight_from_payload(bundle_payload: dict[str, Any] | None, bundle: str) -> float:
    if not isinstance(bundle_payload, dict):
        return float("nan")
    raw = bundle_payload.get("raw", {})
    if not isinstance(raw, dict):
        return float("nan")
    tri = raw.get("triplet_risk_norm", {})
    if not isinstance(tri, dict):
        return float("nan")
    b = tri.get(bundle, {})
    if not isinstance(b, dict):
        return float("nan")
    return _safe_float(b.get("bundle_or_risk", np.nan), default=float("nan"))


def _sem_not_applicable(bundle_payload: dict[str, Any] | None) -> bool:
    if not isinstance(bundle_payload, dict):
        return False
    raw = bundle_payload.get("raw", {})
    if not isinstance(raw, dict):
        return False
    sem = raw.get("semantic", {})
    if not isinstance(sem, dict):
        return False
    return bool(sem.get("sem_not_applicable", False) or sem.get("sem_uninformative", False))


def _build_warn_rows_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "rank",
                "row_index",
                "row_id",
                "tags",
                "rank_score",
                "row_cont",
                "state_severity",
                "hard_count",
                "fail_count",
                "hard_rules",
                "fail_rules",
                "dominant_bundle",
                "dominant_rule",
                "bundle_risk_contrib",
                "detail_fail_any",
                "detail_fail_leaf_count",
                "detail_eval_leaf_count",
                "detail_failed_leaf_paths",
                "detail_primary_leaf",
                "source_input",
                "source_output",
            ]
        )
    return pd.DataFrame(rows)


def _normalize_weights(weights: dict[str, float], keys: list[str]) -> dict[str, float]:
    vals = [max(0.0, float(weights.get(k, 0.0))) for k in keys]
    s = float(np.sum(vals))
    if s <= float(EPS):
        uni = 1.0 / float(len(keys)) if keys else 0.0
        return {k: uni for k in keys}
    return {k: float(max(0.0, float(weights.get(k, 0.0))) / s) for k in keys}


def _flatten_for_csv(rows_df: pd.DataFrame) -> pd.DataFrame:
    if rows_df.empty:
        return rows_df.copy()
    out = rows_df.copy()
    out["tags"] = out["tags"].apply(lambda x: "|".join([str(v) for v in x]) if isinstance(x, list) else str(x))
    out["hard_rules"] = out["hard_rules"].apply(lambda x: "|".join([str(v) for v in x]) if isinstance(x, list) else str(x))
    out["fail_rules"] = out["fail_rules"].apply(lambda x: "|".join([str(v) for v in x]) if isinstance(x, list) else str(x))
    for b in ["OUT", "RID", "DIAG", "SEM"]:
        out[f"contrib_{b.lower()}"] = out["bundle_risk_contrib"].apply(
            lambda d: (_safe_float(d.get(b, np.nan), default=np.nan) if isinstance(d, dict) else np.nan)
        )
    out["bundle_risk_contrib"] = out["bundle_risk_contrib"].apply(
        lambda d: json_dumps_compact(d) if isinstance(d, dict) else "{}"
    )
    return out


def json_dumps_compact(value: Any) -> str:
    try:
        import json

        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return "{}"


def compute_warn_inspect(
    *,
    row_df: pd.DataFrame,
    threshold_summary_df: pd.DataFrame,
    score_runtime: Any,
    bundle_payload: dict[str, Any] | None = None,
) -> WarnInspectArtifacts:
    del threshold_summary_df  # kept for interface parity; currently not required in score input.

    n = len(row_df)
    if n <= 0:
        empty_rows = _build_warn_rows_dataframe([])
        payload = build_warn_inspect_payload(
            rows_df=empty_rows,
            top_k=int(getattr(score_runtime, "new_score_risk_case_top_k", 20)),
            multi_fail_cut=int(getattr(score_runtime, "new_score_risk_case_multi_fail_cut", 3)),
            mix_cont=float(getattr(score_runtime, "new_score_risk_case_mix_cont", 0.70)),
            mix_state=float(getattr(score_runtime, "new_score_risk_case_mix_state", 0.30)),
            state_hard_weight=float(getattr(score_runtime, "new_score_risk_case_state_hard_weight", 8.0)),
            state_fail_weight=float(getattr(score_runtime, "new_score_risk_case_state_fail_weight", 1.0)),
            detail_boost=float(getattr(score_runtime, "new_score_risk_case_detail_boost", 0.20)),
            preview_chars=int(getattr(score_runtime, "new_score_risk_case_preview_chars", 180)),
            n_rows=0,
            n_candidates=0,
            n_hard=0,
            n_multi=0,
            n_detail=0,
        )
        return WarnInspectArtifacts(rows_df=empty_rows, payload=payload, summary=dict(payload.get("summary", {})))

    hard_gate = _to_bool_array(row_df.get("hard_gate_pass", True), n)
    z_fail_ref = float(getattr(score_runtime, "new_score_z_fail_ref", 2.0))
    z_hard_ref = float(getattr(score_runtime, "new_score_z_hard_ref", 3.5))
    tau_rate = float(getattr(score_runtime, "new_score_tau_rate", 0.60))
    tau_mass = float(getattr(score_runtime, "new_score_tau_mass", 0.80))
    rate_lambda_output = float(getattr(score_runtime, "new_score_rate_lambda_output", 8.0))
    rate_lambda_residual = float(getattr(score_runtime, "new_score_rate_lambda_residual", 8.0))
    rate_lambda_semantic = float(getattr(score_runtime, "new_score_rate_lambda_semantic", 8.0))
    eta_hard = float(getattr(score_runtime, "eta_hard", 3.0))
    alpha = float(getattr(score_runtime, "new_score_hard_tail_penalty_alpha", 0.25))
    w_rate = float(getattr(score_runtime, "new_score_w_rate", 0.55))
    w_mass = float(getattr(score_runtime, "new_score_w_mass", 0.30))

    s_mass_output = float(getattr(score_runtime, "new_score_s_mass_output", 0.50))
    s_mass_diff = float(getattr(score_runtime, "new_score_s_mass_diff_residual", 0.50))
    s_mass_ridge = float(getattr(score_runtime, "new_score_s_mass_delta_ridge_ens", 0.50))
    s_mass_disc = float(getattr(score_runtime, "new_score_s_mass_discourse_instability", 0.50))
    s_mass_contr = float(getattr(score_runtime, "new_score_s_mass_contradiction", 0.50))

    top_k = int(max(1, getattr(score_runtime, "new_score_risk_case_top_k", 20)))
    multi_fail_cut = int(max(1, getattr(score_runtime, "new_score_risk_case_multi_fail_cut", 3)))
    mix_cont = float(np.clip(getattr(score_runtime, "new_score_risk_case_mix_cont", 0.70), 0.0, 1.0))
    mix_state = float(np.clip(getattr(score_runtime, "new_score_risk_case_mix_state", 0.30), 0.0, 1.0))
    mix_sum = mix_cont + mix_state
    if mix_sum <= float(EPS):
        mix_cont = 0.70
        mix_state = 0.30
    else:
        mix_cont = mix_cont / mix_sum
        mix_state = mix_state / mix_sum
    state_hard_weight = float(max(0.0, getattr(score_runtime, "new_score_risk_case_state_hard_weight", 8.0)))
    state_fail_weight = float(max(0.0, getattr(score_runtime, "new_score_risk_case_state_fail_weight", 1.0)))
    detail_boost = float(np.clip(getattr(score_runtime, "new_score_risk_case_detail_boost", 0.20), 0.0, 1.0))
    # Ensure one hard event outweighs seven fail events.
    state_hard_weight = float(max(state_hard_weight, (7.0 * state_fail_weight) + float(EPS)))
    preview_chars = int(max(32, getattr(score_runtime, "new_score_risk_case_preview_chars", 180)))

    signal_inputs: dict[str, dict[str, Any]] = {}
    for rule in TARGET_RULES:
        signal_col = str(RUNTIME_RULE_SIGNAL_COL_NOMASK.get(rule, f"{rule}_signal_nomask"))
        avail_col = str(RUNTIME_RULE_AVAILABLE_COL_NOMASK.get(rule, ""))
        if signal_col in row_df.columns:
            signal = pd.to_numeric(row_df[signal_col], errors="coerce").to_numpy(dtype=float)
        else:
            signal = np.full(n, np.nan, dtype=float)
        available = (
            _to_bool_array(row_df.get(avail_col, True), n)
            if avail_col and avail_col in row_df.columns
            else np.ones(n, dtype=bool)
        )
        valid = hard_gate & available & np.isfinite(signal)
        m, sigma = _robust_center_scale(signal[valid])
        if np.isfinite(m) and np.isfinite(sigma):
            z = (signal - m) / (sigma + EPS)
        else:
            z = np.full(n, np.nan, dtype=float)
        z_pos = np.maximum(0.0, z)
        z_pos[~valid] = np.nan
        pf = np.full(n, np.nan, dtype=float)
        ph = np.full(n, np.nan, dtype=float)
        if np.any(valid):
            pf_v = _sigmoid_safe((z_pos[valid] - z_fail_ref) / max(float(EPS), tau_rate))
            ph_v = _sigmoid_safe((z_pos[valid] - z_hard_ref) / max(float(EPS), tau_rate))
            pf[valid] = pf_v
            ph[valid] = ph_v
        signal_inputs[rule] = {
            "signal_col": signal_col,
            "available_col": avail_col,
            "valid": valid,
            "z_pos": z_pos,
            "p_fail_soft": pf,
            "p_hard_soft": ph,
        }

    def _rule_row_risk(rule: str, rate_lambda: float, s_mass: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        src = signal_inputs[rule]
        z_pos = np.asarray(src["z_pos"], dtype=float)
        valid = np.asarray(src["valid"], dtype=bool)
        pf = np.nan_to_num(np.asarray(src["p_fail_soft"], dtype=float), nan=0.0)
        ph = np.nan_to_num(np.asarray(src["p_hard_soft"], dtype=float), nan=0.0)
        rate_r = _clip01(pf + float(rate_lambda) * ph)
        ef = _softplus_safe((np.nan_to_num(z_pos, nan=0.0) - z_fail_ref) / max(float(EPS), tau_mass))
        eh = _softplus_safe((np.nan_to_num(z_pos, nan=0.0) - z_hard_ref) / max(float(EPS), tau_mass))
        mass_raw = ef + float(max(0.0, eta_hard)) * eh
        mass_r = _risk_from_scale(mass_raw, s_mass)
        base = _noisy_or_two_axis(rate_r, mass_r, w_rate, w_mass)
        risk = _clip01(base + _hard_tail_penalty(ph, mass_r, alpha))
        risk[~valid] = np.nan
        rate_r[~valid] = np.nan
        mass_r[~valid] = np.nan
        ph[~valid] = np.nan
        return risk, rate_r, mass_r, ph

    out_risk, out_rate_r, out_mass_r, out_ph = _rule_row_risk(
        "output", rate_lambda=rate_lambda_output, s_mass=s_mass_output
    )
    diff_risk, diff_rate_r, diff_mass_r, diff_ph = _rule_row_risk(
        "diff_residual", rate_lambda=rate_lambda_residual, s_mass=s_mass_diff
    )
    ridge_risk, ridge_rate_r, ridge_mass_r, ridge_ph = _rule_row_risk(
        "delta_ridge_ens", rate_lambda=rate_lambda_residual, s_mass=s_mass_ridge
    )
    disc_risk, disc_rate_r, disc_mass_r, disc_ph = _rule_row_risk(
        "discourse_instability", rate_lambda=rate_lambda_semantic, s_mass=s_mass_disc
    )
    contr_risk, contr_rate_r, contr_mass_r, contr_ph = _rule_row_risk(
        "contradiction", rate_lambda=rate_lambda_semantic, s_mass=s_mass_contr
    )

    a = np.nan_to_num(signal_inputs["diff_residual"]["p_fail_soft"], nan=0.0)
    b = np.nan_to_num(signal_inputs["delta_ridge_ens"]["p_fail_soft"], nan=0.0)
    diag_rate = _clip01(a + b - (a * b))
    q10 = a * (1.0 - b)
    q01 = (1.0 - a) * b
    q11 = a * b
    diag_mass = _clip01(q11 + 0.5 * np.minimum(q10, q01))
    diag_ph = np.nanmean(
        np.vstack(
            [
                np.nan_to_num(signal_inputs["diff_residual"]["p_hard_soft"], nan=np.nan),
                np.nan_to_num(signal_inputs["delta_ridge_ens"]["p_hard_soft"], nan=np.nan),
            ]
        ),
        axis=0,
    )
    diag_base = _noisy_or_two_axis(diag_rate, diag_mass, w_rate, w_mass)
    diag_risk = _clip01(diag_base + _hard_tail_penalty(np.nan_to_num(diag_ph, nan=0.0), diag_mass, alpha))
    diag_valid = np.asarray(signal_inputs["diff_residual"]["valid"], dtype=bool) & np.asarray(
        signal_inputs["delta_ridge_ens"]["valid"], dtype=bool
    )
    diag_risk[~diag_valid] = np.nan

    sem_skip = _sem_not_applicable(bundle_payload)
    if sem_skip:
        sem_risk = np.full(n, np.nan, dtype=float)
    else:
        sem_rate = _clip01(np.nanmean(np.vstack([disc_rate_r, contr_rate_r]), axis=0))
        sem_mass = _clip01(np.nanmean(np.vstack([disc_mass_r, contr_mass_r]), axis=0))
        sem_ph = np.nanmean(np.vstack([disc_ph, contr_ph]), axis=0)
        sem_base = _noisy_or_two_axis(sem_rate, sem_mass, w_rate, w_mass)
        sem_risk = _clip01(sem_base + _hard_tail_penalty(np.nan_to_num(sem_ph, nan=0.0), sem_mass, alpha))
        sem_valid = np.asarray(signal_inputs["discourse_instability"]["valid"], dtype=bool) & np.asarray(
            signal_inputs["contradiction"]["valid"], dtype=bool
        )
        sem_risk[~sem_valid] = np.nan

    bundle_contrib_map: dict[str, np.ndarray] = {
        "OUT": out_risk,
        "RID": diff_risk,
        "DIAG": diag_risk,
        "SEM": sem_risk,
    }

    bundle_weight_raw = {
        "OUT": _bundle_weight_from_payload(bundle_payload, "OUT"),
        "RID": _bundle_weight_from_payload(bundle_payload, "RID"),
        "DIAG": _bundle_weight_from_payload(bundle_payload, "DIAG"),
        "SEM": _bundle_weight_from_payload(bundle_payload, "SEM"),
    }

    row_cont = np.zeros(n, dtype=float)
    for i in range(n):
        valid_keys = [k for k, arr in bundle_contrib_map.items() if np.isfinite(float(arr[i]))]
        if not valid_keys:
            row_cont[i] = 0.0
            continue
        w_norm = _normalize_weights(bundle_weight_raw, valid_keys)
        row_cont[i] = float(
            np.clip(
                np.sum([w_norm[k] * float(bundle_contrib_map[k][i]) for k in valid_keys]),
                0.0,
                1.0,
            )
        )

    hard_rules_map: list[list[str]] = [[] for _ in range(n)]
    fail_rules_map: list[list[str]] = [[] for _ in range(n)]
    hard_count = np.zeros(n, dtype=int)
    fail_count = np.zeros(n, dtype=int)
    active_state_rules = 0
    for rule in TARGET_RULES:
        hard_arr, fail_arr = _state_arrays(row_df, rule, n)
        if f"{rule}_state_nomask" in row_df.columns:
            active_state_rules += 1
        idx_h = np.where(hard_arr)[0]
        idx_f = np.where(fail_arr)[0]
        for i in idx_h.tolist():
            hard_rules_map[i].append(rule)
        for i in idx_f.tolist():
            fail_rules_map[i].append(rule)
        hard_count += hard_arr.astype(int)
        fail_count += fail_arr.astype(int)
    n_active_rules = max(1, active_state_rules)

    nonhard_fail_count = np.maximum(0, fail_count - hard_count)
    state_raw = (state_hard_weight * hard_count.astype(float)) + (
        state_fail_weight * nonhard_fail_count.astype(float)
    )
    state_den = float(max(float(EPS), max(state_hard_weight, state_fail_weight) * float(n_active_rules)))
    state_severity = np.clip(state_raw.astype(float) / state_den, 0.0, 1.0)
    rank_score = (float(mix_cont) * row_cont) + (float(mix_state) * state_severity)
    detail_eval_any = _to_bool_array(row_df.get("detail_evaluated_nomask", False), n)
    detail_fail_any = _to_bool_array(row_df.get("detail_fail_any_leaf_nomask", False), n)
    detail_target = hard_gate & detail_eval_any & detail_fail_any
    rank_score = np.clip(rank_score + (detail_boost * detail_target.astype(float)), 0.0, 1.0)
    detail_fail_leaf_count = _nonneg_int_col(row_df, "detail_fail_leaf_count_nomask", n, default=0)
    detail_eval_leaf_count = _nonneg_int_col(row_df, "detail_eval_leaf_count_nomask", n, default=0)
    detail_failed_leaf_paths = _string_col(row_df, "detail_failed_leaf_paths_nomask", n)

    row_id = _string_col(row_df, "row_id", n, fallback_prefix="row_")
    source_input = _string_col(row_df, "source_input", n)
    source_output = _string_col(row_df, "source_output", n)

    tags_map: list[list[str]] = [[] for _ in range(n)]
    is_hard = hard_count > 0
    is_multi = fail_count >= int(multi_fail_cut)
    for i in range(n):
        if is_hard[i]:
            tags_map[i].append("HARD")
        if is_multi[i]:
            tags_map[i].append("MULTI")
        if detail_target[i]:
            tags_map[i].append("DETAIL")
    candidates = is_hard | is_multi | detail_target

    rule_risk_map = {
        "output": out_risk,
        "diff_residual": diff_risk,
        "delta_ridge_ens": ridge_risk,
        "discourse_instability": disc_risk,
        "contradiction": contr_risk,
    }

    candidate_idx = np.where(candidates)[0]
    rows: list[dict[str, Any]] = []
    for i in candidate_idx.tolist():
        contrib = {k: (_safe_float(bundle_contrib_map[k][i], default=np.nan)) for k in ["OUT", "RID", "DIAG", "SEM"]}
        finite_contrib = {k: v for k, v in contrib.items() if np.isfinite(v)}
        dominant_bundle = max(finite_contrib.items(), key=lambda x: x[1])[0] if finite_contrib else ""

        rr = {k: _safe_float(v[i], default=np.nan) for k, v in rule_risk_map.items()}
        rr_f = {k: v for k, v in rr.items() if np.isfinite(v)}
        dominant_rule = max(rr_f.items(), key=lambda x: x[1])[0] if rr_f else ""
        detail_paths = _parse_detail_paths(detail_failed_leaf_paths[i])

        rows.append(
            {
                "rank": 0,
                "row_index": int(i),
                "row_id": str(row_id[i]),
                "tags": list(tags_map[i]),
                "rank_score": float(rank_score[i]),
                "row_cont": float(row_cont[i]),
                "state_severity": float(state_severity[i]),
                "hard_count": int(hard_count[i]),
                "fail_count": int(fail_count[i]),
                "hard_rules": list(hard_rules_map[i]),
                "fail_rules": list(fail_rules_map[i]),
                "dominant_bundle": str(dominant_bundle),
                "dominant_rule": str(dominant_rule),
                "bundle_risk_contrib": {
                    "OUT": (float(contrib["OUT"]) if np.isfinite(contrib["OUT"]) else None),
                    "RID": (float(contrib["RID"]) if np.isfinite(contrib["RID"]) else None),
                    "DIAG": (float(contrib["DIAG"]) if np.isfinite(contrib["DIAG"]) else None),
                    "SEM": (float(contrib["SEM"]) if np.isfinite(contrib["SEM"]) else None),
                },
                "detail_fail_any": bool(detail_target[i]),
                "detail_fail_leaf_count": int(detail_fail_leaf_count[i]),
                "detail_eval_leaf_count": int(detail_eval_leaf_count[i]),
                "detail_failed_leaf_paths": str(detail_failed_leaf_paths[i]),
                "detail_primary_leaf": (str(detail_paths[0]) if detail_paths else ""),
                "source_input": str(source_input[i]),
                "source_output": str(source_output[i]),
            }
        )

    # Sort -> dedupe row_id -> top-k.
    rows.sort(
        key=lambda r: (
            -float(r["rank_score"]),
            -int(r["hard_count"]),
            -int(r["fail_count"]),
            int(r["row_index"]),
        )
    )
    dedup_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for r in rows:
        rid = str(r["row_id"])
        if rid in seen_ids:
            continue
        seen_ids.add(rid)
        dedup_rows.append(r)
        if len(dedup_rows) >= int(top_k):
            break
    for idx, r in enumerate(dedup_rows, start=1):
        r["rank"] = int(idx)

    rows_df = _build_warn_rows_dataframe(dedup_rows)
    rows_df_for_csv = _flatten_for_csv(rows_df)
    detail_leaf_summary = _build_detail_leaf_summary(
        rows_df=rows_df,
        n_detail_rows=int(np.sum(detail_target)),
        max_items=5,
    )

    summary = {
        "n_rows": int(n),
        "n_candidates": int(np.sum(candidates)),
        "n_hard": int(np.sum(is_hard)),
        "n_multi": int(np.sum(is_multi)),
        "n_detail": int(np.sum(detail_target)),
        "n_detail_with_paths": int(detail_leaf_summary.get("n_rows_with_paths", 0)),
        "n_selected": int(len(rows_df)),
    }
    payload = build_warn_inspect_payload(
        rows_df=rows_df,
        top_k=int(top_k),
        multi_fail_cut=int(multi_fail_cut),
        mix_cont=float(mix_cont),
        mix_state=float(mix_state),
        state_hard_weight=float(state_hard_weight),
        state_fail_weight=float(state_fail_weight),
        detail_boost=float(detail_boost),
        preview_chars=int(preview_chars),
        n_rows=summary["n_rows"],
        n_candidates=summary["n_candidates"],
        n_hard=summary["n_hard"],
        n_multi=summary["n_multi"],
        n_detail=summary["n_detail"],
        detail_leaf_summary=detail_leaf_summary,
    )
    payload["summary"]["n_selected"] = int(summary["n_selected"])
    payload["rows_csv"] = rows_df_for_csv.to_dict(orient="records")

    return WarnInspectArtifacts(rows_df=rows_df_for_csv, payload=payload, summary=summary)


def build_warn_inspect_payload(
    *,
    rows_df: pd.DataFrame,
    top_k: int,
    multi_fail_cut: int,
    mix_cont: float,
    mix_state: float,
    state_hard_weight: float,
    state_fail_weight: float,
    detail_boost: float,
    preview_chars: int,
    n_rows: int,
    n_candidates: int,
    n_hard: int,
    n_multi: int,
    n_detail: int,
    detail_leaf_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if not rows_df.empty:
        for _, r in rows_df.iterrows():
            rows.append(
                {
                    "rank": int(r.get("rank", 0)),
                    "row_index": int(r.get("row_index", -1)),
                    "row_id": str(r.get("row_id", "")),
                    "tags": list(r.get("tags", [])) if isinstance(r.get("tags", []), list) else str(r.get("tags", "")).split("|"),
                    "rank_score": _safe_float(r.get("rank_score", np.nan), default=np.nan),
                    "row_cont": _safe_float(r.get("row_cont", np.nan), default=np.nan),
                    "state_severity": _safe_float(r.get("state_severity", np.nan), default=np.nan),
                    "hard_count": int(r.get("hard_count", 0)),
                    "fail_count": int(r.get("fail_count", 0)),
                    "hard_rules": list(r.get("hard_rules", [])) if isinstance(r.get("hard_rules", []), list) else str(r.get("hard_rules", "")).split("|"),
                    "fail_rules": list(r.get("fail_rules", [])) if isinstance(r.get("fail_rules", []), list) else str(r.get("fail_rules", "")).split("|"),
                    "dominant_bundle": str(r.get("dominant_bundle", "")),
                    "dominant_rule": str(r.get("dominant_rule", "")),
                    "bundle_risk_contrib": (
                        r.get("bundle_risk_contrib", {})
                        if isinstance(r.get("bundle_risk_contrib", {}), dict)
                        else {}
                    ),
                    "detail_fail_any": bool(r.get("detail_fail_any", False)),
                    "detail_fail_leaf_count": int(_safe_float(r.get("detail_fail_leaf_count", 0), default=0.0)),
                    "detail_eval_leaf_count": int(_safe_float(r.get("detail_eval_leaf_count", 0), default=0.0)),
                    "detail_failed_leaf_paths": str(r.get("detail_failed_leaf_paths", "")),
                    "detail_primary_leaf": str(r.get("detail_primary_leaf", "")),
                    "source_input": str(r.get("source_input", "")),
                    "source_output": str(r.get("source_output", "")),
                }
            )
    leaf_summary = (
        detail_leaf_summary
        if isinstance(detail_leaf_summary, dict)
        else {
            "enabled": False,
            "n_detail_rows": int(max(0, n_detail)),
            "n_rows_with_paths": 0,
            "top_leaves": [],
            "issue_summary": "상세 모드 leaf 하락 신호가 없습니다.",
        }
    )
    return {
        "version": "v1",
        "selection": {
            "top_k": int(top_k),
            "multi_fail_cut": int(multi_fail_cut),
            "mix_cont": float(mix_cont),
            "mix_state": float(mix_state),
            "state_hard_weight": float(state_hard_weight),
            "state_fail_weight": float(state_fail_weight),
            "detail_boost": float(detail_boost),
            "preview_chars": int(preview_chars),
        },
        "summary": {
            "n_rows": int(n_rows),
            "n_candidates": int(n_candidates),
            "n_hard": int(n_hard),
            "n_multi": int(n_multi),
            "n_detail": int(n_detail),
            "n_detail_with_paths": int(leaf_summary.get("n_rows_with_paths", 0)),
            "n_selected": int(len(rows)),
        },
        "detail_leaf_summary": leaf_summary,
        "rows": rows,
    }
