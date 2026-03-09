"""Bundle score calculator for final_metric runtime."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from final_metric_refactor.config import RUNTIME_RULE_AVAILABLE_COL_NOMASK, RUNTIME_RULE_SIGNAL_COL_NOMASK
from final_metric_refactor.shared.geometry import knn_self

EPS = 1e-9
BUNDLE_ORDER = ["COV", "OUT", "RID", "DIAG", "SEM", "CONF"]
BUNDLE_LABEL_KO = {
    "COV": "COV (운영 커버리지)",
    "OUT": "OUT (출력 안정성)",
    "RID": "RID (설명가능성)",
    "DIAG": "DIAG (원인 분해)",
    "SEM": "SEM (의미 안정성)",
    "CONF": "CONF (점수 신뢰도)",
}
BUNDLE_DESC_KO = {
    "COV": "HG/finite/available 기반 측정 가능성(coverage)을 나타냅니다.",
    "OUT": "출력 분포 이탈 위험을 Core/Rate/Mass 축으로 평가합니다.",
    "RID": "분포 형태 기준 잔차(diff_residual) 이탈 위험을 평가합니다.",
    "DIAG": "Residual vs Ridge 동시 관찰로 이탈 원인을 q10/q01/q11로 분해합니다.",
    "SEM": "discourse_instability + contradiction 기반 의미 안정성을 평가합니다.",
    "CONF": "품질 점수와 분리된 신뢰도(Conf_data/calc/th/op)를 나타냅니다.",
}
SUBMETRIC_LABEL_KO = {
    "HG_score": "하드게이트 통과율",
    "Finite_score": "수치 계산 가능성",
    "Available_score": "룰 계산 가능성",
    "OUT_core": "코어 응집도",
    "OUT_rate": "경계 이탈 비율",
    "OUT_mass": "이탈 깊이 위험",
    "RID_core": "설명가능 코어 안정성",
    "RID_rate": "설명불가 이동 비율",
    "RID_mass": "설명불가 이동 깊이",
    "DIAG_core": "코어 참고치(RID core)",
    "DIAG_rate": "불안정 비율",
    "DIAG_mass": "붕괴 심각도",
    "DIAG_stable": "안정 상태(q00)",
    "DIAG_local": "분포형 이탈(q10)",
    "DIAG_unexplainable": "예측 이탈(q01)",
    "DIAG_systemic": "동시 붕괴(q11)",
    "SEM_core": "의미 코어 안정성",
    "SEM_rate": "의미 이탈 비율",
    "SEM_mass": "의미 이탈 깊이",
    "CONF_data": "CONF_data",
    "CONF_calc": "CONF_calc",
    "CONF_th": "CONF_th",
    "CONF_op": "CONF_op",
}
SUBMETRIC_DESC_KO = {
    "HG_score": "형식/스키마/길이 게이트 통과 안정성",
    "Finite_score": "NaN/Inf 없이 계산 가능한 비율",
    "Available_score": "필수 룰의 계산 가능 비율",
    "OUT_core": "정상 코어 구간의 조밀함",
    "OUT_rate": "fail/hard 기반 경계 이탈 비율 위험",
    "OUT_mass": "경계 이탈 이후 붕괴 깊이 위험",
    "RID_core": "설명가능한 이동의 코어 안정성",
    "RID_rate": "설명불가 이동의 발생 비율 위험",
    "RID_mass": "설명되지 않는 이동 강도 위험",
    "DIAG_core": "DIAG 집계에서 제외되는 참고 코어축(RID_core)",
    "DIAG_rate": "불안정 상태(Q10+Q01+Q11) 비율",
    "DIAG_mass": "q11 + 0.5*min(q10,q01) 기반 심각도",
    "DIAG_stable": "q00: Residual/Ridge 모두 안정인 비율(높을수록 좋음)",
    "DIAG_local": "q10: Residual 중심 분포형 이탈 비율",
    "DIAG_unexplainable": "q01: Ridge 중심 입력조건 예측 이탈 비율",
    "DIAG_systemic": "q11: 형태+예측 동시 붕괴 비율",
    "SEM_core": "의미/근거 구조의 코어 안정성",
    "SEM_rate": "의미 구조 이탈 비율 위험",
    "SEM_mass": "의미 구조 붕괴 심각도",
    "CONF_data": "샘플 지원도 기반 신뢰도",
    "CONF_calc": "available/finite/NA 상태 기반 신뢰도",
    "CONF_th": "threshold 안정성 기반 신뢰도",
    "CONF_op": "운영 마스크/적용 조건 반영",
}


@dataclass(frozen=True)
class RuleStat:
    rule: str
    signal_col: str
    available_col: str
    selected_method: str
    threshold_source: str
    support_rows: int
    signal: np.ndarray
    available_mask: np.ndarray
    finite_mask: np.ndarray
    valid_mask: np.ndarray
    m: float
    sigma: float
    scale_collapse: bool
    fail_threshold_raw: float
    hard_threshold_raw: float
    k_fail: float
    k_hard: float
    z: np.ndarray
    pass_mask: np.ndarray
    fail_mask: np.ndarray
    hard_mask: np.ndarray
    p_fail: float
    p_hard: float
    tail_mass: float
    mass_mode: str
    mass_scale: float
    mass_cap: float
    core_mad: float
    n_valid: int


@dataclass(frozen=True)
class SubMetric:
    bundle: str
    name: str
    kind: str
    raw_value: float
    score: int
    bucket_source: str
    detail: dict[str, Any]


@dataclass(frozen=True)
class BundleScoreArtifacts:
    summary_df: pd.DataFrame
    detail_df: pd.DataFrame
    payload: dict[str, Any]


def to_bool_array(series: pd.Series | np.ndarray | list[Any] | bool, size: int) -> np.ndarray:
    if isinstance(series, bool):
        return np.full(size, bool(series), dtype=bool)
    arr = pd.Series(series) if not isinstance(series, pd.Series) else series
    if len(arr) != size:
        out = np.zeros(size, dtype=bool)
        n = min(size, len(arr))
        if n > 0:
            out[:n] = to_bool_array(arr.iloc[:n], n)
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


def robust_center_scale(values: np.ndarray, eps: float = EPS) -> tuple[float, float, bool]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), True
    m = float(np.median(x))
    mad = float(np.median(np.abs(x - m)))
    sigma = float(1.4826 * mad)
    scale_collapse = (not np.isfinite(sigma)) or sigma <= float(eps)
    if scale_collapse:
        sigma = float(eps)
    return m, sigma, scale_collapse


def mad(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def bucket_pass_rate(rate: float) -> int:
    if not np.isfinite(rate):
        return 0
    r = float(np.clip(rate, 0.0, 1.0))
    if r >= 0.99:
        return 5
    if r >= 0.95:
        return 4
    if r >= 0.90:
        return 3
    if r >= 0.80:
        return 2
    if r >= 0.60:
        return 1
    return 0


def bucket_pass_rate_relaxed(rate: float) -> int:
    if not np.isfinite(rate):
        return 0
    r = float(np.clip(rate, 0.0, 1.0))
    if r >= 0.95:
        return 5
    if r >= 0.85:
        return 4
    if r >= 0.70:
        return 3
    if r >= 0.50:
        return 2
    if r >= 0.30:
        return 1
    return 0


def bucket_good_ratio(rate: float) -> int:
    if not np.isfinite(rate):
        return 0
    r = float(np.clip(rate, 0.0, 1.0))
    if r >= 0.90:
        return 5
    if r >= 0.80:
        return 4
    if r >= 0.70:
        return 3
    if r >= 0.55:
        return 2
    if r >= 0.40:
        return 1
    return 0


def bucket_good_with_cuts(value: float, cuts_desc: tuple[float, float, float, float, float]) -> int:
    """Map goodness ratio [0,1] to 0-5 with descending cut thresholds."""
    if not np.isfinite(value):
        return 0
    v = float(np.clip(value, 0.0, 1.0))
    if v >= float(cuts_desc[0]):
        return 5
    if v >= float(cuts_desc[1]):
        return 4
    if v >= float(cuts_desc[2]):
        return 3
    if v >= float(cuts_desc[3]):
        return 2
    if v >= float(cuts_desc[4]):
        return 1
    return 0


def or_risk_aggregate(scores: list[float], weights: list[float]) -> tuple[float, float]:
    """Aggregate sub-scores with OR-risk composition.

    NOTE:
    This follows the vNext triplet policy:
    ``r_bundle = 1 - Π(1 - w_i * r_i)`` where ``r_i = 1 - score_i/5``.
    Weights are normalized to sum to 1 across valid axes.
    """
    if len(scores) != len(weights) or len(scores) == 0:
        return float("nan"), float("nan")
    s = np.asarray(scores, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(s) & np.isfinite(w)
    if not np.any(mask):
        return float("nan"), float("nan")
    s = s[mask]
    w = np.clip(w[mask], 0.0, np.inf)
    if float(np.sum(w)) <= 0.0:
        w = np.full(shape=s.shape, fill_value=(1.0 / float(max(1, s.size))), dtype=float)
    else:
        w = w / float(np.sum(w))
    r = np.clip(1.0 - (s / 5.0), 0.0, 1.0)
    terms = np.clip(1.0 - (w * r), 0.0, 1.0)
    bundle_risk = float(1.0 - np.prod(terms))
    bundle_score = float(np.clip(np.rint(5.0 * (1.0 - bundle_risk)), 0.0, 5.0))
    return bundle_score, bundle_risk


def quantile_bucket_cuts(
    values: list[float] | np.ndarray,
    quantiles: tuple[float, ...] = (0.50, 0.75, 0.90, 0.95, 0.99),
    fallback: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20, 0.40),
) -> tuple[float, float, float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return tuple(float(x) for x in fallback)  # type: ignore[return-value]
    qs = [float(np.quantile(arr, q)) for q in quantiles]
    # Force monotonic non-decreasing.
    for i in range(1, len(qs)):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return tuple(qs)  # type: ignore[return-value]


def bucket_inverse(value: float, cuts: tuple[float, float, float, float, float]) -> int:
    if not np.isfinite(value):
        return 0
    v = float(value)
    if v <= cuts[0]:
        return 5
    if v <= cuts[1]:
        return 4
    if v <= cuts[2]:
        return 3
    if v <= cuts[3]:
        return 2
    if v <= cuts[4]:
        return 1
    return 0


def bucket_inverse_relaxed(
    value: float,
    cuts: tuple[float, float, float, float, float],
    *,
    floor_score: int = 1,
    zero_cutoff: float = 0.95,
    is_catastrophic: bool = False,
) -> int:
    """Relax inverse bucket so 0-score is reserved for truly extreme risks."""
    base = int(bucket_inverse(value, cuts))
    if base > 0:
        return base
    if not np.isfinite(value):
        return 0
    floor = int(np.clip(int(floor_score), 0, 5))
    if floor <= 0:
        return 0
    if bool(is_catastrophic) and np.isfinite(zero_cutoff) and float(value) >= float(zero_cutoff):
        return 0
    return floor


def bucket_good(value: float, cuts: tuple[float, float, float, float, float]) -> int:
    if not np.isfinite(value):
        return 0
    v = float(value)
    if v >= cuts[4]:
        return 5
    if v >= cuts[3]:
        return 4
    if v >= cuts[2]:
        return 3
    if v >= cuts[1]:
        return 2
    if v >= cuts[0]:
        return 1
    return 0


def clamp_score(x: float) -> int:
    if not np.isfinite(x):
        return 0
    return int(np.clip(int(np.rint(float(x))), 0, 5))


def clamp_score_value(x: float) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(np.clip(float(x), 0.0, 5.0))


def round4(x: float) -> float:
    if not np.isfinite(x):
        return float("nan")
    return float(np.round(float(x), 4))


def apply_non_catastrophic_floor(
    score: int,
    *,
    floor_score: int,
    is_catastrophic: bool,
) -> tuple[int, bool]:
    """Apply a minimum floor only when the case is not catastrophic."""
    s = int(np.clip(int(score), 0, 5))
    floor = int(np.clip(int(floor_score), 0, 5))
    if bool(is_catastrophic):
        return s, False
    if s < floor:
        return floor, True
    return s, False


def has_meaningful_hard_events(
    *,
    p_hard: float,
    n_valid: int,
    hard_rate_cutoff: float,
    min_hard_count: int,
    strict_gt: bool = False,
) -> bool:
    """True when hard events are both frequent enough and count-significant."""
    if (not np.isfinite(p_hard)) or n_valid <= 0:
        return False
    hard_count = float(p_hard) * float(max(0, int(n_valid)))
    rate_ok = float(p_hard) > float(max(0.0, hard_rate_cutoff)) if bool(strict_gt) else float(p_hard) >= float(max(0.0, hard_rate_cutoff))
    return bool(
        rate_ok
        and hard_count >= float(max(1, int(min_hard_count)))
    )


def effective_out_catastrophic_cutoff(
    *,
    n_valid: int,
    p_hard_base: float,
    n_correction_k: float,
) -> float:
    if n_valid <= 0:
        return float(max(0.0, p_hard_base))
    return float(max(float(max(0.0, p_hard_base)), float(n_correction_k) / float(max(1, int(n_valid)))))


def is_out_catastrophic_case(
    *,
    p_hard: float,
    n_valid: int,
    mass_norm_raw: float,
    hard_rate_cutoff_effective: float,
    min_hard_count: int,
    mass_norm_raw_cutoff: float,
    strict_gt: bool = True,
) -> bool:
    """OUT catastrophic: meaningful hard events + high raw mass risk."""
    meaningful_hard = has_meaningful_hard_events(
        p_hard=p_hard,
        n_valid=n_valid,
        hard_rate_cutoff=hard_rate_cutoff_effective,
        min_hard_count=min_hard_count,
        strict_gt=bool(strict_gt),
    )
    if not np.isfinite(mass_norm_raw):
        return False
    mass_cut = float(max(0.0, mass_norm_raw_cutoff))
    mass_high = bool(float(mass_norm_raw) > mass_cut) if bool(strict_gt) else bool(float(mass_norm_raw) >= mass_cut)
    return bool(meaningful_hard and mass_high)


def should_apply_out_bundle_floor(
    *,
    floor_score: int,
    is_catastrophic: bool,
    out_rate_raw: float,
    out_mass_norm: float,
    max_rate_raw: float,
    max_mass_norm: float,
) -> bool:
    """Apply OUT floor only in clearly low-risk non-catastrophic zone."""
    floor = int(np.clip(int(floor_score), 0, 5))
    if floor <= 0 or bool(is_catastrophic):
        return False
    return bool(
        np.isfinite(out_rate_raw)
        and np.isfinite(out_mass_norm)
        and float(out_rate_raw) <= float(max(0.0, max_rate_raw))
        and float(out_mass_norm) <= float(max(0.0, max_mass_norm))
    )


def build_na_annotation(
    *,
    bundle: str,
    submetric: str,
    bucket_source: str,
    raw_value: float,
    bucket_value: float | None,
    detail: dict[str, Any] | None,
    na_short_circuit: dict[str, dict[str, Any]],
) -> tuple[bool, str, str]:
    """Derive human-readable NA annotation for detail rows."""
    source = str(bucket_source or "")
    raw_finite = np.isfinite(float(raw_value))
    bucket_finite = bucket_value is not None and np.isfinite(float(bucket_value))
    is_na = bool(source in {"na", "na_neutral"} or (not raw_finite) or (not bucket_finite))
    if not is_na:
        return False, "", ""

    reasons: list[str] = []
    if source:
        reasons.append(source)

    short = na_short_circuit.get(bundle, {})
    if bool(short.get("enabled", False)):
        axes = short.get("axes", [])
        if isinstance(axes, list) and str(submetric) in [str(x) for x in axes]:
            r = str(short.get("reason", "")).strip()
            if r:
                reasons.append(r)

    meta = detail or {}
    for key in ("reason", "edge_score_reason"):
        v = str(meta.get(key, "")).strip()
        if v:
            reasons.append(v)

    dedup: list[str] = []
    for r in reasons:
        if r and r not in dedup:
            dedup.append(r)

    if "insufficient_edge_samples" in dedup:
        label = "NA(엣지 샘플 부족)"
    elif "required_axis_na" in dedup:
        label = "NA(필수축 계산불가)"
    elif "sem_zero_dominance" in dedup:
        label = "NA(SEM 정보량 부족)"
    elif source == "na_neutral":
        label = "NA(중립처리)"
    else:
        label = "NA(미산출)"

    return True, label, ";".join(dedup)


def derive_subscore_precise(sub: SubMetric) -> float:
    """Continuous 0~5 value used for visualization (falls back to bucketed score)."""
    bucket_value = sub.detail.get("bucket_value", None)
    kind = str(sub.kind or "")
    if bucket_value is not None and np.isfinite(float(bucket_value)):
        b = float(bucket_value)
        if kind in {"continuous_good", "inverse"}:
            # bucket_value is normalized risk [0,1] for these kinds.
            good = float(np.clip(1.0 - b, 0.0, 1.0))
            return float(np.clip(5.0 * good, 0.0, 5.0))
        if kind in {"rate_good", "good_ratio_relaxed", "pass_rate_relaxed"}:
            # bucket_value is normalized goodness [0,1] for these kinds.
            good = float(np.clip(b, 0.0, 1.0))
            return float(np.clip(5.0 * good, 0.0, 5.0))
    return float(np.clip(float(sub.score), 0.0, 5.0))


def weighted_score(parts: list[tuple[float, float]]) -> int:
    valid = [(w, v) for w, v in parts if np.isfinite(v)]
    if not valid:
        return 0
    ws = np.asarray([w for w, _ in valid], dtype=float)
    vs = np.asarray([v for _, v in valid], dtype=float)
    denom = float(np.sum(ws))
    if denom <= 0.0:
        return 0
    return clamp_score(float(np.sum(ws * vs) / denom))


def weighted_score_value(parts: list[tuple[float, float]]) -> float:
    valid = [(w, v) for w, v in parts if np.isfinite(v)]
    if not valid:
        return float("nan")
    ws = np.asarray([w for w, _ in valid], dtype=float)
    vs = np.asarray([v for _, v in valid], dtype=float)
    denom = float(np.sum(ws))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(ws * vs) / denom)


def weighted_minus_zero_penalty_value(
    parts: list[tuple[float, float]],
    *,
    zero_penalty: float,
    max_penalty: float | None = None,
) -> tuple[float, int, float, float]:
    base = weighted_score_value(parts)
    zero_count = int(np.sum([1 for _, v in parts if np.isfinite(v) and float(v) <= 0.0]))
    penalty = float(zero_penalty) * float(zero_count)
    if max_penalty is not None and np.isfinite(float(max_penalty)):
        penalty = float(min(max(0.0, float(max_penalty)), max(0.0, penalty)))
    penalized = float(base) - penalty if np.isfinite(base) else float("nan")
    return float(base), int(zero_count), float(penalty), float(penalized)


def min_plus_consistency_score(
    parts: list[tuple[float, float]],
    *,
    tight_spread: float = 1.0,
    consistency_bonus: int = 1,
) -> int:
    vals = [float(v) for _, v in parts if np.isfinite(v)]
    if not vals:
        return 0
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    bonus = int(consistency_bonus) if (hi - lo) <= float(tight_spread) else 0
    return clamp_score(lo + float(bonus))


def guarded_weighted_score(parts: list[tuple[float, float]], *, min_guard_margin: int) -> int:
    base_score = weighted_score(parts)
    vals = [float(v) for _, v in parts if np.isfinite(v)]
    if not vals:
        return int(base_score)
    floor_score = float(np.min(vals))
    margin = max(0, int(min_guard_margin))
    cap = int(np.floor(floor_score + float(margin)))
    cap = int(np.clip(cap, 0, 5))
    return int(min(base_score, cap))


def state_array(rule: RuleStat) -> np.ndarray:
    out = np.full(len(rule.signal), -1, dtype=int)
    out[rule.valid_mask & rule.pass_mask] = 0
    out[rule.valid_mask & rule.fail_mask] = 1
    out[rule.valid_mask & rule.hard_mask] = 2
    return out


def classify_rule_states(
    z: np.ndarray,
    *,
    k_fail: float,
    k_hard: float,
    valid_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    valid = np.asarray(valid_mask, dtype=bool)
    zz = np.asarray(z, dtype=float)
    pass_mask = np.zeros(len(zz), dtype=bool)
    fail_mask = np.zeros(len(zz), dtype=bool)
    hard_mask = np.zeros(len(zz), dtype=bool)
    if not np.isfinite(k_fail):
        return {"pass": pass_mask, "fail": fail_mask, "hard": hard_mask}
    if np.isfinite(k_hard) and k_hard < k_fail:
        k_hard = k_fail
    hard_mask = valid & np.isfinite(zz) & (zz >= k_hard) if np.isfinite(k_hard) else np.zeros(len(zz), dtype=bool)
    fail_mask = valid & np.isfinite(zz) & (zz >= k_fail) & (~hard_mask)
    pass_mask = valid & np.isfinite(zz) & (zz < k_fail)
    return {"pass": pass_mask, "fail": fail_mask, "hard": hard_mask}


def compute_rule_common_metrics(
    z: np.ndarray,
    *,
    k_fail: float,
    k_hard: float,
    valid_mask: np.ndarray,
    eta: float = 3.0,
    z_mass_cap: float | None = None,
    signal: np.ndarray | None = None,
    fail_threshold_raw: float = float("nan"),
    hard_threshold_raw: float = float("nan"),
    center_raw: float = float("nan"),
    scale_collapse: bool = False,
    raw_mass_cap: float | None = None,
    raw_mass_kfail_trigger: float | None = 1e6,
) -> dict[str, float]:
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(np.asarray(z, dtype=float))
    if np.sum(valid) == 0 or (not np.isfinite(k_fail)):
        return {
            "p_fail": float("nan"),
            "p_hard": float("nan"),
            "tail_mass": float("nan"),
            "mass_mode": "na",
            "mass_scale": float("nan"),
            "mass_cap": float("nan"),
            "core_mad": float("nan"),
            "n_valid": int(np.sum(valid)),
        }
    zz = np.asarray(z, dtype=float)[valid]
    p_fail = float(np.mean(zz >= k_fail))
    p_hard = float(np.mean(zz >= k_hard)) if np.isfinite(k_hard) else 0.0
    use_raw_mass = bool(scale_collapse)
    if (
        (not use_raw_mass)
        and np.isfinite(raw_mass_kfail_trigger)
        and np.isfinite(k_fail)
        and abs(float(k_fail)) >= float(raw_mass_kfail_trigger)
    ):
        use_raw_mass = True
    if (signal is None) or (not np.isfinite(fail_threshold_raw)):
        use_raw_mass = False

    mass_mode = "z_excess"
    mass_scale = float("nan")
    if use_raw_mass:
        xx = np.asarray(signal, dtype=float)[valid]
        xx = xx[np.isfinite(xx)]
        if xx.size == 0:
            fail_excess = np.zeros_like(zz)
            hard_excess = np.zeros_like(zz)
            mass_mode = "z_excess"
        else:
            candidates: list[float] = []
            if np.isfinite(hard_threshold_raw) and hard_threshold_raw > fail_threshold_raw:
                candidates.append(float(hard_threshold_raw - fail_threshold_raw))
            if np.isfinite(center_raw):
                candidates.append(float(abs(fail_threshold_raw - center_raw)))
            candidates.append(float(abs(fail_threshold_raw)))
            candidates.append(1.0)
            mass_scale = float(max(float(EPS), np.nanmax(np.asarray(candidates, dtype=float))))
            fail_excess_raw = np.maximum(0.0, xx - float(fail_threshold_raw))
            hard_excess_raw = (
                np.maximum(0.0, xx - float(hard_threshold_raw))
                if np.isfinite(hard_threshold_raw)
                else np.zeros_like(xx)
            )
            fail_excess = np.log1p(fail_excess_raw / (mass_scale + EPS))
            hard_excess = np.log1p(hard_excess_raw / (mass_scale + EPS))
            if raw_mass_cap is not None and np.isfinite(float(raw_mass_cap)):
                cap = float(max(0.0, raw_mass_cap))
                fail_excess = np.minimum(fail_excess, cap)
                hard_excess = np.minimum(hard_excess, cap)
                mass_mode = "raw_log_ratio_capped"
            else:
                mass_mode = "raw_log_ratio"
    else:
        fail_excess = np.maximum(0.0, zz - k_fail)
        hard_excess = np.maximum(0.0, zz - k_hard) if np.isfinite(k_hard) else np.zeros_like(zz)
        if z_mass_cap is not None and np.isfinite(float(z_mass_cap)):
            cap = float(max(0.0, z_mass_cap))
            fail_excess = np.minimum(fail_excess, cap)
            hard_excess = np.minimum(hard_excess, cap)
            mass_mode = "z_excess_capped"
    mass = float(np.mean(fail_excess) + float(eta) * np.mean(hard_excess))
    core = zz[zz < k_fail]
    return {
        "p_fail": p_fail,
        "p_hard": p_hard,
        "tail_mass": mass,
        "mass_mode": mass_mode,
        "mass_scale": mass_scale,
        "mass_cap": (
            float(raw_mass_cap)
            if mass_mode.startswith("raw_") and raw_mass_cap is not None and np.isfinite(float(raw_mass_cap))
            else (
                float(z_mass_cap)
                if mass_mode.startswith("z_") and z_mass_cap is not None and np.isfinite(float(z_mass_cap))
                else float("nan")
            )
        ),
        "core_mad": mad(core),
        "n_valid": int(np.sum(valid)),
    }


def classify_diag_quadrants(
    diff_state: np.ndarray,
    ridge_state: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, float]:
    valid = np.asarray(valid_mask, dtype=bool)
    n = int(np.sum(valid))
    if n <= 0:
        return {"q00": float("nan"), "q10": float("nan"), "q01": float("nan"), "q11": float("nan")}
    diff = np.asarray(diff_state, dtype=int)
    ridge = np.asarray(ridge_state, dtype=int)
    q00 = float(np.mean((diff == 0) & (ridge == 0) & valid))
    q10 = float(np.mean((diff >= 1) & (ridge == 0) & valid))
    q01 = float(np.mean((diff == 0) & (ridge >= 1) & valid))
    q11 = float(np.mean((diff >= 1) & (ridge >= 1) & valid))
    # Mean() above used full length; convert to valid denominator.
    scale = float(len(valid) / n)
    return {
        "q00": q00 * scale,
        "q10": q10 * scale,
        "q01": q01 * scale,
        "q11": q11 * scale,
    }


def confidence_to_hex(conf: float) -> str:
    if not np.isfinite(conf):
        return "#9ca3af"
    t = float(np.clip(conf / 5.0, 0.0, 1.0))
    if t <= 0.5:
        u = t / 0.5
        r, g, b = 255, int(round(255.0 * u)), 0
    else:
        u = (t - 0.5) / 0.5
        r, g, b = int(round(255.0 * (1.0 - u))), 255, 0
    return f"#{r:02x}{g:02x}{b:02x}"


def canonical_output(value: Any) -> str:
    text = "" if value is None else str(value)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, (dict, list)):
            return json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        pass
    return " ".join(text.split())


def compute_duplicate_rate(outputs: pd.Series | list[str]) -> tuple[float, list[dict[str, Any]]]:
    vals = pd.Series(outputs).fillna("").astype(str).tolist()
    n = len(vals)
    if n == 0:
        return float("nan"), []
    canonical = [canonical_output(v) for v in vals]
    hashed = [hashlib.sha256(v.encode("utf-8", errors="ignore")).hexdigest() for v in canonical]
    counts = pd.Series(hashed).value_counts(dropna=False)
    unique_n = int(counts.shape[0])
    dup_rate = 1.0 - (float(unique_n) / float(n))
    top = [
        {"hash": str(h), "count": int(c), "rate": float(c / n)}
        for h, c in counts.head(10).items()
        if int(c) >= 2
    ]
    return float(dup_rate), top


def compute_iosens(
    input_norm: np.ndarray,
    output_norm: np.ndarray,
    valid_mask: np.ndarray,
    k_neighbors: int,
) -> tuple[float, int]:
    valid = np.asarray(valid_mask, dtype=bool)
    idx = np.where(valid)[0]
    if idx.size < 3:
        return float("nan"), 0
    xin = np.asarray(input_norm[idx, :], dtype=float)
    yout = np.asarray(output_norm[idx, :], dtype=float)
    if xin.ndim != 2 or yout.ndim != 2 or xin.shape != yout.shape:
        return float("nan"), 0
    if not np.all(np.isfinite(xin)) or not np.all(np.isfinite(yout)):
        return float("nan"), 0
    k = int(min(max(2, k_neighbors), idx.size - 1))
    dist_x, nbr = knn_self(xin, n_neighbors=k, metric="euclidean")
    if dist_x.size == 0:
        return float("nan"), 0
    rows = np.arange(idx.size)[:, None]
    nbr_y = yout[nbr]
    cur_y = yout[rows]
    dy = np.linalg.norm(nbr_y - cur_y, axis=2)
    med_dx = np.median(dist_x, axis=1)
    med_dy = np.median(dy, axis=1)
    ratio = med_dy / (med_dx + EPS)
    ratio = ratio[np.isfinite(ratio)]
    if ratio.size == 0:
        return float("nan"), 0
    return float(np.median(ratio)), int(ratio.size)


def _threshold_map(threshold_summary_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if threshold_summary_df is None or threshold_summary_df.empty:
        return out
    for _, row in threshold_summary_df.iterrows():
        rule = str(row.get("rule", "")).strip().lower()
        if not rule:
            continue
        out[rule] = dict(row)
    return out


def _finite_float(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    return np.where(np.isfinite(x), x, np.nan)


def _build_rule_stat(
    row_df: pd.DataFrame,
    threshold_map: dict[str, dict[str, Any]],
    rule: str,
    hard_gate: np.ndarray,
    eta: float,
    z_mass_cap: float | None,
    raw_mass_cap: float | None,
    raw_mass_kfail_trigger: float | None,
) -> RuleStat:
    n = len(row_df)
    signal_col = str(RUNTIME_RULE_SIGNAL_COL_NOMASK.get(rule, f"{rule}_signal_nomask"))
    available_col = str(RUNTIME_RULE_AVAILABLE_COL_NOMASK.get(rule, ""))

    # Handle both Series and scalar values from row_df.get()
    signal_raw = row_df.get(signal_col, np.nan)
    if isinstance(signal_raw, (int, float, np.number)):
        # Scalar value - broadcast to array of length n
        signal = np.full(n, signal_raw, dtype=float)
    else:
        # Series or other - use pd.to_numeric
        signal = pd.to_numeric(signal_raw, errors="coerce")
        if isinstance(signal, (int, float, np.number)):
            signal = np.full(n, signal, dtype=float)
        else:
            signal = signal.to_numpy(dtype=float)
    available = (
        to_bool_array(row_df.get(available_col, True), n)
        if available_col and available_col in row_df.columns
        else np.ones(n, dtype=bool)
    )
    finite = np.isfinite(signal)
    valid = np.asarray(hard_gate, dtype=bool) & available & finite
    m, sigma, scale_collapse = robust_center_scale(signal[valid], eps=EPS)

    thr_row = threshold_map.get(rule, {})
    fail_threshold = float(
        thr_row.get(
            "tail_start_threshold",
            thr_row.get("fail_threshold", np.nan),
        )
    )
    hard_threshold = float(
        thr_row.get(
            "exceptional_out_threshold",
            thr_row.get("hard_fail_threshold", np.nan),
        )
    )
    if (not np.isfinite(fail_threshold)) and f"{rule}_tail_start_threshold_nomask" in row_df.columns:
        vals = pd.to_numeric(row_df[f"{rule}_tail_start_threshold_nomask"], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            fail_threshold = float(vals[0])
    if (not np.isfinite(hard_threshold)) and f"{rule}_hard_fail_threshold_nomask" in row_df.columns:
        vals = pd.to_numeric(row_df[f"{rule}_hard_fail_threshold_nomask"], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            hard_threshold = float(vals[0])
    if (not np.isfinite(hard_threshold)) and f"{rule}_exceptional_out_threshold_nomask" in row_df.columns:
        vals = pd.to_numeric(row_df[f"{rule}_exceptional_out_threshold_nomask"], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            hard_threshold = float(vals[0])

    z = (signal - m) / (sigma + EPS) if np.isfinite(m) and np.isfinite(sigma) else np.full(n, np.nan, dtype=float)
    if np.isfinite(fail_threshold) and np.isfinite(m) and np.isfinite(sigma):
        k_fail = float((fail_threshold - m) / (sigma + EPS))
    else:
        k_fail = float("nan")
    if np.isfinite(hard_threshold) and np.isfinite(m) and np.isfinite(sigma):
        k_hard = float((hard_threshold - m) / (sigma + EPS))
    else:
        k_hard = float("nan")
    if np.isfinite(k_fail) and np.isfinite(k_hard) and k_hard < k_fail:
        k_hard = k_fail

    states = classify_rule_states(z=z, k_fail=k_fail, k_hard=k_hard, valid_mask=valid)
    metrics = compute_rule_common_metrics(
        z=z,
        k_fail=k_fail,
        k_hard=k_hard,
        valid_mask=valid,
        eta=eta,
        z_mass_cap=z_mass_cap,
        signal=signal,
        fail_threshold_raw=fail_threshold,
        hard_threshold_raw=hard_threshold,
        center_raw=m,
        scale_collapse=scale_collapse,
        raw_mass_cap=raw_mass_cap,
        raw_mass_kfail_trigger=raw_mass_kfail_trigger,
    )
    pass_mask = states["pass"]
    fail_mask = states["fail"]
    hard_mask = states["hard"]
    p_fail = float(metrics["p_fail"])
    p_hard = float(metrics["p_hard"])
    tail_mass = float(metrics["tail_mass"])
    mass_mode = str(metrics.get("mass_mode", "z_excess"))
    mass_scale = float(metrics.get("mass_scale", np.nan))
    mass_cap = float(metrics.get("mass_cap", np.nan))
    core_mad = float(metrics["core_mad"])
    n_valid = int(metrics["n_valid"])

    return RuleStat(
        rule=rule,
        signal_col=signal_col,
        available_col=available_col,
        selected_method=str(thr_row.get("selected_method", "missing")),
        threshold_source=str(thr_row.get("threshold_source", "missing")),
        support_rows=int(thr_row.get("support_rows", n_valid) or 0),
        signal=signal,
        available_mask=available,
        finite_mask=finite,
        valid_mask=valid,
        m=float(m),
        sigma=float(sigma),
        scale_collapse=bool(scale_collapse),
        fail_threshold_raw=float(fail_threshold),
        hard_threshold_raw=float(hard_threshold),
        k_fail=float(k_fail),
        k_hard=float(k_hard),
        z=_finite_float(z),
        pass_mask=pass_mask,
        fail_mask=fail_mask,
        hard_mask=hard_mask,
        p_fail=float(p_fail),
        p_hard=float(p_hard),
        tail_mass=float(tail_mass),
        mass_mode=mass_mode,
        mass_scale=mass_scale,
        mass_cap=mass_cap,
        core_mad=float(core_mad),
        n_valid=n_valid,
    )


def _risk_in_group(rule: RuleStat, group_mask: np.ndarray, eta: float, z_mass_cap: float | None) -> float:
    mask = np.asarray(group_mask, dtype=bool) & rule.valid_mask & np.isfinite(rule.z)
    if np.sum(mask) == 0 or (not np.isfinite(rule.k_fail)):
        return float("nan")
    z = rule.z[mask]
    p_fail = float(np.mean(z >= rule.k_fail))
    p_hard = float(np.mean(z >= rule.k_hard)) if np.isfinite(rule.k_hard) else 0.0
    if str(rule.mass_mode).startswith("raw_") and np.isfinite(rule.fail_threshold_raw):
        x = np.asarray(rule.signal, dtype=float)[mask]
        x = x[np.isfinite(x)]
        if x.size == 0:
            return float("nan")
        mass_scale = float(rule.mass_scale) if np.isfinite(rule.mass_scale) and rule.mass_scale > 0.0 else 1.0
        fail_excess_raw = np.maximum(0.0, x - float(rule.fail_threshold_raw))
        hard_excess_raw = (
            np.maximum(0.0, x - float(rule.hard_threshold_raw))
            if np.isfinite(rule.hard_threshold_raw)
            else np.zeros_like(x)
        )
        fail_excess = np.log1p(fail_excess_raw / (mass_scale + EPS))
        hard_excess = np.log1p(hard_excess_raw / (mass_scale + EPS))
        if np.isfinite(rule.mass_cap):
            cap = float(max(0.0, rule.mass_cap))
            fail_excess = np.minimum(fail_excess, cap)
            hard_excess = np.minimum(hard_excess, cap)
    else:
        fail_excess = np.maximum(0.0, z - rule.k_fail)
        hard_excess = np.maximum(0.0, z - rule.k_hard)
        if z_mass_cap is not None and np.isfinite(float(z_mass_cap)):
            cap = float(max(0.0, z_mass_cap))
            fail_excess = np.minimum(fail_excess, cap)
            hard_excess = np.minimum(hard_excess, cap)
    mass = float(
        np.mean(fail_excess)
        + float(eta) * np.mean(hard_excess)
    )
    return float((p_fail + 3.0 * p_hard) + 0.5 * mass)


def _safe_rate(mask: np.ndarray) -> float:
    arr = np.asarray(mask, dtype=bool)
    return float(np.mean(arr)) if arr.size > 0 else float("nan")


def _rate_on_base(mask: np.ndarray, base: np.ndarray) -> float:
    b = np.asarray(base, dtype=bool)
    if np.sum(b) <= 0:
        return float("nan")
    m = np.asarray(mask, dtype=bool)
    if len(m) != len(b):
        return float("nan")
    return float(np.mean(m[b]))


def _nanmean_or_nan(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _nanmax_or_nan(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.max(arr))


def _normalize_inverse_risk_values(
    *,
    raw_values: dict[str, float],
    inverse_keys: list[str],
    use_log_scale: bool,
    clip_quantile: float,
) -> tuple[dict[str, float], dict[str, float]]:
    transformed: dict[str, float] = {}
    finite_vals: list[float] = []
    for key in inverse_keys:
        raw = float(raw_values.get(key, np.nan))
        if not np.isfinite(raw):
            transformed[key] = float("nan")
            continue
        x = max(0.0, raw)
        if use_log_scale:
            x = float(np.log1p(x))
        transformed[key] = float(x)
        finite_vals.append(float(x))

    if not finite_vals:
        return (
            {k: float("nan") for k in inverse_keys},
            {
                "use_log_scale": float(1.0 if use_log_scale else 0.0),
                "clip_quantile": float(np.clip(clip_quantile, 0.50, 1.0)),
                "center": float("nan"),
                "scale": float("nan"),
                "clip_cap": float("nan"),
            },
        )

    arr = np.asarray(finite_vals, dtype=float)
    center, scale, scale_collapse = robust_center_scale(arr, eps=EPS)
    if not np.isfinite(center):
        center = float(np.nanmedian(arr))
    if (not np.isfinite(scale)) or scale <= float(EPS):
        scale = float(np.nanstd(arr)) if np.isfinite(np.nanstd(arr)) and np.nanstd(arr) > float(EPS) else 1.0

    # Positive-side robust z risk.
    pos_z: dict[str, float] = {}
    pos_vals: list[float] = []
    for key, x in transformed.items():
        if not np.isfinite(x):
            pos_z[key] = float("nan")
            continue
        v = float(max(0.0, (x - center) / (scale + EPS)))
        pos_z[key] = v
        pos_vals.append(v)

    q = float(np.clip(clip_quantile, 0.50, 1.0))
    if pos_vals:
        z_arr = np.asarray(pos_vals, dtype=float)
        if q < 1.0:
            cap = float(np.quantile(z_arr, q))
        else:
            cap = float(np.max(z_arr))
        if (not np.isfinite(cap)) or cap <= float(EPS):
            cap = float(np.max(z_arr)) if np.max(z_arr) > float(EPS) else 1.0
    else:
        cap = 1.0

    norm: dict[str, float] = {}
    for key, z in pos_z.items():
        if not np.isfinite(z):
            norm[key] = float("nan")
            continue
        norm[key] = float(np.clip(min(z, cap) / (cap + EPS), 0.0, 1.0))

    meta = {
        "use_log_scale": float(1.0 if use_log_scale else 0.0),
        "clip_quantile": float(q),
        "center": float(center),
        "scale": float(scale),
        "scale_collapse": float(1.0 if scale_collapse else 0.0),
        "clip_cap": float(cap),
    }
    return norm, meta


# New score runtime (2026-03): continuous risk axes + noisy-OR bundle aggregation.


def _clip01(value: float) -> float:
    if not np.isfinite(value):
        return float("nan")
    return float(np.clip(float(value), 0.0, 1.0))


def _score_to_int(score: float) -> int:
    return int(np.clip(np.rint(float(score if np.isfinite(score) else 0.0)), 0, 5))


def _quality_from_risk(risk: float, gamma: float = 1.0) -> float:
    if not np.isfinite(risk):
        return float("nan")
    r = float(np.clip(risk, 0.0, 1.0))
    g = float(max(0.1, gamma))
    return float(np.clip(5.0 * ((1.0 - r) ** g), 0.0, 5.0))


def _risk_from_scale(raw: float, scale: float) -> float:
    if (not np.isfinite(raw)) or (not np.isfinite(scale)):
        return float("nan")
    r = float(max(0.0, raw))
    s = float(max(float(EPS), scale))
    return float(np.clip(r / (r + s), 0.0, 1.0))


def _band_label_from_rates(p_fail: float, p_hard: float, hard_cut: float) -> str:
    if np.isfinite(p_hard) and float(p_hard) >= float(max(0.0, hard_cut)):
        return "exceptional"
    if np.isfinite(p_fail) and float(p_fail) > 0.0:
        return "tail"
    return "core"


def _bundle_risk_noisy_or(axis_risks: dict[str, float], axis_weights: dict[str, float]) -> tuple[float, float]:
    pairs: list[tuple[float, float]] = []
    for key, risk in axis_risks.items():
        if (not np.isfinite(risk)) or key not in axis_weights:
            continue
        w = float(max(0.0, axis_weights[key]))
        if w <= 0.0:
            continue
        pairs.append((w, float(np.clip(risk, 0.0, 1.0))))
    if not pairs:
        return float("nan"), float("nan")
    ws = np.asarray([w for w, _ in pairs], dtype=float)
    rs = np.asarray([r for _, r in pairs], dtype=float)
    ws = ws / float(np.sum(ws))
    terms = np.clip(1.0 - (ws * rs), 0.0, 1.0)
    risk = float(1.0 - np.prod(terms))
    score = _quality_from_risk(risk, gamma=1.0)
    return score, risk


def _apply_catastrophic_guard(
    *,
    score: float,
    p_hard: float,
    hard_zero_cutoff: float,
    hard_cap_cutoff: float,
) -> tuple[float, str]:
    if not np.isfinite(score):
        return float("nan"), "score_nan"
    if np.isfinite(p_hard) and float(p_hard) >= float(hard_zero_cutoff):
        return 0.0, "hard_zero"
    if np.isfinite(p_hard) and float(p_hard) >= float(hard_cap_cutoff):
        return float(min(float(score), 1.0)), "hard_cap"
    return float(score), "none"


def _apply_soft_hard_tail_penalty(
    *,
    base_risk: float,
    p_hard: float,
    mass_risk: float,
    alpha: float,
) -> tuple[float, float]:
    if not np.isfinite(base_risk):
        return float("nan"), float("nan")
    risk_base = float(np.clip(base_risk, 0.0, 1.0))
    hard_rate = float(np.clip(p_hard, 0.0, 1.0)) if np.isfinite(p_hard) else 0.0
    mass = float(np.clip(mass_risk, 0.0, 1.0)) if np.isfinite(mass_risk) else 0.0
    alpha_n = float(max(0.0, alpha))
    # Increase penalty when hard events and tail thickness are both high.
    penalty = float(alpha_n * hard_rate * (0.5 + 0.5 * mass))
    risk = float(np.clip(risk_base + penalty, 0.0, 1.0))
    return risk, penalty


def _submetric_from_risk(
    *,
    bundle: str,
    name: str,
    raw_value: float,
    risk: float,
    neutral_score: float,
    detail: dict[str, Any] | None = None,
) -> SubMetric:
    info = dict(detail or {})
    if np.isfinite(risk):
        risk_n = float(np.clip(risk, 0.0, 1.0))
        score = _score_to_int(_quality_from_risk(risk_n, gamma=1.0))
        info.update(
            {
                "bucket_value": risk_n,
                "bucket_value_is": "risk_norm_used",
                "risk_norm_raw": risk_n,
                "risk_norm_used": risk_n,
                "good_norm_used": float(np.clip(1.0 - risk_n, 0.0, 1.0)),
            }
        )
        return SubMetric(
            bundle=bundle,
            name=name,
            kind="continuous_good",
            raw_value=float(raw_value),
            score=int(score),
            bucket_source="new_score",
            detail=info,
        )
    score = _score_to_int(float(neutral_score))
    info.update(
        {
            "bucket_value": None,
            "bucket_value_is": "risk_norm_used",
            "risk_norm_raw": None,
            "risk_norm_used": None,
            "good_norm_used": None,
        }
    )
    return SubMetric(
        bundle=bundle,
        name=name,
        kind="continuous_good",
        raw_value=float(raw_value),
        score=int(score),
        bucket_source="na_neutral",
        detail=info,
    )


def _submetric_from_good(
    *,
    bundle: str,
    name: str,
    raw_value: float,
    good: float,
    neutral_score: float,
    detail: dict[str, Any] | None = None,
) -> SubMetric:
    info = dict(detail or {})
    if np.isfinite(good):
        good_n = float(np.clip(good, 0.0, 1.0))
        score = _score_to_int(5.0 * good_n)
        info.update(
            {
                "bucket_value": good_n,
                "bucket_value_is": "good_norm_used",
                "risk_norm_raw": float(np.clip(1.0 - good_n, 0.0, 1.0)),
                "risk_norm_used": float(np.clip(1.0 - good_n, 0.0, 1.0)),
                "good_norm_used": good_n,
            }
        )
        return SubMetric(
            bundle=bundle,
            name=name,
            kind="pass_rate_relaxed",
            raw_value=float(raw_value),
            score=int(score),
            bucket_source="new_score",
            detail=info,
        )
    score = _score_to_int(float(neutral_score))
    info.update(
        {
            "bucket_value": None,
            "bucket_value_is": "good_norm_used",
            "risk_norm_raw": None,
            "risk_norm_used": None,
            "good_norm_used": None,
        }
    )
    return SubMetric(
        bundle=bundle,
        name=name,
        kind="pass_rate_relaxed",
        raw_value=float(raw_value),
        score=int(score),
        bucket_source="na_neutral",
        detail=info,
    )


def _rule_axis_profile(
    *,
    stat: RuleStat,
    rate_lambda: float,
    s_core: float,
    s_mass: float,
    gamma: float,
    hard_band_cut: float,
) -> dict[str, float | str]:
    r_rate = _clip01(float(stat.p_fail + float(rate_lambda) * stat.p_hard)) if np.isfinite(stat.p_fail) else float("nan")
    r_mass = _risk_from_scale(float(stat.tail_mass), float(s_mass))
    r_core = _risk_from_scale(float(stat.core_mad), float(s_core))
    q_rate = _quality_from_risk(r_rate, gamma=gamma)
    q_mass = _quality_from_risk(r_mass, gamma=gamma)
    q_core = _quality_from_risk(r_core, gamma=gamma)
    return {
        "r_rate": float(r_rate),
        "r_mass": float(r_mass),
        "r_core": float(r_core),
        "q_rate": float(q_rate),
        "q_mass": float(q_mass),
        "q_core": float(q_core),
        "band": _band_label_from_rates(float(stat.p_fail), float(stat.p_hard), hard_cut=float(hard_band_cut)),
        "p_fail": float(stat.p_fail),
        "p_hard": float(stat.p_hard),
        "tail_mass": float(stat.tail_mass),
        "core_mad": float(stat.core_mad),
        "n_valid": float(stat.n_valid),
    }


def _sigmoid_safe(x: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(x, dtype=float)
    z = np.clip(arr, -60.0, 60.0)
    out = 1.0 / (1.0 + np.exp(-z))
    if np.isscalar(x):
        return float(out.item())
    return out


def _softplus_safe(x: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(x, dtype=float)
    out = np.log1p(np.exp(np.clip(arr, -60.0, 60.0)))
    if np.isscalar(x):
        return float(out.item())
    return out


def _state_rate_map(
    *,
    row_df: pd.DataFrame,
    rule: str,
    valid_mask: np.ndarray,
) -> dict[str, float]:
    state_col = f"{rule}_state_nomask"
    if state_col not in row_df.columns:
        return {
            "state_pass_rate": float("nan"),
            "state_warn_rate": float("nan"),
            "state_fail_rate": float("nan"),
            "state_hard_rate": float("nan"),
        }
    valid = np.asarray(valid_mask, dtype=bool)
    n_valid = int(np.sum(valid))
    if n_valid <= 0:
        return {
            "state_pass_rate": float("nan"),
            "state_warn_rate": float("nan"),
            "state_fail_rate": float("nan"),
            "state_hard_rate": float("nan"),
        }
    states = row_df[state_col].fillna("").astype(str).str.strip().str.lower().to_numpy()
    s = states[valid]
    return {
        "state_pass_rate": float(np.mean(s == "pass")),
        "state_warn_rate": float(np.mean(s == "warn")),
        "state_fail_rate": float(np.mean(s == "fail")),
        "state_hard_rate": float(np.mean(s == "hard_fail")),
    }


def _threshold_explain_profile(
    *,
    row_df: pd.DataFrame,
    rule: str,
    stat: RuleStat,
) -> dict[str, float | None]:
    valid = np.asarray(stat.valid_mask, dtype=bool) & np.isfinite(np.asarray(stat.z, dtype=float))
    n_valid = int(np.sum(valid))
    z_pos = np.maximum(0.0, np.asarray(stat.z, dtype=float))
    zv = z_pos[valid]

    k_fail = float(stat.k_fail) if np.isfinite(stat.k_fail) else float("nan")
    k_hard = float(stat.k_hard) if np.isfinite(stat.k_hard) else float("nan")
    if zv.size <= 0 or (not np.isfinite(k_fail)):
        rate_fail = float("nan")
        excess_fail = float("nan")
    else:
        rate_fail = float(np.mean(zv >= k_fail))
        excess_fail = float(np.mean(np.maximum(0.0, zv - k_fail)))
    if zv.size <= 0 or (not np.isfinite(k_hard)):
        rate_hard = float("nan")
        excess_hard = float("nan")
    else:
        rate_hard = float(np.mean(zv >= k_hard))
        excess_hard = float(np.mean(np.maximum(0.0, zv - k_hard)))

    out: dict[str, float | None] = {
        "n_valid": float(n_valid),
        "k_fail": (float(k_fail) if np.isfinite(k_fail) else None),
        "k_hard": (float(k_hard) if np.isfinite(k_hard) else None),
        "rate_fail": (float(rate_fail) if np.isfinite(rate_fail) else None),
        "rate_hard": (float(rate_hard) if np.isfinite(rate_hard) else None),
        "excess_fail": (float(excess_fail) if np.isfinite(excess_fail) else None),
        "excess_hard": (float(excess_hard) if np.isfinite(excess_hard) else None),
    }
    out.update(_state_rate_map(row_df=row_df, rule=rule, valid_mask=valid))
    return out


def _state_band_label(
    *,
    state_fail_rate: float,
    state_hard_rate: float,
) -> str:
    if np.isfinite(state_hard_rate) and float(state_hard_rate) > 0.0:
        return "exceptional"
    if np.isfinite(state_fail_rate) and float(state_fail_rate) > 0.0:
        return "tail"
    return "core"


def _apply_out_kfail_fallback(
    *,
    out_axes_soft: dict[str, Any],
    explain_output: dict[str, float | None],
    output_rule: RuleStat,
    rate_lambda_output: float,
    eta_hard: float,
    gamma: float,
    kfail_fallback_cutoff: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    used = dict(out_axes_soft)

    k_fail = float(output_rule.k_fail) if np.isfinite(output_rule.k_fail) else float("nan")
    cutoff = float(max(0.0, kfail_fallback_cutoff))
    state_fail = explain_output.get("state_fail_rate")
    state_hard = explain_output.get("state_hard_rate")
    state_fail_f = float(state_fail) if state_fail is not None and np.isfinite(float(state_fail)) else float("nan")
    state_hard_f = float(state_hard) if state_hard is not None and np.isfinite(float(state_hard)) else float("nan")

    soft_r_rate = float(out_axes_soft.get("r_rate", np.nan))
    soft_r_mass = float(out_axes_soft.get("r_mass", np.nan))
    soft_p_fail = float(out_axes_soft.get("p_fail", np.nan))
    soft_p_hard = float(out_axes_soft.get("p_hard", np.nan))

    applied = False
    reason = "below_cutoff"
    if np.isfinite(k_fail) and abs(k_fail) >= cutoff:
        if np.isfinite(state_fail_f) or np.isfinite(state_hard_f):
            sf = float(np.clip(state_fail_f, 0.0, 1.0)) if np.isfinite(state_fail_f) else 0.0
            sh = float(np.clip(state_hard_f, 0.0, 1.0)) if np.isfinite(state_hard_f) else 0.0
            used_r_rate = _clip01(sf + float(rate_lambda_output) * sh)
            used_r_mass = _clip01(sf + float(max(0.0, eta_hard)) * sh)
            used.update(
                {
                    "r_rate": float(used_r_rate),
                    "r_mass": float(used_r_mass),
                    "q_rate": float(_quality_from_risk(used_r_rate, gamma=gamma)),
                    "q_mass": float(_quality_from_risk(used_r_mass, gamma=gamma)),
                    "p_fail": float(sf),
                    "p_hard": float(sh),
                }
            )
            applied = True
            reason = "k_fail_cutoff_exceeded"
        else:
            reason = "k_fail_cutoff_exceeded_but_state_nan"

    meta: dict[str, Any] = {
        "applied": bool(applied),
        "reason": str(reason),
        "k_fail": (float(k_fail) if np.isfinite(k_fail) else None),
        "k_fail_abs": (float(abs(k_fail)) if np.isfinite(k_fail) else None),
        "k_fail_cutoff": float(cutoff),
        "state_fail_rate": (float(state_fail_f) if np.isfinite(state_fail_f) else None),
        "state_hard_rate": (float(state_hard_f) if np.isfinite(state_hard_f) else None),
        "r_rate_soft_raw": (float(soft_r_rate) if np.isfinite(soft_r_rate) else None),
        "r_mass_soft_raw": (float(soft_r_mass) if np.isfinite(soft_r_mass) else None),
        "p_fail_soft_raw": (float(soft_p_fail) if np.isfinite(soft_p_fail) else None),
        "p_hard_soft_raw": (float(soft_p_hard) if np.isfinite(soft_p_hard) else None),
        "r_rate_used": (float(used.get("r_rate")) if np.isfinite(float(used.get("r_rate", np.nan))) else None),
        "r_mass_used": (float(used.get("r_mass")) if np.isfinite(float(used.get("r_mass", np.nan))) else None),
        "p_fail_used": (float(used.get("p_fail")) if np.isfinite(float(used.get("p_fail", np.nan))) else None),
        "p_hard_used": (float(used.get("p_hard")) if np.isfinite(float(used.get("p_hard", np.nan))) else None),
    }
    return used, meta


def _continuous_rule_axis_profile(
    *,
    stat: RuleStat,
    rate_lambda: float,
    s_core: float,
    s_mass: float,
    gamma: float,
    z_fail_ref: float,
    z_hard_ref: float,
    tau_rate: float,
    tau_mass: float,
    core_quantile: float,
    eta_hard: float,
) -> dict[str, Any]:
    n = len(stat.z)
    valid = np.asarray(stat.valid_mask, dtype=bool) & np.isfinite(np.asarray(stat.z, dtype=float))
    z_cont = np.full(n, np.nan, dtype=float)
    p_fail_soft = np.full(n, np.nan, dtype=float)
    p_hard_soft = np.full(n, np.nan, dtype=float)
    if np.sum(valid) <= 0:
        return {
            "z_cont": z_cont,
            "p_fail_soft": p_fail_soft,
            "p_hard_soft": p_hard_soft,
            "r_rate": float("nan"),
            "r_mass": float("nan"),
            "r_core": float("nan"),
            "q_rate": float("nan"),
            "q_mass": float("nan"),
            "q_core": float("nan"),
            "p_fail": float("nan"),
            "p_hard": float("nan"),
            "tail_mass": float("nan"),
            "core_mad": float("nan"),
            "n_valid": float(0.0),
        }

    z = np.maximum(0.0, np.asarray(stat.z, dtype=float)[valid])
    z_cont[valid] = z

    tr = float(max(float(EPS), tau_rate))
    tm = float(max(float(EPS), tau_mass))
    pf = _sigmoid_safe((z - float(z_fail_ref)) / tr)
    ph = _sigmoid_safe((z - float(z_hard_ref)) / tr)
    p_fail_soft[valid] = pf
    p_hard_soft[valid] = ph

    rate_fail_soft = float(np.mean(pf))
    rate_hard_soft = float(np.mean(ph))
    r_rate = _clip01(rate_fail_soft + float(rate_lambda) * rate_hard_soft)

    ef = _softplus_safe((z - float(z_fail_ref)) / tm)
    eh = _softplus_safe((z - float(z_hard_ref)) / tm)
    tail_mass = float(np.mean(ef) + float(max(0.0, eta_hard)) * np.mean(eh))
    r_mass = _risk_from_scale(tail_mass, float(s_mass))

    if z.size >= 3:
        q = float(np.quantile(z, float(np.clip(core_quantile, 0.50, 0.95))))
        core = z[z <= q]
        if core.size < 3:
            core = z
    else:
        core = z
    core_mad = float(mad(core))
    r_core = _risk_from_scale(core_mad, float(s_core))

    q_rate = _quality_from_risk(r_rate, gamma=gamma)
    q_mass = _quality_from_risk(r_mass, gamma=gamma)
    q_core = _quality_from_risk(r_core, gamma=gamma)
    return {
        "z_cont": z_cont,
        "p_fail_soft": p_fail_soft,
        "p_hard_soft": p_hard_soft,
        "r_rate": float(r_rate),
        "r_mass": float(r_mass),
        "r_core": float(r_core),
        "q_rate": float(q_rate),
        "q_mass": float(q_mass),
        "q_core": float(q_core),
        "p_fail": float(rate_fail_soft),
        "p_hard": float(rate_hard_soft),
        "tail_mass": float(tail_mass),
        "core_mad": float(core_mad),
        "n_valid": float(np.sum(valid)),
    }


def _diag_cause_text(q10: float, q01: float, q11: float) -> str:
    values = {
        "q10": float(q10) if np.isfinite(q10) else float("nan"),
        "q01": float(q01) if np.isfinite(q01) else float("nan"),
        "q11": float(q11) if np.isfinite(q11) else float("nan"),
    }
    finite = {k: v for k, v in values.items() if np.isfinite(v)}
    if not finite:
        return "진단 불가(유효 샘플 부족)"
    top = max(finite.items(), key=lambda x: x[1])[0]
    if top == "q10":
        return "분포 형태 이탈(Residual) 중심"
    if top == "q01":
        return "입력조건 예측 이탈(Ridge) 중심"
    return "둘 다(형태+예측) 동시 붕괴"


def _conf_components(
    *,
    n_valid: int,
    finite_rate: float,
    available_rate: float,
    rules: list[RuleStat],
    min_support_rows: int,
    conf_op_norm: float = 1.0,
) -> dict[str, float]:
    support_floor = int(max(1, min_support_rows))
    conf_data = float(np.clip(float(n_valid) / float(support_floor), 0.0, 1.0))
    if int(n_valid) < 8:
        conf_data = 0.0
    calc_vals = [finite_rate, available_rate]
    calc_vals = [float(np.clip(v, 0.0, 1.0)) for v in calc_vals if np.isfinite(v)]
    conf_calc = float(min(calc_vals)) if calc_vals else 0.0
    if rules:
        method_scores: list[float] = []
        scale_scores: list[float] = []
        for r in rules:
            method_ok = str(r.selected_method or "").strip().lower() not in {"", "missing", "na"}
            source_ok = str(r.threshold_source or "").strip().lower() not in {"", "missing", "na"}
            method_scores.append(1.0 if (method_ok and source_ok) else 0.0)
            scale_scores.append(1.0 if (not bool(r.scale_collapse)) else 0.0)
        conf_th = float(min(min(method_scores), min(scale_scores)))
    else:
        conf_th = 1.0
    conf_op = float(np.clip(conf_op_norm, 0.0, 1.0))
    conf_bundle = float(np.clip(min(conf_data, conf_calc, conf_th) * conf_op, 0.0, 1.0))
    return {
        "Conf_data_norm": float(conf_data),
        "Conf_calc_norm": float(conf_calc),
        "Conf_th_norm": float(conf_th),
        "Conf_op_norm": float(conf_op),
        "CONF_norm": float(conf_bundle),
        "Conf_data": _score_to_int(5.0 * conf_data),
        "Conf_calc": _score_to_int(5.0 * conf_calc),
        "Conf_th": _score_to_int(5.0 * conf_th),
        "Conf_op": _score_to_int(5.0 * conf_op),
        "CONF_bundle": _score_to_int(5.0 * conf_bundle),
    }


def compute_bundle_scores(
    *,
    row_df: pd.DataFrame,
    threshold_summary_df: pd.DataFrame,
    score_runtime: Any,
    input_norm: np.ndarray | None = None,
    output_norm: np.ndarray | None = None,
    embedding_meta: dict[str, Any] | None = None,
) -> BundleScoreArtifacts:
    del input_norm, output_norm, embedding_meta  # New score does not use EDGE/DEG geometry.

    n = len(row_df)
    hard_gate = to_bool_array(row_df.get("hard_gate_pass", True), n)
    detail_evaluated = to_bool_array(row_df.get("detail_evaluated_nomask", False), n)
    detail_fail_any = to_bool_array(row_df.get("detail_fail_any_leaf_nomask", False), n)
    hard_detail_eval = hard_gate & detail_evaluated
    detail_fail_rate = float(np.mean(detail_fail_any[hard_detail_eval])) if np.any(hard_detail_eval) else 0.0
    eta = float(getattr(score_runtime, "eta_hard", 3.0))
    mass_excess_cap = float(getattr(score_runtime, "mass_excess_cap", getattr(score_runtime, "z_mass_cap", 12.0)))
    raw_mass_cap = float(getattr(score_runtime, "raw_mass_cap", 6.0))
    raw_mass_kfail_trigger = float(getattr(score_runtime, "raw_mass_kfail_trigger", 1e6))
    min_support_rows = int(getattr(score_runtime, "min_support_rows", 8))

    gamma = float(getattr(score_runtime, "new_score_gamma", 1.0))
    hard_band_cut = float(getattr(score_runtime, "new_score_band_hard_cut", 1e-12))
    na_neutral_score = float(getattr(score_runtime, "new_score_na_neutral_score", 3.0))
    sem_na_neutral_score = float(getattr(score_runtime, "new_score_sem_na_neutral_score", 3.0))

    w_rate = float(getattr(score_runtime, "new_score_w_rate", 0.55))
    w_mass = float(getattr(score_runtime, "new_score_w_mass", 0.30))
    w_core = float(getattr(score_runtime, "new_score_w_core", 0.15))
    axis_weights = {"core": w_core, "rate": w_rate, "mass": w_mass}
    diag_weights = {
        "rate": (w_rate / max(w_rate + w_mass, EPS)),
        "mass": (w_mass / max(w_rate + w_mass, EPS)),
    }

    # Threshold-free scoring controls (score input does not use rule thresholds).
    z_fail_ref = float(getattr(score_runtime, "new_score_z_fail_ref", 2.0))
    z_hard_ref = float(getattr(score_runtime, "new_score_z_hard_ref", 3.5))
    tau_rate = float(getattr(score_runtime, "new_score_tau_rate", 0.60))
    tau_mass = float(getattr(score_runtime, "new_score_tau_mass", 0.80))
    rate_lambda_output = float(getattr(score_runtime, "new_score_rate_lambda_output", 8.0))
    rate_lambda_residual = float(getattr(score_runtime, "new_score_rate_lambda_residual", 8.0))
    rate_lambda_semantic = float(getattr(score_runtime, "new_score_rate_lambda_semantic", 8.0))
    core_quantile = float(getattr(score_runtime, "new_score_core_quantile", 0.70))
    # Keep guard params in payload for backward compatibility, but score path does not use guard.
    hard_zero_cutoff = float(getattr(score_runtime, "new_score_hard_zero_cutoff", 0.05))
    hard_cap_cutoff = float(getattr(score_runtime, "new_score_hard_cap_cutoff", 0.02))
    use_hard_guard = False
    hard_tail_penalty_alpha = float(getattr(score_runtime, "new_score_hard_tail_penalty_alpha", 0.25))
    detail_penalty_alpha = float(getattr(score_runtime, "new_score_detail_penalty_alpha", 0.35))
    out_kfail_fallback_cutoff = float(getattr(score_runtime, "new_score_out_kfail_fallback_cutoff", 20.0))

    s_core_output = float(getattr(score_runtime, "new_score_s_core_output", 0.50))
    s_core_diff = float(getattr(score_runtime, "new_score_s_core_diff_residual", 0.50))
    s_core_ridge = float(getattr(score_runtime, "new_score_s_core_delta_ridge_ens", 0.50))
    s_core_disc = float(getattr(score_runtime, "new_score_s_core_discourse_instability", 0.50))
    s_core_contr = float(getattr(score_runtime, "new_score_s_core_contradiction", 0.50))

    s_mass_output = float(getattr(score_runtime, "new_score_s_mass_output", 0.50))
    s_mass_diff = float(getattr(score_runtime, "new_score_s_mass_diff_residual", 0.50))
    s_mass_ridge = float(getattr(score_runtime, "new_score_s_mass_delta_ridge_ens", 0.50))
    s_mass_disc = float(getattr(score_runtime, "new_score_s_mass_discourse_instability", 0.50))
    s_mass_contr = float(getattr(score_runtime, "new_score_s_mass_contradiction", 0.50))

    sem_zero_dominance_cutoff = float(getattr(score_runtime, "sem_zero_dominance_cutoff", 0.90))
    sem_zero_dominance_eps = float(getattr(score_runtime, "sem_zero_dominance_eps", 1e-12))

    threshold_map = _threshold_map(threshold_summary_df)
    required_rules = [
        "output",
        "diff_residual",
        "delta_ridge_ens",
        "discourse_instability",
        "contradiction",
    ]
    rules: dict[str, RuleStat] = {
        r: _build_rule_stat(
            row_df=row_df,
            threshold_map=threshold_map,
            rule=r,
            hard_gate=hard_gate,
            eta=eta,
            z_mass_cap=mass_excess_cap,
            raw_mass_cap=raw_mass_cap,
            raw_mass_kfail_trigger=raw_mass_kfail_trigger,
        )
        for r in required_rules
    }

    coverage_rules = ["output", "diff_residual", "delta_ridge_ens"]
    cov_finite_row = np.ones(n, dtype=bool)
    cov_available_row = np.ones(n, dtype=bool)
    for rule in coverage_rules:
        rs = rules[rule]
        cov_finite_row &= (~rs.available_mask) | rs.finite_mask
        cov_available_row &= rs.available_mask
    coverage_mask = hard_gate & cov_finite_row & cov_available_row
    coverage = _safe_rate(coverage_mask)
    hg_pass_rate = _safe_rate(hard_gate)
    finite_rate = _safe_rate(cov_finite_row)
    available_rate = _safe_rate(cov_available_row)
    q_cov = float(np.clip(5.0 * coverage, 0.0, 5.0)) if np.isfinite(coverage) else float(na_neutral_score)
    cov_band = "core" if q_cov >= 4.0 else ("tail" if q_cov >= 2.0 else "exceptional")

    out_axes = _continuous_rule_axis_profile(
        stat=rules["output"],
        rate_lambda=rate_lambda_output,
        s_core=s_core_output,
        s_mass=s_mass_output,
        gamma=gamma,
        z_fail_ref=z_fail_ref,
        z_hard_ref=z_hard_ref,
        tau_rate=tau_rate,
        tau_mass=tau_mass,
        core_quantile=core_quantile,
        eta_hard=eta,
    )
    rid_axes = _continuous_rule_axis_profile(
        stat=rules["diff_residual"],
        rate_lambda=rate_lambda_residual,
        s_core=s_core_diff,
        s_mass=s_mass_diff,
        gamma=gamma,
        z_fail_ref=z_fail_ref,
        z_hard_ref=z_hard_ref,
        tau_rate=tau_rate,
        tau_mass=tau_mass,
        core_quantile=core_quantile,
        eta_hard=eta,
    )
    sem_disc_axes = _continuous_rule_axis_profile(
        stat=rules["discourse_instability"],
        rate_lambda=rate_lambda_semantic,
        s_core=s_core_disc,
        s_mass=s_mass_disc,
        gamma=gamma,
        z_fail_ref=z_fail_ref,
        z_hard_ref=z_hard_ref,
        tau_rate=tau_rate,
        tau_mass=tau_mass,
        core_quantile=core_quantile,
        eta_hard=eta,
    )
    sem_contr_axes = _continuous_rule_axis_profile(
        stat=rules["contradiction"],
        rate_lambda=rate_lambda_semantic,
        s_core=s_core_contr,
        s_mass=s_mass_contr,
        gamma=gamma,
        z_fail_ref=z_fail_ref,
        z_hard_ref=z_hard_ref,
        tau_rate=tau_rate,
        tau_mass=tau_mass,
        core_quantile=core_quantile,
        eta_hard=eta,
    )
    ridge_axes = _continuous_rule_axis_profile(
        stat=rules["delta_ridge_ens"],
        rate_lambda=rate_lambda_residual,
        s_core=s_core_ridge,
        s_mass=s_mass_ridge,
        gamma=gamma,
        z_fail_ref=z_fail_ref,
        z_hard_ref=z_hard_ref,
        tau_rate=tau_rate,
        tau_mass=tau_mass,
        core_quantile=core_quantile,
        eta_hard=eta,
    )

    # Thresholds are downgraded to state/why explainer only.
    explain_output = _threshold_explain_profile(row_df=row_df, rule="output", stat=rules["output"])
    explain_diff = _threshold_explain_profile(row_df=row_df, rule="diff_residual", stat=rules["diff_residual"])
    explain_ridge = _threshold_explain_profile(row_df=row_df, rule="delta_ridge_ens", stat=rules["delta_ridge_ens"])
    explain_disc = _threshold_explain_profile(row_df=row_df, rule="discourse_instability", stat=rules["discourse_instability"])
    explain_contr = _threshold_explain_profile(row_df=row_df, rule="contradiction", stat=rules["contradiction"])
    out_axes, out_fallback_meta = _apply_out_kfail_fallback(
        out_axes_soft=out_axes,
        explain_output=explain_output,
        output_rule=rules["output"],
        rate_lambda_output=rate_lambda_output,
        eta_hard=eta,
        gamma=gamma,
        kfail_fallback_cutoff=out_kfail_fallback_cutoff,
    )

    def _zero_dominance_ratio(rule_stat: RuleStat) -> float:
        mask = rule_stat.valid_mask & np.isfinite(rule_stat.signal)
        if int(np.sum(mask)) <= 0:
            return float("nan")
        sig = np.asarray(rule_stat.signal, dtype=float)[mask]
        return float(np.mean(np.abs(sig) <= float(max(0.0, sem_zero_dominance_eps))))

    sem_zero_ratio_disc = _zero_dominance_ratio(rules["discourse_instability"])
    sem_zero_ratio_contr = _zero_dominance_ratio(rules["contradiction"])
    sem_uninformative = bool(
        np.isfinite(sem_zero_ratio_disc)
        and np.isfinite(sem_zero_ratio_contr)
        and float(sem_zero_ratio_disc) >= float(sem_zero_dominance_cutoff)
        and float(sem_zero_ratio_contr) >= float(sem_zero_dominance_cutoff)
    )

    def _axis_mean_quality(key: str) -> float:
        a = float(sem_disc_axes.get(f"q_{key}", np.nan))
        b = float(sem_contr_axes.get(f"q_{key}", np.nan))
        vals = [x for x in [a, b] if np.isfinite(x)]
        if not vals:
            return float("nan")
        return float(np.mean(vals))

    sem_q_core = _axis_mean_quality("core")
    sem_q_rate = _axis_mean_quality("rate")
    sem_q_mass = _axis_mean_quality("mass")
    sem_r_core = float(np.clip(1.0 - (sem_q_core / 5.0), 0.0, 1.0)) if np.isfinite(sem_q_core) else float("nan")
    sem_r_rate = float(np.clip(1.0 - (sem_q_rate / 5.0), 0.0, 1.0)) if np.isfinite(sem_q_rate) else float("nan")
    sem_r_mass = float(np.clip(1.0 - (sem_q_mass / 5.0), 0.0, 1.0)) if np.isfinite(sem_q_mass) else float("nan")
    sem_state_fail = _nanmean_or_nan(
        [
            float(explain_disc.get("state_fail_rate", np.nan)),
            float(explain_contr.get("state_fail_rate", np.nan)),
        ]
    )
    sem_state_hard = _nanmean_or_nan(
        [
            float(explain_disc.get("state_hard_rate", np.nan)),
            float(explain_contr.get("state_hard_rate", np.nan)),
        ]
    )
    sem_band = _state_band_label(state_fail_rate=sem_state_fail, state_hard_rate=sem_state_hard)
    sem_na = bool(sem_uninformative or (not np.isfinite(sem_r_core)) or (not np.isfinite(sem_r_rate)) or (not np.isfinite(sem_r_mass)))

    diff_state = state_array(rules["diff_residual"])
    ridge_state = state_array(rules["delta_ridge_ens"])
    diag_valid = rules["diff_residual"].valid_mask & rules["delta_ridge_ens"].valid_mask
    diag_q_state = classify_diag_quadrants(diff_state=diff_state, ridge_state=ridge_state, valid_mask=diag_valid)
    q00 = float(diag_q_state.get("q00", np.nan))
    q10 = float(diag_q_state.get("q10", np.nan))
    q01 = float(diag_q_state.get("q01", np.nan))
    q11 = float(diag_q_state.get("q11", np.nan))
    # DIAG score uses threshold-free soft quadrant probabilities.
    pa = np.asarray(rid_axes.get("p_fail_soft", np.full(n, np.nan)), dtype=float)
    pb = np.asarray(ridge_axes.get("p_fail_soft", np.full(n, np.nan)), dtype=float)
    soft_valid = np.isfinite(pa) & np.isfinite(pb)
    if np.sum(soft_valid) <= 0:
        q00_soft = q10_soft = q01_soft = q11_soft = float("nan")
    else:
        aa = pa[soft_valid]
        bb = pb[soft_valid]
        q10_soft = float(np.mean(aa * (1.0 - bb)))
        q01_soft = float(np.mean((1.0 - aa) * bb))
        q11_soft = float(np.mean(aa * bb))
        q00_soft = float(np.mean((1.0 - aa) * (1.0 - bb)))
    diag_rate_risk = (
        _clip01(float(q10_soft + q01_soft + q11_soft))
        if np.isfinite(q10_soft) and np.isfinite(q01_soft) and np.isfinite(q11_soft)
        else float("nan")
    )
    diag_mass_risk = (
        _clip01(float(q11_soft + 0.5 * min(q10_soft, q01_soft)))
        if np.isfinite(q11_soft) and np.isfinite(q10_soft) and np.isfinite(q01_soft)
        else float("nan")
    )
    diag_core_risk_ref = float(rid_axes["r_core"]) if np.isfinite(float(rid_axes["r_core"])) else float("nan")
    diag_core_q_ref = float(rid_axes["q_core"]) if np.isfinite(float(rid_axes["q_core"])) else float("nan")
    diag_cause = _diag_cause_text(q10=q10_soft, q01=q01_soft, q11=q11_soft)
    diag_cause_state = _diag_cause_text(q10=q10, q01=q01, q11=q11)
    diag_band = (
        "exceptional"
        if np.isfinite(q11) and q11 > 0.0
        else (
            "tail"
            if np.isfinite(q10) and np.isfinite(q01) and np.isfinite(q11) and (q10 + q01 + q11) > 0.0
            else "core"
        )
    )

    _, out_risk_base = _bundle_risk_noisy_or(
        {"core": float(out_axes["r_core"]), "rate": float(out_axes["r_rate"]), "mass": float(out_axes["r_mass"])},
        axis_weights,
    )
    out_risk, out_hard_tail_penalty = _apply_soft_hard_tail_penalty(
        base_risk=out_risk_base,
        p_hard=float(out_axes.get("p_hard", np.nan)),
        mass_risk=float(out_axes["r_mass"]),
        alpha=hard_tail_penalty_alpha,
    )
    detail_penalty_applied = float(np.clip(detail_penalty_alpha * detail_fail_rate, 0.0, 1.0))
    if np.isfinite(out_risk):
        out_risk = float(np.clip(out_risk + detail_penalty_applied, 0.0, 1.0))
    elif detail_penalty_applied > 0.0:
        out_risk = float(np.clip(detail_penalty_applied, 0.0, 1.0))
    out_score_penalized = _quality_from_risk(out_risk, gamma=1.0)
    out_score, out_guard = float(out_score_penalized), "disabled"
    if not np.isfinite(out_score):
        out_score = float(na_neutral_score)
    _, rid_risk_base = _bundle_risk_noisy_or(
        {"core": float(rid_axes["r_core"]), "rate": float(rid_axes["r_rate"]), "mass": float(rid_axes["r_mass"])},
        axis_weights,
    )
    rid_risk, rid_hard_tail_penalty = _apply_soft_hard_tail_penalty(
        base_risk=rid_risk_base,
        p_hard=float(rid_axes.get("p_hard", np.nan)),
        mass_risk=float(rid_axes["r_mass"]),
        alpha=hard_tail_penalty_alpha,
    )
    rid_score_penalized = _quality_from_risk(rid_risk, gamma=1.0)
    rid_score, rid_guard = float(rid_score_penalized), "disabled"
    if not np.isfinite(rid_score):
        rid_score = float(na_neutral_score)

    _, diag_risk_base = _bundle_risk_noisy_or(
        {"rate": float(diag_rate_risk), "mass": float(diag_mass_risk)},
        diag_weights,
    )
    diag_guard_p_hard = _nanmean_or_nan(
        [
            float(rid_axes.get("p_hard", np.nan)),
            float(ridge_axes.get("p_hard", np.nan)),
        ]
    )
    diag_risk, diag_hard_tail_penalty = _apply_soft_hard_tail_penalty(
        base_risk=diag_risk_base,
        p_hard=float(diag_guard_p_hard),
        mass_risk=float(diag_mass_risk),
        alpha=hard_tail_penalty_alpha,
    )
    diag_score_penalized = _quality_from_risk(diag_risk, gamma=1.0)
    diag_score, diag_guard = float(diag_score_penalized), "disabled"
    if not np.isfinite(diag_score):
        diag_score = float(na_neutral_score)

    if sem_na:
        sem_score = float(sem_na_neutral_score)
        sem_risk = float("nan")
        sem_risk_base = float("nan")
        sem_hard_tail_penalty = float("nan")
        sem_guard = "sem_na"
    else:
        _, sem_risk_base = _bundle_risk_noisy_or(
            {"core": sem_r_core, "rate": sem_r_rate, "mass": sem_r_mass},
            axis_weights,
        )
        sem_guard_p_hard = _nanmean_or_nan(
            [
                float(sem_disc_axes.get("p_hard", np.nan)),
                float(sem_contr_axes.get("p_hard", np.nan)),
            ]
        )
        sem_risk, sem_hard_tail_penalty = _apply_soft_hard_tail_penalty(
            base_risk=sem_risk_base,
            p_hard=float(sem_guard_p_hard),
            mass_risk=float(sem_r_mass),
            alpha=hard_tail_penalty_alpha,
        )
        sem_score_penalized = _quality_from_risk(sem_risk, gamma=1.0)
        sem_score, sem_guard = float(sem_score_penalized), "disabled"
        if not np.isfinite(sem_score):
            sem_score = float(sem_na_neutral_score)

    na_short_circuit: dict[str, dict[str, Any]] = {
        "OUT": {"enabled": False, "count": 0, "axes": [], "reason": ""},
        "RID": {"enabled": False, "count": 0, "axes": [], "reason": ""},
        "DIAG": {"enabled": False, "count": 0, "axes": [], "reason": ""},
        "SEM": {"enabled": False, "count": 0, "axes": [], "reason": ""},
    }

    out_missing = [k for k in ["OUT_core", "OUT_rate", "OUT_mass"] if not np.isfinite(float(out_axes[f"r_{k.split('_')[-1]}"]))]
    if out_missing:
        na_short_circuit["OUT"] = {"enabled": True, "count": len(out_missing), "axes": out_missing, "reason": "required_axis_na"}
        out_score = float(na_neutral_score)
    rid_missing = [k for k in ["RID_core", "RID_rate", "RID_mass"] if not np.isfinite(float(rid_axes[f"r_{k.split('_')[-1]}"]))]
    if rid_missing:
        na_short_circuit["RID"] = {"enabled": True, "count": len(rid_missing), "axes": rid_missing, "reason": "required_axis_na"}
        rid_score = float(na_neutral_score)
    diag_missing = []
    if not np.isfinite(diag_rate_risk):
        diag_missing.append("DIAG_rate")
    if not np.isfinite(diag_mass_risk):
        diag_missing.append("DIAG_mass")
    if diag_missing:
        na_short_circuit["DIAG"] = {"enabled": True, "count": len(diag_missing), "axes": diag_missing, "reason": "required_axis_na"}
        diag_score = float(na_neutral_score)
    if sem_na:
        na_short_circuit["SEM"] = {
            "enabled": True,
            "count": 3,
            "axes": ["SEM_core", "SEM_rate", "SEM_mass"],
            "reason": "sem_not_applicable",
        }

    cov_conf = _conf_components(
        n_valid=int(np.sum(coverage_mask)),
        finite_rate=_rate_on_base(cov_finite_row, hard_gate),
        available_rate=_rate_on_base(cov_available_row, hard_gate),
        rules=[rules[r] for r in coverage_rules],
        min_support_rows=min_support_rows,
        conf_op_norm=1.0,
    )
    out_conf = _conf_components(
        n_valid=int(rules["output"].n_valid),
        finite_rate=_rate_on_base(rules["output"].finite_mask, hard_gate),
        available_rate=_rate_on_base(rules["output"].available_mask, hard_gate),
        rules=[rules["output"]],
        min_support_rows=min_support_rows,
        conf_op_norm=1.0,
    )
    rid_conf = _conf_components(
        n_valid=int(rules["diff_residual"].n_valid),
        finite_rate=_rate_on_base(rules["diff_residual"].finite_mask, hard_gate),
        available_rate=_rate_on_base(rules["diff_residual"].available_mask, hard_gate),
        rules=[rules["diff_residual"]],
        min_support_rows=min_support_rows,
        conf_op_norm=1.0,
    )
    diag_conf = _conf_components(
        n_valid=int(np.sum(diag_valid)),
        finite_rate=_rate_on_base(rules["diff_residual"].finite_mask & rules["delta_ridge_ens"].finite_mask, hard_gate),
        available_rate=_rate_on_base(
            rules["diff_residual"].available_mask & rules["delta_ridge_ens"].available_mask,
            hard_gate,
        ),
        rules=[rules["diff_residual"], rules["delta_ridge_ens"]],
        min_support_rows=min_support_rows,
        conf_op_norm=1.0,
    )
    sem_conf = _conf_components(
        n_valid=int(np.sum(rules["discourse_instability"].valid_mask & rules["contradiction"].valid_mask)),
        finite_rate=_rate_on_base(
            rules["discourse_instability"].finite_mask & rules["contradiction"].finite_mask,
            hard_gate,
        ),
        available_rate=_rate_on_base(
            rules["discourse_instability"].available_mask & rules["contradiction"].available_mask,
            hard_gate,
        ),
        rules=[rules["discourse_instability"], rules["contradiction"]],
        min_support_rows=min_support_rows,
        conf_op_norm=1.0,
    )

    conf_data_norm = min(
        float(cov_conf["Conf_data_norm"]),
        float(out_conf["Conf_data_norm"]),
        float(rid_conf["Conf_data_norm"]),
        float(diag_conf["Conf_data_norm"]),
    )
    conf_calc_norm = min(
        float(cov_conf["Conf_calc_norm"]),
        float(out_conf["Conf_calc_norm"]),
        float(rid_conf["Conf_calc_norm"]),
        float(diag_conf["Conf_calc_norm"]),
    )
    conf_th_norm = min(
        float(cov_conf["Conf_th_norm"]),
        float(out_conf["Conf_th_norm"]),
        float(rid_conf["Conf_th_norm"]),
        float(diag_conf["Conf_th_norm"]),
    )
    conf_op_norm = 1.0
    conf_note = "SEM not applicable" if sem_na else ""
    if int(min_support_rows) < 8:
        conf_data_norm = 0.0
        conf_calc_norm = 0.0
        conf_th_norm = 0.0
        conf_note = "min_support_rows_lt_8"
    conf_norm = float(np.clip(min(conf_data_norm, conf_calc_norm, conf_th_norm) * conf_op_norm, 0.0, 1.0))
    conf_score = float(np.clip(5.0 * conf_norm, 0.0, 5.0))
    conf_band = "core" if conf_score >= 4.0 else ("tail" if conf_score >= 2.0 else "exceptional")
    conf_bundle = {
        "Conf_data_norm": float(conf_data_norm),
        "Conf_calc_norm": float(conf_calc_norm),
        "Conf_th_norm": float(conf_th_norm),
        "Conf_op_norm": float(conf_op_norm),
        "CONF_norm": float(conf_norm),
        "Conf_data": _score_to_int(5.0 * conf_data_norm),
        "Conf_calc": _score_to_int(5.0 * conf_calc_norm),
        "Conf_th": _score_to_int(5.0 * conf_th_norm),
        "Conf_op": _score_to_int(5.0 * conf_op_norm),
        "CONF_bundle": _score_to_int(conf_score),
        "note": conf_note,
    }

    bundle_scores_precise: dict[str, float] = {
        "COV": float(q_cov),
        "OUT": float(out_score),
        "RID": float(rid_score),
        "DIAG": float(diag_score),
        "SEM": float(sem_score),
        "CONF": float(conf_score),
    }
    bundle_scores_bucket: dict[str, int] = {b: _score_to_int(v) for b, v in bundle_scores_precise.items()}
    new_scores = {b: float(round4(bundle_scores_precise[b])) for b in BUNDLE_ORDER}

    bundle_band = {
        "COV": cov_band,
        "OUT": _state_band_label(
            state_fail_rate=float(explain_output.get("state_fail_rate", np.nan)),
            state_hard_rate=float(explain_output.get("state_hard_rate", np.nan)),
        ),
        "RID": _state_band_label(
            state_fail_rate=float(explain_diff.get("state_fail_rate", np.nan)),
            state_hard_rate=float(explain_diff.get("state_hard_rate", np.nan)),
        ),
        "DIAG": diag_band,
        "SEM": ("NA" if sem_na else sem_band),
        "CONF": conf_band,
    }

    subs: list[SubMetric] = [
        _submetric_from_good(
            bundle="COV",
            name="HG_score",
            raw_value=float(hg_pass_rate),
            good=float(hg_pass_rate),
            neutral_score=na_neutral_score,
            detail={"hg_pass_rate": float(hg_pass_rate)},
        ),
        _submetric_from_good(
            bundle="COV",
            name="Finite_score",
            raw_value=float(finite_rate),
            good=float(finite_rate),
            neutral_score=na_neutral_score,
            detail={"finite_rate": float(finite_rate)},
        ),
        _submetric_from_good(
            bundle="COV",
            name="Available_score",
            raw_value=float(available_rate),
            good=float(available_rate),
            neutral_score=na_neutral_score,
            detail={"available_rate": float(available_rate)},
        ),
        _submetric_from_risk(
            bundle="OUT",
            name="OUT_core",
            raw_value=float(out_axes["core_mad"]),
            risk=float(out_axes["r_core"]),
            neutral_score=na_neutral_score,
            detail={
                "core_mad": float(out_axes["core_mad"]),
                "band": str(bundle_band.get("OUT", "")),
            },
        ),
        _submetric_from_risk(
            bundle="OUT",
            name="OUT_rate",
            raw_value=(
                float(out_fallback_meta["r_rate_soft_raw"])
                if out_fallback_meta.get("r_rate_soft_raw") is not None
                else float("nan")
            ),
            risk=float(out_axes["r_rate"]),
            neutral_score=na_neutral_score,
            detail={
                "p_fail_soft": out_fallback_meta.get("p_fail_soft_raw"),
                "p_hard_soft": out_fallback_meta.get("p_hard_soft_raw"),
                "p_fail_soft_raw": out_fallback_meta.get("p_fail_soft_raw"),
                "p_hard_soft_raw": out_fallback_meta.get("p_hard_soft_raw"),
                "p_fail_used": out_fallback_meta.get("p_fail_used"),
                "p_hard_used": out_fallback_meta.get("p_hard_used"),
                "r_rate_soft_raw": out_fallback_meta.get("r_rate_soft_raw"),
                "r_rate_used": out_fallback_meta.get("r_rate_used"),
                "out_kfail_fallback_applied": bool(out_fallback_meta.get("applied", False)),
                "out_kfail_fallback_reason": str(out_fallback_meta.get("reason", "")),
                "out_kfail_fallback_cutoff": out_fallback_meta.get("k_fail_cutoff"),
                "out_kfail": out_fallback_meta.get("k_fail"),
                "rate_fail": explain_output.get("rate_fail"),
                "rate_hard": explain_output.get("rate_hard"),
                "excess_fail": explain_output.get("excess_fail"),
                "excess_hard": explain_output.get("excess_hard"),
                "state_fail_rate": explain_output.get("state_fail_rate"),
                "state_hard_rate": explain_output.get("state_hard_rate"),
                "band": str(bundle_band.get("OUT", "")),
            },
        ),
        _submetric_from_risk(
            bundle="OUT",
            name="OUT_mass",
            raw_value=float(out_axes["tail_mass"]),
            risk=float(out_axes["r_mass"]),
            neutral_score=na_neutral_score,
            detail={
                "tail_mass": float(out_axes["tail_mass"]),
                "tail_mass_soft_raw": float(out_axes["tail_mass"]),
                "r_mass_soft_raw": out_fallback_meta.get("r_mass_soft_raw"),
                "r_mass_used": out_fallback_meta.get("r_mass_used"),
                "out_kfail_fallback_applied": bool(out_fallback_meta.get("applied", False)),
                "out_kfail_fallback_reason": str(out_fallback_meta.get("reason", "")),
                "out_kfail_fallback_cutoff": out_fallback_meta.get("k_fail_cutoff"),
                "out_kfail": out_fallback_meta.get("k_fail"),
                "rate_fail": explain_output.get("rate_fail"),
                "rate_hard": explain_output.get("rate_hard"),
                "excess_fail": explain_output.get("excess_fail"),
                "excess_hard": explain_output.get("excess_hard"),
                "state_fail_rate": explain_output.get("state_fail_rate"),
                "state_hard_rate": explain_output.get("state_hard_rate"),
                "band": str(bundle_band.get("OUT", "")),
            },
        ),
        _submetric_from_risk(
            bundle="RID",
            name="RID_core",
            raw_value=float(rid_axes["core_mad"]),
            risk=float(rid_axes["r_core"]),
            neutral_score=na_neutral_score,
            detail={
                "core_mad": float(rid_axes["core_mad"]),
                "band": str(bundle_band.get("RID", "")),
            },
        ),
        _submetric_from_risk(
            bundle="RID",
            name="RID_rate",
            raw_value=float(rid_axes["r_rate"]),
            risk=float(rid_axes["r_rate"]),
            neutral_score=na_neutral_score,
            detail={
                "p_fail_soft": float(rid_axes["p_fail"]),
                "p_hard_soft": float(rid_axes["p_hard"]),
                "rate_fail": explain_diff.get("rate_fail"),
                "rate_hard": explain_diff.get("rate_hard"),
                "excess_fail": explain_diff.get("excess_fail"),
                "excess_hard": explain_diff.get("excess_hard"),
                "state_fail_rate": explain_diff.get("state_fail_rate"),
                "state_hard_rate": explain_diff.get("state_hard_rate"),
                "band": str(bundle_band.get("RID", "")),
            },
        ),
        _submetric_from_risk(
            bundle="RID",
            name="RID_mass",
            raw_value=float(rid_axes["tail_mass"]),
            risk=float(rid_axes["r_mass"]),
            neutral_score=na_neutral_score,
            detail={
                "tail_mass": float(rid_axes["tail_mass"]),
                "rate_fail": explain_diff.get("rate_fail"),
                "rate_hard": explain_diff.get("rate_hard"),
                "excess_fail": explain_diff.get("excess_fail"),
                "excess_hard": explain_diff.get("excess_hard"),
                "state_fail_rate": explain_diff.get("state_fail_rate"),
                "state_hard_rate": explain_diff.get("state_hard_rate"),
                "band": str(bundle_band.get("RID", "")),
            },
        ),
        _submetric_from_risk(
            bundle="DIAG",
            name="DIAG_core",
            raw_value=float(rid_axes["core_mad"]),
            risk=float(diag_core_risk_ref),
            neutral_score=na_neutral_score,
            detail={
                "core_reference": "RID_core",
                "q00": float(q00) if np.isfinite(q00) else None,
                "reference_only": True,
            },
        ),
        _submetric_from_risk(
            bundle="DIAG",
            name="DIAG_rate",
            raw_value=float(diag_rate_risk),
            risk=float(diag_rate_risk),
            neutral_score=na_neutral_score,
            detail={
                "q10_soft": float(q10_soft) if np.isfinite(q10_soft) else None,
                "q01_soft": float(q01_soft) if np.isfinite(q01_soft) else None,
                "q11_soft": float(q11_soft) if np.isfinite(q11_soft) else None,
                "q10_state": float(q10) if np.isfinite(q10) else None,
                "q01_state": float(q01) if np.isfinite(q01) else None,
                "q11_state": float(q11) if np.isfinite(q11) else None,
                "diag_cause": str(diag_cause),
                "diag_cause_state": str(diag_cause_state),
            },
        ),
        _submetric_from_risk(
            bundle="DIAG",
            name="DIAG_mass",
            raw_value=float(diag_mass_risk),
            risk=float(diag_mass_risk),
            neutral_score=na_neutral_score,
            detail={
                "q10_soft": float(q10_soft) if np.isfinite(q10_soft) else None,
                "q01_soft": float(q01_soft) if np.isfinite(q01_soft) else None,
                "q11_soft": float(q11_soft) if np.isfinite(q11_soft) else None,
                "q10_state": float(q10) if np.isfinite(q10) else None,
                "q01_state": float(q01) if np.isfinite(q01) else None,
                "q11_state": float(q11) if np.isfinite(q11) else None,
                "diag_cause": str(diag_cause),
                "diag_cause_state": str(diag_cause_state),
            },
        ),
        _submetric_from_good(
            bundle="DIAG",
            name="DIAG_stable",
            raw_value=float(q00_soft),
            good=float(q00_soft),
            neutral_score=na_neutral_score,
            detail={
                "q00_soft": float(q00_soft) if np.isfinite(q00_soft) else None,
                "q00_state": float(q00) if np.isfinite(q00) else None,
                "diag_cause": str(diag_cause),
                "diag_cause_state": str(diag_cause_state),
                "reference_only": True,
            },
        ),
        _submetric_from_risk(
            bundle="DIAG",
            name="DIAG_local",
            raw_value=float(q10_soft),
            risk=float(q10_soft),
            neutral_score=na_neutral_score,
            detail={
                "q10_soft": float(q10_soft) if np.isfinite(q10_soft) else None,
                "q10_state": float(q10) if np.isfinite(q10) else None,
                "diag_cause": str(diag_cause),
                "diag_cause_state": str(diag_cause_state),
                "reference_only": True,
            },
        ),
        _submetric_from_risk(
            bundle="DIAG",
            name="DIAG_unexplainable",
            raw_value=float(q01_soft),
            risk=float(q01_soft),
            neutral_score=na_neutral_score,
            detail={
                "q01_soft": float(q01_soft) if np.isfinite(q01_soft) else None,
                "q01_state": float(q01) if np.isfinite(q01) else None,
                "diag_cause": str(diag_cause),
                "diag_cause_state": str(diag_cause_state),
                "reference_only": True,
            },
        ),
        _submetric_from_risk(
            bundle="DIAG",
            name="DIAG_systemic",
            raw_value=float(q11_soft),
            risk=float(q11_soft),
            neutral_score=na_neutral_score,
            detail={
                "q11_soft": float(q11_soft) if np.isfinite(q11_soft) else None,
                "q11_state": float(q11) if np.isfinite(q11) else None,
                "diag_cause": str(diag_cause),
                "diag_cause_state": str(diag_cause_state),
                "reference_only": True,
            },
        ),
    ]

    if sem_na:
        sem_detail = {
            "reason": ("sem_zero_dominance" if sem_uninformative else "sem_not_applicable"),
            "sem_uninformative": bool(sem_uninformative),
            "sem_zero_ratio_disc": float(sem_zero_ratio_disc) if np.isfinite(sem_zero_ratio_disc) else None,
            "sem_zero_ratio_contr": float(sem_zero_ratio_contr) if np.isfinite(sem_zero_ratio_contr) else None,
            "sem_zero_dominance_cutoff": float(sem_zero_dominance_cutoff),
        }
        subs.extend(
            [
                _submetric_from_risk(
                    bundle="SEM",
                    name="SEM_core",
                    raw_value=float("nan"),
                    risk=float("nan"),
                    neutral_score=sem_na_neutral_score,
                    detail=sem_detail,
                ),
                _submetric_from_risk(
                    bundle="SEM",
                    name="SEM_rate",
                    raw_value=float("nan"),
                    risk=float("nan"),
                    neutral_score=sem_na_neutral_score,
                    detail=sem_detail,
                ),
                _submetric_from_risk(
                    bundle="SEM",
                    name="SEM_mass",
                    raw_value=float("nan"),
                    risk=float("nan"),
                    neutral_score=sem_na_neutral_score,
                    detail=sem_detail,
                ),
            ]
        )
    else:
        subs.extend(
            [
                _submetric_from_risk(
                    bundle="SEM",
                    name="SEM_core",
                    raw_value=float(_nanmean_or_nan([float(sem_disc_axes["core_mad"]), float(sem_contr_axes["core_mad"])])),
                    risk=float(sem_r_core),
                    neutral_score=na_neutral_score,
                    detail={"band": sem_band},
                ),
                _submetric_from_risk(
                    bundle="SEM",
                    name="SEM_rate",
                    raw_value=float(sem_r_rate),
                    risk=float(sem_r_rate),
                    neutral_score=na_neutral_score,
                    detail={
                        "p_fail_disc_soft": float(sem_disc_axes["p_fail"]),
                        "p_fail_contr_soft": float(sem_contr_axes["p_fail"]),
                        "p_hard_disc_soft": float(sem_disc_axes["p_hard"]),
                        "p_hard_contr_soft": float(sem_contr_axes["p_hard"]),
                        "rate_fail_disc": explain_disc.get("rate_fail"),
                        "rate_fail_contr": explain_contr.get("rate_fail"),
                        "rate_hard_disc": explain_disc.get("rate_hard"),
                        "rate_hard_contr": explain_contr.get("rate_hard"),
                        "state_fail_disc": explain_disc.get("state_fail_rate"),
                        "state_fail_contr": explain_contr.get("state_fail_rate"),
                        "state_hard_disc": explain_disc.get("state_hard_rate"),
                        "state_hard_contr": explain_contr.get("state_hard_rate"),
                        "band": sem_band,
                    },
                ),
                _submetric_from_risk(
                    bundle="SEM",
                    name="SEM_mass",
                    raw_value=float(_nanmean_or_nan([float(sem_disc_axes["tail_mass"]), float(sem_contr_axes["tail_mass"])])),
                    risk=float(sem_r_mass),
                    neutral_score=na_neutral_score,
                    detail={"band": sem_band},
                ),
            ]
        )

    subs.extend(
        [
            _submetric_from_good(
                bundle="CONF",
                name="CONF_data",
                raw_value=float(conf_data_norm),
                good=float(conf_data_norm),
                neutral_score=na_neutral_score,
                detail={"note": conf_note},
            ),
            _submetric_from_good(
                bundle="CONF",
                name="CONF_calc",
                raw_value=float(conf_calc_norm),
                good=float(conf_calc_norm),
                neutral_score=na_neutral_score,
                detail={"note": conf_note},
            ),
            _submetric_from_good(
                bundle="CONF",
                name="CONF_th",
                raw_value=float(conf_th_norm),
                good=float(conf_th_norm),
                neutral_score=na_neutral_score,
                detail={"note": conf_note},
            ),
            _submetric_from_good(
                bundle="CONF",
                name="CONF_op",
                raw_value=float(conf_op_norm),
                good=float(conf_op_norm),
                neutral_score=na_neutral_score,
                detail={"note": conf_note},
            ),
        ]
    )

    sub_lookup = {(s.bundle, s.name): s for s in subs}

    triplet_aggregation_payload: dict[str, dict[str, Any]] = {
        "OUT": {
            "mode": "threshold_free_noisy_or",
            "formula": "R_base=1-Π(1-w_axis*R_axis_cont), R=clip(R_base+soft_hard_tail), Q=5*(1-R)",
            "weights": [float(w_core), float(w_rate), float(w_mass)],
            "subscores": [float(_score_to_int(float(out_axes["q_core"]))), float(_score_to_int(float(out_axes["q_rate"]))), float(_score_to_int(float(out_axes["q_mass"])))],
            "subscores_precise": [float(round4(float(out_axes["q_core"]))), float(round4(float(out_axes["q_rate"]))), float(round4(float(out_axes["q_mass"])))],
            "risks": [float(round4(float(out_axes["r_core"]))), float(round4(float(out_axes["r_rate"]))), float(round4(float(out_axes["r_mass"])))],
            "risks_raw": [
                float(round4(float(out_axes["r_core"]))),
                (
                    float(round4(float(out_fallback_meta["r_rate_soft_raw"])))
                    if out_fallback_meta.get("r_rate_soft_raw") is not None
                    else None
                ),
                (
                    float(round4(float(out_fallback_meta["r_mass_soft_raw"])))
                    if out_fallback_meta.get("r_mass_soft_raw") is not None
                    else None
                ),
            ],
            "p_fail_soft_raw": out_fallback_meta.get("p_fail_soft_raw"),
            "p_hard_soft_raw": out_fallback_meta.get("p_hard_soft_raw"),
            "p_fail_used": out_fallback_meta.get("p_fail_used"),
            "p_hard_used": out_fallback_meta.get("p_hard_used"),
            "out_kfail_fallback": out_fallback_meta,
            "risk_or_base": float(round4(out_risk_base)) if np.isfinite(out_risk_base) else None,
            "hard_tail_penalty": float(round4(out_hard_tail_penalty)) if np.isfinite(out_hard_tail_penalty) else None,
            "detail_penalty_alpha": float(round4(detail_penalty_alpha)),
            "detail_fail_rate": float(round4(detail_fail_rate)),
            "detail_penalty_applied": float(round4(detail_penalty_applied)),
            "risk_or": float(round4(out_risk)) if np.isfinite(out_risk) else None,
            "score_precise": float(round4(out_score)),
            "score_bucket": int(bundle_scores_bucket["OUT"]),
            "guard": out_guard,
        },
        "RID": {
            "mode": "threshold_free_noisy_or",
            "formula": "R_base=1-Π(1-w_axis*R_axis_cont), R=clip(R_base+soft_hard_tail), Q=5*(1-R)",
            "weights": [float(w_core), float(w_rate), float(w_mass)],
            "subscores": [float(_score_to_int(float(rid_axes["q_core"]))), float(_score_to_int(float(rid_axes["q_rate"]))), float(_score_to_int(float(rid_axes["q_mass"])))],
            "subscores_precise": [float(round4(float(rid_axes["q_core"]))), float(round4(float(rid_axes["q_rate"]))), float(round4(float(rid_axes["q_mass"])))],
            "risks": [float(round4(float(rid_axes["r_core"]))), float(round4(float(rid_axes["r_rate"]))), float(round4(float(rid_axes["r_mass"])))],
            "risks_raw": [float(round4(float(rid_axes["r_core"]))), float(round4(float(rid_axes["r_rate"]))), float(round4(float(rid_axes["r_mass"])))],
            "risk_or_base": float(round4(rid_risk_base)) if np.isfinite(rid_risk_base) else None,
            "hard_tail_penalty": float(round4(rid_hard_tail_penalty)) if np.isfinite(rid_hard_tail_penalty) else None,
            "risk_or": float(round4(rid_risk)) if np.isfinite(rid_risk) else None,
            "score_precise": float(round4(rid_score)),
            "score_bucket": int(bundle_scores_bucket["RID"]),
            "guard": rid_guard,
        },
        "DIAG": {
            "mode": "threshold_free_noisy_or_diag",
            "formula": "R_base=1-Π(1-w_axis*R_axis_cont), R=clip(R_base+soft_hard_tail), axis={rate,mass}; core is reference only",
            "weights": [0.0, float(diag_weights["rate"]), float(diag_weights["mass"])],
            "subscores": [float(_score_to_int(diag_core_q_ref)), float(_score_to_int(_quality_from_risk(diag_rate_risk, gamma=1.0))), float(_score_to_int(_quality_from_risk(diag_mass_risk, gamma=1.0)))],
            "subscores_precise": [float(round4(diag_core_q_ref)) if np.isfinite(diag_core_q_ref) else None, float(round4(_quality_from_risk(diag_rate_risk, gamma=1.0))) if np.isfinite(diag_rate_risk) else None, float(round4(_quality_from_risk(diag_mass_risk, gamma=1.0))) if np.isfinite(diag_mass_risk) else None],
            "risks": [float(round4(diag_core_risk_ref)) if np.isfinite(diag_core_risk_ref) else None, float(round4(diag_rate_risk)) if np.isfinite(diag_rate_risk) else None, float(round4(diag_mass_risk)) if np.isfinite(diag_mass_risk) else None],
            "risks_raw": [float(round4(diag_core_risk_ref)) if np.isfinite(diag_core_risk_ref) else None, float(round4(diag_rate_risk)) if np.isfinite(diag_rate_risk) else None, float(round4(diag_mass_risk)) if np.isfinite(diag_mass_risk) else None],
            "risk_or_base": float(round4(diag_risk_base)) if np.isfinite(diag_risk_base) else None,
            "hard_tail_penalty": float(round4(diag_hard_tail_penalty)) if np.isfinite(diag_hard_tail_penalty) else None,
            "risk_or": float(round4(diag_risk)) if np.isfinite(diag_risk) else None,
            "score_precise": float(round4(diag_score)),
            "score_bucket": int(bundle_scores_bucket["DIAG"]),
            "guard": diag_guard,
        },
        "SEM": {
            "mode": ("sem_na" if sem_na else "threshold_free_noisy_or"),
            "formula": ("SEM is NA-neutral when uninformative" if sem_na else "R_base=1-Π(1-w_axis*R_axis_cont), R=clip(R_base+soft_hard_tail), Q=5*(1-R)"),
            "weights": [float(w_core), float(w_rate), float(w_mass)],
            "subscores": ([] if sem_na else [float(_score_to_int(sem_q_core)), float(_score_to_int(sem_q_rate)), float(_score_to_int(sem_q_mass))]),
            "subscores_precise": ([] if sem_na else [float(round4(sem_q_core)), float(round4(sem_q_rate)), float(round4(sem_q_mass))]),
            "risks": ([] if sem_na else [float(round4(sem_r_core)), float(round4(sem_r_rate)), float(round4(sem_r_mass))]),
            "risks_raw": ([] if sem_na else [float(round4(sem_r_core)), float(round4(sem_r_rate)), float(round4(sem_r_mass))]),
            "risk_or_base": (None if not np.isfinite(sem_risk_base) else float(round4(sem_risk_base))),
            "hard_tail_penalty": (None if not np.isfinite(sem_hard_tail_penalty) else float(round4(sem_hard_tail_penalty))),
            "risk_or": (None if not np.isfinite(sem_risk) else float(round4(sem_risk))),
            "score_precise": float(round4(sem_score)),
            "score_bucket": int(bundle_scores_bucket["SEM"]),
            "guard": sem_guard,
        },
    }

    triplet_risk_norm_raw: dict[str, dict[str, Any]] = {
        "OUT": {
            "core": float(out_axes["r_core"]) if np.isfinite(float(out_axes["r_core"])) else None,
            "rate": float(out_axes["r_rate"]) if np.isfinite(float(out_axes["r_rate"])) else None,
            "mass": float(out_axes["r_mass"]) if np.isfinite(float(out_axes["r_mass"])) else None,
            "rate_soft_raw": out_fallback_meta.get("r_rate_soft_raw"),
            "mass_soft_raw": out_fallback_meta.get("r_mass_soft_raw"),
            "p_fail_soft_raw": out_fallback_meta.get("p_fail_soft_raw"),
            "p_hard_soft_raw": out_fallback_meta.get("p_hard_soft_raw"),
            "p_fail_used": out_fallback_meta.get("p_fail_used"),
            "p_hard_used": out_fallback_meta.get("p_hard_used"),
            "out_kfail_fallback": out_fallback_meta,
            "bundle_or_risk_base": float(out_risk_base) if np.isfinite(out_risk_base) else None,
            "hard_tail_penalty": float(out_hard_tail_penalty) if np.isfinite(out_hard_tail_penalty) else None,
            "detail_penalty_alpha": float(detail_penalty_alpha),
            "detail_fail_rate": float(detail_fail_rate),
            "detail_penalty_applied": float(detail_penalty_applied),
            "bundle_or_risk": float(out_risk) if np.isfinite(out_risk) else None,
        },
        "RID": {
            "core": float(rid_axes["r_core"]) if np.isfinite(float(rid_axes["r_core"])) else None,
            "rate": float(rid_axes["r_rate"]) if np.isfinite(float(rid_axes["r_rate"])) else None,
            "mass": float(rid_axes["r_mass"]) if np.isfinite(float(rid_axes["r_mass"])) else None,
            "bundle_or_risk_base": float(rid_risk_base) if np.isfinite(rid_risk_base) else None,
            "hard_tail_penalty": float(rid_hard_tail_penalty) if np.isfinite(rid_hard_tail_penalty) else None,
            "bundle_or_risk": float(rid_risk) if np.isfinite(rid_risk) else None,
        },
        "DIAG": {
            "core": float(diag_core_risk_ref) if np.isfinite(diag_core_risk_ref) else None,
            "rate": float(diag_rate_risk) if np.isfinite(diag_rate_risk) else None,
            "mass": float(diag_mass_risk) if np.isfinite(diag_mass_risk) else None,
            "bundle_or_risk_base": float(diag_risk_base) if np.isfinite(diag_risk_base) else None,
            "hard_tail_penalty": float(diag_hard_tail_penalty) if np.isfinite(diag_hard_tail_penalty) else None,
            "bundle_or_risk": float(diag_risk) if np.isfinite(diag_risk) else None,
        },
        "SEM": {
            "core": float(sem_r_core) if np.isfinite(sem_r_core) else None,
            "rate": float(sem_r_rate) if np.isfinite(sem_r_rate) else None,
            "mass": float(sem_r_mass) if np.isfinite(sem_r_mass) else None,
            "bundle_or_risk_base": float(sem_risk_base) if np.isfinite(sem_risk_base) else None,
            "hard_tail_penalty": float(sem_hard_tail_penalty) if np.isfinite(sem_hard_tail_penalty) else None,
            "bundle_or_risk": float(sem_risk) if np.isfinite(sem_risk) else None,
        },
    }

    conf_by_bundle = {
        "COV": cov_conf,
        "OUT": out_conf,
        "RID": rid_conf,
        "DIAG": diag_conf,
        "SEM": sem_conf,
        "CONF": conf_bundle,
    }
    bundle_conf_eval = {b: int(conf_by_bundle[b]["CONF_bundle"]) for b in BUNDLE_ORDER}
    bundle_conf_warn = {b: int(conf_by_bundle[b]["Conf_th"]) for b in BUNDLE_ORDER}

    key_risk_map = {
        "COV": float(1.0 - coverage) if np.isfinite(coverage) else float("nan"),
        "OUT": float(out_risk) if np.isfinite(out_risk) else float("nan"),
        "RID": float(rid_risk) if np.isfinite(rid_risk) else float("nan"),
        "DIAG": float(diag_risk) if np.isfinite(diag_risk) else float("nan"),
        "SEM": float(sem_risk) if np.isfinite(sem_risk) else float("nan"),
        "CONF": float(1.0 - conf_norm),
    }
    key_risk_raw_map = dict(key_risk_map)
    key_risk_note = {
        "COV": "coverage=HG∧finite∧available",
            "OUT": "threshold_free noisyOR(core,rate,mass)+soft_hard_tail+detail_penalty",
        "RID": "threshold_free noisyOR(core,rate,mass)+soft_hard_tail",
        "DIAG": "threshold_free noisyOR(rate,mass)+soft_hard_tail, core is reference",
        "SEM": ("SEM NA-neutral" if sem_na else "threshold_free noisyOR(core,rate,mass)+soft_hard_tail"),
        "CONF": "min(Conf_data,Conf_calc,Conf_th)*Conf_op",
    }

    summary_rows: list[dict[str, Any]] = []
    for b in BUNDLE_ORDER:
        conf = int(bundle_conf_eval[b])
        conf_warn = int(bundle_conf_warn[b])
        color_basis = conf_warn
        summary_rows.append(
            {
                "bundle": b,
                "bundle_score": float(round4(bundle_scores_precise[b])),
                "bundle_score_bucket": int(bundle_scores_bucket[b]),
                "New_score": float(round4(bundle_scores_precise[b])),
                "bundle_band": str(bundle_band.get(b, "")),
                "bundle_confidence": int(conf),
                "bundle_conf_warning": int(conf_warn),
                "conf_data": int(conf_by_bundle[b]["Conf_data"]),
                "conf_calc": int(conf_by_bundle[b]["Conf_calc"]),
                "conf_th": int(conf_by_bundle[b]["Conf_th"]),
                "conf_op": int(conf_by_bundle[b]["Conf_op"]),
                "key_risk": float(key_risk_map[b]) if np.isfinite(key_risk_map[b]) else np.nan,
                "key_risk_raw": float(key_risk_raw_map[b]) if np.isfinite(key_risk_raw_map[b]) else np.nan,
                "key_risk_note": str(key_risk_note[b]),
                "conf_is_placeholder": False,
                "conf_placeholder_reason": "",
                "conf_color_source": "conf_warning",
                "conf_warning_reason": (str(conf_bundle.get("note", "")) if b == "CONF" else ""),
                "conf_color": confidence_to_hex(float(color_basis)),
                "detail_fail_rate": (float(detail_fail_rate) if b == "OUT" else 0.0),
                "detail_penalty_applied": (float(detail_penalty_applied) if b == "OUT" else 0.0),
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    sub_order = [
        ("COV", "HG_score"),
        ("COV", "Finite_score"),
        ("COV", "Available_score"),
        ("OUT", "OUT_core"),
        ("OUT", "OUT_rate"),
        ("OUT", "OUT_mass"),
        ("RID", "RID_core"),
        ("RID", "RID_rate"),
        ("RID", "RID_mass"),
        ("DIAG", "DIAG_core"),
        ("DIAG", "DIAG_rate"),
        ("DIAG", "DIAG_mass"),
        ("DIAG", "DIAG_stable"),
        ("DIAG", "DIAG_local"),
        ("DIAG", "DIAG_unexplainable"),
        ("DIAG", "DIAG_systemic"),
        ("SEM", "SEM_core"),
        ("SEM", "SEM_rate"),
        ("SEM", "SEM_mass"),
        ("CONF", "CONF_data"),
        ("CONF", "CONF_calc"),
        ("CONF", "CONF_th"),
        ("CONF", "CONF_op"),
    ]
    detail_rows: list[dict[str, Any]] = []
    for idx, (bundle, sub_name) in enumerate(sub_order):
        sub = sub_lookup[(bundle, sub_name)]
        bucket_value = sub.detail.get("bucket_value", None)
        is_na, na_label, na_reason = build_na_annotation(
            bundle=bundle,
            submetric=sub_name,
            bucket_source=str(sub.bucket_source),
            raw_value=float(sub.raw_value),
            bucket_value=(float(bucket_value) if bucket_value is not None and np.isfinite(float(bucket_value)) else None),
            detail=sub.detail,
            na_short_circuit=na_short_circuit,
        )
        detail_rows.append(
            {
                "bundle": bundle,
                "submetric": sub_name,
                "label": f"{bundle}.{sub_name}",
                "subscore": int(sub.score),
                "subscore_precise": float(round4(derive_subscore_precise(sub))),
                "raw_value": float(sub.raw_value) if np.isfinite(sub.raw_value) else np.nan,
                "bucket_value": float(bucket_value) if bucket_value is not None and np.isfinite(float(bucket_value)) else np.nan,
                "bucket_value_is": str(sub.detail.get("bucket_value_is", "")),
                "risk_norm_raw": (
                    float(sub.detail.get("risk_norm_raw"))
                    if sub.detail.get("risk_norm_raw") is not None and np.isfinite(float(sub.detail.get("risk_norm_raw")))
                    else np.nan
                ),
                "risk_norm_used": (
                    float(sub.detail.get("risk_norm_used"))
                    if sub.detail.get("risk_norm_used") is not None and np.isfinite(float(sub.detail.get("risk_norm_used")))
                    else np.nan
                ),
                "good_norm_used": (
                    float(sub.detail.get("good_norm_used"))
                    if sub.detail.get("good_norm_used") is not None and np.isfinite(float(sub.detail.get("good_norm_used")))
                    else np.nan
                ),
                "bucket_source": str(sub.bucket_source),
                "bucket_kind": str(sub.kind),
                "is_na": bool(is_na),
                "na_label": str(na_label),
                "na_reason": str(na_reason),
                "bundle_confidence": int(bundle_conf_eval[bundle]),
                "bundle_conf_warning": int(bundle_conf_warn[bundle]),
                "warning_adjusted": False,
                "warning_adjust_reason": "",
                "small_pool_guard": False,
                "conf_color_source": "conf_warning",
                "conf_color": confidence_to_hex(float(bundle_conf_warn[bundle])),
                "order": int(idx),
            }
        )
    detail_df = pd.DataFrame(detail_rows).sort_values("order").reset_index(drop=True)

    conf_overall = int(_score_to_int(conf_score))

    payload: dict[str, Any] = {
        "bundle_scores": {b: float(round4(bundle_scores_precise[b])) for b in BUNDLE_ORDER},
        "bundle_scores_bucket": {b: int(bundle_scores_bucket[b]) for b in BUNDLE_ORDER},
        "New_score": {b: float(round4(bundle_scores_precise[b])) for b in BUNDLE_ORDER},
        "new_scores": {b: float(round4(bundle_scores_precise[b])) for b in BUNDLE_ORDER},
        "bundle_band": {b: str(bundle_band.get(b, "")) for b in BUNDLE_ORDER},
        "bundle_confidence": {b: int(bundle_conf_eval[b]) for b in BUNDLE_ORDER},
        "bundle_conf_warning": {b: int(bundle_conf_warn[b]) for b in BUNDLE_ORDER},
        "confidence": {b: {k: (int(v) if str(k).startswith("Conf") or str(k).startswith("CONF") else v) for k, v in conf_by_bundle[b].items()} for b in BUNDLE_ORDER},
        "conf_overall": int(conf_overall),
        "conf_color_source": "conf_warning",
        "bucket_policy": {
            "risk": "threshold_free_continuous",
            "new_score_gamma": float(gamma),
            "new_score_z_fail_ref": float(z_fail_ref),
            "new_score_z_hard_ref": float(z_hard_ref),
            "new_score_tau_rate": float(tau_rate),
            "new_score_tau_mass": float(tau_mass),
            "new_score_rate_lambda_output": float(rate_lambda_output),
            "new_score_rate_lambda_residual": float(rate_lambda_residual),
            "new_score_rate_lambda_semantic": float(rate_lambda_semantic),
            "new_score_core_quantile": float(core_quantile),
            "new_score_w_core": float(w_core),
            "new_score_w_rate": float(w_rate),
            "new_score_w_mass": float(w_mass),
            "new_score_hard_zero_cutoff": float(hard_zero_cutoff),
            "new_score_hard_cap_cutoff": float(hard_cap_cutoff),
            "new_score_use_hard_guard": bool(use_hard_guard),
            "new_score_hard_tail_penalty_alpha": float(hard_tail_penalty_alpha),
            "new_score_detail_penalty_alpha": float(detail_penalty_alpha),
            "new_score_out_kfail_fallback_cutoff": float(out_kfail_fallback_cutoff),
            "new_score_na_neutral_score": float(na_neutral_score),
            "new_score_sem_na_neutral_score": float(sem_na_neutral_score),
            "new_score_scales": {
                "core": {
                    "output": float(s_core_output),
                    "diff_residual": float(s_core_diff),
                    "delta_ridge_ens": float(s_core_ridge),
                    "discourse_instability": float(s_core_disc),
                    "contradiction": float(s_core_contr),
                },
                "mass": {
                    "output": float(s_mass_output),
                    "diff_residual": float(s_mass_diff),
                    "delta_ridge_ens": float(s_mass_ridge),
                    "discourse_instability": float(s_mass_disc),
                    "contradiction": float(s_mass_contr),
                },
            },
        },
        "submetrics": [
            {
                "bundle": s.bundle,
                "submetric": s.name,
                "subscore_precise": float(round4(derive_subscore_precise(s))),
                "raw_value": float(s.raw_value) if np.isfinite(s.raw_value) else None,
                "bucket_value": float(s.detail.get("bucket_value")) if s.detail.get("bucket_value") is not None else None,
                "bucket_value_is": str(s.detail.get("bucket_value_is", "")),
                "risk_norm_raw": (
                    float(s.detail.get("risk_norm_raw"))
                    if s.detail.get("risk_norm_raw") is not None and np.isfinite(float(s.detail.get("risk_norm_raw")))
                    else None
                ),
                "risk_norm_used": (
                    float(s.detail.get("risk_norm_used"))
                    if s.detail.get("risk_norm_used") is not None and np.isfinite(float(s.detail.get("risk_norm_used")))
                    else None
                ),
                "good_norm_used": (
                    float(s.detail.get("good_norm_used"))
                    if s.detail.get("good_norm_used") is not None and np.isfinite(float(s.detail.get("good_norm_used")))
                    else None
                ),
                "score": int(s.score),
                "bucket_source": s.bucket_source,
                "bucket_kind": s.kind,
                "detail": s.detail,
            }
            for s in subs
        ],
        "raw": {
            "hg_pass_rate": float(hg_pass_rate),
            "finite_rate": float(finite_rate),
            "available_rate": float(available_rate),
            "coverage_rate": float(coverage) if np.isfinite(coverage) else None,
            "bundle_band": {b: str(bundle_band.get(b, "")) for b in BUNDLE_ORDER},
            "triplet_risk_norm_raw": triplet_risk_norm_raw,
            "triplet_risk_norm": triplet_risk_norm_raw,
            "triplet_formula": triplet_aggregation_payload,
            "triplet_aggregation": triplet_aggregation_payload,
            "na_short_circuit": na_short_circuit,
            "diag_quadrants": {
                "q00": (float(q00) if np.isfinite(q00) else None),
                "q10": (float(q10) if np.isfinite(q10) else None),
                "q01": (float(q01) if np.isfinite(q01) else None),
                "q11": (float(q11) if np.isfinite(q11) else None),
            },
            "diag_quadrants_soft": {
                "q00": (float(q00_soft) if np.isfinite(q00_soft) else None),
                "q10": (float(q10_soft) if np.isfinite(q10_soft) else None),
                "q01": (float(q01_soft) if np.isfinite(q01_soft) else None),
                "q11": (float(q11_soft) if np.isfinite(q11_soft) else None),
            },
            "diag_cause": str(diag_cause),
            "diag_cause_state": str(diag_cause_state),
            "threshold_explain": {
                "output": explain_output,
                "diff_residual": explain_diff,
                "delta_ridge_ens": explain_ridge,
                "discourse_instability": explain_disc,
                "contradiction": explain_contr,
                "output_out_kfail_fallback": out_fallback_meta,
            },
            "semantic": {
                "sem_uninformative": bool(sem_uninformative),
                "sem_not_applicable": bool(sem_na),
                "sem_zero_ratio_disc": float(sem_zero_ratio_disc) if np.isfinite(sem_zero_ratio_disc) else None,
                "sem_zero_ratio_contr": float(sem_zero_ratio_contr) if np.isfinite(sem_zero_ratio_contr) else None,
                "sem_zero_dominance_cutoff": float(sem_zero_dominance_cutoff),
            },
            "conf_components": {
                "Conf_data": float(conf_data_norm),
                "Conf_calc": float(conf_calc_norm),
                "Conf_th": float(conf_th_norm),
                "Conf_op": float(conf_op_norm),
                "CONF": float(conf_norm),
                "note": conf_note,
            },
            "detail_inspection": {
                "detail_evaluated_rows": int(np.sum(detail_evaluated)),
                "detail_hard_gate_evaluated_rows": int(np.sum(hard_detail_eval)),
                "detail_fail_rows": int(np.sum(detail_fail_any)),
                "detail_fail_rate_hard_eval": float(detail_fail_rate),
                "detail_penalty_alpha": float(detail_penalty_alpha),
                "detail_penalty_applied": float(detail_penalty_applied),
            },
            "new_score": new_scores,
        },
        "logs": {
            "key_risk_definition": key_risk_note,
            "diag_cause": str(diag_cause),
        },
        "bundle_label_ko": BUNDLE_LABEL_KO,
        "bundle_desc_ko": BUNDLE_DESC_KO,
        "submetric_label_ko": SUBMETRIC_LABEL_KO,
        "submetric_desc_ko": SUBMETRIC_DESC_KO,
    }

    return BundleScoreArtifacts(summary_df=summary_df, detail_df=detail_df, payload=payload)
