"""Shared semantic helpers for semantic distribution metrics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from final_metric_refactor.shared.geometry import normalize_rows
from final_metric_refactor.distribution._shared import weighted_robust_z

TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


@dataclass
class SemanticContext:
    evidence_embeddings: np.ndarray
    labels: np.ndarray
    available_mask: np.ndarray
    ref_available_mask: np.ndarray
    centroids: dict[int, np.ndarray]
    class_values: dict[int, str]
    selected_verdict_key: str | None
    selected_evidence_key: str | None
    class_count: int
    eligible_rows: int
    label_kind: str | None
    unavailable_reason: str | None


def _token_count(text: str) -> int:
    return len(TOKEN_RE.findall(str(text)))


def _get_value(row: dict[str, Any] | None, key: str) -> Any:
    if not isinstance(row, dict):
        return None
    if key in row:
        return row.get(key)
    return None


def _normalized_entropy(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    total = float(np.sum(counts))
    if total <= 0.0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0.0]
    if len(probs) <= 1:
        return 0.0
    ent = float(-np.sum(probs * np.log(probs)))
    ent_max = float(np.log(len(probs)))
    if ent_max <= 0.0:
        return 0.0
    return float(np.clip(ent / ent_max, 0.0, 1.0))


def _scalar_informativeness(values: list[Any]) -> float:
    if not values:
        return 0.0

    arr = np.array([str(v).strip().lower() for v in values if str(v).strip()], dtype=object)
    if arr.size == 0:
        return 0.0

    nums = np.full(arr.size, np.nan, dtype=float)
    numeric_ok = np.zeros(arr.size, dtype=bool)
    for i, v in enumerate(arr.tolist()):
        try:
            x = float(v)
            if np.isfinite(x):
                nums[i] = x
                numeric_ok[i] = True
        except Exception:
            continue

    numeric_ratio = float(np.mean(numeric_ok)) if arr.size > 0 else 0.0
    if numeric_ratio >= 0.8:
        finite = nums[numeric_ok]
        if finite.size < 3:
            return 0.0
        q = np.quantile(finite, [0.0, 0.25, 0.5, 0.75, 1.0])
        q = np.unique(q)
        if q.size < 3:
            return 0.0
        bins = np.digitize(finite, q[1:-1], right=False)
        _, counts = np.unique(bins, return_counts=True)
        ent = _normalized_entropy(counts)
        spread = float(np.percentile(finite, 90) - np.percentile(finite, 10))
        spread_score = float(np.clip(spread, 0.0, 1.0))
        return float(np.clip((0.8 * ent) + (0.2 * spread_score), 0.0, 1.0))

    uniq, counts = np.unique(arr, return_counts=True)
    ent = _normalized_entropy(counts)
    uniq_ratio = float(len(uniq) / max(len(arr), 1))
    # Prefer moderately diverse verdict fields; penalize near-unique id-like keys.
    uniq_penalty = float(np.clip((uniq_ratio - 0.70) / 0.30, 0.0, 1.0))
    return float(np.clip(ent - (0.35 * uniq_penalty), 0.0, 1.0))


def _text_informativeness(values: list[Any], min_tokens: int) -> float:
    if not values:
        return 0.0
    lengths = []
    for v in values:
        t = str(v).strip()
        if not t:
            continue
        lengths.append(_token_count(t))
    if not lengths:
        return 0.0
    med = float(np.median(lengths))
    # Reward fields that contain enough lexical evidence without over-weighting verbosity.
    return float(np.clip((med - float(min_tokens)) / 12.0, 0.0, 1.0))


def _pick_best_key(
    output_dicts: list[dict[str, Any] | None],
    candidates: list[str],
    min_support_ratio: float,
    min_tokens: int = 1,
    value_kind: str = "text",
) -> tuple[str | None, np.ndarray]:
    n = len(output_dicts)
    best_key: str | None = None
    best_mask = np.zeros(n, dtype=bool)
    best_score = -np.inf

    for key in candidates:
        mask = np.zeros(n, dtype=bool)
        selected_values: list[Any] = []
        for i, row in enumerate(output_dicts):
            value = _get_value(row, key)
            if value is None:
                continue
            if value_kind == "text":
                text = str(value).strip()
                if not text:
                    continue
                if _token_count(text) < min_tokens:
                    continue
                mask[i] = True
                selected_values.append(text)
            else:
                if isinstance(value, (dict, list)):
                    continue
                text = str(value).strip()
                if not text:
                    continue
                mask[i] = True
                selected_values.append(value)

        support = float(mask.mean()) if n > 0 else 0.0
        if support < min_support_ratio:
            continue

        if value_kind == "scalar":
            info = _scalar_informativeness(selected_values)
        else:
            info = _text_informativeness(selected_values, min_tokens=min_tokens)

        score = support + (0.20 * info)
        if score > best_score:
            best_score = score
            best_key = key
            best_mask = mask

    return best_key, best_mask


def _build_verdict_labels(
    output_dicts: list[dict[str, Any] | None],
    verdict_key: str,
    mask: np.ndarray,
    max_classes: int,
    score_bins: int,
) -> tuple[np.ndarray, dict[int, str], str | None]:
    n = len(output_dicts)
    labels = np.full(n, -1, dtype=int)

    raw_vals: list[Any] = [_get_value(row, verdict_key) for row in output_dicts]

    nums = np.full(n, np.nan, dtype=float)
    numeric_ok = np.zeros(n, dtype=bool)
    for i, v in enumerate(raw_vals):
        if not mask[i]:
            continue
        try:
            nums[i] = float(v)
            if np.isfinite(nums[i]):
                numeric_ok[i] = True
        except Exception:
            continue

    numeric_ratio = float(numeric_ok[mask].mean()) if mask.any() else 0.0

    if numeric_ratio >= 0.8:
        finite_vals = nums[numeric_ok]
        uniq = np.unique(np.round(finite_vals, 8))
        if uniq.size >= 2:
            n_bins = int(max(2, min(score_bins, max_classes)))
            edges = np.quantile(finite_vals, np.linspace(0.0, 1.0, n_bins + 1))
            edges = np.unique(edges)
            if len(edges) >= 3:
                b = np.digitize(nums, edges[1:-1], right=False)
                b = np.where(numeric_ok, b, -1)
                labels = b.astype(int)
                class_values = {}
                for cls in sorted(set(labels[labels >= 0].tolist())):
                    class_values[int(cls)] = f"bin_{int(cls)}"
                return labels, class_values, "numeric"

    # Categorical fallback
    vals = np.array([str(v).strip().lower() if v is not None else "" for v in raw_vals], dtype=object)
    vals = np.where(mask, vals, "")
    valid = vals != ""
    if not valid.any():
        return labels, {}, None

    uniq, counts = np.unique(vals[valid], return_counts=True)
    order = np.argsort(-counts)
    uniq = uniq[order]
    if len(uniq) > max_classes:
        uniq = uniq[:max_classes]

    val_to_id = {str(v): i for i, v in enumerate(uniq.tolist())}
    for i in range(n):
        if valid[i] and str(vals[i]) in val_to_id:
            labels[i] = int(val_to_id[str(vals[i])])

    class_values = {int(v): k for k, v in val_to_id.items()}
    return labels, class_values, "categorical"


def build_semantic_context(
    model: Any,
    output_dicts: list[dict[str, Any] | None],
    ref_mask: np.ndarray,
    verdict_candidate_keys: list[str],
    evidence_candidate_keys: list[str],
    min_support_ratio: float,
    min_class_size: int,
    max_classes: int,
    score_bins: int,
    min_evidence_tokens: int,
    batch_size: int,
    ref_weights: np.ndarray | None = None,
) -> SemanticContext:
    n = len(output_dicts)
    ref_mask = np.asarray(ref_mask, dtype=bool)
    if len(ref_mask) != n:
        ref_mask = np.ones(n, dtype=bool)

    if n == 0:
        return SemanticContext(
            evidence_embeddings=np.zeros((0, 1), dtype=np.float32),
            labels=np.zeros(0, dtype=int),
            available_mask=np.zeros(0, dtype=bool),
            ref_available_mask=np.zeros(0, dtype=bool),
            centroids={},
            class_values={},
            selected_verdict_key=None,
            selected_evidence_key=None,
            class_count=0,
            eligible_rows=0,
            label_kind=None,
            unavailable_reason="empty_input",
        )

    evidence_key, evidence_mask = _pick_best_key(
        output_dicts=output_dicts,
        candidates=evidence_candidate_keys,
        min_support_ratio=min_support_ratio,
        min_tokens=min_evidence_tokens,
        value_kind="text",
    )
    if evidence_key is None:
        return SemanticContext(
            evidence_embeddings=np.zeros((n, 1), dtype=np.float32),
            labels=np.full(n, -1, dtype=int),
            available_mask=np.zeros(n, dtype=bool),
            ref_available_mask=np.zeros(n, dtype=bool),
            centroids={},
            class_values={},
            selected_verdict_key=None,
            selected_evidence_key=None,
            class_count=0,
            eligible_rows=0,
            label_kind=None,
            unavailable_reason="no_evidence_key",
        )

    verdict_key, verdict_mask = _pick_best_key(
        output_dicts=output_dicts,
        candidates=verdict_candidate_keys,
        min_support_ratio=min_support_ratio,
        value_kind="scalar",
    )
    if verdict_key is None:
        return SemanticContext(
            evidence_embeddings=np.zeros((n, 1), dtype=np.float32),
            labels=np.full(n, -1, dtype=int),
            available_mask=np.zeros(n, dtype=bool),
            ref_available_mask=np.zeros(n, dtype=bool),
            centroids={},
            class_values={},
            selected_verdict_key=None,
            selected_evidence_key=evidence_key,
            class_count=0,
            eligible_rows=0,
            label_kind=None,
            unavailable_reason="no_valid_verdict_candidate",
        )

    labels, class_values, label_kind = _build_verdict_labels(
        output_dicts=output_dicts,
        verdict_key=verdict_key,
        mask=verdict_mask,
        max_classes=max_classes,
        score_bins=score_bins,
    )

    base_available = evidence_mask & (labels >= 0)
    if not base_available.any():
        return SemanticContext(
            evidence_embeddings=np.zeros((n, 1), dtype=np.float32),
            labels=labels,
            available_mask=np.zeros(n, dtype=bool),
            ref_available_mask=np.zeros(n, dtype=bool),
            centroids={},
            class_values=class_values,
            selected_verdict_key=verdict_key,
            selected_evidence_key=evidence_key,
            class_count=0,
            eligible_rows=0,
            label_kind=label_kind,
            unavailable_reason="insufficient_evidence_rows",
        )

    evidence_texts = ["" for _ in range(n)]
    to_encode_idx = np.where(evidence_mask)[0]
    encode_texts = []
    for i in to_encode_idx:
        val = _get_value(output_dicts[i], evidence_key)
        encode_texts.append(str(val).strip())
        evidence_texts[i] = str(val).strip()

    if not encode_texts:
        return SemanticContext(
            evidence_embeddings=np.zeros((n, 1), dtype=np.float32),
            labels=labels,
            available_mask=np.zeros(n, dtype=bool),
            ref_available_mask=np.zeros(n, dtype=bool),
            centroids={},
            class_values=class_values,
            selected_verdict_key=verdict_key,
            selected_evidence_key=evidence_key,
            class_count=0,
            eligible_rows=0,
            label_kind=label_kind,
            unavailable_reason="insufficient_evidence_rows",
        )

    enc = model.encode(encode_texts, batch_size=batch_size)
    enc = normalize_rows(np.asarray(enc, dtype=np.float32))
    emb_dim = enc.shape[1] if enc.ndim == 2 and enc.shape[0] > 0 else 1
    evidence_embeddings = np.zeros((n, emb_dim), dtype=np.float32)
    evidence_embeddings[to_encode_idx] = enc

    ref_base = base_available & ref_mask
    class_counts: dict[int, int] = {}
    for cls in sorted(set(labels[ref_base].tolist())):
        class_counts[int(cls)] = int(np.sum(ref_base & (labels == cls)))
    valid_classes = [cls for cls, cnt in class_counts.items() if cnt >= int(min_class_size)]

    if len(valid_classes) < 2:
        return SemanticContext(
            evidence_embeddings=evidence_embeddings,
            labels=labels,
            available_mask=np.zeros(n, dtype=bool),
            ref_available_mask=np.zeros(n, dtype=bool),
            centroids={},
            class_values=class_values,
            selected_verdict_key=verdict_key,
            selected_evidence_key=evidence_key,
            class_count=0,
            eligible_rows=0,
            label_kind=label_kind,
            unavailable_reason="insufficient_ref_support",
        )

    valid_class_set = set(valid_classes)
    available_mask = base_available & np.array([int(v) in valid_class_set for v in labels], dtype=bool)
    ref_available_mask = available_mask & ref_mask

    if not ref_available_mask.any():
        return SemanticContext(
            evidence_embeddings=evidence_embeddings,
            labels=labels,
            available_mask=np.zeros(n, dtype=bool),
            ref_available_mask=np.zeros(n, dtype=bool),
            centroids={},
            class_values=class_values,
            selected_verdict_key=verdict_key,
            selected_evidence_key=evidence_key,
            class_count=0,
            eligible_rows=0,
            label_kind=label_kind,
            unavailable_reason="insufficient_ref_support",
        )

    centroids: dict[int, np.ndarray] = {}
    for cls in sorted(valid_classes):
        idx = np.where(ref_available_mask & (labels == cls))[0]
        if len(idx) == 0:
            continue
        vecs = evidence_embeddings[idx].astype(float)
        if ref_weights is not None:
            rw = np.asarray(ref_weights, dtype=float)
            if len(rw) == n:
                w = np.where(np.isfinite(rw[idx]) & (rw[idx] > 0), rw[idx], 0.0)
                sw = float(w.sum())
            else:
                w = None
                sw = 0.0
            if w is not None and sw > 0:
                center = (vecs * w[:, None]).sum(axis=0) / sw
            else:
                center = vecs.mean(axis=0)
        else:
            center = vecs.mean(axis=0)
        c_norm = float(np.linalg.norm(center))
        if c_norm <= 0:
            continue
        centroids[int(cls)] = (center / c_norm).astype(np.float32)

    valid_centroid_classes = set(centroids.keys())
    available_mask = available_mask & np.array([int(v) in valid_centroid_classes for v in labels], dtype=bool)
    ref_available_mask = available_mask & ref_mask

    if len(valid_centroid_classes) < 2 or not ref_available_mask.any():
        return SemanticContext(
            evidence_embeddings=evidence_embeddings,
            labels=labels,
            available_mask=np.zeros(n, dtype=bool),
            ref_available_mask=np.zeros(n, dtype=bool),
            centroids={},
            class_values=class_values,
            selected_verdict_key=verdict_key,
            selected_evidence_key=evidence_key,
            class_count=0,
            eligible_rows=0,
            label_kind=label_kind,
            unavailable_reason="insufficient_ref_support",
        )

    return SemanticContext(
        evidence_embeddings=evidence_embeddings,
        labels=labels,
        available_mask=available_mask,
        ref_available_mask=ref_available_mask,
        centroids=centroids,
        class_values={int(k): str(class_values.get(k, str(k))) for k in centroids.keys()},
        selected_verdict_key=verdict_key,
        selected_evidence_key=evidence_key,
        class_count=int(len(centroids)),
        eligible_rows=int(available_mask.sum()),
        label_kind=label_kind,
        unavailable_reason=None,
    )


def safe_semantic_z(
    raw_values: np.ndarray,
    context: SemanticContext,
    ref_weights: np.ndarray | None,
    z_clip: float = 30.0,
) -> np.ndarray:
    """Compute robust z for semantic raw values using reference-eligible rows."""
    z = weighted_robust_z(
        values=np.asarray(raw_values, dtype=float),
        ref_mask=np.asarray(context.ref_available_mask, dtype=bool),
        ref_weights=ref_weights,
    )
    clip = float(max(z_clip, 1.0))
    z = np.clip(z, -clip, clip)
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
