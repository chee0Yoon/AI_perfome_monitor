"""Contradiction Metric - verdict/evidence contradiction risk score."""

from __future__ import annotations

from typing import Any

import numpy as np

from final_metric_refactor.distribution._semantic import build_semantic_context, safe_semantic_z


class ContradictionMetric:
    """Detect evidence vectors that are closer to opposite-class anchors than own anchors."""

    def __init__(
        self,
        enabled: bool = True,
        verdict_candidate_keys: list[str] | None = None,
        evidence_candidate_keys: list[str] | None = None,
        min_support_ratio: float = 0.60,
        min_class_size: int = 20,
        max_classes: int = 8,
        score_bins: int = 3,
        min_evidence_tokens: int = 2,
        margin_shift: float = 0.0,
        margin_weight: float = 0.7,
        severity_weight: float = 0.3,
    ):
        self.enabled = enabled
        self.verdict_candidate_keys = verdict_candidate_keys or [
            "is_correct",
            "decision",
            "label",
            "result",
            "verdict",
            "score",
            "rating",
        ]
        self.evidence_candidate_keys = evidence_candidate_keys or [
            "feedback",
            "explanation",
            "rationale",
            "reason",
            "analysis",
            "comment",
        ]
        self.min_support_ratio = min_support_ratio
        self.min_class_size = min_class_size
        self.max_classes = max_classes
        self.score_bins = score_bins
        self.min_evidence_tokens = min_evidence_tokens
        self.margin_shift = margin_shift
        self.margin_weight = margin_weight
        self.severity_weight = severity_weight

    def compute(
        self,
        model: Any,
        output_dicts: list[dict[str, Any] | None],
        ref_mask: np.ndarray,
        batch_size: int = 64,
        ref_weights: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        n = len(output_dicts)
        if not self.enabled:
            return (
                np.zeros(n),
                np.zeros(n, dtype=bool),
                {
                    "selected_verdict_key": None,
                    "selected_evidence_key": None,
                    "class_count": 0,
                    "eligible_rows": 0,
                    "unavailable_reason": "disabled",
                    "label_kind": None,
                },
            )

        context = build_semantic_context(
            model=model,
            output_dicts=output_dicts,
            ref_mask=ref_mask,
            verdict_candidate_keys=self.verdict_candidate_keys,
            evidence_candidate_keys=self.evidence_candidate_keys,
            min_support_ratio=float(self.min_support_ratio),
            min_class_size=int(self.min_class_size),
            max_classes=int(self.max_classes),
            score_bins=int(self.score_bins),
            min_evidence_tokens=int(self.min_evidence_tokens),
            batch_size=int(batch_size),
            ref_weights=ref_weights,
        )

        available = np.asarray(context.available_mask, dtype=bool)
        if not available.any() or len(context.centroids) < 2:
            return (
                np.zeros(n),
                np.zeros(n, dtype=bool),
                {
                    "selected_verdict_key": context.selected_verdict_key,
                    "selected_evidence_key": context.selected_evidence_key,
                    "class_count": int(context.class_count),
                    "eligible_rows": int(context.eligible_rows),
                    "unavailable_reason": context.unavailable_reason or "insufficient_ref_support",
                    "label_kind": context.label_kind,
                },
            )

        emb = np.nan_to_num(np.asarray(context.evidence_embeddings, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        labels = np.asarray(context.labels, dtype=int)

        margin_raw = np.zeros(n, dtype=float)
        severity_raw = np.zeros(n, dtype=float)

        classes = sorted(context.centroids.keys())
        centroid_mat = np.nan_to_num(
            np.vstack([context.centroids[c] for c in classes]).astype(float),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        cls_to_col = {c: i for i, c in enumerate(classes)}

        idx = np.where(available)[0]
        sims = emb[idx] @ centroid_mat.T
        own_cols = np.array([cls_to_col[int(labels[i])] for i in idx], dtype=int)
        own_sim = sims[np.arange(len(idx)), own_cols]

        other_sim = np.full(len(idx), -1.0, dtype=float)
        for j in range(len(idx)):
            row = sims[j].copy()
            row[own_cols[j]] = -np.inf
            row_max = np.max(row)
            other_sim[j] = float(row_max) if np.isfinite(row_max) else -1.0

        margin = other_sim - own_sim - float(self.margin_shift)
        severity = margin * np.maximum(0.0, 1.0 - own_sim)

        margin_raw[idx] = margin
        severity_raw[idx] = np.maximum(0.0, severity)

        z_margin = safe_semantic_z(margin_raw, context=context, ref_weights=ref_weights)
        z_severity = safe_semantic_z(severity_raw, context=context, ref_weights=ref_weights)

        w_margin = float(max(self.margin_weight, 0.0))
        w_severity = float(max(self.severity_weight, 0.0))
        w_sum = max(w_margin + w_severity, 1e-12)
        score = ((w_margin * z_margin) + (w_severity * z_severity)) / w_sum
        score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        score[~available] = 0.0

        meta = {
            "selected_verdict_key": context.selected_verdict_key,
            "selected_evidence_key": context.selected_evidence_key,
            "class_count": int(context.class_count),
            "eligible_rows": int(context.eligible_rows),
            "unavailable_reason": context.unavailable_reason or "",
            "label_kind": context.label_kind,
        }
        return score, available, meta
