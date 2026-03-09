"""Discourse Instability Metric - semantic ambiguity within verdict-conditioned evidence."""

from __future__ import annotations

from typing import Any

import numpy as np

from final_metric_refactor.distribution._semantic import build_semantic_context, safe_semantic_z


class DiscourseInstabilityMetric:
    """Detect semantic ambiguity of evidence relative to verdict classes."""

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
        intra_weight: float = 0.4,
        cross_weight: float = 0.6,
        entropy_temperature: float = 5.0,
        class_deviation_weight: float = 0.25,
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
        self.intra_weight = intra_weight
        self.cross_weight = cross_weight
        self.entropy_temperature = entropy_temperature
        self.class_deviation_weight = class_deviation_weight

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

        intra_raw = np.zeros(n, dtype=float)
        entropy_raw = np.zeros(n, dtype=float)
        class_dev_raw = np.zeros(n, dtype=float)

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

        intra_raw[idx] = 1.0 - own_sim

        temp = float(max(self.entropy_temperature, 1e-3))
        logits = sims * temp
        logits = logits - np.max(logits, axis=1, keepdims=True)
        probs = np.exp(logits)
        probs = probs / np.maximum(np.sum(probs, axis=1, keepdims=True), 1e-12)
        ent = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1)
        ent_max = float(np.log(max(len(classes), 2)))
        entropy_raw[idx] = ent / max(ent_max, 1e-12)

        for cls in classes:
            col = cls_to_col[int(cls)]
            cls_mask = own_cols == col
            if not np.any(cls_mask):
                continue
            cls_vals = own_sim[cls_mask]
            cls_med = float(np.median(cls_vals))
            class_dev_raw[idx[cls_mask]] = np.abs(cls_vals - cls_med)

        z_intra = safe_semantic_z(intra_raw, context=context, ref_weights=ref_weights)
        z_entropy = safe_semantic_z(entropy_raw, context=context, ref_weights=ref_weights)
        z_class_dev = safe_semantic_z(class_dev_raw, context=context, ref_weights=ref_weights)

        w_intra = float(max(self.intra_weight, 0.0))
        w_entropy = float(max(self.cross_weight, 0.0))
        w_dev = float(max(self.class_deviation_weight, 0.0))
        w_sum = max(w_intra + w_entropy + w_dev, 1e-12)

        score = ((w_intra * z_intra) + (w_entropy * z_entropy) + (w_dev * z_class_dev)) / w_sum
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
