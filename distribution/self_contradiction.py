"""Self Contradiction Metric - verdict/evidence contradiction risk score."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from final_metric_refactor.shared.geometry import normalize_rows
from final_metric_refactor.distribution._semantic import build_semantic_context, safe_semantic_z

SENT_SPLIT_RE = re.compile(r"[.!?;:\n,]|[。！？；]|(?:\s-\s)")
TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


def _split_evidence_segments(text: str, min_tokens: int = 2, max_segments: int = 4) -> list[str]:
    parts = SENT_SPLIT_RE.split(str(text))
    out: list[str] = []
    for p in parts:
        seg = str(p).strip()
        if not seg:
            continue
        if len(TOKEN_RE.findall(seg)) < int(min_tokens):
            continue
        out.append(seg)
        if len(out) >= int(max_segments):
            break
    return out


class SelfContradictionMetric:
    """Detect evidence vectors closer to opposite-class anchors than own-class anchors."""

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
        clause_dispersion_weight: float = 0.30,
        clause_flip_weight: float = 0.15,
        max_clause_segments: int = 4,
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
        self.clause_dispersion_weight = clause_dispersion_weight
        self.clause_flip_weight = clause_flip_weight
        self.max_clause_segments = max_clause_segments

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
        dispersion_raw = np.zeros(n, dtype=float)
        clause_flip_raw = np.zeros(n, dtype=float)

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
            other_sim[j] = float(np.max(row)) if np.isfinite(np.max(row)) else -1.0

        margin_raw[idx] = other_sim - own_sim

        evidence_key = str(context.selected_evidence_key or "").strip()
        if evidence_key:
            segment_texts: list[str] = []
            segment_rows: list[int] = []
            row_to_seg_pos: dict[int, list[int]] = {}
            for row_idx in idx.tolist():
                row = output_dicts[row_idx] if row_idx < len(output_dicts) else None
                if not isinstance(row, dict):
                    continue
                txt = row.get(evidence_key)
                if txt is None:
                    continue
                segs = _split_evidence_segments(
                    text=str(txt),
                    min_tokens=max(1, int(self.min_evidence_tokens)),
                    max_segments=max(2, int(self.max_clause_segments)),
                )
                if len(segs) < 2:
                    continue
                for seg in segs:
                    pos = len(segment_texts)
                    segment_texts.append(seg)
                    segment_rows.append(int(row_idx))
                    row_to_seg_pos.setdefault(int(row_idx), []).append(pos)

            if segment_texts:
                seg_emb = model.encode(segment_texts, batch_size=batch_size)
                seg_emb = normalize_rows(np.asarray(seg_emb, dtype=float))
                for row_idx, pos_list in row_to_seg_pos.items():
                    if len(pos_list) < 2:
                        continue
                    seg_vecs = seg_emb[np.asarray(pos_list, dtype=int)]
                    sim_mat = np.asarray(seg_vecs @ seg_vecs.T, dtype=float)
                    tri_u = np.triu_indices(sim_mat.shape[0], k=1)
                    if len(tri_u[0]) > 0:
                        mean_pair_sim = float(np.mean(sim_mat[tri_u]))
                        dispersion_raw[int(row_idx)] = max(0.0, 1.0 - mean_pair_sim)

                    own_col = int(cls_to_col[int(labels[int(row_idx)])])
                    seg_cent_sims = np.asarray(seg_vecs @ centroid_mat.T, dtype=float)
                    seg_own = seg_cent_sims[:, own_col]
                    seg_other = np.full(len(seg_own), -1.0, dtype=float)
                    for j in range(seg_cent_sims.shape[0]):
                        r = seg_cent_sims[j].copy()
                        r[own_col] = -np.inf
                        r_max = np.max(r)
                        seg_other[j] = float(r_max) if np.isfinite(r_max) else -1.0
                    clause_flip_raw[int(row_idx)] = float(np.mean(seg_other > seg_own))

        z_margin = safe_semantic_z(margin_raw, context=context, ref_weights=ref_weights)
        z_dispersion = safe_semantic_z(dispersion_raw, context=context, ref_weights=ref_weights)
        z_clause_flip = safe_semantic_z(clause_flip_raw, context=context, ref_weights=ref_weights)

        w_margin = 1.0
        w_disp = float(max(self.clause_dispersion_weight, 0.0))
        w_flip = float(max(self.clause_flip_weight, 0.0))
        w_sum = max(w_margin + w_disp + w_flip, 1e-12)

        score = ((w_margin * z_margin) + (w_disp * z_dispersion) + (w_flip * z_clause_flip)) / w_sum
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
