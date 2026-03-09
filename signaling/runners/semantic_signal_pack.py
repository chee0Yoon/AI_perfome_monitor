from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from final_metric_refactor.distribution.contradiction import ContradictionMetric
from final_metric_refactor.distribution.discourse_instability import DiscourseInstabilityMetric
from final_metric_refactor.distribution.self_contradiction import SelfContradictionMetric
from final_metric_refactor.embedding.embedder import TextEmbedder


@dataclass
class SemanticSignalPackResult:
    discourse_signal: np.ndarray
    discourse_available: np.ndarray
    discourse_meta: dict[str, Any]
    contradiction_signal: np.ndarray
    contradiction_available: np.ndarray
    contradiction_meta: dict[str, Any]
    self_contradiction_signal: np.ndarray
    self_contradiction_available: np.ndarray
    self_contradiction_meta: dict[str, Any]


class SemanticSignalPack:
    """Bundle wrapper for semantic anomaly signals (3 rules)."""

    def __init__(
        self,
        *,
        enable_discourse_instability_rule: bool,
        discourse_instability_min_support_ratio: float,
        discourse_instability_min_class_size: int,
        discourse_instability_max_classes: int,
        discourse_instability_signal_bins: int,
        discourse_instability_min_evidence_tokens: int,
        discourse_instability_intra_weight: float,
        discourse_instability_cross_weight: float,
        discourse_instability_candidate_keys: list[str] | None,
        discourse_instability_evidence_keys: list[str] | None,
        enable_contradiction_rule: bool,
        contradiction_min_support_ratio: float,
        contradiction_min_class_size: int,
        contradiction_max_classes: int,
        contradiction_signal_bins: int,
        contradiction_candidate_keys: list[str] | None,
        contradiction_evidence_keys: list[str] | None,
        contradiction_min_evidence_tokens: int,
        enable_self_contradiction_rule: bool,
        self_contradiction_min_support_ratio: float,
        self_contradiction_min_class_size: int,
        self_contradiction_max_classes: int,
        self_contradiction_signal_bins: int,
        self_contradiction_candidate_keys: list[str] | None,
        self_contradiction_evidence_keys: list[str] | None,
        self_contradiction_min_evidence_tokens: int,
    ) -> None:
        self.discourse = DiscourseInstabilityMetric(
            enabled=enable_discourse_instability_rule,
            verdict_candidate_keys=discourse_instability_candidate_keys,
            evidence_candidate_keys=discourse_instability_evidence_keys,
            min_support_ratio=discourse_instability_min_support_ratio,
            min_class_size=discourse_instability_min_class_size,
            max_classes=discourse_instability_max_classes,
            score_bins=discourse_instability_signal_bins,
            min_evidence_tokens=discourse_instability_min_evidence_tokens,
            intra_weight=discourse_instability_intra_weight,
            cross_weight=discourse_instability_cross_weight,
        )
        self.contradiction = ContradictionMetric(
            enabled=enable_contradiction_rule,
            verdict_candidate_keys=contradiction_candidate_keys,
            evidence_candidate_keys=contradiction_evidence_keys,
            min_support_ratio=contradiction_min_support_ratio,
            min_class_size=contradiction_min_class_size,
            max_classes=contradiction_max_classes,
            score_bins=contradiction_signal_bins,
            min_evidence_tokens=contradiction_min_evidence_tokens,
        )
        self.self_contradiction = SelfContradictionMetric(
            enabled=enable_self_contradiction_rule,
            verdict_candidate_keys=self_contradiction_candidate_keys,
            evidence_candidate_keys=self_contradiction_evidence_keys,
            min_support_ratio=self_contradiction_min_support_ratio,
            min_class_size=self_contradiction_min_class_size,
            max_classes=self_contradiction_max_classes,
            score_bins=self_contradiction_signal_bins,
            min_evidence_tokens=self_contradiction_min_evidence_tokens,
        )

    def compute(
        self,
        *,
        embedder: TextEmbedder,
        output_dicts: list[dict[str, Any] | None] | None,
        ref_mask: np.ndarray,
        batch_size: int,
        weights: np.ndarray | None,
        n_rows: int,
    ) -> SemanticSignalPackResult:
        if output_dicts is None:
            return SemanticSignalPackResult(
                discourse_signal=np.zeros(n_rows),
                discourse_available=np.zeros(n_rows, dtype=bool),
                discourse_meta={
                    "selected_verdict_key": None,
                    "selected_evidence_key": None,
                    "class_count": 0,
                    "eligible_rows": 0,
                    "unavailable_reason": "missing_output_dicts",
                    "label_kind": None,
                },
                contradiction_signal=np.zeros(n_rows),
                contradiction_available=np.zeros(n_rows, dtype=bool),
                contradiction_meta={
                    "selected_verdict_key": None,
                    "selected_evidence_key": None,
                    "class_count": 0,
                    "eligible_rows": 0,
                    "unavailable_reason": "missing_output_dicts",
                    "label_kind": None,
                },
                self_contradiction_signal=np.zeros(n_rows),
                self_contradiction_available=np.zeros(n_rows, dtype=bool),
                self_contradiction_meta={
                    "selected_verdict_key": None,
                    "selected_evidence_key": None,
                    "class_count": 0,
                    "eligible_rows": 0,
                    "unavailable_reason": "missing_output_dicts",
                    "label_kind": None,
                },
            )

        discourse_signal, discourse_available, discourse_meta = self.discourse.compute(
            model=embedder,
            output_dicts=output_dicts,
            ref_mask=np.asarray(ref_mask, dtype=bool),
            batch_size=batch_size,
            ref_weights=weights,
        )
        contradiction_signal, contradiction_available, contradiction_meta = self.contradiction.compute(
            model=embedder,
            output_dicts=output_dicts,
            ref_mask=np.asarray(ref_mask, dtype=bool),
            batch_size=batch_size,
            ref_weights=weights,
        )
        self_contradiction_signal, self_contradiction_available, self_contradiction_meta = (
            self.self_contradiction.compute(
                model=embedder,
                output_dicts=output_dicts,
                ref_mask=np.asarray(ref_mask, dtype=bool),
                batch_size=batch_size,
                ref_weights=weights,
            )
        )

        return SemanticSignalPackResult(
            discourse_signal=discourse_signal,
            discourse_available=discourse_available,
            discourse_meta=discourse_meta,
            contradiction_signal=contradiction_signal,
            contradiction_available=contradiction_available,
            contradiction_meta=contradiction_meta,
            self_contradiction_signal=self_contradiction_signal,
            self_contradiction_available=self_contradiction_available,
            self_contradiction_meta=self_contradiction_meta,
        )
