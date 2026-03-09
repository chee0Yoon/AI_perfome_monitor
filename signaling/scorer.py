"""Distribution signal wrapper - orchestrates embedding-based anomaly signals."""

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import numpy as np

from final_metric_refactor.embedding.embedder import TextEmbedder
from final_metric_refactor.shared.geometry import normalize_rows, sanitize_matrix
from final_metric_refactor.distribution._shared import build_local_knn_context
from final_metric_refactor.signaling.runners import (
    DeltaRidgeEnsSignalRunner,
    DiffResidualSignalRunner,
    DirectionSignalRunner,
    LengthSignalRunner,
    OutputSignalRunner,
    SemanticSignalPack,
    SimilarInputConflictSignalRunner,
)

if TYPE_CHECKING:
    from final_metric_refactor.config import FinalMetricConfig


@dataclass
class DistributionSignalResult:
    output_signal: np.ndarray
    direction_signal: np.ndarray
    length_signal: np.ndarray
    diff_residual_signal: np.ndarray
    delta_ridge_ens_signal: np.ndarray
    similar_input_conflict_signal: np.ndarray
    discourse_instability_signal: np.ndarray
    discourse_instability_available: np.ndarray
    discourse_instability_meta: dict[str, Any]
    contradiction_signal: np.ndarray
    contradiction_available: np.ndarray
    contradiction_meta: dict[str, Any]
    self_contradiction_signal: np.ndarray
    self_contradiction_available: np.ndarray
    self_contradiction_meta: dict[str, Any]
    consistency_rule_signal: np.ndarray
    consistency_rule_available: np.ndarray
    local_k: np.ndarray
    tau: np.ndarray
    used_output_density_ks: list[int]
    x_norm: np.ndarray
    y_norm: np.ndarray
    diff_residual_meta: dict[str, Any]
    delta_ridge_ens_meta: dict[str, Any]


class DistributionScorer:
    """Orchestrates embedding and all distribution-based anomaly signals.

    Notes:
        Result field names use ``*_signal`` consistently.
    """

    def __init__(
        self,
        embedder: TextEmbedder,
        var_target: float = 0.90,
        pca_min_dims: int = 8,
        pca_max_dims: int = 64,
        min_k: int = 3,
        max_k: int = 50,
        gap_ratio: float = 1.5,
        cov_shrinkage: float = 0.20,
        diff_residual_aux_enabled: bool = False,
        diff_residual_aux_lambda: float = 0.7,
        diff_residual_aux_model: str = "linear",
        diff_residual_row_chunk_workers: int = 0,
        delta_ens_rp_dims: int = 96,
        delta_ens_alpha: float = 0.1,
        delta_ens_cv_mode: str = "auto",
        delta_ens_kfolds: int = 5,
        delta_ens_split_train_ratio: float = 0.8,
        delta_ens_random_state: int = 42,
        delta_ens_residual: str = "l2",
        delta_ens_fit_intercept: bool = True,
        delta_ens_members_nystrom: int = 4,
        delta_ens_members_lowrank: int = 4,
        delta_ens_row_subsample: float = 0.85,
        delta_ens_ranks: list[int] | tuple[int, ...] = (16, 32),
        delta_ens_landmark_policy: str = "sqrt",
        delta_ens_landmark_cap: int = 512,
        delta_ens_fusion: str = "robust_z_weighted",
        delta_ens_debug_members: bool = True,
        similarity_threshold: float = 0.9,
        similarity_k: int = 50,
        enable_discourse_instability_rule: bool = True,
        discourse_instability_min_support_ratio: float = 0.60,
        discourse_instability_min_class_size: int = 20,
        discourse_instability_max_classes: int = 8,
        discourse_instability_signal_bins: int = 3,
        discourse_instability_min_evidence_tokens: int = 2,
        discourse_instability_intra_weight: float = 0.4,
        discourse_instability_cross_weight: float = 0.6,
        discourse_instability_candidate_keys: list[str] | None = None,
        discourse_instability_evidence_keys: list[str] | None = None,
        enable_contradiction_rule: bool = True,
        contradiction_min_support_ratio: float = 0.60,
        contradiction_min_class_size: int = 20,
        contradiction_max_classes: int = 8,
        contradiction_signal_bins: int = 3,
        contradiction_candidate_keys: list[str] | None = None,
        contradiction_evidence_keys: list[str] | None = None,
        contradiction_min_evidence_tokens: int = 2,
        enable_self_contradiction_rule: bool = True,
        self_contradiction_min_support_ratio: float = 0.60,
        self_contradiction_min_class_size: int = 20,
        self_contradiction_max_classes: int = 8,
        self_contradiction_signal_bins: int = 3,
        self_contradiction_candidate_keys: list[str] | None = None,
        self_contradiction_evidence_keys: list[str] | None = None,
        self_contradiction_min_evidence_tokens: int = 2,
    ):
        # Kept for backward-compatible constructor shape; signal computation is PCA-free.
        self.embedder = embedder
        self.var_target = var_target
        self.pca_min_dims = pca_min_dims
        self.pca_max_dims = pca_max_dims
        self.min_k = min_k
        self.max_k = max_k
        self.gap_ratio = gap_ratio
        self.cov_shrinkage = cov_shrinkage
        self.similarity_threshold = similarity_threshold
        self.similarity_k = similarity_k

        self.output_signal_runner = OutputSignalRunner(min_k=min_k, max_k=max_k)
        self.direction_signal_runner = DirectionSignalRunner()
        self.length_signal_runner = LengthSignalRunner()
        self.diff_residual_signal_runner = DiffResidualSignalRunner(
            cov_shrinkage=cov_shrinkage,
            aux_enabled=diff_residual_aux_enabled,
            aux_lambda=diff_residual_aux_lambda,
            aux_model=diff_residual_aux_model,
            row_chunk_workers=diff_residual_row_chunk_workers,
        )
        self.delta_ridge_ens_signal_runner = DeltaRidgeEnsSignalRunner(
            rp_dims=delta_ens_rp_dims,
            alpha=delta_ens_alpha,
            cv_mode=delta_ens_cv_mode,
            kfolds=delta_ens_kfolds,
            split_train_ratio=delta_ens_split_train_ratio,
            random_state=delta_ens_random_state,
            residual=delta_ens_residual,
            fit_intercept=delta_ens_fit_intercept,
            members_nystrom=delta_ens_members_nystrom,
            members_lowrank=delta_ens_members_lowrank,
            row_subsample=delta_ens_row_subsample,
            ranks=delta_ens_ranks,
            landmark_policy=delta_ens_landmark_policy,
            landmark_cap=delta_ens_landmark_cap,
            fusion=delta_ens_fusion,
            debug_members=delta_ens_debug_members,
        )
        self.similar_input_conflict_signal_runner = SimilarInputConflictSignalRunner(
            similarity_threshold=similarity_threshold,
            similarity_k=similarity_k,
        )
        self.semantic_signal_pack = SemanticSignalPack(
            enable_discourse_instability_rule=enable_discourse_instability_rule,
            discourse_instability_candidate_keys=discourse_instability_candidate_keys,
            discourse_instability_evidence_keys=discourse_instability_evidence_keys,
            discourse_instability_min_support_ratio=discourse_instability_min_support_ratio,
            discourse_instability_min_class_size=discourse_instability_min_class_size,
            discourse_instability_max_classes=discourse_instability_max_classes,
            discourse_instability_signal_bins=discourse_instability_signal_bins,
            discourse_instability_min_evidence_tokens=discourse_instability_min_evidence_tokens,
            discourse_instability_intra_weight=discourse_instability_intra_weight,
            discourse_instability_cross_weight=discourse_instability_cross_weight,
            enable_contradiction_rule=enable_contradiction_rule,
            contradiction_min_support_ratio=contradiction_min_support_ratio,
            contradiction_min_class_size=contradiction_min_class_size,
            contradiction_max_classes=contradiction_max_classes,
            contradiction_signal_bins=contradiction_signal_bins,
            contradiction_candidate_keys=contradiction_candidate_keys,
            contradiction_evidence_keys=contradiction_evidence_keys,
            contradiction_min_evidence_tokens=contradiction_min_evidence_tokens,
            enable_self_contradiction_rule=enable_self_contradiction_rule,
            self_contradiction_min_support_ratio=self_contradiction_min_support_ratio,
            self_contradiction_min_class_size=self_contradiction_min_class_size,
            self_contradiction_max_classes=self_contradiction_max_classes,
            self_contradiction_signal_bins=self_contradiction_signal_bins,
            self_contradiction_candidate_keys=self_contradiction_candidate_keys,
            self_contradiction_evidence_keys=self_contradiction_evidence_keys,
            self_contradiction_min_evidence_tokens=self_contradiction_min_evidence_tokens,
        )

    def compute_signals(
        self,
        input_texts: list[str],
        output_texts: list[str],
        source_output_texts: list[str],
        ref_mask: np.ndarray,
        output_dicts: list[dict[str, Any] | None] | None = None,
        batch_size: int = 64,
        sample_weights: np.ndarray | None = None,
    ) -> DistributionSignalResult:
        n = len(input_texts)

        input_embs = self.embedder.encode(input_texts, batch_size=batch_size)
        output_embs = self.embedder.encode(output_texts, batch_size=batch_size)

        input_embs = sanitize_matrix(input_embs)
        output_embs = sanitize_matrix(output_embs)
        x_norm = normalize_rows(input_embs)
        y_norm = normalize_rows(output_embs)

        d_raw = output_embs - input_embs
        tau = np.linalg.norm(d_raw, axis=1)
        u = normalize_rows(d_raw)

        weights = None
        if sample_weights is not None:
            weights = np.asarray(sample_weights, dtype=float)
            if len(weights) != n:
                raise ValueError("sample_weights length mismatch in DistributionScorer.compute_signals")
            weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
            weights = np.clip(weights, 0.0, 1e12)

        output_signal, output_ks = self.output_signal_runner.compute(
            y_norm=y_norm,
            ref_mask=ref_mask,
            weights=weights,
        )

        knn_context = build_local_knn_context(x_norm, self.max_k, self.min_k, self.gap_ratio)
        direction_signal = self.direction_signal_runner.compute(
            unit_delta=u,
            knn_context=knn_context,
            weights=weights,
            tau=tau,
        )
        length_signal = self.length_signal_runner.compute(
            tau=tau,
            knn_context=knn_context,
            weights=weights,
        )
        diff_residual_signal, diff_residual_meta = self.diff_residual_signal_runner.compute(
            d_raw=d_raw,
            knn_context=knn_context,
            weights=weights,
            input_embs=input_embs,
            direction_signal=direction_signal,
            length_signal=length_signal,
            ref_mask=ref_mask,
        )
        delta_ridge_ens_signal, delta_ridge_ens_meta = self.delta_ridge_ens_signal_runner.compute(
            d_raw=d_raw,
            knn_context=knn_context,
            weights=weights,
            input_embs=input_embs,
        )

        sim_conflict_signal = self.similar_input_conflict_signal_runner.compute(
            x_norm=x_norm,
            unit_delta=u,
            tau=tau,
        )

        semantic_result = self.semantic_signal_pack.compute(
            embedder=self.embedder,
            output_dicts=output_dicts,
            ref_mask=ref_mask,
            batch_size=batch_size,
            weights=weights,
            n_rows=n,
        )
        discourse_instability_signal = semantic_result.discourse_signal
        discourse_instability_available = semantic_result.discourse_available
        discourse_instability_meta = semantic_result.discourse_meta
        contradiction_signal = semantic_result.contradiction_signal
        contradiction_available = semantic_result.contradiction_available
        contradiction_meta = semantic_result.contradiction_meta
        self_contradiction_signal = semantic_result.self_contradiction_signal
        self_contradiction_available = semantic_result.self_contradiction_available
        self_contradiction_meta = semantic_result.self_contradiction_meta

        consistency_rule_signal = np.zeros(n)
        consistency_rule_available = np.zeros(n, dtype=bool)

        return DistributionSignalResult(
            output_signal=output_signal,
            direction_signal=direction_signal,
            length_signal=length_signal,
            diff_residual_signal=diff_residual_signal,
            delta_ridge_ens_signal=delta_ridge_ens_signal,
            similar_input_conflict_signal=sim_conflict_signal,
            discourse_instability_signal=discourse_instability_signal,
            discourse_instability_available=discourse_instability_available,
            discourse_instability_meta=discourse_instability_meta,
            contradiction_signal=contradiction_signal,
            contradiction_available=contradiction_available,
            contradiction_meta=contradiction_meta,
            self_contradiction_signal=self_contradiction_signal,
            self_contradiction_available=self_contradiction_available,
            self_contradiction_meta=self_contradiction_meta,
            consistency_rule_signal=consistency_rule_signal,
            consistency_rule_available=consistency_rule_available,
            local_k=knn_context.used_k,
            tau=tau,
            used_output_density_ks=output_ks,
            x_norm=x_norm,
            y_norm=y_norm,
            diff_residual_meta=diff_residual_meta,
            delta_ridge_ens_meta=delta_ridge_ens_meta,
        )

    @classmethod
    def from_config(
        cls,
        embedder: TextEmbedder,
        config: "FinalMetricConfig",
    ) -> "DistributionScorer":
        """Create DistributionScorer from FinalMetricConfig (single-source-of-truth pattern).

        This factory method eliminates the need to pass 50+ individual parameters.
        Instead, configuration is loaded from FinalMetricConfig and RUNTIME objects.

        Args:
            embedder: TextEmbedder instance for text vectorization.
            config: FinalMetricConfig instance with unified configuration.

        Returns:
            DistributionScorer instance fully configured from the config object.
        """
        from final_metric_refactor.config import (
            DISTRIBUTION_SIGNAL_RUNTIME,
            SEMANTIC_SIGNAL_RUNTIME,
        )

        # Extract parameters from RUNTIME objects
        sig_rt = DISTRIBUTION_SIGNAL_RUNTIME
        sem_rt = SEMANTIC_SIGNAL_RUNTIME

        return cls(
            embedder=embedder,
            # Distribution signal parameters
            var_target=float(getattr(sig_rt, "var_target", 0.90)),
            pca_min_dims=int(getattr(sig_rt, "pca_min_dims", 8)),
            pca_max_dims=int(getattr(sig_rt, "pca_max_dims", 64)),
            min_k=int(getattr(sig_rt, "min_k", 15)),
            max_k=int(getattr(sig_rt, "max_k", 80)),
            gap_ratio=float(getattr(sig_rt, "k_gap_ratio", 1.35)),
            cov_shrinkage=float(getattr(sig_rt, "cov_shrinkage", 0.20)),
            # Diff residual parameters (from config + runtime)
            diff_residual_aux_enabled=bool(config.diff_residual_aux_enabled),
            diff_residual_aux_lambda=float(config.diff_residual_aux_lambda),
            diff_residual_aux_model=str(config.diff_residual_aux_model),
            diff_residual_row_chunk_workers=int(getattr(sig_rt, "diff_residual_row_chunk_workers", 2)),
            # Delta ensemble parameters
            delta_ens_rp_dims=int(getattr(sig_rt, "delta_ens_rp_dims", 96)),
            delta_ens_alpha=float(getattr(sig_rt, "delta_ens_alpha", 0.1)),
            delta_ens_cv_mode=str(getattr(sig_rt, "delta_ens_cv_mode", "auto")),
            delta_ens_kfolds=int(getattr(sig_rt, "delta_ens_kfolds", 5)),
            delta_ens_split_train_ratio=float(getattr(sig_rt, "delta_ens_split_train_ratio", 0.8)),
            delta_ens_random_state=int(getattr(sig_rt, "delta_ens_random_state", 42)),
            delta_ens_residual=str(getattr(sig_rt, "delta_ens_residual", "l2")),
            delta_ens_fit_intercept=bool(getattr(sig_rt, "delta_ens_fit_intercept", True)),
            delta_ens_members_nystrom=int(getattr(sig_rt, "delta_ens_members_nystrom", 4)),
            delta_ens_members_lowrank=int(getattr(sig_rt, "delta_ens_members_lowrank", 4)),
            delta_ens_row_subsample=float(getattr(sig_rt, "delta_ens_row_subsample", 0.85)),
            delta_ens_ranks=getattr(sig_rt, "delta_ens_ranks", (16, 32)),
            delta_ens_landmark_policy=str(getattr(sig_rt, "delta_ens_landmark_policy", "sqrt")),
            delta_ens_landmark_cap=int(getattr(sig_rt, "delta_ens_landmark_cap", 512)),
            delta_ens_fusion=str(getattr(sig_rt, "delta_ens_fusion", "robust_z_weighted")),
            delta_ens_debug_members=bool(getattr(sig_rt, "delta_ens_debug_members", True)),
            # Similarity parameters
            similarity_threshold=float(getattr(sig_rt, "similarity_threshold", 0.9)),
            similarity_k=int(getattr(sig_rt, "similarity_k", 30)),
            # Semantic parameters
            enable_discourse_instability_rule=bool(getattr(sem_rt, "enable_discourse_instability_rule", True)),
            discourse_instability_min_support_ratio=float(getattr(sem_rt, "discourse_instability_min_support_ratio", 0.60)),
            discourse_instability_min_class_size=int(getattr(sem_rt, "discourse_instability_min_class_size", 20)),
            discourse_instability_max_classes=int(getattr(sem_rt, "discourse_instability_max_classes", 8)),
            discourse_instability_signal_bins=int(getattr(sem_rt, "discourse_instability_signal_bins", 3)),
            discourse_instability_min_evidence_tokens=int(getattr(sem_rt, "discourse_instability_min_evidence_tokens", 2)),
            discourse_instability_intra_weight=float(getattr(sem_rt, "discourse_instability_intra_weight", 0.4)),
            discourse_instability_cross_weight=float(getattr(sem_rt, "discourse_instability_cross_weight", 0.6)),
            discourse_instability_candidate_keys=list(getattr(sem_rt, "discourse_instability_candidate_keys", [])),
            discourse_instability_evidence_keys=list(getattr(sem_rt, "discourse_instability_evidence_keys", [])),
            enable_contradiction_rule=bool(getattr(sem_rt, "enable_contradiction_rule", True)),
            contradiction_min_support_ratio=float(getattr(sem_rt, "contradiction_min_support_ratio", 0.60)),
            contradiction_min_class_size=int(getattr(sem_rt, "contradiction_min_class_size", 20)),
            contradiction_max_classes=int(getattr(sem_rt, "contradiction_max_classes", 8)),
            contradiction_signal_bins=int(getattr(sem_rt, "contradiction_signal_bins", 3)),
            contradiction_candidate_keys=list(getattr(sem_rt, "contradiction_candidate_keys", [])),
            contradiction_evidence_keys=list(getattr(sem_rt, "contradiction_evidence_keys", [])),
            contradiction_min_evidence_tokens=int(getattr(sem_rt, "contradiction_min_evidence_tokens", 2)),
            enable_self_contradiction_rule=bool(getattr(sem_rt, "enable_self_contradiction_rule", True)),
            self_contradiction_min_support_ratio=float(getattr(sem_rt, "self_contradiction_min_support_ratio", 0.60)),
            self_contradiction_min_class_size=int(getattr(sem_rt, "self_contradiction_min_class_size", 20)),
            self_contradiction_max_classes=int(getattr(sem_rt, "self_contradiction_max_classes", 8)),
            self_contradiction_signal_bins=int(getattr(sem_rt, "self_contradiction_signal_bins", 3)),
            self_contradiction_candidate_keys=list(getattr(sem_rt, "self_contradiction_candidate_keys", [])),
            self_contradiction_evidence_keys=list(getattr(sem_rt, "self_contradiction_evidence_keys", [])),
            self_contradiction_min_evidence_tokens=int(getattr(sem_rt, "self_contradiction_min_evidence_tokens", 2)),
        )

    # Backward-compatibility shim for existing callsites.
    def score(
        self,
        input_texts: list[str],
        output_texts: list[str],
        source_output_texts: list[str],
        ref_mask: np.ndarray,
        output_dicts: list[dict[str, Any] | None] | None = None,
        batch_size: int = 64,
        sample_weights: np.ndarray | None = None,
    ) -> DistributionSignalResult:
        return self.compute_signals(
            input_texts=input_texts,
            output_texts=output_texts,
            source_output_texts=source_output_texts,
            ref_mask=ref_mask,
            output_dicts=output_dicts,
            batch_size=batch_size,
            sample_weights=sample_weights,
        )

