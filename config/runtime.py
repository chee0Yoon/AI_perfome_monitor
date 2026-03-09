from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field, fields, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_IFEVAL_IDS = ["detectable_format:json_format"]
DEFAULT_IFEVAL_KWARGS = {"detectable_format:json_format": {}}
DEFAULT_SCHEMA = {
    "error_check_process": "text",
    "total_clauses": "integer",
    "num_error_clause": "integer",
    "highlight": "text",
    "feedback_eng": "text",
    "example_correction": "text",
}
DEFAULT_TRY_ENCODINGS = ["utf-8", "utf-8-sig", "euc-kr", "cp949"]


@dataclass(frozen=True)
class TemplateRuntimeConfig:
    n_min: int = 3
    n_max: int = 7
    freq_threshold: float = 0.70
    position_std: float = 0.05
    coverage_threshold: float = 0.70
    mask_token: str = "<TPL>"


@dataclass(frozen=True)
class UdfRuntimeConfig:
    enabled: bool = True
    iterations: int = 1
    core_rules: tuple[str, ...] = ("output", "direction", "length", "diff_resid")
    q_clean: float = 0.90
    soft_alpha: float = 6.0
    min_weight: float = 0.05


@dataclass(frozen=True)
class FinalThresholdRuntimeConfig:
    threshold_points: int = 260
    robust_z_k: float = 3.5
    tail_start_floor_k: float = float("nan")
    tail_start_max_k: float = 8.0
    tail_start_min_survival: float = 0.005
    tail_start_max_survival: float = 0.30
    tail_start_grid_points: int = 160
    tail_direction: str = "two_sided"
    mad_eps: float = 1e-9
    fallback_quantile: float = 0.995
    dist_q_min: int = 1
    dist_q_max: int = 30
    dist_q_step: int = 1
    min_support_rows: int = 8
    # Core threshold (warn replacement): T_core = median(C) + kappa * 1.4826 * MAD(C),
    # where C = {score <= T_tail_start} (or mirrored for lower / radius for two-sided).
    core_kappa: float = 1.0
    core_fallback_quantile: float = 0.85
    core_min_count: int = 12
    # Exceptional out (hard replacement): second-derivative floor on tail segment.
    exceptional_d1_lambda: float = 1.0
    exceptional_d2_lambda: float = 1.0
    exceptional_consecutive: int = 4
    exceptional_grid_points: int = 140
    exceptional_min_tail_points: int = 24
    exceptional_fallback_quantile: float = 0.90
    exceptional_min_delta_ratio: float = 0.05
    # Legacy ratios are kept as emergency fallback only.
    warn_ratio_from_fail: float = 0.8
    hard_ratio_from_fail: float = 1.2


@dataclass(frozen=True)
class FinalPlotRuntimeConfig:
    hist_bins: int = 60
    ratio_bins: int = 16


@dataclass(frozen=True)
class ScoreRuntimeConfig:
    eta_hard: float = 3.0
    mass_excess_cap: float = 12.0
    z_mass_cap: float = 12.0
    raw_mass_cap: float = 6.0
    raw_mass_kfail_trigger: float = 1e6
    min_support_rows: int = 8
    sem_zero_dominance_cutoff: float = 0.90
    sem_zero_dominance_eps: float = 1e-12
    # Threshold-free score (continuous axis risk + noisy-OR bundle aggregation).
    new_score_gamma: float = 1.0
    new_score_w_core: float = 0.15
    new_score_w_rate: float = 0.55
    new_score_w_mass: float = 0.30
    new_score_hard_zero_cutoff: float = 0.05
    new_score_hard_cap_cutoff: float = 0.02
    new_score_use_hard_guard: bool = False
    new_score_hard_tail_penalty_alpha: float = 0.25
    # Threshold-free axis controls (score input uses continuous z-risk only).
    new_score_z_fail_ref: float = 2.0
    new_score_z_hard_ref: float = 3.5
    new_score_tau_rate: float = 0.60
    new_score_tau_mass: float = 0.80
    # Hard-rate weight in rate axis. Keep >7 to satisfy: fail 7 < hard 1.
    new_score_rate_lambda_output: float = 8.0
    new_score_rate_lambda_residual: float = 8.0
    new_score_rate_lambda_semantic: float = 8.0
    new_score_core_quantile: float = 0.70
    new_score_band_hard_cut: float = 1e-12
    new_score_na_neutral_score: float = 3.0
    new_score_sem_na_neutral_score: float = 3.0
    new_score_s_core_output: float = 0.50
    new_score_s_core_diff_residual: float = 0.50
    new_score_s_core_delta_ridge_ens: float = 0.50
    new_score_s_core_discourse_instability: float = 0.50
    new_score_s_core_contradiction: float = 0.50
    new_score_s_mass_output: float = 0.50
    new_score_s_mass_diff_residual: float = 0.50
    new_score_s_mass_delta_ridge_ens: float = 0.50
    new_score_s_mass_discourse_instability: float = 0.50
    new_score_s_mass_contradiction: float = 0.50
    # OUT axis fallback: when threshold-derived k_fail is abnormally large,
    # use threshold-state rates to stabilize OUT_rate/OUT_mass.
    new_score_out_kfail_fallback_cutoff: float = 20.0
    # Detailed inspection coupling (output leaf fail from detailed mode).
    new_score_detail_penalty_alpha: float = 0.35
    # Warn inspect runtime (independent from bundle scoring).
    new_score_risk_case_top_k: int = 20
    new_score_risk_case_multi_fail_cut: int = 3
    new_score_risk_case_mix_cont: float = 0.70
    new_score_risk_case_mix_state: float = 0.30
    new_score_risk_case_detail_boost: float = 0.20
    # State severity weights. Keep hard > 7*fail to satisfy fail 7 < hard 1.
    new_score_risk_case_state_hard_weight: float = 8.0
    new_score_risk_case_state_fail_weight: float = 1.0
    new_score_risk_case_preview_chars: int = 180


@dataclass(frozen=True)
class DistributionSignalRuntimeConfig:
    var_target: float = 0.90
    pca_min_dims: int = 8
    pca_max_dims: int = 64
    cov_shrinkage: float = 0.20
    min_k: int = 15
    max_k: int = 80
    k_gap_ratio: float = 1.35
    signal_quantile: float = 0.99
    delta_ens_rp_dims: int = 96
    delta_ens_alpha: float = 0.1
    delta_ens_cv_mode: str = "auto"
    delta_ens_kfolds: int = 5
    delta_ens_split_train_ratio: float = 0.8
    delta_ens_random_state: int = 42
    delta_ens_residual: str = "l2"
    delta_ens_fit_intercept: bool = True
    delta_ens_members_nystrom: int = 4
    delta_ens_members_lowrank: int = 4
    delta_ens_row_subsample: float = 0.85
    delta_ens_ranks: tuple[int, ...] = (16, 32)
    delta_ens_landmark_policy: str = "sqrt"
    delta_ens_landmark_cap: int = 512
    delta_ens_fusion: str = "robust_z_weighted"
    delta_ens_debug_members: bool = True
    diff_residual_aux_enabled: bool = True
    diff_residual_aux_lambda: float = 0.1
    diff_residual_aux_model: str = "linear"
    # Production default: keep diff_residual row-chunk parallel workers fixed.
    diff_residual_row_chunk_workers: int = 2


@dataclass(frozen=True)
class DistributionGateRuntimeConfig:
    rule_quantiles_json: str | None = None
    active_distribution_rules: tuple[str, ...] = ("output", "direction", "length", "diff_resid", "delta_ridge_ens")
    threshold_refine: bool = True
    threshold_refine_iterations: int = 2
    threshold_refine_rules: tuple[str, ...] = ("output", "direction", "length", "diff_resid")
    threshold_refine_min_size: int = 30
    min_reference_size: int = 200
    similarity_threshold: float = 0.90
    similarity_k: int = 30
    consistency_observe_only: bool = True


@dataclass(frozen=True)
class SemanticSignalRuntimeConfig:
    enable_discourse_instability_rule: bool = True
    discourse_instability_quantile: float = 0.95
    discourse_instability_min_support_ratio: float = 0.60
    discourse_instability_min_class_size: int = 20
    discourse_instability_max_classes: int = 8
    discourse_instability_signal_bins: int = 3
    discourse_instability_min_evidence_tokens: int = 2
    discourse_instability_intra_weight: float = 0.4
    discourse_instability_cross_weight: float = 0.6
    discourse_instability_candidate_keys: tuple[str, ...] = (
        "is_correct",
        "decision",
        "label",
        "result",
        "verdict",
        "score",
        "rating",
    )
    discourse_instability_evidence_keys: tuple[str, ...] = (
        "feedback",
        "explanation",
        "rationale",
        "reason",
        "analysis",
        "comment",
    )

    enable_contradiction_rule: bool = True
    contradiction_quantile: float = 0.95
    contradiction_min_support_ratio: float = 0.60
    contradiction_min_class_size: int = 20
    contradiction_max_classes: int = 8
    contradiction_signal_bins: int = 3
    contradiction_min_evidence_tokens: int = 2
    contradiction_candidate_keys: tuple[str, ...] = (
        "is_correct",
        "decision",
        "label",
        "result",
        "verdict",
        "score",
        "rating",
    )
    contradiction_evidence_keys: tuple[str, ...] = (
        "feedback",
        "explanation",
        "rationale",
        "reason",
        "analysis",
        "comment",
    )

    enable_self_contradiction_rule: bool = True
    self_contradiction_quantile: float = 0.95
    self_contradiction_min_support_ratio: float = 0.60
    self_contradiction_min_class_size: int = 20
    self_contradiction_max_classes: int = 8
    self_contradiction_signal_bins: int = 3
    self_contradiction_min_evidence_tokens: int = 2
    self_contradiction_candidate_keys: tuple[str, ...] = (
        "is_correct",
        "decision",
        "label",
        "result",
        "verdict",
        "score",
        "rating",
    )
    self_contradiction_evidence_keys: tuple[str, ...] = (
        "feedback",
        "explanation",
        "rationale",
        "reason",
        "analysis",
        "comment",
    )


@dataclass(frozen=True)
class DistanceCalibrationRuntimeConfig:
    mode: str = "off"
    path: str | None = None
    label_col: str = "sample_label"
    bad_value: str = "bad"
    cv_folds: int = 5
    min_precision: float = 0.95
    rules: tuple[str, ...] = ("output", "direction", "length", "diff_resid")
    apply_modes: tuple[str, ...] = ("nomask", "mask")
    output_quantiles: str = "0.70~0.99(step 0.01)"
    other_quantiles: str = "0.90,0.92,0.94,0.96,0.98,0.99"


@dataclass(frozen=True)
class TristateRuntimeConfig:
    enabled: bool = True
    rules: tuple[str, ...] = (
        "output",
        "direction",
        "length",
        "diff_resid",
        "delta_ridge_ens",
        "sim_conflict",
        "discourse_instability",
        "contradiction",
        "self_contradiction",
    )
    warn_quantile: float = 0.95
    warn_rule_quantiles_json: str | None = None
    calibration_mode: str = "off"
    calibration_path: str | None = None
    calibration_label_col: str = "sample_label"
    calibration_bad_value: str = "bad"
    calibration_cv_folds: int = 5
    calibration_min_fail_precision: float = 0.95
    grid_warn_quantiles: str = "0.80,0.85,0.90,0.92,0.94,0.96"
    grid_fail_quantiles: str = "0.90,0.92,0.94,0.96,0.98,0.99"
    apply_modes: tuple[str, ...] = ("nomask", "mask")


# ============================================================================
# Unified Configuration for Final Metric Execution
# ============================================================================
# Single Source of Truth for all CLI and runtime configuration


@dataclass(frozen=True)
class FinalMetricConfig:
    """Unified configuration for final metric execution.

    This consolidates all configuration sources:
    - CLI arguments (via from_cli_args)
    - Environment variables (via from_env)
    - Hardcoded defaults (dataclass field defaults)

    Replaces: RunConfig, _default_cli_values(), RunConfig defaults in run.py
    Integrates: RUNTIME objects from config/runtime.py
    """

    # ========== Required ==========
    run_tag: str

    # ========== Input Data ==========
    source_csv: Path = field(default_factory=lambda: Path(__import__("final_metric_refactor.config.data_paths", fromlist=["default_ambiguous_csv"]).default_ambiguous_csv()))
    row_results_csv: Path | None = None
    max_rows: int = 0

    # ========== Output ==========
    output_tag: str = "final_metric"
    output_dir: Path | None = None

    # ========== Execution Mode ==========
    inspection_mode: str = "integrated"
    rules: tuple[str, ...] = field(default_factory=lambda: tuple(__import__("final_metric_refactor.config.rules", fromlist=["RUNTIME_DEFAULT_ACTIVE_RULES"]).RUNTIME_DEFAULT_ACTIVE_RULES))
    tail_direction: str = "two_sided"

    # ========== Embedding ==========
    embedding_backend: str = "auto"
    embedding_model: str = "google/embeddinggemma-300m"
    embedding_batch_size: int = 64
    rebuild_embedding_cache: bool = True

    # ========== Signal Computation (from SIGNAL_RUNTIME) ==========
    diff_residual_aux_enabled: bool = True
    diff_residual_aux_lambda: float = 0.1
    diff_residual_aux_model: str = "linear"

    # ========== CSV Column Names ==========
    source_id_col: str = "id"
    results_id_col: str = "row_id"
    label_col: str = "eval"
    bad_label: str = "incorrect"
    id_col: str = "id"
    prompt_col: str = "Prompt"
    input_col: str = "input"
    output_col: str = "expectedOutput"
    schema_path: str = ""
    schema_json: str = ""

    # ========== Plot Settings (from FINAL_PLOT_RUNTIME) ==========
    hist_bins: int = 60
    ratio_bins: int = 16

    # ========== Cache Metadata ==========
    embedding_cache_meta_json: str = ""

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> FinalMetricConfig:
        """Generate FinalMetricConfig from CLI arguments (argparse.Namespace)."""
        from final_metric_refactor.config.data_paths import default_ambiguous_csv

        return cls(
            run_tag=getattr(args, "run_tag", None) or datetime.now().strftime("runtime_%Y%m%d_%H%M%S"),
            source_csv=Path(getattr(args, "source_csv", str(default_ambiguous_csv()))) if hasattr(args, "source_csv") and args.source_csv else Path(default_ambiguous_csv()),
            row_results_csv=Path(getattr(args, "row_results_csv", "")) if hasattr(args, "row_results_csv") and getattr(args, "row_results_csv", "") else None,
            output_tag=getattr(args, "tag", "final_metric"),
            output_dir=Path(getattr(args, "output_dir", "")) if hasattr(args, "output_dir") and getattr(args, "output_dir", "") else None,
            inspection_mode=getattr(args, "inspection_mode", "integrated"),
            rules=tuple(getattr(args, "rules", "").split(",")) if hasattr(args, "rules") and isinstance(getattr(args, "rules", ""), str) else tuple(__import__("final_metric_refactor.config.rules", fromlist=["RUNTIME_DEFAULT_ACTIVE_RULES"]).RUNTIME_DEFAULT_ACTIVE_RULES),
            max_rows=getattr(args, "max_rows", 0),
            tail_direction=getattr(args, "tail_direction", "two_sided"),
            embedding_backend=getattr(args, "embedding_backend", "auto"),
            embedding_model=getattr(args, "embedding_model", "google/embeddinggemma-300m"),
            embedding_batch_size=getattr(args, "embedding_batch_size", 64),
            diff_residual_aux_enabled=getattr(args, "diff_residual_aux_enabled", True),
            diff_residual_aux_lambda=getattr(args, "diff_residual_aux_lambda", 0.1),
            diff_residual_aux_model=getattr(args, "diff_residual_aux_model", "linear"),
            source_id_col=getattr(args, "source_id_col", "id"),
            results_id_col=getattr(args, "results_id_col", "row_id"),
            label_col=getattr(args, "label_col", "eval"),
            bad_label=getattr(args, "bad_label", "incorrect"),
            id_col=getattr(args, "id_col", "id"),
            prompt_col=getattr(args, "prompt_col", "Prompt"),
            input_col=getattr(args, "input_col", "input"),
            output_col=getattr(args, "output_col", "expectedOutput"),
            schema_path=getattr(args, "schema_path", ""),
            schema_json=getattr(args, "schema_json", ""),
            hist_bins=getattr(args, "hist_bins", 60),
            ratio_bins=getattr(args, "ratio_bins", 16),
            embedding_cache_meta_json=getattr(args, "embedding_cache_meta_json", ""),
            rebuild_embedding_cache=getattr(args, "rebuild_embedding_cache", True),
        )

    @classmethod
    def from_env(cls) -> FinalMetricConfig:
        """Generate FinalMetricConfig from environment variables."""
        from final_metric_refactor.config.data_paths import default_ambiguous_csv

        return cls(
            run_tag=os.environ.get("FINAL_METRIC_RUN_TAG", datetime.now().strftime("runtime_%Y%m%d_%H%M%S")),
            source_csv=Path(os.environ.get("FINAL_METRIC_SOURCE_CSV", str(default_ambiguous_csv()))),
            row_results_csv=Path(os.environ.get("FINAL_METRIC_ROW_RESULTS_CSV")) if os.environ.get("FINAL_METRIC_ROW_RESULTS_CSV") else None,
            output_tag=os.environ.get("FINAL_METRIC_OUTPUT_TAG", "final_metric"),
            output_dir=Path(os.environ.get("FINAL_METRIC_OUTPUT_DIR")) if os.environ.get("FINAL_METRIC_OUTPUT_DIR") else None,
            inspection_mode=os.environ.get("FINAL_METRIC_INSPECTION_MODE", "integrated"),
            rules=tuple(os.environ.get("FINAL_METRIC_RULES", "").split(",")) if os.environ.get("FINAL_METRIC_RULES") else tuple(__import__("final_metric_refactor.config.rules", fromlist=["RUNTIME_DEFAULT_ACTIVE_RULES"]).RUNTIME_DEFAULT_ACTIVE_RULES),
            max_rows=int(os.environ.get("FINAL_METRIC_MAX_ROWS", "0")),
            tail_direction=os.environ.get("FINAL_METRIC_TAIL_DIRECTION", "two_sided"),
            embedding_backend=os.environ.get("FINAL_METRIC_EMBEDDING_BACKEND", "auto"),
            embedding_model=os.environ.get("FINAL_METRIC_EMBEDDING_MODEL", "google/embeddinggemma-300m"),
            embedding_batch_size=int(os.environ.get("FINAL_METRIC_EMBEDDING_BATCH_SIZE", "64")),
            diff_residual_aux_enabled=os.environ.get("FINAL_METRIC_DIFF_RESIDUAL_AUX_ENABLED", "true").lower() == "true",
            diff_residual_aux_lambda=float(os.environ.get("FINAL_METRIC_DIFF_RESIDUAL_AUX_LAMBDA", "0.1")),
            diff_residual_aux_model=os.environ.get("FINAL_METRIC_DIFF_RESIDUAL_AUX_MODEL", "linear"),
            source_id_col=os.environ.get("FINAL_METRIC_SOURCE_ID_COL", "id"),
            results_id_col=os.environ.get("FINAL_METRIC_RESULTS_ID_COL", "row_id"),
            label_col=os.environ.get("FINAL_METRIC_LABEL_COL", "eval"),
            bad_label=os.environ.get("FINAL_METRIC_BAD_LABEL", "incorrect"),
            id_col=os.environ.get("FINAL_METRIC_ID_COL", "id"),
            prompt_col=os.environ.get("FINAL_METRIC_PROMPT_COL", "Prompt"),
            input_col=os.environ.get("FINAL_METRIC_INPUT_COL", "input"),
            output_col=os.environ.get("FINAL_METRIC_OUTPUT_COL", "expectedOutput"),
            schema_path=os.environ.get("FINAL_METRIC_SCHEMA_PATH", ""),
            schema_json=os.environ.get("FINAL_METRIC_SCHEMA_JSON", ""),
            hist_bins=int(os.environ.get("FINAL_METRIC_HIST_BINS", "60")),
            ratio_bins=int(os.environ.get("FINAL_METRIC_RATIO_BINS", "16")),
            embedding_cache_meta_json=os.environ.get("FINAL_METRIC_EMBEDDING_CACHE_META_JSON", ""),
            rebuild_embedding_cache=os.environ.get("FINAL_METRIC_REBUILD_EMBEDDING_CACHE", "true").lower() == "true",
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        d["source_csv"] = str(self.source_csv)
        d["row_results_csv"] = str(self.row_results_csv) if self.row_results_csv else None
        d["output_dir"] = str(self.output_dir) if self.output_dir else None
        return d

    def resolve_output_dir(self) -> Path:
        """Resolve output directory (with results/<run_tag> pattern if not specified)."""
        if self.output_dir:
            return self.output_dir
        # Default to results/<run_tag>/
        return Path(__import__("final_metric_refactor", fromlist=["__file__"]).__file__).parent.parent / "results" / self.run_tag


TEMPLATE_RUNTIME = TemplateRuntimeConfig()
UDF_RUNTIME = UdfRuntimeConfig()
FINAL_THRESHOLD_RUNTIME = FinalThresholdRuntimeConfig()
FINAL_PLOT_RUNTIME = FinalPlotRuntimeConfig()
SCORE_RUNTIME = ScoreRuntimeConfig()
DISTRIBUTION_SIGNAL_RUNTIME = DistributionSignalRuntimeConfig()
DISTRIBUTION_GATE_RUNTIME = DistributionGateRuntimeConfig()
SEMANTIC_SIGNAL_RUNTIME = SemanticSignalRuntimeConfig()
DISTANCE_CALIBRATION_RUNTIME = DistanceCalibrationRuntimeConfig()
TRISTATE_RUNTIME = TristateRuntimeConfig()
