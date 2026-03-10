#!/usr/bin/env python3
"""Final metric runtime for deployment-focused threshold extraction.

This script consumes an existing row-results CSV (with distribution signals)
and produces:
- row_results.csv (updated pass/fail and thresholds)
- summary.csv (operational summary)
- run_config.json
- rule_thresholds.csv (rule-level core/tail_start/exceptional_out + warn/fail/hard aliases)
- rule_thresholds_compact.csv (compact view)
- distribution diagnostics HTML (Graph1/2/3)
- diagnostics summary CSV

Threshold policy is fixed to hybrid derivative:
1) robust_z_tail_start
2) dist_stability_jump fallback on strict abnormal conditions
3) quantile_tail guard fallback

Tri-threshold naming:
- core (warn alias)
- tail_start (fail alias)
- exceptional_out (hard alias)
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
try:
    import plotly.graph_objects as go
    import plotly.io as pio
except Exception:  # pragma: no cover
    go = None
    pio = None


FINAL_DIR = Path(__file__).resolve().parent

# Final-metric local threshold and diagnostics helpers.
from final_metric_refactor.threshold import (
    AVAILABLE_COL,
    SIGNAL_COL,
    binary_metrics,
    choose_rule_threshold_and_fail,
    compute_labels_bad,
    compute_policy_features,
    derive_tristate_thresholds_from_fail,
    resolve_signal_col,
    trigger_mask,
)
from final_metric_refactor.report.plots import (
    RULE_LABEL,
    make_signal_box_fig,
    make_signal_hist_fig,
    signal_hist_ratio,
)
from final_metric_refactor.shared.geometry import knn_self  # type: ignore
from final_metric_refactor.embedding.embedder import build_embedder as _build_embedder_module
from final_metric_refactor.embedding.cache import (
    build_embedding_cache_paths,
    load_embedding_cache,
    load_or_rebuild_embedding_cache,
    resolve_embedding_cache_paths,
)
from final_metric_refactor.config import (
    DISTRIBUTION_SIGNAL_RUNTIME,
    FINAL_PLOT_RUNTIME,
    SCORE_RUNTIME,
    FINAL_THRESHOLD_RUNTIME,
    RUNTIME_DEFAULT_ACTIVE_RULES,
    normalize_runtime_rule_key,
    FinalMetricConfig,
)
from final_metric_refactor.scoring import compute_bundle_scores, render_bundle_score_dashboard
from final_metric_refactor.scoring.warn_inspect import compute_warn_inspect
from final_metric_refactor.config.data_paths import default_ambiguous_csv
from final_metric_refactor.signaling import run_distribution_pipeline
from final_metric_refactor.report.writer import ensure_report_dir, write_csv, write_json


DEFAULT_RULES = list(RUNTIME_DEFAULT_ACTIVE_RULES)
SIGNAL_RUNTIME = DISTRIBUTION_SIGNAL_RUNTIME


# ============================================================================
# Legacy RunConfig - Now using FinalMetricConfig for all configuration
# Kept as alias for backward compatibility during transition
# ============================================================================
RunConfig = FinalMetricConfig


@dataclass(frozen=True)
class RunArtifacts:
    run_root: Path
    report_dir: Path
    row_results_csv: Path
    summary_csv: Path
    run_config_json: Path
    diagnostics_html: Path
    diagnostics_summary_csv: Path


def build_parser() -> argparse.ArgumentParser:
    """Build argparse.ArgumentParser for FinalMetricConfig from CLI."""
    p = argparse.ArgumentParser(description="Run final metric thresholding + diagnostics plot")

    # Input data
    p.add_argument(
        "--source-csv",
        default=str(default_ambiguous_csv()),
        help="Source CSV path.",
    )
    p.add_argument("--row-results-csv", default="", help="Existing row_results CSV. Omit to bootstrap from source.")
    p.add_argument("--max-rows", type=int, default=0, help="0 means full dataset.")

    # Output
    p.add_argument("--output-dir", required=True, help="Output root directory.")
    p.add_argument("--tag", default="final_metric", help="Output filename tag.")
    p.add_argument("--run-tag", default="", help="Run tag/identifier (default: auto-generated timestamp)")

    # Execution mode
    p.add_argument(
        "--inspection-mode",
        default="integrated",
        choices=["integrated", "detailed"],
        help="Use detailed when leaf diagnostics are required.",
    )
    p.add_argument("--rules", default=",".join(DEFAULT_RULES), help="Comma-separated active rule list.")
    p.add_argument(
        "--tail-direction",
        default=str(FINAL_THRESHOLD_RUNTIME.tail_direction),
        choices=["upper", "lower", "two_sided"],
        help="Tail direction for thresholding.",
    )

    # Embedding
    p.add_argument("--embedding-backend", default="auto", help="Embedding backend (auto/sentence-transformers/hash)")
    p.add_argument("--embedding-model", default="google/embeddinggemma-300m", help="Embedding model name")
    p.add_argument("--embedding-batch-size", type=int, default=64, help="Embedding batch size")
    p.add_argument("--rebuild-embedding-cache", action="store_true", default=True, help="Rebuild embedding cache")

    # Signal computation
    p.add_argument(
        "--diff-residual-aux-enabled",
        action="store_true",
        default=bool(SIGNAL_RUNTIME.diff_residual_aux_enabled),
        help="Enable diff residual auxiliary",
    )
    p.add_argument(
        "--diff-residual-aux-lambda",
        type=float,
        default=float(SIGNAL_RUNTIME.diff_residual_aux_lambda),
        help="Diff residual auxiliary lambda",
    )
    p.add_argument(
        "--diff-residual-aux-model",
        default=str(SIGNAL_RUNTIME.diff_residual_aux_model),
        help="Diff residual auxiliary model",
    )

    # CSV columns
    p.add_argument("--source-id-col", default="id", help="Source ID column name")
    p.add_argument("--results-id-col", default="row_id", help="Results ID column name")
    p.add_argument("--label-col", default="eval", help="Label column name")
    p.add_argument("--bad-label", default="incorrect", help="Bad label value")
    p.add_argument("--id-col", default="id", help="ID column name")
    p.add_argument("--prompt-col", default="Prompt", help="Prompt column name")
    p.add_argument("--input-col", default="input", help="Input column name")
    p.add_argument("--output-col", default="expectedOutput", help="Output column name")

    # Plot settings
    p.add_argument("--hist-bins", type=int, default=int(FINAL_PLOT_RUNTIME.hist_bins), help="Histogram bins")
    p.add_argument("--ratio-bins", type=int, default=int(FINAL_PLOT_RUNTIME.ratio_bins), help="Ratio bins")

    # Cache
    p.add_argument("--embedding-cache-meta-json", default="", help="Embedding cache metadata JSON")

    return p


def parse_args() -> FinalMetricConfig:
    """Parse CLI arguments and return FinalMetricConfig."""
    p = build_parser()
    args = p.parse_args()
    return FinalMetricConfig.from_cli_args(args)


def _config_to_namespace(config: FinalMetricConfig) -> argparse.Namespace:
    """Convert FinalMetricConfig to argparse.Namespace for internal use."""
    output_root = config.resolve_output_dir()
    return argparse.Namespace(
        run_tag=str(config.run_tag),
        source_csv=str(Path(config.source_csv).resolve()),
        row_results_csv=(str(Path(config.row_results_csv).resolve()) if config.row_results_csv else ""),
        output_dir=str(output_root),
        report_dir_name="report",
        tag=str(config.output_tag),
        inspection_mode=str(config.inspection_mode),
        rules=",".join([str(r) for r in config.rules]),
        max_rows=int(config.max_rows),
        tail_direction=str(config.tail_direction),
        embedding_backend=str(config.embedding_backend),
        embedding_model=str(config.embedding_model),
        embedding_batch_size=int(config.embedding_batch_size),
        diff_residual_aux_enabled=bool(config.diff_residual_aux_enabled),
        diff_residual_aux_lambda=float(config.diff_residual_aux_lambda),
        diff_residual_aux_model=str(config.diff_residual_aux_model),
        embedding_cache_meta_json=str(config.embedding_cache_meta_json),
        rebuild_embedding_cache=bool(config.rebuild_embedding_cache),
        source_id_col=str(config.source_id_col),
        results_id_col=str(config.results_id_col),
        label_col=str(config.label_col),
        bad_label=str(config.bad_label),
        id_col=str(config.id_col),
        prompt_col=str(config.prompt_col),
        input_col=str(config.input_col),
        output_col=str(config.output_col),
        hist_bins=int(config.hist_bins),
        ratio_bins=int(config.ratio_bins),
    )


def parse_rules(raw: str) -> list[str]:
    out: list[str] = []
    for tok in str(raw).split(","):
        t = normalize_runtime_rule_key(tok)
        if not t:
            continue
        if t not in SIGNAL_COL:
            continue
        if t not in out:
            out.append(t)
    return out or list(DEFAULT_RULES)


def bool_series(s: pd.Series) -> np.ndarray:
    if s.dtype == bool:
        return s.fillna(False).to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).to_numpy(dtype=float) != 0.0
    mapped = (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False})
        .fillna(False)
    )
    return mapped.to_numpy(dtype=bool)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    c = str(hex_color).strip().lstrip("#")
    if len(c) != 6:
        return f"rgba(0,0,0,{alpha})"
    r = int(c[0:2], 16)
    g = int(c[2:4], 16)
    b = int(c[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _vector_trace(
    ix: np.ndarray,
    iy: np.ndarray,
    ox: np.ndarray,
    oy: np.ndarray,
    mask: np.ndarray,
    color: str,
    name: str,
    width: float,
    opacity: float,
) -> go.Scattergl | None:
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None
    x_vals: list[float | None] = []
    y_vals: list[float | None] = []
    for i in idx.tolist():
        x_vals.extend([float(ix[i]), float(ox[i]), None])
        y_vals.extend([float(iy[i]), float(oy[i]), None])
    return go.Scattergl(
        x=x_vals,
        y=y_vals,
        mode="lines",
        name=name,
        line=dict(color=color, width=width),
        opacity=opacity,
        hoverinfo="skip",
    )


def _knn_cluster_circles(
    x: np.ndarray,
    y: np.ndarray,
    *,
    k: int = 8,
    edge_quantile: float = 0.65,
    min_cluster_size: int = 20,
    radius_pad: float = 1.12,
) -> list[tuple[float, float, float, int]]:
    pts = np.column_stack([x, y]).astype(float, copy=False)
    finite = np.all(np.isfinite(pts), axis=1)
    pts = pts[finite]
    n = len(pts)
    if n < max(min_cluster_size, k + 1):
        return []

    diag = float(np.hypot(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])))
    min_radius = max(0.02 * diag, 0.02)

    # Many embedding runs collapse to repeated PCA coordinates. In that case,
    # KNN edge thresholds can become 0 and graph components disappear.
    def coord_fallback() -> list[tuple[float, float, float, int]]:
        rounded = np.round(pts, 6)
        _, inv, cnt = np.unique(rounded, axis=0, return_inverse=True, return_counts=True)
        out: list[tuple[float, float, float, int]] = []
        for gi, c in enumerate(cnt.tolist()):
            if c < int(min_cluster_size):
                continue
            g = pts[inv == gi]
            cx = float(np.mean(g[:, 0]))
            cy = float(np.mean(g[:, 1]))
            r = float(np.max(np.sqrt((g[:, 0] - cx) ** 2 + (g[:, 1] - cy) ** 2)))
            if not np.isfinite(r):
                continue
            out.append((cx, cy, max(r * float(radius_pad), min_radius), int(c)))
        return out

    dist, nbr = knn_self(pts, n_neighbors=min(k, n - 1), metric="euclidean")
    ok = np.isfinite(dist)
    if not np.any(ok):
        return coord_fallback()
    thr = float(np.quantile(dist[ok], float(np.clip(edge_quantile, 0.3, 0.95))))
    if (not np.isfinite(thr)) or thr <= 0.0:
        thr = float(np.nanmedian(dist[ok]))
    if (not np.isfinite(thr)) or thr <= 0.0:
        return coord_fallback()

    adj: list[set[int]] = [set() for _ in range(n)]
    for i in range(n):
        for j, d in zip(nbr[i].tolist(), dist[i].tolist(), strict=False):
            if not np.isfinite(d) or d > thr:
                continue
            jj = int(j)
            if jj == i:
                continue
            adj[i].add(jj)
            adj[jj].add(i)

    visited = np.zeros(n, dtype=bool)
    circles: list[tuple[float, float, float, int]] = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp: list[int] = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in adj[cur]:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        if len(comp) < int(min_cluster_size):
            continue
        cpts = pts[np.asarray(comp, dtype=int)]
        cx = float(np.mean(cpts[:, 0]))
        cy = float(np.mean(cpts[:, 1]))
        r = float(np.max(np.sqrt((cpts[:, 0] - cx) ** 2 + (cpts[:, 1] - cy) ** 2)))
        if not np.isfinite(r):
            continue
        circles.append((cx, cy, max(r * float(radius_pad), min_radius), int(len(comp))))
    if circles:
        return circles
    return coord_fallback()


def make_single_pca_geometry_fig(
    *,
    rule: str,
    ix: np.ndarray,
    iy: np.ndarray,
    ox: np.ndarray,
    oy: np.ndarray,
    base_mask: np.ndarray,
    y_bad: np.ndarray,
    label_known: np.ndarray,
) -> go.Figure:
    good_color = "#16a34a"
    bad_color = "#dc2626"
    unk_color = "#6b7280"

    base = np.asarray(base_mask, dtype=bool)
    known = base & np.asarray(label_known, dtype=bool)
    bad = known & np.asarray(y_bad, dtype=bool)
    good = known & (~np.asarray(y_bad, dtype=bool))
    unknown = base & (~np.asarray(label_known, dtype=bool))

    fig = go.Figure()

    # Vectors first (under points)
    for tr in [
        _vector_trace(ix, iy, ox, oy, good, good_color, "vector(good)", width=1.0, opacity=0.35),
        _vector_trace(ix, iy, ox, oy, bad, bad_color, "vector(bad)", width=1.2, opacity=0.45),
        _vector_trace(ix, iy, ox, oy, unknown, unk_color, "vector(unknown)", width=0.9, opacity=0.25),
    ]:
        if tr is not None:
            fig.add_trace(tr)

    # Input circles
    if np.any(good):
        fig.add_trace(
            go.Scattergl(
                x=ix[good],
                y=iy[good],
                mode="markers",
                name="input(good)",
                marker=dict(symbol="circle", color=good_color, size=6, opacity=0.75),
            )
        )
    if np.any(bad):
        fig.add_trace(
            go.Scattergl(
                x=ix[bad],
                y=iy[bad],
                mode="markers",
                name="input(bad)",
                marker=dict(symbol="circle", color=bad_color, size=6, opacity=0.80),
            )
        )
    if np.any(unknown):
        fig.add_trace(
            go.Scattergl(
                x=ix[unknown],
                y=iy[unknown],
                mode="markers",
                name="input(unknown)",
                marker=dict(symbol="circle", color=unk_color, size=5, opacity=0.45),
            )
        )

    # Output stars
    if np.any(good):
        fig.add_trace(
            go.Scattergl(
                x=ox[good],
                y=oy[good],
                mode="markers",
                name="output(good)",
                marker=dict(symbol="star", color=good_color, size=9, opacity=0.70),
            )
        )
    if np.any(bad):
        fig.add_trace(
            go.Scattergl(
                x=ox[bad],
                y=oy[bad],
                mode="markers",
                name="output(bad)",
                marker=dict(symbol="star", color=bad_color, size=9, opacity=0.78),
            )
        )
    if np.any(unknown):
        fig.add_trace(
            go.Scattergl(
                x=ox[unknown],
                y=oy[unknown],
                mode="markers",
                name="output(unknown)",
                marker=dict(symbol="star", color=unk_color, size=8, opacity=0.45),
            )
        )

    # KNN-like cluster grouping circles (per label class on output points).
    shapes: list[dict[str, Any]] = []
    for cls_mask, color in [(good, good_color), (bad, bad_color)]:
        if not np.any(cls_mask):
            continue
        circles = _knn_cluster_circles(ox[cls_mask], oy[cls_mask], k=8, edge_quantile=0.65, min_cluster_size=20)
        for cx, cy, r, _ in circles:
            shapes.append(
                {
                    "type": "circle",
                    "xref": "x",
                    "yref": "y",
                    "x0": cx - r,
                    "x1": cx + r,
                    "y0": cy - r,
                    "y1": cy + r,
                    "line": {"color": color, "width": 1.4, "dash": "dot"},
                    "fillcolor": _hex_to_rgba(color, 0.05),
                    "layer": "below",
                }
            )

    fig.update_layout(
        title=f"{RULE_LABEL.get(rule, rule)} | PCA Vector Geometry (single-view)",
        xaxis=dict(title="PCA 1"),
        yaxis=dict(title="PCA 2", scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h"),
        shapes=shapes,
        margin=dict(t=48, l=52, r=24, b=48),
    )
    return fig


def ensure_required_columns(row_df: pd.DataFrame, source_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = row_df.copy()

    if "row_id" not in out.columns:
        if args.id_col in source_df.columns and len(source_df) == len(out):
            out["row_id"] = source_df[args.id_col].astype(str)
        else:
            out["row_id"] = [f"row_{i}" for i in range(len(out))]

    if "source_input" not in out.columns and args.input_col in source_df.columns and len(source_df) == len(out):
        out["source_input"] = source_df[args.input_col].fillna("").astype(str)
    if "source_output" not in out.columns and args.output_col in source_df.columns and len(source_df) == len(out):
        out["source_output"] = source_df[args.output_col].fillna("").astype(str)

    if "hard_gate_pass" not in out.columns:
        out["hard_gate_pass"] = True

    # Minimum columns for diagnostics compatibility.
    for col in [
        "input_pca_x_nomask",
        "input_pca_y_nomask",
        "output_pca_x_nomask",
        "output_pca_y_nomask",
    ]:
        if col not in out.columns:
            out[col] = 0.0

    if "distribution_group_id" not in out.columns:
        out["distribution_group_id"] = "g0000"
    if "distribution_group_col" not in out.columns:
        out["distribution_group_col"] = "__all__"
    if "distribution_group_size" not in out.columns:
        out["distribution_group_size"] = len(out)

    return out


def resolve_input_distribution_signal(
    *,
    row_df: pd.DataFrame,
    source_df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, str]:
    """Resolve a numeric input-side signal for diagnostics.

    Priority:
    1) input PCA radius (if both PCA axes are available)
    2) source_input character length
    3) input_col character length from source CSV
    """
    ix_col = "input_pca_x_nomask"
    iy_col = "input_pca_y_nomask"
    if ix_col in row_df.columns and iy_col in row_df.columns:
        ix = pd.to_numeric(row_df[ix_col], errors="coerce").to_numpy(dtype=float)
        iy = pd.to_numeric(row_df[iy_col], errors="coerce").to_numpy(dtype=float)
        radius = np.sqrt(np.square(ix) + np.square(iy))
        if np.any(np.isfinite(radius)):
            return radius, "input_pca_radius_nomask"

    if "source_input" in row_df.columns:
        sig = row_df["source_input"].fillna("").astype(str).str.len().to_numpy(dtype=float)
        if np.any(np.isfinite(sig)):
            return sig, "source_input_char_length"

    input_col = str(getattr(args, "input_col", "")).strip()
    if input_col and (input_col in source_df.columns) and (len(source_df) == len(row_df)):
        sig = source_df[input_col].fillna("").astype(str).str.len().to_numpy(dtype=float)
        if np.any(np.isfinite(sig)):
            return sig, f"{input_col}_char_length"

    return None, ""


def compute_boxplot_summary(values: np.ndarray) -> dict[str, float]:
    """Compute compact boxplot statistics for fallback rendering."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {
            "min": float("nan"),
            "q1": float("nan"),
            "median": float("nan"),
            "q3": float("nan"),
            "max": float("nan"),
            "iqr": float("nan"),
            "whisker_low": float("nan"),
            "whisker_high": float("nan"),
            "outlier_n": 0.0,
        }
    q1 = float(np.quantile(v, 0.25))
    q2 = float(np.quantile(v, 0.50))
    q3 = float(np.quantile(v, 0.75))
    iqr = float(q3 - q1)
    lo = float(np.min(v))
    hi = float(np.max(v))
    whisker_low = max(lo, q1 - (1.5 * iqr))
    whisker_high = min(hi, q3 + (1.5 * iqr))
    outlier_n = float(np.sum((v < whisker_low) | (v > whisker_high)))
    return {
        "min": lo,
        "q1": q1,
        "median": q2,
        "q3": q3,
        "max": hi,
        "iqr": iqr,
        "whisker_low": float(whisker_low),
        "whisker_high": float(whisker_high),
        "outlier_n": float(outlier_n),
    }


def render_boxplot_mini_html(stats: dict[str, float]) -> str:
    """Render a tiny HTML/CSS boxplot-like track for fallback pages."""
    lo = float(stats.get("min", np.nan))
    hi = float(stats.get("max", np.nan))
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "<div class='boxplot-wrap'></div>"
    span = max(float(hi - lo), 1e-12)

    def pct(v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(((v - lo) / span) * 100.0, 0.0, 100.0))

    wl = pct(float(stats.get("whisker_low", lo)))
    wh = pct(float(stats.get("whisker_high", hi)))
    q1 = pct(float(stats.get("q1", lo)))
    q2 = pct(float(stats.get("median", lo)))
    q3 = pct(float(stats.get("q3", hi)))
    if q3 < q1:
        q1, q3 = q3, q1
    whisker_w = max(wh - wl, 0.0)
    box_w = max(q3 - q1, 0.8)
    return (
        "<div class='boxplot-wrap'>"
        f"<div class='boxplot-whisker' style='left:{wl:.3f}%;width:{whisker_w:.3f}%;'></div>"
        f"<div class='boxplot-box' style='left:{q1:.3f}%;width:{box_w:.3f}%;'></div>"
        f"<div class='boxplot-med' style='left:{q2:.3f}%;'></div>"
        "</div>"
    )


DETAIL_REQUIRED_COLUMNS = [
    "detail_evaluated_nomask",
    "detail_fail_any_leaf_nomask",
    "detail_fail_leaf_count_nomask",
    "detail_eval_leaf_count_nomask",
    "detail_failed_leaf_paths_nomask",
]


def validate_detail_columns_or_raise(row_df: pd.DataFrame) -> None:
    missing = [c for c in DETAIL_REQUIRED_COLUMNS if c not in row_df.columns]
    if missing:
        raise ValueError(
            "inspection_mode=detailed requires detail columns in row_results. "
            f"Missing: {missing}"
        )


def apply_detail_override_nomask(row_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = row_df.copy()
    required = [
        "hard_gate_pass",
        "distribution_pass_nomask",
        "distribution_state_nomask",
        "detail_evaluated_nomask",
        "detail_fail_any_leaf_nomask",
    ]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"detail override requires columns: {missing}")

    hard_gate = bool_series(out["hard_gate_pass"])
    detail_eval = bool_series(out["detail_evaluated_nomask"])
    detail_fail = bool_series(out["detail_fail_any_leaf_nomask"])
    detail_target = hard_gate & detail_eval & detail_fail

    pre_dist_pass = bool_series(out["distribution_pass_nomask"])
    pre_final_pass = bool_series(out["final_pass_nomask"]) if "final_pass_nomask" in out.columns else (hard_gate & pre_dist_pass)
    pre_state = out["distribution_state_nomask"].fillna("na").astype(str).str.strip().str.lower().to_numpy(dtype=object)

    dist_hard = bool_series(out["distribution_hard_fail_nomask"]) if "distribution_hard_fail_nomask" in out.columns else np.zeros(len(out), dtype=bool)
    dist_fail = bool_series(out["distribution_fail_nomask"]) if "distribution_fail_nomask" in out.columns else (~pre_dist_pass)
    dist_warn = bool_series(out["distribution_warn_nomask"]) if "distribution_warn_nomask" in out.columns else np.zeros(len(out), dtype=bool)

    dist_pass_new = pre_dist_pass & (~detail_target)
    dist_fail_new = dist_fail | (detail_target & (~dist_hard))
    dist_warn_new = dist_warn & (~detail_target)
    dist_hard_new = dist_hard

    dist_state_new = np.full(len(out), "na", dtype=object)
    dist_state_new[hard_gate] = "pass"
    dist_state_new[hard_gate & dist_warn_new] = "warn"
    dist_state_new[hard_gate & dist_fail_new] = "fail"
    dist_state_new[hard_gate & dist_hard_new] = "hard_fail"

    out["distribution_pass_nomask"] = dist_pass_new
    out["distribution_warn_nomask"] = dist_warn_new
    out["distribution_fail_nomask"] = dist_fail_new
    out["distribution_hard_fail_nomask"] = dist_hard_new
    out["distribution_state_nomask"] = dist_state_new

    final_pass_new = hard_gate & dist_pass_new
    final_state_new = np.full(len(out), "hard_gate_fail", dtype=object)
    final_state_new[hard_gate] = dist_state_new[hard_gate]
    out["final_pass_nomask"] = final_pass_new
    out["final_state_nomask"] = final_state_new

    override_applied = detail_target & np.isin(pre_state, ["pass", "warn"])
    out["detail_override_applied_nomask"] = override_applied.astype(bool)

    hard_eval = hard_gate & detail_eval
    stats = {
        "enabled": True,
        "detail_target_rows": int(np.sum(detail_target)),
        "detail_override_rows": int(np.sum(override_applied)),
        "detail_fail_rate_hard_eval": float(np.mean(detail_fail[hard_eval])) if np.any(hard_eval) else 0.0,
        "distribution_pass_rate_pre": float(np.mean(pre_dist_pass[hard_gate])) if np.any(hard_gate) else 0.0,
        "distribution_pass_rate_post": float(np.mean(dist_pass_new[hard_gate])) if np.any(hard_gate) else 0.0,
        "final_pass_rate_pre": float(np.mean(pre_final_pass[hard_gate])) if np.any(hard_gate) else 0.0,
        "final_pass_rate_post": float(np.mean(final_pass_new[hard_gate])) if np.any(hard_gate) else 0.0,
    }
    return out, stats


def bootstrap_row_results(args: argparse.Namespace, source_csv: Path, out_dir: Path) -> Path:
    """Run distribution pipeline API to obtain signal columns when row-results is absent."""
    tmp_dir = out_dir / "_bootstrap"
    artifacts = run_distribution_pipeline(
        {
            "csv_path": str(source_csv.resolve()),
            "id_col": str(args.id_col),
            "prompt_col": str(args.prompt_col),
            "input_col": str(args.input_col),
            "output_col": str(args.output_col),
            "output_dir": str(tmp_dir.resolve()),
            "report_dir_name": "report",
            "tag": f"{args.tag}_bootstrap",
            "inspection_mode": str(args.inspection_mode),
            "embedding_backend": str(args.embedding_backend),
            "embedding_model": str(args.embedding_model),
            "embedding_batch_size": int(args.embedding_batch_size),
            "diff_residual_aux_enabled": bool(args.diff_residual_aux_enabled),
            "diff_residual_aux_lambda": float(args.diff_residual_aux_lambda),
            "diff_residual_aux_model": str(args.diff_residual_aux_model),
            "max_rows": (int(args.max_rows) if args.max_rows and int(args.max_rows) > 0 else None),
        }
    )
    if not artifacts.row_results_csv.exists():
        raise FileNotFoundError(f"Bootstrap row_results not found: {artifacts.row_results_csv}")
    return artifacts.row_results_csv


def _find_peer_run_config(row_results_csv: Path) -> Path | None:
    stem = row_results_csv.stem
    if stem.endswith("_row_results"):
        cfg_name = f"{stem[:-len('_row_results')]}_run_config.json"
        candidate = row_results_csv.with_name(cfg_name)
        if candidate.exists():
            return candidate
    return None


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return raw if isinstance(raw, dict) else None


def _resolve_embedding_cache_meta_path(
    *,
    args: argparse.Namespace,
    row_csv: Path,
    out_dir: Path,
    tag: str,
    stem: str,
) -> tuple[Path, dict[str, Any]]:
    notes: dict[str, Any] = {}
    if str(args.embedding_cache_meta_json).strip():
        p = Path(str(args.embedding_cache_meta_json)).resolve()
        notes["selected_from"] = "cli"
        return p, notes

    peer_cfg = _find_peer_run_config(row_csv)
    if peer_cfg is not None:
        payload = _load_json(peer_cfg)
        if payload:
            emb = payload.get("embedding_cache", {})
            if isinstance(emb, dict):
                meta = str(emb.get("meta_json", "")).strip()
                if meta:
                    p = Path(meta).resolve()
                    notes["selected_from"] = "row_run_config"
                    notes["row_run_config"] = str(peer_cfg)
                    return p, notes
            # Nested run_config from pipeline in some artifacts.
            if isinstance(payload.get("run_config"), dict):
                emb2 = payload["run_config"].get("embedding_cache", {})
                if isinstance(emb2, dict):
                    meta = str(emb2.get("meta_json", "")).strip()
                    if meta:
                        p = Path(meta).resolve()
                        notes["selected_from"] = "row_run_config_nested"
                        notes["row_run_config"] = str(peer_cfg)
                        return p, notes

    default_paths = build_embedding_cache_paths(output_dir=out_dir, tag=tag, stem=stem)
    notes["selected_from"] = "default_output_dir"
    return default_paths.meta_json_path.resolve(), notes


def _resolve_dashboard_diagnostics_path(
    *,
    diagnostics_html_out: Path,
) -> tuple[Path | None, dict[str, Any]]:
    if diagnostics_html_out.exists():
        return diagnostics_html_out.resolve(), {"selected_from": "current_run_or_existing"}
    legacy = diagnostics_html_out.with_name(
        diagnostics_html_out.name.replace(
            "_distribution_diagnostics.html",
            "_nomask_hist_geometry_hybrid_derivative.html",
        )
    )
    if legacy.exists():
        return legacy.resolve(), {"selected_from": "legacy_distribution_diagnostics"}
    # run_final_metric always emits diagnostics HTML.
    return diagnostics_html_out.resolve(), {"selected_from": "current_run"}


def _resolve_peer_detailed_inspection_paths(row_csv: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    peer_cfg = _find_peer_run_config(row_csv)
    if peer_cfg is None:
        return out
    payload = _load_json(peer_cfg)
    if not payload:
        return out
    detailed = payload.get("detailed_inspection", {})
    if not isinstance(detailed, dict):
        return out
    for key in (
        "summary_csv",
        "row_hits_csv",
        "distribution_html",
        "distribution_stats_csv",
        "distribution_bins_csv",
        "gate_summary_csv",
        "gate_row_hits_csv",
    ):
        raw = str(detailed.get(key, "")).strip()
        if not raw:
            continue
        p = Path(raw).expanduser()
        if p.exists():
            out[key] = p.resolve()
    return out


def _path_to_href(path: Path) -> str:
    try:
        return path.resolve().as_uri()
    except Exception:
        return str(path)


def _leaf_label_from_hist_name(path: Path) -> str:
    name = path.name
    marker = "__leaf__"
    suffixes = (
        "_leaf_distribution_signal_hist.html",
        "_leaf_nomask_signal_hist.html",
    )
    for suffix in suffixes:
        if marker in name and name.endswith(suffix):
            return name.split(marker, 1)[1][: -len(suffix)]
    return path.stem


def _render_leaf_distribution_section_html(
    *,
    detail_paths: dict[str, Path],
    include_iframe: bool,
    max_leaf_files: int = 120,
) -> str:
    parts: list[str] = []
    parts.append("<h2>Leaf Distribution Diagnostics</h2>")

    if not detail_paths:
        parts.append(
            "<div class='meta'>"
            "detailed leaf artifacts not found from peer run_config. "
            "final diagnostics currently contains row-level distribution only."
            "</div>"
        )
        return "\n".join(parts)

    dist_html = detail_paths.get("distribution_html")
    dist_stats_csv = detail_paths.get("distribution_stats_csv")
    dist_bins_csv = detail_paths.get("distribution_bins_csv")

    report_dir: Path | None = None
    for p in (dist_html, dist_stats_csv, dist_bins_csv):
        if p is not None and p.exists():
            report_dir = p.parent
            break
    if report_dir is None:
        for p in detail_paths.values():
            if p.exists():
                report_dir = p.parent
                break

    leaf_hist_files: list[Path] = []
    if report_dir is not None and report_dir.exists():
        seen: dict[Path, None] = {}
        for pattern in ("*_leaf_distribution_signal_hist.html", "*_leaf_nomask_signal_hist.html"):
            for path in sorted(report_dir.glob(pattern)):
                seen[path] = None
        leaf_hist_files = list(seen.keys())

    parts.append(
        "<div class='meta'>"
        f"source_report_dir=<code>{html_lib.escape(str(report_dir) if report_dir else '')}</code>, "
        f"leaf_hist_files=<code>{len(leaf_hist_files)}</code>"
        "</div>"
    )

    parts.append("<h3>Global Leaf Distribution</h3>")
    if dist_html is not None and dist_html.exists():
        href = html_lib.escape(_path_to_href(dist_html))
        parts.append(
            "<div class='meta'>"
            f"<a href='{href}' target='_blank' rel='noopener'>open detail_leaf_distribution.html</a>"
            "</div>"
        )
        if include_iframe:
            parts.append(f"<iframe class='leaf-frame' src='{href}' loading='lazy'></iframe>")
    else:
        parts.append("<div class='meta'>distribution_html not found.</div>")

    if dist_stats_csv is not None and dist_stats_csv.exists():
        parts.append(
            "<div class='meta'>"
            f"distribution_stats_csv: <a href='{html_lib.escape(_path_to_href(dist_stats_csv))}' target='_blank' rel='noopener'>{html_lib.escape(dist_stats_csv.name)}</a>"
            "</div>"
        )
        try:
            stats_df = pd.read_csv(dist_stats_csv)
            cols = [
                c
                for c in (
                    "leaf_path",
                    "support",
                    "output_mean",
                    "output_std",
                    "output_q95",
                    "output_q99",
                    "output_max",
                )
                if c in stats_df.columns
            ]
            if not cols:
                cols = list(stats_df.columns[: min(8, len(stats_df.columns))])
            preview = stats_df[cols].copy().head(200)
            for c in preview.columns:
                if pd.api.types.is_numeric_dtype(preview[c]):
                    preview[c] = preview[c].map(lambda x: f"{float(x):.6g}" if pd.notna(x) else "")
                else:
                    preview[c] = preview[c].fillna("").astype(str)
            parts.append("<h3>Leaf Stats Preview</h3>")
            parts.append(preview.to_html(index=False, escape=True))
        except Exception as exc:
            parts.append(
                "<div class='meta'>"
                f"distribution_stats_csv preview failed: <code>{html_lib.escape(str(exc))}</code>"
                "</div>"
            )
    if dist_bins_csv is not None and dist_bins_csv.exists():
        parts.append(
            "<div class='meta'>"
            f"distribution_bins_csv: <a href='{html_lib.escape(_path_to_href(dist_bins_csv))}' target='_blank' rel='noopener'>{html_lib.escape(dist_bins_csv.name)}</a>"
            "</div>"
        )

    parts.append("<h3>Per-Leaf Signal Distribution</h3>")
    if not leaf_hist_files:
        parts.append("<div class='meta'>leaf signal histogram html files not found.</div>")
    else:
        parts.append("<div class='leaf-list'>")
        for path in leaf_hist_files[:max_leaf_files]:
            leaf_label = html_lib.escape(_leaf_label_from_hist_name(path))
            href = html_lib.escape(_path_to_href(path))
            fname = html_lib.escape(path.name)
            parts.append(
                "<details>"
                f"<summary><code>{leaf_label}</code> - <a href='{href}' target='_blank' rel='noopener'>{fname}</a></summary>"
            )
            if include_iframe:
                parts.append(f"<iframe class='leaf-frame' src='{href}' loading='lazy'></iframe>")
            parts.append("</details>")
        if len(leaf_hist_files) > max_leaf_files:
            parts.append(
                "<div class='meta'>"
                f"truncated: showing first {max_leaf_files} / {len(leaf_hist_files)} leaf histogram files."
                "</div>"
            )
        parts.append("</div>")

    return "\n".join(parts)


def build_detail_leaf_gate_tables_payload(
    *,
    gate_summary_csv: Path | None,
    gate_row_hits_csv: Path | None,
    max_rule_rows: int = 240,
    max_case_rows: int = 120,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "enabled": False,
        "leaf_rows": [],
        "rule_rows": [],
        "case_rows": [],
    }
    if gate_summary_csv is None or gate_row_hits_csv is None:
        payload["reason"] = "missing_artifacts"
        return payload
    if not gate_summary_csv.exists() or not gate_row_hits_csv.exists():
        payload["reason"] = "artifacts_not_found"
        return payload

    try:
        sum_df = pd.read_csv(gate_summary_csv)
        hit_df = pd.read_csv(gate_row_hits_csv)
    except Exception as exc:
        payload["reason"] = f"csv_read_error:{exc}"
        return payload

    if sum_df.empty or hit_df.empty:
        payload["reason"] = "empty_artifacts"
        return payload
    for c in ("group_id", "leaf_path"):
        if c not in hit_df.columns:
            payload["reason"] = f"missing_column:{c}"
            return payload

    rule_names = sorted(
        {
            str(c).replace("_pass_leaf_nomask", "")
            for c in hit_df.columns
            if str(c).endswith("_pass_leaf_nomask") and str(c) != "distribution_pass_leaf_nomask"
        }
    )
    if not rule_names:
        payload["reason"] = "no_rule_columns"
        return payload

    sum_map: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in sum_df.to_dict(orient="records"):
        gid = str(rec.get("group_id_raw", rec.get("group_id", "")))
        leaf = str(rec.get("leaf_path", ""))
        if gid and leaf:
            sum_map[(gid, leaf)] = rec

    leaf_rows: list[dict[str, Any]] = []
    rule_rows: list[dict[str, Any]] = []
    grouped = hit_df.groupby(["group_id", "leaf_path"], sort=True, dropna=False)
    for (gid_raw, leaf_path_raw), sub in grouped:
        gid = str(gid_raw)
        leaf_path = str(leaf_path_raw)
        support = int(len(sub))
        srow = sum_map.get((gid, leaf_path), {})

        dist_pass = (
            sub["distribution_pass_leaf_nomask"].fillna(True).astype(bool).to_numpy(dtype=bool)
            if "distribution_pass_leaf_nomask" in sub.columns
            else np.ones(support, dtype=bool)
        )
        failed_rule_count = (
            pd.to_numeric(sub.get("failed_rule_count_leaf_nomask", 0), errors="coerce")
            .fillna(0)
            .astype(int)
            .to_numpy(dtype=int)
        )
        fail_any_n = int((~dist_pass).sum())
        fail_any_rate = float(fail_any_n / support) if support > 0 else 0.0
        total_failed_rules = int(np.sum(failed_rule_count))

        rule_counter: Counter[str] = Counter()
        if "failed_rules_leaf_nomask" in sub.columns:
            for raw in sub["failed_rules_leaf_nomask"].fillna("").astype(str).tolist():
                for rule in [x.strip() for x in raw.split("|") if x.strip()]:
                    rule_counter[rule] += 1
        top_rules = ", ".join([f"{k}:{v}" for k, v in rule_counter.most_common(3)])

        leaf_rows.append(
            {
                "group_id": gid,
                "leaf_path": leaf_path,
                "leaf_support": support,
                "distribution_pass_rate": float(srow.get("distribution_pass_rate", np.nan)),
                "row_fail_any_n": fail_any_n,
                "row_fail_any_rate": fail_any_rate,
                "total_failed_rules": total_failed_rules,
                "top_failed_rules": top_rules,
            }
        )

        for rule in rule_names:
            pass_col = f"{rule}_pass_leaf_nomask"
            if pass_col not in sub.columns:
                continue
            avail_col = f"{rule}_available_leaf_nomask"
            pass_vals = sub[pass_col].fillna(True).astype(bool).to_numpy(dtype=bool)
            avail_vals = (
                sub[avail_col].fillna(False).astype(bool).to_numpy(dtype=bool)
                if avail_col in sub.columns
                else np.ones(support, dtype=bool)
            )
            available_n = int(np.sum(avail_vals))
            fail_n = int(np.sum((~pass_vals) & avail_vals))
            fail_rate = (float(fail_n) / float(available_n)) if available_n > 0 else np.nan
            available_rate = (float(available_n) / float(support)) if support > 0 else np.nan
            threshold = float(srow.get(f"{rule}_threshold", np.nan))
            status = "unavailable" if available_n == 0 else ("fail" if fail_n > 0 else "pass")
            rule_rows.append(
                {
                    "group_id": gid,
                    "leaf_path": leaf_path,
                    "rule": rule,
                    "status": status,
                    "threshold": threshold,
                    "leaf_support": support,
                    "available_n": available_n,
                    "available_rate": available_rate,
                    "fail_n": fail_n,
                    "fail_rate": fail_rate,
                }
            )

    leaf_rows = sorted(
        leaf_rows,
        key=lambda x: (
            -int(x.get("row_fail_any_n", 0)),
            -float(x.get("row_fail_any_rate", 0.0) or 0.0),
            str(x.get("group_id", "")),
            str(x.get("leaf_path", "")),
        ),
    )
    status_rank = {"fail": 0, "pass": 1, "unavailable": 2}
    rule_rows = sorted(
        rule_rows,
        key=lambda x: (
            status_rank.get(str(x.get("status", "")), 9),
            -int(x.get("fail_n", 0)),
            -float(x.get("fail_rate", 0.0) if np.isfinite(float(x.get("fail_rate", np.nan))) else -1.0),
            str(x.get("group_id", "")),
            str(x.get("leaf_path", "")),
            str(x.get("rule", "")),
        ),
    )

    case_rows: list[dict[str, Any]] = []
    if "failed_rule_count_leaf_nomask" in hit_df.columns:
        cases = hit_df.copy()
        cases["_failed_rule_count"] = pd.to_numeric(cases["failed_rule_count_leaf_nomask"], errors="coerce").fillna(0).astype(int)
        cases = cases[cases["_failed_rule_count"] > 0]
        if not cases.empty:
            cases = cases.sort_values(["_failed_rule_count", "row_index"], ascending=[False, True], kind="stable")
            for rec in cases.head(max(1, int(max_case_rows))).to_dict(orient="records"):
                case_rows.append(
                    {
                        "row_index": int(rec.get("row_index", -1)),
                        "row_id": str(rec.get("row_id", "")),
                        "group_id": str(rec.get("group_id", "")),
                        "leaf_path": str(rec.get("leaf_path", "")),
                        "failed_rule_count": int(rec.get("_failed_rule_count", 0)),
                        "failed_rules": str(rec.get("failed_rules_leaf_nomask", "")),
                        "distribution_pass_leaf_nomask": bool(rec.get("distribution_pass_leaf_nomask", True)),
                    }
                )

    payload.update(
        {
            "enabled": True,
            "source": {
                "gate_summary_csv": str(gate_summary_csv),
                "gate_row_hits_csv": str(gate_row_hits_csv),
            },
            "leaf_count": int(pd.DataFrame(leaf_rows).leaf_path.nunique()) if leaf_rows else 0,
            "row_hits": int(len(hit_df)),
            "rule_names": rule_names,
            "leaf_rows": leaf_rows[: max(1, int(max_rule_rows // 2))],
            "rule_rows": rule_rows[: max(1, int(max_rule_rows))],
            "case_rows": case_rows,
        }
    )
    return payload


def build_detail_leaf_triplet_payload(
    *,
    row_df: pd.DataFrame,
    threshold_summary_df: pd.DataFrame,
    score_runtime: Any,
    detail_leaf_gate_tables_payload: dict[str, Any],
    input_norm: np.ndarray | None,
    output_norm: np.ndarray | None,
    embedding_meta: dict[str, Any] | None,
    max_leaves: int = 8,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "enabled": False,
        "rows": [],
        "max_leaves": int(max(1, max_leaves)),
    }
    if not bool(detail_leaf_gate_tables_payload.get("enabled", False)):
        payload["reason"] = "leaf_gate_tables_disabled"
        return payload

    source_info = detail_leaf_gate_tables_payload.get("source", {})
    gate_hits_raw = str(source_info.get("gate_row_hits_csv", "")).strip() if isinstance(source_info, dict) else ""
    if not gate_hits_raw:
        payload["reason"] = "missing_gate_row_hits_csv"
        return payload
    gate_hits_path = Path(gate_hits_raw).expanduser()
    if not gate_hits_path.exists():
        payload["reason"] = "gate_row_hits_csv_not_found"
        payload["gate_row_hits_csv"] = str(gate_hits_path)
        return payload

    try:
        hit_df = pd.read_csv(gate_hits_path)
    except Exception as exc:
        payload["reason"] = f"csv_read_error:{exc}"
        payload["gate_row_hits_csv"] = str(gate_hits_path)
        return payload
    if hit_df.empty:
        payload["reason"] = "empty_gate_hits"
        payload["gate_row_hits_csv"] = str(gate_hits_path)
        return payload
    for c in ("group_id", "leaf_path", "row_index"):
        if c not in hit_df.columns:
            payload["reason"] = f"missing_column:{c}"
            payload["gate_row_hits_csv"] = str(gate_hits_path)
            return payload

    hit_df = hit_df.copy()
    hit_df["group_id"] = hit_df["group_id"].fillna("").astype(str)
    hit_df["leaf_path"] = hit_df["leaf_path"].fillna("").astype(str)
    hit_df["row_index"] = pd.to_numeric(hit_df["row_index"], errors="coerce").fillna(-1).astype(int)
    hit_df = hit_df[(hit_df["row_index"] >= 0) & (hit_df["row_index"] < len(row_df))].copy()
    if hit_df.empty:
        payload["reason"] = "no_valid_row_indices"
        payload["gate_row_hits_csv"] = str(gate_hits_path)
        return payload

    leaf_rows = detail_leaf_gate_tables_payload.get("leaf_rows", [])
    leaf_keys: list[tuple[str, str]] = []
    if isinstance(leaf_rows, list):
        for rec in leaf_rows:
            if not isinstance(rec, dict):
                continue
            gid = str(rec.get("group_id", "")).strip()
            leaf = str(rec.get("leaf_path", "")).strip()
            if gid and leaf:
                leaf_keys.append((gid, leaf))
    if not leaf_keys:
        leaf_keys = [
            (str(gid), str(leaf))
            for gid, leaf in hit_df.groupby(["group_id", "leaf_path"], sort=True, dropna=False).groups.keys()
            if str(gid).strip() and str(leaf).strip()
        ]
    if not leaf_keys:
        payload["reason"] = "no_leaf_keys"
        payload["gate_row_hits_csv"] = str(gate_hits_path)
        return payload

    leaf_keys = list(dict.fromkeys(leaf_keys))[: max(1, int(max_leaves))]
    out_rows: list[dict[str, Any]] = []

    in_norm_arr = None if input_norm is None else np.asarray(input_norm, dtype=float)
    out_norm_arr = None if output_norm is None else np.asarray(output_norm, dtype=float)

    for gid, leaf in leaf_keys:
        sub = hit_df[(hit_df["group_id"] == gid) & (hit_df["leaf_path"] == leaf)].copy()
        if sub.empty:
            continue
        idx = np.unique(pd.to_numeric(sub["row_index"], errors="coerce").fillna(-1).astype(int).to_numpy(dtype=int))
        idx = idx[(idx >= 0) & (idx < len(row_df))]
        if len(idx) == 0:
            continue

        leaf_row_df = row_df.iloc[idx].copy().reset_index(drop=True)
        leaf_input_norm = None
        leaf_output_norm = None
        if in_norm_arr is not None and len(in_norm_arr) == len(row_df):
            leaf_input_norm = in_norm_arr[idx]
        if out_norm_arr is not None and len(out_norm_arr) == len(row_df):
            leaf_output_norm = out_norm_arr[idx]

        try:
            leaf_art = compute_bundle_scores(
                row_df=leaf_row_df,
                threshold_summary_df=threshold_summary_df,
                score_runtime=score_runtime,
                input_norm=leaf_input_norm,
                output_norm=leaf_output_norm,
                embedding_meta=embedding_meta,
            )
        except Exception as exc:
            out_rows.append(
                {
                    "group_id": gid,
                    "leaf_path": leaf,
                    "n_rows": int(len(idx)),
                    "error": str(exc),
                    "summary": [],
                    "detail": [],
                }
            )
            continue

        row_fail_any_rate = float(
            np.mean(~sub["distribution_pass_leaf_nomask"].fillna(True).astype(bool).to_numpy(dtype=bool))
        ) if "distribution_pass_leaf_nomask" in sub.columns else float("nan")
        row_fail_any_n = int(
            np.sum(~sub["distribution_pass_leaf_nomask"].fillna(True).astype(bool).to_numpy(dtype=bool))
        ) if "distribution_pass_leaf_nomask" in sub.columns else 0

        out_rows.append(
            {
                "group_id": gid,
                "leaf_path": leaf,
                "n_rows": int(len(idx)),
                "row_hits": int(len(sub)),
                "row_fail_any_n": row_fail_any_n,
                "row_fail_any_rate": row_fail_any_rate,
                "diag_cause": str(leaf_art.payload.get("raw", {}).get("diag_cause", "")),
                "summary": leaf_art.summary_df.to_dict(orient="records"),
                "detail": leaf_art.detail_df.to_dict(orient="records"),
            }
        )

    if not out_rows:
        payload["reason"] = "leaf_bundle_compute_empty"
        payload["gate_row_hits_csv"] = str(gate_hits_path)
        return payload

    payload.update(
        {
            "enabled": True,
            "reason": "",
            "gate_row_hits_csv": str(gate_hits_path),
            "leaf_count": int(len(out_rows)),
            "rows": out_rows,
        }
    )
    return payload


@dataclass
class RuleThresholdResult:
    rule: str
    selected_method: str
    threshold_applied: float
    threshold_low: float
    threshold_high: float
    threshold_source: str
    support_rows: int
    label_support_rows: int
    precision: float
    recall: float
    f1: float
    fpr: float
    pred_bad_rate: float
    warn_threshold: float
    warn_threshold_low: float
    warn_threshold_high: float
    core_threshold: float
    core_threshold_low: float
    core_threshold_high: float
    fail_threshold: float
    fail_threshold_low: float
    fail_threshold_high: float
    tail_start_threshold: float
    tail_start_threshold_low: float
    tail_start_threshold_high: float
    hard_fail_threshold: float
    hard_fail_threshold_low: float
    hard_fail_threshold_high: float
    exceptional_out_threshold: float
    exceptional_out_threshold_low: float
    exceptional_out_threshold_high: float
    core_source: str
    exceptional_out_source: str
    hard_n: int
    hard_bad_precision: float
    fail_n: int
    fail_bad_precision: float
    warn_n: int
    warn_good_ratio: float


def is_strict_abnormal(meta: dict[str, Any], support_rows: int, min_support_rows: int) -> bool:
    thr = float(meta.get("threshold_applied", float("nan")))
    if not np.isfinite(thr):
        return True
    if support_rows < int(min_support_rows):
        return True
    source = str(meta.get("source", "") or "").strip().lower()
    if "tailstart_missing" in source:
        return True
    if source in {"", "missing"}:
        return True
    return False


def run_policy(
    *,
    rule: str,
    signal: np.ndarray,
    base: np.ndarray,
    policy: str,
    y_bad: np.ndarray,
    label_known: np.ndarray,
    tail_direction: str,
    features: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, Any]]:
    cfg = FINAL_THRESHOLD_RUNTIME
    fallback_q = float(cfg.fallback_quantile)
    return choose_rule_threshold_and_fail(
        rule=rule,
        score=signal,
        base=base,
        policy=policy,
        y_bad=y_bad,
        label_known=label_known,
        threshold_points=int(cfg.threshold_points),
        robust_z_k=float(cfg.robust_z_k),
        tail_start_floor_k=float(cfg.tail_start_floor_k),
        tail_start_max_k=float(cfg.tail_start_max_k),
        tail_start_min_survival=float(cfg.tail_start_min_survival),
        tail_start_max_survival=float(cfg.tail_start_max_survival),
        tail_start_grid_points=int(cfg.tail_start_grid_points),
        tail_direction=str(tail_direction),
        mad_eps=float(cfg.mad_eps),
        fallback_quantile=fallback_q,
        quantile_tail_q=float(1.0 - np.clip(fallback_q, 1e-6, 0.999999)),
        dist_q_min=int(cfg.dist_q_min),
        dist_q_max=int(cfg.dist_q_max),
        dist_q_step=int(cfg.dist_q_step),
        features=features,
    )


def apply_hybrid_thresholds_nomask(
    source_df: pd.DataFrame,
    row_df: pd.DataFrame,
    rules: list[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = FINAL_THRESHOLD_RUNTIME
    tail_direction = str(args.tail_direction)
    # Operational policy: tail-based thresholds are right-tail focused.
    # Even if runtime arg is two_sided, threshold derivation/evaluation uses upper.
    tail_direction_eff = "upper" if tail_direction == "two_sided" else tail_direction
    out = row_df.copy()

    y_bad, label_known, has_labels = compute_labels_bad(
        source_df=source_df,
        row_df=out,
        source_id_col=args.source_id_col,
        results_id_col=args.results_id_col,
        label_col=args.label_col,
        bad_label=args.bad_label,
    )

    hard_gate = bool_series(out["hard_gate_pass"]) if "hard_gate_pass" in out.columns else np.ones(len(out), dtype=bool)
    base_new_cols: dict[str, pd.Series] = {}
    if "distribution_pass_nomask" not in out.columns:
        base_new_cols["distribution_pass_nomask"] = pd.Series([False] * len(out), index=out.index, dtype=bool)
    if "distribution_warn_nomask" not in out.columns:
        base_new_cols["distribution_warn_nomask"] = pd.Series([False] * len(out), index=out.index, dtype=bool)
    if "distribution_fail_nomask" not in out.columns:
        base_new_cols["distribution_fail_nomask"] = pd.Series([False] * len(out), index=out.index, dtype=bool)
    if "distribution_hard_fail_nomask" not in out.columns:
        base_new_cols["distribution_hard_fail_nomask"] = pd.Series([False] * len(out), index=out.index, dtype=bool)
    if "distribution_state_nomask" not in out.columns:
        base_new_cols["distribution_state_nomask"] = pd.Series(["na"] * len(out), index=out.index, dtype=object)
    if "final_state_nomask" not in out.columns:
        base_new_cols["final_state_nomask"] = pd.Series(["hard_gate_fail"] * len(out), index=out.index, dtype=object)
    if base_new_cols:
        out = pd.concat([out, pd.DataFrame(base_new_cols)], axis=1)
    # Diagnostics should cover all inspected rows; plotting later still applies
    # per-signal availability/finite checks.
    out["distribution_evaluated_nomask"] = np.ones(len(out), dtype=bool)

    features = compute_policy_features(row_df=out)

    active_rules: list[str] = []
    rows: list[RuleThresholdResult] = []

    # Pre-create per-rule output columns once to avoid fragmented frame inserts.
    new_cols: dict[str, pd.Series] = {}
    for rule in rules:
        pass_col = f"{rule}_pass_nomask"
        warn_col = f"{rule}_warn_nomask"
        fail_col = f"{rule}_fail_nomask"
        hard_col = f"{rule}_hard_fail_nomask"
        state_col = f"{rule}_state_nomask"
        method_col = f"{rule}_threshold_method_nomask"
        source_col = f"{rule}_threshold_source_nomask"
        thr_col = f"{rule}_threshold_nomask"
        thr_lo_col = f"{rule}_threshold_low_nomask"
        thr_hi_col = f"{rule}_threshold_high_nomask"
        warn_thr_col = f"{rule}_warn_threshold_nomask"
        warn_thr_lo_col = f"{rule}_warn_threshold_low_nomask"
        warn_thr_hi_col = f"{rule}_warn_threshold_high_nomask"
        core_thr_col = f"{rule}_core_threshold_nomask"
        core_thr_lo_col = f"{rule}_core_threshold_low_nomask"
        core_thr_hi_col = f"{rule}_core_threshold_high_nomask"
        fail_thr_col = f"{rule}_fail_threshold_nomask"
        fail_thr_lo_col = f"{rule}_fail_threshold_low_nomask"
        fail_thr_hi_col = f"{rule}_fail_threshold_high_nomask"
        tail_thr_col = f"{rule}_tail_start_threshold_nomask"
        tail_thr_lo_col = f"{rule}_tail_start_threshold_low_nomask"
        tail_thr_hi_col = f"{rule}_tail_start_threshold_high_nomask"
        hard_thr_col = f"{rule}_hard_fail_threshold_nomask"
        hard_thr_lo_col = f"{rule}_hard_fail_threshold_low_nomask"
        hard_thr_hi_col = f"{rule}_hard_fail_threshold_high_nomask"
        exc_thr_col = f"{rule}_exceptional_out_threshold_nomask"
        exc_thr_lo_col = f"{rule}_exceptional_out_threshold_low_nomask"
        exc_thr_hi_col = f"{rule}_exceptional_out_threshold_high_nomask"

        if pass_col not in out.columns:
            new_cols[pass_col] = pd.Series([True] * len(out), index=out.index, dtype=bool)
        if warn_col not in out.columns:
            new_cols[warn_col] = pd.Series([False] * len(out), index=out.index, dtype=bool)
        if fail_col not in out.columns:
            new_cols[fail_col] = pd.Series([False] * len(out), index=out.index, dtype=bool)
        if hard_col not in out.columns:
            new_cols[hard_col] = pd.Series([False] * len(out), index=out.index, dtype=bool)
        if state_col not in out.columns:
            new_cols[state_col] = pd.Series(["na"] * len(out), index=out.index, dtype=object)
        if method_col not in out.columns:
            new_cols[method_col] = pd.Series(["missing"] * len(out), index=out.index, dtype=object)
        if source_col not in out.columns:
            new_cols[source_col] = pd.Series(["missing"] * len(out), index=out.index, dtype=object)

        for c in [
            thr_col,
            thr_lo_col,
            thr_hi_col,
            warn_thr_col,
            warn_thr_lo_col,
            warn_thr_hi_col,
            core_thr_col,
            core_thr_lo_col,
            core_thr_hi_col,
            fail_thr_col,
            fail_thr_lo_col,
            fail_thr_hi_col,
            tail_thr_col,
            tail_thr_lo_col,
            tail_thr_hi_col,
            hard_thr_col,
            hard_thr_lo_col,
            hard_thr_hi_col,
            exc_thr_col,
            exc_thr_lo_col,
            exc_thr_hi_col,
        ]:
            if c not in out.columns:
                new_cols[c] = pd.Series([np.nan] * len(out), index=out.index, dtype=float)

    if new_cols:
        out = pd.concat([out, pd.DataFrame(new_cols)], axis=1)

    for rule in rules:
        signal_col = resolve_signal_col(rule, out)
        if not signal_col or signal_col not in out.columns:
            continue

        signal = pd.to_numeric(out[signal_col], errors="coerce").to_numpy(dtype=float)
        avail_col = AVAILABLE_COL.get(rule, "")
        if avail_col and avail_col in out.columns:
            available = bool_series(out[avail_col])
        else:
            available = np.ones(len(out), dtype=bool)

        base = hard_gate & available & np.isfinite(signal)
        support_rows = int(np.sum(base))
        label_support_rows = int(np.sum(base & label_known))

        pass_col = f"{rule}_pass_nomask"
        warn_col = f"{rule}_warn_nomask"
        fail_col = f"{rule}_fail_nomask"
        hard_col = f"{rule}_hard_fail_nomask"
        state_col = f"{rule}_state_nomask"
        method_col = f"{rule}_threshold_method_nomask"
        source_col = f"{rule}_threshold_source_nomask"
        thr_col = f"{rule}_threshold_nomask"
        thr_lo_col = f"{rule}_threshold_low_nomask"
        thr_hi_col = f"{rule}_threshold_high_nomask"
        warn_thr_col = f"{rule}_warn_threshold_nomask"
        warn_thr_lo_col = f"{rule}_warn_threshold_low_nomask"
        warn_thr_hi_col = f"{rule}_warn_threshold_high_nomask"
        core_thr_col = f"{rule}_core_threshold_nomask"
        core_thr_lo_col = f"{rule}_core_threshold_low_nomask"
        core_thr_hi_col = f"{rule}_core_threshold_high_nomask"
        fail_thr_col = f"{rule}_fail_threshold_nomask"
        fail_thr_lo_col = f"{rule}_fail_threshold_low_nomask"
        fail_thr_hi_col = f"{rule}_fail_threshold_high_nomask"
        tail_thr_col = f"{rule}_tail_start_threshold_nomask"
        tail_thr_lo_col = f"{rule}_tail_start_threshold_low_nomask"
        tail_thr_hi_col = f"{rule}_tail_start_threshold_high_nomask"
        hard_thr_col = f"{rule}_hard_fail_threshold_nomask"
        hard_thr_lo_col = f"{rule}_hard_fail_threshold_low_nomask"
        hard_thr_hi_col = f"{rule}_hard_fail_threshold_high_nomask"
        exc_thr_col = f"{rule}_exceptional_out_threshold_nomask"
        exc_thr_lo_col = f"{rule}_exceptional_out_threshold_low_nomask"
        exc_thr_hi_col = f"{rule}_exceptional_out_threshold_high_nomask"

        if support_rows < int(cfg.min_support_rows):
            out[pass_col] = True
            out[warn_col] = False
            out[fail_col] = False
            out[hard_col] = False
            out[state_col] = "na"
            out[method_col] = "missing"
            out[source_col] = "insufficient_support"
            out[thr_col] = np.nan
            out[thr_lo_col] = np.nan
            out[thr_hi_col] = np.nan
            out[warn_thr_col] = np.nan
            out[warn_thr_lo_col] = np.nan
            out[warn_thr_hi_col] = np.nan
            out[core_thr_col] = np.nan
            out[core_thr_lo_col] = np.nan
            out[core_thr_hi_col] = np.nan
            out[fail_thr_col] = np.nan
            out[fail_thr_lo_col] = np.nan
            out[fail_thr_hi_col] = np.nan
            out[tail_thr_col] = np.nan
            out[tail_thr_lo_col] = np.nan
            out[tail_thr_hi_col] = np.nan
            out[hard_thr_col] = np.nan
            out[hard_thr_lo_col] = np.nan
            out[hard_thr_hi_col] = np.nan
            out[exc_thr_col] = np.nan
            out[exc_thr_lo_col] = np.nan
            out[exc_thr_hi_col] = np.nan
            continue

        fail_1, meta_1 = run_policy(
            rule=rule,
            signal=signal,
            base=base,
            policy="robust_z_tail_start",
            y_bad=y_bad,
            label_known=label_known,
            tail_direction=tail_direction_eff,
            features=features,
        )
        abnormal_1 = is_strict_abnormal(meta_1, support_rows=support_rows, min_support_rows=cfg.min_support_rows)

        if not abnormal_1:
            fail = fail_1
            meta = meta_1
            selected_method = "tail_start"
        else:
            fail_2, meta_2 = run_policy(
                rule=rule,
                signal=signal,
                base=base,
                policy="dist_stability_jump",
                y_bad=y_bad,
                label_known=label_known,
                tail_direction=tail_direction_eff,
                features=features,
            )
            abnormal_2 = is_strict_abnormal(meta_2, support_rows=support_rows, min_support_rows=cfg.min_support_rows)
            if not abnormal_2:
                fail = fail_2
                meta = meta_2
                selected_method = "dist_stability_jump"
            else:
                fail_3, meta_3 = run_policy(
                    rule=rule,
                    signal=signal,
                    base=base,
                    policy="quantile_tail",
                    y_bad=y_bad,
                    label_known=label_known,
                    tail_direction=tail_direction_eff,
                    features=features,
                )
                fail = fail_3
                meta = meta_3
                selected_method = "quantile_tail_guard"

        fail_t = float(meta.get("threshold_applied", np.nan))
        fail_lo = float(meta.get("threshold_low", np.nan))
        fail_hi = float(meta.get("threshold_high", np.nan))
        tri = derive_tristate_thresholds_from_fail(
            values=signal[base],
            tail_direction=tail_direction_eff,
            fail_threshold=fail_t,
            fail_low=fail_lo,
            fail_high=fail_hi,
            warn_ratio=float(cfg.warn_ratio_from_fail),
            hard_ratio=float(cfg.hard_ratio_from_fail),
            core_kappa=float(cfg.core_kappa),
            core_quantile=float(cfg.core_fallback_quantile),
            core_min_count=int(cfg.core_min_count),
            mad_eps=float(cfg.mad_eps),
            exceptional_d1_lambda=float(cfg.exceptional_d1_lambda),
            exceptional_d2_lambda=float(cfg.exceptional_d2_lambda),
            exceptional_consecutive=int(cfg.exceptional_consecutive),
            exceptional_grid_points=int(cfg.exceptional_grid_points),
            exceptional_min_tail_points=int(cfg.exceptional_min_tail_points),
            exceptional_fallback_quantile=float(cfg.exceptional_fallback_quantile),
            exceptional_min_delta_ratio=float(cfg.exceptional_min_delta_ratio),
        )
        core_t = float(tri.get("core_threshold", tri.get("warn_threshold", np.nan)))
        core_lo = float(tri.get("core_threshold_low", tri.get("warn_threshold_low", np.nan)))
        core_hi = float(tri.get("core_threshold_high", tri.get("warn_threshold_high", np.nan)))
        tail_t = float(tri.get("tail_start_threshold", fail_t))
        tail_lo = float(tri.get("tail_start_threshold_low", fail_lo))
        tail_hi = float(tri.get("tail_start_threshold_high", fail_hi))
        exc_t = float(tri.get("exceptional_out_threshold", tri.get("hard_fail_threshold", np.nan)))
        exc_lo = float(tri.get("exceptional_out_threshold_low", tri.get("hard_fail_threshold_low", np.nan)))
        exc_hi = float(tri.get("exceptional_out_threshold_high", tri.get("hard_fail_threshold_high", np.nan)))
        core_src = str(tri.get("core_source", ""))
        exc_src = str(tri.get("exceptional_out_source", ""))

        fail_mask = np.zeros(len(out), dtype=bool)
        fail_mask[base] = np.asarray(fail[base], dtype=bool)
        warn_tail_direction = (
            "two_sided"
            if np.isfinite(core_lo) and np.isfinite(core_hi) and (core_hi > core_lo)
            else tail_direction_eff
        )
        warn_mask = trigger_mask(
            score=signal,
            base=base,
            tail_direction=warn_tail_direction,
            threshold=float(core_t),
            threshold_low=float(core_lo),
            threshold_high=float(core_hi),
        )
        hard_mask = trigger_mask(
            score=signal,
            base=base,
            tail_direction=tail_direction_eff,
            threshold=float(exc_t),
            threshold_low=float(exc_lo),
            threshold_high=float(exc_hi),
        )

        hard_only = hard_mask & fail_mask & base
        fail_only = fail_mask & base & (~hard_only)
        warn_only = warn_mask & base & (~fail_mask) & (~hard_only)

        passed = np.ones(len(out), dtype=bool)
        passed[base] = ~fail_mask[base]

        states = np.full(len(out), "na", dtype=object)
        states[base] = "pass"
        states[warn_only] = "warn"
        states[fail_only] = "fail"
        states[hard_only] = "hard_fail"

        out[pass_col] = passed
        out[warn_col] = warn_only
        out[fail_col] = fail_only
        out[hard_col] = hard_only
        out[state_col] = states
        out[method_col] = selected_method
        out[source_col] = str(meta.get("source", ""))
        out[thr_col] = fail_t
        out[thr_lo_col] = fail_lo
        out[thr_hi_col] = fail_hi
        out[warn_thr_col] = core_t
        out[warn_thr_lo_col] = core_lo
        out[warn_thr_hi_col] = core_hi
        out[core_thr_col] = core_t
        out[core_thr_lo_col] = core_lo
        out[core_thr_hi_col] = core_hi
        out[fail_thr_col] = fail_t
        out[fail_thr_lo_col] = fail_lo
        out[fail_thr_hi_col] = fail_hi
        out[tail_thr_col] = tail_t
        out[tail_thr_lo_col] = tail_lo
        out[tail_thr_hi_col] = tail_hi
        out[hard_thr_col] = exc_t
        out[hard_thr_lo_col] = exc_lo
        out[hard_thr_hi_col] = exc_hi
        out[exc_thr_col] = exc_t
        out[exc_thr_lo_col] = exc_lo
        out[exc_thr_hi_col] = exc_hi

        if has_labels and np.any(base & label_known):
            mm = binary_metrics(
                y_true_bad=y_bad[base & label_known],
                y_pred_bad=fail_mask[base & label_known],
                known_mask=np.ones(int(np.sum(base & label_known)), dtype=bool),
            )
        else:
            mm = {
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
                "fpr": float("nan"),
                "pred_bad_rate": float(np.mean(fail_mask[base])) if np.any(base) else float("nan"),
            }

        hard_n = int(np.sum(hard_only))
        fail_n = int(np.sum(fail_only))
        warn_n = int(np.sum(warn_only))
        if has_labels and np.any(base & label_known):
            hard_known = hard_only & label_known
            fail_known = fail_only & label_known
            warn_known = warn_only & label_known
            hard_bad_precision = (
                float(np.mean(y_bad[hard_known])) if np.any(hard_known) else float("nan")
            )
            fail_bad_precision = (
                float(np.mean(y_bad[fail_known])) if np.any(fail_known) else float("nan")
            )
            warn_good_ratio = (
                float(np.mean(~y_bad[warn_known])) if np.any(warn_known) else float("nan")
            )
        else:
            hard_bad_precision = float("nan")
            fail_bad_precision = float("nan")
            warn_good_ratio = float("nan")

        rows.append(
            RuleThresholdResult(
                rule=rule,
                selected_method=selected_method,
                threshold_applied=fail_t,
                threshold_low=fail_lo,
                threshold_high=fail_hi,
                threshold_source=str(meta.get("source", "")),
                support_rows=support_rows,
                label_support_rows=label_support_rows,
                precision=float(mm.get("precision", np.nan)),
                recall=float(mm.get("recall", np.nan)),
                f1=float(mm.get("f1", np.nan)),
                fpr=float(mm.get("fpr", np.nan)),
                pred_bad_rate=float(mm.get("pred_bad_rate", np.nan)),
                warn_threshold=core_t,
                warn_threshold_low=core_lo,
                warn_threshold_high=core_hi,
                core_threshold=core_t,
                core_threshold_low=core_lo,
                core_threshold_high=core_hi,
                fail_threshold=fail_t,
                fail_threshold_low=fail_lo,
                fail_threshold_high=fail_hi,
                tail_start_threshold=tail_t,
                tail_start_threshold_low=tail_lo,
                tail_start_threshold_high=tail_hi,
                hard_fail_threshold=exc_t,
                hard_fail_threshold_low=exc_lo,
                hard_fail_threshold_high=exc_hi,
                exceptional_out_threshold=exc_t,
                exceptional_out_threshold_low=exc_lo,
                exceptional_out_threshold_high=exc_hi,
                core_source=core_src,
                exceptional_out_source=exc_src,
                hard_n=hard_n,
                hard_bad_precision=hard_bad_precision,
                fail_n=fail_n,
                fail_bad_precision=fail_bad_precision,
                warn_n=warn_n,
                warn_good_ratio=warn_good_ratio,
            )
        )
        active_rules.append(rule)

    dist_pass = np.ones(len(out), dtype=bool)
    dist_warn = np.zeros(len(out), dtype=bool)
    dist_fail = np.zeros(len(out), dtype=bool)
    dist_hard = np.zeros(len(out), dtype=bool)
    for rule in active_rules:
        dist_pass &= bool_series(out[f"{rule}_pass_nomask"])
        dist_warn |= bool_series(out[f"{rule}_warn_nomask"])
        dist_fail |= bool_series(out[f"{rule}_fail_nomask"])
        dist_hard |= bool_series(out[f"{rule}_hard_fail_nomask"])

    out["distribution_pass_nomask"] = False
    out.loc[hard_gate, "distribution_pass_nomask"] = dist_pass[hard_gate]
    out["distribution_warn_nomask"] = False
    out["distribution_fail_nomask"] = False
    out["distribution_hard_fail_nomask"] = False
    out.loc[hard_gate, "distribution_warn_nomask"] = dist_warn[hard_gate]
    out.loc[hard_gate, "distribution_fail_nomask"] = dist_fail[hard_gate]
    out.loc[hard_gate, "distribution_hard_fail_nomask"] = dist_hard[hard_gate]

    dist_state = np.full(len(out), "na", dtype=object)
    dist_state[hard_gate] = "pass"
    dist_state[hard_gate & dist_warn] = "warn"
    dist_state[hard_gate & dist_fail] = "fail"
    dist_state[hard_gate & dist_hard] = "hard_fail"
    out["distribution_state_nomask"] = dist_state

    out["final_pass_nomask"] = bool_series(out["hard_gate_pass"]) & bool_series(out["distribution_pass_nomask"])
    final_state = np.full(len(out), "hard_gate_fail", dtype=object)
    final_state[hard_gate] = dist_state[hard_gate]
    out["final_state_nomask"] = final_state

    summary_rows = []
    for r in rows:
        summary_rows.append(
            {
                "rule": r.rule,
                "threshold_policy": "hybrid_tailstart_core_exceptional",
                "selected_method": r.selected_method,
                "threshold_applied": r.threshold_applied,
                "threshold_low": r.threshold_low,
                "threshold_high": r.threshold_high,
                "core_threshold": r.core_threshold,
                "core_threshold_low": r.core_threshold_low,
                "core_threshold_high": r.core_threshold_high,
                "warn_threshold": r.warn_threshold,
                "warn_threshold_low": r.warn_threshold_low,
                "warn_threshold_high": r.warn_threshold_high,
                "tail_start_threshold": r.tail_start_threshold,
                "tail_start_threshold_low": r.tail_start_threshold_low,
                "tail_start_threshold_high": r.tail_start_threshold_high,
                "fail_threshold": r.fail_threshold,
                "fail_threshold_low": r.fail_threshold_low,
                "fail_threshold_high": r.fail_threshold_high,
                "exceptional_out_threshold": r.exceptional_out_threshold,
                "exceptional_out_threshold_low": r.exceptional_out_threshold_low,
                "exceptional_out_threshold_high": r.exceptional_out_threshold_high,
                "hard_fail_threshold": r.hard_fail_threshold,
                "hard_fail_threshold_low": r.hard_fail_threshold_low,
                "hard_fail_threshold_high": r.hard_fail_threshold_high,
                "threshold_source": r.threshold_source,
                "core_source": r.core_source,
                "exceptional_out_source": r.exceptional_out_source,
                "precision": r.precision,
                "recall": r.recall,
                "f1": r.f1,
                "fpr": r.fpr,
                "pred_bad_rate": r.pred_bad_rate,
                "hard_n": r.hard_n,
                "hard_bad_precision": r.hard_bad_precision,
                "fail_n": r.fail_n,
                "fail_bad_precision": r.fail_bad_precision,
                "warn_n": r.warn_n,
                "warn_good_ratio": r.warn_good_ratio,
                "support_rows": r.support_rows,
                "label_support_rows": r.label_support_rows,
            }
        )

    return out, pd.DataFrame(summary_rows)


def render_distribution_diagnostics_plot(
    source_df: pd.DataFrame,
    row_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    rules: list[str],
    args: argparse.Namespace,
    out_html: Path,
    out_summary_csv: Path,
    detail_paths: dict[str, Path] | None = None,
) -> None:
    y_bad, label_known, has_labels = compute_labels_bad(
        source_df=source_df,
        row_df=row_df,
        source_id_col=args.source_id_col,
        results_id_col=args.results_id_col,
        label_col=args.label_col,
        bad_label=args.bad_label,
    )

    # Diagnostics scope: use all rows that were actually evaluated by distribution.
    # Fallback to all rows when the coverage column is unavailable.
    diag_eval_mask = (
        bool_series(row_df["distribution_evaluated_nomask"])
        if "distribution_evaluated_nomask" in row_df.columns
        else np.ones(len(row_df), dtype=bool)
    )

    plot_tail_direction = "upper" if str(args.tail_direction) == "two_sided" else str(args.tail_direction)

    html_parts: list[str] = []
    html_parts.append(
        "<html><head><meta charset='utf-8'><title>Final Metric Distribution Diagnostics</title>"
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:24px;}"
        "h1{margin-bottom:8px;}h2{margin-top:34px;margin-bottom:8px;}h3{margin-top:16px;margin-bottom:8px;}"
        ".meta{color:#374151;font-size:13px;margin-bottom:16px;}.chart{margin-bottom:18px;}"
        ".chart-half{width:min(900px,48vw);} .chart-full{width:min(1600px,95vw);}"
        "@media (max-width:1200px){.chart-half,.chart-full{width:95vw;}}"
        ".leaf-list details{margin:8px 0;}.leaf-list summary{cursor:pointer;}"
        ".leaf-frame{width:min(1600px,95vw);height:460px;border:1px solid #e5e7eb;border-radius:8px;margin-top:8px;}"
        "table{border-collapse:collapse;width:min(1300px,95vw);font-size:12px;margin-bottom:14px;}"
        "th,td{border:1px solid #e5e7eb;padding:6px 8px;text-align:right;}"
        "th:first-child,td:first-child{text-align:left;}th{background:#f8fafc;}"
        "code{background:#f3f4f6;padding:2px 6px;border-radius:6px;}</style></head><body>"
    )
    html_parts.append("<h1>Final Metric Distribution Diagnostics</h1>")
    html_parts.append(
        "<div class='meta'>"
        f"row_results=<code>{args.row_results_csv if args.row_results_csv else '(bootstrapped)'}</code><br>"
        f"rules={', '.join(rules)}<br>"
        "threshold_policy=<code>hybrid_tailstart_core_exceptional</code> "
        "(robust_z_tail_start -> dist_stability_jump -> quantile_tail_guard), "
        f"tail_direction(config)=<code>{args.tail_direction}</code>, effective=<code>{plot_tail_direction}</code><br>"
        f"labels_available=<code>{str(has_labels).lower()}</code><br>"
        "diagnostic_scope=<code>distribution_evaluated_nomask</code> (all inspected rows)<br>"
        "green=good, red=bad, orange=core-threshold(warn), blue=tail-start(fail), red-line=exceptional-out(hard)"
        "</div>"
    )

    include_js: str | bool = "cdn"
    plot_summary_rows: list[dict[str, Any]] = []

    sidx = {str(r["rule"]): i for i, r in summary_df.iterrows()} if not summary_df.empty else {}

    for rule in rules:
        signal_col = resolve_signal_col(rule, row_df)
        if not signal_col or signal_col not in row_df.columns:
            continue

        signal = pd.to_numeric(row_df[signal_col], errors="coerce").to_numpy(dtype=float)
        avail_col = AVAILABLE_COL.get(rule, "")
        available = bool_series(row_df[avail_col]) if (avail_col and avail_col in row_df.columns) else np.ones(len(row_df), dtype=bool)
        finite_signal = np.isfinite(signal)
        base_primary = diag_eval_mask & available & finite_signal
        base_relaxed = diag_eval_mask & finite_signal
        use_relaxed_base = False
        if int(np.sum(base_primary)) == 0 and int(np.sum(base_relaxed)) > 0:
            base = base_relaxed
            use_relaxed_base = True
        else:
            base = base_primary
        if int(np.sum(base)) == 0:
            continue

        pass_col = f"{rule}_pass_nomask"
        pred_pass = bool_series(row_df[pass_col]) if pass_col in row_df.columns else np.ones(len(row_df), dtype=bool)
        fail = ~pred_pass
        warn_only = (
            bool_series(row_df[f"{rule}_warn_nomask"])
            if f"{rule}_warn_nomask" in row_df.columns
            else np.zeros(len(row_df), dtype=bool)
        )
        fail_only = (
            bool_series(row_df[f"{rule}_fail_nomask"])
            if f"{rule}_fail_nomask" in row_df.columns
            else fail.copy()
        )
        hard_only = (
            bool_series(row_df[f"{rule}_hard_fail_nomask"])
            if f"{rule}_hard_fail_nomask" in row_df.columns
            else np.zeros(len(row_df), dtype=bool)
        )

        ridx = sidx.get(rule)
        if ridx is not None:
            row = summary_df.iloc[int(ridx)]
            thr = float(row.get("threshold_applied", np.nan))
            core_thr = float(row.get("core_threshold", row.get("warn_threshold", np.nan)))
            core_thr_lo = float(row.get("core_threshold_low", row.get("warn_threshold_low", np.nan)))
            core_thr_hi = float(row.get("core_threshold_high", row.get("warn_threshold_high", np.nan)))
            tail_thr = float(row.get("tail_start_threshold", row.get("fail_threshold", thr)))
            tail_thr_lo = float(row.get("tail_start_threshold_low", row.get("fail_threshold_low", np.nan)))
            tail_thr_hi = float(row.get("tail_start_threshold_high", row.get("fail_threshold_high", np.nan)))
            exc_thr = float(row.get("exceptional_out_threshold", row.get("hard_fail_threshold", np.nan)))
            exc_thr_lo = float(row.get("exceptional_out_threshold_low", row.get("hard_fail_threshold_low", np.nan)))
            exc_thr_hi = float(row.get("exceptional_out_threshold_high", row.get("hard_fail_threshold_high", np.nan)))
            selected_method = str(row.get("selected_method", "unknown"))
            source = str(row.get("threshold_source", ""))
        else:
            thr = float("nan")
            core_thr = float("nan")
            core_thr_lo = float("nan")
            core_thr_hi = float("nan")
            tail_thr = float(thr)
            tail_thr_lo = float("nan")
            tail_thr_hi = float("nan")
            exc_thr = float("nan")
            exc_thr_lo = float("nan")
            exc_thr_hi = float("nan")
            selected_method = "unknown"
            source = ""

        plot_td_rule = plot_tail_direction
        if np.isfinite(core_thr_lo) and np.isfinite(core_thr_hi) and (core_thr_hi > core_thr_lo):
            plot_td_rule = "two_sided"

        if has_labels and np.any(base & label_known):
            mm = binary_metrics(
                y_true_bad=y_bad[base & label_known],
                y_pred_bad=fail[base & label_known],
                known_mask=np.ones(int(np.sum(base & label_known)), dtype=bool),
            )
        else:
            mm = {
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
                "fpr": float("nan"),
                "pred_bad_rate": float(np.mean(fail[base])) if np.any(base) else float("nan"),
            }

        xh, good_count, bad_count = signal_hist_ratio(signals=signal[base], y_bad=y_bad[base], bins=int(args.hist_bins))
        fig_hist = make_signal_hist_fig(
            rule=rule,
            centers=xh,
            good_ratio=good_count,
            bad_ratio=bad_count,
            threshold=thr,
            tail_direction=plot_td_rule,
            warn_threshold=core_thr,
            warn_threshold_low=core_thr_lo,
            warn_threshold_high=core_thr_hi,
            fail_threshold=tail_thr,
            fail_threshold_low=tail_thr_lo,
            fail_threshold_high=tail_thr_hi,
            hard_threshold=exc_thr,
            hard_threshold_low=exc_thr_lo,
            hard_threshold_high=exc_thr_hi,
        )
        fig_box = make_signal_box_fig(
            rule=rule,
            signals=signal[base],
            threshold=thr,
            tail_direction=plot_td_rule,
            warn_threshold=core_thr,
            warn_threshold_low=core_thr_lo,
            warn_threshold_high=core_thr_hi,
            fail_threshold=tail_thr,
            fail_threshold_low=tail_thr_lo,
            fail_threshold_high=tail_thr_hi,
            hard_threshold=exc_thr,
            hard_threshold_low=exc_thr_lo,
            hard_threshold_high=exc_thr_hi,
        )
        html_parts.append(f"<h2>{RULE_LABEL.get(rule, rule)}</h2>")
        html_parts.append(
            "<div class='meta'>"
            f"thresholds(core-range/tail_start/exceptional_out)=<code>[{core_thr_lo:.6g}, {core_thr_hi:.6g}]</code> / "
            f"<code>{tail_thr:.6g}</code> / <code>{exc_thr:.6g}</code>, "
            f"selected=<code>{selected_method}</code>, "
            f"source=<code>{source}</code>, support=<code>{int(np.sum(base))}</code>, "
            f"support(available)=<code>{int(np.sum(base_primary))}</code>, "
            f"warn_n=<code>{int(np.sum(warn_only & base))}</code>, fail_n=<code>{int(np.sum(fail_only & base))}</code>, hard_n=<code>{int(np.sum(hard_only & base))}</code>, "
            f"precision=<code>{mm['precision']:.3f}</code>, recall=<code>{mm['recall']:.3f}</code>, f1=<code>{mm['f1']:.3f}</code>"
            + ("<br><span class='warn'>available mask had zero support; diagnostics fallback used finite-signal rows.</span>" if use_relaxed_base else "")
            + "</div>"
        )
        html_parts.append("<h3>Graph 1. Signal Histogram Count</h3>")
        html_parts.append(f"<div class='chart chart-half'>{pio.to_html(fig_hist, full_html=False, include_plotlyjs=include_js)}</div>")
        html_parts.append("<h3>Graph 2. Signal Boxplot</h3>")
        html_parts.append(f"<div class='chart chart-half'>{pio.to_html(fig_box, full_html=False, include_plotlyjs=False)}</div>")
        include_js = False

        plot_summary_rows.append(
            {
                "rule": rule,
                "threshold_policy": "hybrid_tailstart_core_exceptional",
                "selected_method": selected_method,
                "threshold_applied": float(thr),
                "core_threshold": float(core_thr),
                "tail_start_threshold": float(tail_thr),
                "exceptional_out_threshold": float(exc_thr),
                "warn_threshold": float(core_thr),
                "fail_threshold": float(tail_thr),
                "hard_fail_threshold": float(exc_thr),
                "threshold_source": source,
                "precision": float(mm["precision"]),
                "recall": float(mm["recall"]),
                "f1": float(mm["f1"]),
                "fpr": float(mm["fpr"]),
                "pred_bad_rate": float(mm["pred_bad_rate"]),
                "warn_n": int(np.sum(warn_only & base)),
                "fail_n": int(np.sum(fail_only & base)),
                "hard_n": int(np.sum(hard_only & base)),
                "support_rows": int(np.sum(base)),
                "label_support_rows": int(np.sum(base & label_known)),
            }
        )

    # Input-side distribution (explicit section).
    input_signal, input_signal_name = resolve_input_distribution_signal(
        row_df=row_df,
        source_df=source_df,
        args=args,
    )
    if input_signal is not None:
        input_base = diag_eval_mask & np.isfinite(input_signal)
        input_support = int(np.sum(input_base))
        if input_support > 0:
            xh, good_count, bad_count = signal_hist_ratio(
                signals=input_signal[input_base],
                y_bad=y_bad[input_base],
                bins=int(args.hist_bins),
            )
            fig_input_hist = make_signal_hist_fig(
                rule="input_distribution",
                centers=xh,
                good_ratio=good_count,
                bad_ratio=bad_count,
                threshold=float("nan"),
                tail_direction="upper",
                warn_threshold=float("nan"),
                warn_threshold_low=float("nan"),
                warn_threshold_high=float("nan"),
                fail_threshold=float("nan"),
                fail_threshold_low=float("nan"),
                fail_threshold_high=float("nan"),
                hard_threshold=float("nan"),
                hard_threshold_low=float("nan"),
                hard_threshold_high=float("nan"),
            )
            fig_input_box = make_signal_box_fig(
                rule="input_distribution",
                signals=input_signal[input_base],
                threshold=float("nan"),
                tail_direction="upper",
                warn_threshold=float("nan"),
                warn_threshold_low=float("nan"),
                warn_threshold_high=float("nan"),
                fail_threshold=float("nan"),
                fail_threshold_low=float("nan"),
                fail_threshold_high=float("nan"),
                hard_threshold=float("nan"),
                hard_threshold_low=float("nan"),
                hard_threshold_high=float("nan"),
            )
            html_parts.append("<h2>Input Distribution</h2>")
            html_parts.append(
                "<div class='meta'>"
                f"signal=<code>{html_lib.escape(input_signal_name)}</code>, "
                f"support=<code>{input_support}</code>, "
                f"label_support=<code>{int(np.sum(input_base & label_known))}</code>"
                "</div>"
            )
            html_parts.append("<h3>Graph 1. Signal Histogram Count</h3>")
            html_parts.append(
                f"<div class='chart chart-half'>{pio.to_html(fig_input_hist, full_html=False, include_plotlyjs=include_js)}</div>"
            )
            html_parts.append("<h3>Graph 2. Signal Boxplot</h3>")
            html_parts.append(
                f"<div class='chart chart-half'>{pio.to_html(fig_input_box, full_html=False, include_plotlyjs=False)}</div>"
            )
            include_js = False

            plot_summary_rows.append(
                {
                    "rule": "input_distribution",
                    "threshold_policy": "diagnostic_input_signal_only",
                    "selected_method": input_signal_name,
                    "threshold_applied": float("nan"),
                    "core_threshold": float("nan"),
                    "tail_start_threshold": float("nan"),
                    "exceptional_out_threshold": float("nan"),
                    "warn_threshold": float("nan"),
                    "fail_threshold": float("nan"),
                    "hard_fail_threshold": float("nan"),
                    "threshold_source": "diagnostic_only",
                    "precision": float("nan"),
                    "recall": float("nan"),
                    "f1": float("nan"),
                    "fpr": float("nan"),
                    "pred_bad_rate": float("nan"),
                    "warn_n": 0,
                    "fail_n": 0,
                    "hard_n": 0,
                    "support_rows": int(input_support),
                    "label_support_rows": int(np.sum(input_base & label_known)),
                }
            )

    if detail_paths is not None:
        gate_summary_csv = detail_paths.get("gate_summary_csv")
        gate_row_hits_csv = detail_paths.get("gate_row_hits_csv")
        html_parts.append("<h2>Leaf Diagnostics (Graph 1 + Graph 2)</h2>")
        if gate_summary_csv is None or gate_row_hits_csv is None or (not gate_summary_csv.exists()) or (not gate_row_hits_csv.exists()):
            html_parts.append(
                "<div class='meta'>"
                "leaf gate artifacts not found "
                "(required: gate_summary_csv, gate_row_hits_csv)."
                "</div>"
            )
        else:
            try:
                leaf_summary_df = pd.read_csv(gate_summary_csv)
                leaf_hits_df = pd.read_csv(gate_row_hits_csv)
                if leaf_summary_df.empty or leaf_hits_df.empty:
                    html_parts.append("<div class='meta'>leaf gate artifacts are empty.</div>")
                else:
                    leaf_hits_df = leaf_hits_df.copy()
                    leaf_hits_df["group_id"] = leaf_hits_df.get("group_id", "").fillna("").astype(str)
                    leaf_hits_df["leaf_path"] = leaf_hits_df.get("leaf_path", "").fillna("").astype(str)
                    leaf_hits_df["row_index"] = pd.to_numeric(leaf_hits_df.get("row_index", -1), errors="coerce").fillna(-1).astype(int)

                    for _, srow in leaf_summary_df.iterrows():
                        leaf_path = str(srow.get("leaf_path", "")).strip()
                        gid_raw = str(srow.get("group_id_raw", "")).strip()
                        if not gid_raw:
                            gid_full = str(srow.get("group_id", "")).strip()
                            gid_raw = gid_full.split("__leaf__", 1)[0] if "__leaf__" in gid_full else gid_full
                        if not leaf_path:
                            continue
                        sub = leaf_hits_df[(leaf_hits_df["group_id"] == gid_raw) & (leaf_hits_df["leaf_path"] == leaf_path)].copy()
                        if sub.empty:
                            continue

                        signal = pd.to_numeric(sub.get("output_signal_leaf_nomask", np.nan), errors="coerce").to_numpy(dtype=float)
                        available = (
                            bool_series(sub["output_available_leaf_nomask"])
                            if "output_available_leaf_nomask" in sub.columns
                            else np.ones(len(sub), dtype=bool)
                        )
                        row_idx = pd.to_numeric(sub["row_index"], errors="coerce").fillna(-1).astype(int).to_numpy(dtype=int)
                        row_valid = (row_idx >= 0) & (row_idx < len(y_bad))
                        base = np.isfinite(signal) & available & row_valid
                        if int(np.sum(base)) == 0:
                            continue

                        y_bad_leaf = np.zeros(len(sub), dtype=bool)
                        label_known_leaf = np.zeros(len(sub), dtype=bool)
                        y_bad_leaf[row_valid] = y_bad[row_idx[row_valid]]
                        label_known_leaf[row_valid] = label_known[row_idx[row_valid]]

                        if "output_pass_leaf_nomask" in sub.columns:
                            pred_pass = bool_series(sub["output_pass_leaf_nomask"])
                        else:
                            pred_pass = bool_series(sub["distribution_pass_leaf_nomask"])
                        fail = ~pred_pass

                        if has_labels and np.any(base & label_known_leaf):
                            mm = binary_metrics(
                                y_true_bad=y_bad_leaf[base & label_known_leaf],
                                y_pred_bad=fail[base & label_known_leaf],
                                known_mask=np.ones(int(np.sum(base & label_known_leaf)), dtype=bool),
                            )
                        else:
                            mm = {
                                "precision": float("nan"),
                                "recall": float("nan"),
                                "f1": float("nan"),
                                "fpr": float("nan"),
                                "pred_bad_rate": float(np.mean(fail[base])) if np.any(base) else float("nan"),
                            }

                        thr = float(pd.to_numeric(srow.get("output_threshold", np.nan), errors="coerce"))

                        xh, good_count, bad_count = signal_hist_ratio(
                            signals=signal[base],
                            y_bad=y_bad_leaf[base],
                            bins=int(args.hist_bins),
                        )
                        fig_hist = make_signal_hist_fig(
                            rule=f"leaf:{leaf_path}",
                            centers=xh,
                            good_ratio=good_count,
                            bad_ratio=bad_count,
                            threshold=thr,
                            tail_direction=plot_tail_direction,
                            warn_threshold=float("nan"),
                            warn_threshold_low=float("nan"),
                            warn_threshold_high=float("nan"),
                            fail_threshold=thr,
                            fail_threshold_low=float("nan"),
                            fail_threshold_high=float("nan"),
                            hard_threshold=float("nan"),
                            hard_threshold_low=float("nan"),
                            hard_threshold_high=float("nan"),
                        )
                        fig_box = make_signal_box_fig(
                            rule=f"leaf:{leaf_path}",
                            signals=signal[base],
                            threshold=thr,
                            tail_direction=plot_tail_direction,
                            warn_threshold=float("nan"),
                            warn_threshold_low=float("nan"),
                            warn_threshold_high=float("nan"),
                            fail_threshold=thr,
                            fail_threshold_low=float("nan"),
                            fail_threshold_high=float("nan"),
                            hard_threshold=float("nan"),
                            hard_threshold_low=float("nan"),
                            hard_threshold_high=float("nan"),
                        )
                        html_parts.append(
                            "<h3>"
                            f"Leaf <code>{html_lib.escape(leaf_path)}</code> "
                            f"(group=<code>{html_lib.escape(gid_raw)}</code>)"
                            "</h3>"
                        )
                        html_parts.append(
                            "<div class='meta'>"
                            f"rule=<code>output</code>, support=<code>{int(np.sum(base))}</code>, "
                            f"threshold=<code>{thr:.6g}</code>, "
                            f"fail_n=<code>{int(np.sum(fail & base))}</code>, "
                            f"precision=<code>{mm['precision']:.3f}</code>, recall=<code>{mm['recall']:.3f}</code>, f1=<code>{mm['f1']:.3f}</code>"
                            "</div>"
                        )
                        html_parts.append("<h3>Graph 1. Signal Histogram Count</h3>")
                        html_parts.append(f"<div class='chart chart-half'>{pio.to_html(fig_hist, full_html=False, include_plotlyjs=include_js)}</div>")
                        html_parts.append("<h3>Graph 2. Signal Boxplot</h3>")
                        html_parts.append(f"<div class='chart chart-half'>{pio.to_html(fig_box, full_html=False, include_plotlyjs=False)}</div>")
                        include_js = False
            except Exception as exc:
                html_parts.append(
                    "<div class='meta'>"
                    f"leaf diagnostics render failed: <code>{html_lib.escape(str(exc))}</code>"
                    "</div>"
                )

    html_parts.append("</body></html>")
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text("\n".join(html_parts), encoding="utf-8")

    pd.DataFrame(plot_summary_rows).to_csv(out_summary_csv, index=False)


def render_distribution_diagnostics_plot_fallback(
    source_df: pd.DataFrame,
    row_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    rules: list[str],
    args: argparse.Namespace,
    out_html: Path,
    out_summary_csv: Path,
    detail_paths: dict[str, Path] | None = None,
) -> None:
    """Fallback diagnostics HTML when plotly is unavailable.

    This keeps distribution visibility using lightweight HTML tables/bars.
    """
    y_bad, label_known, has_labels = compute_labels_bad(
        source_df=source_df,
        row_df=row_df,
        source_id_col=args.source_id_col,
        results_id_col=args.results_id_col,
        label_col=args.label_col,
        bad_label=args.bad_label,
    )
    diag_eval_mask = (
        bool_series(row_df["distribution_evaluated_nomask"])
        if "distribution_evaluated_nomask" in row_df.columns
        else np.ones(len(row_df), dtype=bool)
    )

    plot_tail_direction = "upper" if str(args.tail_direction) == "two_sided" else str(args.tail_direction)
    sidx = {str(r["rule"]): i for i, r in summary_df.iterrows()} if not summary_df.empty else {}

    html_parts: list[str] = []
    html_parts.append(
        "<html><head><meta charset='utf-8'><title>Final Metric Distribution Diagnostics (Fallback)</title>"
        "<style>"
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:24px;}"
        "h1{margin-bottom:10px;}h2{margin-top:30px;margin-bottom:10px;}"
        ".meta{color:#374151;font-size:13px;margin-bottom:14px;line-height:1.5;}"
        "table{border-collapse:collapse;width:min(1100px,96vw);font-size:12px;margin-bottom:14px;}"
        "th,td{border:1px solid #e5e7eb;padding:6px 8px;text-align:right;}"
        "th:first-child,td:first-child{text-align:left;}"
        "th{background:#f8fafc;}"
        ".barwrap{width:180px;height:10px;background:#f1f5f9;border-radius:999px;overflow:hidden;}"
        ".bar{height:10px;background:#2563eb;}"
        ".boxplot-wrap{position:relative;width:220px;height:14px;background:#f8fafc;border:1px solid #e5e7eb;border-radius:999px;overflow:hidden;}"
        ".boxplot-whisker{position:absolute;top:6px;height:2px;background:#94a3b8;}"
        ".boxplot-box{position:absolute;top:2px;height:10px;background:#93c5fd;border:1px solid #2563eb;border-radius:4px;}"
        ".boxplot-med{position:absolute;top:1px;height:12px;width:2px;background:#1d4ed8;}"
        ".leaf-list details{margin:8px 0;}.leaf-list summary{cursor:pointer;}"
        ".leaf-frame{width:min(1600px,95vw);height:460px;border:1px solid #e5e7eb;border-radius:8px;margin-top:8px;}"
        "code{background:#f3f4f6;padding:2px 6px;border-radius:6px;}"
        ".warn{color:#92400e;}"
        "</style></head><body>"
    )
    html_parts.append("<h1>Final Metric Distribution Diagnostics (Fallback)</h1>")
    html_parts.append(
        "<div class='meta'>"
        f"plotly 설치가 없어 fallback 뷰로 생성됨. tail_direction(config)=<code>{args.tail_direction}</code>, effective=<code>{plot_tail_direction}</code><br>"
        f"rules={', '.join(rules)}<br>"
        f"labels_available=<code>{str(has_labels).lower()}</code><br>"
        "diagnostic_scope=<code>distribution_evaluated_nomask</code> (all inspected rows)"
        "</div>"
    )

    plot_summary_rows: list[dict[str, Any]] = []

    for rule in rules:
        signal_col = resolve_signal_col(rule, row_df)
        if not signal_col or signal_col not in row_df.columns:
            continue

        signal = pd.to_numeric(row_df[signal_col], errors="coerce").to_numpy(dtype=float)
        avail_col = AVAILABLE_COL.get(rule, "")
        available = bool_series(row_df[avail_col]) if (avail_col and avail_col in row_df.columns) else np.ones(len(row_df), dtype=bool)
        finite_signal = np.isfinite(signal)
        base_primary = diag_eval_mask & available & finite_signal
        base_relaxed = diag_eval_mask & finite_signal
        use_relaxed_base = False
        if int(np.sum(base_primary)) == 0 and int(np.sum(base_relaxed)) > 0:
            base = base_relaxed
            use_relaxed_base = True
        else:
            base = base_primary
        if int(np.sum(base)) == 0:
            continue

        pass_col = f"{rule}_pass_nomask"
        pred_pass = bool_series(row_df[pass_col]) if pass_col in row_df.columns else np.ones(len(row_df), dtype=bool)
        fail = ~pred_pass
        warn_only = bool_series(row_df[f"{rule}_warn_nomask"]) if f"{rule}_warn_nomask" in row_df.columns else np.zeros(len(row_df), dtype=bool)
        fail_only = bool_series(row_df[f"{rule}_fail_nomask"]) if f"{rule}_fail_nomask" in row_df.columns else fail.copy()
        hard_only = bool_series(row_df[f"{rule}_hard_fail_nomask"]) if f"{rule}_hard_fail_nomask" in row_df.columns else np.zeros(len(row_df), dtype=bool)

        ridx = sidx.get(rule)
        if ridx is not None:
            row = summary_df.iloc[int(ridx)]
            core_thr = float(row.get("core_threshold", row.get("warn_threshold", np.nan)))
            tail_thr = float(row.get("tail_start_threshold", row.get("fail_threshold", np.nan)))
            exc_thr = float(row.get("exceptional_out_threshold", row.get("hard_fail_threshold", np.nan)))
            selected_method = str(row.get("selected_method", "unknown"))
            source = str(row.get("threshold_source", ""))
        else:
            core_thr = float("nan")
            tail_thr = float("nan")
            exc_thr = float("nan")
            selected_method = "unknown"
            source = ""

        if has_labels and np.any(base & label_known):
            mm = binary_metrics(
                y_true_bad=y_bad[base & label_known],
                y_pred_bad=fail[base & label_known],
                known_mask=np.ones(int(np.sum(base & label_known)), dtype=bool),
            )
        else:
            mm = {
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
                "fpr": float("nan"),
                "pred_bad_rate": float(np.mean(fail[base])) if np.any(base) else float("nan"),
            }

        sig = signal[base]
        xh, good_count, bad_count = signal_hist_ratio(
            signals=sig,
            y_bad=y_bad[base],
            bins=int(args.hist_bins),
        )
        q50, q90, q95, q99 = [float(np.quantile(sig, q)) for q in [0.50, 0.90, 0.95, 0.99]]
        smax = float(np.max(sig))
        smean = float(np.mean(sig))

        html_parts.append(f"<h2>{RULE_LABEL.get(rule, rule)}</h2>")
        html_parts.append(
            "<div class='meta'>"
            f"signal_col=<code>{signal_col}</code>, support=<code>{int(np.sum(base))}</code>, "
            f"support(available)=<code>{int(np.sum(base_primary))}</code>, "
            f"selected=<code>{selected_method}</code>, source=<code>{source}</code><br>"
            f"core/tail_start/exceptional_out=<code>{core_thr:.6g}</code> / <code>{tail_thr:.6g}</code> / <code>{exc_thr:.6g}</code><br>"
            f"warn_n=<code>{int(np.sum(warn_only & base))}</code>, fail_n=<code>{int(np.sum(fail_only & base))}</code>, hard_n=<code>{int(np.sum(hard_only & base))}</code>, "
            f"precision=<code>{mm['precision']:.3f}</code>, recall=<code>{mm['recall']:.3f}</code>, f1=<code>{mm['f1']:.3f}</code>"
            + ("<br><span class='warn'>available mask had zero support; diagnostics fallback used finite-signal rows.</span>" if use_relaxed_base else "")
            + "</div>"
        )
        html_parts.append(
            "<table><thead><tr><th>Stats</th><th>mean</th><th>q50</th><th>q90</th><th>q95</th><th>q99</th><th>max</th></tr></thead>"
            "<tbody>"
            f"<tr><td>signal distribution</td><td>{smean:.6g}</td><td>{q50:.6g}</td><td>{q90:.6g}</td><td>{q95:.6g}</td><td>{q99:.6g}</td><td>{smax:.6g}</td></tr>"
            "</tbody></table>"
        )

        html_parts.append("<h3>Graph 1. Signal Histogram Count</h3>")
        rows_html: list[str] = []
        total_count = np.asarray(good_count, dtype=float) + np.asarray(bad_count, dtype=float)
        max_total = float(np.nanmax(total_count)) if len(total_count) and np.any(np.isfinite(total_count)) else 1.0
        max_total = max(max_total, 1e-9)
        for x, good_cnt, bad_cnt, t_cnt in zip(xh, good_count, bad_count, total_count):
            width = float(np.clip((float(t_cnt) / max_total) * 180.0, 0.0, 180.0)) if np.isfinite(t_cnt) else 0.0
            rows_html.append(
                "<tr>"
                f"<td>{float(x):.6g}</td>"
                f"<td>{int(good_cnt)}</td>"
                f"<td>{int(bad_cnt)}</td>"
                f"<td>{int(t_cnt)}</td>"
                f"<td><div class='barwrap'><div class='bar' style='width:{width:.1f}px'></div></div></td>"
                "</tr>"
            )
        html_parts.append(
            "<table><thead><tr><th>bin_center</th><th>good_count</th><th>bad_count</th><th>total_count</th><th>relative bar</th></tr></thead>"
            f"<tbody>{''.join(rows_html)}</tbody></table>"
        )
        box = compute_boxplot_summary(sig)
        html_parts.append("<h3>Graph 2. Signal Boxplot</h3>")
        html_parts.append(
            "<table><thead><tr><th>boxplot</th><th>min</th><th>q1</th><th>median</th><th>q3</th><th>max</th><th>iqr</th><th>outlier_n</th><th>shape</th></tr></thead>"
            "<tbody>"
            f"<tr><td>signal boxplot</td><td>{box['min']:.6g}</td><td>{box['q1']:.6g}</td><td>{box['median']:.6g}</td><td>{box['q3']:.6g}</td><td>{box['max']:.6g}</td><td>{box['iqr']:.6g}</td><td>{int(box['outlier_n'])}</td><td>{render_boxplot_mini_html(box)}</td></tr>"
            "</tbody></table>"
        )

        plot_summary_rows.append(
            {
                "rule": rule,
                "threshold_policy": "hybrid_tailstart_core_exceptional_fallback_html",
                "selected_method": selected_method,
                "core_threshold": float(core_thr),
                "tail_start_threshold": float(tail_thr),
                "exceptional_out_threshold": float(exc_thr),
                "threshold_source": source,
                "precision": float(mm["precision"]),
                "recall": float(mm["recall"]),
                "f1": float(mm["f1"]),
                "fpr": float(mm["fpr"]),
                "pred_bad_rate": float(mm["pred_bad_rate"]),
                "warn_n": int(np.sum(warn_only & base)),
                "fail_n": int(np.sum(fail_only & base)),
                "hard_n": int(np.sum(hard_only & base)),
                "support_rows": int(np.sum(base)),
                "label_support_rows": int(np.sum(base & label_known)),
                "signal_mean": float(smean),
                "signal_q95": float(q95),
                "signal_max": float(smax),
            }
        )

    # Input-side distribution (explicit section).
    input_signal, input_signal_name = resolve_input_distribution_signal(
        row_df=row_df,
        source_df=source_df,
        args=args,
    )
    if input_signal is not None:
        input_base = diag_eval_mask & np.isfinite(input_signal)
        input_support = int(np.sum(input_base))
        if input_support > 0:
            sig = input_signal[input_base]
            xh, good_count, bad_count = signal_hist_ratio(
                signals=sig,
                y_bad=y_bad[input_base],
                bins=int(args.hist_bins),
            )
            q50, q90, q95, q99 = [float(np.quantile(sig, q)) for q in [0.50, 0.90, 0.95, 0.99]]
            smax = float(np.max(sig))
            smean = float(np.mean(sig))

            html_parts.append("<h2>Input Distribution</h2>")
            html_parts.append(
                "<div class='meta'>"
                f"signal=<code>{html_lib.escape(input_signal_name)}</code>, "
                f"support=<code>{input_support}</code>, "
                f"label_support=<code>{int(np.sum(input_base & label_known))}</code>"
                "</div>"
            )
            html_parts.append(
                "<table><thead><tr><th>Stats</th><th>mean</th><th>q50</th><th>q90</th><th>q95</th><th>q99</th><th>max</th></tr></thead>"
                "<tbody>"
                f"<tr><td>signal distribution</td><td>{smean:.6g}</td><td>{q50:.6g}</td><td>{q90:.6g}</td><td>{q95:.6g}</td><td>{q99:.6g}</td><td>{smax:.6g}</td></tr>"
                "</tbody></table>"
            )
            html_parts.append("<h3>Graph 1. Signal Histogram Count</h3>")
            rows_html: list[str] = []
            total_count = np.asarray(good_count, dtype=float) + np.asarray(bad_count, dtype=float)
            max_total = float(np.nanmax(total_count)) if len(total_count) and np.any(np.isfinite(total_count)) else 1.0
            max_total = max(max_total, 1e-9)
            for x, good_cnt, bad_cnt, t_cnt in zip(xh, good_count, bad_count, total_count):
                width = float(np.clip((float(t_cnt) / max_total) * 180.0, 0.0, 180.0)) if np.isfinite(t_cnt) else 0.0
                rows_html.append(
                    "<tr>"
                    f"<td>{float(x):.6g}</td>"
                    f"<td>{int(good_cnt)}</td>"
                    f"<td>{int(bad_cnt)}</td>"
                    f"<td>{int(t_cnt)}</td>"
                    f"<td><div class='barwrap'><div class='bar' style='width:{width:.1f}px'></div></div></td>"
                    "</tr>"
                )
            html_parts.append(
                "<table><thead><tr><th>bin_center</th><th>good_count</th><th>bad_count</th><th>total_count</th><th>relative bar</th></tr></thead>"
                f"<tbody>{''.join(rows_html)}</tbody></table>"
            )
            box = compute_boxplot_summary(sig)
            html_parts.append("<h3>Graph 2. Signal Boxplot</h3>")
            html_parts.append(
                "<table><thead><tr><th>boxplot</th><th>min</th><th>q1</th><th>median</th><th>q3</th><th>max</th><th>iqr</th><th>outlier_n</th><th>shape</th></tr></thead>"
                "<tbody>"
                f"<tr><td>signal boxplot</td><td>{box['min']:.6g}</td><td>{box['q1']:.6g}</td><td>{box['median']:.6g}</td><td>{box['q3']:.6g}</td><td>{box['max']:.6g}</td><td>{box['iqr']:.6g}</td><td>{int(box['outlier_n'])}</td><td>{render_boxplot_mini_html(box)}</td></tr>"
                "</tbody></table>"
            )

            plot_summary_rows.append(
                {
                    "rule": "input_distribution",
                    "threshold_policy": "diagnostic_input_signal_only_fallback_html",
                    "selected_method": input_signal_name,
                    "core_threshold": float("nan"),
                    "tail_start_threshold": float("nan"),
                    "exceptional_out_threshold": float("nan"),
                    "threshold_source": "diagnostic_only",
                    "precision": float("nan"),
                    "recall": float("nan"),
                    "f1": float("nan"),
                    "fpr": float("nan"),
                    "pred_bad_rate": float("nan"),
                    "warn_n": 0,
                    "fail_n": 0,
                    "hard_n": 0,
                    "support_rows": int(input_support),
                    "label_support_rows": int(np.sum(input_base & label_known)),
                    "signal_mean": float(smean),
                    "signal_q95": float(q95),
                    "signal_max": float(smax),
                }
            )

    if detail_paths is not None:
        html_parts.append(
            _render_leaf_distribution_section_html(
                detail_paths=detail_paths,
                include_iframe=True,
            )
        )

    html_parts.append("<div class='meta warn'>fallback 뷰는 인터랙티브 Plotly 그래프 대신 분포 통계/빈도표를 제공합니다.</div>")
    html_parts.append("</body></html>")
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text("\n".join(html_parts), encoding="utf-8")
    pd.DataFrame(plot_summary_rows).to_csv(out_summary_csv, index=False)


def build_operational_summary(row_df: pd.DataFrame, threshold_summary_df: pd.DataFrame) -> pd.DataFrame:
    hard_gate = bool_series(row_df["hard_gate_pass"]) if "hard_gate_pass" in row_df.columns else np.ones(len(row_df), dtype=bool)
    eval_mask = bool_series(row_df["distribution_evaluated_nomask"]) if "distribution_evaluated_nomask" in row_df.columns else hard_gate
    dist_pass = bool_series(row_df["distribution_pass_nomask"]) if "distribution_pass_nomask" in row_df.columns else np.zeros(len(row_df), dtype=bool)
    dist_warn = bool_series(row_df["distribution_warn_nomask"]) if "distribution_warn_nomask" in row_df.columns else np.zeros(len(row_df), dtype=bool)
    dist_fail = bool_series(row_df["distribution_fail_nomask"]) if "distribution_fail_nomask" in row_df.columns else np.zeros(len(row_df), dtype=bool)
    dist_hard = bool_series(row_df["distribution_hard_fail_nomask"]) if "distribution_hard_fail_nomask" in row_df.columns else np.zeros(len(row_df), dtype=bool)
    final_pass = bool_series(row_df["final_pass_nomask"]) if "final_pass_nomask" in row_df.columns else (hard_gate & dist_pass)

    row = {
        "mode": "nomask",
        "rows": int(len(row_df)),
        "hard_gate_pass_rate": float(np.mean(hard_gate)) if len(hard_gate) else 0.0,
        "distribution_eval_coverage": float(np.mean(eval_mask)) if len(eval_mask) else 0.0,
        "distribution_pass_rate": float(np.mean(dist_pass[eval_mask])) if np.any(eval_mask) else 0.0,
        "distribution_warn_rate": float(np.mean(dist_warn[eval_mask])) if np.any(eval_mask) else 0.0,
        "distribution_fail_rate": float(np.mean(dist_fail[eval_mask])) if np.any(eval_mask) else 0.0,
        "distribution_hard_fail_rate": float(np.mean(dist_hard[eval_mask])) if np.any(eval_mask) else 0.0,
        "final_pass_rate": float(np.mean(final_pass)) if len(final_pass) else 0.0,
    }

    for _, r in threshold_summary_df.iterrows():
        rule = str(r.get("rule", "")).strip().lower()
        if not rule:
            continue
        row[f"{rule}_threshold"] = float(r.get("threshold_applied", np.nan))
        row[f"{rule}_core_threshold"] = float(r.get("core_threshold", r.get("warn_threshold", np.nan)))
        row[f"{rule}_warn_threshold"] = float(r.get("warn_threshold", np.nan))
        row[f"{rule}_tail_start_threshold"] = float(r.get("tail_start_threshold", r.get("fail_threshold", np.nan)))
        row[f"{rule}_fail_threshold"] = float(r.get("fail_threshold", np.nan))
        row[f"{rule}_exceptional_out_threshold"] = float(r.get("exceptional_out_threshold", r.get("hard_fail_threshold", np.nan)))
        row[f"{rule}_hard_fail_threshold"] = float(r.get("hard_fail_threshold", np.nan))
        row[f"{rule}_threshold_method"] = str(r.get("selected_method", ""))

    return pd.DataFrame([row])


def _axis_role_from_submetric(submetric: str) -> str:
    s = str(submetric or "").strip()
    if s.endswith("_core"):
        return "core"
    if s.endswith("_rate"):
        return "rate"
    if s.endswith("_mass"):
        return "mass"
    return "aux"


def _score_band_text(score: float) -> str:
    if not np.isfinite(score):
        return "NA"
    v = float(score)
    if v >= 4.0:
        return "양호"
    if v >= 3.0:
        return "보통"
    if v >= 2.0:
        return "주의"
    return "위험"


def _axis_interpretation_text(axis_role: str, axis_score: float, is_na: bool) -> str:
    if bool(is_na) or (not np.isfinite(axis_score)):
        return "NA(미산출)"
    v = float(axis_score)
    if axis_role == "core":
        if v >= 4.0:
            return "정상 구간 흔들림이 작음"
        if v >= 3.0:
            return "코어 흔들림이 경미함"
        if v >= 2.0:
            return "코어 분산이 다소 큼"
        return "코어 안정성이 낮음"
    if axis_role == "rate":
        if v >= 4.0:
            return "경계 이탈 빈도가 낮음"
        if v >= 3.0:
            return "이탈 빈도는 관리 가능"
        if v >= 2.0:
            return "이탈 빈도 주의 필요"
        return "이탈 빈도가 높음"
    if axis_role == "mass":
        if v >= 4.0:
            return "이탈 시 붕괴 깊이가 얕음"
        if v >= 3.0:
            return "붕괴 깊이는 중간 수준"
        if v >= 2.0:
            return "붕괴 깊이가 다소 큼"
        return "붕괴 깊이가 큼"
    return "보조 지표(해석 참고)"


def build_bundle_ab_interpretation_csv(
    *,
    tag: str,
    stem: str,
    summary_df: pd.DataFrame,
    detail_df: pd.DataFrame,
    payload: dict[str, Any],
    out_csv: Path,
) -> pd.DataFrame:
    summary_by_bundle: dict[str, dict[str, Any]] = {
        str(r.get("bundle")): dict(r)
        for r in summary_df.to_dict(orient="records")
    }
    triplet_meta: dict[str, Any] = (
        payload.get("raw", {}).get("triplet_aggregation", {})
        if isinstance(payload.get("raw", {}), dict)
        else {}
    )
    axis_order_by_bundle: dict[str, list[str]] = {
        "OUT": ["OUT_core", "OUT_rate", "OUT_mass"],
        "RID": ["RID_core", "RID_rate", "RID_mass"],
        "SEM": ["SEM_core", "SEM_rate", "SEM_mass"],
        "DIAG": ["DIAG_core", "DIAG_rate", "DIAG_mass"],
        "CONF": ["CONF_data", "CONF_calc", "CONF_th", "CONF_op"],
    }

    rows: list[dict[str, Any]] = []
    for r in detail_df.to_dict(orient="records"):
        bundle = str(r.get("bundle", ""))
        submetric = str(r.get("submetric", ""))
        sum_row = summary_by_bundle.get(bundle, {})
        axis_role = _axis_role_from_submetric(submetric)
        subscore_precise = float(r.get("subscore_precise", np.nan))
        is_na = bool(r.get("is_na", False))
        axis_interp = _axis_interpretation_text(axis_role, subscore_precise, is_na)

        tmeta = dict(triplet_meta.get(bundle, {})) if isinstance(triplet_meta, dict) else {}
        order = axis_order_by_bundle.get(bundle, [])
        axis_idx = order.index(submetric) if submetric in order else -1
        t_risks = tmeta.get("risks") if isinstance(tmeta.get("risks"), list) else []
        t_risks_raw = tmeta.get("risks_raw") if isinstance(tmeta.get("risks_raw"), list) else []
        axis_triplet_risk = (
            float(t_risks[axis_idx])
            if axis_idx >= 0 and axis_idx < len(t_risks) and t_risks[axis_idx] is not None and np.isfinite(float(t_risks[axis_idx]))
            else np.nan
        )
        axis_triplet_risk_raw = (
            float(t_risks_raw[axis_idx])
            if axis_idx >= 0 and axis_idx < len(t_risks_raw) and t_risks_raw[axis_idx] is not None and np.isfinite(float(t_risks_raw[axis_idx]))
            else np.nan
        )

        bundle_score = float(sum_row.get("bundle_score", np.nan))
        new_score = float(sum_row.get("New_score", bundle_score))
        bundle_bucket = (
            int(sum_row.get("bundle_score_bucket", 0))
            if np.isfinite(float(sum_row.get("bundle_score_bucket", np.nan)))
            else np.nan
        )
        subscore = (
            int(r.get("subscore", 0))
            if np.isfinite(float(r.get("subscore", np.nan)))
            else np.nan
        )
        rows.append(
            {
                "case_id": f"{tag}_{stem}",
                "bundle": bundle,
                "submetric": submetric,
                "axis_role": axis_role,
                "bundle_score": bundle_score if np.isfinite(bundle_score) else np.nan,
                "New_score": new_score if np.isfinite(new_score) else np.nan,
                "bundle_score_bucket": bundle_bucket,
                "bundle_confidence": int(sum_row.get("bundle_confidence", 0)) if sum_row else 0,
                "bundle_conf_warning": int(sum_row.get("bundle_conf_warning", 0)) if sum_row else 0,
                "bundle_interpretation": (_score_band_text(float(bundle_bucket)) if np.isfinite(bundle_bucket) else ""),
                "subscore": subscore,
                "subscore_precise": subscore_precise if np.isfinite(subscore_precise) else np.nan,
                "raw_value": float(r.get("raw_value", np.nan)) if np.isfinite(float(r.get("raw_value", np.nan))) else np.nan,
                "bucket_value": float(r.get("bucket_value", np.nan)) if np.isfinite(float(r.get("bucket_value", np.nan))) else np.nan,
                "risk_norm_used": float(r.get("risk_norm_used", np.nan)) if np.isfinite(float(r.get("risk_norm_used", np.nan))) else np.nan,
                "axis_triplet_risk": axis_triplet_risk,
                "axis_triplet_risk_raw": axis_triplet_risk_raw,
                "axis_interpretation": axis_interp,
                "is_na": bool(is_na),
                "na_label": str(r.get("na_label", "")),
                "na_reason": str(r.get("na_reason", "")),
                "bucket_source": str(r.get("bucket_source", "")),
                "bucket_kind": str(r.get("bucket_kind", "")),
                "triplet_formula": str(tmeta.get("formula", "")),
                "triplet_weights": (
                    json.dumps(tmeta.get("weights", []), ensure_ascii=False)
                    if isinstance(tmeta.get("weights", []), list)
                    else "[]"
                ),
                "triplet_subscores": (
                    json.dumps(tmeta.get("subscores", []), ensure_ascii=False)
                    if isinstance(tmeta.get("subscores", []), list)
                    else "[]"
                ),
                "triplet_risks": (
                    json.dumps(tmeta.get("risks", []), ensure_ascii=False)
                    if isinstance(tmeta.get("risks", []), list)
                    else "[]"
                ),
                "triplet_risks_raw": (
                    json.dumps(tmeta.get("risks_raw", []), ensure_ascii=False)
                    if isinstance(tmeta.get("risks_raw", []), list)
                    else "[]"
                ),
                "triplet_risk_or": (
                    float(tmeta.get("risk_or"))
                    if tmeta.get("risk_or") is not None and np.isfinite(float(tmeta.get("risk_or")))
                    else np.nan
                ),
                "triplet_score_precise": (
                    float(tmeta.get("score_precise"))
                    if tmeta.get("score_precise") is not None and np.isfinite(float(tmeta.get("score_precise")))
                    else np.nan
                ),
                "triplet_score_bucket": (
                    int(tmeta.get("score_bucket"))
                    if tmeta.get("score_bucket") is not None and np.isfinite(float(tmeta.get("score_bucket")))
                    else np.nan
                ),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False, float_format="%.4f")
    return out_df


def build_edge_input_aux_payload(
    *,
    row_df: pd.DataFrame,
    top_k: int = 20,
    preview_chars: int = 180,
) -> dict[str, Any]:
    n_rows = int(len(row_df))
    if n_rows <= 0:
        return {
            "enabled": False,
            "reason": "empty_row_df",
            "n_rows": 0,
            "base_rows": 0,
            "active_rules": [],
            "rule_stats": [],
            "top_cases": [],
        }

    hard_gate = bool_series(row_df["hard_gate_pass"]) if "hard_gate_pass" in row_df.columns else np.ones(n_rows, dtype=bool)
    base_rows = int(np.sum(hard_gate))
    if base_rows <= 0:
        return {
            "enabled": False,
            "reason": "no_hard_gate_base_rows",
            "n_rows": n_rows,
            "base_rows": 0,
            "active_rules": [],
            "rule_stats": [],
            "top_cases": [],
        }

    def _pick_col(candidates: list[str]) -> str:
        for c in candidates:
            if c in row_df.columns:
                return c
        return ""

    def _preview_text(v: Any, lim: int) -> str:
        s = "" if pd.isna(v) else str(v)
        t = s.strip()
        if len(t) <= lim:
            return t
        return f"{t[:lim]}…"

    input_rules = ["similar_input_conflict", "direction", "length"]
    active_rules = [r for r in input_rules if f"{r}_state_nomask" in row_df.columns]
    if not active_rules:
        return {
            "enabled": False,
            "reason": "missing_input_rule_states",
            "n_rows": n_rows,
            "base_rows": base_rows,
            "active_rules": [],
            "rule_stats": [],
            "top_cases": [],
        }

    any_warn = np.zeros(n_rows, dtype=bool)
    any_fail = np.zeros(n_rows, dtype=bool)
    any_hard = np.zeros(n_rows, dtype=bool)
    severity = np.zeros(n_rows, dtype=float)
    rule_stats: list[dict[str, Any]] = []

    for rule in active_rules:
        state_col = f"{rule}_state_nomask"
        state = row_df[state_col].astype(str).str.strip().str.lower().to_numpy(dtype=object)
        warn_mask = hard_gate & (state == "warn")
        fail_mask = hard_gate & (state == "fail")
        hard_mask = hard_gate & (state == "hard_fail")

        any_warn |= warn_mask
        any_fail |= fail_mask
        any_hard |= hard_mask

        severity += (hard_mask.astype(float) * 8.0) + (fail_mask.astype(float) * 1.0) + (warn_mask.astype(float) * 0.25)

        signal_col = resolve_signal_col(rule, row_df)
        signal = (
            pd.to_numeric(row_df[signal_col], errors="coerce").to_numpy(dtype=float)
            if signal_col and signal_col in row_df.columns
            else np.full(n_rows, np.nan, dtype=float)
        )
        base_signal = signal[hard_gate & np.isfinite(signal)]
        hard_signal = signal[hard_mask & np.isfinite(signal)]

        rule_stats.append(
            {
                "rule": str(rule),
                "state_col": state_col,
                "signal_col": str(signal_col),
                "warn_rows": int(np.sum(warn_mask)),
                "fail_rows": int(np.sum(fail_mask)),
                "hard_rows": int(np.sum(hard_mask)),
                "warn_rate_base": (float(np.mean(warn_mask[hard_gate])) if base_rows > 0 else 0.0),
                "fail_rate_base": (float(np.mean(fail_mask[hard_gate])) if base_rows > 0 else 0.0),
                "hard_rate_base": (float(np.mean(hard_mask[hard_gate])) if base_rows > 0 else 0.0),
                "signal_median_base": (float(np.median(base_signal)) if base_signal.size > 0 else np.nan),
                "signal_median_hard": (float(np.median(hard_signal)) if hard_signal.size > 0 else np.nan),
            }
        )

    any_issue = any_warn | any_fail | any_hard
    hard_rows = int(np.sum(any_hard))
    fail_rows = int(np.sum(any_fail))
    warn_rows = int(np.sum(any_warn))
    issue_rows = int(np.sum(any_issue))

    candidate_mask = any_hard.copy()
    if not np.any(candidate_mask):
        candidate_mask = any_fail.copy()
    if not np.any(candidate_mask):
        candidate_mask = any_issue.copy()
    candidate_idx = np.where(candidate_mask)[0]

    order = sorted(candidate_idx.tolist(), key=lambda i: (-float(severity[i]), int(i)))
    limit = max(1, int(top_k))
    top_idx = order[:limit]

    row_id_col = _pick_col(["row_id", "id"])
    input_col = _pick_col(["source_input", "input", "prompt", "Prompt"])
    output_col = _pick_col(["source_output", "output", "expectedOutput"])
    final_state_col = _pick_col(["final_state_nomask"])

    top_cases: list[dict[str, Any]] = []
    for i in top_idx:
        hard_rules = [r for r in active_rules if str(row_df.at[i, f"{r}_state_nomask"]).strip().lower() == "hard_fail"]
        fail_rules = [r for r in active_rules if str(row_df.at[i, f"{r}_state_nomask"]).strip().lower() == "fail"]
        warn_rules = [r for r in active_rules if str(row_df.at[i, f"{r}_state_nomask"]).strip().lower() == "warn"]
        top_cases.append(
            {
                "row_index": int(i),
                "row_id": (str(row_df.at[i, row_id_col]) if row_id_col else str(i)),
                "severity": float(severity[i]),
                "final_state_nomask": (str(row_df.at[i, final_state_col]) if final_state_col else ""),
                "hard_rules": hard_rules,
                "fail_rules": fail_rules,
                "warn_rules": warn_rules,
                "source_input_preview": (_preview_text(row_df.at[i, input_col], max(24, int(preview_chars))) if input_col else ""),
                "source_output_preview": (_preview_text(row_df.at[i, output_col], max(24, int(preview_chars))) if output_col else ""),
            }
        )

    return {
        "enabled": True,
        "definition": (
            "EdgeInput는 입력 분포 기반 hard/fail 신호를 보조 진단으로 보여줍니다. "
            "bundle/New_score에는 직접 반영되지 않습니다."
        ),
        "score_usage": "aux_only_not_in_bundle_score",
        "n_rows": n_rows,
        "base_rows": base_rows,
        "active_rules": active_rules,
        "hard_rows": hard_rows,
        "fail_rows": fail_rows,
        "warn_rows": warn_rows,
        "issue_rows": issue_rows,
        "hard_rate_base": (float(hard_rows) / float(base_rows) if base_rows > 0 else 0.0),
        "fail_rate_base": (float(fail_rows) / float(base_rows) if base_rows > 0 else 0.0),
        "warn_rate_base": (float(warn_rows) / float(base_rows) if base_rows > 0 else 0.0),
        "issue_rate_base": (float(issue_rows) / float(base_rows) if base_rows > 0 else 0.0),
        "rule_stats": rule_stats,
        "top_cases": top_cases,
    }


def _run_with_args(args: argparse.Namespace) -> RunArtifacts:
    threshold_cfg = FINAL_THRESHOLD_RUNTIME
    rules = parse_rules(args.rules)

    source_csv = Path(args.source_csv)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir = ensure_report_dir(out_dir, str(args.report_dir_name))

    source_df = pd.read_csv(source_csv)
    if args.max_rows and int(args.max_rows) > 0:
        source_df = source_df.head(int(args.max_rows)).copy()

    if args.row_results_csv:
        row_csv = Path(args.row_results_csv)
    else:
        row_csv = bootstrap_row_results(args=args, source_csv=source_csv, out_dir=out_dir)

    row_df = pd.read_csv(row_csv)
    if args.max_rows and int(args.max_rows) > 0:
        row_df = row_df.head(int(args.max_rows)).copy()
    row_df = ensure_required_columns(row_df=row_df, source_df=source_df, args=args)
    if str(args.inspection_mode) == "detailed":
        validate_detail_columns_or_raise(row_df)

    updated_row_df, threshold_summary_df = apply_hybrid_thresholds_nomask(
        source_df=source_df,
        row_df=row_df,
        rules=rules,
        args=args,
    )

    detail_override_stats: dict[str, Any] = {
        "enabled": False,
        "detail_target_rows": 0,
        "detail_override_rows": 0,
        "detail_fail_rate_hard_eval": 0.0,
        "distribution_pass_rate_pre": float("nan"),
        "distribution_pass_rate_post": float("nan"),
        "final_pass_rate_pre": float("nan"),
        "final_pass_rate_post": float("nan"),
    }
    if str(args.inspection_mode) == "detailed":
        updated_row_df, detail_override_stats = apply_detail_override_nomask(updated_row_df)
        print(
            "[INFO] Applied detailed override (nomask): "
            f"target_rows={int(detail_override_stats['detail_target_rows'])}, "
            f"override_rows={int(detail_override_stats['detail_override_rows'])}, "
            f"final_pass_rate(pre={float(detail_override_stats['final_pass_rate_pre']):.4f}, "
            f"post={float(detail_override_stats['final_pass_rate_post']):.4f})"
        )
    else:
        updated_row_df["detail_override_applied_nomask"] = False

    summary_df = build_operational_summary(updated_row_df, threshold_summary_df)
    summary_df["inspection_mode"] = str(args.inspection_mode)
    summary_df["detail_mode_enabled"] = bool(str(args.inspection_mode) == "detailed")
    summary_df["detail_override_rows"] = int(detail_override_stats["detail_override_rows"])
    summary_df["detail_fail_rate_hard_eval"] = float(detail_override_stats["detail_fail_rate_hard_eval"])
    summary_df["detail_distribution_pass_rate_pre"] = float(detail_override_stats["distribution_pass_rate_pre"])
    summary_df["detail_distribution_pass_rate_post"] = float(detail_override_stats["distribution_pass_rate_post"])
    summary_df["detail_final_pass_rate_pre"] = float(detail_override_stats["final_pass_rate_pre"])
    summary_df["detail_final_pass_rate_post"] = float(detail_override_stats["final_pass_rate_post"])

    stem = source_csv.stem
    tag = args.tag

    row_out = report_dir / f"{tag}_{stem}_row_results.csv"
    summary_out = report_dir / f"{tag}_{stem}_summary.csv"
    cfg_out = report_dir / f"{tag}_{stem}_run_config.json"

    write_csv(updated_row_df, row_out, index=False)
    write_csv(summary_df, summary_out, index=False)

    cfg = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(source_csv),
        "input_row_results_csv": str(row_csv),
        "output_row_results_csv": str(row_out),
        "output_summary_csv": str(summary_out),
        "inspection_mode": str(args.inspection_mode),
        "rules": rules,
        "threshold_policy": "hybrid_tailstart_core_exceptional",
        "hybrid_order": ["robust_z_tail_start", "dist_stability_jump", "quantile_tail_guard"],
        "fallback_trigger": {
            "non_finite_threshold": True,
            "insufficient_support_rows": int(threshold_cfg.min_support_rows),
            "tailstart_missing_signature": True,
        },
        "tail_direction_effective": "upper" if str(args.tail_direction) == "two_sided" else str(args.tail_direction),
        "tri_threshold_policy": {
            "enabled": True,
            "core_definition": "T_core = median(C) + kappa * 1.4826*MAD(C), C={s<=T_tail_start}",
            "core_kappa": float(threshold_cfg.core_kappa),
            "core_fallback_quantile": float(threshold_cfg.core_fallback_quantile),
            "core_min_count": int(threshold_cfg.core_min_count),
            "exceptional_definition": "2nd-derivative floor from tail_start to right tail",
            "exceptional_d1_lambda": float(threshold_cfg.exceptional_d1_lambda),
            "exceptional_d2_lambda": float(threshold_cfg.exceptional_d2_lambda),
            "exceptional_consecutive": int(threshold_cfg.exceptional_consecutive),
            "exceptional_grid_points": int(threshold_cfg.exceptional_grid_points),
            "exceptional_min_tail_points": int(threshold_cfg.exceptional_min_tail_points),
            "exceptional_fallback_quantile": float(threshold_cfg.exceptional_fallback_quantile),
            "exceptional_min_delta_ratio": float(threshold_cfg.exceptional_min_delta_ratio),
        },
        "tristate_policy": {
            "enabled": True,
            "warn_ratio_from_fail": float(threshold_cfg.warn_ratio_from_fail),
            "hard_ratio_from_fail": float(threshold_cfg.hard_ratio_from_fail),
        },
        "detail_override": detail_override_stats,
        "runtime_args": {
            "source_csv": str(source_csv),
            "input_row_results_csv": str(row_csv),
            "inspection_mode": str(args.inspection_mode),
            "rules": rules,
            "max_rows": int(args.max_rows),
            "tail_direction": str(args.tail_direction),
        },
    }

    thr_rule_out = report_dir / f"{tag}_{stem}_rule_thresholds.csv"
    thr_summary_out = report_dir / f"{tag}_{stem}_thresholds_summary.csv"
    thr_compact_out = report_dir / f"{tag}_{stem}_rule_thresholds_compact.csv"
    write_csv(threshold_summary_df, thr_rule_out, index=False)
    write_csv(threshold_summary_df, thr_summary_out, index=False)
    compact_cols = [
        "rule",
        "selected_method",
        "threshold_source",
        "support_rows",
        "core_threshold",
        "warn_threshold",
        "tail_start_threshold",
        "fail_threshold",
        "exceptional_out_threshold",
        "hard_fail_threshold",
        "hard_n",
        "fail_n",
        "warn_n",
    ]
    use_compact = [c for c in compact_cols if c in threshold_summary_df.columns]
    write_csv(threshold_summary_df[use_compact], thr_compact_out, index=False)

    cfg["output_rule_thresholds_csv"] = str(thr_rule_out)
    cfg["output_rule_thresholds_compact_csv"] = str(thr_compact_out)
    cfg["output_thresholds_summary_csv"] = str(thr_summary_out)

    score_summary_out = report_dir / f"{tag}_{stem}_bundle_scores_summary.csv"
    score_detail_out = report_dir / f"{tag}_{stem}_bundle_scores_detail.csv"
    score_ab_interpret_out = report_dir / f"{tag}_{stem}_bundle_scores_ab_interpretation.csv"
    score_json_out = report_dir / f"{tag}_{stem}_bundle_scores.json"
    score_dashboard_out = report_dir / f"{tag}_{stem}_bundle_scores_dashboard.html"
    warn_inspect_json_out = report_dir / f"{tag}_{stem}_warn_inspect.json"
    warn_inspect_csv_out = report_dir / f"{tag}_{stem}_warn_inspect.csv"
    diagnostics_html_out = report_dir / f"{tag}_{stem}_distribution_diagnostics.html"
    diagnostics_summary_out = report_dir / f"{tag}_{stem}_distribution_diagnostics_summary.csv"
    dashboard_diagnostics_path, dashboard_diagnostics_meta = _resolve_dashboard_diagnostics_path(
        diagnostics_html_out=diagnostics_html_out,
    )
    plot_detail_paths: dict[str, Path] | None = None
    if str(args.inspection_mode) == "detailed":
        plot_detail_paths = _resolve_peer_detailed_inspection_paths(row_csv.resolve())
    detail_leaf_gate_tables_payload: dict[str, Any] = {"enabled": False}
    if str(args.inspection_mode) == "detailed":
        peer_detail_paths = plot_detail_paths or {}
        gate_summary_path = peer_detail_paths.get("gate_summary_csv")
        gate_rows_path = peer_detail_paths.get("gate_row_hits_csv")
        distribution_html_path = peer_detail_paths.get("distribution_html")
        detail_leaf_gate_tables_payload = build_detail_leaf_gate_tables_payload(
            gate_summary_csv=gate_summary_path,
            gate_row_hits_csv=gate_rows_path,
        )
        dashboard_diagnostics_meta = {
            **dashboard_diagnostics_meta,
            "detail_leaf_gate_tables_enabled": bool(detail_leaf_gate_tables_payload.get("enabled", False)),
            "gate_summary_csv": str(gate_summary_path) if gate_summary_path is not None else "",
            "gate_row_hits_csv": str(gate_rows_path) if gate_rows_path is not None else "",
            "distribution_html": str(distribution_html_path) if distribution_html_path is not None else "",
            "detail_leaf_gate_tables_reason": str(detail_leaf_gate_tables_payload.get("reason", "")),
        }
    if dashboard_diagnostics_path is not None:
        dashboard_diagnostics_meta = {
            **dashboard_diagnostics_meta,
            "path": str(dashboard_diagnostics_path),
        }

    score_status: dict[str, Any] = {"enabled": True, "status": "pending", "embedding_cache": {}}
    detail_leaf_triplet_payload: dict[str, Any] = {
        "enabled": False,
        "rows": [],
        "reason": "integrated_mode",
    }
    cache_meta_path, cache_note = _resolve_embedding_cache_meta_path(
        args=args,
        row_csv=row_csv.resolve(),
        out_dir=out_dir.resolve(),
        tag=tag,
        stem=stem,
    )
    cache_paths = resolve_embedding_cache_paths(
        output_dir=out_dir.resolve(),
        tag=tag,
        stem=stem,
        meta_json_path=cache_meta_path,
    )
    cache_load_note = dict(cache_note)
    cache_load_note["meta_json"] = str(cache_paths.meta_json_path.resolve())

    loaded_cache = None
    cache_error = None
    try:
        loaded_cache = load_embedding_cache(paths=cache_paths)
        if int(loaded_cache.input_norm.shape[0]) != len(updated_row_df):
            raise ValueError(
                f"cache row mismatch: cache={loaded_cache.input_norm.shape[0]}, rows={len(updated_row_df)}"
            )
        cache_load_note["status"] = "loaded"
        cache_load_note["source"] = "existing"
    except Exception as exc:
        cache_error = str(exc)
        if bool(args.rebuild_embedding_cache):
            embedder = _build_embedder_module(
                backend=str(args.embedding_backend),
                embedding_model=str(args.embedding_model),
                hash_dim=768,
            )
            loaded_cache = load_or_rebuild_embedding_cache(
                paths=build_embedding_cache_paths(output_dir=out_dir.resolve(), tag=tag, stem=stem),
                expected_rows=len(updated_row_df),
                input_texts=updated_row_df["source_input"].fillna("").astype(str).tolist(),
                output_texts=updated_row_df["source_output"].fillna("").astype(str).tolist(),
                embedder=embedder,
                batch_size=int(args.embedding_batch_size),
                allow_rebuild=True,
            )
            cache_load_note["status"] = "rebuilt"
            cache_load_note["source"] = "auto_rebuild"
        else:
            cache_load_note["status"] = "missing"
            cache_load_note["source"] = "none"

    input_norm = None
    output_norm = None
    if loaded_cache is not None:
        cache_load_note["meta_json"] = str(loaded_cache.paths.meta_json_path)
        input_norm = np.asarray(loaded_cache.input_norm, dtype=float)
        output_norm = np.asarray(loaded_cache.output_norm, dtype=float)
        cache_load_note["input_norm_path"] = str(loaded_cache.paths.input_norm_path)
        cache_load_note["output_norm_path"] = str(loaded_cache.paths.output_norm_path)
        cache_load_note["valid_rows"] = int(loaded_cache.meta.get("valid_rows", len(updated_row_df)))
    elif cache_error is not None:
        cache_load_note["error"] = cache_error

    artifacts = compute_bundle_scores(
        row_df=updated_row_df,
        threshold_summary_df=threshold_summary_df,
        score_runtime=SCORE_RUNTIME,
        input_norm=input_norm,
        output_norm=output_norm,
        embedding_meta=loaded_cache.meta if loaded_cache is not None else None,
    )
    write_csv(artifacts.summary_df, score_summary_out, index=False, float_format="%.4f")
    write_csv(artifacts.detail_df, score_detail_out, index=False, float_format="%.4f")
    warn_artifacts = compute_warn_inspect(
        row_df=updated_row_df,
        threshold_summary_df=threshold_summary_df,
        score_runtime=SCORE_RUNTIME,
        bundle_payload=artifacts.payload,
    )
    write_csv(warn_artifacts.rows_df, warn_inspect_csv_out, index=False, float_format="%.6f")
    write_json(warn_artifacts.payload, warn_inspect_json_out)
    if str(args.inspection_mode) == "detailed":
        detail_leaf_triplet_payload = build_detail_leaf_triplet_payload(
            row_df=updated_row_df,
            threshold_summary_df=threshold_summary_df,
            score_runtime=SCORE_RUNTIME,
            detail_leaf_gate_tables_payload=detail_leaf_gate_tables_payload,
            input_norm=input_norm,
            output_norm=output_norm,
            embedding_meta=loaded_cache.meta if loaded_cache is not None else None,
            max_leaves=8,
        )
    edge_input_aux_payload = build_edge_input_aux_payload(
        row_df=updated_row_df,
        top_k=int(getattr(SCORE_RUNTIME, "new_score_risk_case_top_k", 20)),
        preview_chars=int(getattr(SCORE_RUNTIME, "new_score_risk_case_preview_chars", 180)),
    )
    merged_payload = dict(artifacts.payload)
    merged_payload["warn_inspect"] = warn_artifacts.payload
    merged_payload["detail_leaf_gate_tables"] = detail_leaf_gate_tables_payload
    merged_payload["detail_leaf_triplet_interp"] = detail_leaf_triplet_payload
    merged_payload["edge_input_aux"] = edge_input_aux_payload
    build_bundle_ab_interpretation_csv(
        tag=tag,
        stem=stem,
        summary_df=artifacts.summary_df,
        detail_df=artifacts.detail_df,
        payload=merged_payload,
        out_csv=score_ab_interpret_out,
    )
    write_json(merged_payload, score_json_out)
    render_bundle_score_dashboard(
        output_html=score_dashboard_out,
        summary_df=artifacts.summary_df,
        detail_df=artifacts.detail_df,
        payload=merged_payload,
        diagnostics_html_path=dashboard_diagnostics_path,
    )
    score_status = {
        "enabled": True,
        "status": "ready",
        "output_summary_csv": str(score_summary_out),
        "output_detail_csv": str(score_detail_out),
        "output_ab_interpretation_csv": str(score_ab_interpret_out),
        "output_json": str(score_json_out),
        "output_dashboard_html": str(score_dashboard_out),
        "warn_inspect": {
            "enabled": True,
            "status": "ready",
            "output_json": str(warn_inspect_json_out),
            "output_csv": str(warn_inspect_csv_out),
            "summary": warn_artifacts.summary,
        },
        "detail_leaf_gate_tables": {
            "enabled": bool(detail_leaf_gate_tables_payload.get("enabled", False)),
            "reason": str(detail_leaf_gate_tables_payload.get("reason", "")),
            "leaf_count": int(detail_leaf_gate_tables_payload.get("leaf_count", 0) or 0),
            "row_hits": int(detail_leaf_gate_tables_payload.get("row_hits", 0) or 0),
        },
        "detail_leaf_triplet_interp": {
            "enabled": bool(detail_leaf_triplet_payload.get("enabled", False)),
            "reason": str(detail_leaf_triplet_payload.get("reason", "")),
            "leaf_count": int(detail_leaf_triplet_payload.get("leaf_count", 0) or 0),
        },
        "edge_input_aux": {
            "enabled": bool(edge_input_aux_payload.get("enabled", False)),
            "reason": str(edge_input_aux_payload.get("reason", "")),
            "base_rows": int(edge_input_aux_payload.get("base_rows", 0) or 0),
            "hard_rows": int(edge_input_aux_payload.get("hard_rows", 0) or 0),
            "hard_rate_base": float(edge_input_aux_payload.get("hard_rate_base", 0.0) or 0.0),
        },
        "dashboard_diagnostics": dashboard_diagnostics_meta,
        "embedding_cache": cache_load_note,
        "conf_overall": int(merged_payload.get("conf_overall", 0)),
    }

    cfg["score"] = score_status
    cfg["runtime_profile"] = {
        "distribution_mode": "nomask_only",
        "plot_export": "always_on",
        "score_export": "always_on",
    }
    write_json(cfg, cfg_out)

    print(f"[DONE] row_results: {row_out}")
    print(f"[DONE] summary: {summary_out}")
    print(f"[DONE] run_config: {cfg_out}")
    print(f"[DONE] rule_thresholds: {thr_rule_out}")
    print(f"[DONE] rule_thresholds_compact: {thr_compact_out}")
    print(f"[DONE] thresholds_summary: {thr_summary_out}")
    print(f"[DONE] bundle_scores_summary: {score_summary_out}")
    print(f"[DONE] bundle_scores_detail: {score_detail_out}")
    print(f"[DONE] bundle_scores_ab_interpretation: {score_ab_interpret_out}")
    print(f"[DONE] bundle_scores_json: {score_json_out}")
    print(f"[DONE] bundle_scores_dashboard: {score_dashboard_out}")
    print(f"[DONE] warn_inspect_json: {warn_inspect_json_out}")
    print(f"[DONE] warn_inspect_csv: {warn_inspect_csv_out}")

    if go is None or pio is None:
        print("[WARN] plotly is not installed; generating fallback diagnostics HTML.", file=sys.stderr)
        render_distribution_diagnostics_plot_fallback(
            source_df=source_df,
            row_df=updated_row_df,
            summary_df=threshold_summary_df,
            rules=rules,
            args=args,
            out_html=diagnostics_html_out,
            out_summary_csv=diagnostics_summary_out,
            detail_paths=plot_detail_paths,
        )
    else:
        render_distribution_diagnostics_plot(
            source_df=source_df,
            row_df=updated_row_df,
            summary_df=threshold_summary_df,
            rules=rules,
            args=args,
            out_html=diagnostics_html_out,
            out_summary_csv=diagnostics_summary_out,
            detail_paths=plot_detail_paths,
        )
    print(f"[DONE] diagnostics_html: {diagnostics_html_out}")
    print(f"[DONE] diagnostics_summary: {diagnostics_summary_out}")

    return RunArtifacts(
        run_root=out_dir.resolve(),
        report_dir=report_dir.resolve(),
        row_results_csv=row_out.resolve(),
        summary_csv=summary_out.resolve(),
        run_config_json=cfg_out.resolve(),
        diagnostics_html=diagnostics_html_out.resolve(),
        diagnostics_summary_csv=diagnostics_summary_out.resolve(),
    )


def run(config: FinalMetricConfig) -> RunArtifacts:
    """Primary API entrypoint for final_metric_refactor.

    Args:
        config: FinalMetricConfig instance with all run parameters.

    Returns:
        RunArtifacts with paths to generated output files.
    """
    args = _config_to_namespace(config)
    return _run_with_args(args)


def main() -> RunArtifacts:
    """CLI entrypoint - parse args and run final_metric."""
    config = parse_args()
    return run(config)


if __name__ == "__main__":
    main()
