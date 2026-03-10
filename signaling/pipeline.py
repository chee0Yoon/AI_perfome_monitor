#!/usr/bin/env python3
"""
Distribution-based anomaly pipeline for input->output diffs.

What this script does:
1) Dynamic data loading and hard gates (IFEval / schema / text-length)
2) JSON leaf-value extraction for bundle text construction (distribution runtime only)
3) Distribution-based anomaly signals (no clustering):
   - output density outlier
   - input-conditioned direction/length/diff residual outlier
   - similar-input conflict signal
4) Final AND-gate pass/fail export
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import umap.umap_ as umap_module
except Exception:
    umap_module = None

try:
    from sklearn.covariance import LedoitWolf as _LedoitWolf
except ImportError:
    _LedoitWolf = None


os_env = {"TOKENIZERS_PARALLELISM": "false"}
for k, v in os_env.items():
    try:
        os.environ[k] = v
    except Exception:
        pass


ROOT = Path(__file__).resolve().parent

from final_metric_refactor.hard_gate.ifeval import IFEvalMetric
from final_metric_refactor.hard_gate.schema_gate import JSONSchemaGate
from final_metric_refactor.hard_gate.textlen_gate import TextLengthGate
from final_metric_refactor.signaling.scorer import DistributionScorer
from final_metric_refactor.shared.preprocessor import flatten_json_leaves, safe_json_load
from final_metric_refactor.embedding.embedder import TextEmbedder, build_embedder as _build_embedder_module
from final_metric_refactor.embedding.cache import EmbeddingCacheWriter, build_embedding_cache_paths
from final_metric_refactor.shared.geometry import (
    ensure_2d_coords,
    knn_self,
    normalize_rows,
    pca_fit_transform,
    robust_z,
    sanitize_matrix,
)
from final_metric_refactor.distribution.output_density import OutputDensityMetric
from final_metric_refactor.distribution._shared import weighted_quantile
from final_metric_refactor.threshold import (
    apply_mode_quantiles,
    apply_rule_tristate,
    calibrate_mode_quantiles,
    calibrate_rule_tristate,
    evaluate_rule_states,
    flatten_calibration_json,
    parse_csv_tokens,
    parse_quantile_csv_spec,
    parse_quantile_range_spec,
    validate_calibration_rules,
)
from final_metric_refactor.config import (
    DEFAULT_IFEVAL_IDS,
    DEFAULT_IFEVAL_KWARGS,
    DEFAULT_SCHEMA,
    DEFAULT_TRY_ENCODINGS,
    DISTANCE_CALIBRATION_RUNTIME,
    DISTRIBUTION_GATE_RUNTIME,
    DISTRIBUTION_SIGNAL_RUNTIME,
    PIPELINE_RULE_KEYS,
    PIPELINE_RULE_PASS_PREFIX,
    PIPELINE_RULE_SIGNAL_PREFIX,
    SEMANTIC_SIGNAL_RUNTIME,
    TEMPLATE_RUNTIME,
    TRISTATE_RUNTIME,
    UDF_RUNTIME,
    normalize_pipeline_rule_key,
    parse_pipeline_rule_keys,
)
from final_metric_refactor.config.data_paths import default_ambiguous_csv

TEMPLATE_BUNDLE_CONFIG = {
    "n_min": int(TEMPLATE_RUNTIME.n_min),
    "n_max": int(TEMPLATE_RUNTIME.n_max),
    "freq_threshold": float(TEMPLATE_RUNTIME.freq_threshold),
    "position_std": float(TEMPLATE_RUNTIME.position_std),
    "coverage_threshold": float(TEMPLATE_RUNTIME.coverage_threshold),
    "mask_token": str(TEMPLATE_RUNTIME.mask_token),
}

EPS = 1e-9
RULE_KEYS = list(PIPELINE_RULE_KEYS)
RULE_SIGNAL_PREFIX = dict(PIPELINE_RULE_SIGNAL_PREFIX)
RULE_PASS_PREFIX = dict(PIPELINE_RULE_PASS_PREFIX)

# UDF is fixed to one-pass mode for production runtime stability.
UDF_FIXED_ENABLED = UDF_RUNTIME.enabled
UDF_FIXED_ITERATIONS = UDF_RUNTIME.iterations
UDF_FIXED_CORE_RULES = list(UDF_RUNTIME.core_rules)
UDF_FIXED_Q_CLEAN = UDF_RUNTIME.q_clean
UDF_FIXED_SOFT_ALPHA = UDF_RUNTIME.soft_alpha
UDF_FIXED_MIN_WEIGHT = UDF_RUNTIME.min_weight
SIGNAL_RUNTIME = DISTRIBUTION_SIGNAL_RUNTIME
GATE_RUNTIME = DISTRIBUTION_GATE_RUNTIME
SEMANTIC_RUNTIME = SEMANTIC_SIGNAL_RUNTIME
CALIBRATION_RUNTIME = DISTANCE_CALIBRATION_RUNTIME
TRISTATE_CFG = TRISTATE_RUNTIME


@dataclass(frozen=True)
class DistributionRunArtifacts:
    output_dir: Path
    report_dir: Path
    row_results_csv: Path
    summary_csv: Path
    run_config_json: Path

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Distribution-based anomaly detection for input->output embedding diffs."
    )
    parser.add_argument(
        "--csv-path",
        default=str(default_ambiguous_csv()),
        help="Path to input CSV (default: shared /data ambiguous benchmark).",
    )
    parser.add_argument("--encoding", default=None, help="Optional CSV encoding")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row cap for quick runs")
    parser.add_argument(
        "--auto-max-rows",
        type=int,
        default=8000,
        help="Automatic row cap when max-rows is not set (for tractable distribution signaling).",
    )
    parser.add_argument("--sample-seed", type=int, default=42, help="Seed for row sampling")

    parser.add_argument("--id-col", default="id", help="ID column name")
    parser.add_argument("--prompt-col", default="Prompt", help="Prompt column name for IFEval")
    parser.add_argument("--input-col", default="input", help="Input column name")
    parser.add_argument("--output-col", default="expectedOutput", help="Output column name")
    parser.add_argument(
        "--label-col",
        default="eval",
        help="Optional label column used for plot coloring (correct/incorrect).",
    )
    parser.add_argument(
        "--label-correct-values",
        default="correct,good,pass,true,1",
        help="Comma-separated values interpreted as correct labels (case-insensitive).",
    )
    parser.add_argument(
        "--label-incorrect-values",
        default="incorrect,bad,fail,false,0",
        help="Comma-separated values interpreted as incorrect labels (case-insensitive).",
    )
    parser.add_argument(
        "--use-label-coloring",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When label-col exists, color input/output/diff plots by correct/incorrect label.",
    )
    parser.add_argument(
        "--group-by-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run template/diff distribution signaling separately per prompt.",
    )
    parser.add_argument(
        "--group-by-col",
        default=None,
        help="Optional explicit grouping column (overrides --group-by-prompt).",
    )
    parser.add_argument(
        "--inspection-mode",
        default="integrated",
        choices=["integrated", "detailed"],
        help="Inspection mode. integrated keeps bundled checks; detailed adds per-leaf output checks.",
    )
    parser.add_argument(
        "--detail-leaf-min-support",
        type=int,
        default=30,
        help="Detailed mode: minimum non-empty support rows per output leaf path in each group.",
    )
    parser.add_argument(
        "--detail-output-quantile",
        type=float,
        default=0.995,
        help="Detailed mode: quantile for per-leaf output-density fail threshold.",
    )
    parser.add_argument(
        "--detail-leaf-dist-bins",
        type=int,
        default=20,
        help="Detailed mode: histogram bins for per-leaf signal distribution artifacts.",
    )
    parser.add_argument(
        "--ifeval-ids",
        default=",".join(DEFAULT_IFEVAL_IDS),
        help="Comma-separated IFEval instruction ids",
    )
    parser.add_argument(
        "--ifeval-kwargs-json",
        default=json.dumps(DEFAULT_IFEVAL_KWARGS),
        help="JSON dict: instruction_id -> kwargs",
    )
    parser.add_argument(
        "--ifeval-mode",
        default="strict",
        choices=["strict", "loose", "both"],
        help="IFEval mode",
    )

    parser.add_argument("--schema-path", default=None, help="Optional schema JSON file path")
    parser.add_argument("--schema-json", default=None, help="Optional inline schema JSON")
    parser.add_argument(
        "--schema-strict-keys",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require all schema keys",
    )
    parser.add_argument(
        "--schema-allow-extra-keys",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow extra keys not in schema",
    )
    parser.add_argument(
        "--schema-infer-min-key-ratio",
        type=float,
        default=0.90,
        help="Key inclusion ratio for inferred schema",
    )
    parser.add_argument(
        "--schema-infer-max-enum",
        type=int,
        default=8,
        help="Max enum size when inferring schema",
    )

    parser.add_argument(
        "--textlen-min-ratio",
        type=float,
        default=0.30,
        help="Per-path threshold = median_length * ratio",
    )
    parser.add_argument(
        "--textlen-min-support-ratio",
        type=float,
        default=0.20,
        help="Run text-length check only on paths appearing in >= this ratio",
    )

    parser.add_argument("--embedding-model", default="google/embeddinggemma-300m")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument(
        "--embedding-backend",
        default="auto",
        choices=["auto", "sentence-transformers", "hash"],
        help="Embedding backend. auto tries sentence-transformers then falls back to hash.",
    )
    parser.add_argument(
        "--hash-dim",
        type=int,
        default=768,
        help="Vector dimension for hash backend",
    )
    parser.add_argument(
        "--diff-residual-aux-enabled",
        action=argparse.BooleanOptionalAction,
        default=bool(SIGNAL_RUNTIME.diff_residual_aux_enabled),
        help="Enable diff_residual auxiliary residual boost from direction/length.",
    )
    parser.add_argument(
        "--diff-residual-aux-lambda",
        type=float,
        default=float(SIGNAL_RUNTIME.diff_residual_aux_lambda),
        help="Weight for positive auxiliary residual boost.",
    )
    parser.add_argument(
        "--diff-residual-aux-model",
        default=str(SIGNAL_RUNTIME.diff_residual_aux_model),
        choices=["linear", "poly2"],
        help="Model for expected diff residual from direction/length.",
    )

    parser.add_argument("--top-n-anomalies", type=int, default=30, help="Top-N rows exported per signal")
    parser.add_argument("--plot-max-arrows", type=int, default=1000, help="Max arrows drawn in plot")
    parser.add_argument("--plot-seed", type=int, default=42, help="Random seed for plot subsampling")
    parser.add_argument("--hist-bins", type=int, default=60, help="Histogram bins for distribution report")

    parser.add_argument("--output-dir", default=str(ROOT / "results"), help="Directory for exported CSVs")
    parser.add_argument("--report-dir-name", default="report", help="Sub-directory under output-dir for report artifacts.")
    parser.add_argument("--tag", default="dist_gate", help="Output filename tag")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def run_distribution_pipeline(
    config: dict[str, Any] | None = None,
    **overrides: Any,
) -> DistributionRunArtifacts:
    """Run distribution pipeline via Python API without subprocess."""
    parser = build_parser()
    defaults = vars(parser.parse_args([]))
    merged: dict[str, Any] = dict(defaults)
    if config:
        merged.update(dict(config))
    if overrides:
        merged.update(dict(overrides))
    args = argparse.Namespace(**merged)
    return _run_with_args(args)


def read_csv_dynamic(path: str, encoding: str | None) -> tuple[pd.DataFrame, str]:
    if encoding:
        return pd.read_csv(path, encoding=encoding), encoding
    last_err = None
    for enc in DEFAULT_TRY_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc), enc
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"Could not read CSV with tried encodings {DEFAULT_TRY_ENCODINGS}: {last_err}")


def parse_json_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    def _safe_parse(val: Any) -> dict[str, Any]:
        if pd.isna(val):
            return {"_parse_error": "NaN"}
        try:
            parsed = json.loads(str(val))
            if isinstance(parsed, dict):
                return parsed
            return {"_parse_error": f"non_dict:{type(parsed).__name__}"}
        except Exception:
            return {"_parse_error": str(val)[:120]}

    parsed = df[column].apply(_safe_parse)
    return pd.json_normalize(parsed)


def parse_label_value_set(raw_values: str) -> set[str]:
    return {x.strip().lower() for x in str(raw_values).split(",") if x.strip()}


def map_label_correctness(
    values: list[str],
    correct_values: set[str],
    incorrect_values: set[str],
) -> np.ndarray:
    """
    Map raw label strings to:
    - 1.0: correct
    - 0.0: incorrect
    - NaN: unknown
    """
    out = np.full(len(values), np.nan, dtype=float)
    for i, v in enumerate(values):
        key = str(v).strip().lower()
        if key in correct_values:
            out[i] = 1.0
        elif key in incorrect_values:
            out[i] = 0.0
    return out


def resolve_rule_quantiles() -> dict[str, float]:
    qmap = {k: float(SIGNAL_RUNTIME.signal_quantile) for k in RULE_KEYS}
    qmap["discourse_instability"] = float(SEMANTIC_RUNTIME.discourse_instability_quantile)
    qmap["contradiction"] = float(SEMANTIC_RUNTIME.contradiction_quantile)
    qmap["self_contradiction"] = float(SEMANTIC_RUNTIME.self_contradiction_quantile)
    if GATE_RUNTIME.rule_quantiles_json:
        raw = json.loads(GATE_RUNTIME.rule_quantiles_json)
        if not isinstance(raw, dict):
            raise ValueError("DISTRIBUTION_GATE_RUNTIME.rule_quantiles_json must be a JSON object.")
        for k, v in raw.items():
            nk = normalize_pipeline_rule_key(k)
            if nk not in RULE_KEYS:
                raise ValueError(f"Unknown rule key in DISTRIBUTION_GATE_RUNTIME.rule_quantiles_json: {k}")
            q = float(v)
            if not (0.0 < q < 1.0):
                raise ValueError(f"Quantile must be in (0,1): {k}={v}")
            qmap[nk] = q
    return qmap


def compute_thresholds(
    signal_map: dict[str, np.ndarray],
    ref_mask: np.ndarray,
    rule_quantiles: dict[str, float],
    ref_weights: np.ndarray | None = None,
    signal_available_map: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    weights = None
    if ref_weights is not None:
        weights = np.asarray(ref_weights, dtype=float)
        if len(weights) != len(next(iter(signal_map.values()))):
            weights = None
    for key in RULE_KEYS:
        q = float(rule_quantiles.get(key, 0.99))
        vals = np.asarray(signal_map[key], dtype=float)
        if len(vals) == 0:
            thresholds[key] = float("nan")
            continue

        finite = np.isfinite(vals)
        base_ref = np.asarray(ref_mask, dtype=bool).copy()
        if len(base_ref) != len(vals):
            base_ref = np.ones(len(vals), dtype=bool)
        mask = base_ref & finite

        if signal_available_map is not None and key in signal_available_map:
            avail = np.asarray(signal_available_map[key], dtype=bool)
            if len(avail) == len(vals):
                mask = base_ref & avail & finite
                if not mask.any():
                    # Semantic-style unavailable case: no eligible rows for this rule.
                    avail_any = avail & finite
                    if avail_any.any():
                        # If ref rows miss eligibility, fallback to all available rows.
                        mask = avail_any
                    else:
                        thresholds[key] = float("nan")
                        continue
        if not mask.any():
            mask = finite
        if not mask.any():
            thresholds[key] = float("nan")
            continue

        if weights is not None:
            th = weighted_quantile(vals[mask], weights[mask], q)
        else:
            th = weighted_quantile(vals[mask], None, q)
        if not np.isfinite(th):
            th = weighted_quantile(vals[mask], None, q)
        thresholds[key] = float(th) if np.isfinite(th) else float("nan")
    return thresholds


def refine_reference_mask(
    signal_map: dict[str, np.ndarray],
    initial_ref_mask: np.ndarray,
    rule_quantiles: dict[str, float],
    refine_rules: list[str],
    iterations: int,
    min_size: int,
    signal_available_map: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, float], int]:
    ref_mask = np.asarray(initial_ref_mask, dtype=bool).copy()
    if ref_mask.sum() == 0:
        ref_mask = np.ones(len(next(iter(signal_map.values()))), dtype=bool)
    thresholds = compute_thresholds(
        signal_map=signal_map,
        ref_mask=ref_mask,
        rule_quantiles=rule_quantiles,
        signal_available_map=signal_available_map,
    )

    applied_rounds = 0
    iters = max(0, int(iterations))
    min_size = max(5, int(min_size))

    for _ in range(iters):
        if ref_mask.sum() < min_size:
            break
        keep = np.ones(len(ref_mask), dtype=bool)
        for key in refine_rules:
            pass_mask = signal_map[key] <= thresholds[key]
            if signal_available_map is not None and key in signal_available_map:
                avail = np.asarray(signal_available_map[key], dtype=bool)
                if len(avail) == len(pass_mask):
                    pass_mask = (~avail) | pass_mask
            keep &= pass_mask
        new_ref_mask = ref_mask & keep
        if new_ref_mask.sum() < min_size:
            break
        if np.array_equal(new_ref_mask, ref_mask):
            break
        ref_mask = new_ref_mask
        thresholds = compute_thresholds(
            signal_map=signal_map,
            ref_mask=ref_mask,
            rule_quantiles=rule_quantiles,
            signal_available_map=signal_available_map,
        )
        applied_rounds += 1

    return ref_mask, thresholds, applied_rounds


def build_embedder(args: argparse.Namespace):
    """Build embedder from argparse namespace (wrapper for processor.embedder.build_embedder)."""
    return _build_embedder_module(
        backend=args.embedding_backend,
        embedding_model=args.embedding_model,
        hash_dim=args.hash_dim,
    )


def run_ifeval(
    df: pd.DataFrame,
    prompt_col: str,
    output_col: str,
    instruction_ids: list[str],
    kwargs_dict: dict[str, dict[str, Any]],
    eval_mode: str,
) -> pd.DataFrame:
    kwargs_list = [kwargs_dict.get(inst_id, {}) for inst_id in instruction_ids]
    metric = IFEvalMetric(
        instruction_id_list=instruction_ids,
        kwargs_list=kwargs_list,
        eval_mode=eval_mode,
    )

    prompt_exists = prompt_col in df.columns
    results: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        system_prompt = str(row[prompt_col]) if prompt_exists and pd.notna(row[prompt_col]) else ""
        response = str(row[output_col]) if pd.notna(row[output_col]) else ""
        result = metric.evaluate_single(
            system_prompt=system_prompt,
            user_prompt="",
            response=response,
        )
        failed = [r["instruction_id"] for r in result.instruction_results if not r["strict_passed"]]
        results.append(
            {
                "ifeval_pass": bool(result.prompt_level_strict_acc),
                "ifeval_failed": failed if failed else None,
            }
        )
    return pd.DataFrame(results)


def infer_schema_from_outputs(
    outputs: list[str],
    min_key_ratio: float = 0.90,
    max_enum: int = 8,
    min_parse_ratio: float = 0.30,
) -> dict[str, Any] | None:
    parsed = [safe_json_load(x) for x in outputs]
    dict_rows = [x for x in parsed if isinstance(x, dict)]
    if not dict_rows:
        return None
    if len(dict_rows) / max(len(outputs), 1) < min_parse_ratio:
        return None
    return infer_schema_node(dict_rows, min_key_ratio=min_key_ratio, max_enum=max_enum, depth=0)


def infer_schema_node(
    rows: list[dict[str, Any]],
    min_key_ratio: float,
    max_enum: int,
    depth: int,
    max_depth: int = 2,
) -> dict[str, Any]:
    schema: dict[str, Any] = {}
    n = len(rows)
    key_counts = Counter()
    for row in rows:
        key_counts.update(row.keys())

    required_keys = [k for k, c in key_counts.items() if c / max(n, 1) >= min_key_ratio]
    for key in sorted(required_keys):
        vals = [row.get(key) for row in rows if key in row and row.get(key) is not None]
        if not vals:
            schema[key] = "text"
            continue

        dict_ratio = np.mean([isinstance(v, dict) for v in vals])
        if dict_ratio >= 0.7 and depth < max_depth:
            nested = [v for v in vals if isinstance(v, dict)]
            nested_schema = infer_schema_node(nested, min_key_ratio=min_key_ratio, max_enum=max_enum, depth=depth + 1)
            schema[key] = nested_schema if nested_schema else "text"
            continue

        flat_vals = [str(v).strip() for v in vals if str(v).strip()]
        uniq = sorted(set(flat_vals))
        uniq_ratio = len(uniq) / max(len(flat_vals), 1)
        if 1 < len(uniq) <= max_enum and uniq_ratio <= 0.25:
            schema[key] = uniq
        elif len(uniq) == 1 and len(flat_vals) >= 3:
            schema[key] = uniq
        else:
            schema[key] = "text"
    return schema


def run_schema_gate(
    df: pd.DataFrame,
    output_col: str,
    schema: dict[str, Any],
    strict_keys: bool,
    allow_extra_keys: bool,
) -> tuple[Any, pd.DataFrame]:
    gate = JSONSchemaGate(schema=schema, strict_keys=strict_keys, allow_extra_keys=allow_extra_keys)
    outputs = df[output_col].fillna("").astype(str).tolist()
    result = gate.validate(outputs)
    per_output = []
    for r in result.per_output_results:
        per_output.append(
            {
                "index": r["index"],
                "schema_pass": r["passed"],
                "schema_missing_keys": r["missing_keys"],
                "schema_error": r["error"],
            }
        )
    return result, pd.DataFrame(per_output)




def extract_ngrams(text: str, n: int = 3) -> list[str]:
    words = re.findall(r"\b\w+\b", text.lower())
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def detect_template_candidates(texts: list[str], n_min: int, n_max: int, freq_threshold: float) -> dict[int, set[str]]:
    if len(texts) < 3:
        return {}
    min_count = max(2, int(len(texts) * freq_threshold))
    candidates: dict[int, set[str]] = {}
    for n in range(n_min, n_max + 1):
        ngram_counts: Counter[str] = Counter()
        for text in texts:
            for ng in set(extract_ngrams(text, n)):
                ngram_counts[ng] += 1
        freq = {ng for ng, c in ngram_counts.items() if c >= min_count}
        if freq:
            candidates[n] = freq
    return candidates


def validate_position_alignment(
    texts: list[str], candidates: dict[int, set[str]], position_std: float
) -> dict[int, set[str]]:
    if not candidates:
        return {}
    ngram_positions: dict[tuple[int, str], list[float]] = {}
    for text in texts:
        words = re.findall(r"\b\w+\b", text.lower())
        n_words = len(words)
        if n_words == 0:
            continue
        seen: set[tuple[int, str]] = set()
        for n, ng_set in candidates.items():
            if n > n_words:
                continue
            for i in range(n_words - n + 1):
                ng = " ".join(words[i : i + n])
                key = (n, ng)
                if ng in ng_set and key not in seen:
                    seen.add(key)
                    ngram_positions.setdefault(key, []).append(i / max(n_words, 1))
    structural: dict[int, set[str]] = {}
    for (n, ng), positions in ngram_positions.items():
        if len(positions) >= 2 and np.std(positions) < position_std:
            structural.setdefault(n, set()).add(ng)
    return structural


def compute_template_coverage(text: str, structural_templates: dict[int, set[str]]) -> float:
    words = re.findall(r"\b\w+\b", text.lower())
    n_words = len(words)
    if n_words == 0:
        return 0.0
    covered: set[int] = set()
    for n, ng_set in structural_templates.items():
        if n > n_words:
            continue
        for i in range(n_words - n + 1):
            ng = " ".join(words[i : i + n])
            if ng in ng_set:
                covered.update(range(i, i + n))
    return len(covered) / max(n_words, 1)


def mask_structural_spans(text: str, structural_templates: dict[int, set[str]], mask: str) -> str:
    word_spans = [(m.start(), m.end()) for m in re.finditer(r"\b\w+\b", text)]
    if not word_spans or not structural_templates:
        return text
    words_lc = [text[s:e].lower() for s, e in word_spans]
    n_words = len(words_lc)
    matches: list[tuple[int, int, int]] = []
    for n, ng_set in structural_templates.items():
        if n > n_words:
            continue
        for i in range(n_words - n + 1):
            ng = " ".join(words_lc[i : i + n])
            if ng in ng_set:
                matches.append((i, i + n - 1, n))
    if not matches:
        return text
    matches.sort(key=lambda x: (-x[2], x[0]))
    occupied: set[int] = set()
    selected: list[tuple[int, int]] = []
    for s, e, _ in matches:
        span = set(range(s, e + 1))
        if not span & occupied:
            selected.append((s, e))
            occupied |= span
    selected.sort()
    parts = []
    prev_end = 0
    for ws, we in selected:
        c_start = word_spans[ws][0]
        c_end = word_spans[we][1]
        parts.append(text[prev_end:c_start])
        parts.append(mask)
        prev_end = c_end
    parts.append(text[prev_end:])
    return "".join(parts)


def collapse_mask_token(text: str, mask: str) -> str:
    count = text.count(mask)
    if count <= 1:
        return text
    first_pos = text.find(mask)
    head = text[: first_pos + len(mask)]
    tail = text[first_pos + len(mask) :].replace(mask, " ")
    collapsed = head + tail
    collapsed = re.sub(r"\s+", " ", collapsed).strip()
    collapsed = re.sub(rf"\s*{re.escape(mask)}\s*", f" {mask} ", collapsed).strip()
    collapsed = re.sub(r"\s+([,.;:!?])", r"\1", collapsed)
    return collapsed


def run_template_detection(
    texts: list[str],
    n_min: int,
    n_max: int,
    freq_threshold: float,
    position_std: float,
    coverage_threshold: float,
    mask_token: str,
) -> tuple[dict[int, set[str]], pd.DataFrame]:
    candidates = detect_template_candidates(texts, n_min=n_min, n_max=n_max, freq_threshold=freq_threshold)
    structural = validate_position_alignment(texts, candidates, position_std=position_std)

    rows: list[dict[str, Any]] = []
    for text in texts:
        coverage = compute_template_coverage(text, structural)
        is_template_doc = coverage >= coverage_threshold
        if structural:
            masked = mask_structural_spans(text, structural, mask=mask_token)
            masked = collapse_mask_token(masked, mask=mask_token)
            if not masked.strip():
                masked = mask_token
        else:
            masked = text
        rows.append(
            {
                "original": text,
                "masked": masked,
                "coverage": round(coverage, 4),
                "is_template_doc": bool(is_template_doc),
            }
        )
    return structural, pd.DataFrame(rows)


def build_column_bundles(
    series: pd.Series,
    n_min: int,
    n_max: int,
    freq_threshold: float,
    position_std: float,
    coverage_threshold: float,
    mask_token: str,
) -> tuple[list[str], list[str], pd.DataFrame]:
    """
    Parse each row -> JSON leaf values.
    Bundle row text by concatenating all leaf values.
    """
    row_leaf_maps: list[dict[str, str]] = []
    for value in series.fillna("").tolist():
        parsed = safe_json_load(value)
        if isinstance(parsed, dict):
            leaves = flatten_json_leaves(parsed)
            leaf_map = {k: str(v) for k, v in leaves.items() if str(v).strip()}
            row_leaf_maps.append(leaf_map or {"_raw_text": str(value)})
        else:
            row_leaf_maps.append({"_raw_text": str(value)})

    n = len(series)
    all_paths = sorted({p for m in row_leaf_maps for p in m.keys()})
    path_stats_rows: list[dict[str, Any]] = []
    raw_bundles: list[str] = []
    for i in range(n):
        raw_values = [row_leaf_maps[i].get(p, "") for p in all_paths]
        raw_bundle = " || ".join([v for v in raw_values if v.strip()]).strip()
        raw_bundles.append(raw_bundle)
    for path in all_paths:
        non_empty = sum(1 for m in row_leaf_maps if str(m.get(path, "")).strip())
        path_stats_rows.append(
            {
                "path": path,
                "num_rows": n,
                "non_empty": non_empty,
                "num_structural_ngrams": 0,
                "template_doc_count": 0,
                "coverage_mean": 0.0,
            }
        )

    # Single distribution runtime: keep return signature for compatibility.
    return raw_bundles, list(raw_bundles), pd.DataFrame(path_stats_rows)


def resolve_distribution_groups(
    df: pd.DataFrame,
    prompt_col: str,
    group_by_prompt: bool,
    group_by_col: str | None,
) -> tuple[pd.Series, pd.DataFrame, str]:
    if group_by_col:
        if group_by_col not in df.columns:
            raise ValueError(f"--group-by-col '{group_by_col}' not found in dataframe columns")
        selected_col = group_by_col
    elif group_by_prompt and prompt_col in df.columns:
        selected_col = prompt_col
    else:
        selected_col = "__all__"

    if selected_col == "__all__":
        group_ids = pd.Series(["g0000"] * len(df), index=df.index, dtype=object)
        group_meta = pd.DataFrame(
            [
                {
                    "group_id": "g0000",
                    "group_col": "__all__",
                    "group_size": int(len(df)),
                    "group_value_preview": "__all__",
                }
            ]
        )
        return group_ids, group_meta, selected_col

    raw_values = df[selected_col].fillna("__MISSING__").astype(str)
    codes, uniques = pd.factorize(raw_values, sort=False)
    group_ids = pd.Series([f"g{int(c):04d}" for c in codes], index=df.index, dtype=object)

    counts = np.bincount(codes, minlength=len(uniques))
    rows: list[dict[str, Any]] = []
    for i, value in enumerate(uniques):
        preview = str(value).replace("\n", " ").strip()
        if len(preview) > 180:
            preview = preview[:180] + "..."
        rows.append(
            {
                "group_id": f"g{i:04d}",
                "group_col": selected_col,
                "group_size": int(counts[i]),
                "group_value_preview": preview,
            }
        )
    group_meta = pd.DataFrame(rows).sort_values("group_size", ascending=False).reset_index(drop=True)
    return group_ids, group_meta, selected_col


def build_column_bundles_grouped(
    series: pd.Series,
    group_ids: pd.Series,
    n_min: int,
    n_max: int,
    freq_threshold: float,
    position_std: float,
    coverage_threshold: float,
    mask_token: str,
) -> tuple[list[str], list[str], pd.DataFrame]:
    n = len(series)
    raw_all = [""] * n
    masked_all = [""] * n
    stats_parts: list[pd.DataFrame] = []

    gid_values = group_ids.to_numpy()
    unique_gids = pd.unique(gid_values)
    for gid in unique_gids:
        idx = np.where(gid_values == gid)[0]
        sub_series = series.iloc[idx]
        raw_sub, masked_sub, stats_sub = build_column_bundles(
            sub_series,
            n_min=n_min,
            n_max=n_max,
            freq_threshold=freq_threshold,
            position_std=position_std,
            coverage_threshold=coverage_threshold,
            mask_token=mask_token,
        )
        for p, v in zip(idx, raw_sub):
            raw_all[int(p)] = v
        for p, v in zip(idx, masked_sub):
            masked_all[int(p)] = v
        stats_sub = stats_sub.copy()
        stats_sub["group_id"] = gid
        stats_sub["group_size"] = len(idx)
        stats_parts.append(stats_sub)

    stats_df = pd.concat(stats_parts, ignore_index=True) if stats_parts else pd.DataFrame()
    return raw_all, masked_all, stats_df


def flatten_string_leaves(obj: Any, prefix: str = "") -> dict[str, str]:
    leaves: dict[str, str] = {}
    if isinstance(obj, dict):
        for key in sorted(obj.keys()):
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            leaves.update(flatten_string_leaves(obj[key], next_prefix))
        return leaves
    if isinstance(obj, list):
        for idx, item in enumerate(obj):
            next_prefix = f"{prefix}[{idx}]"
            leaves.update(flatten_string_leaves(item, next_prefix))
        return leaves
    if isinstance(obj, str):
        key = prefix if prefix else "_value"
        leaves[key] = obj
    return leaves


def build_output_string_leaf_maps(series: pd.Series) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for value in series.fillna("").tolist():
        parsed = safe_json_load(value)
        if isinstance(parsed, dict):
            out.append(flatten_string_leaves(parsed))
        else:
            out.append({})
    return out


def sanitize_leaf_path_token(path: str, max_len: int = 96) -> str:
    text = str(path).strip()
    if not text:
        return "leaf"
    tok = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    if not tok:
        tok = "leaf"
    return tok[: int(max_len)]


def run_detailed_output_leaf_inspection(
    *,
    output_series: pd.Series,
    group_ids: pd.Series,
    row_ids: pd.Series,
    hard_gate_mask: np.ndarray,
    model: TextEmbedder,
    batch_size: int,
    min_support: int,
    output_quantile: float,
    max_failed_paths_per_row: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, np.ndarray]]:
    n = len(output_series)
    if len(group_ids) != n or len(row_ids) != n or len(hard_gate_mask) != n:
        raise ValueError("Detailed inspection input length mismatch.")
    if not (0.0 < float(output_quantile) < 1.0):
        raise ValueError("--detail-output-quantile must be in (0,1).")
    min_support = int(max(1, min_support))
    max_failed_paths_per_row = int(max(1, max_failed_paths_per_row))

    row_leaf_maps = build_output_string_leaf_maps(output_series)
    gid_values = group_ids.fillna("g0000").astype(str).to_numpy(dtype=object)
    rid_values = row_ids.fillna("").astype(str).to_numpy(dtype=object)
    hard_gate = np.asarray(hard_gate_mask, dtype=bool)

    detail_evaluated = np.zeros(n, dtype=bool)
    detail_fail_any = np.zeros(n, dtype=bool)
    detail_fail_count = np.zeros(n, dtype=int)
    detail_eval_count = np.zeros(n, dtype=int)
    failed_paths_with_signal: list[list[tuple[str, float]]] = [[] for _ in range(n)]

    summary_rows: list[dict[str, Any]] = []
    row_hit_rows: list[dict[str, Any]] = []

    unique_groups = pd.unique(gid_values)
    for gid in unique_groups:
        idx_group = np.where(gid_values == gid)[0]
        if len(idx_group) == 0:
            continue
        idx_eval = idx_group[hard_gate[idx_group]]
        if len(idx_eval) == 0:
            continue
        all_paths = sorted({p for i in idx_eval for p in row_leaf_maps[int(i)].keys()})
        for path in all_paths:
            path_row_idx: list[int] = []
            path_texts: list[str] = []
            for i in idx_eval:
                text = str(row_leaf_maps[int(i)].get(path, ""))
                if text.strip():
                    path_row_idx.append(int(i))
                    path_texts.append(text)
            support = len(path_row_idx)
            if support < min_support:
                continue

            emb = model.encode(path_texts, batch_size=batch_size)
            y_norm = normalize_rows(sanitize_matrix(np.asarray(emb, dtype=float)))
            metric = OutputDensityMetric(min_k=int(SIGNAL_RUNTIME.min_k), max_k=int(SIGNAL_RUNTIME.max_k))
            signal, used_ks = metric.compute(
                y_norm,
                ref_mask=np.ones(support, dtype=bool),
                ref_weights=None,
            )
            threshold = float(np.quantile(signal, float(output_quantile)))
            failed = np.asarray(signal > threshold, dtype=bool)

            summary_rows.append(
                {
                    "group_id": str(gid),
                    "leaf_path": str(path),
                    "support": int(support),
                    "threshold": float(threshold),
                    "signal_mean": float(np.mean(signal)),
                    "signal_std": float(np.std(signal)),
                    "fail_count": int(np.sum(failed)),
                    "fail_rate": float(np.mean(failed)),
                    "output_quantile": float(output_quantile),
                    "used_ks": ",".join(map(str, used_ks)),
                }
            )
            for j, row_idx in enumerate(path_row_idx):
                s = float(signal[j])
                is_failed = bool(failed[j])
                detail_eval_count[row_idx] += 1
                if is_failed:
                    detail_fail_count[row_idx] += 1
                    failed_paths_with_signal[row_idx].append((str(path), s))
                row_hit_rows.append(
                    {
                        "row_index": int(row_idx),
                        "row_id": str(rid_values[row_idx]),
                        "group_id": str(gid),
                        "leaf_path": str(path),
                        "signal": s,
                        "threshold": float(threshold),
                        "failed": is_failed,
                    }
                )

    detail_evaluated = detail_eval_count > 0
    detail_fail_any = detail_fail_count > 0
    failed_paths_text = np.full(n, "", dtype=object)
    for i in range(n):
        failed_items = failed_paths_with_signal[i]
        if not failed_items:
            continue
        failed_items_sorted = sorted(failed_items, key=lambda x: float(x[1]), reverse=True)
        top_paths = [str(p) for p, _ in failed_items_sorted[:max_failed_paths_per_row]]
        failed_paths_text[i] = "|".join(top_paths)

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["fail_rate", "support"], ascending=[False, False]).reset_index(drop=True)
    row_hits_df = pd.DataFrame(row_hit_rows)

    hard_eval = hard_gate & detail_evaluated
    report = {
        "mode": "detailed",
        "rows": int(n),
        "hard_gate_rows": int(np.sum(hard_gate)),
        "detail_evaluated_rows": int(np.sum(detail_evaluated)),
        "detail_hard_gate_evaluated_rows": int(np.sum(hard_eval)),
        "detail_fail_rows": int(np.sum(detail_fail_any)),
        "detail_fail_rate_on_hard_evaluated": (
            float(np.mean(detail_fail_any[hard_eval])) if np.any(hard_eval) else 0.0
        ),
        "detail_leaf_paths_evaluated": int(len(summary_df)),
        "detail_row_hits": int(len(row_hits_df)),
        "detail_leaf_min_support": int(min_support),
        "detail_output_quantile": float(output_quantile),
        "max_failed_paths_per_row": int(max_failed_paths_per_row),
    }
    row_metrics = {
        "detail_evaluated_nomask": detail_evaluated.astype(bool),
        "detail_fail_any_leaf_nomask": detail_fail_any.astype(bool),
        "detail_fail_leaf_count_nomask": detail_fail_count.astype(int),
        "detail_eval_leaf_count_nomask": detail_eval_count.astype(int),
        "detail_failed_leaf_paths_nomask": failed_paths_text.astype(object),
    }
    return summary_df, row_hits_df, report, row_metrics


def build_detail_leaf_distribution_artifacts(
    detail_leaf_hits_df: pd.DataFrame,
    *,
    bins: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    bins = max(4, int(bins))
    if detail_leaf_hits_df.empty:
        report = {
            "enabled": False,
            "bins": int(bins),
            "leaf_paths": 0,
            "row_hits": 0,
        }
        return pd.DataFrame(), pd.DataFrame(), report

    required_cols = {"group_id", "leaf_path", "signal", "threshold", "failed"}
    missing = sorted(required_cols - set(detail_leaf_hits_df.columns))
    if missing:
        raise ValueError(f"detail_leaf_hits_df is missing columns: {missing}")

    stats_rows: list[dict[str, Any]] = []
    hist_rows: list[dict[str, Any]] = []
    grouped = detail_leaf_hits_df.groupby(["group_id", "leaf_path"], sort=True, dropna=False)

    for (group_id, leaf_path), sub in grouped:
        signals = pd.to_numeric(sub["signal"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(signals)
        if not np.any(valid):
            continue
        signals = signals[valid]
        failed_all = sub["failed"].fillna(False).astype(bool).to_numpy(dtype=bool)
        failed = failed_all[valid]
        thresholds = pd.to_numeric(sub["threshold"], errors="coerce").to_numpy(dtype=float)
        threshold = float(np.nanmedian(thresholds)) if np.any(np.isfinite(thresholds)) else float("nan")

        support = int(len(signals))
        fail_count = int(np.sum(failed))
        fail_rate = float(fail_count / support) if support > 0 else 0.0
        sig_min = float(np.min(signals))
        sig_max = float(np.max(signals))
        q50, q90, q95, q99 = [float(np.quantile(signals, q)) for q in (0.50, 0.90, 0.95, 0.99)]
        stats_rows.append(
            {
                "group_id": str(group_id),
                "leaf_path": str(leaf_path),
                "support": int(support),
                "threshold": float(threshold),
                "signal_mean": float(np.mean(signals)),
                "signal_std": float(np.std(signals)),
                "signal_min": float(sig_min),
                "signal_q50": float(q50),
                "signal_q90": float(q90),
                "signal_q95": float(q95),
                "signal_q99": float(q99),
                "signal_max": float(sig_max),
                "fail_count": int(fail_count),
                "fail_rate": float(fail_rate),
                "bins": int(bins),
            }
        )

        if sig_max > sig_min:
            edges = np.linspace(sig_min, sig_max, bins + 1)
        else:
            width = max(abs(sig_min) * 0.05, 1e-6)
            edges = np.linspace(sig_min - width, sig_max + width, bins + 1)

        counts, _ = np.histogram(signals, bins=edges)
        fail_counts = np.zeros(len(counts), dtype=int)
        if fail_count > 0:
            fail_counts, _ = np.histogram(signals[failed], bins=edges)

        for b in range(len(counts)):
            count = int(counts[b])
            fail_n = int(fail_counts[b])
            ratio = float(fail_n / count) if count > 0 else 0.0
            left = float(edges[b])
            right = float(edges[b + 1])
            center = float(0.5 * (left + right))
            hist_rows.append(
                {
                    "group_id": str(group_id),
                    "leaf_path": str(leaf_path),
                    "bin_index": int(b),
                    "bin_left": float(left),
                    "bin_right": float(right),
                    "bin_center": float(center),
                    "count": int(count),
                    "fail_count": int(fail_n),
                    "fail_ratio": float(ratio),
                    "threshold": float(threshold),
                }
            )

    stats_df = pd.DataFrame(stats_rows)
    hist_df = pd.DataFrame(hist_rows)
    if not stats_df.empty:
        stats_df = stats_df.sort_values(["fail_rate", "support"], ascending=[False, False]).reset_index(drop=True)
    if not hist_df.empty:
        hist_df = hist_df.sort_values(["group_id", "leaf_path", "bin_index"]).reset_index(drop=True)
    report = {
        "enabled": bool((not stats_df.empty) and (not hist_df.empty)),
        "bins": int(bins),
        "leaf_paths": int(len(stats_df)),
        "row_hits": int(len(detail_leaf_hits_df)),
    }
    return stats_df, hist_df, report


def render_detail_leaf_distribution_html(
    *,
    stats_df: pd.DataFrame,
    hist_df: pd.DataFrame,
    out_html: Path,
    max_leaf_sections: int = 120,
) -> None:
    out_html.parent.mkdir(parents=True, exist_ok=True)
    if stats_df.empty or hist_df.empty:
        out_html.write_text(
            (
                "<html><head><meta charset='utf-8'><title>Detail Leaf Distribution</title></head><body>"
                "<h1>Detail Leaf Distribution</h1><p>No detailed leaf distribution data.</p></body></html>"
            ),
            encoding="utf-8",
        )
        return

    html_parts: list[str] = []
    html_parts.append(
        "<html><head><meta charset='utf-8'><title>Detail Leaf Distribution</title>"
        "<style>"
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:24px;}"
        "h1{margin-bottom:10px;}h2{margin-top:26px;margin-bottom:8px;}"
        ".meta{color:#374151;font-size:13px;margin-bottom:10px;line-height:1.5;}"
        "table{border-collapse:collapse;width:min(1100px,96vw);font-size:12px;margin-bottom:14px;}"
        "th,td{border:1px solid #e5e7eb;padding:6px 8px;text-align:right;}"
        "th:first-child,td:first-child{text-align:left;}"
        "th{background:#f8fafc;}"
        ".barwrap{width:180px;height:10px;background:#f1f5f9;border-radius:999px;overflow:hidden;}"
        ".bar{height:10px;background:#2563eb;}"
        "</style></head><body>"
    )
    html_parts.append("<h1>Detailed Leaf Signal Distribution (nomask)</h1>")
    html_parts.append(
        "<div class='meta'>"
        f"leaf_paths=<code>{int(len(stats_df))}</code>, hist_bins=<code>{int(len(hist_df))}</code>"
        "</div>"
    )

    limited = stats_df.head(int(max_leaf_sections)).copy()
    for row in limited.itertuples(index=False):
        gid = str(getattr(row, "group_id", ""))
        path = str(getattr(row, "leaf_path", ""))
        support = int(getattr(row, "support", 0))
        threshold = float(getattr(row, "threshold", np.nan))
        fail_count = int(getattr(row, "fail_count", 0))
        fail_rate = float(getattr(row, "fail_rate", 0.0))
        title = f"{gid} :: {path}"
        html_parts.append(f"<h2>{html_lib.escape(title)}</h2>")
        html_parts.append(
            "<div class='meta'>"
            f"support=<code>{support}</code>, threshold=<code>{threshold:.6g}</code>, "
            f"fail=<code>{fail_count}</code>, fail_rate=<code>{fail_rate:.4f}</code>"
            "</div>"
        )

        mask = (
            hist_df["group_id"].astype(str).eq(gid)
            & hist_df["leaf_path"].astype(str).eq(path)
        )
        hsub = hist_df.loc[mask].copy()
        if hsub.empty:
            html_parts.append("<p>No histogram bins.</p>")
            continue
        max_count = int(pd.to_numeric(hsub["count"], errors="coerce").fillna(0).max())
        if max_count <= 0:
            max_count = 1
        rows_html: list[str] = []
        for b in hsub.itertuples(index=False):
            left = float(getattr(b, "bin_left", np.nan))
            right = float(getattr(b, "bin_right", np.nan))
            center = float(getattr(b, "bin_center", np.nan))
            count = int(getattr(b, "count", 0))
            fail_n = int(getattr(b, "fail_count", 0))
            fail_r = float(getattr(b, "fail_ratio", 0.0))
            width = float(np.clip((count / max_count) * 180.0, 0.0, 180.0))
            rows_html.append(
                "<tr>"
                f"<td>{left:.6g} ~ {right:.6g}</td>"
                f"<td>{center:.6g}</td>"
                f"<td>{count}</td>"
                f"<td>{fail_n}</td>"
                f"<td>{fail_r:.4f}</td>"
                f"<td><div class='barwrap'><div class='bar' style='width:{width:.1f}px'></div></div></td>"
                "</tr>"
            )
        html_parts.append(
            "<table><thead><tr><th>bin range</th><th>center</th><th>count</th><th>fail_count</th><th>fail_ratio</th><th>density</th></tr></thead>"
            f"<tbody>{''.join(rows_html)}</tbody></table>"
        )

    html_parts.append("</body></html>")
    out_html.write_text("\n".join(html_parts), encoding="utf-8")


def apply_detailed_override_nomask(result_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = result_df.copy()
    if "distribution_pass_nomask" not in out.columns:
        out["distribution_pass_nomask"] = False
    if "final_pass_nomask" not in out.columns:
        out["final_pass_nomask"] = False
    required_cols = [
        "hard_gate_pass",
        "detail_evaluated_nomask",
        "detail_fail_any_leaf_nomask",
    ]
    missing = [c for c in required_cols if c not in out.columns]
    if missing:
        raise ValueError(f"Detailed override requires columns: {missing}")

    hard_gate = out["hard_gate_pass"].fillna(False).astype(bool).to_numpy()
    detail_eval = out["detail_evaluated_nomask"].fillna(False).astype(bool).to_numpy()
    detail_fail = out["detail_fail_any_leaf_nomask"].fillna(False).astype(bool).to_numpy()
    detail_target = hard_gate & detail_eval & detail_fail

    pre_dist_pass = out["distribution_pass_nomask"].fillna(False).astype(bool).to_numpy()
    pre_final_pass = (
        out["final_pass_nomask"].fillna(False).astype(bool).to_numpy()
        if "final_pass_nomask" in out.columns
        else (hard_gate & pre_dist_pass)
    )
    if "distribution_state_nomask" in out.columns:
        pre_state = out["distribution_state_nomask"].fillna("na").astype(str).str.strip().str.lower().to_numpy(dtype=object)
    else:
        pre_state = np.full(len(out), "na", dtype=object)
        pre_state[hard_gate] = "pass"
        pre_state[hard_gate & (~pre_dist_pass)] = "fail"

    dist_hard = (
        out["distribution_hard_fail_nomask"].fillna(False).astype(bool).to_numpy()
        if "distribution_hard_fail_nomask" in out.columns
        else np.zeros(len(out), dtype=bool)
    )
    dist_fail = (
        out["distribution_fail_nomask"].fillna(False).astype(bool).to_numpy()
        if "distribution_fail_nomask" in out.columns
        else (~pre_dist_pass)
    )
    dist_warn = (
        out["distribution_warn_nomask"].fillna(False).astype(bool).to_numpy()
        if "distribution_warn_nomask" in out.columns
        else np.zeros(len(out), dtype=bool)
    )

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

    stats = {
        "detail_mode_enabled": True,
        "detail_target_rows": int(np.sum(detail_target)),
        "detail_override_rows": int(np.sum(override_applied)),
        "distribution_pass_rate_pre": (
            float(np.mean(pre_dist_pass[hard_gate])) if np.any(hard_gate) else 0.0
        ),
        "distribution_pass_rate_post": (
            float(np.mean(dist_pass_new[hard_gate])) if np.any(hard_gate) else 0.0
        ),
        "final_pass_rate_pre": (
            float(np.mean(pre_final_pass[hard_gate])) if np.any(hard_gate) else 0.0
        ),
        "final_pass_rate_post": (
            float(np.mean(final_pass_new[hard_gate])) if np.any(hard_gate) else 0.0
        ),
    }
    return out, stats


def assign_subset_columns(dest_df: pd.DataFrame, subset_index: pd.Index, src_df: pd.DataFrame) -> pd.DataFrame:
    missing_cols = [col for col in src_df.columns if col not in dest_df.columns]
    if missing_cols:
        new_cols: dict[str, pd.Series] = {}
        for col in missing_cols:
            s = src_df[col]
            if pd.api.types.is_bool_dtype(s):
                new_cols[col] = pd.Series([False] * len(dest_df), index=dest_df.index, dtype=bool)
            elif pd.api.types.is_numeric_dtype(s):
                new_cols[col] = pd.Series([np.nan] * len(dest_df), index=dest_df.index, dtype=float)
            else:
                new_cols[col] = pd.Series([None] * len(dest_df), index=dest_df.index, dtype=object)
        dest_df = pd.concat([dest_df, pd.DataFrame(new_cols, index=dest_df.index)], axis=1)

    dest_df.loc[subset_index, src_df.columns] = src_df.to_numpy(copy=False)
    return dest_df


def parse_apply_modes(raw: str) -> list[str]:
    # Operational runtime is single-pass distribution only.
    modes: list[str] = []
    for tok in parse_csv_tokens(raw):
        key = tok.strip().lower()
        if key not in {"nomask", "mask"}:
            raise ValueError(f"Unknown mode in apply_modes: {tok}. Allowed=['nomask', 'mask']")
        if key == "mask":
            continue
        if key not in modes:
            modes.append(key)
    if not modes:
        return ["nomask"]
    return modes


def parse_calibration_rules(raw: str) -> list[str]:
    rules = [x.strip().lower() for x in parse_csv_tokens(raw)]
    return validate_calibration_rules(rules)


def parse_calibration_label(y: pd.Series, bad_value: str) -> np.ndarray:
    target = str(bad_value).strip().lower()
    vals = y.fillna("").astype(str).str.strip().str.lower()
    return vals.eq(target).to_numpy(dtype=bool)


def get_rule_available_mask(result_df: pd.DataFrame, rule: str, mode: str) -> np.ndarray:
    col = f"{rule}_available_{mode}"
    if col in result_df.columns:
        return result_df[col].fillna(False).astype(bool).to_numpy()
    if rule == "discourse_instability":
        alt = f"contradiction_available_{mode}"
        if alt in result_df.columns:
            return result_df[alt].fillna(False).astype(bool).to_numpy()
    if rule == "contradiction":
        alt = f"discourse_instability_available_{mode}"
        if alt in result_df.columns:
            return result_df[alt].fillna(False).astype(bool).to_numpy()
    return np.ones(len(result_df), dtype=bool)


def resolve_tristate_warn_quantiles(
    fail_quantiles: dict[str, float],
    tristate_rules: list[str],
) -> dict[str, float]:
    base_q = float(TRISTATE_CFG.warn_quantile)
    if not (0.0 < base_q < 1.0):
        raise ValueError("TRISTATE_RUNTIME.warn_quantile must be in (0,1).")
    qmap = {r: base_q for r in tristate_rules}
    if TRISTATE_CFG.warn_rule_quantiles_json:
        raw = json.loads(TRISTATE_CFG.warn_rule_quantiles_json)
        if not isinstance(raw, dict):
            raise ValueError("TRISTATE_RUNTIME.warn_rule_quantiles_json must be JSON object.")
        for k, v in raw.items():
            nk = normalize_pipeline_rule_key(k)
            if nk not in tristate_rules:
                continue
            qv = float(v)
            if not (0.0 < qv < 1.0):
                raise ValueError(f"Warn quantile must be in (0,1): {k}={v}")
            qmap[nk] = qv
    for r in tristate_rules:
        fq = float(fail_quantiles.get(r, 0.99))
        wq = float(qmap[r])
        if wq >= fq:
            wq = max(0.0001, fq - 0.01)
        qmap[r] = wq
    return qmap


def save_signal_histograms(
    mode_name: str,
    signal_map: dict[str, np.ndarray],
    thresholds: dict[str, float],
    ref_mask: np.ndarray,
    bins: int,
    output_dir: Path,
    base_name: str,
) -> Path | None:
    keys = [k for k in RULE_KEYS if k in signal_map]
    payload: list[dict[str, Any]] = []
    for key in keys:
        vals = sanitize_matrix(signal_map[key]).ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        edges = np.histogram_bin_edges(vals, bins=bins)
        all_density, _ = np.histogram(vals, bins=edges, density=True)
        ref_density = np.zeros_like(all_density, dtype=float)
        if ref_mask.any():
            ref_vals = sanitize_matrix(signal_map[key][ref_mask]).ravel()
            ref_vals = ref_vals[np.isfinite(ref_vals)]
            if ref_vals.size > 0:
                ref_density, _ = np.histogram(ref_vals, bins=edges, density=True)
        payload.append(
            {
                "rule": key,
                "bin_centers": ((edges[:-1] + edges[1:]) * 0.5).tolist(),
                "bin_widths": np.diff(edges).tolist(),
                "all_density": all_density.astype(float).tolist(),
                "ref_density": ref_density.astype(float).tolist(),
                "threshold": float(thresholds.get(key, np.nan)),
                "all_size": int(vals.size),
                "ref_size": int(ref_mask.sum()) if ref_mask.any() else 0,
            }
        )

    if not payload:
        return None

    payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    html_template = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Signal Histograms (__MODE__)</title>
  <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
  <style>
    body { font-family: Arial, Helvetica, sans-serif; margin: 20px; color: #111827; }
    .card { border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px; margin-bottom: 16px; }
    .title { margin: 0 0 8px 0; font-size: 18px; }
    .meta { color: #6b7280; font-size: 12px; margin-bottom: 8px; }
  </style>
</head>
<body>
  <h1 class="title">Signal Histograms (__MODE__)</h1>
  <div class="meta">source: __BASENAME__</div>
  <div id="charts"></div>
  <script>
  const payload = __PAYLOAD_JSON__;
  const charts = document.getElementById('charts');
  payload.forEach((item, idx) => {
    const card = document.createElement('div');
    card.className = 'card';
    const chartId = `signal_hist_${idx}`;
    card.innerHTML = `<div class="title">${item.rule}</div><div class="meta">all=${item.all_size}, ref=${item.ref_size}</div><div id="${chartId}" style="height: 360px;"></div>`;
    charts.appendChild(card);

    const traces = [
      {
        x: item.bin_centers,
        y: item.all_density,
        type: 'bar',
        name: 'all',
        marker: { color: '#4C78A8', opacity: 0.60 },
      },
      {
        x: item.bin_centers,
        y: item.ref_density,
        type: 'scatter',
        mode: 'lines',
        name: 'ref',
        line: { color: '#111827', width: 2.0 },
      },
    ];

    const shapes = [];
    if (Number.isFinite(item.threshold)) {
      shapes.push({
        type: 'line',
        x0: item.threshold,
        x1: item.threshold,
        y0: 0,
        y1: 1,
        xref: 'x',
        yref: 'paper',
        line: { color: '#E45756', dash: 'dash', width: 2.0 },
      });
    }

    Plotly.newPlot(
      chartId,
      traces,
      {
        barmode: 'overlay',
        xaxis: { title: 'signal' },
        yaxis: { title: 'density' },
        legend: { orientation: 'h' },
        shapes,
      },
      { responsive: true, displaylogo: false, margin: { t: 20, l: 52, r: 24, b: 50 } },
    );
  });
  </script>
</body>
</html>
"""
    html = (
        html_template
        .replace("__MODE__", mode_name)
        .replace("__BASENAME__", base_name)
        .replace("__PAYLOAD_JSON__", payload_json)
    )
    out_path = output_dir / f"{base_name}_{mode_name}_signal_hist.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


def save_signal_hist_data(
    mode_name: str,
    signal_map: dict[str, np.ndarray],
    thresholds: dict[str, float],
    ref_mask: np.ndarray,
    bins: int,
    output_dir: Path,
    base_name: str,
) -> Path:
    rows: list[dict[str, Any]] = []
    keys = [k for k in RULE_KEYS if k in signal_map]
    for key in keys:
        vals = sanitize_matrix(signal_map[key]).ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        edges = np.histogram_bin_edges(vals, bins=bins)
        all_counts, _ = np.histogram(vals, bins=edges)
        ref_vals = sanitize_matrix(signal_map[key][ref_mask]).ravel() if ref_mask.any() else np.array([])
        ref_vals = ref_vals[np.isfinite(ref_vals)] if ref_vals.size > 0 else ref_vals
        ref_counts, _ = np.histogram(ref_vals, bins=edges) if ref_vals.size > 0 else (np.zeros_like(all_counts), edges)
        for i in range(len(edges) - 1):
            rows.append(
                {
                    "mode": mode_name,
                    "signal_key": key,
                    "score_key": key,
                    "bin_left": float(edges[i]),
                    "bin_right": float(edges[i + 1]),
                    "all_count": int(all_counts[i]),
                    "ref_count": int(ref_counts[i]),
                    "threshold": float(thresholds[key]),
                }
            )
    out_path = output_dir / f"{base_name}_{mode_name}_signal_hist_data.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def _sample_arrow_indices(n: int, max_arrows: int, seed: int, priority_mask: np.ndarray) -> np.ndarray:
    idx = np.arange(n)
    pri = idx[priority_mask]
    if len(pri) >= max_arrows:
        return pri[:max_arrows]
    remain = idx[~priority_mask]
    take = max_arrows - len(pri)
    if take <= 0 or len(remain) == 0:
        return pri
    rng = np.random.default_rng(seed)
    if len(remain) > take:
        remain = rng.choice(remain, size=take, replace=False)
    return np.concatenate([pri, remain])


def save_arrow_plot(
    mode_name: str,
    method_name: str,
    in_2d: np.ndarray,
    out_2d: np.ndarray,
    anomaly_mask: np.ndarray,
    conflict_mask: np.ndarray,
    max_arrows: int,
    seed: int,
    output_dir: Path,
    base_name: str,
    label_is_correct: np.ndarray | None = None,
    pred_pass: np.ndarray | None = None,
    pred_name: str = "distribution_pass",
) -> Path | None:
    n = len(in_2d)
    priority = anomaly_mask | conflict_mask
    draw_idxs = _sample_arrow_indices(n=n, max_arrows=min(max_arrows, n), seed=seed, priority_mask=priority)

    label_vec: np.ndarray | None = None
    has_label_coloring = False
    if label_is_correct is not None:
        try:
            label_vec = sanitize_matrix(np.asarray(label_is_correct, dtype=float)).ravel()
            if len(label_vec) == n and np.isfinite(label_vec).any():
                has_label_coloring = True
        except Exception:
            label_vec = None
            has_label_coloring = False

    pred_vec: np.ndarray | None = None
    has_confusion_coloring = False
    if pred_pass is not None and has_label_coloring and label_vec is not None:
        try:
            pred_vec = np.asarray(pred_pass, dtype=bool).ravel()
            if len(pred_vec) == n:
                has_confusion_coloring = True
        except Exception:
            pred_vec = None
            has_confusion_coloring = False

    traces: list[dict[str, Any]] = []
    if has_confusion_coloring and label_vec is not None and pred_vec is not None:
        known_mask = np.isfinite(label_vec)
        true_bad = known_mask & (label_vec < 0.5)
        true_good = known_mask & (label_vec >= 0.5)
        pred_bad = ~pred_vec
        pred_good = pred_vec

        tp_mask = true_bad & pred_bad
        fp_mask = true_good & pred_bad
        tn_mask = true_good & pred_good
        fn_mask = true_bad & pred_good
        unknown_mask = ~known_mask

        groups = [
            ("TN", tn_mask, "#2CA02C"),
            ("TP", tp_mask, "#D62728"),
            ("FP", fp_mask, "#FF7F0E"),
            ("FN", fn_mask, "#1F77B4"),
        ]
        for name, mask, color in groups:
            if not mask.any():
                continue
            traces.append(
                {
                    "type": "scattergl",
                    "mode": "markers",
                    "name": f"input({name})",
                    "x": in_2d[mask, 0].astype(float).tolist(),
                    "y": in_2d[mask, 1].astype(float).tolist(),
                    "marker": {"size": 6, "color": color, "opacity": 0.50, "symbol": "circle"},
                }
            )
            traces.append(
                {
                    "type": "scattergl",
                    "mode": "markers",
                    "name": f"output({name})",
                    "x": out_2d[mask, 0].astype(float).tolist(),
                    "y": out_2d[mask, 1].astype(float).tolist(),
                    "marker": {"size": 7, "color": color, "opacity": 0.35, "symbol": "triangle-up"},
                }
            )
        if unknown_mask.any():
            traces.append(
                {
                    "type": "scattergl",
                    "mode": "markers",
                    "name": "input(unknown)",
                    "x": in_2d[unknown_mask, 0].astype(float).tolist(),
                    "y": in_2d[unknown_mask, 1].astype(float).tolist(),
                    "marker": {"size": 5, "color": "gray", "opacity": 0.25, "symbol": "circle"},
                }
            )
            traces.append(
                {
                    "type": "scattergl",
                    "mode": "markers",
                    "name": "output(unknown)",
                    "x": out_2d[unknown_mask, 0].astype(float).tolist(),
                    "y": out_2d[unknown_mask, 1].astype(float).tolist(),
                    "marker": {"size": 6, "color": "gray", "opacity": 0.25, "symbol": "triangle-up"},
                }
            )
    elif has_label_coloring and label_vec is not None:
        correct_mask = np.isfinite(label_vec) & (label_vec >= 0.5)
        incorrect_mask = np.isfinite(label_vec) & (label_vec < 0.5)
        unknown_mask = ~np.isfinite(label_vec)

        if correct_mask.any():
            traces.append(
                {
                    "type": "scattergl",
                    "mode": "markers",
                    "name": "input(correct)",
                    "x": in_2d[correct_mask, 0].astype(float).tolist(),
                    "y": in_2d[correct_mask, 1].astype(float).tolist(),
                    "marker": {"size": 6, "color": "#2CA02C", "opacity": 0.50, "symbol": "circle"},
                }
            )
            traces.append(
                {
                    "type": "scattergl",
                    "mode": "markers",
                    "name": "output(correct)",
                    "x": out_2d[correct_mask, 0].astype(float).tolist(),
                    "y": out_2d[correct_mask, 1].astype(float).tolist(),
                    "marker": {"size": 7, "color": "#2CA02C", "opacity": 0.35, "symbol": "triangle-up"},
                }
            )
        if incorrect_mask.any():
            traces.append(
                {
                    "type": "scattergl",
                    "mode": "markers",
                    "name": "input(incorrect)",
                    "x": in_2d[incorrect_mask, 0].astype(float).tolist(),
                    "y": in_2d[incorrect_mask, 1].astype(float).tolist(),
                    "marker": {"size": 6, "color": "#D62728", "opacity": 0.50, "symbol": "circle"},
                }
            )
            traces.append(
                {
                    "type": "scattergl",
                    "mode": "markers",
                    "name": "output(incorrect)",
                    "x": out_2d[incorrect_mask, 0].astype(float).tolist(),
                    "y": out_2d[incorrect_mask, 1].astype(float).tolist(),
                    "marker": {"size": 7, "color": "#D62728", "opacity": 0.35, "symbol": "triangle-up"},
                }
            )
        if unknown_mask.any():
            traces.append(
                {
                    "type": "scattergl",
                    "mode": "markers",
                    "name": "input(unknown)",
                    "x": in_2d[unknown_mask, 0].astype(float).tolist(),
                    "y": in_2d[unknown_mask, 1].astype(float).tolist(),
                    "marker": {"size": 5, "color": "gray", "opacity": 0.25, "symbol": "circle"},
                }
            )
            traces.append(
                {
                    "type": "scattergl",
                    "mode": "markers",
                    "name": "output(unknown)",
                    "x": out_2d[unknown_mask, 0].astype(float).tolist(),
                    "y": out_2d[unknown_mask, 1].astype(float).tolist(),
                    "marker": {"size": 6, "color": "gray", "opacity": 0.25, "symbol": "triangle-up"},
                }
            )
    else:
        traces.append(
            {
                "type": "scattergl",
                "mode": "markers",
                "name": "input",
                "x": in_2d[:, 0].astype(float).tolist(),
                "y": in_2d[:, 1].astype(float).tolist(),
                "marker": {"size": 5, "color": "#4C78A8", "opacity": 0.35, "symbol": "circle"},
            }
        )
        traces.append(
            {
                "type": "scattergl",
                "mode": "markers",
                "name": "output",
                "x": out_2d[:, 0].astype(float).tolist(),
                "y": out_2d[:, 1].astype(float).tolist(),
                "marker": {"size": 6, "color": "#72B7B2", "opacity": 0.25, "symbol": "triangle-up"},
            }
        )

    def build_vector_trace(idxs: np.ndarray, color: str, width: float, opacity: float, name: str) -> dict[str, Any] | None:
        if len(idxs) == 0:
            return None
        x_vals: list[float | None] = []
        y_vals: list[float | None] = []
        for i in idxs:
            x_vals.extend([float(in_2d[i, 0]), float(out_2d[i, 0]), None])
            y_vals.extend([float(in_2d[i, 1]), float(out_2d[i, 1]), None])
        return {
            "type": "scattergl",
            "mode": "lines",
            "name": name,
            "x": x_vals,
            "y": y_vals,
            "line": {"color": color, "width": width},
            "opacity": opacity,
            "hoverinfo": "skip",
        }

    draw_idxs = np.asarray(draw_idxs, dtype=int)
    if has_confusion_coloring and label_vec is not None and pred_vec is not None:
        known_mask = np.isfinite(label_vec)
        true_bad = known_mask & (label_vec < 0.5)
        true_good = known_mask & (label_vec >= 0.5)
        pred_bad = ~pred_vec
        pred_good = pred_vec

        idx_tp = draw_idxs[true_bad[draw_idxs] & pred_bad[draw_idxs]]
        idx_fp = draw_idxs[true_good[draw_idxs] & pred_bad[draw_idxs]]
        idx_fn = draw_idxs[true_bad[draw_idxs] & pred_good[draw_idxs]]
        idx_tn = draw_idxs[true_good[draw_idxs] & pred_good[draw_idxs]]
        idx_unk = draw_idxs[~known_mask[draw_idxs]]

        tr_tn = build_vector_trace(idx_tn, "#2CA02C", 0.9, 0.20, "vectors(TN)")
        tr_fn = build_vector_trace(idx_fn, "#1F77B4", 1.1, 0.55, "vectors(FN)")
        tr_fp = build_vector_trace(idx_fp, "#FF7F0E", 1.2, 0.60, "vectors(FP)")
        tr_tp = build_vector_trace(idx_tp, "#D62728", 1.5, 0.80, "vectors(TP)")
        tr_unk = build_vector_trace(idx_unk, "gray", 0.8, 0.18, "vectors(unknown)")
        for tr in (tr_tn, tr_fn, tr_fp, tr_tp, tr_unk):
            if tr is not None:
                traces.append(tr)
    else:
        idx_anom = draw_idxs[anomaly_mask[draw_idxs]]
        idx_conf = draw_idxs[(~anomaly_mask[draw_idxs]) & conflict_mask[draw_idxs]]
        idx_norm = draw_idxs[(~anomaly_mask[draw_idxs]) & (~conflict_mask[draw_idxs])]

        tr_anom = build_vector_trace(idx_anom, "#E45756", 1.5, 0.75, "vectors(anomaly)")
        tr_conf = build_vector_trace(idx_conf, "#F58518", 1.2, 0.55, "vectors(conflict)")
        tr_norm = build_vector_trace(idx_norm, "gray", 0.8, 0.18, "vectors(normal)")
        for tr in (tr_norm, tr_conf, tr_anom):
            if tr is not None:
                traces.append(tr)

    if has_confusion_coloring:
        title_suffix = f", confusion-colored ({pred_name})"
    elif has_label_coloring and label_vec is not None:
        title_suffix = ", label-colored"
    else:
        title_suffix = ""

    xaxis_cfg: dict[str, Any] = {"title": f"{method_name.upper()}-1"}
    if method_name.lower() == "pca":
        xaxis_cfg["range"] = [-0.5, 0.5]

    layout = {
        "title": f"{mode_name} input->output vectors ({method_name}{title_suffix})",
        "xaxis": xaxis_cfg,
        "yaxis": {"title": f"{method_name.upper()}-2", "scaleanchor": "x", "scaleratio": 1},
        "legend": {"orientation": "h"},
        "hovermode": "closest",
        "margin": {"t": 48, "l": 52, "r": 24, "b": 48},
    }

    traces_json = json.dumps(traces, ensure_ascii=False).replace("</", "<\\/")
    layout_json = json.dumps(layout, ensure_ascii=False).replace("</", "<\\/")
    html_template = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Arrow Plot (__MODE__ / __METHOD__)</title>
  <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
  <style>
    body { font-family: Arial, Helvetica, sans-serif; margin: 20px; color: #111827; }
    .meta { color: #6b7280; font-size: 12px; margin-bottom: 10px; }
  </style>
</head>
<body>
  <h1>Input→Output Vectors (__MODE__ / __METHOD__)</h1>
  <div class="meta">source: __BASENAME__ | drawn vectors: __DRAWN_COUNT__ / __TOTAL_COUNT__</div>
  <div id="plot" style="height: 760px;"></div>
  <script>
    const traces = __TRACES_JSON__;
    const layout = __LAYOUT_JSON__;
    Plotly.newPlot(
      'plot',
      traces,
      layout,
      { responsive: true, displaylogo: false }
    );
  </script>
</body>
</html>
"""
    html = (
        html_template
        .replace("__MODE__", mode_name)
        .replace("__METHOD__", method_name)
        .replace("__BASENAME__", base_name)
        .replace("__DRAWN_COUNT__", str(int(len(draw_idxs))))
        .replace("__TOTAL_COUNT__", str(int(n)))
        .replace("__TRACES_JSON__", traces_json)
        .replace("__LAYOUT_JSON__", layout_json)
    )

    out_path = output_dir / f"{base_name}_{mode_name}_arrow_{method_name}.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


def save_arrow_data(
    mode_name: str,
    method_name: str,
    in_2d: np.ndarray,
    out_2d: np.ndarray,
    anomaly_mask: np.ndarray,
    conflict_mask: np.ndarray,
    output_dir: Path,
    base_name: str,
    label_is_correct: np.ndarray | None = None,
    label_raw: np.ndarray | None = None,
    pred_pass: np.ndarray | None = None,
) -> Path:
    rows: dict[str, Any] = {
        "mode": mode_name,
        "method": method_name,
        "row_index": np.arange(len(in_2d)),
        "input_x": in_2d[:, 0],
        "input_y": in_2d[:, 1],
        "output_x": out_2d[:, 0],
        "output_y": out_2d[:, 1],
        "is_distribution_anomaly": anomaly_mask.astype(bool),
        "is_conflict_anomaly": conflict_mask.astype(bool),
    }
    if label_is_correct is not None:
        label_arr = sanitize_matrix(np.asarray(label_is_correct, dtype=float)).ravel()
        if len(label_arr) == len(in_2d):
            rows["label_is_correct"] = np.where(np.isfinite(label_arr), label_arr, np.nan)
    if label_raw is not None:
        label_raw_arr = np.asarray(label_raw, dtype=object).ravel()
        if len(label_raw_arr) == len(in_2d):
            rows["label_raw"] = label_raw_arr
    if pred_pass is not None:
        pred_arr = np.asarray(pred_pass, dtype=bool).ravel()
        if len(pred_arr) == len(in_2d):
            rows["pred_pass"] = pred_arr
            rows["pred_bad"] = ~pred_arr
            if "label_is_correct" in rows:
                label_arr = np.asarray(rows["label_is_correct"], dtype=float)
                known = np.isfinite(label_arr)
                y_bad = known & (label_arr < 0.5)
                y_good = known & (label_arr >= 0.5)
                pred_bad = ~pred_arr
                confusion = np.full(len(pred_arr), "unknown", dtype=object)
                confusion[y_bad & pred_bad] = "TP"
                confusion[y_good & pred_bad] = "FP"
                confusion[y_bad & (~pred_bad)] = "FN"
                confusion[y_good & (~pred_bad)] = "TN"
                rows["confusion_label"] = confusion
    df = pd.DataFrame(rows)
    out_path = output_dir / f"{base_name}_{mode_name}_arrow_{method_name}_data.csv"
    df.to_csv(out_path, index=False)
    return out_path


def export_top_anomalies(
    result_df: pd.DataFrame,
    source_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
    stem: str,
    tag: str,
) -> Path:
    rows: list[dict[str, Any]] = []
    rule_keys = RULE_KEYS.copy()
    signal_specs = [(rk, RULE_SIGNAL_PREFIX[rk]) for rk in RULE_KEYS]

    for _, srow in summary_df.iterrows():
        mode = str(srow["mode"])
        group_id = str(srow["group_id"]) if "group_id" in summary_df.columns else None
        thresholds: dict[str, float] = {}
        for rk in RULE_KEYS:
            col = f"{rk}_threshold"
            if col in summary_df.columns and pd.notna(srow.get(col)):
                thresholds[rk] = float(srow[col])
        if group_id is not None and "distribution_group_id" in result_df.columns:
            subset_idx = result_df.index[result_df["distribution_group_id"].astype(str) == group_id]
        else:
            subset_idx = result_df.index
        if "hard_gate_pass" in result_df.columns and len(subset_idx) > 0:
            hard_gate_subset_mask = result_df.loc[subset_idx, "hard_gate_pass"].to_numpy(dtype=bool)
            subset_idx = subset_idx[hard_gate_subset_mask]
        if len(subset_idx) == 0:
            continue
        for signal_key, signal_prefix in signal_specs:
            signal_col = f"{signal_prefix}_{mode}"
            if signal_col not in result_df.columns:
                continue
            subset_signal_idx = subset_idx
            signal_vals = sanitize_matrix(result_df.loc[subset_signal_idx, signal_col].to_numpy(dtype=float)).ravel()
            finite_mask = np.isfinite(signal_vals)
            if not finite_mask.any():
                continue
            subset_signal_idx = subset_signal_idx[finite_mask]
            signal_vals = signal_vals[finite_mask]
            order = np.argsort(-signal_vals)
            top_n = min(int(args.top_n_anomalies), len(order))
            for rank in range(top_n):
                local_pos = int(order[rank])
                idx = int(subset_signal_idx[local_pos])
                signal_value = float(signal_vals[local_pos])
                fail_rules = [
                    rk
                    for rk in rule_keys
                    if f"{RULE_PASS_PREFIX[rk]}_{mode}" in result_df.columns
                    and not bool(result_df.at[idx, f"{RULE_PASS_PREFIX[rk]}_{mode}"])
                ]
                rows.append(
                    {
                        "mode": mode,
                        "group_id": group_id,
                        "group_col": srow.get("group_col", ""),
                        "group_value_preview": srow.get("group_value_preview", ""),
                        "signal_key": signal_key,
                        "score_key": signal_key,
                        "rank": rank + 1,
                        "row_index": idx,
                        "row_id": result_df.at[idx, "row_id"] if "row_id" in result_df.columns else idx,
                        "signal": signal_value,
                        "score": signal_value,
                        "threshold": thresholds.get(signal_key),
                        "failed_this_rule": bool(
                            signal_key in thresholds and signal_value > float(thresholds[signal_key])
                        ),
                        "failed_rules": "|".join(fail_rules) if fail_rules else "",
                        "hard_gate_pass": bool(result_df.at[idx, "hard_gate_pass"]),
                        "distribution_evaluated": bool(result_df.at[idx, f"distribution_evaluated_{mode}"])
                        if f"distribution_evaluated_{mode}" in result_df.columns
                        else True,
                        "distribution_pass": bool(result_df.at[idx, f"distribution_pass_{mode}"])
                        if f"distribution_pass_{mode}" in result_df.columns
                        else None,
                        "final_pass": bool(result_df.at[idx, f"final_pass_{mode}"])
                        if f"final_pass_{mode}" in result_df.columns
                        else None,
                        "label_raw": str(result_df.at[idx, "label_raw"]) if "label_raw" in result_df.columns else "",
                        "label_is_correct": (
                            float(result_df.at[idx, "label_is_correct"])
                            if "label_is_correct" in result_df.columns and pd.notna(result_df.at[idx, "label_is_correct"])
                            else np.nan
                        ),
                        "source_input": str(source_df.iloc[idx][args.input_col]),
                        "source_output": str(source_df.iloc[idx][args.output_col]),
                        "input_bundle_nomask": str(result_df.at[idx, "input_bundle_nomask"]),
                        "output_bundle_nomask": str(result_df.at[idx, "output_bundle_nomask"]),
                    }
                )

    out_df = pd.DataFrame(rows)
    out_path = output_dir / f"{tag}_{stem}_top_anomalies.csv"
    out_df.to_csv(out_path, index=False)
    return out_path


def run_distribution_mode(
    mode_name: str,
    input_texts: list[str],
    output_texts: list[str],
    source_output_texts: list[str],
    model: TextEmbedder,
    ref_mask: np.ndarray,
    active_rules: list[str],
    rule_quantiles: dict[str, float],
    refine_reference: bool,
    refine_rules: list[str],
    refine_iterations: int,
    refine_min_size: int,
    args: argparse.Namespace,
    output_dir: Path,
    stem: str,
    tag: str,
    group_id: str,
    label_is_correct: np.ndarray | None = None,
    label_raw: np.ndarray | None = None,
    cache_writer: EmbeddingCacheWriter | None = None,
    cache_row_indices: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    delta_ens_ranks = list(SIGNAL_RUNTIME.delta_ens_ranks)
    scorer = DistributionScorer(
        embedder=model,
        var_target=SIGNAL_RUNTIME.var_target,
        pca_min_dims=SIGNAL_RUNTIME.pca_min_dims,
        pca_max_dims=SIGNAL_RUNTIME.pca_max_dims,
        min_k=SIGNAL_RUNTIME.min_k,
        max_k=SIGNAL_RUNTIME.max_k,
        gap_ratio=SIGNAL_RUNTIME.k_gap_ratio,
        cov_shrinkage=SIGNAL_RUNTIME.cov_shrinkage,
        diff_residual_aux_enabled=bool(args.diff_residual_aux_enabled),
        diff_residual_aux_lambda=float(args.diff_residual_aux_lambda),
        diff_residual_aux_model=str(args.diff_residual_aux_model),
        diff_residual_row_chunk_workers=int(SIGNAL_RUNTIME.diff_residual_row_chunk_workers),
        delta_ens_rp_dims=SIGNAL_RUNTIME.delta_ens_rp_dims,
        delta_ens_alpha=SIGNAL_RUNTIME.delta_ens_alpha,
        delta_ens_cv_mode=SIGNAL_RUNTIME.delta_ens_cv_mode,
        delta_ens_kfolds=SIGNAL_RUNTIME.delta_ens_kfolds,
        delta_ens_split_train_ratio=SIGNAL_RUNTIME.delta_ens_split_train_ratio,
        delta_ens_random_state=SIGNAL_RUNTIME.delta_ens_random_state,
        delta_ens_residual=SIGNAL_RUNTIME.delta_ens_residual,
        delta_ens_fit_intercept=bool(SIGNAL_RUNTIME.delta_ens_fit_intercept),
        delta_ens_members_nystrom=SIGNAL_RUNTIME.delta_ens_members_nystrom,
        delta_ens_members_lowrank=SIGNAL_RUNTIME.delta_ens_members_lowrank,
        delta_ens_row_subsample=SIGNAL_RUNTIME.delta_ens_row_subsample,
        delta_ens_ranks=delta_ens_ranks,
        delta_ens_landmark_policy=SIGNAL_RUNTIME.delta_ens_landmark_policy,
        delta_ens_landmark_cap=SIGNAL_RUNTIME.delta_ens_landmark_cap,
        delta_ens_fusion=SIGNAL_RUNTIME.delta_ens_fusion,
        delta_ens_debug_members=bool(SIGNAL_RUNTIME.delta_ens_debug_members),
        similarity_threshold=float(GATE_RUNTIME.similarity_threshold),
        similarity_k=int(GATE_RUNTIME.similarity_k),
        enable_discourse_instability_rule=bool(SEMANTIC_RUNTIME.enable_discourse_instability_rule),
        discourse_instability_min_support_ratio=float(SEMANTIC_RUNTIME.discourse_instability_min_support_ratio),
        discourse_instability_min_class_size=int(SEMANTIC_RUNTIME.discourse_instability_min_class_size),
        discourse_instability_max_classes=int(SEMANTIC_RUNTIME.discourse_instability_max_classes),
        discourse_instability_signal_bins=int(SEMANTIC_RUNTIME.discourse_instability_signal_bins),
        discourse_instability_min_evidence_tokens=int(SEMANTIC_RUNTIME.discourse_instability_min_evidence_tokens),
        discourse_instability_intra_weight=float(SEMANTIC_RUNTIME.discourse_instability_intra_weight),
        discourse_instability_cross_weight=float(SEMANTIC_RUNTIME.discourse_instability_cross_weight),
        discourse_instability_candidate_keys=list(SEMANTIC_RUNTIME.discourse_instability_candidate_keys),
        discourse_instability_evidence_keys=list(SEMANTIC_RUNTIME.discourse_instability_evidence_keys),
        enable_contradiction_rule=bool(SEMANTIC_RUNTIME.enable_contradiction_rule),
        contradiction_min_support_ratio=float(SEMANTIC_RUNTIME.contradiction_min_support_ratio),
        contradiction_min_class_size=int(SEMANTIC_RUNTIME.contradiction_min_class_size),
        contradiction_max_classes=int(SEMANTIC_RUNTIME.contradiction_max_classes),
        contradiction_signal_bins=int(SEMANTIC_RUNTIME.contradiction_signal_bins),
        contradiction_candidate_keys=list(SEMANTIC_RUNTIME.contradiction_candidate_keys),
        contradiction_evidence_keys=list(SEMANTIC_RUNTIME.contradiction_evidence_keys),
        contradiction_min_evidence_tokens=int(SEMANTIC_RUNTIME.contradiction_min_evidence_tokens),
        enable_self_contradiction_rule=bool(SEMANTIC_RUNTIME.enable_self_contradiction_rule),
        self_contradiction_min_support_ratio=float(SEMANTIC_RUNTIME.self_contradiction_min_support_ratio),
        self_contradiction_min_class_size=int(SEMANTIC_RUNTIME.self_contradiction_min_class_size),
        self_contradiction_max_classes=int(SEMANTIC_RUNTIME.self_contradiction_max_classes),
        self_contradiction_signal_bins=int(SEMANTIC_RUNTIME.self_contradiction_signal_bins),
        self_contradiction_candidate_keys=list(SEMANTIC_RUNTIME.self_contradiction_candidate_keys),
        self_contradiction_evidence_keys=list(SEMANTIC_RUNTIME.self_contradiction_evidence_keys),
        self_contradiction_min_evidence_tokens=int(SEMANTIC_RUNTIME.self_contradiction_min_evidence_tokens),
    )

    output_dicts = [safe_json_load(x) if isinstance(x, str) else x for x in source_output_texts]
    output_dicts = [
        {str(k).strip().lower(): v for k, v in (parsed.items() if isinstance(parsed, dict) else [])}
        if isinstance(parsed, dict)
        else None
        for parsed in output_dicts
    ]

    n_rows = len(input_texts)
    ref_mask_used = np.asarray(ref_mask, dtype=bool).copy()
    if ref_mask_used.sum() == 0:
        ref_mask_used = np.ones(n_rows, dtype=bool)

    udf_enabled = bool(UDF_FIXED_ENABLED)
    udf_core_rules = list(UDF_FIXED_CORE_RULES)
    sample_weights = np.ones(n_rows, dtype=float)
    udf_rounds = 0
    udf_converged = True

    result = None
    signal_map: dict[str, np.ndarray] = {}
    signal_available_map: dict[str, np.ndarray] = {}

    n_iters = int(max(1, UDF_FIXED_ITERATIONS))
    for iter_idx in range(n_iters):
        result = scorer.compute_signals(
            input_texts=input_texts,
            output_texts=output_texts,
            source_output_texts=source_output_texts,
            ref_mask=ref_mask_used,
            output_dicts=output_dicts,
            batch_size=args.embedding_batch_size,
            sample_weights=sample_weights,
        )

        output_signal = result.output_signal
        direction_signal = result.direction_signal
        length_signal = result.length_signal
        diff_resid_signal = result.diff_residual_signal
        delta_ridge_ens_signal = result.delta_ridge_ens_signal
        sim_conflict_signal = result.similar_input_conflict_signal
        discourse_instability_signal = result.discourse_instability_signal
        discourse_instability_available = np.asarray(result.discourse_instability_available, dtype=bool)
        contradiction_signal = result.contradiction_signal
        contradiction_available = np.asarray(result.contradiction_available, dtype=bool)
        self_contradiction_signal = result.self_contradiction_signal
        self_contradiction_available = np.asarray(result.self_contradiction_available, dtype=bool)

        signal_map = {
            "output": output_signal,
            "direction": direction_signal,
            "length": length_signal,
            "diff_resid": diff_resid_signal,
            "delta_ridge_ens": delta_ridge_ens_signal,
            "sim_conflict": sim_conflict_signal,
            "discourse_instability": discourse_instability_signal,
            "contradiction": contradiction_signal,
            "self_contradiction": self_contradiction_signal,
        }
        signal_available_map = {
            "output": np.ones(n_rows, dtype=bool),
            "direction": np.ones(n_rows, dtype=bool),
            "length": np.ones(n_rows, dtype=bool),
            "diff_resid": np.ones(n_rows, dtype=bool),
            "delta_ridge_ens": np.ones(n_rows, dtype=bool),
            "sim_conflict": np.ones(n_rows, dtype=bool),
            "discourse_instability": discourse_instability_available,
            "contradiction": contradiction_available,
            "self_contradiction": self_contradiction_available,
        }

        clean_q = float(np.clip(UDF_FIXED_Q_CLEAN, 0.01, 0.99))
        clean_q_map = {k: float(rule_quantiles.get(k, SIGNAL_RUNTIME.signal_quantile)) for k in RULE_KEYS}
        for rk in udf_core_rules:
            clean_q_map[rk] = clean_q

        clean_th = compute_thresholds(
            signal_map=signal_map,
            ref_mask=ref_mask_used,
            rule_quantiles=clean_q_map,
            ref_weights=sample_weights,
            signal_available_map=signal_available_map,
        )

        risk = np.zeros(n_rows, dtype=float)
        for rk in udf_core_rules:
            if rk not in signal_map:
                continue
            vals = np.asarray(signal_map[rk], dtype=float)
            avail = np.asarray(signal_available_map.get(rk, np.ones(n_rows, dtype=bool)), dtype=bool)
            th = float(clean_th.get(rk, np.nan))
            if not np.isfinite(th):
                continue
            denom = max(abs(th), 1e-6)
            exceed = np.maximum(0.0, (vals - th) / denom)
            exceed[~avail] = 0.0
            risk += exceed
        if len(udf_core_rules) > 0:
            risk /= float(len(udf_core_rules))

        risk_rank = np.zeros(n_rows, dtype=float)
        if n_rows > 0:
            order = np.argsort(risk)
            risk_rank[order] = (np.arange(n_rows) + 1) / max(float(n_rows), 1.0)

        alpha = float(max(0.0, UDF_FIXED_SOFT_ALPHA))
        min_w = float(np.clip(UDF_FIXED_MIN_WEIGHT, 0.0, 1.0))
        new_weights = np.exp(-alpha * np.maximum(0.0, risk_rank - clean_q))
        new_weights = np.clip(new_weights, min_w, 1.0)
        new_weights = np.where(ref_mask_used, new_weights, 1.0)

        udf_rounds = iter_idx + 1
        delta = float(np.mean(np.abs(new_weights - sample_weights))) if iter_idx > 0 else np.inf
        sample_weights = new_weights

        if iter_idx > 0 and delta < 1e-3:
            udf_converged = True
            break
        if iter_idx == n_iters - 1:
            udf_converged = False

    if result is None:
        raise RuntimeError("Distribution signaling failed to produce a result.")

    output_signal = result.output_signal
    direction_signal = result.direction_signal
    length_signal = result.length_signal
    diff_resid_signal = result.diff_residual_signal
    delta_ridge_ens_signal = result.delta_ridge_ens_signal
    sim_conflict_signal = result.similar_input_conflict_signal
    discourse_instability_signal = result.discourse_instability_signal
    discourse_instability_available = np.asarray(result.discourse_instability_available, dtype=bool)
    discourse_instability_meta = dict(result.discourse_instability_meta or {})
    contradiction_signal = result.contradiction_signal
    contradiction_available = np.asarray(result.contradiction_available, dtype=bool)
    contradiction_meta = dict(result.contradiction_meta or {})
    self_contradiction_signal = result.self_contradiction_signal
    self_contradiction_available = np.asarray(result.self_contradiction_available, dtype=bool)
    self_contradiction_meta = dict(result.self_contradiction_meta or {})
    consistency_signal = result.consistency_rule_signal
    consistency_available = np.asarray(result.consistency_rule_available, dtype=bool)
    local_k = result.local_k
    tau = result.tau
    output_ks = result.used_output_density_ks
    x_norm = result.x_norm
    y_norm = result.y_norm
    if cache_writer is not None and cache_row_indices is not None:
        cache_writer.write(
            row_indices=np.asarray(cache_row_indices, dtype=int),
            input_norm=np.asarray(x_norm, dtype=np.float32),
            output_norm=np.asarray(y_norm, dtype=np.float32),
        )
    diff_residual_meta = dict(result.diff_residual_meta or {})
    delta_ridge_ens_meta = dict(result.delta_ridge_ens_meta or {})

    x_meta = {"used_dims": x_norm.shape[1], "target_var_reached": None, "pca_first2_var": None}
    y_meta = {"used_dims": y_norm.shape[1], "target_var_reached": None, "pca_first2_var": None}
    d_meta = {"used_dims": x_norm.shape[1], "target_var_reached": None, "pca_first2_var": None}
    d_mahal_dims = int(diff_residual_meta.get("reduced_dims", x_norm.shape[1]))
    diff_resid_method = str(diff_residual_meta.get("method", "local_mahalanobis")).strip().lower()
    diff_residual_method_label = "local_mahalanobis_ledoitwolf" if _LedoitWolf is not None else "local_mahalanobis_shrunk_fallback"
    consistency_meta = {
        "selected_verdict_key": "",
        "selected_evidence_key": "",
        "available_rows": int(consistency_available.sum()),
        "conflict_rows": 0,
        "conflict_rate": 0.0,
    }

    thresholds = compute_thresholds(
        signal_map=signal_map,
        ref_mask=ref_mask_used,
        rule_quantiles=rule_quantiles,
        ref_weights=sample_weights,
        signal_available_map=signal_available_map,
    )
    refine_rounds = 0
    if refine_reference and not udf_enabled:
        ref_mask_used, thresholds, refine_rounds = refine_reference_mask(
            signal_map=signal_map,
            initial_ref_mask=ref_mask_used,
            rule_quantiles=rule_quantiles,
            refine_rules=refine_rules,
            iterations=refine_iterations,
            min_size=max(refine_min_size, min(int(GATE_RUNTIME.min_reference_size), len(ref_mask_used))),
            signal_available_map=signal_available_map,
        )

    output_pass = output_signal <= thresholds["output"]
    direction_pass = direction_signal <= thresholds["direction"]
    length_pass = length_signal <= thresholds["length"]
    diff_resid_pass = diff_resid_signal <= thresholds["diff_resid"]
    delta_ridge_ens_pass = delta_ridge_ens_signal <= thresholds["delta_ridge_ens"]
    sim_conflict_pass = sim_conflict_signal <= thresholds["sim_conflict"]
    discourse_instability_pass = np.ones(len(output_pass), dtype=bool)
    discourse_instability_pass[discourse_instability_available] = (
        discourse_instability_signal[discourse_instability_available] <= thresholds["discourse_instability"]
    )
    contradiction_pass = np.ones(len(output_pass), dtype=bool)
    contradiction_pass[contradiction_available] = (
        contradiction_signal[contradiction_available] <= thresholds["contradiction"]
    )
    self_contradiction_pass = np.ones(len(output_pass), dtype=bool)
    self_contradiction_pass[self_contradiction_available] = (
        self_contradiction_signal[self_contradiction_available] <= thresholds["self_contradiction"]
    )

    rule_pass_map = {
        "output": output_pass,
        "direction": direction_pass,
        "length": length_pass,
        "diff_resid": diff_resid_pass,
        "delta_ridge_ens": delta_ridge_ens_pass,
        "sim_conflict": sim_conflict_pass,
        "discourse_instability": discourse_instability_pass,
        "contradiction": contradiction_pass,
        "self_contradiction": self_contradiction_pass,
    }
    distribution_pass = np.ones(len(output_pass), dtype=bool)
    for key in active_rules:
        distribution_pass &= rule_pass_map[key]

    base_name = f"{tag}_{stem}_{group_id}"
    stacked = np.vstack([x_norm, y_norm])
    coords2d, coords_evr = pca_fit_transform(stacked, n_components=2)
    coords2d = ensure_2d_coords(coords2d)
    in_2d = coords2d[: len(x_norm)]
    out_2d = coords2d[len(x_norm) :]
    pca_var2 = float(np.sum(coords_evr[:2])) if len(coords_evr) > 0 else None

    discourse_unavailable_reason = str(discourse_instability_meta.get("unavailable_reason", "") or "")
    contradiction_unavailable_reason = str(contradiction_meta.get("unavailable_reason", "") or "")
    self_contra_unavailable_reason = str(self_contradiction_meta.get("unavailable_reason", "") or "")

    result_df = pd.DataFrame(
        {
            f"output_signal_{mode_name}": output_signal,
            f"direction_signal_{mode_name}": direction_signal,
            f"length_signal_{mode_name}": length_signal,
            f"diff_residual_signal_{mode_name}": diff_resid_signal,
            f"delta_ridge_ens_signal_{mode_name}": delta_ridge_ens_signal,
            f"similar_input_conflict_signal_{mode_name}": sim_conflict_signal,
            f"discourse_instability_signal_{mode_name}": discourse_instability_signal,
            f"contradiction_signal_{mode_name}": contradiction_signal,
            f"self_contradiction_signal_{mode_name}": self_contradiction_signal,
            f"consistency_rule_signal_{mode_name}": consistency_signal,
            f"consistency_rule_available_{mode_name}": consistency_available,
            f"output_pass_{mode_name}": output_pass,
            f"direction_pass_{mode_name}": direction_pass,
            f"length_pass_{mode_name}": length_pass,
            f"diff_residual_pass_{mode_name}": diff_resid_pass,
            f"delta_ridge_ens_pass_{mode_name}": delta_ridge_ens_pass,
            f"similar_input_conflict_pass_{mode_name}": sim_conflict_pass,
            f"discourse_instability_pass_{mode_name}": discourse_instability_pass,
            f"contradiction_pass_{mode_name}": contradiction_pass,
            f"self_contradiction_pass_{mode_name}": self_contradiction_pass,
            f"discourse_instability_available_{mode_name}": discourse_instability_available,
            f"contradiction_available_{mode_name}": contradiction_available,
            f"self_contradiction_available_{mode_name}": self_contradiction_available,
            f"discourse_instability_verdict_key_{mode_name}": discourse_instability_meta.get("selected_verdict_key", ""),
            f"contradiction_verdict_key_{mode_name}": contradiction_meta.get("selected_verdict_key", ""),
            f"self_contradiction_verdict_key_{mode_name}": self_contradiction_meta.get("selected_verdict_key", ""),
            f"discourse_instability_evidence_key_{mode_name}": discourse_instability_meta.get("selected_evidence_key", ""),
            f"contradiction_evidence_key_{mode_name}": contradiction_meta.get("selected_evidence_key", ""),
            f"self_contradiction_evidence_key_{mode_name}": self_contradiction_meta.get("selected_evidence_key", ""),
            f"contradiction_unavailable_reason_{mode_name}": np.where(
                contradiction_available, "", contradiction_unavailable_reason
            ),
            f"self_contradiction_unavailable_reason_{mode_name}": np.where(
                self_contradiction_available, "", self_contra_unavailable_reason
            ),
            f"distribution_pass_{mode_name}": distribution_pass,
            f"local_k_{mode_name}": local_k,
            f"tau_{mode_name}": tau,
            f"udf_weight_{mode_name}": sample_weights,
            f"udf_round_{mode_name}": int(udf_rounds),
            f"udf_ref_selected_{mode_name}": ref_mask_used.astype(bool),
            f"input_pca_x_{mode_name}": in_2d[:, 0],
            f"input_pca_y_{mode_name}": in_2d[:, 1],
            f"output_pca_x_{mode_name}": out_2d[:, 0],
            f"output_pca_y_{mode_name}": out_2d[:, 1],
        }
    )
    if bool(SIGNAL_RUNTIME.delta_ens_debug_members):
        for k, v in delta_ridge_ens_meta.items():
            k_text = str(k)
            if not (k_text.startswith("member_signal_") or k_text.startswith("member_score_")):
                continue
            arr = np.asarray(v, dtype=float).ravel()
            if len(arr) != len(result_df):
                continue
            member_name = k_text.replace("member_signal_", "", 1).replace("member_score_", "", 1)
            result_df[f"delta_ridge_ens_member_{member_name}_signal_{mode_name}"] = arr

    hist_data_path = save_signal_hist_data(
        mode_name=mode_name,
        signal_map=signal_map,
        thresholds=thresholds,
        ref_mask=ref_mask_used,
        bins=args.hist_bins,
        output_dir=output_dir,
        base_name=base_name,
    )
    arrow_pca_data_path = save_arrow_data(
        mode_name=mode_name,
        method_name="pca",
        in_2d=in_2d,
        out_2d=out_2d,
        anomaly_mask=~distribution_pass,
        conflict_mask=~sim_conflict_pass,
        output_dir=output_dir,
        base_name=base_name,
        label_is_correct=label_is_correct,
        label_raw=label_raw,
        pred_pass=distribution_pass,
    )

    hist_path = None
    arrow_pca_path = None
    arrow_umap_path = None
    arrow_umap_data_path = None
    hist_path = save_signal_histograms(
        mode_name=mode_name,
        signal_map=signal_map,
        thresholds=thresholds,
        ref_mask=ref_mask_used,
        bins=args.hist_bins,
        output_dir=output_dir,
        base_name=base_name,
    )

    arrow_pca_path = save_arrow_plot(
        mode_name=mode_name,
        method_name="pca",
        in_2d=in_2d,
        out_2d=out_2d,
        anomaly_mask=~distribution_pass,
        conflict_mask=~sim_conflict_pass,
        max_arrows=args.plot_max_arrows,
        seed=args.plot_seed,
        output_dir=output_dir,
        base_name=base_name,
        label_is_correct=label_is_correct,
        pred_pass=distribution_pass,
        pred_name="distribution_pass",
    )

    if umap_module is not None and len(stacked) >= 4:
        try:
            n_neighbors = min(30, max(4, len(stacked) - 1))
            reducer = umap_module.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=0.05,
                metric="cosine",
                random_state=args.plot_seed,
            )
            u2d = reducer.fit_transform(stacked)
            in_u2d = u2d[: len(x_norm)]
            out_u2d = u2d[len(x_norm) :]
            arrow_umap_data_path = save_arrow_data(
                mode_name=mode_name,
                method_name="umap",
                in_2d=in_u2d,
                out_2d=out_u2d,
                anomaly_mask=~distribution_pass,
                conflict_mask=~sim_conflict_pass,
                output_dir=output_dir,
                base_name=base_name,
                label_is_correct=label_is_correct,
                label_raw=label_raw,
                pred_pass=distribution_pass,
            )
            arrow_umap_path = save_arrow_plot(
                mode_name=mode_name,
                method_name="umap",
                in_2d=in_u2d,
                out_2d=out_u2d,
                anomaly_mask=~distribution_pass,
                conflict_mask=~sim_conflict_pass,
                max_arrows=args.plot_max_arrows,
                seed=args.plot_seed,
                output_dir=output_dir,
                base_name=base_name,
                label_is_correct=label_is_correct,
                pred_pass=distribution_pass,
                pred_name="distribution_pass",
            )
        except Exception:
            arrow_umap_path = None
            arrow_umap_data_path = None

    summary = {
        "mode": mode_name,
        "group_id": group_id,
        "rows": int(len(result_df)),
        "output_density_k_candidates": ",".join(map(str, output_ks)),
        "local_k_mean": float(np.mean(local_k)),
        "local_k_min": int(np.min(local_k)),
        "local_k_max": int(np.max(local_k)),
        "x_dims": x_meta["used_dims"],
        "y_dims": y_meta["used_dims"],
        "d_dims": d_meta["used_dims"],
        "d_mahal_dims": d_mahal_dims,
        "diff_residual_method": diff_residual_method_label,
        "diff_residual_method_family": diff_resid_method,
        "diff_residual_reduced_dims": int(diff_residual_meta.get("reduced_dims", d_mahal_dims)),
        "diff_residual_input_dims": int(diff_residual_meta.get("input_dims", x_norm.shape[1])),
        "diff_residual_target_dims": int(diff_residual_meta.get("target_dims", d_mahal_dims)),
        "diff_residual_alpha": np.nan,
        "diff_residual_cv_mode": "",
        "diff_residual_residual_norm": "",
        "diff_residual_rp_used": False,
        "diff_residual_fit_intercept": np.nan,
        "delta_ridge_ens_method": str(delta_ridge_ens_meta.get("method", "delta_ridge_ens")),
        "delta_ridge_ens_cv_mode": str(delta_ridge_ens_meta.get("cv_mode", SIGNAL_RUNTIME.delta_ens_cv_mode)),
        "delta_ridge_ens_rp_dims": int(delta_ridge_ens_meta.get("rp_dims", SIGNAL_RUNTIME.delta_ens_rp_dims)),
        "delta_ridge_ens_target_dims": int(delta_ridge_ens_meta.get("target_dims", x_norm.shape[1])),
        "delta_ridge_ens_alpha": float(delta_ridge_ens_meta.get("alpha", SIGNAL_RUNTIME.delta_ens_alpha)),
        "delta_ridge_ens_residual_norm": str(delta_ridge_ens_meta.get("residual_norm", SIGNAL_RUNTIME.delta_ens_residual)),
        "delta_ridge_ens_fit_intercept": bool(delta_ridge_ens_meta.get("fit_intercept", SIGNAL_RUNTIME.delta_ens_fit_intercept)),
        "delta_ridge_ens_members_nystrom": int(delta_ridge_ens_meta.get("members_nystrom", SIGNAL_RUNTIME.delta_ens_members_nystrom)),
        "delta_ridge_ens_members_lowrank": int(delta_ridge_ens_meta.get("members_lowrank", SIGNAL_RUNTIME.delta_ens_members_lowrank)),
        "delta_ridge_ens_members_total": int(
            delta_ridge_ens_meta.get(
                "members_total",
                SIGNAL_RUNTIME.delta_ens_members_nystrom + SIGNAL_RUNTIME.delta_ens_members_lowrank,
            )
        ),
        "delta_ridge_ens_row_subsample": float(delta_ridge_ens_meta.get("row_subsample", SIGNAL_RUNTIME.delta_ens_row_subsample)),
        "delta_ridge_ens_ranks": str(delta_ridge_ens_meta.get("ranks", ",".join(map(str, SIGNAL_RUNTIME.delta_ens_ranks)))),
        "delta_ridge_ens_landmark_policy": str(delta_ridge_ens_meta.get("landmark_policy", SIGNAL_RUNTIME.delta_ens_landmark_policy)),
        "delta_ridge_ens_landmark_cap": int(delta_ridge_ens_meta.get("landmark_cap", SIGNAL_RUNTIME.delta_ens_landmark_cap)),
        "delta_ridge_ens_fusion": str(delta_ridge_ens_meta.get("fusion", SIGNAL_RUNTIME.delta_ens_fusion)),
        "delta_ridge_ens_debug_members": bool(delta_ridge_ens_meta.get("debug_members", SIGNAL_RUNTIME.delta_ens_debug_members)),
        "x_first2_var": x_meta["pca_first2_var"],
        "y_first2_var": y_meta["pca_first2_var"],
        "d_first2_var": d_meta["pca_first2_var"],
        "arrow_pca_var2": pca_var2,
        "output_threshold": thresholds["output"],
        "direction_threshold": thresholds["direction"],
        "length_threshold": thresholds["length"],
        "diff_resid_threshold": thresholds["diff_resid"],
        "delta_ridge_ens_threshold": thresholds["delta_ridge_ens"],
        "sim_conflict_threshold": thresholds["sim_conflict"],
        "discourse_instability_threshold": thresholds["discourse_instability"],
        "contradiction_threshold": thresholds["contradiction"],
        "self_contradiction_threshold": thresholds["self_contradiction"],
        "output_outlier_rate": float((~output_pass).mean()),
        "direction_outlier_rate": float((~direction_pass).mean()),
        "length_outlier_rate": float((~length_pass).mean()),
        "diff_resid_outlier_rate": float((~diff_resid_pass).mean()),
        "delta_ridge_ens_outlier_rate": float((~delta_ridge_ens_pass).mean()),
        "sim_conflict_outlier_rate": float((~sim_conflict_pass).mean()),
        "discourse_instability_outlier_rate": float((~discourse_instability_pass).mean()),
        "contradiction_outlier_rate": float((~contradiction_pass).mean()),
        "self_contradiction_outlier_rate": float((~self_contradiction_pass).mean()),
        "discourse_instability_available_rate": float(np.mean(np.asarray(discourse_instability_available, dtype=bool))),
        "contradiction_available_rate": float(np.mean(np.asarray(contradiction_available, dtype=bool))),
        "self_contradiction_available_rate": float(np.mean(np.asarray(self_contradiction_available, dtype=bool))),
        "discourse_instability_selected_key": discourse_instability_meta.get("selected_verdict_key", ""),
        "contradiction_selected_key": contradiction_meta.get("selected_verdict_key", ""),
        "self_contradiction_selected_key": self_contradiction_meta.get("selected_verdict_key", ""),
        "discourse_instability_selected_evidence_key": discourse_instability_meta.get("selected_evidence_key", ""),
        "contradiction_selected_evidence_key": contradiction_meta.get("selected_evidence_key", ""),
        "self_contradiction_selected_evidence_key": self_contradiction_meta.get("selected_evidence_key", ""),
        "discourse_instability_class_count": int(discourse_instability_meta.get("class_count", 0)),
        "contradiction_class_count": int(contradiction_meta.get("class_count", 0)),
        "self_contradiction_class_count": int(self_contradiction_meta.get("class_count", 0)),
        "discourse_instability_eligible_rows": int(discourse_instability_meta.get("eligible_rows", 0)),
        "contradiction_eligible_rows": int(contradiction_meta.get("eligible_rows", 0)),
        "self_contradiction_eligible_rows": int(self_contradiction_meta.get("eligible_rows", 0)),
        "discourse_instability_unavailable_reason": discourse_instability_meta.get("unavailable_reason", ""),
        "contradiction_unavailable_reason": contradiction_meta.get("unavailable_reason", ""),
        "self_contradiction_unavailable_reason": self_contradiction_meta.get("unavailable_reason", ""),
        "distribution_pass_rate": float(distribution_pass.mean()),
        "active_distribution_rules": ",".join(active_rules),
        "rule_quantiles_json": json.dumps(rule_quantiles, ensure_ascii=False),
        "consistency_observe_only": bool(GATE_RUNTIME.consistency_observe_only),
        "consistency_available_rate": float(np.mean(consistency_available.astype(bool))),
        "consistency_conflict_rate": float(consistency_meta.get("conflict_rate", 0.0)),
        "consistency_selected_key": consistency_meta.get("selected_verdict_key", ""),
        "consistency_selected_evidence_key": consistency_meta.get("selected_evidence_key", ""),
        "threshold_refine_enabled": bool(refine_reference and (not udf_enabled)),
        "threshold_refine_rules": ",".join(refine_rules),
        "threshold_refine_rounds": int(refine_rounds),
        "threshold_ref_size": int(ref_mask_used.sum()),
        "udf_enabled": bool(udf_enabled),
        "udf_iterations_applied": int(udf_rounds),
        "udf_core_rules": ",".join(udf_core_rules),
        "udf_weight_mean": float(np.mean(sample_weights)) if len(sample_weights) > 0 else 1.0,
        "udf_weight_p10": float(np.percentile(sample_weights, 10)) if len(sample_weights) > 0 else 1.0,
        "udf_weight_p50": float(np.percentile(sample_weights, 50)) if len(sample_weights) > 0 else 1.0,
        "udf_weight_p90": float(np.percentile(sample_weights, 90)) if len(sample_weights) > 0 else 1.0,
        "udf_converged": bool(udf_converged),
        "label_known_ratio": (
            float(np.mean(np.isfinite(np.asarray(label_is_correct, dtype=float))))
            if label_is_correct is not None and len(label_is_correct) == len(result_df)
            else 0.0
        ),
        "signal_hist_data_path": str(hist_data_path),
        "score_hist_data_path": str(hist_data_path),
        "arrow_pca_data_path": str(arrow_pca_data_path),
        "arrow_umap_data_path": str(arrow_umap_data_path) if arrow_umap_data_path else "",
        "signal_hist_path": str(hist_path) if hist_path else "",
        "score_hist_path": str(hist_path) if hist_path else "",
        "arrow_pca_path": str(arrow_pca_path) if arrow_pca_path else "",
        "arrow_umap_path": str(arrow_umap_path) if arrow_umap_path else "",
    }
    return result_df, summary


def collect_mode_text_triplet_by_indices(
    *,
    row_indices: np.ndarray,
    input_text_arr: np.ndarray,
    output_text_arr: np.ndarray,
    source_output_arr: np.ndarray,
) -> tuple[list[str], list[str], list[str]]:
    idx = np.asarray(row_indices, dtype=int)
    in_texts = np.asarray(input_text_arr, dtype=object)[idx].astype(str).tolist()
    out_texts = np.asarray(output_text_arr, dtype=object)[idx].astype(str).tolist()
    src_out_texts = np.asarray(source_output_arr, dtype=object)[idx].astype(str).tolist()
    return in_texts, out_texts, src_out_texts


def run_distribution_mode_job(
    *,
    mode_name: str,
    input_texts: list[str],
    output_texts: list[str],
    source_output_texts: list[str],
    model: TextEmbedder,
    active_rules: list[str],
    rule_quantiles: dict[str, float],
    refine_reference: bool,
    refine_rules: list[str],
    refine_iterations: int,
    refine_min_size: int,
    args: argparse.Namespace,
    output_dir: Path,
    stem: str,
    tag: str,
    group_id: str,
    label_is_correct: np.ndarray | None = None,
    label_raw: np.ndarray | None = None,
    cache_writer: EmbeddingCacheWriter | None = None,
    cache_row_indices: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray]:
    n = len(input_texts)
    if len(output_texts) != n or len(source_output_texts) != n:
        raise ValueError("Distribution mode job input length mismatch.")
    ref_mask = np.ones(n, dtype=bool)
    mode_df, mode_summary = run_distribution_mode(
        mode_name=mode_name,
        input_texts=input_texts,
        output_texts=output_texts,
        source_output_texts=source_output_texts,
        model=model,
        ref_mask=ref_mask,
        active_rules=active_rules,
        rule_quantiles=rule_quantiles,
        refine_reference=refine_reference,
        refine_rules=refine_rules,
        refine_iterations=refine_iterations,
        refine_min_size=refine_min_size,
        args=args,
        output_dir=output_dir,
        stem=stem,
        tag=tag,
        group_id=group_id,
        label_is_correct=label_is_correct,
        label_raw=label_raw,
        cache_writer=cache_writer,
        cache_row_indices=cache_row_indices,
    )
    return mode_df, mode_summary, ref_mask


def run_detailed_leaf_distribution_gate(
    *,
    input_text_series: np.ndarray,
    output_series: pd.Series,
    group_ids: pd.Series,
    row_ids: pd.Series,
    hard_gate_mask: np.ndarray,
    model: TextEmbedder,
    active_rules: list[str],
    rule_quantiles: dict[str, float],
    refine_reference: bool,
    refine_rules: list[str],
    refine_iterations: int,
    refine_min_size: int,
    args: argparse.Namespace,
    output_dir: Path,
    stem: str,
    tag: str,
    min_support: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    n = len(output_series)
    if len(input_text_series) != n or len(group_ids) != n or len(row_ids) != n or len(hard_gate_mask) != n:
        raise ValueError("Detailed leaf distribution input length mismatch.")

    min_support = int(max(1, min_support))
    row_leaf_maps = build_output_string_leaf_maps(output_series)
    gid_values = group_ids.fillna("g0000").astype(str).to_numpy(dtype=object)
    rid_values = row_ids.fillna("").astype(str).to_numpy(dtype=object)
    hard_gate = np.asarray(hard_gate_mask, dtype=bool)

    summary_rows: list[dict[str, Any]] = []
    row_rows: list[dict[str, Any]] = []
    evaluated_paths = 0

    for gid in pd.unique(gid_values):
        idx_group = np.where(gid_values == gid)[0]
        if len(idx_group) == 0:
            continue
        idx_eval = idx_group[hard_gate[idx_group]]
        if len(idx_eval) == 0:
            continue
        all_paths = sorted({p for i in idx_eval for p in row_leaf_maps[int(i)].keys()})
        for leaf_path in all_paths:
            idx_leaf: list[int] = []
            leaf_texts: list[str] = []
            for i in idx_eval:
                text = str(row_leaf_maps[int(i)].get(leaf_path, ""))
                if text.strip():
                    idx_leaf.append(int(i))
                    leaf_texts.append(text)
            support = len(idx_leaf)
            if support < min_support:
                continue

            evaluated_paths += 1
            idx_leaf_arr = np.asarray(idx_leaf, dtype=int)
            in_texts = np.asarray(input_text_series, dtype=object)[idx_leaf_arr].astype(str).tolist()
            source_out = list(leaf_texts)
            safe_leaf = sanitize_leaf_path_token(leaf_path)
            leaf_group_id = f"{str(gid)}__leaf__{safe_leaf}"

            mode_df, mode_summary, _ = run_distribution_mode_job(
                mode_name="nomask",
                input_texts=in_texts,
                output_texts=leaf_texts,
                source_output_texts=source_out,
                model=model,
                active_rules=active_rules,
                rule_quantiles=rule_quantiles,
                refine_reference=refine_reference,
                refine_rules=refine_rules,
                refine_iterations=refine_iterations,
                refine_min_size=refine_min_size,
                args=args,
                output_dir=output_dir,
                stem=stem,
                tag=f"{tag}_leafgate",
                group_id=leaf_group_id,
                label_is_correct=None,
                label_raw=None,
                cache_writer=None,
                cache_row_indices=None,
            )

            summary_row = dict(mode_summary)
            summary_row["group_id_raw"] = str(gid)
            summary_row["leaf_path"] = str(leaf_path)
            summary_row["leaf_support"] = int(support)
            summary_rows.append(summary_row)

            dist_col = "distribution_pass_nomask"
            for j, row_idx in enumerate(idx_leaf):
                rec: dict[str, Any] = {
                    "row_index": int(row_idx),
                    "row_id": str(rid_values[row_idx]),
                    "group_id": str(gid),
                    "leaf_path": str(leaf_path),
                    "leaf_support": int(support),
                    "distribution_pass_leaf_nomask": bool(mode_df.at[j, dist_col]) if dist_col in mode_df.columns else True,
                }
                failed_rules: list[str] = []
                for rule in active_rules:
                    sig_col = f"{RULE_SIGNAL_PREFIX[rule]}_nomask"
                    pass_col = f"{RULE_PASS_PREFIX[rule]}_nomask"
                    avail_col = f"{rule}_available_nomask"
                    sig = float(mode_df.at[j, sig_col]) if sig_col in mode_df.columns else np.nan
                    passed = bool(mode_df.at[j, pass_col]) if pass_col in mode_df.columns else True
                    available = bool(mode_df.at[j, avail_col]) if avail_col in mode_df.columns else True
                    rec[f"{rule}_signal_leaf_nomask"] = sig
                    rec[f"{rule}_pass_leaf_nomask"] = passed
                    rec[f"{rule}_available_leaf_nomask"] = available
                    if not passed:
                        failed_rules.append(str(rule))
                rec["failed_rules_leaf_nomask"] = "|".join(failed_rules)
                rec["failed_rule_count_leaf_nomask"] = int(len(failed_rules))
                row_rows.append(rec)

    summary_df = pd.DataFrame(summary_rows)
    row_df = pd.DataFrame(row_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["distribution_pass_rate", "leaf_support"], ascending=[True, False]).reset_index(drop=True)
    if not row_df.empty:
        row_df = row_df.sort_values(["group_id", "leaf_path", "row_index"]).reset_index(drop=True)

    report = {
        "enabled": bool(not summary_df.empty),
        "mode": "nomask",
        "rows": int(n),
        "hard_gate_rows": int(np.sum(hard_gate)),
        "leaf_paths_evaluated": int(evaluated_paths),
        "leaf_row_hits": int(len(row_df)),
        "leaf_min_support": int(min_support),
        "active_rules": list(active_rules),
    }
    return summary_df, row_df, report


def build_detail_row_metrics_from_leaf_gate_hits(
    *,
    leaf_gate_row_df: pd.DataFrame,
    n_rows: int,
    max_failed_paths_per_row: int = 8,
) -> dict[str, np.ndarray]:
    max_failed_paths_per_row = int(max(1, max_failed_paths_per_row))
    detail_eval_count = np.zeros(int(n_rows), dtype=int)
    detail_fail_count = np.zeros(int(n_rows), dtype=int)
    failed_paths_text = np.full(int(n_rows), "", dtype=object)

    if leaf_gate_row_df.empty:
        return {
            "detail_evaluated_nomask": detail_eval_count > 0,
            "detail_fail_any_leaf_nomask": detail_fail_count > 0,
            "detail_fail_leaf_count_nomask": detail_fail_count,
            "detail_eval_leaf_count_nomask": detail_eval_count,
            "detail_failed_leaf_paths_nomask": failed_paths_text,
        }

    required_cols = {"row_index", "leaf_path", "distribution_pass_leaf_nomask"}
    missing = sorted(required_cols - set(leaf_gate_row_df.columns))
    if missing:
        raise ValueError(f"detail leaf gate row hits missing columns: {missing}")

    rows = leaf_gate_row_df.copy()
    rows["row_index"] = pd.to_numeric(rows["row_index"], errors="coerce").fillna(-1).astype(int)
    rows = rows[(rows["row_index"] >= 0) & (rows["row_index"] < int(n_rows))].copy()
    if rows.empty:
        return {
            "detail_evaluated_nomask": detail_eval_count > 0,
            "detail_fail_any_leaf_nomask": detail_fail_count > 0,
            "detail_fail_leaf_count_nomask": detail_fail_count,
            "detail_eval_leaf_count_nomask": detail_eval_count,
            "detail_failed_leaf_paths_nomask": failed_paths_text,
        }

    rows["_dist_pass"] = rows["distribution_pass_leaf_nomask"].fillna(True).astype(bool)
    if "failed_rule_count_leaf_nomask" in rows.columns:
        rows["_failed_rule_count"] = (
            pd.to_numeric(rows["failed_rule_count_leaf_nomask"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
    else:
        rows["_failed_rule_count"] = (~rows["_dist_pass"]).astype(int)

    grouped = rows.groupby("row_index", sort=False, dropna=False)
    for row_idx_raw, sub in grouped:
        row_idx = int(row_idx_raw)
        detail_eval_count[row_idx] = int(len(sub))
        failed = sub[~sub["_dist_pass"]].copy()
        fail_n = int(len(failed))
        detail_fail_count[row_idx] = fail_n
        if fail_n <= 0:
            continue

        failed = failed.sort_values(
            ["_failed_rule_count", "leaf_path"],
            ascending=[False, True],
            kind="stable",
        )
        uniq_paths: list[str] = []
        seen: set[str] = set()
        for path in failed["leaf_path"].fillna("").astype(str).tolist():
            p = path.strip()
            if not p or p in seen:
                continue
            seen.add(p)
            uniq_paths.append(p)
            if len(uniq_paths) >= max_failed_paths_per_row:
                break
        failed_paths_text[row_idx] = "|".join(uniq_paths)

    return {
        "detail_evaluated_nomask": detail_eval_count > 0,
        "detail_fail_any_leaf_nomask": detail_fail_count > 0,
        "detail_fail_leaf_count_nomask": detail_fail_count,
        "detail_eval_leaf_count_nomask": detail_eval_count,
        "detail_failed_leaf_paths_nomask": failed_paths_text,
    }


def build_detail_leaf_distribution_hits_from_gate(
    *,
    leaf_gate_summary_df: pd.DataFrame,
    leaf_gate_row_df: pd.DataFrame,
    signal_rule: str = "output",
) -> pd.DataFrame:
    if leaf_gate_row_df.empty:
        return pd.DataFrame(
            columns=["row_index", "row_id", "group_id", "leaf_path", "signal", "threshold", "failed"]
        )

    rule = normalize_pipeline_rule_key(signal_rule)
    if not rule:
        rule = "output"
    signal_col = f"{rule}_signal_leaf_nomask"
    pass_col = f"{rule}_pass_leaf_nomask"
    threshold_col = f"{rule}_threshold"

    if signal_col not in leaf_gate_row_df.columns:
        return pd.DataFrame(
            columns=["row_index", "row_id", "group_id", "leaf_path", "signal", "threshold", "failed"]
        )

    hits = leaf_gate_row_df.copy()
    hits["row_index"] = pd.to_numeric(hits["row_index"], errors="coerce").fillna(-1).astype(int)
    hits["row_id"] = hits["row_id"].fillna("").astype(str)
    hits["group_id"] = hits["group_id"].fillna("").astype(str)
    hits["leaf_path"] = hits["leaf_path"].fillna("").astype(str)
    hits["signal"] = pd.to_numeric(hits[signal_col], errors="coerce")

    if pass_col in hits.columns:
        hits["failed"] = ~hits[pass_col].fillna(True).astype(bool)
    elif "distribution_pass_leaf_nomask" in hits.columns:
        hits["failed"] = ~hits["distribution_pass_leaf_nomask"].fillna(True).astype(bool)
    else:
        hits["failed"] = False

    hits["threshold"] = np.nan
    if (not leaf_gate_summary_df.empty) and threshold_col in leaf_gate_summary_df.columns:
        gid_col = "group_id_raw" if "group_id_raw" in leaf_gate_summary_df.columns else "group_id"
        if gid_col in leaf_gate_summary_df.columns and "leaf_path" in leaf_gate_summary_df.columns:
            th_df = leaf_gate_summary_df[[gid_col, "leaf_path", threshold_col]].copy()
            th_df[gid_col] = th_df[gid_col].fillna("").astype(str)
            th_df["leaf_path"] = th_df["leaf_path"].fillna("").astype(str)
            th_df[threshold_col] = pd.to_numeric(th_df[threshold_col], errors="coerce")
            th_df = th_df.rename(columns={gid_col: "group_id", threshold_col: "threshold"})
            hits = hits.merge(th_df, on=["group_id", "leaf_path"], how="left", suffixes=("", "_summary"))
            if "threshold_summary" in hits.columns:
                hits["threshold"] = pd.to_numeric(hits["threshold_summary"], errors="coerce")
                hits = hits.drop(columns=["threshold_summary"])

    return hits[["row_index", "row_id", "group_id", "leaf_path", "signal", "threshold", "failed"]].copy()


def apply_detail_leaf_gate_to_row_result(
    *,
    result_df: pd.DataFrame,
    leaf_gate_row_df: pd.DataFrame,
    gate_rules: list[str],
    distribution_rules: list[str] | None = None,
) -> pd.DataFrame:
    out = result_df.copy()
    n_rows = int(len(out))
    if n_rows == 0:
        return out

    hard_gate = out["hard_gate_pass"].fillna(False).astype(bool).to_numpy()
    uniq_gate_rules: list[str] = []
    for r in gate_rules:
        rk = normalize_pipeline_rule_key(r)
        if rk and rk not in uniq_gate_rules:
            uniq_gate_rules.append(rk)

    uniq_dist_rules = [
        normalize_pipeline_rule_key(r)
        for r in (distribution_rules if distribution_rules is not None else uniq_gate_rules)
    ]
    uniq_dist_rules = [r for r in uniq_dist_rules if r and r in uniq_gate_rules]
    if not uniq_dist_rules:
        uniq_dist_rules = list(uniq_gate_rules)

    dist_eval = np.zeros(n_rows, dtype=bool)
    dist_pass_leaf = np.ones(n_rows, dtype=bool)

    rule_pass_map: dict[str, np.ndarray] = {
        rule: np.ones(n_rows, dtype=bool) for rule in uniq_gate_rules
    }
    rule_available_map: dict[str, np.ndarray] = {
        rule: np.zeros(n_rows, dtype=bool) for rule in uniq_gate_rules
    }
    rule_signal_map: dict[str, np.ndarray] = {
        rule: np.zeros(n_rows, dtype=float) for rule in uniq_gate_rules
    }

    if not leaf_gate_row_df.empty:
        rows = leaf_gate_row_df.copy()
        if "row_index" not in rows.columns:
            raise ValueError("detail leaf gate row hits missing column: row_index")
        rows["row_index"] = pd.to_numeric(rows["row_index"], errors="coerce").fillna(-1).astype(int)
        rows = rows[(rows["row_index"] >= 0) & (rows["row_index"] < n_rows)].copy()

        if not rows.empty:
            if "distribution_pass_leaf_nomask" not in rows.columns:
                rows["distribution_pass_leaf_nomask"] = True
            rows["distribution_pass_leaf_nomask"] = rows["distribution_pass_leaf_nomask"].fillna(True).astype(bool)

            grouped = rows.groupby("row_index", sort=False, dropna=False)
            for row_idx_raw, sub in grouped:
                row_idx = int(row_idx_raw)
                dist_eval[row_idx] = True
                dist_pass_leaf[row_idx] = bool(np.all(sub["distribution_pass_leaf_nomask"].to_numpy(dtype=bool)))

                for rule in uniq_gate_rules:
                    pass_col = f"{rule}_pass_leaf_nomask"
                    avail_col = f"{rule}_available_leaf_nomask"
                    sig_col = f"{rule}_signal_leaf_nomask"

                    if avail_col in sub.columns:
                        avail = sub[avail_col].fillna(False).astype(bool).to_numpy()
                    elif pass_col in sub.columns or sig_col in sub.columns:
                        avail = np.ones(len(sub), dtype=bool)
                    else:
                        continue

                    if pass_col in sub.columns:
                        passed = sub[pass_col].fillna(True).astype(bool).to_numpy()
                    else:
                        passed = np.ones(len(sub), dtype=bool)

                    if sig_col in sub.columns:
                        sig_vals = pd.to_numeric(sub[sig_col], errors="coerce").to_numpy(dtype=float)
                    else:
                        sig_vals = np.full(len(sub), np.nan, dtype=float)

                    has_avail = bool(np.any(avail))
                    rule_available_map[rule][row_idx] = has_avail
                    if not has_avail:
                        rule_pass_map[rule][row_idx] = True
                        rule_signal_map[rule][row_idx] = 0.0
                        continue

                    rule_pass_map[rule][row_idx] = bool(np.all(passed[avail]))
                    finite_sig = sig_vals[avail & np.isfinite(sig_vals)]
                    rule_signal_map[rule][row_idx] = (
                        float(np.max(finite_sig)) if finite_sig.size > 0 else 0.0
                    )

    for rule in uniq_gate_rules:
        sig_prefix = RULE_SIGNAL_PREFIX.get(rule, f"{rule}_signal")
        pass_prefix = RULE_PASS_PREFIX.get(rule, f"{rule}_pass")
        out[f"{sig_prefix}_nomask"] = rule_signal_map[rule]
        out[f"{pass_prefix}_nomask"] = rule_pass_map[rule].astype(bool)
        out[f"{rule}_available_nomask"] = rule_available_map[rule].astype(bool)

    distribution_pass = np.ones(n_rows, dtype=bool)
    for rule in uniq_dist_rules:
        distribution_pass &= rule_pass_map[rule]
    # If explicit distribution_rules are provided, prioritize those rules only.
    # Otherwise keep legacy behavior (AND with leaf-level distribution pass flag).
    if distribution_rules is None:
        distribution_pass &= dist_pass_leaf
    distribution_pass = hard_gate & distribution_pass

    distribution_fail = hard_gate & (~distribution_pass)
    distribution_warn = np.zeros(n_rows, dtype=bool)
    distribution_hard = np.zeros(n_rows, dtype=bool)

    distribution_state = np.full(n_rows, "na", dtype=object)
    distribution_state[hard_gate] = "pass"
    distribution_state[hard_gate & distribution_fail] = "fail"
    distribution_state[hard_gate & distribution_warn] = "warn"
    distribution_state[hard_gate & distribution_hard] = "hard_fail"

    final_pass = hard_gate & distribution_pass
    final_state = np.full(n_rows, "hard_gate_fail", dtype=object)
    final_state[hard_gate] = distribution_state[hard_gate]

    out["distribution_evaluated_nomask"] = dist_eval.astype(bool)
    out["distribution_pass_nomask"] = distribution_pass.astype(bool)
    out["distribution_fail_nomask"] = distribution_fail.astype(bool)
    out["distribution_warn_nomask"] = distribution_warn.astype(bool)
    out["distribution_hard_fail_nomask"] = distribution_hard.astype(bool)
    out["distribution_state_nomask"] = distribution_state
    out["final_pass_nomask"] = final_pass.astype(bool)
    out["final_state_nomask"] = final_state
    out["detail_override_applied_nomask"] = False

    return out


def run_detailed_leaf_group_fallback(
    *,
    result_df: pd.DataFrame,
    group_meta: pd.DataFrame,
    in_text_arr: np.ndarray,
    out_text_arr: np.ndarray,
    source_output_arr: np.ndarray,
    model: TextEmbedder,
    active_rules: list[str],
    rule_quantiles: dict[str, float],
    refine_reference: bool,
    refine_rules: list[str],
    refine_iterations: int,
    refine_min_size: int,
    args: argparse.Namespace,
    output_dir: Path,
    stem: str,
    tag: str,
    label_coloring_enabled: bool,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    out = result_df.copy()
    eval_col = "distribution_evaluated_nomask"
    pass_col = "distribution_pass_nomask"
    final_col = "final_pass_nomask"

    if eval_col not in out.columns:
        out[eval_col] = False
    if final_col not in out.columns:
        out[final_col] = False

    groups = out["distribution_group_id"].fillna("g0000").astype(str).to_numpy(dtype=object)
    hard_gate = out["hard_gate_pass"].fillna(False).astype(bool).to_numpy()
    evaluated = out[eval_col].fillna(False).astype(bool).to_numpy()
    miss_mask = hard_gate & (~evaluated)

    summaries: list[dict[str, Any]] = []
    stats = {
        "enabled": True,
        "fallback_groups": 0,
        "fallback_rows": int(np.sum(miss_mask)),
    }
    if not np.any(miss_mask):
        return out, summaries, stats

    for g in group_meta.itertuples(index=False):
        gid = str(g.group_id)
        idx_group = np.where(groups == gid)[0]
        if len(idx_group) == 0:
            continue
        idx_eval = idx_group[miss_mask[idx_group]]
        if len(idx_eval) == 0:
            continue

        idx_eval_arr = np.asarray(idx_eval, dtype=int)
        in_texts, out_texts, src_out_texts = collect_mode_text_triplet_by_indices(
            row_indices=idx_eval_arr,
            input_text_arr=in_text_arr,
            output_text_arr=out_text_arr,
            source_output_arr=source_output_arr,
        )

        if label_coloring_enabled:
            group_label_is_correct = out.loc[idx_eval_arr, "label_is_correct"].to_numpy(dtype=float)
            group_label_raw = out.loc[idx_eval_arr, "label_raw"].fillna("").astype(str).to_numpy(dtype=object)
        else:
            group_label_is_correct = None
            group_label_raw = None

        mode_df, summary, group_ref_mask = run_distribution_mode_job(
            mode_name="nomask",
            input_texts=in_texts,
            output_texts=out_texts,
            source_output_texts=src_out_texts,
            model=model,
            active_rules=active_rules,
            rule_quantiles=rule_quantiles,
            refine_reference=refine_reference,
            refine_rules=refine_rules,
            refine_iterations=refine_iterations,
            refine_min_size=refine_min_size,
            args=args,
            output_dir=output_dir,
            stem=stem,
            tag=tag,
            group_id=gid,
            label_is_correct=group_label_is_correct,
            label_raw=group_label_raw,
            cache_writer=None,
            cache_row_indices=None,
        )
        out = assign_subset_columns(out, pd.Index(idx_eval_arr), mode_df)
        out.loc[idx_eval_arr, eval_col] = True
        if pass_col in out.columns:
            out.loc[idx_eval_arr, final_col] = out.loc[idx_eval_arr, pass_col].fillna(False).astype(bool).to_numpy()

        group_hard = hard_gate[idx_group]
        group_eval = out.loc[idx_group, eval_col].fillna(False).astype(bool).to_numpy()
        group_final = out.loc[idx_group, final_col].fillna(False).astype(bool).to_numpy()
        summary["group_col"] = getattr(g, "group_col", "distribution_group_id")
        summary["group_size"] = int(getattr(g, "group_size", len(idx_group)))
        summary["group_eval_size"] = int(np.sum(group_eval))
        summary["group_value_preview"] = getattr(g, "group_value_preview", "")
        summary["group_ref_init_size"] = int(group_ref_mask.sum())
        summary["group_ref_size"] = int(summary.get("threshold_ref_size", group_ref_mask.sum()))
        summary["hard_gate_pass_rate"] = float(np.mean(group_hard)) if len(group_hard) > 0 else 0.0
        summary["distribution_eval_coverage"] = float(np.mean(group_eval)) if len(group_eval) > 0 else 0.0
        summary["final_pass_rate"] = (
            float(np.mean(group_final[group_hard])) if np.any(group_hard) else 0.0
        )
        summaries.append(summary)
        stats["fallback_groups"] = int(stats["fallback_groups"]) + 1

    return out, summaries, stats


def build_leaf_only_group_summary(
    *,
    result_df: pd.DataFrame,
    group_meta: pd.DataFrame,
    mode_name: str,
    active_rules: list[str],
    rule_quantiles: dict[str, float],
) -> pd.DataFrame:
    mode = str(mode_name).strip().lower()
    if mode != "nomask":
        raise ValueError(f"Unsupported mode for leaf-only summary: {mode_name}")
    if result_df.empty:
        return pd.DataFrame()

    groups = result_df["distribution_group_id"].fillna("g0000").astype(str).to_numpy(dtype=object)
    hard_gate = result_df["hard_gate_pass"].fillna(False).astype(bool).to_numpy()
    eval_col = f"distribution_evaluated_{mode}"
    dist_col = f"distribution_pass_{mode}"
    final_col = f"final_pass_{mode}"
    eval_mask = (
        result_df[eval_col].fillna(False).astype(bool).to_numpy()
        if eval_col in result_df.columns
        else np.zeros(len(result_df), dtype=bool)
    )
    dist_pass = (
        result_df[dist_col].fillna(False).astype(bool).to_numpy()
        if dist_col in result_df.columns
        else np.zeros(len(result_df), dtype=bool)
    )
    final_pass = (
        result_df[final_col].fillna(False).astype(bool).to_numpy()
        if final_col in result_df.columns
        else (hard_gate & dist_pass)
    )

    meta_df = group_meta.copy()
    if meta_df.empty:
        uniq = pd.unique(groups)
        meta_df = pd.DataFrame(
            {
                "group_id": uniq,
                "group_col": "__all__",
                "group_value_preview": "",
                "group_size": [int(np.sum(groups == gid)) for gid in uniq],
            }
        )

    rows: list[dict[str, Any]] = []
    for g in meta_df.itertuples(index=False):
        gid = str(getattr(g, "group_id", "g0000"))
        gmask = groups == gid
        if not np.any(gmask):
            continue

        gsize = int(np.sum(gmask))
        ghard = gmask & hard_gate
        geval = gmask & eval_mask
        ghard_n = int(np.sum(ghard))
        geval_n = int(np.sum(geval))

        row: dict[str, Any] = {
            "mode": mode,
            "group_id": gid,
            "rows": int(ghard_n),
            "group_col": str(getattr(g, "group_col", "distribution_group_id")),
            "group_size": int(getattr(g, "group_size", gsize)),
            "group_eval_size": int(geval_n),
            "group_value_preview": str(getattr(g, "group_value_preview", "")),
            "group_ref_init_size": int(geval_n),
            "group_ref_size": int(geval_n),
            "threshold_ref_size": int(geval_n),
            "hard_gate_pass_rate": float(ghard_n / max(gsize, 1)),
            "distribution_eval_coverage": float(geval_n / max(gsize, 1)),
            "distribution_pass_rate": float(np.mean(dist_pass[ghard])) if ghard_n > 0 else 0.0,
            "final_pass_rate": float(np.mean(final_pass[ghard])) if ghard_n > 0 else 0.0,
            "active_distribution_rules": ",".join(active_rules),
            "rule_quantiles_json": json.dumps(rule_quantiles, ensure_ascii=False),
            "consistency_observe_only": bool(GATE_RUNTIME.consistency_observe_only),
            "threshold_refine_enabled": False,
            "threshold_refine_rules": "",
            "threshold_refine_rounds": 0,
        }

        for rule in active_rules:
            pass_col = f"{RULE_PASS_PREFIX[rule]}_{mode}"
            if pass_col in result_df.columns and ghard_n > 0:
                p = result_df.loc[ghard, pass_col].fillna(True).astype(bool).to_numpy()
                row[_rule_outlier_rate_col(rule)] = float((~p).mean())
            else:
                row[_rule_outlier_rate_col(rule)] = np.nan
            row[f"{rule}_threshold"] = np.nan

        rows.append(row)

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ["distribution_pass_rate", "group_size"],
            ascending=[True, False],
            kind="stable",
        ).reset_index(drop=True)
    return summary_df


def load_schema(args: argparse.Namespace, outputs: list[str]) -> tuple[dict[str, Any] | None, str]:
    if args.schema_path:
        schema = json.loads(Path(args.schema_path).read_text())
        return schema, f"loaded:{args.schema_path}"

    if args.schema_json:
        schema = json.loads(args.schema_json)
        return schema, "loaded:inline"

    if DEFAULT_SCHEMA:
        return dict(DEFAULT_SCHEMA), "default_runtime"

    inferred = infer_schema_from_outputs(
        outputs,
        min_key_ratio=args.schema_infer_min_key_ratio,
        max_enum=args.schema_infer_max_enum,
    )
    if inferred:
        return inferred, "inferred"

    dict_ratio = float(np.mean([isinstance(safe_json_load(x), dict) for x in outputs])) if outputs else 0.0
    if dict_ratio < 0.30:
        return None, f"auto_disabled_non_json(dict_ratio={dict_ratio:.2f})"
    return None, f"auto_disabled_no_stable_schema(dict_ratio={dict_ratio:.2f})"


def _rule_outlier_rate_col(rule: str) -> str:
    if rule == "diff_resid":
        return "diff_resid_outlier_rate"
    return f"{rule}_outlier_rate"


def _apply_distance_calibration_to_mode(
    result_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    mode: str,
    active_rules: list[str],
    rules: list[str],
    selected_quantiles: dict[str, float],
    group_thresholds: dict[str, dict[str, float]],
    global_thresholds: dict[str, float],
    selected_metrics: dict[str, Any] | None,
    calibration_mode: str,
) -> None:
    eval_col = f"distribution_evaluated_{mode}"
    if eval_col not in result_df.columns:
        return
    eval_mask = result_df[eval_col].astype(bool).to_numpy()
    if not eval_mask.any():
        return

    groups = result_df["distribution_group_id"].astype(str).to_numpy(dtype=object)
    selected_col = f"distance_calibration_selected_{mode}"
    if selected_col not in result_df.columns:
        result_df[selected_col] = False

    for rule in rules:
        pass_col = f"{RULE_PASS_PREFIX[rule]}_{mode}"
        signal_col = f"{RULE_SIGNAL_PREFIX[rule]}_{mode}"
        if pass_col not in result_df.columns or signal_col not in result_df.columns:
            continue
        signal = result_df[signal_col].to_numpy(dtype=float)
        passed = result_df[pass_col].fillna(True).astype(bool).to_numpy()
        idx_eval = np.where(eval_mask)[0]
        for i in idx_eval:
            gid = str(groups[i])
            th_map = group_thresholds.get(gid)
            th = float(global_thresholds.get(rule, np.nan))
            if th_map is not None and rule in th_map:
                th = float(th_map[rule])
            if np.isfinite(signal[i]) and np.isfinite(th):
                passed[i] = bool(signal[i] <= th)
        result_df[pass_col] = passed

    dist_pass = np.ones(len(result_df), dtype=bool)
    for rule in active_rules:
        col = f"{RULE_PASS_PREFIX[rule]}_{mode}"
        if col in result_df.columns:
            dist_pass &= result_df[col].fillna(True).astype(bool).to_numpy()
    if f"distribution_pass_{mode}" in result_df.columns:
        cur = result_df[f"distribution_pass_{mode}"].fillna(True).astype(bool).to_numpy()
        cur[eval_mask] = dist_pass[eval_mask]
        result_df[f"distribution_pass_{mode}"] = cur
    if f"final_pass_{mode}" in result_df.columns:
        final = result_df[f"final_pass_{mode}"].fillna(False).astype(bool).to_numpy()
        final[eval_mask] = dist_pass[eval_mask]
        result_df[f"final_pass_{mode}"] = final
    sel = result_df[selected_col].fillna(False).astype(bool).to_numpy()
    sel[eval_mask] = True
    result_df[selected_col] = sel

    if summary_df.empty:
        return
    s_mask = summary_df["mode"].astype(str) == mode
    if not s_mask.any():
        return

    for sidx in summary_df.index[s_mask]:
        gid = str(summary_df.at[sidx, "group_id"]) if "group_id" in summary_df.columns else "g0000"
        g_eval = eval_mask & (groups == gid)
        if not g_eval.any():
            continue
        for rule in rules:
            th = float(global_thresholds.get(rule, np.nan))
            th_map = group_thresholds.get(gid)
            if th_map is not None and rule in th_map:
                th = float(th_map[rule])
            summary_df.at[sidx, f"{rule}_threshold"] = th
            pass_col = f"{RULE_PASS_PREFIX[rule]}_{mode}"
            if pass_col in result_df.columns:
                p = result_df.loc[g_eval, pass_col].fillna(True).astype(bool).to_numpy()
                summary_df.at[sidx, _rule_outlier_rate_col(rule)] = float((~p).mean())
        if f"distribution_pass_{mode}" in result_df.columns:
            summary_df.at[sidx, "distribution_pass_rate"] = float(
                result_df.loc[g_eval, f"distribution_pass_{mode}"].fillna(True).astype(bool).mean()
            )
        if f"final_pass_{mode}" in result_df.columns:
            summary_df.at[sidx, "final_pass_rate"] = float(
                result_df.loc[g_eval, f"final_pass_{mode}"].fillna(False).astype(bool).mean()
            )
        summary_df.at[sidx, "distance_calibration_mode"] = calibration_mode
        summary_df.at[sidx, "distance_calibration_rules"] = ",".join(rules)
        summary_df.at[sidx, "distance_calibration_selected_quantiles_json"] = json.dumps(
            selected_quantiles, ensure_ascii=False
        )
        if selected_metrics:
            summary_df.at[sidx, "distance_calibration_cv_f1"] = float(selected_metrics.get("mean_f1", np.nan))
            summary_df.at[sidx, "distance_calibration_cv_precision"] = float(
                selected_metrics.get("mean_precision", np.nan)
            )
            summary_df.at[sidx, "distance_calibration_cv_fp"] = float(selected_metrics.get("mean_fp", np.nan))


def _ensure_distance_calibration_columns(
    result_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    new_result_cols: dict[str, pd.Series] = {}
    col = "distance_calibration_selected_nomask"
    if col not in result_df.columns:
        new_result_cols[col] = pd.Series([False] * len(result_df), index=result_df.index, dtype=bool)
    if new_result_cols:
        result_df = pd.concat([result_df, pd.DataFrame(new_result_cols, index=result_df.index)], axis=1)

    new_summary_cols: dict[str, pd.Series] = {}
    for col in [
        "distance_calibration_mode",
        "distance_calibration_rules",
        "distance_calibration_cv_f1",
        "distance_calibration_cv_precision",
        "distance_calibration_cv_fp",
        "distance_calibration_selected_quantiles_json",
    ]:
        if col not in summary_df.columns:
            if col.endswith(("_f1", "_precision", "_fp")):
                new_summary_cols[col] = pd.Series([np.nan] * len(summary_df), index=summary_df.index, dtype=float)
            else:
                new_summary_cols[col] = pd.Series([""] * len(summary_df), index=summary_df.index, dtype=object)
    if new_summary_cols:
        summary_df = pd.concat([summary_df, pd.DataFrame(new_summary_cols, index=summary_df.index)], axis=1)

    return result_df, summary_df


def _ensure_tristate_columns(
    result_df: pd.DataFrame, summary_df: pd.DataFrame, rules: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    existing_result_cols = set(result_df.columns)
    new_result_cols: dict[str, Any] = {}
    for rule in rules:
        st_col = f"{rule}_status_nomask"
        pass_col = f"{rule}_pass3_nomask"
        warn_col = f"{rule}_warn_nomask"
        fail_col = f"{rule}_fail_nomask"
        if st_col not in existing_result_cols:
            new_result_cols[st_col] = pd.Series(["na"] * len(result_df), index=result_df.index)
        if pass_col not in existing_result_cols:
            new_result_cols[pass_col] = pd.Series([False] * len(result_df), index=result_df.index, dtype=bool)
        if warn_col not in existing_result_cols:
            new_result_cols[warn_col] = pd.Series([False] * len(result_df), index=result_df.index, dtype=bool)
        if fail_col not in existing_result_cols:
            new_result_cols[fail_col] = pd.Series([False] * len(result_df), index=result_df.index, dtype=bool)
    if new_result_cols:
        add_df = pd.DataFrame(new_result_cols, index=result_df.index)
        result_df = pd.concat([result_df, add_df], axis=1)
    for rule in rules:
        for col in [
            f"{rule}_warn_threshold",
            f"{rule}_fail_threshold",
            f"{rule}_warn_rate",
            f"{rule}_fail_rate",
        ]:
            if col not in summary_df.columns:
                summary_df[col] = np.nan
    for col in [
        "tristate_enabled",
        "tristate_calibration_mode",
        "tristate_calibration_rules",
        "tristate_calibration_selected_json",
    ]:
        if col not in summary_df.columns:
            if col == "tristate_enabled":
                summary_df[col] = False
            else:
                summary_df[col] = ""
    return result_df, summary_df


def _apply_tristate_rule_mode(
    result_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    mode: str,
    rule: str,
    states: dict[str, np.ndarray],
    global_thresholds: dict[str, float],
    group_thresholds: dict[str, tuple[float, float]],
) -> None:
    eval_col = f"distribution_evaluated_{mode}"
    if eval_col not in result_df.columns:
        return
    eval_mask = result_df[eval_col].fillna(False).astype(bool).to_numpy()
    if not eval_mask.any():
        return
    st_col = f"{rule}_status_{mode}"
    pass_col = f"{rule}_pass3_{mode}"
    warn_col = f"{rule}_warn_{mode}"
    fail_col = f"{rule}_fail_{mode}"
    result_df.loc[eval_mask, st_col] = np.asarray(states["status"], dtype=object)[eval_mask]
    result_df.loc[eval_mask, pass_col] = np.asarray(states["pass"], dtype=bool)[eval_mask]
    result_df.loc[eval_mask, warn_col] = np.asarray(states["warn"], dtype=bool)[eval_mask]
    result_df.loc[eval_mask, fail_col] = np.asarray(states["fail"], dtype=bool)[eval_mask]

    if summary_df.empty:
        return
    groups = result_df["distribution_group_id"].astype(str).to_numpy(dtype=object)
    s_mask = summary_df["mode"].astype(str) == mode
    for sidx in summary_df.index[s_mask]:
        gid = str(summary_df.at[sidx, "group_id"]) if "group_id" in summary_df.columns else "g0000"
        gmask = eval_mask & (groups == gid)
        if not gmask.any():
            continue
        w, f = group_thresholds.get(gid, (global_thresholds["warn"], global_thresholds["fail"]))
        summary_df.at[sidx, f"{rule}_warn_threshold"] = float(w)
        summary_df.at[sidx, f"{rule}_fail_threshold"] = float(f)
        summary_df.at[sidx, f"{rule}_warn_rate"] = float(np.asarray(states["warn"], dtype=bool)[gmask].mean())
        summary_df.at[sidx, f"{rule}_fail_rate"] = float(np.asarray(states["fail"], dtype=bool)[gmask].mean())

def _run_with_args(args: argparse.Namespace) -> DistributionRunArtifacts:
    root_output_dir = Path(args.output_dir)
    root_output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = root_output_dir / str(args.report_dir_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("[INFO] Plot export: enabled (interactive Plotly HTML).")
    if umap_module is None:
        print("[INFO] umap-learn not available: skipping UMAP projection.")

    df, used_encoding = read_csv_dynamic(args.csv_path, encoding=args.encoding)
    source_rows = len(df)
    if args.max_rows is not None and len(df) > args.max_rows:
        df = df.sample(args.max_rows, random_state=args.sample_seed).reset_index(drop=True)
        print(f"[INFO] Applied --max-rows sampling: {source_rows} -> {len(df)}")
    elif args.max_rows is None and args.auto_max_rows is not None and len(df) > args.auto_max_rows:
        df = df.sample(args.auto_max_rows, random_state=args.sample_seed).reset_index(drop=True)
        print(f"[INFO] Applied --auto-max-rows sampling: {source_rows} -> {len(df)}")

    required_cols = [args.input_col, args.output_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    print(
        f"[INFO] Loaded rows={len(df)} (source_rows={source_rows}) "
        f"from {args.csv_path} (encoding={used_encoding})"
    )

    result_df = pd.DataFrame(index=df.index)
    if args.id_col in df.columns:
        result_df["row_id"] = df[args.id_col].astype(str)
    else:
        result_df["row_id"] = df.index.astype(str)

    # Optional label parsing for plot coloring
    label_correct_set = parse_label_value_set(args.label_correct_values)
    label_incorrect_set = parse_label_value_set(args.label_incorrect_values)
    label_coloring_enabled = bool(args.use_label_coloring and args.label_col in df.columns)
    if label_coloring_enabled:
        raw_labels = df[args.label_col].fillna("").astype(str).tolist()
        label_is_correct = map_label_correctness(
            values=raw_labels,
            correct_values=label_correct_set,
            incorrect_values=label_incorrect_set,
        )
        result_df["label_raw"] = raw_labels
        result_df["label_is_correct"] = label_is_correct
        known_ratio = float(np.mean(np.isfinite(label_is_correct))) if len(label_is_correct) > 0 else 0.0
        print(
            f"[INFO] Label coloring: col={args.label_col}, known_ratio={known_ratio:.2%}, "
            f"correct_values={sorted(label_correct_set)}, incorrect_values={sorted(label_incorrect_set)}"
        )
    else:
        result_df["label_raw"] = ""
        result_df["label_is_correct"] = np.nan
        if args.use_label_coloring:
            print(f"[INFO] Label coloring: disabled (column '{args.label_col}' not found).")
        else:
            print("[INFO] Label coloring: disabled by --no-use-label-coloring.")

    # Hard gate: IFEval
    instruction_ids = [x.strip() for x in args.ifeval_ids.split(",") if x.strip()]
    kwargs_dict = json.loads(args.ifeval_kwargs_json) if args.ifeval_kwargs_json else {}
    ifeval_df = run_ifeval(
        df=df,
        prompt_col=args.prompt_col,
        output_col=args.output_col,
        instruction_ids=instruction_ids,
        kwargs_dict=kwargs_dict,
        eval_mode=args.ifeval_mode,
    )
    result_df["ifeval_pass"] = ifeval_df["ifeval_pass"].values
    result_df["ifeval_failed"] = ifeval_df["ifeval_failed"].values
    print(f"[INFO] IFEval pass rate: {result_df['ifeval_pass'].mean():.2%}")

    # Hard gate: schema
    schema, schema_source = load_schema(args, outputs=df[args.output_col].fillna("").astype(str).tolist())
    if schema is None:
        result_df["schema_pass"] = True
        result_df["schema_missing_keys"] = None
        result_df["schema_error"] = None
        print("[INFO] Schema gate: disabled")
    else:
        schema_result, schema_df = run_schema_gate(
            df=df,
            output_col=args.output_col,
            schema=schema,
            strict_keys=args.schema_strict_keys,
            allow_extra_keys=args.schema_allow_extra_keys,
        )
        result_df["schema_pass"] = schema_df["schema_pass"].values
        result_df["schema_missing_keys"] = schema_df["schema_missing_keys"].values
        result_df["schema_error"] = schema_df["schema_error"].values
        print(f"[INFO] Schema source: {schema_source}, pass rate: {schema_result.pass_rate:.2%}")

    # Hard gate: text-length
    gate = TextLengthGate(
        min_ratio=args.textlen_min_ratio,
        min_support_ratio=args.textlen_min_support_ratio,
    )
    gate.compute_thresholds(df[args.output_col].fillna("").astype(str).tolist())
    textlen_pass, textlen_failed = gate.validate_with_dataframe(
        df,
        output_col=args.output_col,
    )
    result_df["textlen_pass"] = textlen_pass.values
    result_df["textlen_failed"] = textlen_failed.values
    print(f"[INFO] Text-length pass rate: {result_df['textlen_pass'].mean():.2%}")

    result_df["hard_gate_pass"] = (
        result_df["ifeval_pass"].astype(bool)
        & result_df["schema_pass"].astype(bool)
        & result_df["textlen_pass"].astype(bool)
    )
    print(f"[INFO] Hard gate pass rate: {result_df['hard_gate_pass'].mean():.2%}")
    print(
        f"[INFO] Distribution signaling target rows (hard_gate_pass=True): "
        f"{int(result_df['hard_gate_pass'].sum())}/{len(result_df)}"
    )

    # Distribution grouping (default: by prompt)
    group_ids, group_meta, group_col_used = resolve_distribution_groups(
        df=df,
        prompt_col=args.prompt_col,
        group_by_prompt=args.group_by_prompt,
        group_by_col=args.group_by_col,
    )
    result_df["distribution_group_id"] = group_ids.values
    result_df["distribution_group_col"] = group_col_used
    group_size_map = group_meta.set_index("group_id")["group_size"].to_dict()
    group_preview_map = group_meta.set_index("group_id")["group_value_preview"].to_dict()
    result_df["distribution_group_size"] = result_df["distribution_group_id"].map(group_size_map)
    result_df["distribution_group_value_preview"] = result_df["distribution_group_id"].map(group_preview_map)
    print(f"[INFO] Distribution grouping: col={group_col_used}, groups={len(group_meta)}")
    if len(group_meta) > 1:
        top_group_info = ", ".join(
            [f"{r.group_id}:{r.group_size}" for r in group_meta.head(5).itertuples(index=False)]
        )
        print(f"[INFO] Largest groups: {top_group_info}")

    # Build nomask bundles for input/output, independently per group
    in_raw, _, in_path_stats = build_column_bundles_grouped(
        series=df[args.input_col],
        group_ids=result_df["distribution_group_id"],
        **TEMPLATE_BUNDLE_CONFIG,
    )
    out_raw, _, out_path_stats = build_column_bundles_grouped(
        series=df[args.output_col],
        group_ids=result_df["distribution_group_id"],
        **TEMPLATE_BUNDLE_CONFIG,
    )
    if not in_path_stats.empty:
        in_path_stats = in_path_stats.merge(
            group_meta[["group_id", "group_col", "group_value_preview"]],
            on="group_id",
            how="left",
        )
    if not out_path_stats.empty:
        out_path_stats = out_path_stats.merge(
            group_meta[["group_id", "group_col", "group_value_preview"]],
            on="group_id",
            how="left",
        )
    print("[INFO] Built JSON value-level bundles per group: nomask")
    result_df["input_bundle_nomask"] = in_raw
    result_df["output_bundle_nomask"] = out_raw
    result_df["source_input"] = df[args.input_col].fillna("").astype(str).values
    result_df["source_output"] = df[args.output_col].fillna("").astype(str).values

    model = build_embedder(args)
    print(f"[INFO] Embedding backend: {model.name}")
    result_df["detail_evaluated_nomask"] = False
    result_df["detail_fail_any_leaf_nomask"] = False
    result_df["detail_fail_leaf_count_nomask"] = 0
    result_df["detail_eval_leaf_count_nomask"] = 0
    result_df["detail_failed_leaf_paths_nomask"] = ""
    result_df["detail_override_applied_nomask"] = False
    detail_leaf_summary_df = pd.DataFrame()
    detail_leaf_hits_df = pd.DataFrame()
    detail_leaf_dist_stats_df = pd.DataFrame()
    detail_leaf_dist_bins_df = pd.DataFrame()
    detail_leaf_gate_summary_df = pd.DataFrame()
    detail_leaf_gate_row_df = pd.DataFrame()
    detail_leaf_report: dict[str, Any] = {
        "mode": str(args.inspection_mode),
        "enabled": bool(str(args.inspection_mode) == "detailed"),
        "detail_leaf_paths_evaluated": 0,
        "detail_row_hits": 0,
        "detail_fail_rows": 0,
    }
    detail_leaf_distribution_report: dict[str, Any] = {
        "enabled": False,
        "bins": int(args.detail_leaf_dist_bins),
        "leaf_paths": 0,
        "row_hits": 0,
    }
    detail_leaf_gate_report: dict[str, Any] = {
        "enabled": False,
        "mode": "nomask",
        "rows": int(len(df)),
        "hard_gate_rows": int(result_df["hard_gate_pass"].sum()),
        "leaf_paths_evaluated": 0,
        "leaf_row_hits": 0,
        "leaf_min_support": int(args.detail_leaf_min_support),
        "active_rules": [],
    }
    if str(args.inspection_mode) == "detailed":
        print("[INFO] Detailed observe-only inspection: disabled (leaf gate is the single source).")
    else:
        print("[INFO] Detailed output-leaf inspection: skipped (inspection_mode=integrated)")

    active_rules = parse_pipeline_rule_keys(",".join(GATE_RUNTIME.active_distribution_rules))
    refine_rules = parse_pipeline_rule_keys(",".join(GATE_RUNTIME.threshold_refine_rules))
    semantic_rules_leaf: list[str] = []
    if bool(SEMANTIC_RUNTIME.enable_discourse_instability_rule):
        semantic_rules_leaf.append("discourse_instability")
    if bool(SEMANTIC_RUNTIME.enable_contradiction_rule):
        semantic_rules_leaf.append("contradiction")
    if bool(SEMANTIC_RUNTIME.enable_self_contradiction_rule):
        semantic_rules_leaf.append("self_contradiction")
    leaf_gate_rules = list(dict.fromkeys([*active_rules, *semantic_rules_leaf]))
    udf_core_rules = list(UDF_FIXED_CORE_RULES)
    rule_quantiles = resolve_rule_quantiles()
    distribution_modes = ["nomask"]
    calibration_rules = parse_calibration_rules(",".join(CALIBRATION_RUNTIME.rules))
    calibration_apply_modes = parse_apply_modes(",".join(CALIBRATION_RUNTIME.apply_modes))
    output_quantile_candidates = parse_quantile_range_spec(CALIBRATION_RUNTIME.output_quantiles)
    other_quantile_candidates = parse_quantile_csv_spec(CALIBRATION_RUNTIME.other_quantiles)
    tristate_rules = parse_pipeline_rule_keys(",".join(TRISTATE_CFG.rules))
    tristate_apply_modes = parse_apply_modes(",".join(TRISTATE_CFG.apply_modes))
    tristate_grid_warn = parse_quantile_csv_spec(TRISTATE_CFG.grid_warn_quantiles)
    tristate_grid_fail = parse_quantile_csv_spec(TRISTATE_CFG.grid_fail_quantiles)
    print(
        "[INFO] Distribution rules: "
        f"active={active_rules}, refine_rules={refine_rules}, "
        f"threshold_refine={bool(GATE_RUNTIME.threshold_refine)}"
    )
    print(f"[INFO] Detailed leaf gate rules: {leaf_gate_rules}")
    print(f"[INFO] Distribution execution modes: {distribution_modes}")
    print(
        "[INFO] Diff residual: "
        f"method=local_mahalanobis_fixed, cov_shrinkage={float(SIGNAL_RUNTIME.cov_shrinkage):.3f}, "
        f"aux_enabled={bool(args.diff_residual_aux_enabled)}, "
        f"aux_lambda={float(args.diff_residual_aux_lambda):.3f}, "
        f"aux_model={str(args.diff_residual_aux_model)}, "
        f"row_chunk_workers={int(SIGNAL_RUNTIME.diff_residual_row_chunk_workers)}"
    )
    print(
        "[INFO] Delta ensemble: "
        f"rp_dims={int(SIGNAL_RUNTIME.delta_ens_rp_dims)}, alpha={float(SIGNAL_RUNTIME.delta_ens_alpha):.3f}, "
        f"cv_mode={SIGNAL_RUNTIME.delta_ens_cv_mode}, kfolds={int(SIGNAL_RUNTIME.delta_ens_kfolds)}, "
        f"split_train_ratio={float(SIGNAL_RUNTIME.delta_ens_split_train_ratio):.2f}, "
        f"residual={SIGNAL_RUNTIME.delta_ens_residual}, fit_intercept={bool(SIGNAL_RUNTIME.delta_ens_fit_intercept)}, "
        f"members=(nystrom:{int(SIGNAL_RUNTIME.delta_ens_members_nystrom)},lowrank:{int(SIGNAL_RUNTIME.delta_ens_members_lowrank)}), "
        f"row_subsample={float(SIGNAL_RUNTIME.delta_ens_row_subsample):.2f}, "
        f"ranks={','.join(map(str, SIGNAL_RUNTIME.delta_ens_ranks))}, "
        f"landmark={SIGNAL_RUNTIME.delta_ens_landmark_policy}:{int(SIGNAL_RUNTIME.delta_ens_landmark_cap)}, "
        f"fusion={SIGNAL_RUNTIME.delta_ens_fusion}, debug_members={bool(SIGNAL_RUNTIME.delta_ens_debug_members)}"
    )
    print(
        "[INFO] UDF: "
        f"enabled={bool(UDF_FIXED_ENABLED)}, iterations={int(UDF_FIXED_ITERATIONS)}, "
        f"core_rules={udf_core_rules}, q_clean={float(UDF_FIXED_Q_CLEAN):.3f}, "
        f"alpha={float(UDF_FIXED_SOFT_ALPHA):.3f}, min_weight={float(UDF_FIXED_MIN_WEIGHT):.3f}"
    )
    print(f"[INFO] Rule quantiles: {rule_quantiles}")
    print(
        "[INFO] Distance calibration: "
        f"mode={CALIBRATION_RUNTIME.mode}, rules={calibration_rules}, "
        f"apply_modes={calibration_apply_modes}, min_precision={CALIBRATION_RUNTIME.min_precision:.3f}, "
        f"cv_folds={CALIBRATION_RUNTIME.cv_folds}"
    )
    print(
        "[INFO] Tri-state: "
        f"enabled={bool(TRISTATE_CFG.enabled)}, mode={TRISTATE_CFG.calibration_mode}, "
        f"rules={tristate_rules}, apply_modes={tristate_apply_modes}, "
        f"min_fail_precision={TRISTATE_CFG.calibration_min_fail_precision:.3f}, "
        f"cv_folds={TRISTATE_CFG.calibration_cv_folds}"
    )
    stem = Path(args.csv_path).stem
    tag = args.tag

    if str(args.inspection_mode) == "detailed":
        detail_leaf_gate_summary_df, detail_leaf_gate_row_df, detail_leaf_gate_report = run_detailed_leaf_distribution_gate(
            input_text_series=np.asarray(in_raw, dtype=object),
            output_series=df[args.output_col],
            group_ids=result_df["distribution_group_id"],
            row_ids=result_df["row_id"],
            hard_gate_mask=result_df["hard_gate_pass"].to_numpy(dtype=bool),
            model=model,
            active_rules=leaf_gate_rules,
            rule_quantiles=rule_quantiles,
            refine_reference=bool(GATE_RUNTIME.threshold_refine),
            refine_rules=refine_rules,
            refine_iterations=int(GATE_RUNTIME.threshold_refine_iterations),
            refine_min_size=int(GATE_RUNTIME.threshold_refine_min_size),
            args=args,
            output_dir=output_dir,
            stem=stem,
            tag=tag,
            min_support=int(args.detail_leaf_min_support),
        )
        print(
            "[INFO] Detailed leaf distribution gate: "
            f"paths={int(detail_leaf_gate_report.get('leaf_paths_evaluated', 0))}, "
            f"row_hits={int(detail_leaf_gate_report.get('leaf_row_hits', 0))}"
        )
        result_df = apply_detail_leaf_gate_to_row_result(
            result_df=result_df,
            leaf_gate_row_df=detail_leaf_gate_row_df,
            gate_rules=leaf_gate_rules,
            distribution_rules=active_rules,
        )
        detail_metrics = build_detail_row_metrics_from_leaf_gate_hits(
            leaf_gate_row_df=detail_leaf_gate_row_df,
            n_rows=len(result_df),
            max_failed_paths_per_row=8,
        )
        result_df["detail_evaluated_nomask"] = detail_metrics["detail_evaluated_nomask"].astype(bool)
        result_df["detail_fail_any_leaf_nomask"] = detail_metrics["detail_fail_any_leaf_nomask"].astype(bool)
        result_df["detail_fail_leaf_count_nomask"] = detail_metrics["detail_fail_leaf_count_nomask"].astype(int)
        result_df["detail_eval_leaf_count_nomask"] = detail_metrics["detail_eval_leaf_count_nomask"].astype(int)
        result_df["detail_failed_leaf_paths_nomask"] = detail_metrics["detail_failed_leaf_paths_nomask"].astype(object)
        detail_leaf_summary_df = detail_leaf_gate_summary_df.copy()
        detail_leaf_hits_df = build_detail_leaf_distribution_hits_from_gate(
            leaf_gate_summary_df=detail_leaf_gate_summary_df,
            leaf_gate_row_df=detail_leaf_gate_row_df,
            signal_rule="output",
        )
        detail_leaf_dist_stats_df, detail_leaf_dist_bins_df, detail_leaf_distribution_report = build_detail_leaf_distribution_artifacts(
            detail_leaf_hits_df,
            bins=int(args.detail_leaf_dist_bins),
        )

        hard_gate_arr = result_df["hard_gate_pass"].fillna(False).astype(bool).to_numpy()
        hard_eval_arr = hard_gate_arr & detail_metrics["detail_evaluated_nomask"].astype(bool)
        detail_fail_arr = detail_metrics["detail_fail_any_leaf_nomask"].astype(bool)
        detail_leaf_report = {
            **detail_leaf_report,
            "mode": "detailed",
            "enabled": True,
            "detail_metric_source": "leaf_distribution_gate",
            "detail_leaf_paths_evaluated": int(detail_leaf_gate_report.get("leaf_paths_evaluated", 0)),
            "detail_row_hits": int(detail_leaf_gate_report.get("leaf_row_hits", 0)),
            "detail_evaluated_rows": int(np.sum(detail_metrics["detail_evaluated_nomask"])),
            "detail_hard_gate_evaluated_rows": int(np.sum(hard_eval_arr)),
            "detail_fail_rows": int(np.sum(detail_fail_arr)),
            "detail_fail_rate_on_hard_evaluated": (
                float(np.mean(detail_fail_arr[hard_eval_arr])) if np.any(hard_eval_arr) else 0.0
            ),
            "distribution": dict(detail_leaf_distribution_report),
        }
        print(
            "[INFO] Detailed decision metrics (leaf distribution gate): "
            f"rows_evaluated={int(np.sum(detail_metrics['detail_evaluated_nomask']))}, "
            f"rows_failed={int(np.sum(detail_metrics['detail_fail_any_leaf_nomask']))}"
        )
    cache_paths = build_embedding_cache_paths(output_dir=root_output_dir, tag=tag, stem=stem)
    cache_writer = EmbeddingCacheWriter(
        paths=cache_paths,
        total_rows=len(result_df),
        dtype="float32",
    )
    cache_meta: dict[str, Any] = {}

    in_text_arr = np.asarray(in_raw, dtype=object)
    out_text_arr = np.asarray(out_raw, dtype=object)
    source_output_arr = result_df["source_output"].fillna("").astype(str).to_numpy(dtype=object)
    summary_df = pd.DataFrame()
    if str(args.inspection_mode) == "detailed":
        hard_idx = np.where(result_df["hard_gate_pass"].fillna(False).astype(bool).to_numpy())[0]
        if len(hard_idx) > 0:
            in_eval_texts = in_text_arr[hard_idx].astype(str).tolist()
            out_eval_texts = out_text_arr[hard_idx].astype(str).tolist()
            in_norm = normalize_rows(sanitize_matrix(model.encode(in_eval_texts, batch_size=int(args.embedding_batch_size))))
            out_norm = normalize_rows(sanitize_matrix(model.encode(out_eval_texts, batch_size=int(args.embedding_batch_size))))
            cache_writer.write(
                row_indices=hard_idx,
                input_norm=np.asarray(in_norm, dtype=np.float32),
                output_norm=np.asarray(out_norm, dtype=np.float32),
            )

        result_df, fallback_summaries, fallback_stats = run_detailed_leaf_group_fallback(
            result_df=result_df,
            group_meta=group_meta,
            in_text_arr=in_text_arr,
            out_text_arr=out_text_arr,
            source_output_arr=source_output_arr,
            model=model,
            active_rules=active_rules,
            rule_quantiles=rule_quantiles,
            refine_reference=bool(GATE_RUNTIME.threshold_refine),
            refine_rules=refine_rules,
            refine_iterations=int(GATE_RUNTIME.threshold_refine_iterations),
            refine_min_size=int(GATE_RUNTIME.threshold_refine_min_size),
            args=args,
            output_dir=output_dir,
            stem=stem,
            tag=f"{tag}_detail_fallback",
            label_coloring_enabled=label_coloring_enabled,
        )
        print(
            "[INFO] Detailed group fallback: "
            f"groups={int(fallback_stats.get('fallback_groups', 0))}, "
            f"rows={int(fallback_stats.get('fallback_rows', 0))}"
        )
        detail_leaf_report["leaf_group_fallback"] = dict(fallback_stats)
        summary_df = build_leaf_only_group_summary(
            result_df=result_df,
            group_meta=group_meta,
            mode_name="nomask",
            active_rules=active_rules,
            rule_quantiles=rule_quantiles,
        )
        if not summary_df.empty and not detail_leaf_gate_summary_df.empty:
            gid_col = "group_id_raw" if "group_id_raw" in detail_leaf_gate_summary_df.columns else "group_id"
            if gid_col in detail_leaf_gate_summary_df.columns:
                gate_sum = detail_leaf_gate_summary_df.copy()
                gate_sum[gid_col] = gate_sum[gid_col].fillna("").astype(str)
                for sidx in summary_df.index:
                    gid = str(summary_df.at[sidx, "group_id"]) if "group_id" in summary_df.columns else ""
                    if not gid:
                        continue
                    sub = gate_sum[gate_sum[gid_col] == gid]
                    if sub.empty:
                        continue
                    for rule in active_rules:
                        key = f"{rule}_threshold"
                        if key not in sub.columns:
                            continue
                        vals = pd.to_numeric(sub[key], errors="coerce").to_numpy(dtype=float)
                        finite = vals[np.isfinite(vals)]
                        if finite.size > 0:
                            summary_df.at[sidx, key] = float(np.median(finite))
        if not summary_df.empty and fallback_summaries:
            for summary in fallback_summaries:
                gid = str(summary.get("group_id", ""))
                if not gid:
                    continue
                s_mask = (
                    summary_df["mode"].astype(str).eq("nomask")
                    & summary_df["group_id"].astype(str).eq(gid)
                )
                if not bool(np.any(s_mask)):
                    continue
                sidx = summary_df.index[s_mask][0]
                for rule in active_rules:
                    key = f"{rule}_threshold"
                    if key in summary:
                        summary_df.at[sidx, key] = float(summary.get(key, np.nan))
    else:
        summaries: list[dict[str, Any]] = []
        print("[INFO] Running distribution signaling mode=nomask")
        if "final_pass_nomask" not in result_df.columns:
            result_df["final_pass_nomask"] = False
        if "distribution_evaluated_nomask" not in result_df.columns:
            result_df["distribution_evaluated_nomask"] = False
        mode_eval_rows = 0
        mode_skip_rows = 0

        for g in group_meta.itertuples(index=False):
            gid = str(g.group_id)
            idx_group = result_df.index[result_df["distribution_group_id"].astype(str) == gid]
            if len(idx_group) == 0:
                continue

            group_hard_mask = result_df.loc[idx_group, "hard_gate_pass"].to_numpy(dtype=bool)
            if group_hard_mask.sum() == 0:
                mode_skip_rows += len(idx_group)
                continue

            idx_eval = idx_group[group_hard_mask]
            idx_eval_arr = np.asarray(idx_eval, dtype=int)
            in_texts, out_texts, src_out_texts = collect_mode_text_triplet_by_indices(
                row_indices=idx_eval_arr,
                input_text_arr=in_text_arr,
                output_text_arr=out_text_arr,
                source_output_arr=source_output_arr,
            )
            mode_eval_rows += len(idx_eval)
            mode_skip_rows += int(len(idx_group) - len(idx_eval))

            if label_coloring_enabled:
                group_label_is_correct = result_df.loc[idx_eval, "label_is_correct"].to_numpy(dtype=float)
                group_label_raw = result_df.loc[idx_eval, "label_raw"].fillna("").astype(str).to_numpy(dtype=object)
            else:
                group_label_is_correct = None
                group_label_raw = None

            mode_df, summary, group_ref_mask = run_distribution_mode_job(
                mode_name="nomask",
                input_texts=in_texts,
                output_texts=out_texts,
                source_output_texts=src_out_texts,
                model=model,
                active_rules=active_rules,
                rule_quantiles=rule_quantiles,
                refine_reference=bool(GATE_RUNTIME.threshold_refine),
                refine_rules=refine_rules,
                refine_iterations=int(GATE_RUNTIME.threshold_refine_iterations),
                refine_min_size=int(GATE_RUNTIME.threshold_refine_min_size),
                args=args,
                output_dir=output_dir,
                stem=stem,
                tag=tag,
                group_id=gid,
                label_is_correct=group_label_is_correct,
                label_raw=group_label_raw,
                cache_writer=cache_writer,
                cache_row_indices=idx_eval_arr,
            )
            result_df = assign_subset_columns(result_df, idx_eval, mode_df)
            result_df.loc[idx_eval, "distribution_evaluated_nomask"] = True
            result_df.loc[idx_eval, "final_pass_nomask"] = (
                result_df.loc[idx_eval, "distribution_pass_nomask"].astype(bool)
            )

            summary["group_col"] = getattr(g, "group_col", group_col_used)
            summary["group_size"] = int(getattr(g, "group_size", len(idx_group)))
            summary["group_eval_size"] = int(len(idx_eval))
            summary["group_value_preview"] = getattr(g, "group_value_preview", "")
            summary["group_ref_init_size"] = int(group_ref_mask.sum())
            summary["group_ref_size"] = int(summary.get("threshold_ref_size", group_ref_mask.sum()))
            summary["hard_gate_pass_rate"] = float(result_df.loc[idx_group, "hard_gate_pass"].mean())
            summary["distribution_eval_coverage"] = float(len(idx_eval) / max(len(idx_group), 1))
            summary["final_pass_rate"] = float(result_df.loc[idx_eval, "final_pass_nomask"].mean())
            summaries.append(summary)
        print(
            f"[INFO] mode=nomask distribution evaluated rows={mode_eval_rows}, "
            f"skipped_by_hard_gate={mode_skip_rows}"
        )
        summary_df = pd.DataFrame(summaries)

    cache_meta = cache_writer.finalize(
        extra_meta={
            "source": "distribution_outlier_pipeline",
            "scope": "nomask",
            "tag": str(tag),
            "stem": str(stem),
            "rows": int(len(result_df)),
        }
    )

    result_df, summary_df = _ensure_distance_calibration_columns(result_df, summary_df)

    for col in ["consistency_observe_only"]:
        if col not in summary_df.columns:
            summary_df[col] = bool(GATE_RUNTIME.consistency_observe_only)
        else:
            summary_df[col] = bool(GATE_RUNTIME.consistency_observe_only)
    if "distance_calibration_mode" in summary_df.columns:
        summary_df["distance_calibration_mode"] = str(CALIBRATION_RUNTIME.mode)
    if "distance_calibration_rules" in summary_df.columns:
        summary_df["distance_calibration_rules"] = ",".join(calibration_rules)

    calibration_base_path = output_dir / f"{tag}_{stem}_distance_calibration"
    calibration_json_path = (
        Path(CALIBRATION_RUNTIME.path) if CALIBRATION_RUNTIME.path else calibration_base_path.with_suffix(".json")
    )
    calibration_cv_metrics_path = output_dir / f"{tag}_{stem}_distance_calibration_cv_metrics.csv"
    calibration_candidates_path = output_dir / f"{tag}_{stem}_distance_calibration_candidates.csv"
    calibration_summary_path = output_dir / f"{tag}_{stem}_distance_calibration_summary.json"

    calibration_bundle: dict[str, Any] = {
        "rules": calibration_rules,
        "cv_folds": int(CALIBRATION_RUNTIME.cv_folds),
        "min_precision": float(CALIBRATION_RUNTIME.min_precision),
        "modes": {},
    }
    calibration_candidate_frames: list[pd.DataFrame] = []
    calibration_cv_frames: list[pd.DataFrame] = []

    if CALIBRATION_RUNTIME.mode == "fit":
        if CALIBRATION_RUNTIME.label_col not in df.columns:
            raise ValueError(
                f"DISTANCE_CALIBRATION_RUNTIME.mode=fit requires label column: {CALIBRATION_RUNTIME.label_col}"
            )
        raw_label_series = df[CALIBRATION_RUNTIME.label_col].fillna("").astype(str)
        known_label_mask = raw_label_series.str.strip().ne("")
        y_is_bad_all = parse_calibration_label(raw_label_series, bad_value=CALIBRATION_RUNTIME.bad_value)

        for mode in calibration_apply_modes:
            eval_col = f"distribution_evaluated_{mode}"
            if eval_col not in result_df.columns:
                continue
            mode_eval_mask = (
                result_df[eval_col].fillna(False).astype(bool).to_numpy()
                & known_label_mask.to_numpy(dtype=bool)
            )
            if mode_eval_mask.sum() < 20:
                print(
                    f"[WARN] Distance calibration skipped for mode={mode}: "
                    f"insufficient eval+label rows ({int(mode_eval_mask.sum())})"
                )
                continue
            fitted = calibrate_mode_quantiles(
                df=result_df,
                mode=mode,
                y_is_bad=y_is_bad_all,
                eval_mask=mode_eval_mask,
                rules=calibration_rules,
                output_quantiles=output_quantile_candidates,
                other_quantiles=other_quantile_candidates,
                cv_folds=CALIBRATION_RUNTIME.cv_folds,
                min_precision=CALIBRATION_RUNTIME.min_precision,
                group_col="distribution_group_id",
                seed=args.sample_seed,
            )
            selected_quantiles = dict(fitted["selected_quantiles"])
            selected_metrics = dict(fitted["selected_metrics"])
            print(
                f"[INFO] Distance calibration selected mode={mode}: "
                f"quantiles={selected_quantiles}, "
                f"cv_precision={selected_metrics.get('mean_precision', np.nan):.4f}, "
                f"cv_f1={selected_metrics.get('mean_f1', np.nan):.4f}, "
                f"cv_fp={selected_metrics.get('mean_fp', np.nan):.2f}, "
                f"constraint_met={bool(selected_metrics.get('precision_constraint_met', False))}"
            )

            _apply_distance_calibration_to_mode(
                result_df=result_df,
                summary_df=summary_df,
                mode=mode,
                active_rules=active_rules,
                rules=calibration_rules,
                selected_quantiles=selected_quantiles,
                group_thresholds=fitted["group_thresholds"],
                global_thresholds=fitted["global_thresholds"],
                selected_metrics=selected_metrics,
                calibration_mode="fit",
            )
            calibration_bundle["modes"][mode] = {
                "rules": calibration_rules,
                "selected_quantiles": selected_quantiles,
                "selected_metrics": selected_metrics,
            }
            cand_df = fitted["candidate_df"].copy()
            cand_df["mode"] = mode
            calibration_candidate_frames.append(cand_df)
            cv_df = fitted["cv_df"].copy()
            cv_df["mode"] = mode
            calibration_cv_frames.append(cv_df)

        flat = flatten_calibration_json(calibration_bundle)
        calibration_json_path.write_text(json.dumps(flat, ensure_ascii=False, indent=2))
        if calibration_candidate_frames:
            pd.concat(calibration_candidate_frames, ignore_index=True).to_csv(calibration_candidates_path, index=False)
        else:
            pd.DataFrame().to_csv(calibration_candidates_path, index=False)
        if calibration_cv_frames:
            pd.concat(calibration_cv_frames, ignore_index=True).to_csv(calibration_cv_metrics_path, index=False)
        else:
            pd.DataFrame().to_csv(calibration_cv_metrics_path, index=False)
        calibration_summary_path.write_text(
            json.dumps(
                {
                    "mode": "fit",
                    "rules": calibration_rules,
                    "apply_modes": calibration_apply_modes,
                    "min_precision": float(CALIBRATION_RUNTIME.min_precision),
                    "cv_folds": int(CALIBRATION_RUNTIME.cv_folds),
                    "output_quantiles": output_quantile_candidates,
                    "other_quantiles": other_quantile_candidates,
                    "calibration_json_path": str(calibration_json_path),
                    "candidates_csv_path": str(calibration_candidates_path),
                    "cv_metrics_csv_path": str(calibration_cv_metrics_path),
                    "modes": flat.get("modes", {}),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        print(f"[DONE] Distance calibration json: {calibration_json_path}")
        print(f"[DONE] Distance calibration candidates: {calibration_candidates_path}")
        print(f"[DONE] Distance calibration CV metrics: {calibration_cv_metrics_path}")
        print(f"[DONE] Distance calibration summary: {calibration_summary_path}")

    elif CALIBRATION_RUNTIME.mode == "apply":
        if not calibration_json_path.exists():
            raise FileNotFoundError(
                f"DISTANCE_CALIBRATION_RUNTIME.mode=apply requires calibration json: {calibration_json_path}"
            )
        loaded = json.loads(calibration_json_path.read_text())
        modes_map = loaded.get("modes", {})
        for mode in calibration_apply_modes:
            eval_col = f"distribution_evaluated_{mode}"
            if eval_col not in result_df.columns:
                continue
            mode_info = modes_map.get(mode)
            if not mode_info:
                print(f"[WARN] Calibration JSON has no mode={mode}; skipping.")
                continue
            selected_quantiles = {k: float(v) for k, v in dict(mode_info.get("selected_quantiles", {})).items()}
            if not selected_quantiles:
                print(f"[WARN] Calibration JSON has empty selected_quantiles for mode={mode}; skipping.")
                continue
            mode_eval_mask = result_df[eval_col].fillna(False).astype(bool).to_numpy()
            if mode_eval_mask.sum() < 1:
                continue
            applied = apply_mode_quantiles(
                df=result_df,
                mode=mode,
                rules=calibration_rules,
                selected_quantiles=selected_quantiles,
                eval_mask=mode_eval_mask,
                group_col="distribution_group_id",
            )
            _apply_distance_calibration_to_mode(
                result_df=result_df,
                summary_df=summary_df,
                mode=mode,
                active_rules=active_rules,
                rules=calibration_rules,
                selected_quantiles=selected_quantiles,
                group_thresholds=applied["group_thresholds"],
                global_thresholds=applied["global_thresholds"],
                selected_metrics=dict(mode_info.get("selected_metrics", {})),
                calibration_mode="apply",
            )
            print(f"[INFO] Applied distance calibration for mode={mode}: quantiles={selected_quantiles}")

        if calibration_summary_path.exists():
            prev = json.loads(calibration_summary_path.read_text())
        else:
            prev = {}
        prev.update(
            {
                "mode": "apply",
                "rules": calibration_rules,
                "apply_modes": calibration_apply_modes,
                "calibration_json_path": str(calibration_json_path),
            }
        )
        calibration_summary_path.write_text(json.dumps(prev, ensure_ascii=False, indent=2))
        print(f"[DONE] Distance calibration summary: {calibration_summary_path}")

    result_df, summary_df = _ensure_tristate_columns(result_df, summary_df, tristate_rules)
    if not summary_df.empty:
        summary_df["tristate_enabled"] = bool(TRISTATE_CFG.enabled)
        summary_df["tristate_calibration_mode"] = str(TRISTATE_CFG.calibration_mode)
        summary_df["tristate_calibration_rules"] = ",".join(tristate_rules)

    tristate_fail_quantiles = {r: float(rule_quantiles.get(r, SIGNAL_RUNTIME.signal_quantile)) for r in tristate_rules}
    tristate_warn_quantiles = resolve_tristate_warn_quantiles(
        fail_quantiles=tristate_fail_quantiles,
        tristate_rules=tristate_rules,
    )
    tristate_base_path = output_dir / f"{tag}_{stem}_tristate_calibration"
    tristate_json_path = (
        Path(TRISTATE_CFG.calibration_path) if TRISTATE_CFG.calibration_path else tristate_base_path.with_suffix(".json")
    )
    tristate_cv_metrics_path = output_dir / f"{tag}_{stem}_tristate_calibration_cv_metrics.csv"
    tristate_candidates_path = output_dir / f"{tag}_{stem}_tristate_calibration_candidates.csv"
    tristate_summary_path = output_dir / f"{tag}_{stem}_tristate_calibration_summary.json"

    tristate_selected_map: dict[str, dict[str, Any]] = {}
    tristate_candidate_frames: list[pd.DataFrame] = []
    tristate_cv_frames: list[pd.DataFrame] = []
    tristate_bundle: dict[str, Any] = {
        "rules": tristate_rules,
        "cv_folds": int(TRISTATE_CFG.calibration_cv_folds),
        "min_fail_precision": float(TRISTATE_CFG.calibration_min_fail_precision),
        "warn_grid_quantiles": tristate_grid_warn,
        "fail_grid_quantiles": tristate_grid_fail,
        "modes": {},
    }

    tri_label_known_mask = np.zeros(len(df), dtype=bool)
    tri_y_is_bad_all = np.zeros(len(df), dtype=bool)
    if TRISTATE_CFG.enabled and (
        TRISTATE_CFG.calibration_mode == "fit" or TRISTATE_CFG.calibration_label_col in df.columns
    ):
        if TRISTATE_CFG.calibration_label_col not in df.columns and TRISTATE_CFG.calibration_mode == "fit":
            raise ValueError(
                f"TRISTATE_RUNTIME.calibration_mode=fit requires label column: {TRISTATE_CFG.calibration_label_col}"
            )
        if TRISTATE_CFG.calibration_label_col in df.columns:
            tri_label_series = df[TRISTATE_CFG.calibration_label_col].fillna("").astype(str)
            tri_label_known_mask = tri_label_series.str.strip().ne("").to_numpy(dtype=bool)
            tri_y_is_bad_all = parse_calibration_label(
                tri_label_series,
                bad_value=TRISTATE_CFG.calibration_bad_value,
            )

    tristate_loaded = None
    if TRISTATE_CFG.enabled and TRISTATE_CFG.calibration_mode == "apply":
        if not tristate_json_path.exists():
            raise FileNotFoundError(
                f"TRISTATE_RUNTIME.calibration_mode=apply requires calibration json: {tristate_json_path}"
            )
        tristate_loaded = json.loads(tristate_json_path.read_text())

    if TRISTATE_CFG.enabled:
        groups = result_df["distribution_group_id"].astype(str).to_numpy(dtype=object)
        min_fit_rows = max(20, int(TRISTATE_CFG.calibration_cv_folds) * 2)

        for mode in tristate_apply_modes:
            eval_col = f"distribution_evaluated_{mode}"
            if eval_col not in result_df.columns:
                continue
            mode_eval_mask = result_df[eval_col].fillna(False).astype(bool).to_numpy()
            if mode_eval_mask.sum() == 0:
                continue

            mode_selected: dict[str, Any] = {}
            for rule in tristate_rules:
                signal_col = f"{RULE_SIGNAL_PREFIX[rule]}_{mode}"
                if signal_col not in result_df.columns:
                    continue

                signals = result_df[signal_col].to_numpy(dtype=float)
                available_mask = get_rule_available_mask(result_df=result_df, rule=rule, mode=mode)

                selected_q_warn = float(tristate_warn_quantiles[rule])
                selected_q_fail = float(tristate_fail_quantiles[rule])
                selected_metrics: dict[str, Any] = {}

                if TRISTATE_CFG.calibration_mode == "fit":
                    fit_eval_mask = mode_eval_mask & tri_label_known_mask & available_mask
                    if fit_eval_mask.sum() >= min_fit_rows:
                        fitted = calibrate_rule_tristate(
                            scores=signals,
                            groups=groups,
                            y_bad=tri_y_is_bad_all,
                            eval_mask=fit_eval_mask,
                            available_mask=available_mask,
                            warn_quantiles=tristate_grid_warn,
                            fail_quantiles=tristate_grid_fail,
                            cv_folds=TRISTATE_CFG.calibration_cv_folds,
                            min_fail_precision=TRISTATE_CFG.calibration_min_fail_precision,
                            seed=args.sample_seed,
                        )
                        selected_q_warn = float(fitted["selected_quantiles"]["warn"])
                        selected_q_fail = float(fitted["selected_quantiles"]["fail"])
                        selected_metrics = dict(fitted.get("selected_metrics", {}))

                        cand_df = fitted["candidate_df"].copy()
                        cand_df["mode"] = mode
                        cand_df["rule"] = rule
                        tristate_candidate_frames.append(cand_df)

                        cv_df = fitted["cv_df"].copy()
                        cv_df["mode"] = mode
                        cv_df["rule"] = rule
                        tristate_cv_frames.append(cv_df)
                    else:
                        print(
                            f"[WARN] Tri-state calibration fallback to default quantiles "
                            f"for mode={mode}, rule={rule}: insufficient labeled rows ({int(fit_eval_mask.sum())})"
                        )

                elif TRISTATE_CFG.calibration_mode == "apply":
                    mode_info = (tristate_loaded or {}).get("modes", {}).get(mode, {})
                    mode_rules = mode_info.get("rules", {}) if isinstance(mode_info, dict) else {}
                    rule_info = mode_rules.get(rule) if isinstance(mode_rules, dict) else None
                    if rule_info is None and isinstance(mode_info, dict):
                        rule_info = mode_info.get(rule)
                    if rule_info is not None:
                        qmap = dict(rule_info.get("selected_quantiles", {}))
                        if "warn" in qmap and "fail" in qmap:
                            selected_q_warn = float(qmap["warn"])
                            selected_q_fail = float(qmap["fail"])
                            selected_metrics = dict(rule_info.get("selected_metrics", {}))
                        else:
                            print(
                                f"[WARN] Tri-state apply missing warn/fail quantiles "
                                f"for mode={mode}, rule={rule}; fallback to defaults."
                            )
                    else:
                        print(f"[WARN] Tri-state apply missing mode={mode}, rule={rule}; fallback to defaults.")

                applied = apply_rule_tristate(
                    scores=signals,
                    groups=groups,
                    eval_mask=mode_eval_mask,
                    available_mask=available_mask,
                    q_warn=selected_q_warn,
                    q_fail=selected_q_fail,
                )
                if rule in {"discourse_instability", "contradiction", "self_contradiction"}:
                    # Semantic unavailable rows are warn-only and non-blocking.
                    states = applied["states"]
                    unavailable = mode_eval_mask & (~available_mask)
                    if np.any(unavailable):
                        status_arr = np.asarray(states["status"], dtype=object).copy()
                        pass_arr = np.asarray(states["pass"], dtype=bool).copy()
                        warn_arr = np.asarray(states["warn"], dtype=bool).copy()
                        fail_arr = np.asarray(states["fail"], dtype=bool).copy()
                        status_arr[unavailable] = "warn"
                        pass_arr[unavailable] = False
                        warn_arr[unavailable] = True
                        fail_arr[unavailable] = False
                        applied["states"] = {
                            "status": status_arr,
                            "pass": pass_arr,
                            "warn": warn_arr,
                            "fail": fail_arr,
                        }

                if tri_label_known_mask.any():
                    eval_label_mask = mode_eval_mask & tri_label_known_mask
                    eval_metrics = evaluate_rule_states(
                        y_bad=tri_y_is_bad_all,
                        states=applied["states"],
                        eval_mask=eval_label_mask,
                    )
                    if eval_metrics:
                        selected_metrics = {**selected_metrics, **eval_metrics}

                _apply_tristate_rule_mode(
                    result_df=result_df,
                    summary_df=summary_df,
                    mode=mode,
                    rule=rule,
                    states=applied["states"],
                    global_thresholds=dict(applied["global_thresholds"]),
                    group_thresholds=dict(applied["group_thresholds"]),
                )

                metrics_jsonable: dict[str, Any] = {}
                for k, v in selected_metrics.items():
                    if isinstance(v, (bool, np.bool_)):
                        metrics_jsonable[k] = bool(v)
                    elif isinstance(v, (int, float, np.integer, np.floating)):
                        metrics_jsonable[k] = float(v)
                    else:
                        metrics_jsonable[k] = v
                mode_selected[rule] = {
                    "selected_quantiles": {"warn": float(selected_q_warn), "fail": float(selected_q_fail)},
                    "selected_metrics": metrics_jsonable,
                    "global_thresholds": {
                        "warn": float(applied["global_thresholds"]["warn"]),
                        "fail": float(applied["global_thresholds"]["fail"]),
                    },
                    "group_thresholds": dict(applied["group_thresholds"]),
                    "eval_support": int(applied.get("eval_support", 0)),
                }

            tristate_selected_map[mode] = mode_selected

        if TRISTATE_CFG.calibration_mode == "fit":
            for mode in tristate_apply_modes:
                if mode in tristate_selected_map:
                    tristate_bundle["modes"][mode] = {"rules": tristate_selected_map[mode]}
            tristate_json_path.write_text(json.dumps(tristate_bundle, ensure_ascii=False, indent=2))
            if tristate_candidate_frames:
                pd.concat(tristate_candidate_frames, ignore_index=True).to_csv(tristate_candidates_path, index=False)
            else:
                pd.DataFrame().to_csv(tristate_candidates_path, index=False)
            if tristate_cv_frames:
                pd.concat(tristate_cv_frames, ignore_index=True).to_csv(tristate_cv_metrics_path, index=False)
            else:
                pd.DataFrame().to_csv(tristate_cv_metrics_path, index=False)
            tristate_summary_path.write_text(
                json.dumps(
                    {
                        "mode": "fit",
                        "rules": tristate_rules,
                        "apply_modes": tristate_apply_modes,
                        "cv_folds": int(TRISTATE_CFG.calibration_cv_folds),
                        "min_fail_precision": float(TRISTATE_CFG.calibration_min_fail_precision),
                        "warn_grid_quantiles": tristate_grid_warn,
                        "fail_grid_quantiles": tristate_grid_fail,
                        "calibration_json_path": str(tristate_json_path),
                        "candidates_csv_path": str(tristate_candidates_path),
                        "cv_metrics_csv_path": str(tristate_cv_metrics_path),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            print(f"[DONE] Tri-state calibration json: {tristate_json_path}")
            print(f"[DONE] Tri-state calibration candidates: {tristate_candidates_path}")
            print(f"[DONE] Tri-state calibration CV metrics: {tristate_cv_metrics_path}")
            print(f"[DONE] Tri-state calibration summary: {tristate_summary_path}")
        elif TRISTATE_CFG.calibration_mode == "apply":
            prev_tri = {}
            if tristate_summary_path.exists():
                prev_tri = json.loads(tristate_summary_path.read_text())
            prev_tri.update(
                {
                    "mode": "apply",
                    "rules": tristate_rules,
                    "apply_modes": tristate_apply_modes,
                    "calibration_json_path": str(tristate_json_path),
                }
            )
            tristate_summary_path.write_text(json.dumps(prev_tri, ensure_ascii=False, indent=2))
            print(f"[DONE] Tri-state calibration summary: {tristate_summary_path}")
    else:
        print("[INFO] Tri-state output disabled by TRISTATE_RUNTIME.enabled=False.")

    if not summary_df.empty:
        for mode in ("nomask",):
            s_mask = summary_df["mode"].astype(str) == mode
            if not s_mask.any():
                continue
            mode_payload = tristate_selected_map.get(mode, {})
            summary_df.loc[s_mask, "tristate_calibration_selected_json"] = json.dumps(
                mode_payload, ensure_ascii=False
            )

    detail_override_stats: dict[str, Any] = {
        "detail_mode_enabled": False,
        "detail_target_rows": 0,
        "detail_override_rows": 0,
        "distribution_pass_rate_pre": float("nan"),
        "distribution_pass_rate_post": float("nan"),
        "final_pass_rate_pre": float("nan"),
        "final_pass_rate_post": float("nan"),
    }
    if str(args.inspection_mode) == "detailed":
        print(
            "[INFO] Detailed override(nomask): skipped in distribution pipeline "
            "(deferred to run_final_metric)."
        )

    hard_gate_all = result_df["hard_gate_pass"].fillna(False).astype(bool).to_numpy()
    detail_eval_all = result_df["detail_evaluated_nomask"].fillna(False).astype(bool).to_numpy()
    detail_fail_all = result_df["detail_fail_any_leaf_nomask"].fillna(False).astype(bool).to_numpy()
    hard_eval_all = hard_gate_all & detail_eval_all
    detail_fail_rate_hard_eval = float(np.mean(detail_fail_all[hard_eval_all])) if np.any(hard_eval_all) else 0.0
    if not summary_df.empty:
        summary_df["inspection_mode"] = str(args.inspection_mode)
        summary_df["detail_mode_enabled"] = bool(str(args.inspection_mode) == "detailed")
        summary_df["detail_fail_rate_hard_eval"] = float(detail_fail_rate_hard_eval)
        summary_df["detail_override_rows"] = int(detail_override_stats.get("detail_override_rows", 0))
        summary_df["detail_distribution_pass_rate_pre"] = float(
            detail_override_stats.get("distribution_pass_rate_pre", np.nan)
        )
        summary_df["detail_distribution_pass_rate_post"] = float(
            detail_override_stats.get("distribution_pass_rate_post", np.nan)
        )
        summary_df["detail_final_pass_rate_pre"] = float(detail_override_stats.get("final_pass_rate_pre", np.nan))
        summary_df["detail_final_pass_rate_post"] = float(detail_override_stats.get("final_pass_rate_post", np.nan))

    top_anom_path = export_top_anomalies(
        result_df=result_df,
        source_df=df,
        summary_df=summary_df,
        args=args,
        output_dir=output_dir,
        stem=stem,
        tag=tag,
    )

    row_path = output_dir / f"{tag}_{stem}_row_results.csv"
    summary_path = output_dir / f"{tag}_{stem}_summary.csv"
    input_tpl_path = output_dir / f"{tag}_{stem}_input_template_stats.csv"
    output_tpl_path = output_dir / f"{tag}_{stem}_output_template_stats.csv"
    detail_leaf_summary_path = output_dir / f"{tag}_{stem}_detail_leaf_summary.csv"
    detail_leaf_hits_path = output_dir / f"{tag}_{stem}_detail_leaf_row_hits.csv"
    detail_leaf_report_path = output_dir / f"{tag}_{stem}_detail_leaf_report.json"
    detail_leaf_dist_stats_path = output_dir / f"{tag}_{stem}_detail_leaf_distribution_stats.csv"
    detail_leaf_dist_bins_path = output_dir / f"{tag}_{stem}_detail_leaf_distribution_bins.csv"
    detail_leaf_dist_html_path = output_dir / f"{tag}_{stem}_detail_leaf_distribution.html"
    detail_leaf_gate_summary_path = output_dir / f"{tag}_{stem}_detail_leaf_gate_summary.csv"
    detail_leaf_gate_rows_path = output_dir / f"{tag}_{stem}_detail_leaf_gate_row_hits.csv"
    detail_leaf_gate_report_path = output_dir / f"{tag}_{stem}_detail_leaf_gate_report.json"
    top_anom_print_path = top_anom_path
    schema_path = output_dir / f"{tag}_{stem}_schema_used.json"
    config_path = output_dir / f"{tag}_{stem}_run_config.json"

    result_df.to_csv(row_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    in_path_stats.to_csv(input_tpl_path, index=False)
    out_path_stats.to_csv(output_tpl_path, index=False)
    if str(args.inspection_mode) == "detailed":
        detail_leaf_summary_df.to_csv(detail_leaf_summary_path, index=False)
        detail_leaf_hits_df.to_csv(detail_leaf_hits_path, index=False)
        detail_leaf_dist_stats_df.to_csv(detail_leaf_dist_stats_path, index=False)
        detail_leaf_dist_bins_df.to_csv(detail_leaf_dist_bins_path, index=False)
        render_detail_leaf_distribution_html(
            stats_df=detail_leaf_dist_stats_df,
            hist_df=detail_leaf_dist_bins_df,
            out_html=detail_leaf_dist_html_path,
        )
        detail_leaf_gate_summary_df.to_csv(detail_leaf_gate_summary_path, index=False)
        detail_leaf_gate_row_df.to_csv(detail_leaf_gate_rows_path, index=False)
        detail_leaf_gate_report_path.write_text(
            json.dumps(detail_leaf_gate_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        detail_leaf_report_path.write_text(
            json.dumps(detail_leaf_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if schema is not None:
        schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2))
    config_path.write_text(
        json.dumps(
            {
                "args": vars(args),
                "inspection_mode": str(args.inspection_mode),
                "distribution_modes": distribution_modes,
                "detailed_inspection": {
                    "enabled": bool(str(args.inspection_mode) == "detailed"),
                    "detail_leaf_min_support": int(args.detail_leaf_min_support),
                    "detail_output_quantile": float(args.detail_output_quantile),
                    "detail_leaf_dist_bins": int(args.detail_leaf_dist_bins),
                    "summary_csv": (str(detail_leaf_summary_path) if str(args.inspection_mode) == "detailed" else ""),
                    "row_hits_csv": (str(detail_leaf_hits_path) if str(args.inspection_mode) == "detailed" else ""),
                    "report_json": (str(detail_leaf_report_path) if str(args.inspection_mode) == "detailed" else ""),
                    "distribution_stats_csv": (
                        str(detail_leaf_dist_stats_path) if str(args.inspection_mode) == "detailed" else ""
                    ),
                    "distribution_bins_csv": (
                        str(detail_leaf_dist_bins_path) if str(args.inspection_mode) == "detailed" else ""
                    ),
                    "distribution_html": (
                        str(detail_leaf_dist_html_path) if str(args.inspection_mode) == "detailed" else ""
                    ),
                    "gate_summary_csv": (
                        str(detail_leaf_gate_summary_path) if str(args.inspection_mode) == "detailed" else ""
                    ),
                    "gate_row_hits_csv": (
                        str(detail_leaf_gate_rows_path) if str(args.inspection_mode) == "detailed" else ""
                    ),
                    "gate_report_json": (
                        str(detail_leaf_gate_report_path) if str(args.inspection_mode) == "detailed" else ""
                    ),
                    "report": detail_leaf_report,
                    "distribution_report": detail_leaf_distribution_report,
                    "gate_report": detail_leaf_gate_report,
                    "override": detail_override_stats,
                },
                "runtime_profile": {
                    "distribution_mode": "nomask_only",
                    "plot_export": "always_on",
                    "hard_gate": "always_on",
                    "diff_residual_aux": {
                        "enabled": bool(args.diff_residual_aux_enabled),
                        "lambda": float(args.diff_residual_aux_lambda),
                        "model": str(args.diff_residual_aux_model),
                        "row_chunk_workers": int(SIGNAL_RUNTIME.diff_residual_row_chunk_workers),
                    },
                    "active_distribution_rules": list(active_rules),
                    "leaf_gate_rules": list(leaf_gate_rules),
                },
                "used_encoding": used_encoding,
                "source_rows": source_rows,
                "rows": len(df),
                "schema_source": schema_source,
                "embedding_cache": {
                    "scope": "nomask",
                    "meta_json": str(cache_paths.meta_json_path.resolve()),
                    "input_norm_path": str(cache_paths.input_norm_path.resolve()),
                    "output_norm_path": str(cache_paths.output_norm_path.resolve()),
                    "status": "ready" if bool(cache_meta) else "empty",
                    "valid_rows": int(cache_meta.get("valid_rows", 0)),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    print(f"[DONE] Row results: {row_path}")
    print(f"[DONE] Summary: {summary_path}")
    print(f"[DONE] Input template stats: {input_tpl_path}")
    print(f"[DONE] Output template stats: {output_tpl_path}")
    if str(args.inspection_mode) == "detailed":
        print(f"[DONE] Detail leaf summary: {detail_leaf_summary_path}")
        print(f"[DONE] Detail leaf row hits: {detail_leaf_hits_path}")
        print(f"[DONE] Detail leaf report: {detail_leaf_report_path}")
        print(f"[DONE] Detail leaf distribution stats: {detail_leaf_dist_stats_path}")
        print(f"[DONE] Detail leaf distribution bins: {detail_leaf_dist_bins_path}")
        print(f"[DONE] Detail leaf distribution html: {detail_leaf_dist_html_path}")
        print(f"[DONE] Detail leaf gate summary: {detail_leaf_gate_summary_path}")
        print(f"[DONE] Detail leaf gate row hits: {detail_leaf_gate_rows_path}")
        print(f"[DONE] Detail leaf gate report: {detail_leaf_gate_report_path}")
    print(f"[DONE] Top anomalies: {top_anom_print_path}")
    print(f"[DONE] Run config: {config_path}")
    if cache_meta:
        print(f"[DONE] Embedding cache meta: {cache_paths.meta_json_path.resolve()}")
    if schema is not None:
        print(f"[DONE] Schema used: {schema_path}")

    return DistributionRunArtifacts(
        output_dir=root_output_dir.resolve(),
        report_dir=output_dir.resolve(),
        row_results_csv=row_path.resolve(),
        summary_csv=summary_path.resolve(),
        run_config_json=config_path.resolve(),
    )


def main() -> DistributionRunArtifacts:
    return _run_with_args(parse_args())
