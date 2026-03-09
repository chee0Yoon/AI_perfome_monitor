#!/usr/bin/env python3
"""Generate independent mock datasets and validate final_metric on each set.

This runner creates N datasets (default 20), each with 100-200 rows and:
- 90%~98% correct labels
- independent prompt/topic and output style per set
- mostly hard-gate pass with a few intentionally fail-heavy sets
- short/long student inputs and simple/complex JSON outputs

For each dataset it writes:
- source CSV (same schema as ambiguous_prompt_benchmark_v3_large.csv)
- synthetic row_results CSV (nomask score columns)
- final_metric outputs (row_results/summary/thresholds/diagnostics html)

Finally it writes aggregate validation reports.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SOURCE_COLUMNS = [
    "id",
    "Prompt",
    "input",
    "expectedOutput",
    "sample_label",
    "truth_label",
    "input_edge_type",
    "bad_type",
    "bad_family",
    "output_json_valid",
    "eval",
]

RULES = [
    "output",
    "direction",
    "length",
    "diff_residual",
    "delta_ridge_ens",
    "similar_input_conflict",
    "discourse_instability",
    "contradiction",
]

TOPICS = [
    "climate policy reflection",
    "digital citizenship essay",
    "literary character analysis",
    "historical source critique",
    "community service journal",
    "scientific claim evaluation",
    "economics argument response",
    "media bias reflection",
    "ethics case analysis",
    "college application draft review",
    "lab report reasoning check",
    "civic debate rebuttal",
    "comparative poetry response",
    "environmental justice memo",
    "public health awareness brief",
    "startup pitch reflection",
    "STEM project retrospective",
    "sociology observation report",
    "cross-cultural narrative review",
    "philosophy short argument",
]

BAD_TYPES = [
    ("wrong_label", "direction"),
    ("empty_feedback", "length"),
    ("missing_feedback_key", "output"),
    ("malformed_json", "output"),
    ("non_json_text", "output"),
    ("wrong_enum", "contradiction"),
    ("off_topic_long", "diff_residual"),
]

FORMAT_STYLES = [
    "simple",
    "simple_plus",
    "nested_rubric",
    "evidence_list",
    "deep_nested",
]

TEXT_PROFILES = [
    "short",
    "mixed",
    "long",
    "mixed",
]

ADJECTIVES = [
    "clear",
    "inconsistent",
    "nuanced",
    "direct",
    "partial",
    "careful",
    "uncertain",
    "coherent",
    "shallow",
    "focused",
]

NOUNS = [
    "claim",
    "evidence",
    "reasoning",
    "argument",
    "example",
    "structure",
    "tone",
    "interpretation",
    "summary",
    "conclusion",
]

VERBS = [
    "supports",
    "misses",
    "connects",
    "extends",
    "confuses",
    "reframes",
    "clarifies",
    "overstates",
    "balances",
    "summarizes",
]


@dataclass
class DatasetSpec:
    dataset_id: str
    topic: str
    n_rows: int
    incorrect_ratio: float
    hard_gate_target: float
    format_style: str
    text_profile: str
    seed: int
    fail_heavy: bool


@dataclass
class RunOutcome:
    dataset_id: str
    n_rows: int
    incorrect_ratio: float
    hard_gate_pass_rate: float
    selected_methods: str
    all_threshold_finite: bool
    mean_f1: float
    min_f1: float
    mean_recall: float
    mean_fpr: float
    output_threshold: float
    output_f1: float
    final_pass_rate: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate 20 mock datasets and run final_metric validation")
    p.add_argument("--output-root", default="", help="Optional output root. Default: final_metric/results/mock_suite_<ts>")
    p.add_argument("--set-count", type=int, default=20)
    p.add_argument("--min-rows", type=int, default=100)
    p.add_argument("--max-rows", type=int, default=200)
    p.add_argument("--seed", type=int, default=20260225)
    p.add_argument("--tail-direction", default="two_sided", choices=["upper", "lower", "two_sided"])
    p.add_argument("--rules", default=",".join(RULES))
    p.add_argument("--skip-run", action="store_true", help="Only generate datasets and row_results")
    return p.parse_args()


def _topic_for(idx: int) -> str:
    if idx < len(TOPICS):
        return TOPICS[idx]
    return f"{TOPICS[idx % len(TOPICS)]} variant {idx // len(TOPICS) + 1}"


def _sentence(rng: np.random.Generator, topic: str) -> str:
    a = rng.choice(ADJECTIVES)
    n = rng.choice(NOUNS)
    v = rng.choice(VERBS)
    return f"The student {v} a {a} {n} about {topic}."


def _make_student_response(rng: np.random.Generator, topic: str, profile: str) -> str:
    if profile == "short":
        n_sent = int(rng.integers(1, 3))
    elif profile == "long":
        n_sent = int(rng.integers(8, 14))
    else:
        n_sent = int(rng.integers(3, 9))
    sents = [_sentence(rng, topic) for _ in range(n_sent)]
    return " ".join(sents)


def _truth_label_from_response(student_response: str) -> str:
    length = len(student_response)
    if length < 120:
        return "Incorrect"
    # Simple deterministic split with realistic mix.
    return "Correct" if (length % 7) >= 2 else "Incorrect"


def _prompt_text(topic: str, format_style: str) -> str:
    return (
        "You are grading whether a student's written response is acceptable for the assignment.\n\n"
        f"Assignment topic: {topic}.\n"
        "Return JSON with key \"is_correct\" (Correct/Incorrect) and key \"feedback\" (1-3 concrete sentences).\n"
        f"Output style profile: {format_style}.\n"
        "Rules:\n"
        "1) Judge by claim-evidence alignment and logic.\n"
        "2) Minor grammar errors are acceptable.\n"
        "3) Keep feedback specific to the student response.\n"
        "4) Do not invent facts not present in the input.\n"
    )


def _good_output_payload(
    rng: np.random.Generator,
    truth_label: str,
    style: str,
    topic: str,
    long_feedback: bool,
) -> dict[str, Any]:
    feedback = (
        f"The response is {('well grounded' if truth_label == 'Correct' else 'missing key support')} "
        f"for {topic}. It references the core idea with {'consistent' if truth_label == 'Correct' else 'insufficient'} evidence."
    )
    if long_feedback:
        feedback += " The final sentence could still be clearer, but the reasoning trail is understandable from beginning to end."

    payload: dict[str, Any] = {"is_correct": truth_label, "feedback": feedback}

    if style == "simple_plus":
        payload["score"] = int(4 if truth_label == "Correct" else 2)
        payload["focus"] = ["claim", "evidence", "organization"]
    elif style == "nested_rubric":
        payload["rubric"] = {
            "claim": "meets" if truth_label == "Correct" else "partial",
            "evidence": "meets" if truth_label == "Correct" else "missing",
            "logic": "meets" if truth_label == "Correct" else "weak",
        }
        payload["confidence"] = round(float(rng.uniform(0.62, 0.94)), 3)
    elif style == "evidence_list":
        payload["evidence"] = [
            "references the assignment context",
            "links examples to conclusion",
            "keeps a coherent claim",
        ]
        payload["improvement_steps"] = [
            "tighten one supporting example",
            "remove repetitive wording",
        ]
    elif style == "deep_nested":
        payload["analysis"] = {
            "rubric_scores": {
                "claim": int(4 if truth_label == "Correct" else 2),
                "evidence": int(4 if truth_label == "Correct" else 2),
                "coherence": int(4 if truth_label == "Correct" else 2),
            },
            "notes": {
                "strength": "maintains line of reasoning" if truth_label == "Correct" else "attempts a claim",
                "risk": "minor ambiguity" if truth_label == "Correct" else "unsupported conclusion",
            },
        }
        payload["next_action"] = "retain structure" if truth_label == "Correct" else "add concrete textual evidence"

    return payload


def _make_bad_output(
    rng: np.random.Generator,
    truth_label: str,
    style: str,
    topic: str,
) -> tuple[str, str, str, bool]:
    bad_type, bad_family = BAD_TYPES[int(rng.integers(0, len(BAD_TYPES)))]

    if bad_type == "wrong_label":
        wrong = "Incorrect" if truth_label == "Correct" else "Correct"
        payload = _good_output_payload(rng, wrong, style, topic, long_feedback=False)
        return json.dumps(payload, ensure_ascii=False), bad_type, bad_family, True

    if bad_type == "empty_feedback":
        payload = _good_output_payload(rng, truth_label, style, topic, long_feedback=False)
        payload["feedback"] = ""
        return json.dumps(payload, ensure_ascii=False), bad_type, bad_family, True

    if bad_type == "missing_feedback_key":
        payload = _good_output_payload(rng, truth_label, style, topic, long_feedback=False)
        payload.pop("feedback", None)
        return json.dumps(payload, ensure_ascii=False), bad_type, bad_family, True

    if bad_type == "malformed_json":
        return '{"is_correct": "Correct", "feedback": "Unfinished output"', bad_type, bad_family, False

    if bad_type == "non_json_text":
        return "Student seems fine overall; probably correct.", bad_type, bad_family, False

    if bad_type == "wrong_enum":
        payload = _good_output_payload(rng, truth_label, style, topic, long_feedback=False)
        payload["is_correct"] = "Maybe"
        return json.dumps(payload, ensure_ascii=False), bad_type, bad_family, True

    # off_topic_long
    payload = _good_output_payload(rng, truth_label, style, topic, long_feedback=True)
    payload["feedback"] = (
        "This response suddenly talks about unrelated sports history and ignores the assignment criteria. "
        "The explanation keeps repeating broad statements without tying them to the student text. "
        "As written, this does not evaluate the actual submission and should be revised."
    )
    return json.dumps(payload, ensure_ascii=False), bad_type, bad_family, True


def _safe_json_dict(s: str) -> bool:
    try:
        parsed = json.loads(s)
        return isinstance(parsed, dict)
    except Exception:
        return False


def _inject_overlap(
    arr: np.ndarray,
    y_bad: np.ndarray,
    rng: np.random.Generator,
    *,
    good_outlier_rate: float,
    bad_easy_rate: float,
    up_factor: float,
    down_factor: float,
) -> np.ndarray:
    out = arr.copy()
    good_idx = np.where(~y_bad)[0]
    bad_idx = np.where(y_bad)[0]

    g_n = int(len(good_idx) * good_outlier_rate)
    b_n = int(len(bad_idx) * bad_easy_rate)

    if g_n > 0:
        g_pick = rng.choice(good_idx, size=g_n, replace=False)
        out[g_pick] = out[g_pick] * (up_factor + rng.uniform(0.0, 0.25, size=g_n))
    if b_n > 0:
        b_pick = rng.choice(bad_idx, size=b_n, replace=False)
        out[b_pick] = out[b_pick] * (down_factor - rng.uniform(0.0, 0.2, size=b_n))

    return np.maximum(out, 1e-9)


def _build_source_and_results(spec: DatasetSpec) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(spec.seed)

    n = spec.n_rows
    n_bad = max(2, int(round(n * spec.incorrect_ratio)))
    bad_idx = set(rng.choice(np.arange(n), size=n_bad, replace=False).tolist())

    prompt = _prompt_text(spec.topic, spec.format_style)

    source_rows: list[dict[str, Any]] = []
    output_json_valid_flags: list[bool] = []

    for i in range(n):
        rid = f"{spec.dataset_id}_{i:04d}"

        row_profile = spec.text_profile
        if spec.text_profile == "mixed":
            row_profile = str(rng.choice(["short", "mixed", "long"], p=[0.3, 0.4, 0.3]))

        student_response = _make_student_response(rng, spec.topic, row_profile)
        truth_label = _truth_label_from_response(student_response)

        input_payload = {
            "assignment": spec.topic,
            "rubric_focus": ["claim", "evidence", "coherence", "language"],
            "student_response": student_response,
            "teacher_note": "Judge core meaning and support quality.",
        }
        input_str = json.dumps(input_payload, ensure_ascii=False)

        is_bad = i in bad_idx
        if not is_bad:
            expected_payload = _good_output_payload(
                rng=rng,
                truth_label=truth_label,
                style=spec.format_style,
                topic=spec.topic,
                long_feedback=(row_profile == "long"),
            )
            expected_output = json.dumps(expected_payload, ensure_ascii=False)
            bad_type = np.nan
            bad_family = "none"
            eval_label = "correct"
        else:
            expected_output, bad_type_str, bad_family_str, _ = _make_bad_output(
                rng=rng,
                truth_label=truth_label,
                style=spec.format_style,
                topic=spec.topic,
            )
            bad_type = bad_type_str
            bad_family = bad_family_str
            eval_label = "incorrect"

        output_json_valid = _safe_json_dict(expected_output)
        output_json_valid_flags.append(output_json_valid)

        edge_type = "clean"
        if row_profile == "short":
            edge_type = str(rng.choice(["clean", "short_text", "newline_tail"], p=[0.6, 0.25, 0.15]))
        elif row_profile == "long":
            edge_type = str(rng.choice(["clean", "long_text", "mixed_style"], p=[0.55, 0.30, 0.15]))

        source_rows.append(
            {
                "id": rid,
                "Prompt": prompt,
                "input": input_str,
                "expectedOutput": expected_output,
                "sample_label": "bad" if eval_label == "incorrect" else "good",
                "truth_label": truth_label,
                "input_edge_type": edge_type,
                "bad_type": bad_type,
                "bad_family": bad_family,
                "output_json_valid": bool(output_json_valid),
                "eval": eval_label,
            }
        )

    source_df = pd.DataFrame(source_rows, columns=SOURCE_COLUMNS)
    y_bad = source_df["eval"].astype(str).str.lower().eq("incorrect").to_numpy(dtype=bool)

    # Hard-gate pass: mostly pass, with fail-heavy datasets intentionally lower.
    hard_gate = np.ones(n, dtype=bool)
    severe_fail = source_df["bad_type"].isin(["malformed_json", "non_json_text", "missing_feedback_key"]).to_numpy(dtype=bool)
    hard_gate[severe_fail] = False

    target_pass = float(spec.hard_gate_target)
    current_pass = float(np.mean(hard_gate))

    if current_pass > target_pass:
        need_drop = int(round((current_pass - target_pass) * n))
        candidates = np.where(hard_gate)[0]
        if need_drop > 0 and len(candidates) > 0:
            pick = rng.choice(candidates, size=min(need_drop, len(candidates)), replace=False)
            hard_gate[pick] = False
    elif current_pass < target_pass:
        need_raise = int(round((target_pass - current_pass) * n))
        candidates = np.where(~hard_gate & ~severe_fail)[0]
        if need_raise > 0 and len(candidates) > 0:
            pick = rng.choice(candidates, size=min(need_raise, len(candidates)), replace=False)
            hard_gate[pick] = True

    # Score synthesis with overlap between good/bad.
    set_shift = float(rng.normal(0.0, 0.18))
    bad_f = y_bad.astype(float)

    output_signal = np.exp(rng.normal(12.0 + set_shift + 2.25 * bad_f, 0.40 + 0.15 * bad_f, size=n))
    output_signal = np.clip(output_signal, 1e3, 2.5e8)

    direction_signal = np.abs(rng.normal(0.35 + 0.04 * set_shift, 0.14, size=n)) + bad_f * np.abs(
        rng.normal(0.82, 0.34, size=n)
    )
    length_signal = np.abs(rng.normal(0.42 + 0.05 * set_shift, 0.18, size=n)) + bad_f * np.abs(
        rng.normal(0.95, 0.44, size=n)
    )

    diff_residual_signal = (
        3.2
        + 5.8 * direction_signal
        + 2.9 * length_signal
        + rng.normal(0.0, 1.5, size=n)
        + bad_f * rng.normal(4.5, 1.9, size=n)
    )
    diff_residual_signal = np.maximum(diff_residual_signal, 0.1)

    delta_ridge_ens_signal = np.abs(rng.normal(0.17 + 0.03 * set_shift, 0.08, size=n)) + bad_f * np.abs(
        rng.normal(0.60, 0.28, size=n)
    )
    similar_input_conflict_signal = np.abs(rng.normal(0.85, 0.45, size=n)) + bad_f * np.abs(
        rng.normal(1.55, 0.70, size=n)
    )

    discourse_instability_signal = np.abs(rng.normal(0.16 + 0.03 * set_shift, 0.11, size=n)) + bad_f * np.abs(
        rng.normal(0.55, 0.25, size=n)
    )
    contradiction_signal = np.abs(rng.normal(0.20 + 0.03 * set_shift, 0.12, size=n)) + bad_f * np.abs(
        rng.normal(0.90, 0.35, size=n)
    )

    output_signal = _inject_overlap(
        output_signal,
        y_bad,
        rng,
        good_outlier_rate=0.06,
        bad_easy_rate=0.15,
        up_factor=5.5,
        down_factor=0.45,
    )
    direction_signal = _inject_overlap(
        direction_signal,
        y_bad,
        rng,
        good_outlier_rate=0.07,
        bad_easy_rate=0.18,
        up_factor=3.6,
        down_factor=0.55,
    )
    length_signal = _inject_overlap(
        length_signal,
        y_bad,
        rng,
        good_outlier_rate=0.07,
        bad_easy_rate=0.18,
        up_factor=3.3,
        down_factor=0.60,
    )
    diff_residual_signal = _inject_overlap(
        diff_residual_signal,
        y_bad,
        rng,
        good_outlier_rate=0.08,
        bad_easy_rate=0.16,
        up_factor=2.3,
        down_factor=0.65,
    )

    # Availability flags (few missing for discourse/contradiction).
    discourse_avail_rate = 0.84 if not spec.fail_heavy else 0.72
    contradiction_avail_rate = 0.82 if not spec.fail_heavy else 0.70
    discourse_available = rng.random(n) < discourse_avail_rate
    contradiction_available = rng.random(n) < contradiction_avail_rate
    discourse_instability_signal = discourse_instability_signal.astype(float)
    contradiction_signal = contradiction_signal.astype(float)
    discourse_instability_signal[~discourse_available] = np.nan
    contradiction_signal[~contradiction_available] = np.nan

    # PCA geometry: input circles, output stars, vectors.
    k_clusters = int(rng.integers(3, 7))
    base_angles = np.linspace(0.0, 2.0 * math.pi, num=k_clusters, endpoint=False)
    base_angles = base_angles + float(rng.uniform(-0.3, 0.3))
    radii = rng.uniform(0.8, 2.0, size=k_clusters)
    centers = np.column_stack([radii * np.cos(base_angles), radii * np.sin(base_angles)])
    weights_raw = rng.uniform(0.2, 1.0, size=k_clusters)
    weights = weights_raw / np.sum(weights_raw)
    cluster_id = rng.choice(np.arange(k_clusters), size=n, p=weights)

    input_xy = centers[cluster_id] + rng.normal(0.0, 0.10, size=(n, 2))

    vec_angle = base_angles[cluster_id] + rng.normal(0.0, 0.35, size=n)
    good_mag = np.clip(np.abs(rng.normal(0.08, 0.03, size=n)), 0.01, 0.25)
    bad_mag = np.clip(np.abs(rng.normal(0.30, 0.11, size=n)), 0.05, 0.70)
    mag = np.where(y_bad, bad_mag, good_mag)

    vec = np.column_stack([mag * np.cos(vec_angle), mag * np.sin(vec_angle)])
    output_xy = input_xy + vec + rng.normal(0.0, 0.03, size=(n, 2))

    row_df = pd.DataFrame(
        {
            "row_id": source_df["id"].astype(str),
            "label_raw": source_df["eval"].astype(str),
            "label_is_correct": source_df["eval"].astype(str).str.lower().eq("correct"),
            "ifeval_pass": hard_gate,
            "schema_pass": hard_gate,
            "textlen_pass": hard_gate,
            "hard_gate_pass": hard_gate,
            "distribution_group_id": "g0000",
            "distribution_group_col": "Prompt",
            "distribution_group_size": int(n),
            "source_input": source_df["input"].astype(str),
            "source_output": source_df["expectedOutput"].astype(str),
            "output_signal_nomask": output_signal,
            "direction_signal_nomask": direction_signal,
            "length_signal_nomask": length_signal,
            "diff_residual_signal_nomask": diff_residual_signal,
            "delta_ridge_ens_signal_nomask": delta_ridge_ens_signal,
            "similar_input_conflict_signal_nomask": similar_input_conflict_signal,
            "discourse_instability_signal_nomask": discourse_instability_signal,
            "contradiction_signal_nomask": contradiction_signal,
            "discourse_instability_available_nomask": discourse_available,
            "contradiction_available_nomask": contradiction_available,
            "input_pca_x_nomask": input_xy[:, 0],
            "input_pca_y_nomask": input_xy[:, 1],
            "output_pca_x_nomask": output_xy[:, 0],
            "output_pca_y_nomask": output_xy[:, 1],
        }
    )

    metadata = {
        "dataset_id": spec.dataset_id,
        "topic": spec.topic,
        "format_style": spec.format_style,
        "text_profile": spec.text_profile,
        "seed": spec.seed,
        "n_rows": int(n),
        "n_incorrect": int(np.sum(y_bad)),
        "incorrect_ratio": float(np.mean(y_bad)),
        "hard_gate_pass_rate": float(np.mean(hard_gate)),
        "target_hard_gate_pass_rate": float(spec.hard_gate_target),
        "json_valid_ratio": float(np.mean(source_df["output_json_valid"].to_numpy(dtype=bool))),
        "fail_heavy": bool(spec.fail_heavy),
    }

    return source_df, row_df, metadata


def _build_specs(args: argparse.Namespace) -> list[DatasetSpec]:
    rng = np.random.default_rng(args.seed)
    fail_heavy_set = set(rng.choice(np.arange(args.set_count), size=max(3, args.set_count // 5), replace=False).tolist())

    specs: list[DatasetSpec] = []
    for i in range(args.set_count):
        n_rows = int(rng.integers(args.min_rows, args.max_rows + 1))
        incorrect_ratio = float(rng.uniform(0.02, 0.10))
        if i in fail_heavy_set:
            hard_gate_target = float(rng.uniform(0.72, 0.88))
        else:
            hard_gate_target = float(rng.uniform(0.90, 0.98))

        specs.append(
            DatasetSpec(
                dataset_id=f"mock_{i:02d}",
                topic=_topic_for(i),
                n_rows=n_rows,
                incorrect_ratio=incorrect_ratio,
                hard_gate_target=hard_gate_target,
                format_style=FORMAT_STYLES[i % len(FORMAT_STYLES)],
                text_profile=TEXT_PROFILES[i % len(TEXT_PROFILES)],
                seed=int(args.seed + i * 97 + 11),
                fail_heavy=bool(i in fail_heavy_set),
            )
        )

    return specs


def _run_final_metric(
    run_script: Path,
    source_csv: Path,
    row_results_csv: Path,
    out_dir: Path,
    dataset_id: str,
    rules: str,
    tail_direction: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cmd = [
        sys.executable,
        str(run_script),
        "--source-csv",
        str(source_csv),
        "--row-results-csv",
        str(row_results_csv),
        "--output-dir",
        str(out_dir),
        "--tag",
        dataset_id,
        "--rules",
        rules,
        "--tail-direction",
        tail_direction,
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "final_metric run failed\n"
            f"dataset={dataset_id}\n"
            f"cmd={' '.join(cmd)}\n"
            f"stdout={proc.stdout}\n"
            f"stderr={proc.stderr}"
        )

    stem = source_csv.stem
    report_dir = out_dir / "report"
    thr_path = report_dir / f"{dataset_id}_{stem}_thresholds_summary.csv"
    summary_path = report_dir / f"{dataset_id}_{stem}_summary.csv"

    if not thr_path.exists() or not summary_path.exists():
        raise FileNotFoundError(f"Missing output files for {dataset_id}")

    return pd.read_csv(thr_path), pd.read_csv(summary_path)


def _summarize_outcome(
    spec: DatasetSpec,
    metadata: dict[str, Any],
    thresholds_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> RunOutcome:
    threshold_finite = bool(np.all(np.isfinite(pd.to_numeric(thresholds_df["threshold_applied"], errors="coerce"))))
    selected_methods = ",".join(sorted(set(thresholds_df["selected_method"].astype(str).tolist())))

    f1_vals = pd.to_numeric(thresholds_df["f1"], errors="coerce")
    recall_vals = pd.to_numeric(thresholds_df["recall"], errors="coerce")
    fpr_vals = pd.to_numeric(thresholds_df["fpr"], errors="coerce")

    output_row = thresholds_df[thresholds_df["rule"] == "output"]
    if output_row.empty:
        output_threshold = float("nan")
        output_f1 = float("nan")
    else:
        output_threshold = float(pd.to_numeric(output_row.iloc[0]["threshold_applied"], errors="coerce"))
        output_f1 = float(pd.to_numeric(output_row.iloc[0]["f1"], errors="coerce"))

    final_pass_rate = float(pd.to_numeric(summary_df.iloc[0].get("final_pass_rate", np.nan), errors="coerce"))

    return RunOutcome(
        dataset_id=spec.dataset_id,
        n_rows=int(metadata["n_rows"]),
        incorrect_ratio=float(metadata["incorrect_ratio"]),
        hard_gate_pass_rate=float(metadata["hard_gate_pass_rate"]),
        selected_methods=selected_methods,
        all_threshold_finite=threshold_finite,
        mean_f1=float(np.nanmean(f1_vals)),
        min_f1=float(np.nanmin(f1_vals)),
        mean_recall=float(np.nanmean(recall_vals)),
        mean_fpr=float(np.nanmean(fpr_vals)),
        output_threshold=output_threshold,
        output_f1=output_f1,
        final_pass_rate=final_pass_rate,
    )


def _write_reports(
    report_dir: Path,
    specs_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    long_rules_df: pd.DataFrame,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)

    specs_df.to_csv(report_dir / "suite_dataset_specs.csv", index=False)
    meta_df.to_csv(report_dir / "suite_generation_metadata.csv", index=False)
    outcomes_df.to_csv(report_dir / "suite_validation_summary.csv", index=False)
    long_rules_df.to_csv(report_dir / "suite_rule_metrics_long.csv", index=False)

    method_counts: dict[str, int] = {}
    for methods in outcomes_df["selected_methods"].astype(str):
        for m in methods.split(","):
            mm = m.strip()
            if not mm:
                continue
            method_counts[mm] = method_counts.get(mm, 0) + 1

    overview = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_count": int(len(outcomes_df)),
        "rows_total": int(outcomes_df["n_rows"].sum()),
        "incorrect_ratio_mean": float(outcomes_df["incorrect_ratio"].mean()),
        "hard_gate_pass_rate_mean": float(outcomes_df["hard_gate_pass_rate"].mean()),
        "final_pass_rate_mean": float(outcomes_df["final_pass_rate"].mean()),
        "mean_f1_mean": float(outcomes_df["mean_f1"].mean()),
        "mean_recall_mean": float(outcomes_df["mean_recall"].mean()),
        "mean_fpr_mean": float(outcomes_df["mean_fpr"].mean()),
        "all_threshold_finite_sets": int(outcomes_df["all_threshold_finite"].sum()),
        "selected_method_counts": method_counts,
    }
    (report_dir / "suite_overview.json").write_text(json.dumps(overview, indent=2), encoding="utf-8")

    md = []
    md.append("# Mock Suite Validation Report")
    md.append("")
    md.append(f"- datasets: {overview['dataset_count']}")
    md.append(f"- total rows: {overview['rows_total']}")
    md.append(f"- mean incorrect ratio: {overview['incorrect_ratio_mean']:.4f}")
    md.append(f"- mean hard-gate pass rate: {overview['hard_gate_pass_rate_mean']:.4f}")
    md.append(f"- mean final pass rate: {overview['final_pass_rate_mean']:.4f}")
    md.append(f"- mean F1 (rule avg): {overview['mean_f1_mean']:.4f}")
    md.append(f"- mean recall (rule avg): {overview['mean_recall_mean']:.4f}")
    md.append(f"- mean FPR (rule avg): {overview['mean_fpr_mean']:.4f}")
    md.append(f"- all-threshold-finite sets: {overview['all_threshold_finite_sets']}/{overview['dataset_count']}")
    md.append("")
    md.append("## Selected Method Counts")
    for k, v in sorted(method_counts.items()):
        md.append(f"- {k}: {v}")
    md.append("")
    md.append("## Per-dataset Summary")
    md.append("")
    md.append(outcomes_df.to_markdown(index=False))
    (report_dir / "suite_report.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    args = parse_args()

    root = Path(__file__).resolve().parents[1]
    run_script = root / "run_final_metric.py"

    if args.output_root:
        out_root = Path(args.output_root)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = root / "results" / f"mock_suite_{ts}"

    datasets_root = out_root / "datasets"
    runs_root = out_root / "runs"
    reports_root = out_root / "reports"
    datasets_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)

    specs = _build_specs(args)

    specs_rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    outcome_rows: list[dict[str, Any]] = []
    long_rule_rows: list[pd.DataFrame] = []

    for idx, spec in enumerate(specs):
        source_df, row_df, metadata = _build_source_and_results(spec)

        ddir = datasets_root / spec.dataset_id
        ddir.mkdir(parents=True, exist_ok=True)
        source_csv = ddir / f"{spec.dataset_id}_source.csv"
        row_csv = ddir / f"{spec.dataset_id}_row_results.csv"
        source_df.to_csv(source_csv, index=False)
        row_df.to_csv(row_csv, index=False)
        (ddir / f"{spec.dataset_id}_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        specs_rows.append(
            {
                "dataset_id": spec.dataset_id,
                "topic": spec.topic,
                "n_rows": spec.n_rows,
                "incorrect_ratio_target": spec.incorrect_ratio,
                "hard_gate_target": spec.hard_gate_target,
                "format_style": spec.format_style,
                "text_profile": spec.text_profile,
                "seed": spec.seed,
                "fail_heavy": spec.fail_heavy,
            }
        )
        meta_rows.append(metadata)

        print(f"[{idx + 1:02d}/{len(specs):02d}] generated {spec.dataset_id} rows={spec.n_rows}")

        if args.skip_run:
            continue

        rdir = runs_root / spec.dataset_id
        rdir.mkdir(parents=True, exist_ok=True)
        thresholds_df, summary_df = _run_final_metric(
            run_script=run_script,
            source_csv=source_csv,
            row_results_csv=row_csv,
            out_dir=rdir,
            dataset_id=spec.dataset_id,
            rules=args.rules,
            tail_direction=args.tail_direction,
        )

        outcome = _summarize_outcome(spec=spec, metadata=metadata, thresholds_df=thresholds_df, summary_df=summary_df)
        outcome_rows.append(outcome.__dict__)

        tf = thresholds_df.copy()
        tf.insert(0, "dataset_id", spec.dataset_id)
        long_rule_rows.append(tf)

        print(
            f"    run ok | hard_gate={metadata['hard_gate_pass_rate']:.3f} "
            f"mean_f1={outcome.mean_f1:.3f} methods={outcome.selected_methods}"
        )

    specs_df = pd.DataFrame(specs_rows)
    meta_df = pd.DataFrame(meta_rows)

    if args.skip_run:
        specs_df.to_csv(reports_root / "suite_dataset_specs.csv", index=False)
        meta_df.to_csv(reports_root / "suite_generation_metadata.csv", index=False)
        print(f"[DONE] generated-only suite at: {out_root}")
        return

    outcomes_df = pd.DataFrame(outcome_rows)
    long_rules_df = pd.concat(long_rule_rows, ignore_index=True) if long_rule_rows else pd.DataFrame()

    _write_reports(
        report_dir=reports_root,
        specs_df=specs_df,
        meta_df=meta_df,
        outcomes_df=outcomes_df,
        long_rules_df=long_rules_df,
    )

    print(f"[DONE] mock suite root: {out_root}")
    print(f"[DONE] reports: {reports_root}")


if __name__ == "__main__":
    main()
