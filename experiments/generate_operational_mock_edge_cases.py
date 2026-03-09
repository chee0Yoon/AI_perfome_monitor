#!/usr/bin/env python3
"""Generate operational-like mock datasets for final_metric edge-case tests.

This script creates 16 datasets:
- sample sizes: 10, 50, 100, 500
- scenario per size:
  1) good_performance
  2) errors_1_to_2
  3) errors_about_1pct
  4) errors_many

For each dataset it writes:
- source CSV (same schema as ambiguous_prompt_benchmark_v3_large.csv)
- synthetic row_results CSV (nomask signal columns)
- metadata JSON

And at suite level:
- dataset_manifest.csv
- README.md
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
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

BAD_TYPES: list[tuple[str, str]] = [
    ("wrong_label", "direction"),
    ("empty_feedback", "length"),
    ("missing_feedback_key", "output"),
    ("malformed_json", "output"),
    ("non_json_text", "output"),
    ("wrong_enum", "contradiction"),
    ("off_topic_long", "diff_residual"),
    ("null_feedback", "schema"),
    ("extra_noise_tail", "similar_input_conflict"),
]

SEVERE_BAD_TYPES = {"missing_feedback_key", "malformed_json", "non_json_text"}

NAMES = [
    "Mina",
    "Joon",
    "Ari",
    "Theo",
    "Sora",
    "Lina",
    "Hana",
    "Evan",
    "Kai",
    "Nora",
]

TOPICS = [
    "education policy memo",
    "public transport incident review",
    "customer quality complaint audit",
    "energy saving campaign summary",
    "school safety case analysis",
    "community health outreach report",
    "supply chain delay retrospective",
    "library digital service evaluation",
]


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    sample_size: int
    scenario: str
    bad_count: int
    seed: int


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1] / "data" / "operational_mock_edge_suite"
    p = argparse.ArgumentParser(description="Generate operational mock edge-case datasets for final_metric.")
    p.add_argument("--output-root", default=str(root))
    p.add_argument("--seed", type=int, default=20260227)
    p.add_argument("--sizes", default="10,50,100,500")
    return p.parse_args()


def _safe_json_dict(text: str) -> bool:
    try:
        parsed = json.loads(str(text))
        return isinstance(parsed, dict)
    except Exception:
        return False


def _prompt_text(topic: str) -> str:
    return (
        "You are grading whether a student's short response matches the required facts.\n\n"
        f"Topic: {topic}.\n"
        "Return JSON with keys:\n"
        '- "is_correct": "Correct" or "Incorrect"\n'
        '- "feedback": 1-3 concise sentences tied to evidence\n'
        "Rules:\n"
        "1) Judge by meaning and evidence alignment.\n"
        "2) Minor wording differences are acceptable.\n"
        "3) Do not add external facts.\n"
    )


def _build_case_bank(rng: np.random.Generator, n_cases: int) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for i in range(n_cases):
        person = str(rng.choice(NAMES))
        partner = str(rng.choice([x for x in NAMES if x != person]))
        topic = str(rng.choice(TOPICS))
        point_1 = f"{topic} reduced average handling time"
        point_2 = f"{person} validated evidence from two independent records"
        point_3 = "the final recommendation prioritized safety over speed"
        wrong_1 = f"{topic} increased delays"
        wrong_2 = f"{partner} skipped verification"
        wrong_3 = "the final recommendation prioritized speed over safety"

        cases.append(
            {
                "case_id": f"case_{i:04d}",
                "topic": topic,
                "question": f"What conclusion is supported in the {topic}?",
                "answer": f"The report supports that {point_1}, {point_2}, and {point_3}.",
                "section_text": (
                    f"Meeting notes: {person} requested verification. "
                    f"Audit logs show {point_1}. "
                    f"Risk review confirms {point_3}."
                ),
                "correct_variants": [
                    f"{point_1}. {point_2}. {point_3}.",
                    f"The supported conclusion is that {point_1}, and {point_2}, while {point_3}.",
                    f"{point_2}; therefore {point_1}, and {point_3}.",
                ],
                "incorrect_variants": [
                    f"The report says {wrong_1} and {wrong_2}.",
                    f"It claims {wrong_3}.",
                    "No clear answer can be confirmed from the text.",
                    "This is about food and music, not the report.",
                ],
            }
        )
    return cases


def _student_answer(rng: np.random.Generator, case: dict[str, Any], truth_label: str, profile: str) -> str:
    if truth_label == "Correct":
        base = str(rng.choice(case["correct_variants"]))
    else:
        base = str(rng.choice(case["incorrect_variants"]))

    if profile == "short":
        return base
    if profile == "long":
        tail = (
            " The answer also explains why evidence consistency matters for deployment quality checks. "
            "It references the conclusion and keeps the argument bounded to the provided context."
        )
        return base + tail
    return base + (" I think this matches the key evidence." if truth_label == "Correct" else " This seems uncertain.")


def _good_output_payload(
    rng: np.random.Generator,
    truth_label: str,
    topic: str,
    *,
    long_feedback: bool,
) -> dict[str, Any]:
    feedback = (
        f"The response is {'aligned' if truth_label == 'Correct' else 'not aligned'} with the required points for {topic}. "
        f"It {'keeps' if truth_label == 'Correct' else 'does not keep'} evidence consistency."
    )
    if long_feedback:
        feedback += " The explanation is actionable and linked to the student text."

    payload: dict[str, Any] = {"is_correct": truth_label, "feedback": feedback}
    if rng.random() < 0.45:
        payload["confidence"] = round(float(rng.uniform(0.62, 0.95)), 3)
    if rng.random() < 0.30:
        payload["rubric"] = {
            "claim": "meets" if truth_label == "Correct" else "partial",
            "evidence": "meets" if truth_label == "Correct" else "missing",
            "logic": "meets" if truth_label == "Correct" else "weak",
        }
    return payload


def _make_bad_output(
    rng: np.random.Generator,
    *,
    truth_label: str,
    topic: str,
    bad_type: str,
) -> str:
    if bad_type == "wrong_label":
        wrong = "Incorrect" if truth_label == "Correct" else "Correct"
        return json.dumps(_good_output_payload(rng, wrong, topic, long_feedback=False), ensure_ascii=False)

    if bad_type == "empty_feedback":
        payload = _good_output_payload(rng, truth_label, topic, long_feedback=False)
        payload["feedback"] = ""
        return json.dumps(payload, ensure_ascii=False)

    if bad_type == "missing_feedback_key":
        payload = _good_output_payload(rng, truth_label, topic, long_feedback=False)
        payload.pop("feedback", None)
        return json.dumps(payload, ensure_ascii=False)

    if bad_type == "malformed_json":
        return '{"is_correct": "Correct", "feedback": "truncated output"'

    if bad_type == "non_json_text":
        return "Looks mostly acceptable, maybe correct, maybe not."

    if bad_type == "wrong_enum":
        payload = _good_output_payload(rng, truth_label, topic, long_feedback=False)
        payload["is_correct"] = "PartiallyCorrect"
        return json.dumps(payload, ensure_ascii=False)

    if bad_type == "off_topic_long":
        payload = _good_output_payload(rng, truth_label, topic, long_feedback=True)
        payload["feedback"] = (
            "This feedback drifts into unrelated sports history and travel stories. "
            "It repeats generic claims without checking the assignment evidence. "
            "As written, this does not evaluate the student's answer."
        )
        return json.dumps(payload, ensure_ascii=False)

    if bad_type == "null_feedback":
        payload = _good_output_payload(rng, truth_label, topic, long_feedback=False)
        payload["feedback"] = None
        return json.dumps(payload, ensure_ascii=False)

    # extra_noise_tail
    payload = _good_output_payload(rng, truth_label, topic, long_feedback=False)
    payload["feedback"] = (
        str(payload["feedback"])
        + " Additional unrelated commentary about weather, movies, and lunch plans is appended repeatedly."
    )
    return json.dumps(payload, ensure_ascii=False)


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

    g_n = int(round(len(good_idx) * good_outlier_rate))
    b_n = int(round(len(bad_idx) * bad_easy_rate))

    if g_n > 0 and len(good_idx) > 0:
        g_pick = rng.choice(good_idx, size=min(g_n, len(good_idx)), replace=False)
        out[g_pick] = out[g_pick] * (up_factor + rng.uniform(0.0, 0.25, size=len(g_pick)))
    if b_n > 0 and len(bad_idx) > 0:
        b_pick = rng.choice(bad_idx, size=min(b_n, len(bad_idx)), replace=False)
        out[b_pick] = out[b_pick] * np.maximum(0.1, down_factor - rng.uniform(0.0, 0.2, size=len(b_pick)))

    return np.maximum(out, 1e-9)


def _target_bad_count(sample_size: int, scenario: str) -> int:
    if scenario == "good_performance":
        return 0
    if scenario == "errors_1_to_2":
        return 1 if sample_size <= 10 else 2
    if scenario == "errors_about_1pct":
        return max(1, int(round(sample_size * 0.01)))
    if scenario == "errors_many":
        if sample_size <= 10:
            return 4
        return int(round(sample_size * 0.30))
    raise ValueError(f"Unknown scenario: {scenario}")


def _build_specs(seed: int, sizes: list[int]) -> list[DatasetSpec]:
    scenarios = ["good_performance", "errors_1_to_2", "errors_about_1pct", "errors_many"]
    specs: list[DatasetSpec] = []
    idx = 0
    for n in sizes:
        for sc in scenarios:
            bad_count = _target_bad_count(n, sc)
            specs.append(
                DatasetSpec(
                    dataset_id=f"n{n:03d}_{sc}",
                    sample_size=n,
                    scenario=sc,
                    bad_count=min(max(bad_count, 0), n),
                    seed=int(seed + idx * 131 + n * 17),
                )
            )
            idx += 1
    return specs


def _edge_type(profile: str, bad_type: str | None, rng: np.random.Generator) -> str:
    if bad_type == "off_topic_long":
        return "long_text"
    if bad_type == "malformed_json":
        return "json_truncated"
    if bad_type == "non_json_text":
        return "non_json_output"
    if bad_type == "extra_noise_tail":
        return "noisy_tail"
    if profile == "short":
        return str(rng.choice(["clean", "short_text", "newline_tail"], p=[0.55, 0.30, 0.15]))
    if profile == "long":
        return str(rng.choice(["clean", "long_text", "mixed_style"], p=[0.55, 0.30, 0.15]))
    return str(rng.choice(["clean", "mixed_style", "quoted_escape"], p=[0.65, 0.20, 0.15]))


def _build_source_df(spec: DatasetSpec) -> pd.DataFrame:
    rng = np.random.default_rng(spec.seed)
    case_bank = _build_case_bank(rng, n_cases=max(32, min(256, spec.sample_size * 4)))
    prompt = _prompt_text(str(rng.choice(TOPICS)))

    bad_indices: set[int] = set()
    if spec.bad_count > 0:
        bad_indices = set(rng.choice(np.arange(spec.sample_size), size=spec.bad_count, replace=False).tolist())

    bad_cycle = [BAD_TYPES[i % len(BAD_TYPES)] for i in range(max(1, spec.bad_count))]
    rng.shuffle(bad_cycle)
    bad_type_by_idx: dict[int, tuple[str, str]] = {}
    for rank, idx in enumerate(sorted(bad_indices)):
        bad_type_by_idx[idx] = bad_cycle[rank % len(bad_cycle)]

    rows: list[dict[str, Any]] = []

    for i in range(spec.sample_size):
        case = case_bank[int(rng.integers(0, len(case_bank)))]
        profile = str(rng.choice(["short", "mixed", "long"], p=[0.30, 0.45, 0.25]))
        truth_label = "Correct" if rng.random() < 0.62 else "Incorrect"
        student_answer = _student_answer(rng, case, truth_label, profile)
        is_bad = i in bad_indices

        input_payload = {
            "question": case["question"],
            "answer": case["answer"],
            "section_text": case["section_text"],
            "student_answer": student_answer,
        }
        input_str = json.dumps(input_payload, ensure_ascii=False)

        bad_type: str | None = None
        bad_family: str = "none"
        if not is_bad:
            expected_output = json.dumps(
                _good_output_payload(rng, truth_label, case["topic"], long_feedback=(profile == "long")),
                ensure_ascii=False,
            )
            eval_label = "correct"
            sample_label = "good"
        else:
            bad_type, bad_family = bad_type_by_idx[i]
            expected_output = _make_bad_output(
                rng,
                truth_label=truth_label,
                topic=case["topic"],
                bad_type=bad_type,
            )
            eval_label = "incorrect"
            sample_label = "bad"

        rows.append(
            {
                "id": f"{spec.dataset_id}_{i:04d}",
                "Prompt": prompt,
                "input": input_str,
                "expectedOutput": expected_output,
                "sample_label": sample_label,
                "truth_label": truth_label,
                "input_edge_type": _edge_type(profile, bad_type, rng),
                "bad_type": bad_type if bad_type is not None else np.nan,
                "bad_family": bad_family,
                "output_json_valid": bool(_safe_json_dict(expected_output)),
                "eval": eval_label,
            }
        )

    return pd.DataFrame(rows, columns=SOURCE_COLUMNS)


def _build_geometry(
    rng: np.random.Generator,
    n: int,
    y_bad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    k_clusters = int(np.clip(max(2, n // 90 + 2), 2, 6))
    base_angles = np.linspace(0.0, 2.0 * math.pi, num=k_clusters, endpoint=False)
    base_angles = base_angles + float(rng.uniform(-0.25, 0.25))
    radii = rng.uniform(0.9, 2.0, size=k_clusters)
    centers = np.column_stack([radii * np.cos(base_angles), radii * np.sin(base_angles)])
    raw_w = rng.uniform(0.2, 1.0, size=k_clusters)
    weights = raw_w / np.sum(raw_w)
    cluster_id = rng.choice(np.arange(k_clusters), size=n, p=weights)

    input_xy = centers[cluster_id] + rng.normal(0.0, 0.11, size=(n, 2))
    vec_angle = base_angles[cluster_id] + rng.normal(0.0, 0.30, size=n)
    good_mag = np.clip(np.abs(rng.normal(0.09, 0.03, size=n)), 0.01, 0.26)
    bad_mag = np.clip(np.abs(rng.normal(0.30, 0.10, size=n)), 0.05, 0.72)
    mag = np.where(y_bad, bad_mag, good_mag)
    vec = np.column_stack([mag * np.cos(vec_angle), mag * np.sin(vec_angle)])
    output_xy = input_xy + vec + rng.normal(0.0, 0.03, size=(n, 2))
    return input_xy, output_xy


def _build_row_results_df(spec: DatasetSpec, source_df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(spec.seed + 1009)
    n = len(source_df)
    y_bad = source_df["eval"].astype(str).str.lower().eq("incorrect").to_numpy(dtype=bool)
    bad_type_arr = source_df["bad_type"].fillna("").astype(str).to_numpy()
    bad_f = y_bad.astype(float)

    hard_gate = np.ones(n, dtype=bool)
    severe = np.isin(bad_type_arr, list(SEVERE_BAD_TYPES))
    hard_gate[severe] = False
    if spec.scenario == "errors_many":
        additional_drop = int(round(0.05 * n))
        candidates = np.where(hard_gate)[0]
        if additional_drop > 0 and len(candidates) > 0:
            pick = rng.choice(candidates, size=min(additional_drop, len(candidates)), replace=False)
            hard_gate[pick] = False

    set_shift = float(rng.normal(0.0, 0.16))
    output_signal = np.exp(rng.normal(11.8 + set_shift + 1.9 * bad_f, 0.45 + 0.12 * bad_f, size=n))
    output_signal = np.clip(output_signal, 8e2, 2.8e8)

    direction_signal = np.abs(rng.normal(0.32 + 0.03 * set_shift, 0.13, size=n)) + bad_f * np.abs(
        rng.normal(0.85, 0.28, size=n)
    )
    length_signal = np.abs(rng.normal(0.37 + 0.04 * set_shift, 0.15, size=n)) + bad_f * np.abs(
        rng.normal(0.92, 0.32, size=n)
    )
    diff_residual_signal = (
        2.8
        + 5.2 * direction_signal
        + 2.7 * length_signal
        + rng.normal(0.0, 1.2, size=n)
        + bad_f * rng.normal(4.0, 1.6, size=n)
    )
    diff_residual_signal = np.maximum(diff_residual_signal, 0.1)

    delta_ridge_ens_signal = np.abs(rng.normal(0.16 + 0.03 * set_shift, 0.07, size=n)) + bad_f * np.abs(
        rng.normal(0.58, 0.22, size=n)
    )
    similar_input_conflict_signal = np.abs(rng.normal(0.78, 0.34, size=n)) + bad_f * np.abs(
        rng.normal(1.40, 0.50, size=n)
    )
    discourse_instability_signal = np.abs(rng.normal(0.14 + 0.03 * set_shift, 0.10, size=n)) + bad_f * np.abs(
        rng.normal(0.50, 0.19, size=n)
    )
    contradiction_signal = np.abs(rng.normal(0.19 + 0.03 * set_shift, 0.10, size=n)) + bad_f * np.abs(
        rng.normal(0.82, 0.25, size=n)
    )
    self_contradiction_signal = np.abs(rng.normal(0.13 + 0.03 * set_shift, 0.08, size=n)) + bad_f * np.abs(
        rng.normal(0.63, 0.20, size=n)
    )

    wrong_label_mask = bad_type_arr == "wrong_label"
    empty_feedback_mask = bad_type_arr == "empty_feedback"
    missing_feedback_mask = bad_type_arr == "missing_feedback_key"
    malformed_mask = bad_type_arr == "malformed_json"
    non_json_mask = bad_type_arr == "non_json_text"
    wrong_enum_mask = bad_type_arr == "wrong_enum"
    off_topic_mask = bad_type_arr == "off_topic_long"
    null_feedback_mask = bad_type_arr == "null_feedback"
    extra_noise_mask = bad_type_arr == "extra_noise_tail"

    output_signal[missing_feedback_mask | malformed_mask | non_json_mask] *= 5.5
    direction_signal[wrong_label_mask] *= 2.7
    contradiction_signal[wrong_label_mask | wrong_enum_mask] *= 2.5
    length_signal[empty_feedback_mask | null_feedback_mask] *= 2.4
    diff_residual_signal[off_topic_mask | extra_noise_mask] *= 1.9
    similar_input_conflict_signal[extra_noise_mask] *= 2.8
    discourse_instability_signal[off_topic_mask] *= 2.0

    output_signal = _inject_overlap(
        output_signal,
        y_bad,
        rng,
        good_outlier_rate=0.05,
        bad_easy_rate=0.14,
        up_factor=4.6,
        down_factor=0.50,
    )
    direction_signal = _inject_overlap(
        direction_signal,
        y_bad,
        rng,
        good_outlier_rate=0.06,
        bad_easy_rate=0.16,
        up_factor=3.3,
        down_factor=0.58,
    )
    length_signal = _inject_overlap(
        length_signal,
        y_bad,
        rng,
        good_outlier_rate=0.06,
        bad_easy_rate=0.17,
        up_factor=3.1,
        down_factor=0.60,
    )
    diff_residual_signal = _inject_overlap(
        diff_residual_signal,
        y_bad,
        rng,
        good_outlier_rate=0.07,
        bad_easy_rate=0.14,
        up_factor=2.2,
        down_factor=0.68,
    )

    discourse_avail_rate = 0.84 if n >= 100 else 0.70
    contradiction_avail_rate = 0.82 if n >= 100 else 0.68
    self_contradiction_avail_rate = 0.80 if n >= 100 else 0.66

    discourse_available = rng.random(n) < discourse_avail_rate
    contradiction_available = rng.random(n) < contradiction_avail_rate
    self_contradiction_available = rng.random(n) < self_contradiction_avail_rate

    discourse_instability_signal = discourse_instability_signal.astype(float)
    contradiction_signal = contradiction_signal.astype(float)
    self_contradiction_signal = self_contradiction_signal.astype(float)
    discourse_instability_signal[~discourse_available] = np.nan
    contradiction_signal[~contradiction_available] = np.nan
    self_contradiction_signal[~self_contradiction_available] = np.nan

    input_xy, output_xy = _build_geometry(rng, n, y_bad)

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
            "self_contradiction_signal_nomask": self_contradiction_signal,
            "discourse_instability_available_nomask": discourse_available,
            "contradiction_available_nomask": contradiction_available,
            "self_contradiction_available_nomask": self_contradiction_available,
            "input_pca_x_nomask": input_xy[:, 0],
            "input_pca_y_nomask": input_xy[:, 1],
            "output_pca_x_nomask": output_xy[:, 0],
            "output_pca_y_nomask": output_xy[:, 1],
        }
    )
    return row_df


def _scenario_description(scenario: str) -> str:
    descriptions = {
        "good_performance": "No injected error rows.",
        "errors_1_to_2": "Exactly 1-2 injected error rows.",
        "errors_about_1pct": "About 1% injected error rows (rounded to integer, minimum 1).",
        "errors_many": "Many injected error rows (about 30%, 40% when n=10).",
    }
    return descriptions.get(scenario, scenario)


def _write_readme(output_root: Path, manifest_df: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append("# operational_mock_edge_suite")
    lines.append("")
    lines.append("Generated datasets for final_metric edge-case tests.")
    lines.append("")
    lines.append("## Dataset Matrix")
    lines.append("")
    lines.append(manifest_df.to_markdown(index=False))
    lines.append("")
    lines.append("## File Layout")
    lines.append("")
    lines.append("- datasets/<dataset_id>/<dataset_id>_source.csv")
    lines.append("- datasets/<dataset_id>/<dataset_id>_row_results.csv")
    lines.append("- datasets/<dataset_id>/<dataset_id>_meta.json")
    lines.append("- dataset_manifest.csv")
    lines.append("")
    lines.append("## Run Example")
    lines.append("")
    lines.append("```bash")
    lines.append(
        "python final_metric/run_final_metric.py "
        "--source-csv <source.csv> "
        "--row-results-csv <row_results.csv> "
        "--output-dir <output_dir> "
        "--tag edge_case_test"
    )
    lines.append("```")
    (output_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    datasets_root = output_root / "datasets"
    datasets_root.mkdir(parents=True, exist_ok=True)

    sizes = [int(x.strip()) for x in str(args.sizes).split(",") if x.strip()]
    specs = _build_specs(seed=int(args.seed), sizes=sizes)

    manifest_rows: list[dict[str, Any]] = []

    for spec in specs:
        source_df = _build_source_df(spec)
        row_df = _build_row_results_df(spec, source_df)

        dataset_dir = datasets_root / spec.dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)

        source_csv = dataset_dir / f"{spec.dataset_id}_source.csv"
        row_csv = dataset_dir / f"{spec.dataset_id}_row_results.csv"
        meta_json = dataset_dir / f"{spec.dataset_id}_meta.json"

        source_df.to_csv(source_csv, index=False)
        row_df.to_csv(row_csv, index=False)

        actual_bad = int(source_df["eval"].astype(str).str.lower().eq("incorrect").sum())
        json_invalid = int((~source_df["output_json_valid"].astype(bool)).sum())
        bad_type_counts = (
            source_df["bad_type"]
            .dropna()
            .astype(str)
            .value_counts()
            .to_dict()
        )
        hard_gate_pass_rate = float(np.mean(row_df["hard_gate_pass"].to_numpy(dtype=bool)))

        meta = {
            "dataset_id": spec.dataset_id,
            "sample_size": int(spec.sample_size),
            "scenario": spec.scenario,
            "scenario_description": _scenario_description(spec.scenario),
            "seed": int(spec.seed),
            "bad_count_target": int(spec.bad_count),
            "bad_count_actual": int(actual_bad),
            "bad_ratio_actual": float(actual_bad / spec.sample_size) if spec.sample_size else 0.0,
            "json_invalid_count": int(json_invalid),
            "hard_gate_pass_rate": float(hard_gate_pass_rate),
            "bad_type_counts": bad_type_counts,
            "source_csv": str(source_csv),
            "row_results_csv": str(row_csv),
        }
        meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        manifest_rows.append(
            {
                "dataset_id": spec.dataset_id,
                "sample_size": spec.sample_size,
                "scenario": spec.scenario,
                "scenario_desc": _scenario_description(spec.scenario),
                "bad_count_target": spec.bad_count,
                "bad_count_actual": actual_bad,
                "bad_ratio_actual": round(float(actual_bad / spec.sample_size), 6),
                "json_invalid_count": json_invalid,
                "hard_gate_pass_rate": round(hard_gate_pass_rate, 6),
                "bad_types_used": ",".join(sorted(bad_type_counts.keys())),
            }
        )

        print(
            f"[DONE] {spec.dataset_id} "
            f"n={spec.sample_size} "
            f"bad={actual_bad} "
            f"hard_gate_pass={hard_gate_pass_rate:.3f}"
        )

    manifest_df = pd.DataFrame(manifest_rows).sort_values(["sample_size", "scenario"]).reset_index(drop=True)
    manifest_path = output_root / "dataset_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    _write_readme(output_root, manifest_df)

    print(f"[DONE] suite_root={output_root}")
    print(f"[DONE] manifest={manifest_path}")


if __name__ == "__main__":
    main()
