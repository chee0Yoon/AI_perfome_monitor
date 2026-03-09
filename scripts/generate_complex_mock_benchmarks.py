#!/usr/bin/env python3
"""
Generate two complex mock benchmarks with the same column schema as ambiguous_prompt_benchmark_v3_large.csv.

Outputs:
1) Large and diverse set (more rows, richer nested input/output structures, wider edge-case coverage)
2) Smaller low-error set (fewer rows, small number of bad samples)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd


PROMPTS: list[str] = [
    """
You are an evaluator for complex student responses.

Return JSON with these keys:
- "is_correct": "Correct" or "Incorrect"
- "feedback": concise explanation grounded in evidence
- "reasoning_steps": list of short reasoning points
- "cited_evidence": list of objects with "doc_id" and "quote"
- "risk_flags": object with boolean checks
- "confidence": number in [0,1]
- "verdict_trace": object with score details

Keep judgments strictly tied to provided context and rubric.
""".strip(),
    """
You are grading whether a student complied with policy constraints and captured key facts.

Return JSON keys:
- is_correct
- feedback
- reasoning_steps
- cited_evidence
- risk_flags
- confidence
- verdict_trace

Do not add external facts.
""".strip(),
    """
You are a rubric checker for multi-document answers.

Output strict JSON with:
is_correct, feedback, reasoning_steps, cited_evidence, risk_flags, confidence, verdict_trace
""".strip(),
]

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
    "Rin",
    "Dino",
]

DOMAINS = [
    "education policy",
    "public transport planning",
    "product quality review",
    "incident triage",
    "community health outreach",
    "energy savings campaign",
    "supply chain update",
    "library operation report",
]

GOOD_FEEDBACK_CORRECT = [
    "The response captures the required facts and follows the rubric constraints.",
    "Core meaning and cited evidence align with the reference, so this is correct.",
    "The answer includes the key points without violating the must-not constraints.",
]

GOOD_FEEDBACK_INCORRECT = [
    "The response misses key facts or violates the rubric constraints, so it is incorrect.",
    "Evidence coverage is insufficient and key meaning does not align with the reference.",
    "Critical required points are missing, so this should be marked incorrect.",
]

BAD_RAMBLING = [
    "This might be right depending on interpretation, but also maybe not, and there are many perspectives.",
    "There is some overlap yet uncertainty remains high because wording can be interpreted in multiple ways.",
    "I can argue both sides and am not fully sure, but I still made a decision.",
]

BAD_CONTRADICT = [
    "The response is fully aligned with all required points, but I still mark it Incorrect.",
    "The response misses key facts and violates constraints, but I still mark it Correct.",
    "Evidence supports the opposite conclusion, yet the final label remains unchanged.",
]

NOISY_TAILS = [
    "By the way this reminds me of a different project.",
    "Also the weather was nice and everyone felt happy.",
    "Additional unrelated commentary is attached for context.",
]

OFFTOPIC = [
    "I like pizza and movies.",
    "Blue is my favorite color.",
    "No clear answer, maybe anything is fine.",
    "This is unrelated to the question.",
]

EDGE_JSON_FRAGMENTS = [
    '{"note":"partial","status":"draft"',
    '{"hint":"review pending", "flags":["low_conf"]',
    '{"analysis":"incomplete',
]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2] / "data"
    parser = argparse.ArgumentParser(description="Generate complex mock benchmarks (large + low-error).")
    parser.add_argument(
        "--large-output-csv",
        default=str(root / "complex_mock_benchmark_large.csv"),
        help="Output path for large complex set.",
    )
    parser.add_argument(
        "--small-output-csv",
        default=str(root / "complex_mock_benchmark_low_error.csv"),
        help="Output path for smaller low-error set.",
    )
    parser.add_argument("--n-large-good", type=int, default=5000, help="Good row count for large set.")
    parser.add_argument("--n-large-bad", type=int, default=2000, help="Bad row count for large set.")
    parser.add_argument("--n-small-good", type=int, default=1170, help="Good row count for low-error set.")
    parser.add_argument("--n-small-bad", type=int, default=30, help="Bad row count for low-error set.")
    parser.add_argument("--n-cases", type=int, default=420, help="Unique synthetic case count.")
    parser.add_argument("--truth-correct-ratio", type=float, default=0.58, help="Probability of truth=Correct.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def json_validity(text: str) -> bool:
    try:
        json.loads(str(text))
        return True
    except Exception:
        return False


def to_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def build_case_bank(n_cases: int, rng: random.Random) -> list[dict[str, Any]]:
    case_bank: list[dict[str, Any]] = []
    for i in range(max(1, int(n_cases))):
        person = rng.choice(NAMES)
        partner = rng.choice([x for x in NAMES if x != person])
        domain = rng.choice(DOMAINS)
        k1 = f"{domain} reduced processing delays"
        k2 = f"{person} validated two independent sources"
        k3 = f"final recommendation prioritized safety over speed"
        wrong1 = f"{domain} increased delays"
        wrong2 = f"{partner} ignored evidence checks"
        wrong3 = "recommendation prioritized speed over safety"
        q = f"In the {domain} report, what conclusion was supported by evidence?"
        a = f"The report concluded that {k1}, {k2}, and {k3}."

        doc_a = (
            f"Meeting notes: {person} requested verification. "
            f"Audit findings showed that {domain} reduced processing delays."
        )
        doc_b = (
            f"Evidence log: {person} validated two independent sources before approval."
        )
        doc_c = (
            "Risk review: the final recommendation prioritized safety over speed due to risk exposure."
        )
        distractor = (
            f"Older draft by {partner} claimed that {wrong1}, but this draft was superseded."
        )

        case_bank.append(
            {
                "case_id": f"cx_case_{i:05d}",
                "prompt": rng.choice(PROMPTS),
                "task_type": rng.choice(["evidence_alignment", "policy_check", "multi_doc_review"]),
                "question": q,
                "answer": a,
                "key_points": [k1, k2, k3],
                "forbidden_points": [wrong1, wrong2, wrong3],
                "correct_variants": [
                    f"{k1}; {k2}; {k3}.",
                    f"It said {k1} and that {k2}, while {k3}.",
                    f"The supported conclusion was: {k1}, {k2}, and {k3}.",
                ],
                "incorrect_variants": [
                    f"The report said {wrong1} and {wrong2}.",
                    f"{wrong3}.",
                    rng.choice(OFFTOPIC),
                ],
                "documents": [
                    {"doc_id": "doc_a", "title": "Meeting Notes", "text": doc_a, "reliability": 0.91},
                    {"doc_id": "doc_b", "title": "Evidence Log", "text": doc_b, "reliability": 0.94},
                    {"doc_id": "doc_c", "title": "Risk Review", "text": doc_c, "reliability": 0.90},
                    {"doc_id": "doc_d", "title": "Superseded Draft", "text": distractor, "reliability": 0.41},
                ],
            }
        )
    return case_bank


def apply_input_edge(base_answer: str, edge_type: str, rng: random.Random) -> str:
    if edge_type == "clean":
        return base_answer
    if edge_type == "lower":
        return base_answer.lower()
    if edge_type == "extra_space":
        return "  " + "   ".join(base_answer.split()) + "  "
    if edge_type == "quoted":
        return f"\"{base_answer}\""
    if edge_type == "chain_of_thought":
        return f"I checked all documents. Step 1 done. Step 2 done. Final: {base_answer}"
    if edge_type == "newline_tail":
        return f"{base_answer}\nI might be missing one detail."
    if edge_type == "context_tail":
        return f"{base_answer} {rng.choice(NOISY_TAILS)}"
    if edge_type == "offtopic":
        return rng.choice(OFFTOPIC)
    if edge_type == "empty":
        return ""
    if edge_type == "noisy_long":
        return f"{base_answer} {' '.join(rng.choice(NOISY_TAILS) for _ in range(4))}"
    if edge_type == "markdown_block":
        return (
            "```text\n"
            + base_answer
            + "\n```\n"
            + "Checklist: evidence verified, constraints checked."
        )
    if edge_type == "bullet_dump":
        return (
            "- claim: "
            + base_answer
            + "\n- note: "
            + rng.choice(NOISY_TAILS)
            + "\n- extra: unstructured details"
        )
    if edge_type == "json_fragment":
        return rng.choice(EDGE_JSON_FRAGMENTS) + " " + base_answer
    if edge_type == "field_leak":
        return (
            base_answer
            + " Evidence refs: doc_a|doc_b|doc_x, confidence_tag=approx, parser_hint=legacy."
        )
    if edge_type == "long_repetition":
        repeated = " ".join([base_answer] * 3)
        return repeated + " " + " ".join(rng.choice(NOISY_TAILS) for _ in range(6))
    return base_answer


def choose_input_edge(sample_label: str, rng: random.Random) -> str:
    if sample_label == "good":
        return rng.choice(
            [
                "clean",
                "lower",
                "extra_space",
                "quoted",
                "chain_of_thought",
                "context_tail",
                "markdown_block",
                "bullet_dump",
                "field_leak",
            ]
        )
    return rng.choice(
        [
            "clean",
            "offtopic",
            "empty",
            "noisy_long",
            "newline_tail",
            "extra_space",
            "quoted",
            "context_tail",
            "markdown_block",
            "bullet_dump",
            "json_fragment",
            "field_leak",
            "long_repetition",
        ]
    )


def build_good_output(case: dict[str, Any], truth_is_correct: bool, rng: random.Random) -> dict[str, Any]:
    label = "Correct" if truth_is_correct else "Incorrect"
    feedback = rng.choice(GOOD_FEEDBACK_CORRECT if truth_is_correct else GOOD_FEEDBACK_INCORRECT)
    docs = case["documents"]
    cited = []
    for d in docs[:3]:
        quote = d["text"].split(".")[0].strip()[:120]
        cited.append({"doc_id": d["doc_id"], "quote": quote})
    semantic_score = rng.uniform(0.72, 0.97) if truth_is_correct else rng.uniform(0.22, 0.56)
    coverage_score = rng.uniform(0.68, 0.95) if truth_is_correct else rng.uniform(0.20, 0.52)
    consistency_score = rng.uniform(0.70, 0.96) if truth_is_correct else rng.uniform(0.18, 0.49)
    confidence = float(
        min(
            0.99,
            max(
                0.01,
                0.35 * semantic_score + 0.35 * coverage_score + 0.30 * consistency_score,
            ),
        )
    )
    return {
        "is_correct": label,
        "feedback": feedback,
        "reasoning_steps": [
            "Extracted required key points from high-reliability documents.",
            "Compared student claim against required and forbidden points.",
            "Checked internal consistency between label and textual rationale.",
        ],
        "cited_evidence": cited,
        "risk_flags": {
            "hallucination_risk": not truth_is_correct and rng.random() < 0.35,
            "missing_keypoint": not truth_is_correct,
            "forbidden_claim_present": not truth_is_correct and rng.random() < 0.50,
        },
        "confidence": round(confidence, 3),
        "verdict_trace": {
            "label_source": "rubric_alignment",
            "score_breakdown": {
                "semantic": round(semantic_score, 3),
                "coverage": round(coverage_score, 3),
                "consistency": round(consistency_score, 3),
            },
            "decision_margin": round(abs(semantic_score - 0.5), 3),
        },
    }


def build_bad_output(
    case: dict[str, Any],
    good_payload: dict[str, Any],
    truth_is_correct: bool,
    rng: random.Random,
) -> tuple[str, str, str]:
    bad_type = rng.choice(
        [
            "flip_label",
            "contradict_feedback",
            "rambling_feedback",
            "empty_feedback",
            "ultra_short_feedback",
            "wrong_keys",
            "missing_label_key",
            "wrong_type_confidence",
            "wrong_type_cited_evidence",
            "wrong_type_reasoning_steps",
            "wrong_type_risk_flags",
            "confidence_out_of_range",
            "confidence_nan_string",
            "wrong_enum_label",
            "null_label",
            "empty_cited_evidence",
            "partial_cited_evidence",
            "verdict_trace_flattened",
            "missing_verdict_trace",
            "extra_unknown_top_key",
            "fenced_json",
            "yaml_like",
            "json_with_trailing_text",
            "json_array_output",
            "single_quotes_json",
            "truncated_json",
            "malformed_json",
        ]
    )

    label = str(good_payload["is_correct"])
    flipped = "Incorrect" if label == "Correct" else "Correct"

    if bad_type == "flip_label":
        payload = dict(good_payload)
        payload["is_correct"] = flipped
        payload["feedback"] = (
            "I flipped the final verdict despite overlap in evidence."
            if truth_is_correct
            else "I flipped the verdict even though key points were missing."
        )
        return to_json(payload), bad_type, "semantic"

    if bad_type == "contradict_feedback":
        payload = dict(good_payload)
        payload["feedback"] = rng.choice(BAD_CONTRADICT)
        return to_json(payload), bad_type, "semantic"

    if bad_type == "rambling_feedback":
        payload = dict(good_payload)
        payload["feedback"] = rng.choice(BAD_RAMBLING)
        return to_json(payload), bad_type, "semantic"

    if bad_type == "empty_feedback":
        payload = dict(good_payload)
        payload["feedback"] = ""
        return to_json(payload), bad_type, "length"

    if bad_type == "ultra_short_feedback":
        payload = dict(good_payload)
        payload["feedback"] = rng.choice(["ok", "bad", "?"])
        return to_json(payload), bad_type, "length"

    if bad_type == "wrong_keys":
        payload = {
            "decision": label,
            "comment": good_payload["feedback"],
            "trace": good_payload["verdict_trace"],
        }
        return to_json(payload), bad_type, "schema"

    if bad_type == "missing_label_key":
        payload = dict(good_payload)
        payload.pop("is_correct", None)
        return to_json(payload), bad_type, "schema"

    if bad_type == "wrong_type_confidence":
        payload = dict(good_payload)
        payload["confidence"] = f"{good_payload['confidence']}"
        return to_json(payload), bad_type, "schema"

    if bad_type == "wrong_type_cited_evidence":
        payload = dict(good_payload)
        payload["cited_evidence"] = "doc_a, doc_b, doc_c"
        return to_json(payload), bad_type, "schema"

    if bad_type == "wrong_type_reasoning_steps":
        payload = dict(good_payload)
        payload["reasoning_steps"] = "step1 -> step2 -> step3"
        return to_json(payload), bad_type, "schema"

    if bad_type == "wrong_type_risk_flags":
        payload = dict(good_payload)
        payload["risk_flags"] = ["hallucination_risk", "missing_keypoint"]
        return to_json(payload), bad_type, "schema"

    if bad_type == "confidence_out_of_range":
        payload = dict(good_payload)
        payload["confidence"] = round(rng.uniform(1.05, 1.50), 3)
        return to_json(payload), bad_type, "schema"

    if bad_type == "confidence_nan_string":
        payload = dict(good_payload)
        payload["confidence"] = "NaN"
        return to_json(payload), bad_type, "schema"

    if bad_type == "wrong_enum_label":
        payload = dict(good_payload)
        payload["is_correct"] = rng.choice(["Maybe", "Unknown", "PartiallyCorrect"])
        return to_json(payload), bad_type, "schema"

    if bad_type == "null_label":
        payload = dict(good_payload)
        payload["is_correct"] = None
        return to_json(payload), bad_type, "schema"

    if bad_type == "empty_cited_evidence":
        payload = dict(good_payload)
        payload["cited_evidence"] = []
        return to_json(payload), bad_type, "length"

    if bad_type == "partial_cited_evidence":
        payload = dict(good_payload)
        payload["cited_evidence"] = [
            {"doc_id": "doc_a"},
            {"quote": "Evidence snippet without doc id"},
            {"doc_id": 7, "quote": 123},
        ]
        return to_json(payload), bad_type, "schema"

    if bad_type == "verdict_trace_flattened":
        payload = dict(good_payload)
        payload["verdict_trace"] = "semantic=0.8,coverage=0.7,consistency=0.6"
        return to_json(payload), bad_type, "schema"

    if bad_type == "missing_verdict_trace":
        payload = dict(good_payload)
        payload.pop("verdict_trace", None)
        return to_json(payload), bad_type, "schema"

    if bad_type == "extra_unknown_top_key":
        payload = dict(good_payload)
        payload["debug_payload"] = {
            "raw_logits": [0.10, 0.20, 0.70],
            "internal_notes": "for development only",
        }
        return to_json(payload), bad_type, "schema"

    if bad_type == "fenced_json":
        return f"```json\n{to_json(good_payload)}\n```", bad_type, "format"

    if bad_type == "yaml_like":
        return (
            f"is_correct: {label}\n"
            f"feedback: {good_payload['feedback']}\n"
            f"confidence: {good_payload['confidence']}"
        ), bad_type, "format"

    if bad_type == "json_with_trailing_text":
        return to_json(good_payload) + "\nFinal decision is above.", bad_type, "format"

    if bad_type == "json_array_output":
        payload = [good_payload, {"note": "secondary object"}]
        return json.dumps(payload, ensure_ascii=False), bad_type, "format"

    if bad_type == "single_quotes_json":
        return str(good_payload), bad_type, "format"

    if bad_type == "truncated_json":
        as_json = to_json(good_payload)
        cut = max(1, len(as_json) - rng.randint(5, 40))
        return as_json[:cut], bad_type, "format"

    # malformed_json
    malformed = (
        '{"is_correct":"'
        + label
        + '","feedback":"'
        + str(good_payload["feedback"]).replace('"', "'")
        + '"'
    )
    return malformed, bad_type, "format"


def make_row(
    idx: int,
    id_prefix: str,
    sample_label: str,
    case: dict[str, Any],
    truth_is_correct: bool,
    rng: random.Random,
) -> dict[str, Any]:
    base_student = rng.choice(case["correct_variants"] if truth_is_correct else case["incorrect_variants"])
    input_edge_type = choose_input_edge(sample_label, rng)
    student_answer = apply_input_edge(base_student, input_edge_type, rng)

    input_obj = {
        "task": {
            "task_id": f"{id_prefix}_task_{idx:07d}",
            "type": case["task_type"],
            "question": case["question"],
            "expected_answer": case["answer"],
            "language": rng.choice(["en", "en-US", "en-GB"]),
        },
        "context": {
            "documents": case["documents"],
            "timeline": [
                {"t": "T-2", "event": "draft prepared"},
                {"t": "T-1", "event": "evidence validated"},
                {"t": "T0", "event": "final recommendation issued"},
            ],
            "entity_graph": {
                "primary_actor": rng.choice(NAMES),
                "reviewer": rng.choice(NAMES),
                "relations": [
                    {"src": "primary_actor", "dst": "reviewer", "type": "reviewed_by"},
                ],
            },
            "metadata": {
                "domain": rng.choice(DOMAINS),
                "source_count": len(case["documents"]),
                "contains_superseded_doc": True,
            },
        },
        "rubric": {
            "must_include": case["key_points"][:2],
            "must_not_include": case["forbidden_points"][:2],
            "weights": {"semantic": 0.60, "coverage": 0.25, "consistency": 0.15},
            "pass_threshold": 0.72,
        },
        "student_submission": {
            "answer": student_answer,
            "evidence": [d["doc_id"] for d in case["documents"][: rng.choice([1, 2, 3])]],
            "confidence": round(rng.uniform(0.22, 0.96), 3),
            "revision_history": [
                {"rev": 1, "note": "initial draft"},
                {"rev": 2, "note": "minor wording update"},
            ],
            "channels": {"source": "chat", "locale": "en-US"},
        },
    }
    input_text = to_json(input_obj)

    good_payload = build_good_output(case=case, truth_is_correct=truth_is_correct, rng=rng)
    bad_type = ""
    bad_family = "none"
    if sample_label == "good":
        expected_output = to_json(good_payload)
    else:
        expected_output, bad_type, bad_family = build_bad_output(
            case=case,
            good_payload=good_payload,
            truth_is_correct=truth_is_correct,
            rng=rng,
        )

    return {
        "id": f"{id_prefix}_{idx:07d}",
        "Prompt": case["prompt"],
        "input": input_text,
        "expectedOutput": expected_output,
        "sample_label": sample_label,
        "truth_label": "Correct" if truth_is_correct else "Incorrect",
        "input_edge_type": input_edge_type,
        "bad_type": bad_type,
        "bad_family": bad_family,
        "output_json_valid": json_validity(expected_output),
        "eval": "correct" if sample_label == "good" else "incorrect",
    }


def generate_dataset(
    id_prefix: str,
    case_bank: list[dict[str, Any]],
    n_good: int,
    n_bad: int,
    truth_correct_ratio: float,
    seed: int,
) -> pd.DataFrame:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    idx = 0
    for _ in range(max(0, int(n_good))):
        case = rng.choice(case_bank)
        truth = bool(rng.random() < float(truth_correct_ratio))
        rows.append(make_row(idx=idx, id_prefix=id_prefix, sample_label="good", case=case, truth_is_correct=truth, rng=rng))
        idx += 1
    for _ in range(max(0, int(n_bad))):
        case = rng.choice(case_bank)
        truth = bool(rng.random() < float(truth_correct_ratio))
        rows.append(make_row(idx=idx, id_prefix=id_prefix, sample_label="bad", case=case, truth_is_correct=truth, rng=rng))
        idx += 1
    rng.shuffle(rows)
    return pd.DataFrame(rows)


def print_stats(name: str, df: pd.DataFrame) -> None:
    print(f"[INFO] {name}: rows={len(df)}")
    print(f"[INFO] {name}: sample_label={df['sample_label'].value_counts().to_dict()}")
    print(f"[INFO] {name}: truth_label={df['truth_label'].value_counts().to_dict()}")
    print(f"[INFO] {name}: prompt_count={df['Prompt'].nunique()}")
    bad = df[df["sample_label"] == "bad"]
    if not bad.empty:
        print(f"[INFO] {name}: bad_type={bad['bad_type'].value_counts().to_dict()}")
        print(f"[INFO] {name}: bad_family={bad['bad_family'].value_counts().to_dict()}")
        print(f"[INFO] {name}: output_json_valid_bad={bad['output_json_valid'].value_counts().to_dict()}")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed + 17)
    case_bank = build_case_bank(args.n_cases, rng=rng)

    large_df = generate_dataset(
        id_prefix="cxl",
        case_bank=case_bank,
        n_good=args.n_large_good,
        n_bad=args.n_large_bad,
        truth_correct_ratio=args.truth_correct_ratio,
        seed=args.seed + 101,
    )
    small_df = generate_dataset(
        id_prefix="cxs",
        case_bank=case_bank,
        n_good=args.n_small_good,
        n_bad=args.n_small_bad,
        truth_correct_ratio=args.truth_correct_ratio,
        seed=args.seed + 202,
    )

    cols = [
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
    large_df = large_df[cols]
    small_df = small_df[cols]

    large_path = Path(args.large_output_csv)
    small_path = Path(args.small_output_csv)
    large_path.parent.mkdir(parents=True, exist_ok=True)
    small_path.parent.mkdir(parents=True, exist_ok=True)
    large_df.to_csv(large_path, index=False, encoding="utf-8")
    small_df.to_csv(small_path, index=False, encoding="utf-8")

    print(f"[DONE] Large complex set -> {large_path}")
    print_stats("large_complex", large_df)
    print(f"[DONE] Small low-error set -> {small_path}")
    print_stats("small_low_error", small_df)


if __name__ == "__main__":
    main()
