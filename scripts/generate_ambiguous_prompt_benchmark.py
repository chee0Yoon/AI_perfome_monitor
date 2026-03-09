#!/usr/bin/env python3
"""
Generate a large synthetic benchmark with one ambiguous prompt and diverse edge cases.

Output includes:
- good samples: schema-valid and semantically aligned outputs
- bad samples: semantic, schema, format, and length edge cases
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd


AMBIGUOUS_PROMPT = """
You are checking whether a student's short answer matches the reference answer.

Return JSON with exactly these keys:
- "is_correct": "Correct" or "Incorrect"
- "feedback": 1-3 short sentences

Rules:
1) Focus on key meaning, not exact wording.
2) Minor grammar mistakes are acceptable.
3) If the evidence is borderline, choose whichever label seems more plausible.
4) Keep feedback concrete, concise, and directly tied to the answer.
5) Avoid adding new facts not present in the context.
""".strip()


BASE_CASE_BANK: list[dict[str, Any]] = [
    {
        "question": "What did Mina buy at the market?",
        "answer": "Mina bought fresh apples.",
        "section_text": "After school, Mina visited the market and bought fresh apples for her family.",
        "correct_variants": ["fresh apples", "Mina bought apples", "she bought fresh apples"],
        "incorrect_variants": ["bananas", "she bought milk", "I am not sure"],
    },
    {
        "question": "Where did the team practice?",
        "answer": "The team practiced in the gym.",
        "section_text": "Because it was raining, the team practiced in the gym instead of the field.",
        "correct_variants": ["in the gym", "they practiced at the gym", "gym"],
        "incorrect_variants": ["on the field", "at home", "outside"],
    },
    {
        "question": "Why was Leo late?",
        "answer": "Leo was late because he missed the bus.",
        "section_text": "Leo ran to the stop but missed the bus, so he arrived late to class.",
        "correct_variants": ["he missed the bus", "because Leo missed the bus", "missed the bus"],
        "incorrect_variants": ["he woke up early", "traffic was heavy", "he forgot homework"],
    },
    {
        "question": "What is the main benefit of recycling in the passage?",
        "answer": "Recycling reduces waste in landfills.",
        "section_text": "The text explains that recycling helps reduce waste in landfills and saves resources.",
        "correct_variants": ["it reduces landfill waste", "reduces waste in landfills", "less trash in landfills"],
        "incorrect_variants": ["it makes products cheaper", "it increases waste", "no benefit is given"],
    },
    {
        "question": "Who led the science project?",
        "answer": "Jin led the science project.",
        "section_text": "During the semester, Jin organized meetings and led the science project.",
        "correct_variants": ["Jin", "Jin led it", "the leader was Jin"],
        "incorrect_variants": ["Maya", "the teacher led it", "nobody led it"],
    },
    {
        "question": "When does the library close?",
        "answer": "The library closes at 8 PM.",
        "section_text": "On weekdays, the library closes at 8 PM and opens again at 9 AM.",
        "correct_variants": ["8 PM", "it closes at 8", "at eight in the evening"],
        "incorrect_variants": ["6 PM", "9 PM", "morning"],
    },
    {
        "question": "What did the writer recommend for beginners?",
        "answer": "The writer recommended starting with short daily practice.",
        "section_text": "For beginners, the writer suggests short daily practice instead of long weekly sessions.",
        "correct_variants": ["short daily practice", "start with short daily practice", "practice daily for short time"],
        "incorrect_variants": ["long weekly sessions", "skip practice", "buy expensive tools"],
    },
    {
        "question": "How did the class travel to the museum?",
        "answer": "The class traveled by train.",
        "section_text": "The students met early and traveled by train to the museum downtown.",
        "correct_variants": ["by train", "they took the train", "train"],
        "incorrect_variants": ["by bus", "walking", "by airplane"],
    },
    {
        "question": "What changed after the update?",
        "answer": "The app became faster.",
        "section_text": "After the update, users noticed that the app became faster and more stable.",
        "correct_variants": ["it became faster", "the app got faster", "faster performance"],
        "incorrect_variants": ["it became slower", "nothing changed", "it lost features only"],
    },
    {
        "question": "Why did Nari bring an umbrella?",
        "answer": "Nari brought an umbrella because rain was expected.",
        "section_text": "The forecast predicted rain, so Nari brought an umbrella before leaving.",
        "correct_variants": ["rain was expected", "because rain was predicted", "it might rain"],
        "incorrect_variants": ["for sun only", "to lend a friend", "no reason"],
    },
    {
        "question": "Which two tools did Owen pack for hiking?",
        "answer": "Owen packed a map and a flashlight.",
        "section_text": "Before sunrise, Owen packed a map and a flashlight for the trail.",
        "correct_variants": ["a map and a flashlight", "map and flashlight", "he packed both map and flashlight"],
        "incorrect_variants": ["a map only", "a flashlight only", "water bottle and snacks"],
    },
    {
        "question": "What were the two reasons Mira chose the bike?",
        "answer": "Mira chose the bike because it was cheaper and faster in traffic.",
        "section_text": "Mira compared options and chose the bike because it was cheaper and faster in traffic.",
        "correct_variants": ["cheaper and faster in traffic", "it was cheaper and faster", "cost and traffic speed"],
        "incorrect_variants": ["it looked cool", "only because it was cheaper", "she had no reason"],
    },
]


GOOD_FEEDBACK_CORRECT = [
    "The answer captures the key idea from the reference, so it is correct.",
    "The core meaning matches the reference answer, even with different wording.",
    "This response conveys the essential point and should be marked correct.",
]

GOOD_FEEDBACK_INCORRECT = [
    "The response misses or changes the key idea from the reference, so it is incorrect.",
    "The main meaning does not match the reference answer, so this should be incorrect.",
    "This answer does not capture the required core information and is incorrect.",
]

BAD_RAMBLING_TEMPLATES = [
    "The answer maybe looks somewhat fine in a broad sense, but there are many interpretive possibilities, and depending on perspective it could be read either way; still I made a choice.",
    "This response might be acceptable in one interpretation while also being problematic in another, and the confidence is moderate at best, so the decision is tentative.",
    "I can see reasons for both sides and the wording is ambiguous, but I picked one side with uncertainty due to limited explicit overlap.",
]

NOISY_TAILS = [
    "By the way the weather was sunny and everyone smiled.",
    "Also this reminds me of another topic from yesterday.",
    "I am adding extra words that are not useful to the question.",
]

OFFTOPIC_ANSWERS = [
    "I like pizza.",
    "The movie was interesting.",
    "My favorite color is blue.",
    "No idea, maybe something else.",
]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Create large ambiguous-prompt benchmark CSV.")
    parser.add_argument(
        "--output-csv",
        default=str(project_root / "data" / "ambiguous_prompt_benchmark.csv"),
        help="Path to output CSV",
    )
    parser.add_argument("--n-good", type=int, default=700, help="Number of well-formed samples")
    parser.add_argument("--n-bad", type=int, default=500, help="Number of intentionally degraded samples")
    parser.add_argument("--n-cases", type=int, default=180, help="Number of unique QA cases to synthesize")
    parser.add_argument("--truth-correct-ratio", type=float, default=0.55, help="Ratio of truth=Correct rows")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def to_output_json(is_correct: str, feedback: str, extra: dict[str, Any] | None = None) -> str:
    payload: dict[str, Any] = {"is_correct": is_correct, "feedback": feedback}
    if extra:
        payload.update(extra)
    return json.dumps(payload, ensure_ascii=False)


def build_good_output(truth_is_correct: bool, rng: random.Random) -> tuple[str, str]:
    if truth_is_correct:
        return "Correct", rng.choice(GOOD_FEEDBACK_CORRECT)
    return "Incorrect", rng.choice(GOOD_FEEDBACK_INCORRECT)


def generate_template_cases(n_cases: int, rng: random.Random) -> list[dict[str, Any]]:
    names = [
        "Ari",
        "Nora",
        "Joon",
        "Mila",
        "Theo",
        "Rin",
        "Hana",
        "Dino",
        "Sora",
        "Evan",
        "Lina",
        "Kai",
    ]
    places = ["library", "community center", "school lab", "museum", "sports hall", "station"]
    items = ["fresh oranges", "a new notebook", "paint brushes", "garden seeds", "a map", "safety gloves"]
    wrong_items = ["old batteries", "video games", "a sandwich", "a camera", "nothing"]
    reasons = [
        "the bus was delayed",
        "there was heavy rain",
        "the train stopped unexpectedly",
        "the road was blocked",
        "the alarm did not ring",
    ]
    wrong_reasons = [
        "she woke up very early",
        "traffic was perfect",
        "nothing happened",
        "he arrived before everyone",
    ]
    cases: list[dict[str, Any]] = []

    while len(cases) < max(0, int(n_cases)):
        pattern = rng.choice(["buy", "where", "why_late", "who_lead", "multi_part"])
        name = rng.choice(names)
        place = rng.choice(places)
        item = rng.choice(items)
        w_item = rng.choice([x for x in wrong_items if x != item])
        reason = rng.choice(reasons)
        w_reason = rng.choice(wrong_reasons)

        if pattern == "buy":
            cases.append(
                {
                    "question": f"What did {name} buy at the {place}?",
                    "answer": f"{name} bought {item}.",
                    "section_text": f"After class, {name} visited the {place} and bought {item} for a project.",
                    "correct_variants": [item, f"{name} bought {item}", f"bought {item}"],
                    "incorrect_variants": [w_item, "I do not know", OFFTOPIC_ANSWERS[rng.randrange(len(OFFTOPIC_ANSWERS))]],
                }
            )
        elif pattern == "where":
            team = rng.choice(["the robotics team", "the debate club", "the choir", "the science class"])
            cases.append(
                {
                    "question": f"Where did {team} practice?",
                    "answer": f"{team} practiced in the {place}.",
                    "section_text": f"Due to schedule changes, {team} practiced in the {place} this week.",
                    "correct_variants": [f"in the {place}", place, f"they practiced in the {place}"],
                    "incorrect_variants": ["outside", "at home", "on the field"],
                }
            )
        elif pattern == "why_late":
            cases.append(
                {
                    "question": f"Why was {name} late?",
                    "answer": f"{name} was late because {reason}.",
                    "section_text": f"{name} arrived late because {reason}, according to the note.",
                    "correct_variants": [reason, f"because {reason}", f"{name} was late because {reason}"],
                    "incorrect_variants": [w_reason, "no reason", "not late"],
                }
            )
        elif pattern == "who_lead":
            project = rng.choice(["the design project", "the cleanup campaign", "the coding sprint", "the history fair"])
            other = rng.choice([x for x in names if x != name])
            cases.append(
                {
                    "question": f"Who led {project}?",
                    "answer": f"{name} led {project}.",
                    "section_text": f"For the semester showcase, {name} coordinated tasks and led {project}.",
                    "correct_variants": [name, f"{name} led it", f"the leader was {name}"],
                    "incorrect_variants": [other, "the teacher", "nobody"],
                }
            )
        else:
            benefit_a = rng.choice(["cheaper", "faster", "safer", "more reliable"])
            benefit_b = rng.choice([x for x in ["cheaper", "faster", "safer", "more reliable"] if x != benefit_a])
            cases.append(
                {
                    "question": f"What were the two reasons {name} chose the plan?",
                    "answer": f"{name} chose the plan because it was {benefit_a} and {benefit_b}.",
                    "section_text": f"{name} compared options and chose the plan because it was {benefit_a} and {benefit_b}.",
                    "correct_variants": [f"{benefit_a} and {benefit_b}", f"because it was {benefit_a} and {benefit_b}"],
                    "incorrect_variants": [benefit_a, benefit_b, "no reason was given"],
                }
            )
    return cases


def build_case_bank(target_n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed + 17)
    merged = list(BASE_CASE_BANK)
    need = max(target_n - len(BASE_CASE_BANK), 0)
    attempts = 0
    while need > 0 and attempts < 8:
        attempts += 1
        merged.extend(generate_template_cases(n_cases=max(need * 2, 20), rng=rng))
        # Keep deterministic uniqueness by (question, answer).
        seen: set[tuple[str, str]] = set()
        uniq: list[dict[str, Any]] = []
        for c in merged:
            key = (str(c["question"]), str(c["answer"]))
            if key not in seen:
                seen.add(key)
                uniq.append(c)
        merged = uniq
        need = max(target_n - len(merged), 0)
    return merged


def apply_input_edge(base_answer: str, edge_type: str, rng: random.Random) -> str:
    if edge_type == "clean":
        return base_answer
    if edge_type == "lower":
        return base_answer.lower()
    if edge_type == "extra_space":
        return "  " + "   ".join(base_answer.split()) + "  "
    if edge_type == "quoted":
        return f"\"{base_answer}\""
    if edge_type == "newline_tail":
        return f"{base_answer}\nI think this is right."
    if edge_type == "context_tail":
        return f"{base_answer} {rng.choice(NOISY_TAILS)}"
    if edge_type == "empty":
        return ""
    if edge_type == "uncertain":
        return f"maybe {base_answer}?"
    if edge_type == "offtopic":
        return rng.choice(OFFTOPIC_ANSWERS)
    if edge_type == "noisy_long":
        return f"{base_answer} {' '.join(rng.choice(NOISY_TAILS) for _ in range(3))}"
    return base_answer


def choose_input_edge(sample_label: str, rng: random.Random) -> str:
    if sample_label == "good":
        return rng.choice(["clean", "lower", "extra_space", "quoted", "newline_tail", "context_tail"])
    return rng.choice(
        ["clean", "uncertain", "offtopic", "noisy_long", "extra_space", "quoted", "empty", "newline_tail"]
    )


def build_bad_output(
    truth_is_correct: bool,
    nominal_is_correct: str,
    nominal_feedback: str,
    rng: random.Random,
) -> tuple[str, str, str]:
    """
    Returns:
    - output_text
    - bad_type
    - bad_family (semantic/schema/format/length)
    """
    truth_label = "Correct" if truth_is_correct else "Incorrect"
    bad_type = rng.choice(
        [
            "flip_label",
            "contradict_feedback",
            "rambling",
            "empty_feedback",
            "ultra_long_feedback",
            "wrong_keys",
            "wrong_type",
            "extra_keys",
            "fenced_json",
            "malformed_json",
            "yaml_like",
            "json_with_trailing_text",
        ]
    )

    if bad_type == "flip_label":
        flipped = "Incorrect" if truth_label == "Correct" else "Correct"
        feedback = (
            "The response does not really match the main point, so this should be incorrect."
            if flipped == "Incorrect"
            else "The response seems close enough to the key meaning and can be accepted as correct."
        )
        return to_output_json(flipped, feedback), bad_type, "semantic"

    if bad_type == "contradict_feedback":
        contradictory_feedback = (
            "The response misses the key idea, but I am marking it correct."
            if nominal_is_correct == "Correct"
            else "The response captures the key idea, but I am marking it incorrect."
        )
        return to_output_json(nominal_is_correct, contradictory_feedback), bad_type, "semantic"

    if bad_type == "rambling":
        return to_output_json(nominal_is_correct, rng.choice(BAD_RAMBLING_TEMPLATES)), bad_type, "semantic"

    if bad_type == "empty_feedback":
        return to_output_json(nominal_is_correct, ""), bad_type, "length"

    if bad_type == "ultra_long_feedback":
        long_feedback = " ".join(rng.choice(BAD_RAMBLING_TEMPLATES) for _ in range(8))
        return to_output_json(nominal_is_correct, long_feedback), bad_type, "length"

    if bad_type == "wrong_keys":
        payload = {"decision": nominal_is_correct, "reason": nominal_feedback}
        return json.dumps(payload, ensure_ascii=False), bad_type, "schema"

    if bad_type == "wrong_type":
        payload = {"is_correct": nominal_is_correct == "Correct", "feedback": nominal_feedback}
        return json.dumps(payload, ensure_ascii=False), bad_type, "schema"

    if bad_type == "extra_keys":
        payload = {
            "is_correct": nominal_is_correct,
            "feedback": nominal_feedback,
            "confidence": round(rng.uniform(0.31, 0.92), 2),
            "notes": "extra field not requested",
        }
        return json.dumps(payload, ensure_ascii=False), bad_type, "schema"

    if bad_type == "fenced_json":
        inner = to_output_json(nominal_is_correct, nominal_feedback)
        return f"```json\n{inner}\n```", bad_type, "format"

    if bad_type == "yaml_like":
        return f"is_correct: {nominal_is_correct}\nfeedback: {nominal_feedback}", bad_type, "format"

    if bad_type == "json_with_trailing_text":
        inner = to_output_json(nominal_is_correct, nominal_feedback)
        return inner + "\nFinal answer above.", bad_type, "format"

    # malformed_json
    malformed = '{"is_correct":"' + nominal_is_correct + '","feedback":"' + nominal_feedback.replace('"', "'")
    return malformed, bad_type, "format"


def json_validity(text: str) -> bool:
    try:
        json.loads(str(text))
        return True
    except Exception:
        return False


def make_row(
    idx: int,
    sample_label: str,
    case: dict[str, Any],
    truth_is_correct: bool,
    rng: random.Random,
) -> dict[str, Any]:
    base_student_answer = (
        rng.choice(case["correct_variants"]) if truth_is_correct else rng.choice(case["incorrect_variants"])
    )
    input_edge_type = choose_input_edge(sample_label=sample_label, rng=rng)
    student_answer = apply_input_edge(base_answer=base_student_answer, edge_type=input_edge_type, rng=rng)

    input_obj = {
        "question": case["question"],
        "answer": case["answer"],
        "section_text": case["section_text"],
        "student_answer": student_answer,
    }
    input_text = json.dumps(input_obj, ensure_ascii=False)

    nominal_is_correct, nominal_feedback = build_good_output(truth_is_correct=truth_is_correct, rng=rng)
    bad_type = ""
    bad_family = "none"
    if sample_label == "good":
        expected_output = to_output_json(nominal_is_correct, nominal_feedback)
    else:
        expected_output, bad_type, bad_family = build_bad_output(
            truth_is_correct=truth_is_correct,
            nominal_is_correct=nominal_is_correct,
            nominal_feedback=nominal_feedback,
            rng=rng,
        )

    return {
        "id": f"amb_{idx:07d}",
        "Prompt": AMBIGUOUS_PROMPT,
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


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []

    case_bank = build_case_bank(target_n=int(args.n_cases), seed=args.seed)
    if not case_bank:
        raise RuntimeError("No cases generated.")

    next_idx = 0
    for _ in range(max(0, int(args.n_good))):
        case = rng.choice(case_bank)
        truth_is_correct = bool(rng.random() < float(args.truth_correct_ratio))
        rows.append(make_row(next_idx, "good", case, truth_is_correct, rng))
        next_idx += 1

    for _ in range(max(0, int(args.n_bad))):
        case = rng.choice(case_bank)
        truth_is_correct = bool(rng.random() < float(args.truth_correct_ratio))
        rows.append(make_row(next_idx, "bad", case, truth_is_correct, rng))
        next_idx += 1

    rng.shuffle(rows)
    df = pd.DataFrame(rows)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"[DONE] Wrote {len(df)} rows -> {out_path}")
    print(f"[INFO] case_count: {len(case_bank)}")
    print(f"[INFO] sample_label counts: {df['sample_label'].value_counts().to_dict()}")
    print(f"[INFO] truth_label counts: {df['truth_label'].value_counts().to_dict()}")
    print(f"[INFO] input_edge_type counts: {df['input_edge_type'].value_counts().head(12).to_dict()}")
    if "bad_type" in df.columns:
        bad_rows = df[df["sample_label"] == "bad"]
        print(f"[INFO] bad_type counts: {bad_rows['bad_type'].value_counts().to_dict()}")
        print(f"[INFO] bad_family counts: {bad_rows['bad_family'].value_counts().to_dict()}")
        print(f"[INFO] output_json_valid (bad only): {bad_rows['output_json_valid'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
