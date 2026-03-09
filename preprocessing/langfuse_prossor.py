#!/usr/bin/env python3
"""Preprocess Langfuse-exported CSV for final_metric_refactor.

Converts a production-style dataset (e.g., ``data/prod_data_sample.csv``) into
the source schema expected by final_metric_refactor:

- id
- Prompt
- input
- expectedOutput
- eval
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_EVAL = "correct"
EVAL_COL_CANDIDATES = (
    "eval",
    "correctness",
    "Score Correctness",
    "Feedback Correctness",
    "label",
    "is_correct",
)
INPUT_COL_CANDIDATES = ("input", "user_input", "request", "prompt_input")
OUTPUT_COL_CANDIDATES = ("output", "assistant_output", "response", "completion")
ID_COL_CANDIDATES = ("id", "row_id", "traceId")


def _safe_json_load(value: Any) -> Any | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return None


def _normalize_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _normalize_id(value: Any, fallback: int) -> str:
    raw = _normalize_str(value)
    if not raw:
        return str(fallback)
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {'"', "'"}:
        raw = raw[1:-1].strip()
    return raw or str(fallback)


def _pick_existing_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _extract_messages_container(parsed: Any) -> list[dict[str, Any]] | None:
    if isinstance(parsed, list):
        items = [x for x in parsed if isinstance(x, dict)]
        return items if items else None
    if isinstance(parsed, dict):
        msgs = parsed.get("messages")
        if isinstance(msgs, list):
            items = [x for x in msgs if isinstance(x, dict)]
            return items if items else None
        if "role" in parsed and "content" in parsed:
            return [parsed]
    return None


def extract_system_prompt(raw_input: Any) -> str:
    parsed = _safe_json_load(raw_input)
    if parsed is None:
        return ""
    messages = _extract_messages_container(parsed)
    if not messages:
        return ""

    systems: list[str] = []
    for msg in messages:
        role = _normalize_str(msg.get("role")).lower()
        if role == "system":
            content = _normalize_str(msg.get("content"))
            if content:
                systems.append(content)
    return "\n\n".join(systems)


def extract_user_context(raw_input: Any) -> str:
    parsed = _safe_json_load(raw_input)
    if parsed is None:
        return _normalize_str(raw_input)

    messages = _extract_messages_container(parsed)
    if not messages:
        if isinstance(parsed, dict):
            for key in ("user_input", "input", "query", "question", "content"):
                if key in parsed:
                    val = _normalize_str(parsed.get(key))
                    if val:
                        return val
        return _normalize_str(raw_input)

    user_msgs: list[str] = []
    for msg in messages:
        role = _normalize_str(msg.get("role")).lower()
        if role == "user":
            content = _normalize_str(msg.get("content"))
            if content:
                user_msgs.append(content)
    if user_msgs:
        return "\n\n".join(user_msgs)

    # Fallback: first non-empty message content.
    for msg in messages:
        content = _normalize_str(msg.get("content"))
        if content:
            return content
    return ""


def extract_assistant_response(raw_output: Any) -> str:
    parsed = _safe_json_load(raw_output)
    if parsed is None:
        return _normalize_str(raw_output)

    messages = _extract_messages_container(parsed)
    if messages:
        assistant_msgs: list[str] = []
        for msg in messages:
            role = _normalize_str(msg.get("role")).lower()
            if role == "assistant":
                content = _normalize_str(msg.get("content"))
                if content:
                    assistant_msgs.append(content)
        if assistant_msgs:
            return "\n\n".join(assistant_msgs)
        for msg in messages:
            content = _normalize_str(msg.get("content"))
            if content:
                return content
        return ""

    if isinstance(parsed, dict):
        for key in ("content", "response", "answer", "output", "text"):
            if key in parsed:
                val = _normalize_str(parsed.get(key))
                if val:
                    return val
    return _normalize_str(parsed)


def _select_eval_series(df: pd.DataFrame, default_eval: str) -> pd.Series:
    for col in EVAL_COL_CANDIDATES:
        if col not in df.columns:
            continue
        series = df[col].astype(str).str.strip().str.replace('"', "", regex=False)
        non_empty = series.ne("") & series.str.lower().ne("nan")
        if bool(non_empty.any()):
            return series.mask(~non_empty, default_eval)
    return pd.Series([default_eval] * len(df), index=df.index, dtype=object)


def preprocess_langfuse_csv(
    source_csv: str | Path,
    output_csv: str | Path | None = None,
    *,
    max_rows: int = 0,
    default_eval: str = DEFAULT_EVAL,
    keep_original_cols: bool = False,
) -> Path:
    """Convert Langfuse CSV to final_metric_refactor source schema."""
    source_path = Path(source_csv).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Source CSV not found: {source_path}")

    df = pd.read_csv(source_path)
    if max_rows > 0:
        df = df.head(max_rows).copy()

    input_col = _pick_existing_column(df, INPUT_COL_CANDIDATES)
    output_col = _pick_existing_column(df, OUTPUT_COL_CANDIDATES)
    id_col = _pick_existing_column(df, ID_COL_CANDIDATES)

    if input_col is None:
        raise ValueError(f"Input column not found. Tried: {INPUT_COL_CANDIDATES}")
    if output_col is None:
        raise ValueError(f"Output column not found. Tried: {OUTPUT_COL_CANDIDATES}")

    eval_series = _select_eval_series(df, default_eval=default_eval)
    if id_col is not None:
        id_series = df[id_col]
    else:
        id_series = pd.Series(range(len(df)), index=df.index, dtype=object)

    out_df = pd.DataFrame(
        {
            "id": [_normalize_id(v, idx) for idx, v in enumerate(id_series.tolist())],
            "Prompt": df[input_col].apply(extract_system_prompt).astype(str),
            "input": df[input_col].apply(extract_user_context).astype(str),
            "expectedOutput": df[output_col].apply(extract_assistant_response).astype(str),
            "eval": eval_series.astype(str),
        }
    )

    if keep_original_cols:
        existing = set(out_df.columns)
        for col in df.columns:
            if col not in existing:
                out_df[col] = df[col]

    if output_csv is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = source_path.parent / f"{source_path.stem}_langfuse_preprocessed_{ts}.csv"
    else:
        output_path = Path(output_csv).resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Preprocess Langfuse CSV for final_metric_refactor")
    p.add_argument("--source-csv", type=Path, required=True, help="Path to raw Langfuse CSV")
    p.add_argument("--output-csv", type=Path, default=None, help="Path to save preprocessed CSV")
    p.add_argument("--max-rows", type=int, default=0, help="Limit rows (0 means all)")
    p.add_argument("--default-eval", default=DEFAULT_EVAL, help="Default eval value when label is missing")
    p.add_argument(
        "--keep-original-cols",
        action="store_true",
        help="Append original columns after the required final_metric columns",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    output = preprocess_langfuse_csv(
        source_csv=args.source_csv,
        output_csv=args.output_csv,
        max_rows=int(args.max_rows),
        default_eval=str(args.default_eval),
        keep_original_cols=bool(args.keep_original_cols),
    )
    print(f"[DONE] preprocessed_csv: {output}")


if __name__ == "__main__":
    main()
