#!/usr/bin/env python3
"""
Preprocess prod_data_sample.csv and run final_metric_refactor.

This script:
1. Extracts user_context (user message) from input JSON
2. Extracts assistant response from output JSON
3. Creates a minimal CSV format for final_metric_refactor
4. Runs final_metric_refactor on the processed data
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Any

import pandas as pd
import numpy as np

# Add parent directory to path for imports
FINAL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FINAL_DIR))

from final_metric_refactor.config import FinalMetricConfig
from final_metric_refactor.run import run
from final_metric_refactor.shared.preprocessor import safe_json_load


def extract_user_context(input_data: str | dict) -> str:
    """Extract user context (user message) from input JSON.

    Args:
        input_data: Input as string or dict. Expected format:
            - List of dicts with 'role' and 'content'
            - Or direct string

    Returns:
        Extracted user message content
    """
    if not input_data or (isinstance(input_data, float) and pd.isna(input_data)):
        return ""

    # Parse if string
    if isinstance(input_data, str):
        try:
            parsed = json.loads(input_data)
        except (json.JSONDecodeError, TypeError):
            return str(input_data)
    else:
        parsed = input_data

    # Handle list of messages
    if isinstance(parsed, list):
        for msg in parsed:
            if isinstance(msg, dict):
                if msg.get("role") == "user":
                    return str(msg.get("content", ""))
        # If no user message, return first message content
        if parsed and isinstance(parsed[0], dict):
            return str(parsed[0].get("content", ""))

    # Handle direct dict
    if isinstance(parsed, dict):
        if "content" in parsed:
            return str(parsed["content"])
        return str(parsed)

    return str(parsed) if parsed else ""


def extract_assistant_response(output_data: str | dict) -> str:
    """Extract assistant response from output JSON.

    Args:
        output_data: Output as string or dict. Expected format:
            - Dict with 'content' or 'role'/'content'
            - Or direct string

    Returns:
        Extracted assistant response
    """
    if not output_data or (isinstance(output_data, float) and pd.isna(output_data)):
        return ""

    # Parse if string
    if isinstance(output_data, str):
        try:
            parsed = json.loads(output_data)
        except (json.JSONDecodeError, TypeError):
            return str(output_data)
    else:
        parsed = output_data

    # Handle dict
    if isinstance(parsed, dict):
        if "content" in parsed:
            return str(parsed["content"])
        # Try to find any text-like value
        for key in ["text", "response", "answer", "output"]:
            if key in parsed:
                return str(parsed[key])
        return str(parsed)

    # Handle list
    if isinstance(parsed, list):
        if parsed and isinstance(parsed[0], dict) and "content" in parsed[0]:
            return str(parsed[0]["content"])
        return str(parsed)

    return str(parsed) if parsed else ""


def preprocess_prod_data(
    source_csv: Path,
    output_csv: Path | None = None,
    max_rows: int = 0,
    keep_original_cols: bool = False,
) -> Path:
    """Preprocess prod_data_sample.csv for final_metric_refactor.

    Args:
        source_csv: Path to prod_data_sample.csv
        output_csv: Path to save processed CSV. If None, saves to same dir as source
        max_rows: Max rows to process (0 = all)
        keep_original_cols: Whether to keep original columns

    Returns:
        Path to processed CSV
    """
    print(f"Loading data from {source_csv}...")
    df = pd.read_csv(source_csv)

    print(f"Loaded {len(df)} rows")

    if max_rows > 0:
        df = df.head(max_rows)
        print(f"Limited to {max_rows} rows")

    # Extract user context and assistant response
    print("Extracting user_context from input...")
    df["user_context"] = df["input"].apply(extract_user_context)

    print("Extracting assistant_response from output...")
    df["assistant_response"] = df["output"].apply(extract_assistant_response)

    # Use existing evaluation columns if available
    # Try to find a quality/correctness column
    eval_col_candidates = [
        "correctness",
        "score",
        "A Score",
        "correctness",
        "is_correct",
        "label",
        "eval",
    ]

    eval_col = None
    for col in eval_col_candidates:
        if col in df.columns:
            eval_col = col
            print(f"Found evaluation column: {eval_col}")
            break

    if eval_col:
        # 평가 컬럼이 모두 비어있는 경우 처리
        # 빈 값 패턴: "", NaN, None, 공백 등
        cleaned = df[eval_col].astype(str).str.replace('"', '').str.strip()
        is_empty = (cleaned == "") | df[eval_col].isna()

        if is_empty.all():
            print(f"WARNING: {eval_col} 컬럼이 모두 비어있습니다.")
            print("         기본 평가값 'correct'를 사용합니다.")
            df["eval"] = "correct"
        else:
            df["eval"] = df[eval_col]
    else:
        print("WARNING: No evaluation column found. Using default value 'correct'.")
        df["eval"] = "correct"

    # Build output dataframe with minimal required columns
    output_df = pd.DataFrame({
        "id": df.get("id", range(len(df))),
        "user_context": df["user_context"],
        "assistant_response": df["assistant_response"],
        "eval": df["eval"],
    })

    # Add optional columns if they exist
    if "traceId" in df.columns:
        output_df["traceId"] = df["traceId"]
    if "userId" in df.columns:
        output_df["userId"] = df["userId"]
    if "model" in df.columns:
        output_df["model"] = df["model"]

    # Keep original columns if requested
    if keep_original_cols:
        for col in df.columns:
            if col not in output_df.columns:
                output_df[col] = df[col]

    # Save processed data
    if output_csv is None:
        output_csv = source_csv.parent / f"prod_data_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv, index=False)
    print(f"Saved processed data to {output_csv}")
    print(f"  Rows: {len(output_df)}")
    print(f"  Columns: {list(output_df.columns)}")

    # Print sample
    print("\nSample row:")
    if len(output_df) > 0:
        row = output_df.iloc[0]
        for col in ["user_context", "assistant_response", "eval"]:
            val = str(row[col])[:200]
            print(f"  {col}: {val}...")

    return output_csv


def run_final_metric_on_prod_data(
    source_csv: Path,
    max_rows: int = 0,
    inspection_mode: str = "integrated",
    output_dir: Path | None = None,
) -> None:
    """Preprocess prod_data and run final_metric_refactor.

    Args:
        source_csv: Path to prod_data_sample.csv
        max_rows: Max rows to process (0 = all)
        inspection_mode: "integrated" or "detailed"
        output_dir: Custom output directory (optional)
    """
    # Step 1: Preprocess data
    print("=" * 80)
    print("STEP 1: Preprocessing prod_data_sample.csv")
    print("=" * 80)

    processed_csv = preprocess_prod_data(
        source_csv=source_csv,
        max_rows=max_rows,
    )

    # Step 2: Run final_metric_refactor
    print("\n" + "=" * 80)
    print("STEP 2: Running final_metric_refactor")
    print("=" * 80)

    run_tag = f"prod_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    config = FinalMetricConfig(
        run_tag=run_tag,
        source_csv=processed_csv,
        max_rows=0,  # Already limited in preprocessing
        inspection_mode=inspection_mode,
        output_dir=output_dir,
        # Column mapping for processed data
        input_col="user_context",
        output_col="assistant_response",
        label_col="eval",
        source_id_col="id",
    )

    print(f"\nConfig:")
    print(f"  run_tag: {config.run_tag}")
    print(f"  source_csv: {config.source_csv}")
    print(f"  inspection_mode: {config.inspection_mode}")
    print(f"  input_col: {config.input_col}")
    print(f"  output_col: {config.output_col}")
    print(f"  label_col: {config.label_col}")
    print()

    artifacts = run(config)

    print("\n" + "=" * 80)
    print("STEP 3: Final Metric Complete")
    print("=" * 80)
    print(f"Output directory: {artifacts.output_dir}")
    print(f"Files created:")
    for f in artifacts.output_dir.glob("*"):
        print(f"  - {f.name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess prod_data_sample.csv and run final_metric_refactor"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "prod_data_sample.csv",
        help="Path to prod_data_sample.csv",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Max rows to process (0 = all)",
    )
    parser.add_argument(
        "--mode",
        choices=["integrated", "detailed"],
        default="integrated",
        help="Inspection mode",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Custom output directory",
    )
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Only preprocess, don't run final_metric",
    )

    args = parser.parse_args()

    source_csv = args.source
    if not source_csv.exists():
        print(f"Error: {source_csv} not found")
        sys.exit(1)

    if args.preprocess_only:
        processed_csv = preprocess_prod_data(source_csv, max_rows=args.max_rows)
        print(f"\nProcessed CSV: {processed_csv}")
    else:
        run_final_metric_on_prod_data(
            source_csv=source_csv,
            max_rows=args.max_rows,
            inspection_mode=args.mode,
            output_dir=args.output_dir,
        )
