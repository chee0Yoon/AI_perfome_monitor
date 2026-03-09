from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHARED_DATA_DIR = PROJECT_ROOT / "data"


def shared_csv_path(filename: str) -> Path:
    return SHARED_DATA_DIR / filename


def default_ambiguous_csv() -> Path:
    return shared_csv_path("ambiguous_prompt_benchmark_v3_large.csv")
