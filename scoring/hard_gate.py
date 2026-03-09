#!/usr/bin/env python3
"""Hard-gate score calculator.

Score mapping:
- 5: pass rate == 100% (perfect only)
- 4: pass rate >= 90%
- 3: pass rate >= 80%
- 2: pass rate >= 70%
- 1: pass rate >= 60%
- 0: otherwise
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = ROOT / "results"
EPS = 1e-12


@dataclass(frozen=True)
class HardGateScoreResult:
    score: int
    pass_rate: float
    pass_rate_percent: float
    summary_csv: Path
    row_index: int
    mode: str | None
    generated_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": int(self.score),
            "hard_gate_pass_rate": float(self.pass_rate),
            "hard_gate_pass_rate_percent": float(self.pass_rate_percent),
            "summary_csv": str(self.summary_csv),
            "row_index": int(self.row_index),
            "mode": self.mode,
            "generated_at_utc": str(self.generated_at_utc),
        }


def normalize_pass_rate(pass_rate: float) -> float:
    """Normalize pass rate into [0.0, 1.0].

    Supports either ratio scale (0.0~1.0) or percent scale (0.0~100.0).
    """
    rate = float(pass_rate)
    if not math.isfinite(rate):
        raise ValueError(f"pass_rate must be finite, got: {pass_rate}")

    if rate > 1.0 + EPS:
        if rate <= 100.0 + EPS:
            rate = rate / 100.0
        else:
            raise ValueError(f"pass_rate out of range: {pass_rate}")

    if rate < -EPS:
        raise ValueError(f"pass_rate out of range: {pass_rate}")

    if rate < 0.0:
        rate = 0.0
    if rate > 1.0:
        rate = 1.0
    return float(rate)


def hard_gate_score_from_pass_rate(pass_rate: float) -> int:
    """Map pass rate to hard-gate score [0..5]."""
    rate = normalize_pass_rate(pass_rate)

    if math.isclose(rate, 1.0, rel_tol=0.0, abs_tol=EPS):
        return 5
    if rate >= 0.9 - EPS:
        return 4
    if rate >= 0.8 - EPS:
        return 3
    if rate >= 0.7 - EPS:
        return 2
    if rate >= 0.6 - EPS:
        return 1
    return 0


def _csv_has_hard_gate_pass_rate(csv_path: Path) -> bool:
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
    except Exception:
        return False
    return "hard_gate_pass_rate" in header


def find_latest_hard_gate_summary_csv(results_root: Path | str = DEFAULT_RESULTS_ROOT) -> Path:
    """Find the most recently modified summary CSV that includes hard_gate_pass_rate."""
    root = Path(results_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"results root not found: {root}")

    candidates = sorted(
        (
            p
            for p in root.rglob("*summary.csv")
            if p.is_file() and "thresholds_summary" not in p.name
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for csv_path in candidates:
        if _csv_has_hard_gate_pass_rate(csv_path):
            return csv_path.resolve()
    raise FileNotFoundError(f"no summary csv with hard_gate_pass_rate found under {root}")


def _load_summary_rows(summary_csv: Path | str) -> tuple[list[str], list[dict[str, str]]]:
    csv_path = Path(summary_csv).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"summary csv not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError(f"summary csv has no header: {csv_path}")
        rows = list(reader)

    if not rows:
        raise ValueError(f"summary csv has no data rows: {csv_path}")
    if "hard_gate_pass_rate" not in fieldnames:
        raise ValueError(f"hard_gate_pass_rate column not found: {csv_path}")
    return fieldnames, rows


def _select_row(rows: list[dict[str, str]], fieldnames: list[str], mode: str) -> tuple[int, dict[str, str], str | None]:
    if "mode" not in fieldnames:
        return 0, rows[0], None

    mode_target = str(mode).strip().lower()
    for idx, row in enumerate(rows):
        row_mode = str(row.get("mode", "")).strip().lower()
        if row_mode == mode_target:
            return idx, row, row.get("mode")

    raise ValueError(f'mode "{mode}" not found in summary rows')


def compute_hard_gate_score(summary_csv: Path | str, mode: str = "nomask") -> HardGateScoreResult:
    """Compute hard-gate score from a specific summary csv."""
    csv_path = Path(summary_csv).resolve()
    fieldnames, rows = _load_summary_rows(csv_path)
    row_idx, row, row_mode = _select_row(rows=rows, fieldnames=fieldnames, mode=mode)

    raw_rate = row.get("hard_gate_pass_rate", "")
    if raw_rate is None or str(raw_rate).strip() == "":
        raise ValueError(f"hard_gate_pass_rate is empty at row index {row_idx}: {csv_path}")

    rate = normalize_pass_rate(float(raw_rate))
    score = hard_gate_score_from_pass_rate(rate)

    return HardGateScoreResult(
        score=score,
        pass_rate=rate,
        pass_rate_percent=rate * 100.0,
        summary_csv=csv_path,
        row_index=row_idx,
        mode=row_mode,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def compute_hard_gate_score_from_latest(
    results_root: Path | str = DEFAULT_RESULTS_ROOT,
    mode: str = "nomask",
) -> HardGateScoreResult:
    """Compute hard-gate score from the latest summary csv in results root."""
    summary_csv = find_latest_hard_gate_summary_csv(results_root=results_root)
    return compute_hard_gate_score(summary_csv=summary_csv, mode=mode)
