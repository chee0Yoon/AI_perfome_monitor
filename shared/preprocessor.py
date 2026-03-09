"""
Text preprocessing utilities and JSON parsing.
"""

import json
import math
import re
import unicodedata
from typing import Any

import pandas as pd
import numpy as np


def preprocess_text(text: str) -> str:
    """Preprocess text for similarity comparison.

    Applies the following normalizations:
    1. Unicode normalization (NFKC) - normalizes unicode characters
    2. Lowercase conversion
    3. Punctuation normalization - standardizes quotes, dashes, ellipsis
    4. Whitespace normalization - collapses multiple spaces, trims

    Args:
        text: Input text string.

    Returns:
        Normalized text string.
    """
    if not text:
        return ""

    # 1. Unicode normalization (NFKC normalizes compatibility characters)
    text = unicodedata.normalize("NFKC", text)

    # 2. Lowercase
    text = text.lower()

    # 3. Punctuation normalization
    # Normalize various quote styles to standard ASCII quotes
    text = re.sub(r"[''‚‛]", "'", text)  # Single quotes
    text = re.sub(r'[""„‟]', '"', text)  # Double quotes

    # Normalize dashes (em-dash, en-dash, etc.) to hyphen
    text = re.sub(r"[–—―‒]", "-", text)

    # Normalize ellipsis
    text = re.sub(r"…", "...", text)

    # 4. Whitespace normalization
    # Replace various unicode whitespace with regular space
    text = re.sub(r"[\u00a0\u2000-\u200b\u202f\u205f\u3000]", " ", text)
    # Remove invisible control characters
    text = re.sub(r"[\u200c\u200d\u200e\u200f\u00ad]", "", text)

    # Collapse repeated punctuation
    text = re.sub(r"([.!?]){2,}", r"\1", text)
    text = re.sub(r"(-){2,}", r"-", text)

    # Collapse multiple whitespace to single space
    text = re.sub(r"\s+", " ", text)
    # Trim leading/trailing whitespace
    text = text.strip()

    return text


def safe_json_load(value: Any) -> Any | None:
    """Safely parse a value as JSON.

    Args:
        value: Value to parse (can be dict, list, string, or None).

    Returns:
        Parsed JSON object, or None if parsing fails.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return None


def flatten_json_leaves(obj: Any, prefix: str = "") -> dict[str, str]:
    """Flatten nested JSON structure to dot-notation leaf paths.

    Args:
        obj: JSON object (dict, list, or primitive).
        prefix: Current path prefix (used in recursion).

    Returns:
        Dictionary mapping dot-notation paths to string values.

    Example:
        >>> flatten_json_leaves({"a": {"b": 1}, "c": [2, 3]})
        {'a.b': '1', 'c[0]': '2', 'c[1]': '3'}
    """
    leaves: dict[str, str] = {}
    if isinstance(obj, dict):
        for key in sorted(obj.keys()):
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            leaves.update(flatten_json_leaves(obj[key], next_prefix))
        return leaves
    if isinstance(obj, list):
        for idx, item in enumerate(obj):
            next_prefix = f"{prefix}[{idx}]"
            leaves.update(flatten_json_leaves(item, next_prefix))
        return leaves
    if obj is None:
        return leaves
    leaf_key = prefix if prefix else "_value"
    leaves[leaf_key] = str(obj)
    return leaves


def safe_bool_series(series: pd.Series) -> pd.Series:
    """Ensure a pandas Series is properly converted to boolean dtype.

    Handles various input types: bool, numeric, string representations.
    Missing values are treated as False.

    Args:
        series: Input Series of any type.

    Returns:
        Series with dtype bool, where all values are True or False (no NaN/None).
    """
    # Already proper boolean
    if series.dtype == bool:
        return series.fillna(False)

    # Numeric: convert 0/1/NaN to bool
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(bool).fillna(False)

    # String: map common representations
    mapped = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
            "y": True,
            "n": False,
        })
    )
    return mapped.fillna(False).astype(bool)
