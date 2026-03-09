from __future__ import annotations

from typing import Iterable


# Pipeline-internal rule keys (short form).
PIPELINE_RULE_KEYS: tuple[str, ...] = (
    "output",
    "direction",
    "length",
    "diff_resid",
    "delta_ridge_ens",
    "sim_conflict",
    "discourse_instability",
    "contradiction",
    "self_contradiction",
)

PIPELINE_RULE_SIGNAL_PREFIX: dict[str, str] = {
    "output": "output_signal",
    "direction": "direction_signal",
    "length": "length_signal",
    "diff_resid": "diff_residual_signal",
    "delta_ridge_ens": "delta_ridge_ens_signal",
    "sim_conflict": "similar_input_conflict_signal",
    "discourse_instability": "discourse_instability_signal",
    "contradiction": "contradiction_signal",
    "self_contradiction": "self_contradiction_signal",
}

PIPELINE_RULE_PASS_PREFIX: dict[str, str] = {
    "output": "output_pass",
    "direction": "direction_pass",
    "length": "length_pass",
    "diff_resid": "diff_residual_pass",
    "delta_ridge_ens": "delta_ridge_ens_pass",
    "sim_conflict": "similar_input_conflict_pass",
    "discourse_instability": "discourse_instability_pass",
    "contradiction": "contradiction_pass",
    "self_contradiction": "self_contradiction_pass",
}

PIPELINE_RULE_ALIASES: dict[str, str] = {
    "diff_residual": "diff_resid",
    "delta_ridge": "delta_ridge_ens",
    "delta-ridge": "delta_ridge_ens",
    "delta_ridge_ensemble": "delta_ridge_ens",
    "delta-ridge-ensemble": "delta_ridge_ens",
    "delta-ridge-ens": "delta_ridge_ens",
    "similar_input_conflict": "sim_conflict",
    "sim_conflict": "sim_conflict",
    "semantic_contradiction": "contradiction",
    "contradiction": "contradiction",
    "selfcontradiction": "self_contradiction",
    "self-contradiction": "self_contradiction",
    "self contradiction": "self_contradiction",
}


# Runtime/export rule keys (long form).
RUNTIME_RULE_ORDER: tuple[str, ...] = (
    "output",
    "direction",
    "length",
    "diff_residual",
    "delta_ridge_ens",
    "similar_input_conflict",
    "discourse_instability",
    "contradiction",
    "self_contradiction",
)

RUNTIME_DEFAULT_ACTIVE_RULES: tuple[str, ...] = (
    "output",
    "direction",
    "length",
    "diff_residual",
    "delta_ridge_ens",
    "similar_input_conflict",
    "discourse_instability",
    "contradiction",
)

RUNTIME_RULE_SIGNAL_COL_NOMASK: dict[str, str] = {
    "output": "output_signal_nomask",
    "direction": "direction_signal_nomask",
    "length": "length_signal_nomask",
    "diff_residual": "diff_residual_signal_nomask",
    "delta_ridge_ens": "delta_ridge_ens_signal_nomask",
    "similar_input_conflict": "similar_input_conflict_signal_nomask",
    "discourse_instability": "discourse_instability_signal_nomask",
    "contradiction": "contradiction_signal_nomask",
    "self_contradiction": "self_contradiction_signal_nomask",
}

RUNTIME_RULE_AVAILABLE_COL_NOMASK: dict[str, str] = {
    "discourse_instability": "discourse_instability_available_nomask",
    "contradiction": "contradiction_available_nomask",
    "self_contradiction": "self_contradiction_available_nomask",
}

RUNTIME_RULE_ALIASES: dict[str, str] = {
    "output": "output",
    "direction": "direction",
    "length": "length",
    "diff_residual": "diff_residual",
    "diff_resid": "diff_residual",
    "delta_ridge_ens": "delta_ridge_ens",
    "delta_ridge": "delta_ridge_ens",
    "delta-ridge": "delta_ridge_ens",
    "delta_ridge_ensemble": "delta_ridge_ens",
    "delta-ridge-ensemble": "delta_ridge_ens",
    "delta-ridge-ens": "delta_ridge_ens",
    "similar_input_conflict": "similar_input_conflict",
    "sim_conflict": "similar_input_conflict",
    "discourse_instability": "discourse_instability",
    "contradiction": "contradiction",
    "self_contradiction": "self_contradiction",
    "selfcontradiction": "self_contradiction",
    "self-contradiction": "self_contradiction",
    "self contradiction": "self_contradiction",
}


def normalize_pipeline_rule_key(key: str) -> str:
    k = str(key).strip().lower()
    return PIPELINE_RULE_ALIASES.get(k, k)


def normalize_runtime_rule_key(key: str) -> str:
    k = str(key).strip().lower()
    return RUNTIME_RULE_ALIASES.get(k, k)


def parse_pipeline_rule_keys(raw: str) -> list[str]:
    keys: list[str] = []
    for token in str(raw).split(","):
        t = normalize_pipeline_rule_key(token)
        if not t:
            continue
        if t not in PIPELINE_RULE_KEYS:
            raise ValueError(f"Unknown rule key: {token}. Available={list(PIPELINE_RULE_KEYS)}")
        if t not in keys:
            keys.append(t)
    if not keys:
        raise ValueError("No rule keys provided.")
    return keys


def parse_runtime_rule_keys(raw: str, *, default: Iterable[str] | None = None) -> list[str]:
    keys: list[str] = []
    for token in str(raw).split(","):
        t = normalize_runtime_rule_key(token)
        if not t:
            continue
        if t not in RUNTIME_RULE_ORDER:
            raise ValueError(f"Unknown runtime rule key: {token}. Available={list(RUNTIME_RULE_ORDER)}")
        if t not in keys:
            keys.append(t)
    if keys:
        return keys
    return list(default) if default is not None else list(RUNTIME_RULE_ORDER)

