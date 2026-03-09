"""
JSON Schema Wrapper - Wrapper for strict JSON schema validation.

Provides a hard gate to filter outputs that don't conform to the expected schema.
"""

from dataclasses import dataclass
from typing import Any

from final_metric_refactor.hard_gate.schema_metric import JSONSchemaMetric


@dataclass
class SchemaGateResult:
    """Result from the JSON schema gate."""

    num_outputs: int
    num_passed: int
    pass_rate: float
    valid_outputs: list[str]
    invalid_outputs: list[dict]
    per_output_results: list[dict]


class JSONSchemaGate:
    """
    Hard gate for JSON Schema validation.

    Use this after IFEval and before similarity evaluation to ensure
    only schema-conforming outputs are passed through.

    Usage:
        gate = JSONSchemaGate(schema={
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": False,
            "required": ["score", "explanation"],
            "properties": {
                "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                "explanation": {"type": "string"},
            },
        })
        result = gate.validate(outputs=["...", "..."])
        # Use result.valid_outputs for downstream processing
    """

    def __init__(
        self,
        schema: dict[str, Any],
        strict_keys: bool = True,
        allow_extra_keys: bool = True,
    ):
        """
        Initialize the JSON schema gate.

        Args:
            schema: JSON Schema defining expected keys and their types.
            strict_keys: If True, all schema keys must be present.
            allow_extra_keys: If True, extra keys in output are allowed.
        """
        self.schema = schema
        self.metric = JSONSchemaMetric(
            schema=schema,
            strict_keys=strict_keys,
            allow_extra_keys=allow_extra_keys,
        )

    def validate(self, outputs: list[str]) -> SchemaGateResult:
        """
        Validate outputs against the schema.

        Args:
            outputs: List of output strings to validate.

        Returns:
            SchemaGateResult with valid/invalid outputs separated.
        """
        result = self.metric.validate(outputs)

        # Build per-output results for detailed reporting
        per_output_results = []
        for individual in result.individual_results:
            per_output_results.append({
                "index": individual.output_index,
                "output": (
                    individual.raw_output[:50] + "..."
                    if len(individual.raw_output) > 50
                    else individual.raw_output
                ),
                "is_valid_json": individual.is_valid_json,
                "is_schema_valid": individual.is_schema_valid,
                "passed": individual.is_valid_json and individual.is_schema_valid,
                "missing_keys": individual.missing_keys,
                "extra_keys": individual.extra_keys,
                "key_validations": {
                    kr.key: {
                        "present": kr.present,
                        "valid": kr.valid_value,
                        "error": kr.error,
                    }
                    for kr in individual.key_results
                },
                "error": individual.error,
            })

        return SchemaGateResult(
            num_outputs=result.num_outputs,
            num_passed=result.num_schema_valid,
            pass_rate=result.schema_valid_rate,
            valid_outputs=result.valid_outputs,
            invalid_outputs=result.invalid_outputs,
            per_output_results=per_output_results,
        )


if __name__ == "__main__":
    # Example usage
    schema = {
        "answer": [1, 2, 3, 4, 5],
        "explanation": "text",
    }

    gate = JSONSchemaGate(schema=schema)

    outputs = [
        '{"answer": 4, "explanation": "Two plus two equals four."}',
        '{"answer": 4.5, "explanation": "Two plus two equal four."}',  # 4.5 not in list
        '{"answers": 4, "explanation": "Two plus two is four."}',  # wrong key
        '{"answer": 4, "explanation": ""}',  # empty text
        '{"answer": 4, "explanation": "Valid answer."}',
        'not json',
    ]

    result = gate.validate(outputs)

    print(f"Total: {result.num_outputs}")
    print(f"Passed: {result.num_passed}")
    print(f"Pass rate: {result.pass_rate:.2%}")

    print("\nValid outputs:")
    for output in result.valid_outputs:
        print(f"  {output}")

    print("\nPer-output results:")
    for item in result.per_output_results:
        status = "✓" if item["passed"] else "✗"
        print(f"  [{item['index']}] {status} {item['output']}")
        if not item["passed"]:
            if item["error"]:
                print(f"        Error: {item['error']}")
            if item["missing_keys"]:
                print(f"        Missing: {item['missing_keys']}")
            for key, validation in item["key_validations"].items():
                if validation["error"]:
                    print(f"        {key}: {validation['error']}")
