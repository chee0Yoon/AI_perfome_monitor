"""
JSON Schema Metric - Strict validation of JSON output against a schema.

Supports:
1) Legacy lightweight schema (list / "text" / nested dict)
2) Full JSON Schema (Draft 2020-12)
"""

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class KeyValidationResult:
    """Validation result for a single key."""

    key: str  # Dot-notation path for nested keys, e.g., "correctness.score"
    present: bool
    valid_type: bool
    valid_value: bool
    expected_type: str  # "categorical", "text", or "nested"
    expected_values: list[Any] | None  # For categorical keys
    actual_value: Any
    error: str | None = None


@dataclass
class SchemaValidationResult:
    """Validation result for a single output."""

    output_index: int
    raw_output: str
    is_valid_json: bool
    parsed_json: dict | None
    is_schema_valid: bool
    key_results: list[KeyValidationResult]
    missing_keys: list[str]  # Dot-notation paths for missing keys
    extra_keys: list[str]
    error: str | None = None


@dataclass
class AggregatedSchemaResult:
    """Aggregated results for multiple outputs."""

    num_outputs: int
    num_valid_json: int
    num_schema_valid: int
    json_parse_rate: float
    schema_valid_rate: float
    individual_results: list[SchemaValidationResult]
    valid_outputs: list[str]  # Outputs that passed all validations
    invalid_outputs: list[dict]  # Details of failed outputs


class JSONSchemaMetric:
    """
    Strict JSON schema validation metric with nested object support.

    Validates that JSON outputs conform to a specified schema.

    Usage:
        metric = JSONSchemaMetric(schema={...})
        result = metric.validate(outputs=["{'score': 3, 'explanation': 'Good', 'details': {'rating': 2, 'comment': 'OK'}}"])
    """

    def __init__(
        self,
        schema: dict[str, Any],
        strict_keys: bool = True,
        allow_extra_keys: bool = True,
    ):
        """
        Initialize the JSON schema metric.

        Args:
            schema: Schema defining expected keys and their types.
                   - list value = categorical (value must be in list)
                   - "text" = free text (must be non-empty string)
                   - dict value = nested object (validated recursively)
            strict_keys: If True, all schema keys must be present.
            allow_extra_keys: If True, extra keys in output are allowed.
        """
        self.schema = schema
        self.strict_keys = strict_keys
        self.allow_extra_keys = allow_extra_keys
        self._use_jsonschema = self._is_jsonschema(schema)
        self._validator = None
        if self._use_jsonschema:
            from jsonschema import Draft202012Validator
            self._validator = Draft202012Validator(schema)

    def _is_jsonschema(self, schema: dict[str, Any]) -> bool:
        if not isinstance(schema, dict):
            return False
        if "$schema" in schema:
            return True
        if schema.get("type") == "object":
            return True
        if "properties" in schema or "required" in schema:
            return True
        return False

    @staticmethod
    def _expected_type_name(expected: Any) -> str:
        if isinstance(expected, list):
            return "categorical"
        if expected == "text":
            return "text"
        if expected in {"integer", "int"}:
            return "integer"
        if expected in {"number", "float"}:
            return "number"
        if expected in {"boolean", "bool"}:
            return "boolean"
        if isinstance(expected, dict):
            return "nested"
        return "any"

    def _validate_value(
        self,
        key_path: str,
        expected: Any,
        actual_value: Any,
        key_results: list[KeyValidationResult],
        missing_keys: list[str],
    ) -> bool:
        """Legacy schema validation for list/'text'/nested dict."""
        if isinstance(expected, list):
            is_valid_value = str(actual_value) in [str(v) for v in expected]
            key_results.append(
                KeyValidationResult(
                    key=key_path,
                    present=True,
                    valid_type=True,
                    valid_value=is_valid_value,
                    expected_type="categorical",
                    expected_values=expected,
                    actual_value=actual_value,
                    error=None if is_valid_value else f"Value '{actual_value}' not in {expected}",
                )
            )
            return is_valid_value

        if expected == "text":
            is_string = isinstance(actual_value, str)
            is_non_empty = bool(actual_value) if is_string else False
            is_valid = is_string and is_non_empty
            key_results.append(
                KeyValidationResult(
                    key=key_path,
                    present=True,
                    valid_type=is_string,
                    valid_value=is_valid,
                    expected_type="text",
                    expected_values=None,
                    actual_value=actual_value,
                    error=None if is_valid else "Must be a non-empty string",
                )
            )
            return is_valid

        if expected in {"integer", "int"}:
            is_valid = isinstance(actual_value, int) and not isinstance(actual_value, bool)
            key_results.append(
                KeyValidationResult(
                    key=key_path,
                    present=True,
                    valid_type=is_valid,
                    valid_value=is_valid,
                    expected_type="integer",
                    expected_values=None,
                    actual_value=actual_value,
                    error=None if is_valid else "Must be an integer",
                )
            )
            return is_valid

        if expected in {"number", "float"}:
            is_valid = isinstance(actual_value, (int, float)) and not isinstance(actual_value, bool)
            key_results.append(
                KeyValidationResult(
                    key=key_path,
                    present=True,
                    valid_type=is_valid,
                    valid_value=is_valid,
                    expected_type="number",
                    expected_values=None,
                    actual_value=actual_value,
                    error=None if is_valid else "Must be a number",
                )
            )
            return is_valid

        if expected in {"boolean", "bool"}:
            is_valid = isinstance(actual_value, bool)
            key_results.append(
                KeyValidationResult(
                    key=key_path,
                    present=True,
                    valid_type=is_valid,
                    valid_value=is_valid,
                    expected_type="boolean",
                    expected_values=None,
                    actual_value=actual_value,
                    error=None if is_valid else "Must be a boolean",
                )
            )
            return is_valid

        if isinstance(expected, dict):
            if not isinstance(actual_value, dict):
                key_results.append(
                    KeyValidationResult(
                        key=key_path,
                        present=True,
                        valid_type=False,
                        valid_value=False,
                        expected_type="nested",
                        expected_values=None,
                        actual_value=actual_value,
                        error="Expected nested object, got " + type(actual_value).__name__,
                    )
                )
                return False

            nested_valid = True
            for nested_key, nested_expected in expected.items():
                nested_path = f"{key_path}.{nested_key}"
                if nested_key not in actual_value:
                    key_results.append(
                        KeyValidationResult(
                            key=nested_path,
                            present=False,
                            valid_type=False,
                            valid_value=False,
                            expected_type=(
                                self._expected_type_name(nested_expected)
                            ),
                            expected_values=nested_expected if isinstance(nested_expected, list) else None,
                            actual_value=None,
                            error="Key missing",
                        )
                    )
                    missing_keys.append(nested_path)
                    nested_valid = False
                else:
                    if not self._validate_value(
                        nested_path,
                        nested_expected,
                        actual_value[nested_key],
                        key_results,
                        missing_keys,
                    ):
                        nested_valid = False
            return nested_valid

        return True

    def _validate_single_output(
        self, output: str, index: int
    ) -> SchemaValidationResult:
        """Validate a single output against the schema."""
        # Step 1: Check if valid JSON
        try:
            parsed = json.loads(output)
            if not isinstance(parsed, dict):
                return SchemaValidationResult(
                    output_index=index,
                    raw_output=output,
                    is_valid_json=False,
                    parsed_json=None,
                    is_schema_valid=False,
                    key_results=[],
                    missing_keys=[],
                    extra_keys=[],
                    error="JSON parsed but is not a dictionary",
                )
        except json.JSONDecodeError as e:
            return SchemaValidationResult(
                output_index=index,
                raw_output=output,
                is_valid_json=False,
                parsed_json=None,
                is_schema_valid=False,
                key_results=[],
                missing_keys=[],
                extra_keys=[],
                error=f"Invalid JSON: {e}",
            )

        if self._use_jsonschema and self._validator is not None:
            errors = sorted(self._validator.iter_errors(parsed), key=lambda e: e.path)
            missing_keys: list[str] = []
            extra_keys: list[str] = []
            invalid_keys: list[KeyValidationResult] = []

            for err in errors:
                if err.validator == "required":
                    missing = err.message.split("'")[1] if "'" in err.message else None
                    if missing:
                        path = ".".join([str(p) for p in err.path]) if err.path else ""
                        missing_keys.append(f"{path}.{missing}".strip("."))
                elif err.validator == "additionalProperties":
                    extras = err.params.get("additionalProperties", [])
                    path = ".".join([str(p) for p in err.path]) if err.path else ""
                    for extra in extras:
                        extra_keys.append(f"{path}.{extra}".strip("."))
                else:
                    path = ".".join([str(p) for p in err.path]) if err.path else ""
                    invalid_keys.append(
                        KeyValidationResult(
                            key=path or "<root>",
                            present=True,
                            valid_type=False,
                            valid_value=False,
                            expected_type="schema",
                            expected_values=None,
                            actual_value=None,
                            error=err.message,
                        )
                    )

            is_schema_valid = len(errors) == 0
        else:
            schema_keys = set(self.schema.keys())
            output_keys = set(parsed.keys())

            missing_keys = list(schema_keys - output_keys)
            extra_keys = list(output_keys - schema_keys)

            key_results: list[KeyValidationResult] = []
            all_keys_valid = True

            for key, expected in self.schema.items():
                if key not in parsed:
                    expected_type = self._expected_type_name(expected)
                    key_results.append(
                        KeyValidationResult(
                            key=key,
                            present=False,
                            valid_type=False,
                            valid_value=False,
                            expected_type=expected_type,
                            expected_values=expected if isinstance(expected, list) else None,
                            actual_value=None,
                            error="Key missing",
                        )
                    )
                    all_keys_valid = False
                    continue

                actual_value = parsed[key]
                if not self._validate_value(key, expected, actual_value, key_results, missing_keys):
                    all_keys_valid = False

            invalid_keys = key_results
            is_schema_valid = (
                all_keys_valid
                and (not self.strict_keys or len(missing_keys) == 0)
                and (self.allow_extra_keys or len(extra_keys) == 0)
            )

        return SchemaValidationResult(
            output_index=index,
            raw_output=output,
            is_valid_json=True,
            parsed_json=parsed,
            is_schema_valid=is_schema_valid,
            key_results=invalid_keys,
            missing_keys=missing_keys,
            extra_keys=extra_keys,
            error=None if is_schema_valid else "Schema validation failed",
        )

    def validate(self, outputs: list[str]) -> AggregatedSchemaResult:
        """
        Validate multiple outputs against the schema.

        Args:
            outputs: List of JSON string outputs to validate.

        Returns:
            AggregatedSchemaResult with validation results.
        """
        individual_results = []
        valid_outputs = []
        invalid_outputs = []

        for idx, output in enumerate(outputs):
            result = self._validate_single_output(output, idx)
            individual_results.append(result)

            if result.is_valid_json and result.is_schema_valid:
                valid_outputs.append(output)
            else:
                invalid_outputs.append({
                    "index": idx,
                    "output": output[:100] + "..." if len(output) > 100 else output,
                    "is_valid_json": result.is_valid_json,
                    "is_schema_valid": result.is_schema_valid,
                    "error": result.error,
                    "missing_keys": result.missing_keys,
                    "invalid_keys": [
                        {"key": kr.key, "error": kr.error}
                        for kr in result.key_results
                        if not kr.valid_value
                    ],
                })

        num_outputs = len(outputs)
        num_valid_json = sum(1 for r in individual_results if r.is_valid_json)
        num_schema_valid = sum(1 for r in individual_results if r.is_schema_valid)

        return AggregatedSchemaResult(
            num_outputs=num_outputs,
            num_valid_json=num_valid_json,
            num_schema_valid=num_schema_valid,
            json_parse_rate=num_valid_json / num_outputs if num_outputs > 0 else 0.0,
            schema_valid_rate=num_schema_valid / num_outputs if num_outputs > 0 else 0.0,
            individual_results=individual_results,
            valid_outputs=valid_outputs,
            invalid_outputs=invalid_outputs,
        )


if __name__ == "__main__":
    # Example usage with nested schema
    print("=" * 60)
    print("Example 1: Simple flat schema")
    print("=" * 60)

    schema_simple = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["score", "explanation"],
        "properties": {
            "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
            "explanation": {"type": "string"},
        },
    }

    metric_simple = JSONSchemaMetric(schema=schema_simple)

    outputs_simple = [
        '{"score": 4, "explanation": "Good answer"}',  # Valid
        '{"score": 6, "explanation": "Invalid score"}',  # Invalid: score not in list
        '{"score": 3}',  # Invalid: missing explanation
    ]

    result_simple = metric_simple.validate(outputs_simple)
    print(f"Schema valid: {result_simple.num_schema_valid}/{result_simple.num_outputs}")

    print("\n" + "=" * 60)
    print("Example 2: Nested schema")
    print("=" * 60)

    schema_nested = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["overall_score", "summary", "correctness", "readability", "recommendation"],
        "properties": {
            "overall_score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
            "summary": {"type": "string"},
            "correctness": {
                "type": "object",
                "additionalProperties": False,
                "required": ["score", "feedback"],
                "properties": {
                    "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                    "feedback": {"type": "string"},
                },
            },
            "readability": {
                "type": "object",
                "additionalProperties": False,
                "required": ["score", "feedback"],
                "properties": {
                    "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                    "feedback": {"type": "string"},
                },
            },
            "recommendation": {
                "type": "string",
                "enum": ["approve", "approve_with_changes", "request_changes", "reject"],
            },
        },
    }

    metric_nested = JSONSchemaMetric(schema=schema_nested)

    outputs_nested = [
        # Valid nested output
        '{"overall_score": 4, "summary": "Good code.", "correctness": {"score": 5, "feedback": "Correct logic."}, "readability": {"score": 4, "feedback": "Clean code."}, "recommendation": "approve"}',
        # Missing nested key (correctness.feedback)
        '{"overall_score": 4, "summary": "Good.", "correctness": {"score": 5}, "readability": {"score": 4, "feedback": "OK."}, "recommendation": "approve"}',
        # Invalid nested value (correctness.score = 6)
        '{"overall_score": 4, "summary": "Good.", "correctness": {"score": 6, "feedback": "OK."}, "readability": {"score": 4, "feedback": "OK."}, "recommendation": "approve"}',
        # Missing entire nested object
        '{"overall_score": 4, "summary": "Good.", "readability": {"score": 4, "feedback": "OK."}, "recommendation": "approve"}',
        # Nested value is wrong type (correctness is string instead of object)
        '{"overall_score": 4, "summary": "Good.", "correctness": "good", "readability": {"score": 4, "feedback": "OK."}, "recommendation": "approve"}',
    ]

    result_nested = metric_nested.validate(outputs_nested)

    print(f"Total outputs: {result_nested.num_outputs}")
    print(f"Valid JSON: {result_nested.num_valid_json}")
    print(f"Schema valid: {result_nested.num_schema_valid}")
    print(f"Schema valid rate: {result_nested.schema_valid_rate:.2%}")

    print("\nValid outputs:")
    for output in result_nested.valid_outputs:
        print(f"  {output[:80]}...")

    print("\nInvalid outputs:")
    for item in result_nested.invalid_outputs:
        print(f"  [{item['index']}] {item['output'][:60]}...")
        if item['missing_keys']:
            print(f"      Missing: {item['missing_keys']}")
        if item['invalid_keys']:
            for inv in item['invalid_keys']:
                print(f"      Invalid key '{inv['key']}': {inv['error']}")
