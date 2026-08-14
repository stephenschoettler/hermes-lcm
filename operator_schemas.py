"""Tool schemas for the engine-managed LCM map operators."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any


_JSON_TYPES = {"object", "array", "string", "number", "integer", "boolean", "null"}
_SCHEMA_KEYWORDS = {
    "$id", "$schema", "title", "description", "default", "examples",
    "type", "properties", "required", "additionalProperties", "items",
    "enum", "minimum", "maximum", "anyOf",
}


class JSONSchemaError(ValueError):
    """The caller supplied a schema outside the supported bounded subset."""


class JSONSchemaValidationError(ValueError):
    """An operator result did not satisfy its declared output schema."""

    def __init__(self, message: str, path: tuple[str | int, ...] = ()):
        super().__init__(message)
        self.message = message
        self.path = path


class JSONSchemaValidator:
    """Dependency-free validator for the operator output-schema subset."""

    def __init__(self, schema: dict[str, Any]):
        _check_schema(schema, path=("schema",))
        self.schema = deepcopy(schema)

    def validate(self, instance: Any) -> None:
        _validate_instance(instance, self.schema, path=())


def compile_json_schema(schema: dict[str, Any]) -> JSONSchemaValidator:
    if not isinstance(schema, dict):
        raise JSONSchemaError("output_schema must be a JSON Schema object")
    return JSONSchemaValidator(schema)


def _check_schema(schema: Any, *, path: tuple[str | int, ...]) -> None:
    if not isinstance(schema, dict):
        raise JSONSchemaError(f"{_path(path)} must be an object")
    unsupported = sorted(set(schema) - _SCHEMA_KEYWORDS)
    if unsupported:
        raise JSONSchemaError(
            f"{_path(path)} uses unsupported keyword(s): {', '.join(unsupported)}"
        )
    expected_type = schema.get("type")
    if expected_type is not None and expected_type not in _JSON_TYPES:
        raise JSONSchemaError(f"{_path(path + ('type',))} is not a supported JSON type")
    enum = schema.get("enum")
    if enum is not None:
        if not isinstance(enum, list) or not enum:
            raise JSONSchemaError(f"{_path(path + ('enum',))} must be a non-empty array")
        try:
            for value in enum:
                _canonical_json(value)
        except (TypeError, ValueError) as exc:
            raise JSONSchemaError(f"{_path(path + ('enum',))} must contain JSON values") from exc
    for keyword in ("minimum", "maximum"):
        value = schema.get(keyword)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise JSONSchemaError(f"{_path(path + (keyword,))} must be a number")
    if "minimum" in schema and "maximum" in schema and schema["minimum"] > schema["maximum"]:
        raise JSONSchemaError(f"{_path(path)} minimum cannot exceed maximum")
    properties = schema.get("properties")
    if properties is not None:
        if not isinstance(properties, dict) or not all(isinstance(key, str) for key in properties):
            raise JSONSchemaError(f"{_path(path + ('properties',))} must be an object")
        for key, child in properties.items():
            _check_schema(child, path=path + ("properties", key))
    required = schema.get("required")
    if required is not None:
        if (
            not isinstance(required, list)
            or not all(isinstance(key, str) for key in required)
            or len(set(required)) != len(required)
        ):
            raise JSONSchemaError(f"{_path(path + ('required',))} must be unique strings")
    additional = schema.get("additionalProperties")
    if additional is not None and not isinstance(additional, (bool, dict)):
        raise JSONSchemaError(
            f"{_path(path + ('additionalProperties',))} must be boolean or a schema"
        )
    if isinstance(additional, dict):
        _check_schema(additional, path=path + ("additionalProperties",))
    if "items" in schema:
        _check_schema(schema["items"], path=path + ("items",))
    any_of = schema.get("anyOf")
    if any_of is not None:
        if not isinstance(any_of, list) or not any_of:
            raise JSONSchemaError(f"{_path(path + ('anyOf',))} must be a non-empty array")
        for index, child in enumerate(any_of):
            _check_schema(child, path=path + ("anyOf", index))


def _validate_instance(
    instance: Any,
    schema: dict[str, Any],
    *,
    path: tuple[str | int, ...],
) -> None:
    any_of = schema.get("anyOf")
    if any_of is not None:
        failures: list[str] = []
        for child in any_of:
            try:
                _validate_instance(instance, child, path=path)
            except JSONSchemaValidationError as exc:
                failures.append(exc.message)
            else:
                break
        else:
            detail = "; ".join(failures[:3])
            raise JSONSchemaValidationError(
                "does not satisfy anyOf" + (f" ({detail})" if detail else ""), path
            )

    expected_type = schema.get("type")
    if expected_type is not None and not _is_json_type(instance, expected_type):
        raise JSONSchemaValidationError(f"must be of type {expected_type}", path)

    enum = schema.get("enum")
    if enum is not None and all(not _json_equal(instance, allowed) for allowed in enum):
        raise JSONSchemaValidationError(f"must be one of {enum!r}", path)

    if _is_json_type(instance, "number"):
        if "minimum" in schema and instance < schema["minimum"]:
            raise JSONSchemaValidationError(f"must be >= {schema['minimum']}", path)
        if "maximum" in schema and instance > schema["maximum"]:
            raise JSONSchemaValidationError(f"must be <= {schema['maximum']}", path)

    if isinstance(instance, dict):
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in instance:
                raise JSONSchemaValidationError(f"missing required property {key!r}", path)
        for key, value in instance.items():
            if key in properties:
                _validate_instance(value, properties[key], path=path + (key,))
                continue
            additional = schema.get("additionalProperties", True)
            if additional is False:
                raise JSONSchemaValidationError(
                    f"additional property {key!r} is not allowed", path + (key,)
                )
            if isinstance(additional, dict):
                _validate_instance(value, additional, path=path + (key,))

    if isinstance(instance, list) and "items" in schema:
        for index, value in enumerate(instance):
            _validate_instance(value, schema["items"], path=path + (index,))


def _is_json_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return False


def _json_equal(left: Any, right: Any) -> bool:
    try:
        return _canonical_json(left) == _canonical_json(right)
    except (TypeError, ValueError):
        return False


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _path(parts: tuple[str | int, ...]) -> str:
    result = ""
    for part in parts:
        if isinstance(part, int):
            result += f"[{part}]"
        else:
            result += ("." if result else "") + part
    return result or "output"


_COMMON_PROPERTIES: dict[str, Any] = {
    "input_path": {
        "type": "string",
        "description": "Path to a JSONL file containing one input item per line.",
    },
    "output_path": {
        "type": "string",
        "description": "Optional destination JSONL path for ordered item results.",
    },
    "prompt": {
        "type": "string",
        "description": "Instruction applied independently to every input item.",
    },
    "output_schema": {
        "type": "object",
        "description": "JSON Schema that every successful output must satisfy.",
    },
    "concurrency": {
        "type": "integer",
        "minimum": 1,
        "maximum": 256,
        "default": 16,
    },
    "max_retries": {
        "type": "integer",
        "minimum": 0,
        "maximum": 20,
        "default": 2,
    },
    "batch_id": {
        "type": "string",
        "description": "Existing batch ID to resume instead of creating a new batch.",
    },
}


LLM_MAP_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "llm_map",
        "description": (
            "Apply a stateless LLM call to every JSONL item. The engine handles "
            "persistent claims, bounded concurrency, schema validation, and retries."
        ),
        "parameters": {
            "type": "object",
            "properties": deepcopy(_COMMON_PROPERTIES),
            "required": [],
            "anyOf": [
                {"required": ["batch_id"]},
                {"required": ["input_path", "prompt", "output_schema"]},
            ],
            "additionalProperties": False,
        },
    },
}


_agentic_properties = deepcopy(_COMMON_PROPERTIES)
_agentic_properties["read_only"] = {
    "type": "boolean",
    "description": "Capability boundary enforced for every spawned sub-agent.",
}
AGENTIC_MAP_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "agentic_map",
        "description": (
            "Apply an isolated tool-capable sub-agent to every JSONL item. The "
            "engine handles persistent claims, schema validation, and retries."
        ),
        "parameters": {
            "type": "object",
            "properties": _agentic_properties,
            "required": [],
            "anyOf": [
                {"required": ["batch_id"]},
                {"required": ["input_path", "prompt", "output_schema", "read_only"]},
            ],
            "additionalProperties": False,
        },
    },
}


OPERATOR_TOOL_SCHEMAS = (LLM_MAP_SCHEMA, AGENTIC_MAP_SCHEMA)
