from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
from inspect import Parameter, signature
from json import dumps
from types import NoneType, UnionType
from typing import Any, Literal, get_args, get_origin, is_typeddict

_JSON_DEFAULT_MISSING = object()


def _parse_docstring_sections(docstring: str | None) -> dict[str, str]:
    sections: dict[str, str] = {}
    if not docstring:
        return sections

    current_key: str | None = None
    for line in docstring.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.endswith(":") and stripped in {"Args:", "Returns:"}:
            current_key = stripped[:-1].lower()
            continue
        if current_key == "args" and ":" in stripped:
            name, description = stripped.split(":", maxsplit=1)
            sections[f"arg:{name.strip()}"] = description.strip()
        elif current_key == "returns":
            sections["return"] = stripped
    return sections


def _literal_schema(annotation: object) -> dict[str, Any] | None:
    if get_origin(annotation) is not Literal:
        return None

    values = list(get_args(annotation))
    if not values:
        return {"type": "object"}

    value_types = {type(value) for value in values}
    if len(value_types) != 1:
        return {"enum": values}

    base_type = _json_type(next(iter(value_types)))
    return {"type": base_type, "enum": values}


def _json_type(annotation: object) -> str:
    schema = _annotation_schema(annotation)
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        return schema_type
    if isinstance(schema_type, list):
        types = [value for value in schema_type if value != "null"]
        if len(types) == 1:
            return str(types[0])
    return "object"


def _annotation_schema(annotation: object) -> dict[str, Any]:
    literal_schema = _literal_schema(annotation)
    if literal_schema is not None:
        return literal_schema

    enum_schema = _enum_schema(annotation)
    if enum_schema is not None:
        return enum_schema

    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is None or annotation is NoneType:
        return {"type": "null"}
    if annotation is dict:
        return {"type": "object"}
    if annotation in {list, tuple, set, frozenset}:
        return {"type": "array"}
    if annotation in {Any, object}:
        return {"type": "object"}

    if isinstance(annotation, type) and is_typeddict(annotation):
        return _typeddict_schema(annotation)
    if isinstance(annotation, type) and is_dataclass(annotation):
        return _dataclass_schema(annotation)

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin in {UnionType} or str(origin) == "typing.Union":
        return _union_schema(args)
    if origin in {list, set, frozenset, Sequence}:
        return _array_schema(args)
    if origin is tuple:
        return _tuple_schema(args)
    if origin in {dict, Mapping, MutableMapping}:
        return _mapping_schema(args)

    return {"type": "object"}


def _enum_schema(annotation: object) -> dict[str, Any] | None:
    if not isinstance(annotation, type) or not issubclass(annotation, Enum):
        return None

    values = [member.value for member in annotation]
    if not values:
        return {"type": "object"}

    value_types = {type(value) for value in values}
    if len(value_types) == 1:
        return {"type": _json_type(next(iter(value_types))), "enum": values}
    return {"enum": values}


def _union_schema(args: tuple[object, ...]) -> dict[str, Any]:
    if not args:
        return {"type": "object"}

    schemas = [_annotation_schema(arg) for arg in args]
    non_null_schemas = [
        schema for schema in schemas if schema.get("type") != "null"
    ]
    has_null = len(non_null_schemas) < len(schemas)

    if has_null and len(non_null_schemas) == 1:
        schema = dict(non_null_schemas[0])
        schema_type = schema.get("type")
        if isinstance(schema_type, str):
            schema["type"] = [schema_type, "null"]
        else:
            schema = {"anyOf": [schema, {"type": "null"}]}
        return schema

    return {"anyOf": _unique_schemas(schemas)}


def _array_schema(args: tuple[object, ...]) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "array"}
    if args:
        schema["items"] = _annotation_schema(args[0])
    return schema


def _tuple_schema(args: tuple[object, ...]) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "array"}
    if not args:
        return schema
    if len(args) == 2 and args[1] is Ellipsis:
        schema["items"] = _annotation_schema(args[0])
        return schema
    schema["prefixItems"] = [_annotation_schema(arg) for arg in args]
    schema["minItems"] = len(args)
    schema["maxItems"] = len(args)
    return schema


def _mapping_schema(args: tuple[object, ...]) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "object"}
    if len(args) != 2:
        return schema

    value_annotation = args[1]
    if value_annotation not in {Any, object}:
        schema["additionalProperties"] = _annotation_schema(value_annotation)
    return schema


def _typeddict_schema(annotation: type) -> dict[str, Any]:
    properties: dict[str, dict[str, Any]] = {}
    annotations = getattr(annotation, "__annotations__", {})
    for name, field_annotation in annotations.items():
        properties[name] = _annotation_schema(field_annotation)

    return {
        "type": "object",
        "properties": properties,
        "required": sorted(getattr(annotation, "__required_keys__", set())),
        "additionalProperties": False,
    }


def _dataclass_schema(annotation: type) -> dict[str, Any]:
    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []
    annotations = getattr(annotation, "__annotations__", {})
    for field in fields(annotation):
        field_annotation = annotations.get(field.name, field.type)
        properties[field.name] = _annotation_schema(field_annotation)
        if field.default is MISSING and field.default_factory is MISSING:
            required.append(field.name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _json_default(value: object) -> object:
    if isinstance(value, Enum):
        value = value.value
    try:
        dumps(value)
    except TypeError:
        return _JSON_DEFAULT_MISSING
    return value


def _unique_schemas(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for schema in schemas:
        key = dumps(schema, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        unique.append(schema)
    return unique


def get_json_schema(function: Callable[..., Any]) -> dict[str, Any]:
    """Build an OpenAI-compatible function JSON schema from a callable."""
    function_name = getattr(function, "__name__", function.__class__.__name__)
    function_doc = (function.__doc__ or "").strip()
    summary = function_doc.splitlines()[0].strip() if function_doc else ""
    docs = _parse_docstring_sections(function_doc)

    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []
    function_signature = signature(function)
    for name, parameter in function_signature.parameters.items():
        if name == "context":
            continue
        if parameter.kind not in {
            Parameter.POSITIONAL_OR_KEYWORD,
            Parameter.KEYWORD_ONLY,
        }:
            continue

        annotation = (
            parameter.annotation
            if parameter.annotation is not Parameter.empty
            else Any
        )
        properties[name] = _annotation_schema(annotation)
        properties[name]["description"] = docs.get(f"arg:{name}", "")
        if parameter.default is Parameter.empty:
            required.append(name)
        else:
            default = _json_default(parameter.default)
            if default is not _JSON_DEFAULT_MISSING:
                properties[name]["default"] = default

    return_annotation = (
        function_signature.return_annotation
        if function_signature.return_annotation is not Parameter.empty
        else Any
    )
    return_schema = _annotation_schema(return_annotation)
    return_schema["description"] = docs.get("return", "")

    return {
        "type": "function",
        "function": {
            "name": function_name,
            "description": summary,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
            "return": return_schema,
        },
    }
