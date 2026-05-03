from collections.abc import Callable
from inspect import Parameter, signature
from typing import Any, get_args, get_origin


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


def _json_type(annotation: object) -> str:
    if annotation is str:
        return "string"
    if annotation is int:
        return "integer"
    if annotation is float:
        return "number"
    if annotation is bool:
        return "boolean"
    if annotation is dict:
        return "object"
    if annotation in {list, tuple, set}:
        return "array"

    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in {list, tuple, set}:
        return "array"
    if origin is dict:
        return "object"
    if origin is None and annotation in {Any, object}:
        return "object"
    if origin is None:
        return "object"

    non_none_args = [arg for arg in args if arg is not type(None)]
    if len(non_none_args) == 1:
        return _json_type(non_none_args[0])
    return "object"


def get_json_schema(function: Callable[..., Any]) -> dict[str, Any]:
    """Build an OpenAI-compatible function JSON schema from a callable."""
    function_name = getattr(function, "__name__", function.__class__.__name__)
    function_doc = (function.__doc__ or "").strip()
    summary = function_doc.splitlines()[0].strip() if function_doc else ""
    docs = _parse_docstring_sections(function_doc)

    properties: dict[str, dict[str, str]] = {}
    required: list[str] = []
    function_signature = signature(function)
    for name, parameter in function_signature.parameters.items():
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
        properties[name] = {
            "type": _json_type(annotation),
            "description": docs.get(f"arg:{name}", ""),
        }
        if parameter.default is Parameter.empty:
            required.append(name)

    return_annotation = (
        function_signature.return_annotation
        if function_signature.return_annotation is not Parameter.empty
        else Any
    )

    return {
        "type": "function",
        "function": {
            "name": function_name,
            "description": summary,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
            "return": {
                "type": _json_type(return_annotation),
                "description": docs.get("return", ""),
            },
        },
    }
