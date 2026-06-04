from ..types import LooseJsonValue
from .definition import (
    FrozenMetadata,
    TaskDefinition,
    TaskInputContract,
    TaskOutputContract,
)

from collections.abc import Mapping
from dataclasses import dataclass, replace
from hashlib import sha256
from json import JSONDecodeError, dumps, loads
from math import isfinite
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import cast
from urllib.parse import urlsplit

CanonicalSchemaValue = LooseJsonValue

_CANONICAL_JSON_SEPARATORS = (",", ":")
_ORDER_INSENSITIVE_SCHEMA_ARRAYS = frozenset(("enum", "required"))


class TaskSchemaResolutionError(ValueError):
    path: str

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        super().__init__(f"{path} {message}")


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolvedTaskSchema:
    schema: FrozenMetadata
    digest: str

    def __post_init__(self) -> None:
        assert isinstance(self.schema, Mapping)
        assert isinstance(self.digest, str) and self.digest


def resolve_task_definition_schemas(
    definition: TaskDefinition,
    *,
    schema_base_path: str | Path | None,
) -> TaskDefinition:
    assert isinstance(definition, TaskDefinition)
    input_contract = definition.input
    output_contract = definition.output
    if input_contract.schema_ref is not None:
        input_contract = _resolved_input_contract(
            input_contract,
            schema_base_path=schema_base_path,
        )
    if output_contract.schema_ref is not None:
        output_contract = _resolved_output_contract(
            output_contract,
            schema_base_path=schema_base_path,
        )
    if (
        input_contract is definition.input
        and output_contract is definition.output
    ):
        return definition
    return replace(
        definition,
        input=input_contract,
        output=output_contract,
    )


def resolve_schema_ref(
    schema_ref: str,
    *,
    schema_base_path: str | Path | None,
    path: str,
) -> ResolvedTaskSchema:
    assert isinstance(path, str) and path.strip()
    ref = _portable_schema_ref(schema_ref, path)
    base_dir = _schema_base_dir(schema_base_path, path)
    source_path = _schema_source_path(ref, base_dir, path)
    try:
        source = source_path.read_text(encoding="utf-8")
    except OSError as error:
        raise TaskSchemaResolutionError(
            path,
            "could not be read",
        ) from error
    try:
        raw_schema = loads(source)
    except JSONDecodeError as error:
        raise TaskSchemaResolutionError(
            path,
            "must point to a JSON schema file",
        ) from error
    if not isinstance(raw_schema, Mapping):
        raise TaskSchemaResolutionError(
            path,
            "must point to a JSON object schema",
        )
    _reject_external_schema_refs(raw_schema, path)
    schema = cast(FrozenMetadata, normalize_schema_value(raw_schema))
    canonical = canonical_schema_json(schema)
    return ResolvedTaskSchema(
        schema=schema,
        digest=sha256(canonical.encode("utf-8")).hexdigest(),
    )


def canonical_schema_json(schema: Mapping[str, object]) -> str:
    assert isinstance(schema, Mapping)
    return dumps(
        normalize_schema_value(schema),
        allow_nan=False,
        ensure_ascii=False,
        separators=_CANONICAL_JSON_SEPARATORS,
        sort_keys=True,
    )


def normalize_schema_value(
    value: object,
    *,
    parent_key: str | None = None,
) -> CanonicalSchemaValue:
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TaskSchemaResolutionError(
                    "schema",
                    "keys must be strings",
                )
            normalized[key] = normalize_schema_value(
                item,
                parent_key=key,
            )
        return normalized
    if isinstance(value, list | tuple):
        normalized_list = [normalize_schema_value(item) for item in value]
        if parent_key in _ORDER_INSENSITIVE_SCHEMA_ARRAYS:
            return cast(
                list[object],
                sorted(
                    normalized_list,
                    key=lambda item: dumps(
                        item,
                        allow_nan=False,
                        separators=_CANONICAL_JSON_SEPARATORS,
                        sort_keys=True,
                    ),
                ),
            )
        return cast(list[object], normalized_list)
    return _normalize_scalar(value)


def _resolved_input_contract(
    contract: TaskInputContract,
    *,
    schema_base_path: str | Path | None,
) -> TaskInputContract:
    assert contract.schema_ref is not None
    resolved = resolve_schema_ref(
        contract.schema_ref,
        schema_base_path=schema_base_path,
        path="input.schema_ref",
    )
    return replace(contract, schema=resolved.schema, schema_ref=None)


def _resolved_output_contract(
    contract: TaskOutputContract,
    *,
    schema_base_path: str | Path | None,
) -> TaskOutputContract:
    assert contract.schema_ref is not None
    resolved = resolve_schema_ref(
        contract.schema_ref,
        schema_base_path=schema_base_path,
        path="output.schema_ref",
    )
    return replace(contract, schema=resolved.schema, schema_ref=None)


def _portable_schema_ref(schema_ref: object, path: str) -> str:
    if not isinstance(schema_ref, str) or not schema_ref.strip():
        raise TaskSchemaResolutionError(path, "is invalid")
    ref = schema_ref.strip()
    parsed = urlsplit(ref)
    if parsed.scheme or "://" in ref:
        raise TaskSchemaResolutionError(
            path,
            "remote schema references are not supported",
        )
    if "\\" in ref:
        raise TaskSchemaResolutionError(path, "must use portable separators")
    if "#" in ref:
        raise TaskSchemaResolutionError(path, "must point to a local file")
    if Path(ref).is_absolute() or PureWindowsPath(ref).is_absolute():
        raise TaskSchemaResolutionError(
            path,
            "absolute schema references are not supported",
        )
    pure_ref = PurePosixPath(ref)
    if not pure_ref.parts or any(part == ".." for part in pure_ref.parts):
        raise TaskSchemaResolutionError(path, "cannot escape its base path")
    return ref


def _schema_base_dir(
    schema_base_path: str | Path | None,
    path: str,
) -> Path:
    if schema_base_path is None:
        raise TaskSchemaResolutionError(
            path,
            "requires a definition base path",
        )
    base_path = Path(schema_base_path)
    if base_path.exists() and base_path.is_dir():
        base_dir = base_path
    elif base_path.suffix:
        base_dir = base_path.parent
    else:
        base_dir = base_path
    try:
        return base_dir.resolve(strict=True)
    except OSError as error:
        raise TaskSchemaResolutionError(
            path,
            "base path could not be resolved",
        ) from error


def _schema_source_path(ref: str, base_dir: Path, path: str) -> Path:
    try:
        source_path = (base_dir / ref).resolve(strict=True)
    except OSError as error:
        raise TaskSchemaResolutionError(
            path,
            "could not be read",
        ) from error
    if not source_path.is_relative_to(base_dir):
        raise TaskSchemaResolutionError(path, "cannot escape its base path")
    return source_path


def _reject_external_schema_refs(schema: object, path: str) -> None:
    if isinstance(schema, Mapping):
        for key, value in schema.items():
            if key == "$ref":
                if not isinstance(value, str) or not value.startswith("#"):
                    raise TaskSchemaResolutionError(
                        path,
                        "cannot contain external $ref targets",
                    )
            else:
                _reject_external_schema_refs(value, path)
    elif isinstance(schema, list | tuple):
        for item in schema:
            _reject_external_schema_refs(item, path)


def _normalize_scalar(value: object) -> CanonicalSchemaValue:
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float) and isfinite(value):
        return value
    raise TaskSchemaResolutionError("schema", "contains a non-JSON value")
