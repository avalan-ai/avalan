from ..types import LooseJsonValue
from .definition import (
    FrozenMetadata,
    ObservabilitySinkType,
    TaskDefinition,
)

from collections.abc import Mapping
from enum import StrEnum
from hashlib import sha256
from json import JSONDecodeError, dumps, loads
from math import isfinite
from pathlib import Path
from typing import TypeAlias, cast

CanonicalValue: TypeAlias = LooseJsonValue

_CANONICAL_JSON_SEPARATORS = (",", ":")
_SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "credential",
    "database_url",
    "dsn",
    "passwd",
    "password",
    "private_key",
    "secret",
    "token",
)
_DSN_SCHEMES = (
    "clickhouse://",
    "mariadb://",
    "mongodb://",
    "mysql://",
    "postgres://",
    "postgresql://",
    "redis://",
)
_ORDER_INSENSITIVE_SCHEMA_ARRAYS = frozenset(("enum", "required"))


class TaskCanonicalizationError(ValueError):
    pass


def canonical_definition(
    definition: TaskDefinition,
    *,
    schema_base_path: str | Path | None = None,
) -> dict[str, object]:
    assert isinstance(definition, TaskDefinition)
    return {
        "artifact": {
            "encrypt": definition.artifact.encrypt,
            "max_bytes": definition.artifact.max_bytes,
            "max_count": definition.artifact.max_count,
            "retention_days": definition.artifact.retention_days,
            "storage": _normalize_definition_string(
                definition.artifact.storage
            ),
            "store_bytes": definition.artifact.store_bytes,
        },
        "execution": {
            "ref": _normalize_ref(definition.execution.ref),
            "type": definition.execution.type.value,
            "variables": _normalize_definition_mapping(
                definition.execution.variables
            ),
        },
        "input": {
            "description": definition.input.description,
            "file_conversions": list(definition.input.file_conversions),
            "mime_types": list(definition.input.mime_types),
            "required": definition.input.required,
            "schema": _canonical_schema(
                definition.input.schema,
                definition.input.schema_ref,
                schema_base_path,
                "input.schema_ref",
            ),
            "type": definition.input.type.value,
        },
        "limits": {
            "artifact_bytes": definition.limits.artifact_bytes,
            "artifact_count": definition.limits.artifact_count,
            "file_bytes": definition.limits.file_bytes,
            "file_count": definition.limits.file_count,
            "input_bytes": definition.limits.input_bytes,
            "output_bytes": definition.limits.output_bytes,
            "total_tokens": definition.limits.total_tokens,
        },
        "observability": {
            "capture_events": definition.observability.capture_events,
            "metrics": definition.observability.metrics,
            "sinks": _canonical_sinks(definition.observability.sinks),
            "trace": definition.observability.trace,
        },
        "output": {
            "description": definition.output.description,
            "schema": _canonical_schema(
                definition.output.schema,
                definition.output.schema_ref,
                schema_base_path,
                "output.schema_ref",
            ),
            "type": definition.output.type.value,
        },
        "privacy": {
            "errors": definition.privacy.errors.value,
            "events": definition.privacy.events.value,
            "file_bytes": definition.privacy.file_bytes.value,
            "files": definition.privacy.files.value,
            "input": definition.privacy.input.value,
            "output": definition.privacy.output.value,
            "prompt": definition.privacy.prompt.value,
            "raw_retention_days": definition.privacy.raw_retention_days,
            "token_text": definition.privacy.token_text.value,
            "tool_arguments": definition.privacy.tool_arguments.value,
            "tool_results": definition.privacy.tool_results.value,
        },
        "retry": {
            "backoff": definition.retry.backoff.value,
            "jitter": definition.retry.jitter,
            "max_attempts": definition.retry.max_attempts,
            "max_delay_seconds": definition.retry.max_delay_seconds,
        },
        "run": {
            "concurrency": definition.run.concurrency,
            "idempotency": definition.run.idempotency.value,
            "idempotency_key_path": definition.run.idempotency_key_path,
            "mode": definition.run.mode.value,
            "priority": definition.run.priority,
            "queue": definition.run.queue,
            "timeout_seconds": definition.run.timeout_seconds,
        },
        "task": {
            "annotations": _normalize_definition_mapping(
                definition.task.annotations
            ),
            "description": definition.task.description,
            "labels": list(definition.task.labels),
            "name": definition.task.name,
            "version": definition.task.version,
        },
    }


def canonical_json(
    definition: TaskDefinition,
    *,
    schema_base_path: str | Path | None = None,
) -> str:
    return dumps(
        canonical_definition(definition, schema_base_path=schema_base_path),
        allow_nan=False,
        ensure_ascii=False,
        separators=_CANONICAL_JSON_SEPARATORS,
        sort_keys=True,
    )


def spec_hash(
    definition: TaskDefinition,
    *,
    schema_base_path: str | Path | None = None,
) -> str:
    canonical = canonical_json(definition, schema_base_path=schema_base_path)
    return sha256(canonical.encode("utf-8")).hexdigest()


def _canonical_schema(
    schema: FrozenMetadata | None,
    schema_ref: str | None,
    schema_base_path: str | Path | None,
    path: str,
) -> object:
    if schema is not None:
        return _normalize_schema_value(schema)
    if schema_ref is None:
        return None
    return _load_schema_ref(schema_ref, schema_base_path, path)


def _load_schema_ref(
    schema_ref: str,
    schema_base_path: str | Path | None,
    path: str,
) -> object:
    if "://" in schema_ref:
        raise TaskCanonicalizationError(
            f"{path} remote schema references are not supported"
        )
    source_path = _schema_source_path(schema_ref, schema_base_path)
    try:
        source = source_path.read_text(encoding="utf-8")
    except OSError as error:
        raise TaskCanonicalizationError(f"{path} could not be read") from error
    try:
        raw_schema = loads(source)
    except JSONDecodeError as error:
        raise TaskCanonicalizationError(
            f"{path} must point to a JSON schema file"
        ) from error
    if not isinstance(raw_schema, Mapping):
        raise TaskCanonicalizationError(
            f"{path} must point to a JSON object schema"
        )
    return _normalize_schema_value(raw_schema)


def _schema_source_path(
    schema_ref: str, schema_base_path: str | Path | None
) -> Path:
    path = Path(schema_ref)
    if path.is_absolute():
        return path
    if schema_base_path is None:
        return path
    base_path = Path(schema_base_path)
    if base_path.suffix:
        base_path = base_path.parent
    return base_path / path


def _normalize_schema_value(
    value: object, *, parent_key: str | None = None
) -> CanonicalValue:
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TaskCanonicalizationError("schema keys must be strings")
            normalized[key] = _normalize_schema_value(item, parent_key=key)
        return normalized
    if isinstance(value, list | tuple):
        normalized_list = [_normalize_schema_value(item) for item in value]
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
    return _normalize_scalar(value, context="schema")


def _normalize_definition_mapping(
    value: Mapping[str, object],
) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key, item in value.items():
        if _sensitive_key(key):
            normalized[key] = "<redacted>"
        else:
            normalized[key] = _normalize_definition_value(item)
    return normalized


def _normalize_definition_value(value: object) -> CanonicalValue:
    if isinstance(value, Mapping):
        return _normalize_definition_mapping(cast(Mapping[str, object], value))
    if isinstance(value, list | tuple):
        return [_normalize_definition_value(item) for item in value]
    if isinstance(value, StrEnum):
        return _normalize_scalar(value, context="definition")
    if isinstance(value, str):
        return _normalize_definition_string(value)
    return _normalize_scalar(value, context="definition")


def _normalize_definition_string(value: str | None) -> str | None:
    if value is None:
        return None
    if _looks_like_dsn(value):
        return "<dsn>"
    if Path(value).is_absolute():
        return "<absolute-path>"
    return value


def _normalize_ref(ref: str) -> str:
    if ref.startswith("ai://env:") and "@" in ref:
        return "ai://env:@" + ref.split("@", maxsplit=1)[1]
    normalized = _normalize_definition_string(ref)
    assert normalized is not None
    return normalized


def _normalize_scalar(value: object, *, context: str) -> CanonicalValue:
    if isinstance(value, StrEnum):
        return value.value
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float) and isfinite(value):
        return value
    raise TaskCanonicalizationError(f"{context} contains a non-JSON value")


def _sensitive_key(key: str) -> bool:
    normalized = key.replace("-", "_").lower()
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def _looks_like_dsn(value: str) -> bool:
    normalized = value.lower()
    return normalized.startswith(_DSN_SCHEMES)


def _canonical_sinks(
    sinks: tuple[ObservabilitySinkType, ...],
) -> list[str]:
    return [sink.value for sink in sinks]
