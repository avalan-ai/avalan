from ..filesystem import read_text
from ..types import LooseJsonValue
from .container import task_container_canonical_value
from .definition import (
    FrozenMetadata,
    ObservabilitySinkType,
    TaskDefinition,
    TaskTargetType,
)
from .schema import (
    TaskSchemaResolutionError,
    normalize_schema_value,
    resolve_schema_ref,
    task_definition_schema_base_path,
)
from .skills import task_definition_with_skills_identity

from collections.abc import Mapping
from enum import StrEnum
from hashlib import sha256
from json import dumps
from math import isfinite
from pathlib import Path
from tomllib import TOMLDecodeError
from tomllib import loads as toml_loads
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


class TaskCanonicalizationError(ValueError):
    pass


async def canonical_definition(
    definition: TaskDefinition,
    *,
    schema_base_path: str | Path | None = None,
) -> dict[str, object]:
    assert isinstance(definition, TaskDefinition)
    schema_base_path = task_definition_schema_base_path(
        definition,
        schema_base_path=schema_base_path,
    )
    definition = await task_definition_with_skills_identity(
        definition,
        schema_base_path=schema_base_path,
    )
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
        "container": task_container_canonical_value(definition),
        "execution": {
            "provider_instructions_sha256": (
                await _provider_instructions_digest(
                    definition,
                    schema_base_path,
                )
            ),
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
            "schema": await _canonical_schema(
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
            "schema": await _canonical_schema(
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
        "skills": (
            _normalize_definition_mapping(definition.skills_identity)
            if definition.skills_identity is not None
            else None
        ),
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


async def canonical_json(
    definition: TaskDefinition,
    *,
    schema_base_path: str | Path | None = None,
) -> str:
    return dumps(
        await canonical_definition(
            definition,
            schema_base_path=schema_base_path,
        ),
        allow_nan=False,
        ensure_ascii=False,
        separators=_CANONICAL_JSON_SEPARATORS,
        sort_keys=True,
    )


async def spec_hash(
    definition: TaskDefinition,
    *,
    schema_base_path: str | Path | None = None,
) -> str:
    canonical = await canonical_json(
        definition,
        schema_base_path=schema_base_path,
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


async def _provider_instructions_digest(
    definition: TaskDefinition,
    schema_base_path: str | Path | None,
) -> str | None:
    if definition.execution.type != TaskTargetType.AGENT:
        return None
    if schema_base_path is None:
        return None
    ref = Path(definition.execution.ref)
    if ref.is_absolute():
        return None

    base = Path(schema_base_path)
    base_dir = base.parent if base.suffix else base
    try:
        root = base_dir.resolve(strict=False)
        source_path = (root / ref).resolve(strict=False)
        source_path.relative_to(root)
        source = await read_text(source_path)
        raw = toml_loads(source)
    except (OSError, TOMLDecodeError, ValueError):
        return None

    agent = raw.get("agent")
    if not isinstance(agent, Mapping):
        return None
    instructions = agent.get("instructions")
    if not isinstance(instructions, str):
        return None
    return sha256(instructions.encode("utf-8")).hexdigest()


async def _canonical_schema(
    schema: FrozenMetadata | None,
    schema_ref: str | None,
    schema_base_path: str | Path | None,
    path: str,
) -> object:
    if schema is not None:
        return _normalize_schema_value(schema)
    if schema_ref is None:
        return None
    return await _load_schema_ref(schema_ref, schema_base_path, path)


async def _load_schema_ref(
    schema_ref: str,
    schema_base_path: str | Path | None,
    path: str,
) -> object:
    try:
        return (
            await resolve_schema_ref(
                schema_ref,
                schema_base_path=schema_base_path,
                path=path,
            )
        ).schema
    except TaskSchemaResolutionError as error:
        raise TaskCanonicalizationError(str(error)) from error


def _normalize_schema_value(
    value: object, *, parent_key: str | None = None
) -> CanonicalValue:
    try:
        return normalize_schema_value(value, parent_key=parent_key)
    except TaskSchemaResolutionError as error:
        raise TaskCanonicalizationError(str(error)) from error


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
