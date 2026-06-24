from ..container import ContainerEffectiveSettings
from ..types import (
    assert_non_empty_string as _assert_non_empty_string,
)
from ..types import (
    assert_optional_positive_int as _assert_positive_int,
)

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import TypeAlias

FrozenMetadata: TypeAlias = Mapping[str, object]


class TaskInputType(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    OBJECT = "object"
    ARRAY = "array"
    FILE = "file"
    FILE_ARRAY = "file[]"


class TaskOutputType(StrEnum):
    TEXT = "text"
    JSON = "json"
    OBJECT = "object"
    ARRAY = "array"
    FILE = "file"
    FILE_ARRAY = "file[]"
    ARTIFACT_ARRAY = "artifact[]"


class TaskTargetType(StrEnum):
    AGENT = "agent"
    FLOW = "flow"
    TASK = "task"
    MODEL = "model"
    CALLABLE = "callable"
    TOOL = "tool"


class RunMode(StrEnum):
    DIRECT = "direct"
    QUEUE = "queue"


class IdempotencyMode(StrEnum):
    NONE = "none"
    INPUT_HASH = "input_hash"
    INPUT_AND_FILES_HASH = "input_and_files_hash"
    CUSTOM = "custom"


class RetryBackoff(StrEnum):
    NONE = "none"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


class PrivacyAction(StrEnum):
    DROP = "drop"
    HASH = "hash"
    REDACT = "redact"
    STORE = "store"
    ENCRYPT = "encrypt"


class ObservabilitySinkType(StrEnum):
    PGSQL = "pgsql"
    PROMETHEUS = "prometheus"
    OTEL = "otel"
    NOOP = "noop"


def _empty_mapping() -> FrozenMetadata:
    return MappingProxyType({})


def _freeze_mapping(value: FrozenMetadata) -> FrozenMetadata:
    assert isinstance(value, Mapping), "metadata must be a mapping"
    frozen: dict[str, object] = {}
    for key, item in value.items():
        assert isinstance(key, str), "metadata keys must be strings"
        frozen[key] = _freeze_value(item)
    return MappingProxyType(frozen)


def _freeze_value(value: object) -> object:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, list | tuple):
        return tuple(_freeze_value(item) for item in value)
    return value


def _assert_enum(
    value: StrEnum, enum_type: type[StrEnum], field_name: str
) -> None:
    assert isinstance(
        value, enum_type
    ), f"{field_name} must be a {enum_type.__name__}"


def _assert_string_tuple(values: tuple[str, ...], field_name: str) -> None:
    assert isinstance(values, tuple), f"{field_name} must be a tuple"
    for value in values:
        _assert_non_empty_string(value, field_name)


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskMetadata:
    name: str
    version: str
    description: str | None = None
    labels: tuple[str, ...] = ()
    annotations: FrozenMetadata = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.name, "name")
        _assert_non_empty_string(self.version, "version")
        if self.description is not None:
            _assert_non_empty_string(self.description, "description")
        _assert_string_tuple(self.labels, "labels")
        object.__setattr__(
            self, "annotations", _freeze_mapping(self.annotations)
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskInputContract:
    type: TaskInputType
    schema: FrozenMetadata | None = None
    schema_ref: str | None = None
    description: str | None = None
    required: bool = True
    file_conversions: tuple[str, ...] = ()
    mime_types: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _assert_enum(self.type, TaskInputType, "type")
        assert isinstance(self.required, bool), "required must be a bool"
        if self.description is not None:
            _assert_non_empty_string(self.description, "description")
        if self.schema is not None:
            assert self.type in {
                TaskInputType.OBJECT,
                TaskInputType.ARRAY,
            }, "schema is only supported for structured input contracts"
            object.__setattr__(self, "schema", _freeze_mapping(self.schema))
        if self.schema_ref is not None:
            assert self.type in {
                TaskInputType.OBJECT,
                TaskInputType.ARRAY,
            }, "schema_ref is only supported for structured input contracts"
            assert self.schema is None, "schema and schema_ref are exclusive"
            _assert_non_empty_string(self.schema_ref, "schema_ref")
        _assert_string_tuple(self.file_conversions, "file_conversions")
        _assert_string_tuple(self.mime_types, "mime_types")
        if self.file_conversions or self.mime_types:
            assert self.type in {
                TaskInputType.FILE,
                TaskInputType.FILE_ARRAY,
            }, "file options require a file input contract"

    @classmethod
    def string(cls, *, description: str | None = None) -> "TaskInputContract":
        return cls(type=TaskInputType.STRING, description=description)

    @classmethod
    def integer(cls, *, description: str | None = None) -> "TaskInputContract":
        return cls(type=TaskInputType.INTEGER, description=description)

    @classmethod
    def number(cls, *, description: str | None = None) -> "TaskInputContract":
        return cls(type=TaskInputType.NUMBER, description=description)

    @classmethod
    def boolean(cls, *, description: str | None = None) -> "TaskInputContract":
        return cls(type=TaskInputType.BOOLEAN, description=description)

    @classmethod
    def object(
        cls,
        schema: FrozenMetadata | None = None,
        *,
        schema_ref: str | None = None,
        description: str | None = None,
    ) -> "TaskInputContract":
        return cls(
            type=TaskInputType.OBJECT,
            schema=schema,
            schema_ref=schema_ref,
            description=description,
        )

    @classmethod
    def array(
        cls,
        schema: FrozenMetadata | None = None,
        *,
        schema_ref: str | None = None,
        description: str | None = None,
    ) -> "TaskInputContract":
        return cls(
            type=TaskInputType.ARRAY,
            schema=schema,
            schema_ref=schema_ref,
            description=description,
        )

    @classmethod
    def file(
        cls,
        *,
        conversions: Iterable[str] = (),
        mime_types: Iterable[str] = (),
        description: str | None = None,
    ) -> "TaskInputContract":
        return cls(
            type=TaskInputType.FILE,
            file_conversions=tuple(conversions),
            mime_types=tuple(mime_types),
            description=description,
        )

    @classmethod
    def file_array(
        cls,
        *,
        conversions: Iterable[str] = (),
        mime_types: Iterable[str] = (),
        description: str | None = None,
    ) -> "TaskInputContract":
        return cls(
            type=TaskInputType.FILE_ARRAY,
            file_conversions=tuple(conversions),
            mime_types=tuple(mime_types),
            description=description,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskOutputContract:
    type: TaskOutputType
    schema: FrozenMetadata | None = None
    schema_ref: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        _assert_enum(self.type, TaskOutputType, "type")
        if self.description is not None:
            _assert_non_empty_string(self.description, "description")
        if self.schema is not None:
            assert self.type in {
                TaskOutputType.JSON,
                TaskOutputType.OBJECT,
                TaskOutputType.ARRAY,
            }, "schema is only supported for structured output contracts"
            object.__setattr__(self, "schema", _freeze_mapping(self.schema))
        if self.schema_ref is not None:
            assert self.type in {
                TaskOutputType.JSON,
                TaskOutputType.OBJECT,
                TaskOutputType.ARRAY,
            }, "schema_ref is only supported for structured output contracts"
            assert self.schema is None, "schema and schema_ref are exclusive"
            _assert_non_empty_string(self.schema_ref, "schema_ref")

    @classmethod
    def text(cls, *, description: str | None = None) -> "TaskOutputContract":
        return cls(type=TaskOutputType.TEXT, description=description)

    @classmethod
    def json(
        cls,
        schema: FrozenMetadata | None = None,
        *,
        schema_ref: str | None = None,
        description: str | None = None,
    ) -> "TaskOutputContract":
        return cls(
            type=TaskOutputType.JSON,
            schema=schema,
            schema_ref=schema_ref,
            description=description,
        )

    @classmethod
    def object(
        cls,
        schema: FrozenMetadata | None = None,
        *,
        schema_ref: str | None = None,
        description: str | None = None,
    ) -> "TaskOutputContract":
        return cls(
            type=TaskOutputType.OBJECT,
            schema=schema,
            schema_ref=schema_ref,
            description=description,
        )

    @classmethod
    def array(
        cls,
        schema: FrozenMetadata | None = None,
        *,
        schema_ref: str | None = None,
        description: str | None = None,
    ) -> "TaskOutputContract":
        return cls(
            type=TaskOutputType.ARRAY,
            schema=schema,
            schema_ref=schema_ref,
            description=description,
        )

    @classmethod
    def file(cls, *, description: str | None = None) -> "TaskOutputContract":
        return cls(type=TaskOutputType.FILE, description=description)

    @classmethod
    def file_array(
        cls, *, description: str | None = None
    ) -> "TaskOutputContract":
        return cls(type=TaskOutputType.FILE_ARRAY, description=description)

    @classmethod
    def artifact_array(
        cls, *, description: str | None = None
    ) -> "TaskOutputContract":
        return cls(
            type=TaskOutputType.ARTIFACT_ARRAY,
            description=description,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskExecutionTarget:
    type: TaskTargetType
    ref: str
    variables: FrozenMetadata = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        _assert_enum(self.type, TaskTargetType, "type")
        _assert_non_empty_string(self.ref, "ref")
        object.__setattr__(self, "variables", _freeze_mapping(self.variables))

    @classmethod
    def agent(
        cls, ref: str, *, variables: FrozenMetadata | None = None
    ) -> "TaskExecutionTarget":
        return cls(
            type=TaskTargetType.AGENT,
            ref=ref,
            variables=variables or _empty_mapping(),
        )

    @classmethod
    def flow(
        cls, ref: str, *, variables: FrozenMetadata | None = None
    ) -> "TaskExecutionTarget":
        return cls(
            type=TaskTargetType.FLOW,
            ref=ref,
            variables=variables or _empty_mapping(),
        )

    @classmethod
    def task(
        cls, ref: str, *, variables: FrozenMetadata | None = None
    ) -> "TaskExecutionTarget":
        return cls(
            type=TaskTargetType.TASK,
            ref=ref,
            variables=variables or _empty_mapping(),
        )

    @classmethod
    def model(
        cls, ref: str, *, variables: FrozenMetadata | None = None
    ) -> "TaskExecutionTarget":
        return cls(
            type=TaskTargetType.MODEL,
            ref=ref,
            variables=variables or _empty_mapping(),
        )

    @classmethod
    def callable(
        cls, ref: str, *, variables: FrozenMetadata | None = None
    ) -> "TaskExecutionTarget":
        return cls(
            type=TaskTargetType.CALLABLE,
            ref=ref,
            variables=variables or _empty_mapping(),
        )

    @classmethod
    def tool(
        cls, ref: str, *, variables: FrozenMetadata | None = None
    ) -> "TaskExecutionTarget":
        return cls(
            type=TaskTargetType.TOOL,
            ref=ref,
            variables=variables or _empty_mapping(),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskRunPolicy:
    mode: RunMode = RunMode.DIRECT
    timeout_seconds: int = 300
    idempotency: IdempotencyMode = IdempotencyMode.NONE
    queue: str | None = None
    priority: int | None = None
    concurrency: int | None = None
    idempotency_key_path: str | None = None

    def __post_init__(self) -> None:
        _assert_enum(self.mode, RunMode, "mode")
        _assert_positive_int(self.timeout_seconds, "timeout_seconds")
        _assert_enum(self.idempotency, IdempotencyMode, "idempotency")
        if self.mode == RunMode.QUEUE:
            _assert_non_empty_string(self.queue, "queue")
        elif self.queue is not None:
            _assert_non_empty_string(self.queue, "queue")
        if self.priority is not None:
            assert isinstance(self.priority, int)
            assert not isinstance(self.priority, bool)
            assert self.priority >= 0, "priority must be non-negative"
        _assert_positive_int(self.concurrency, "concurrency")
        if self.idempotency == IdempotencyMode.CUSTOM:
            _assert_non_empty_string(
                self.idempotency_key_path, "idempotency_key_path"
            )
        elif self.idempotency_key_path is not None:
            _assert_non_empty_string(
                self.idempotency_key_path, "idempotency_key_path"
            )

    @classmethod
    def direct(
        cls,
        *,
        timeout_seconds: int = 300,
        idempotency: IdempotencyMode = IdempotencyMode.NONE,
    ) -> "TaskRunPolicy":
        return cls(timeout_seconds=timeout_seconds, idempotency=idempotency)

    @classmethod
    def queued(
        cls,
        queue: str,
        *,
        timeout_seconds: int = 300,
        idempotency: IdempotencyMode = IdempotencyMode.INPUT_AND_FILES_HASH,
        priority: int | None = None,
        concurrency: int | None = None,
    ) -> "TaskRunPolicy":
        return cls(
            mode=RunMode.QUEUE,
            timeout_seconds=timeout_seconds,
            idempotency=idempotency,
            queue=queue,
            priority=priority,
            concurrency=concurrency,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskRetryPolicy:
    max_attempts: int = 1
    backoff: RetryBackoff = RetryBackoff.NONE
    max_delay_seconds: int | None = None
    jitter: bool = False

    def __post_init__(self) -> None:
        _assert_positive_int(self.max_attempts, "max_attempts")
        _assert_enum(self.backoff, RetryBackoff, "backoff")
        _assert_positive_int(self.max_delay_seconds, "max_delay_seconds")
        assert isinstance(self.jitter, bool), "jitter must be a bool"


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskPrivacyPolicy:
    input: PrivacyAction = PrivacyAction.HASH
    prompt: PrivacyAction = PrivacyAction.REDACT
    output: PrivacyAction = PrivacyAction.REDACT
    files: PrivacyAction = PrivacyAction.HASH
    file_bytes: PrivacyAction = PrivacyAction.DROP
    token_text: PrivacyAction = PrivacyAction.DROP
    tool_arguments: PrivacyAction = PrivacyAction.REDACT
    tool_results: PrivacyAction = PrivacyAction.REDACT
    events: PrivacyAction = PrivacyAction.REDACT
    errors: PrivacyAction = PrivacyAction.REDACT
    raw_retention_days: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "input",
            "prompt",
            "output",
            "files",
            "file_bytes",
            "token_text",
            "tool_arguments",
            "tool_results",
            "events",
            "errors",
        ):
            _assert_enum(getattr(self, field_name), PrivacyAction, field_name)
        assert isinstance(self.raw_retention_days, int)
        assert not isinstance(self.raw_retention_days, bool)
        assert self.raw_retention_days >= 0

    @classmethod
    def default(cls) -> "TaskPrivacyPolicy":
        return cls()


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskArtifactPolicy:
    retention_days: int | None = None
    store_bytes: bool = False
    storage: str | None = None
    max_count: int | None = None
    max_bytes: int | None = None
    encrypt: bool = True

    def __post_init__(self) -> None:
        _assert_positive_int(self.retention_days, "retention_days")
        assert isinstance(self.store_bytes, bool), "store_bytes must be a bool"
        if self.storage is not None:
            _assert_non_empty_string(self.storage, "storage")
        _assert_positive_int(self.max_count, "max_count")
        _assert_positive_int(self.max_bytes, "max_bytes")
        assert isinstance(self.encrypt, bool), "encrypt must be a bool"

    @classmethod
    def references_only(
        cls,
        *,
        retention_days: int | None = None,
    ) -> "TaskArtifactPolicy":
        return cls(retention_days=retention_days)

    @classmethod
    def raw_storage(
        cls,
        *,
        retention_days: int,
        storage: str | None = None,
        encrypt: bool = True,
    ) -> "TaskArtifactPolicy":
        return cls(
            retention_days=retention_days,
            store_bytes=True,
            storage=storage,
            encrypt=encrypt,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskLimitsPolicy:
    input_bytes: int | None = None
    file_count: int | None = None
    file_bytes: int | None = None
    output_bytes: int | None = None
    artifact_count: int | None = None
    artifact_bytes: int | None = None
    total_tokens: int | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "input_bytes",
            "file_count",
            "file_bytes",
            "output_bytes",
            "artifact_count",
            "artifact_bytes",
            "total_tokens",
        ):
            _assert_positive_int(getattr(self, field_name), field_name)


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskObservabilityPolicy:
    sinks: tuple[ObservabilitySinkType, ...] = (ObservabilitySinkType.PGSQL,)
    metrics: bool = True
    trace: bool = True
    capture_events: bool = True

    def __post_init__(self) -> None:
        assert self.sinks, "sinks must not be empty"
        for sink in self.sinks:
            _assert_enum(sink, ObservabilitySinkType, "sinks")
        if ObservabilitySinkType.NOOP in self.sinks:
            assert self.sinks == (ObservabilitySinkType.NOOP,)
        assert isinstance(self.metrics, bool), "metrics must be a bool"
        assert isinstance(self.trace, bool), "trace must be a bool"
        assert isinstance(
            self.capture_events, bool
        ), "capture_events must be a bool"

    @classmethod
    def noop(cls) -> "TaskObservabilityPolicy":
        return cls(
            sinks=(ObservabilitySinkType.NOOP,),
            metrics=False,
            trace=False,
            capture_events=False,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskContainerExecutionSettings:
    attempt: ContainerEffectiveSettings | None = None
    worker_envelope: ContainerEffectiveSettings | None = None
    readiness_timeout_seconds: int = 30

    def __post_init__(self) -> None:
        if self.attempt is not None:
            assert isinstance(self.attempt, ContainerEffectiveSettings)
        if self.worker_envelope is not None:
            assert isinstance(
                self.worker_envelope,
                ContainerEffectiveSettings,
            )
        assert isinstance(self.readiness_timeout_seconds, int)
        assert not isinstance(self.readiness_timeout_seconds, bool)
        assert (
            self.readiness_timeout_seconds > 0
        ), "readiness_timeout_seconds must be positive"

    @property
    def enabled(self) -> bool:
        return bool(
            (self.attempt is not None and self.attempt.enabled)
            or (
                self.worker_envelope is not None
                and self.worker_envelope.enabled
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "attempt": (
                self.attempt.to_dict() if self.attempt is not None else None
            ),
            "worker_envelope": (
                self.worker_envelope.to_dict()
                if self.worker_envelope is not None
                else None
            ),
            "readiness_timeout_seconds": self.readiness_timeout_seconds,
        }

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
    ) -> "TaskContainerExecutionSettings":
        assert isinstance(raw, Mapping), "container settings must be a mapping"
        for field_name in (
            "attempt",
            "worker_envelope",
            "readiness_timeout_seconds",
        ):
            assert (
                field_name in raw
            ), "container settings are missing a required field"
        attempt = raw["attempt"]
        worker_envelope = raw["worker_envelope"]
        readiness_timeout_seconds = raw["readiness_timeout_seconds"]
        assert isinstance(readiness_timeout_seconds, int)
        assert not isinstance(readiness_timeout_seconds, bool)
        return cls(
            attempt=(
                None
                if attempt is None
                else ContainerEffectiveSettings.from_dict(
                    _mapping_value(attempt, "attempt")
                )
            ),
            worker_envelope=(
                None
                if worker_envelope is None
                else ContainerEffectiveSettings.from_dict(
                    _mapping_value(worker_envelope, "worker_envelope")
                )
            ),
            readiness_timeout_seconds=readiness_timeout_seconds,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskDefinition:
    task: TaskMetadata
    input: TaskInputContract
    output: TaskOutputContract
    execution: TaskExecutionTarget
    definition_base: str | Path | None = None
    run: TaskRunPolicy = field(default_factory=TaskRunPolicy)
    retry: TaskRetryPolicy = field(default_factory=TaskRetryPolicy)
    privacy: TaskPrivacyPolicy = field(default_factory=TaskPrivacyPolicy)
    artifact: TaskArtifactPolicy = field(default_factory=TaskArtifactPolicy)
    limits: TaskLimitsPolicy = field(default_factory=TaskLimitsPolicy)
    observability: TaskObservabilityPolicy = field(
        default_factory=TaskObservabilityPolicy
    )
    container: TaskContainerExecutionSettings = field(
        default_factory=TaskContainerExecutionSettings
    )

    def __post_init__(self) -> None:
        assert isinstance(self.task, TaskMetadata)
        assert isinstance(self.input, TaskInputContract)
        assert isinstance(self.output, TaskOutputContract)
        assert isinstance(self.execution, TaskExecutionTarget)
        if self.definition_base is not None:
            assert isinstance(self.definition_base, str | Path)
            _assert_non_empty_string(
                str(self.definition_base), "definition_base"
            )
        assert isinstance(self.run, TaskRunPolicy)
        assert isinstance(self.retry, TaskRetryPolicy)
        assert isinstance(self.privacy, TaskPrivacyPolicy)
        assert isinstance(self.artifact, TaskArtifactPolicy)
        assert isinstance(self.limits, TaskLimitsPolicy)
        assert isinstance(self.observability, TaskObservabilityPolicy)
        assert isinstance(self.container, TaskContainerExecutionSettings)


def _mapping_value(value: object, field_name: str) -> Mapping[str, object]:
    assert isinstance(value, Mapping), f"{field_name} must be a mapping"
    return value
