from ..event import Event
from ..types import assert_non_empty_string as _assert_non_empty_string
from .artifact import ArtifactStore, TaskArtifactRef
from .converters import FileConverter
from .definition import TaskDefinition
from .input import TaskProviderReference
from .store import (
    TaskExecutionContext,
    TaskStore,
    freeze_snapshot_metadata,
    freeze_snapshot_value,
)

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

_REDACTED_METADATA_SUMMARY = MappingProxyType({"privacy": "<redacted>"})
_SAFE_LOGICAL_PATH_PREFIXES = ("artifact:", "inline:", "provider:")

TaskCancellationChecker = Callable[[], Awaitable[None]]
TaskEventListener = Callable[[Event], Awaitable[None] | None]
TaskUsageObserver = Callable[[object], Awaitable[None] | None]


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskInputFile:
    logical_path: str
    artifact_ref: TaskArtifactRef | None = None
    provider_reference: TaskProviderReference | None = None
    media_type: str | None = None
    size_bytes: int | None = None
    metadata: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.logical_path, "logical_path")
        if self.artifact_ref is not None:
            assert isinstance(self.artifact_ref, TaskArtifactRef)
        if self.provider_reference is not None:
            assert isinstance(
                self.provider_reference,
                TaskProviderReference,
            )
        if self.media_type is not None:
            _assert_non_empty_string(self.media_type, "media_type")
        if self.size_bytes is not None:
            assert isinstance(self.size_bytes, int)
            assert not isinstance(self.size_bytes, bool)
            assert self.size_bytes >= 0
        object.__setattr__(
            self,
            "metadata",
            freeze_snapshot_metadata(self.metadata),
        )

    def summary(self) -> Mapping[str, object]:
        value: dict[str, object] = {
            "logical_path": _safe_logical_path(self.logical_path)
        }
        if self.artifact_ref is not None:
            value["artifact"] = self.artifact_ref.summary()
        if self.provider_reference is not None:
            value["provider_reference"] = self.provider_reference.summary()
        if self.media_type is not None:
            value["media_type"] = self.media_type
        if self.size_bytes is not None:
            value["size_bytes"] = self.size_bytes
        if self.metadata:
            value["metadata"] = _REDACTED_METADATA_SUMMARY
        return MappingProxyType(value)


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskTargetContext:
    definition: TaskDefinition
    execution: TaskExecutionContext
    input_value: object = None
    files: tuple[TaskInputFile, ...] = ()
    metadata: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )
    cancellation_checker: TaskCancellationChecker | None = None
    event_listener: TaskEventListener | None = None
    usage_observer: TaskUsageObserver | None = None
    artifact_store: ArtifactStore | None = None
    task_store: TaskStore | None = None
    file_converters: Mapping[str, FileConverter] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        assert isinstance(self.definition, TaskDefinition)
        assert isinstance(self.execution, TaskExecutionContext)
        assert isinstance(self.files, tuple)
        for file in self.files:
            assert isinstance(file, TaskInputFile)
        object.__setattr__(
            self,
            "metadata",
            freeze_snapshot_metadata(self.metadata),
        )
        if self.cancellation_checker is not None:
            assert callable(self.cancellation_checker)
        if self.event_listener is not None:
            assert callable(self.event_listener)
        if self.usage_observer is not None:
            assert callable(self.usage_observer)
        if self.artifact_store is not None:
            assert callable(getattr(self.artifact_store, "open_stream", None))
        if self.task_store is not None:
            assert callable(getattr(self.task_store, "append_artifact", None))
        assert isinstance(self.file_converters, Mapping)
        object.__setattr__(
            self,
            "file_converters",
            MappingProxyType(dict(self.file_converters)),
        )

    async def check_cancelled(self) -> None:
        if self.cancellation_checker is not None:
            await self.cancellation_checker()

    async def observe_usage(self, response: object) -> None:
        if self.usage_observer is None:
            return
        result = self.usage_observer(response)
        if result is not None:
            await result


def safe_target_metadata(
    value: Mapping[str, object] | None,
) -> Mapping[str, object]:
    return freeze_snapshot_metadata(value)


def safe_target_value(value: object) -> object:
    return freeze_snapshot_value(value)


def _safe_logical_path(value: str) -> str | Mapping[str, str]:
    if value.startswith(_SAFE_LOGICAL_PATH_PREFIXES):
        return value
    return _REDACTED_METADATA_SUMMARY
