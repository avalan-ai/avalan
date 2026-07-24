from ..event import Event
from ..interaction.continuation import (
    ContinuationCompletionCommand,
    ContinuationRejectionCommand,
    DurableContinuationResumeState,
)
from ..types import assert_non_empty_string as _assert_non_empty_string
from .artifact import ArtifactStore, TaskArtifactRef
from .converters import FileConverter
from .definition import TaskDefinition
from .input import TaskProviderReference
from .settlement import TaskDurableResumeFailure, TaskDurableResumeSettlement
from .store import (
    TaskExecutionContext,
    TaskStore,
    freeze_snapshot_metadata,
    freeze_snapshot_value,
)

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

_REDACTED_METADATA_SUMMARY = MappingProxyType({"privacy": "<redacted>"})
_SAFE_LOGICAL_PATH_PREFIXES = ("artifact:", "inline:", "provider:")

TaskCancellationChecker = Callable[[], Awaitable[None]]
TaskEventListener = Callable[[Event], Awaitable[None] | None]
TaskUsageObserver = Callable[[object], Awaitable[None] | None]
TaskUsageObservationPredicate = Callable[[object], bool]


class TaskEventListenerRegistration(Protocol):
    """Own one task event listener registration."""

    def close(self) -> None:
        """Remove the registered task event listener exactly once."""
        ...


class TaskDurableResumeHandle(Protocol):
    """Resume one exact durable task continuation."""

    def register_event_listener(
        self,
        listener: TaskEventListener,
    ) -> TaskEventListenerRegistration:
        """Register one task listener before resumed provider dispatch."""
        ...

    async def dispatch(self) -> object:
        """Dispatch the reconstructed continuation."""
        ...

    async def wait_dispatch_settled(
        self,
    ) -> DurableContinuationResumeState:
        """Wait for an owned provider dispatch to settle durably."""
        ...

    async def interrupt_dispatch(self) -> DurableContinuationResumeState:
        """Stop owned work at a durable pre- or post-dispatch boundary."""
        ...

    async def complete_output(self, output: object) -> None:
        """Complete continuation metadata after task success."""
        ...

    def completion_command_for_output(
        self,
        output: object,
    ) -> ContinuationCompletionCommand:
        """Return a fenced command for atomic task settlement."""
        ...

    def completion_command_for_settlement(
        self,
        settlement: TaskDurableResumeSettlement,
    ) -> ContinuationCompletionCommand:
        """Return a fenced command for atomic terminal task settlement."""
        ...

    def completed_completion_command(
        self,
    ) -> ContinuationCompletionCommand:
        """Return the pinned fence for an already completed continuation."""
        ...

    def completion_command_for_suspension(
        self,
        *,
        request_id: str,
        continuation_id: str,
        checkpoint_id: str,
    ) -> ContinuationCompletionCommand:
        """Return a fenced command for an atomic successor suspension."""
        ...

    def rejection_command_for_settlement(
        self,
        failure: TaskDurableResumeFailure,
    ) -> ContinuationRejectionCommand:
        """Return a fenced command for deterministic setup rejection."""
        ...

    async def release(self) -> None:
        """Release a claim before provider dispatch starts."""
        ...

    async def release_if_pre_dispatch(self) -> bool:
        """Release safely only when provider dispatch has not started."""
        ...

    async def close(self) -> None:
        """Close resources owned by this durable resume admission."""
        ...


class TaskUsageObservationTracker:
    def __init__(
        self,
        observer: TaskUsageObserver | None,
        *,
        has_observations: TaskUsageObservationPredicate,
    ) -> None:
        if observer is not None:
            assert callable(observer)
        assert callable(has_observations)
        self._observer = observer
        self._has_observations = has_observations
        self._observed = False

    @property
    def observed(self) -> bool:
        return self._observed

    async def observe(self, response: object) -> None:
        if self._observer is None:
            return
        has_observations = self._has_observations(response)
        previous_observed = self._observed
        result = self._observer(response)
        if result is not None:
            await result
        self._observed = previous_observed or has_observations


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
    durable_resume: TaskDurableResumeHandle | None = None
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
        if self.durable_resume is not None:
            for method in (
                "register_event_listener",
                "dispatch",
                "wait_dispatch_settled",
                "interrupt_dispatch",
                "complete_output",
                "completion_command_for_output",
                "completion_command_for_settlement",
                "completion_command_for_suspension",
                "rejection_command_for_settlement",
                "release",
                "release_if_pre_dispatch",
                "close",
            ):
                assert callable(getattr(self.durable_resume, method, None))
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
