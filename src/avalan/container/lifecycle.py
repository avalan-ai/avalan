from ..types import (
    assert_bool as _assert_bool,
)
from ..types import (
    assert_non_empty_string as _assert_non_empty_string,
)
from .backend import (
    ContainerAsyncBackend,
    ContainerBackendContainer,
    ContainerBackendDiagnostic,
    ContainerBackendDiagnosticCode,
    ContainerBackendError,
    ContainerBackendImageResolution,
    ContainerBackendLifecycleResult,
    ContainerBackendOperation,
    ContainerBackendOperationResult,
    ContainerBackendStats,
    ContainerBackendStream,
    ContainerBackendStreamChunk,
    ContainerBackendWaitResult,
)
from .output import (
    ContainerOutputContract,
    ContainerOutputDecisionType,
    ContainerOutputValidationResult,
)
from .settings import (
    ContainerBuildPolicy,
    ContainerExecutionResult,
    ContainerPullPolicy,
    ContainerResultStatus,
    ContainerRunPlan,
)

from asyncio import (
    CancelledError,
    create_task,
    get_running_loop,
    shield,
    wait_for,
)
from collections.abc import (
    AsyncIterable,
    Awaitable,
    Iterable,
    Mapping,
    Sequence,
)
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import TypeVar, cast, final

EnumValue = TypeVar("EnumValue", bound=StrEnum)

_TRUNCATION_MARKER = b"[container stream truncated]"


class ContainerLifecyclePhase(StrEnum):
    POLICY_NORMALIZATION = "policy_normalization"
    BACKEND_SELECTION = "backend_selection"
    IMAGE_RESOLUTION = "image_resolution"
    IMAGE_PULL = "image_pull"
    IMAGE_BUILD = "image_build"
    CREATE = "create"
    ATTACH = "attach"
    START = "start"
    STREAM = "stream"
    STATS = "stats"
    WAIT = "wait"
    INSPECT = "inspect"
    COPY_OUTPUTS = "copy_outputs"
    STOP = "stop"
    KILL = "kill"
    REMOVE = "remove"
    CLEANUP = "cleanup"
    RESULT = "result"


class ContainerLifecycleEventStatus(StrEnum):
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


_PHASE_DEADLINE_FIELDS: Mapping[ContainerLifecyclePhase, str] = {
    ContainerLifecyclePhase.IMAGE_PULL: "pull_seconds",
    ContainerLifecyclePhase.IMAGE_BUILD: "build_seconds",
    ContainerLifecyclePhase.CREATE: "create_seconds",
    ContainerLifecyclePhase.ATTACH: "start_seconds",
    ContainerLifecyclePhase.START: "start_seconds",
    ContainerLifecyclePhase.STREAM: "execution_seconds",
    ContainerLifecyclePhase.WAIT: "execution_seconds",
    ContainerLifecyclePhase.COPY_OUTPUTS: "copy_seconds",
    ContainerLifecyclePhase.STOP: "cleanup_seconds",
    ContainerLifecyclePhase.KILL: "cleanup_seconds",
    ContainerLifecyclePhase.REMOVE: "cleanup_seconds",
    ContainerLifecyclePhase.CLEANUP: "cleanup_seconds",
}

_PHASE_OPERATIONS: Mapping[
    ContainerLifecyclePhase,
    ContainerBackendOperation,
] = {
    ContainerLifecyclePhase.IMAGE_RESOLUTION: (
        ContainerBackendOperation.IMAGE_RESOLUTION
    ),
    ContainerLifecyclePhase.IMAGE_PULL: ContainerBackendOperation.IMAGE_PULL,
    ContainerLifecyclePhase.IMAGE_BUILD: ContainerBackendOperation.IMAGE_BUILD,
    ContainerLifecyclePhase.CREATE: ContainerBackendOperation.CREATE,
    ContainerLifecyclePhase.ATTACH: ContainerBackendOperation.ATTACH,
    ContainerLifecyclePhase.START: ContainerBackendOperation.START,
    ContainerLifecyclePhase.STREAM: ContainerBackendOperation.STREAM,
    ContainerLifecyclePhase.STATS: ContainerBackendOperation.STATS,
    ContainerLifecyclePhase.WAIT: ContainerBackendOperation.WAIT,
    ContainerLifecyclePhase.INSPECT: ContainerBackendOperation.INSPECT,
    ContainerLifecyclePhase.COPY_OUTPUTS: (
        ContainerBackendOperation.COPY_OUTPUTS
    ),
    ContainerLifecyclePhase.STOP: ContainerBackendOperation.STOP,
    ContainerLifecyclePhase.KILL: ContainerBackendOperation.KILL,
    ContainerLifecyclePhase.REMOVE: ContainerBackendOperation.REMOVE,
    ContainerLifecyclePhase.CLEANUP: ContainerBackendOperation.CLEANUP,
}


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerLifecycleEvent:
    phase: ContainerLifecyclePhase | str
    status: ContainerLifecycleEventStatus | str
    sequence: int
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "phase",
            _enum_value(self.phase, ContainerLifecyclePhase, "phase"),
        )
        object.__setattr__(
            self,
            "status",
            _enum_value(
                self.status,
                ContainerLifecycleEventStatus,
                "status",
            ),
        )
        assert isinstance(self.sequence, int)
        assert self.sequence >= 0, "sequence must not be negative"
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(_string_mapping(self.metadata, "metadata")),
        )

    def to_dict(self) -> dict[str, object]:
        phase = cast(ContainerLifecyclePhase, self.phase)
        status = cast(ContainerLifecycleEventStatus, self.status)
        return {
            "phase": phase.value,
            "status": status.value,
            "sequence": self.sequence,
            "metadata": dict(self.metadata),
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerLifecycleDeadlines:
    pull_seconds: float | None = None
    build_seconds: float | None = None
    create_seconds: float | None = None
    start_seconds: float | None = None
    execution_seconds: float | None = None
    idle_seconds: float | None = None
    copy_seconds: float | None = None
    cleanup_seconds: float | None = None
    parent_seconds: float | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "pull_seconds",
            "build_seconds",
            "create_seconds",
            "start_seconds",
            "execution_seconds",
            "idle_seconds",
            "copy_seconds",
            "cleanup_seconds",
            "parent_seconds",
        ):
            _assert_optional_positive_number(
                getattr(self, field_name),
                field_name,
            )

    def effective_seconds(
        self,
        phase: ContainerLifecyclePhase | str,
    ) -> float | None:
        resolved = _enum_value(phase, ContainerLifecyclePhase, "phase")
        phase_seconds = self._phase_seconds(resolved)
        if self.parent_seconds is None:
            return phase_seconds
        if phase_seconds is None:
            return self.parent_seconds
        return min(phase_seconds, self.parent_seconds)

    def to_dict(self) -> dict[str, float | None]:
        return {
            "pull_seconds": self.pull_seconds,
            "build_seconds": self.build_seconds,
            "create_seconds": self.create_seconds,
            "start_seconds": self.start_seconds,
            "execution_seconds": self.execution_seconds,
            "idle_seconds": self.idle_seconds,
            "copy_seconds": self.copy_seconds,
            "cleanup_seconds": self.cleanup_seconds,
            "parent_seconds": self.parent_seconds,
        }

    def _phase_seconds(
        self,
        phase: ContainerLifecyclePhase,
    ) -> float | None:
        field_name = _PHASE_DEADLINE_FIELDS.get(phase)
        return (
            None
            if field_name is None
            else cast(
                float | None,
                getattr(
                    self,
                    field_name,
                ),
            )
        )


class _LifecycleDeadlineBudget:
    def __init__(self, deadlines: ContainerLifecycleDeadlines) -> None:
        assert isinstance(deadlines, ContainerLifecycleDeadlines)
        self._deadlines = deadlines
        self._parent_deadline = (
            None
            if deadlines.parent_seconds is None
            else get_running_loop().time() + deadlines.parent_seconds
        )

    def effective_seconds(
        self,
        phase: ContainerLifecyclePhase,
        *,
        timeout_override: float | None = None,
    ) -> float | None:
        phase_seconds = (
            timeout_override
            if timeout_override is not None
            else self._deadlines._phase_seconds(phase)
        )
        parent_seconds = self._parent_remaining_seconds()
        if parent_seconds is None:
            return phase_seconds
        if parent_seconds <= 0:
            return 0
        if phase_seconds is None:
            return parent_seconds
        return min(phase_seconds, parent_seconds)

    def _parent_remaining_seconds(self) -> float | None:
        if self._parent_deadline is None:
            return None
        return self._parent_deadline - get_running_loop().time()


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerStreamDrainPolicy:
    max_chunks: int = 128
    max_bytes: int = 65536
    max_chunk_bytes: int = 8192
    max_stdout_bytes: int | None = None
    max_stderr_bytes: int | None = None
    max_non_output_chunks: int | None = None
    max_non_output_bytes: int | None = None
    preserve_truncated_prefix: bool = False

    def __post_init__(self) -> None:
        _assert_positive_int(self.max_chunks, "max_chunks")
        _assert_positive_int(self.max_bytes, "max_bytes")
        _assert_positive_int(self.max_chunk_bytes, "max_chunk_bytes")
        _assert_optional_non_negative_int(
            self.max_stdout_bytes,
            "max_stdout_bytes",
        )
        _assert_optional_non_negative_int(
            self.max_stderr_bytes,
            "max_stderr_bytes",
        )
        _assert_optional_non_negative_int(
            self.max_non_output_chunks,
            "max_non_output_chunks",
        )
        _assert_optional_non_negative_int(
            self.max_non_output_bytes,
            "max_non_output_bytes",
        )
        _assert_bool(
            self.preserve_truncated_prefix,
            "preserve_truncated_prefix",
        )

    def to_dict(self) -> dict[str, int | bool | None]:
        return {
            "max_chunks": self.max_chunks,
            "max_bytes": self.max_bytes,
            "max_chunk_bytes": self.max_chunk_bytes,
            "max_stdout_bytes": self.max_stdout_bytes,
            "max_stderr_bytes": self.max_stderr_bytes,
            "max_non_output_chunks": self.max_non_output_chunks,
            "max_non_output_bytes": self.max_non_output_bytes,
            "preserve_truncated_prefix": self.preserve_truncated_prefix,
        }


class _StreamDrainAccumulator:
    def __init__(self, policy: ContainerStreamDrainPolicy) -> None:
        assert isinstance(policy, ContainerStreamDrainPolicy)
        self._policy = policy
        self._kept: list[ContainerBackendStreamChunk] = []
        self._total_bytes = 0
        self._output_bytes = 0
        self._stdout_bytes = 0
        self._stderr_bytes = 0
        self._output_chunks = 0
        self._non_output_chunks = 0
        self._non_output_bytes = 0
        self._dropped_chunks = 0
        self._truncated_chunks = 0

    def append(self, chunk: ContainerBackendStreamChunk) -> None:
        assert isinstance(chunk, ContainerBackendStreamChunk)
        if self._chunk_limit_reached(chunk):
            self._dropped_chunks += 1
            return
        remaining = self._remaining_bytes(chunk)
        if remaining <= 0:
            self._dropped_chunks += 1
            return
        content = chunk.content
        chunk_limit = min(self._policy.max_chunk_bytes, remaining)
        if len(content) > chunk_limit:
            if self._policy.preserve_truncated_prefix:
                content = content[:chunk_limit]
            else:
                content = _truncate_content(content, chunk_limit)
            self._truncated_chunks += 1
        kept_chunk = ContainerBackendStreamChunk(
            stream=chunk.stream,
            content=content,
            sequence=chunk.sequence,
        )
        self._kept.append(kept_chunk)
        self._total_bytes += len(content)
        if self._is_output_chunk(kept_chunk):
            self._output_bytes += len(content)
            self._output_chunks += 1
        else:
            self._non_output_bytes += len(content)
            self._non_output_chunks += 1
        if kept_chunk.stream is ContainerBackendStream.STDOUT:
            self._stdout_bytes += len(content)
        if kept_chunk.stream is ContainerBackendStream.STDERR:
            self._stderr_bytes += len(content)

    def _chunk_limit_reached(
        self,
        chunk: ContainerBackendStreamChunk,
    ) -> bool:
        if self._policy.max_non_output_chunks is None:
            return len(self._kept) >= self._policy.max_chunks
        if self._is_output_chunk(chunk):
            return self._output_chunks >= self._policy.max_chunks
        return self._non_output_chunks >= self._policy.max_non_output_chunks

    def _remaining_bytes(self, chunk: ContainerBackendStreamChunk) -> int:
        if (
            self._policy.max_non_output_bytes is not None
            and not self._is_output_chunk(chunk)
        ):
            return self._policy.max_non_output_bytes - self._non_output_bytes
        total_used = self._total_bytes
        if self._policy.max_non_output_bytes is not None:
            total_used = self._output_bytes
        total_remaining = self._policy.max_bytes - total_used
        if total_remaining <= 0:
            return 0
        stream_limit: int | None = None
        if chunk.stream is ContainerBackendStream.STDOUT:
            stream_limit = self._policy.max_stdout_bytes
            stream_used = self._stdout_bytes
        elif chunk.stream is ContainerBackendStream.STDERR:
            stream_limit = self._policy.max_stderr_bytes
            stream_used = self._stderr_bytes
        else:
            stream_used = 0
        if stream_limit is None:
            return total_remaining
        return min(total_remaining, stream_limit - stream_used)

    def _is_output_chunk(self, chunk: ContainerBackendStreamChunk) -> bool:
        return chunk.stream in {
            ContainerBackendStream.STDOUT,
            ContainerBackendStream.STDERR,
        }

    def result(self) -> "ContainerStreamDrainResult":
        diagnostics: list[ContainerBackendDiagnostic] = []
        if self._truncated_chunks:
            diagnostics.append(
                _diagnostic(
                    ContainerBackendDiagnosticCode.STREAM_TRUNCATED,
                    ContainerBackendOperation.STREAM,
                    "container stream output was truncated",
                )
            )
        if self._dropped_chunks:
            diagnostics.append(
                _diagnostic(
                    ContainerBackendDiagnosticCode.EVENT_DROPPED,
                    ContainerBackendOperation.STREAM,
                    "container stream events were dropped",
                )
            )
        return ContainerStreamDrainResult(
            chunks=self._kept,
            diagnostics=diagnostics,
            stdout_bytes=self._stdout_bytes,
            stderr_bytes=self._stderr_bytes,
            dropped_chunks=self._dropped_chunks,
            truncated_chunks=self._truncated_chunks,
        )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerLifecycleEventPolicy:
    max_events: int = 256

    def __post_init__(self) -> None:
        _assert_positive_int(self.max_events, "max_events")

    def to_dict(self) -> dict[str, int]:
        return {"max_events": self.max_events}


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerStreamDrainResult:
    chunks: Sequence[ContainerBackendStreamChunk] = field(
        default_factory=tuple,
    )
    diagnostics: Sequence[ContainerBackendDiagnostic] = field(
        default_factory=tuple,
    )
    stdout_bytes: int = 0
    stderr_bytes: int = 0
    dropped_chunks: int = 0
    truncated_chunks: int = 0

    def __post_init__(self) -> None:
        for chunk in self.chunks:
            assert isinstance(chunk, ContainerBackendStreamChunk)
        for diagnostic in self.diagnostics:
            assert isinstance(diagnostic, ContainerBackendDiagnostic)
        _assert_non_negative_int(self.stdout_bytes, "stdout_bytes")
        _assert_non_negative_int(self.stderr_bytes, "stderr_bytes")
        _assert_non_negative_int(self.dropped_chunks, "dropped_chunks")
        _assert_non_negative_int(self.truncated_chunks, "truncated_chunks")
        object.__setattr__(self, "chunks", tuple(self.chunks))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    @property
    def stdout_chunks(self) -> tuple[ContainerBackendStreamChunk, ...]:
        return tuple(
            chunk
            for chunk in self.chunks
            if chunk.stream is ContainerBackendStream.STDOUT
        )

    @property
    def stderr_chunks(self) -> tuple[ContainerBackendStreamChunk, ...]:
        return tuple(
            chunk
            for chunk in self.chunks
            if chunk.stream is ContainerBackendStream.STDERR
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "diagnostics": [
                diagnostic.to_dict() for diagnostic in self.diagnostics
            ],
            "stdout_bytes": self.stdout_bytes,
            "stderr_bytes": self.stderr_bytes,
            "dropped_chunks": self.dropped_chunks,
            "truncated_chunks": self.truncated_chunks,
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerLifecycleCleanupResult:
    diagnostics: Sequence[ContainerBackendDiagnostic] = field(
        default_factory=tuple,
    )
    cleanup_uncertain: bool = False
    orphan_quarantined: bool = False
    already_cleaned: bool = False

    def __post_init__(self) -> None:
        for diagnostic in self.diagnostics:
            assert isinstance(diagnostic, ContainerBackendDiagnostic)
        _assert_bool(self.cleanup_uncertain, "cleanup_uncertain")
        _assert_bool(self.orphan_quarantined, "orphan_quarantined")
        _assert_bool(self.already_cleaned, "already_cleaned")
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    def to_dict(self) -> dict[str, object]:
        return {
            "diagnostics": [
                diagnostic.to_dict() for diagnostic in self.diagnostics
            ],
            "cleanup_uncertain": self.cleanup_uncertain,
            "orphan_quarantined": self.orphan_quarantined,
            "already_cleaned": self.already_cleaned,
        }


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ContainerManagedLifecycleResult:
    execution: ContainerExecutionResult
    events: Sequence[ContainerLifecycleEvent] = field(default_factory=tuple)
    diagnostics: Sequence[ContainerBackendDiagnostic] = field(
        default_factory=tuple,
    )
    stream: ContainerStreamDrainResult = field(
        default_factory=ContainerStreamDrainResult,
    )
    stats: Sequence[ContainerBackendStats] = field(default_factory=tuple)
    output: ContainerOutputValidationResult | None = None
    cleanup_uncertain: bool = False
    orphan_quarantined: bool = False
    cleanup_completed: bool = False
    timed_out_phase: ContainerLifecyclePhase | str | None = None
    cancelled_phase: ContainerLifecyclePhase | str | None = None
    dropped_events: int = 0

    def __post_init__(self) -> None:
        assert isinstance(self.execution, ContainerExecutionResult)
        for event in self.events:
            assert isinstance(event, ContainerLifecycleEvent)
        for diagnostic in self.diagnostics:
            assert isinstance(diagnostic, ContainerBackendDiagnostic)
        assert isinstance(self.stream, ContainerStreamDrainResult)
        for stat in self.stats:
            assert isinstance(stat, ContainerBackendStats)
        if self.output is not None:
            assert isinstance(self.output, ContainerOutputValidationResult)
        _assert_bool(self.cleanup_uncertain, "cleanup_uncertain")
        _assert_bool(self.orphan_quarantined, "orphan_quarantined")
        _assert_bool(self.cleanup_completed, "cleanup_completed")
        if self.timed_out_phase is not None:
            object.__setattr__(
                self,
                "timed_out_phase",
                _enum_value(
                    self.timed_out_phase,
                    ContainerLifecyclePhase,
                    "timed_out_phase",
                ),
            )
        if self.cancelled_phase is not None:
            object.__setattr__(
                self,
                "cancelled_phase",
                _enum_value(
                    self.cancelled_phase,
                    ContainerLifecyclePhase,
                    "cancelled_phase",
                ),
            )
        _assert_non_negative_int(self.dropped_events, "dropped_events")
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(self, "stats", tuple(self.stats))

    def to_backend_result(self) -> ContainerBackendLifecycleResult:
        return ContainerBackendLifecycleResult(
            execution=self.execution,
            diagnostics=self.diagnostics,
            stream_chunks=self.stream.chunks,
            stats=self.stats,
            output=self.output,
            cleanup_uncertain=self.cleanup_uncertain,
            orphan_quarantined=self.orphan_quarantined,
        )

    def to_dict(self) -> dict[str, object]:
        timed_out_phase = cast(
            ContainerLifecyclePhase | None,
            self.timed_out_phase,
        )
        cancelled_phase = cast(
            ContainerLifecyclePhase | None,
            self.cancelled_phase,
        )
        return {
            "execution": self.execution.to_dict(),
            "events": [event.to_dict() for event in self.events],
            "diagnostics": [
                diagnostic.to_dict() for diagnostic in self.diagnostics
            ],
            "stream": self.stream.to_dict(),
            "stats": [stat.to_dict() for stat in self.stats],
            "output": None if self.output is None else self.output.to_dict(),
            "cleanup_uncertain": self.cleanup_uncertain,
            "orphan_quarantined": self.orphan_quarantined,
            "cleanup_completed": self.cleanup_completed,
            "timed_out_phase": (
                None if timed_out_phase is None else timed_out_phase.value
            ),
            "cancelled_phase": (
                None if cancelled_phase is None else cancelled_phase.value
            ),
            "dropped_events": self.dropped_events,
        }


class ContainerLifecycleCleanup:
    def __init__(self) -> None:
        self._cleaned_container_ids: set[str] = set()

    async def cleanup(
        self,
        backend: ContainerAsyncBackend,
        container: ContainerBackendContainer,
        *,
        deadlines: ContainerLifecycleDeadlines | None = None,
        deadline_budget: "_LifecycleDeadlineBudget | None" = None,
        recorder: "_LifecycleRecorder | None" = None,
        force_kill: bool = False,
    ) -> ContainerLifecycleCleanupResult:
        assert isinstance(backend, ContainerAsyncBackend)
        assert isinstance(container, ContainerBackendContainer)
        resolved_deadlines = deadlines or ContainerLifecycleDeadlines()
        budget = deadline_budget or _LifecycleDeadlineBudget(
            resolved_deadlines,
        )
        if container.container_id in self._cleaned_container_ids:
            return ContainerLifecycleCleanupResult(already_cleaned=True)
        diagnostics: list[ContainerBackendDiagnostic] = []
        cleanup_uncertain = False
        orphan_quarantined = False
        if force_kill:
            stop_result = await _cleanup_operation(
                backend.stop(container),
                ContainerLifecyclePhase.STOP,
                ContainerBackendOperation.STOP,
                budget,
                recorder,
            )
            diagnostics.extend(stop_result.diagnostics)
            if stop_result.diagnostics:
                cleanup_uncertain = True
            kill_result = await _cleanup_operation(
                backend.kill(container),
                ContainerLifecyclePhase.KILL,
                ContainerBackendOperation.KILL,
                budget,
                recorder,
            )
            diagnostics.extend(kill_result.diagnostics)
            if kill_result.diagnostics:
                cleanup_uncertain = True
                orphan_quarantined = _contains_orphan(kill_result.diagnostics)
        remove_result = await _cleanup_operation(
            backend.remove(container),
            ContainerLifecyclePhase.REMOVE,
            ContainerBackendOperation.REMOVE,
            budget,
            recorder,
        )
        diagnostics.extend(remove_result.diagnostics)
        if remove_result.diagnostics:
            cleanup_uncertain = True
            orphan_quarantined = orphan_quarantined or _contains_orphan(
                remove_result.diagnostics
            )
        cleanup_result = await _cleanup_operation(
            backend.cleanup(container),
            ContainerLifecyclePhase.CLEANUP,
            ContainerBackendOperation.CLEANUP,
            budget,
            recorder,
        )
        diagnostics.extend(cleanup_result.diagnostics)
        if cleanup_result.diagnostics:
            cleanup_uncertain = True
            orphan_quarantined = orphan_quarantined or _contains_orphan(
                cleanup_result.diagnostics
            )
        if not cleanup_uncertain:
            self._cleaned_container_ids.add(container.container_id)
        return ContainerLifecycleCleanupResult(
            diagnostics=diagnostics,
            cleanup_uncertain=cleanup_uncertain,
            orphan_quarantined=orphan_quarantined,
        )


async def run_container_managed_lifecycle(
    backend: ContainerAsyncBackend,
    plan: ContainerRunPlan,
    *,
    output_contract: ContainerOutputContract | None = None,
    deadlines: ContainerLifecycleDeadlines | None = None,
    stream_policy: ContainerStreamDrainPolicy | None = None,
    event_policy: ContainerLifecycleEventPolicy | None = None,
    shutdown_requested: bool = False,
) -> ContainerManagedLifecycleResult:
    assert isinstance(backend, ContainerAsyncBackend)
    assert isinstance(plan, ContainerRunPlan)
    if output_contract is not None:
        assert isinstance(output_contract, ContainerOutputContract)
    resolved_deadlines = deadlines or ContainerLifecycleDeadlines()
    deadline_budget = _LifecycleDeadlineBudget(resolved_deadlines)
    resolved_stream_policy = stream_policy or ContainerStreamDrainPolicy()
    recorder = _LifecycleRecorder(
        event_policy or ContainerLifecycleEventPolicy(),
    )
    cleanup = ContainerLifecycleCleanup()
    diagnostics: list[ContainerBackendDiagnostic] = []
    stream_result = ContainerStreamDrainResult()
    stats: tuple[ContainerBackendStats, ...] = ()
    output: ContainerOutputValidationResult | None = None
    container: ContainerBackendContainer | None = None
    exit_code: int | None = None
    status = ContainerResultStatus.COMPLETED
    timed_out_phase: ContainerLifecyclePhase | None = None
    cancelled_phase: ContainerLifecyclePhase | None = None
    cleanup_uncertain = False
    orphan_quarantined = False
    cleanup_completed = False
    force_kill = False
    try:
        await _record_noop_phase(
            recorder,
            ContainerLifecyclePhase.POLICY_NORMALIZATION,
        )
        await _record_noop_phase(
            recorder,
            ContainerLifecyclePhase.BACKEND_SELECTION,
        )
        image = cast(
            ContainerBackendImageResolution,
            await _run_phase(
                recorder,
                deadline_budget,
                ContainerLifecyclePhase.IMAGE_RESOLUTION,
                ContainerBackendOperation.IMAGE_RESOLUTION,
                backend.resolve_image(plan),
            ),
        )
        diagnostics.extend(image.diagnostics)
        if not image.ok:
            raise _ManagedLifecycleFailure(
                _status_for_diagnostics(image.diagnostics)
            )
        if plan.image.pull_policy is not ContainerPullPolicy.NEVER:
            _append_operation_result(
                diagnostics,
                await _run_phase(
                    recorder,
                    deadline_budget,
                    ContainerLifecyclePhase.IMAGE_PULL,
                    ContainerBackendOperation.IMAGE_PULL,
                    backend.pull_image(plan, image),
                ),
            )
        if plan.image.build_policy is not ContainerBuildPolicy.DISABLED:
            _append_operation_result(
                diagnostics,
                await _run_phase(
                    recorder,
                    deadline_budget,
                    ContainerLifecyclePhase.IMAGE_BUILD,
                    ContainerBackendOperation.IMAGE_BUILD,
                    backend.build_image(plan),
                ),
            )
        container = cast(
            ContainerBackendContainer,
            await _run_phase(
                recorder,
                deadline_budget,
                ContainerLifecyclePhase.CREATE,
                ContainerBackendOperation.CREATE,
                backend.create(plan),
            ),
        )
        _append_operation_result(
            diagnostics,
            await _run_phase(
                recorder,
                deadline_budget,
                ContainerLifecyclePhase.ATTACH,
                ContainerBackendOperation.ATTACH,
                backend.attach(container),
            ),
        )
        _append_operation_result(
            diagnostics,
            await _run_phase(
                recorder,
                deadline_budget,
                ContainerLifecyclePhase.START,
                ContainerBackendOperation.START,
                backend.start(container),
            ),
        )
        if shutdown_requested:
            force_kill = True
            raise _ManagedLifecycleCancellation(
                ContainerLifecyclePhase.START,
                ContainerBackendOperation.STOP,
                "container lifecycle shutdown requested",
            )
        stream_result = cast(
            ContainerStreamDrainResult,
            await _run_phase(
                recorder,
                deadline_budget,
                ContainerLifecyclePhase.STREAM,
                ContainerBackendOperation.STREAM,
                _drain_stream_phase(
                    backend.stream(container),
                    resolved_stream_policy,
                    resolved_deadlines,
                    deadline_budget,
                ),
            ),
        )
        diagnostics.extend(stream_result.diagnostics)
        stats = cast(
            tuple[ContainerBackendStats, ...],
            await _run_phase(
                recorder,
                deadline_budget,
                ContainerLifecyclePhase.STATS,
                ContainerBackendOperation.STATS,
                backend.stats(container),
            ),
        )
        wait_result = cast(
            ContainerBackendWaitResult,
            await _run_phase(
                recorder,
                deadline_budget,
                ContainerLifecyclePhase.WAIT,
                ContainerBackendOperation.WAIT,
                backend.wait(container),
            ),
        )
        diagnostics.extend(wait_result.diagnostics)
        if wait_result.timed_out:
            diagnostics.append(
                _diagnostic(
                    ContainerBackendDiagnosticCode.TIMEOUT,
                    ContainerBackendOperation.WAIT,
                    "container execution timed out",
                )
            )
            force_kill = True
            raise _ManagedLifecycleTimeout(ContainerLifecyclePhase.WAIT)
        if wait_result.diagnostics:
            raise _ManagedLifecycleFailure(
                _status_for_diagnostics(wait_result.diagnostics)
            )
        exit_code = wait_result.exit_code
        if exit_code != 0:
            status = ContainerResultStatus.FAILED
        await _run_phase(
            recorder,
            deadline_budget,
            ContainerLifecyclePhase.INSPECT,
            ContainerBackendOperation.INSPECT,
            backend.inspect(container),
        )
        if output_contract is not None:
            output = cast(
                ContainerOutputValidationResult,
                await _run_phase(
                    recorder,
                    deadline_budget,
                    ContainerLifecyclePhase.COPY_OUTPUTS,
                    ContainerBackendOperation.COPY_OUTPUTS,
                    backend.copy_outputs(container, output_contract),
                ),
            )
            if output.decision is not ContainerOutputDecisionType.ACCEPT:
                status = ContainerResultStatus.FAILED
    except _ManagedLifecycleTimeout as error:
        status = ContainerResultStatus.FAILED
        timed_out_phase = error.phase
        diagnostics.append(
            _diagnostic(
                ContainerBackendDiagnosticCode.TIMEOUT,
                error.operation,
                f"container lifecycle timed out during {error.phase.value}",
            )
        )
        force_kill = True
    except _ManagedLifecycleCancellation as error:
        status = ContainerResultStatus.CANCELLED
        cancelled_phase = error.phase
        diagnostics.append(
            _diagnostic(
                ContainerBackendDiagnosticCode.CANCELLED,
                error.operation,
                error.message,
            )
        )
        force_kill = True
    except _ManagedLifecycleFailure as error:
        status = error.status
        force_kill = True
    except ContainerBackendError as error:
        diagnostics.append(error.diagnostic)
        status = _status_for_diagnostic(error.diagnostic)
        force_kill = True
    finally:
        if container is not None:
            cleanup_result, cleanup_cancelled = await _run_cleanup_shielded(
                cleanup,
                backend,
                container,
                deadlines=resolved_deadlines,
                deadline_budget=deadline_budget,
                recorder=recorder,
                force_kill=force_kill,
            )
            diagnostics.extend(cleanup_result.diagnostics)
            cleanup_uncertain = cleanup_result.cleanup_uncertain
            orphan_quarantined = cleanup_result.orphan_quarantined
            cleanup_completed = not cleanup_result.already_cleaned
            if cleanup_cancelled:
                status = ContainerResultStatus.CANCELLED
                cancelled_phase = ContainerLifecyclePhase.CLEANUP
                diagnostics.append(
                    _diagnostic(
                        ContainerBackendDiagnosticCode.CANCELLED,
                        ContainerBackendOperation.CLEANUP,
                        "container lifecycle cancelled during cleanup",
                    )
                )
            if cleanup_uncertain and status is ContainerResultStatus.COMPLETED:
                status = ContainerResultStatus.FAILED
        recorder.record(
            ContainerLifecyclePhase.RESULT,
            ContainerLifecycleEventStatus.COMPLETED,
            {"status": status.value},
        )
    return ContainerManagedLifecycleResult(
        execution=ContainerExecutionResult(
            status=status,
            exit_code=exit_code,
            diagnostics=tuple(_diagnostic_text(item) for item in diagnostics),
            metadata={
                "cleanup_completed": str(cleanup_completed).lower(),
                "cleanup_uncertain": str(cleanup_uncertain).lower(),
                "dropped_events": str(recorder.dropped_events),
            },
        ),
        events=recorder.events,
        diagnostics=diagnostics,
        stream=stream_result,
        stats=stats,
        output=output,
        cleanup_uncertain=cleanup_uncertain,
        orphan_quarantined=orphan_quarantined,
        cleanup_completed=cleanup_completed,
        timed_out_phase=timed_out_phase,
        cancelled_phase=cancelled_phase,
        dropped_events=recorder.dropped_events,
    )


def drain_container_streams(
    chunks: Sequence[ContainerBackendStreamChunk],
    policy: ContainerStreamDrainPolicy | None = None,
) -> ContainerStreamDrainResult:
    resolved_policy = policy or ContainerStreamDrainPolicy()
    assert isinstance(resolved_policy, ContainerStreamDrainPolicy)
    accumulator = _StreamDrainAccumulator(resolved_policy)
    for chunk in chunks:
        accumulator.append(chunk)
    return accumulator.result()


async def _drain_stream_phase(
    stream_awaitable: Awaitable[
        Sequence[ContainerBackendStreamChunk]
        | AsyncIterable[ContainerBackendStreamChunk]
    ],
    policy: ContainerStreamDrainPolicy,
    deadlines: ContainerLifecycleDeadlines,
    deadline_budget: _LifecycleDeadlineBudget,
) -> ContainerStreamDrainResult:
    source = await stream_awaitable
    accumulator = _StreamDrainAccumulator(policy)
    if isinstance(source, AsyncIterable):
        async_iterator = source.__aiter__()
        while True:
            try:
                chunk = await wait_for(
                    async_iterator.__anext__(),
                    timeout=deadline_budget.effective_seconds(
                        ContainerLifecyclePhase.STREAM,
                        timeout_override=deadlines.idle_seconds,
                    ),
                )
            except StopAsyncIteration:
                break
            accumulator.append(chunk)
        return accumulator.result()
    assert isinstance(source, Iterable), "stream source must be iterable"
    for chunk in source:
        accumulator.append(chunk)
    return accumulator.result()


async def _run_cleanup_shielded(
    cleanup: ContainerLifecycleCleanup,
    backend: ContainerAsyncBackend,
    container: ContainerBackendContainer,
    *,
    deadlines: ContainerLifecycleDeadlines,
    deadline_budget: _LifecycleDeadlineBudget,
    recorder: "_LifecycleRecorder",
    force_kill: bool,
) -> tuple[ContainerLifecycleCleanupResult, bool]:
    cleanup_task = create_task(
        cleanup.cleanup(
            backend,
            container,
            deadlines=deadlines,
            deadline_budget=deadline_budget,
            recorder=recorder,
            force_kill=force_kill,
        )
    )
    try:
        return await shield(cleanup_task), False
    except CancelledError:
        try:
            return (
                await wait_for(
                    shield(cleanup_task),
                    timeout=deadline_budget.effective_seconds(
                        ContainerLifecyclePhase.CLEANUP,
                    ),
                ),
                True,
            )
        except TimeoutError:
            return (
                ContainerLifecycleCleanupResult(
                    diagnostics=(
                        _diagnostic(
                            ContainerBackendDiagnosticCode.TIMEOUT,
                            ContainerBackendOperation.CLEANUP,
                            "cleanup did not finish after caller cancellation",
                        ),
                    ),
                    cleanup_uncertain=True,
                ),
                True,
            )
        except CancelledError:
            return (
                ContainerLifecycleCleanupResult(
                    diagnostics=(
                        _diagnostic(
                            ContainerBackendDiagnosticCode.CANCELLED,
                            ContainerBackendOperation.CLEANUP,
                            "cleanup was interrupted by repeated cancellation",
                        ),
                    ),
                    cleanup_uncertain=True,
                ),
                True,
            )


class _LifecycleRecorder:
    def __init__(self, policy: ContainerLifecycleEventPolicy) -> None:
        assert isinstance(policy, ContainerLifecycleEventPolicy)
        self._policy = policy
        self._events: list[ContainerLifecycleEvent] = []
        self._dropped_events = 0

    @property
    def events(self) -> tuple[ContainerLifecycleEvent, ...]:
        return tuple(self._events)

    @property
    def dropped_events(self) -> int:
        return self._dropped_events

    def record(
        self,
        phase: ContainerLifecyclePhase,
        status: ContainerLifecycleEventStatus,
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        if len(self._events) >= self._policy.max_events:
            self._dropped_events += 1
            return
        self._events.append(
            ContainerLifecycleEvent(
                phase=phase,
                status=status,
                sequence=len(self._events),
                metadata=metadata or {},
            )
        )


class _ManagedLifecycleFailure(Exception):
    def __init__(self, status: ContainerResultStatus) -> None:
        assert isinstance(status, ContainerResultStatus)
        super().__init__(status.value)
        self.status = status


class _ManagedLifecycleTimeout(Exception):
    def __init__(self, phase: ContainerLifecyclePhase) -> None:
        assert isinstance(phase, ContainerLifecyclePhase)
        super().__init__(phase.value)
        self.phase = phase
        self.operation = _operation_for_phase(phase)


class _ManagedLifecycleCancellation(Exception):
    def __init__(
        self,
        phase: ContainerLifecyclePhase,
        operation: ContainerBackendOperation,
        message: str,
    ) -> None:
        assert isinstance(phase, ContainerLifecyclePhase)
        assert isinstance(operation, ContainerBackendOperation)
        _assert_non_empty_string(message, "message")
        super().__init__(phase.value)
        self.phase = phase
        self.operation = operation
        self.message = message


async def _record_noop_phase(
    recorder: _LifecycleRecorder,
    phase: ContainerLifecyclePhase,
) -> None:
    recorder.record(phase, ContainerLifecycleEventStatus.STARTED)
    recorder.record(phase, ContainerLifecycleEventStatus.COMPLETED)


async def _run_phase(
    recorder: _LifecycleRecorder,
    deadline_budget: _LifecycleDeadlineBudget,
    phase: ContainerLifecyclePhase,
    operation: ContainerBackendOperation,
    awaitable: Awaitable[object],
    *,
    timeout_override: float | None = None,
) -> object:
    recorder.record(phase, ContainerLifecycleEventStatus.STARTED)
    timeout = deadline_budget.effective_seconds(
        phase,
        timeout_override=timeout_override,
    )
    try:
        result = await wait_for(awaitable, timeout=timeout)
    except TimeoutError as error:
        recorder.record(phase, ContainerLifecycleEventStatus.FAILED)
        raise _ManagedLifecycleTimeout(phase) from error
    except CancelledError as error:
        recorder.record(phase, ContainerLifecycleEventStatus.FAILED)
        raise _ManagedLifecycleCancellation(
            phase,
            operation,
            f"container lifecycle cancelled during {phase.value}",
        ) from error
    recorder.record(phase, ContainerLifecycleEventStatus.COMPLETED)
    return result


async def _cleanup_operation(
    awaitable: Awaitable[ContainerBackendOperationResult],
    phase: ContainerLifecyclePhase,
    operation: ContainerBackendOperation,
    deadline_budget: _LifecycleDeadlineBudget,
    recorder: _LifecycleRecorder | None,
) -> ContainerBackendOperationResult:
    if recorder is not None:
        recorder.record(phase, ContainerLifecycleEventStatus.STARTED)
    try:
        result = await wait_for(
            awaitable,
            timeout=deadline_budget.effective_seconds(phase),
        )
    except ContainerBackendError as error:
        if recorder is not None:
            recorder.record(phase, ContainerLifecycleEventStatus.FAILED)
        return ContainerBackendOperationResult(
            operation=operation,
            diagnostics=(error.diagnostic,),
        )
    except TimeoutError:
        if recorder is not None:
            recorder.record(phase, ContainerLifecycleEventStatus.FAILED)
        return ContainerBackendOperationResult(
            operation=operation,
            diagnostics=(
                _diagnostic(
                    ContainerBackendDiagnosticCode.TIMEOUT,
                    operation,
                    f"cleanup timed out during {phase.value}",
                ),
            ),
        )
    except CancelledError:
        if recorder is not None:
            recorder.record(phase, ContainerLifecycleEventStatus.FAILED)
        return ContainerBackendOperationResult(
            operation=operation,
            diagnostics=(
                _diagnostic(
                    ContainerBackendDiagnosticCode.CANCELLED,
                    operation,
                    f"cleanup cancelled during {phase.value}",
                ),
            ),
        )
    if recorder is not None:
        recorder.record(
            phase,
            (
                ContainerLifecycleEventStatus.FAILED
                if result.diagnostics
                else ContainerLifecycleEventStatus.COMPLETED
            ),
        )
    return result


def _append_operation_result(
    diagnostics: list[ContainerBackendDiagnostic],
    result: object,
) -> None:
    assert isinstance(result, ContainerBackendOperationResult)
    diagnostics.extend(result.diagnostics)
    if not result.ok:
        raise _ManagedLifecycleFailure(
            _status_for_diagnostics(result.diagnostics)
        )


def _truncate_content(content: bytes, limit: int) -> bytes:
    if limit <= len(_TRUNCATION_MARKER):
        return _TRUNCATION_MARKER[:limit]
    keep = limit - len(_TRUNCATION_MARKER)
    return content[:keep] + _TRUNCATION_MARKER


def _contains_orphan(
    diagnostics: Sequence[ContainerBackendDiagnostic],
) -> bool:
    return any(
        diagnostic.code is ContainerBackendDiagnosticCode.ORPHAN_QUARANTINED
        for diagnostic in diagnostics
    )


def _operation_for_phase(
    phase: ContainerLifecyclePhase,
) -> ContainerBackendOperation:
    return _PHASE_OPERATIONS.get(phase, ContainerBackendOperation.CLEANUP)


def _status_for_diagnostic(
    diagnostic: ContainerBackendDiagnostic,
) -> ContainerResultStatus:
    if diagnostic.code in {
        ContainerBackendDiagnosticCode.IMAGE_DENIED,
        ContainerBackendDiagnosticCode.PULL_DENIED,
        ContainerBackendDiagnosticCode.BUILD_DENIED,
        ContainerBackendDiagnosticCode.CAPABILITY_MISMATCH,
        ContainerBackendDiagnosticCode.ROOTFUL_NOT_AUTHORIZED,
    }:
        return ContainerResultStatus.DENIED
    return ContainerResultStatus.FAILED


def _status_for_diagnostics(
    diagnostics: Sequence[ContainerBackendDiagnostic],
) -> ContainerResultStatus:
    for diagnostic in diagnostics:
        if _status_for_diagnostic(diagnostic) is ContainerResultStatus.DENIED:
            return ContainerResultStatus.DENIED
    return ContainerResultStatus.FAILED


def _diagnostic(
    code: ContainerBackendDiagnosticCode,
    operation: ContainerBackendOperation,
    message: str,
) -> ContainerBackendDiagnostic:
    return ContainerBackendDiagnostic(
        code=code,
        operation=operation,
        message=message,
        retryable=True,
    )


def _diagnostic_text(diagnostic: ContainerBackendDiagnostic) -> str:
    code = cast(ContainerBackendDiagnosticCode, diagnostic.code)
    operation = cast(ContainerBackendOperation, diagnostic.operation)
    return f"{code.value}:{operation.value}:{diagnostic.message}"


def _assert_positive_int(value: int, field_name: str) -> None:
    assert isinstance(value, int), f"{field_name} must be an integer"
    assert value > 0, f"{field_name} must be positive"


def _assert_non_negative_int(value: int, field_name: str) -> None:
    assert isinstance(value, int), f"{field_name} must be an integer"
    assert value >= 0, f"{field_name} must not be negative"


def _assert_optional_non_negative_int(
    value: int | None,
    field_name: str,
) -> None:
    if value is None:
        return
    _assert_non_negative_int(value, field_name)


def _assert_optional_positive_number(
    value: float | None,
    field_name: str,
) -> None:
    if value is None:
        return
    assert isinstance(value, int | float), f"{field_name} must be numeric"
    assert value > 0, f"{field_name} must be positive"


def _string_mapping(
    value: Mapping[str, str],
    field_name: str,
) -> dict[str, str]:
    assert isinstance(value, Mapping), f"{field_name} must be a mapping"
    result = dict(value)
    for key, item in result.items():
        _assert_non_empty_string(key, field_name)
        _assert_non_empty_string(item, field_name)
        assert isinstance(key, str)
        assert isinstance(item, str)
    return result


def _enum_value(
    value: object,
    enum_type: type[EnumValue],
    field_name: str,
) -> EnumValue:
    if isinstance(value, enum_type):
        return value
    _assert_non_empty_string(value, field_name)
    assert isinstance(value, str)
    assert value in {
        member.value for member in enum_type
    }, f"{field_name} contains unsupported value"
    return enum_type(value)
