from ..types import JsonObject

from asyncio import Event, Lock, Task, create_task, wait, wait_for
from collections.abc import Awaitable, Callable, Hashable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from logging import getLogger
from typing import Literal, TypeAlias, cast
from uuid import uuid4

MCP_TASK_PROTOCOL_VERSION = "2025-11-25"
MCP_RELATED_TASK_METADATA_KEY = "io.modelcontextprotocol/related-task"
MCP_TASK_INPUT_METADATA_KEY = "https://avalan.ai/extensions/task-input/v1"

MCPTaskStatus: TypeAlias = Literal[
    "working",
    "input_required",
    "completed",
    "failed",
    "cancelled",
]
MCPTaskExecutionMode: TypeAlias = Literal[
    "forbidden",
    "optional",
    "required",
]
MCPTaskRequestor: TypeAlias = Hashable
MCPTaskCancellationCallback: TypeAlias = Callable[[], Awaitable[None]]

_TERMINAL = frozenset({"completed", "failed", "cancelled"})
_REQUEST_PATHS: Mapping[str, tuple[str, ...]] = {
    "tools/call": ("tools", "call"),
    "sampling/createMessage": ("sampling", "createMessage"),
    "elicitation/create": ("elicitation", "create"),
}
_LOGGER = getLogger(__name__)
_MISSING = object()


class MCPTaskProtocolError(RuntimeError):
    """Represent a sanitized MCP JSON-RPC task error."""

    def __init__(
        self,
        *,
        code: int,
        message: str,
        data: Mapping[str, object] | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.data = _copy(data) if data is not None else None
        super().__init__(message)

    def as_error(self) -> JsonObject:
        """Return the JSON-RPC error fields."""
        error: JsonObject = {"code": self.code, "message": self.message}
        if self.data is not None:
            error["data"] = _copy(self.data)
        return error


@dataclass(frozen=True, slots=True, kw_only=True)
class MCPTaskPolicy:
    """Bound one MCP receiver's task projection."""

    request_types: frozenset[str] = frozenset({"tools/call"})
    allow_cancel: bool = True
    allow_list: bool = True
    default_ttl_ms: int = 300_000
    maximum_ttl_ms: int = 3_600_000
    poll_interval_ms: int = 1_000
    page_size: int = 50
    maximum_tasks_per_requestor: int = 128
    expired_tombstone_limit: int = 256
    cancellation_timeout_ms: int = 1_000

    def __post_init__(self) -> None:
        assert self.request_types.issubset(_REQUEST_PATHS)
        assert isinstance(self.allow_cancel, bool)
        assert isinstance(self.allow_list, bool)
        for value in (
            self.default_ttl_ms,
            self.maximum_ttl_ms,
            self.poll_interval_ms,
            self.page_size,
            self.maximum_tasks_per_requestor,
            self.expired_tombstone_limit,
            self.cancellation_timeout_ms,
        ):
            assert _is_int(value) and value > 0
        assert self.default_ttl_ms <= self.maximum_ttl_ms


@dataclass(frozen=True, slots=True, kw_only=True)
class MCPTaskCapabilities:
    """Describe negotiated MCP task support."""

    request_types: frozenset[str] = frozenset()
    list_tasks: bool = False
    cancel_tasks: bool = False

    @classmethod
    def parse(
        cls,
        capabilities: Mapping[str, object],
    ) -> "MCPTaskCapabilities":
        """Parse an initialize capabilities object."""
        tasks_value = capabilities.get("tasks")
        if tasks_value is None:
            return cls()
        tasks = _mapping(tasks_value, "Invalid MCP task capabilities.")
        requests_value = tasks.get("requests")
        requests = (
            {}
            if requests_value is None
            else _mapping(
                requests_value,
                "Invalid MCP task request capabilities.",
            )
        )
        return cls(
            request_types=frozenset(
                method
                for method, path in _REQUEST_PATHS.items()
                if _has_path(requests, path)
            ),
            list_tasks=_present_capability(tasks, "list"),
            cancel_tasks=_present_capability(tasks, "cancel"),
        )

    @classmethod
    def advertise(
        cls,
        policy: MCPTaskPolicy,
        *,
        requestor: MCPTaskRequestor | None,
    ) -> "MCPTaskCapabilities":
        """Advertise only implemented and authorized operations."""
        _requestor(requestor)
        return cls(
            request_types=policy.request_types,
            list_tasks=policy.allow_list and requestor is not None,
            cancel_tasks=policy.allow_cancel,
        )

    def supports(self, request_type: str) -> bool:
        """Return whether one request may be task augmented."""
        return request_type in self.request_types

    def as_dict(self) -> JsonObject:
        """Return the `tasks` capabilities value."""
        tasks: JsonObject = {}
        if self.list_tasks:
            tasks["list"] = {}
        if self.cancel_tasks:
            tasks["cancel"] = {}
        requests: JsonObject = {}
        for method in sorted(self.request_types):
            _set_path(requests, _REQUEST_PATHS[method])
        if requests:
            tasks["requests"] = requests
        return tasks


@dataclass(frozen=True, slots=True, kw_only=True)
class MCPTaskRequest:
    """Hold validated task augmentation parameters."""

    request_type: str = "tools/call"
    requested_ttl_ms: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class MCPTaskOutcome:
    """Preserve an exact underlying result or JSON-RPC error."""

    result: JsonObject | None = None
    error: JsonObject | None = None

    def __post_init__(self) -> None:
        if (self.result is None) == (self.error is None):
            raise ValueError("exactly one outcome is required")
        if self.result is not None:
            object.__setattr__(self, "result", _copy(self.result))
        if self.error is not None:
            error = _copy(self.error)
            if not _is_int(error.get("code")) or not isinstance(
                error.get("message"), str
            ):
                raise ValueError("task error requires code and message")
            object.__setattr__(self, "error", error)

    @classmethod
    def success(cls, result: Mapping[str, object]) -> "MCPTaskOutcome":
        """Create a successful operation outcome."""
        return cls(result=_copy(result))

    @classmethod
    def failure(cls, error: Mapping[str, object]) -> "MCPTaskOutcome":
        """Create a failed operation outcome."""
        return cls(error=_copy(error))

    def for_task(self, task_id: str) -> "MCPTaskOutcome":
        """Attach required metadata to a successful task result."""
        if self.error is not None:
            return MCPTaskOutcome.failure(self.error)
        assert self.result is not None
        return MCPTaskOutcome.success(
            with_related_task_metadata(self.result, task_id)
        )


@dataclass(slots=True)
class _Record:
    task_id: str
    requestor: MCPTaskRequestor | None
    request_type: str
    status: MCPTaskStatus
    message: str
    created_at: datetime
    updated_at: datetime
    ttl_ms: int
    poll_interval_ms: int
    cancel_callback: MCPTaskCancellationCallback | None
    outcome: MCPTaskOutcome | None = None
    input_request_id: str | None = None
    changed: Event = field(default_factory=Event)
    cancelled: Event = field(default_factory=Event)


@dataclass(frozen=True, slots=True, kw_only=True)
class MCPTaskCreation:
    """Pair an immediate task result with its background handle."""

    task: JsonObject
    handle: "MCPTaskHandle"

    def as_dict(self) -> JsonObject:
        """Return the immediate CreateTaskResult."""
        return {"task": _copy(self.task)}


@dataclass(frozen=True, slots=True, kw_only=True)
class MCPTaskHandle:
    """Drive the operation associated with one MCP task."""

    _controller: "MCPTaskController" = field(repr=False)
    _cancelled: Event = field(repr=False)
    task_id: str
    requestor: MCPTaskRequestor | None = field(repr=False)

    @property
    def cancellation_requested(self) -> bool:
        """Return whether cancellation or cleanup was requested."""
        return self._cancelled.is_set()

    async def wait_cancelled(self) -> None:
        """Wait for cancellation or cleanup."""
        await self._cancelled.wait()

    async def transition_input_required(
        self,
        request_id: str | None = None,
    ) -> JsonObject:
        """Move from working to input required."""
        return await self._controller.transition_input_required(
            self.task_id,
            requestor=self.requestor,
            request_id=request_id,
        )

    async def transition_working(self) -> JsonObject:
        """Move from input required back to working."""
        return await self._controller.transition_working(
            self.task_id,
            requestor=self.requestor,
        )

    async def complete(self, result: Mapping[str, object]) -> JsonObject:
        """Complete with the underlying result."""
        return await self._controller.complete(
            self.task_id,
            result,
            requestor=self.requestor,
        )

    async def fail(self, error: Mapping[str, object]) -> JsonObject:
        """Fail with the underlying JSON-RPC error."""
        return await self._controller.fail(
            self.task_id,
            error,
            requestor=self.requestor,
        )


class MCPTaskController:
    """Coordinate bounded, requestor-scoped MCP tasks."""

    def __init__(
        self,
        policy: MCPTaskPolicy | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._policy = policy or MCPTaskPolicy()
        self._clock = clock or _utc_now
        self._id_factory = id_factory or _uuid_id
        self._tasks: dict[str, _Record] = {}
        self._expired: dict[str, MCPTaskRequestor | None] = {}
        self._cancellation_tasks: dict[str, Task[None]] = {}
        self._lock = Lock()
        self._closed = False

    def capabilities(
        self,
        *,
        requestor: MCPTaskRequestor | None,
    ) -> MCPTaskCapabilities:
        """Return safe task capabilities for one requestor."""
        return MCPTaskCapabilities.advertise(
            self._policy,
            requestor=requestor,
        )

    def capability_dict(
        self,
        *,
        requestor: MCPTaskRequestor | None,
    ) -> JsonObject:
        """Return an initialize capabilities fragment."""
        tasks = self.capabilities(requestor=requestor).as_dict()
        return {"tasks": tasks} if tasks else {}

    def task_request(
        self,
        params: Mapping[str, object],
        *,
        request_type: str,
        execution_task_support: MCPTaskExecutionMode | None = None,
        requestor: MCPTaskRequestor | None,
    ) -> MCPTaskRequest | None:
        """Parse task augmentation against advertised support."""
        return parse_task_request(
            params,
            request_type=request_type,
            capabilities=self.capabilities(requestor=requestor),
            execution_task_support=execution_task_support,
        )

    async def create(
        self,
        request: MCPTaskRequest,
        *,
        requestor: MCPTaskRequestor | None,
        cancellation_callback: MCPTaskCancellationCallback | None = None,
        task_id: str | None = None,
    ) -> MCPTaskCreation:
        """Create a task before its operation starts."""
        assert isinstance(request, MCPTaskRequest)
        if request.request_type not in _REQUEST_PATHS:
            raise MCPTaskProtocolError(
                code=-32602,
                message="Unsupported task request type.",
            )
        if request.request_type not in self._policy.request_types:
            raise MCPTaskProtocolError(
                code=-32601,
                message=(
                    "Task augmentation is not supported for this request type."
                ),
            )
        if request.requested_ttl_ms is not None and (
            not _is_int(request.requested_ttl_ms)
            or request.requested_ttl_ms <= 0
        ):
            raise MCPTaskProtocolError(
                code=-32602,
                message="Task ttl must be a positive integer.",
            )
        _requestor(requestor)
        assert cancellation_callback is None or callable(cancellation_callback)
        async with self._lock:
            self._ensure_open()
            self._prune()
            actual_id = task_id or self._new_id()
            assert isinstance(actual_id, str) and actual_id
            if actual_id in self._tasks or actual_id in self._expired:
                raise _policy_error(
                    "duplicate_task",
                    "Task creation conflicts with existing state.",
                )
            if (
                sum(r.requestor == requestor for r in self._tasks.values())
                >= self._policy.maximum_tasks_per_requestor
            ):
                raise _policy_error(
                    "capacity",
                    "Task capacity has been reached for this requestor.",
                )
            now = self._now()
            ttl = min(
                request.requested_ttl_ms or self._policy.default_ttl_ms,
                self._policy.maximum_ttl_ms,
            )
            record = _Record(
                task_id=actual_id,
                requestor=requestor,
                request_type=request.request_type,
                status="working",
                message="The operation is in progress.",
                created_at=now,
                updated_at=now,
                ttl_ms=ttl,
                poll_interval_ms=self._policy.poll_interval_ms,
                cancel_callback=cancellation_callback,
            )
            self._tasks[actual_id] = record
            return MCPTaskCreation(
                task=_view(record),
                handle=MCPTaskHandle(
                    _controller=self,
                    _cancelled=record.cancelled,
                    task_id=actual_id,
                    requestor=requestor,
                ),
            )

    async def get(
        self,
        task_id: str,
        *,
        requestor: MCPTaskRequestor | None,
    ) -> JsonObject:
        """Return one owned task."""
        _requestor(requestor)
        async with self._lock:
            return _view(self._owned(task_id, requestor))

    async def list(
        self,
        *,
        requestor: MCPTaskRequestor | None,
        cursor: str | None = None,
    ) -> JsonObject:
        """Return an authenticated, requestor-scoped task page."""
        _requestor(requestor)
        if not self._policy.allow_list:
            raise MCPTaskProtocolError(
                code=-32601,
                message="Task listing is not supported.",
            )
        if requestor is None:
            raise _policy_error(
                "absent_requestor",
                "Task listing requires an authenticated requestor.",
            )
        async with self._lock:
            self._ensure_open()
            self._prune()
            owned = [
                record
                for record in self._tasks.values()
                if record.requestor == requestor
            ]
            start = 0
            if cursor is not None:
                if not isinstance(cursor, str) or not cursor:
                    raise _invalid_cursor()
                positions = [
                    index
                    for index, record in enumerate(owned)
                    if record.task_id == cursor
                ]
                if not positions:
                    if cursor in self._tasks:
                        raise _authorization_error()
                    raise _invalid_cursor()
                start = positions[0] + 1
            records = owned[start : start + self._policy.page_size]
            page: JsonObject = {"tasks": [_view(record) for record in records]}
            if start + len(records) < len(owned) and records:
                page["nextCursor"] = records[-1].task_id
            return page

    async def result(
        self,
        task_id: str,
        *,
        requestor: MCPTaskRequestor | None,
    ) -> MCPTaskOutcome:
        """Block across working and input-required states."""
        _requestor(requestor)
        while True:
            async with self._lock:
                record = self._owned(task_id, requestor)
                if record.status in _TERMINAL:
                    assert record.outcome is not None
                    return record.outcome.for_task(task_id)
                changed = record.changed
                timeout = self._remaining(record)
            try:
                await wait_for(changed.wait(), timeout=timeout)
            except TimeoutError:
                pass

    async def transition_input_required(
        self,
        task_id: str,
        *,
        requestor: MCPTaskRequestor | None,
        request_id: str | None = None,
    ) -> JsonObject:
        """Move a working task to input required."""
        if request_id is not None and (
            not isinstance(request_id, str) or not request_id
        ):
            raise TypeError("request_id must be a non-empty string")
        return await self._transition(
            task_id,
            requestor,
            "working",
            "input_required",
            "Additional input is required.",
            input_request_id=request_id,
        )

    async def transition_working(
        self,
        task_id: str,
        *,
        requestor: MCPTaskRequestor | None,
    ) -> JsonObject:
        """Move an input-required task back to working."""
        return await self._transition(
            task_id,
            requestor,
            "input_required",
            "working",
            "The operation is in progress.",
        )

    async def complete(
        self,
        task_id: str,
        result: Mapping[str, object],
        *,
        requestor: MCPTaskRequestor | None,
    ) -> JsonObject:
        """Complete with the exact operation result."""
        return await self._finish(
            task_id,
            requestor,
            None,
            None,
            MCPTaskOutcome.success(result),
        )

    async def fail(
        self,
        task_id: str,
        error: Mapping[str, object],
        *,
        requestor: MCPTaskRequestor | None,
    ) -> JsonObject:
        """Fail with the exact operation error."""
        return await self._finish(
            task_id,
            requestor,
            "failed",
            "The operation failed.",
            MCPTaskOutcome.failure(error),
        )

    async def cancel(
        self,
        task_id: str,
        *,
        requestor: MCPTaskRequestor | None,
    ) -> JsonObject:
        """Make cancellation sticky before bounded operation cleanup."""
        _requestor(requestor)
        if not self._policy.allow_cancel:
            raise MCPTaskProtocolError(
                code=-32601,
                message="Task cancellation is not supported.",
            )
        callback_task: Task[None] | None = None
        async with self._lock:
            record = self._owned(task_id, requestor)
            if record.status in _TERMINAL:
                raise MCPTaskProtocolError(
                    code=-32602,
                    message="Cannot cancel a task in terminal state.",
                )
            record.status = "cancelled"
            record.message = "The task was cancelled by request."
            record.updated_at = self._now()
            record.outcome = MCPTaskOutcome.failure(
                {
                    "code": -32000,
                    "message": "Request cancelled",
                }
            )
            record.cancelled.set()
            callback = record.cancel_callback
            if callback is not None:
                callback_task = create_task(_invoke_callback(callback))
                self._cancellation_tasks[task_id] = callback_task
            self._notify(record)
            task = _view(record)
        if callback_task is not None:
            await self._bound_cancellation_task(task_id, callback_task)
        return task

    async def cleanup(self) -> int:
        """Remove expired task state."""
        async with self._lock:
            self._ensure_open()
            return self._prune()

    async def cleanup_requestor(
        self,
        requestor: MCPTaskRequestor | None,
    ) -> int:
        """Delete task state for one disconnected requestor."""
        _requestor(requestor)
        async with self._lock:
            task_ids = [
                task_id
                for task_id, record in self._tasks.items()
                if record.requestor == requestor
            ]
            for task_id in task_ids:
                self._delete(task_id)
            self._expired = {
                task_id: owner
                for task_id, owner in self._expired.items()
                if owner != requestor
            }
            return len(task_ids)

    async def close(self) -> None:
        """Delete all state and release waiters."""
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            cancellation_tasks = tuple(self._cancellation_tasks.items())
            for task_id in tuple(self._tasks):
                self._delete(task_id)
            self._expired.clear()
        await self._cancel_and_drain(cancellation_tasks)

    async def _transition(
        self,
        task_id: str,
        requestor: MCPTaskRequestor | None,
        expected: MCPTaskStatus,
        target: MCPTaskStatus,
        message: str,
        *,
        input_request_id: str | None = None,
    ) -> JsonObject:
        _requestor(requestor)
        async with self._lock:
            record = self._owned(task_id, requestor)
            if record.status == "cancelled":
                return _view(record)
            if record.status != expected:
                raise _state_error()
            if input_request_id is not None:
                record.input_request_id = input_request_id
            record.status = target
            record.message = message
            record.updated_at = self._now()
            self._notify(record)
            return _view(record)

    async def _finish(
        self,
        task_id: str,
        requestor: MCPTaskRequestor | None,
        status: Literal["completed", "failed"] | None,
        message: str | None,
        outcome: MCPTaskOutcome,
    ) -> JsonObject:
        _requestor(requestor)
        async with self._lock:
            record = self._owned(task_id, requestor)
            if record.status == "cancelled":
                return _view(record)
            if record.status not in {"working", "input_required"}:
                raise _state_error()
            if outcome.result is not None:
                with_related_task_metadata(outcome.result, task_id)
            if status is None:
                failed = (
                    record.request_type == "tools/call"
                    and outcome.result is not None
                    and outcome.result.get("isError") is True
                )
                status = "failed" if failed else "completed"
                message = (
                    "The operation failed."
                    if failed
                    else "The operation completed."
                )
            assert message is not None
            record.status = status
            record.message = message
            record.updated_at = self._now()
            record.outcome = outcome
            self._notify(record)
            return _view(record)

    def _owned(
        self,
        task_id: str,
        requestor: MCPTaskRequestor | None,
    ) -> _Record:
        self._ensure_open()
        if not isinstance(task_id, str) or not task_id:
            raise _invalid_task()
        self._prune()
        record = self._tasks.get(task_id)
        if record is not None:
            if record.requestor != requestor:
                raise _authorization_error()
            return record
        expired_owner = self._expired.get(task_id, _MISSING)
        if expired_owner is not _MISSING:
            if expired_owner != requestor:
                raise _authorization_error()
            raise MCPTaskProtocolError(
                code=-32602,
                message="Task has expired.",
            )
        raise _invalid_task()

    def _prune(self) -> int:
        now = self._now()
        expired = [
            task_id
            for task_id, record in self._tasks.items()
            if now >= record.created_at + timedelta(milliseconds=record.ttl_ms)
        ]
        for task_id in expired:
            self._expired[task_id] = self._tasks[task_id].requestor
            self._delete(task_id)
        while len(self._expired) > self._policy.expired_tombstone_limit:
            self._expired.pop(next(iter(self._expired)))
        return len(expired)

    def _delete(self, task_id: str) -> None:
        record = self._tasks.pop(task_id)
        record.cancelled.set()
        record.changed.set()
        callback_task = self._cancellation_tasks.get(task_id)
        if callback_task is not None:
            callback_task.cancel()
            self._watch_cancellation_task(task_id, callback_task)

    async def _bound_cancellation_task(
        self,
        task_id: str,
        task: Task[None],
    ) -> None:
        done, _ = await wait(
            {task},
            timeout=self._policy.cancellation_timeout_ms / 1000,
        )
        if task in done:
            self._drain_cancellation_task(task_id, task)
            return
        task.cancel()
        self._watch_cancellation_task(task_id, task)
        _LOGGER.warning("MCP task cancellation callback timed out")

    async def _cancel_and_drain(
        self,
        tasks: tuple[tuple[str, Task[None]], ...],
    ) -> None:
        pending = {task for _, task in tasks if not task.done()}
        for task in pending:
            task.cancel()
        for _ in range(2):
            if not pending:
                break
            done, pending = await wait(
                pending,
                timeout=self._policy.cancellation_timeout_ms / 1000,
            )
            for task_id, task in tasks:
                if task in done:
                    self._drain_cancellation_task(task_id, task)
            for task in pending:
                task.cancel()
        for task_id, task in tasks:
            if task.done():
                self._drain_cancellation_task(task_id, task)
        if pending:
            for task_id, task in tasks:
                if (
                    task in pending
                    and self._cancellation_tasks.get(task_id) is task
                ):
                    self._cancellation_tasks.pop(task_id, None)
                    task.add_done_callback(
                        self._drain_detached_cancellation_task
                    )
            _LOGGER.warning(
                "MCP cancellation callback detached after bounded cleanup"
            )

    def _watch_cancellation_task(
        self,
        task_id: str,
        task: Task[None],
    ) -> None:
        task.add_done_callback(
            lambda completed: self._drain_cancellation_task(
                task_id,
                completed,
            )
        )

    def _drain_cancellation_task(
        self,
        task_id: str,
        task: Task[None],
    ) -> None:
        if self._cancellation_tasks.get(task_id) is not task:
            return
        self._cancellation_tasks.pop(task_id, None)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception:
            _LOGGER.exception("MCP task cancellation callback failed")

    @staticmethod
    def _drain_detached_cancellation_task(task: Task[None]) -> None:
        try:
            task.result()
        except BaseException:
            _LOGGER.warning(
                "Detached MCP cancellation callback ended exceptionally"
            )

    @staticmethod
    def _notify(record: _Record) -> None:
        record.changed.set()
        record.changed = Event()

    def _new_id(self) -> str:
        task_id = self._id_factory()
        if not isinstance(task_id, str) or not task_id:
            raise RuntimeError("task ID factory returned an invalid value")
        return task_id

    def _now(self) -> datetime:
        now = self._clock()
        if not isinstance(now, datetime) or now.tzinfo is None:
            raise RuntimeError("task clock must return an aware datetime")
        return now.astimezone(UTC)

    def _remaining(self, record: _Record) -> float:
        expires = record.created_at + timedelta(milliseconds=record.ttl_ms)
        return max((expires - self._now()).total_seconds(), 0)

    def _ensure_open(self) -> None:
        if self._closed:
            raise MCPTaskProtocolError(
                code=-32603,
                message="Task controller is closed.",
            )


def parse_task_request(
    params: Mapping[str, object],
    *,
    request_type: str,
    capabilities: MCPTaskCapabilities,
    execution_task_support: MCPTaskExecutionMode | None = None,
) -> MCPTaskRequest | None:
    """Parse `params.task` after both negotiation layers."""
    if request_type not in _REQUEST_PATHS:
        raise MCPTaskProtocolError(
            code=-32602,
            message="Unsupported task request type.",
        )
    assert isinstance(capabilities, MCPTaskCapabilities)
    present = "task" in params and params["task"] is not None
    mode: MCPTaskExecutionMode = execution_task_support or "forbidden"
    if execution_task_support not in {
        None,
        "forbidden",
        "optional",
        "required",
    }:
        raise MCPTaskProtocolError(
            code=-32602,
            message="Invalid tool task support.",
        )
    if (
        request_type == "tools/call"
        and capabilities.supports(request_type)
        and mode == "required"
        and not present
    ):
        raise MCPTaskProtocolError(
            code=-32601,
            message="Tool requires task-augmented execution.",
        )
    if not present:
        return None
    if not capabilities.supports(request_type):
        return None
    if request_type == "tools/call" and mode == "forbidden":
        raise MCPTaskProtocolError(
            code=-32601,
            message="Tool does not support task-augmented execution.",
        )
    task = _mapping(params["task"], "Invalid task parameters.")
    ttl = task.get("ttl")
    if ttl is not None and (not _is_int(ttl) or cast(int, ttl) <= 0):
        raise MCPTaskProtocolError(
            code=-32602,
            message="Task ttl must be a positive integer.",
        )
    return MCPTaskRequest(
        request_type=request_type,
        requested_ttl_ms=cast(int | None, ttl),
    )


def with_related_task_metadata(
    payload: Mapping[str, object],
    task_id: str,
) -> JsonObject:
    """Attach related-task metadata without mutating a message."""
    if not isinstance(task_id, str) or not task_id:
        raise _invalid_task()
    value = _copy(payload)
    raw_meta = value.get("_meta")
    if raw_meta is None:
        meta: JsonObject = {}
    elif isinstance(raw_meta, Mapping):
        meta = _copy(cast(Mapping[str, object], raw_meta))
    else:
        raise MCPTaskProtocolError(
            code=-32602,
            message="Invalid related task metadata.",
        )
    existing = meta.get(MCP_RELATED_TASK_METADATA_KEY)
    if existing is not None:
        _require_related_value(existing, task_id)
    meta[MCP_RELATED_TASK_METADATA_KEY] = {"taskId": task_id}
    value["_meta"] = meta
    return value


def require_related_task_metadata(
    payload: Mapping[str, object],
    task_id: str,
) -> None:
    """Require matching metadata on a task-associated message."""
    meta = payload.get("_meta")
    if not isinstance(meta, Mapping):
        raise MCPTaskProtocolError(
            code=-32602,
            message="Related task metadata is required.",
        )
    related = meta.get(MCP_RELATED_TASK_METADATA_KEY)
    if related is None:
        raise MCPTaskProtocolError(
            code=-32602,
            message="Related task metadata is required.",
        )
    _require_related_value(related, task_id)


def without_related_task_metadata(
    payload: Mapping[str, object],
) -> JsonObject:
    """Remove related metadata from task-control traffic."""
    value = _copy(payload)
    raw_meta = value.get("_meta")
    if not isinstance(raw_meta, Mapping):
        return value
    meta = _copy(cast(Mapping[str, object], raw_meta))
    meta.pop(MCP_RELATED_TASK_METADATA_KEY, None)
    if meta:
        value["_meta"] = meta
    else:
        value.pop("_meta", None)
    return value


def _view(record: _Record) -> JsonObject:
    task: JsonObject = {
        "taskId": record.task_id,
        "status": record.status,
        "statusMessage": record.message,
        "createdAt": _timestamp(record.created_at),
        "lastUpdatedAt": _timestamp(record.updated_at),
        "ttl": record.ttl_ms,
        "pollInterval": record.poll_interval_ms,
    }
    if record.input_request_id is not None:
        task["_meta"] = {
            MCP_TASK_INPUT_METADATA_KEY: {
                "kind": (
                    "request"
                    if record.status == "input_required"
                    else "resolution"
                ),
                "request_id": record.input_request_id,
            }
        }
    return task


def _require_related_value(value: object, task_id: str) -> None:
    if not isinstance(value, Mapping):
        raise MCPTaskProtocolError(
            code=-32602,
            message="Invalid related task metadata.",
        )
    related_id = value.get("taskId")
    if not isinstance(related_id, str) or not related_id:
        raise MCPTaskProtocolError(
            code=-32602,
            message="Invalid related task metadata.",
        )
    if related_id != task_id:
        raise _state_error()


def _has_path(value: Mapping[str, object], path: tuple[str, ...]) -> bool:
    current = value
    for name in path:
        child = current.get(name)
        if child is None:
            return False
        current = _mapping(
            child,
            "Invalid MCP task request capabilities.",
        )
    return True


def _set_path(value: JsonObject, path: tuple[str, ...]) -> None:
    current = value
    for name in path:
        child = current.setdefault(name, {})
        assert isinstance(child, dict)
        current = child


def _present_capability(value: Mapping[str, object], name: str) -> bool:
    item = value.get(name)
    if item is None:
        return False
    _mapping(item, "Invalid MCP task capabilities.")
    return True


def _mapping(value: object, message: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise MCPTaskProtocolError(code=-32602, message=message)
    return cast(Mapping[str, object], value)


def _copy(value: Mapping[str, object]) -> JsonObject:
    return cast(JsonObject, deepcopy(dict(value)))


def _requestor(value: MCPTaskRequestor | None) -> None:
    if value is None:
        return
    try:
        hash(value)
    except TypeError as error:
        raise TypeError("requestor must be hashable") from error


def _policy_error(reason: str, message: str) -> MCPTaskProtocolError:
    return MCPTaskProtocolError(
        code=-32602,
        message=message,
        data={"policy": "avalan", "reason": reason},
    )


def _authorization_error() -> MCPTaskProtocolError:
    return _policy_error(
        "authorization",
        "Task is not available to this requestor.",
    )


def _state_error() -> MCPTaskProtocolError:
    return _policy_error(
        "state_mismatch",
        "Task state does not permit this operation.",
    )


def _invalid_task() -> MCPTaskProtocolError:
    return MCPTaskProtocolError(code=-32602, message="Task not found.")


def _invalid_cursor() -> MCPTaskProtocolError:
    return MCPTaskProtocolError(
        code=-32602,
        message="Invalid task cursor.",
    )


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _uuid_id() -> str:
    return str(uuid4())


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


async def _invoke_callback(
    callback: MCPTaskCancellationCallback,
) -> None:
    await callback()
