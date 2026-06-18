from ...model.stream import (
    CanonicalStreamAccumulator,
    CanonicalStreamItem,
    StreamConsumerProjection,
    StreamItemKind,
    StreamProjectionState,
    StreamRetentionPolicy,
    StreamTerminalOutcome,
    canonical_item_from_consumer_projection,
    project_canonical_stream_item,
)
from ...model.stream import (
    stream_consumer_iterator as _stream_consumer_iterator,
)
from ...model.stream import (
    stream_iterator as _stream_iterator,
)
from ...types import LooseJsonValue

from asyncio import (
    FIRST_COMPLETED,
    CancelledError,
    Task,
    create_task,
    wait,
)
from asyncio import (
    Event as AsyncEvent,
)
from collections.abc import AsyncIterator, Awaitable
from contextlib import suppress
from dataclasses import dataclass
from inspect import isawaitable
from typing import Any, cast

_FLOW_PUBLIC_METADATA_FIELDS = frozenset(
    {
        "event_type",
        "started",
        "finished",
        "elapsed",
        "state",
        "status",
        "attempt",
        "attempts",
        "duration_ms",
        "elapsed_ms",
        "route_kind",
        "edge_kind",
        "edge_index",
        "source",
        "target",
        "output_name",
        "node_count",
        "edge_count",
        "progress",
        "progress_percent",
        "matched",
        "eligible",
        "ready",
        "parent_node_id",
        "child_node_id",
    }
)


@dataclass(frozen=True, kw_only=True, slots=True)
class ProtocolStreamTerminalSnapshot:
    outcome: StreamTerminalOutcome | None
    sequence: int | None
    data: LooseJsonValue | None
    succeeded: bool

    def __post_init__(self) -> None:
        if self.outcome is not None:
            assert isinstance(self.outcome, StreamTerminalOutcome)
        if self.sequence is not None:
            assert isinstance(self.sequence, int)
            assert self.sequence >= 0
        assert isinstance(self.succeeded, bool)


@dataclass(frozen=True, kw_only=True, slots=True)
class ProtocolStreamSnapshot:
    answer_text: str
    reasoning_text: str
    usage: LooseJsonValue | None
    terminal_outcome: StreamTerminalOutcome | None
    terminal_succeeded: bool
    terminal_snapshot: ProtocolStreamTerminalSnapshot
    tool_call_arguments: dict[str, str]
    tool_execution_outputs: dict[str, str]
    diagnostics: tuple[CanonicalStreamItem, ...]
    flow_items: tuple[CanonicalStreamItem, ...]
    usage_items: tuple[CanonicalStreamItem, ...]
    control_items: tuple[CanonicalStreamItem, ...]


@dataclass(frozen=True, kw_only=True, slots=True)
class ProtocolStreamRetentionSettings:
    resource_item_limit: int
    resource_text_byte_limit: int
    task_record_item_limit: int
    task_event_byte_limit: int
    flow_history_item_limit: int
    active_session_lossless: bool

    def __post_init__(self) -> None:
        for field_name, value in (
            ("resource_item_limit", self.resource_item_limit),
            ("resource_text_byte_limit", self.resource_text_byte_limit),
            ("task_record_item_limit", self.task_record_item_limit),
            ("task_event_byte_limit", self.task_event_byte_limit),
            ("flow_history_item_limit", self.flow_history_item_limit),
        ):
            assert isinstance(value, int), f"{field_name} must be an integer"
            assert not isinstance(
                value, bool
            ), f"{field_name} must be an integer"
            assert value >= 0, f"{field_name} must not be negative"
        assert (
            self.resource_text_byte_limit > 0
        ), "resource_text_byte_limit must be positive"
        assert (
            self.task_event_byte_limit >= 2
        ), "task_event_byte_limit must be at least 2"
        assert isinstance(
            self.active_session_lossless, bool
        ), "active_session_lossless must be a boolean"
        assert self.active_session_lossless


class ProtocolStreamAccumulator:
    def __init__(
        self,
        *,
        retention_policy: StreamRetentionPolicy | None = None,
    ) -> None:
        if retention_policy is not None:
            assert isinstance(retention_policy, StreamRetentionPolicy)
        self._accumulator = CanonicalStreamAccumulator(
            retention_policy=retention_policy
        )

    @property
    def answer_text(self) -> str:
        return self._accumulator.answer_text

    @property
    def reasoning_text(self) -> str:
        return self._accumulator.reasoning_text

    @property
    def usage(self) -> LooseJsonValue | None:
        return self._accumulator.final_usage

    @property
    def terminal_outcome(self) -> StreamTerminalOutcome | None:
        return self._accumulator.terminal_outcome

    @property
    def terminal_snapshot(self) -> ProtocolStreamTerminalSnapshot:
        terminal_item = self._accumulator.terminal_item
        if terminal_item is None:
            return protocol_stream_terminal_snapshot(self.terminal_outcome)
        return protocol_stream_terminal_snapshot(
            project_canonical_stream_item(terminal_item)
        )

    @property
    def tool_call_arguments(self) -> dict[str, str]:
        return self._accumulator.tool_call_arguments

    @property
    def tool_execution_outputs(self) -> dict[str, str]:
        return self._accumulator.tool_execution_outputs

    @property
    def diagnostics(self) -> tuple[CanonicalStreamItem, ...]:
        return self._accumulator.diagnostics

    @property
    def flow_items(self) -> tuple[CanonicalStreamItem, ...]:
        return self._accumulator.flow_items

    @property
    def usage_items(self) -> tuple[CanonicalStreamItem, ...]:
        return self._accumulator.usage_items

    @property
    def control_items(self) -> tuple[CanonicalStreamItem, ...]:
        return self._accumulator.control_items

    def add(
        self,
        item: CanonicalStreamItem | StreamConsumerProjection,
    ) -> None:
        assert isinstance(
            item, (CanonicalStreamItem, StreamConsumerProjection)
        )
        canonical_item = (
            canonical_item_from_consumer_projection(item)
            if isinstance(item, StreamConsumerProjection)
            else item
        )
        self._accumulator.add(canonical_item)

    def snapshot(self) -> ProtocolStreamSnapshot:
        terminal_outcome = self.terminal_outcome
        return ProtocolStreamSnapshot(
            answer_text=self.answer_text,
            reasoning_text=self.reasoning_text,
            usage=self.usage,
            terminal_outcome=terminal_outcome,
            terminal_succeeded=stream_terminal_succeeded(terminal_outcome),
            terminal_snapshot=self.terminal_snapshot,
            tool_call_arguments=self.tool_call_arguments,
            tool_execution_outputs=self.tool_execution_outputs,
            diagnostics=self.diagnostics,
            flow_items=self.flow_items,
            usage_items=self.usage_items,
            control_items=self.control_items,
        )

    def validate_complete(self) -> None:
        self._accumulator.validate_complete()


def canonical_flow_public_metadata(
    item: CanonicalStreamItem,
) -> dict[str, LooseJsonValue]:
    assert isinstance(item, CanonicalStreamItem)
    assert item.kind is StreamItemKind.FLOW_EVENT
    return {
        key: value
        for key, value in item.metadata.items()
        if key in _FLOW_PUBLIC_METADATA_FIELDS
    }


ProtocolStreamProjectionState = StreamProjectionState
stream_iterator = _stream_iterator
stream_consumer_iterator = _stream_consumer_iterator


async def cancellable_stream_iterator(
    iterator: AsyncIterator[Any],
    cancel_event: AsyncEvent,
) -> AsyncIterator[Any]:
    assert hasattr(iterator, "__anext__")
    assert isinstance(cancel_event, AsyncEvent)

    while not cancel_event.is_set():
        item_task: Task[Any] = create_task(_pull_stream_item(iterator))
        cancel_task = create_task(cancel_event.wait())
        try:
            done, _ = await wait(
                {item_task, cancel_task}, return_when=FIRST_COMPLETED
            )
            if cancel_task in done or cancel_event.is_set():
                item_task.cancel()
                try:
                    await item_task
                except (CancelledError, StopAsyncIteration):
                    pass
                except Exception:
                    pass
                break
            cancel_task.cancel()
            with suppress(CancelledError):
                await cancel_task
            try:
                item = item_task.result()
            except StopAsyncIteration:
                break
            except Exception:
                if cancel_event.is_set():  # pragma: no cover
                    break
                raise  # pragma: no cover - defensive race guard
            yield item
        finally:
            if not item_task.done():
                item_task.cancel()
                with suppress(CancelledError):
                    await item_task
            if not cancel_task.done():
                cancel_task.cancel()
                with suppress(CancelledError):
                    await cancel_task


async def _pull_stream_item(iterator: AsyncIterator[Any]) -> Any:
    return await anext(iterator)


def stream_terminal_succeeded(
    terminal: StreamConsumerProjection | StreamTerminalOutcome | None,
) -> bool:
    assert terminal is None or isinstance(
        terminal, (StreamConsumerProjection, StreamTerminalOutcome)
    )
    outcome = (
        terminal.terminal_outcome
        if isinstance(terminal, StreamConsumerProjection)
        else terminal
    )
    return outcome is None or outcome is StreamTerminalOutcome.COMPLETED


def protocol_stream_terminal_snapshot(
    terminal: StreamConsumerProjection | StreamTerminalOutcome | None,
) -> ProtocolStreamTerminalSnapshot:
    assert terminal is None or isinstance(
        terminal, (StreamConsumerProjection, StreamTerminalOutcome)
    )
    if isinstance(terminal, StreamConsumerProjection):
        assert terminal.is_stream_terminal
        outcome = terminal.terminal_outcome
        sequence = terminal.sequence
        data = terminal.data
    else:
        outcome = terminal
        sequence = None
        data = None
    return ProtocolStreamTerminalSnapshot(
        outcome=outcome,
        sequence=sequence,
        data=data,
        succeeded=stream_terminal_succeeded(outcome),
    )


def protocol_stream_retention_settings(
    policy: StreamRetentionPolicy | None = None,
) -> ProtocolStreamRetentionSettings:
    assert policy is None or isinstance(policy, StreamRetentionPolicy)
    policy = policy or StreamRetentionPolicy()
    return ProtocolStreamRetentionSettings(
        resource_item_limit=policy.mcp_resource_item_limit,
        resource_text_byte_limit=policy.mcp_resource_text_byte_limit,
        task_record_item_limit=policy.a2a_task_record_item_limit,
        task_event_byte_limit=policy.a2a_task_event_byte_limit,
        flow_history_item_limit=policy.flow_history_item_limit,
        active_session_lossless=policy.active_session_lossless,
    )


async def cleanup_stream_sources(
    *sources: object,
    cancelled: bool = False,
) -> None:
    assert isinstance(cancelled, bool)
    seen: set[int] = set()
    unique_sources: list[object] = []
    for source in sources:
        source_id = id(source)
        if source_id in seen:
            continue
        seen.add(source_id)
        unique_sources.append(source)

    errors: list[BaseException] = []
    if cancelled:
        for source in unique_sources:
            await _call_optional_collecting_errors(source, "cancel", errors)

    for source in unique_sources:
        await _call_optional_collecting_errors(source, "aclose", errors)

    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise BaseExceptionGroup("stream source cleanup failed", errors)


async def _call_optional_collecting_errors(
    source: object, method_name: str, errors: list[BaseException]
) -> None:
    try:
        await _call_optional(source, method_name)
    except (Exception, CancelledError) as exc:
        errors.append(exc)


async def _call_optional(source: object, method_name: str) -> None:
    method = getattr(source, method_name, None)
    if method is None:
        return
    assert callable(method)
    result = method()
    if isawaitable(result):
        awaited_result = await cast(Awaitable[object], result)
        assert awaited_result is None
    else:
        assert result is None
