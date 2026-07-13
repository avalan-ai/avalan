from ...model.stream import (
    CanonicalStreamAccumulator,
    CanonicalStreamItem,
    StreamConsumerProjection,
    StreamItemKind,
    StreamProjectionState,
    StreamReasoningRepresentation,
    StreamReasoningSegment,
    StreamReasoningTruncation,
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
from ..entities import (
    SKILL_CONTENT_REDACTION,
    ModelVisibleServerProtocolTextRedactor,
    ServerOutputRedactionProtocol,
    ServerOutputRedactionSettings,
)

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
from collections.abc import AsyncIterator, Awaitable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
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
PROTOCOL_REASONING_REDACTION_MARKER_CHARACTER_COUNT = len(
    SKILL_CONTENT_REDACTION
)
PROTOCOL_REASONING_REDACTION_MARKER_UTF8_BYTE_COUNT = len(
    SKILL_CONTENT_REDACTION.encode("utf-8")
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
    reasoning_segments: tuple[StreamReasoningSegment, ...] = ()
    reasoning_truncation: StreamReasoningTruncation = field(
        default_factory=StreamReasoningTruncation
    )


@dataclass(frozen=True, kw_only=True, slots=True)
class ProtocolStreamRetentionSettings:
    resource_item_limit: int
    resource_text_byte_limit: int
    task_record_item_limit: int
    task_event_byte_limit: int
    flow_history_item_limit: int
    active_session_lossless: bool
    mcp_reasoning_segment_limit: int = 512
    mcp_reasoning_character_limit: int = 1048576
    mcp_reasoning_text_byte_limit: int = 1048576
    a2a_reasoning_segment_limit: int = 512
    a2a_reasoning_character_limit: int = 1048576
    a2a_reasoning_text_byte_limit: int = 1048576

    def __post_init__(self) -> None:
        for field_name, value in (
            ("mcp_reasoning_segment_limit", self.mcp_reasoning_segment_limit),
            (
                "mcp_reasoning_character_limit",
                self.mcp_reasoning_character_limit,
            ),
            (
                "mcp_reasoning_text_byte_limit",
                self.mcp_reasoning_text_byte_limit,
            ),
            ("a2a_reasoning_segment_limit", self.a2a_reasoning_segment_limit),
            (
                "a2a_reasoning_character_limit",
                self.a2a_reasoning_character_limit,
            ),
            (
                "a2a_reasoning_text_byte_limit",
                self.a2a_reasoning_text_byte_limit,
            ),
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


@dataclass(frozen=True, kw_only=True, slots=True)
class ProtocolReasoningIdentity:
    """Identify one protocol reasoning redaction segment."""

    representation: StreamReasoningRepresentation
    segment_instance_ordinal: int
    provider_item_id: str | None = None
    output_index: int | None = None
    summary_index: int | None = None
    continuation_id: str | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.representation, StreamReasoningRepresentation)
        assert isinstance(self.segment_instance_ordinal, int)
        assert not isinstance(self.segment_instance_ordinal, bool)
        assert self.segment_instance_ordinal >= 0
        for field_name, string_value in (
            ("provider_item_id", self.provider_item_id),
            ("continuation_id", self.continuation_id),
        ):
            if string_value is not None:
                assert isinstance(
                    string_value, str
                ), f"{field_name} must be a string"
                assert string_value.strip(), f"{field_name} must not be empty"
        for field_name, index_value in (
            ("output_index", self.output_index),
            ("summary_index", self.summary_index),
        ):
            if index_value is not None:
                assert isinstance(
                    index_value, int
                ), f"{field_name} must be an integer"
                assert not isinstance(
                    index_value, bool
                ), f"{field_name} must be an integer"
                assert index_value >= 0, f"{field_name} must not be negative"

    @classmethod
    def from_item(
        cls,
        item: CanonicalStreamItem | StreamConsumerProjection,
    ) -> "ProtocolReasoningIdentity":
        """Return the reasoning identity carried by a canonical item."""
        assert isinstance(
            item, (CanonicalStreamItem, StreamConsumerProjection)
        )
        assert item.kind is StreamItemKind.REASONING_DELTA
        assert item.reasoning_representation is not None
        assert item.segment_instance_ordinal is not None
        correlation = item.correlation
        return cls(
            representation=item.reasoning_representation,
            segment_instance_ordinal=item.segment_instance_ordinal,
            provider_item_id=correlation.protocol_item_id,
            output_index=correlation.provider_output_index,
            summary_index=correlation.provider_summary_index,
            continuation_id=correlation.model_continuation_id,
        )

    def loses_correlation_from(
        self,
        previous: "ProtocolReasoningIdentity",
    ) -> bool:
        """Return whether a previously known correlation became unknown."""
        assert isinstance(previous, ProtocolReasoningIdentity)
        return any(
            previous_value is not None and value is None
            for previous_value, value in zip(
                (
                    previous.provider_item_id,
                    previous.output_index,
                    previous.summary_index,
                    previous.continuation_id,
                ),
                (
                    self.provider_item_id,
                    self.output_index,
                    self.summary_index,
                    self.continuation_id,
                ),
                strict=True,
            )
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class ProtocolReasoningRedactedText:
    """Pair protocol-safe reasoning text with its original identity."""

    identity: ProtocolReasoningIdentity
    text: str

    def __post_init__(self) -> None:
        assert isinstance(self.identity, ProtocolReasoningIdentity)
        assert isinstance(self.text, str)
        assert self.text


@dataclass(frozen=True, kw_only=True, slots=True)
class ProtocolReasoningAdmission:
    """Describe capacity required before mutating a reasoning redactor."""

    candidate_character_count: int
    candidate_utf8_byte_count: int
    required_character_count: int
    required_utf8_byte_count: int
    marker_reserved: bool
    suppressed: bool

    def __post_init__(self) -> None:
        for field_name, value in (
            ("candidate_character_count", self.candidate_character_count),
            ("candidate_utf8_byte_count", self.candidate_utf8_byte_count),
            ("required_character_count", self.required_character_count),
            ("required_utf8_byte_count", self.required_utf8_byte_count),
        ):
            assert isinstance(value, int), f"{field_name} must be an integer"
            assert not isinstance(
                value, bool
            ), f"{field_name} must be an integer"
            assert value >= 0, f"{field_name} must not be negative"
        assert isinstance(self.marker_reserved, bool)
        assert isinstance(self.suppressed, bool)


class ProtocolReasoningRedactionState:
    """Preserve reasoning identity across bounded protocol redaction."""

    def __init__(
        self,
        output_redaction_settings: ServerOutputRedactionSettings,
        *,
        protocol: ServerOutputRedactionProtocol,
    ) -> None:
        assert isinstance(
            output_redaction_settings, ServerOutputRedactionSettings
        )
        assert protocol in ("mcp", "a2a")
        self._settings = output_redaction_settings
        self._protocol = protocol
        self._identity: ProtocolReasoningIdentity | None = None
        self._redactor = self._new_redactor()
        self._quarantined = False
        self._quarantine_next_identity = False
        self._redaction_latched = False

    @property
    def identity(self) -> ProtocolReasoningIdentity | None:
        """Return the active reasoning identity, if any."""
        return self._identity

    @property
    def pending_character_count(self) -> int:
        """Return admitted raw characters awaiting a safe decision."""
        return self._redactor.pending_character_count

    @property
    def pending_utf8_byte_count(self) -> int:
        """Return admitted raw bytes awaiting a safe decision."""
        return self._redactor.pending_utf8_byte_count

    @property
    def marker_reserved(self) -> bool:
        """Return whether the fixed marker must remain reserved."""
        return self._redactor.has_pending

    @property
    def redaction_latched(self) -> bool:
        """Return whether all later reasoning text must be suppressed."""
        return self._redaction_latched

    def preview_push(
        self,
        identity: ProtocolReasoningIdentity | None,
        value: str,
    ) -> ProtocolReasoningAdmission:
        """Return required capacity without mutating redaction state."""
        assert identity is None or isinstance(
            identity, ProtocolReasoningIdentity
        )
        assert isinstance(value, str)
        assert value
        candidate_characters = len(value)
        candidate_bytes = len(value.encode("utf-8"))
        if self._redaction_latched:
            return self._admission(
                candidate_characters,
                candidate_bytes,
                suppressed=True,
            )

        current_identity = self._identity
        identity_lost = bool(
            current_identity is not None
            and (
                identity is None
                or (
                    identity is not None
                    and identity.loses_correlation_from(current_identity)
                )
            )
        )
        crossed_pending_boundary = bool(
            current_identity is not None
            and identity != current_identity
            and self._redactor.has_pending
        )
        if crossed_pending_boundary:
            return self._admission(
                candidate_characters,
                candidate_bytes,
                marker_reserved=True,
                suppressed=True,
            )
        if (
            identity is None
            or identity_lost
            or (self._quarantined and identity == current_identity)
            or (current_identity is None and self._quarantine_next_identity)
        ):
            return self._admission(
                candidate_characters,
                candidate_bytes,
                suppressed=True,
            )

        if current_identity is None or current_identity != identity:
            redactor = self._new_redactor()
            existing_pending_characters = 0
            existing_pending_bytes = 0
        else:
            redactor = self._redactor
            existing_pending_characters = self.pending_character_count
            existing_pending_bytes = self.pending_utf8_byte_count
        (
            _chunks,
            preview_pending_characters,
            preview_pending_bytes,
            preview_redacted,
        ) = redactor.preview_push(value)
        reserve_marker = bool(preview_pending_characters or preview_redacted)
        marker_characters = (
            PROTOCOL_REASONING_REDACTION_MARKER_CHARACTER_COUNT
            if reserve_marker
            else 0
        )
        marker_bytes = (
            PROTOCOL_REASONING_REDACTION_MARKER_UTF8_BYTE_COUNT
            if reserve_marker
            else 0
        )
        return ProtocolReasoningAdmission(
            candidate_character_count=candidate_characters,
            candidate_utf8_byte_count=candidate_bytes,
            required_character_count=(
                existing_pending_characters
                + candidate_characters
                + marker_characters
            ),
            required_utf8_byte_count=(
                existing_pending_bytes + candidate_bytes + marker_bytes
            ),
            marker_reserved=reserve_marker,
            suppressed=False,
        )

    def push(
        self,
        identity: ProtocolReasoningIdentity | None,
        value: str,
    ) -> tuple[ProtocolReasoningRedactedText, ...]:
        """Return identity-tagged safe text for one admitted delta."""
        assert identity is None or isinstance(
            identity, ProtocolReasoningIdentity
        )
        assert isinstance(value, str)
        assert value
        if self._redaction_latched:
            return ()

        outputs: list[ProtocolReasoningRedactedText] = []
        current_identity = self._identity
        if identity is None:
            outputs.extend(self._resolve_pending_before_boundary())
            self._reset_segment()
            if not self._redaction_latched:
                self._quarantine_next_identity = True
            return tuple(outputs)

        if current_identity is not None and identity.loses_correlation_from(
            current_identity
        ):
            outputs.extend(self._resolve_pending_before_boundary())
            self._reset_segment()
            if self._redaction_latched:
                return tuple(outputs)
            self._identity = identity
            self._quarantined = True
            return tuple(outputs)

        if current_identity is None:
            self._identity = identity
            self._quarantined = self._quarantine_next_identity
            self._quarantine_next_identity = False
        elif identity != current_identity:
            outputs.extend(self._resolve_pending_before_boundary())
            self._reset_segment()
            if self._redaction_latched:
                return tuple(outputs)
            self._identity = identity

        if self._quarantined:
            return tuple(outputs)
        outputs.extend(
            ProtocolReasoningRedactedText(identity=identity, text=text)
            for text in self._redactor.push(value)
        )
        if any(output.text == SKILL_CONTENT_REDACTION for output in outputs):
            self._redaction_latched = True
        return tuple(outputs)

    def complete(
        self,
        identity: ProtocolReasoningIdentity | None,
    ) -> tuple[ProtocolReasoningRedactedText, ...]:
        """Resolve pending text before one reasoning identity closes."""
        assert identity is None or isinstance(
            identity, ProtocolReasoningIdentity
        )
        if self._redaction_latched:
            self._reset_segment()
            return ()
        if (
            self._identity is None
            and self._quarantine_next_identity
            and identity is not None
        ):
            self._quarantine_next_identity = False
            return ()
        identity_matches = identity is not None and identity == self._identity
        outputs = self._resolve_pending_before_boundary()
        self._reset_segment()
        if not identity_matches and not self._redaction_latched:
            self._quarantine_next_identity = True
        return outputs

    def _admission(
        self,
        candidate_characters: int,
        candidate_bytes: int,
        *,
        marker_reserved: bool = False,
        suppressed: bool,
    ) -> ProtocolReasoningAdmission:
        marker_characters = (
            PROTOCOL_REASONING_REDACTION_MARKER_CHARACTER_COUNT
            if marker_reserved
            else 0
        )
        marker_bytes = (
            PROTOCOL_REASONING_REDACTION_MARKER_UTF8_BYTE_COUNT
            if marker_reserved
            else 0
        )
        return ProtocolReasoningAdmission(
            candidate_character_count=candidate_characters,
            candidate_utf8_byte_count=candidate_bytes,
            required_character_count=marker_characters,
            required_utf8_byte_count=marker_bytes,
            marker_reserved=marker_reserved,
            suppressed=suppressed,
        )

    def _resolve_pending_before_boundary(
        self,
    ) -> tuple[ProtocolReasoningRedactedText, ...]:
        if (
            self._identity is None
            or self._quarantined
            or not self._redactor.has_pending
        ):
            return ()
        self._redaction_latched = True
        return (
            ProtocolReasoningRedactedText(
                identity=self._identity,
                text=SKILL_CONTENT_REDACTION,
            ),
        )

    def _reset_segment(self) -> None:
        self._identity = None
        self._redactor = self._new_redactor()
        self._quarantined = False

    def _new_redactor(self) -> ModelVisibleServerProtocolTextRedactor:
        return ModelVisibleServerProtocolTextRedactor(
            self._settings,
            protocol=self._protocol,
            channel="reasoning",
        )


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
    def reasoning_segments(self) -> tuple[StreamReasoningSegment, ...]:
        return self._accumulator.reasoning_segments

    @property
    def reasoning_truncation(self) -> StreamReasoningTruncation:
        return self._accumulator.reasoning_truncation

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
            reasoning_segments=self.reasoning_segments,
            reasoning_truncation=self.reasoning_truncation,
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


def protocol_stream_usage_mappings(
    usage: object | None,
) -> tuple[Mapping[object, object], ...]:
    if not isinstance(usage, Mapping):
        return ()
    usage_mapping = cast(Mapping[object, object], usage)
    totals = usage.get("totals")
    if isinstance(totals, Mapping):
        return (usage_mapping, cast(Mapping[object, object], totals))
    return (usage_mapping,)


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
        mcp_reasoning_segment_limit=policy.mcp_reasoning_segment_limit,
        mcp_reasoning_character_limit=policy.mcp_reasoning_character_limit,
        mcp_reasoning_text_byte_limit=(policy.mcp_reasoning_text_byte_limit),
        a2a_reasoning_segment_limit=policy.a2a_reasoning_segment_limit,
        a2a_reasoning_character_limit=policy.a2a_reasoning_character_limit,
        a2a_reasoning_text_byte_limit=(policy.a2a_reasoning_text_byte_limit),
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
