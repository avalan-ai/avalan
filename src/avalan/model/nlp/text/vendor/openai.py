from .....entities import (
    GenerationSettings,
    Input,
    Message,
    MessageRole,
    PromptCacheRetention,
    ReasoningEffort,
    ReasoningSummaryMode,
    ToolCallDiagnostic,
    ToolCallResult,
)
from .....model.provider import ProviderFamily, provider_string_option
from .....model.reasoning import validate_reasoning_summary_request
from .....model.response.text import TextGenerationResponse
from .....model.stream import (
    REASONING_SEGMENT_BOUNDARY_METADATA_KEY,
    CanonicalStreamItem,
    StreamItemCorrelation,
    StreamItemKind,
    StreamProducerBackend,
    StreamProviderAdapterError,
    StreamProviderCapabilities,
    StreamProviderEvent,
    StreamReasoningRepresentation,
    StreamReasoningSegmentState,
    StreamRetentionPolicy,
    StreamValidationError,
    StreamVisibility,
    TextGenerationSingleStream,
    TextGenerationStream,
)
from .....tool.manager import ToolManager
from .....types import (
    LooseJsonValue,
    assert_non_negative_int,
    assert_non_negative_number,
    assert_positive_number,
)
from .....utils import to_json, tool_call_diagnostic_payload
from ....message import TemplateMessage, TemplateMessageRole
from ....vendor import TextGenerationVendor, TextGenerationVendorStream
from . import (
    DiffusionPipeline,
    PreTrainedModel,
    TextGenerationVendorModel,
)

from asyncio import CancelledError, sleep
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Mapping,
    Sequence,
)
from copy import deepcopy
from dataclasses import dataclass, field
from importlib import import_module
from inspect import isawaitable
from json import dumps
from math import isfinite
from mimetypes import guess_type
from typing import Any, cast
from urllib.parse import urlparse


class _OmitPlaceholder:  # noqa: D101
    pass


Omit: type[Any] = _OmitPlaceholder


class _ReasoningReplayRetentionError(RuntimeError):
    """Report a content-free OpenAI replay admission failure."""

    code = "reasoning_replay_retention_exceeded"

    def __init__(self) -> None:
        super().__init__(
            "OpenAI reasoning replay state is invalid or exceeds its "
            "retention limit."
        )


class _ReplayOwnerAssociationError(RuntimeError):
    """Report an ambiguous continuation without exposing request state."""

    code = "reasoning_replay_owner_ambiguous"

    def __init__(self) -> None:
        super().__init__("OpenAI replay continuation id is ambiguous")


class _OpenAIClientClosedError(RuntimeError):
    """Report a request attempted after OpenAI client shutdown."""

    code = "openai_client_closed"

    def __init__(self) -> None:
        super().__init__("OpenAI client is closed")


class _OpenAICleanupError(RuntimeError):
    """Report content-free OpenAI cleanup failure diagnostics."""

    code = "openai_cleanup_failed"

    def __init__(self, cleanup_target: str) -> None:
        assert cleanup_target in {"client", "response", "stream"}
        self.cleanup_target = cleanup_target
        super().__init__(f"OpenAI {cleanup_target} cleanup failed")


class _OpenAIProviderRequestError(RuntimeError):
    """Report a content-free private-replay provider failure."""

    code = "openai_provider_request_failed"

    def __init__(self) -> None:
        super().__init__("OpenAI provider request failed")


@dataclass(frozen=True, slots=True)
class _ReplayItemAccounting:
    reasoning_items: int = 0
    summary_nodes: int = 0
    summary_characters: int = 0
    summary_serialized_bytes: int = 0


def _assign_replay_json_value(
    target: dict[str, object] | list[object],
    key: str | None,
    value: object,
) -> None:
    if isinstance(target, list):
        assert key is None
        target.append(value)
        return
    assert key is not None
    target[key] = value


def _strict_replay_json_copy(value: object) -> LooseJsonValue:
    root: list[object] = []
    work: list[
        tuple[
            object,
            dict[str, object] | list[object],
            str | None,
            bool,
        ]
    ] = [(value, root, None, False)]
    ancestor_containers: set[int] = set()
    while work:
        source, target, key, exiting = work.pop()
        if exiting:
            ancestor_containers.remove(cast(int, source))
            continue
        if source is None or type(source) in {bool, int, float, str}:
            if type(source) is float and not isfinite(source):
                raise _ReasoningReplayRetentionError()
            _assign_replay_json_value(target, key, source)
            continue
        if type(source) is dict:
            source_id = id(source)
            if source_id in ancestor_containers:
                raise _ReasoningReplayRetentionError()
            ancestor_containers.add(source_id)
            source_mapping = cast(dict[object, object], source)
            normalized_mapping: dict[str, object] = {}
            _assign_replay_json_value(target, key, normalized_mapping)
            items = list(source_mapping.items())
            work.append((source_id, target, key, True))
            for item_key, item_value in reversed(items):
                if type(item_key) is not str:
                    raise _ReasoningReplayRetentionError()
                work.append(
                    (
                        item_value,
                        normalized_mapping,
                        item_key,
                        False,
                    )
                )
            continue
        if type(source) is list:
            source_id = id(source)
            if source_id in ancestor_containers:
                raise _ReasoningReplayRetentionError()
            ancestor_containers.add(source_id)
            normalized_sequence: list[object] = []
            _assign_replay_json_value(target, key, normalized_sequence)
            work.append((source_id, target, key, True))
            for item in reversed(cast(list[object], source)):
                work.append((item, normalized_sequence, None, False))
            continue
        raise _ReasoningReplayRetentionError()
    assert len(root) == 1
    return cast(LooseJsonValue, root[0])


def _clean_replay_input_payload(
    payload: dict[str, object],
) -> dict[str, Any]:
    root: list[object] = []
    work: list[
        tuple[
            object,
            dict[str, object] | list[object],
            str | None,
            bool,
        ]
    ] = [(payload, root, None, True)]
    while work:
        source, target, key, clean_fields = work.pop()
        if isinstance(source, dict):
            normalized_mapping: dict[str, object] = {}
            _assign_replay_json_value(target, key, normalized_mapping)
            item_type = source.get("type") if clean_fields else None
            entries: list[tuple[str, object, bool]] = []
            for item_key, item_value in source.items():
                if clean_fields and (
                    item_key == "status" or item_value is None
                ):
                    continue
                if (
                    clean_fields
                    and item_key == "id"
                    and item_type == "function_call"
                ):
                    continue
                if (
                    clean_fields
                    and item_type == "reasoning"
                    and item_key == "content"
                    and item_value == []
                ):
                    continue
                child_clean_fields = not (
                    clean_fields
                    and item_type == "reasoning"
                    and item_key == "summary"
                )
                entries.append((item_key, item_value, child_clean_fields))
            for item_key, item_value, child_clean_fields in reversed(entries):
                work.append(
                    (
                        item_value,
                        normalized_mapping,
                        item_key,
                        child_clean_fields,
                    )
                )
            continue
        if isinstance(source, list):
            normalized_sequence: list[object] = []
            _assign_replay_json_value(target, key, normalized_sequence)
            for item in reversed(source):
                work.append((item, normalized_sequence, None, clean_fields))
            continue
        _assign_replay_json_value(target, key, source)
    assert len(root) == 1
    assert isinstance(root[0], dict)
    return cast(dict[str, Any], root[0])


def _sanitize_provider_json_payload(
    payload: dict[str, Any],
) -> LooseJsonValue:
    root: list[object] = []
    work: list[
        tuple[
            object,
            dict[str, object] | list[object],
            str | None,
            bool,
        ]
    ] = [(payload, root, None, False)]
    ancestor_containers: set[int] = set()
    while work:
        source, target, key, exiting = work.pop()
        if exiting:
            ancestor_containers.remove(cast(int, source))
            continue
        if source is None or type(source) in {bool, int, float, str}:
            if type(source) is float and not isfinite(source):
                raise _ReasoningReplayRetentionError()
            _assign_replay_json_value(target, key, source)
            continue
        if type(source) is dict:
            source_id = id(source)
            if source_id in ancestor_containers:
                raise _ReasoningReplayRetentionError()
            ancestor_containers.add(source_id)
            source_mapping = cast(dict[object, object], source)
            item_type = source_mapping.get("type")
            normalized_mapping: dict[str, object] = {}
            _assign_replay_json_value(target, key, normalized_mapping)
            work.append((source_id, target, key, True))
            for item_key, item_value in reversed(list(source_mapping.items())):
                if type(item_key) is not str:
                    raise _ReasoningReplayRetentionError()
                if item_key == "encrypted_content":
                    continue
                if item_type == "reasoning" and item_key in {
                    "content",
                    "summary",
                }:
                    continue
                work.append(
                    (
                        item_value,
                        normalized_mapping,
                        item_key,
                        False,
                    )
                )
            continue
        if type(source) is list:
            source_id = id(source)
            if source_id in ancestor_containers:
                raise _ReasoningReplayRetentionError()
            ancestor_containers.add(source_id)
            normalized_sequence: list[object] = []
            _assign_replay_json_value(target, key, normalized_sequence)
            work.append((source_id, target, key, True))
            for item in reversed(cast(list[object], source)):
                work.append((item, normalized_sequence, None, False))
            continue
        raise _ReasoningReplayRetentionError()
    assert len(root) == 1
    return cast(LooseJsonValue, root[0])


def _replay_json_accounting(value: LooseJsonValue) -> tuple[int, int]:
    nodes = 0
    characters = 0
    work: list[object] = [value]
    while work:
        current = work.pop()
        if isinstance(current, dict):
            nodes += len(current)
            characters += sum(len(key) for key in current)
            work.extend(current.values())
        elif isinstance(current, list):
            nodes += len(current)
            work.extend(current)
        else:
            nodes += 1
            if isinstance(current, str):
                characters += len(current)
    return nodes, characters


def _replay_json_serialized_bytes(value: LooseJsonValue) -> int:
    byte_count = 0
    work: list[object] = [value]
    while work:
        current = work.pop()
        if isinstance(current, dict):
            byte_count += 2
            if current:
                byte_count += len(current) - 1
            for key in sorted(current, reverse=True):
                byte_count += _replay_json_scalar_serialized_bytes(key)
                byte_count += 1
                work.append(current[key])
        elif isinstance(current, list):
            byte_count += 2
            if current:
                byte_count += len(current) - 1
            work.extend(reversed(current))
        elif type(current) is int:
            byte_count += _replay_json_integer_serialized_bytes(current)
        else:
            byte_count += _replay_json_scalar_serialized_bytes(current)
    return byte_count


def _replay_json_integer_serialized_bytes(value: int) -> int:
    assert type(value) is int
    magnitude = abs(value)
    if magnitude == 0:
        return 1
    decimal_digits = (((magnitude.bit_length() - 1) * 30103) // 100000) + 1
    threshold = 10**decimal_digits
    while magnitude >= threshold:
        threshold *= 10
        decimal_digits += 1
    while magnitude < threshold // 10:
        threshold //= 10
        decimal_digits -= 1
    return decimal_digits + int(value < 0)


def _replay_json_scalar_serialized_bytes(value: object) -> int:
    serialization_failed = False
    serialized = ""
    try:
        serialized = dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        encoded = serialized.encode("utf-8")
    except (OverflowError, TypeError, UnicodeError, ValueError):
        serialization_failed = True
        encoded = b""
    if serialization_failed:
        raise _ReasoningReplayRetentionError() from None
    return len(encoded)


@dataclass(slots=True, repr=False)
class _OpenAIReplayOwner:
    policy: StreamRetentionPolicy
    _items: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _ledger: list[_ReplayItemAccounting] = field(
        default_factory=list,
        repr=False,
    )
    _reasoning_item_count: int = 0
    _summary_node_count: int = 0
    _summary_character_count: int = 0
    _summary_serialized_byte_count: int = 0
    _attempt_checkpoint: int = 0
    _attempt_active: bool = False
    _released: bool = False
    _release_count: int = 0

    def __post_init__(self) -> None:
        assert isinstance(self.policy, StreamRetentionPolicy)

    def __repr__(self) -> str:
        return (
            "_OpenAIReplayOwner("
            f"item_count={len(self._items)}, "
            f"reasoning_item_count={self._reasoning_item_count}, "
            f"released={self._released})"
        )

    @property
    def released(self) -> bool:
        return self._released

    @property
    def release_count(self) -> int:
        return self._release_count

    @property
    def item_count(self) -> int:
        return len(self._items)

    @property
    def counters(self) -> tuple[int, int, int, int]:
        return (
            self._reasoning_item_count,
            self._summary_node_count,
            self._summary_character_count,
            self._summary_serialized_byte_count,
        )

    def replay_items(self) -> tuple[dict[str, Any], ...]:
        if self._released:
            return ()
        items: list[dict[str, Any]] = []
        for item in self._items:
            copied = _strict_replay_json_copy(item)
            assert isinstance(copied, dict)
            items.append(cast(dict[str, Any], copied))
        return tuple(items)

    def begin_attempt(self) -> None:
        if self._released:
            raise RuntimeError("OpenAI replay owner is released")
        if self._attempt_active:
            return
        self._attempt_checkpoint = len(self._items)
        self._attempt_active = True

    def admit(self, item: dict[str, Any]) -> bool:
        if self._released or not self._attempt_active:
            raise RuntimeError("OpenAI replay owner has no active attempt")
        normalized = OpenAIStream._response_input_item_payload(item)
        item_type = normalized.get("type")
        if item_type == "reasoning":
            if not OpenAIStream._is_replayable_reasoning_item(normalized):
                return False
            accounting = self._reasoning_accounting(normalized)
        elif item_type == "function_call":
            accounting = _ReplayItemAccounting()
        else:
            return False
        self._assert_fits(accounting)
        self._items.append(normalized)
        self._ledger.append(accounting)
        self._apply(accounting, direction=1)
        return True

    def commit_attempt(self) -> None:
        if self._released or not self._attempt_active:
            return
        self._attempt_checkpoint = len(self._items)
        self._attempt_active = False

    def rollback_attempt(self) -> None:
        if self._released or not self._attempt_active:
            return
        while len(self._items) > self._attempt_checkpoint:
            self._items.pop()
            accounting = self._ledger.pop()
            self._apply(accounting, direction=-1)
        self._attempt_active = False

    def release(self) -> None:
        if self._released:
            return
        self._items.clear()
        self._ledger.clear()
        self._reasoning_item_count = 0
        self._summary_node_count = 0
        self._summary_character_count = 0
        self._summary_serialized_byte_count = 0
        self._attempt_checkpoint = 0
        self._attempt_active = False
        self._released = True
        self._release_count += 1

    @staticmethod
    def _reasoning_accounting(
        item: dict[str, Any],
    ) -> _ReplayItemAccounting:
        summary_fields = {
            key: value
            for key, value in item.items()
            if key not in {"encrypted_content", "id", "type"}
        }
        if not summary_fields:
            return _ReplayItemAccounting(reasoning_items=1)
        normalized_summary = _strict_replay_json_copy(summary_fields)
        assert isinstance(normalized_summary, dict)
        nodes, characters = _replay_json_accounting(normalized_summary)
        return _ReplayItemAccounting(
            reasoning_items=1,
            summary_nodes=nodes,
            summary_characters=characters,
            summary_serialized_bytes=_replay_json_serialized_bytes(
                normalized_summary
            ),
        )

    def _assert_fits(self, accounting: _ReplayItemAccounting) -> None:
        limits_and_values = (
            (
                self.policy.openai_replay_reasoning_item_limit,
                self._reasoning_item_count + accounting.reasoning_items,
            ),
            (
                self.policy.openai_replay_reasoning_summary_node_limit,
                self._summary_node_count + accounting.summary_nodes,
            ),
            (
                self.policy.openai_replay_reasoning_summary_character_limit,
                self._summary_character_count + accounting.summary_characters,
            ),
            (
                self.policy.openai_replay_reasoning_summary_serialized_byte_limit,
                self._summary_serialized_byte_count
                + accounting.summary_serialized_bytes,
            ),
        )
        if any(value > limit for limit, value in limits_and_values):
            raise _ReasoningReplayRetentionError()

    def _apply(
        self,
        accounting: _ReplayItemAccounting,
        *,
        direction: int,
    ) -> None:
        assert direction in {-1, 1}
        self._reasoning_item_count += direction * accounting.reasoning_items
        self._summary_node_count += direction * accounting.summary_nodes
        self._summary_character_count += (
            direction * accounting.summary_characters
        )
        self._summary_serialized_byte_count += (
            direction * accounting.summary_serialized_bytes
        )


class OpenAIStream(TextGenerationVendorStream):
    _STREAM_RETRY_MAX_DELAY_SECONDS = 8.0
    _TEXT_DELTA_EVENTS = {"response.text.delta", "response.output_text.delta"}
    _TEXT_DONE_EVENTS = {"response.text.done", "response.output_text.done"}
    _REASONING_DELTA_EVENTS = {"response.reasoning_text.delta"}
    _REASONING_DONE_EVENTS = {"response.reasoning_text.done"}
    _TOOL_CALL_ITEM_TYPES = {
        "custom_tool_call",
        "function_call",
        "tool_call",
    }
    _TOOL_ARGUMENT_DELTA_EVENTS = {
        "response.custom_tool_call_input.delta",
        "response.function_call_arguments.delta",
    }
    _TOOL_ARGUMENT_DONE_EVENTS = {
        "response.custom_tool_call_input.done",
        "response.function_call_arguments.done",
    }
    _ERROR_EVENTS = {"response.error", "response.failed", "error"}
    _CANCELLED_EVENTS = {"response.cancelled", "response.canceled"}
    _INCOMPLETE_EVENTS = {"response.incomplete"}
    _stream: AsyncIterator[Any]
    _canonical_tool_calls: dict[str, dict[str, str | bool | None]]
    _tool_call_ids_by_item_id: dict[str, str]
    _canonical_ready_tool_call_ids: set[str]
    _canonical_done_tool_call_ids: set[str]
    _answer_text_seen: bool
    _answer_done_seen: bool
    _output_item_sink: Callable[[dict[str, Any]], None] | None
    _stream_factory: Callable[[], Awaitable[AsyncIterator[Any]]] | None
    _stream_retry_delay_seconds: float
    _stream_retries: int
    _tool_manager: ToolManager | None
    _last_text_delta_alias_key: tuple[object, ...] | None
    _last_text_delta_alias_event_type: str | None
    _attempt_output_item_count: int
    _output_item_rollback: Callable[[int], None] | None
    _reasoning_segments: StreamReasoningSegmentState
    _replay_owner: _OpenAIReplayOwner | None
    _replay_owner_retainer: (
        Callable[[_OpenAIReplayOwner, tuple[str, ...]], None] | None
    )
    _replay_owner_releaser: Callable[[_OpenAIReplayOwner], None] | None
    _replay_owner_terminal_handled: bool
    _request_has_replay_items: bool

    def __init__(
        self,
        stream: AsyncIterator[Any],
        *,
        provider_family: ProviderFamily | str = ProviderFamily.OPENAI,
        output_item_sink: Callable[[dict[str, Any]], None] | None = None,
        output_item_rollback: Callable[[int], None] | None = None,
        replay_owner: _OpenAIReplayOwner | None = None,
        replay_owner_retainer: (
            Callable[[_OpenAIReplayOwner, tuple[str, ...]], None] | None
        ) = None,
        replay_owner_releaser: (
            Callable[[_OpenAIReplayOwner], None] | None
        ) = None,
        request_has_replay_items: bool = False,
        stream_factory: (
            Callable[[], Awaitable[AsyncIterator[Any]]] | None
        ) = None,
        stream_retry_delay_seconds: float = 0,
        stream_retries: int = 0,
        tool: ToolManager | None = None,
    ) -> None:
        self._stream = stream
        self._canonical_tool_calls = {}
        self._tool_call_ids_by_item_id = {}
        self._canonical_ready_tool_call_ids = set()
        self._canonical_done_tool_call_ids = set()
        self._answer_text_seen = False
        self._answer_done_seen = False
        self._output_item_sink = output_item_sink
        self._output_item_rollback = output_item_rollback
        self._stream_factory = stream_factory
        self._stream_retry_delay_seconds = stream_retry_delay_seconds
        self._stream_retries = stream_retries
        self._tool_manager = tool
        self._last_text_delta_alias_key = None
        self._last_text_delta_alias_event_type = None
        self._attempt_output_item_count = 0
        self._reasoning_segments = StreamReasoningSegmentState()
        self._replay_owner = replay_owner
        self._replay_owner_retainer = replay_owner_retainer
        self._replay_owner_releaser = replay_owner_releaser
        self._replay_owner_terminal_handled = False
        self._request_has_replay_items = request_has_replay_items or bool(
            replay_owner is not None and replay_owner.item_count
        )

        async def generator() -> AsyncIterator[CanonicalStreamItem]:
            async for item in self.canonical_stream(
                stream_session_id=self._DEFAULT_STREAM_SESSION_ID,
                run_id=self._DEFAULT_RUN_ID,
                turn_id=self._DEFAULT_TURN_ID,
            ):
                yield item

        super().__init__(
            generator(),
            provider_family=provider_family,
            sources=(stream,),
        )

    def __aiter__(self) -> AsyncIterator[CanonicalStreamItem]:
        assert self._generator
        return self._generator

    async def __anext__(self) -> CanonicalStreamItem:
        return await super().__anext__()

    async def cancel(self) -> None:
        await self._finish_and_cleanup("cancel")

    async def aclose(self) -> None:
        await self._finish_and_cleanup("aclose")

    async def _finish_and_cleanup(self, method_name: str) -> None:
        assert method_name in {"aclose", "cancel"}
        errors: list[BaseException] = []
        try:
            self._finish_replay_owner(succeeded=False)
        except BaseException as error:
            errors.append(self._cleanup_boundary_error(error))
        try:
            if method_name == "cancel":
                await super().cancel()
            else:
                await super().aclose()
        except BaseException as error:
            errors.append(self._cleanup_boundary_error(error))
        if len(errors) == 1:
            raise errors[0] from None
        if errors:
            raise BaseExceptionGroup(
                "OpenAI stream cleanup failed",
                errors,
            ) from None

    def _cleanup_boundary_error(
        self,
        error: BaseException,
    ) -> BaseException:
        if not self._request_has_replay_items:
            return error
        if isinstance(error, _OpenAIClientClosedError):
            return _OpenAIClientClosedError()
        if isinstance(error, _ReasoningReplayRetentionError):
            return _ReasoningReplayRetentionError()
        if isinstance(error, _ReplayOwnerAssociationError):
            return _ReplayOwnerAssociationError()
        if isinstance(error, _OpenAIProviderRequestError):
            return _OpenAIProviderRequestError()
        if isinstance(error, _OpenAICleanupError):
            return _OpenAICleanupError(error.cleanup_target)
        return _OpenAICleanupError("stream")

    def _cleanup_sources(self) -> tuple[object, ...]:
        return self._stream_sources

    def canonical_stream(
        self,
        *,
        stream_session_id: str,
        run_id: str,
        turn_id: str,
        provider_family: ProviderFamily | str | None = None,
        capabilities: StreamProviderCapabilities | None = None,
        close_after_terminal: bool = True,
    ) -> AsyncIterator[CanonicalStreamItem]:
        self._reset_response_attempt_state()
        return self._provider_canonical_stream(
            self._provider_events(),
            stream_session_id=stream_session_id,
            run_id=run_id,
            turn_id=turn_id,
            provider_family=provider_family,
            capabilities=capabilities
            or StreamProviderCapabilities(
                backend=StreamProducerBackend.HOSTED,
                provider_family=self._provider_family,
                supports_reasoning=True,
                supports_tool_calls=True,
                supports_usage=True,
                supports_terminal_events=True,
                supports_cancellation=True,
            ),
            close_after_terminal=close_after_terminal,
        )

    async def _provider_events(self) -> AsyncIterator[StreamProviderEvent]:
        private_provider_failure = False
        private_cleanup_failure = False
        try:
            try:
                attempts = 0
                while True:
                    retry = False
                    output_seen = False
                    async for event in self._stream:
                        event_type_value = OpenAIClient._response_field(
                            event, "type"
                        )
                        provider_event_type = (
                            event_type_value
                            if isinstance(event_type_value, str)
                            else None
                        )
                        try:
                            provider_events = self._provider_events_from_event(
                                event
                            )
                        except _ReasoningReplayRetentionError as exc:
                            provider_events = (
                                StreamProviderEvent(
                                    kind=StreamItemKind.STREAM_ERRORED,
                                    data={
                                        "error": {
                                            "type": "server_error",
                                            "code": exc.code,
                                            "message": str(exc),
                                        }
                                    },
                                    provider_event_type=provider_event_type,
                                ),
                            )
                        except Exception as exc:
                            if self._request_has_replay_items:
                                provider_events = (
                                    self._private_replay_provider_failure_event(),
                                )
                            else:
                                raise StreamProviderAdapterError(
                                    exc,
                                    provider_payload=self._provider_payload(
                                        event
                                    ),
                                    provider_event_type=provider_event_type,
                                ) from exc
                        if self._request_has_replay_items:
                            self._usage = self._private_replay_usage(
                                self._usage
                            )
                            sanitized_events: list[StreamProviderEvent] = []
                            for provider_event in provider_events:
                                sanitized_event = (
                                    self._sanitize_private_replay_event(
                                        provider_event
                                    )
                                )
                                if (
                                    sanitized_event.kind
                                    is StreamItemKind.USAGE_COMPLETED
                                    and sanitized_event.usage is None
                                ):
                                    continue
                                sanitized_events.append(sanitized_event)
                            provider_events = tuple(sanitized_events)
                        if self._should_retry_stream_failure(
                            event,
                            provider_events,
                            output_seen=output_seen,
                            attempts=attempts,
                        ):
                            retry = True
                            break
                        for provider_event in provider_events:
                            if self._is_model_output_event(provider_event):
                                output_seen = True
                            try:
                                if (
                                    provider_event.kind
                                    is StreamItemKind.STREAM_COMPLETED
                                ):
                                    self._finish_replay_owner(succeeded=True)
                                elif provider_event.kind in {
                                    StreamItemKind.STREAM_CANCELLED,
                                    StreamItemKind.STREAM_ERRORED,
                                }:
                                    self._finish_replay_owner(succeeded=False)
                            except (
                                _OpenAIClientClosedError,
                                _ReasoningReplayRetentionError,
                                _ReplayOwnerAssociationError,
                            ) as exc:
                                yield self._replay_error_event(
                                    exc,
                                    provider_event_type,
                                )
                                return
                            yield provider_event
                            if (
                                provider_event.kind
                                is not StreamItemKind.REASONING_DELTA
                            ):
                                self._reasoning_segments.complete_segment()
                    if not retry:
                        try:
                            self._finish_replay_owner(succeeded=True)
                        except (
                            _OpenAIClientClosedError,
                            _ReasoningReplayRetentionError,
                            _ReplayOwnerAssociationError,
                        ) as exc:
                            yield self._replay_error_event(exc, None)
                        break
                    await self._close_current_stream()
                    self._rollback_response_attempt_output_items()
                    self._reset_response_attempt_state()
                    await self._raise_if_retry_interrupted()
                    assert self._stream_factory is not None
                    delay = min(
                        self._stream_retry_delay_seconds * (2**attempts),
                        max(
                            self._stream_retry_delay_seconds,
                            self._STREAM_RETRY_MAX_DELAY_SECONDS,
                        ),
                    )
                    if delay > 0:
                        await sleep(delay)
                    await self._raise_if_retry_interrupted()
                    attempts += 1
                    stream = await self._stream_factory()
                    await self._raise_if_retry_interrupted(stream)
                    self._stream = stream
                    self._stream_sources = (self._stream,)
            except Exception as error:
                if not self._request_has_replay_items:
                    raise
                private_provider_failure = True
                private_cleanup_failure = isinstance(
                    error,
                    _OpenAICleanupError,
                )
            if private_provider_failure:
                cleanup_failed = False
                try:
                    await self.aclose()
                except BaseException:
                    cleanup_failed = True
                yield self._private_replay_provider_failure_event(
                    cleanup_failed=(cleanup_failed or private_cleanup_failure)
                )
                return
        finally:
            await self.aclose()

    def _should_retry_stream_failure(
        self,
        event: object,
        provider_events: tuple[StreamProviderEvent, ...],
        *,
        output_seen: bool,
        attempts: int,
    ) -> bool:
        if (
            self._stream_factory is None
            or attempts >= self._stream_retries
            or output_seen
            or any(
                self._is_model_output_event(provider_event)
                for provider_event in provider_events
            )
        ):
            return False
        if OpenAIClient._response_field(event, "type") != "response.failed":
            return False
        response = OpenAIClient._response_field(event, "response")
        response_error = OpenAIClient._response_field(response, "error")
        event_error = OpenAIClient._response_field(event, "error")
        if not self._is_retryable_response_failed_error_pair(
            response_error,
            event_error,
        ):
            return False
        return any(
            provider_event.kind is StreamItemKind.STREAM_ERRORED
            for provider_event in provider_events
        )

    @staticmethod
    def _is_retryable_response_failed_error_pair(
        response_error: object,
        event_error: object,
    ) -> bool:
        if response_error is None and event_error is None:
            return True
        present_errors = tuple(
            error
            for error in (response_error, event_error)
            if error is not None
        )
        return bool(present_errors) and all(
            OpenAIClient._response_field(error, "code") == "response_failed"
            for error in present_errors
        )

    @staticmethod
    def _is_model_output_event(event: StreamProviderEvent) -> bool:
        return event.kind in {
            StreamItemKind.ANSWER_DELTA,
            StreamItemKind.ANSWER_DONE,
            StreamItemKind.REASONING_DELTA,
            StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
            StreamItemKind.TOOL_CALL_READY,
            StreamItemKind.TOOL_CALL_DONE,
        }

    def _is_duplicate_text_delta_alias(
        self,
        event: object,
        event_type: str,
        delta: str,
    ) -> bool:
        key = (
            delta,
            OpenAIClient._response_field(event, "item_id"),
            OpenAIClient._response_field(event, "output_index"),
            OpenAIClient._response_field(event, "content_index"),
            OpenAIClient._response_field(event, "sequence_number"),
        )
        duplicate = (
            self._last_text_delta_alias_key == key
            and self._last_text_delta_alias_event_type != event_type
            and self._last_text_delta_alias_event_type
            in self._TEXT_DELTA_EVENTS
        )
        self._last_text_delta_alias_key = key
        self._last_text_delta_alias_event_type = event_type
        return duplicate

    def _reset_response_attempt_state(self) -> None:
        self._canonical_tool_calls = {}
        self._tool_call_ids_by_item_id = {}
        self._canonical_ready_tool_call_ids = set()
        self._canonical_done_tool_call_ids = set()
        self._answer_text_seen = False
        self._answer_done_seen = False
        self._last_text_delta_alias_key = None
        self._last_text_delta_alias_event_type = None
        self._attempt_output_item_count = 0
        self._reasoning_segments = StreamReasoningSegmentState()
        if self._replay_owner is not None and not self._replay_owner.released:
            self._replay_owner.begin_attempt()

    def _rollback_response_attempt_output_items(self) -> None:
        if self._replay_owner is not None and not self._replay_owner.released:
            self._replay_owner.rollback_attempt()
        if self._attempt_output_item_count <= 0:
            return
        rollback = self._output_item_rollback
        if rollback is None:
            return
        rollback(self._attempt_output_item_count)

    def _finish_replay_owner(self, *, succeeded: bool) -> None:
        if self._replay_owner_terminal_handled:
            return
        owner = self._replay_owner
        if owner is None:
            self._replay_owner_terminal_handled = True
            return
        if not succeeded:
            self._replay_owner_terminal_handled = True
            self._release_replay_owner(owner)
            return
        owner.commit_attempt()
        call_ids = tuple(sorted(self._canonical_done_tool_call_ids))
        if not call_ids or self._replay_owner_retainer is None:
            self._replay_owner_terminal_handled = True
            self._release_replay_owner(owner)
            return
        self._replay_owner_retainer(owner, call_ids)
        self._replay_owner_terminal_handled = True

    @staticmethod
    def _private_replay_provider_failure_event(
        *,
        cleanup_failed: bool = False,
    ) -> StreamProviderEvent:
        data: dict[str, object] = {
            "error": {
                "type": "server_error",
                "code": _OpenAIProviderRequestError.code,
                "status": "failed",
                "message": "OpenAI provider request failed",
            }
        }
        if cleanup_failed:
            data["cleanup_error"] = {
                "type": "server_error",
                "code": _OpenAICleanupError.code,
                "message": "OpenAI stream cleanup failed",
            }
        return StreamProviderEvent(
            kind=StreamItemKind.STREAM_ERRORED,
            data=cast(LooseJsonValue, data),
        )

    @staticmethod
    def _sanitize_private_replay_event(
        event: StreamProviderEvent,
    ) -> StreamProviderEvent:
        if event.kind is StreamItemKind.STREAM_CANCELLED:
            data: LooseJsonValue = {
                "error": {
                    "type": "server_error",
                    "code": _OpenAIProviderRequestError.code,
                    "status": "cancelled",
                    "message": "OpenAI provider request cancelled",
                }
            }
        elif event.kind is StreamItemKind.STREAM_ERRORED:
            data = OpenAIStream._private_replay_provider_failure_event().data
        else:
            data = event.data
        safe_event_type = (
            event.provider_event_type
            if event.provider_event_type
            in {
                *OpenAIStream._ERROR_EVENTS,
                *OpenAIStream._CANCELLED_EVENTS,
                *OpenAIStream._INCOMPLETE_EVENTS,
                *OpenAIStream._TEXT_DELTA_EVENTS,
                *OpenAIStream._TEXT_DONE_EVENTS,
                *OpenAIStream._REASONING_DELTA_EVENTS,
                *OpenAIStream._REASONING_DONE_EVENTS,
                *OpenAIStream._TOOL_ARGUMENT_DELTA_EVENTS,
                *OpenAIStream._TOOL_ARGUMENT_DONE_EVENTS,
                "response.completed",
                "response.output_item.added",
                "response.output_item.done",
            }
            else None
        )
        return StreamProviderEvent(
            kind=event.kind,
            text_delta=event.text_delta,
            correlation=event.correlation,
            data=data,
            usage=OpenAIStream._private_replay_usage(event.usage),
            visibility=event.visibility,
            reasoning_representation=event.reasoning_representation,
            segment_instance_ordinal=event.segment_instance_ordinal,
            metadata=event.metadata,
            provider_payload=None,
            provider_event_type=safe_event_type,
        )

    @staticmethod
    def _private_replay_usage(
        usage: object | None,
    ) -> LooseJsonValue | None:
        if usage is None:
            return None
        if type(usage) in {
            bool,
            bytearray,
            bytes,
            float,
            int,
            list,
            str,
            tuple,
        }:
            return None

        def field(value: object, name: str) -> object | None:
            access_failed = False
            result: object | None = None
            try:
                result = OpenAIClient._response_field(value, name)
            except BaseException:
                access_failed = True
            if access_failed:
                raise _ReasoningReplayRetentionError() from None
            return result

        def counter(value: object) -> int | float | None:
            if type(value) is int:
                normalized: int | float = value
            elif type(value) is float:
                normalized = value
            else:
                return None
            if normalized < 0 or (
                isinstance(normalized, float) and not isfinite(normalized)
            ):
                return None
            return normalized

        sanitized: dict[str, object] = {}
        try:
            for name in (
                "cacheCreationInputTokens",
                "cacheReadInputTokens",
                "cacheWriteInputTokens",
                "cache_read_input_tokens",
                "cache_creation_input_tokens",
                "cache_creation_input_token_count",
                "cache_write_input_tokens",
                "cachedContentTokenCount",
                "cached_input_tokens",
                "cached_input_token_count",
                "cached_content_token_count",
                "candidatesTokenCount",
                "candidates_token_count",
                "completion_tokens",
                "inputTokens",
                "input_tokens",
                "input_token_count",
                "outputTokens",
                "output_tokens",
                "output_token_count",
                "promptTokenCount",
                "prompt_tokens",
                "prompt_token_count",
                "reasoningTokens",
                "reasoning_tokens",
                "reasoning_token_count",
                "thoughtsTokenCount",
                "thoughts_token_count",
                "totalTokenCount",
                "totalTokens",
                "total_tokens",
                "total_token_count",
            ):
                value = field(usage, name)
                if value is None:
                    continue
                normalized = counter(value)
                if normalized is None:
                    return None
                sanitized[name] = normalized
            for detail_name, counter_names in (
                (
                    "input_tokens_details",
                    ("audio_tokens", "cached_tokens"),
                ),
                (
                    "prompt_tokens_details",
                    ("cached_tokens",),
                ),
                (
                    "output_tokens_details",
                    (
                        "accepted_prediction_tokens",
                        "audio_tokens",
                        "reasoning_tokens",
                        "rejected_prediction_tokens",
                        "thinking_tokens",
                    ),
                ),
                (
                    "completion_tokens_details",
                    ("reasoning_tokens",),
                ),
            ):
                details = field(usage, detail_name)
                if details is None:
                    continue
                sanitized_details: dict[str, object] = {}
                for name in counter_names:
                    value = field(details, name)
                    if value is None:
                        continue
                    normalized = counter(value)
                    if normalized is None:
                        return None
                    sanitized_details[name] = normalized
                sanitized[detail_name] = sanitized_details
        except _ReasoningReplayRetentionError:
            return None
        return cast(LooseJsonValue, sanitized)

    @staticmethod
    def _replay_error_event(
        error: (
            _OpenAIClientClosedError
            | _ReasoningReplayRetentionError
            | _ReplayOwnerAssociationError
        ),
        provider_event_type: str | None,
    ) -> StreamProviderEvent:
        return StreamProviderEvent(
            kind=StreamItemKind.STREAM_ERRORED,
            data={
                "error": {
                    "type": "server_error",
                    "code": error.code,
                    "message": str(error),
                }
            },
            provider_event_type=provider_event_type,
        )

    def _release_replay_owner(self, owner: _OpenAIReplayOwner) -> None:
        if self._replay_owner_releaser is None:
            owner.release()
            return
        self._replay_owner_releaser(owner)

    async def _close_current_stream(self) -> None:
        cleanup_failed = False
        try:
            await self._call_stream_source_cleanup(self._stream, "aclose")
        except BaseException:
            if not self._request_has_replay_items:
                raise
            cleanup_failed = True
        finally:
            self._stream_sources = ()
        if cleanup_failed:
            raise _OpenAICleanupError("stream") from None

    async def _raise_if_retry_interrupted(
        self,
        stream: AsyncIterator[Any] | None = None,
    ) -> None:
        if not (self._stream_cancelled or self._stream_closed):
            return
        if stream is not None:
            await self._call_stream_source_cleanup(stream, "aclose")
        raise CancelledError()

    async def _call_stream_cleanup(self, method_name: str) -> None:
        assert method_name in ("cancel", "aclose")
        errors: list[Exception] = []
        for source in self._cleanup_sources():
            try:
                await self._call_stream_source_cleanup(source, method_name)
            except Exception as exc:
                errors.append(exc)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("vendor stream cleanup failed", errors)

    @staticmethod
    async def _call_stream_source_cleanup(
        source: object,
        method_name: str,
    ) -> None:
        method_names = (
            ("cancel", "close", "aclose")
            if method_name == "cancel"
            else ("aclose", "close")
        )
        method = None
        for cleanup_method_name in method_names:
            method = getattr(source, cleanup_method_name, None)
            if method is not None:
                break
        else:
            return
        assert callable(method)
        result = method()
        if isawaitable(result):
            awaited_result = await cast(Awaitable[object], result)
            assert awaited_result is None
        else:
            assert result is None

    def _provider_events_from_event(
        self, event: object
    ) -> tuple[StreamProviderEvent, ...]:
        event_type_value = OpenAIClient._response_field(event, "type")
        if event_type_value is not None and not isinstance(
            event_type_value, str
        ):
            raise ValueError("response event type must be a string")
        event_type = event_type_value
        provider_payload = self._provider_payload(event)
        response = OpenAIClient._response_field(event, "response")
        error = OpenAIClient._response_field(
            event, "error"
        ) or OpenAIClient._response_field(response, "error")

        if event_type in self._CANCELLED_EVENTS:
            return (
                StreamProviderEvent(
                    kind=StreamItemKind.STREAM_CANCELLED,
                    data=self._response_event_data(event),
                    provider_payload=provider_payload,
                    provider_event_type=event_type,
                ),
            )
        if event_type in self._ERROR_EVENTS or error is not None:
            data = (
                self._response_failure_data(response)
                if error is None and event_type == "response.failed"
                else self._response_error_data(error or event)
            )
            return (
                StreamProviderEvent(
                    kind=StreamItemKind.STREAM_ERRORED,
                    data=data,
                    provider_payload=provider_payload,
                    provider_event_type=event_type,
                ),
            )
        if event_type in self._INCOMPLETE_EVENTS or (
            self._response_is_incomplete(response)
        ):
            return self._incomplete_events(event, provider_payload, event_type)
        if event_type == "response.completed":
            return self._completion_events(event, provider_payload, event_type)
        if event_type in self._TEXT_DELTA_EVENTS:
            if self._answer_done_seen:
                return ()
            delta = self._response_string_field(event, "delta", event_type)
            if self._is_duplicate_text_delta_alias(
                event,
                event_type,
                delta,
            ):
                return ()
            self._answer_text_seen = True
            return (
                StreamProviderEvent(
                    kind=StreamItemKind.ANSWER_DELTA,
                    text_delta=delta,
                    provider_payload=provider_payload,
                    provider_event_type=event_type,
                ),
            )
        if event_type in self._TEXT_DONE_EVENTS:
            if self._answer_done_seen:
                return ()
            text = self._response_optional_string_field(
                event, event_type, "text", "delta"
            )
            if text and not self._answer_text_seen:
                self._answer_text_seen = True
                self._answer_done_seen = True
                return (
                    StreamProviderEvent(
                        kind=StreamItemKind.ANSWER_DELTA,
                        text_delta=text,
                        provider_payload=provider_payload,
                        provider_event_type=event_type,
                    ),
                    StreamProviderEvent(
                        kind=StreamItemKind.ANSWER_DONE,
                        provider_payload=provider_payload,
                        provider_event_type=event_type,
                    ),
                )
            if not self._answer_text_seen:
                return ()
            self._answer_done_seen = True
            return (
                StreamProviderEvent(
                    kind=StreamItemKind.ANSWER_DONE,
                    provider_payload=provider_payload,
                    provider_event_type=event_type,
                ),
            )
        if event_type in self._REASONING_DELTA_EVENTS:
            delta = self._response_string_field(event, "delta", event_type)
            if not delta:
                return ()
            representation = StreamReasoningRepresentation.NATIVE_TEXT
            correlation = self._reasoning_correlation(event)
            follows_boundary = (
                self._reasoning_segments.next_allocation_follows_boundary
            )
            segment_instance_ordinal = self._reasoning_segments.allocate(
                representation, correlation
            )
            return (
                StreamProviderEvent(
                    kind=StreamItemKind.REASONING_DELTA,
                    text_delta=delta,
                    correlation=correlation,
                    visibility=StreamVisibility.PRIVATE,
                    reasoning_representation=representation,
                    segment_instance_ordinal=segment_instance_ordinal,
                    metadata=(
                        {REASONING_SEGMENT_BOUNDARY_METADATA_KEY: "completed"}
                        if follows_boundary
                        else {}
                    ),
                    provider_payload=provider_payload,
                    provider_event_type=event_type,
                ),
            )
        if event_type in self._REASONING_DONE_EVENTS:
            self._reasoning_segments.complete_segment()
            return ()
        if event_type == "response.output_item.added":
            self._record_output_item(event)
            return ()
        if event_type in self._TOOL_ARGUMENT_DELTA_EVENTS:
            return self._tool_argument_delta_events(
                event, provider_payload, event_type
            )
        if event_type in self._TOOL_ARGUMENT_DONE_EVENTS:
            return self._tool_ready_events(event, provider_payload, event_type)
        if event_type == "response.output_item.done":
            self._record_done_output_item(event)
            item = OpenAIClient._response_field(event, "item")
            if OpenAIClient._response_field(item, "type") == "reasoning":
                self._reasoning_segments.complete_segment()
                return ()
            return self._tool_done_events(event, provider_payload, event_type)
        return ()

    @staticmethod
    def _reasoning_correlation(event: object) -> StreamItemCorrelation:
        item_id = OpenAIClient._response_field(event, "item_id")
        if item_id is not None and (
            not isinstance(item_id, str) or not item_id.strip()
        ):
            raise ValueError(
                "response reasoning item id must be a non-empty string"
            )

        indices: dict[str, int] = {}
        for source_name, target_name in (
            ("output_index", "provider_output_index"),
            ("summary_index", "provider_summary_index"),
        ):
            value = OpenAIClient._response_field(event, source_name)
            if value is None:
                continue
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"response reasoning {source_name} must be a "
                    "non-negative integer"
                )
            indices[target_name] = value
        return StreamItemCorrelation(
            protocol_item_id=item_id,
            provider_output_index=indices.get("provider_output_index"),
            provider_summary_index=indices.get("provider_summary_index"),
        )

    def _record_done_output_item(self, event: object) -> None:
        if self._output_item_sink is None and self._replay_owner is None:
            return
        item = OpenAIClient._response_field(event, "item")
        payload = self._raw_provider_payload(item)
        if not isinstance(payload, dict):
            return
        payload = self._response_input_item_payload(payload)
        item_type = payload.get("type")
        if item_type == "reasoning":
            if not self._is_replayable_reasoning_item(payload):
                return
        elif item_type != "function_call":
            return
        if self._replay_owner is not None and not self._replay_owner.released:
            if self._replay_owner.admit(payload):
                self._request_has_replay_items = True
        if self._output_item_sink is not None:
            self._output_item_sink(payload)
            self._attempt_output_item_count += 1

    @staticmethod
    def _is_replayable_reasoning_item(
        payload: Mapping[str, Any],
    ) -> bool:
        encrypted_content = payload.get("encrypted_content")
        return isinstance(encrypted_content, str) and bool(encrypted_content)

    @staticmethod
    def _response_input_item_payload(
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        normalized = _strict_replay_json_copy(payload)
        assert isinstance(normalized, dict)
        return _clean_replay_input_payload(normalized)

    def _incomplete_events(
        self,
        event: object,
        provider_payload: LooseJsonValue | None,
        event_type: str | None,
    ) -> tuple[StreamProviderEvent, ...]:
        response = OpenAIClient._response_field(event, "response")
        usage = OpenAIClient._response_field(response, "usage")
        result: list[StreamProviderEvent] = []
        if usage is not None:
            self._usage = usage
            result.append(
                StreamProviderEvent(
                    kind=StreamItemKind.USAGE_COMPLETED,
                    usage=cast(LooseJsonValue, usage),
                    provider_payload=provider_payload,
                    provider_event_type=event_type,
                )
            )
        result.append(
            StreamProviderEvent(
                kind=StreamItemKind.STREAM_ERRORED,
                data=self._response_incomplete_data(response or event),
                provider_payload=provider_payload,
                provider_event_type=event_type,
            )
        )
        return tuple(result)

    def _completion_events(
        self,
        event: object,
        provider_payload: LooseJsonValue | None,
        event_type: str,
    ) -> tuple[StreamProviderEvent, ...]:
        response = OpenAIClient._response_field(event, "response")
        usage = OpenAIClient._response_field(response, "usage")
        result: list[StreamProviderEvent] = []
        if not self._answer_text_seen and not self._answer_done_seen:
            text = self._completed_response_text(response)
            if text:
                self._answer_text_seen = True
                self._answer_done_seen = True
                result.extend(
                    (
                        StreamProviderEvent(
                            kind=StreamItemKind.ANSWER_DELTA,
                            text_delta=text,
                            provider_payload=provider_payload,
                            provider_event_type=event_type,
                        ),
                        StreamProviderEvent(
                            kind=StreamItemKind.ANSWER_DONE,
                            provider_payload=provider_payload,
                            provider_event_type=event_type,
                        ),
                    )
                )
        if usage is not None:
            self._usage = usage
            result.append(
                StreamProviderEvent(
                    kind=StreamItemKind.USAGE_COMPLETED,
                    usage=cast(LooseJsonValue, usage),
                    provider_payload=provider_payload,
                    provider_event_type=event_type,
                )
            )
        result.append(
            StreamProviderEvent(
                kind=StreamItemKind.STREAM_COMPLETED,
                provider_payload=provider_payload,
                provider_event_type=event_type,
            )
        )
        return tuple(result)

    def _record_output_item(self, event: object) -> None:
        item = OpenAIClient._response_field(event, "item")
        if not self._is_tool_call_item(item):
            return
        call_id = self._tool_call_id_from_item(item)
        item_id = self._tool_item_id_from_item(item)
        name = self._tool_call_name_from_item(item)
        if call_id is None:
            return
        self._record_tool_call_item_id(item_id, call_id)
        state = self._canonical_tool_calls.setdefault(
            call_id,
            {"name": None, "arguments_seen": False},
        )
        if name is not None:
            state["name"] = name
        if item_id is not None:
            state["protocol_item_id"] = item_id

    def _tool_argument_delta_events(
        self,
        event: object,
        provider_payload: LooseJsonValue | None,
        event_type: str,
    ) -> tuple[StreamProviderEvent, ...]:
        call_id = self._tool_call_id_from_event(event)
        assert call_id is not None
        delta = self._response_string_field(event, "delta", event_type)
        state = self._canonical_tool_calls.setdefault(
            call_id,
            {"name": None, "arguments_seen": False},
        )
        state["arguments_seen"] = True
        return (
            StreamProviderEvent(
                kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                correlation=self._tool_call_correlation(event, call_id),
                text_delta=delta,
                provider_payload=provider_payload,
                provider_event_type=event_type,
            ),
        )

    def _tool_ready_events(
        self,
        event: object,
        provider_payload: LooseJsonValue | None,
        event_type: str,
    ) -> tuple[StreamProviderEvent, ...]:
        call_id = self._tool_call_id_from_event(event)
        assert call_id is not None
        return self._mark_tool_ready(call_id, provider_payload, event_type)

    def _tool_done_events(
        self,
        event: object,
        provider_payload: LooseJsonValue | None,
        event_type: str,
    ) -> tuple[StreamProviderEvent, ...]:
        item = OpenAIClient._response_field(event, "item")
        if item is not None and not self._is_tool_call_item(item):
            return self._message_done_events(
                item, provider_payload, event_type
            )
        call_id = self._tool_call_id_from_item(item)
        item_id = self._tool_item_id_from_item(item)
        item_call_id = OpenAIClient._response_field(item, "call_id")
        if (
            call_id is not None
            and item_id is not None
            and call_id == item_id
            and item_call_id is None
        ):
            call_id = self._tool_call_ids_by_item_id.get(item_id, call_id)
        if call_id is None:
            call_id = self._tool_call_id_from_event(event, required=False)
        if call_id is None:
            return ()
        if item_id is not None and item_id != call_id:
            item_state = self._canonical_tool_calls.get(item_id)
            if item_state is not None and item_state.get("arguments_seen"):
                call_id = item_id
            else:
                self._record_tool_call_item_id(item_id, call_id)
        else:
            self._record_tool_call_item_id(item_id, call_id)
        if call_id not in self._canonical_tool_calls:
            pending_call_ids = [
                pending_call_id
                for pending_call_id, state in (
                    self._canonical_tool_calls.items()
                )
                if state.get("arguments_seen")
                and pending_call_id not in self._canonical_done_tool_call_ids
            ]
            if pending_call_ids:
                raise StreamValidationError(
                    "response tool call item id "
                    f"{call_id} does not match pending tool call item "
                    f"{pending_call_ids[0]}"
                )
        if call_id in self._canonical_done_tool_call_ids:
            raise ValueError("response tool call already completed")
        state = self._canonical_tool_calls.setdefault(
            call_id,
            {
                "name": self._tool_call_name_from_item(item),
                "arguments_seen": False,
            },
        )
        name = self._tool_call_name_from_item(item)
        if name is not None:
            state["name"] = name

        result = list(
            self._tool_argument_from_done_item(
                item, call_id, provider_payload, event_type
            )
        )
        result.extend(
            self._mark_tool_ready(call_id, provider_payload, event_type)
        )
        result.append(
            StreamProviderEvent(
                kind=StreamItemKind.TOOL_CALL_DONE,
                correlation=self._tool_call_correlation(item, call_id),
                provider_payload=provider_payload,
                provider_event_type=event_type,
            )
        )
        self._canonical_done_tool_call_ids.add(call_id)
        return tuple(result)

    def _message_done_events(
        self,
        item: object,
        provider_payload: LooseJsonValue | None,
        event_type: str,
    ) -> tuple[StreamProviderEvent, ...]:
        if self._answer_text_seen or self._answer_done_seen:
            return ()
        text = self._message_done_text(item)
        if not text:
            return ()

        self._answer_text_seen = True
        result = [
            StreamProviderEvent(
                kind=StreamItemKind.ANSWER_DELTA,
                text_delta=text,
                provider_payload=provider_payload,
                provider_event_type=event_type,
            )
        ]
        if not self._answer_done_seen:
            self._answer_done_seen = True
            result.append(
                StreamProviderEvent(
                    kind=StreamItemKind.ANSWER_DONE,
                    provider_payload=provider_payload,
                    provider_event_type=event_type,
                )
            )
        return tuple(result)

    @staticmethod
    def _completed_response_text(response: object) -> str:
        output_text = OpenAIClient._response_field(response, "output_text")
        if isinstance(output_text, str) and output_text:
            return output_text

        parts: list[str] = []
        output = OpenAIClient._response_field(response, "output")
        if isinstance(output, list):
            for item in output:
                item_type = OpenAIClient._response_field(item, "type")
                content = OpenAIClient._response_field(item, "content")
                has_text_content = isinstance(content, list) or isinstance(
                    OpenAIClient._response_field(item, "text"), str
                )
                if item_type in {"message", "output_text"} or (
                    item_type is None and has_text_content
                ):
                    parts.append(OpenAIStream._message_done_text(item))
        return "".join(parts)

    @staticmethod
    def _message_done_text(item: object) -> str:
        direct_text = OpenAIClient._response_field(item, "text")
        if isinstance(direct_text, str) and direct_text:
            return direct_text

        parts: list[str] = []
        contents = OpenAIClient._response_field(item, "content")
        if isinstance(contents, list):
            for content in contents:
                text = OpenAIClient._response_field(content, "text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)

    def _is_tool_call_item(self, item: object) -> bool:
        if item is None:
            return False
        if OpenAIClient._response_field(item, "custom_tool_call") is not None:
            return True
        item_type = OpenAIClient._response_field(item, "type")
        return item_type is None or item_type in self._TOOL_CALL_ITEM_TYPES

    def _tool_argument_from_done_item(
        self,
        item: object,
        call_id: str,
        provider_payload: LooseJsonValue | None,
        event_type: str,
    ) -> tuple[StreamProviderEvent, ...]:
        state = self._canonical_tool_calls.setdefault(
            call_id,
            {"name": None, "arguments_seen": False},
        )
        if state["arguments_seen"]:
            return ()
        arguments = self._tool_call_arguments_from_item(item)
        if arguments is None:
            return ()
        state["arguments_seen"] = True
        return (
            StreamProviderEvent(
                kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                correlation=self._tool_call_correlation(item, call_id),
                text_delta=arguments,
                provider_payload=provider_payload,
                provider_event_type=event_type,
            ),
        )

    def _mark_tool_ready(
        self,
        call_id: str,
        provider_payload: LooseJsonValue | None,
        event_type: str,
    ) -> tuple[StreamProviderEvent, ...]:
        if call_id in self._canonical_ready_tool_call_ids:
            return ()
        self._canonical_ready_tool_call_ids.add(call_id)
        state = self._canonical_tool_calls.setdefault(
            call_id,
            {"name": None, "arguments_seen": False},
        )
        return (
            StreamProviderEvent(
                kind=StreamItemKind.TOOL_CALL_READY,
                correlation=self._state_tool_call_correlation(call_id, state),
                data={"name": state.get("name")},
                provider_payload=provider_payload,
                provider_event_type=event_type,
            ),
        )

    def _tool_call_id_from_event(
        self, event: object, *, required: bool = True
    ) -> str | None:
        for field_name in ("call_id", "id", "item_id"):
            value = OpenAIClient._response_field(event, field_name)
            if value is None:
                continue
            if isinstance(value, str) and value.strip():
                if field_name == "call_id":
                    return value
                return self._tool_call_ids_by_item_id.get(value, value)
            raise ValueError(
                "response tool call id must be a non-empty string"
            )
        if required:
            raise ValueError("response tool call id is missing")
        return None

    def _tool_call_id_from_item(self, item: object) -> str | None:
        if item is None:
            return None
        custom = OpenAIClient._response_field(item, "custom_tool_call")
        for value in (
            OpenAIClient._response_field(custom, "id"),
            OpenAIClient._response_field(item, "call_id"),
            OpenAIClient._response_field(item, "id"),
        ):
            if value is None:
                continue
            if isinstance(value, str) and value.strip():
                return value
            raise ValueError(
                "response tool call id must be a non-empty string"
            )
        return None

    @staticmethod
    def _tool_item_id_from_item(item: object) -> str | None:
        if item is None:
            return None
        for field_name in ("item_id", "id"):
            value = OpenAIClient._response_field(item, field_name)
            if value is None:
                continue
            if isinstance(value, str) and value.strip():
                return value
            raise ValueError("response tool call item id must be a string")
        return None

    def _record_tool_call_item_id(
        self,
        item_id: str | None,
        call_id: str,
    ) -> None:
        assert isinstance(call_id, str)
        assert call_id.strip()
        if item_id is None:
            return
        self._tool_call_ids_by_item_id[item_id] = call_id

    def _tool_call_correlation(
        self,
        source: object,
        call_id: str,
    ) -> StreamItemCorrelation:
        assert isinstance(call_id, str)
        assert call_id.strip()
        item_id = self._tool_item_id_from_item(source)
        return StreamItemCorrelation(
            tool_call_id=call_id,
            protocol_item_id=(
                item_id if item_id is not None and item_id != call_id else None
            ),
        )

    @staticmethod
    def _state_tool_call_correlation(
        call_id: str,
        state: dict[str, str | bool | None],
    ) -> StreamItemCorrelation:
        assert isinstance(call_id, str)
        assert call_id.strip()
        item_id = state.get("protocol_item_id")
        return StreamItemCorrelation(
            tool_call_id=call_id,
            protocol_item_id=(
                item_id
                if isinstance(item_id, str) and item_id != call_id
                else None
            ),
        )

    def _tool_call_name_from_item(self, item: object) -> str | None:
        if item is None:
            return None
        custom = OpenAIClient._response_field(item, "custom_tool_call")
        function = OpenAIClient._response_field(item, "function") or item
        for value in (
            OpenAIClient._response_field(custom, "name"),
            OpenAIClient._response_field(function, "name"),
        ):
            if value is None:
                continue
            if isinstance(value, str):
                try:
                    return TextGenerationVendor.canonical_tool_name(
                        value,
                        tool=self._tool_manager,
                        provider_family=self._provider_family,
                    )
                except AssertionError:
                    return value
            raise ValueError("response tool call name must be a string")
        return None

    def _tool_call_arguments_from_item(self, item: object) -> str | None:
        if item is None:
            return None
        custom = OpenAIClient._response_field(item, "custom_tool_call")
        for value in (
            OpenAIClient._response_field(item, "arguments"),
            OpenAIClient._response_field(custom, "input"),
        ):
            if value is None:
                continue
            if isinstance(value, str):
                return value
            if isinstance(value, Mapping):
                return to_json(value)
            raise ValueError("response tool call arguments must be a string")
        return None

    @staticmethod
    def _response_string_field(
        event: object, field_name: str, event_type: str
    ) -> str:
        value = OpenAIClient._response_field(event, field_name)
        if isinstance(value, str):
            return value
        raise ValueError(f"{event_type} {field_name} must be a string")

    @staticmethod
    def _response_optional_string_field(
        event: object,
        event_type: str,
        *field_names: str,
    ) -> str | None:
        for field_name in field_names:
            value = OpenAIClient._response_field(event, field_name)
            if value is None:
                continue
            if isinstance(value, str):
                return value
            raise ValueError(f"{event_type} {field_name} must be a string")
        return None

    @staticmethod
    def _response_event_data(event: object) -> LooseJsonValue:
        reason = OpenAIClient._response_field(event, "reason")
        if isinstance(reason, str):
            return {"reason": reason}
        return {}

    @staticmethod
    def _response_error_data(error: object) -> LooseJsonValue:
        payload = OpenAIStream._provider_payload(error)
        if isinstance(payload, dict):
            return {"error": payload}
        message = OpenAIClient._response_field(error, "message")
        if isinstance(message, str):
            return {"error": {"message": message}}
        return {"error": {"message": "provider error"}}

    @staticmethod
    def _response_failure_data(response: object) -> LooseJsonValue:
        message = OpenAIClient._response_field(response, "message")
        error: dict[str, LooseJsonValue] = {
            "code": "response_failed",
            "message": (
                message
                if isinstance(message, str) and message
                else "response failed"
            ),
        }
        status = OpenAIClient._response_field(response, "status")
        if isinstance(status, str) and status:
            error["status"] = status
        response_id = OpenAIClient._response_field(response, "id")
        if isinstance(response_id, str) and response_id:
            error["response_id"] = response_id
        return {"error": error}

    @staticmethod
    def _response_is_incomplete(response: object) -> bool:
        status = OpenAIClient._response_field(response, "status")
        return status == "incomplete" or (
            OpenAIClient._response_field(response, "incomplete_details")
            is not None
        )

    @staticmethod
    def _response_incomplete_data(response: object) -> LooseJsonValue:
        details = OpenAIClient._response_field(response, "incomplete_details")
        reason = OpenAIClient._response_field(details, "reason")
        message = "response incomplete"
        error: dict[str, LooseJsonValue] = {
            "code": "response_incomplete",
            "message": message,
        }
        if isinstance(reason, str) and reason:
            error["reason"] = reason
            error["message"] = f"{message}: {reason}"
        status = OpenAIClient._response_field(response, "status")
        if isinstance(status, str) and status:
            error["status"] = status
        response_id = OpenAIClient._response_field(response, "id")
        if isinstance(response_id, str) and response_id:
            error["response_id"] = response_id
        return {"error": error}

    @staticmethod
    def _provider_payload(event: object) -> LooseJsonValue | None:
        try:
            payload = OpenAIStream._raw_provider_payload(event)
            if payload is None:
                return None
            return _sanitize_provider_json_payload(payload)
        except _ReasoningReplayRetentionError:
            return None

    @staticmethod
    def _raw_provider_payload(event: object) -> dict[str, Any] | None:
        if isinstance(event, Mapping):
            try:
                return dict(event)
            except Exception:
                raise _ReasoningReplayRetentionError() from None
        model_dump = getattr(event, "model_dump", None)
        if callable(model_dump):
            try:
                payload = model_dump(mode="json")
            except Exception:
                raise _ReasoningReplayRetentionError() from None
            if isinstance(payload, Mapping):
                try:
                    return dict(payload)
                except Exception:
                    raise _ReasoningReplayRetentionError() from None
        return None


class OpenAIClient(TextGenerationVendor):
    _DEFAULT_MODEL_ID = "default"
    _STREAM_RESPONSE_FAILED_RETRIES = 24
    _STREAM_RESPONSE_FAILED_RETRY_DELAY_SECONDS = 1.0
    _client: Any
    _extra_query: dict[str, str] | None
    _is_azure: bool
    _stream_response_failed_retries: int
    _stream_response_failed_retry_delay_seconds: float
    _stream_retention_policy: StreamRetentionPolicy
    _replay_owners_by_call_id: dict[str, _OpenAIReplayOwner]
    _active_replay_owners: dict[int, _OpenAIReplayOwner]
    _active_replay_streams: dict[int, OpenAIStream]
    _active_replay_call_ids: dict[str, _OpenAIReplayOwner]
    _ambiguous_replay_call_ids: dict[str, None]
    _replay_association_poisoned: bool
    _closed: bool
    _reasoning_summary_provider = "openai"

    def __init__(
        self,
        api_key: str | None,
        base_url: str | None,
        *,
        azure_api_version: str | None = None,
        max_retries: int | None = None,
        stream_response_failed_retries: int = (
            _STREAM_RESPONSE_FAILED_RETRIES
        ),
        stream_response_failed_retry_delay_seconds: int | float = (
            _STREAM_RESPONSE_FAILED_RETRY_DELAY_SECONDS
        ),
        timeout_seconds: int | float | None = None,
        stream_retention_policy: StreamRetentionPolicy | None = None,
    ):
        global Omit

        self._stream_response_failed_retries = (
            self._normalize_response_failed_retries(
                stream_response_failed_retries
            )
        )
        self._stream_response_failed_retry_delay_seconds = (
            self._normalize_response_failed_retry_delay_seconds(
                stream_response_failed_retry_delay_seconds
            )
        )
        self._is_azure = self._is_azure_base_url(base_url)
        self._extra_query = self._azure_extra_query(
            base_url, azure_api_version
        )
        self._stream_retention_policy = (
            stream_retention_policy or StreamRetentionPolicy()
        )
        assert isinstance(self._stream_retention_policy, StreamRetentionPolicy)
        self._replay_owners_by_call_id = {}
        self._active_replay_owners = {}
        self._active_replay_streams = {}
        self._active_replay_call_ids = {}
        self._ambiguous_replay_call_ids = {}
        self._replay_association_poisoned = False
        self._closed = False
        if self._is_azure and api_key is None:
            raise AssertionError(
                "Azure OpenAI Responses requires api-key authentication"
            )

        openai_module = import_module("openai")
        async_openai_type = getattr(openai_module, "AsyncOpenAI")
        client_kwargs: dict[str, Any] = {"base_url": base_url}
        if api_key is None:
            assert base_url
            Omit = cast(type[Any], getattr(openai_module, "Omit"))
            client_kwargs.update(
                api_key="",
                default_headers=cast(Any, {"Authorization": Omit()}),
            )
        else:
            client_kwargs["api_key"] = api_key
        if max_retries is not None:
            client_kwargs["max_retries"] = self._normalize_max_retries(
                max_retries
            )
        if timeout_seconds is not None:
            client_kwargs["timeout"] = self._normalize_timeout_seconds(
                timeout_seconds
            )
        self._client = async_openai_type(**client_kwargs)

    async def __call__(
        self,
        model_id: str,
        messages: list[Message],
        settings: GenerationSettings | None = None,
        *,
        instructions: str | None = None,
        timeout: int | float | None = None,
        tool: ToolManager | None = None,
        use_async_generator: bool = True,
    ) -> TextGenerationStream:
        self._raise_if_closed()
        self._validate_reasoning_summary_request(settings)
        replay_owner = self._replay_owner_for_messages(messages)
        replay_owner.begin_attempt()
        request_has_replay_items = replay_owner.item_count > 0
        try:
            template_messages = self._template_messages(
                messages,
                tool=tool,
                replay_items=replay_owner.replay_items(),
            )
            use_reasoning_profile = self._uses_reasoning_profile(model_id)
            kwargs: dict[str, Any] = {
                "extra_headers": {
                    "X-Title": "Avalan",
                    "HTTP-Referer": "https://github.com/avalan-ai/avalan",
                },
                "model": model_id or self._DEFAULT_MODEL_ID,
                "input": template_messages,
                "store": False,
                "stream": use_async_generator,
            }
            request_client = self._client
            request_timeout = timeout
            request_max_retries: int | None = None
            include_values: list[str] = []
            if instructions is not None:
                assert isinstance(
                    instructions, str
                ), "OpenAI Responses instructions must be a string"
                kwargs["instructions"] = instructions
            if self._extra_query is not None:
                kwargs["extra_query"] = self._extra_query
            if settings:
                if settings.max_new_tokens is not None:
                    kwargs["max_output_tokens"] = settings.max_new_tokens
                if (
                    settings.temperature is not None
                    and not use_reasoning_profile
                ):
                    kwargs["temperature"] = settings.temperature
                if settings.top_p is not None and not use_reasoning_profile:
                    kwargs["top_p"] = settings.top_p
                text = OpenAIClient._text_config(settings)
                if text:
                    kwargs["text"] = text
                reasoning = OpenAIClient._reasoning_config(settings)
                if reasoning:
                    kwargs["reasoning"] = reasoning
                    include_values.append("reasoning.encrypted_content")
                prompt_cache_retention = (
                    OpenAIClient._prompt_cache_retention_config(settings)
                )
                if prompt_cache_retention is not None:
                    kwargs["prompt_cache_retention"] = prompt_cache_retention
                if settings.openai_max_retries is not None:
                    request_max_retries = OpenAIClient._normalize_max_retries(
                        settings.openai_max_retries
                    )
                if settings.openai_timeout_seconds is not None:
                    request_timeout = OpenAIClient._normalize_timeout_seconds(
                        settings.openai_timeout_seconds
                    )
            if request_timeout is not None:
                kwargs["timeout"] = request_timeout
            if request_max_retries is not None:
                request_client = self._client.with_options(
                    max_retries=request_max_retries
                )
            stream_response_failed_retries = (
                OpenAIClient._response_failed_retries(
                    settings,
                    default=self._stream_response_failed_retries,
                )
            )
            stream_response_failed_retry_delay_seconds = (
                OpenAIClient._response_failed_retry_delay_seconds(
                    settings,
                    default=(self._stream_response_failed_retry_delay_seconds),
                )
            )
            if tool:
                schemas = OpenAIClient._tool_schemas(tool)
                if schemas:
                    kwargs["tools"] = schemas
                    if (
                        use_reasoning_profile
                        and "reasoning.encrypted_content" not in include_values
                    ):
                        include_values.append("reasoning.encrypted_content")
                    if settings and settings.tool_choice is not None:
                        kwargs["tool_choice"] = OpenAIClient._tool_choice(
                            settings.tool_choice,
                            schemas,
                            tool=tool,
                        )
            if include_values:
                kwargs["include"] = include_values
            normalized_request_kwargs = _strict_replay_json_copy(kwargs)
            assert isinstance(normalized_request_kwargs, dict)
            request_kwargs = cast(dict[str, Any], normalized_request_kwargs)

            async def create_response() -> Any:
                self._raise_if_closed()
                attempt_kwargs = _strict_replay_json_copy(request_kwargs)
                assert isinstance(attempt_kwargs, dict)
                provider_request_failed = False
                provider_request_cancelled = False
                created_response: Any = None
                try:
                    created_response = await request_client.responses.create(
                        **cast(dict[str, Any], attempt_kwargs)
                    )
                except CancelledError:
                    if not request_has_replay_items:
                        raise
                    provider_request_cancelled = True
                except BaseException:
                    if not request_has_replay_items:
                        raise
                    provider_request_failed = True
                if provider_request_failed:
                    raise _OpenAIProviderRequestError() from None
                if provider_request_cancelled:
                    raise CancelledError() from None
                if getattr(self, "_closed", False):
                    cleanup_failed = False
                    try:
                        await OpenAIStream._call_stream_source_cleanup(
                            created_response,
                            "aclose",
                        )
                    except BaseException:
                        cleanup_failed = True
                    if cleanup_failed:
                        raise BaseExceptionGroup(
                            "OpenAI response cleanup failed",
                            [
                                _OpenAIClientClosedError(),
                                _OpenAICleanupError("response"),
                            ],
                        ) from None
                    raise _OpenAIClientClosedError()
                return created_response

            async def stream_factory() -> AsyncIterator[Any]:
                return cast(AsyncIterator[Any], await create_response())

            client_stream = await create_response()
            if use_async_generator:
                stream_kwargs: dict[str, Any] = {}
                if isinstance(tool, ToolManager):
                    stream_kwargs["tool"] = tool
                if request_has_replay_items:
                    stream_kwargs["request_has_replay_items"] = True
                response_stream = OpenAIStream(
                    stream=cast(AsyncIterator[Any], client_stream),
                    provider_family=self._usage_provider_family.value,
                    replay_owner=replay_owner,
                    replay_owner_retainer=self._retain_replay_owner,
                    replay_owner_releaser=self._discard_replay_owner,
                    stream_factory=stream_factory,
                    stream_retry_delay_seconds=(
                        stream_response_failed_retry_delay_seconds
                    ),
                    stream_retries=stream_response_failed_retries,
                    **stream_kwargs,
                )
                self._register_active_replay_stream(
                    replay_owner,
                    response_stream,
                )
                return response_stream

            non_stream_adapter_failed = False
            response: TextGenerationSingleStream | None = None
            try:
                content = OpenAIClient._non_stream_response_content(
                    client_stream,
                    tool=tool,
                    provider_family=self._usage_provider_family,
                )
                response_usage = OpenAIClient._response_field(
                    client_stream,
                    "usage",
                )
                if request_has_replay_items:
                    response_usage = OpenAIStream._private_replay_usage(
                        response_usage
                    )
                response = TextGenerationSingleStream(
                    content,
                    provider_family=self._usage_provider_family,
                    usage=response_usage,
                )
            except BaseException:
                if not request_has_replay_items:
                    raise
                non_stream_adapter_failed = True
            if non_stream_adapter_failed:
                raise _OpenAIProviderRequestError() from None
            assert response is not None
        except BaseException:
            self._discard_replay_owner(replay_owner)
            raise
        self._discard_replay_owner(replay_owner)
        return response

    @property
    def _usage_provider_family(self) -> ProviderFamily:
        return (
            ProviderFamily.AZURE_OPENAI
            if self._is_azure
            else ProviderFamily.OPENAI
        )

    @property
    def reasoning_summary_provider(self) -> str:
        """Return the configured OpenAI client provider family."""
        compatible_provider = super().reasoning_summary_provider
        if compatible_provider != "openai":
            return compatible_provider
        if self._is_azure:
            return "azure_openai"
        return "openai"

    async def aclose(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        errors: list[BaseException] = []
        streams = tuple(self._active_replay_stream_registry().values())
        for stream in streams:
            try:
                await stream.aclose()
            except BaseException:
                errors.append(_OpenAICleanupError("stream"))
        owners = [
            *self._active_replay_owner_registry().values(),
            *self._replay_owner_registry().values(),
        ]
        released_owner_ids: set[int] = set()
        for owner in owners:
            owner_id = id(owner)
            if owner_id in released_owner_ids:
                continue
            released_owner_ids.add(owner_id)
            owner.release()
        self._active_replay_owner_registry().clear()
        self._active_replay_stream_registry().clear()
        self._active_replay_call_id_registry().clear()
        self._replay_owner_registry().clear()
        self._ambiguous_replay_ids().clear()
        self._replay_association_poisoned = False
        close = getattr(self._client, "close", None)
        if close is None:
            close = getattr(self._client, "aclose", None)
        if close is not None:
            assert callable(close)
            try:
                result = close()
                if isawaitable(result):
                    awaited_result = await cast(Awaitable[object], result)
                    assert awaited_result is None
                else:
                    assert result is None
            except BaseException:
                errors.append(_OpenAICleanupError("client"))
        if len(errors) == 1:
            raise errors[0] from None
        if errors:
            raise BaseExceptionGroup(
                "OpenAI client cleanup failed",
                errors,
            ) from None

    def _raise_if_closed(self) -> None:
        if getattr(self, "_closed", False):
            raise _OpenAIClientClosedError()

    @staticmethod
    def _response_failed_retries(
        settings: GenerationSettings | None,
        *,
        default: int,
    ) -> int:
        if settings is None or settings.openai_response_failed_retries is None:
            return default
        return OpenAIClient._normalize_response_failed_retries(
            settings.openai_response_failed_retries
        )

    @staticmethod
    def _response_failed_retry_delay_seconds(
        settings: GenerationSettings | None,
        *,
        default: float,
    ) -> float:
        if (
            settings is None
            or settings.openai_response_failed_retry_delay_seconds is None
        ):
            return default
        return OpenAIClient._normalize_response_failed_retry_delay_seconds(
            settings.openai_response_failed_retry_delay_seconds
        )

    @staticmethod
    def _normalize_response_failed_retries(value: object) -> int:
        assert_non_negative_int(value, "openai_response_failed_retries")
        assert isinstance(value, int)
        return value

    @staticmethod
    def _normalize_max_retries(value: object) -> int:
        assert_non_negative_int(value, "openai_max_retries")
        assert isinstance(value, int)
        return value

    @staticmethod
    def _normalize_response_failed_retry_delay_seconds(
        value: object,
    ) -> float:
        assert_non_negative_number(
            value,
            "openai_response_failed_retry_delay_seconds",
        )
        assert isinstance(value, int | float)
        return float(value)

    @staticmethod
    def _normalize_timeout_seconds(value: object) -> float:
        assert_positive_number(value, "openai_timeout_seconds")
        assert isinstance(value, int | float)
        return float(value)

    @staticmethod
    def _provider_response_failed_retries(
        provider_options: Mapping[str, object],
    ) -> int:
        return OpenAIClient._normalize_response_failed_retries(
            provider_options["openai_response_failed_retries"]
        )

    @staticmethod
    def _provider_max_retries(
        provider_options: Mapping[str, object],
    ) -> int:
        return OpenAIClient._normalize_max_retries(
            provider_options["openai_max_retries"]
        )

    @staticmethod
    def _provider_retry_delay_seconds(
        provider_options: Mapping[str, object],
    ) -> float:
        return OpenAIClient._normalize_response_failed_retry_delay_seconds(
            provider_options["openai_response_failed_retry_delay_seconds"]
        )

    @staticmethod
    def _provider_timeout_seconds(
        provider_options: Mapping[str, object],
    ) -> float:
        return OpenAIClient._normalize_timeout_seconds(
            provider_options["openai_timeout_seconds"]
        )

    def _replay_owner_for_messages(
        self,
        messages: list[Message],
    ) -> _OpenAIReplayOwner:
        self._raise_if_closed()
        registry = self._replay_owner_registry()
        call_ids = self._tool_result_call_ids(messages)
        if call_ids and self._replay_association_poisoned:
            raise _ReplayOwnerAssociationError()
        ambiguous_call_ids = self._ambiguous_replay_ids()
        ambiguous_matches = [
            call_id for call_id in call_ids if call_id in ambiguous_call_ids
        ]
        if ambiguous_matches:
            raise _ReplayOwnerAssociationError()
        active_registry = self._active_replay_call_id_registry()
        active_matches: list[_OpenAIReplayOwner] = []
        for call_id in call_ids:
            owner = active_registry.get(call_id)
            if owner is not None and all(
                existing is not owner for existing in active_matches
            ):
                active_matches.append(owner)
        if active_matches:
            ambiguity_call_ids = self._ambiguity_call_ids(
                call_ids,
                active_matches,
            )
            for owner in active_matches:
                self._dissociate_replay_owner(owner)
                owner.release()
            for call_id in ambiguity_call_ids:
                self._record_ambiguous_replay_call_id(call_id)
            raise _ReplayOwnerAssociationError()
        matching_owners: list[_OpenAIReplayOwner] = []
        for call_id in call_ids:
            owner = registry.get(call_id)
            if owner is not None and all(
                existing is not owner for existing in matching_owners
            ):
                matching_owners.append(owner)
        if len(matching_owners) > 1:
            ambiguity_call_ids = self._ambiguity_call_ids(
                call_ids,
                matching_owners,
            )
            for owner in matching_owners:
                self._discard_replay_owner(owner)
            for call_id in ambiguity_call_ids:
                self._record_ambiguous_replay_call_id(call_id)
            raise _ReplayOwnerAssociationError()
        if matching_owners:
            owner = matching_owners[0]
            owner_call_ids = tuple(
                call_id
                for call_id, registered_owner in registry.items()
                if registered_owner is owner
            )
            try:
                self._activate_replay_owner(owner)
            except BaseException:
                self._dissociate_replay_owner(owner)
                raise
            for call_id in owner_call_ids:
                del registry[call_id]
                active_registry[call_id] = owner
            return owner
        owner = _OpenAIReplayOwner(self._stream_retention_policy)
        self._activate_replay_owner(owner)
        return owner

    def _replay_owner_registry(self) -> dict[str, _OpenAIReplayOwner]:
        return self._replay_owners_by_call_id

    def _active_replay_owner_registry(
        self,
    ) -> dict[int, _OpenAIReplayOwner]:
        return self._active_replay_owners

    def _active_replay_stream_registry(self) -> dict[int, OpenAIStream]:
        return self._active_replay_streams

    def _active_replay_call_id_registry(
        self,
    ) -> dict[str, _OpenAIReplayOwner]:
        return self._active_replay_call_ids

    def _register_active_replay_stream(
        self,
        owner: _OpenAIReplayOwner,
        stream: OpenAIStream,
    ) -> None:
        assert self._active_replay_owner_registry().get(id(owner)) is owner
        self._active_replay_stream_registry()[id(owner)] = stream

    def _activate_replay_owner(self, owner: _OpenAIReplayOwner) -> None:
        registry = self._active_replay_owner_registry()
        owner_id = id(owner)
        assert owner_id not in registry
        limit = max(1, self._stream_retention_policy.replay_history_item_limit)
        if len(registry) >= limit:
            owner.release()
            raise _ReasoningReplayRetentionError()
        registry[owner_id] = owner

    def _deactivate_replay_owner(self, owner: _OpenAIReplayOwner) -> None:
        self._active_replay_owner_registry().pop(id(owner), None)
        self._active_replay_stream_registry().pop(id(owner), None)

    def _ambiguous_replay_ids(self) -> dict[str, None]:
        return self._ambiguous_replay_call_ids

    def _record_ambiguous_replay_call_id(self, call_id: str) -> None:
        call_ids = self._ambiguous_replay_ids()
        limit = max(
            1,
            self._stream_retention_policy.replay_history_item_limit,
        )
        if call_id in call_ids:
            return
        if len(call_ids) >= limit:
            self._replay_association_poisoned = True
            return
        call_ids[call_id] = None

    @staticmethod
    def _tool_result_call_ids(messages: list[Message]) -> tuple[str, ...]:
        call_ids: list[str] = []
        for message in messages:
            if message.role != MessageRole.TOOL:
                continue
            outcome = (
                message.tool_call_result
                or message.tool_call_error
                or message.tool_call_diagnostic
            )
            if isinstance(outcome, ToolCallDiagnostic):
                call_id_value = outcome.call_id
            else:
                call = getattr(outcome, "call", None)
                call_id_value = getattr(call, "id", None)
            if call_id_value is None:
                continue
            call_id = str(call_id_value)
            if call_id and call_id not in call_ids:
                call_ids.append(call_id)
        return tuple(call_ids)

    def _retain_replay_owner(
        self,
        owner: _OpenAIReplayOwner,
        call_ids: tuple[str, ...],
    ) -> None:
        assert isinstance(owner, _OpenAIReplayOwner)
        assert call_ids
        assert not owner.released
        if getattr(self, "_closed", False):
            self._dissociate_replay_owner(owner)
            owner.release()
            raise _OpenAIClientClosedError()
        if self._replay_association_poisoned:
            self._dissociate_replay_owner(owner)
            owner.release()
            raise _ReplayOwnerAssociationError()
        registry = self._replay_owner_registry()
        active_registry = self._active_replay_call_id_registry()
        ambiguous_call_ids = self._ambiguous_replay_ids()
        retained_collisions = [
            registry[call_id]
            for call_id in call_ids
            if call_id in registry and registry[call_id] is not owner
        ]
        active_collisions = [
            active_registry[call_id]
            for call_id in call_ids
            if call_id in active_registry
            and active_registry[call_id] is not owner
        ]
        if (
            retained_collisions
            or active_collisions
            or any(call_id in ambiguous_call_ids for call_id in call_ids)
        ):
            ambiguity_call_ids = self._ambiguity_call_ids(
                call_ids,
                [owner, *retained_collisions, *active_collisions],
            )
            colliding_owner_ids: set[int] = set()
            for colliding_owner in retained_collisions:
                if id(colliding_owner) in colliding_owner_ids:
                    continue
                colliding_owner_ids.add(id(colliding_owner))
                self._discard_replay_owner(colliding_owner)
            for colliding_owner in active_collisions:
                if id(colliding_owner) in colliding_owner_ids:
                    continue
                colliding_owner_ids.add(id(colliding_owner))
                self._dissociate_replay_owner(colliding_owner)
                colliding_owner.release()
            for call_id in ambiguity_call_ids:
                self._record_ambiguous_replay_call_id(call_id)
            self._dissociate_replay_owner(owner)
            owner.release()
            raise _ReplayOwnerAssociationError()
        limit = max(1, self._stream_retention_policy.replay_history_item_limit)
        owner_active_call_id_count = sum(
            registered_owner is owner
            for registered_owner in active_registry.values()
        )
        if (
            len(registry)
            + len(active_registry)
            - owner_active_call_id_count
            + len(call_ids)
            > limit
        ):
            self._dissociate_replay_owner(owner)
            owner.release()
            raise _ReasoningReplayRetentionError()
        self._deactivate_replay_owner(owner)
        self._dissociate_replay_owner(owner)
        for call_id in call_ids:
            assert isinstance(call_id, str) and call_id
            registry[call_id] = owner

    def _discard_replay_owner(self, owner: _OpenAIReplayOwner) -> None:
        self._deactivate_replay_owner(owner)
        self._dissociate_replay_owner(owner)
        owner.release()

    def _dissociate_replay_owner(self, owner: _OpenAIReplayOwner) -> None:
        for registry in (
            self._replay_owner_registry(),
            self._active_replay_call_id_registry(),
        ):
            for call_id in tuple(registry):
                if registry[call_id] is owner:
                    del registry[call_id]

    def _ambiguity_call_ids(
        self,
        incoming_call_ids: tuple[str, ...],
        owners: Sequence[_OpenAIReplayOwner],
    ) -> tuple[str, ...]:
        call_ids = list(incoming_call_ids)
        owner_ids = {id(owner) for owner in owners}
        for registry in (
            self._replay_owner_registry(),
            self._active_replay_call_id_registry(),
        ):
            for call_id, registered_owner in registry.items():
                if (
                    id(registered_owner) in owner_ids
                    and call_id not in call_ids
                ):
                    call_ids.append(call_id)
        return tuple(call_ids)

    def _template_messages(
        self,
        messages: list[Message],
        exclude_roles: list[TemplateMessageRole] | None = None,
        *,
        replay_items: tuple[dict[str, Any], ...] = (),
        tool: ToolManager | None = None,
    ) -> list[TemplateMessage] | list[dict[str, Any]]:
        tool_messages = [
            message
            for message in messages
            if message.role == MessageRole.TOOL
            and (
                message.tool_call_result
                or message.tool_call_error
                or message.tool_call_diagnostic
            )
        ]
        do_exclude_roles = [*(exclude_roles or []), "tool"]
        template_messages = super()._template_messages(
            [
                message
                for message in messages
                if not (
                    message.role == MessageRole.ASSISTANT
                    and message.content is None
                    and message.tool_calls
                )
            ],
            do_exclude_roles,
        )
        messages_out = cast(list[dict[str, Any]], template_messages)
        for message in messages_out:
            if "content" not in message:
                continue
            content = message.get("content")
            if isinstance(content, list):
                message["content"] = [
                    OpenAIClient._content_block(block)
                    for block in content
                    if isinstance(block, dict)
                ]
        response_item_messages = self._response_item_tool_messages(
            tool_messages,
            replay_items=replay_items,
            tool=tool,
        )
        if response_item_messages is not None:
            messages_out.extend(response_item_messages)
            return messages_out

        for tool_message in tool_messages:
            messages_out.extend(
                self._synthetic_tool_messages(tool_message, tool=tool)
            )
        return messages_out

    def _response_item_tool_messages(
        self,
        tool_messages: list[Message],
        *,
        replay_items: tuple[dict[str, Any], ...] = (),
        tool: ToolManager | None = None,
    ) -> list[dict[str, Any]] | None:
        if not replay_items:
            return None

        synthetic_records: list[tuple[str | None, list[dict[str, Any]]]] = []
        outputs_by_call_id: dict[str, dict[str, Any]] = {}
        for tool_message in tool_messages:
            synthetic = self._synthetic_tool_messages(
                tool_message,
                tool=tool,
            )
            call_id: str | None = None
            if len(synthetic) == 2:
                synthetic_call_id = synthetic[0].get("call_id")
                if isinstance(synthetic_call_id, str) and synthetic_call_id:
                    call_id = synthetic_call_id
                    outputs_by_call_id[call_id] = synthetic[1]
            synthetic_records.append((call_id, synthetic))

        if not synthetic_records:
            return []

        messages: list[dict[str, Any]] = []
        matched_call_ids: set[str] = set()
        for response_item in replay_items:
            item_type = response_item.get("type")
            if item_type == "reasoning":
                if OpenAIStream._is_replayable_reasoning_item(response_item):
                    copied = _strict_replay_json_copy(response_item)
                    assert isinstance(copied, dict)
                    messages.append(cast(dict[str, Any], copied))
                continue
            if item_type != "function_call":
                continue
            call_id = response_item.get("call_id")
            if not isinstance(call_id, str):
                continue
            result_message = outputs_by_call_id.get(call_id)
            if result_message is not None:
                copied = _strict_replay_json_copy(response_item)
                assert isinstance(copied, dict)
                messages.append(cast(dict[str, Any], copied))
                messages.append(deepcopy(result_message))
                matched_call_ids.add(call_id)

        for call_id, synthetic in synthetic_records:
            if call_id is not None and call_id in matched_call_ids:
                continue
            messages.extend(deepcopy(synthetic))
        return messages

    def _synthetic_tool_messages(
        self,
        tool_message: Message,
        *,
        tool: ToolManager | None = None,
    ) -> list[dict[str, Any]]:
        outcome = (
            tool_message.tool_call_result
            or tool_message.tool_call_error
            or tool_message.tool_call_diagnostic
        )
        assert outcome is not None

        output: Any
        if isinstance(outcome, ToolCallDiagnostic):
            call_id_value = outcome.call_id
            if call_id_value is None:
                return [
                    {
                        "role": str(MessageRole.ASSISTANT),
                        "content": to_json(
                            tool_call_diagnostic_payload(outcome)
                        ),
                    }
                ]
            call_id = str(call_id_value)
            name = (
                tool_message.name
                or outcome.canonical_name
                or outcome.requested_name
                or "tool"
            )
            arguments = tool_message.arguments
            output = tool_call_diagnostic_payload(outcome)
        else:
            call_id = str(outcome.call.id)
            name = outcome.call.name
            arguments = outcome.call.arguments
            output = (
                outcome.result
                if isinstance(outcome, ToolCallResult)
                else {"error": outcome.message}
            )

        return [
            {
                "type": "function_call",
                "name": TextGenerationVendor.provider_tool_name(
                    name,
                    tool=tool,
                    provider_family=self._usage_provider_family,
                ),
                "call_id": call_id,
                "arguments": to_json(arguments),
            },
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": to_json(output),
            },
        ]

    @staticmethod
    def _has_function_call_context(
        messages: list[TemplateMessage] | list[dict[str, Any]],
    ) -> bool:
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            if message.get("type") in {
                "function_call",
                "function_call_output",
            }:
                return True
        return False

    @staticmethod
    def _reasoning_config(
        settings: GenerationSettings,
    ) -> dict[str, str] | None:
        effort = settings.reasoning.effort
        summary = settings.reasoning.summary
        reasoning: dict[str, str] = {}
        if effort is not None:
            assert isinstance(
                effort, ReasoningEffort
            ), "OpenAI Responses reasoning effort is not supported"
            if effort == ReasoningEffort.MAX:
                effort = ReasoningEffort.XHIGH
            if effort != ReasoningEffort.NONE or summary is not None:
                reasoning["effort"] = effort.value
        if summary is not None:
            assert isinstance(
                summary, ReasoningSummaryMode
            ), "OpenAI Responses reasoning summary is not supported"
            reasoning["summary"] = summary.value
        return reasoning or None

    @staticmethod
    def _text_config(settings: GenerationSettings) -> dict[str, Any]:
        text: dict[str, Any] = {}
        if settings.response_format is not None:
            text["format"] = OpenAIClient._response_text_format(
                settings.response_format
            )
        if settings.stop_strings is not None:
            text["stop"] = settings.stop_strings
        return text

    @staticmethod
    def _prompt_cache_retention_config(
        settings: GenerationSettings,
    ) -> str | None:
        retention = settings.prompt_cache_retention
        if retention is None:
            return None
        if isinstance(retention, PromptCacheRetention):
            return retention.value
        assert isinstance(
            retention, str
        ), "OpenAI prompt cache retention must be a string"
        assert retention in {
            item.value for item in PromptCacheRetention
        }, "OpenAI prompt cache retention is not supported"
        return retention

    @staticmethod
    def _response_text_format(
        response_format: dict[str, Any],
    ) -> dict[str, Any]:
        assert isinstance(response_format, dict)
        format_type = response_format.get("type")
        match format_type:
            case "text" | "json_object":
                return {"type": format_type}
            case "json_schema":
                return OpenAIClient._json_schema_format(response_format)
            case _:
                raise AssertionError(
                    "OpenAI Responses response format is not supported"
                )

    @staticmethod
    def _json_schema_format(
        response_format: dict[str, Any],
    ) -> dict[str, Any]:
        has_chat_schema = "json_schema" in response_format
        has_responses_schema = "schema" in response_format
        if has_chat_schema == has_responses_schema:
            raise AssertionError(
                "OpenAI Responses json_schema format is ambiguous"
            )
        if has_chat_schema:
            return OpenAIClient._chat_json_schema_format(response_format)
        return OpenAIClient._responses_json_schema_format(response_format)

    @staticmethod
    def _chat_json_schema_format(
        response_format: dict[str, Any],
    ) -> dict[str, Any]:
        json_schema = response_format["json_schema"]
        assert isinstance(json_schema, dict)
        schema = json_schema.get("schema")
        assert isinstance(schema, dict)
        name = json_schema.get("name") or schema.get("title") or "response"
        assert isinstance(name, str) and name
        output: dict[str, Any] = {
            "type": "json_schema",
            "name": name,
            "schema": schema,
        }
        if "strict" in json_schema:
            strict = json_schema["strict"]
            assert isinstance(strict, bool)
            output["strict"] = strict
        return output

    @staticmethod
    def _responses_json_schema_format(
        response_format: dict[str, Any],
    ) -> dict[str, Any]:
        schema = response_format["schema"]
        name = response_format.get("name")
        assert isinstance(schema, dict)
        assert isinstance(name, str) and name
        output: dict[str, Any] = {
            "type": "json_schema",
            "name": name,
            "schema": schema,
        }
        if "strict" in response_format:
            strict = response_format["strict"]
            assert isinstance(strict, bool)
            output["strict"] = strict
        return output

    @staticmethod
    def _is_azure_base_url(base_url: str | None) -> bool:
        if not isinstance(base_url, str):
            return False
        host = urlparse(base_url).hostname or ""
        return host.endswith(".openai.azure.com") or host.endswith(
            ".cognitiveservices.azure.com"
        )

    @staticmethod
    def _azure_extra_query(
        base_url: str | None,
        azure_api_version: str | None,
    ) -> dict[str, str] | None:
        is_azure = OpenAIClient._is_azure_base_url(base_url)
        parsed = urlparse(base_url or "")
        if is_azure and parsed.query:
            raise AssertionError(
                "Azure OpenAI base_url must not include query parameters"
            )
        if azure_api_version is None:
            if is_azure and not parsed.path.rstrip("/").endswith("/openai/v1"):
                raise AssertionError(
                    "Azure OpenAI Responses base_url must end with /openai/v1/"
                )
            return None
        assert isinstance(azure_api_version, str) and azure_api_version.strip()
        if not is_azure:
            raise AssertionError(
                "azure_api_version is only supported for Azure OpenAI"
            )
        return {"api-version": azure_api_version}

    def _uses_reasoning_profile(self, model_id: str) -> bool:
        normalized = (model_id or self._DEFAULT_MODEL_ID).lower()
        return (
            self._is_azure
            or normalized.startswith("gpt-5")
            or (
                len(normalized) > 1
                and normalized[0] == "o"
                and normalized[1].isdigit()
            )
        )

    @staticmethod
    def _content_block(block: dict[str, Any]) -> dict[str, Any]:
        block_type = block.get("type")
        match block_type:
            case "file":
                file = block.get("file")
                assert isinstance(file, dict), "File blocks require file data"
                return OpenAIClient._file_block(file)
            case "image_url":
                image = block.get("image_url")
                assert isinstance(
                    image, dict
                ), "Image blocks require image data"
                return OpenAIClient._image_block(image)
            case "text":
                text = block.get("text")
                assert isinstance(text, str), "Text blocks require text"
                return {"type": "input_text", "text": text}
            case _:
                return block

    @staticmethod
    def _file_block(file: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": "input_file"}
        file_id = file.get("file_id")
        file_url = file.get("file_url") or file.get("url")
        file_data = file.get("file_data") or file.get("data")
        filename = file.get("filename")
        mime_type = file.get("mime_type")

        if isinstance(file_id, str):
            payload["file_id"] = file_id
        elif isinstance(file_url, str):
            payload["file_url"] = file_url
        elif isinstance(file_data, str):
            file_mime_type = (
                mime_type
                if isinstance(mime_type, str)
                else (
                    guess_type(filename)[0]
                    if isinstance(filename, str)
                    else None
                )
            )
            payload["file_data"] = (
                file_data
                if file_data.startswith("data:")
                or not isinstance(file_mime_type, str)
                else f"data:{file_mime_type};base64,{file_data}"
            )
        else:
            raise AssertionError(
                "OpenAI file blocks require file_id, file_url, or file_data"
            )

        if isinstance(filename, str):
            payload["filename"] = filename
        return payload

    @staticmethod
    def _image_block(image: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": "input_image"}
        file_id = image.get("file_id")
        image_url = image.get("url") or image.get("uri")
        image_data = image.get("data")
        mime_type = image.get("mime_type") or "image/png"
        detail = image.get("detail")

        if isinstance(file_id, str):
            payload["file_id"] = file_id
        elif isinstance(image_url, str):
            payload["image_url"] = image_url
        elif isinstance(image_data, str):
            payload["image_url"] = OpenAIClient._image_data_url(
                image_data,
                mime_type,
            )
        else:
            raise AssertionError(
                "OpenAI image blocks require file_id, url, or data"
            )

        if isinstance(detail, str):
            payload["detail"] = detail
        return payload

    @staticmethod
    def _image_data_url(image_data: str, mime_type: object) -> str:
        if image_data.startswith("data:"):
            return image_data
        assert isinstance(
            mime_type, str
        ), "OpenAI image blocks require an image MIME type"
        assert mime_type.startswith(
            "image/"
        ), "OpenAI image blocks require an image MIME type"
        return f"data:{mime_type};base64,{image_data}"

    @staticmethod
    def _tool_schemas(tool: ToolManager) -> list[dict[str, Any]] | None:
        provider_ready = isinstance(tool, ToolManager)
        schemas = (
            tool.provider_json_schemas(
                provider_family=ProviderFamily.OPENAI.value
            )
            if provider_ready
            else tool.json_schemas()
        )
        return (
            [
                {
                    "type": t["type"],
                    **t["function"],
                    **(
                        {}
                        if provider_ready
                        else {
                            "name": TextGenerationVendor.encode_tool_name(
                                t["function"]["name"]
                            )
                        }
                    ),
                }
                for t in schemas
                if t["type"] == "function"
            ]
            if schemas
            else None
        )

    @staticmethod
    def _tool_choice(
        tool_choice: str,
        schemas: Sequence[Mapping[str, Any]],
        *,
        tool: ToolManager | None = None,
    ) -> dict[str, str]:
        assert tool_choice, "OpenAI tool_choice must be a tool name"
        name = TextGenerationVendor.provider_tool_name(
            tool_choice,
            tool=tool,
            provider_family=ProviderFamily.OPENAI,
        )
        schema_names = {schema.get("name") for schema in schemas}
        assert (
            name in schema_names
        ), "OpenAI tool_choice must match an available tool"
        return {"type": "function", "name": name}

    @staticmethod
    def _non_stream_response_content(
        response: object,
        *,
        tool: ToolManager | None = None,
        provider_family: ProviderFamily | str | None = ProviderFamily.OPENAI,
    ) -> str:
        parts: list[str] = []
        output = OpenAIClient._response_field(response, "output")
        if not isinstance(output, list):
            return "".join(parts)

        for item in output:
            item_type = OpenAIClient._response_field(item, "type")
            contents = OpenAIClient._response_field(item, "content")
            if not isinstance(contents, list):
                contents = []

            if item_type in {None, "message", "output_text"}:
                for content in contents:
                    text = OpenAIClient._response_field(content, "text")
                    if isinstance(text, str):
                        parts.append(text)
                continue

            if item_type in {"tool_call", "function_call"}:
                call = OpenAIClient._response_field(item, "call") or item
                function = (
                    OpenAIClient._response_field(call, "function") or call
                )
                provider_name = OpenAIClient._response_field(function, "name")
                canonical_name = (
                    TextGenerationVendor.canonical_tool_name(
                        provider_name,
                        tool=tool,
                        provider_family=provider_family,
                    )
                    if isinstance(provider_name, str)
                    and isinstance(tool, ToolManager)
                    else provider_name
                )
                call_id = OpenAIClient._response_field(call, "id")
                arguments = OpenAIClient._response_field(function, "arguments")
                if isinstance(tool, ToolManager):
                    parts.append(
                        TextGenerationVendor.build_tool_call_text(
                            call_id,
                            canonical_name,
                            arguments,
                            tool_name_is_canonical=isinstance(
                                canonical_name, str
                            ),
                        )
                    )
                else:
                    parts.append(
                        TextGenerationVendor.build_tool_call_text(
                            call_id,
                            canonical_name,
                            arguments,
                        )
                    )

        return "".join(parts)

    @staticmethod
    def _response_field(value: object, attribute: str) -> object | None:
        if isinstance(value, dict):
            return value.get(attribute)
        return getattr(value, attribute, None)


class OpenAINonStreamingResponse(TextGenerationResponse):
    _static_response_text: str | None

    def __init__(
        self,
        *args: Any,
        static_response_text: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._static_response_text = static_response_text

    def __str__(self) -> str:
        if self._static_response_text is not None:
            return self._static_response_text

        buffered = self._buffer.getvalue()
        if buffered is not None:
            return buffered

        return object.__repr__(self)

    async def to_str(self) -> str:
        text = await super().to_str()
        self._static_response_text = text
        return text


class OpenAIModel(TextGenerationVendorModel):
    @property
    def reasoning_summary_provider(self) -> str:
        """Return the configured OpenAI provider family."""
        compatible_provider = super().reasoning_summary_provider
        if compatible_provider != "openai":
            return compatible_provider
        client_is_azure = getattr(
            getattr(self, "_model", None),
            "_is_azure",
            None,
        )
        if client_is_azure is True:
            return "azure_openai"
        if client_is_azure is False:
            return "openai"
        settings = getattr(self, "_settings", None)
        base_url = getattr(settings, "base_url", None)
        if OpenAIClient._is_azure_base_url(base_url):
            return "azure_openai"
        return "openai"

    def _load(
        self,
        *args: object,
        load_tokenizer: bool,
        tokenizer_name_or_path: str | None,
    ) -> None:
        super()._load(
            *args,
            load_tokenizer=load_tokenizer,
            tokenizer_name_or_path=tokenizer_name_or_path,
        )
        if isinstance(self._model, OpenAIClient):
            self._exit_stack.push_async_callback(self._model.aclose)

    def _load_model(
        self,
    ) -> PreTrainedModel | TextGenerationVendor | DiffusionPipeline:
        assert self._settings.base_url or self._settings.access_token
        azure_api_version = provider_string_option(
            self._settings.provider_options,
            "azure_api_version",
        )
        client_kwargs: dict[str, Any] = {
            "api_key": self._settings.access_token,
            "base_url": self._settings.base_url,
        }
        if azure_api_version is not None:
            client_kwargs["azure_api_version"] = azure_api_version
        provider_options = self._settings.provider_options
        if (
            provider_options is not None
            and "openai_max_retries" in provider_options
        ):
            client_kwargs["max_retries"] = OpenAIClient._provider_max_retries(
                provider_options
            )
        if (
            provider_options is not None
            and "openai_response_failed_retries" in provider_options
        ):
            client_kwargs["stream_response_failed_retries"] = (
                OpenAIClient._provider_response_failed_retries(
                    provider_options
                )
            )
        if (
            provider_options is not None
            and "openai_response_failed_retry_delay_seconds"
            in provider_options
        ):
            retry_delay = OpenAIClient._provider_retry_delay_seconds(
                provider_options
            )
            client_kwargs["stream_response_failed_retry_delay_seconds"] = (
                retry_delay
            )
        if (
            provider_options is not None
            and "openai_timeout_seconds" in provider_options
        ):
            client_kwargs["timeout_seconds"] = (
                OpenAIClient._provider_timeout_seconds(provider_options)
            )
        return OpenAIClient(**client_kwargs)

    async def __call__(
        self,
        input: Input,
        system_prompt: str | None = None,
        developer_prompt: str | None = None,
        settings: GenerationSettings | None = None,
        *,
        instructions: str | None = None,
        tool: ToolManager | None = None,
    ) -> TextGenerationResponse:
        generation_settings = settings or GenerationSettings()
        validate_reasoning_summary_request(self, generation_settings)
        messages = self._messages(input, system_prompt, developer_prompt, tool)
        streamer = await self._model(
            self._model_id,
            messages,
            generation_settings,
            instructions=instructions,
            tool=tool,
            use_async_generator=generation_settings.use_async_generator,
        )

        if generation_settings.use_async_generator:
            return TextGenerationResponse(
                streamer,
                logger=self._logger,
                generation_settings=generation_settings,
                settings=generation_settings,
                use_async_generator=True,
            )

        static_text: str | None = None
        if isinstance(streamer, TextGenerationSingleStream):
            content = streamer.content
            static_text = content if isinstance(content, str) else None

        return OpenAINonStreamingResponse(
            streamer,
            logger=self._logger,
            generation_settings=generation_settings,
            settings=generation_settings,
            use_async_generator=False,
            static_response_text=static_text,
        )
