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
from .....model.reasoning import (
    ReasoningSummaryRequestCapability,
    validate_reasoning_summary_request,
)
from .....model.response.text import TextGenerationResponse
from .....model.stream import (
    REASONING_SEGMENT_BOUNDARY_METADATA_KEY,
    CanonicalStreamItem,
    StreamConsumerCancellation,
    StreamConsumerClosure,
    StreamItemCorrelation,
    StreamItemKind,
    StreamProducerBackend,
    StreamProviderAdapterError,
    StreamProviderCapabilities,
    StreamProviderEvent,
    StreamReasoningRepresentation,
    StreamReasoningSegmentState,
    StreamRetentionPolicy,
    StreamVisibility,
    TextGenerationNonStreamResult,
    TextGenerationSingleStream,
    TextGenerationStream,
    normalize_provider_stream,
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

from asyncio import (
    CancelledError,
    Event,
    Lock,
    Task,
    create_task,
    current_task,
    sleep,
)
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Mapping,
    Sequence,
)
from copy import deepcopy
from dataclasses import dataclass, field, replace
from importlib import import_module
from inspect import isawaitable
from json import dumps
from math import isfinite
from mimetypes import guess_type
from types import SimpleNamespace
from typing import Any, Never, cast, get_args
from urllib.parse import urlparse


class _OmitPlaceholder:  # noqa: D101
    pass


Omit: type[Any] = _OmitPlaceholder

_OPENAI_REASONING_SUMMARY_CAPABILITY = ReasoningSummaryRequestCapability(
    supported_modes=frozenset(ReasoningSummaryMode)
)
_OPENAI_REASONING_SUMMARY_PROVIDERS = frozenset(
    {
        ProviderFamily.OPENAI.value,
        ProviderFamily.AZURE_OPENAI.value,
    }
)
_OPENAI_VENDOR_MODULE = "avalan.model.nlp.text.vendor.openai"


def _is_exact_native_openai_type(value: object, expected_name: str) -> bool:
    value_type = type(value)
    return (
        value_type.__module__ == _OPENAI_VENDOR_MODULE
        and value_type.__name__ == expected_name
        and value_type.__qualname__ == expected_name
    )


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


class _OpenAIConcurrentProviderConsumerError(RuntimeError):
    """Report a concurrent pull against the single-consumer stream."""


class _MissingProviderField:  # noqa: D101
    pass


_MISSING_PROVIDER_FIELD = _MissingProviderField()
_OPENAI_RESPONSE_STREAM_EVENT_TYPES: frozenset[type[object]] | None = None
_OPENAI_RESPONSE_OUTPUT_ITEM_TYPES: frozenset[type[object]] | None = None
_OPENAI_TOOL_CALL_ID_ERROR_MESSAGE = (
    "response tool call id must be a non-empty string"
)


def _is_trusted_openai_response_stream_event(value: object) -> bool:
    global _OPENAI_RESPONSE_STREAM_EVENT_TYPES

    trusted_types = _OPENAI_RESPONSE_STREAM_EVENT_TYPES
    if trusted_types is None:
        try:
            event_alias = getattr(
                import_module("openai.types.responses"),
                "ResponseStreamEvent",
            )
        except (AttributeError, ImportError):
            return False
        discovered: set[type[object]] = set()
        pending = [event_alias]
        while pending:
            candidate = pending.pop()
            if isinstance(candidate, type):
                discovered.add(candidate)
                continue
            pending.extend(get_args(candidate))
        trusted_types = frozenset(discovered)
        _OPENAI_RESPONSE_STREAM_EVENT_TYPES = trusted_types
    return type(value) in trusted_types


def _is_trusted_openai_response_output_item(value: object) -> bool:
    global _OPENAI_RESPONSE_OUTPUT_ITEM_TYPES

    trusted_types = _OPENAI_RESPONSE_OUTPUT_ITEM_TYPES
    if trusted_types is None:
        try:
            item_alias = getattr(
                import_module("openai.types.responses"),
                "ResponseOutputItem",
            )
        except (AttributeError, ImportError):
            return False
        discovered: set[type[object]] = set()
        pending = [item_alias]
        while pending:
            candidate = pending.pop()
            if isinstance(candidate, type):
                discovered.add(candidate)
                continue
            pending.extend(get_args(candidate))
        trusted_types = frozenset(discovered)
        _OPENAI_RESPONSE_OUTPUT_ITEM_TYPES = trusted_types
    return type(value) in trusted_types


def _provider_value_shape(value: object) -> str:
    if value is _MISSING_PROVIDER_FIELD:
        return "missing"
    if value is None:
        return "null"
    if type(value) is bool:
        return "boolean"
    if type(value) is int:
        return "negative_integer" if value < 0 else "integer"
    if type(value) is float:
        return "float"
    if type(value) is str:
        return "empty_string" if not value.strip() else "string"
    if isinstance(value, Mapping):
        return "mapping"
    if isinstance(value, Sequence):
        return "sequence"
    return "object"


class _OpenAIReasoningSummaryEventError(ValueError):
    code = "invalid_reasoning_summary_event"
    event_type: str
    field: str
    output_index: int | None
    summary_index: int | None
    value_shape: str
    public_message: str | None

    def __init__(
        self,
        *,
        event_type: str,
        field: str,
        value: object = _MISSING_PROVIDER_FIELD,
        output_index: object = _MISSING_PROVIDER_FIELD,
        summary_index: object = _MISSING_PROVIDER_FIELD,
        value_shape: str | None = None,
        public_message: str | None = None,
    ) -> None:
        assert isinstance(event_type, str) and event_type
        assert isinstance(field, str) and field
        assert value_shape is None or (
            isinstance(value_shape, str) and value_shape
        )
        assert public_message is None or (
            type(public_message) is str and bool(public_message)
        )
        super().__init__("OpenAI reasoning summary event is invalid.")
        self.event_type = event_type
        self.field = field
        self.output_index = self._safe_index(output_index)
        self.summary_index = self._safe_index(summary_index)
        self.value_shape = value_shape or _provider_value_shape(value)
        self.public_message = public_message

    @staticmethod
    def _safe_index(value: object) -> int | None:
        return value if type(value) is int and value >= 0 else None

    def safe_data(self) -> LooseJsonValue:
        data: dict[str, object] = {
            "error": {
                "type": "invalid_provider_event",
                "code": self.code,
                "message": str(self),
                "event_type": self.event_type,
                "field": self.field,
                "index": {
                    "output_index": self.output_index,
                    "summary_index": self.summary_index,
                },
                "value_shape": self.value_shape,
            }
        }
        if self.public_message is not None:
            data["message"] = self.public_message
        return cast(LooseJsonValue, data)


class _OpenAIToolCallIdError(ValueError):
    """Report a fixed safe invalid tool-call identifier diagnostic."""

    value_shape: str

    def __init__(self, value: object) -> None:
        self.value_shape = _provider_value_shape(value)
        super().__init__(_OPENAI_TOOL_CALL_ID_ERROR_MESSAGE)


@dataclass(frozen=True, slots=True)
class _OpenAIReasoningSummaryEmission:
    text: str
    item_id: str
    output_index: int
    summary_index: int
    provider_event_type: str


@dataclass(slots=True)
class _OpenAIReasoningSummaryPartState:
    item_id: str
    output_index: int
    summary_index: int
    part_added: bool = False
    delta_fragments: list[str] = field(default_factory=list)
    streamed_text_present: bool = False
    text_done_seen: bool = False
    text_done_text: str = ""
    part_done_seen: bool = False
    part_done_text: str = ""
    fallback_text: str | None = None
    closed: bool = False


@dataclass(slots=True)
class _OpenAIReasoningSummaryItemState:
    item_id: str
    output_index: int
    parts: dict[int, _OpenAIReasoningSummaryPartState] = field(
        default_factory=dict
    )
    completed_fingerprint: tuple[tuple[object, ...], ...] | None = None
    completed: bool = False
    incomplete: bool = False


@dataclass(slots=True)
class _OpenAINativeReasoningPartState:
    delta_fragments: list[str] = field(default_factory=list)
    completed_text: str | None = None
    completed: bool = False


@dataclass(slots=True)
class _OpenAIOutputItemFingerprint:
    item_id: str | None
    output_index: int | None
    item_type: str | None
    call_id: str | None = None
    canonical_name: str | None = None
    content_index: int | None = None
    completed_payload_fingerprint: tuple[tuple[object, ...], ...] | None = None
    argument_fragments: list[str] = field(default_factory=list)
    final_arguments: str | None = None
    arguments_closed: bool = False
    channel_closed: bool = False
    completed: bool = False
    native_parts: dict[int, _OpenAINativeReasoningPartState] = field(
        default_factory=dict
    )


@dataclass(frozen=True, slots=True)
class _OpenAIProviderTerminal:
    event: StreamProviderEvent
    succeeded: bool
    cleanup_method: str = "aclose"
    cleanup_failed: bool = False
    include_cleanup_diagnostic: bool = False

    def __post_init__(self) -> None:
        assert self.event.kind in {
            StreamItemKind.STREAM_CANCELLED,
            StreamItemKind.STREAM_COMPLETED,
            StreamItemKind.STREAM_ERRORED,
        }
        assert self.succeeded is (
            self.event.kind is StreamItemKind.STREAM_COMPLETED
        )
        assert self.cleanup_method in {"aclose", "cancel"}
        assert isinstance(self.cleanup_failed, bool)
        assert isinstance(self.include_cleanup_diagnostic, bool)


@dataclass(slots=True)
class _OpenAIReasoningSummaryState:
    _items: dict[tuple[str, int], _OpenAIReasoningSummaryItemState] = field(
        default_factory=dict
    )
    _fingerprint_by_item_id: dict[str, _OpenAIOutputItemFingerprint] = field(
        default_factory=dict
    )
    _fingerprint_by_output_index: dict[int, _OpenAIOutputItemFingerprint] = (
        field(default_factory=dict)
    )
    _fingerprint_by_call_id: dict[str, _OpenAIOutputItemFingerprint] = field(
        default_factory=dict
    )
    _standalone_completed_item_ids: set[str] = field(default_factory=set)
    _response_closed: bool = False
    _response_terminal_type: str | None = None

    def abort(self) -> None:
        for item in self._items.values():
            for part in item.parts.values():
                part.delta_fragments.clear()
                part.text_done_text = ""
                part.part_done_text = ""
                part.fallback_text = None
            item.parts.clear()
            item.completed_fingerprint = None
        self._items.clear()
        fingerprints = {
            id(fingerprint): fingerprint
            for fingerprint in (
                *self._fingerprint_by_item_id.values(),
                *self._fingerprint_by_output_index.values(),
                *self._fingerprint_by_call_id.values(),
            )
        }
        for fingerprint in fingerprints.values():
            fingerprint.argument_fragments.clear()
            fingerprint.final_arguments = None
            fingerprint.canonical_name = None
            fingerprint.call_id = None
            fingerprint.completed_payload_fingerprint = None
            for native_part in fingerprint.native_parts.values():
                native_part.delta_fragments.clear()
                native_part.completed_text = None
            fingerprint.native_parts.clear()
        self._fingerprint_by_item_id.clear()
        self._fingerprint_by_output_index.clear()
        self._fingerprint_by_call_id.clear()
        self._standalone_completed_item_ids.clear()
        self._response_closed = True
        self._response_terminal_type = "response.consumer_closed"

    def add_item(
        self,
        event: object,
        event_type: str,
        item_type: str | None,
    ) -> bool:
        item = self._field(event, "item", event_type)
        if item_type != "reasoning":
            return False
        output_index_value = self._field(event, "output_index", event_type)
        item_id_value = self._field(item, "id", event_type)
        output_index = self._required_index(
            output_index_value,
            event_type=event_type,
            field_name="output_index",
            output_index=output_index_value,
        )
        item_id = self._required_item_id(
            item_id_value,
            event_type=event_type,
            field_name="item.id",
            output_index=output_index,
        )
        summary = self._field(item, "summary", event_type)
        if type(summary) is not list or summary:
            self._error(
                event_type,
                "item.summary",
                summary,
                output_index=output_index,
                value_shape="unexpected_value",
            )
        key = (item_id, output_index)
        current = self._items.get(key)
        if current is not None:
            return True
        self._items[key] = _OpenAIReasoningSummaryItemState(
            item_id=item_id,
            output_index=output_index,
        )
        return True

    def observe_output_item(
        self,
        event: object,
        event_type: str,
        item_type: str | None,
        *,
        call_id: str | None = None,
        canonical_name: str | None = None,
    ) -> tuple[str | None, _OpenAIOutputItemFingerprint | None]:
        item = self._field(event, "item", event_type)
        item_id_value = self._field(item, "id", event_type)
        output_index_value = self._field(event, "output_index", event_type)
        item_type_value = self._field(item, "type", event_type)
        self._validate_output_item_status(
            item,
            event_type,
            output_index=output_index_value,
        )
        item_id = self._optional_item_id(
            item_id_value,
            event_type=event_type,
            field_name="item.id",
            output_index=output_index_value,
        )
        output_index = self._optional_index(
            output_index_value,
            event_type=event_type,
            field_name="output_index",
        )
        fingerprint = self._observe_fingerprint(
            event_type=event_type,
            item_id=item_id,
            output_index=output_index,
            output_index_value=output_index_value,
            item_type=item_type,
            item_type_value=item_type_value,
            call_id=call_id,
        )
        if fingerprint is None:
            return item_type, None
        if (
            fingerprint.channel_closed
            and event_type == "response.output_item.added"
        ):
            self._error(
                event_type,
                "item",
                output_index=output_index_value,
                value_shape="closed",
            )
        if canonical_name is not None:
            if fingerprint.canonical_name is None:
                fingerprint.canonical_name = canonical_name
            elif fingerprint.canonical_name != canonical_name:
                self._error(
                    event_type,
                    "item.name",
                    output_index=output_index_value,
                    value_shape="conflict",
                )
        return fingerprint.item_type, fingerprint

    @classmethod
    def _validate_output_item_status(
        cls,
        item: object,
        event_type: str,
        *,
        output_index: object,
    ) -> None:
        status = cls._field(item, "status", event_type)
        if status is _MISSING_PROVIDER_FIELD or status is None:
            return
        expected = {
            (
                "in_progress"
                if event_type == "response.output_item.added"
                else "completed"
            )
        }
        if event_type in {
            "error",
            "response.error",
            "response.failed",
            "response.incomplete",
        }:
            expected.add("incomplete")
        if type(status) is not str or status not in expected:
            cls._error(
                event_type,
                "item.status",
                status,
                output_index=output_index,
                value_shape="unexpected_value",
            )

    def output_item_type(
        self,
        event: object,
        event_type: str,
    ) -> str | None:
        item = self._field(event, "item", event_type)
        output_index_value = self._field(event, "output_index", event_type)
        raw_item_type = self._field(item, "type", event_type)
        item_type = self._optional_item_type(
            raw_item_type,
            event_type=event_type,
            output_index=output_index_value,
        )
        nested_custom_tool = self._field(item, "custom_tool_call", event_type)
        if (
            nested_custom_tool is not _MISSING_PROVIDER_FIELD
            and nested_custom_tool is not None
        ):
            if item_type is None:
                return "tool_call"
            if item_type != "custom_tool_call":
                self._error(
                    event_type,
                    "item.type",
                    output_index=output_index_value,
                    value_shape="conflict",
                )
            return item_type
        if item_type is not None:
            return item_type
        if self._reasoning_fields_are_present(item, event_type):
            self._error(
                event_type,
                "item.type",
                raw_item_type,
                output_index=output_index_value,
            )
        if event_type == "response.output_item.added":
            return None
        item_id_value = self._field(item, "id", event_type)
        item_id = (
            item_id_value
            if type(item_id_value) is str and bool(item_id_value.strip())
            else None
        )
        output_index = (
            output_index_value
            if type(output_index_value) is int and output_index_value >= 0
            else None
        )
        fingerprints = tuple(
            fingerprint
            for fingerprint in (
                (
                    self._fingerprint_by_item_id.get(item_id)
                    if item_id is not None
                    else None
                ),
                (
                    self._fingerprint_by_output_index.get(output_index)
                    if output_index is not None
                    else None
                ),
            )
            if fingerprint is not None
        )
        if fingerprints:
            for fingerprint in fingerprints:
                if fingerprint.item_type in {
                    "custom_tool_call",
                    "function_call",
                    "tool_call",
                }:
                    return fingerprint.item_type
            return None
        return "function_call"

    @classmethod
    def _reasoning_fields_are_present(
        cls,
        item: object,
        event_type: str,
    ) -> bool:
        return any(
            cls._field(item, field_name, event_type)
            is not _MISSING_PROVIDER_FIELD
            for field_name in ("summary", "encrypted_content")
        )

    @staticmethod
    def _item_types_compatible(left: str, right: str) -> bool:
        if left == right:
            return True
        if {left, right} <= {"message", "output_text"}:
            return True
        refinements = {"custom_tool_call", "function_call"}
        return (left == "tool_call" and right in refinements) or (
            right == "tool_call" and left in refinements
        )

    def validate_semantic_event(
        self,
        event: object,
        event_type: str,
        *,
        allowed_item_types: frozenset[str],
        implied_item_type: str | None = None,
        call_id: str | None = None,
        canonical_name: str | None = None,
        require_open: bool = False,
        validate_content_index: bool = False,
    ) -> _OpenAIOutputItemFingerprint | None:
        item_id_value = self._field(event, "item_id", event_type)
        output_index_value = self._field(event, "output_index", event_type)
        item_id = self._optional_item_id(
            item_id_value,
            event_type=event_type,
            field_name="item_id",
            output_index=output_index_value,
        )
        output_index = self._optional_index(
            output_index_value,
            event_type=event_type,
            field_name="output_index",
        )
        content_index_value = (
            self._field(event, "content_index", event_type)
            if validate_content_index
            else _MISSING_PROVIDER_FIELD
        )
        content_index = self._optional_index(
            content_index_value,
            event_type=event_type,
            field_name="content_index",
            output_index=output_index_value,
        )
        fingerprint = self._observe_fingerprint(
            event_type=event_type,
            item_id=item_id,
            output_index=output_index,
            output_index_value=output_index_value,
            item_type=implied_item_type,
            item_type_value=implied_item_type,
            call_id=call_id,
        )
        assert fingerprint is not None
        if require_open and fingerprint.channel_closed:
            self._error(
                event_type,
                "item",
                output_index=output_index_value,
                value_shape="closed",
            )
        assert fingerprint.item_type in allowed_item_types
        if content_index is not None:
            if fingerprint.content_index is None:
                fingerprint.content_index = content_index
            elif fingerprint.content_index != content_index:
                self._error(
                    event_type,
                    "content_index",
                    content_index_value,
                    output_index=output_index_value,
                    value_shape="conflict",
                )
        if canonical_name is not None:
            if fingerprint.canonical_name is None:
                fingerprint.canonical_name = canonical_name
            elif fingerprint.canonical_name != canonical_name:
                self._error(
                    event_type,
                    "item.name",
                    output_index=output_index_value,
                    value_shape="conflict",
                )
        return fingerprint

    def add_tool_argument_delta(
        self,
        fingerprint: _OpenAIOutputItemFingerprint,
        delta: str,
        *,
        event_type: str,
        output_index: object = _MISSING_PROVIDER_FIELD,
    ) -> None:
        if fingerprint.arguments_closed or fingerprint.completed:
            self._error(
                event_type,
                "arguments",
                output_index=output_index,
                value_shape="closed",
            )
        fingerprint.argument_fragments.append(delta)

    def finish_tool_arguments(
        self,
        fingerprint: _OpenAIOutputItemFingerprint,
        arguments: str | None,
        *,
        event_type: str,
        output_index: object = _MISSING_PROVIDER_FIELD,
        close: bool,
        allow_closed: bool = False,
    ) -> None:
        if (
            fingerprint.completed
            or fingerprint.arguments_closed
            and not allow_closed
        ):
            self._error(
                event_type,
                "arguments",
                output_index=output_index,
                value_shape="closed",
            )
        assembled = "".join(fingerprint.argument_fragments)
        if arguments is not None and fingerprint.argument_fragments:
            if arguments != assembled:
                self._error(
                    event_type,
                    "arguments",
                    output_index=output_index,
                    value_shape="conflict",
                )
        if (
            arguments is not None
            and fingerprint.final_arguments is not None
            and arguments != fingerprint.final_arguments
        ):
            self._error(
                event_type,
                "arguments",
                output_index=output_index,
                value_shape="conflict",
            )
        if fingerprint.final_arguments is None:
            if arguments is not None:
                fingerprint.final_arguments = arguments
            elif fingerprint.argument_fragments:
                fingerprint.final_arguments = assembled
        if close:
            fingerprint.arguments_closed = True

    def add_native_reasoning_delta(
        self,
        event: object,
        event_type: str,
        fingerprint: _OpenAIOutputItemFingerprint,
        delta: str,
    ) -> int:
        content_index = self._native_content_index(event, event_type)
        part = fingerprint.native_parts.setdefault(
            content_index,
            _OpenAINativeReasoningPartState(),
        )
        if part.completed:
            self._error(
                event_type,
                "content",
                output_index=fingerprint.output_index,
                value_shape="closed",
            )
        part.delta_fragments.append(delta)
        return content_index

    def finish_native_reasoning(
        self,
        event: object,
        event_type: str,
        fingerprint: _OpenAIOutputItemFingerprint,
    ) -> bool:
        content_index = self._native_content_index(event, event_type)
        part = fingerprint.native_parts.setdefault(
            content_index,
            _OpenAINativeReasoningPartState(),
        )
        text_value = self._field(event, "text", event_type)
        assembled = "".join(part.delta_fragments)
        if text_value is _MISSING_PROVIDER_FIELD or text_value is None:
            text = assembled
        elif type(text_value) is str:
            text = text_value
        else:
            self._error(
                event_type,
                "text",
                text_value,
                output_index=fingerprint.output_index,
            )
        if part.completed:
            if text == part.completed_text:
                return False
            self._error(
                event_type,
                "text",
                text_value,
                output_index=fingerprint.output_index,
                value_shape="conflict",
            )
        if part.delta_fragments and text != assembled:
            self._error(
                event_type,
                "text",
                text_value,
                output_index=fingerprint.output_index,
                value_shape="conflict",
            )
        part.completed_text = text
        part.completed = True
        return True

    @classmethod
    def _native_content_index(
        cls,
        event: object,
        event_type: str,
    ) -> int:
        content_index = cls._field(event, "content_index", event_type)
        if content_index is _MISSING_PROVIDER_FIELD or content_index is None:
            return 0
        if type(content_index) is not int or content_index < 0:
            cls._error(
                event_type,
                "content_index",
                content_index,
                output_index=cls._field(event, "output_index", event_type),
            )
        return content_index

    def output_item_completion_is_duplicate(
        self,
        event: object,
        event_type: str,
        fingerprint: _OpenAIOutputItemFingerprint | None,
    ) -> bool:
        if fingerprint is None or not fingerprint.completed:
            return False
        item = self._field(event, "item", event_type)
        completed_fingerprint = self._replay_significant_fingerprint(
            item,
            event_type=event_type,
            output_index=fingerprint.output_index,
        )
        if fingerprint.completed_payload_fingerprint == completed_fingerprint:
            return True
        self._error(
            event_type,
            "item",
            output_index=fingerprint.output_index,
            value_shape="conflict",
        )

    def mark_output_item_completed(
        self,
        event: object,
        event_type: str,
        fingerprint: _OpenAIOutputItemFingerprint | None,
    ) -> None:
        if fingerprint is not None:
            item = self._field(event, "item", event_type)
            completed_fingerprint = self._replay_significant_fingerprint(
                item,
                event_type=event_type,
                output_index=fingerprint.output_index,
            )
            fingerprint.completed_payload_fingerprint = completed_fingerprint
            fingerprint.completed = True
            fingerprint.channel_closed = True

    @staticmethod
    def mark_channel_closed(
        fingerprint: _OpenAIOutputItemFingerprint | None,
    ) -> None:
        if fingerprint is not None:
            fingerprint.channel_closed = True

    def _observe_fingerprint(
        self,
        *,
        event_type: str,
        item_id: str | None,
        output_index: int | None,
        output_index_value: object,
        item_type: str | None,
        item_type_value: object,
        call_id: str | None,
    ) -> _OpenAIOutputItemFingerprint | None:
        candidates: list[_OpenAIOutputItemFingerprint] = []
        for fingerprint in (
            (
                self._fingerprint_by_item_id.get(item_id)
                if item_id is not None
                else None
            ),
            (
                self._fingerprint_by_output_index.get(output_index)
                if output_index is not None
                else None
            ),
            (
                self._fingerprint_by_call_id.get(call_id)
                if call_id is not None
                else None
            ),
        ):
            if fingerprint is not None and all(
                fingerprint is not candidate for candidate in candidates
            ):
                candidates.append(fingerprint)
        if len(candidates) > 1:
            self._error(
                event_type,
                "item.identity",
                output_index=output_index_value,
                value_shape="conflict",
            )
        if candidates:
            fingerprint = candidates[0]
        elif item_type is None and call_id is None:
            return None
        else:
            fingerprint = _OpenAIOutputItemFingerprint(
                item_id=None,
                output_index=None,
                item_type=None,
            )
        if (
            item_id is not None
            and fingerprint.item_id is not None
            and item_id != fingerprint.item_id
        ) or (
            output_index is not None
            and fingerprint.output_index is not None
            and output_index != fingerprint.output_index
        ):
            self._error(
                event_type,
                "item.identity",
                output_index=output_index_value,
                value_shape="conflict",
            )
        if (
            call_id is not None
            and fingerprint.call_id is not None
            and call_id != fingerprint.call_id
        ):
            self._error(
                event_type,
                "item.call_id",
                output_index=output_index_value,
                value_shape="conflict",
            )
        if item_type is None and candidates:
            self._error(
                event_type,
                "item.type",
                item_type_value,
                output_index=output_index_value,
            )
        if item_type is not None and fingerprint.item_type is not None:
            if not self._item_types_compatible(
                item_type, fingerprint.item_type
            ):
                self._error(
                    event_type,
                    "item.type",
                    output_index=output_index_value,
                    value_shape="conflict",
                )
            if fingerprint.item_type == "tool_call" and item_type in {
                "custom_tool_call",
                "function_call",
            }:
                fingerprint.item_type = item_type
        elif item_type is not None:
            fingerprint.item_type = item_type
        if fingerprint.item_id is None:
            fingerprint.item_id = item_id
        if fingerprint.output_index is None:
            fingerprint.output_index = output_index
        if fingerprint.call_id is None:
            fingerprint.call_id = call_id
        if fingerprint.item_id is not None:
            self._fingerprint_by_item_id[fingerprint.item_id] = fingerprint
        if fingerprint.output_index is not None:
            self._fingerprint_by_output_index[fingerprint.output_index] = (
                fingerprint
            )
        if fingerprint.call_id is not None:
            self._fingerprint_by_call_id[fingerprint.call_id] = fingerprint
        return fingerprint

    def validate_terminal_output(
        self,
        response: object,
        event_type: str,
    ) -> None:
        output = self._field(response, "output", event_type)
        if output is _MISSING_PROVIDER_FIELD:
            return
        if type(output) is not list:
            self._error(
                event_type,
                "response.output",
                output,
            )
        observed_fingerprints: set[int] = set()
        for output_index, item in enumerate(output):
            if not (
                type(item) is dict
                or type(item) is SimpleNamespace
                or _is_trusted_openai_response_output_item(item)
            ):
                self._error(
                    event_type,
                    "response.output",
                    output_index=output_index,
                    value_shape="unreadable",
                )
            item_id_value = self._field(item, "id", event_type)
            item_type_value = self._field(item, "type", event_type)
            self._validate_output_item_status(
                item,
                event_type,
                output_index=output_index,
            )
            item_id = self._optional_item_id(
                item_id_value,
                event_type=event_type,
                field_name="item.id",
                output_index=output_index,
            )
            item_type = self._optional_item_type(
                item_type_value,
                event_type=event_type,
                output_index=output_index,
            )
            item_type = self._terminal_output_item_type(
                item,
                event_type,
                item_type,
                item_type_value=item_type_value,
                output_index=output_index,
            )
            call_id = self._terminal_call_id(
                item,
                event_type,
                item_type,
                item_id=item_id,
                output_index=output_index,
            )
            if (
                item_type in {"custom_tool_call", "function_call", "tool_call"}
                and call_id is None
            ):
                self._error(
                    event_type,
                    "item.call_id",
                    output_index=output_index,
                )
            fingerprints = tuple(
                {
                    id(fingerprint): fingerprint
                    for fingerprint in (
                        (
                            self._fingerprint_by_item_id.get(item_id)
                            if item_id is not None
                            else None
                        ),
                        self._fingerprint_by_output_index.get(output_index),
                        (
                            self._fingerprint_by_call_id.get(call_id)
                            if call_id is not None
                            else None
                        ),
                    )
                    if fingerprint is not None
                }.values()
            )
            if len(fingerprints) > 1:
                self._error(
                    event_type,
                    "item.identity",
                    output_index=output_index,
                    value_shape="conflict",
                )
            observed_fingerprints.update(map(id, fingerprints))
            for fingerprint in fingerprints:
                if (
                    (
                        fingerprint.item_id is not None
                        and item_id != fingerprint.item_id
                    )
                    or (fingerprint.item_id is None and item_id is not None)
                    or (
                        fingerprint.output_index is not None
                        and output_index != fingerprint.output_index
                    )
                    or (
                        fingerprint.call_id is not None
                        and call_id != fingerprint.call_id
                    )
                    or (fingerprint.call_id is None and call_id is not None)
                ):
                    self._error(
                        event_type,
                        "item.identity",
                        output_index=output_index,
                        value_shape="conflict",
                    )
                if fingerprint.output_index is None:
                    fingerprint.output_index = output_index
                    self._fingerprint_by_output_index[output_index] = (
                        fingerprint
                    )
            for fingerprint in fingerprints:
                if (
                    fingerprint.item_type is not None
                    and not self._item_types_compatible(
                        item_type, fingerprint.item_type
                    )
                ):
                    self._error(
                        event_type,
                        "item.type",
                        output_index=output_index,
                        value_shape="conflict",
                    )
                if fingerprint.completed_payload_fingerprint is not None:
                    terminal_fingerprint = (
                        self._replay_significant_fingerprint(
                            item,
                            event_type=event_type,
                            output_index=output_index,
                        )
                    )
                    if (
                        terminal_fingerprint
                        != fingerprint.completed_payload_fingerprint
                    ):
                        self._error(
                            event_type,
                            "item",
                            output_index=output_index,
                            value_shape="conflict",
                        )
        tracked_fingerprints = {
            id(fingerprint): fingerprint
            for fingerprint in (
                *self._fingerprint_by_item_id.values(),
                *self._fingerprint_by_output_index.values(),
                *self._fingerprint_by_call_id.values(),
            )
        }
        for identity, fingerprint in tracked_fingerprints.items():
            if fingerprint.completed and identity not in observed_fingerprints:
                self._error(
                    event_type,
                    "response.output",
                    output_index=fingerprint.output_index,
                    value_shape="missing_position",
                )

    @classmethod
    def _terminal_call_id(
        cls,
        item: object,
        event_type: str,
        item_type: str | None,
        *,
        item_id: str | None,
        output_index: int,
    ) -> str | None:
        nested = cls._field(item, "custom_tool_call", event_type)
        values = (
            cls._field(item, "call_id", event_type),
            cls._field(nested, "id", event_type),
        )
        call_id: str | None = None
        for value in values:
            if value is _MISSING_PROVIDER_FIELD or value is None:
                continue
            if type(value) is not str or not value.strip():
                cls._error(
                    event_type,
                    "item.call_id",
                    value,
                    output_index=output_index,
                )
            if call_id is not None and value != call_id:
                cls._error(
                    event_type,
                    "item.call_id",
                    output_index=output_index,
                    value_shape="conflict",
                )
            call_id = value
        if call_id is not None:
            if item_type not in {
                "custom_tool_call",
                "function_call",
                "tool_call",
            }:
                cls._error(
                    event_type,
                    "item.call_id",
                    output_index=output_index,
                    value_shape="unexpected_value",
                )
            return call_id
        if item_type in {
            "custom_tool_call",
            "function_call",
            "tool_call",
        }:
            return item_id
        return None

    @classmethod
    def _terminal_output_item_type(
        cls,
        item: object,
        event_type: str,
        item_type: str | None,
        *,
        item_type_value: object,
        output_index: int,
    ) -> str:
        supported_types = {
            "custom_tool_call",
            "function_call",
            "message",
            "output_text",
            "reasoning",
            "tool_call",
        }
        if item_type is not None:
            if item_type not in supported_types:
                cls._error(
                    event_type,
                    "item.type",
                    item_type_value,
                    output_index=output_index,
                    value_shape="unexpected_value",
                )
            effective_type = item_type
        else:
            cls._error(
                event_type,
                "item.type",
                item_type_value,
                output_index=output_index,
            )
        if effective_type in {"message", "output_text"}:
            OpenAIStream._message_done_text(
                item,
                event_type=event_type,
                output_index=output_index,
            )
        elif effective_type in {
            "custom_tool_call",
            "function_call",
            "tool_call",
        }:
            cls._validate_terminal_tool_item_shape(
                item,
                event_type,
                output_index=output_index,
            )
        else:
            summary = cls._field(item, "summary", event_type)
            encrypted_content = cls._field(
                item,
                "encrypted_content",
                event_type,
            )
            content = cls._field(item, "content", event_type)
            legacy_replay_item = (
                summary is _MISSING_PROVIDER_FIELD
                and type(encrypted_content) is str
                and bool(encrypted_content)
                and type(content) is list
            )
            if type(summary) is not list and not legacy_replay_item:
                cls._error(
                    event_type,
                    "item.summary",
                    summary,
                    output_index=output_index,
                )
        return effective_type

    @classmethod
    def _validate_terminal_tool_item_shape(
        cls,
        item: object,
        event_type: str,
        *,
        output_index: int,
    ) -> None:
        nested = cls._field(item, "custom_tool_call", event_type)
        function = cls._field(item, "function", event_type)
        names = (
            cls._field(item, "name", event_type),
            cls._field(nested, "name", event_type),
            cls._field(function, "name", event_type),
        )
        present_names = tuple(
            value
            for value in names
            if value is not _MISSING_PROVIDER_FIELD and value is not None
        )
        arguments = (
            cls._field(item, "arguments", event_type),
            cls._field(item, "input", event_type),
            cls._field(nested, "input", event_type),
        )
        present_arguments = tuple(
            value
            for value in arguments
            if value is not _MISSING_PROVIDER_FIELD and value is not None
        )
        if not present_names and not present_arguments:
            return
        if (
            not present_names
            or any(
                type(value) is not str or not value.strip()
                for value in present_names
            )
            or len(set(present_names)) > 1
        ):
            cls._error(
                event_type,
                "item.name",
                output_index=output_index,
            )
        if (
            not present_arguments
            or any(
                type(value) not in {dict, str} for value in present_arguments
            )
            or any(
                value != present_arguments[0]
                for value in present_arguments[1:]
            )
        ):
            cls._error(
                event_type,
                "item.arguments",
                output_index=output_index,
            )

    def add_part(self, event: object, event_type: str) -> None:
        item, part, output_index, summary_index = self._open_part_identity(
            event, event_type
        )
        part_value = self._field(event, "part", event_type)
        part_type = self._field(part_value, "type", event_type)
        text = self._field(part_value, "text", event_type)
        if type(part_type) is not str or part_type != "summary_text":
            self._error(
                event_type,
                "part.type",
                part_type,
                output_index=output_index,
                summary_index=summary_index,
                value_shape="unexpected_value",
            )
        if type(text) is not str or text:
            self._error(
                event_type,
                "part.text",
                text,
                output_index=output_index,
                summary_index=summary_index,
                value_shape="unexpected_value",
            )
        if part.part_added:
            if part.closed:
                self._error(
                    event_type,
                    "part",
                    part_value,
                    output_index=output_index,
                    summary_index=summary_index,
                    value_shape="closed",
                )
            return
        assert item.parts[summary_index] is part
        part.part_added = True

    def add_delta(
        self, event: object, event_type: str
    ) -> _OpenAIReasoningSummaryEmission | None:
        _, part, output_index, summary_index = self._open_part_identity(
            event, event_type
        )
        delta = self._field(event, "delta", event_type)
        if type(delta) is not str:
            self._error(
                event_type,
                "delta",
                delta,
                output_index=output_index,
                summary_index=summary_index,
            )
        if not part.part_added:
            self._error(
                event_type,
                "part",
                output_index=output_index,
                summary_index=summary_index,
                value_shape="not_added",
            )
        if part.text_done_seen or part.part_done_seen or part.closed:
            self._error(
                event_type,
                "part",
                output_index=output_index,
                summary_index=summary_index,
                value_shape="closed",
            )
        part.delta_fragments.append(delta)
        if not delta:
            return None
        part.streamed_text_present = True
        return _OpenAIReasoningSummaryEmission(
            text=delta,
            item_id=part.item_id,
            output_index=output_index,
            summary_index=summary_index,
            provider_event_type=event_type,
        )

    def finish_text(self, event: object, event_type: str) -> None:
        _, part, output_index, summary_index = self._open_part_identity(
            event, event_type
        )
        text = self._field(event, "text", event_type)
        if type(text) is not str:
            self._error(
                event_type,
                "text",
                text,
                output_index=output_index,
                summary_index=summary_index,
            )
        expected = "".join(part.delta_fragments)
        if part.text_done_seen:
            if text == part.text_done_text:
                return
            self._error(
                event_type,
                "text",
                text,
                output_index=output_index,
                summary_index=summary_index,
                value_shape="conflict",
            )
        if not part.part_added:
            self._error(
                event_type,
                "part",
                output_index=output_index,
                summary_index=summary_index,
                value_shape="not_added",
            )
        if text != expected:
            self._error(
                event_type,
                "text",
                text,
                output_index=output_index,
                summary_index=summary_index,
                value_shape="conflict",
            )
        part.text_done_seen = True
        part.text_done_text = text

    def finish_part(self, event: object, event_type: str) -> None:
        _, part, output_index, summary_index = self._open_part_identity(
            event, event_type
        )
        part_value = self._field(event, "part", event_type)
        part_type = self._field(part_value, "type", event_type)
        text = self._field(part_value, "text", event_type)
        if type(part_type) is not str or part_type != "summary_text":
            self._error(
                event_type,
                "part.type",
                part_type,
                output_index=output_index,
                summary_index=summary_index,
                value_shape="unexpected_value",
            )
        if type(text) is not str:
            self._error(
                event_type,
                "part.text",
                text,
                output_index=output_index,
                summary_index=summary_index,
            )
        if part.part_done_seen:
            if text == part.part_done_text:
                return
            self._error(
                event_type,
                "part.text",
                text,
                output_index=output_index,
                summary_index=summary_index,
                value_shape="conflict",
            )
        if not part.text_done_seen:
            self._error(
                event_type,
                "text.done",
                output_index=output_index,
                summary_index=summary_index,
                value_shape="missing",
            )
        if text != part.text_done_text:
            self._error(
                event_type,
                "part.text",
                text,
                output_index=output_index,
                summary_index=summary_index,
                value_shape="conflict",
            )
        part.part_done_seen = True
        part.part_done_text = text
        part.closed = True

    def complete_item(
        self,
        event: object,
        event_type: str,
    ) -> tuple[tuple[_OpenAIReasoningSummaryEmission, ...], bool, bool]:
        item_value = self._field(event, "item", event_type)
        item_id_value = self._field(item_value, "id", event_type)
        item_id = self._required_item_id(
            item_id_value,
            event_type=event_type,
            field_name="item.id",
        )
        output_index_value = self._field(event, "output_index", event_type)
        output_item_fingerprint = self._fingerprint_by_item_id.get(item_id)
        if (
            output_item_fingerprint is None
            or output_item_fingerprint.output_index is None
        ):
            self._standalone_completed_item_ids.add(item_id)
            return (), True, True
        tracked_output_index = output_item_fingerprint.output_index
        state = self._items.get((item_id, tracked_output_index))
        if state is None:
            self._standalone_completed_item_ids.add(item_id)
            return (), True, True
        output_index = self._required_index(
            output_index_value,
            event_type=event_type,
            field_name="output_index",
            output_index=output_index_value,
        )
        assert output_index == tracked_output_index
        summary_value = self._field(item_value, "summary", event_type)
        summary_fingerprint = self._summary_fingerprint(
            summary_value,
            event_type=event_type,
            output_index=output_index,
        )
        completed_fingerprint = self._replay_significant_fingerprint(
            item_value,
            event_type=event_type,
            output_index=output_index,
        )
        for part in state.parts.values():
            if part.part_added and not part.part_done_seen:
                self._error(
                    event_type,
                    "part",
                    output_index=output_index,
                    summary_index=part.summary_index,
                    value_shape="open",
                )
        emissions: list[_OpenAIReasoningSummaryEmission] = []
        for summary_index, (_, text) in enumerate(summary_fingerprint):
            part_state = state.parts.get(summary_index)
            if part_state is not None and part_state.streamed_text_present:
                if text != part_state.part_done_text:
                    self._error(
                        event_type,
                        "item.summary",
                        summary_value,
                        output_index=output_index,
                        summary_index=summary_index,
                        value_shape="conflict",
                    )
                continue
            if not text:
                continue
            if part_state is None:
                part_state = _OpenAIReasoningSummaryPartState(
                    item_id=item_id,
                    output_index=output_index,
                    summary_index=summary_index,
                    closed=True,
                )
                state.parts[summary_index] = part_state
            part_state.fallback_text = text
            part_state.closed = True
            emissions.append(
                _OpenAIReasoningSummaryEmission(
                    text=text,
                    item_id=item_id,
                    output_index=output_index,
                    summary_index=summary_index,
                    provider_event_type=event_type,
                )
            )
        for summary_index in state.parts:
            if summary_index >= len(summary_fingerprint):
                self._error(
                    event_type,
                    "item.summary",
                    summary_value,
                    output_index=output_index,
                    summary_index=summary_index,
                    value_shape="missing_position",
                )
        state.completed_fingerprint = completed_fingerprint
        output_item_fingerprint.completed_payload_fingerprint = (
            completed_fingerprint
        )
        state.completed = True
        return tuple(emissions), True, False

    @classmethod
    def _replay_significant_fingerprint(
        cls,
        item: object,
        *,
        event_type: str,
        output_index: int | None,
    ) -> tuple[tuple[object, ...], ...]:
        try:
            dumped_payload = OpenAIStream._raw_provider_payload(item)
        except _ReasoningReplayRetentionError:
            dumped_payload = None
        if dumped_payload is not None:
            raw_payload: object = dumped_payload
        else:
            payload: dict[str, object] = {}
            for field_name in (
                "id",
                "type",
                "encrypted_content",
                "summary",
                "content",
                "call_id",
                "name",
                "arguments",
                "input",
                "custom_tool_call",
            ):
                field_value = cls._field(item, field_name, event_type)
                if field_value is not _MISSING_PROVIDER_FIELD:
                    payload[field_name] = field_value
            raw_payload = payload
        try:
            normalized = _strict_provider_fingerprint_copy(raw_payload)
            assert type(normalized) is dict
            cleaned = _clean_replay_input_payload(normalized)
            return _stable_replay_json_fingerprint(cleaned)
        except _ReasoningReplayRetentionError:
            raise _OpenAIReasoningSummaryEventError(
                event_type=event_type,
                field="item",
                output_index=output_index,
                value_shape="unreadable",
            ) from None

    def close_response(self, event_type: str, *, completed: bool) -> None:
        if self._response_closed:
            if self._response_terminal_type == event_type:
                return
            self._error(
                event_type,
                "response",
                value_shape="closed",
            )
        if completed:
            for item in self._items.values():
                if not item.completed:
                    self._error(
                        event_type,
                        "item",
                        output_index=item.output_index,
                        value_shape="open",
                    )
        else:
            for item in self._items.values():
                if item.completed:
                    continue
                item.incomplete = True
                for part in item.parts.values():
                    part.closed = True
        self._response_closed = True
        self._response_terminal_type = event_type

    def close_source(self) -> None:
        if self._response_closed:
            return
        for item in self._items.values():
            if not item.completed:
                self._error(
                    "response.source_exhausted",
                    "item",
                    output_index=item.output_index,
                    value_shape="open",
                )
        self._response_closed = True
        self._response_terminal_type = "response.source_exhausted"

    def _open_part_identity(
        self,
        event: object,
        event_type: str,
    ) -> tuple[
        _OpenAIReasoningSummaryItemState,
        _OpenAIReasoningSummaryPartState,
        int,
        int,
    ]:
        item_id_value = self._field(event, "item_id", event_type)
        output_index_value = self._field(event, "output_index", event_type)
        summary_index_value = self._field(event, "summary_index", event_type)
        output_index = self._required_index(
            output_index_value,
            event_type=event_type,
            field_name="output_index",
            output_index=output_index_value,
            summary_index=summary_index_value,
        )
        summary_index = self._required_index(
            summary_index_value,
            event_type=event_type,
            field_name="summary_index",
            output_index=output_index,
            summary_index=summary_index_value,
        )
        item_id = self._required_item_id(
            item_id_value,
            event_type=event_type,
            field_name="item_id",
            output_index=output_index,
            summary_index=summary_index,
        )
        item_fingerprint = self._fingerprint_by_item_id.get(item_id)
        index_fingerprint = self._fingerprint_by_output_index.get(output_index)
        if item_fingerprint is None or index_fingerprint is None:
            self._error(
                event_type,
                "item.identity",
                output_index=output_index,
                summary_index=summary_index,
                value_shape=(
                    "unrecognized"
                    if item_fingerprint is None and index_fingerprint is None
                    else "conflict"
                ),
            )
        assert item_fingerprint is not None
        assert index_fingerprint is not None
        assert item_fingerprint is index_fingerprint
        assert item_fingerprint.item_id == item_id
        assert item_fingerprint.output_index == output_index
        if (
            item_fingerprint.item_type != "reasoning"
            or index_fingerprint.item_type != "reasoning"
        ):
            self._error(
                event_type,
                "item.type",
                output_index=output_index,
                summary_index=summary_index,
                value_shape="unexpected_value",
            )
        if item_id in self._standalone_completed_item_ids:
            self._error(
                event_type,
                "item",
                output_index=output_index,
                summary_index=summary_index,
                value_shape="closed",
            )
        item = self._items.get((item_id, output_index))
        if item is None:
            self._error(
                event_type,
                "item.state",
                output_index=output_index,
                summary_index=summary_index,
                value_shape="missing",
            )
        assert item is not None
        if item.completed or item.incomplete:
            self._error(
                event_type,
                "item",
                item_id,
                output_index=output_index,
                summary_index=summary_index,
                value_shape="closed",
            )
        part = item.parts.get(summary_index)
        if part is None:
            part = _OpenAIReasoningSummaryPartState(
                item_id=item_id,
                output_index=output_index,
                summary_index=summary_index,
            )
            item.parts[summary_index] = part
        return item, part, output_index, summary_index

    @staticmethod
    def _summary_fingerprint(
        summary: object,
        *,
        event_type: str,
        output_index: int,
    ) -> tuple[tuple[str, str], ...]:
        if type(summary) is not list:
            raise _OpenAIReasoningSummaryEventError(
                event_type=event_type,
                field="item.summary",
                value=summary,
                output_index=output_index,
            )
        result: list[tuple[str, str]] = []
        for summary_index, raw_part in enumerate(summary):
            part_type = _OpenAIReasoningSummaryState._nested_field(
                raw_part,
                "type",
                event_type=event_type,
                field_name="item.summary.type",
                output_index=output_index,
                summary_index=summary_index,
            )
            text = _OpenAIReasoningSummaryState._nested_field(
                raw_part,
                "text",
                event_type=event_type,
                field_name="item.summary.text",
                output_index=output_index,
                summary_index=summary_index,
            )
            if type(part_type) is not str or part_type != "summary_text":
                raise _OpenAIReasoningSummaryEventError(
                    event_type=event_type,
                    field="item.summary.type",
                    value=part_type,
                    output_index=output_index,
                    summary_index=summary_index,
                    value_shape="unexpected_value",
                )
            if type(text) is not str:
                raise _OpenAIReasoningSummaryEventError(
                    event_type=event_type,
                    field="item.summary.text",
                    value=text,
                    output_index=output_index,
                    summary_index=summary_index,
                )
            result.append((part_type, text))
        return tuple(result)

    @staticmethod
    def _nested_field(
        value: object,
        name: str,
        *,
        event_type: str,
        field_name: str,
        output_index: int,
        summary_index: int,
    ) -> object:
        try:
            return _OpenAIReasoningSummaryState._field(value, name, event_type)
        except _OpenAIReasoningSummaryEventError:
            raise _OpenAIReasoningSummaryEventError(
                event_type=event_type,
                field=field_name,
                output_index=output_index,
                summary_index=summary_index,
                value_shape="unreadable",
            ) from None

    @staticmethod
    def _field(value: object, name: str, event_type: str) -> object:
        try:
            if isinstance(value, Mapping):
                if name not in value:
                    return _MISSING_PROVIDER_FIELD
                return value[name]
            return getattr(value, name, _MISSING_PROVIDER_FIELD)
        except Exception:
            raise _OpenAIReasoningSummaryEventError(
                event_type=event_type,
                field=name,
                value_shape="unreadable",
            ) from None

    @staticmethod
    def _required_index(
        value: object,
        *,
        event_type: str,
        field_name: str,
        output_index: object = _MISSING_PROVIDER_FIELD,
        summary_index: object = _MISSING_PROVIDER_FIELD,
    ) -> int:
        if type(value) is not int or value < 0:
            raise _OpenAIReasoningSummaryEventError(
                event_type=event_type,
                field=field_name,
                value=value,
                output_index=output_index,
                summary_index=summary_index,
            )
        return value

    @staticmethod
    def _optional_index(
        value: object,
        *,
        event_type: str,
        field_name: str,
        output_index: object = _MISSING_PROVIDER_FIELD,
    ) -> int | None:
        if value is _MISSING_PROVIDER_FIELD or value is None:
            return None
        if type(value) is not int or value < 0:
            raise _OpenAIReasoningSummaryEventError(
                event_type=event_type,
                field=field_name,
                value=value,
                output_index=output_index,
            )
        return value

    @staticmethod
    def _required_item_id(
        value: object,
        *,
        event_type: str,
        field_name: str,
        output_index: object = _MISSING_PROVIDER_FIELD,
        summary_index: object = _MISSING_PROVIDER_FIELD,
    ) -> str:
        if type(value) is not str or not value.strip():
            raise _OpenAIReasoningSummaryEventError(
                event_type=event_type,
                field=field_name,
                value=value,
                output_index=output_index,
                summary_index=summary_index,
            )
        return value

    @staticmethod
    def _optional_item_id(
        value: object,
        *,
        event_type: str,
        field_name: str,
        output_index: object = _MISSING_PROVIDER_FIELD,
    ) -> str | None:
        if value is _MISSING_PROVIDER_FIELD or value is None:
            return None
        if type(value) is not str or not value.strip():
            raise _OpenAIReasoningSummaryEventError(
                event_type=event_type,
                field=field_name,
                value=value,
                output_index=output_index,
            )
        return value

    @staticmethod
    def _optional_item_type(
        value: object,
        *,
        event_type: str,
        output_index: object = _MISSING_PROVIDER_FIELD,
    ) -> str | None:
        if value is _MISSING_PROVIDER_FIELD or value is None:
            return None
        if type(value) is not str or not value.strip():
            raise _OpenAIReasoningSummaryEventError(
                event_type=event_type,
                field="item.type",
                value=value,
                output_index=output_index,
            )
        return value

    @staticmethod
    def _error(
        event_type: str,
        field_name: str,
        value: object = _MISSING_PROVIDER_FIELD,
        *,
        output_index: object = _MISSING_PROVIDER_FIELD,
        summary_index: object = _MISSING_PROVIDER_FIELD,
        value_shape: str | None = None,
    ) -> Never:
        raise _OpenAIReasoningSummaryEventError(
            event_type=event_type,
            field=field_name,
            value=value,
            output_index=output_index,
            summary_index=summary_index,
            value_shape=value_shape,
        )


@dataclass(frozen=True, slots=True)
class _ReplayItemAccounting:
    items: int = 0
    serialized_bytes: int = 0
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


def _strict_provider_fingerprint_copy(value: object) -> LooseJsonValue:
    if type(value) is SimpleNamespace:
        value = value.__dict__
    if type(value) is list:
        return [
            _strict_provider_fingerprint_copy(item)
            for item in cast(list[object], value)
        ]
    if type(value) is dict:
        result: dict[str, object] = {}
        for key, item in cast(dict[object, object], value).items():
            if type(key) is not str:
                raise _ReasoningReplayRetentionError()
            result[key] = _strict_provider_fingerprint_copy(item)
        return cast(LooseJsonValue, result)
    return _strict_replay_json_copy(value)


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
            raw_item_type = source.get("type") if clean_fields else None
            item_type = raw_item_type if type(raw_item_type) is str else None
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
            raw_item_type = source_mapping.get("type")
            item_type = raw_item_type if type(raw_item_type) is str else None
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
            if type(current) is str:
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


def _stable_replay_json_fingerprint(
    value: LooseJsonValue,
) -> tuple[tuple[object, ...], ...]:
    tokens: list[tuple[object, ...]] = []
    work: list[tuple[str, object]] = [("value", value)]
    while work:
        token_type, current = work.pop()
        if token_type == "key":
            assert type(current) is str
            tokens.append(("key", current))
            continue
        if type(current) is dict:
            mapping = cast(dict[str, LooseJsonValue], current)
            tokens.append(("mapping", len(mapping)))
            for key in sorted(mapping, reverse=True):
                work.append(("value", mapping[key]))
                work.append(("key", key))
            continue
        if type(current) is list:
            sequence = cast(list[LooseJsonValue], current)
            tokens.append(("sequence", len(sequence)))
            for item in reversed(sequence):
                work.append(("value", item))
            continue
        assert current is None or type(current) in {
            bool,
            float,
            int,
            str,
        }
        tokens.append(("scalar", type(current).__name__, current))
    return tuple(tokens)


@dataclass(slots=True, repr=False)
class _OpenAIReplayOwner:
    policy: StreamRetentionPolicy
    _items: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _ledger: list[_ReplayItemAccounting] = field(
        default_factory=list,
        repr=False,
    )
    _replay_item_count: int = 0
    _replay_serialized_byte_count: int = 0
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

    @property
    def generic_counters(self) -> tuple[int, int]:
        return (
            self._replay_item_count,
            self._replay_serialized_byte_count,
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
        raw_item_type = normalized.get("type")
        item_type = raw_item_type if type(raw_item_type) is str else None
        if item_type == "reasoning":
            if not OpenAIStream._is_replayable_reasoning_item(normalized):
                return False
            accounting = self._reasoning_accounting(normalized)
        elif item_type == "function_call":
            accounting = self._generic_accounting(normalized)
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
        self._replay_item_count = 0
        self._replay_serialized_byte_count = 0
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
        generic = _OpenAIReplayOwner._generic_accounting(item)
        summary_fields = {
            key: value
            for key, value in item.items()
            if key not in {"encrypted_content", "id", "type"}
        }
        if not summary_fields:
            return _ReplayItemAccounting(
                items=generic.items,
                serialized_bytes=generic.serialized_bytes,
                reasoning_items=1,
            )
        normalized_summary = _strict_replay_json_copy(summary_fields)
        assert isinstance(normalized_summary, dict)
        nodes, characters = _replay_json_accounting(normalized_summary)
        return _ReplayItemAccounting(
            items=generic.items,
            serialized_bytes=generic.serialized_bytes,
            reasoning_items=1,
            summary_nodes=nodes,
            summary_characters=characters,
            summary_serialized_bytes=_replay_json_serialized_bytes(
                normalized_summary
            ),
        )

    @staticmethod
    def _generic_accounting(
        item: dict[str, Any],
    ) -> _ReplayItemAccounting:
        return _ReplayItemAccounting(
            items=1,
            serialized_bytes=_replay_json_serialized_bytes(item),
        )

    def _assert_fits(self, accounting: _ReplayItemAccounting) -> None:
        limits_and_values = (
            (
                self.policy.openai_replay_item_limit,
                self._replay_item_count + accounting.items,
            ),
            (
                self.policy.openai_replay_serialized_byte_limit,
                self._replay_serialized_byte_count
                + accounting.serialized_bytes,
            ),
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
        self._replay_item_count += direction * accounting.items
        self._replay_serialized_byte_count += (
            direction * accounting.serialized_bytes
        )
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
    _REASONING_SUMMARY_PART_ADDED_EVENT = (
        "response.reasoning_summary_part.added"
    )
    _REASONING_SUMMARY_TEXT_DELTA_EVENT = (
        "response.reasoning_summary_text.delta"
    )
    _REASONING_SUMMARY_TEXT_DONE_EVENT = "response.reasoning_summary_text.done"
    _REASONING_SUMMARY_PART_DONE_EVENT = "response.reasoning_summary_part.done"
    _REASONING_SUMMARY_EVENTS = {
        _REASONING_SUMMARY_PART_ADDED_EVENT,
        _REASONING_SUMMARY_TEXT_DELTA_EVENT,
        _REASONING_SUMMARY_TEXT_DONE_EVENT,
        _REASONING_SUMMARY_PART_DONE_EVENT,
    }
    _TOOL_CALL_ITEM_TYPES = {
        "custom_tool_call",
        "function_call",
        "tool_call",
    }
    _TEXT_OUTPUT_ITEM_TYPES = frozenset({"message", "output_text"})
    _NATIVE_REASONING_ITEM_TYPES = frozenset({"reasoning"})
    _FUNCTION_ARGUMENT_ITEM_TYPES = frozenset({"function_call", "tool_call"})
    _CUSTOM_ARGUMENT_ITEM_TYPES = frozenset({"custom_tool_call", "tool_call"})
    _TOOL_ARGUMENT_DELTA_EVENTS = {
        "response.custom_tool_call_input.delta",
        "response.function_call_arguments.delta",
    }
    _TOOL_ARGUMENT_DONE_EVENTS = {
        "response.custom_tool_call_input.done",
        "response.function_call_arguments.done",
    }
    _ERROR_EVENTS = {"response.error", "response.failed", "error"}
    _INCOMPLETE_EVENTS = {"response.incomplete"}
    _INCOMPLETE_REASONS = {"content_filter", "max_output_tokens"}
    _stream: AsyncIterator[Any] | None
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
    _reasoning_summary_state: _OpenAIReasoningSummaryState
    _replay_owner: _OpenAIReplayOwner | None
    _replay_owner_retainer: (
        Callable[[_OpenAIReplayOwner, tuple[str, ...]], None] | None
    )
    _replay_owner_releaser: Callable[[_OpenAIReplayOwner], None] | None
    _replay_owner_terminal_handled: bool
    _request_has_replay_items: bool
    _active_provider_task: Task[Any] | None
    _external_finish_method: str | None
    _private_output_seen: bool
    _provider_terminal_prepared: bool
    _provider_terminal_event: StreamProviderEvent | None
    _provider_terminal_emitted: bool
    _provider_terminal_cleanup_failed: bool
    _provider_terminal_lock: Lock
    _provider_lifetime_claimed: bool
    _provider_consumer_owner: object | None
    _provider_consumer_done: Event
    _provider_finish_done: Event
    _provider_finish_error: BaseException | None
    _provider_silent_closed: bool
    _external_finish_lock: Lock

    def __init__(
        self,
        stream: AsyncIterator[Any],
        *,
        provider_family: ProviderFamily | str = ProviderFamily.OPENAI,
        supports_reasoning_summary: bool | None = None,
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
        normalized_provider_family = (
            provider_family.value
            if isinstance(provider_family, ProviderFamily)
            else provider_family
        )
        if supports_reasoning_summary is None:
            supports_reasoning_summary = (
                normalized_provider_family
                in _OPENAI_REASONING_SUMMARY_PROVIDERS
            )
        assert isinstance(supports_reasoning_summary, bool)
        self._stream = stream
        self._supports_reasoning_summary = supports_reasoning_summary
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
        self._reasoning_summary_state = _OpenAIReasoningSummaryState()
        self._replay_owner = replay_owner
        self._replay_owner_retainer = replay_owner_retainer
        self._replay_owner_releaser = replay_owner_releaser
        self._replay_owner_terminal_handled = False
        self._request_has_replay_items = request_has_replay_items or bool(
            replay_owner is not None and replay_owner.item_count
        )
        self._active_provider_task = None
        self._external_finish_method = None
        self._private_output_seen = False
        self._provider_terminal_prepared = False
        self._provider_terminal_event = None
        self._provider_terminal_emitted = False
        self._provider_terminal_cleanup_failed = False
        self._provider_terminal_lock = Lock()
        self._provider_lifetime_claimed = False
        self._provider_consumer_owner = None
        self._provider_consumer_done = Event()
        self._provider_consumer_done.set()
        self._provider_finish_done = Event()
        self._provider_finish_error = None
        self._provider_silent_closed = False
        self._external_finish_lock = Lock()

        async def generator() -> AsyncIterator[CanonicalStreamItem]:
            canonical = self.canonical_stream(
                stream_session_id=self._DEFAULT_STREAM_SESSION_ID,
                run_id=self._DEFAULT_RUN_ID,
                turn_id=self._DEFAULT_TURN_ID,
            )
            try:
                async for item in canonical:
                    yield item
            finally:
                await self._call_stream_source_cleanup(canonical, "aclose")

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
        await self._request_external_finish("cancel")

    async def aclose(self) -> None:
        await self._request_external_finish("aclose")

    async def _request_external_finish(self, method_name: str) -> None:
        assert method_name in {"aclose", "cancel"}
        async with self._external_finish_lock:
            existing_method = self._external_finish_method
            first_request = existing_method is None
            if first_request:
                self._external_finish_method = method_name
            already_finished = (
                self._provider_terminal_prepared
                or self._provider_silent_closed
            )

        if not first_request:
            await self._provider_finish_done.wait()
            if method_name == "aclose":
                await super().aclose()
            return

        if not already_finished:
            active_operation = self._active_provider_task
            consumer_active = self._provider_consumer_owner is not None
            if active_operation is current_task():
                return
            await self._cancel_active_provider_pull()
            if active_operation is not None and consumer_active:
                await self._provider_finish_done.wait()
                if method_name == "aclose":
                    await self._provider_consumer_done.wait()
            elif method_name == "cancel":
                await self._finalize_provider_terminal(
                    _OpenAIProviderTerminal(
                        event=StreamProviderEvent(
                            kind=StreamItemKind.STREAM_CANCELLED
                        ),
                        succeeded=False,
                        cleanup_method="cancel",
                    )
                )
            else:
                await self._finalize_provider_close()

        cleanup_errors: list[BaseException] = []
        if self._provider_finish_error is not None:
            cleanup_errors.append(self._provider_finish_error)
        if method_name == "aclose":
            try:
                await super().aclose()
            except BaseException as error:
                cleanup_errors.append(self._cleanup_boundary_error(error))
        self._raise_cleanup_errors(cleanup_errors)

    @staticmethod
    def _raise_cleanup_errors(errors: list[BaseException]) -> None:
        if len(errors) == 1:
            raise errors[0] from None
        if errors:
            raise BaseExceptionGroup(
                "OpenAI stream cleanup failed",
                errors,
            ) from None

    async def _cleanup_provider_sources_locked(
        self,
        method_name: str,
    ) -> list[BaseException]:
        cleanup_methods, errors = self._detach_provider_cleanup_methods(
            method_name
        )
        for cleanup_method in cleanup_methods:
            try:
                result = cleanup_method()
                if isawaitable(result):
                    awaited_result = await cast(Awaitable[object], result)
                    assert awaited_result is None
                else:
                    assert result is None
            except BaseException as error:
                errors.append(error)
                if not isinstance(error, Exception):
                    break
        return errors

    def _detach_provider_cleanup_methods(
        self,
        method_name: str,
    ) -> tuple[list[Callable[[], object]], list[BaseException]]:
        assert method_name in {"aclose", "cancel"}
        sources = (self._stream, *self._stream_sources)
        self._stream = None
        self._stream_sources = ()
        if method_name == "cancel":
            self._stream_cancelled = True
        method_names = (
            ("cancel", "close", "aclose")
            if method_name == "cancel"
            else ("aclose", "close")
        )
        cleanup_methods: list[Callable[[], object]] = []
        errors: list[BaseException] = []
        seen: set[int] = set()
        for source in sources:
            if source is None:
                continue
            source_id = id(source)
            if source_id in seen:
                continue
            seen.add(source_id)
            try:
                cleanup_method: object | None = None
                for cleanup_method_name in method_names:
                    cleanup_method = getattr(
                        source,
                        cleanup_method_name,
                        None,
                    )
                    if cleanup_method is not None:
                        break
                if cleanup_method is None:
                    continue
                assert callable(cleanup_method)
                cleanup_methods.append(cleanup_method)
            except BaseException as error:
                errors.append(error)
                if not isinstance(error, Exception):
                    break
        return cleanup_methods, errors

    async def _cancel_active_provider_pull(self) -> None:
        active_task = self._active_provider_task
        if active_task is None or active_task is current_task():
            return
        if not active_task.done():
            active_task.cancel()
        try:
            await active_task
        except BaseException:
            pass
        finally:
            if self._active_provider_task is active_task:
                self._active_provider_task = None

    def _cleanup_boundary_error(
        self,
        error: BaseException,
    ) -> BaseException:
        if not (self._request_has_replay_items or self._private_output_seen):
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
        canonical = normalize_provider_stream(
            self._provider_events(),
            stream_session_id=stream_session_id,
            run_id=run_id,
            turn_id=turn_id,
            provider_family=self._effective_provider_family(
                provider_family,
                capabilities,
            ),
            capabilities=capabilities
            or StreamProviderCapabilities(
                backend=StreamProducerBackend.HOSTED,
                provider_family=self._provider_family,
                supports_reasoning=True,
                supports_reasoning_summary=(self._supports_reasoning_summary),
                supports_tool_calls=True,
                supports_usage=True,
                supports_terminal_events=True,
                supports_cancellation=True,
            ),
            close_after_terminal=close_after_terminal,
        )
        return self._guard_canonical_stream(canonical)

    async def _guard_canonical_stream(
        self,
        items: AsyncIterator[CanonicalStreamItem],
    ) -> AsyncIterator[CanonicalStreamItem]:
        iterator = items.__aiter__()
        try:
            while True:
                if self._external_finish_method == "aclose":
                    return
                try:
                    item = await iterator.__anext__()
                except StopAsyncIteration:
                    return
                if self._external_finish_method == "aclose":
                    return
                yield item
        finally:
            await self._call_stream_source_cleanup(iterator, "aclose")
            if (
                not self._provider_lifetime_claimed
                and self._external_finish_method is None
                and not self._provider_terminal_prepared
                and not self._provider_silent_closed
            ):
                self._provider_lifetime_claimed = True
                await self._finalize_provider_close()

    async def _provider_events(self) -> AsyncIterator[StreamProviderEvent]:
        consumer_owner = object()
        if (
            self._provider_lifetime_claimed
            or self._provider_consumer_owner is not None
            or self._active_provider_task is not None
        ):
            raise _OpenAIConcurrentProviderConsumerError(
                "OpenAI stream already has an active consumer"
            )
        self._provider_lifetime_claimed = True
        self._provider_consumer_owner = consumer_owner
        self._provider_consumer_done.clear()
        self._reset_response_attempt_state()
        terminal: _OpenAIProviderTerminal | None = None
        provider_iterator: AsyncIterator[Any] | None = None
        event: object | None = None
        provider_events: tuple[StreamProviderEvent, ...] = ()
        provider_event: StreamProviderEvent | None = None
        sanitized_events: list[StreamProviderEvent] = []
        sanitized_event: StreamProviderEvent | None = None
        replacement_stream: AsyncIterator[Any] | None = None
        try:
            try:
                attempts = 0
                if self._provider_terminal_prepared:
                    assert self._provider_terminal_event is not None
                    stored_event = self._provider_terminal_event
                    terminal = _OpenAIProviderTerminal(
                        event=stored_event,
                        succeeded=(
                            stored_event.kind
                            is StreamItemKind.STREAM_COMPLETED
                        ),
                    )
                while terminal is None:
                    retry = False
                    output_seen = False
                    self._raise_if_provider_interrupted()
                    assert self._stream is not None
                    provider_iterator = self._stream.__aiter__()
                    while True:
                        try:
                            self._raise_if_provider_interrupted()
                            assert provider_iterator is not None
                            event = await self._pull_provider_event(
                                provider_iterator
                            )
                        except StopAsyncIteration:
                            break
                        self._raise_if_provider_interrupted()
                        provider_event_type: str | None = None
                        try:
                            provider_events = self._provider_events_from_event(
                                event
                            )
                            provider_event_type = next(
                                (
                                    provider_event.provider_event_type
                                    for provider_event in provider_events
                                    if provider_event.provider_event_type
                                    is not None
                                ),
                                self._best_effort_event_type(event),
                            )
                        except _OpenAIReasoningSummaryEventError as exc:
                            raise self._reasoning_summary_adapter_error(
                                exc
                            ) from None
                        except StreamProviderAdapterError:
                            raise
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
                            provider_event_type = self._best_effort_event_type(
                                event
                            )
                            if (
                                self._request_has_replay_items
                                or self._private_output_seen
                            ):
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
                        self._raise_if_provider_interrupted()
                        if (
                            self._request_has_replay_items
                            or self._private_output_seen
                        ):
                            self._usage = self._private_replay_usage(
                                self._usage
                            )
                            sanitized_events = []
                            for provider_event in provider_events:
                                sanitized_event = (
                                    self._sanitize_private_replay_event(
                                        provider_event
                                    )
                                    if self._request_has_replay_items
                                    else self._sanitize_private_output_event(
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
                            self._raise_if_provider_interrupted()
                            if self._is_model_output_event(provider_event):
                                output_seen = True
                            if provider_event.kind in {
                                StreamItemKind.STREAM_CANCELLED,
                                StreamItemKind.STREAM_COMPLETED,
                                StreamItemKind.STREAM_ERRORED,
                            }:
                                terminal = _OpenAIProviderTerminal(
                                    event=provider_event,
                                    succeeded=(
                                        provider_event.kind
                                        is StreamItemKind.STREAM_COMPLETED
                                    ),
                                )
                                break
                            yield provider_event
                            if (
                                provider_event.kind
                                is not StreamItemKind.REASONING_DELTA
                            ):
                                self._reasoning_segments.complete_segment()
                        if terminal is not None:
                            break
                        event = None
                        provider_event = None
                        provider_events = ()
                    if terminal is not None:
                        break
                    if not retry:
                        self._raise_if_provider_interrupted()
                        try:
                            self._reasoning_summary_state.close_source()
                        except _OpenAIReasoningSummaryEventError as exc:
                            raise self._reasoning_summary_adapter_error(
                                exc
                            ) from None
                        terminal = _OpenAIProviderTerminal(
                            event=StreamProviderEvent(
                                kind=StreamItemKind.STREAM_COMPLETED
                            ),
                            succeeded=True,
                        )
                        break
                    retry_terminal = next(
                        (
                            candidate
                            for candidate in reversed(provider_events)
                            if candidate.kind
                            in {
                                StreamItemKind.STREAM_CANCELLED,
                                StreamItemKind.STREAM_COMPLETED,
                                StreamItemKind.STREAM_ERRORED,
                            }
                        ),
                        self._stream_cleanup_failure_event(),
                    )
                    try:
                        await self._close_current_stream()
                        self._rollback_response_attempt_output_items()
                    except BaseException:
                        terminal = _OpenAIProviderTerminal(
                            event=retry_terminal,
                            succeeded=(
                                retry_terminal.kind
                                is StreamItemKind.STREAM_COMPLETED
                            ),
                            cleanup_failed=True,
                        )
                        break
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
                        await self._run_provider_operation(sleep(delay))
                    await self._raise_if_retry_interrupted()
                    attempts += 1
                    replacement_stream = cast(
                        AsyncIterator[Any],
                        await self._run_provider_operation(
                            self._stream_factory()
                        ),
                    )
                    await self._raise_if_retry_interrupted(replacement_stream)
                    self._stream = replacement_stream
                    self._stream_sources = (self._stream,)
                    event = None
                    provider_iterator = None
                    provider_event = None
                    provider_events = ()
                    replacement_stream = None
            except StreamConsumerClosure:
                raise
            except StreamProviderAdapterError as error:
                terminal = _OpenAIProviderTerminal(
                    event=self._provider_adapter_error_event(error),
                    succeeded=False,
                )
            except Exception as error:
                if not (
                    self._request_has_replay_items or self._private_output_seen
                ):
                    terminal = _OpenAIProviderTerminal(
                        event=self._provider_exception_event(error),
                        succeeded=False,
                    )
                else:
                    terminal = _OpenAIProviderTerminal(
                        event=self._private_replay_provider_failure_event(),
                        succeeded=False,
                        include_cleanup_diagnostic=True,
                    )
        except CancelledError:
            if self._external_finish_method == "cancel":
                terminal = _OpenAIProviderTerminal(
                    event=StreamProviderEvent(
                        kind=StreamItemKind.STREAM_CANCELLED
                    ),
                    succeeded=False,
                    cleanup_method="cancel",
                )
            else:
                task = current_task()
                if task is not None and task.cancelling():
                    await self._finalize_provider_close()
                    self._release_provider_consumer(consumer_owner)
                    raise StreamConsumerCancellation() from None
                await self._finalize_provider_close()
                self._release_provider_consumer(consumer_owner)
                raise
        except BaseException:
            await self._finalize_provider_close()
            self._release_provider_consumer(consumer_owner)
            raise
        event = None
        provider_iterator = None
        provider_events = ()
        provider_event = None
        sanitized_events.clear()
        sanitized_event = None
        replacement_stream = None
        assert terminal is not None
        try:
            terminal_event = await self._finalize_provider_terminal(terminal)
        except StreamConsumerCancellation:
            self._release_provider_consumer(consumer_owner)
            raise
        except StreamConsumerClosure:
            self._release_provider_consumer(consumer_owner)
            raise
        try:
            if not self._provider_terminal_emitted:
                self._provider_terminal_emitted = True
                yield terminal_event
        finally:
            self._release_provider_consumer(consumer_owner)

    def _release_provider_consumer(self, owner: object) -> None:
        assert self._provider_consumer_owner is owner
        self._provider_consumer_owner = None
        self._provider_consumer_done.set()

    @staticmethod
    def _provider_adapter_error_event(
        error: StreamProviderAdapterError,
    ) -> StreamProviderEvent:
        return StreamProviderEvent(
            kind=StreamItemKind.STREAM_ERRORED,
            data=(
                error.safe_data
                if error.safe_data is not None
                else {
                    "error_type": error.error.__class__.__name__,
                    "message": str(error.error),
                }
            ),
            provider_payload=error.provider_payload,
            provider_event_type=error.provider_event_type,
        )

    @staticmethod
    def _provider_exception_event(error: Exception) -> StreamProviderEvent:
        return StreamProviderEvent(
            kind=StreamItemKind.STREAM_ERRORED,
            data={
                "error_type": error.__class__.__name__,
                "message": str(error),
            },
        )

    async def _pull_provider_event(
        self,
        iterator: AsyncIterator[Any],
    ) -> object:
        pull: Task[object] = create_task(self._pull_provider_next(iterator))
        self._active_provider_task = pull
        try:
            return await pull
        except BaseException:
            self._raise_if_provider_interrupted()
            raise
        finally:
            if self._active_provider_task is pull:
                self._active_provider_task = None

    @staticmethod
    async def _pull_provider_next(iterator: AsyncIterator[Any]) -> object:
        return await iterator.__anext__()

    async def _run_provider_operation(
        self,
        operation: Awaitable[object],
    ) -> object:
        async def await_operation() -> object:
            return await operation

        task = create_task(await_operation())
        self._active_provider_task = task
        try:
            return await task
        except BaseException:
            self._raise_if_provider_interrupted()
            raise
        finally:
            if self._active_provider_task is task:
                self._active_provider_task = None

    async def _finalize_provider_terminal(
        self,
        outcome: _OpenAIProviderTerminal,
    ) -> StreamProviderEvent:
        async with self._provider_terminal_lock:
            if self._provider_terminal_prepared:
                assert self._provider_terminal_event is not None
                return self._provider_terminal_event
            cleanup_errors = await self._cleanup_provider_sources_locked(
                outcome.cleanup_method
            )
            task = current_task()
            consumer_cleanup_cancelled = bool(
                self._external_finish_method is None
                and task is not None
                and task.cancelling()
            )
            cleanup_failed = outcome.cleanup_failed or bool(cleanup_errors)

            event = outcome.event
            succeeded = outcome.succeeded
            external_finish_method = self._external_finish_method
            if external_finish_method == "aclose":
                succeeded = False
            elif external_finish_method == "cancel":
                event = StreamProviderEvent(
                    kind=StreamItemKind.STREAM_CANCELLED
                )
                succeeded = False
            elif consumer_cleanup_cancelled:
                succeeded = False
            elif cleanup_failed and succeeded:
                event = self._stream_cleanup_failure_event()
                succeeded = False
            elif cleanup_failed and outcome.include_cleanup_diagnostic:
                event = self._private_replay_provider_failure_event(
                    cleanup_failed=True
                )

            if succeeded:
                replay_error: (
                    _OpenAIClientClosedError
                    | _ReasoningReplayRetentionError
                    | _ReplayOwnerAssociationError
                    | None
                ) = None
                try:
                    self._finish_replay_owner(succeeded=True)
                except (
                    _OpenAIClientClosedError,
                    _ReasoningReplayRetentionError,
                    _ReplayOwnerAssociationError,
                ) as error:
                    replay_error = error
                except BaseException:
                    replay_error = _ReplayOwnerAssociationError()
                if replay_error is not None:
                    event = self._replay_error_event(
                        replay_error,
                        event.provider_event_type,
                    )
                    succeeded = False

            if succeeded:
                self._attempt_output_item_count = 0
            else:
                cleanup_errors.extend(
                    self._rollback_and_release_provider_attempt()
                )

            self._reasoning_summary_state.abort()
            self._provider_terminal_cleanup_failed = cleanup_failed
            finish_error = self._cleanup_error_from_errors(cleanup_errors)
            if finish_error is not None:
                if self._provider_finish_error is None:
                    self._provider_finish_error = finish_error
                else:
                    self._provider_finish_error = BaseExceptionGroup(
                        "OpenAI stream cleanup failed",
                        [self._provider_finish_error, finish_error],
                    )
            if (
                consumer_cleanup_cancelled
                or external_finish_method == "aclose"
            ):
                self._provider_silent_closed = True
            else:
                self._provider_terminal_event = event
                self._provider_terminal_prepared = True
            self._provider_finish_done.set()
            if consumer_cleanup_cancelled:
                raise StreamConsumerCancellation() from None
            if external_finish_method == "aclose":
                raise StreamConsumerClosure() from None
            return event

    async def _finalize_provider_close(self) -> None:
        async with self._provider_terminal_lock:
            if (
                self._provider_terminal_prepared
                or self._provider_silent_closed
            ):
                self._provider_finish_done.set()
                return
            cleanup_errors = await self._cleanup_provider_sources_locked(
                "aclose"
            )
            cleanup_errors.extend(
                self._rollback_and_release_provider_attempt()
            )
            self._reasoning_summary_state.abort()
            finish_error = self._cleanup_error_from_errors(cleanup_errors)
            if finish_error is not None:
                if self._provider_finish_error is None:
                    self._provider_finish_error = finish_error
                else:
                    self._provider_finish_error = BaseExceptionGroup(
                        "OpenAI stream cleanup failed",
                        [self._provider_finish_error, finish_error],
                    )
            self._provider_silent_closed = True
            self._provider_finish_done.set()

    def _rollback_and_release_provider_attempt(
        self,
    ) -> list[BaseException]:
        errors: list[BaseException] = []
        try:
            self._rollback_response_attempt_output_items()
        except BaseException as error:
            errors.append(error)
        try:
            self._finish_replay_owner(succeeded=False)
        except BaseException as error:
            errors.append(error)
        return errors

    def _cleanup_error_from_errors(
        self,
        errors: list[BaseException],
    ) -> BaseException | None:
        sanitized = [self._cleanup_boundary_error(error) for error in errors]
        if len(sanitized) == 1:
            return sanitized[0]
        if sanitized:
            return BaseExceptionGroup(
                "OpenAI stream cleanup failed",
                sanitized,
            )
        return None

    @staticmethod
    def _stream_cleanup_failure_event() -> StreamProviderEvent:
        return StreamProviderEvent(
            kind=StreamItemKind.STREAM_ERRORED,
            data={
                "error": {
                    "type": "server_error",
                    "code": _OpenAICleanupError.code,
                    "message": "OpenAI stream cleanup failed",
                }
            },
        )

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
        if not any(
            provider_event.provider_event_type == "response.failed"
            for provider_event in provider_events
        ):
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
        self._reasoning_summary_state = _OpenAIReasoningSummaryState()
        if (
            not self._provider_terminal_prepared
            and self._external_finish_method is None
        ):
            self._provider_terminal_event = None
            self._provider_terminal_emitted = False
            self._provider_terminal_cleanup_failed = False
            self._provider_finish_error = None
            self._provider_finish_done.clear()
            self._provider_silent_closed = False
        if self._replay_owner is not None and not self._replay_owner.released:
            self._replay_owner.begin_attempt()

    @staticmethod
    def _reasoning_summary_adapter_error(
        error: _OpenAIReasoningSummaryEventError,
    ) -> StreamProviderAdapterError:
        return StreamProviderAdapterError(
            error,
            provider_payload=None,
            provider_event_type=error.event_type,
            safe_data=error.safe_data(),
        )

    def _reasoning_summary_event(
        self,
        emission: _OpenAIReasoningSummaryEmission,
    ) -> StreamProviderEvent:
        correlation = StreamItemCorrelation(
            protocol_item_id=emission.item_id,
            provider_output_index=emission.output_index,
            provider_summary_index=emission.summary_index,
        )
        follows_boundary = (
            self._reasoning_segments.next_allocation_follows_boundary
        )
        ordinal = self._reasoning_segments.allocate(
            StreamReasoningRepresentation.SUMMARY,
            correlation,
        )
        return StreamProviderEvent(
            kind=StreamItemKind.REASONING_DELTA,
            text_delta=emission.text,
            correlation=correlation,
            visibility=StreamVisibility.PRIVATE,
            reasoning_representation=StreamReasoningRepresentation.SUMMARY,
            segment_instance_ordinal=ordinal,
            metadata=(
                {REASONING_SEGMENT_BOUNDARY_METADATA_KEY: "completed"}
                if follows_boundary
                else {}
            ),
            provider_payload=None,
            provider_event_type=emission.provider_event_type,
        )

    def _rollback_response_attempt_output_items(self) -> None:
        if self._replay_owner is not None and not self._replay_owner.released:
            self._replay_owner.rollback_attempt()
        output_item_count = self._attempt_output_item_count
        self._attempt_output_item_count = 0
        if output_item_count <= 0:
            return
        rollback = self._output_item_rollback
        if rollback is None:
            return
        rollback(output_item_count)

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
                *OpenAIStream._INCOMPLETE_EVENTS,
                *OpenAIStream._TEXT_DELTA_EVENTS,
                *OpenAIStream._TEXT_DONE_EVENTS,
                *OpenAIStream._REASONING_DELTA_EVENTS,
                *OpenAIStream._REASONING_DONE_EVENTS,
                *OpenAIStream._REASONING_SUMMARY_EVENTS,
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
    def _sanitize_private_output_event(
        event: StreamProviderEvent,
    ) -> StreamProviderEvent:
        sanitized = OpenAIStream._sanitize_private_replay_event(event)
        event_data = event.data
        error_data = (
            event_data.get("error") if type(event_data) is dict else None
        )
        error_code = (
            error_data.get("code") if type(error_data) is dict else None
        )
        if (
            event.kind is StreamItemKind.STREAM_ERRORED
            and type(error_code) is str
            and error_code == _ReasoningReplayRetentionError.code
        ):
            return replace(
                sanitized,
                data={
                    "error": {
                        "type": "server_error",
                        "code": _ReasoningReplayRetentionError.code,
                        "message": (
                            "OpenAI reasoning replay state is invalid or "
                            "exceeds its retention limit."
                        ),
                    }
                },
            )
        if (
            event.kind is StreamItemKind.STREAM_ERRORED
            and event.provider_event_type in OpenAIStream._INCOMPLETE_EVENTS
        ):
            return replace(sanitized, data=event.data)
        return sanitized

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
        try:
            self._replay_owner_releaser(owner)
        finally:
            owner.release()

    async def _close_current_stream(self) -> None:
        async with self._provider_terminal_lock:
            errors = await self._cleanup_provider_sources_locked("aclose")
        if errors and (
            self._request_has_replay_items or self._private_output_seen
        ):
            raise _OpenAICleanupError("stream") from None
        self._raise_cleanup_errors(errors)

    async def _raise_if_retry_interrupted(
        self,
        stream: AsyncIterator[Any] | None = None,
    ) -> None:
        try:
            self._raise_if_provider_interrupted()
        except BaseException:
            if stream is not None:
                try:
                    await self._call_stream_source_cleanup(stream, "aclose")
                except BaseException:
                    cleanup_error: BaseException = _OpenAICleanupError(
                        "stream"
                    )
                    assert self._provider_finish_error is None
                    self._provider_finish_error = cleanup_error
            raise

    def _raise_if_provider_interrupted(self) -> None:
        if self._external_finish_method == "aclose":
            raise StreamConsumerClosure()
        if self._external_finish_method == "cancel":
            raise CancelledError()
        task = current_task()
        if task is None or not task.cancelling():
            return
        raise StreamConsumerCancellation()

    @staticmethod
    def _best_effort_event_type(event: object) -> str | None:
        try:
            event_type = OpenAIClient._response_field(event, "type")
        except Exception:
            return None
        return event_type if type(event_type) is str else None

    @staticmethod
    def _response_event_type(event: object) -> str | None:
        try:
            event_type = OpenAIClient._response_field(event, "type")
        except Exception:
            raise _OpenAIReasoningSummaryEventError(
                event_type="response.unknown",
                field="type",
                value_shape="unreadable",
            ) from None
        if isinstance(event_type, str) and type(event_type) is not str:
            raise _OpenAIReasoningSummaryEventError(
                event_type="response.unknown",
                field="type",
                value=event_type,
            ) from None
        if event_type is None:
            raise _OpenAIReasoningSummaryEventError(
                event_type="response.unknown",
                field="type",
                value_shape="missing",
            )
        if type(event_type) is not str:
            raise ValueError("response event type must be a string")
        return event_type

    @staticmethod
    def _validate_reasoning_summary_provider_payload(
        event: object,
        event_type: str,
    ) -> None:
        if isinstance(event, Mapping):
            return
        try:
            payload = OpenAIStream._raw_provider_payload(event)
            if payload is not None:
                _sanitize_provider_json_payload(payload)
        except _ReasoningReplayRetentionError:
            raise _OpenAIReasoningSummaryEventError(
                event_type=event_type,
                field="provider_payload",
                value_shape="unreadable",
            ) from None

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
        if self._event_may_contain_private_output(event, None):
            self._private_output_seen = True
        if not self._provider_event_graph_is_readable(event):
            raise _OpenAIReasoningSummaryEventError(
                event_type=self._preclassified_event_type(event),
                field="provider_payload",
                value_shape="unreadable",
            )
        event_type = self._response_event_type(event)
        assert event_type is not None
        if event_type in self._REASONING_SUMMARY_EVENTS:
            self._validate_reasoning_summary_provider_payload(
                event, event_type
            )
        try:
            provider_events = self._map_provider_events_from_event(
                event, event_type
            )
        except _OpenAIToolCallIdError as error:
            raise _OpenAIReasoningSummaryEventError(
                event_type=event_type,
                field="item.call_id",
                output_index=OpenAIClient._response_field(
                    event,
                    "output_index",
                ),
                value_shape=error.value_shape,
                public_message=_OPENAI_TOOL_CALL_ID_ERROR_MESSAGE,
            ) from None
        if not self._private_output_seen:
            return provider_events
        return tuple(
            self._sanitize_private_output_event(provider_event)
            for provider_event in provider_events
        )

    @classmethod
    def _event_may_contain_private_output(
        cls,
        event: object,
        event_type: str | None,
    ) -> bool:
        _ = event_type
        payload: object
        if type(event) is dict:
            raw_event_type = event.get("type", _MISSING_PROVIDER_FIELD)
            if type(raw_event_type) is not str:
                return True
            return cls._privacy_graph_may_be_private(event)
        if type(event) is SimpleNamespace:
            payload = event.__dict__
        elif _is_trusted_openai_response_stream_event(event):
            try:
                payload = cls._raw_provider_payload(event)
            except BaseException:
                return True
            if type(payload) is not dict:
                return True
        else:
            return True
        assert type(payload) is dict
        raw_event_type = payload.get("type", _MISSING_PROVIDER_FIELD)
        if type(raw_event_type) is not str:
            return True
        return cls._privacy_graph_may_be_private(payload)

    @classmethod
    def _provider_event_graph_is_readable(cls, value: object) -> bool:
        if not (
            type(value) in {dict, SimpleNamespace}
            or _is_trusted_openai_response_stream_event(value)
        ):
            return False
        container_fields = {
            "content",
            "item",
            "output",
            "part",
            "response",
            "summary",
        }
        work: list[tuple[object, str | None]] = [(value, None)]
        seen: set[int] = set()
        while work:
            current, parent_field = work.pop()
            if (
                isinstance(current, str)
                or type(current) in {bool, float, int}
                or current is None
            ):
                continue
            current_id = id(current)
            if current_id in seen:
                return False
            seen.add(current_id)
            if type(current) is dict:
                for key, item in current.items():
                    if type(key) is not str:
                        return False
                    work.append((item, key))
                continue
            if type(current) is list:
                work.extend((item, parent_field) for item in current)
                continue
            if type(current) is SimpleNamespace:
                work.append((current.__dict__, parent_field))
                continue
            if _is_trusted_openai_response_stream_event(current):
                continue
            if isinstance(current, (Mapping, Sequence)):
                return False
            if parent_field in container_fields:
                return False
        return True

    @staticmethod
    def _preclassified_event_type(event: object) -> str:
        if type(event) is dict:
            event_type = event.get("type")
            if type(event_type) is str and event_type:
                return event_type
        if type(event) is SimpleNamespace:
            event_type = event.__dict__.get("type")
            if type(event_type) is str and event_type:
                return event_type
        return "response.unknown"

    @classmethod
    def _privacy_graph_may_be_private(
        cls,
        value: object,
        *,
        seen: set[int] | None = None,
    ) -> bool:
        if type(value) in {bool, float, int, str} or value is None:
            return False
        if seen is None:
            seen = set()
        value_id = id(value)
        if value_id in seen:
            return True
        seen.add(value_id)
        if type(value) is list:
            return any(
                cls._privacy_graph_may_be_private(item, seen=seen)
                for item in value
            )
        if type(value) is dict:
            for key, item in value.items():
                if type(key) is not str:
                    return True
                if key in {"encrypted_content", "summary"}:
                    return True
                if key == "type":
                    if type(item) is not str:
                        return True
                    if "reasoning" in item:
                        return True
                if cls._privacy_graph_may_be_private(item, seen=seen):
                    return True
            return False
        if isinstance(value, (Mapping, Sequence)):
            return True
        if type(value) is SimpleNamespace:
            return cls._privacy_graph_may_be_private(
                value.__dict__,
                seen=seen,
            )
        if _is_trusted_openai_response_stream_event(value):
            try:
                payload = cls._raw_provider_payload(value)
            except BaseException:
                return True
            return cls._privacy_graph_may_be_private(payload, seen=seen)
        return True

    def _map_provider_events_from_event(
        self,
        event: object,
        event_type: str | None,
    ) -> tuple[StreamProviderEvent, ...]:
        provider_payload = (
            None
            if self._private_output_seen
            else self._provider_payload(event)
        )
        if event_type in self._REASONING_SUMMARY_EVENTS:
            response = None
            error = None
        else:
            response = OpenAIClient._response_field(event, "response")
            error = OpenAIClient._response_field(
                event, "error"
            ) or OpenAIClient._response_field(response, "error")

        if event_type in self._ERROR_EVENTS or error is not None:
            assert event_type is not None
            self._reasoning_summary_state.validate_terminal_output(
                response,
                event_type,
            )
            self._reasoning_summary_state.close_response(
                event_type, completed=False
            )
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
            assert event_type is not None
            self._reasoning_summary_state.validate_terminal_output(
                response,
                event_type,
            )
            self._reasoning_summary_state.close_response(
                event_type, completed=False
            )
            return self._incomplete_events(event, provider_payload, event_type)
        if event_type == "response.completed":
            self._reasoning_summary_state.validate_terminal_output(
                response, event_type
            )
            self._reasoning_summary_state.close_response(
                event_type, completed=True
            )
            return self._completion_events(event, provider_payload, event_type)
        if event_type in self._TEXT_DELTA_EVENTS:
            self._reasoning_summary_state.validate_semantic_event(
                event,
                event_type,
                allowed_item_types=self._TEXT_OUTPUT_ITEM_TYPES,
                implied_item_type="output_text",
                require_open=True,
                validate_content_index=True,
            )
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
            fingerprint = (
                self._reasoning_summary_state.validate_semantic_event(
                    event,
                    event_type,
                    allowed_item_types=self._TEXT_OUTPUT_ITEM_TYPES,
                    implied_item_type="output_text",
                    validate_content_index=True,
                )
            )
            if self._answer_done_seen:
                self._reasoning_summary_state.mark_channel_closed(fingerprint)
                return ()
            text = self._response_optional_string_field(
                event, event_type, "text", "delta"
            )
            if text and not self._answer_text_seen:
                self._answer_text_seen = True
                self._answer_done_seen = True
                self._reasoning_summary_state.mark_channel_closed(fingerprint)
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
                self._reasoning_summary_state.mark_channel_closed(fingerprint)
                return ()
            self._answer_done_seen = True
            self._reasoning_summary_state.mark_channel_closed(fingerprint)
            return (
                StreamProviderEvent(
                    kind=StreamItemKind.ANSWER_DONE,
                    provider_payload=provider_payload,
                    provider_event_type=event_type,
                ),
            )
        if event_type in self._REASONING_DELTA_EVENTS:
            fingerprint = (
                self._reasoning_summary_state.validate_semantic_event(
                    event,
                    event_type,
                    allowed_item_types=self._NATIVE_REASONING_ITEM_TYPES,
                    implied_item_type="reasoning",
                    require_open=True,
                )
            )
            assert fingerprint is not None
            delta = self._response_string_field(event, "delta", event_type)
            self._reasoning_summary_state.add_native_reasoning_delta(
                event,
                event_type,
                fingerprint,
                delta,
            )
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
            fingerprint = (
                self._reasoning_summary_state.validate_semantic_event(
                    event,
                    event_type,
                    allowed_item_types=self._NATIVE_REASONING_ITEM_TYPES,
                    implied_item_type="reasoning",
                    require_open=True,
                )
            )
            assert fingerprint is not None
            if self._reasoning_summary_state.finish_native_reasoning(
                event,
                event_type,
                fingerprint,
            ):
                self._reasoning_segments.complete_segment()
            return ()
        if event_type == self._REASONING_SUMMARY_PART_ADDED_EVENT:
            self._reasoning_summary_state.add_part(event, event_type)
            return ()
        if event_type == self._REASONING_SUMMARY_TEXT_DELTA_EVENT:
            emission = self._reasoning_summary_state.add_delta(
                event, event_type
            )
            if emission is None:
                return ()
            return (self._reasoning_summary_event(emission),)
        if event_type == self._REASONING_SUMMARY_TEXT_DONE_EVENT:
            self._reasoning_summary_state.finish_text(event, event_type)
            return ()
        if event_type == self._REASONING_SUMMARY_PART_DONE_EVENT:
            self._reasoning_summary_state.finish_part(event, event_type)
            self._reasoning_segments.complete_segment()
            return ()
        if event_type == "response.output_item.added":
            item_type = self._reasoning_summary_state.output_item_type(
                event, event_type
            )
            if item_type == "reasoning":
                self._private_output_seen = True
            item = OpenAIClient._response_field(event, "item")
            call_id = (
                self._tool_call_id_from_item(item)
                if item_type in self._TOOL_CALL_ITEM_TYPES
                else None
            )
            canonical_name = (
                self._tool_call_name_from_item(item)
                if item_type in self._TOOL_CALL_ITEM_TYPES
                else None
            )
            item_type, _ = self._reasoning_summary_state.observe_output_item(
                event,
                event_type,
                item_type,
                call_id=call_id,
                canonical_name=canonical_name,
            )
            self._reasoning_summary_state.add_item(
                event, event_type, item_type
            )
            self._record_output_item(
                event,
                call_id=call_id,
                canonical_name=canonical_name,
            )
            return ()
        if event_type in self._TOOL_ARGUMENT_DELTA_EVENTS:
            implied_item_type = (
                "custom_tool_call"
                if event_type.startswith("response.custom_")
                else "function_call"
            )
            call_id = self._tool_call_id_from_event(event)
            assert call_id is not None
            canonical_name = self._tool_call_name_from_item(event)
            fingerprint = (
                self._reasoning_summary_state.validate_semantic_event(
                    event,
                    event_type,
                    allowed_item_types=(
                        self._CUSTOM_ARGUMENT_ITEM_TYPES
                        if event_type.startswith("response.custom_")
                        else self._FUNCTION_ARGUMENT_ITEM_TYPES
                    ),
                    implied_item_type=implied_item_type,
                    call_id=call_id,
                    canonical_name=canonical_name,
                    require_open=True,
                )
            )
            assert fingerprint is not None
            delta = self._response_string_field(event, "delta", event_type)
            self._reasoning_summary_state.add_tool_argument_delta(
                fingerprint,
                delta,
                event_type=event_type,
                output_index=OpenAIClient._response_field(
                    event, "output_index"
                ),
            )
            self._bind_canonical_tool_state(
                call_id,
                fingerprint.item_id,
                canonical_name,
                event_type=event_type,
            )
            return self._tool_argument_delta_events(
                event,
                provider_payload,
                event_type,
                call_id=call_id,
                delta=delta,
            )
        if event_type in self._TOOL_ARGUMENT_DONE_EVENTS:
            implied_item_type = (
                "custom_tool_call"
                if event_type.startswith("response.custom_")
                else "function_call"
            )
            call_id = self._tool_call_id_from_event(event)
            assert call_id is not None
            canonical_name = self._tool_call_name_from_item(event)
            fingerprint = (
                self._reasoning_summary_state.validate_semantic_event(
                    event,
                    event_type,
                    allowed_item_types=(
                        self._CUSTOM_ARGUMENT_ITEM_TYPES
                        if event_type.startswith("response.custom_")
                        else self._FUNCTION_ARGUMENT_ITEM_TYPES
                    ),
                    implied_item_type=implied_item_type,
                    call_id=call_id,
                    canonical_name=canonical_name,
                )
            )
            assert fingerprint is not None
            arguments = self._tool_call_arguments_from_item(event)
            self._reasoning_summary_state.finish_tool_arguments(
                fingerprint,
                arguments,
                event_type=event_type,
                output_index=OpenAIClient._response_field(
                    event, "output_index"
                ),
                close=True,
            )
            self._bind_canonical_tool_state(
                call_id,
                fingerprint.item_id,
                canonical_name,
                event_type=event_type,
            )
            return self._tool_argument_done_events(
                event,
                provider_payload,
                event_type,
                call_id=call_id,
                arguments=arguments,
            )
        if event_type == "response.output_item.done":
            item_type = self._reasoning_summary_state.output_item_type(
                event, event_type
            )
            if item_type == "reasoning":
                self._private_output_seen = True
            item = OpenAIClient._response_field(event, "item")
            call_id = (
                self._tool_call_id_from_item(item)
                if item_type in self._TOOL_CALL_ITEM_TYPES
                else None
            )
            canonical_name = (
                self._tool_call_name_from_item(item)
                if item_type in self._TOOL_CALL_ITEM_TYPES
                else None
            )
            item_type, fingerprint = (
                self._reasoning_summary_state.observe_output_item(
                    event,
                    event_type,
                    item_type,
                    call_id=call_id,
                    canonical_name=canonical_name,
                )
            )
            state = self._reasoning_summary_state
            completion_is_duplicate = (
                state.output_item_completion_is_duplicate(
                    event,
                    event_type,
                    fingerprint,
                )
            )
            if completion_is_duplicate:
                return ()
            if item_type == "reasoning":
                emissions, record_output_item, _ = (
                    self._reasoning_summary_state.complete_item(
                        event, event_type
                    )
                )
                provider_events = tuple(
                    self._reasoning_summary_event(emission)
                    for emission in emissions
                )
                if record_output_item:
                    self._record_done_output_item(event)
                self._reasoning_segments.complete_segment()
                self._reasoning_summary_state.mark_output_item_completed(
                    event,
                    event_type,
                    fingerprint,
                )
                return provider_events
            if item_type in self._TOOL_CALL_ITEM_TYPES:
                if call_id is not None and fingerprint is not None:
                    arguments = self._tool_call_arguments_from_item(item)
                    self._reasoning_summary_state.finish_tool_arguments(
                        fingerprint,
                        arguments,
                        event_type=event_type,
                        output_index=OpenAIClient._response_field(
                            event, "output_index"
                        ),
                        close=True,
                        allow_closed=True,
                    )
                    self._bind_canonical_tool_state(
                        call_id,
                        fingerprint.item_id,
                        canonical_name,
                        event_type=event_type,
                    )
            provider_events = self._tool_done_events(
                event,
                provider_payload,
                event_type,
                item_type=item_type,
                call_id=call_id,
                canonical_name=canonical_name,
            )
            self._record_done_output_item(event)
            self._reasoning_summary_state.mark_output_item_completed(
                event,
                event_type,
                fingerprint,
            )
            return provider_events
        return ()

    @staticmethod
    def _reasoning_correlation(event: object) -> StreamItemCorrelation:
        item_id = OpenAIClient._response_field(event, "item_id")
        if item_id is not None and (
            type(item_id) is not str or not item_id.strip()
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

    def _record_done_output_item(
        self,
        event: object,
    ) -> None:
        item = OpenAIClient._response_field(event, "item")
        payload = self._raw_provider_payload(item)
        if not isinstance(payload, dict):
            return
        payload = self._response_input_item_payload(payload)
        raw_item_type = payload.get("type")
        item_type = raw_item_type if type(raw_item_type) is str else None
        if item_type == "reasoning":
            if not self._is_replayable_reasoning_item(payload):
                return
        elif item_type != "function_call":
            return
        if self._output_item_sink is None and self._replay_owner is None:
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
        return type(encrypted_content) is str and bool(encrypted_content)

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

    def _record_output_item(
        self,
        event: object,
        *,
        call_id: str | None = None,
        canonical_name: str | None = None,
    ) -> None:
        item = OpenAIClient._response_field(event, "item")
        if not self._is_tool_call_item(item):
            return
        if call_id is None:
            call_id = self._tool_call_id_from_item(item)
        item_id = self._tool_item_id_from_item(item)
        if canonical_name is None:
            canonical_name = self._tool_call_name_from_item(item)
        if call_id is None:
            return
        self._record_tool_call_item_id(item_id, call_id)
        state = self._canonical_tool_calls.setdefault(
            call_id,
            {"name": None, "arguments_seen": False},
        )
        self._record_tool_call_name(
            state,
            canonical_name,
            event_type="response.output_item.added",
        )
        if item_id is not None:
            state["protocol_item_id"] = item_id

    def _bind_canonical_tool_state(
        self,
        call_id: str,
        item_id: str | None,
        canonical_name: str | None,
        *,
        event_type: str,
    ) -> None:
        self._record_tool_call_item_id(item_id, call_id)
        state = self._canonical_tool_calls.setdefault(
            call_id,
            {"name": None, "arguments_seen": False},
        )
        self._record_tool_call_name(
            state,
            canonical_name,
            event_type=event_type,
        )
        if item_id is not None:
            state["protocol_item_id"] = item_id

    def _tool_argument_delta_events(
        self,
        event: object,
        provider_payload: LooseJsonValue | None,
        event_type: str,
        *,
        call_id: str,
        delta: str,
    ) -> tuple[StreamProviderEvent, ...]:
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

    def _tool_argument_done_events(
        self,
        event: object,
        provider_payload: LooseJsonValue | None,
        event_type: str,
        *,
        call_id: str,
        arguments: str | None,
    ) -> tuple[StreamProviderEvent, ...]:
        state = self._canonical_tool_calls.setdefault(
            call_id,
            {"name": None, "arguments_seen": False},
        )
        result: list[StreamProviderEvent] = []
        if arguments is not None and not state["arguments_seen"]:
            state["arguments_seen"] = True
            result.append(
                StreamProviderEvent(
                    kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    correlation=self._tool_call_correlation(event, call_id),
                    text_delta=arguments,
                    provider_payload=provider_payload,
                    provider_event_type=event_type,
                )
            )
        result.extend(
            self._mark_tool_ready(call_id, provider_payload, event_type)
        )
        return tuple(result)

    def _tool_done_events(
        self,
        event: object,
        provider_payload: LooseJsonValue | None,
        event_type: str,
        *,
        item_type: str | None,
        call_id: str | None,
        canonical_name: str | None,
    ) -> tuple[StreamProviderEvent, ...]:
        item = OpenAIClient._response_field(event, "item")
        if item_type in self._TEXT_OUTPUT_ITEM_TYPES:
            return self._message_done_events(
                item,
                provider_payload,
                event_type,
                output_index=OpenAIClient._response_field(
                    event, "output_index"
                ),
            )
        if item_type not in self._TOOL_CALL_ITEM_TYPES:
            return ()
        item_id = self._tool_item_id_from_item(item)
        if call_id is None:
            call_id = self._tool_call_id_from_event(event, required=False)
        if call_id is None:
            return ()
        self._record_tool_call_item_id(item_id, call_id)
        state = self._canonical_tool_calls.setdefault(
            call_id,
            {
                "name": canonical_name,
                "arguments_seen": False,
            },
        )
        self._record_tool_call_name(
            state,
            canonical_name,
            event_type=event_type,
        )

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
        *,
        output_index: object,
    ) -> tuple[StreamProviderEvent, ...]:
        if self._answer_text_seen or self._answer_done_seen:
            return ()
        text = self._message_done_text(
            item,
            event_type=event_type,
            output_index=output_index,
        )
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

    @classmethod
    def _completed_response_text(
        cls,
        response: object,
        *,
        event_type: str = "response.completed",
    ) -> str:
        output_text = OpenAIClient._response_field(response, "output_text")
        output = OpenAIClient._response_field(response, "output")
        if output is None:
            if output_text is None:
                return ""
            if type(output_text) is not str:
                raise _OpenAIReasoningSummaryEventError(
                    event_type=event_type,
                    field="response.output_text",
                    value=output_text,
                )
            return output_text
        assert type(output) is list
        parts: list[str] = []
        validated_message = False
        for output_index, item in enumerate(output):
            item_type = OpenAIClient._response_field(item, "type")
            if type(item_type) is str and item_type in {
                "message",
                "output_text",
            }:
                validated_message = True
                parts.append(
                    cls._message_done_text(
                        item,
                        event_type=event_type,
                        output_index=output_index,
                    )
                )
        if not validated_message:
            return ""
        derived_text = "".join(parts)
        if output_text is None:
            return derived_text
        if type(output_text) is not str:
            raise _OpenAIReasoningSummaryEventError(
                event_type=event_type,
                field="response.output_text",
                value=output_text,
            )
        if output_text != derived_text:
            raise _OpenAIReasoningSummaryEventError(
                event_type=event_type,
                field="response.output_text",
                output_index=0,
                value_shape="conflict",
            )
        return derived_text

    @staticmethod
    def _message_done_text(
        item: object,
        *,
        event_type: str,
        output_index: object,
    ) -> str:
        parts: list[str] = []
        contents = OpenAIClient._response_field(item, "content")
        if type(contents) is list:
            for content in contents:
                part_type = _OpenAIReasoningSummaryState._field(
                    content,
                    "type",
                    event_type,
                )
                if (
                    part_type is not _MISSING_PROVIDER_FIELD
                    and part_type is not None
                    and (
                        type(part_type) is not str
                        or part_type != "output_text"
                    )
                ):
                    raise _OpenAIReasoningSummaryEventError(
                        event_type=event_type,
                        field="item.content.type",
                        value=part_type,
                        output_index=output_index,
                        value_shape="unexpected_value",
                    )
                text = OpenAIClient._response_field(content, "text")
                if type(text) is str:
                    parts.append(text)
                elif text is not None:
                    raise _OpenAIReasoningSummaryEventError(
                        event_type=event_type,
                        field="item.content.text",
                        value=text,
                        output_index=output_index,
                    )
        if parts:
            return "".join(parts)
        direct_text = OpenAIClient._response_field(item, "text")
        if type(direct_text) is str:
            return direct_text
        if direct_text is not None:
            raise _OpenAIReasoningSummaryEventError(
                event_type=event_type,
                field="item.text",
                value=direct_text,
                output_index=output_index,
            )
        return "".join(parts)

    def _is_tool_call_item(self, item: object) -> bool:
        if item is None:
            return False
        if OpenAIClient._response_field(item, "custom_tool_call") is not None:
            return True
        item_type = OpenAIClient._response_field(item, "type")
        return item_type is None or (
            type(item_type) is str and item_type in self._TOOL_CALL_ITEM_TYPES
        )

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
            if type(value) is str and value.strip():
                if field_name == "call_id":
                    return value
                return self._tool_call_ids_by_item_id.get(value, value)
            raise _OpenAIToolCallIdError(value)
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
        ):
            if value is None:
                continue
            if type(value) is str and value.strip():
                return value
            raise _OpenAIToolCallIdError(value)
        item_id = OpenAIClient._response_field(item, "id")
        if item_id is None:
            return None
        if type(item_id) is str and item_id.strip():
            return self._tool_call_ids_by_item_id.get(item_id, item_id)
        raise _OpenAIToolCallIdError(item_id)

    @staticmethod
    def _tool_item_id_from_item(item: object) -> str | None:
        if item is None:
            return None
        for field_name in ("item_id", "id"):
            value = OpenAIClient._response_field(item, field_name)
            if value is None:
                continue
            assert type(value) is str and value.strip()
            return value
        return None

    def _record_tool_call_item_id(
        self,
        item_id: str | None,
        call_id: str,
    ) -> None:
        assert type(call_id) is str
        assert call_id.strip()
        if item_id is None:
            return
        self._tool_call_ids_by_item_id[item_id] = call_id

    @staticmethod
    def _record_tool_call_name(
        state: dict[str, str | bool | None],
        name: str | None,
        *,
        event_type: str,
    ) -> None:
        if name is None:
            return
        current = state.get("name")
        assert current is None or current == name, event_type
        state["name"] = name

    def _tool_call_correlation(
        self,
        source: object,
        call_id: str,
    ) -> StreamItemCorrelation:
        assert type(call_id) is str
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
        assert type(call_id) is str
        assert call_id.strip()
        item_id = state.get("protocol_item_id")
        return StreamItemCorrelation(
            tool_call_id=call_id,
            protocol_item_id=(
                item_id
                if type(item_id) is str and item_id != call_id
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
            if type(value) is str:
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
            OpenAIClient._response_field(item, "input"),
            OpenAIClient._response_field(custom, "input"),
        ):
            if value is None:
                continue
            if type(value) is str:
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
        if type(value) is str:
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
            if type(value) is str:
                return value
            raise ValueError(f"{event_type} {field_name} must be a string")
        return None

    @staticmethod
    def _response_error_data(error: object) -> LooseJsonValue:
        payload = OpenAIStream._provider_payload(error)
        if isinstance(payload, dict):
            return {"error": payload}
        message = OpenAIClient._response_field(error, "message")
        if type(message) is str:
            return {"error": {"message": message}}
        return {"error": {"message": "provider error"}}

    @staticmethod
    def _response_failure_data(response: object) -> LooseJsonValue:
        message = OpenAIClient._response_field(response, "message")
        error: dict[str, LooseJsonValue] = {
            "code": "response_failed",
            "message": (
                message
                if type(message) is str and message
                else "response failed"
            ),
        }
        status = OpenAIClient._response_field(response, "status")
        if type(status) is str and status:
            error["status"] = status
        response_id = OpenAIClient._response_field(response, "id")
        if type(response_id) is str and response_id:
            error["response_id"] = response_id
        return {"error": error}

    @staticmethod
    def _response_is_incomplete(response: object) -> bool:
        status = OpenAIClient._response_field(response, "status")
        return (type(status) is str and status == "incomplete") or (
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
        if type(reason) is str and reason in OpenAIStream._INCOMPLETE_REASONS:
            error["reason"] = reason
            error["message"] = f"{message}: {reason}"
        status = OpenAIClient._response_field(response, "status")
        if type(status) is str and status == "incomplete":
            error["status"] = status
        response_id = OpenAIClient._response_field(response, "id")
        if not isinstance(
            response, Mapping
        ) and OpenAIStream._is_safe_response_id(response_id):
            assert type(response_id) is str
            error["response_id"] = response_id
        return {"error": error}

    @staticmethod
    def _is_safe_response_id(value: object) -> bool:
        if type(value) is not str or not value.startswith(
            (
                "resp_",
                "resp-",
            )
        ):
            return False
        suffix = value[5:]
        return (
            bool(suffix)
            and len(suffix) <= 128
            and all(
                character.isascii()
                and (character.isalnum() or character in {"_", "-"})
                for character in suffix
            )
        )

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
        try:
            model_dump = getattr(event, "model_dump", None)
        except Exception:
            raise _ReasoningReplayRetentionError() from None
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

    @property
    def reasoning_summary_request_capability(
        self,
    ) -> ReasoningSummaryRequestCapability:
        """Return native OpenAI and Azure summary request support."""
        if _is_exact_native_openai_type(self, "OpenAIClient"):
            return _OPENAI_REASONING_SUMMARY_CAPABILITY
        return super().reasoning_summary_request_capability

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
                    supports_reasoning_summary=bool(
                        self.reasoning_summary_request_capability.supported_modes
                    ),
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
            response: TextGenerationNonStreamResult | None = None
            try:
                response = await self._non_stream_result(
                    client_stream,
                    tool=tool,
                    replay_owner=replay_owner,
                    request_has_replay_items=request_has_replay_items,
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
        close: object | None = None
        try:
            close = getattr(self._client, "close", None)
            if close is None:
                close = getattr(self._client, "aclose", None)
            if close is not None:
                assert callable(close)
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
            assert type(call_id) is str and call_id
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
                if type(synthetic_call_id) is str and synthetic_call_id:
                    call_id = synthetic_call_id
                    outputs_by_call_id[call_id] = synthetic[1]
            synthetic_records.append((call_id, synthetic))

        if not synthetic_records:
            return []

        messages: list[dict[str, Any]] = []
        matched_call_ids: set[str] = set()
        for response_item in replay_items:
            raw_item_type = response_item.get("type")
            item_type = raw_item_type if type(raw_item_type) is str else None
            if item_type == "reasoning":
                if OpenAIStream._is_replayable_reasoning_item(response_item):
                    copied = _strict_replay_json_copy(response_item)
                    assert isinstance(copied, dict)
                    messages.append(cast(dict[str, Any], copied))
                continue
            if item_type != "function_call":
                continue
            call_id = response_item.get("call_id")
            if type(call_id) is not str:
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

    async def _non_stream_result(
        self,
        response: object,
        *,
        tool: ToolManager | None,
        replay_owner: _OpenAIReplayOwner,
        request_has_replay_items: bool,
    ) -> TextGenerationNonStreamResult:
        synthetic_events = self._non_stream_response_events(response)
        source = self._iterate_non_stream_events(synthetic_events)
        stream = OpenAIStream(
            stream=source,
            provider_family=self._usage_provider_family.value,
            supports_reasoning_summary=bool(
                self.reasoning_summary_request_capability.supported_modes
            ),
            replay_owner=replay_owner,
            replay_owner_retainer=self._retain_replay_owner,
            replay_owner_releaser=self._discard_replay_owner,
            request_has_replay_items=request_has_replay_items,
            tool=tool,
        )
        self._raise_if_closed()
        self._register_active_replay_stream(replay_owner, stream)
        collected_events: list[StreamProviderEvent] = []
        async for event in stream._provider_events():
            collected_events.append(replace(event, provider_payload=None))
        events = tuple(collected_events)
        usage = self._response_field(response, "usage")
        if request_has_replay_items or stream._private_output_seen:
            usage = OpenAIStream._private_replay_usage(usage)
        answer_text = "".join(
            event.text_delta or ""
            for event in events
            if event.kind is StreamItemKind.ANSWER_DELTA
        )
        return TextGenerationNonStreamResult(
            events,
            answer_text=answer_text,
            provider_family=self._usage_provider_family,
            usage=usage,
        )

    @staticmethod
    async def _iterate_non_stream_events(
        events: tuple[dict[str, object], ...],
    ) -> AsyncIterator[object]:
        for event in events:
            yield event

    @classmethod
    def _non_stream_response_events(
        cls,
        response: object,
    ) -> tuple[dict[str, object], ...]:
        terminal_type = cls._non_stream_terminal_type(response)
        output = cls._response_field(response, "output")
        if output is None:
            raw_items: list[object] = []
        elif type(output) is list:
            raw_items = cast(list[object], output)
        else:
            raise _OpenAIReasoningSummaryEventError(
                event_type=terminal_type,
                field="response.output",
                value=output,
            )

        normalized_items: list[dict[str, object]] = []
        source_indices: list[int] = []
        for source_index, item in enumerate(raw_items):
            normalized = cls._non_stream_output_item(item, source_index)
            if normalized is None:
                continue
            normalized_items.append(normalized)
            source_indices.append(source_index)

        output_text = cls._response_field(response, "output_text")
        if not normalized_items and output_text is not None:
            if type(output_text) is not str:
                raise _OpenAIReasoningSummaryEventError(
                    event_type=terminal_type,
                    field="response.output_text",
                    value=output_text,
                )
            normalized_items.append(
                {
                    "id": "msg_non_stream_output_text",
                    "type": "message",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": output_text}],
                }
            )
            source_indices.append(0)

        events: list[dict[str, object]] = []
        terminal_item_count = max(
            len(raw_items),
            max(source_indices, default=-1) + 1,
        )
        terminal_items = [
            cls._non_stream_terminal_placeholder(output_index)
            for output_index in range(terminal_item_count)
        ]
        for item, source_index in zip(
            normalized_items,
            source_indices,
            strict=True,
        ):
            output_index = source_index
            added_item = cls._non_stream_added_item(item, source_index)
            completed_item = dict(item)
            completed_item["status"] = "completed"
            terminal_item = dict(item)
            terminal_item["status"] = cls._non_stream_terminal_item_status(
                item,
                terminal_type=terminal_type,
                output_index=output_index,
            )
            events.append(
                {
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": added_item,
                }
            )
            if item["type"] == "reasoning":
                events.extend(
                    cls._non_stream_native_reasoning_events(
                        item,
                        output_index=output_index,
                    )
                )
            if not (
                item["type"] in OpenAIStream._TOOL_CALL_ITEM_TYPES
                and item["status"] == "incomplete"
            ):
                events.append(
                    {
                        "type": "response.output_item.done",
                        "output_index": output_index,
                        "item": completed_item,
                    }
                )
            terminal_items[output_index] = terminal_item

        terminal_response: dict[str, object] = {"output": terminal_items}
        usage = cls._response_field(response, "usage")
        if usage is not None:
            terminal_response["usage"] = usage
        response_id = cls._response_field(response, "id")
        if OpenAIStream._is_safe_response_id(response_id):
            terminal_response["id"] = response_id
        response_status = cls._response_field(response, "status")
        if type(response_status) is str:
            terminal_response["status"] = response_status
        elif terminal_type == "response.failed":
            terminal_response["status"] = "failed"
        elif terminal_type == "response.incomplete":
            terminal_response["status"] = "incomplete"
        else:
            terminal_response["status"] = "completed"
        error = cls._response_field(response, "error")
        if error is not None:
            terminal_response["error"] = error
        incomplete_details = cls._response_field(
            response,
            "incomplete_details",
        )
        if incomplete_details is not None:
            terminal_response["incomplete_details"] = incomplete_details
        events.append({"type": terminal_type, "response": terminal_response})
        return tuple(events)

    @staticmethod
    def _non_stream_terminal_placeholder(
        output_index: int,
    ) -> dict[str, object]:
        assert type(output_index) is int and output_index >= 0
        return {
            "type": "message",
            "status": "completed",
            "content": [],
        }

    @classmethod
    def _non_stream_native_reasoning_events(
        cls,
        item: dict[str, object],
        *,
        output_index: int,
    ) -> tuple[dict[str, object], ...]:
        content = item.get("content")
        if content is None:
            return ()
        if type(content) is not list:
            raise _OpenAIReasoningSummaryEventError(
                event_type="response.output_item.done",
                field="item.content",
                value=content,
                output_index=output_index,
            )
        item_id = item["id"]
        assert type(item_id) is str and item_id.strip()
        events: list[dict[str, object]] = []
        for content_index, part in enumerate(cast(list[object], content)):
            part_type = cls._response_field(part, "type")
            text = cls._response_field(part, "text")
            if type(part_type) is not str or part_type != "reasoning_text":
                raise _OpenAIReasoningSummaryEventError(
                    event_type="response.output_item.done",
                    field="item.content.type",
                    value=part_type,
                    output_index=output_index,
                    value_shape="unexpected_value",
                )
            if type(text) is not str:
                raise _OpenAIReasoningSummaryEventError(
                    event_type="response.output_item.done",
                    field="item.content.text",
                    value=text,
                    output_index=output_index,
                )
            identity = {
                "item_id": item_id,
                "output_index": output_index,
                "content_index": content_index,
            }
            events.extend(
                (
                    {
                        "type": "response.reasoning_text.delta",
                        "delta": text,
                        **identity,
                    },
                    {
                        "type": "response.reasoning_text.done",
                        "text": text,
                        **identity,
                    },
                )
            )
        return tuple(events)

    @classmethod
    def _non_stream_terminal_type(cls, response: object) -> str:
        status = cls._response_field(response, "status")
        error = cls._response_field(response, "error")
        incomplete_details = cls._response_field(
            response,
            "incomplete_details",
        )
        if status is not None and type(status) is not str:
            raise _OpenAIReasoningSummaryEventError(
                event_type="response.completed",
                field="response.status",
                value=status,
            )
        if error is not None or status in {"failed", "cancelled"}:
            return "response.failed"
        if incomplete_details is not None or status == "incomplete":
            return "response.incomplete"
        if status not in {None, "completed"}:
            raise _OpenAIReasoningSummaryEventError(
                event_type="response.completed",
                field="response.status",
                value=status,
                value_shape="unexpected_value",
            )
        return "response.completed"

    @classmethod
    def _non_stream_output_item(
        cls,
        item: object,
        source_index: int,
    ) -> dict[str, object] | None:
        item_type = cls._response_field(item, "type")
        content = cls._response_field(item, "content")
        direct_text = cls._response_field(item, "text")
        nested_call = cls._response_field(item, "call")
        if item_type is None:
            if type(content) is list or direct_text is not None:
                item_type = "message"
            elif nested_call is not None:
                item_type = "function_call"
            else:
                return None
        if type(item_type) is not str:
            raise _OpenAIReasoningSummaryEventError(
                event_type="response.output_item.done",
                field="item.type",
                value=item_type,
                output_index=source_index,
            )
        if item_type in {"message", "output_text"}:
            return cls._non_stream_message_item(
                item,
                item_type=item_type,
                source_index=source_index,
            )
        if item_type == "reasoning":
            return cls._non_stream_reasoning_item(item, source_index)
        if item_type in OpenAIStream._TOOL_CALL_ITEM_TYPES:
            return cls._non_stream_tool_item(
                item,
                item_type=item_type,
                source_index=source_index,
            )
        return None

    @classmethod
    def _non_stream_message_item(
        cls,
        item: object,
        *,
        item_type: str,
        source_index: int,
    ) -> dict[str, object]:
        identifier = cls._non_stream_item_id(
            item,
            fallback=f"msg_non_stream_{source_index}",
            output_index=source_index,
        )
        normalized_content: list[dict[str, object]] = []
        content = cls._response_field(item, "content")
        if type(content) is list:
            for part in cast(list[object], content):
                part_type = cls._response_field(part, "type")
                if part_type is None:
                    part_type = "output_text"
                normalized_content.append(
                    {
                        "type": part_type,
                        "text": cls._response_field(part, "text"),
                    }
                )
        if not normalized_content:
            direct_text = cls._response_field(item, "text")
            if direct_text is not None:
                normalized_content.append(
                    {
                        "type": "output_text",
                        "text": direct_text,
                    }
                )
        return {
            "id": identifier,
            "type": item_type,
            "status": cls._non_stream_raw_item_status(item, source_index),
            "content": normalized_content,
        }

    @classmethod
    def _non_stream_reasoning_item(
        cls,
        item: object,
        source_index: int,
    ) -> dict[str, object]:
        identifier = cls._non_stream_item_id(
            item,
            fallback=None,
            output_index=source_index,
        )
        summary = cls._response_field(item, "summary")
        normalized_summary: object
        if summary is None:
            normalized_summary = []
        elif type(summary) is list:
            normalized_summary = [
                {
                    "type": cls._response_field(part, "type"),
                    "text": cls._response_field(part, "text"),
                }
                for part in cast(list[object], summary)
            ]
        else:
            normalized_summary = summary
        normalized: dict[str, object] = {
            "id": identifier,
            "type": "reasoning",
            "status": cls._non_stream_raw_item_status(item, source_index),
            "summary": normalized_summary,
        }
        encrypted_content = cls._response_field(item, "encrypted_content")
        if encrypted_content is not None:
            normalized["encrypted_content"] = encrypted_content
        content = cls._response_field(item, "content")
        if type(content) is list:
            normalized["content"] = [
                {
                    "type": cls._response_field(part, "type"),
                    "text": cls._response_field(part, "text"),
                }
                for part in cast(list[object], content)
            ]
        elif content is not None:
            normalized["content"] = content
        return normalized

    @classmethod
    def _non_stream_tool_item(
        cls,
        item: object,
        *,
        item_type: str,
        source_index: int,
    ) -> dict[str, object]:
        call = cls._response_field(item, "call") or item
        function = cls._response_field(call, "function") or call
        custom = cls._response_field(item, "custom_tool_call")
        effective = custom or function
        call_id = (
            cls._response_field(item, "call_id")
            or cls._response_field(call, "call_id")
            or cls._response_field(call, "id")
            or cls._response_field(custom, "id")
        )
        if type(call_id) is not str or not call_id.strip():
            raise _OpenAIReasoningSummaryEventError(
                event_type="response.output_item.done",
                field="item.call_id",
                value=call_id,
                output_index=source_index,
            )
        identifier = cls._non_stream_item_id(
            item,
            fallback=f"fc_non_stream_{source_index}",
            output_index=source_index,
        )
        name = cls._response_field(effective, "name")
        arguments = (
            cls._response_field(item, "arguments")
            or cls._response_field(item, "input")
            or cls._response_field(effective, "arguments")
            or cls._response_field(effective, "input")
        )
        normalized_type = (
            "custom_tool_call"
            if item_type == "custom_tool_call"
            else "function_call"
        )
        normalized: dict[str, object] = {
            "id": identifier,
            "type": normalized_type,
            "status": cls._non_stream_raw_item_status(item, source_index),
            "call_id": call_id,
            "name": name,
        }
        field_name = (
            "input" if normalized_type == "custom_tool_call" else "arguments"
        )
        normalized[field_name] = arguments
        return normalized

    @classmethod
    def _non_stream_item_id(
        cls,
        item: object,
        *,
        fallback: str | None,
        output_index: int,
    ) -> str:
        identifier = cls._response_field(item, "id")
        if identifier is None:
            if fallback is not None:
                return fallback
            raise _OpenAIReasoningSummaryEventError(
                event_type="response.output_item.done",
                field="item.id",
                output_index=output_index,
            )
        if type(identifier) is not str or not identifier.strip():
            raise _OpenAIReasoningSummaryEventError(
                event_type="response.output_item.done",
                field="item.id",
                value=identifier,
                output_index=output_index,
            )
        return identifier

    @classmethod
    def _non_stream_raw_item_status(
        cls,
        item: object,
        output_index: int,
    ) -> str:
        status = cls._response_field(item, "status")
        if status is None:
            return "completed"
        if type(status) is not str or status not in {
            "completed",
            "incomplete",
        }:
            raise _OpenAIReasoningSummaryEventError(
                event_type="response.output_item.done",
                field="item.status",
                value=status,
                output_index=output_index,
                value_shape="unexpected_value",
            )
        return status

    @staticmethod
    def _non_stream_added_item(
        item: dict[str, object],
        source_index: int,
    ) -> dict[str, object]:
        item_type = item["type"]
        identifier = item["id"]
        added: dict[str, object] = {
            "id": identifier,
            "type": item_type,
            "status": "in_progress",
        }
        if item_type == "reasoning":
            added["summary"] = []
        elif item_type in {"message", "output_text"}:
            added["content"] = []
        elif item_type in OpenAIStream._TOOL_CALL_ITEM_TYPES:
            added["call_id"] = item["call_id"]
            added["name"] = item["name"]
            field_name = (
                "input" if item_type == "custom_tool_call" else "arguments"
            )
            added[field_name] = ""
        else:
            raise AssertionError(
                f"unsupported non-stream item at index {source_index}"
            )
        return added

    @classmethod
    def _non_stream_terminal_item_status(
        cls,
        item: dict[str, object],
        *,
        terminal_type: str,
        output_index: int,
    ) -> str:
        status = item["status"]
        assert type(status) is str
        if terminal_type == "response.completed" and status != "completed":
            raise _OpenAIReasoningSummaryEventError(
                event_type=terminal_type,
                field="item.status",
                value=status,
                output_index=output_index,
                value_shape="unexpected_value",
            )
        return status

    @staticmethod
    def _non_stream_response_content(
        response: object,
        *,
        tool: ToolManager | None = None,
        provider_family: ProviderFamily | str | None = ProviderFamily.OPENAI,
    ) -> str:
        parts: list[str] = []
        output = OpenAIClient._response_field(response, "output")
        if type(output) is not list:
            return "".join(parts)

        for item in output:
            raw_item_type = OpenAIClient._response_field(item, "type")
            item_type = raw_item_type if type(raw_item_type) is str else None
            contents = OpenAIClient._response_field(item, "content")
            if type(contents) is not list:
                contents = []

            if item_type in {None, "message", "output_text"}:
                for content in contents:
                    text = OpenAIClient._response_field(content, "text")
                if type(text) is str:
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
    def reasoning_summary_request_capability(
        self,
    ) -> ReasoningSummaryRequestCapability:
        """Return pre-load native OpenAI and Azure summary support."""
        if _is_exact_native_openai_type(self, "OpenAIModel"):
            return _OPENAI_REASONING_SUMMARY_CAPABILITY
        return super().reasoning_summary_request_capability

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
        if isinstance(
            streamer,
            (TextGenerationNonStreamResult, TextGenerationSingleStream),
        ):
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
