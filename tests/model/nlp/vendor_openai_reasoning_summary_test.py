"""Test OpenAI reasoning-summary requests, retry, and opaque replay."""

from ast import Name, Yield, parse, walk
from asyncio import CancelledError, Event, create_task, gather, run, wait_for
from collections.abc import AsyncIterator, Iterator, Mapping
from copy import deepcopy
from inspect import getsource
from json import dumps, loads
from math import inf, nan
from pathlib import Path
from textwrap import dedent
from time import perf_counter
from traceback import format_exception
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai.types.responses import ResponseStreamEvent
from openai.types.responses.response_function_tool_call import (
    ResponseFunctionToolCall,
)
from openai.types.responses.response_function_web_search import (
    ResponseFunctionWebSearch,
)
from openai.types.responses.response_reasoning_item import (
    Content as ResponseReasoningContent,
)
from openai.types.responses.response_reasoning_item import (
    ResponseReasoningItem,
)
from openai.types.responses.response_reasoning_item import (
    Summary as ResponseReasoningSummary,
)
from pydantic import TypeAdapter, ValidationError

from avalan.entities import (
    GenerationSettings,
    Message,
    MessageRole,
    ReasoningEffort,
    ReasoningSettings,
    ReasoningSummaryMode,
    ToolCall,
    ToolCallDiagnostic,
    ToolCallDiagnosticCode,
    ToolCallDiagnosticStage,
    ToolCallResult,
)
from avalan.model.capability import ModelCapabilityCatalog
from avalan.model.nlp.text.vendor import openai as openai_module
from avalan.model.nlp.text.vendor.openai import OpenAIClient, OpenAIStream
from avalan.model.reasoning import (
    ReasoningSummaryCapabilityError,
    ReasoningSummaryRequestCapability,
)
from avalan.model.stream import (
    REASONING_SEGMENT_BOUNDARY_METADATA_KEY,
    CanonicalStreamItem,
    StreamItemKind,
    StreamPerformanceBudget,
    StreamProviderAdapterError,
    StreamProviderEvent,
    StreamReasoningRepresentation,
    StreamRetentionPolicy,
    StreamTerminalOutcome,
    StreamVisibility,
    TextGenerationNonStreamResult,
    accumulate_canonical_stream_items,
    project_canonical_stream_item,
    stream_observability_payload,
)
from avalan.task.usage import usage_totals_from_response

_PRIVATE_ENCRYPTED_SENTINEL = "encrypted-private-sentinel"
_PRIVATE_SUMMARY_SENTINEL = "summary-private-sentinel"
_RESPONSE_STREAM_EVENT = TypeAdapter(ResponseStreamEvent)
_PHASE4_NEGATIVE_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "reasoning_summary"
    / "phase4_negative_traces.json"
)
_REASONING_SUMMARY_TRACE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "reasoning_summary"
    / "provider_traces"
)


class _AsyncEvents:
    def __init__(self, events: list[object]) -> None:
        self._events = iter(events)
        self.read_count = 0
        self.close_count = 0

    def __aiter__(self) -> "_AsyncEvents":
        return self

    async def __anext__(self) -> object:
        self.read_count += 1
        try:
            return next(self._events)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def aclose(self) -> None:
        self.close_count += 1


class _FailingAsyncEvents(_AsyncEvents):
    def __init__(
        self,
        events: list[object],
        error: BaseException,
    ) -> None:
        super().__init__(events)
        self._error = error

    async def __anext__(self) -> object:
        try:
            return next(self._events)
        except StopIteration:
            raise self._error from None


class _UnsupportedOpenAIClient(OpenAIClient):
    reasoning_summary_request_capability = ReasoningSummaryRequestCapability()


class _InjectedSummaryStream(OpenAIStream):
    def _provider_events_from_event(
        self,
        event: object,
    ) -> tuple[StreamProviderEvent, ...]:
        if OpenAIClient._response_field(event, "type") == "test.summary.delta":
            return (
                StreamProviderEvent(
                    kind=StreamItemKind.REASONING_DELTA,
                    text_delta="summary",
                    visibility=StreamVisibility.PRIVATE,
                    reasoning_representation=(
                        StreamReasoningRepresentation.SUMMARY
                    ),
                    segment_instance_ordinal=0,
                ),
            )
        return super()._provider_events_from_event(event)


class _CompoundFailureStream(OpenAIStream):
    def _provider_events_from_event(
        self,
        event: object,
    ) -> tuple[StreamProviderEvent, ...]:
        if OpenAIClient._response_field(event, "type") == "response.failed":
            return (
                StreamProviderEvent(
                    kind=StreamItemKind.REASONING_DELTA,
                    text_delta="summary",
                    visibility=StreamVisibility.PRIVATE,
                    reasoning_representation=(
                        StreamReasoningRepresentation.SUMMARY
                    ),
                    segment_instance_ordinal=0,
                ),
                StreamProviderEvent(
                    kind=StreamItemKind.STREAM_ERRORED,
                    data={"error": {"code": "response_failed"}},
                ),
            )
        return super()._provider_events_from_event(event)


class _CoercionTrap:
    calls = 0

    def __str__(self) -> str:
        type(self).calls += 1
        return "must-not-coerce"


def _client(
    create: AsyncMock,
    *,
    capable: bool = True,
    is_azure: bool = False,
    policy: StreamRetentionPolicy | None = None,
) -> OpenAIClient:
    client_type = OpenAIClient if capable else _UnsupportedOpenAIClient
    client = object.__new__(client_type)
    cast(Any, client)._client = SimpleNamespace(
        responses=SimpleNamespace(create=create)
    )
    cast(Any, client)._extra_query = None
    cast(Any, client)._is_azure = is_azure
    cast(Any, client)._stream_response_failed_retries = 2
    cast(Any, client)._stream_response_failed_retry_delay_seconds = 0.0
    cast(Any, client)._stream_retention_policy = (
        policy or StreamRetentionPolicy()
    )
    cast(Any, client)._replay_owners_by_call_id = {}
    cast(Any, client)._active_replay_owners = {}
    cast(Any, client)._active_replay_streams = {}
    cast(Any, client)._active_replay_call_ids = {}
    cast(Any, client)._ambiguous_replay_call_ids = {}
    cast(Any, client)._replay_association_poisoned = False
    cast(Any, client)._closed = False
    return client


def _settings(
    summary: ReasoningSummaryMode | None = ReasoningSummaryMode.AUTO,
    *,
    effort: ReasoningEffort | None = None,
    enabled: bool = True,
) -> GenerationSettings:
    return GenerationSettings(
        reasoning=ReasoningSettings(
            effort=effort,
            summary=summary,
            enabled=enabled,
        )
    )


def _completed_event(*, text: str | None = None) -> object:
    events: dict[str, object] = {
        "type": "response.completed",
        "response": {"usage": {}},
    }
    if text is not None:
        events["response"] = {"usage": {}, "output_text": text}
    return events


def _non_stream_response() -> object:
    return SimpleNamespace(output=[], usage=None)


def _rich_non_stream_response(*, include_tool: bool = True) -> object:
    output: list[object] = [
        SimpleNamespace(
            id="rs_non_stream",
            type="reasoning",
            status="completed",
            encrypted_content=_PRIVATE_ENCRYPTED_SENTINEL,
            summary=[
                SimpleNamespace(type="summary_text", text="first"),
                SimpleNamespace(type="summary_text", text="second"),
            ],
        )
    ]
    if include_tool:
        output.append(
            SimpleNamespace(
                id="fc_non_stream",
                type="function_call",
                status="completed",
                call_id="call_non_stream",
                name="lookup",
                arguments='{"id":1}',
            )
        )
    output.append(
        SimpleNamespace(
            id="msg_non_stream",
            type="message",
            status="completed",
            content=[SimpleNamespace(type="output_text", text='{"ok":true}')],
        )
    )
    return SimpleNamespace(
        id="resp_non_stream",
        status="completed",
        output=output,
        usage=SimpleNamespace(
            input_tokens=4,
            output_tokens=6,
            output_tokens_details=SimpleNamespace(reasoning_tokens=3),
            total_tokens=10,
        ),
    )


def _reasoning_item(
    identifier: str,
    encrypted: object = "cipher",
    summary: object = None,
) -> dict[str, object]:
    item: dict[str, object] = {
        "type": "reasoning",
        "id": identifier,
        "encrypted_content": encrypted,
        "summary": [] if summary is None else summary,
        "status": "completed",
    }
    return item


def _function_item(call_id: str, *, suffix: str = "") -> dict[str, object]:
    return {
        "type": "function_call",
        "id": f"fc_{call_id}{suffix}",
        "call_id": call_id,
        "name": "lookup",
        "arguments": '{"id":1}',
        "status": "completed",
    }


def _output_done(item: dict[str, object]) -> dict[str, object]:
    return {"type": "response.output_item.done", "item": item}


def _message_item_event(
    event_type: str,
    item_id: object,
    output_index: object,
    text: object = "message",
) -> dict[str, object]:
    return {
        "type": event_type,
        "output_index": output_index,
        "item": {
            "id": item_id,
            "type": "message",
            "status": (
                "in_progress"
                if event_type == "response.output_item.added"
                else "completed"
            ),
            "content": [{"type": "output_text", "text": text}],
        },
    }


def _tool_item_done(
    item_id: object,
    output_index: object,
    arguments: object = '{"value":1}',
) -> dict[str, object]:
    return {
        "type": "response.output_item.done",
        "output_index": output_index,
        "item": {
            "id": item_id,
            "type": "function_call",
            "status": "completed",
            "call_id": "call-safe",
            "name": "safe_tool",
            "arguments": arguments,
        },
    }


def _typed_item_added(
    item_type: object,
    item_id: object,
    output_index: object,
) -> dict[str, object]:
    return {
        "type": "response.output_item.added",
        "output_index": output_index,
        "item": {
            "id": item_id,
            "type": item_type,
            "status": "in_progress",
            "call_id": "call-safe",
            "name": "safe_tool",
            "arguments": "",
            "content": [],
        },
    }


def _trace_events(
    name: str,
    *,
    response_index: int = 0,
    attempt_index: int | None = None,
) -> list[object]:
    payload = loads(
        (_REASONING_SUMMARY_TRACE_ROOT / f"{name}.json").read_text()
    )
    assert isinstance(payload, dict)
    responses = payload["responses"]
    assert isinstance(responses, list)
    response = responses[response_index]
    assert isinstance(response, dict)
    if attempt_index is None:
        events = response["events"]
        assert isinstance(events, list)
        return deepcopy(events)
    attempts = response["attempts"]
    assert isinstance(attempts, list)
    attempt = attempts[attempt_index]
    assert isinstance(attempt, list)
    return deepcopy(attempt)


def _summary_item_added(
    item_id: str = "rs_test",
    output_index: object = 0,
) -> dict[str, object]:
    return {
        "type": "response.output_item.added",
        "output_index": output_index,
        "item": {
            "id": item_id,
            "type": "reasoning",
            "status": "in_progress",
            "summary": [],
        },
    }


def _summary_part_added(
    item_id: str = "rs_test",
    output_index: object = 0,
    summary_index: object = 0,
) -> dict[str, object]:
    return {
        "type": "response.reasoning_summary_part.added",
        "item_id": item_id,
        "output_index": output_index,
        "summary_index": summary_index,
        "part": {"type": "summary_text", "text": ""},
    }


def _summary_delta(
    text: object,
    item_id: str = "rs_test",
    output_index: object = 0,
    summary_index: object = 0,
) -> dict[str, object]:
    return {
        "type": "response.reasoning_summary_text.delta",
        "item_id": item_id,
        "output_index": output_index,
        "summary_index": summary_index,
        "delta": text,
    }


def _summary_text_done(
    text: object,
    item_id: str = "rs_test",
    output_index: object = 0,
    summary_index: object = 0,
) -> dict[str, object]:
    return {
        "type": "response.reasoning_summary_text.done",
        "item_id": item_id,
        "output_index": output_index,
        "summary_index": summary_index,
        "text": text,
    }


def _summary_part_done(
    text: object,
    item_id: str = "rs_test",
    output_index: object = 0,
    summary_index: object = 0,
) -> dict[str, object]:
    return {
        "type": "response.reasoning_summary_part.done",
        "item_id": item_id,
        "output_index": output_index,
        "summary_index": summary_index,
        "part": {"type": "summary_text", "text": text},
    }


def _summary_item_done(
    texts: list[object],
    item_id: str = "rs_test",
    output_index: object = 0,
    *,
    content: object | None = None,
) -> dict[str, object]:
    item: dict[str, object] = {
        "id": item_id,
        "type": "reasoning",
        "status": "completed",
        "summary": [{"type": "summary_text", "text": text} for text in texts],
        "encrypted_content": f"cipher-{item_id}",
    }
    if content is not None:
        item["content"] = content
    return {
        "type": "response.output_item.done",
        "output_index": output_index,
        "item": item,
    }


def _summary_event_cases(
    secret: str,
    item_id: str,
) -> tuple[tuple[str, dict[str, object]], ...]:
    part_added = _summary_part_added(item_id)
    cast(dict[str, object], part_added["part"])["text"] = secret
    return (
        ("response.reasoning_summary_part.added", part_added),
        (
            "response.reasoning_summary_text.delta",
            _summary_delta(secret, item_id),
        ),
        (
            "response.reasoning_summary_text.done",
            _summary_text_done(secret, item_id),
        ),
        (
            "response.reasoning_summary_part.done",
            _summary_part_done(secret, item_id),
        ),
    )


def _summary_trace(
    *parts: tuple[str, ...],
    item_id: str = "rs_test",
    output_index: int = 0,
) -> list[object]:
    events: list[object] = [_summary_item_added(item_id, output_index)]
    completed: list[object] = []
    for summary_index, deltas in enumerate(parts):
        events.append(
            _summary_part_added(item_id, output_index, summary_index)
        )
        for delta in deltas:
            events.append(
                _summary_delta(delta, item_id, output_index, summary_index)
            )
        text = "".join(deltas)
        events.extend(
            (
                _summary_text_done(text, item_id, output_index, summary_index),
                _summary_part_done(text, item_id, output_index, summary_index),
            )
        )
        completed.append(text)
    events.extend(
        (
            _summary_item_done(completed, item_id, output_index),
            _completed_event(),
        )
    )
    return events


def _reasoning_items(
    items: list[CanonicalStreamItem],
) -> list[CanonicalStreamItem]:
    return [
        item for item in items if item.kind is StreamItemKind.REASONING_DELTA
    ]


def _error_item(items: list[CanonicalStreamItem]) -> CanonicalStreamItem:
    errors = [
        item for item in items if item.kind is StreamItemKind.STREAM_ERRORED
    ]
    assert len(errors) == 1
    return errors[0]


def _phase4_negative_fixture() -> dict[str, Any]:
    payload = loads(_PHASE4_NEGATIVE_FIXTURE.read_text())
    assert isinstance(payload, dict)
    assert payload["schema_version"] == 1
    assert payload["error_code"] == "invalid_reasoning_summary_event"
    return cast(dict[str, Any], payload)


def _fixture_runtime_value(row: Mapping[str, object]) -> object:
    if row["shape"] == "object":
        return _CoercionTrap()
    return deepcopy(row["value"])


def _assert_structured_summary_error(
    items: list[CanonicalStreamItem],
    *,
    event_type: str,
    field: str,
    value_shape: str,
    output_index: int | None = 0,
    summary_index: int | None = 0,
) -> CanonicalStreamItem:
    terminal = _error_item(items)
    assert terminal.terminal_outcome is StreamTerminalOutcome.ERRORED
    assert terminal.provider_payload is None
    assert terminal.provider_event_type == event_type
    assert terminal.data == {
        "error": {
            "type": "invalid_provider_event",
            "code": "invalid_reasoning_summary_event",
            "message": "OpenAI reasoning summary event is invalid.",
            "event_type": event_type,
            "field": field,
            "index": {
                "output_index": output_index,
                "summary_index": summary_index,
            },
            "value_shape": value_shape,
        }
    }
    assert sum(item.is_stream_terminal for item in items) == 1
    return terminal


def _assert_preparse_private_error(
    items: list[CanonicalStreamItem],
    *,
    event_type: str,
) -> CanonicalStreamItem:
    terminal = _error_item(items)
    data = cast(dict[str, Any], terminal.data)
    error = cast(dict[str, Any], data["error"])
    assert terminal.terminal_outcome is StreamTerminalOutcome.ERRORED
    assert terminal.provider_payload is None
    assert terminal.provider_event_type == event_type
    assert error["type"] == "invalid_provider_event"
    assert error["code"] == "invalid_reasoning_summary_event"
    assert error["event_type"] == event_type
    assert error["field"] == "provider_payload"
    assert error["index"] == {
        "output_index": None,
        "summary_index": None,
    }
    assert error["value_shape"] == "unreadable"
    assert sum(item.is_stream_terminal for item in items) == 1
    return terminal


def _assert_output_item_conflict(
    events: list[object],
    *,
    field: str,
    output_index: int,
    secret: str,
) -> list[CanonicalStreamItem]:
    items = run(_consume(OpenAIStream(_AsyncEvents(events))))
    terminal = _assert_structured_summary_error(
        items,
        event_type="response.output_item.done",
        field=field,
        value_shape="conflict",
        output_index=output_index,
        summary_index=None,
    )
    assert not any(
        item.kind
        in {
            StreamItemKind.ANSWER_DELTA,
            StreamItemKind.ANSWER_DONE,
            StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
            StreamItemKind.TOOL_CALL_READY,
            StreamItemKind.TOOL_CALL_DONE,
        }
        for item in items
    )
    outward = repr([terminal, terminal.to_trace_dict()])
    assert secret not in outward
    return items


def _index_case_events(
    event_type: str,
) -> tuple[list[object], dict[str, object]]:
    item_added = _summary_item_added("rs_bad", 0)
    part_added = _summary_part_added("rs_bad", 0, 0)
    delta = _summary_delta("", "rs_bad", 0, 0)
    text_done = _summary_text_done("", "rs_bad", 0, 0)
    part_done = _summary_part_done("", "rs_bad", 0, 0)
    item_done = _summary_item_done([""], "rs_bad", 0)
    cases: dict[str, tuple[list[object], dict[str, object]]] = {
        "response.output_item.added": ([], item_added),
        "response.reasoning_summary_part.added": (
            [item_added],
            part_added,
        ),
        "response.reasoning_summary_text.delta": (
            [item_added, part_added],
            delta,
        ),
        "response.reasoning_summary_text.done": (
            [item_added, part_added, delta],
            text_done,
        ),
        "response.reasoning_summary_part.done": (
            [item_added, part_added, delta, text_done],
            part_done,
        ),
        "response.output_item.done": (
            [item_added, part_added, delta, text_done, part_done],
            item_done,
        ),
    }
    prefix, event = cases[event_type]
    return deepcopy(prefix), deepcopy(event)


def _tool_result(call_id: str, value: str) -> Message:
    call = ToolCall(id=call_id, name="lookup", arguments={"id": 1})
    result = ToolCallResult(
        id=f"result_{call_id}",
        name="lookup",
        arguments=call.arguments,
        call=call,
        result=value,
    )
    return Message(role=MessageRole.TOOL, tool_call_result=result)


async def _consume(stream: object) -> list[CanonicalStreamItem]:
    return [
        item
        async for item in cast(
            OpenAIStream,
            stream,
        )
    ]


async def _consume_provider_events(
    stream: OpenAIStream,
) -> list[StreamProviderEvent]:
    return [event async for event in stream._provider_events()]


async def _consume_with_error(
    stream: OpenAIStream,
) -> tuple[list[CanonicalStreamItem], BaseException | None]:
    consumed: list[CanonicalStreamItem] = []
    iterator = stream.__aiter__()
    while True:
        try:
            consumed.append(await iterator.__anext__())
        except StopAsyncIteration:
            return consumed, None
        except BaseException as error:
            return consumed, error


def _owner(
    policy: StreamRetentionPolicy | None = None,
) -> Any:
    owner = openai_module._OpenAIReplayOwner(policy or StreamRetentionPolicy())
    owner.begin_attempt()
    return owner


class _CountingReplayOwner(openai_module._OpenAIReplayOwner):
    def __init__(self) -> None:
        super().__init__(StreamRetentionPolicy())
        self.admit_calls = 0

    def admit(self, item: dict[str, Any]) -> bool:
        self.admit_calls += 1
        return super().admit(item)


def _retained_private_client(
    create: AsyncMock,
    call_id: str,
) -> OpenAIClient:
    owner = _owner()
    owner.admit(
        _reasoning_item(
            f"reasoning-{call_id}",
            _PRIVATE_ENCRYPTED_SENTINEL,
            [
                {
                    "type": "summary_text",
                    "text": _PRIVATE_SUMMARY_SENTINEL,
                }
            ],
        )
    )
    owner.admit(_function_item(call_id))
    owner.commit_attempt()
    client = _client(create)
    cast(Any, client)._retain_replay_owner(owner, (call_id,))
    return client


def _summary_accounting(summary: object) -> tuple[int, int, int]:
    normalized = openai_module._strict_replay_json_copy({"summary": summary})
    nodes, characters = openai_module._replay_json_accounting(normalized)
    serialized_bytes = openai_module._replay_json_serialized_bytes(normalized)
    return nodes, characters, serialized_bytes


def _capability_mock(count: int) -> MagicMock | None:
    if count == 0:
        return None
    capability = MagicMock(spec=ModelCapabilityCatalog)
    capability.project.return_value.schemas = tuple(
        {
            "type": "function",
            "function": {"name": f"tool_{index}"},
        }
        for index in range(count)
    )
    return capability


def _awaited_kwargs(create: AsyncMock) -> dict[str, Any]:
    call = create.await_args
    assert call is not None
    return dict(call.kwargs)


def _safe_exception_diagnostics(
    error: BaseException,
) -> tuple[list[BaseException], str]:
    nodes: list[BaseException] = []
    work = [error]
    while work:
        node = work.pop()
        nodes.append(node)
        assert node.__cause__ is None
        assert node.__context__ is None
        if isinstance(node, BaseExceptionGroup):
            work.extend(node.exceptions)
    diagnostics = "\n".join(
        [
            *(repr(node) for node in nodes),
            *(str(node) for node in nodes),
            *("".join(format_exception(node)) for node in nodes),
        ]
    )
    return nodes, diagnostics


def test_reasoning_request_shapes_are_exact_and_omission_safe() -> None:
    cases = (
        (None, None),
        (GenerationSettings(), None),
        (_settings(None, effort=ReasoningEffort.HIGH), {"effort": "high"}),
        (_settings(ReasoningSummaryMode.AUTO), {"summary": "auto"}),
        (_settings(ReasoningSummaryMode.CONCISE), {"summary": "concise"}),
        (_settings(ReasoningSummaryMode.DETAILED), {"summary": "detailed"}),
        (
            _settings(
                ReasoningSummaryMode.AUTO,
                effort=ReasoningEffort.HIGH,
            ),
            {"effort": "high", "summary": "auto"},
        ),
    )
    for settings, expected in cases:
        create = AsyncMock(return_value=_non_stream_response())
        client = _client(create)
        run(
            client(
                "plain-model",
                [Message(role=MessageRole.USER, content="hello")],
                settings,
                use_async_generator=False,
            )
        )
        request = _awaited_kwargs(create)
        if expected is None:
            assert "reasoning" not in request
            assert "include" not in request
        else:
            assert request["reasoning"] == expected
            assert request["include"] == ["reasoning.encrypted_content"]


def test_summary_only_request_is_forwarded() -> None:
    create = AsyncMock(return_value=_non_stream_response())
    run(
        _client(create)(
            "plain-model",
            [Message(role=MessageRole.USER, content="hello")],
            _settings(ReasoningSummaryMode.CONCISE),
            use_async_generator=False,
        )
    )

    assert _awaited_kwargs(create)["reasoning"] == {"summary": "concise"}


def test_reasoning_request_streaming_and_non_streaming_are_identical() -> None:
    requests: list[dict[str, object]] = []
    for streaming in (False, True):
        response = (
            _AsyncEvents([_completed_event()])
            if streaming
            else _non_stream_response()
        )
        create = AsyncMock(return_value=response)
        result = run(
            _client(create)(
                "plain-model",
                [Message(role=MessageRole.USER, content="hello")],
                _settings(
                    ReasoningSummaryMode.DETAILED,
                    effort=ReasoningEffort.HIGH,
                ),
                use_async_generator=streaming,
            )
        )
        if streaming:
            run(_consume(result))
        request = deepcopy(_awaited_kwargs(create))
        del request["stream"]
        requests.append(request)

    assert requests[0] == requests[1]


def test_summary_request_tool_matrix_keeps_include_deduplicated() -> None:
    for tool_count in (0, 1, 2):
        create = AsyncMock(return_value=_non_stream_response())
        run(
            _client(create)(
                "gpt-5",
                [Message(role=MessageRole.USER, content="hello")],
                _settings(),
                capability=cast(Any, _capability_mock(tool_count)),
                use_async_generator=False,
            )
        )
        request = _awaited_kwargs(create)
        assert request["reasoning"] == {"summary": "auto"}
        assert request["include"] == ["reasoning.encrypted_content"]
        assert request["include"].count("reasoning.encrypted_content") == 1
        assert ("tools" in request) is (tool_count > 0)


def test_effort_none_max_and_disabled_request_semantics() -> None:
    cases = (
        (
            _settings(
                ReasoningSummaryMode.CONCISE,
                effort=ReasoningEffort.NONE,
            ),
            {"effort": "none", "summary": "concise"},
        ),
        (
            _settings(
                ReasoningSummaryMode.DETAILED,
                effort=ReasoningEffort.MAX,
            ),
            {"effort": "xhigh", "summary": "detailed"},
        ),
        (
            _settings(None, effort=ReasoningEffort.HIGH, enabled=False),
            {"effort": "high"},
        ),
        (
            _settings(None, effort=ReasoningEffort.NONE, enabled=False),
            None,
        ),
    )
    for settings, expected in cases:
        create = AsyncMock(return_value=_non_stream_response())
        run(
            _client(create)(
                "plain-model",
                [],
                settings,
                use_async_generator=False,
            )
        )
        assert _awaited_kwargs(create).get("reasoning") == expected
    with pytest.raises(AssertionError, match="cannot be requested"):
        _settings(ReasoningSummaryMode.AUTO, enabled=False)


def test_invalid_summary_fails_before_responses_create() -> None:
    settings = GenerationSettings()
    object.__setattr__(settings.reasoning, "summary", "auto")
    create = AsyncMock()

    with pytest.raises(AssertionError):
        run(_client(create)("plain-model", [], settings))

    create.assert_not_awaited()


def test_declared_unsupported_summary_fails_before_provider_call() -> None:
    for is_azure, provider in ((False, "openai"), (True, "azure_openai")):
        create = AsyncMock()
        with pytest.raises(ReasoningSummaryCapabilityError) as error:
            run(
                _client(create, capable=False, is_azure=is_azure)(
                    "summary-looking-model",
                    [],
                    _settings(ReasoningSummaryMode.DETAILED),
                )
            )
        assert error.value.provider == provider
        assert error.value.requested_mode is ReasoningSummaryMode.DETAILED
        create.assert_not_awaited()


def test_upstream_summary_rejection_is_actionable_and_non_retryable() -> None:
    rejection = RuntimeError("model rejects reasoning summary mode 'concise'")
    create = AsyncMock(side_effect=rejection)

    with pytest.raises(RuntimeError) as error:
        run(
            _client(create)(
                "plain-model",
                [],
                _settings(ReasoningSummaryMode.CONCISE),
            )
        )

    assert error.value is rejection
    assert create.await_count == 1


def test_rejected_summary_is_not_retried_without_summary() -> None:
    failed = _AsyncEvents(
        [
            {
                "type": "response.failed",
                "response": {
                    "status": "failed",
                    "error": {
                        "code": "unsupported_reasoning_summary",
                        "message": "summary is unavailable for this model",
                    },
                    "output": [],
                },
            }
        ]
    )
    create = AsyncMock(return_value=failed)
    stream = run(
        _client(create)(
            "plain-model",
            [],
            _settings(ReasoningSummaryMode.DETAILED),
        )
    )
    items = run(_consume(stream))

    assert create.await_count == 1
    assert _awaited_kwargs(create)["reasoning"] == {"summary": "detailed"}
    assert any(item.kind is StreamItemKind.STREAM_ERRORED for item in items)

    mixed_create = AsyncMock(
        side_effect=[
            _AsyncEvents(
                [
                    {
                        "type": "response.failed",
                        "error": {"code": "response_failed"},
                        "response": {
                            "status": "failed",
                            "error": {"code": "unsupported_reasoning_summary"},
                            "output": [],
                        },
                    }
                ]
            ),
            _AsyncEvents([_completed_event()]),
        ]
    )
    mixed_items = run(
        _consume(
            run(
                _client(mixed_create)(
                    "plain-model",
                    [],
                    _settings(ReasoningSummaryMode.DETAILED),
                )
            )
        )
    )
    assert mixed_create.await_count == 1
    assert any(
        item.kind is StreamItemKind.STREAM_ERRORED for item in mixed_items
    )


def test_summary_preserves_encrypted_replay() -> None:
    first = _AsyncEvents(
        [
            _output_done(
                _reasoning_item(
                    "rs_1",
                    "encrypted-sentinel",
                    [{"type": "summary_text", "text": "summary-sentinel"}],
                )
            ),
            _output_done(_function_item("call_1")),
            _completed_event(),
        ]
    )
    second = _AsyncEvents([_completed_event(text="done")])
    create = AsyncMock(side_effect=[first, second])
    client = _client(create)
    first_stream = run(client("plain-model", [], _settings()))
    run(_consume(first_stream))
    second_stream = run(
        client(
            "plain-model",
            [_tool_result("call_1", "result")],
            _settings(),
        )
    )
    run(_consume(second_stream))

    replay_input = create.await_args_list[1].kwargs["input"]
    assert replay_input[0]["id"] == "rs_1"
    assert replay_input[0]["encrypted_content"] == "encrypted-sentinel"
    assert replay_input[0]["summary"][0]["text"] == "summary-sentinel"


def test_encrypted_replay_survives_tool_cycles() -> None:
    streams = [
        _AsyncEvents(
            [
                _output_done(
                    _reasoning_item(
                        "rs_1",
                        "cipher-1",
                        [{"type": "summary_text", "text": "first"}],
                    )
                ),
                _output_done(_function_item("call_1")),
                _completed_event(),
            ]
        ),
        _AsyncEvents(
            [
                _output_done(
                    _reasoning_item(
                        "rs_2",
                        "cipher-2",
                        [{"type": "summary_text", "text": "second"}],
                    )
                ),
                _output_done(_function_item("call_2")),
                _completed_event(),
            ]
        ),
        _AsyncEvents([_completed_event(text="answer")]),
    ]
    create = AsyncMock(side_effect=streams)
    client = _client(create)
    user = Message(role=MessageRole.USER, content="start")
    first = run(client("plain-model", [user], _settings()))
    run(_consume(first))
    result_1 = _tool_result("call_1", "one")
    second = run(client("plain-model", [user, result_1], _settings()))
    run(_consume(second))
    result_2 = _tool_result("call_2", "two")
    third = run(client("plain-model", [user, result_1, result_2], _settings()))
    final_items = run(_consume(third))

    second_input = create.await_args_list[1].kwargs["input"]
    third_input = create.await_args_list[2].kwargs["input"]
    assert [
        item.get("id")
        for item in second_input
        if isinstance(item, dict) and item.get("type") == "reasoning"
    ] == ["rs_1"]
    assert [
        item.get("id")
        for item in third_input
        if isinstance(item, dict) and item.get("type") == "reasoning"
    ] == ["rs_1", "rs_2"]
    assert [
        item.get("call_id")
        for item in third_input
        if isinstance(item, dict) and item.get("type") == "function_call"
    ] == ["call_1", "call_2"]
    assert (
        accumulate_canonical_stream_items(final_items).answer_text == "answer"
    )
    assert cast(Any, client)._replay_owners_by_call_id == {}


def test_replay_requires_encrypted_content_and_preserves_provider_fields() -> (
    None
):
    owner = _owner()
    assert not owner.admit(_reasoning_item("missing", None))
    assert not owner.admit(_reasoning_item("empty", ""))
    replayable = _reasoning_item(
        "rs_kept",
        "cipher-kept",
        [
            {
                "type": "summary_text",
                "text": "kept",
                "metadata": None,
            }
        ],
    )
    assert owner.admit(replayable)

    items = owner.replay_items()
    assert len(items) == 1
    assert items[0]["id"] == "rs_kept"
    assert items[0]["encrypted_content"] == "cipher-kept"
    assert items[0]["summary"] == replayable["summary"]
    assert "status" not in items[0]
    assert owner.counters[0] == 1

    repr_encrypted = "REPR_ENCRYPTED_CONTENT_SENTINEL"
    repr_summary = "REPR_SUMMARY_SENTINEL"
    repr_owner = _owner()
    assert repr_owner.admit(
        _reasoning_item(
            "repr-private",
            repr_encrypted,
            [{"type": "summary_text", "text": repr_summary}],
        )
    )
    owner_repr = repr(repr_owner)
    assert (
        owner_repr
        == "_OpenAIReplayOwner(item_count=1, reasoning_item_count=1, "
        "released=False)"
    )
    assert repr_encrypted not in owner_repr
    assert repr_summary not in owner_repr
    repr_owner.release()

    inactive_owner = openai_module._OpenAIReplayOwner(StreamRetentionPolicy())
    inactive_owner.commit_attempt()
    with pytest.raises(RuntimeError, match="no active attempt"):
        inactive_owner.admit(_reasoning_item("inactive"))
    inactive_owner.begin_attempt()
    assert not inactive_owner.admit({"type": "message", "content": "safe"})
    inactive_owner.release()
    assert inactive_owner.replay_items() == ()
    with pytest.raises(RuntimeError, match="released"):
        inactive_owner.begin_attempt()


def test_replay_retention_policy_has_dedicated_exact_limits() -> None:
    policy = StreamRetentionPolicy()
    assert policy.openai_replay_item_limit == 4096
    assert policy.openai_replay_serialized_byte_limit == 4194304
    assert policy.openai_replay_reasoning_item_limit == 1024
    assert policy.openai_replay_reasoning_summary_node_limit == 4096
    assert policy.openai_replay_reasoning_summary_character_limit == 262144
    assert (
        policy.openai_replay_reasoning_summary_serialized_byte_limit == 1048576
    )
    field_names = (
        "openai_replay_item_limit",
        "openai_replay_serialized_byte_limit",
        "openai_replay_reasoning_item_limit",
        "openai_replay_reasoning_summary_node_limit",
        "openai_replay_reasoning_summary_character_limit",
        "openai_replay_reasoning_summary_serialized_byte_limit",
    )
    for field_name in field_names:
        for invalid in (-1, True, 1.0, "1"):
            with pytest.raises(AssertionError):
                StreamRetentionPolicy(**cast(Any, {field_name: invalid}))
        assert (
            getattr(
                StreamRetentionPolicy(**cast(Any, {field_name: 0})),
                field_name,
            )
            == 0
        )


def test_replay_item_limit_is_prospective_at_limit_boundaries() -> None:
    policy = StreamRetentionPolicy(
        openai_replay_item_limit=3,
        openai_replay_reasoning_item_limit=2,
    )
    owner = _owner(policy)
    assert owner.admit(_reasoning_item("rs_1"))
    assert owner.admit(_function_item("call_1"))
    assert owner.admit(_reasoning_item("rs_2"))
    assert owner.counters[0] == 2
    assert owner.generic_counters[0] == 3
    before = (owner.replay_items(), owner.counters, owner.generic_counters)
    with pytest.raises(openai_module._ReasoningReplayRetentionError) as error:
        owner.admit(_function_item("call_2"))
    assert error.value.code == "reasoning_replay_retention_exceeded"
    assert (
        owner.replay_items(),
        owner.counters,
        owner.generic_counters,
    ) == before

    function_owner = _owner(
        StreamRetentionPolicy(
            replay_history_item_limit=0,
            openai_replay_item_limit=3,
            openai_replay_reasoning_item_limit=0,
            openai_replay_reasoning_summary_node_limit=0,
            openai_replay_reasoning_summary_character_limit=0,
            openai_replay_reasoning_summary_serialized_byte_limit=0,
        )
    )
    assert function_owner.admit(_function_item("call_1"))
    assert function_owner.admit(_function_item("call_2"))
    assert function_owner.admit(_function_item("call_3"))
    assert function_owner.item_count == 3
    assert function_owner.counters == (0, 0, 0, 0)
    assert function_owner.generic_counters[0] == 3


@pytest.mark.parametrize(
    "item",
    (
        _reasoning_item("generic-byte-reasoning", "á🙂", []),
        {
            **_function_item("generic-byte-function"),
            "arguments": '{"value":"á🙂"}',
        },
    ),
)
def test_replay_serialized_byte_limit_counts_complete_normalized_items(
    item: dict[str, object],
) -> None:
    normalized = OpenAIStream._response_input_item_payload(
        cast(dict[str, Any], item)
    )
    expected = openai_module._replay_json_serialized_bytes(normalized)
    exact = _owner(
        StreamRetentionPolicy(openai_replay_serialized_byte_limit=expected)
    )
    assert exact.admit(item)
    assert exact.generic_counters == (1, expected)

    below = _owner(
        StreamRetentionPolicy(openai_replay_serialized_byte_limit=expected - 1)
    )
    before = (below.replay_items(), below.counters, below.generic_counters)
    with pytest.raises(
        openai_module._ReasoningReplayRetentionError
    ) as context:
        below.admit(item)
    assert (
        below.replay_items(),
        below.counters,
        below.generic_counters,
    ) == before
    _, diagnostics = _safe_exception_diagnostics(context.value)
    assert "á🙂" not in diagnostics


def test_replay_summary_node_limit_counts_nested_empty_and_scalar_entries(
    # Keep this no-argument signature within the project line limit.
) -> None:
    nested: object = "leaf"
    for index in range(1200):
        nested = [nested] if index % 2 else {f"k{index}": nested}
    nodes, _, _ = _summary_accounting(nested)
    assert nodes == 1202
    exact = _owner(
        StreamRetentionPolicy(openai_replay_reasoning_summary_node_limit=nodes)
    )
    assert exact.admit(_reasoning_item("exact", summary=nested))
    assert exact.counters[1] == nodes
    below = _owner(
        StreamRetentionPolicy(
            openai_replay_reasoning_summary_node_limit=nodes - 1
        )
    )
    with pytest.raises(openai_module._ReasoningReplayRetentionError):
        below.admit(_reasoning_item("over", summary=nested))
    assert below.item_count == 0
    assert below.counters == (0, 0, 0, 0)

    metadata: object = "metadata-leaf"
    for index in range(1500):
        metadata = {f"m{index}": metadata}
    function_item = _function_item("deep-metadata")
    function_item["metadata"] = metadata
    metadata_owner = _owner()
    assert metadata_owner.admit(function_item)
    replay_metadata = metadata_owner.replay_items()[0]["metadata"]
    for index in reversed(range(1500)):
        assert isinstance(replay_metadata, dict)
        replay_metadata = replay_metadata[f"m{index}"]
    assert replay_metadata == "metadata-leaf"

    metadata_owner.commit_attempt()

    async def create_continuation(**kwargs: object) -> _AsyncEvents:
        request_input = cast(list[dict[str, Any]], kwargs["input"])
        assert len(request_input) == 2
        request_metadata = request_input[0]["metadata"]
        for index in reversed(range(1500)):
            assert isinstance(request_metadata, dict)
            request_metadata = request_metadata[f"m{index}"]
        assert request_metadata == "metadata-leaf"
        if continuation_create.await_count == 1:
            first_metadata = request_input[0]["metadata"]
            assert isinstance(first_metadata, dict)
            first_metadata["m1499"] = "mutated-first-attempt"
            return _AsyncEvents(
                [
                    {
                        "type": "response.failed",
                        "response": {
                            "status": "failed",
                            "error": None,
                            "output": [],
                        },
                    }
                ]
            )
        return _AsyncEvents([_completed_event()])

    continuation_create = AsyncMock(side_effect=create_continuation)
    continuation_client = _client(continuation_create)
    cast(Any, continuation_client)._retain_replay_owner(
        metadata_owner,
        ("deep-metadata",),
    )
    continuation = run(
        continuation_client(
            "plain-model",
            [_tool_result("deep-metadata", "complete")],
        )
    )
    continuation_items = run(_consume(continuation))
    assert continuation_create.await_count == 2
    assert any(
        item.kind is StreamItemKind.STREAM_COMPLETED
        for item in continuation_items
    )


def test_replay_summary_character_limit_counts_keys_and_values() -> None:
    summary = [{"α-key": "β-value", "empty": {}}]
    _, characters, _ = _summary_accounting(summary)
    assert characters == len("summaryα-keyβ-valueempty")
    exact = _owner(
        StreamRetentionPolicy(
            openai_replay_reasoning_summary_character_limit=characters
        )
    )
    assert exact.admit(_reasoning_item("exact", summary=summary))
    over = _owner(
        StreamRetentionPolicy(
            openai_replay_reasoning_summary_character_limit=characters - 1
        )
    )
    with pytest.raises(openai_module._ReasoningReplayRetentionError):
        over.admit(_reasoning_item("over", summary=summary))


def test_replay_summary_serialized_byte_limit_uses_compact_unicode_json() -> (
    None
):
    summary = [{"type": "summary_text", "text": "á🙂"}]
    normalized = {"summary": summary}
    expected = len(
        dumps(
            normalized,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
    )
    _, characters, serialized_bytes = _summary_accounting(summary)
    assert serialized_bytes == expected
    assert serialized_bytes > characters
    exact = _owner(
        StreamRetentionPolicy(
            openai_replay_reasoning_summary_serialized_byte_limit=expected
        )
    )
    assert exact.admit(_reasoning_item("exact", summary=summary))
    over = _owner(
        StreamRetentionPolicy(
            openai_replay_reasoning_summary_serialized_byte_limit=expected - 1
        )
    )
    with pytest.raises(openai_module._ReasoningReplayRetentionError):
        over.admit(_reasoning_item("over", summary=summary))

    huge_magnitude = 10**5000
    for label, huge_integer, integer_bytes in (
        ("positive", huge_magnitude, 5001),
        ("negative", -huge_magnitude, 5002),
    ):
        expected_huge_bytes = 14 + integer_bytes
        _, _, huge_bytes = _summary_accounting([huge_integer])
        assert huge_bytes == expected_huge_bytes
        exact_huge = _owner(
            StreamRetentionPolicy(
                openai_replay_reasoning_summary_serialized_byte_limit=(
                    expected_huge_bytes
                )
            )
        )
        assert exact_huge.admit(_reasoning_item(label, summary=[huge_integer]))
        below_huge = _owner(
            StreamRetentionPolicy(
                openai_replay_reasoning_summary_serialized_byte_limit=(
                    expected_huge_bytes - 1
                )
            )
        )
        before = (below_huge.replay_items(), below_huge.counters)
        with pytest.raises(
            openai_module._ReasoningReplayRetentionError
        ) as context:
            below_huge.admit(_reasoning_item(label, summary=[huge_integer]))
        assert (below_huge.replay_items(), below_huge.counters) == before
        nodes, diagnostics = _safe_exception_diagnostics(context.value)
        assert [getattr(node, "code", None) for node in nodes] == [
            "reasoning_replay_retention_exceeded"
        ]
        assert "Exceeds the limit" not in diagnostics

    _, _, zero_bytes = _summary_accounting([0])
    assert zero_bytes == len('{"summary":[0]}'.encode())

    correction_magnitude = 1 << 13301
    _, _, correction_bytes = _summary_accounting([correction_magnitude])
    assert correction_bytes == 14 + 4004


def test_replay_rejects_non_json_values_without_coercion() -> None:
    shared = {"type": "summary_text", "text": "shared"}
    alias_owner = _owner()
    assert alias_owner.admit(
        _reasoning_item("alias", summary=[shared, shared])
    )
    alias_summary = alias_owner.replay_items()[0]["summary"]
    assert alias_summary == [shared, shared]

    cycle: list[object] = []
    cycle.append(cycle)
    mapping_cycle: dict[str, object] = {}
    mapping_cycle["self"] = mapping_cycle
    invalid_values = (
        b"bytes",
        bytearray(b"bytes"),
        nan,
        inf,
        -inf,
        {1: "non-string-key"},
        ("tuple",),
        SimpleNamespace(value=1),
        _CoercionTrap(),
        cycle,
        mapping_cycle,
    )
    _CoercionTrap.calls = 0
    for invalid in invalid_values:
        owner = _owner()
        before = (owner.replay_items(), owner.counters)
        with pytest.raises(openai_module._ReasoningReplayRetentionError):
            owner.admit(_reasoning_item("invalid", summary=invalid))
        assert (owner.replay_items(), owner.counters) == before
    assert _CoercionTrap.calls == 0

    surrogate_cases = (
        (
            "SURROGATE_VALUE_SECRET_\ud800",
            {"type": "summary_text", "text": "VALUE_SECRET_\ud800"},
        ),
        (
            "SURROGATE_KEY_SECRET_\udfff",
            {"KEY_SECRET_\udfff": "value"},
        ),
    )
    for sentinel, invalid_summary in surrogate_cases:
        owner = _owner()
        before = (owner.replay_items(), owner.counters)
        with pytest.raises(
            openai_module._ReasoningReplayRetentionError
        ) as context:
            owner.admit(
                _reasoning_item("invalid-surrogate", summary=invalid_summary)
            )
        assert (owner.replay_items(), owner.counters) == before
        _, diagnostics = _safe_exception_diagnostics(context.value)
        assert sentinel not in diagnostics
        assert repr(sentinel) not in diagnostics

    hostile_items = (
        _reasoning_item("opaque-surrogate", "OPAQUE_ENCRYPTED_\ud800", []),
        {
            **_function_item("function-surrogate"),
            "arguments": "FUNCTION_ARGUMENT_\udfff",
        },
    )
    for hostile_item in hostile_items:
        hostile_owner = _owner()
        before = (
            hostile_owner.replay_items(),
            hostile_owner.counters,
            hostile_owner.generic_counters,
        )
        with pytest.raises(
            openai_module._ReasoningReplayRetentionError
        ) as context:
            hostile_owner.admit(hostile_item)
        assert (
            hostile_owner.replay_items(),
            hostile_owner.counters,
            hostile_owner.generic_counters,
        ) == before
        _, diagnostics = _safe_exception_diagnostics(context.value)
        assert "OPAQUE_ENCRYPTED" not in diagnostics
        assert "FUNCTION_ARGUMENT" not in diagnostics
    ordinary_request = {"prompt": "ORDINARY_REQUEST_\ud800"}
    assert openai_module._strict_replay_json_copy(ordinary_request) == (
        ordinary_request
    )


def test_replay_overflow_is_safe_atomic_non_retryable_and_zero_dispatch() -> (
    None
):
    private = "private-overflow-sentinel"
    policy = StreamRetentionPolicy(
        openai_replay_reasoning_summary_character_limit=0
    )
    events = _AsyncEvents(
        [
            _output_done(
                _reasoning_item(
                    "rs_over",
                    "encrypted-overflow-sentinel",
                    private,
                )
            ),
            _completed_event(),
        ]
    )
    create = AsyncMock(return_value=events)
    client = _client(create, policy=policy)
    stream = run(client("plain-model", [], _settings()))
    items = run(_consume(stream))
    terminal = next(
        item for item in items if item.kind is StreamItemKind.STREAM_ERRORED
    )
    serialized = str([item.to_trace_dict() for item in items])

    assert (
        cast(dict[str, Any], terminal.data)["error"]["code"]
        == "reasoning_replay_retention_exceeded"
    )
    assert private not in serialized
    assert "encrypted-overflow-sentinel" not in serialized
    assert terminal.provider_payload is None
    assert create.await_count == 1
    assert cast(Any, client)._replay_owners_by_call_id == {}


def test_function_replay_overflow_is_one_terminal_without_retry() -> None:
    policy = StreamRetentionPolicy(openai_replay_item_limit=0)
    create = AsyncMock(
        return_value=_AsyncEvents(
            [_output_done(_function_item("function-overflow"))]
        )
    )
    client = _client(create, policy=policy)

    stream = run(client("plain-model", [], _settings()))
    items = run(_consume(stream))

    terminal = [
        item
        for item in items
        if item.kind
        in {
            StreamItemKind.STREAM_COMPLETED,
            StreamItemKind.STREAM_ERRORED,
        }
    ]
    assert len(terminal) == 1
    assert terminal[0].kind is StreamItemKind.STREAM_ERRORED
    assert (
        cast(dict[str, Any], terminal[0].data)["error"]["code"]
        == "reasoning_replay_retention_exceeded"
    )
    assert terminal[0].provider_payload is None
    assert create.await_count == 1
    assert cast(Any, client)._replay_owners_by_call_id == {}
    assert cast(Any, client)._active_replay_owners == {}
    assert cast(Any, client)._active_replay_streams == {}


def test_non_stream_function_replay_overflow_matches_streaming() -> None:
    response = SimpleNamespace(
        id="response-function-overflow",
        status="completed",
        output=[
            SimpleNamespace(
                id="fc-function-overflow",
                type="function_call",
                status="completed",
                call_id="function-overflow",
                name="lookup",
                arguments='{"private":"value"}',
            )
        ],
        usage=None,
    )
    create = AsyncMock(return_value=response)
    client = _client(
        create,
        policy=StreamRetentionPolicy(openai_replay_item_limit=0),
    )

    result = run(client("plain-model", [], use_async_generator=False))

    assert isinstance(result, TextGenerationNonStreamResult)
    terminal = [
        event
        for event in result.events
        if event.kind
        in {
            StreamItemKind.STREAM_COMPLETED,
            StreamItemKind.STREAM_ERRORED,
        }
    ]
    assert len(terminal) == 1
    assert terminal[0].kind is StreamItemKind.STREAM_ERRORED
    assert (
        cast(dict[str, Any], terminal[0].data)["error"]["code"]
        == "reasoning_replay_retention_exceeded"
    )
    assert terminal[0].provider_payload is None
    assert create.await_count == 1
    assert cast(Any, client)._replay_owners_by_call_id == {}
    assert cast(Any, client)._active_replay_owners == {}
    assert cast(Any, client)._active_replay_streams == {}


def test_replay_rollback_release_and_request_isolation() -> None:
    owner_a = _owner()
    owner_b = _owner()
    owner_a.admit(_reasoning_item("a-base", "cipher-a", []))
    owner_a.commit_attempt()
    owner_a.begin_attempt()
    owner_a.admit(_reasoning_item("a-failed", "cipher-failed", []))
    owner_a.admit(_function_item("call-failed"))
    owner_b.admit(_reasoning_item("b", "cipher-b", []))
    owner_b_before = (
        owner_b.replay_items(),
        owner_b.counters,
        owner_b.generic_counters,
    )

    owner_a.rollback_attempt()
    owner_a.rollback_attempt()
    assert [item["id"] for item in owner_a.replay_items()] == ["a-base"]
    assert (
        owner_b.replay_items(),
        owner_b.counters,
        owner_b.generic_counters,
    ) == owner_b_before

    client = _client(AsyncMock())
    cast(Any, client)._retain_replay_owner(owner_a, ("call-a",))
    cast(Any, client)._retain_replay_owner(owner_b, ("call-b",))
    selected = cast(Any, client)._replay_owner_for_messages(
        [_tool_result("call-a", "a")]
    )
    assert selected is owner_a
    assert cast(Any, client)._replay_owners_by_call_id == {"call-b": owner_b}
    cast(Any, client)._discard_replay_owner(owner_a)
    cast(Any, client)._discard_replay_owner(owner_a)
    assert owner_a.release_count == 1
    assert owner_a.generic_counters == (0, 0)
    assert not owner_b.released
    owner_b.release()


def test_multiple_retained_replay_owners_become_ambiguous() -> None:
    multi_match_client = _client(AsyncMock())
    multi_match_a = _owner()
    multi_match_b = _owner()
    multi_match_a.commit_attempt()
    multi_match_b.commit_attempt()
    cast(Any, multi_match_client)._retain_replay_owner(
        multi_match_a,
        ("multi-a",),
    )
    cast(Any, multi_match_client)._retain_replay_owner(
        multi_match_b,
        ("multi-b",),
    )
    with pytest.raises(openai_module._ReplayOwnerAssociationError):
        cast(Any, multi_match_client)._replay_owner_for_messages(
            [
                _tool_result("multi-a", "a"),
                _tool_result("multi-b", "b"),
            ]
        )
    assert multi_match_a.released
    assert multi_match_b.released
    assert cast(Any, multi_match_client)._replay_owners_by_call_id == {}
    assert list(cast(Any, multi_match_client)._ambiguous_replay_call_ids) == [
        "multi-a",
        "multi-b",
    ]


def test_retained_owner_activation_respects_capacity() -> None:
    activation_client = _client(
        AsyncMock(),
        policy=StreamRetentionPolicy(replay_history_item_limit=1),
    )
    activation_retained = _owner()
    activation_retained.commit_attempt()
    cast(Any, activation_client)._retain_replay_owner(
        activation_retained,
        ("activation-retained",),
    )
    activation_blocker = _owner()
    cast(Any, activation_client)._activate_replay_owner(activation_blocker)
    with pytest.raises(openai_module._ReasoningReplayRetentionError):
        cast(Any, activation_client)._replay_owner_for_messages(
            [_tool_result("activation-retained", "value")]
        )
    assert activation_retained.released
    assert cast(Any, activation_client)._replay_owners_by_call_id == {}
    cast(Any, activation_client)._discard_replay_owner(activation_blocker)


def test_tool_diagnostic_call_id_extraction_ignores_unanchored() -> None:
    diagnostic_messages = [
        Message(
            role=MessageRole.TOOL,
            tool_call_diagnostic=ToolCallDiagnostic(
                id="anchored-diagnostic",
                call_id="diagnostic-call",
                code=ToolCallDiagnosticCode.UNKNOWN_TOOL,
                stage=ToolCallDiagnosticStage.RESOLVE,
                message="Tool is unknown.",
            ),
        ),
        Message(
            role=MessageRole.TOOL,
            tool_call_diagnostic=ToolCallDiagnostic(
                id="unanchored-diagnostic",
                code=ToolCallDiagnosticCode.MALFORMED_CALL,
                stage=ToolCallDiagnosticStage.PARSE,
                message="Tool call is malformed.",
            ),
        ),
    ]
    assert OpenAIClient._tool_result_call_ids(diagnostic_messages) == (
        "diagnostic-call",
    )


def test_duplicate_retained_and_active_collisions_release_once() -> None:
    duplicate_retained_client = _client(AsyncMock())
    duplicate_retained = _owner()
    duplicate_retained.commit_attempt()
    cast(Any, duplicate_retained_client)._retain_replay_owner(
        duplicate_retained,
        ("retained-x", "retained-y"),
    )
    duplicate_retained_incoming = _owner()
    duplicate_retained_incoming.commit_attempt()
    with pytest.raises(openai_module._ReplayOwnerAssociationError):
        cast(Any, duplicate_retained_client)._retain_replay_owner(
            duplicate_retained_incoming,
            ("retained-x", "retained-y"),
        )
    assert duplicate_retained.release_count == 1
    assert duplicate_retained_incoming.release_count == 1

    duplicate_active_client = _client(AsyncMock())
    duplicate_active = _owner()
    duplicate_active.commit_attempt()
    cast(Any, duplicate_active_client)._retain_replay_owner(
        duplicate_active,
        ("active-x", "active-y"),
    )
    assert (
        cast(Any, duplicate_active_client)._replay_owner_for_messages(
            [_tool_result("active-x", "value")]
        )
        is duplicate_active
    )
    duplicate_active_incoming = _owner()
    duplicate_active_incoming.commit_attempt()
    with pytest.raises(openai_module._ReplayOwnerAssociationError):
        cast(Any, duplicate_active_client)._retain_replay_owner(
            duplicate_active_incoming,
            ("active-x", "active-y"),
        )
    assert duplicate_active.release_count == 1
    assert duplicate_active_incoming.release_count == 1


def test_client_close_deduplicates_multi_call_owner_release() -> None:
    duplicate_close_client = _client(AsyncMock())
    duplicate_close_owner = _owner()
    duplicate_close_owner.commit_attempt()
    cast(Any, duplicate_close_client)._retain_replay_owner(
        duplicate_close_owner,
        ("duplicate-close-a", "duplicate-close-b"),
    )
    sdk_close = MagicMock(return_value=None)
    cast(Any, duplicate_close_client)._client = SimpleNamespace(
        close=sdk_close
    )
    run(duplicate_close_client.aclose())
    assert duplicate_close_owner.release_count == 1
    sdk_close.assert_called_once_with()


def test_retained_call_id_collision_is_persistent_and_zero_dispatch() -> None:
    collision_create = AsyncMock()
    collision_client = _client(collision_create)
    collision_a = _owner()
    collision_b = _owner()
    collision_a.commit_attempt()
    collision_b.commit_attempt()
    cast(Any, collision_client)._retain_replay_owner(
        collision_a,
        ("duplicate-call",),
    )
    with pytest.raises(openai_module._ReplayOwnerAssociationError):
        cast(Any, collision_client)._retain_replay_owner(
            collision_b,
            ("duplicate-call",),
        )
    assert collision_a.released
    assert collision_b.released
    assert (
        "duplicate-call"
        not in cast(Any, collision_client)._replay_owners_by_call_id
    )
    assert (
        "duplicate-call"
        in cast(Any, collision_client)._ambiguous_replay_call_ids
    )
    with pytest.raises(openai_module._ReplayOwnerAssociationError) as error:
        run(
            collision_client(
                "plain-model",
                [_tool_result("duplicate-call", "value")],
                _settings(),
            )
        )
    assert "duplicate-call" not in str(error.value)
    collision_create.assert_not_awaited()
    assert list(cast(Any, collision_client)._ambiguous_replay_call_ids) == [
        "duplicate-call"
    ]
    with pytest.raises(openai_module._ReplayOwnerAssociationError):
        run(
            collision_client(
                "plain-model",
                [_tool_result("duplicate-call", "again")],
                _settings(),
            )
        )
    collision_create.assert_not_awaited()


def test_active_replay_collision_does_not_leak_colliding_owner() -> None:
    async def collide_with_checked_out_owner() -> None:
        secret_a = "SECRET_A_ACTIVE_REPLAY"
        secret_b = "SECRET_B_COLLIDING_REPLAY"
        create_started = Event()
        create_release = Event()
        dispatched_inputs: list[object] = []

        async def delayed_create(**kwargs: object) -> object:
            dispatched_inputs.append(kwargs["input"])
            create_started.set()
            await wait_for(create_release.wait(), timeout=1.0)
            return _non_stream_response()

        concurrent_create = AsyncMock(side_effect=delayed_create)
        concurrent_client = _client(concurrent_create)
        concurrent_a = _owner()
        concurrent_a.admit(
            _reasoning_item("owner-a", secret_a, [{"text": secret_a}])
        )
        concurrent_a.admit(_function_item("dup-active"))
        concurrent_a.commit_attempt()
        cast(Any, concurrent_client)._retain_replay_owner(
            concurrent_a,
            ("dup-active",),
        )
        active_request = create_task(
            concurrent_client(
                "plain-model",
                [_tool_result("dup-active", "a")],
                use_async_generator=False,
            )
        )
        await wait_for(create_started.wait(), timeout=1.0)
        assert cast(Any, concurrent_client)._active_replay_call_ids == {
            "dup-active": concurrent_a
        }

        concurrent_b = _owner()
        concurrent_b.admit(
            _reasoning_item("owner-b", secret_b, [{"text": secret_b}])
        )
        concurrent_b.admit(_function_item("dup-active"))
        concurrent_b.commit_attempt()
        with pytest.raises(openai_module._ReplayOwnerAssociationError):
            cast(Any, concurrent_client)._retain_replay_owner(
                concurrent_b,
                ("dup-active",),
            )
        assert concurrent_a.released
        assert concurrent_b.released
        assert cast(Any, concurrent_client)._active_replay_call_ids == {}
        assert list(
            cast(Any, concurrent_client)._ambiguous_replay_call_ids
        ) == ["dup-active"]
        with pytest.raises(openai_module._ReplayOwnerAssociationError):
            await concurrent_client(
                "plain-model",
                [_tool_result("dup-active", "late")],
                use_async_generator=False,
            )
        create_release.set()
        await wait_for(active_request, timeout=1.0)
        assert concurrent_create.await_count == 1
        serialized_input = repr(dispatched_inputs)
        assert secret_a in serialized_input
        assert secret_b not in serialized_input
        assert cast(Any, concurrent_client)._active_replay_owners == {}
        assert cast(Any, concurrent_client)._replay_owners_by_call_id == {}

    run(collide_with_checked_out_owner())


def test_checked_out_replay_collision_marks_call_id_union() -> None:
    async def collide_with_checked_out_owner() -> None:
        checkout_owner = _owner()
        checkout_owner.admit(
            _reasoning_item(
                "union-a",
                "SECRET_A_UNION",
                [{"text": "SECRET_A_UNION"}],
            )
        )
        checkout_owner.admit(_function_item("checkout-x"))
        checkout_owner.admit(_function_item("checkout-y"))
        checkout_owner.commit_attempt()
        checkout_create = AsyncMock()
        checkout_client = _client(checkout_create)
        cast(Any, checkout_client)._retain_replay_owner(
            checkout_owner,
            ("checkout-x", "checkout-y"),
        )
        selected_checkout = cast(
            Any,
            checkout_client,
        )._replay_owner_for_messages([_tool_result("checkout-x", "first")])
        assert selected_checkout is checkout_owner
        with pytest.raises(openai_module._ReplayOwnerAssociationError):
            cast(Any, checkout_client)._replay_owner_for_messages(
                [_tool_result("checkout-x", "second")]
            )
        assert checkout_owner.released
        assert cast(Any, checkout_client)._active_replay_call_ids == {}
        assert list(cast(Any, checkout_client)._ambiguous_replay_call_ids) == [
            "checkout-x",
            "checkout-y",
        ]

        union_b = _owner()
        union_b.admit(
            _reasoning_item(
                "union-b",
                "SECRET_B_UNION",
                [{"text": "SECRET_B_UNION"}],
            )
        )
        union_b.admit(_function_item("checkout-y"))
        union_b.commit_attempt()
        with pytest.raises(openai_module._ReplayOwnerAssociationError):
            cast(Any, checkout_client)._retain_replay_owner(
                union_b,
                ("checkout-y",),
            )
        assert union_b.released
        with pytest.raises(openai_module._ReplayOwnerAssociationError):
            await checkout_client(
                "plain-model",
                [_tool_result("checkout-y", "late-a")],
            )
        checkout_create.assert_not_awaited()
        await checkout_client.aclose()
        assert cast(Any, checkout_client)._active_replay_owners == {}

    run(collide_with_checked_out_owner())


def test_ambiguity_marker_capacity_poison_is_fail_closed() -> None:
    bounded_marker_create = AsyncMock()
    bounded_marker_client = _client(
        bounded_marker_create,
        policy=StreamRetentionPolicy(replay_history_item_limit=2),
    )
    for call_id in ("first", "second", "third"):
        cast(Any, bounded_marker_client)._record_ambiguous_replay_call_id(
            call_id
        )
    assert list(
        cast(Any, bounded_marker_client)._ambiguous_replay_call_ids
    ) == ["first", "second"]
    assert cast(Any, bounded_marker_client)._replay_association_poisoned
    with pytest.raises(openai_module._ReplayOwnerAssociationError):
        run(
            bounded_marker_client(
                "plain-model",
                [_tool_result("never-recorded", "value")],
            )
        )
    bounded_marker_create.assert_not_awaited()
    poisoned_owner = _owner()
    poisoned_owner.commit_attempt()
    with pytest.raises(openai_module._ReplayOwnerAssociationError):
        cast(Any, bounded_marker_client)._retain_replay_owner(
            poisoned_owner,
            ("later",),
        )
    assert poisoned_owner.released
    run(bounded_marker_client.aclose())
    assert not cast(Any, bounded_marker_client)._replay_association_poisoned
    assert cast(Any, bounded_marker_client)._ambiguous_replay_call_ids == {}


def test_retained_replay_owner_capacity_releases_rejected_owner() -> None:
    capacity_client = _client(
        AsyncMock(),
        policy=StreamRetentionPolicy(replay_history_item_limit=1),
    )
    capacity_a = _owner()
    capacity_b = _owner()
    capacity_a.commit_attempt()
    capacity_b.commit_attempt()
    cast(Any, capacity_client)._retain_replay_owner(capacity_a, ("one",))
    with pytest.raises(openai_module._ReasoningReplayRetentionError):
        cast(Any, capacity_client)._retain_replay_owner(capacity_b, ("two",))
    assert not capacity_a.released
    assert capacity_b.released
    run(capacity_client.aclose())
    assert capacity_a.released


def test_replay_owner_lifecycle_paths_release_exactly_once() -> None:
    lifecycle_owners = [_owner() for _ in range(4)]
    normal_items = run(
        _consume(
            OpenAIStream(
                _AsyncEvents([_completed_event()]),
                replay_owner=lifecycle_owners[0],
            )
        )
    )
    assert any(
        item.kind is StreamItemKind.STREAM_COMPLETED for item in normal_items
    )

    error_items = run(
        _consume(
            OpenAIStream(
                _AsyncEvents(
                    [
                        {
                            "type": "response.error",
                            "error": {"message": "failed"},
                        }
                    ]
                ),
                replay_owner=lifecycle_owners[1],
            )
        )
    )
    assert any(
        item.kind is StreamItemKind.STREAM_ERRORED for item in error_items
    )

    cancelled_stream = OpenAIStream(
        _AsyncEvents([]),
        replay_owner=lifecycle_owners[2],
    )
    run(cancelled_stream.cancel())
    run(cancelled_stream.cancel())

    async def close_consumer_early() -> None:
        stream = OpenAIStream(
            _AsyncEvents(
                [
                    {"type": "response.output_text.delta", "delta": "later"},
                    _completed_event(),
                ]
            ),
            replay_owner=lifecycle_owners[3],
        )
        iterator = stream.__aiter__()
        started = await iterator.__anext__()
        assert started.kind is StreamItemKind.STREAM_STARTED
        await cast(Any, iterator).aclose()
        await stream.aclose()

    run(close_consumer_early())

    create_failure = RuntimeError("create failed")
    failing_client = _client(AsyncMock(side_effect=create_failure))
    activate_owner = MagicMock(
        wraps=cast(Any, failing_client)._activate_replay_owner
    )
    cast(Any, failing_client)._activate_replay_owner = activate_owner
    with pytest.raises(RuntimeError) as create_error:
        run(failing_client("plain-model", [], _settings()))
    assert create_error.value is create_failure
    cast(
        AsyncMock, failing_client._client.responses.create
    ).assert_awaited_once()
    activate_owner.assert_called_once()
    activated_owner = activate_owner.call_args.args[0]

    assert [
        owner.release_count for owner in [*lifecycle_owners, activated_owner]
    ] == [1] * 5


def test_retainer_and_end_of_stream_failures_are_safe() -> None:
    retainer_owner = _owner()
    retainer_owner.admit(
        _reasoning_item("retainer-error", "cipher-retainer", [])
    )
    retainer = MagicMock(side_effect=RuntimeError("retainer failed"))
    retainer_source = _AsyncEvents(
        [
            _output_done(_function_item("retainer-error")),
            _completed_event(),
        ]
    )
    retainer_items = run(
        _consume(
            OpenAIStream(
                retainer_source,
                replay_owner=retainer_owner,
                replay_owner_retainer=retainer,
            )
        )
    )
    retainer_terminal = next(
        item
        for item in retainer_items
        if item.kind is StreamItemKind.STREAM_ERRORED
    )
    assert (
        cast(dict[str, Any], retainer_terminal.data)["error"]["code"]
        == "reasoning_replay_owner_ambiguous"
    )
    retainer.assert_called_once_with(retainer_owner, ("retainer-error",))
    assert retainer_owner.release_count == 1
    assert retainer_source.close_count == 1

    direct_retainer_owner = _owner()
    direct_retainer_owner.admit(
        _reasoning_item("direct-retainer-error", "cipher-direct", [])
    )
    direct_retainer = MagicMock(
        side_effect=openai_module._ReplayOwnerAssociationError()
    )
    direct_retainer_source = _AsyncEvents(
        [
            _output_done(_function_item("direct-retainer-error")),
            _completed_event(),
        ]
    )
    direct_retainer_events = run(
        _consume_provider_events(
            OpenAIStream(
                direct_retainer_source,
                replay_owner=direct_retainer_owner,
                replay_owner_retainer=direct_retainer,
            )
        )
    )
    assert direct_retainer_events[-1].kind is StreamItemKind.STREAM_ERRORED
    assert (
        cast(dict[str, Any], direct_retainer_events[-1].data)["error"]["code"]
        == "reasoning_replay_owner_ambiguous"
    )
    direct_retainer.assert_called_once_with(
        direct_retainer_owner,
        ("direct-retainer-error",),
    )
    assert direct_retainer_owner.release_count == 1
    assert direct_retainer_source.close_count == 1

    end_owner = _owner()
    end_owner.admit(_reasoning_item("end-error", "cipher-end", []))

    def release_with_retention_error(owner: Any) -> None:
        owner.release()
        raise openai_module._ReasoningReplayRetentionError()

    end_source = _AsyncEvents([])
    end_items = run(
        _consume(
            OpenAIStream(
                end_source,
                replay_owner=end_owner,
                replay_owner_releaser=release_with_retention_error,
            )
        )
    )
    end_terminal = next(
        item
        for item in end_items
        if item.kind is StreamItemKind.STREAM_ERRORED
    )
    assert (
        cast(dict[str, Any], end_terminal.data)["error"]["code"]
        == "reasoning_replay_retention_exceeded"
    )
    assert end_owner.release_count == 1
    assert end_source.close_count == 1


def test_abandoned_stream_capacity_and_close_release_once() -> None:
    abandoned_source = _AsyncEvents([_completed_event()])
    abandoned_create = AsyncMock(return_value=abandoned_source)
    abandoned_client = _client(
        abandoned_create,
        policy=StreamRetentionPolicy(replay_history_item_limit=1),
    )
    abandoned_stream = run(abandoned_client("plain-model", [], _settings()))
    active_owner = next(
        iter(cast(Any, abandoned_client)._active_replay_owners.values())
    )
    with pytest.raises(openai_module._ReasoningReplayRetentionError):
        run(abandoned_client("plain-model", [], _settings()))
    assert abandoned_create.await_count == 1
    run(abandoned_client.aclose())
    assert active_owner.release_count == 1
    assert abandoned_source.close_count == 1
    run(cast(OpenAIStream, abandoned_stream).aclose())
    assert active_owner.release_count == 1
    assert abandoned_source.close_count == 1


def test_client_close_during_create_cleans_late_response() -> None:
    async def close_during_create() -> None:
        create_started = Event()
        create_release = Event()
        raced_source = _AsyncEvents([_completed_event()])

        async def delayed_create(**kwargs: object) -> _AsyncEvents:
            assert kwargs["model"] == "plain-model"
            create_started.set()
            await wait_for(create_release.wait(), timeout=1.0)
            return raced_source

        raced_create = AsyncMock(side_effect=delayed_create)
        raced_client = _client(raced_create)
        request = create_task(raced_client("plain-model", []))
        await wait_for(create_started.wait(), timeout=1.0)
        await raced_client.aclose()
        create_release.set()
        with pytest.raises(openai_module._OpenAIClientClosedError):
            await wait_for(request, timeout=1.0)
        assert raced_source.close_count == 1
        assert cast(Any, raced_client)._active_replay_owners == {}
        assert cast(Any, raced_client)._active_replay_streams == {}
        assert cast(Any, raced_client)._replay_owners_by_call_id == {}
        with pytest.raises(openai_module._OpenAIClientClosedError):
            await raced_client("plain-model", [])
        assert raced_create.await_count == 1

        late_owner = _owner()
        late_owner.commit_attempt()
        with pytest.raises(openai_module._OpenAIClientClosedError):
            cast(Any, raced_client)._retain_replay_owner(
                late_owner,
                ("late",),
            )
        assert late_owner.released
        assert cast(Any, raced_client)._replay_owners_by_call_id == {}

    run(close_during_create())


def test_malformed_provider_payloads_are_safely_omitted() -> None:
    encrypted = _PRIVATE_ENCRYPTED_SENTINEL
    summary = _PRIVATE_SUMMARY_SENTINEL
    provider_mapping_cycle: dict[str, object] = {}
    provider_mapping_cycle["self"] = provider_mapping_cycle
    provider_sequence_cycle: list[object] = []
    provider_sequence_cycle.append(provider_sequence_cycle)
    for malformed_payload in (
        {"value": nan},
        provider_mapping_cycle,
        {1: "non-string-key"},
        {"value": provider_sequence_cycle},
        {"value": object()},
    ):
        assert OpenAIStream._provider_payload(malformed_payload) is None

    class FailingMapping(Mapping[str, object]):
        def __getitem__(self, key: str) -> object:
            raise KeyError(key)

        def __iter__(self) -> Iterator[str]:
            raise RuntimeError(encrypted)

        def __len__(self) -> int:
            return 1

    class FailingModelDump:
        def model_dump(self, *, mode: str) -> object:
            assert mode == "json"
            raise RuntimeError(summary)

    class FailingDumpMapping:
        def model_dump(self, *, mode: str) -> object:
            assert mode == "json"
            return FailingMapping()

    for malformed_event in (
        FailingMapping(),
        FailingModelDump(),
        FailingDumpMapping(),
    ):
        assert OpenAIStream._provider_payload(malformed_event) is None
        with pytest.raises(openai_module._ReasoningReplayRetentionError):
            OpenAIStream._raw_provider_payload(malformed_event)


def test_encrypted_reasoning_remains_opaque() -> None:
    encrypted = _PRIVATE_ENCRYPTED_SENTINEL
    summary = _PRIVATE_SUMMARY_SENTINEL
    payload = {
        "type": "response.completed",
        "response": {
            "usage": {},
            "output": [
                _reasoning_item(
                    "rs_private",
                    encrypted,
                    [{"type": "summary_text", "text": summary}],
                )
            ],
            "metadata": {"provider_secret": _CoercionTrap()},
        },
    }
    _CoercionTrap.calls = 0
    items = run(_consume(OpenAIStream(_AsyncEvents([payload]))))
    error_items = run(
        _consume(
            OpenAIStream(
                _AsyncEvents(
                    [{"type": "response.error", "error": _CoercionTrap()}]
                )
            )
        )
    )
    outward = str([item.to_trace_dict() for item in [*items, *error_items]])

    assert encrypted not in outward
    assert summary not in outward
    assert "must-not-coerce" not in outward
    assert _CoercionTrap.calls == 0
    assert all(
        item.provider_payload is None for item in [*items, *error_items]
    )


def test_replay_provider_cancellation_is_private_and_releases_owner() -> None:
    encrypted = _PRIVATE_ENCRYPTED_SENTINEL
    summary = _PRIVATE_SUMMARY_SENTINEL
    provider_cancel_owner = _owner()
    provider_cancel_owner.admit(
        _reasoning_item(
            "provider-cancel-private",
            encrypted,
            [{"text": summary}],
        )
    )
    provider_cancel_items = run(
        _consume(
            OpenAIStream(
                _FailingAsyncEvents([], CancelledError()),
                replay_owner=provider_cancel_owner,
            )
        )
    )
    assert [item.kind for item in provider_cancel_items] == [
        StreamItemKind.STREAM_STARTED,
        StreamItemKind.STREAM_CANCELLED,
        StreamItemKind.STREAM_CLOSED,
    ]
    provider_cancel_terminal = provider_cancel_items[1]
    assert provider_cancel_terminal.provider_payload is None
    assert provider_cancel_terminal.data is None
    assert encrypted not in repr(provider_cancel_items)
    assert summary not in repr(provider_cancel_items)
    assert provider_cancel_owner.release_count == 1


def test_retained_replay_provider_errors_are_opaque() -> None:
    encrypted = _PRIVATE_ENCRYPTED_SENTINEL
    summary = _PRIVATE_SUMMARY_SENTINEL
    for index, echoed in enumerate((encrypted, summary)):
        call_id = f"echo-{index}"
        echo_stream = _AsyncEvents(
            [
                {
                    "type": "response.error",
                    "error": {"message": echoed},
                    "metadata": {"echo": echoed},
                }
            ]
        )
        echo_client = _retained_private_client(
            AsyncMock(return_value=echo_stream),
            call_id,
        )
        echo_items = run(
            _consume(
                run(
                    echo_client(
                        "plain-model",
                        [_tool_result(call_id, "value")],
                    )
                )
            )
        )
        echo_terminal = next(
            item
            for item in echo_items
            if item.kind is StreamItemKind.STREAM_ERRORED
        )
        echo_outward = "\n".join(
            (
                repr(echo_items),
                str(echo_items),
                repr([item.to_trace_dict() for item in echo_items]),
            )
        )
        assert encrypted not in echo_outward
        assert summary not in echo_outward
        assert echo_terminal.provider_payload is None
        assert cast(dict[str, Any], echo_terminal.data)["error"] == {
            "type": "server_error",
            "code": "openai_provider_request_failed",
            "status": "failed",
            "message": "OpenAI provider request failed",
        }


def test_first_turn_reasoning_latches_provider_privacy() -> None:
    encrypted = _PRIVATE_ENCRYPTED_SENTINEL
    summary = _PRIVATE_SUMMARY_SENTINEL
    first_turn_events = _AsyncEvents(
        [
            _output_done(
                _reasoning_item(
                    "first-turn-private",
                    encrypted,
                    [{"type": "summary_text", "text": summary}],
                )
            ),
            {
                "type": "response.error",
                "error": {"message": encrypted},
                "metadata": {"echo": summary},
            },
        ]
    )
    first_turn_items = run(
        _consume(
            run(
                _client(AsyncMock(return_value=first_turn_events))(
                    "plain-model",
                    [],
                )
            )
        )
    )
    first_turn_outward = repr(
        [item.to_trace_dict() for item in first_turn_items]
    )
    assert encrypted not in first_turn_outward
    assert summary not in first_turn_outward
    assert any(
        item.kind is StreamItemKind.STREAM_ERRORED
        and item.provider_payload is None
        and cast(dict[str, Any], item.data)["error"]["code"]
        == "openai_provider_request_failed"
        for item in first_turn_items
    )


def test_private_replay_standard_usage_is_preserved() -> None:
    encrypted = _PRIVATE_ENCRYPTED_SENTINEL
    summary = _PRIVATE_SUMMARY_SENTINEL
    completed_echo_events = _AsyncEvents(
        [
            _output_done(
                _reasoning_item(
                    "completed-private",
                    encrypted,
                    [{"type": "summary_text", "text": summary}],
                )
            ),
            {
                "type": "response.completed",
                "metadata": {"encrypted_echo": encrypted},
                "response": {
                    "usage": {
                        "input_tokens": 2,
                        "input_tokens_details": {"cached_tokens": 1},
                        "output_tokens": 3,
                        "output_tokens_details": {"reasoning_tokens": 2},
                        "total_tokens": 5,
                    },
                    "output_text": "safe-answer",
                    "metadata": {"summary_echo": summary},
                },
            },
        ]
    )
    completed_echo_stream = run(
        _client(AsyncMock(return_value=completed_echo_events))(
            "plain-model",
            [],
        )
    )
    completed_echo_items = run(_consume(completed_echo_stream))
    completed_echo_outward = repr(
        [item.to_trace_dict() for item in completed_echo_items]
    )
    assert encrypted not in completed_echo_outward
    assert summary not in completed_echo_outward
    assert all(item.provider_payload is None for item in completed_echo_items)
    completed_accumulated = accumulate_canonical_stream_items(
        completed_echo_items
    )
    assert completed_accumulated.answer_text == "safe-answer"
    assert any(
        item.kind is StreamItemKind.USAGE_COMPLETED
        and item.usage
        == {
            "input_tokens": 2,
            "input_tokens_details": {"cached_tokens": 1},
            "output_tokens": 3,
            "output_tokens_details": {"reasoning_tokens": 2},
            "total_tokens": 5,
        }
        for item in completed_echo_items
    )
    assert cast(OpenAIStream, completed_echo_stream).usage == {
        "input_tokens": 2,
        "input_tokens_details": {"cached_tokens": 1},
        "output_tokens": 3,
        "output_tokens_details": {"reasoning_tokens": 2},
        "total_tokens": 5,
    }


def test_private_replay_usage_aliases_remain_compatible() -> None:
    encrypted = _PRIVATE_ENCRYPTED_SENTINEL
    summary = _PRIVATE_SUMMARY_SENTINEL
    alias_usage_events = _AsyncEvents(
        [
            _output_done(
                _reasoning_item(
                    "alias-usage-private",
                    encrypted,
                    [{"text": summary}],
                )
            ),
            {
                "type": "response.completed",
                "response": {
                    "usage": {
                        "prompt_tokens": 7,
                        "prompt_tokens_details": {"cached_tokens": 3},
                        "completion_tokens": 5,
                        "completion_tokens_details": {"reasoning_tokens": 2},
                        "total_tokens": 12,
                    }
                },
            },
        ]
    )
    alias_usage_stream = run(
        _client(AsyncMock(return_value=alias_usage_events))(
            "plain-model",
            [],
        )
    )
    alias_usage_items = run(_consume(alias_usage_stream))
    expected_alias_usage = {
        "prompt_tokens": 7,
        "prompt_tokens_details": {"cached_tokens": 3},
        "completion_tokens": 5,
        "completion_tokens_details": {"reasoning_tokens": 2},
        "total_tokens": 12,
    }
    assert any(
        item.kind is StreamItemKind.USAGE_COMPLETED
        and item.usage == expected_alias_usage
        for item in alias_usage_items
    )
    assert cast(OpenAIStream, alias_usage_stream).usage == expected_alias_usage
    alias_totals = usage_totals_from_response(alias_usage_stream)
    assert alias_totals is not None
    assert alias_totals.input_tokens == 7
    assert alias_totals.cached_input_tokens == 3
    assert alias_totals.output_tokens == 5
    assert alias_totals.reasoning_tokens == 2
    assert alias_totals.total_tokens == 12


def test_private_replay_usage_rejects_opaque_or_invalid_values() -> None:
    encrypted = _PRIVATE_ENCRYPTED_SENTINEL
    summary = _PRIVATE_SUMMARY_SENTINEL
    usage_echo_events = _AsyncEvents(
        [
            _output_done(
                _reasoning_item(
                    "usage-private",
                    encrypted,
                    [{"text": summary}],
                )
            ),
            {
                "type": "response.completed",
                "response": {
                    "usage": {
                        "input_tokens": encrypted,
                        "output_tokens_details": {"reasoning_tokens": summary},
                        "metadata": {"echo": encrypted},
                    }
                },
            },
        ]
    )
    usage_echo_stream = run(
        _client(AsyncMock(return_value=usage_echo_events))(
            "plain-model",
            [],
        )
    )
    usage_echo_items = run(_consume(usage_echo_stream))
    usage_echo_outward = repr(
        [item.to_trace_dict() for item in usage_echo_items]
    )
    assert encrypted not in usage_echo_outward
    assert summary not in usage_echo_outward
    assert not any(
        item.kind is StreamItemKind.USAGE_COMPLETED
        for item in usage_echo_items
    )
    assert cast(OpenAIStream, usage_echo_stream).usage is None

    for primitive_usage in (
        True,
        bytearray(b"usage"),
        b"usage",
        1.0,
        1,
        [],
        "usage",
        (),
    ):
        assert OpenAIStream._private_replay_usage(primitive_usage) is None
    assert OpenAIStream._private_replay_usage({"input_tokens": 1.5}) == {
        "input_tokens": 1.5
    }
    for invalid_usage in (
        {"input_tokens": -1},
        {"input_tokens": inf},
        {"output_tokens_details": {"reasoning_tokens": summary}},
    ):
        assert OpenAIStream._private_replay_usage(invalid_usage) is None

    class ExplodingUsage:
        @property
        def cacheCreationInputTokens(self) -> object:
            raise RuntimeError(f"{encrypted} {summary}")

    assert OpenAIStream._private_replay_usage(ExplodingUsage()) is None


def test_private_replay_adapter_failures_are_sanitized() -> None:
    encrypted = _PRIVATE_ENCRYPTED_SENTINEL
    summary = _PRIVATE_SUMMARY_SENTINEL
    adapter_call_id = "adapter-private"
    adapter_client = _retained_private_client(
        AsyncMock(
            return_value=_AsyncEvents(
                [
                    {
                        "type": "response.output_text.delta",
                        "delta": {"echo": encrypted},
                        "metadata": {"echo": summary},
                    }
                ]
            )
        ),
        adapter_call_id,
    )
    adapter_items = run(
        _consume(
            run(
                adapter_client(
                    "plain-model",
                    [_tool_result(adapter_call_id, "value")],
                )
            )
        )
    )
    adapter_outward = repr([item.to_trace_dict() for item in adapter_items])
    assert encrypted not in adapter_outward
    assert summary not in adapter_outward
    assert any(
        item.kind is StreamItemKind.STREAM_ERRORED
        and item.provider_payload is None
        for item in adapter_items
    )


def test_private_replay_iterator_failures_are_sanitized_and_closed() -> None:
    encrypted = _PRIVATE_ENCRYPTED_SENTINEL
    summary = _PRIVATE_SUMMARY_SENTINEL
    pull_call_id = "pull-private"
    pull_source = _FailingAsyncEvents(
        [],
        RuntimeError(f"{encrypted} {summary}"),
    )
    pull_client = _retained_private_client(
        AsyncMock(return_value=pull_source),
        pull_call_id,
    )
    pull_items = run(
        _consume(
            run(
                pull_client(
                    "plain-model",
                    [_tool_result(pull_call_id, "value")],
                )
            )
        )
    )
    pull_outward = repr([item.to_trace_dict() for item in pull_items])
    assert encrypted not in pull_outward
    assert summary not in pull_outward
    assert (
        sum(item.kind is StreamItemKind.STREAM_ERRORED for item in pull_items)
        == 1
    )
    assert pull_source.close_count == 1
    assert cast(Any, pull_client)._replay_owners_by_call_id == {}

    first_pull_source = _FailingAsyncEvents(
        [
            _output_done(
                _reasoning_item(
                    "first-pull-private",
                    encrypted,
                    [{"text": summary}],
                )
            )
        ],
        RuntimeError(f"{encrypted} {summary}"),
    )
    first_pull_source.aclose = AsyncMock(
        side_effect=RuntimeError(f"cleanup {encrypted} {summary}")
    )
    first_pull_items = run(
        _consume(
            run(
                _client(AsyncMock(return_value=first_pull_source))(
                    "plain-model",
                    [],
                )
            )
        )
    )
    first_pull_outward = repr(
        [item.to_trace_dict() for item in first_pull_items]
    )
    assert encrypted not in first_pull_outward
    assert summary not in first_pull_outward
    first_pull_terminal = [
        item
        for item in first_pull_items
        if item.kind is StreamItemKind.STREAM_ERRORED
    ]
    assert len(first_pull_terminal) == 1
    assert (
        cast(dict[str, Any], first_pull_terminal[0].data)["cleanup_error"][
            "code"
        ]
        == "openai_cleanup_failed"
    )
    first_pull_source.aclose.assert_awaited_once_with()


def test_private_replay_create_errors_and_cancellation_are_sanitized() -> None:
    encrypted = _PRIVATE_ENCRYPTED_SENTINEL
    summary = _PRIVATE_SUMMARY_SENTINEL

    async def echoing_create(**kwargs: object) -> object:
        raise RuntimeError(repr(kwargs["input"]))

    create_call_id = "create-private"
    create_client = _retained_private_client(
        AsyncMock(side_effect=echoing_create),
        create_call_id,
    )
    with pytest.raises(
        openai_module._OpenAIProviderRequestError
    ) as create_context:
        run(
            create_client(
                "plain-model",
                [_tool_result(create_call_id, "value")],
            )
        )
    create_nodes, create_diagnostics = _safe_exception_diagnostics(
        create_context.value
    )
    assert [getattr(node, "code", None) for node in create_nodes] == [
        "openai_provider_request_failed"
    ]
    assert encrypted not in create_diagnostics
    assert summary not in create_diagnostics

    cancelled_call_id = "cancelled-create-private"
    cancelled_owner = _owner()
    cancelled_owner.admit(
        _reasoning_item(
            "cancelled-private",
            encrypted,
            [{"text": summary}],
        )
    )
    cancelled_owner.admit(_function_item(cancelled_call_id))
    cancelled_owner.commit_attempt()
    cancelled_client = _client(
        AsyncMock(side_effect=CancelledError(f"{encrypted} {summary}"))
    )
    cast(Any, cancelled_client)._retain_replay_owner(
        cancelled_owner,
        (cancelled_call_id,),
    )

    async def cancelled_private_create() -> BaseException:
        with pytest.raises(CancelledError) as context:
            await cancelled_client(
                "plain-model",
                [_tool_result(cancelled_call_id, "value")],
            )
        return context.value

    safe_cancellation = run(cancelled_private_create())
    assert safe_cancellation.args == ()
    _, cancellation_diagnostics = _safe_exception_diagnostics(
        safe_cancellation
    )
    assert encrypted not in cancellation_diagnostics
    assert summary not in cancellation_diagnostics
    assert cancelled_owner.release_count == 1

    baseline_cancellation = CancelledError("baseline cancellation")
    baseline_client = _client(AsyncMock(side_effect=baseline_cancellation))

    async def cancelled_baseline_create() -> BaseException:
        with pytest.raises(CancelledError) as context:
            await baseline_client("plain-model", [])
        return context.value

    assert run(cancelled_baseline_create()) is baseline_cancellation

    class ProviderControlFlow(BaseException):
        pass

    control_flow_call_id = "control-flow-create-private"
    provider_control_flow = ProviderControlFlow("provider control flow")
    control_flow_client = _retained_private_client(
        AsyncMock(side_effect=provider_control_flow),
        control_flow_call_id,
    )
    control_flow_owner = cast(
        Any, control_flow_client
    )._replay_owners_by_call_id[control_flow_call_id]

    async def control_flow_private_create() -> BaseException:
        with pytest.raises(ProviderControlFlow) as context:
            await control_flow_client(
                "plain-model",
                [_tool_result(control_flow_call_id, "value")],
            )
        return context.value

    assert run(control_flow_private_create()) is provider_control_flow
    assert control_flow_owner.release_count == 1
    assert cast(Any, control_flow_client)._replay_owners_by_call_id == {}


def test_private_replay_retry_factory_errors_are_sanitized() -> None:
    encrypted = _PRIVATE_ENCRYPTED_SENTINEL
    summary = _PRIVATE_SUMMARY_SENTINEL
    retry_call_id = "retry-private"
    retry_client = _retained_private_client(
        AsyncMock(
            side_effect=[
                _AsyncEvents(
                    [
                        {
                            "type": "response.failed",
                            "response": {
                                "status": "failed",
                                "error": None,
                                "output": [],
                            },
                        }
                    ]
                ),
                RuntimeError(f"{encrypted} {summary}"),
            ]
        ),
        retry_call_id,
    )
    retry_items = run(
        _consume(
            run(
                retry_client(
                    "plain-model",
                    [_tool_result(retry_call_id, "value")],
                )
            )
        )
    )
    retry_outward = repr([item.to_trace_dict() for item in retry_items])
    assert encrypted not in retry_outward
    assert summary not in retry_outward
    assert any(
        item.kind is StreamItemKind.STREAM_ERRORED for item in retry_items
    )


def test_private_replay_non_stream_errors_and_usage_are_sanitized() -> None:
    encrypted = _PRIVATE_ENCRYPTED_SENTINEL
    summary = _PRIVATE_SUMMARY_SENTINEL

    class EchoingInvalidResponse:
        @property
        def output(self) -> object:
            raise RuntimeError(f"{encrypted} {summary}")

    baseline_adapter_error = RuntimeError("baseline adapter failure")

    class BaselineInvalidResponse:
        @property
        def output(self) -> object:
            raise baseline_adapter_error

    baseline_adapter_client = _client(
        AsyncMock(return_value=BaselineInvalidResponse())
    )
    with pytest.raises(RuntimeError) as baseline_adapter_context:
        run(
            baseline_adapter_client(
                "plain-model",
                [],
                use_async_generator=False,
            )
        )
    assert baseline_adapter_context.value is baseline_adapter_error

    non_stream_call_id = "non-stream-private"
    non_stream_client = _retained_private_client(
        AsyncMock(return_value=EchoingInvalidResponse()),
        non_stream_call_id,
    )
    with pytest.raises(
        openai_module._OpenAIProviderRequestError
    ) as non_stream_context:
        run(
            non_stream_client(
                "plain-model",
                [_tool_result(non_stream_call_id, "value")],
                use_async_generator=False,
            )
        )
    _, non_stream_diagnostics = _safe_exception_diagnostics(
        non_stream_context.value
    )
    assert encrypted not in non_stream_diagnostics
    assert summary not in non_stream_diagnostics

    non_stream_usage_call_id = "non-stream-usage-private"
    non_stream_usage_client = _retained_private_client(
        AsyncMock(
            return_value=SimpleNamespace(
                output=[],
                usage={
                    "input_tokens": encrypted,
                    "output_tokens_details": {"reasoning_tokens": summary},
                },
            )
        ),
        non_stream_usage_call_id,
    )
    non_stream_usage = run(
        non_stream_usage_client(
            "plain-model",
            [_tool_result(non_stream_usage_call_id, "value")],
            use_async_generator=False,
        )
    )
    assert non_stream_usage.usage is None
    assert encrypted not in repr(non_stream_usage)
    assert summary not in repr(non_stream_usage)


def test_private_replay_non_stream_cancellation_preserves_semantics() -> None:
    encrypted = _PRIVATE_ENCRYPTED_SENTINEL
    summary = _PRIVATE_SUMMARY_SENTINEL
    call_id = "non-stream-cancelled-private"
    private_cancellation = CancelledError(f"{encrypted} {summary}")
    private_client = _retained_private_client(
        AsyncMock(return_value=_non_stream_response()),
        call_id,
    )
    private_owner = cast(Any, private_client)._replay_owners_by_call_id[
        call_id
    ]
    cast(Any, private_client)._non_stream_result = AsyncMock(
        side_effect=private_cancellation
    )

    async def cancelled_private_adapter() -> BaseException:
        with pytest.raises(CancelledError) as context:
            await private_client(
                "plain-model",
                [_tool_result(call_id, "value")],
                use_async_generator=False,
            )
        return context.value

    safe_cancellation = run(cancelled_private_adapter())
    assert safe_cancellation is not private_cancellation
    assert safe_cancellation.args == ()
    _, cancellation_diagnostics = _safe_exception_diagnostics(
        safe_cancellation
    )
    assert encrypted not in cancellation_diagnostics
    assert summary not in cancellation_diagnostics
    assert private_owner.release_count == 1
    assert cast(Any, private_client)._replay_owners_by_call_id == {}
    assert cast(Any, private_client)._active_replay_owners == {}
    assert cast(Any, private_client)._active_replay_streams == {}

    baseline_cancellation = CancelledError("baseline adapter cancellation")
    baseline_client = _client(AsyncMock(return_value=_non_stream_response()))
    cast(Any, baseline_client)._non_stream_result = AsyncMock(
        side_effect=baseline_cancellation
    )

    async def cancelled_baseline_adapter() -> BaseException:
        with pytest.raises(CancelledError) as context:
            await baseline_client(
                "plain-model",
                [],
                use_async_generator=False,
            )
        return context.value

    assert run(cancelled_baseline_adapter()) is baseline_cancellation


def test_close_during_create_sanitizes_response_cleanup_failure() -> None:
    encrypted = _PRIVATE_ENCRYPTED_SENTINEL
    summary = _PRIVATE_SUMMARY_SENTINEL

    async def close_create_with_cleanup_failure() -> BaseException:
        create_started = Event()
        create_release = Event()
        response = _AsyncEvents([])
        response.aclose = AsyncMock(
            side_effect=RuntimeError(f"{encrypted} {summary}")
        )

        async def delayed_create(**kwargs: object) -> _AsyncEvents:
            assert kwargs["model"] == "plain-model"
            create_started.set()
            await wait_for(create_release.wait(), timeout=1.0)
            return response

        client = _client(AsyncMock(side_effect=delayed_create))
        request = create_task(client("plain-model", []))
        await wait_for(create_started.wait(), timeout=1.0)
        await client.aclose()
        create_release.set()
        with pytest.raises(BaseExceptionGroup) as context:
            await wait_for(request, timeout=1.0)
        response.aclose.assert_awaited_once_with()
        return context.value

    response_cleanup_error = run(close_create_with_cleanup_failure())
    response_nodes, response_diagnostics = _safe_exception_diagnostics(
        response_cleanup_error
    )
    assert {getattr(node, "code", None) for node in response_nodes} >= {
        "openai_client_closed",
        "openai_cleanup_failed",
    }
    assert {
        getattr(node, "cleanup_target", None) for node in response_nodes
    } >= {"response"}
    assert encrypted not in response_diagnostics
    assert summary not in response_diagnostics


def test_client_cleanup_failures_are_content_free_and_aggregated() -> None:
    encrypted = _PRIVATE_ENCRYPTED_SENTINEL
    summary = _PRIVATE_SUMMARY_SENTINEL
    client_cleanup = _client(AsyncMock(return_value=_AsyncEvents([])))
    registered_stream = run(client_cleanup("plain-model", []))
    active_owners = cast(Any, client_cleanup)._active_replay_owners
    active_streams = cast(Any, client_cleanup)._active_replay_streams
    assert len(active_owners) == 1
    assert active_streams == {
        next(iter(active_owners)): registered_stream,
    }
    failing_stream_cleanup = AsyncMock(
        side_effect=RuntimeError(f"stream {encrypted}")
    )
    failing_client_cleanup = AsyncMock(
        side_effect=RuntimeError(f"client {summary}")
    )
    cast(Any, registered_stream).aclose = failing_stream_cleanup
    cast(Any, client_cleanup)._client.close = failing_client_cleanup
    with pytest.raises(BaseExceptionGroup) as cleanup_context:
        run(client_cleanup.aclose())
    cleanup_nodes, cleanup_diagnostics = _safe_exception_diagnostics(
        cleanup_context.value
    )
    assert {
        getattr(node, "cleanup_target", None) for node in cleanup_nodes
    } >= {"client", "stream"}
    assert {getattr(node, "code", None) for node in cleanup_nodes} >= {
        "openai_cleanup_failed"
    }
    assert encrypted not in cleanup_diagnostics
    assert summary not in cleanup_diagnostics
    failing_stream_cleanup.assert_awaited_once_with()
    failing_client_cleanup.assert_awaited_once_with()

    single_cleanup = _client(AsyncMock())
    cast(Any, single_cleanup)._client = SimpleNamespace(
        close=AsyncMock(side_effect=RuntimeError(encrypted))
    )
    with pytest.raises(openai_module._OpenAICleanupError) as single_context:
        run(single_cleanup.aclose())
    single_nodes, single_diagnostics = _safe_exception_diagnostics(
        single_context.value
    )
    assert [getattr(node, "code", None) for node in single_nodes] == [
        "openai_cleanup_failed"
    ]
    assert encrypted not in single_diagnostics


def test_stream_cleanup_methods_sanitize_private_failures() -> None:
    encrypted = _PRIVATE_ENCRYPTED_SENTINEL
    summary = _PRIVATE_SUMMARY_SENTINEL
    for method_name in ("cancel", "aclose"):
        private_owner = _owner()
        private_owner.admit(
            _reasoning_item(
                f"cleanup-{method_name}",
                encrypted,
                [{"text": summary}],
            )
        )
        private_source = _AsyncEvents([])
        private_source.aclose = AsyncMock(
            side_effect=RuntimeError(f"{encrypted} {summary}")
        )
        private_stream = OpenAIStream(
            private_source,
            replay_owner=private_owner,
        )
        with pytest.raises(openai_module._OpenAICleanupError) as context:
            run(getattr(private_stream, method_name)())
        cleanup_nodes, direct_cleanup_diagnostics = (
            _safe_exception_diagnostics(context.value)
        )
        assert [getattr(node, "code", None) for node in cleanup_nodes] == [
            "openai_cleanup_failed"
        ]
        assert encrypted not in direct_cleanup_diagnostics
        assert summary not in direct_cleanup_diagnostics
        private_source.aclose.assert_awaited_once_with()
        assert private_owner.release_count == 1


def test_private_cancel_cleanup_failure_preserves_stored_terminal() -> None:
    secret = "PRIVATE_CANCEL_CLEANUP_SENTINEL"
    owner = _owner()
    owner.admit(_reasoning_item("cancel-cleanup", secret, [{"text": secret}]))
    source = _AsyncEvents([])
    source.aclose = AsyncMock(side_effect=RuntimeError(secret))
    stream = OpenAIStream(source, replay_owner=owner)

    with pytest.raises(openai_module._OpenAICleanupError) as context:
        run(stream.cancel())
    run(stream.cancel())
    items = run(_consume(stream))
    outward = repr(
        [
            context.value,
            items,
            [item.to_trace_dict() for item in items],
        ]
    )

    assert [item.kind for item in items] == [
        StreamItemKind.STREAM_STARTED,
        StreamItemKind.STREAM_CANCELLED,
        StreamItemKind.STREAM_CLOSED,
    ]
    assert secret not in outward
    source.aclose.assert_awaited_once_with()
    assert owner.released
    assert owner.release_count == 1


def test_cleanup_boundary_types_and_groups_are_content_free() -> None:
    encrypted = _PRIVATE_ENCRYPTED_SENTINEL
    summary = _PRIVATE_SUMMARY_SENTINEL
    boundary_owner = _owner()
    boundary_owner.admit(
        _reasoning_item("cleanup-boundaries", encrypted, [{"text": summary}])
    )
    boundary_stream = OpenAIStream(
        _AsyncEvents([]),
        replay_owner=boundary_owner,
    )
    boundary_cases = (
        (
            openai_module._OpenAIClientClosedError(),
            openai_module._OpenAIClientClosedError,
            None,
        ),
        (
            openai_module._ReasoningReplayRetentionError(),
            openai_module._ReasoningReplayRetentionError,
            None,
        ),
        (
            openai_module._ReplayOwnerAssociationError(),
            openai_module._ReplayOwnerAssociationError,
            None,
        ),
        (
            openai_module._OpenAIProviderRequestError(),
            openai_module._OpenAIProviderRequestError,
            None,
        ),
        (
            openai_module._OpenAICleanupError("client"),
            openai_module._OpenAICleanupError,
            "client",
        ),
    )
    for original, expected_type, expected_target in boundary_cases:
        sanitized = boundary_stream._cleanup_boundary_error(original)
        assert isinstance(sanitized, expected_type)
        assert sanitized is not original
        assert getattr(sanitized, "cleanup_target", None) == expected_target
        _, boundary_diagnostics = _safe_exception_diagnostics(sanitized)
        assert encrypted not in boundary_diagnostics
        assert summary not in boundary_diagnostics
    run(boundary_stream.aclose())
    assert boundary_owner.release_count == 1

    grouped_owner = _owner()
    grouped_owner.admit(
        _reasoning_item("grouped-cleanup", encrypted, [{"text": summary}])
    )
    grouped_source = _AsyncEvents([])
    grouped_source.aclose = AsyncMock(
        side_effect=RuntimeError(f"source {encrypted} {summary}")
    )

    def release_then_fail(owner: Any) -> None:
        owner.release()
        raise RuntimeError(f"releaser {encrypted} {summary}")

    grouped_stream = OpenAIStream(
        grouped_source,
        replay_owner=grouped_owner,
        replay_owner_releaser=release_then_fail,
    )
    with pytest.raises(BaseExceptionGroup) as grouped_context:
        run(grouped_stream.aclose())
    grouped_nodes, grouped_diagnostics = _safe_exception_diagnostics(
        grouped_context.value
    )
    assert len(grouped_context.value.exceptions) == 2
    assert {getattr(node, "code", None) for node in grouped_nodes} >= {
        "openai_cleanup_failed"
    }
    assert encrypted not in grouped_diagnostics
    assert summary not in grouped_diagnostics
    assert grouped_owner.release_count == 1
    grouped_source.aclose.assert_awaited_once_with()


def test_retry_source_cleanup_preserves_baseline_and_private_semantics() -> (
    None
):
    encrypted = _PRIVATE_ENCRYPTED_SENTINEL
    summary = _PRIVATE_SUMMARY_SENTINEL
    baseline_retry_cleanup_error = RuntimeError("baseline retry cleanup")
    baseline_retry_source = _AsyncEvents([])
    baseline_retry_source.aclose = AsyncMock(
        side_effect=baseline_retry_cleanup_error
    )
    baseline_retry_stream = OpenAIStream(baseline_retry_source)
    with pytest.raises(RuntimeError) as baseline_retry_context:
        run(baseline_retry_stream._close_current_stream())
    assert baseline_retry_context.value is baseline_retry_cleanup_error
    assert cast(Any, baseline_retry_stream)._stream_sources == ()

    private_retry_owner = _owner()
    private_retry_owner.admit(
        _reasoning_item("retry-cleanup", encrypted, [{"text": summary}])
    )
    private_retry_source = _AsyncEvents([])
    private_retry_source.aclose = AsyncMock(
        side_effect=RuntimeError(f"{encrypted} {summary}")
    )
    private_retry_stream = OpenAIStream(
        private_retry_source,
        replay_owner=private_retry_owner,
    )
    with pytest.raises(openai_module._OpenAICleanupError) as retry_context:
        run(private_retry_stream._close_current_stream())
    retry_nodes, retry_diagnostics = _safe_exception_diagnostics(
        retry_context.value
    )
    assert [getattr(node, "code", None) for node in retry_nodes] == [
        "openai_cleanup_failed"
    ]
    assert encrypted not in retry_diagnostics
    assert summary not in retry_diagnostics
    assert cast(Any, private_retry_stream)._stream_sources == ()
    run(private_retry_stream.aclose())
    assert private_retry_owner.release_count == 1


def test_direct_private_provider_failure_exhaustion_is_content_free() -> None:
    encrypted = _PRIVATE_ENCRYPTED_SENTINEL
    summary = _PRIVATE_SUMMARY_SENTINEL
    direct_failure_owner = _owner()
    direct_failure_owner.admit(
        _reasoning_item("direct-failure", encrypted, [{"text": summary}])
    )
    direct_failure_source = _FailingAsyncEvents(
        [],
        RuntimeError(f"{encrypted} {summary}"),
    )
    direct_failure_events = run(
        _consume_provider_events(
            OpenAIStream(
                direct_failure_source,
                replay_owner=direct_failure_owner,
            )
        )
    )
    assert [event.kind for event in direct_failure_events] == [
        StreamItemKind.STREAM_ERRORED
    ]
    assert encrypted not in repr(direct_failure_events)
    assert summary not in repr(direct_failure_events)
    assert direct_failure_owner.release_count == 1
    assert direct_failure_source.close_count == 1


def test_private_iterator_cancellation_has_exact_terminal_sequence() -> None:
    encrypted = _PRIVATE_ENCRYPTED_SENTINEL
    summary = _PRIVATE_SUMMARY_SENTINEL
    iterator_owner = _owner()
    iterator_owner.admit(
        _reasoning_item(
            "iterator-cancelled",
            encrypted,
            [{"type": "summary_text", "text": summary}],
        )
    )
    iterator_source = _FailingAsyncEvents(
        [],
        CancelledError(f"{encrypted} {summary}"),
    )
    iterator_items = run(
        _consume(
            OpenAIStream(
                iterator_source,
                replay_owner=iterator_owner,
            )
        )
    )
    assert [item.kind for item in iterator_items] == [
        StreamItemKind.STREAM_STARTED,
        StreamItemKind.STREAM_CANCELLED,
        StreamItemKind.STREAM_CLOSED,
    ]
    iterator_outward = "\n".join(
        (
            repr(iterator_items),
            str(iterator_items),
            repr([item.to_trace_dict() for item in iterator_items]),
        )
    )
    assert encrypted not in iterator_outward
    assert summary not in iterator_outward
    iterator_terminal = iterator_items[1]
    assert iterator_terminal.data is None
    assert iterator_terminal.provider_payload is None
    assert iterator_terminal.terminal_outcome is not None
    assert iterator_terminal.terminal_outcome.value == "cancelled"
    assert iterator_owner.release_count == 1
    assert iterator_source.close_count == 1


def test_private_retry_factory_cancellation_has_exact_terminal_sequence() -> (
    None
):
    encrypted = _PRIVATE_ENCRYPTED_SENTINEL
    summary = _PRIVATE_SUMMARY_SENTINEL
    retry_owner = _owner()
    retry_owner.admit(
        _reasoning_item(
            "retry-factory-cancelled",
            encrypted,
            [{"type": "summary_text", "text": summary}],
        )
    )
    retry_source = _AsyncEvents(
        [
            {
                "type": "response.failed",
                "response": {
                    "status": "failed",
                    "error": None,
                    "output": [],
                },
            }
        ]
    )
    retry_factory = AsyncMock(
        side_effect=CancelledError(f"{encrypted} {summary}")
    )
    retry_cancelled_items = run(
        _consume(
            OpenAIStream(
                retry_source,
                replay_owner=retry_owner,
                stream_factory=retry_factory,
                stream_retries=1,
            )
        )
    )
    assert [item.kind for item in retry_cancelled_items] == [
        StreamItemKind.STREAM_STARTED,
        StreamItemKind.STREAM_CANCELLED,
        StreamItemKind.STREAM_CLOSED,
    ]
    retry_cancelled_outward = "\n".join(
        (
            repr(retry_cancelled_items),
            str(retry_cancelled_items),
            repr([item.to_trace_dict() for item in retry_cancelled_items]),
        )
    )
    assert encrypted not in retry_cancelled_outward
    assert summary not in retry_cancelled_outward
    retry_cancelled_terminal = retry_cancelled_items[1]
    assert retry_cancelled_terminal.data is None
    assert retry_cancelled_terminal.provider_payload is None
    assert retry_cancelled_terminal.terminal_outcome is not None
    assert retry_cancelled_terminal.terminal_outcome.value == "cancelled"
    retry_factory.assert_awaited_once_with()
    assert retry_owner.release_count == 1
    assert retry_source.close_count == 1


def test_private_terminal_followed_by_cleanup_failure_is_sanitized() -> None:
    encrypted = _PRIVATE_ENCRYPTED_SENTINEL
    summary = _PRIVATE_SUMMARY_SENTINEL
    terminal_owner = _owner()
    terminal_owner.admit(
        _reasoning_item(
            "terminal-cleanup-failed",
            encrypted,
            [{"type": "summary_text", "text": summary}],
        )
    )
    terminal_source = _AsyncEvents(
        [
            {
                "type": "response.error",
                "error": {"message": encrypted},
                "metadata": {"echo": summary},
            }
        ]
    )
    terminal_source.aclose = AsyncMock(
        side_effect=RuntimeError(f"{encrypted} {summary}")
    )
    terminal_items, terminal_error = run(
        _consume_with_error(
            OpenAIStream(
                terminal_source,
                replay_owner=terminal_owner,
            )
        )
    )
    assert [item.kind for item in terminal_items] == [
        StreamItemKind.STREAM_STARTED,
        StreamItemKind.STREAM_ERRORED,
        StreamItemKind.STREAM_CLOSED,
    ]
    assert terminal_error is None
    terminal_outward = "\n".join(
        (
            repr(terminal_items),
            repr([item.to_trace_dict() for item in terminal_items]),
        )
    )
    assert encrypted not in terminal_outward
    assert summary not in terminal_outward
    terminal_event = terminal_items[1]
    assert terminal_event.provider_payload is None
    assert cast(dict[str, Any], terminal_event.data)["error"] == {
        "type": "server_error",
        "code": "openai_provider_request_failed",
        "status": "failed",
        "message": "OpenAI provider request failed",
    }
    assert terminal_owner.release_count == 1
    terminal_source.aclose.assert_awaited_once_with()


def test_encrypted_reasoning_is_not_displayed_or_relabelled() -> None:
    stream = OpenAIStream(
        _AsyncEvents(
            [
                _output_done(
                    _reasoning_item(
                        "rs_private",
                        "encrypted-private",
                        [{"type": "summary_text", "text": "hidden-summary"}],
                    )
                ),
                {
                    "type": "response.output_text.delta",
                    "delta": "visible-answer",
                },
                _completed_event(),
            ]
        )
    )
    items = run(_consume(stream))
    accumulated = accumulate_canonical_stream_items(items)

    assert accumulated.answer_text == "visible-answer"
    assert accumulated.reasoning_text == ""
    assert not any(
        item.kind is StreamItemKind.REASONING_DELTA for item in items
    )


def test_pre_output_retry_preserves_request_and_resets_state() -> None:
    failed = _AsyncEvents(
        [
            _output_done(
                _reasoning_item(
                    "rs_failed",
                    "cipher-failed",
                    [{"type": "summary_text", "text": "failed"}],
                )
            ),
            {
                "type": "response.failed",
                # Omitted output preserves the legacy retry-compatible shape;
                # an explicit empty list contradicts the completed item above.
                "response": {"status": "failed", "error": None},
            },
        ]
    )
    recovered = _AsyncEvents(
        [
            {
                "type": "response.reasoning_text.delta",
                "delta": "native",
                "item_id": "rs_native",
                "content_index": 0,
            },
            _completed_event(text="answer"),
        ]
    )
    create = AsyncMock(side_effect=[failed, recovered])
    stream = run(
        _client(create)(
            "plain-model",
            [],
            _settings(
                ReasoningSummaryMode.DETAILED,
                effort=ReasoningEffort.MAX,
            ),
        )
    )
    items = run(_consume(stream))

    assert create.await_count == 2
    assert create.await_args_list[0].kwargs == create.await_args_list[1].kwargs
    assert create.await_args_list[1].kwargs["reasoning"] == {
        "effort": "xhigh",
        "summary": "detailed",
    }
    reasoning = next(
        item for item in items if item.kind is StreamItemKind.REASONING_DELTA
    )
    assert reasoning.segment_instance_ordinal == 0


def test_visible_native_or_summary_reasoning_disables_retry() -> None:
    cases: tuple[tuple[type[OpenAIStream], object], ...] = (
        (
            OpenAIStream,
            {
                "type": "response.reasoning_text.delta",
                "delta": "native",
                "content_index": 0,
            },
        ),
        (
            _InjectedSummaryStream,
            {"type": "test.summary.delta"},
        ),
    )
    for stream_type, reasoning_event in cases:
        retries = AsyncMock(return_value=_AsyncEvents([_completed_event()]))
        stream = stream_type(
            _AsyncEvents(
                [
                    reasoning_event,
                    {
                        "type": "response.failed",
                        "response": {
                            "status": "failed",
                            "error": None,
                            "output": [],
                        },
                    },
                ]
            ),
            stream_factory=retries,
            stream_retries=1,
        )
        items = run(_consume(stream))
        retries.assert_not_awaited()
        assert any(
            item.kind is StreamItemKind.STREAM_ERRORED for item in items
        )

    compound_retry = AsyncMock(return_value=_AsyncEvents([_completed_event()]))
    compound_items = run(
        _consume(
            _CompoundFailureStream(
                _AsyncEvents(
                    [
                        {
                            "type": "response.failed",
                            "response": {
                                "status": "failed",
                                "error": None,
                                "output": [],
                            },
                        }
                    ]
                ),
                stream_factory=compound_retry,
                stream_retries=1,
            )
        )
    )
    compound_retry.assert_not_awaited()
    assert [
        item.kind
        for item in compound_items
        if item.kind
        in {StreamItemKind.REASONING_DELTA, StreamItemKind.STREAM_ERRORED}
    ] == [StreamItemKind.REASONING_DELTA, StreamItemKind.STREAM_ERRORED]


def test_failed_attempt_reasoning_and_function_calls_roll_back_together() -> (
    None
):
    owner = _owner(StreamRetentionPolicy(openai_replay_item_limit=4))
    owner.admit(_reasoning_item("kept", "cipher-kept", []))
    owner.commit_attempt()
    baseline = (owner.replay_items(), owner.counters, owner.generic_counters)
    owner.begin_attempt()
    owner.admit(_reasoning_item("failed", "cipher-failed", []))
    owner.admit(_function_item("failed-call"))

    owner.rollback_attempt()

    assert (
        owner.replay_items(),
        owner.counters,
        owner.generic_counters,
    ) == baseline
    assert all(
        item.get("call_id") != "failed-call" for item in owner.replay_items()
    )

    owner.begin_attempt()
    owner.admit(_reasoning_item("committed", "cipher-committed", []))
    owner.admit(_function_item("committed-call"))
    owner.commit_attempt()
    committed = (
        owner.replay_items(),
        owner.counters,
        owner.generic_counters,
    )
    assert [item["type"] for item in owner.replay_items()] == [
        "reasoning",
        "reasoning",
        "function_call",
    ]

    owner.begin_attempt()
    assert owner.admit(_function_item("exact-call"))
    before_overflow = (
        owner.replay_items(),
        owner.counters,
        owner.generic_counters,
    )
    with pytest.raises(openai_module._ReasoningReplayRetentionError):
        owner.admit(_function_item("overflow-call"))
    assert (
        owner.replay_items(),
        owner.counters,
        owner.generic_counters,
    ) == before_overflow
    owner.rollback_attempt()
    assert (
        owner.replay_items(),
        owner.counters,
        owner.generic_counters,
    ) == committed

    owner.release()
    assert owner.replay_items() == ()
    assert owner.counters == (0, 0, 0, 0)
    assert owner.generic_counters == (0, 0)


def test_summary_delta_emits_immediately() -> None:
    items = run(
        _consume(OpenAIStream(_AsyncEvents(_trace_events("one_part"))))
    )
    reasoning = _reasoning_items(items)

    assert [item.text_delta for item in reasoning] == ["Inspect inputs."]
    assert items.index(reasoning[0]) < next(
        index
        for index, item in enumerate(items)
        if item.kind is StreamItemKind.STREAM_COMPLETED
    )


def test_summary_delta_does_not_wait_for_later_output() -> None:
    async def scenario() -> tuple[CanonicalStreamItem, int, int, float]:
        events = _trace_events("one_part")
        source = _AsyncEvents(events)
        canonical = OpenAIStream(source).canonical_stream(
            stream_session_id="summary-immediate",
            run_id="run-1",
            turn_id="turn-1",
        )
        started = await anext(canonical)
        assert started.kind is StreamItemKind.STREAM_STARTED
        started_at = perf_counter()
        delta = await anext(canonical)
        first_summary_ms = (perf_counter() - started_at) * 1000
        reads_at_delta = source.read_count
        await canonical.aclose()
        return delta, reads_at_delta, len(events), first_summary_ms

    delta, reads_at_delta, event_count, first_summary_ms = run(scenario())

    assert delta.kind is StreamItemKind.REASONING_DELTA
    assert delta.text_delta == "Inspect inputs."
    assert reads_at_delta == 3
    assert reads_at_delta < event_count
    assert first_summary_ms <= StreamPerformanceBudget().time_to_first_item_ms


def test_summary_delta_uses_canonical_reasoning() -> None:
    items = run(
        _consume(OpenAIStream(_AsyncEvents(_trace_events("one_part"))))
    )
    reasoning = _reasoning_items(items)

    assert len(reasoning) == 1
    item = reasoning[0]
    assert (
        item.reasoning_representation is StreamReasoningRepresentation.SUMMARY
    )
    assert item.visibility is StreamVisibility.PRIVATE
    assert item.kind is StreamItemKind.REASONING_DELTA
    assert not any(
        candidate.kind is StreamItemKind.ANSWER_DELTA for candidate in items
    )


def test_summary_delta_preserves_lossless_whitespace_and_identity() -> None:
    expected = ("  lead", "\n", "trail  ")
    items = run(_consume(OpenAIStream(_AsyncEvents(_summary_trace(expected)))))
    reasoning = _reasoning_items(items)

    assert [item.text_delta for item in reasoning] == list(expected)
    assert [item.correlation.protocol_item_id for item in reasoning] == [
        "rs_test"
    ] * 3
    assert [item.correlation.provider_output_index for item in reasoning] == [
        0
    ] * 3
    assert [item.correlation.provider_summary_index for item in reasoning] == [
        0
    ] * 3
    assert [item.provider_event_type for item in reasoning] == [
        "response.reasoning_summary_text.delta"
    ] * 3
    assert [item.segment_instance_ordinal for item in reasoning] == [0, 0, 0]


def test_summary_stream_accepts_dict_and_sdk_event_shapes() -> None:
    raw_events = _trace_events("one_part")
    sdk_events = [
        _RESPONSE_STREAM_EVENT.validate_python(event) for event in raw_events
    ]

    mapped = [
        run(_consume(OpenAIStream(_AsyncEvents(events))))
        for events in (raw_events, sdk_events)
    ]
    traces = [
        [
            (
                item.kind,
                item.text_delta,
                item.reasoning_representation,
                item.correlation.to_trace_dict(),
                item.provider_event_type,
            )
            for item in items
        ]
        for items in mapped
    ]

    assert traces[0] == traces[1]


def test_part_completion_does_not_close_reasoning() -> None:
    items = run(
        _consume(OpenAIStream(_AsyncEvents(_trace_events("multipart"))))
    )
    reasoning = _reasoning_items(items)

    assert [item.text_delta for item in reasoning] == [
        "Check records.",
        "Choose the newest.",
    ]
    first_delta_index = items.index(reasoning[0])
    second_delta_index = items.index(reasoning[1])
    done_indices = [
        index
        for index, item in enumerate(items)
        if item.kind is StreamItemKind.REASONING_DONE
    ]
    assert len(done_indices) == 1
    assert first_delta_index < second_delta_index < done_indices[0]


def test_reasoning_item_has_at_most_one_response_done() -> None:
    events = _trace_events("one_part")
    events.insert(-1, deepcopy(events[-2]))
    items = run(_consume(OpenAIStream(_AsyncEvents(events))))

    assert (
        sum(item.kind is StreamItemKind.REASONING_DONE for item in items) == 1
    )
    assert [item.text_delta for item in _reasoning_items(items)] == [
        "Inspect inputs."
    ]


def test_multipart_summary_preserves_emission_order() -> None:
    items = run(
        _consume(OpenAIStream(_AsyncEvents(_trace_events("multipart"))))
    )
    reasoning = _reasoning_items(items)

    assert [item.text_delta for item in reasoning] == [
        "Check records.",
        "Choose the newest.",
    ]
    assert [item.correlation.provider_summary_index for item in reasoning] == [
        0,
        1,
    ]
    assert [item.segment_instance_ordinal for item in reasoning] == [0, 1]
    assert reasoning[1].metadata == {
        REASONING_SEGMENT_BOUNDARY_METADATA_KEY: "completed"
    }


def test_summary_index_is_preserved() -> None:
    items = run(
        _consume(OpenAIStream(_AsyncEvents(_trace_events("sparse_indices"))))
    )
    reasoning = _reasoning_items(items)

    assert [item.correlation.provider_output_index for item in reasoning] == [
        4
    ] * 8
    assert [item.correlation.provider_summary_index for item in reasoning] == [
        2,
        7,
        0,
        1,
        3,
        4,
        5,
        6,
    ]


def test_sparse_summary_indices_preserve_live_then_fallback_order() -> None:
    items = run(
        _consume(OpenAIStream(_AsyncEvents(_trace_events("sparse_indices"))))
    )
    reasoning = _reasoning_items(items)

    assert [item.text_delta for item in reasoning] == [
        "Earlier sparse part.",
        "Later sparse part.",
        "Fallback zero.",
        "Fallback one.",
        "Fallback three.",
        "Fallback four.",
        "Fallback five.",
        "Fallback six.",
    ]
    assert [item.segment_instance_ordinal for item in reasoning] == list(
        range(8)
    )


def test_empty_summary_is_invisible() -> None:
    items = run(_consume(OpenAIStream(_AsyncEvents(_trace_events("empty")))))

    assert _reasoning_items(items) == []
    assert not any(
        item.kind is StreamItemKind.REASONING_DONE for item in items
    )
    assert [item.kind for item in items] == [
        StreamItemKind.STREAM_STARTED,
        StreamItemKind.STREAM_COMPLETED,
        StreamItemKind.STREAM_CLOSED,
    ]


def test_empty_summary_preserves_tools_and_answer() -> None:
    empty_prefix = _trace_events("empty")[:2]
    tool_events = _trace_events("tools_answer", response_index=0)[6:]
    answer_events = _trace_events("tools_answer", response_index=1)[6:]

    tool_items = run(
        _consume(OpenAIStream(_AsyncEvents([*empty_prefix, *tool_events])))
    )
    answer_items = run(
        _consume(OpenAIStream(_AsyncEvents([*empty_prefix, *answer_events])))
    )

    assert _reasoning_items(tool_items) == []
    assert _reasoning_items(answer_items) == []
    assert (
        sum(item.kind is StreamItemKind.TOOL_CALL_DONE for item in tool_items)
        == 1
    )
    assert accumulate_canonical_stream_items(answer_items).answer_text


def test_empty_summary_is_operationally_invisible() -> None:
    empty_prefix = _trace_events("empty")[:2]
    answer_tail = [
        {"type": "response.output_text.delta", "delta": "answer"},
        _completed_event(),
    ]
    with_empty = run(
        _consume(OpenAIStream(_AsyncEvents([*empty_prefix, *answer_tail])))
    )
    without_empty = run(
        _consume(OpenAIStream(_AsyncEvents(deepcopy(answer_tail))))
    )

    assert [item.kind for item in with_empty] == [
        item.kind for item in without_empty
    ]
    assert (
        accumulate_canonical_stream_items(with_empty).answer_text == "answer"
    )


def test_completed_item_fallback_is_deduplicated_per_part() -> None:
    cases = {
        "fallback": ["Recovered from the completed item."],
        "mixed_fallback": ["Streamed part.", "Fallback-only part."],
    }
    for fixture_name, expected in cases.items():
        items = run(
            _consume(OpenAIStream(_AsyncEvents(_trace_events(fixture_name))))
        )
        assert [
            item.text_delta for item in _reasoning_items(items)
        ] == expected


def test_zero_length_delta_remains_fallback_eligible() -> None:
    items = run(
        _consume(
            OpenAIStream(_AsyncEvents(_trace_events("zero_length_fallback")))
        )
    )
    reasoning = _reasoning_items(items)

    assert [item.text_delta for item in reasoning] == [
        "Recovered after empty delta."
    ]
    assert reasoning[0].provider_event_type == "response.output_item.done"
    assert reasoning[0].segment_instance_ordinal == 0


def test_mixed_streamed_and_fallback_parts_emit_once() -> None:
    items = run(
        _consume(OpenAIStream(_AsyncEvents(_trace_events("mixed_fallback"))))
    )
    reasoning = _reasoning_items(items)

    assert [item.text_delta for item in reasoning] == [
        "Streamed part.",
        "Fallback-only part.",
    ]
    assert [item.provider_event_type for item in reasoning] == [
        "response.reasoning_summary_text.delta",
        "response.output_item.done",
    ]


def test_summary_completion_duplicates_are_idempotent() -> None:
    base = _summary_trace(("same",))
    duplicate_events = [
        *base[:4],
        deepcopy(base[3]),
        base[4],
        deepcopy(base[4]),
        base[5],
        deepcopy(base[5]),
        base[6],
    ]
    duplicate_items = run(
        _consume(OpenAIStream(_AsyncEvents(duplicate_events)))
    )
    assert [item.text_delta for item in _reasoning_items(duplicate_items)] == [
        "same"
    ]
    assert (
        sum(
            item.kind is StreamItemKind.REASONING_DONE
            for item in duplicate_items
        )
        == 1
    )

    conflict_cases = []
    text_conflict = deepcopy(base[:4])
    text_conflict.append(_summary_text_done("different"))
    conflict_cases.append(text_conflict)
    part_conflict = deepcopy(base[:5])
    part_conflict.append(_summary_part_done("different"))
    conflict_cases.append(part_conflict)
    item_conflict = deepcopy(base[:6])
    item_conflict.append(_summary_item_done(["different"]))
    conflict_cases.append(item_conflict)

    for events in conflict_cases:
        items = run(_consume(OpenAIStream(_AsyncEvents(events))))
        terminal = _error_item(items)
        assert terminal.provider_payload is None
        assert (
            cast(dict[str, Any], terminal.data)["error"]["code"]
            == "invalid_reasoning_summary_event"
        )


def test_reasoning_output_item_done_never_emits_answer_text() -> None:
    content_sentinel = "must-not-be-answer-content"
    summary_sentinel = "must-not-be-answer-summary"
    events = [
        _summary_item_added(),
        _summary_item_done(
            [summary_sentinel],
            content=[{"type": "output_text", "text": content_sentinel}],
        ),
        _completed_event(),
    ]
    items = run(_consume(OpenAIStream(_AsyncEvents(events))))

    assert [item.text_delta for item in _reasoning_items(items)] == [
        summary_sentinel
    ]
    assert not any(item.kind is StreamItemKind.ANSWER_DELTA for item in items)
    assert (
        content_sentinel
        not in accumulate_canonical_stream_items(items).answer_text
    )


def test_reasoning_output_item_done_records_replay_exactly_once() -> None:
    events = _summary_trace(("record once",))
    events.insert(-1, deepcopy(events[-2]))
    recorded: list[dict[str, Any]] = []
    items = run(
        _consume(
            OpenAIStream(
                _AsyncEvents(events),
                output_item_sink=recorded.append,
            )
        )
    )

    assert len(recorded) == 1
    assert recorded[0]["id"] == "rs_test"
    assert recorded[0]["summary"] == [
        {"type": "summary_text", "text": "record once"}
    ]
    assert [item.text_delta for item in _reasoning_items(items)] == [
        "record once"
    ]


def test_summary_interoperates_with_one_and_multiple_tool_calls() -> None:
    source_events = _trace_events("tools_answer", response_index=0)
    summary_prefix = source_events[:6]
    tool_group = source_events[6:10]
    terminal = source_events[-1]

    for tool_count in (1, 2):
        events = deepcopy(summary_prefix)
        events.extend(deepcopy(tool_group))
        if tool_count == 2:
            second_group = deepcopy(tool_group)
            for event in second_group:
                assert isinstance(event, dict)
                event["output_index"] = 2
                if event.get("item_id") == "fc_tool":
                    event["item_id"] = "fc_tool_2"
                item = event.get("item")
                if isinstance(item, dict):
                    item["id"] = "fc_tool_2"
                    item["call_id"] = "call-tool-2"
            events.extend(second_group)
        events.append(deepcopy(terminal))
        items = run(_consume(OpenAIStream(_AsyncEvents(events))))
        assert [item.text_delta for item in _reasoning_items(items)] == [
            "Use the lookup tool."
        ]
        assert (
            sum(item.kind is StreamItemKind.TOOL_CALL_DONE for item in items)
            == tool_count
        )


def test_summary_followed_by_final_answer_keeps_channels_separate() -> None:
    items = run(
        _consume(
            OpenAIStream(
                _AsyncEvents(_trace_events("tools_answer", response_index=1))
            )
        )
    )

    assert [item.text_delta for item in _reasoning_items(items)] == [
        "Use the tool result."
    ]
    accumulator = accumulate_canonical_stream_items(items)
    assert accumulator.answer_text == '{"ok":true}'
    assert "Use the tool result." not in accumulator.answer_text


def test_native_only_and_mixed_reasoning_preserve_representations() -> None:
    native_events = [
        {
            "type": "response.reasoning_text.delta",
            "delta": "native",
            "content_index": 0,
        },
        {"type": "response.reasoning_text.done", "content_index": 0},
        _completed_event(),
    ]
    summary_prefix = _summary_trace(("summary",))[:-1]
    mixed_events = [*summary_prefix, *deepcopy(native_events)]

    native_items = run(
        _consume(OpenAIStream(_AsyncEvents(deepcopy(native_events))))
    )
    mixed_items = run(_consume(OpenAIStream(_AsyncEvents(mixed_events))))

    assert [
        item.reasoning_representation
        for item in _reasoning_items(native_items)
    ] == [StreamReasoningRepresentation.NATIVE_TEXT]
    mixed_reasoning = _reasoning_items(mixed_items)
    assert [item.reasoning_representation for item in mixed_reasoning] == [
        StreamReasoningRepresentation.SUMMARY,
        StreamReasoningRepresentation.NATIVE_TEXT,
    ]
    assert [item.segment_instance_ordinal for item in mixed_reasoning] == [
        0,
        1,
    ]


def test_multiple_summary_continuations_preserve_response_local_identity() -> (
    None
):
    continuations = [
        run(
            _consume(
                OpenAIStream(
                    _AsyncEvents(
                        _trace_events(
                            "multi_continuation", response_index=index
                        )
                    )
                )
            )
        )
        for index in range(3)
    ]
    reasoning = [_reasoning_items(items) for items in continuations]

    assert all(len(items) == 1 for items in reasoning)
    assert [items[0].segment_instance_ordinal for items in reasoning] == [
        0,
        0,
        0,
    ]
    assert [items[0].correlation.protocol_item_id for items in reasoning] == [
        "rs_1",
        "rs_1",
        "rs_3",
    ]


def test_visible_summary_disables_retry() -> None:
    events = [
        _summary_item_added(),
        _summary_part_added(),
        _summary_delta("visible"),
        {
            "type": "response.failed",
            "response": {"status": "failed", "error": None, "output": []},
        },
    ]
    retry = AsyncMock(return_value=_AsyncEvents([_completed_event()]))
    items = run(
        _consume(
            OpenAIStream(
                _AsyncEvents(events),
                stream_factory=retry,
                stream_retries=1,
            )
        )
    )

    retry.assert_not_awaited()
    assert [item.text_delta for item in _reasoning_items(items)] == ["visible"]
    assert [
        item.kind
        for item in items
        if item.kind
        in {StreamItemKind.REASONING_DONE, StreamItemKind.STREAM_ERRORED}
    ] == [StreamItemKind.REASONING_DONE, StreamItemKind.STREAM_ERRORED]


def test_previsible_retry_resets_multipart_state() -> None:
    failed = _AsyncEvents(
        [
            _summary_item_added(),
            _summary_part_added(),
            _summary_delta(""),
            {
                "type": "response.failed",
                "response": {
                    "status": "failed",
                    "error": None,
                    "output": [],
                },
            },
        ]
    )
    recovered = _AsyncEvents(_summary_trace(("recovered",)))
    retry = AsyncMock(return_value=recovered)
    items = run(
        _consume(
            OpenAIStream(
                failed,
                stream_factory=retry,
                stream_retries=1,
            )
        )
    )

    retry.assert_awaited_once_with()
    reasoning = _reasoning_items(items)
    assert [item.text_delta for item in reasoning] == ["recovered"]
    assert reasoning[0].segment_instance_ordinal == 0


def test_summary_negative_transition_matrix_emits_one_structured_error() -> (
    None
):
    fixture = _phase4_negative_fixture()
    rows = fixture["transition_rows"]
    assert isinstance(rows, list) and len(rows) == 9
    for row in rows:
        assert isinstance(row, dict)
        prefix = deepcopy(row["prefix"])
        assert isinstance(prefix, list)
        event = deepcopy(row["event"])
        assert isinstance(event, dict)
        items = run(_consume(OpenAIStream(_AsyncEvents([*prefix, event]))))
        _assert_structured_summary_error(
            items,
            event_type=cast(str, event["type"]),
            field=cast(str, row["field"]),
            value_shape=cast(str, row["value_shape"]),
            output_index=(
                event.get("output_index")
                if type(event.get("output_index")) is int
                else None
            ),
            summary_index=(
                event.get("summary_index")
                if type(event.get("summary_index")) is int
                else None
            ),
        )
        visible = _reasoning_items(items)
        assert sum(
            item.kind is StreamItemKind.REASONING_DONE for item in items
        ) == (1 if visible else 0)

    terminal_row = fixture["response_terminal_row"]
    assert isinstance(terminal_row, dict)
    terminal_event = cast(dict[str, object], terminal_row["event"])
    terminal_items = run(
        _consume(
            OpenAIStream(
                _AsyncEvents(
                    [
                        *deepcopy(cast(list[object], terminal_row["prefix"])),
                        deepcopy(terminal_event),
                    ]
                )
            )
        )
    )
    _assert_structured_summary_error(
        terminal_items,
        event_type=cast(str, terminal_event["type"]),
        field=cast(str, terminal_row["field"]),
        value_shape=cast(str, terminal_row["value_shape"]),
        output_index=0,
        summary_index=None,
    )
    assert [item.text_delta for item in _reasoning_items(terminal_items)] == [
        "visible-prefix"
    ]


def test_summary_index_type_matrix_emits_one_structured_error() -> None:
    fixture = _phase4_negative_fixture()
    targets = fixture["index_targets"]
    invalid_indices = fixture["invalid_indices"]
    assert isinstance(targets, list) and len(targets) == 6
    assert isinstance(invalid_indices, list) and len(invalid_indices) == 5
    executed = 0

    for target in targets:
        assert isinstance(target, dict)
        event_type = cast(str, target["event_type"])
        fields = target["fields"]
        assert isinstance(fields, list)
        for field_name in fields:
            assert isinstance(field_name, str)
            for invalid in invalid_indices:
                assert isinstance(invalid, dict)
                prefix, event = _index_case_events(event_type)
                event[field_name] = _fixture_runtime_value(invalid)
                items = run(
                    _consume(OpenAIStream(_AsyncEvents([*prefix, event])))
                )
                _assert_structured_summary_error(
                    items,
                    event_type=event_type,
                    field=field_name,
                    value_shape=cast(str, invalid["shape"]),
                    output_index=(None if field_name == "output_index" else 0),
                    summary_index=(
                        None
                        if field_name == "summary_index"
                        or event_type.startswith("response.output_item")
                        else 0
                    ),
                )
                executed += 1

    assert executed == 50


def test_summary_rejects_non_string_delta_with_safe_diagnostics() -> None:
    fixture = _phase4_negative_fixture()
    invalid_deltas = fixture["invalid_deltas"]
    assert isinstance(invalid_deltas, list) and len(invalid_deltas) == 7
    _CoercionTrap.calls = 0

    for invalid in invalid_deltas:
        assert isinstance(invalid, dict)
        events = [
            _summary_item_added("rs_bad"),
            _summary_part_added("rs_bad"),
            _summary_delta(_fixture_runtime_value(invalid), "rs_bad"),
        ]
        items = run(_consume(OpenAIStream(_AsyncEvents(events))))
        _assert_structured_summary_error(
            items,
            event_type="response.reasoning_summary_text.delta",
            field="delta",
            value_shape=cast(str, invalid["shape"]),
        )
    assert _CoercionTrap.calls == 0


def test_summary_identity_conflicts_are_content_free() -> None:
    secret = "identity-private-sentinel"
    cases: list[tuple[list[object], str, str]] = []
    missing = _summary_part_added()
    del missing["item_id"]
    cases.append(([_summary_item_added(), missing], "item_id", "missing"))
    cases.append(
        (
            [_summary_item_added(), _summary_part_added("")],
            "item_id",
            "empty_string",
        )
    )
    cases.append(
        (
            [_summary_item_added(), _summary_part_added(secret)],
            "item.identity",
            "conflict",
        )
    )

    for events, field_name, value_shape in cases:
        items = run(_consume(OpenAIStream(_AsyncEvents(events))))
        terminal = _assert_structured_summary_error(
            items,
            event_type="response.reasoning_summary_part.added",
            field=field_name,
            value_shape=value_shape,
        )
        assert secret not in repr(terminal)
        assert secret not in repr(terminal.to_trace_dict())


def test_reasoning_item_cannot_flip_to_message_after_completion() -> None:
    secret = "REASONING_TO_MESSAGE_PRIVATE_SENTINEL"
    items = _assert_output_item_conflict(
        [
            _summary_item_added("rs_flip", 0),
            _summary_item_done([], "rs_flip", 0),
            _message_item_event(
                "response.output_item.done",
                "rs_flip",
                0,
                secret,
            ),
        ],
        field="item.type",
        output_index=0,
        secret=secret,
    )
    assert _reasoning_items(items) == []


def test_message_item_cannot_flip_to_reasoning() -> None:
    secret = "MESSAGE_TO_REASONING_PRIVATE_SENTINEL"
    _assert_output_item_conflict(
        [
            _message_item_event(
                "response.output_item.added",
                "message-flip",
                0,
                secret,
            ),
            _summary_item_done([], "message-flip", 0),
        ],
        field="item.type",
        output_index=0,
        secret=secret,
    )


def test_reasoning_item_cannot_flip_to_tool_call() -> None:
    secret = "REASONING_TO_TOOL_PRIVATE_SENTINEL"
    _assert_output_item_conflict(
        [
            _summary_item_added("reasoning-tool-flip", 0),
            _summary_item_done([], "reasoning-tool-flip", 0),
            _tool_item_done("reasoning-tool-flip", 0, secret),
        ],
        field="item.type",
        output_index=0,
        secret=secret,
    )


def test_output_item_id_cannot_move_to_another_output_index() -> None:
    secret = "SAME_ID_DIFFERENT_INDEX_PRIVATE_SENTINEL"
    _assert_output_item_conflict(
        [
            _summary_item_added("stable-id", 0),
            _message_item_event(
                "response.output_item.done",
                "stable-id",
                1,
                secret,
            ),
        ],
        field="item.identity",
        output_index=1,
        secret=secret,
    )


def test_output_index_cannot_be_reassigned_to_another_item_id() -> None:
    secret = "SAME_INDEX_DIFFERENT_ID_PRIVATE_SENTINEL"
    _assert_output_item_conflict(
        [
            _summary_item_added("stable-id", 0),
            _message_item_event(
                "response.output_item.done",
                secret,
                0,
                "must-not-emit",
            ),
        ],
        field="item.identity",
        output_index=0,
        secret=secret,
    )


def test_output_item_identity_lock_allows_distinct_message_and_tool() -> None:
    items = run(
        _consume(
            OpenAIStream(
                _AsyncEvents(
                    [
                        _summary_item_added("reasoning-safe", 0),
                        _summary_item_done([], "reasoning-safe", 0),
                        _message_item_event(
                            "response.output_item.done",
                            "message-safe",
                            1,
                            "answer-safe",
                        ),
                        _tool_item_done("tool-safe", 2),
                        _completed_event(),
                    ]
                )
            )
        )
    )
    accumulator = accumulate_canonical_stream_items(items)

    assert accumulator.answer_text == "answer-safe"
    assert accumulator.tool_call_arguments == {"call-safe": '{"value":1}'}
    assert any(item.kind is StreamItemKind.TOOL_CALL_DONE for item in items)


def test_event_type_string_subclass_is_rejected_without_method_calls() -> None:
    secret = "EVENT_TYPE_STRING_SUBCLASS_PRIVATE_SENTINEL"

    class HostileString(str):
        calls = 0

        def _fail(self, *_: object, **__: object) -> object:
            HostileString.calls += 1
            raise RuntimeError(secret)

        __eq__ = _fail
        __hash__ = _fail
        encode = _fail
        strip = _fail

    items = run(
        _consume(
            OpenAIStream(
                _AsyncEvents([{"type": HostileString("response.completed")}])
            )
        )
    )
    terminal = _assert_structured_summary_error(
        items,
        event_type="response.unknown",
        field="type",
        value_shape="sequence",
        output_index=None,
        summary_index=None,
    )

    assert HostileString.calls == 0
    assert secret not in repr([terminal, terminal.to_trace_dict()])


def test_sdk_and_dict_item_id_string_subclasses_are_content_free() -> None:
    secret = "ITEM_ID_STRING_SUBCLASS_PRIVATE_SENTINEL"

    class HostileString(str):
        calls = 0

        def _fail(self, *_: object, **__: object) -> object:
            HostileString.calls += 1
            raise RuntimeError(secret)

        __eq__ = _fail
        __hash__ = _fail
        encode = _fail
        strip = _fail

    hostile_id = HostileString("hostile-item-id")
    cases = (
        (
            _summary_item_added(cast(str, hostile_id), 0),
            "response.output_item.added",
        ),
        (
            SimpleNamespace(
                type="response.output_item.done",
                output_index=0,
                item=SimpleNamespace(
                    id=hostile_id,
                    type="message",
                    status="completed",
                    content=[],
                ),
            ),
            "response.output_item.done",
        ),
    )

    for event, event_type in cases:
        items = run(_consume(OpenAIStream(_AsyncEvents([event]))))
        terminal = _assert_structured_summary_error(
            items,
            event_type=event_type,
            field="item.id",
            value_shape="sequence",
            output_index=0,
            summary_index=None,
        )
        assert secret not in repr([terminal, terminal.to_trace_dict()])
    assert HostileString.calls == 0


def test_summary_text_string_subclasses_are_rejected_without_calls() -> None:
    secret = "SUMMARY_TEXT_STRING_SUBCLASS_PRIVATE_SENTINEL"

    class HostileString(str):
        calls = 0

        def _fail(self, *_: object, **__: object) -> object:
            HostileString.calls += 1
            raise RuntimeError(secret)

        __eq__ = _fail
        __hash__ = _fail
        encode = _fail
        strip = _fail

    cases = (
        (
            [
                _summary_item_added(),
                _summary_part_added(),
                _summary_delta(HostileString("delta")),
            ],
            "response.reasoning_summary_text.delta",
            "delta",
        ),
        (
            [
                _summary_item_added(),
                _summary_part_added(),
                _summary_delta("text"),
                _summary_text_done(HostileString("text")),
            ],
            "response.reasoning_summary_text.done",
            "text",
        ),
    )

    for events, event_type, field in cases:
        items = run(_consume(OpenAIStream(_AsyncEvents(events))))
        terminal = _assert_structured_summary_error(
            items,
            event_type=event_type,
            field=field,
            value_shape="sequence",
        )
        assert secret not in repr([terminal, terminal.to_trace_dict()])
    assert HostileString.calls == 0


def test_duplicate_item_events_require_exact_builtin_type() -> None:
    secret = "DUPLICATE_ITEM_TYPE_PRIVATE_SENTINEL"

    class HostileString(str):
        calls = 0

        def _fail(self, *_: object, **__: object) -> object:
            HostileString.calls += 1
            raise RuntimeError(secret)

        __eq__ = _fail
        __hash__ = _fail
        __str__ = _fail
        encode = _fail
        strip = _fail

    invalid_types: tuple[tuple[str, object, str], ...] = (
        ("missing", object(), "missing"),
        ("integer", 7, "integer"),
        ("boolean", False, "boolean"),
        ("subclass", HostileString("reasoning"), "sequence"),
        ("mismatch", "message", "conflict"),
    )
    for event_type in (
        "response.output_item.added",
        "response.output_item.done",
    ):
        for label, invalid_type, value_shape in invalid_types:
            event = (
                _summary_item_added("type-locked", 0)
                if event_type == "response.output_item.added"
                else _summary_item_done(
                    [secret],
                    "type-locked",
                    0,
                )
            )
            item = cast(dict[str, object], event["item"])
            if label == "missing":
                item.pop("type")
            else:
                item["type"] = invalid_type
            items = run(
                _consume(
                    OpenAIStream(
                        _AsyncEvents(
                            [
                                _summary_item_added("type-locked", 0),
                                event,
                            ]
                        )
                    )
                )
            )
            terminal = _assert_structured_summary_error(
                items,
                event_type=event_type,
                field="item.type",
                value_shape=value_shape,
                output_index=0,
                summary_index=None,
            )
            assert not _reasoning_items(items)
            assert secret not in repr([terminal, terminal.to_trace_dict()])
    assert HostileString.calls == 0


def test_item_type_subclass_is_rejected_before_payload_sanitization() -> None:
    secret = "ITEM_TYPE_SANITIZER_PRIVATE_SENTINEL"

    class HostileString(str):
        calls = 0

        def _fail(self, *_: object, **__: object) -> object:
            HostileString.calls += 1
            raise RuntimeError(secret)

        __eq__ = _fail
        __hash__ = _fail
        __str__ = _fail
        encode = _fail
        strip = _fail

    hostile_type = HostileString("reasoning")
    mapping_event = _summary_item_added("hostile-type", 0)
    cast(dict[str, object], mapping_event["item"])["type"] = hostile_type
    sdk_event = SimpleNamespace(
        type="response.output_item.added",
        output_index=0,
        item=SimpleNamespace(
            id="hostile-type-sdk",
            type=hostile_type,
            status="in_progress",
            summary=[],
        ),
    )

    for event in (mapping_event, sdk_event):
        items = run(_consume(OpenAIStream(_AsyncEvents([event]))))
        terminal = _assert_structured_summary_error(
            items,
            event_type="response.output_item.added",
            field="item.type",
            value_shape="sequence",
            output_index=0,
            summary_index=None,
        )
        assert terminal.provider_payload is None
        assert secret not in repr([terminal, terminal.to_trace_dict()])
    assert HostileString.calls == 0


def test_non_reasoning_identities_reject_every_summary_event() -> None:
    secret = "NON_REASONING_SUMMARY_EVENT_PRIVATE_SENTINEL"
    prefixes = (
        _message_item_event(
            "response.output_item.added",
            "message-summary-target",
            0,
        ),
        _typed_item_added(
            "function_call",
            "tool-summary-target",
            0,
        ),
    )
    for prefix in prefixes:
        item_id = cast(dict[str, object], prefix["item"])["id"]
        assert type(item_id) is str
        for event_type, event in _summary_event_cases(secret, item_id):
            items = run(
                _consume(OpenAIStream(_AsyncEvents([deepcopy(prefix), event])))
            )
            terminal = _assert_structured_summary_error(
                items,
                event_type=event_type,
                field="item.type",
                value_shape="unexpected_value",
                output_index=0,
                summary_index=0,
            )
            assert secret not in repr([terminal, terminal.to_trace_dict()])


def test_stateless_reasoning_identity_rejects_every_summary_event() -> None:
    secret = "STATELESS_REASONING_SUMMARY_EVENT_PRIVATE_SENTINEL"
    item_id = "stateless-reasoning"
    for event_type, event in _summary_event_cases(secret, item_id):
        items = run(
            _consume(
                OpenAIStream(
                    _AsyncEvents(
                        [
                            _summary_item_done([], item_id, 0),
                            event,
                        ]
                    )
                )
            )
        )
        terminal = _assert_structured_summary_error(
            items,
            event_type=event_type,
            field="item",
            value_shape="closed",
            output_index=0,
            summary_index=0,
        )
        assert secret not in repr([terminal, terminal.to_trace_dict()])


def test_safe_adapter_error_does_not_stringify_wrapped_error() -> None:
    secret = "SAFE_ADAPTER_ERROR_STRING_PRIVATE_SENTINEL"

    class HostileError(Exception):
        calls = 0

        def __str__(self) -> str:
            HostileError.calls += 1
            raise RuntimeError(secret)

    class SafeAdapterErrorStream(OpenAIStream):
        def _provider_events_from_event(
            self,
            event: object,
        ) -> tuple[StreamProviderEvent, ...]:
            raise StreamProviderAdapterError(
                HostileError(),
                provider_event_type="response.safe_hostile",
                safe_data={"error": {"code": "safe_hostile_error"}},
            )

    items = run(_consume(SafeAdapterErrorStream(_AsyncEvents([object()]))))
    terminal = _error_item(items)

    assert terminal.data == {"error": {"code": "safe_hostile_error"}}
    assert terminal.provider_event_type == "response.safe_hostile"
    assert HostileError.calls == 0
    assert secret not in repr([terminal, terminal.to_trace_dict()])
    assert (
        str(StreamProviderAdapterError(ValueError("legacy-error")))
        == "legacy-error"
    )


def test_standalone_reasoning_done_latches_private_output_without_replay() -> (
    None
):
    secret = "STANDALONE_REASONING_FAILURE_PRIVATE_SENTINEL"
    reasoning_done = _summary_item_done([secret], "standalone-private", 0)
    failure = {
        "type": "response.failed",
        "error": {"code": "response_failed", "message": secret},
        "response": {
            "status": "failed",
            "error": {"code": "response_failed", "message": secret},
            "output": [cast(dict[str, object], reasoning_done["item"])],
        },
    }

    for sink_enabled in (False, True):
        retained: list[dict[str, Any]] = []
        stream = OpenAIStream(
            _AsyncEvents([deepcopy(reasoning_done), deepcopy(failure)]),
            output_item_sink=retained.append if sink_enabled else None,
        )
        items = run(_consume(stream))
        terminal = _error_item(items)
        error = cast(dict[str, Any], terminal.data)["error"]

        assert error["code"] == "openai_provider_request_failed"
        assert terminal.provider_payload is None
        assert secret not in repr([item.to_trace_dict() for item in items])
        assert len(retained) == int(sink_enabled)


def test_malformed_reasoning_added_latches_before_cleanup() -> None:
    secret = "MALFORMED_REASONING_CLEANUP_PRIVATE_SENTINEL"

    class CleanupFailingEvents(_AsyncEvents):
        async def aclose(self) -> None:
            self.close_count += 1
            raise RuntimeError(secret)

    malformed = _summary_item_added("malformed-private", 0)
    cast(dict[str, object], malformed["item"])["status"] = "invalid"
    source = CleanupFailingEvents([malformed])
    items = run(_consume(OpenAIStream(source)))
    terminal = _assert_structured_summary_error(
        items,
        event_type="response.output_item.added",
        field="item.status",
        value_shape="unexpected_value",
        output_index=0,
        summary_index=None,
    )

    assert source.close_count == 1
    assert secret not in repr([terminal, terminal.to_trace_dict()])


def test_duplicate_standalone_reasoning_done_records_once() -> None:
    reasoning_item = _reasoning_item(
        "standalone-duplicate",
        "standalone-cipher",
        [{"type": "summary_text", "text": "standalone-summary"}],
    )
    reasoning_item["status"] = None
    event = _output_done(reasoning_item)
    for sink_enabled, owner_enabled in (
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ):
        retained: list[dict[str, Any]] = []
        owner = _CountingReplayOwner() if owner_enabled else None
        if owner is not None:
            owner.begin_attempt()
        stream = OpenAIStream(
            _AsyncEvents(
                [
                    deepcopy(event),
                    deepcopy(event),
                    _completed_event(),
                ]
            ),
            output_item_sink=retained.append if sink_enabled else None,
            replay_owner=owner,
        )
        items = run(_consume(stream))

        assert not any(
            item.kind is StreamItemKind.STREAM_ERRORED for item in items
        )
        if owner is not None:
            assert owner.admit_calls == 1
        assert len(retained) == int(sink_enabled)
        if retained:
            assert "status" not in retained[0]


def test_conflicting_standalone_reasoning_done_is_content_free() -> None:
    secret = "STANDALONE_REASONING_CONFLICT_PRIVATE_SENTINEL"
    first = _reasoning_item(
        "standalone-conflict",
        "first-cipher",
        [{"type": "summary_text", "text": "first-summary"}],
    )
    second = _reasoning_item(
        "standalone-conflict",
        secret,
        [{"type": "summary_text", "text": secret}],
    )
    first["status"] = None
    second["status"] = None
    for sink_enabled, owner_enabled in (
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ):
        retained: list[dict[str, Any]] = []
        owner = _CountingReplayOwner() if owner_enabled else None
        if owner is not None:
            owner.begin_attempt()
        stream = OpenAIStream(
            _AsyncEvents(
                [
                    {
                        **_output_done(deepcopy(first)),
                        "output_index": 0,
                    },
                    {
                        **_output_done(deepcopy(second)),
                        "output_index": 0,
                    },
                ]
            ),
            output_item_sink=retained.append if sink_enabled else None,
            replay_owner=owner,
        )
        items = run(_consume(stream))
        terminal = _assert_structured_summary_error(
            items,
            event_type="response.output_item.done",
            field="item",
            value_shape="conflict",
            output_index=0,
            summary_index=None,
        )

        if owner is not None:
            assert owner.admit_calls == 1
        assert len(retained) == int(sink_enabled)
        assert secret not in repr([terminal, terminal.to_trace_dict()])


def test_standalone_reasoning_done_closes_identity_to_late_events() -> None:
    secret = "STANDALONE_LATE_EVENT_PRIVATE_SENTINEL"
    item_id = "standalone-closed"
    added = _summary_item_added(item_id, 0)
    cast(dict[str, object], added["item"])["summary"] = [
        {"type": "summary_text", "text": secret}
    ]
    part = _summary_part_added(item_id, 0, 0)
    cast(dict[str, object], part["part"])["text"] = secret
    late_events = (
        ("response.output_item.added", added, None),
        ("response.reasoning_summary_part.added", part, 0),
        (
            "response.reasoning_summary_text.delta",
            _summary_delta(secret, item_id, 0, 0),
            0,
        ),
    )
    for sink_enabled, owner_enabled in (
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ):
        for event_type, late_event, summary_index in late_events:
            retained: list[dict[str, Any]] = []
            owner = _CountingReplayOwner() if owner_enabled else None
            if owner is not None:
                owner.begin_attempt()
            stream = OpenAIStream(
                _AsyncEvents(
                    [
                        _summary_item_done(["first"], item_id, 0),
                        deepcopy(late_event),
                    ]
                ),
                output_item_sink=retained.append if sink_enabled else None,
                replay_owner=owner,
            )
            items = run(_consume(stream))
            terminal = _assert_structured_summary_error(
                items,
                event_type=event_type,
                field="item",
                value_shape="closed",
                output_index=0,
                summary_index=summary_index,
            )

            if owner is not None:
                assert owner.admit_calls == 1
            assert len(retained) == int(sink_enabled)
            assert secret not in repr([terminal, terminal.to_trace_dict()])


def test_terminal_snapshot_cannot_relabel_reasoning_identity() -> None:
    secret = "TERMINAL_SNAPSHOT_PRIVATE_SENTINEL"

    class HostileString(str):
        calls = 0

        def _fail(self, *_: object, **__: object) -> object:
            HostileString.calls += 1
            raise RuntimeError(secret)

        __eq__ = _fail
        __hash__ = _fail
        __str__ = _fail
        encode = _fail
        strip = _fail

    def output_item(
        item_id: str,
        item_type: object = "reasoning",
    ) -> dict[str, object]:
        return {
            "id": item_id,
            "type": item_type,
            "content": [{"type": "output_text", "text": secret}],
            "summary": [],
        }

    missing_type = output_item("terminal-lock")
    missing_type.pop("type")
    cases = (
        (
            0,
            [output_item("terminal-lock", "message")],
            "item.type",
            "conflict",
            0,
        ),
        (0, [missing_type], "item.type", "missing", 0),
        (
            0,
            [output_item("terminal-lock", HostileString("message"))],
            "item.type",
            "sequence",
            0,
        ),
        (
            1,
            [output_item("terminal-lock")],
            "item.identity",
            "conflict",
            0,
        ),
        (
            0,
            [output_item("different-id")],
            "item.identity",
            "conflict",
            0,
        ),
    )
    for (
        locked_output_index,
        output,
        field,
        value_shape,
        error_output_index,
    ) in cases:
        completed = {
            "type": "response.completed",
            "response": {"usage": {}, "output": output},
        }
        items = run(
            _consume(
                OpenAIStream(
                    _AsyncEvents(
                        [
                            _summary_item_added(
                                "terminal-lock", locked_output_index
                            ),
                            _summary_item_done(
                                [], "terminal-lock", locked_output_index
                            ),
                            completed,
                        ]
                    )
                )
            )
        )
        terminal = _assert_structured_summary_error(
            items,
            event_type="response.completed",
            field=field,
            value_shape=value_shape,
            output_index=error_output_index,
            summary_index=None,
        )
        assert not any(
            item.kind
            in {StreamItemKind.ANSWER_DELTA, StreamItemKind.ANSWER_DONE}
            for item in items
        )
        assert secret not in repr([terminal, terminal.to_trace_dict()])
    assert HostileString.calls == 0


def test_terminal_snapshot_allows_distinct_message_and_legacy_text() -> None:
    reasoning_done = _summary_item_done([], "terminal-safe", 0)
    completed = {
        "type": "response.completed",
        "response": {
            "usage": {},
            "output": [
                deepcopy(cast(dict[str, object], reasoning_done["item"])),
                {
                    "id": "message-safe",
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "safe-answer"}
                    ],
                },
            ],
        },
    }
    tracked_items = run(
        _consume(
            OpenAIStream(
                _AsyncEvents(
                    [
                        _summary_item_added("terminal-safe", 0),
                        reasoning_done,
                        completed,
                    ]
                )
            )
        )
    )
    legacy_items = run(
        _consume(OpenAIStream(_AsyncEvents([_completed_event(text="legacy")])))
    )

    assert (
        accumulate_canonical_stream_items(tracked_items).answer_text
        == "safe-answer"
    )
    assert (
        accumulate_canonical_stream_items(legacy_items).answer_text == "legacy"
    )


def test_text_events_cannot_relabel_non_text_identity() -> None:
    secret = "TEXT_EVENT_NON_TEXT_IDENTITY_PRIVATE_SENTINEL"
    for locked_type in ("reasoning", "function_call"):
        item_id = f"text-{locked_type}-lock"
        prefix = (
            [
                _summary_item_added(item_id, 0),
                _summary_item_done([], item_id, 0),
            ]
            if locked_type == "reasoning"
            else [_typed_item_added("function_call", item_id, 0)]
        )
        for event_type, field_name in (
            ("response.output_text.delta", "delta"),
            ("response.output_text.done", "text"),
        ):
            event = {
                "type": event_type,
                "item_id": item_id,
                "output_index": 0,
                field_name: secret,
            }
            items = run(
                _consume(
                    OpenAIStream(_AsyncEvents([*deepcopy(prefix), event]))
                )
            )
            terminal = _assert_structured_summary_error(
                items,
                event_type=event_type,
                field="item.type",
                value_shape="conflict",
                output_index=0,
                summary_index=None,
            )
            assert not any(
                item.kind
                in {StreamItemKind.ANSWER_DELTA, StreamItemKind.ANSWER_DONE}
                for item in items
            )
            assert secret not in repr([terminal, terminal.to_trace_dict()])


def test_tool_argument_events_cannot_relabel_non_tool_identity() -> None:
    secret = "TOOL_EVENT_IDENTITY_PRIVATE_SENTINEL"
    event_types = (
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.custom_tool_call_input.delta",
        "response.custom_tool_call_input.done",
    )
    for locked_type in ("reasoning", "message"):
        item_id = f"{locked_type}-tool-lock"
        prefix = (
            [
                _summary_item_added(item_id, 0),
                _summary_item_done([], item_id, 0),
            ]
            if locked_type == "reasoning"
            else [
                _message_item_event("response.output_item.added", item_id, 0)
            ]
        )
        for event_type in event_types:
            event = {
                "type": event_type,
                "item_id": item_id,
                "output_index": 0,
                "id": "private-call-id",
                "delta": secret,
                "arguments": secret,
            }
            items = run(
                _consume(
                    OpenAIStream(_AsyncEvents([*deepcopy(prefix), event]))
                )
            )
            terminal = _assert_structured_summary_error(
                items,
                event_type=event_type,
                field="item.type",
                value_shape="conflict",
                output_index=0,
                summary_index=None,
            )
            assert not any(
                item.kind
                in {
                    StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    StreamItemKind.TOOL_CALL_READY,
                    StreamItemKind.TOOL_CALL_DONE,
                }
                for item in items
            )
            assert secret not in repr([terminal, terminal.to_trace_dict()])


def test_unknown_output_item_done_never_emits_answer_or_tool() -> None:
    secret = "UNKNOWN_OUTPUT_ITEM_PRIVATE_SENTINEL"
    event = {
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {
            "id": "unknown-output-item",
            "type": "file_search_call",
            "content": [{"type": "output_text", "text": secret}],
        },
    }
    items = run(_consume(OpenAIStream(_AsyncEvents([event]))))

    assert not any(
        item.kind
        in {
            StreamItemKind.ANSWER_DELTA,
            StreamItemKind.ANSWER_DONE,
            StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
            StreamItemKind.TOOL_CALL_READY,
            StreamItemKind.TOOL_CALL_DONE,
        }
        for item in items
    )
    assert secret not in repr([item.to_trace_dict() for item in items])


def test_nested_custom_tool_shape_requires_matching_outer_type() -> None:
    secret = "NESTED_CUSTOM_TOOL_TYPE_PRIVATE_SENTINEL"

    def event(event_type: str, item_type: str) -> dict[str, object]:
        return {
            "type": event_type,
            "output_index": 0,
            "item": {
                "id": "nested-custom-item",
                "type": item_type,
                "status": (
                    "in_progress"
                    if event_type == "response.output_item.added"
                    else "completed"
                ),
                "custom_tool_call": {
                    "id": "nested-custom-call",
                    "name": "safe_tool",
                    "input": secret,
                },
                "content": [{"type": "output_text", "text": secret}],
            },
        }

    for event_type in (
        "response.output_item.added",
        "response.output_item.done",
    ):
        for item_type in ("message", "function_call", "file_search_call"):
            items = run(
                _consume(
                    OpenAIStream(_AsyncEvents([event(event_type, item_type)]))
                )
            )
            terminal = _assert_structured_summary_error(
                items,
                event_type=event_type,
                field="item.type",
                value_shape="conflict",
                output_index=0,
                summary_index=None,
            )
            assert not any(
                item.kind
                in {
                    StreamItemKind.ANSWER_DELTA,
                    StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    StreamItemKind.TOOL_CALL_READY,
                    StreamItemKind.TOOL_CALL_DONE,
                }
                for item in items
            )
            assert secret not in repr([terminal, terminal.to_trace_dict()])


def test_tool_argument_events_enforce_function_custom_subtype() -> None:
    secret = "TOOL_ARGUMENT_SUBTYPE_PRIVATE_SENTINEL"
    function_added = _typed_item_added("function_call", "function-subtype", 0)
    custom_added = {
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {
            "id": "custom-subtype",
            "type": "custom_tool_call",
            "status": "in_progress",
            "custom_tool_call": {
                "id": "custom-call",
                "name": "safe_tool",
            },
        },
    }
    invalid_cases = (
        (
            function_added,
            "response.custom_tool_call_input.delta",
            "function-subtype",
        ),
        (
            function_added,
            "response.custom_tool_call_input.done",
            "function-subtype",
        ),
        (
            custom_added,
            "response.function_call_arguments.delta",
            "custom-subtype",
        ),
        (
            custom_added,
            "response.function_call_arguments.done",
            "custom-subtype",
        ),
    )
    for added, event_type, item_id in invalid_cases:
        argument_event = {
            "type": event_type,
            "item_id": item_id,
            "output_index": 0,
            "delta": secret,
        }
        items = run(
            _consume(
                OpenAIStream(_AsyncEvents([deepcopy(added), argument_event]))
            )
        )
        terminal = _assert_structured_summary_error(
            items,
            event_type=event_type,
            field="item.type",
            value_shape="conflict",
            output_index=0,
            summary_index=None,
        )
        assert not any(
            item.kind
            in {
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
            }
            for item in items
        )
        assert secret not in repr([terminal, terminal.to_trace_dict()])

    positive_cases = (
        (
            function_added,
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "function-subtype",
        ),
        (
            custom_added,
            "response.custom_tool_call_input.delta",
            "response.custom_tool_call_input.done",
            "custom-subtype",
        ),
    )
    for added, delta_type, done_type, item_id in positive_cases:
        provider_events = run(
            _consume_provider_events(
                OpenAIStream(
                    _AsyncEvents(
                        [
                            deepcopy(added),
                            {
                                "type": delta_type,
                                "item_id": item_id,
                                "output_index": 0,
                                "delta": "{}",
                            },
                            {
                                "type": done_type,
                                "item_id": item_id,
                                "output_index": 0,
                            },
                        ]
                    )
                )
            )
        )
        assert any(
            event.kind is StreamItemKind.TOOL_CALL_ARGUMENT_DELTA
            for event in provider_events
        )
        assert any(
            event.kind is StreamItemKind.TOOL_CALL_READY
            for event in provider_events
        )


def test_native_reasoning_events_require_reasoning_identity() -> None:
    secret = "NATIVE_REASONING_IDENTITY_PRIVATE_SENTINEL"
    for locked_type in ("message", "function_call"):
        item_id = f"native-{locked_type}-lock"
        prefix = (
            _message_item_event("response.output_item.added", item_id, 0)
            if locked_type == "message"
            else _typed_item_added("function_call", item_id, 0)
        )
        for event_type in (
            "response.reasoning_text.delta",
            "response.reasoning_text.done",
        ):
            native_event = {
                "type": event_type,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "delta": secret,
            }
            items = run(
                _consume(
                    OpenAIStream(
                        _AsyncEvents([deepcopy(prefix), native_event])
                    )
                )
            )
            terminal = _assert_structured_summary_error(
                items,
                event_type=event_type,
                field="item.type",
                value_shape="conflict",
                output_index=0,
                summary_index=None,
            )
            assert not _reasoning_items(items)
            assert secret not in repr([terminal, terminal.to_trace_dict()])

    legacy_items = run(
        _consume(
            OpenAIStream(
                _AsyncEvents(
                    [
                        {
                            "type": "response.reasoning_text.delta",
                            "delta": "legacy-native",
                            "content_index": 0,
                        },
                        {
                            "type": "response.reasoning_text.done",
                            "content_index": 0,
                        },
                    ]
                )
            )
        )
    )
    tracked_items = run(
        _consume(
            OpenAIStream(
                _AsyncEvents(
                    [
                        _summary_item_added("native-reasoning", 0),
                        {
                            "type": "response.reasoning_text.delta",
                            "item_id": "native-reasoning",
                            "output_index": 0,
                            "content_index": 0,
                            "delta": "tracked-native",
                        },
                        {
                            "type": "response.reasoning_text.done",
                            "item_id": "native-reasoning",
                            "output_index": 0,
                            "content_index": 0,
                        },
                        _summary_item_done([], "native-reasoning", 0),
                    ]
                )
            )
        )
    )

    assert (
        accumulate_canonical_stream_items(legacy_items).reasoning_text
        == "legacy-native"
    )
    assert (
        accumulate_canonical_stream_items(tracked_items).reasoning_text
        == "tracked-native"
    )


def test_private_output_type_matrix_preserves_primary_error_over_cleanup() -> (
    None
):
    secret = "PRIVATE_TYPE_CLEANUP_SENTINEL"

    class HostileString(str):
        calls = 0

        def _fail(self, *_: object, **__: object) -> object:
            HostileString.calls += 1
            raise RuntimeError(secret)

        __eq__ = _fail
        __hash__ = _fail
        __str__ = _fail
        encode = _fail
        strip = _fail

    class CleanupFailingEvents(_AsyncEvents):
        async def aclose(self) -> None:
            self.close_count += 1
            raise RuntimeError(secret)

    variants: tuple[tuple[str, object, str], ...] = (
        ("exact", "reasoning", "unexpected_value"),
        ("missing", object(), "missing"),
        ("integer", 7, "integer"),
        ("boolean", False, "boolean"),
        ("subclass", HostileString("reasoning"), "sequence"),
    )
    for event_type in (
        "response.output_item.added",
        "response.output_item.done",
    ):
        for label, item_type, value_shape in variants:
            event = (
                _summary_item_added("cleanup-type", 0)
                if event_type == "response.output_item.added"
                else _summary_item_done([], "cleanup-type", 0)
            )
            item = cast(dict[str, object], event["item"])
            if label == "missing":
                item.pop("type")
            else:
                item["type"] = item_type
            if label == "exact":
                item["status"] = "invalid"
            prefix = (
                []
                if event_type == "response.output_item.added"
                else [_summary_item_added("cleanup-type", 0)]
            )
            source = CleanupFailingEvents([*prefix, event])
            items = run(_consume(OpenAIStream(source)))
            terminal = _assert_structured_summary_error(
                items,
                event_type=event_type,
                field="item.status" if label == "exact" else "item.type",
                value_shape=value_shape,
                output_index=0,
                summary_index=None,
            )
            assert source.close_count == 1
            assert secret not in repr([terminal, terminal.to_trace_dict()])

    nested = _message_item_event(
        "response.output_item.done",
        "nested-private-type",
        0,
        secret,
    )
    cast(
        dict[str, object],
        cast(list[object], cast(dict[str, object], nested["item"])["content"])[
            0
        ],
    )["type"] = HostileString("output_text")
    source = CleanupFailingEvents([nested])
    items = run(_consume(OpenAIStream(source)))
    terminal = _assert_structured_summary_error(
        items,
        event_type="response.output_item.done",
        field="item.content.type",
        value_shape="unexpected_value",
        output_index=0,
        summary_index=None,
    )
    assert source.close_count == 1
    assert HostileString.calls == 0
    assert secret not in repr([terminal, terminal.to_trace_dict()])


def test_present_output_fields_and_semantic_identities_are_strict() -> None:
    secret = "STRICT_PRESENT_IDENTITY_SENTINEL"
    cases: tuple[tuple[object, str, str, str, int | bool | None], ...] = (
        (
            _message_item_event("response.output_item.added", 7, 0, secret),
            "response.output_item.added",
            "item.id",
            "integer",
            0,
        ),
        (
            _message_item_event(
                "response.output_item.done", "strict-output", False, secret
            ),
            "response.output_item.done",
            "output_index",
            "boolean",
            None,
        ),
        (
            {
                "type": "response.output_text.delta",
                "item_id": 7,
                "delta": secret,
            },
            "response.output_text.delta",
            "item_id",
            "integer",
            None,
        ),
        (
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "strict-tool",
                "output_index": False,
                "delta": secret,
            },
            "response.function_call_arguments.delta",
            "output_index",
            "boolean",
            None,
        ),
        (
            {
                "type": "response.completed",
                "response": {
                    "usage": {},
                    "output": [
                        {
                            "id": 7,
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": secret}
                            ],
                        }
                    ],
                },
            },
            "response.completed",
            "item.id",
            "integer",
            0,
        ),
    )
    for event, event_type, field, value_shape, output_index in cases:
        items = run(_consume(OpenAIStream(_AsyncEvents([event]))))
        terminal = _assert_structured_summary_error(
            items,
            event_type=event_type,
            field=field,
            value_shape=value_shape,
            output_index=output_index,
            summary_index=None,
        )
        assert not any(
            item.kind
            in {
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
            }
            for item in items
        )
        assert secret not in repr([terminal, terminal.to_trace_dict()])


def test_message_content_types_and_terminal_aggregate_are_authoritative() -> (
    None
):
    secret = "PRIVATE_MESSAGE_CONTENT_SENTINEL"
    invalid_part_types: tuple[object, ...] = (
        "reasoning_text",
        "unknown_private",
        7,
    )
    for event_type in ("response.output_item.done", "response.completed"):
        for part_type in invalid_part_types:
            item = {
                "id": "strict-message",
                "type": "message",
                "status": "completed",
                "content": [{"type": part_type, "text": secret}],
            }
            event = (
                {
                    "type": event_type,
                    "output_index": 0,
                    "item": item,
                }
                if event_type == "response.output_item.done"
                else {
                    "type": event_type,
                    "response": {"usage": {}, "output": [item]},
                }
            )
            items = run(_consume(OpenAIStream(_AsyncEvents([event]))))
            terminal = _assert_structured_summary_error(
                items,
                event_type=event_type,
                field="item.content.type",
                value_shape="unexpected_value",
                output_index=0,
                summary_index=None,
            )
            assert not any(
                item.kind is StreamItemKind.ANSWER_DELTA for item in items
            )
            assert secret not in repr([terminal, terminal.to_trace_dict()])

    for output in (
        [_reasoning_item("aggregate-reasoning", secret, [])],
        [_function_item("aggregate-tool")],
    ):
        items = run(
            _consume(
                OpenAIStream(
                    _AsyncEvents(
                        [
                            {
                                "type": "response.completed",
                                "response": {
                                    "usage": {},
                                    "output": output,
                                    "output_text": secret,
                                },
                            }
                        ]
                    )
                )
            )
        )
        assert not any(
            item.kind is StreamItemKind.ANSWER_DELTA for item in items
        )
        assert (
            secret not in accumulate_canonical_stream_items(items).answer_text
        )

    message = {
        "id": "aggregate-message",
        "type": "message",
        "status": "completed",
        "content": [{"type": "output_text", "text": "safe-answer"}],
    }
    matching = {
        "type": "response.completed",
        "response": {
            "usage": {},
            "output": [_reasoning_item("aggregate-mixed"), message],
            "output_text": "safe-answer",
        },
    }
    matching_items = run(
        _consume(OpenAIStream(_AsyncEvents([deepcopy(matching)])))
    )
    assert (
        accumulate_canonical_stream_items(matching_items).answer_text
        == "safe-answer"
    )

    cast(dict[str, object], matching["response"])["output_text"] = secret
    mismatch_items = run(_consume(OpenAIStream(_AsyncEvents([matching]))))
    terminal = _assert_structured_summary_error(
        mismatch_items,
        event_type="response.completed",
        field="response.output_text",
        value_shape="conflict",
        output_index=0,
        summary_index=None,
    )
    assert secret not in repr([terminal, terminal.to_trace_dict()])


def test_semantic_first_locks_are_monotonic() -> None:
    secret = "SEMANTIC_FIRST_CHANNEL_SENTINEL"
    reverse_cases = (
        (
            [
                {
                    "type": "response.output_text.delta",
                    "item_id": "reverse-text",
                    "output_index": 0,
                    "delta": "safe",
                },
                {
                    "type": "response.output_text.done",
                    "item_id": "reverse-text",
                    "output_index": 0,
                },
            ],
            _summary_item_added("reverse-text", 0),
        ),
        (
            [
                {
                    "type": "response.reasoning_text.delta",
                    "item_id": "reverse-native",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "safe",
                },
                {
                    "type": "response.reasoning_text.done",
                    "item_id": "reverse-native",
                    "output_index": 0,
                    "content_index": 0,
                },
            ],
            _message_item_event(
                "response.output_item.added", "reverse-native", 0, secret
            ),
        ),
        (
            [
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": "reverse-function",
                    "output_index": 0,
                    "delta": "{}",
                },
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": "reverse-function",
                    "output_index": 0,
                    "arguments": "{}",
                },
            ],
            _summary_item_added("reverse-function", 0),
        ),
        (
            [
                {
                    "type": "response.custom_tool_call_input.delta",
                    "item_id": "reverse-custom",
                    "output_index": 0,
                    "delta": "{}",
                },
                {
                    "type": "response.custom_tool_call_input.done",
                    "item_id": "reverse-custom",
                    "output_index": 0,
                    "input": "{}",
                },
            ],
            _summary_item_added("reverse-custom", 0),
        ),
    )
    for prefix, conflicting in reverse_cases:
        items = run(
            _consume(
                OpenAIStream(
                    _AsyncEvents([*deepcopy(prefix), deepcopy(conflicting)])
                )
            )
        )
        terminal = _assert_structured_summary_error(
            items,
            event_type="response.output_item.added",
            field="item.type",
            value_shape="conflict",
            output_index=0,
            summary_index=None,
        )
        assert secret not in repr([terminal, terminal.to_trace_dict()])


def test_tool_identity_and_name_registry_is_one_to_one() -> None:
    secret = "TOOL_REGISTRY_CONFLICT_SENTINEL"
    base = _typed_item_added("function_call", "registry-item", 0)
    base_item = cast(dict[str, object], base["item"])
    base_item["call_id"] = "registry-call"
    base_item["name"] = "safe_tool"
    conflicts = (
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                **base_item,
                "status": "completed",
                "call_id": "other-call",
            },
        },
        {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {
                **base_item,
                "id": "other-item",
                "call_id": "registry-call",
            },
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "registry-item",
            "output_index": 0,
            "call_id": "other-call",
            "delta": secret,
        },
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {**base_item, "name": "other_tool"},
        },
    )
    for conflict in conflicts:
        items = run(
            _consume(
                OpenAIStream(
                    _AsyncEvents([deepcopy(base), deepcopy(conflict)])
                )
            )
        )
        terminal = _error_item(items)
        assert terminal.provider_payload is None
        assert not any(
            item.kind
            in {StreamItemKind.TOOL_CALL_READY, StreamItemKind.TOOL_CALL_DONE}
            for item in items
        )
        assert secret not in repr([terminal, terminal.to_trace_dict()])

    missing_name = deepcopy(base)
    cast(dict[str, object], missing_name["item"]).pop("name")
    completed = {
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {
            **base_item,
            "status": "completed",
            "arguments": "{}",
        },
    }
    positive = run(
        _consume(
            OpenAIStream(
                _AsyncEvents([missing_name, completed, _completed_event()])
            )
        )
    )
    assert any(item.kind is StreamItemKind.TOOL_CALL_DONE for item in positive)
    assert not any(
        item.kind is StreamItemKind.STREAM_ERRORED for item in positive
    )


def test_tool_argument_final_values_reconcile_before_side_effects() -> None:
    secret = "TOOL_ARGUMENT_RECONCILIATION_SENTINEL"
    for custom in (False, True):
        item_id = f"argument-{'custom' if custom else 'function'}"
        call_id = f"call-{'custom' if custom else 'function'}"
        item_type = "custom_tool_call" if custom else "function_call"
        added_item: dict[str, object] = {
            "id": item_id,
            "type": item_type,
            "status": "in_progress",
        }
        if custom:
            added_item["custom_tool_call"] = {
                "id": call_id,
                "name": "safe_tool",
            }
        else:
            added_item.update(call_id=call_id, name="safe_tool")
        added = {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": added_item,
        }
        delta_type = (
            "response.custom_tool_call_input.delta"
            if custom
            else "response.function_call_arguments.delta"
        )
        done_type = (
            "response.custom_tool_call_input.done"
            if custom
            else "response.function_call_arguments.done"
        )
        full_field = "input" if custom else "arguments"
        prefix = [
            added,
            {
                "type": delta_type,
                "item_id": item_id,
                "output_index": 0,
                "delta": '{"x":1}',
            },
        ]
        for final_event in (
            {
                "type": done_type,
                "item_id": item_id,
                "output_index": 0,
                full_field: secret,
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    **added_item,
                    "status": "completed",
                    full_field: secret,
                },
            },
        ):
            retained: list[dict[str, Any]] = []
            items = run(
                _consume(
                    OpenAIStream(
                        _AsyncEvents([*deepcopy(prefix), final_event]),
                        output_item_sink=retained.append,
                    )
                )
            )
            terminal = _assert_structured_summary_error(
                items,
                event_type=cast(str, final_event["type"]),
                field="arguments",
                value_shape="conflict",
                output_index=0,
                summary_index=None,
            )
            assert not any(
                item.kind is StreamItemKind.TOOL_CALL_READY for item in items
            )
            assert all(
                item.metadata.get("tool_call.close_reason") == "error"
                for item in items
                if item.kind is StreamItemKind.TOOL_CALL_DONE
            )
            assert retained == []
            assert secret not in repr([terminal, terminal.to_trace_dict()])

        matching = '{"x":1}'
        positive_events = [
            *deepcopy(prefix),
            {
                "type": done_type,
                "item_id": item_id,
                "output_index": 0,
                full_field: matching,
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    **added_item,
                    "status": "completed",
                    full_field: matching,
                },
            },
        ]
        positive = run(_consume(OpenAIStream(_AsyncEvents(positive_events))))
        assert (
            sum(
                item.kind is StreamItemKind.TOOL_CALL_READY
                for item in positive
            )
            == 1
        )
        assert (
            sum(
                item.kind is StreamItemKind.TOOL_CALL_DONE for item in positive
            )
            == 1
        )


def test_output_status_and_closed_channel_transitions_are_strict() -> None:
    secret = "CLOSED_CHANNEL_TRANSITION_SENTINEL"
    status_cases = (
        _message_item_event(
            "response.output_item.added", "status-message", 0, secret
        ),
        _typed_item_added("function_call", "status-tool", 0),
        _message_item_event(
            "response.output_item.done", "status-done", 0, secret
        ),
    )
    for event in status_cases:
        event_type = cast(str, event["type"])
        item = cast(dict[str, object], event["item"])
        item["status"] = (
            "completed"
            if event_type == "response.output_item.added"
            else "in_progress"
        )
        items = run(_consume(OpenAIStream(_AsyncEvents([event]))))
        terminal = _assert_structured_summary_error(
            items,
            event_type=event_type,
            field="item.status",
            value_shape="unexpected_value",
            output_index=0,
            summary_index=None,
        )
        assert secret not in repr([terminal, terminal.to_trace_dict()])

    late_cases = (
        (
            [
                {
                    "type": "response.output_text.delta",
                    "item_id": "closed-text",
                    "output_index": 0,
                    "delta": "safe",
                },
                {
                    "type": "response.output_text.done",
                    "item_id": "closed-text",
                    "output_index": 0,
                },
            ],
            {
                "type": "response.output_text.delta",
                "item_id": "closed-text",
                "output_index": 0,
                "delta": secret,
            },
        ),
        (
            [
                {
                    "type": "response.reasoning_text.delta",
                    "item_id": "closed-reasoning",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "safe",
                },
                {
                    "type": "response.reasoning_text.done",
                    "item_id": "closed-reasoning",
                    "output_index": 0,
                    "content_index": 0,
                },
            ],
            {
                "type": "response.reasoning_text.delta",
                "item_id": "closed-reasoning",
                "output_index": 0,
                "content_index": 0,
                "delta": secret,
            },
        ),
        (
            [
                _typed_item_added("function_call", "closed-tool", 0),
                _tool_item_done("closed-tool", 0, "{}"),
            ],
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "closed-tool",
                "output_index": 0,
                "delta": secret,
            },
        ),
    )
    for prefix, late_event in late_cases:
        items = run(
            _consume(OpenAIStream(_AsyncEvents([*prefix, late_event])))
        )
        terminal = _assert_structured_summary_error(
            items,
            event_type=cast(str, late_event["type"]),
            field=(
                "content"
                if late_event["type"] == "response.reasoning_text.delta"
                else "item"
            ),
            value_shape="closed",
            output_index=0,
            summary_index=None,
        )
        assert secret not in repr([terminal, terminal.to_trace_dict()])


def test_reasoning_done_is_exact_and_terminal_cipher_is_opaque() -> None:
    secret = "REASONING_CIPHERTEXT_CONFLICT_SENTINEL"
    first = _summary_item_done(["safe"], "exact-reasoning", 0)
    exact_events = [
        _summary_item_added("exact-reasoning", 0),
        deepcopy(first),
        deepcopy(first),
        _completed_event(),
    ]
    exact_items = run(
        _consume(OpenAIStream(_AsyncEvents(deepcopy(exact_events))))
    )
    assert not any(
        item.kind is StreamItemKind.STREAM_ERRORED for item in exact_items
    )

    changed = deepcopy(first)
    cast(dict[str, object], changed["item"])["encrypted_content"] = secret
    changed_items = run(
        _consume(
            OpenAIStream(
                _AsyncEvents(
                    [
                        _summary_item_added("exact-reasoning", 0),
                        deepcopy(first),
                        changed,
                    ]
                )
            )
        )
    )
    terminal = _assert_structured_summary_error(
        changed_items,
        event_type="response.output_item.done",
        field="item",
        value_shape="conflict",
        output_index=0,
        summary_index=None,
    )
    assert secret not in repr([terminal, terminal.to_trace_dict()])

    terminal_item = deepcopy(cast(dict[str, object], first["item"]))
    exact_terminal = {
        "type": "response.completed",
        "response": {"usage": {}, "output": [terminal_item]},
    }
    terminal_items = run(
        _consume(
            OpenAIStream(
                _AsyncEvents(
                    [
                        _summary_item_added("exact-reasoning", 0),
                        deepcopy(first),
                        exact_terminal,
                    ]
                )
            )
        )
    )
    assert not any(
        item.kind is StreamItemKind.STREAM_ERRORED for item in terminal_items
    )

    cast(dict[str, object], terminal_item)["encrypted_content"] = secret
    changed_terminal_items = run(
        _consume(
            OpenAIStream(
                _AsyncEvents(
                    [
                        _summary_item_added("exact-reasoning", 0),
                        deepcopy(first),
                        exact_terminal,
                    ]
                )
            )
        )
    )
    assert not any(
        item.kind is StreamItemKind.STREAM_ERRORED
        for item in changed_terminal_items
    )
    assert secret not in repr(
        [item.to_trace_dict() for item in changed_terminal_items]
    )

    cast(dict[str, object], terminal_item).pop("encrypted_content")
    missing_terminal_items = run(
        _consume(
            OpenAIStream(
                _AsyncEvents(
                    [
                        _summary_item_added("exact-reasoning", 0),
                        deepcopy(first),
                        exact_terminal,
                    ]
                )
            )
        )
    )
    assert not any(
        item.kind is StreamItemKind.STREAM_ERRORED
        for item in missing_terminal_items
    )


def test_text_content_index_is_strict_and_stable() -> None:
    secret = "TEXT_CONTENT_INDEX_SENTINEL"
    for value, value_shape in (
        (-1, "negative_integer"),
        (False, "boolean"),
        (1.5, "float"),
        ("0", "string"),
    ):
        event = {
            "type": "response.output_text.delta",
            "item_id": "content-index",
            "output_index": 0,
            "content_index": value,
            "delta": secret,
        }
        items = run(_consume(OpenAIStream(_AsyncEvents([event]))))
        terminal = _assert_structured_summary_error(
            items,
            event_type="response.output_text.delta",
            field="content_index",
            value_shape=value_shape,
            output_index=0,
            summary_index=None,
        )
        assert secret not in repr([terminal, terminal.to_trace_dict()])

    conflicting = [
        {
            "type": "response.output_text.delta",
            "item_id": "content-index",
            "output_index": 0,
            "content_index": 0,
            "delta": "safe",
        },
        {
            "type": "response.output_text.done",
            "item_id": "content-index",
            "output_index": 0,
            "content_index": 1,
        },
    ]
    items = run(_consume(OpenAIStream(_AsyncEvents(conflicting))))
    terminal = _assert_structured_summary_error(
        items,
        event_type="response.output_text.done",
        field="content_index",
        value_shape="conflict",
        output_index=0,
        summary_index=None,
    )
    assert secret not in repr([terminal, terminal.to_trace_dict()])


def test_text_and_terminal_identity_conflicts_are_strict() -> None:
    missing_type_done = {
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {"id": "locked-message", "status": "completed"},
    }
    items = run(
        _consume(
            OpenAIStream(
                _AsyncEvents(
                    [
                        _message_item_event(
                            "response.output_item.added",
                            "locked-message",
                            0,
                        ),
                        missing_type_done,
                    ]
                )
            )
        )
    )
    _assert_structured_summary_error(
        items,
        event_type="response.output_item.done",
        field="item.type",
        value_shape="missing",
        output_index=0,
        summary_index=None,
    )

    collision = _message_item_event(
        "response.output_item.added", "first-message", 1
    )
    items = run(
        _consume(
            OpenAIStream(
                _AsyncEvents(
                    [
                        _message_item_event(
                            "response.output_item.added", "first-message", 0
                        ),
                        _message_item_event(
                            "response.output_item.added", "second-message", 1
                        ),
                        collision,
                    ]
                )
            )
        )
    )
    _assert_structured_summary_error(
        items,
        event_type="response.output_item.added",
        field="item.identity",
        value_shape="conflict",
        output_index=1,
        summary_index=None,
    )

    terminal_without_id = {
        "type": "response.completed",
        "response": {
            "usage": {},
            "output": [
                {
                    "type": "message",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "must-not-emit"}
                    ],
                }
            ],
        },
    }
    items = run(
        _consume(
            OpenAIStream(
                _AsyncEvents(
                    [
                        _message_item_event(
                            "response.output_item.added", "terminal-id", 0
                        ),
                        terminal_without_id,
                    ]
                )
            )
        )
    )
    _assert_structured_summary_error(
        items,
        event_type="response.completed",
        field="item.identity",
        value_shape="conflict",
        output_index=0,
        summary_index=None,
    )


def test_semantic_tool_metadata_refines_and_rejects_name_conflicts() -> None:
    added = _typed_item_added("function_call", "semantic-name", 0)
    added_item = cast(dict[str, object], added["item"])
    added_item.pop("name")
    added_item["call_id"] = "semantic-call"
    positive = run(
        _consume(
            OpenAIStream(
                _AsyncEvents(
                    [
                        deepcopy(added),
                        {
                            "type": "response.function_call_arguments.done",
                            "item_id": "semantic-name",
                            "output_index": 0,
                            "name": "safe_tool",
                            "arguments": "{}",
                        },
                    ]
                )
            )
        )
    )
    ready = next(
        item
        for item in positive
        if item.kind is StreamItemKind.TOOL_CALL_READY
    )
    assert ready.data == {"name": "safe_tool"}

    conflicting_added = deepcopy(added)
    cast(dict[str, object], conflicting_added["item"])["name"] = "safe_tool"
    items = run(
        _consume(
            OpenAIStream(
                _AsyncEvents(
                    [
                        conflicting_added,
                        {
                            "type": "response.function_call_arguments.done",
                            "item_id": "semantic-name",
                            "output_index": 0,
                            "name": "other_tool",
                            "arguments": "{}",
                        },
                    ]
                )
            )
        )
    )
    _assert_structured_summary_error(
        items,
        event_type="response.function_call_arguments.done",
        field="item.name",
        value_shape="conflict",
        output_index=0,
        summary_index=None,
    )


def test_tool_argument_closure_and_no_delta_fallback_are_consistent() -> None:
    def added(item_id: str) -> dict[str, object]:
        event = _typed_item_added("function_call", item_id, 0)
        cast(dict[str, object], event["item"])["call_id"] = f"call-{item_id}"
        return event

    fallback = run(
        _consume(
            OpenAIStream(
                _AsyncEvents(
                    [
                        added("fallback"),
                        {
                            "type": "response.function_call_arguments.done",
                            "item_id": "fallback",
                            "output_index": 0,
                            "arguments": '{"fallback":true}',
                        },
                    ]
                )
            )
        )
    )
    assert accumulate_canonical_stream_items(fallback).tool_call_arguments == {
        "call-fallback": '{"fallback":true}'
    }

    closed_prefix = [
        added("closed-arguments"),
        {
            "type": "response.function_call_arguments.done",
            "item_id": "closed-arguments",
            "output_index": 0,
            "arguments": "{}",
        },
    ]
    for late_event in (
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "closed-arguments",
            "output_index": 0,
            "delta": "late",
        },
        {
            "type": "response.function_call_arguments.done",
            "item_id": "closed-arguments",
            "output_index": 0,
            "arguments": "{}",
        },
    ):
        items = run(
            _consume(
                OpenAIStream(
                    _AsyncEvents([*deepcopy(closed_prefix), late_event])
                )
            )
        )
        _assert_structured_summary_error(
            items,
            event_type=cast(str, late_event["type"]),
            field="arguments",
            value_shape="closed",
            output_index=0,
            summary_index=None,
        )

    mismatch = {
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {
            "id": "final-mismatch",
            "type": "function_call",
            "status": "completed",
            "call_id": "call-final-mismatch",
            "name": "safe_tool",
            "arguments": "different",
        },
    }
    items = run(
        _consume(
            OpenAIStream(
                _AsyncEvents(
                    [
                        added("final-mismatch"),
                        {
                            "type": "response.function_call_arguments.done",
                            "item_id": "final-mismatch",
                            "output_index": 0,
                            "arguments": "original",
                        },
                        mismatch,
                    ]
                )
            )
        )
    )
    _assert_structured_summary_error(
        items,
        event_type="response.output_item.done",
        field="arguments",
        value_shape="conflict",
        output_index=0,
        summary_index=None,
    )


def test_reasoning_fingerprint_sdk_fallback_and_retention_errors() -> None:
    class DumpFailingItem(SimpleNamespace):
        def model_dump(self, *, mode: str) -> object:
            assert mode == "json"
            raise RuntimeError("dump unavailable")

    def done_item(
        item_type: type[SimpleNamespace], encrypted: object
    ) -> object:
        return item_type(
            id="sdk-fingerprint",
            type="reasoning",
            status="completed",
            summary=[{"type": "summary_text", "text": "sdk-summary"}],
            encrypted_content=encrypted,
        )

    sdk_payload = _summary_item_done(["sdk-summary"], "sdk-fingerprint", 0)
    sdk_payload["sequence_number"] = 1
    sdk_event = _RESPONSE_STREAM_EVENT.validate_python(sdk_payload)
    items = run(
        _consume(
            OpenAIStream(
                _AsyncEvents(
                    [
                        _summary_item_added("sdk-fingerprint", 0),
                        sdk_event,
                    ]
                )
            )
        )
    )
    assert [item.text_delta for item in _reasoning_items(items)] == [
        "sdk-summary"
    ]
    assert not any(
        item.kind is StreamItemKind.STREAM_ERRORED for item in items
    )

    dump_failing = done_item(DumpFailingItem, "sdk-cipher")
    dump_items = run(
        _consume(
            OpenAIStream(
                _AsyncEvents(
                    [
                        _summary_item_added("sdk-fingerprint", 0),
                        SimpleNamespace(
                            type="response.output_item.done",
                            output_index=0,
                            item=dump_failing,
                        ),
                    ]
                )
            )
        )
    )
    _assert_preparse_private_error(
        dump_items,
        event_type="response.output_item.done",
    )

    retained = done_item(SimpleNamespace, object())
    items = run(
        _consume(
            OpenAIStream(
                _AsyncEvents(
                    [
                        _summary_item_added("sdk-fingerprint", 0),
                        SimpleNamespace(
                            type="response.output_item.done",
                            output_index=0,
                            item=retained,
                        ),
                    ]
                )
            )
        )
    )
    _assert_structured_summary_error(
        items,
        event_type="response.output_item.done",
        field="item",
        value_shape="unreadable",
        output_index=0,
        summary_index=None,
    )


def test_private_output_classifier_handles_every_unreadable_boundary() -> None:
    missing = openai_module._MISSING_PROVIDER_FIELD

    class FieldTrap:
        def __init__(
            self,
            field_name: str,
            values: dict[str, object] | None = None,
        ) -> None:
            self._field_name = field_name
            self._values = values or {}

        def __getattr__(self, field_name: str) -> object:
            if field_name == self._field_name:
                raise RuntimeError("unreadable private field")
            if field_name in self._values:
                return self._values[field_name]
            raise AttributeError(field_name)

    cases = (
        (FieldTrap("response"), "response.completed"),
        (
            SimpleNamespace(response=FieldTrap("output")),
            "response.completed",
        ),
        (
            SimpleNamespace(item=FieldTrap("type")),
            "response.output_item.done",
        ),
        (
            SimpleNamespace(item=FieldTrap("summary", {"type": "message"})),
            "response.output_item.done",
        ),
        (
            SimpleNamespace(item=FieldTrap("content", {"type": "message"})),
            "response.output_item.done",
        ),
        (
            SimpleNamespace(
                item=SimpleNamespace(
                    type="message", content=[FieldTrap("type")]
                )
            ),
            "response.output_item.done",
        ),
        (
            SimpleNamespace(
                item=FieldTrap(
                    "custom_tool_call",
                    {"type": "custom_tool_call", "content": []},
                )
            ),
            "response.output_item.done",
        ),
        (
            SimpleNamespace(
                item=SimpleNamespace(
                    type="custom_tool_call",
                    custom_tool_call=FieldTrap("type"),
                )
            ),
            "response.output_item.done",
        ),
        (
            SimpleNamespace(
                item=SimpleNamespace(
                    type="custom_tool_call",
                    custom_tool_call={"type": "reasoning"},
                )
            ),
            "response.output_item.done",
        ),
    )
    for event, event_type in cases:
        assert OpenAIStream._event_may_contain_private_output(
            event, event_type
        )
    assert missing is openai_module._MISSING_PROVIDER_FIELD


def test_terminal_aggregate_and_message_field_shapes_are_strict() -> None:
    cases = (
        (
            {
                "type": "response.completed",
                "response": {"usage": {}, "output_text": object()},
            },
            "response.output_text",
            "object",
            None,
        ),
        (
            {
                "type": "response.completed",
                "response": {"usage": {}, "output": object()},
            },
            "response.output",
            "object",
            None,
        ),
        (
            {
                "type": "response.completed",
                "response": {
                    "usage": {},
                    "output": [
                        {
                            "id": "aggregate-shape",
                            "type": "message",
                            "status": "completed",
                            "content": [
                                {"type": "output_text", "text": "safe"}
                            ],
                        }
                    ],
                    "output_text": object(),
                },
            },
            "response.output_text",
            "object",
            None,
        ),
        (
            _message_item_event(
                "response.output_item.done",
                "content-text-shape",
                0,
                object(),
            ),
            "item.content.text",
            "object",
            0,
        ),
        (
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": "direct-text-shape",
                    "type": "message",
                    "status": "completed",
                    "text": object(),
                },
            },
            "item.text",
            "object",
            0,
        ),
    )
    for event, field, value_shape, output_index in cases:
        items = run(_consume(OpenAIStream(_AsyncEvents([event]))))
        event_type = cast(str, cast(dict[str, object], event)["type"])
        if field == "response.output":
            _assert_preparse_private_error(items, event_type=event_type)
        else:
            _assert_structured_summary_error(
                items,
                event_type=event_type,
                field=field,
                value_shape=value_shape,
                output_index=output_index,
                summary_index=None,
            )


def test_terminal_tool_without_identity_and_malformed_tool_id_are_safe() -> (
    None
):
    secret = "MALFORMED_TOOL_ID_PRIVATE_SENTINEL"

    class HostileToolId:
        calls = 0

        def _fail(self, *_: object, **__: object) -> object:
            type(self).calls += 1
            raise RuntimeError(secret)

        __eq__ = _fail
        __hash__ = _fail
        __repr__ = _fail
        __str__ = _fail

    terminal = {
        "type": "response.completed",
        "response": {
            "usage": {},
            "output": [{"type": "function_call", "status": "completed"}],
        },
    }
    items = run(_consume(OpenAIStream(_AsyncEvents([terminal]))))
    _assert_structured_summary_error(
        items,
        event_type="response.completed",
        field="item.call_id",
        value_shape="missing",
        output_index=0,
        summary_index=None,
    )

    malformed = {
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {
            "id": HostileToolId(),
            "type": "function_call",
            "status": "in_progress",
        },
    }
    malformed_items = run(_consume(OpenAIStream(_AsyncEvents([malformed]))))
    error = _error_item(malformed_items)
    assert error.provider_payload is None
    assert error.data == {
        "message": "response tool call id must be a non-empty string",
        "error": {
            "type": "invalid_provider_event",
            "code": "invalid_reasoning_summary_event",
            "message": "OpenAI reasoning summary event is invalid.",
            "event_type": "response.output_item.added",
            "field": "item.call_id",
            "index": {"output_index": 0, "summary_index": None},
            "value_shape": "object",
        },
    }
    assert error.provider_event_type == "response.output_item.added"
    assert HostileToolId.calls == 0
    assert secret not in repr([error, error.to_trace_dict()])
    assert not OpenAIStream._is_safe_response_id("unsafe-response-id")


def test_native_lock_without_summary_state_rejects_summary_parts() -> None:
    events = [
        {
            "type": "response.reasoning_text.delta",
            "item_id": "native-only-lock",
            "output_index": 0,
            "content_index": 0,
            "delta": "native",
        },
        _summary_part_added("native-only-lock", 0, 0),
    ]
    items = run(_consume(OpenAIStream(_AsyncEvents(events))))
    _assert_structured_summary_error(
        items,
        event_type="response.reasoning_summary_part.added",
        field="item.state",
        value_shape="missing",
        output_index=0,
        summary_index=0,
    )


def test_empty_native_reasoning_delta_is_invisible_and_locks_identity() -> (
    None
):
    events = [
        {
            "type": "response.reasoning_text.delta",
            "item_id": "empty-native-lock",
            "output_index": 0,
            "content_index": 0,
            "delta": "",
        },
        _message_item_event(
            "response.output_item.added", "empty-native-lock", 0
        ),
    ]
    items = run(_consume(OpenAIStream(_AsyncEvents(events))))
    assert not _reasoning_items(items)
    _assert_structured_summary_error(
        items,
        event_type="response.output_item.added",
        field="item.type",
        value_shape="conflict",
        output_index=0,
        summary_index=None,
    )


def test_private_provider_payload_strips_reasoning_replay_fields() -> None:
    payload = openai_module._sanitize_provider_json_payload(
        {
            "item": {
                "id": "private-payload",
                "type": "reasoning",
                "encrypted_content": "cipher",
                "summary": [{"type": "summary_text", "text": "private"}],
                "content": [{"type": "reasoning_text", "text": "private"}],
            }
        }
    )
    assert payload == {"item": {"id": "private-payload", "type": "reasoning"}}


def test_standalone_reasoning_fingerprints_clear_on_terminal_paths() -> None:
    secret = "STANDALONE_FINGERPRINT_SCRUB_PRIVATE_SENTINEL"
    done = _output_done(
        _reasoning_item(
            "fingerprint-scrub",
            secret,
            [{"type": "summary_text", "text": secret}],
        )
    )
    normal = OpenAIStream(_AsyncEvents([deepcopy(done), _completed_event()]))
    run(_consume(normal))
    assert (
        normal._reasoning_summary_state._standalone_completed_item_ids == set()
    )

    failed = OpenAIStream(
        _AsyncEvents(
            [
                deepcopy(done),
                {
                    "type": "response.failed",
                    "error": {"code": "response_failed"},
                    "response": {
                        "status": "failed",
                        "error": {"code": "response_failed"},
                        "output": [],
                    },
                },
            ]
        )
    )
    run(_consume(failed))
    assert (
        failed._reasoning_summary_state._standalone_completed_item_ids == set()
    )

    for method_name in ("cancel", "aclose"):
        stream = OpenAIStream(_AsyncEvents([]))
        stream._provider_events_from_event(deepcopy(done))
        assert stream._reasoning_summary_state._standalone_completed_item_ids
        run(getattr(stream, method_name)())
        assert (
            stream._reasoning_summary_state._standalone_completed_item_ids
            == set()
        )
        assert secret not in repr(stream._reasoning_summary_state)


def test_summary_adapter_error_provider_payload_is_not_projected() -> None:
    secret = "provider-payload-private-sentinel"
    event = _summary_delta(_CoercionTrap())
    event["metadata"] = {"echo": secret}
    items = run(
        _consume(
            OpenAIStream(
                _AsyncEvents(
                    [
                        _summary_item_added(),
                        _summary_part_added(),
                        event,
                    ]
                )
            )
        )
    )
    terminal = _error_item(items)
    outward = "\n".join(
        (
            repr(terminal),
            str(terminal),
            repr(terminal.to_trace_dict()),
        )
    )

    assert terminal.provider_payload is None
    assert secret not in outward
    assert "_CoercionTrap" not in outward


def test_summary_provider_failure_and_incomplete_balance_prefix() -> None:
    incomplete_events = [
        _RESPONSE_STREAM_EVENT.validate_python(event)
        for event in _trace_events("incomplete_after_summary")
    ]
    cases = (
        (
            "response.failed",
            _trace_events("failure_after_summary"),
            StreamItemKind.STREAM_ERRORED,
        ),
        (
            "response.incomplete",
            incomplete_events,
            StreamItemKind.STREAM_ERRORED,
        ),
    )

    for event_type, events, terminal_kind in cases:
        items = run(_consume(OpenAIStream(_AsyncEvents(events))))
        reasoning = _reasoning_items(items)
        assert len(reasoning) == 1
        reasoning_index = items.index(reasoning[0])
        done_index = next(
            index
            for index, item in enumerate(items)
            if item.kind is StreamItemKind.REASONING_DONE
        )
        terminal_index = next(
            index
            for index, item in enumerate(items)
            if item.kind is terminal_kind
        )
        assert reasoning_index < done_index < terminal_index
        assert sum(item.kind is terminal_kind for item in items) == 1
        assert not any(
            item.kind
            in {
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
            }
            for item in items[terminal_index + 1 :]
        )
        assert items[terminal_index].provider_event_type == event_type


def test_summary_response_incomplete_maps_safe_error_after_prefix() -> None:
    retry = AsyncMock(return_value=_AsyncEvents([_completed_event()]))
    events = _trace_events("incomplete_after_summary")
    events[-1] = _RESPONSE_STREAM_EVENT.validate_python(events[-1])
    items = run(
        _consume(
            OpenAIStream(
                _AsyncEvents(events),
                stream_factory=retry,
                stream_retries=1,
            )
        )
    )
    retry.assert_not_awaited()
    terminal = _error_item(items)
    data = cast(dict[str, Any], terminal.data)

    assert terminal.provider_event_type == "response.incomplete"
    assert data["error"] == {
        "code": "response_incomplete",
        "message": "response incomplete: max_output_tokens",
        "reason": "max_output_tokens",
        "status": "incomplete",
        "response_id": "resp-incomplete",
    }
    assert [
        item.kind
        for item in items
        if item.kind
        in {StreamItemKind.REASONING_DONE, StreamItemKind.STREAM_ERRORED}
    ] == [StreamItemKind.REASONING_DONE, StreamItemKind.STREAM_ERRORED]


def test_summary_response_incomplete_rejects_untrusted_metadata() -> None:
    secret = "INCOMPLETE_METADATA_PRIVATE_SENTINEL"
    events = [
        *_summary_trace(("visible-summary",))[:-1],
        {
            "type": "response.incomplete",
            "response": {
                "id": f"resp_{secret}",
                "status": secret,
                "incomplete_details": {"reason": secret},
            },
        },
    ]
    items = run(_consume(OpenAIStream(_AsyncEvents(events))))
    terminal = _error_item(items)
    outward = repr([terminal, terminal.to_trace_dict()])

    assert terminal.provider_payload is None
    assert terminal.provider_event_type == "response.incomplete"
    assert terminal.data == {
        "error": {
            "code": "response_incomplete",
            "message": "response incomplete",
        }
    }
    assert secret not in outward


def test_summary_state_is_isolated_per_stream_instance() -> None:
    first = OpenAIStream(_AsyncEvents(_summary_trace(("first",))))
    second = OpenAIStream(_AsyncEvents(_summary_trace(("second",))))

    first_items, second_items = run(_consume(first)), run(_consume(second))
    first_reasoning = _reasoning_items(first_items)
    second_reasoning = _reasoning_items(second_items)

    assert [item.text_delta for item in first_reasoning] == ["first"]
    assert [item.text_delta for item in second_reasoning] == ["second"]
    assert first_reasoning[0].segment_instance_ordinal == 0
    assert second_reasoning[0].segment_instance_ordinal == 0


def test_private_capable_client_streams_summary_end_to_end() -> None:
    sdk_events = [
        _RESPONSE_STREAM_EVENT.validate_python(event)
        for event in _trace_events("one_part")
    ]
    create = AsyncMock(return_value=_AsyncEvents(sdk_events))
    stream = run(
        _client(create)(
            "plain-model",
            [],
            _settings(ReasoningSummaryMode.CONCISE),
        )
    )
    items = run(_consume(stream))

    create.assert_awaited_once()
    assert create.await_args.kwargs["reasoning"] == {"summary": "concise"}
    assert [item.text_delta for item in _reasoning_items(items)] == [
        "Inspect inputs."
    ]


def test_phase4_public_summary_capability_remains_dormant() -> None:
    create = AsyncMock()
    client = _client(create, capable=False)

    with pytest.raises(ReasoningSummaryCapabilityError):
        run(
            client(
                "plain-model",
                [],
                _settings(ReasoningSummaryMode.AUTO),
            )
        )
    create.assert_not_awaited()


def test_summary_item_and_part_validation_is_strict() -> None:
    cases: list[tuple[list[object], str, str]] = []

    invalid_status = _summary_item_added()
    cast(dict[str, object], invalid_status["item"])["status"] = "completed"
    cases.append(([invalid_status], "item.status", "unexpected_value"))

    invalid_initial_summary = _summary_item_added()
    cast(dict[str, object], invalid_initial_summary["item"])["summary"] = [
        {"type": "summary_text", "text": "private"}
    ]
    cases.append(
        (
            [invalid_initial_summary],
            "item.summary",
            "unexpected_value",
        )
    )

    identity_conflict = _summary_item_added("rs_test", 1)
    cases.append(
        (
            [_summary_item_added(), identity_conflict],
            "item.identity",
            "conflict",
        )
    )

    invalid_part_type = _summary_part_added()
    cast(dict[str, object], invalid_part_type["part"])["type"] = "private"
    cases.append(
        (
            [_summary_item_added(), invalid_part_type],
            "part.type",
            "unexpected_value",
        )
    )

    duplicate_closed_part = [
        _summary_item_added(),
        _summary_part_added(),
        _summary_text_done(""),
        _summary_part_done(""),
        _summary_part_added(),
    ]
    cases.append((duplicate_closed_part, "part", "closed"))

    non_string_text_done = [
        _summary_item_added(),
        _summary_part_added(),
        _summary_text_done({"private": True}),
    ]
    cases.append((non_string_text_done, "text", "mapping"))

    conflicting_text_done = [
        _summary_item_added(),
        _summary_part_added(),
        _summary_delta("prefix"),
        _summary_text_done("different"),
    ]
    cases.append((conflicting_text_done, "text", "conflict"))

    invalid_part_done_type = _summary_part_done("")
    cast(dict[str, object], invalid_part_done_type["part"])["type"] = "private"
    cases.append(
        (
            [
                _summary_item_added(),
                _summary_part_added(),
                _summary_text_done(""),
                invalid_part_done_type,
            ],
            "part.type",
            "unexpected_value",
        )
    )

    cases.append(
        (
            [
                _summary_item_added(),
                _summary_part_added(),
                _summary_text_done(""),
                _summary_part_done(["private"]),
            ],
            "part.text",
            "sequence",
        )
    )

    cases.append(
        (
            [
                _summary_item_added(),
                _summary_part_added(),
                _summary_delta("prefix"),
                _summary_text_done("prefix"),
                _summary_part_done("different"),
            ],
            "part.text",
            "conflict",
        )
    )

    for events, expected_field, expected_shape in cases:
        items = run(_consume(OpenAIStream(_AsyncEvents(events))))
        terminal = _error_item(items)
        data = cast(dict[str, Any], terminal.data)
        assert data["error"]["field"] == expected_field
        assert data["error"]["value_shape"] == expected_shape
        assert terminal.provider_payload is None

    duplicate_open = _summary_item_added()
    duplicate_part = _summary_part_added()
    duplicate_items = run(
        _consume(
            OpenAIStream(
                _AsyncEvents(
                    [
                        duplicate_open,
                        deepcopy(duplicate_open),
                        duplicate_part,
                        deepcopy(duplicate_part),
                        _summary_text_done(""),
                        _summary_part_done(""),
                        _summary_item_done([""]),
                        _completed_event(),
                    ]
                )
            )
        )
    )
    assert not any(
        item.kind is StreamItemKind.STREAM_ERRORED for item in duplicate_items
    )

    completed_then_added = [
        _summary_item_added(),
        _summary_item_done([]),
        _summary_item_added(),
    ]
    terminal = _error_item(
        run(_consume(OpenAIStream(_AsyncEvents(completed_then_added))))
    )
    assert (
        cast(dict[str, Any], terminal.data)["error"]["value_shape"] == "closed"
    )


def test_summary_completed_item_validation_is_strict() -> None:
    cases: list[tuple[list[object], str, str]] = []

    output_conflict = _summary_item_done([], output_index=1)
    cases.append(
        (
            [_summary_item_added(), output_conflict],
            "item.identity",
            "conflict",
        )
    )

    invalid_status = _summary_item_done([])
    cast(dict[str, object], invalid_status["item"])["status"] = "incomplete"
    cases.append(
        (
            [_summary_item_added(), invalid_status],
            "item.status",
            "unexpected_value",
        )
    )

    cases.append(
        (
            [
                _summary_item_added(),
                _summary_part_added(),
                _summary_item_done([]),
            ],
            "part",
            "open",
        )
    )

    cases.append(
        (
            [
                _summary_item_added(),
                _summary_part_added(),
                _summary_delta("streamed"),
                _summary_text_done("streamed"),
                _summary_part_done("streamed"),
                _summary_item_done(["different"]),
            ],
            "item.summary",
            "conflict",
        )
    )

    missing_position = [
        _summary_item_added(),
        _summary_part_added(summary_index=1),
        _summary_delta("streamed", summary_index=1),
        _summary_text_done("streamed", summary_index=1),
        _summary_part_done("streamed", summary_index=1),
        _summary_item_done([]),
    ]
    cases.append((missing_position, "item.summary", "missing_position"))

    invalid_summary = _summary_item_done([])
    cast(dict[str, object], invalid_summary["item"])["summary"] = "private"
    cases.append(
        (
            [_summary_item_added(), invalid_summary],
            "item.summary",
            "string",
        )
    )

    invalid_part_type = _summary_item_done([""])
    summary = cast(
        list[dict[str, object]],
        cast(dict[str, object], invalid_part_type["item"])["summary"],
    )
    summary[0]["type"] = "private"
    cases.append(
        (
            [_summary_item_added(), invalid_part_type],
            "item.summary.type",
            "unexpected_value",
        )
    )

    invalid_part_text = _summary_item_done([""])
    summary = cast(
        list[dict[str, object]],
        cast(dict[str, object], invalid_part_text["item"])["summary"],
    )
    summary[0]["text"] = {"private": True}
    cases.append(
        (
            [_summary_item_added(), invalid_part_text],
            "item.summary.text",
            "mapping",
        )
    )

    for events, expected_field, expected_shape in cases:
        terminal = _error_item(
            run(_consume(OpenAIStream(_AsyncEvents(events))))
        )
        data = cast(dict[str, Any], terminal.data)
        assert data["error"]["field"] == expected_field
        assert data["error"]["value_shape"] == expected_shape

    empty_fallback_items = run(
        _consume(
            OpenAIStream(
                _AsyncEvents(
                    [
                        _summary_item_added(),
                        _summary_item_done([""]),
                        _completed_event(),
                    ]
                )
            )
        )
    )
    assert _reasoning_items(empty_fallback_items) == []


def test_summary_terminal_and_source_closure_are_deterministic() -> None:
    stream = OpenAIStream(_AsyncEvents([]))
    completed = _completed_event()
    stream._provider_events_from_event(completed)
    assert stream._provider_events_from_event(completed)
    stream._reasoning_summary_state.close_source()

    with pytest.raises(
        openai_module._OpenAIReasoningSummaryEventError
    ) as conflicting_terminal:
        stream._provider_events_from_event(
            {
                "type": "response.incomplete",
                "response": {
                    "status": "incomplete",
                    "incomplete_details": {"reason": "late"},
                },
            }
        )
    assert conflicting_terminal.value.field == "response"

    open_item_stream = OpenAIStream(_AsyncEvents([]))
    open_item_stream._provider_events_from_event(_summary_item_added())
    with pytest.raises(
        openai_module._OpenAIReasoningSummaryEventError
    ) as open_completion:
        open_item_stream._provider_events_from_event(_completed_event())
    assert open_completion.value.value_shape == "open"

    completed_then_failed = [
        _summary_item_added(),
        _summary_item_done([]),
        {
            "type": "response.failed",
            "response": {"status": "failed", "error": None, "output": []},
        },
    ]
    items = run(_consume(OpenAIStream(_AsyncEvents(completed_then_failed))))
    assert _error_item(items).provider_event_type == "response.failed"

    exhausted_items = run(
        _consume(OpenAIStream(_AsyncEvents([_summary_item_added()])))
    )
    exhausted = _error_item(exhausted_items)
    assert exhausted.provider_event_type == "response.source_exhausted"
    assert (
        cast(dict[str, Any], exhausted.data)["error"]["value_shape"] == "open"
    )


def test_summary_unreadable_fields_and_adapter_passthrough_are_safe() -> None:
    class UnreadableItemEvent:
        type = "response.output_item.added"
        output_index = 0

        @property
        def item(self) -> object:
            raise RuntimeError("private unreadable value")

    unreadable_items = run(
        _consume(OpenAIStream(_AsyncEvents([UnreadableItemEvent()])))
    )
    _assert_preparse_private_error(
        unreadable_items,
        event_type="response.unknown",
    )

    class SafeAdapterErrorStream(OpenAIStream):
        def _provider_events_from_event(
            self,
            event: object,
        ) -> tuple[StreamProviderEvent, ...]:
            raise StreamProviderAdapterError(
                ValueError("safe adapter error"),
                provider_payload=None,
                provider_event_type="response.safe_test",
                safe_data={"error": {"code": "safe_adapter_error"}},
            )

    passthrough_items = run(
        _consume(
            SafeAdapterErrorStream(
                _AsyncEvents([{"type": "response.safe_test"}])
            )
        )
    )
    passthrough = _error_item(passthrough_items)
    assert passthrough.data == {"error": {"code": "safe_adapter_error"}}
    assert passthrough.provider_event_type == "response.safe_test"


def test_summary_provider_payload_never_reaches_outward_surfaces() -> None:
    provider_secret = "SUMMARY_PROVIDER_PAYLOAD_PRIVATE_SENTINEL"
    delta = _summary_delta("visible-summary")
    delta["metadata"] = {"private_echo": provider_secret}
    items = run(
        _consume(
            OpenAIStream(
                _AsyncEvents(
                    [
                        _summary_item_added(),
                        _summary_part_added(),
                        delta,
                        _summary_text_done("visible-summary"),
                        _summary_part_done("visible-summary"),
                        _summary_item_done(["visible-summary"]),
                        _completed_event(),
                    ]
                )
            )
        )
    )
    reasoning = _reasoning_items(items)
    assert len(reasoning) == 1
    item = reasoning[0]
    projection = project_canonical_stream_item(item)
    observability = stream_observability_payload(item)
    outward = "\n".join(
        (
            repr(item),
            repr(item.to_trace_dict()),
            repr(projection),
            repr(observability),
            repr([stream_item.to_trace_dict() for stream_item in items]),
        )
    )

    assert item.text_delta == "visible-summary"
    assert item.provider_payload is None
    assert not hasattr(projection, "provider_payload")
    assert "has_provider_payload" not in cast(
        dict[str, object], observability["summary"]
    )
    assert all(stream_item.provider_payload is None for stream_item in items)
    assert provider_secret not in outward


def test_hostile_summary_sdk_access_is_structured_and_content_free() -> None:
    type_secret = "HOSTILE_EVENT_TYPE_PRIVATE_SENTINEL"

    class HostileTypeEvent:
        @property
        def type(self) -> object:
            raise RuntimeError(type_secret)

    type_items = run(
        _consume(OpenAIStream(_AsyncEvents([HostileTypeEvent()])))
    )
    type_terminal = _assert_preparse_private_error(
        type_items,
        event_type="response.unknown",
    )

    dump_secret = "HOSTILE_MODEL_DUMP_PRIVATE_SENTINEL"

    class HostileModelDumpEvent:
        type = "response.reasoning_summary_text.delta"
        item_id = "rs_1"
        output_index = 0
        summary_index = 0
        delta = "must-not-emit"

        @property
        def model_dump(self) -> object:
            raise RuntimeError(dump_secret)

    dump_items = run(
        _consume(
            OpenAIStream(
                _AsyncEvents(
                    [
                        _summary_item_added(),
                        _summary_part_added(),
                        HostileModelDumpEvent(),
                    ]
                )
            )
        )
    )
    dump_terminal = _assert_preparse_private_error(
        dump_items,
        event_type="response.unknown",
    )

    nested_secret = "HOSTILE_SUMMARY_PART_PRIVATE_SENTINEL"

    class HostileSummaryPart:
        type = "summary_text"

        @property
        def text(self) -> object:
            raise RuntimeError(nested_secret)

    nested_done = _summary_item_done([])
    cast(dict[str, object], nested_done["item"])["summary"] = [
        HostileSummaryPart()
    ]
    nested_items = run(
        _consume(
            OpenAIStream(_AsyncEvents([_summary_item_added(), nested_done]))
        )
    )
    nested_terminal = _assert_preparse_private_error(
        nested_items,
        event_type="response.output_item.done",
    )
    outward = repr(
        [
            type_terminal.to_trace_dict(),
            dump_terminal.to_trace_dict(),
            nested_terminal.to_trace_dict(),
        ]
    )
    assert type_secret not in outward
    assert dump_secret not in outward
    assert nested_secret not in outward


def test_live_summary_terminal_and_cleanup_echoes_are_sanitized() -> None:
    provider_secret = "LIVE_SUMMARY_PROVIDER_TERMINAL_SENTINEL"
    cleanup_secret = "LIVE_SUMMARY_CLEANUP_PRIVATE_SENTINEL"
    source = _AsyncEvents(
        [
            _summary_item_added(),
            _summary_part_added(),
            _summary_delta("visible-summary"),
            {
                "type": "response.failed",
                "response": {
                    "status": "failed",
                    "error": None,
                    "message": provider_secret,
                    "output": [],
                },
                "metadata": {"private_echo": provider_secret},
            },
        ]
    )
    source.aclose = AsyncMock(side_effect=RuntimeError(cleanup_secret))
    items, error = run(_consume_with_error(OpenAIStream(source)))
    terminal = _error_item(items)
    outward = "\n".join(
        (
            repr(items),
            repr([item.to_trace_dict() for item in items]),
            repr(error),
        )
    )

    assert error is None
    assert terminal.provider_payload is None
    assert terminal.provider_event_type == "response.failed"
    assert terminal.data == {
        "error": {
            "type": "server_error",
            "code": "openai_provider_request_failed",
            "status": "failed",
            "message": "OpenAI provider request failed",
        }
    }
    assert provider_secret not in outward
    assert cleanup_secret not in outward
    source.aclose.assert_awaited_once_with()


def test_summary_source_exhaustion_beats_private_cleanup_failure() -> None:
    cleanup_secret = "SOURCE_EXHAUSTION_CLEANUP_PRIVATE_SENTINEL"
    source = _AsyncEvents(
        [
            _summary_item_added(),
            _summary_part_added(),
            _summary_delta("visible-summary"),
        ]
    )
    source.aclose = AsyncMock(side_effect=RuntimeError(cleanup_secret))
    items, error = run(_consume_with_error(OpenAIStream(source)))
    terminal = _error_item(items)
    terminal_data = cast(dict[str, Any], terminal.data)
    outward = "\n".join(
        (
            repr([item.to_trace_dict() for item in items]),
            repr(error),
        )
    )

    assert error is None
    assert terminal.provider_payload is None
    assert terminal.provider_event_type == "response.source_exhausted"
    assert terminal_data["error"] == {
        "type": "invalid_provider_event",
        "code": "invalid_reasoning_summary_event",
        "message": "OpenAI reasoning summary event is invalid.",
        "event_type": "response.source_exhausted",
        "field": "item",
        "index": {"output_index": 0, "summary_index": None},
        "value_shape": "open",
    }
    assert cleanup_secret not in outward
    source.aclose.assert_awaited_once_with()


def test_private_replay_cancellation_event_is_content_free() -> None:
    secret = "PRIVATE_REPLAY_CANCEL_EVENT_SENTINEL"
    sanitized = OpenAIStream._sanitize_private_replay_event(
        StreamProviderEvent(
            kind=StreamItemKind.STREAM_CANCELLED,
            data={"reason": secret},
            provider_payload={"echo": secret},
        )
    )

    assert sanitized.provider_payload is None
    assert sanitized.data == {
        "error": {
            "type": "server_error",
            "code": "openai_provider_request_failed",
            "status": "cancelled",
            "message": "OpenAI provider request cancelled",
        }
    }
    assert secret not in repr(sanitized)


def test_openai_sdk_rejects_invented_provider_cancellation_event() -> None:
    event = {"type": "response.cancelled", "reason": "provider"}

    with pytest.raises(ValidationError):
        _RESPONSE_STREAM_EVENT.validate_python(event)
    stream = OpenAIStream(_AsyncEvents([]))
    assert stream._provider_events_from_event(event) == ()


def test_second_event_type_access_failure_cannot_leak_provider_data() -> None:
    secret = "SECOND_EVENT_TYPE_ACCESS_PRIVATE_SENTINEL"

    class OneShotTypeEvent:
        delta = 7

        def __init__(self) -> None:
            self.read_count = 0

        @property
        def type(self) -> object:
            self.read_count += 1
            if self.read_count == 1:
                return "response.output_text.delta"
            raise RuntimeError(secret)

    event = OneShotTypeEvent()
    items = run(_consume(OpenAIStream(_AsyncEvents([event]))))
    terminal = _error_item(items)
    outward = repr([terminal, terminal.to_trace_dict()])

    assert event.read_count == 0
    assert terminal.provider_payload is None
    assert terminal.provider_event_type == "response.unknown"
    _assert_preparse_private_error(
        items,
        event_type="response.unknown",
    )
    assert secret not in outward


def test_terminal_snapshots_revalidate_without_replaying_done_items() -> None:
    message_done = _message_item_event(
        "response.output_item.done", "message-duplicate", 0, "answer"
    )
    tool_done = _tool_item_done("tool-duplicate", 1, '{"value":1}')
    completed = {
        "type": "response.completed",
        "response": {
            "usage": {},
            "output": [
                deepcopy(cast(dict[str, object], message_done["item"])),
                deepcopy(cast(dict[str, object], tool_done["item"])),
            ],
        },
    }
    items = run(
        _consume(
            OpenAIStream(
                _AsyncEvents(
                    [
                        message_done,
                        deepcopy(message_done),
                        tool_done,
                        deepcopy(tool_done),
                        completed,
                    ]
                )
            )
        )
    )

    assert not any(
        item.kind is StreamItemKind.STREAM_ERRORED for item in items
    )
    assert accumulate_canonical_stream_items(items).answer_text == "answer"
    assert sum(item.kind is StreamItemKind.ANSWER_DONE for item in items) == 1
    assert (
        sum(item.kind is StreamItemKind.TOOL_CALL_DONE for item in items) == 1
    )

    for response_index in (0, 1):
        fixture_items = run(
            _consume(
                OpenAIStream(
                    _AsyncEvents(
                        _trace_events(
                            "tools_answer", response_index=response_index
                        )
                    )
                )
            )
        )
        assert not any(
            item.kind is StreamItemKind.STREAM_ERRORED
            for item in fixture_items
        )


def test_done_item_conflicts_are_rejected_before_duplicate_side_effects() -> (
    None
):
    secret = "DONE_ITEM_CONFLICT_PRIVATE_SENTINEL"
    message_done = _message_item_event(
        "response.output_item.done", "message-conflict", 0, "safe"
    )
    message_conflict = deepcopy(message_done)
    content = cast(
        list[dict[str, object]],
        cast(dict[str, object], message_conflict["item"])["content"],
    )
    content[0]["text"] = secret

    tool_done = _tool_item_done("tool-conflict", 0, '{"value":1}')
    tool_conflicts: list[dict[str, object]] = []
    for field, value in (
        ("call_id", f"call-{secret}"),
        ("name", f"tool-{secret}"),
        ("arguments", dumps({"private": secret})),
    ):
        conflict = deepcopy(tool_done)
        cast(dict[str, object], conflict["item"])[field] = value
        tool_conflicts.append(conflict)

    cases = (
        ([message_done, message_conflict], StreamItemKind.ANSWER_DONE),
        *(
            ([deepcopy(tool_done), conflict], StreamItemKind.TOOL_CALL_DONE)
            for conflict in tool_conflicts
        ),
    )
    for events, side_effect_kind in cases:
        items = run(_consume(OpenAIStream(_AsyncEvents(list(events)))))
        terminal = _error_item(items)
        data = cast(dict[str, Any], terminal.data)
        assert data["error"]["code"] == "invalid_reasoning_summary_event"
        assert sum(item.kind is side_effect_kind for item in items) <= 1
        assert secret not in repr([terminal, terminal.to_trace_dict()])


def test_terminal_output_cannot_omit_or_conflict_with_completed_items() -> (
    None
):
    secret = "TERMINAL_COMPLETED_ITEM_PRIVATE_SENTINEL"
    completed_items = (
        [
            _message_item_event(
                "response.output_item.done", "terminal-message", 0, "safe"
            )
        ],
        [_tool_item_done("terminal-tool", 0, '{"value":1}')],
        [
            _summary_item_added("terminal-reasoning", 0),
            _summary_item_done([], "terminal-reasoning", 0),
        ],
    )
    for prefix in completed_items:
        omitted = run(
            _consume(
                OpenAIStream(
                    _AsyncEvents(
                        [
                            *deepcopy(prefix),
                            {
                                "type": "response.completed",
                                "response": {"usage": {}, "output": []},
                            },
                        ]
                    )
                )
            )
        )
        terminal = _error_item(omitted)
        assert (
            cast(dict[str, Any], terminal.data)["error"]["code"]
            == "invalid_reasoning_summary_event"
        )

        legacy = run(
            _consume(
                OpenAIStream(
                    _AsyncEvents([*deepcopy(prefix), _completed_event()])
                )
            )
        )
        assert not any(
            item.kind is StreamItemKind.STREAM_ERRORED for item in legacy
        )

    for first in (
        _message_item_event(
            "response.output_item.done", "terminal-message-conflict", 0, "safe"
        ),
        _tool_item_done("terminal-tool-conflict", 0, '{"value":1}'),
    ):
        conflicting_item = deepcopy(cast(dict[str, object], first["item"]))
        if conflicting_item["type"] == "message":
            cast(list[dict[str, object]], conflicting_item["content"])[0][
                "text"
            ] = secret
        else:
            conflicting_item["arguments"] = dumps({"private": secret})
        items = run(
            _consume(
                OpenAIStream(
                    _AsyncEvents(
                        [
                            first,
                            {
                                "type": "response.completed",
                                "response": {
                                    "usage": {},
                                    "output": [conflicting_item],
                                },
                            },
                        ]
                    )
                )
            )
        )
        terminal = _error_item(items)
        assert (
            cast(dict[str, Any], terminal.data)["error"]["code"]
            == "invalid_reasoning_summary_event"
        )
        assert secret not in repr([terminal, terminal.to_trace_dict()])


def test_preparse_privacy_rejects_hostile_container_graph_without_calls() -> (
    None
):
    secret = "PREPARSE_CONTAINER_PRIVATE_SENTINEL"

    class HostileString(str):
        calls = 0

        def _fail(self, *_: object, **__: object) -> object:
            type(self).calls += 1
            raise RuntimeError(secret)

        __eq__ = _fail
        __hash__ = _fail
        __str__ = _fail
        encode = _fail
        startswith = _fail

    class HostileMapping(dict[object, object]):
        calls = 0

        def _fail(self, *_: object, **__: object) -> object:
            type(self).calls += 1
            raise RuntimeError(secret)

        get = _fail
        items = _fail
        values = _fail
        __getitem__ = _fail
        __iter__ = _fail

    class HostileList(list[object]):
        calls = 0

        def _fail(self, *_: object, **__: object) -> object:
            type(self).calls += 1
            raise RuntimeError(secret)

        __getitem__ = _fail
        __iter__ = _fail
        __len__ = _fail

    class HostileObject:
        __module__ = "openai.types.responses.spoof"
        calls = 0

        @property
        def type(self) -> object:
            type(self).calls += 1
            raise RuntimeError(secret)

    events: tuple[object, ...] = (
        HostileObject(),
        HostileMapping(
            {
                "type": "response.output_text.delta",
                "delta": secret,
            }
        ),
        {"metadata": {"encrypted_content": secret}},
        {"type": 7, "metadata": {"summary": secret}},
        {
            "type": HostileString("response.output_text.delta"),
            "delta": secret,
        },
        {
            "type": "response.failed",
            "response": HostileMapping({"status": "failed"}),
        },
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": HostileMapping({"type": "reasoning"}),
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "id": "hostile-content",
                "type": "message",
                "content": HostileList([]),
            },
        },
        {
            "type": "response.completed",
            "response": {"usage": {}, "output": HostileList([])},
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "id": "tuple-content",
                "type": "message",
                "content": ({"type": "output_text", "text": secret},),
            },
        },
    )
    for case_index, event in enumerate(events):
        HostileString.calls = 0
        HostileMapping.calls = 0
        HostileList.calls = 0
        HostileObject.calls = 0
        items = run(_consume(OpenAIStream(_AsyncEvents([event]))))
        errors = [
            item
            for item in items
            if item.kind is StreamItemKind.STREAM_ERRORED
        ]
        assert len(errors) == 1, (case_index, [item.kind for item in items])
        terminal = errors[0]
        assert terminal.provider_payload is None
        assert secret not in repr([terminal, terminal.to_trace_dict()])
        assert HostileString.calls == 0, case_index
        assert HostileMapping.calls == 0, case_index
        assert HostileList.calls == 0, case_index
        assert HostileObject.calls == 0, case_index


def test_preparse_privacy_covers_private_terminal_and_cleanup_matrix() -> None:
    private = "PREPARSE_TERMINAL_PRIVATE_SENTINEL"
    reasoning = {
        "id": "private-reasoning",
        "type": "reasoning",
        "status": "completed",
        "summary": [{"type": "summary_text", "text": private}],
        "encrypted_content": private,
    }
    events = (
        {
            "type": "response.failed",
            "response": {
                "status": "failed",
                "error": {"message": private},
                "output": [reasoning],
            },
        },
        {
            "type": "response.incomplete",
            "response": {
                "status": "incomplete",
                "incomplete_details": {"reason": private},
                "output": [reasoning],
            },
        },
        {
            "type": "error",
            "error": {"message": private},
            "response": {"output": [reasoning]},
        },
        {
            "type": "response.failed",
            "response": {
                "status": "failed",
                "error": None,
                "output": [],
            },
            "metadata": {
                "nested": [{"payload": {"encrypted_content": private}}]
            },
        },
    )
    for event in events:
        items = run(_consume(OpenAIStream(_AsyncEvents([deepcopy(event)]))))
        terminal = _error_item(items)
        assert terminal.provider_payload is None
        assert private not in repr([terminal, terminal.to_trace_dict()])

    cleanup = "PREPARSE_CLEANUP_PRIVATE_SENTINEL"
    source = _AsyncEvents(
        [
            {
                "metadata": {
                    "nested": [{"summary": private}],
                }
            }
        ]
    )
    source.aclose = AsyncMock(side_effect=RuntimeError(cleanup))
    items, error = run(_consume_with_error(OpenAIStream(source)))
    terminal = _error_item(items)
    terminal_error = cast(dict[str, Any], terminal.data)["error"]
    outward = repr([items, [item.to_trace_dict() for item in items], error])

    assert error is None
    assert terminal_error["code"] != "openai_cleanup_failed"
    assert terminal.provider_payload is None
    assert private not in outward
    assert cleanup not in outward
    source.aclose.assert_awaited_once_with()


def test_non_private_omitted_summary_payload_remains_compatible() -> None:
    message_done = _message_item_event(
        "response.output_item.done", "safe-omitted-summary", 0, "safe-answer"
    )
    completed = {
        "type": "response.completed",
        "response": {
            "usage": {},
            "metadata": {"safe": ["diagnostic", {"count": 1}]},
            "output": [
                deepcopy(cast(dict[str, object], message_done["item"]))
            ],
        },
    }
    items = run(
        _consume(OpenAIStream(_AsyncEvents([message_done, completed])))
    )

    assert not any(
        item.kind is StreamItemKind.STREAM_ERRORED for item in items
    )
    assert (
        accumulate_canonical_stream_items(items).answer_text == "safe-answer"
    )


def test_native_reasoning_content_indices_reconcile_and_interleave() -> None:
    item_id = "native-parts"
    events = [
        _summary_item_added(item_id, 0),
        {
            "type": "response.reasoning_text.delta",
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "delta": "native-zero",
        },
        {
            "type": "response.reasoning_text.done",
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "text": "native-zero",
        },
        {
            "type": "response.reasoning_text.done",
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "text": "native-zero",
        },
        _summary_part_added(item_id, 0, 0),
        _summary_delta("summary-middle", item_id, 0, 0),
        _summary_text_done("summary-middle", item_id, 0, 0),
        _summary_part_done("summary-middle", item_id, 0, 0),
        {
            "type": "response.reasoning_text.delta",
            "item_id": item_id,
            "output_index": 0,
            "content_index": 1,
            "delta": "native-one",
        },
        {
            "type": "response.reasoning_text.done",
            "item_id": item_id,
            "output_index": 0,
            "content_index": 1,
            "text": "native-one",
        },
        _summary_item_done(["summary-middle"], item_id, 0),
    ]
    items = run(_consume(OpenAIStream(_AsyncEvents(events))))
    reasoning_items = _reasoning_items(items)

    assert not any(
        item.kind is StreamItemKind.STREAM_ERRORED for item in items
    )
    assert [item.text_delta for item in reasoning_items] == [
        "native-zero",
        "summary-middle",
        "native-one",
    ]
    assert [item.reasoning_representation for item in reasoning_items] == [
        StreamReasoningRepresentation.NATIVE_TEXT,
        StreamReasoningRepresentation.SUMMARY,
        StreamReasoningRepresentation.NATIVE_TEXT,
    ]
    assert [item.segment_instance_ordinal for item in reasoning_items] == [
        0,
        1,
        2,
    ]


def test_native_reasoning_content_index_and_done_conflicts_are_strict() -> (
    None
):
    secret = "NATIVE_CONTENT_INDEX_PRIVATE_SENTINEL"
    for event_type, text_field in (
        ("response.reasoning_text.delta", "delta"),
        ("response.reasoning_text.done", "text"),
    ):
        for invalid in (True, -1, 0.5, "0", SimpleNamespace(value=0)):
            items = run(
                _consume(
                    OpenAIStream(
                        _AsyncEvents(
                            [
                                _summary_item_added("native-invalid", 0),
                                {
                                    "type": event_type,
                                    "item_id": "native-invalid",
                                    "output_index": 0,
                                    "content_index": invalid,
                                    text_field: secret,
                                },
                            ]
                        )
                    )
                )
            )
            terminal = _assert_structured_summary_error(
                items,
                event_type=event_type,
                field="content_index",
                value_shape=(
                    "boolean"
                    if type(invalid) is bool
                    else (
                        "negative_integer"
                        if type(invalid) is int
                        else (
                            "float"
                            if type(invalid) is float
                            else "string" if type(invalid) is str else "object"
                        )
                    )
                ),
                output_index=0,
                summary_index=None,
            )
            assert secret not in repr([terminal, terminal.to_trace_dict()])

    legacy_items = run(
        _consume(
            OpenAIStream(
                _AsyncEvents(
                    [
                        {
                            "type": "response.reasoning_text.delta",
                            "item_id": "native-legacy-index",
                            "output_index": 0,
                            "delta": "legacy",
                        },
                        {
                            "type": "response.reasoning_text.done",
                            "item_id": "native-legacy-index",
                            "output_index": 0,
                            "content_index": None,
                            "text": "legacy",
                        },
                        _completed_event(),
                    ]
                )
            )
        )
    )
    assert not any(
        item.kind is StreamItemKind.STREAM_ERRORED for item in legacy_items
    )
    assert [item.text_delta for item in _reasoning_items(legacy_items)] == [
        "legacy"
    ]
    identity_free = OpenAIStream(_AsyncEvents([]))
    assert (
        identity_free._provider_events_from_event(
            {
                "type": "response.reasoning_text.delta",
                "delta": "",
            }
        )
        == ()
    )
    run(identity_free.aclose())

    prefix: list[object] = [
        _summary_item_added("native-conflict", 0),
        {
            "type": "response.reasoning_text.delta",
            "item_id": "native-conflict",
            "output_index": 0,
            "content_index": 0,
            "delta": "safe",
        },
    ]
    for suffix in (
        [
            {
                "type": "response.reasoning_text.done",
                "item_id": "native-conflict",
                "output_index": 0,
                "content_index": 0,
                "text": secret,
            }
        ],
        [
            {
                "type": "response.reasoning_text.done",
                "item_id": "native-conflict",
                "output_index": 0,
                "content_index": 0,
                "text": "safe",
            },
            {
                "type": "response.reasoning_text.done",
                "item_id": "native-conflict",
                "output_index": 0,
                "content_index": 0,
                "text": secret,
            },
        ],
        [
            {
                "type": "response.reasoning_text.done",
                "item_id": "native-conflict",
                "output_index": 0,
                "content_index": 0,
                "text": "safe",
            },
            {
                "type": "response.reasoning_text.delta",
                "item_id": "native-conflict",
                "output_index": 0,
                "content_index": 0,
                "delta": secret,
            },
        ],
    ):
        items = run(
            _consume(OpenAIStream(_AsyncEvents([*deepcopy(prefix), *suffix])))
        )
        terminal = _error_item(items)
        assert (
            cast(dict[str, Any], terminal.data)["error"]["code"]
            == "invalid_reasoning_summary_event"
        )
        assert secret not in repr([terminal, terminal.to_trace_dict()])


def test_native_content_indices_do_not_define_canonical_segments() -> None:
    def native_delta(content_index: int, text: str) -> dict[str, object]:
        return {
            "type": "response.reasoning_text.delta",
            "item_id": "native-contiguous",
            "output_index": 0,
            "content_index": content_index,
            "delta": text,
        }

    def native_done(content_index: int, text: str) -> dict[str, object]:
        return {
            "type": "response.reasoning_text.done",
            "item_id": "native-contiguous",
            "output_index": 0,
            "content_index": content_index,
            "text": text,
        }

    without_boundaries = run(
        _consume(
            OpenAIStream(
                _AsyncEvents(
                    [
                        _summary_item_added("native-contiguous", 0),
                        native_delta(0, "A"),
                        native_delta(1, "B"),
                        native_delta(0, "A2"),
                    ]
                )
            )
        )
    )
    without_reasoning = _reasoning_items(without_boundaries)
    assert [item.text_delta for item in without_reasoning] == ["A", "B", "A2"]
    assert [item.segment_instance_ordinal for item in without_reasoning] == [
        0,
        0,
        0,
    ]

    with_boundaries = run(
        _consume(
            OpenAIStream(
                _AsyncEvents(
                    [
                        _summary_item_added("native-contiguous", 0),
                        native_delta(0, "A"),
                        native_delta(1, "B"),
                        native_delta(0, "A2"),
                        native_done(0, "AA2"),
                        native_delta(2, "C"),
                    ]
                )
            )
        )
    )
    with_reasoning = _reasoning_items(with_boundaries)
    assert [item.text_delta for item in with_reasoning] == [
        "A",
        "B",
        "A2",
        "C",
    ]
    assert [item.segment_instance_ordinal for item in with_reasoning] == [
        0,
        0,
        0,
        1,
    ]


def test_unified_terminal_finalizer_applies_cleanup_precedence() -> None:
    cleanup_secret = "PUBLIC_TERMINAL_CLEANUP_PRIVATE_SENTINEL"
    terminal_cases = (
        (
            _completed_event(),
            None,
            "openai_cleanup_failed",
        ),
        (
            {
                "type": "response.failed",
                "response": {
                    "status": "failed",
                    "error": None,
                    "output": [],
                },
            },
            "response.failed",
            None,
        ),
        (
            {
                "type": "response.incomplete",
                "response": {
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "output": [],
                },
            },
            "response.incomplete",
            None,
        ),
        (
            {"type": "response.output_text.delta", "delta": 7},
            "response.output_text.delta",
            None,
        ),
    )
    for terminal_event, expected_event_type, expected_code in terminal_cases:
        source = _AsyncEvents([terminal_event])
        source.aclose = AsyncMock(side_effect=RuntimeError(cleanup_secret))
        items, error = run(_consume_with_error(OpenAIStream(source)))
        terminal = _error_item(items)
        terminal_data = cast(dict[str, Any], terminal.data)
        outward = repr(
            [
                items,
                [item.to_trace_dict() for item in items],
                error,
            ]
        )

        assert error is None
        assert terminal.provider_event_type == expected_event_type
        if expected_code is not None:
            assert terminal_data["error"]["code"] == expected_code
        assert cleanup_secret not in outward
        source.aclose.assert_awaited_once_with()


def test_eof_cleanup_failure_rolls_back_replay_success() -> None:
    cleanup_secret = "EOF_REPLAY_CLEANUP_PRIVATE_SENTINEL"
    owner = _owner()
    owner.admit(_reasoning_item("eof-input", "eof-cipher", []))
    retained = MagicMock()
    released = MagicMock()
    output_items: list[dict[str, Any]] = []
    rollback_counts: list[int] = []

    def rollback(count: int) -> None:
        rollback_counts.append(count)
        del output_items[-count:]
        raise RuntimeError("rollback callback failed")

    source = _AsyncEvents([_output_done(_function_item("eof-call"))])
    source.aclose = AsyncMock(side_effect=RuntimeError(cleanup_secret))
    stream = OpenAIStream(
        source,
        output_item_sink=output_items.append,
        output_item_rollback=rollback,
        replay_owner=owner,
        replay_owner_retainer=retained,
        replay_owner_releaser=released,
    )
    items = run(_consume(stream))
    terminal = _error_item(items)
    outward = repr([items, [item.to_trace_dict() for item in items]])

    assert (
        cast(dict[str, Any], terminal.data)["error"]["code"]
        == "openai_cleanup_failed"
    )
    assert cleanup_secret not in outward
    retained.assert_not_called()
    released.assert_called_once_with(owner)
    assert owner.released
    assert owner.release_count == 1
    assert owner.item_count == 0
    assert output_items == []
    assert rollback_counts == [1]
    source.aclose.assert_awaited_once_with()


def test_success_terminal_commits_output_sink_checkpoint() -> None:
    output_items: list[dict[str, Any]] = []
    rollback = MagicMock(
        side_effect=RuntimeError("SUCCESS_ROLLBACK_MUST_NOT_RUN")
    )
    source = _AsyncEvents(
        [
            _output_done(_function_item("success-sink")),
            _completed_event(),
        ]
    )
    items, error = run(
        _consume_with_error(
            OpenAIStream(
                source,
                output_item_sink=output_items.append,
                output_item_rollback=rollback,
            )
        )
    )

    assert error is None
    assert any(item.kind is StreamItemKind.STREAM_COMPLETED for item in items)
    assert len(output_items) == 1
    assert output_items[0]["call_id"] == "success-sink"
    rollback.assert_not_called()
    assert source.close_count == 1


def test_private_provider_failure_terminal_scrubs_live_frames() -> None:
    secret = "PRIVATE_PROVIDER_FRAME_SENTINEL"

    async def scenario() -> None:
        async def provider() -> AsyncIterator[object]:
            yield _summary_item_added("native-frame", 0)
            yield {
                "type": "response.reasoning_text.delta",
                "item_id": "native-frame",
                "output_index": 0,
                "content_index": 0,
                "delta": secret,
            }
            raise RuntimeError(secret)

        source = provider()
        stream = OpenAIStream(source)
        events = stream._provider_events()
        reasoning = await events.__anext__()
        terminal = await events.__anext__()
        frame = events.ag_frame

        assert reasoning.kind is StreamItemKind.REASONING_DELTA
        assert terminal.kind is StreamItemKind.STREAM_ERRORED
        assert secret not in repr(terminal)
        assert source.ag_frame is None
        assert frame is not None
        assert frame.f_locals["event"] is None
        assert frame.f_locals["provider_iterator"] is None
        assert frame.f_locals["provider_events"] == ()
        assert frame.f_locals["provider_event"] is None
        assert secret not in repr(frame.f_locals)
        assert stream._stream is None
        assert stream._stream_sources == ()
        await events.aclose()

    run(scenario())


def test_replay_owner_error_terminal_scrubs_live_frames() -> None:
    secret = "REPLAY_OWNER_FRAME_PRIVATE_SENTINEL"

    async def scenario() -> None:
        finalized = Event()

        async def provider() -> AsyncIterator[object]:
            try:
                yield _output_done(_function_item("frame-call"))
                yield _completed_event()
                yield {"type": "response.output_text.delta", "delta": secret}
            finally:
                finalized.set()

        owner = _owner()
        owner.admit(_reasoning_item("frame-owner", "frame-cipher", []))
        retainer = MagicMock(
            side_effect=openai_module._ReplayOwnerAssociationError()
        )
        source = provider()
        stream = OpenAIStream(
            source,
            replay_owner=owner,
            replay_owner_retainer=retainer,
        )
        events = stream._provider_events()
        terminal: StreamProviderEvent | None = None
        while terminal is None:
            candidate = await events.__anext__()
            if candidate.kind is StreamItemKind.STREAM_ERRORED:
                terminal = candidate
        frame = events.ag_frame

        assert (
            cast(dict[str, Any], terminal.data)["error"]["code"]
            == "reasoning_replay_owner_ambiguous"
        )
        assert finalized.is_set()
        assert source.ag_frame is None
        assert owner.released
        assert owner.release_count == 1
        assert frame is not None
        assert frame.f_locals["event"] is None
        assert frame.f_locals["provider_iterator"] is None
        assert frame.f_locals["provider_events"] == ()
        assert frame.f_locals["provider_event"] is None
        assert secret not in repr(frame.f_locals)
        await events.aclose()

    run(scenario())


def test_failed_and_incomplete_outputs_revalidate_completed_items() -> None:
    secret = "FAILED_OUTPUT_CONFLICT_PRIVATE_SENTINEL"
    prefix = _message_item_event(
        "response.output_item.done",
        "terminal-failure-message",
        0,
        "safe",
    )
    for event_type in ("response.failed", "response.incomplete"):
        response: dict[str, object] = {
            "status": (
                "failed" if event_type == "response.failed" else "incomplete"
            ),
            "output": [],
        }
        if event_type == "response.failed":
            response["error"] = None
        else:
            response["incomplete_details"] = {"reason": "max_output_tokens"}
        omitted = run(
            _consume(
                OpenAIStream(
                    _AsyncEvents(
                        [
                            deepcopy(prefix),
                            {"type": event_type, "response": response},
                        ]
                    )
                )
            )
        )
        omitted_terminal = _error_item(omitted)
        assert (
            cast(dict[str, Any], omitted_terminal.data)["error"]["code"]
            == "invalid_reasoning_summary_event"
        )

        legacy_response = deepcopy(response)
        legacy_response.pop("output")
        legacy = run(
            _consume(
                OpenAIStream(
                    _AsyncEvents(
                        [
                            deepcopy(prefix),
                            {
                                "type": event_type,
                                "response": legacy_response,
                            },
                        ]
                    )
                )
            )
        )
        legacy_terminal = _error_item(legacy)
        assert legacy_terminal.provider_event_type == event_type
        assert (
            cast(dict[str, Any], legacy_terminal.data)["error"].get("code")
            != "invalid_reasoning_summary_event"
        )

        conflicting_item = deepcopy(cast(dict[str, object], prefix["item"]))
        cast(list[dict[str, object]], conflicting_item["content"])[0][
            "text"
        ] = secret
        conflict_response = deepcopy(response)
        conflict_response["output"] = [conflicting_item]
        conflicting = run(
            _consume(
                OpenAIStream(
                    _AsyncEvents(
                        [
                            deepcopy(prefix),
                            {
                                "type": event_type,
                                "response": conflict_response,
                            },
                        ]
                    )
                )
            )
        )
        conflict_terminal = _error_item(conflicting)
        assert (
            cast(dict[str, Any], conflict_terminal.data)["error"]["code"]
            == "invalid_reasoning_summary_event"
        )
        assert secret not in repr(
            [
                conflict_terminal,
                conflict_terminal.to_trace_dict(),
            ]
        )


def test_present_malformed_terminal_output_is_fail_closed() -> None:
    secret = "MALFORMED_TERMINAL_OUTPUT_PRIVATE_SENTINEL"

    class HostileOutput(list[object]):
        calls = 0

        def _fail(self, *_: object, **__: object) -> object:
            type(self).calls += 1
            raise RuntimeError(secret)

        __getitem__ = _fail
        __iter__ = _fail
        __len__ = _fail

    class HostileItem(dict[str, object]):
        calls = 0

        def _fail(self, *_: object, **__: object) -> object:
            type(self).calls += 1
            raise RuntimeError(secret)

        __getattr__ = _fail
        __getitem__ = _fail
        get = _fail
        items = _fail

    def terminal_event(event_type: str, output: object) -> dict[str, object]:
        response: dict[str, object] = {"output": output}
        if event_type == "response.completed":
            response["usage"] = {}
        elif event_type == "response.failed":
            response.update({"status": "failed", "error": None})
        else:
            response.update(
                {
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                }
            )
        return {"type": event_type, "response": response}

    for event_type in (
        "response.completed",
        "response.failed",
        "response.incomplete",
    ):
        for output in (
            None,
            7,
            secret,
            {"private": secret},
            (secret,),
            HostileOutput([secret]),
            [secret],
            [{}],
            [{"type": "unknown", "private": secret}],
            [HostileItem({"type": "message", "private": secret})],
        ):
            HostileOutput.calls = 0
            HostileItem.calls = 0
            items = run(
                _consume(
                    OpenAIStream(
                        _AsyncEvents([terminal_event(event_type, output)])
                    )
                )
            )
            terminal = _error_item(items)
            terminal_data = cast(dict[str, Any], terminal.data)
            outward = repr([terminal, terminal.to_trace_dict()])

            assert (
                terminal_data["error"]["code"]
                == "invalid_reasoning_summary_event"
            )
            assert terminal.provider_payload is None
            assert secret not in outward
            assert HostileOutput.calls == 0
            assert HostileItem.calls == 0


def test_provider_events_has_one_typed_terminal_yield_epilogue() -> None:
    tree = parse(dedent(getsource(OpenAIStream._provider_events)))
    yield_names = [
        node.value.id
        for node in walk(tree)
        if isinstance(node, Yield) and isinstance(node.value, Name)
    ]

    assert yield_names.count("terminal_event") == 1
    assert yield_names.count("provider_event") == 1
    assert set(yield_names) == {"provider_event", "terminal_event"}


def test_zero_length_sparse_part_requires_completed_array_position() -> None:
    item_id = "zero-sparse"
    prefix = [
        _summary_item_added(item_id, 0),
        _summary_part_added(item_id, 0, 3),
        _summary_delta("", item_id, 0, 3),
        _summary_text_done("", item_id, 0, 3),
        _summary_part_done("", item_id, 0, 3),
    ]
    for completed in ([], ["", "", ""]):
        items = run(
            _consume(
                OpenAIStream(
                    _AsyncEvents(
                        [
                            *deepcopy(prefix),
                            _summary_item_done(completed, item_id, 0),
                        ]
                    )
                )
            )
        )
        _assert_structured_summary_error(
            items,
            event_type="response.output_item.done",
            field="item.summary",
            value_shape="missing_position",
            output_index=0,
            summary_index=3,
        )

    positive = run(
        _consume(
            OpenAIStream(
                _AsyncEvents(
                    [
                        *deepcopy(prefix),
                        _summary_item_done(
                            ["", "", "", "fallback-at-three"],
                            item_id,
                            0,
                        ),
                        _completed_event(),
                    ]
                )
            )
        )
    )
    reasoning = _reasoning_items(positive)
    assert not any(
        item.kind is StreamItemKind.STREAM_ERRORED for item in positive
    )
    assert [item.text_delta for item in reasoning] == ["fallback-at-three"]
    assert reasoning[0].correlation.provider_summary_index == 3


def test_optional_sdk_type_discovery_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import_types = MagicMock(side_effect=ImportError)
    monkeypatch.setattr(
        openai_module, "_OPENAI_RESPONSE_STREAM_EVENT_TYPES", None
    )
    monkeypatch.setattr(
        openai_module, "_OPENAI_RESPONSE_OUTPUT_ITEM_TYPES", None
    )
    monkeypatch.setattr(openai_module, "import_module", import_types)

    assert not openai_module._is_trusted_openai_response_stream_event(object())
    assert not openai_module._is_trusted_openai_response_output_item(object())
    assert import_types.call_count == 2


def test_native_reasoning_done_rejects_non_string_text() -> None:
    items = run(
        _consume(
            OpenAIStream(
                _AsyncEvents(
                    [
                        {
                            "type": "response.reasoning_text.done",
                            "item_id": "native-invalid-text",
                            "output_index": 0,
                            "text": {},
                        }
                    ]
                )
            )
        )
    )

    _assert_structured_summary_error(
        items,
        event_type="response.reasoning_text.done",
        field="text",
        value_shape="mapping",
        output_index=0,
        summary_index=None,
    )


def test_terminal_output_rejects_cross_identity_and_invalid_call_ids() -> None:
    first = _message_item_event(
        "response.output_item.added", "terminal-first", 0
    )
    second = _message_item_event(
        "response.output_item.added", "terminal-second", 1
    )
    second_item = deepcopy(cast(dict[str, object], second["item"]))
    second_item["status"] = "completed"
    identity_items = run(
        _consume(
            OpenAIStream(
                _AsyncEvents(
                    [
                        first,
                        second,
                        {
                            "type": "response.completed",
                            "response": {
                                "usage": {},
                                "output": [second_item],
                            },
                        },
                    ]
                )
            )
        )
    )
    _assert_structured_summary_error(
        identity_items,
        event_type="response.completed",
        field="item.identity",
        value_shape="conflict",
        output_index=0,
        summary_index=None,
    )

    invalid_calls = (
        (
            {
                "id": "invalid-call",
                "type": "function_call",
                "status": "completed",
                "call_id": {},
            },
            "mapping",
        ),
        (
            {
                "id": "conflicting-call",
                "type": "custom_tool_call",
                "status": "completed",
                "call_id": "call-outer",
                "name": "lookup",
                "input": "{}",
                "custom_tool_call": {
                    "id": "call-inner",
                    "type": "custom_tool_call",
                    "name": "lookup",
                    "input": "{}",
                },
            },
            "conflict",
        ),
        (
            {
                "id": "message-with-call",
                "type": "message",
                "status": "completed",
                "call_id": "call-unexpected",
                "content": [],
            },
            "unexpected_value",
        ),
    )
    for item, value_shape in invalid_calls:
        items = run(
            _consume(
                OpenAIStream(
                    _AsyncEvents(
                        [
                            {
                                "type": "response.completed",
                                "response": {"usage": {}, "output": [item]},
                            }
                        ]
                    )
                )
            )
        )
        _assert_structured_summary_error(
            items,
            event_type="response.completed",
            field="item.call_id",
            value_shape=value_shape,
            output_index=0,
            summary_index=None,
        )


def test_terminal_output_requires_reasoning_and_tool_shapes() -> None:
    cases = (
        (
            {
                "id": "reasoning-without-summary",
                "type": "reasoning",
                "status": "completed",
                "encrypted_content": "cipher",
                "summary": None,
            },
            "item.summary",
            "null",
        ),
        (
            {
                "id": "tool-without-name",
                "type": "function_call",
                "status": "completed",
                "call_id": "call-without-name",
                "arguments": "{}",
            },
            "item.name",
            "missing",
        ),
        (
            {
                "id": "tool-without-arguments",
                "type": "function_call",
                "status": "completed",
                "call_id": "call-without-arguments",
                "name": "lookup",
            },
            "item.arguments",
            "missing",
        ),
    )
    for item, field, value_shape in cases:
        items = run(
            _consume(
                OpenAIStream(
                    _AsyncEvents(
                        [
                            {
                                "type": "response.completed",
                                "response": {"usage": {}, "output": [item]},
                            }
                        ]
                    )
                )
            )
        )
        _assert_structured_summary_error(
            items,
            event_type="response.completed",
            field=field,
            value_shape=value_shape,
            output_index=0,
            summary_index=None,
        )


def test_provider_graph_cycles_and_non_string_keys_fail_closed() -> None:
    cyclic_event: dict[str, object] = {"type": "response.completed"}
    cyclic_event["self"] = cyclic_event
    non_string_key_event: dict[object, object] = {
        "type": "response.completed",
        1: "unreadable",
    }

    for event in (cyclic_event, non_string_key_event):
        items = run(_consume(OpenAIStream(_AsyncEvents([event]))))
        _assert_preparse_private_error(
            items,
            event_type="response.completed",
        )


def test_trusted_sdk_event_readability_failures_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DumpAttributeTrap:
        type = "response.reasoning_summary_text.delta"
        item_id = "trusted-dump-trap"
        output_index = 0
        summary_index = 0
        delta = "must-not-emit"

        @property
        def model_dump(self) -> object:
            raise RuntimeError("model dump unavailable")

    class NoDumpCompleted:
        type = "response.completed"
        response = {"usage": {}}

    class UnreadableType:
        @property
        def type(self) -> object:
            raise RuntimeError("event type unavailable")

        def model_dump(self, *, mode: str) -> object:
            assert mode == "json"
            return {"type": "response.completed", "response": {"usage": {}}}

    class LateUnreadableType:
        output_index = 0
        item_id = "late-unreadable-type"

        def __init__(self) -> None:
            self.type_reads = 0

        @property
        def type(self) -> str:
            self.type_reads += 1
            if self.type_reads == 1:
                return "response.output_text.delta"
            raise RuntimeError("event type unavailable")

        @property
        def delta(self) -> object:
            raise RuntimeError("delta unavailable")

        def model_dump(self, *, mode: str) -> object:
            assert mode == "json"
            return {
                "type": "response.output_text.delta",
                "item_id": self.item_id,
                "output_index": self.output_index,
                "delta": "safe",
            }

    trusted_types = frozenset(
        {
            DumpAttributeTrap,
            LateUnreadableType,
            NoDumpCompleted,
            UnreadableType,
        }
    )
    monkeypatch.setattr(
        openai_module,
        "_OPENAI_RESPONSE_STREAM_EVENT_TYPES",
        trusted_types,
    )

    dump_items = run(
        _consume(OpenAIStream(_AsyncEvents([DumpAttributeTrap()])))
    )
    _assert_structured_summary_error(
        dump_items,
        event_type="response.reasoning_summary_text.delta",
        field="provider_payload",
        value_shape="unreadable",
        output_index=None,
        summary_index=None,
    )

    completed_items = run(
        _consume(OpenAIStream(_AsyncEvents([NoDumpCompleted()])))
    )
    assert any(
        item.kind is StreamItemKind.STREAM_COMPLETED
        for item in completed_items
    )

    type_items = run(_consume(OpenAIStream(_AsyncEvents([UnreadableType()]))))
    _assert_structured_summary_error(
        type_items,
        event_type="response.unknown",
        field="type",
        value_shape="unreadable",
        output_index=None,
        summary_index=None,
    )

    late_type_items = run(
        _consume(OpenAIStream(_AsyncEvents([LateUnreadableType()])))
    )
    assert _error_item(late_type_items).provider_event_type is None


def test_trusted_nested_sdk_payloads_are_classified_without_coercion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NestedDumpTrap:
        @property
        def model_dump(self) -> object:
            raise RuntimeError("nested dump unavailable")

    class NestedSafeDump:
        def model_dump(self, *, mode: str) -> object:
            assert mode == "json"
            return {"type": "message"}

    monkeypatch.setattr(
        openai_module,
        "_OPENAI_RESPONSE_STREAM_EVENT_TYPES",
        frozenset({NestedDumpTrap, NestedSafeDump}),
    )

    for nested in (NestedDumpTrap(), NestedSafeDump()):
        items = run(
            _consume(
                OpenAIStream(
                    _AsyncEvents(
                        [
                            {
                                "type": "response.completed",
                                "response": {"usage": {}},
                                "metadata": nested,
                            }
                        ]
                    )
                )
            )
        )
        assert any(
            item.kind is StreamItemKind.STREAM_COMPLETED for item in items
        )


def test_trusted_sdk_output_item_fingerprints_are_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DumpTrapItem:
        id = "dump-trap-item"
        type = "reasoning"
        status = "completed"
        encrypted_content = "cipher"
        summary = [{"type": "summary_text", "text": "safe-summary"}]

        @property
        def model_dump(self) -> object:
            raise RuntimeError("item dump unavailable")

    class NonStringKeyItem:
        id = "non-string-key-item"
        type = "reasoning"
        status = "completed"
        encrypted_content = "cipher"
        summary: list[object] = []

        def model_dump(self, *, mode: str) -> object:
            assert mode == "json"
            return {
                "id": self.id,
                "type": self.type,
                "status": self.status,
                "encrypted_content": self.encrypted_content,
                "summary": self.summary,
                1: "invalid-key",
            }

    class NestedFieldTrap:
        @property
        def type(self) -> object:
            raise RuntimeError("nested type unavailable")

    class NestedFieldTrapItem:
        id = "nested-field-trap-item"
        type = "reasoning"
        status = "completed"
        encrypted_content = "cipher"
        summary = [NestedFieldTrap()]

        def model_dump(self, *, mode: str) -> object:
            assert mode == "json"
            return {
                "id": self.id,
                "type": self.type,
                "status": self.status,
                "encrypted_content": self.encrypted_content,
                "summary": [{"type": "summary_text", "text": "safe-summary"}],
            }

    class TrustedDoneEvent:
        type = "response.output_item.done"
        output_index = 0

        def __init__(self, item: object) -> None:
            self.item = item

        def model_dump(self, *, mode: str) -> object:
            assert mode == "json"
            return {
                "type": self.type,
                "output_index": self.output_index,
                "item": {
                    "id": getattr(self.item, "id"),
                    "type": "reasoning",
                    "status": "completed",
                    "encrypted_content": "cipher",
                    "summary": [],
                },
            }

    monkeypatch.setattr(
        openai_module,
        "_OPENAI_RESPONSE_STREAM_EVENT_TYPES",
        frozenset({TrustedDoneEvent}),
    )
    monkeypatch.setattr(
        openai_module,
        "_OPENAI_RESPONSE_OUTPUT_ITEM_TYPES",
        frozenset({DumpTrapItem, NestedFieldTrapItem, NonStringKeyItem}),
    )

    dump_items = run(
        _consume(
            OpenAIStream(
                _AsyncEvents(
                    [
                        _summary_item_added("dump-trap-item", 0),
                        TrustedDoneEvent(DumpTrapItem()),
                    ]
                )
            )
        )
    )
    assert _error_item(dump_items).provider_payload is None

    non_string_key_items = run(
        _consume(
            OpenAIStream(
                _AsyncEvents(
                    [
                        _summary_item_added("non-string-key-item", 0),
                        TrustedDoneEvent(NonStringKeyItem()),
                    ]
                )
            )
        )
    )
    _assert_structured_summary_error(
        non_string_key_items,
        event_type="response.output_item.done",
        field="item",
        value_shape="unreadable",
        output_index=0,
        summary_index=None,
    )

    nested_items = run(
        _consume(
            OpenAIStream(
                _AsyncEvents(
                    [
                        _summary_item_added("nested-field-trap-item", 0),
                        TrustedDoneEvent(NestedFieldTrapItem()),
                    ]
                )
            )
        )
    )
    _assert_structured_summary_error(
        nested_items,
        event_type="response.output_item.done",
        field="item.summary.type",
        value_shape="unreadable",
        output_index=0,
        summary_index=0,
    )


def test_non_stream_openai_preserves_summary_tool_and_answer() -> None:
    client = _client(AsyncMock(return_value=_rich_non_stream_response()))

    result = run(
        client(
            "plain-model",
            [Message(role=MessageRole.USER, content="hello")],
            use_async_generator=False,
        )
    )

    assert isinstance(result, TextGenerationNonStreamResult)
    assert result.answer_text == '{"ok":true}'
    assert run(result.to_str()) == '{"ok":true}'
    assert [event.text_delta for event in result.events[:2]] == [
        "first",
        "second",
    ]
    assert all(
        event.kind is StreamItemKind.REASONING_DELTA
        for event in result.events[:2]
    )
    assert [
        event.correlation.provider_summary_index for event in result.events[:2]
    ] == [0, 1]
    assert all(
        event.correlation.protocol_item_id == "rs_non_stream"
        and event.correlation.provider_output_index == 0
        and event.reasoning_representation
        is StreamReasoningRepresentation.SUMMARY
        for event in result.events[:2]
    )
    tool_done = next(
        event
        for event in result.events
        if event.kind is StreamItemKind.TOOL_CALL_DONE
    )
    assert tool_done.correlation.tool_call_id == "call_non_stream"
    assert tool_done.correlation.protocol_item_id == "fc_non_stream"
    assert all(event.provider_payload is None for event in result.events)
    assert _PRIVATE_ENCRYPTED_SENTINEL not in repr(result.events)
    assert result.events[-1].kind is StreamItemKind.STREAM_COMPLETED
    assert result.usage["output_tokens_details"]["reasoning_tokens"] == 3


def test_non_stream_openai_unknown_output_item_is_ignored() -> None:
    response = SimpleNamespace(
        output=[
            SimpleNamespace(type="computer_call", id="unknown"),
            SimpleNamespace(
                type="message",
                id="msg_after_unknown",
                status="completed",
                content=[SimpleNamespace(type="output_text", text="answer")],
            ),
        ],
        usage=None,
    )

    result = run(
        _client(AsyncMock(return_value=response))(
            "plain-model",
            [],
            use_async_generator=False,
        )
    )

    assert isinstance(result, TextGenerationNonStreamResult)
    assert result.answer_text == "answer"
    assert [event.kind for event in result.events] == [
        StreamItemKind.ANSWER_DELTA,
        StreamItemKind.ANSWER_DONE,
        StreamItemKind.STREAM_COMPLETED,
    ]


def test_non_stream_response_events_preserve_fallback_terminal_metadata() -> (
    None
):
    fallback_events = OpenAIClient._non_stream_response_events(
        SimpleNamespace(output=None, output_text="fallback")
    )

    assert fallback_events[0]["output_index"] == 0
    assert fallback_events[0]["item"] == {
        "id": "msg_non_stream_output_text",
        "type": "message",
        "status": "in_progress",
        "content": [],
    }
    fallback_done = cast(dict[str, object], fallback_events[1]["item"])
    assert fallback_done["content"] == [
        {"type": "output_text", "text": "fallback"}
    ]

    error = {"code": "provider_failure"}
    failed_terminal = OpenAIClient._non_stream_response_events(
        SimpleNamespace(output=[], error=error)
    )[-1]
    failed_response = cast(dict[str, object], failed_terminal["response"])
    assert failed_terminal["type"] == "response.failed"
    assert failed_response["status"] == "failed"
    assert failed_response["error"] is error

    details = {"reason": "max_output_tokens"}
    incomplete_terminal = OpenAIClient._non_stream_response_events(
        SimpleNamespace(output=[], incomplete_details=details)
    )[-1]
    incomplete_response = cast(
        dict[str, object], incomplete_terminal["response"]
    )
    assert incomplete_terminal["type"] == "response.incomplete"
    assert incomplete_response["status"] == "incomplete"
    assert incomplete_response["incomplete_details"] is details


@pytest.mark.parametrize(
    ("response", "field"),
    (
        (SimpleNamespace(output=object()), "response.output"),
        (
            SimpleNamespace(output=[], output_text=object()),
            "response.output_text",
        ),
        (SimpleNamespace(output=[], status=object()), "response.status"),
        (SimpleNamespace(output=[], status="unexpected"), "response.status"),
    ),
)
def test_non_stream_response_events_reject_invalid_aggregate_shapes(
    response: object,
    field: str,
) -> None:
    with pytest.raises(
        openai_module._OpenAIReasoningSummaryEventError
    ) as error:
        OpenAIClient._non_stream_response_events(response)

    assert error.value.field == field


@pytest.mark.parametrize(
    ("content", "field"),
    (
        (object(), "item.content"),
        (
            [SimpleNamespace(type="summary_text", text="wrong")],
            "item.content.type",
        ),
        (
            [SimpleNamespace(type="reasoning_text", text=object())],
            "item.content.text",
        ),
    ),
)
def test_non_stream_native_reasoning_rejects_invalid_content(
    content: object,
    field: str,
) -> None:
    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                id="reasoning-invalid-content",
                type="reasoning",
                status="completed",
                summary=[],
                content=content,
            )
        ]
    )

    with pytest.raises(
        openai_module._OpenAIReasoningSummaryEventError
    ) as error:
        OpenAIClient._non_stream_response_events(response)

    assert error.value.field == field
    assert error.value.output_index == 0


def test_non_stream_output_item_inference_preserves_supported_shapes() -> None:
    direct_message = OpenAIClient._non_stream_output_item(
        SimpleNamespace(text="direct"),
        0,
    )
    nested_tool = OpenAIClient._non_stream_output_item(
        SimpleNamespace(
            call=SimpleNamespace(
                id="nested-call-item",
                call_id="nested-call",
                name="lookup",
                arguments='{"id":1}',
            )
        ),
        1,
    )

    assert direct_message == {
        "id": "msg_non_stream_0",
        "type": "message",
        "status": "completed",
        "content": [{"type": "output_text", "text": "direct"}],
    }
    assert nested_tool == {
        "id": "fc_non_stream_1",
        "type": "function_call",
        "status": "completed",
        "call_id": "nested-call",
        "name": "lookup",
        "arguments": '{"id":1}',
    }
    assert OpenAIClient._non_stream_output_item(SimpleNamespace(), 2) is None


def test_non_stream_reasoning_normalizes_absent_and_opaque_parts() -> None:
    absent = OpenAIClient._non_stream_reasoning_item(
        SimpleNamespace(
            id="reasoning-absent",
            status="completed",
            summary=None,
        ),
        0,
    )
    opaque_summary = object()
    opaque_content = object()
    opaque = OpenAIClient._non_stream_reasoning_item(
        SimpleNamespace(
            id="reasoning-opaque",
            status="completed",
            summary=opaque_summary,
            content=opaque_content,
        ),
        1,
    )

    assert absent["summary"] == []
    assert opaque["summary"] is opaque_summary
    assert opaque["content"] is opaque_content


@pytest.mark.parametrize(
    ("item", "field"),
    (
        (SimpleNamespace(type=object()), "item.type"),
        (
            SimpleNamespace(
                type="function_call",
                id="missing-call-id",
                call=SimpleNamespace(name="lookup", arguments="{}"),
            ),
            "item.call_id",
        ),
        (
            SimpleNamespace(
                type="reasoning",
                id=" ",
                status="completed",
                summary=[],
            ),
            "item.id",
        ),
        (
            SimpleNamespace(
                type="message",
                id="invalid-status",
                status="pending",
                content=[],
            ),
            "item.status",
        ),
    ),
)
def test_non_stream_output_items_reject_invalid_required_fields(
    item: object,
    field: str,
) -> None:
    with pytest.raises(
        openai_module._OpenAIReasoningSummaryEventError
    ) as error:
        OpenAIClient._non_stream_output_item(item, 0)

    assert error.value.field == field
    assert error.value.output_index == 0


def test_non_stream_added_and_terminal_items_enforce_internal_contracts() -> (
    None
):
    with pytest.raises(
        AssertionError,
        match="unsupported non-stream item at index 3",
    ):
        OpenAIClient._non_stream_added_item(
            {"id": "unsupported", "type": "unsupported"},
            3,
        )

    incomplete_tool = SimpleNamespace(
        type="function_call",
        id="incomplete-tool-item",
        status="incomplete",
        call_id="incomplete-tool",
        name="lookup",
        arguments="{}",
    )
    with pytest.raises(
        openai_module._OpenAIReasoningSummaryEventError
    ) as error:
        OpenAIClient._non_stream_response_events(
            SimpleNamespace(output=[incomplete_tool], status="completed")
        )

    assert error.value.field == "item.status"
    assert error.value.output_index == 0


def test_non_stream_openai_replay_owner_follows_valid_tool_cycle() -> None:
    create = AsyncMock(
        side_effect=[
            _rich_non_stream_response(),
            SimpleNamespace(
                output=[
                    SimpleNamespace(
                        id="msg_continuation",
                        type="message",
                        status="completed",
                        content=[
                            SimpleNamespace(
                                type="output_text",
                                text="continued",
                            )
                        ],
                    )
                ],
                usage=None,
            ),
        ]
    )
    client = _client(create)

    first = run(client("plain-model", [], use_async_generator=False))
    owner = cast(Any, client)._replay_owners_by_call_id["call_non_stream"]

    second = run(
        client(
            "plain-model",
            [_tool_result("call_non_stream", "value")],
            use_async_generator=False,
        )
    )

    assert isinstance(first, TextGenerationNonStreamResult)
    assert isinstance(second, TextGenerationNonStreamResult)
    assert second.answer_text == "continued"
    assert owner.release_count == 1
    assert cast(Any, client)._replay_owners_by_call_id == {}
    assert cast(Any, client)._active_replay_owners == {}
    assert cast(Any, client)._active_replay_streams == {}


def test_non_stream_openai_no_tool_releases_replay_owner() -> None:
    client = _client(
        AsyncMock(return_value=_rich_non_stream_response(include_tool=False))
    )
    owners: list[Any] = []
    original = cast(Any, client)._replay_owner_for_messages

    def capture_owner(messages: list[Message]) -> Any:
        owner = original(messages)
        owners.append(owner)
        return owner

    cast(Any, client)._replay_owner_for_messages = capture_owner

    result = run(client("plain-model", [], use_async_generator=False))

    assert isinstance(result, TextGenerationNonStreamResult)
    assert len(owners) == 1
    assert owners[0].release_count == 1
    assert cast(Any, client)._replay_owners_by_call_id == {}
    assert cast(Any, client)._active_replay_owners == {}
    assert cast(Any, client)._active_replay_streams == {}


def test_non_stream_openai_mapping_failure_rolls_back_owner() -> None:
    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="reasoning",
                status="completed",
                summary=[],
            )
        ]
    )
    client = _client(AsyncMock(return_value=response))
    owners: list[Any] = []
    original = cast(Any, client)._replay_owner_for_messages

    def capture_owner(messages: list[Message]) -> Any:
        owner = original(messages)
        owners.append(owner)
        return owner

    cast(Any, client)._replay_owner_for_messages = capture_owner

    with pytest.raises(openai_module._OpenAIReasoningSummaryEventError):
        run(client("plain-model", [], use_async_generator=False))

    assert len(owners) == 1
    assert owners[0].release_count == 1
    assert cast(Any, client)._active_replay_owners == {}
    assert cast(Any, client)._active_replay_streams == {}


def test_openai_stream_and_non_stream_semantics_are_equivalent() -> None:
    reasoning_item = ResponseReasoningItem(
        id="rs_sdk",
        type="reasoning",
        status="completed",
        content=[
            ResponseReasoningContent(
                type="reasoning_text",
                text="native sdk",
            )
        ],
        summary=[
            ResponseReasoningSummary(
                type="summary_text",
                text="summary sdk",
            )
        ],
        encrypted_content=_PRIVATE_ENCRYPTED_SENTINEL,
    )
    ignored_item = ResponseFunctionWebSearch(
        id="web_index_0",
        type="web_search_call",
        status="completed",
        action={"type": "search", "query": "lookup"},
    )
    completed_reasoning = {
        "id": "rs_sdk",
        "type": "reasoning",
        "status": "completed",
        "content": [{"type": "reasoning_text", "text": "native sdk"}],
        "summary": [{"type": "summary_text", "text": "summary sdk"}],
        "encrypted_content": _PRIVATE_ENCRYPTED_SENTINEL,
    }
    provider_trace = [
        {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {
                "id": "rs_sdk",
                "type": "reasoning",
                "status": "in_progress",
                "summary": [],
            },
        },
        {
            "type": "response.reasoning_text.delta",
            "item_id": "rs_sdk",
            "output_index": 1,
            "content_index": 0,
            "delta": "native sdk",
        },
        {
            "type": "response.reasoning_text.done",
            "item_id": "rs_sdk",
            "output_index": 1,
            "content_index": 0,
            "text": "native sdk",
        },
        {
            "type": "response.output_item.done",
            "output_index": 1,
            "item": completed_reasoning,
        },
        {
            "type": "response.completed",
            "response": {"id": "resp_sdk", "status": "completed"},
        },
    ]
    final_response = SimpleNamespace(
        id="resp_sdk",
        status="completed",
        output=[ignored_item, reasoning_item],
        usage=None,
    )
    streaming = run(
        _consume_provider_events(OpenAIStream(_AsyncEvents(provider_trace)))
    )
    non_stream = run(
        _client(AsyncMock(return_value=final_response))(
            "plain-model",
            [],
            use_async_generator=False,
        )
    )
    assert isinstance(non_stream, TextGenerationNonStreamResult)

    def semantic(
        events: list[StreamProviderEvent] | tuple[StreamProviderEvent, ...],
    ) -> list[tuple[object, ...]]:
        return [
            (
                event.kind,
                event.text_delta,
                event.correlation,
                event.visibility,
                event.reasoning_representation,
                event.segment_instance_ordinal,
                event.metadata,
                event.data,
                event.usage,
            )
            for event in events
        ]

    assert semantic(streaming) == semantic(non_stream.events)
    reasoning = [
        event
        for event in non_stream.events
        if event.kind is StreamItemKind.REASONING_DELTA
    ]
    assert [event.text_delta for event in reasoning] == [
        "native sdk",
        "summary sdk",
    ]
    assert [event.reasoning_representation for event in reasoning] == [
        StreamReasoningRepresentation.NATIVE_TEXT,
        StreamReasoningRepresentation.SUMMARY,
    ]
    assert [event.segment_instance_ordinal for event in reasoning] == [0, 1]
    assert all(
        event.correlation.protocol_item_id == "rs_sdk"
        and event.correlation.provider_output_index == 1
        for event in reasoning
    )
    assert [
        event.correlation.provider_summary_index for event in reasoning
    ] == [None, 0]
    assert reasoning[1].metadata == {
        REASONING_SEGMENT_BOUNDARY_METADATA_KEY: "completed"
    }
    assert non_stream.answer_text == ""
    assert all(event.provider_payload is None for event in non_stream.events)
    assert _PRIVATE_ENCRYPTED_SENTINEL not in repr(non_stream.events)


def test_non_stream_openai_incomplete_tool_is_not_executable() -> None:
    response = SimpleNamespace(
        id="resp_incomplete_tool",
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        output=[
            ResponseFunctionToolCall(
                id="fc_incomplete",
                type="function_call",
                status="incomplete",
                call_id="call_incomplete",
                name="lookup",
                arguments='{"id":1}',
            )
        ],
        usage=None,
    )
    client = _client(AsyncMock(return_value=response))

    result = run(client("plain-model", [], use_async_generator=False))

    assert isinstance(result, TextGenerationNonStreamResult)
    assert [event.kind for event in result.events] == [
        StreamItemKind.STREAM_ERRORED
    ]
    assert (
        cast(dict[str, Any], result.events[0].data)["error"]["code"]
        == "response_incomplete"
    )
    assert result.answer_text == ""
    assert cast(Any, client)._replay_owners_by_call_id == {}
    assert cast(Any, client)._active_replay_owners == {}
    assert cast(Any, client)._active_replay_streams == {}


def test_non_stream_openai_client_close_owns_temporary_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise() -> tuple[object, list[Any]]:
        started = Event()
        release = Event()

        async def blocking_source(
            events: tuple[dict[str, object], ...],
        ) -> AsyncIterator[object]:
            started.set()
            await release.wait()
            for event in events:
                yield event

        monkeypatch.setattr(
            OpenAIClient,
            "_iterate_non_stream_events",
            staticmethod(blocking_source),
        )
        client = _client(
            AsyncMock(
                return_value=_rich_non_stream_response(include_tool=False)
            )
        )
        owners: list[Any] = []
        original = cast(Any, client)._replay_owner_for_messages

        def capture_owner(messages: list[Message]) -> Any:
            owner = original(messages)
            owners.append(owner)
            return owner

        cast(Any, client)._replay_owner_for_messages = capture_owner
        call = create_task(
            client("plain-model", [], use_async_generator=False)
        )
        await wait_for(started.wait(), timeout=1.0)
        try:
            await wait_for(client.aclose(), timeout=1.0)
            result = (await gather(call, return_exceptions=True))[0]
        finally:
            release.set()
            if not call.done():
                call.cancel()
                await gather(call, return_exceptions=True)
        assert cast(Any, client)._active_replay_owners == {}
        assert cast(Any, client)._active_replay_streams == {}
        return result, owners

    result, owners = run(exercise())

    assert isinstance(result, openai_module.StreamConsumerClosure)
    assert len(owners) == 1
    assert owners[0].release_count == 1
