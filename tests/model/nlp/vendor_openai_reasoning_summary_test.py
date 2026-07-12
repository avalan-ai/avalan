"""Test OpenAI reasoning-summary requests, retry, and opaque replay."""

from asyncio import CancelledError, Event, create_task, run, wait_for
from collections.abc import Iterator, Mapping
from copy import deepcopy
from json import dumps
from math import inf, nan
from traceback import format_exception
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

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
from avalan.model.nlp.text.vendor import openai as openai_module
from avalan.model.nlp.text.vendor.openai import OpenAIClient, OpenAIStream
from avalan.model.reasoning import (
    ReasoningSummaryCapabilityError,
    ReasoningSummaryRequestCapability,
)
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamItemKind,
    StreamProviderEvent,
    StreamReasoningRepresentation,
    StreamRetentionPolicy,
    StreamVisibility,
    accumulate_canonical_stream_items,
)
from avalan.task.usage import usage_totals_from_response

_PRIVATE_ENCRYPTED_SENTINEL = "encrypted-private-sentinel"
_PRIVATE_SUMMARY_SENTINEL = "summary-private-sentinel"


class _AsyncEvents:
    def __init__(self, events: list[object]) -> None:
        self._events = iter(events)
        self.close_count = 0

    def __aiter__(self) -> "_AsyncEvents":
        return self

    async def __anext__(self) -> object:
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


class _CapableOpenAIClient(OpenAIClient):
    reasoning_summary_request_capability = ReasoningSummaryRequestCapability(
        supported_modes=frozenset(ReasoningSummaryMode)
    )


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
    client_type = _CapableOpenAIClient if capable else OpenAIClient
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


def _tool_mock(count: int) -> MagicMock | None:
    if count == 0:
        return None
    tool = MagicMock()
    tool.json_schemas.return_value = [
        {
            "type": "function",
            "function": {"name": f"pkg.tool_{index}"},
        }
        for index in range(count)
    ]
    return tool


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
                tool=cast(Any, _tool_mock(tool_count)),
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
    assert policy.openai_replay_reasoning_item_limit == 1024
    assert policy.openai_replay_reasoning_summary_node_limit == 4096
    assert policy.openai_replay_reasoning_summary_character_limit == 262144
    assert (
        policy.openai_replay_reasoning_summary_serialized_byte_limit == 1048576
    )
    field_names = (
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
    policy = StreamRetentionPolicy(openai_replay_reasoning_item_limit=3)
    owner = _owner(policy)
    assert owner.admit(_reasoning_item("rs_1"))
    assert owner.admit(_reasoning_item("rs_2"))
    assert owner.counters[0] == 2
    assert owner.admit(_reasoning_item("rs_3"))
    before = (owner.replay_items(), owner.counters)
    with pytest.raises(openai_module._ReasoningReplayRetentionError) as error:
        owner.admit(_reasoning_item("rs_4"))
    assert error.value.code == "reasoning_replay_retention_exceeded"
    assert (owner.replay_items(), owner.counters) == before

    function_owner = _owner(
        StreamRetentionPolicy(
            replay_history_item_limit=0,
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

    opaque_surrogate = "OPAQUE_ENCRYPTED_\ud800"
    opaque_owner = _owner()
    assert opaque_owner.admit(
        _reasoning_item(
            "opaque-surrogate",
            opaque_surrogate,
            [],
        )
    )
    assert (
        opaque_owner.replay_items()[0]["encrypted_content"] == opaque_surrogate
    )
    function_surrogate = _function_item("function-surrogate")
    function_surrogate["arguments"] = "FUNCTION_ARGUMENT_\udfff"
    function_owner = _owner()
    assert function_owner.admit(function_surrogate)
    assert (
        function_owner.replay_items()[0]["arguments"]
        == "FUNCTION_ARGUMENT_\udfff"
    )
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


def test_replay_rollback_release_and_request_isolation() -> None:
    owner_a = _owner()
    owner_b = _owner()
    owner_a.admit(_reasoning_item("a-base", "cipher-a", []))
    owner_a.commit_attempt()
    owner_a.begin_attempt()
    owner_a.admit(_reasoning_item("a-failed", "cipher-failed", []))
    owner_a.admit(_function_item("call-failed"))
    owner_b.admit(_reasoning_item("b", "cipher-b", []))
    owner_b_before = (owner_b.replay_items(), owner_b.counters)

    owner_a.rollback_attempt()
    owner_a.rollback_attempt()
    assert [item["id"] for item in owner_a.replay_items()] == ["a-base"]
    assert (owner_b.replay_items(), owner_b.counters) == owner_b_before

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
    retainer = MagicMock(
        side_effect=openai_module._ReplayOwnerAssociationError()
    )
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
                _AsyncEvents(
                    [
                        {
                            "type": "response.cancelled",
                            "reason": encrypted,
                            "metadata": {"echo": summary},
                        }
                    ]
                ),
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
    assert cast(dict[str, Any], provider_cancel_terminal.data)["error"] == {
        "type": "server_error",
        "code": "openai_provider_request_failed",
        "status": "cancelled",
        "message": "OpenAI provider request cancelled",
    }
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
    assert isinstance(terminal_error, openai_module._OpenAICleanupError)
    assert terminal_error.__cause__ is None
    assert terminal_error.__context__ is None or isinstance(
        terminal_error.__context__,
        GeneratorExit,
    )
    terminal_diagnostics = "\n".join(
        (
            repr(terminal_error),
            str(terminal_error),
            "".join(format_exception(terminal_error)),
        )
    )
    terminal_outward = "\n".join(
        (
            repr(terminal_items),
            repr([item.to_trace_dict() for item in terminal_items]),
            terminal_diagnostics,
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
                "response": {"status": "failed", "error": None, "output": []},
            },
        ]
    )
    recovered = _AsyncEvents(
        [
            {
                "type": "response.reasoning_text.delta",
                "delta": "native",
                "item_id": "rs_native",
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
    owner = _owner()
    owner.admit(_reasoning_item("kept", "cipher-kept", []))
    owner.commit_attempt()
    baseline = (owner.replay_items(), owner.counters)
    owner.begin_attempt()
    owner.admit(_reasoning_item("failed", "cipher-failed", []))
    owner.admit(_function_item("failed-call"))

    owner.rollback_attempt()

    assert (owner.replay_items(), owner.counters) == baseline
    assert all(
        item.get("call_id") != "failed-call" for item in owner.replay_items()
    )
