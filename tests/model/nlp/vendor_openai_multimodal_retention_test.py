"""Exercise bounded multimodal replay without retaining provider secrets."""

from asyncio import CancelledError, Event, create_task, run
from base64 import b64decode
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import replace
from hashlib import sha256
from json import dumps
from tracemalloc import get_traced_memory, start, stop
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, patch

from httpx import Request, Response
from openai import BadRequestError
from pytest import mark, raises

from avalan.agent.orchestrator.response.orchestrator_response import (
    OrchestratorResponse,
)
from avalan.conversation.settings import InlineCompaction
from avalan.entities import (
    GenerationSettings,
    Message,
    MessageRole,
    ToolCall,
    ToolCallResult,
    ToolResultImage,
)
from avalan.model.nlp.text.vendor import openai as vendor
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamItemKind,
    StreamProviderEvent,
    StreamRetentionPolicy,
    TextGenerationNonStreamResult,
)
from avalan.types import LooseJsonValue

# The diagnostic measured this size; these bytes are entirely synthetic.
_MEASURED_BYTES = 23_586_318


def _compaction(
    size: int = 256, identity: str = "compact"
) -> dict[str, LooseJsonValue]:
    item: dict[str, LooseJsonValue] = {
        "type": "compaction",
        "id": identity,
        "encrypted_content": "x",
    }
    overhead = (
        vendor._replay_json_serialized_bytes(cast(LooseJsonValue, item)) - 1
    )
    assert size > overhead
    item["encrypted_content"] = "x" * (size - overhead)
    return item


def _events(output: list[dict[str, LooseJsonValue]]) -> list[object]:
    events: list[object] = []
    for index, item in enumerate(output):
        events.extend(
            [
                {
                    "type": "response.output_item.added",
                    "output_index": index,
                    "item": {**item, "status": "in_progress"},
                },
                {
                    "type": "response.output_item.done",
                    "output_index": index,
                    "item": {**item, "status": "completed"},
                },
            ]
        )
    events.append(
        {
            "type": "response.completed",
            "response": {"status": "completed", "output": output},
        }
    )
    return events


class _Events:
    def __init__(self, events: list[object]) -> None:
        self.events = iter(events)
        self.closed = 0

    def __aiter__(self) -> "_Events":
        return self

    async def __anext__(self) -> object:
        try:
            return next(self.events)
        except StopIteration:
            raise StopAsyncIteration from None

    async def aclose(self) -> None:
        self.closed += 1


def _client(
    create: AsyncMock, policy: StreamRetentionPolicy | None = None
) -> vendor.OpenAIClient:
    with patch("openai.AsyncOpenAI") as constructor:
        constructor.return_value = SimpleNamespace(
            base_url="https://api.openai.com/v1",
            responses=SimpleNamespace(create=create),
            close=AsyncMock(return_value=None),
        )
        return vendor.OpenAIClient(
            api_key="test",
            base_url="https://api.openai.com/v1",
            stream_retention_policy=policy,
        )


def test_realistic_replacements_count_checkpoint_once_and_rollback() -> None:
    owner = vendor._OpenAIDirectReplayExecutionState(StreamRetentionPolicy())
    budget = owner.budget
    assert budget is not None
    for index in range(6):
        owner.begin_attempt()
        checkpoint = owner.replay_items()
        assert owner.admit(_compaction(_MEASURED_BYTES, f"compact-{index}"))
        assert budget.used == _MEASURED_BYTES * (2 if index else 1)
        assert owner.item_count == 1
        if index % 2:
            owner.rollback_attempt()
            assert owner.replay_items() == checkpoint
        else:
            assert owner.commit_attempt() == 1
        assert budget.used == _MEASURED_BYTES
    assert budget.peak == 2 * _MEASURED_BYTES
    owner.release()
    owner.release()
    assert budget.used == 0
    assert owner.release_count == 1


def test_shared_quota_rejects_atomically_without_evicting_other_owner() -> (
    None
):
    policy = StreamRetentionPolicy(
        openai_replay_client_serialized_byte_limit=1000
    )
    client = _client(AsyncMock(), policy)
    first = client._replay_owner_for_messages([])
    second = client._replay_owner_for_messages([])
    assert first.budget is second.budget
    first.begin_attempt()
    first.admit(_compaction(400))
    first.commit_attempt()
    second.begin_attempt()
    second.admit(_compaction(300))
    second.commit_attempt()
    first.begin_attempt()
    with raises(vendor._ReasoningReplayRetentionError) as failure:
        first.admit(_compaction(400, "replacement"))
    assert (
        failure.value.resource == "openai_replay_client_serialized_byte_limit"
    )
    assert (failure.value.observed, failure.value.allowed) == (1100, 1000)
    assert first.replay_items()[0]["id"] == "compact"
    assert second.generic_counters == (1, 300)
    assert client.replay_retention_diagnostics.used_serialized_bytes == 700
    second.release()
    first.admit(_compaction(400, "replacement"))
    first.admit(_compaction(500, "second-replacement"))
    assert client.replay_retention_diagnostics.used_serialized_bytes == 900
    first.rollback_attempt()
    assert client.replay_retention_diagnostics.used_serialized_bytes == 400
    run(client.aclose())
    assert client.replay_retention_diagnostics.used_serialized_bytes == 0


def test_large_string_accounting_and_hashing_have_bounded_scratch_space() -> (
    None
):
    payload = "x" * _MEASURED_BYTES
    expected_hash = sha256(payload.encode()).hexdigest()
    start()
    try:
        assert (
            vendor._replay_json_serialized_bytes(payload) == len(payload) + 2
        )
        assert vendor._replay_text_sha256(payload) == expected_hash
        _, peak = get_traced_memory()
    finally:
        stop()
    assert peak < 1024 * 1024
    for text in ("", '"\\\n\t\x00', "á🙂" * 17000):
        assert vendor._replay_json_serialized_bytes(text) == len(
            dumps(text, ensure_ascii=False).encode()
        )
    with raises(vendor._OpenAIInlineCompactionError):
        vendor._replay_text_sha256("\ud800")


@mark.parametrize("streaming", [True, False])
def test_large_compaction_rejection_preserves_local_code_without_retry(
    streaming: bool,
) -> None:
    compaction = _compaction(_MEASURED_BYTES)
    # First admit reasoning so rejection exercises private replay sanitization.
    reasoning: dict[str, LooseJsonValue] = {
        "id": "reasoning",
        "type": "reasoning",
        "summary": [],
        "encrypted_content": "private-reasoning-sentinel",
    }
    output = [reasoning, compaction]
    response = (
        _Events(_events(output))
        if streaming
        else {
            "status": "completed",
            "output": output,
        }
    )
    create = AsyncMock(return_value=response)
    client = _client(
        create,
        StreamRetentionPolicy(openai_replay_serialized_byte_limit=4194304),
    )

    async def execute() -> list[CanonicalStreamItem | StreamProviderEvent]:
        result = await client(
            "gpt-5.4",
            [],
            GenerationSettings(
                openai_inline_compaction=InlineCompaction(
                    compact_threshold=32768
                )
            ),
            use_async_generator=streaming,
        )
        if isinstance(result, TextGenerationNonStreamResult):
            return list(result.events)
        return [
            item
            async for item in cast(AsyncIterable[CanonicalStreamItem], result)
        ]

    items = run(execute())
    terminal = next(
        item for item in items if item.kind is StreamItemKind.STREAM_ERRORED
    )
    assert isinstance(terminal.data, dict)
    error = terminal.data["error"]
    assert error == {
        "type": "resource_exhausted",
        "code": "reasoning_replay_retention_exceeded",
        "message": "OpenAI replay retention limit exceeded.",
        "retryable": False,
        "resource": "openai_replay_serialized_byte_limit",
        "observed": _MEASURED_BYTES,
        "allowed": 4194304,
    }
    assert terminal.provider_payload is None
    assert create.await_count == 1
    assert client.inline_compaction_diagnostics.committed_boundary_count == 0
    assert client.inline_compaction_diagnostics.rolled_back_boundary_count == 1
    assert client.replay_retention_diagnostics.used_serialized_bytes == 0
    assert "private-reasoning-sentinel" not in repr(items)
    assert "x" * 128 not in repr(items)
    run(client.aclose())


@mark.parametrize("terminal_size", [1024, _MEASURED_BYTES])
def test_large_compaction_continues_with_exact_native_tool_images(
    terminal_size: int,
) -> None:
    compaction = _compaction(_MEASURED_BYTES)
    tool: dict[str, LooseJsonValue] = {
        "type": "function_call",
        "id": "tool-item",
        "call_id": "tool-call",
        "name": "image",
        "arguments": "{}",
    }
    terminal = _compaction(terminal_size)
    first_events = _events([compaction, tool])
    first_events[-1] = {
        "type": "response.completed",
        "response": {"status": "completed", "output": [terminal, tool]},
    }
    create = AsyncMock(
        side_effect=[
            _Events(first_events),
            _Events(
                [
                    {
                        "type": "response.output_text.delta",
                        "delta": "continued",
                    },
                    *_events([]),
                ]
            ),
        ]
    )
    client = _client(create)
    image_bytes = b"exact synthetic tool image bytes"
    result = ToolCallResult(
        id="result",
        call=ToolCall(id="tool-call", name="image", arguments={}),
        name="image",
        result="image produced",
        content=(ToolResultImage(data=image_bytes, media_type="image/png"),),
    )

    async def execute() -> None:
        settings = GenerationSettings(
            openai_inline_compaction=InlineCompaction(compact_threshold=32768)
        )
        first = await client(
            "gpt-5.4",
            [Message(role=MessageRole.USER, content="original")],
            settings,
        )
        first_items = [
            item
            async for item in cast(AsyncIterable[CanonicalStreamItem], first)
        ]
        assert any(
            item.kind is StreamItemKind.INLINE_COMPACTION_COMMITTED
            for item in first_items
        )
        assert (
            client.replay_retention_diagnostics.used_serialized_bytes
            > terminal_size
        )
        second = await client(
            "gpt-5.4",
            OrchestratorResponse._tool_observation_messages(
                result, json_output=False
            ),
            settings,
        )
        second_items = [
            item
            async for item in cast(AsyncIterable[CanonicalStreamItem], second)
        ]
        assert any(item.text_delta == "continued" for item in second_items)
        assert any(
            item.kind is StreamItemKind.STREAM_COMPLETED
            for item in second_items
        )
        await client.aclose()

    run(execute())
    replay = create.await_args_list[1].kwargs["input"]
    assert [item["type"] for item in replay] == [
        "compaction",
        "function_call",
        "function_call_output",
    ]
    assert replay[0] == terminal
    image_block = next(
        block
        for block in replay[2]["output"]
        if block["type"] == "input_image"
    )
    assert b64decode(image_block["image_url"].split(",", 1)[1]) == image_bytes
    assert create.await_args_list[1].kwargs["store"] is False
    assert client.replay_retention_diagnostics.used_serialized_bytes == 0


@mark.parametrize("ending", ["failure", "protocol", "cancel"])
def test_candidate_failure_rolls_back_before_release(ending: str) -> None:
    owner = vendor._OpenAIDirectReplayExecutionState(StreamRetentionPolicy())
    original = _compaction(512, "original")
    owner.begin_attempt()
    owner.admit(original)
    owner.commit_attempt()
    owner.begin_attempt()
    source_events = _events([_compaction(1024, "candidate")])[:-1]
    if ending == "failure":
        source_events.append(
            {
                "type": "response.failed",
                "response": {
                    "status": "failed",
                    "error": {"code": "server_error", "message": "private"},
                },
            }
        )
    elif ending == "protocol":
        source_events.append(
            {
                "type": "response.completed",
                "response": {"status": "completed", "output": []},
            }
        )
    observed: list[object] = []

    def release(
        replay_owner: vendor._OpenAIDirectReplayExecutionState,
    ) -> None:
        observed.append(replay_owner.replay_items())
        assert replay_owner.budget is not None
        assert replay_owner.budget.used == 512
        replay_owner.release()

    async def execute() -> list[CanonicalStreamItem | StreamProviderEvent]:
        blocked = Event()

        async def source() -> AsyncIterator[object]:
            for event in source_events:
                yield event
            if ending == "cancel":
                blocked.set()
                await Event().wait()

        stream = vendor.OpenAIStream(
            source(),
            replay_owner=owner,
            replay_owner_releaser=release,
            inline_compaction_enabled=True,
        )

        async def consume() -> list[CanonicalStreamItem | StreamProviderEvent]:
            return [item async for item in stream]

        task = create_task(consume())
        if ending == "cancel":
            await blocked.wait()
            task.cancel()
            with raises(CancelledError):
                await task
            return []
        return await task

    items = run(execute())
    assert observed == [(original,)]
    assert owner.budget is not None and owner.budget.used == 0
    assert owner.release_count == 1
    if ending != "cancel":
        terminal = next(
            item
            for item in items
            if item.kind is StreamItemKind.STREAM_ERRORED
        )
        assert isinstance(terminal.data, dict)
        error = terminal.data["error"]
        assert isinstance(error, dict)
        assert error["code"] == (
            "inline_compaction_protocol_invalid"
            if ending == "protocol"
            else "openai_provider_request_failed"
        )
        if ending == "protocol":
            assert error["retryable"] is False
        assert "private" not in repr(items)


def test_repeated_compactions_bound_fingerprint_metadata() -> None:
    policy = StreamRetentionPolicy(openai_replay_item_limit=2)
    owner = vendor._OpenAIDirectReplayExecutionState(policy)
    owner.begin_attempt()
    stream = vendor.OpenAIStream(
        _Events([]), replay_owner=owner, inline_compaction_enabled=True
    )
    for index in range(2):
        item = _compaction(256, f"compact-{index}")
        event = {"output_index": index, "item": item}
        stream._record_compaction_added(event)
        stream._complete_compaction_item(event, item)
        owner.admit(item)
    with raises(vendor._ReasoningReplayRetentionError) as failure:
        stream._record_compaction_added(
            {
                "output_index": 2,
                "item": _compaction(),
            }
        )
    assert (failure.value.observed, failure.value.allowed) == (3, 2)
    assert len(stream._compaction_completed_by_item_id) == 2
    assert owner.item_count == 1
    owner.release()


@mark.parametrize("value", [-1, True, 1.0, "1", None])
def test_client_quota_requires_nonnegative_integer(value: object) -> None:
    with raises(AssertionError):
        replace(
            StreamRetentionPolicy(),
            openai_replay_client_serialized_byte_limit=cast(int, value),
        )


def test_replay_association_capacity_reports_pre_cleanup_count() -> None:
    client = _client(
        AsyncMock(), StreamRetentionPolicy(replay_history_item_limit=1)
    )
    owner = client._replay_owner_for_messages([])
    client._active_replay_call_ids["old"] = owner
    with raises(vendor._ReasoningReplayRetentionError) as failure:
        client._retain_replay_owner(owner, ("first", "second"))
    assert (failure.value.observed, failure.value.allowed) == (2, 1)
    assert owner.released
    run(client.aclose())


def test_safe_error_projection_rejects_private_and_untyped_details() -> None:
    invalid_data: tuple[LooseJsonValue, ...] = (
        None,
        {"error": {"code": []}},
        {"error": {"code": "private-code"}},
    )
    for details in invalid_data:
        assert vendor.OpenAIStream._safe_replay_error_data(details) is None
    for resource, observed, allowed in (
        ("private-resource", 10, 2),
        ("openai_replay_serialized_byte_limit", True, 2),
        ("openai_replay_serialized_byte_limit", 10, -1),
    ):
        safe = vendor.OpenAIStream._safe_replay_error_data(
            {
                "error": {
                    "code": "reasoning_replay_retention_exceeded",
                    "resource": resource,
                    "observed": observed,
                    "allowed": allowed,
                    "message": "private-message",
                    "encrypted_content": "private-content",
                }
            }
        )
        assert isinstance(safe, dict)
        error = safe["error"]
        assert isinstance(error, dict)
        assert "resource" not in error
        assert "private" not in repr(safe)
        assert error["retryable"] is False


def test_terminal_reconciliation_preserves_bounds_and_checkpoint() -> None:
    policy = StreamRetentionPolicy(
        openai_replay_serialized_byte_limit=1000,
        openai_replay_client_serialized_byte_limit=1100,
    )
    owner = vendor._OpenAIDirectReplayExecutionState(policy)
    checkpoint = _compaction(400, "checkpoint")
    owner.begin_attempt()
    owner.admit(checkpoint)
    owner.commit_attempt()
    owner.begin_attempt()
    candidate = _compaction(300)
    owner.admit(candidate)
    for size, resource, observed in (
        (1001, "openai_replay_serialized_byte_limit", 1001),
        (800, "openai_replay_client_serialized_byte_limit", 1200),
    ):
        with raises(vendor._ReasoningReplayRetentionError) as failure:
            owner.reconcile_compaction(_compaction(size))
        assert failure.value.resource == resource
        assert failure.value.observed == observed
        assert owner.replay_items() == (candidate,)
        assert owner.budget is not None and owner.budget.used == 700
    owner.reconcile_compaction(_compaction(200))
    assert owner.budget is not None and owner.budget.used == 600
    assert owner.generic_counters == (1, 200)
    owner.rollback_attempt()
    assert owner.replay_items() == (checkpoint,)
    assert owner.budget.used == 400
    owner.release()


def test_terminal_reconciliation_requires_active_matching_boundary() -> None:
    owner = vendor._OpenAIDirectReplayExecutionState(StreamRetentionPolicy())
    with raises(RuntimeError):
        owner.reconcile_compaction(_compaction())
    owner.begin_attempt()
    with raises(vendor._OpenAIInlineCompactionError):
        owner.reconcile_compaction(_compaction())
    owner.admit(_compaction())
    invalid_items: tuple[dict[str, LooseJsonValue], ...] = (
        _compaction(identity="mismatch"),
        {"type": "compaction"},
    )
    for item in invalid_items:
        with raises(vendor._OpenAIInlineCompactionError):
            owner.reconcile_compaction(item)
    owner.release()


@mark.parametrize("streaming", [True, False])
def test_provider_rejects_large_replay_without_echo_or_retry(
    streaming: bool,
) -> None:
    tool: dict[str, LooseJsonValue] = {
        "type": "function_call",
        "id": "tool-item",
        "call_id": "tool-call",
        "name": "image",
        "arguments": "{}",
    }
    secret = "private-provider-echo"
    create = AsyncMock(
        side_effect=[
            _Events(_events([_compaction(), tool])),
            BadRequestError(
                secret,
                response=Response(
                    400,
                    request=Request("POST", "https://provider.invalid"),
                ),
                body={
                    "code": "string_above_max_length",
                    "message": secret,
                    "param": secret,
                },
            ),
        ]
    )
    client = _client(create)

    async def execute() -> None:
        first = await client(
            "gpt-5.4",
            [],
            GenerationSettings(
                openai_inline_compaction=InlineCompaction(
                    compact_threshold=32768
                )
            ),
        )
        async for _ in cast(AsyncIterable[CanonicalStreamItem], first):
            pass
        result = ToolCallResult(
            id="result",
            call=ToolCall(id="tool-call", name="image", arguments={}),
            name="image",
            result="image produced",
        )
        with raises(vendor._OpenAIProviderRequestError) as failure:
            await client(
                "gpt-5.4",
                OrchestratorResponse._tool_observation_messages(
                    result, json_output=False
                ),
                use_async_generator=streaming,
            )
        assert failure.value.code == "string_above_max_length"
        assert failure.value.status_code == 400
        assert failure.value.retryable is False
        assert secret not in str(failure.value)
        assert secret not in repr(failure.value.provider_failure)
        event = vendor.OpenAIStream._private_replay_provider_failure_event(
            provider_failure=failure.value.provider_failure
        )
        assert isinstance(event.data, dict)
        assert event.data["error"] == {
            "type": "invalid_request_error",
            "code": "string_above_max_length",
            "status": "failed",
            "message": "OpenAI provider rejected an input string length.",
            "retryable": False,
            "status_code": 400,
        }
        await client.aclose()

    run(execute())
    assert create.await_count == 2
    assert client.replay_retention_diagnostics.used_serialized_bytes == 0


def test_source_admission_failure_preserves_capacity_classification() -> None:
    owner = vendor._OpenAIDirectReplayExecutionState(
        StreamRetentionPolicy(openai_replay_serialized_byte_limit=1024)
    )
    owner.begin_attempt()

    async def source() -> AsyncIterator[object]:
        yield {
            "type": "response.output_item.done",
            "item": {
                "type": "reasoning",
                "id": "reasoning",
                "encrypted_content": "private-source-sentinel",
                "summary": [],
            },
        }
        owner.admit(_compaction(2048))

    async def execute() -> list[CanonicalStreamItem | StreamProviderEvent]:
        stream = vendor.OpenAIStream(source(), replay_owner=owner)
        return [item async for item in stream]

    items = run(execute())
    terminal = next(
        item for item in items if item.kind is StreamItemKind.STREAM_ERRORED
    )
    assert isinstance(terminal.data, dict)
    error = terminal.data["error"]
    assert isinstance(error, dict)
    assert error["code"] == "reasoning_replay_retention_exceeded"
    assert (error["observed"], error["allowed"]) == (2048, 1024)
    assert error["retryable"] is False
    assert "private-source-sentinel" not in repr(items)
    assert owner.released
    assert owner.budget is not None and owner.budget.used == 0
