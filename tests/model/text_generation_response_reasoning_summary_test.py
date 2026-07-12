"""Test structured reasoning-summary SDK accumulation and isolation."""

from logging import getLogger

import pytest

from avalan.entities import GenerationSettings
from avalan.model.response.text import TextGenerationResponse
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamItemCorrelation,
    StreamItemKind,
    StreamProviderEvent,
    StreamReasoningRepresentation,
    StreamTerminalOutcome,
    StreamVisibility,
    TextGenerationNonStreamResult,
    accumulate_canonical_stream_items,
)
from avalan.task.usage import usage_observation_from_response

_SUMMARY_SENTINEL = "summary-private-sentinel"
_ENCRYPTED_SENTINEL = "encrypted-private-sentinel"
_ANSWER = '{"ok":true}'
_USAGE = {
    "input_tokens": 3,
    "output_tokens": 5,
    "total_tokens": 8,
    "output_tokens_details": {"reasoning_tokens": 2},
}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _rich_result(
    *,
    outcome: StreamTerminalOutcome = StreamTerminalOutcome.COMPLETED,
) -> TextGenerationNonStreamResult:
    terminal_kind = {
        StreamTerminalOutcome.COMPLETED: StreamItemKind.STREAM_COMPLETED,
        StreamTerminalOutcome.ERRORED: StreamItemKind.STREAM_ERRORED,
        StreamTerminalOutcome.CANCELLED: StreamItemKind.STREAM_CANCELLED,
    }[outcome]
    events = [
        StreamProviderEvent(
            kind=StreamItemKind.REASONING_DELTA,
            text_delta=_SUMMARY_SENTINEL,
            correlation=StreamItemCorrelation(
                protocol_item_id="reasoning-item-1",
                provider_output_index=0,
                provider_summary_index=0,
            ),
            visibility=StreamVisibility.PRIVATE,
            reasoning_representation=StreamReasoningRepresentation.SUMMARY,
            segment_instance_ordinal=0,
            provider_event_type="response.output_item.done",
        ),
        StreamProviderEvent(
            kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
            text_delta='{"query":"safe"}',
            correlation=StreamItemCorrelation(tool_call_id="call-1"),
            provider_event_type="response.function_call_arguments.done",
        ),
        StreamProviderEvent(
            kind=StreamItemKind.TOOL_CALL_READY,
            data={"name": "lookup", "arguments": {"query": "safe"}},
            correlation=StreamItemCorrelation(tool_call_id="call-1"),
            provider_event_type="response.output_item.done",
        ),
        StreamProviderEvent(
            kind=StreamItemKind.TOOL_CALL_DONE,
            correlation=StreamItemCorrelation(tool_call_id="call-1"),
            provider_event_type="response.output_item.done",
        ),
        StreamProviderEvent(
            kind=StreamItemKind.ANSWER_DELTA,
            text_delta=_ANSWER,
            provider_event_type="response.output_text.done",
        ),
    ]
    if outcome is StreamTerminalOutcome.COMPLETED:
        events.append(
            StreamProviderEvent(
                kind=StreamItemKind.USAGE_COMPLETED,
                usage=_USAGE,
                provider_event_type="response.completed",
            )
        )
    events.append(
        StreamProviderEvent(
            kind=terminal_kind,
            data=(
                {"code": "provider_failed"}
                if outcome is StreamTerminalOutcome.ERRORED
                else None
            ),
            provider_event_type={
                StreamTerminalOutcome.COMPLETED: "response.completed",
                StreamTerminalOutcome.ERRORED: "response.failed",
                StreamTerminalOutcome.CANCELLED: "response.cancelled",
            }[outcome],
        )
    )
    return TextGenerationNonStreamResult(
        events,
        answer_text=_ANSWER,
        provider_family="openai",
        usage=_USAGE,
    )


def _response(result: TextGenerationNonStreamResult) -> TextGenerationResponse:
    settings = GenerationSettings()
    return TextGenerationResponse(
        lambda **_: result,
        logger=getLogger("reasoning-summary-non-stream"),
        generation_settings=settings,
        settings=settings,
        use_async_generator=False,
    )


async def _canonical_items(
    response: TextGenerationResponse,
) -> tuple[CanonicalStreamItem, ...]:
    return tuple(
        [
            item
            async for item in response.canonical_stream(
                stream_session_id="summary-sdk-stream",
                run_id="summary-sdk-run",
                turn_id="summary-sdk-turn",
                provider_family="openai",
            )
        ]
    )


@pytest.mark.anyio
async def test_summary_never_contaminates_answer() -> None:
    response = _response(_rich_result())

    assert await response.to_str() == _ANSWER
    assert await response.to_json() == _ANSWER
    assert _SUMMARY_SENTINEL not in await response.to_str()


@pytest.mark.anyio
async def test_summary_isolated_from_answer_tools_and_memory() -> None:
    response = _response(_rich_result())
    items = await _canonical_items(response)
    accumulator = accumulate_canonical_stream_items(items)

    assert accumulator.answer_text == _ANSWER
    assert accumulator.reasoning_text == _SUMMARY_SENTINEL
    assert accumulator.tool_call_arguments == {"call-1": '{"query":"safe"}'}
    outward_values = (
        accumulator.answer_text,
        *accumulator.tool_call_arguments.values(),
        *accumulator.tool_execution_outputs.values(),
    )
    assert all(_SUMMARY_SENTINEL not in value for value in outward_values)
    assert all(_ENCRYPTED_SENTINEL not in value for value in outward_values)


@pytest.mark.anyio
async def test_reasoning_tokens_and_summary_characters_are_separate() -> None:
    response = _response(_rich_result())
    items = await _canonical_items(response)
    accumulator = accumulate_canonical_stream_items(items)
    observation = usage_observation_from_response(response)

    assert observation is not None
    assert observation.totals.reasoning_tokens == 2
    assert accumulator.retained_reasoning_characters == len(_SUMMARY_SENTINEL)
    assert accumulator.reasoning_truncation.dropped_characters == 0
    assert len(_SUMMARY_SENTINEL) != observation.totals.reasoning_tokens


@pytest.mark.anyio
async def test_rich_non_stream_result_uses_canonical_normalization() -> None:
    result = _rich_result()
    response = _response(result)
    items = await _canonical_items(response)
    accumulator = accumulate_canonical_stream_items(items)

    assert items[0].kind is StreamItemKind.STREAM_STARTED
    assert [item.kind for item in items[-3:]] == [
        StreamItemKind.USAGE_COMPLETED,
        StreamItemKind.STREAM_COMPLETED,
        StreamItemKind.STREAM_CLOSED,
    ]
    assert (
        accumulator.reasoning_segments[0].provider_item_id
        == "reasoning-item-1"
    )
    assert accumulator.reasoning_segments[0].output_index == 0
    assert accumulator.reasoning_segments[0].summary_index == 0
    assert accumulator.reasoning_segments[0].completed
    assert result.content == _ANSWER
    assert await result.to_str() == _ANSWER


@pytest.mark.anyio
async def test_rich_non_stream_result_supports_direct_async_iteration() -> (
    None
):
    result = _rich_result()

    first = await result.__anext__()

    assert first.kind is StreamItemKind.STREAM_STARTED
    assert result.__aiter__() is result
    restarted = await result.__anext__()
    assert restarted.kind is StreamItemKind.STREAM_STARTED


@pytest.mark.anyio
async def test_non_stream_terminal_status_preserves_partial_answer() -> None:
    for outcome in (
        StreamTerminalOutcome.ERRORED,
        StreamTerminalOutcome.CANCELLED,
    ):
        response = _response(_rich_result(outcome=outcome))
        items = await _canonical_items(response)
        accumulator = accumulate_canonical_stream_items(items)

        assert accumulator.answer_text == _ANSWER
        assert accumulator.terminal_outcome is outcome
        assert all(
            segment.terminal_outcome is outcome
            for segment in accumulator.reasoning_segments
        )
        assert all(
            not segment.completed for segment in accumulator.reasoning_segments
        )
