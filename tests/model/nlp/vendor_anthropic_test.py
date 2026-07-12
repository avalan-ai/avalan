import asyncio
import importlib
import sys
import types
from base64 import b64encode
from contextlib import AsyncExitStack
from dataclasses import dataclass
from json import loads
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from avalan.entities import (
    GenerationSettings,
    Message,
    MessageContentFile,
    MessageContentImage,
    MessageContentText,
    MessageRole,
    MessageToolCall,
    ReasoningEffort,
    ReasoningSettings,
    ToolCall,
    ToolCallDiagnostic,
    ToolCallDiagnosticCode,
    ToolCallDiagnosticStage,
    ToolCallError,
    ToolCallResult,
    ToolManagerSettings,
    ToolNamePolicyMode,
    ToolNamePolicySettings,
    TransformerEngineSettings,
)
from avalan.model.stream import (
    StreamItemCorrelation,
    StreamItemKind,
    StreamReasoningRepresentation,
    StreamVisibility,
    accumulate_canonical_stream_items,
)
from avalan.task.usage import (
    usage_observation_from_response,
    usage_totals_from_response,
)
from avalan.tool import ToolSet
from avalan.tool.manager import ToolManager


class AsyncIter:
    def __init__(self, items):
        self._iter = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration as exc:  # pragma: no cover - helper
            raise StopAsyncIteration from exc


class PolicyAdder:
    def __init__(self) -> None:
        self.__name__ = "adder"

    async def __call__(self, a: int, b: int) -> int:
        """Return the sum of two integers."""
        return a + b


def _sanitized_policy_manager() -> ToolManager:
    return ToolManager.create_instance(
        enable_tools=["math.adder"],
        available_toolsets=[ToolSet(namespace="math", tools=[PolicyAdder()])],
        settings=ToolManagerSettings(
            tool_name_policy=ToolNamePolicySettings(
                mode=ToolNamePolicyMode.SANITIZED
            )
        ),
    )


@pytest.fixture(scope="module")
def anthropic_mod():
    class APIStatusError(Exception):
        def __init__(self, message, *, response=None, body=None):
            super().__init__(message)
            self.status_code = getattr(response, "status_code", None)
            self.body = body

    class NotFoundError(APIStatusError):
        pass

    class DeltaEvent:
        def __init__(self, delta, index=0):
            self.delta = delta
            self.index = index

    class StopEvent:
        pass

    stub = types.ModuleType("anthropic")
    stub.APIStatusError = APIStatusError
    stub.AsyncAnthropic = MagicMock()
    stub.NotFoundError = NotFoundError
    types_mod = types.ModuleType("anthropic.types")
    types_mod.RawContentBlockDeltaEvent = DeltaEvent
    types_mod.RawMessageStopEvent = StopEvent
    stub.types = types_mod
    patcher = patch.dict(
        sys.modules, {"anthropic": stub, "anthropic.types": types_mod}
    )
    patcher.start()
    mod = importlib.import_module("avalan.model.nlp.text.vendor.anthropic")
    yield mod, stub
    patcher.stop()


def test_stream_variants(anthropic_mod):
    mod, _ = anthropic_mod

    async def agen():
        yield SimpleNamespace(
            type="content_block_start",
            content_block=SimpleNamespace(
                type="tool_use", id="tid", name="tname"
            ),
            index=0,
        )
        yield mod.RawContentBlockDeltaEvent(SimpleNamespace())
        yield mod.RawContentBlockDeltaEvent(SimpleNamespace(thinking="think"))
        yield mod.RawContentBlockDeltaEvent(
            SimpleNamespace(partial_json='{"a":1}')
        )
        yield mod.RawContentBlockDeltaEvent(
            SimpleNamespace(partial_json="frag")
        )
        yield mod.RawContentBlockDeltaEvent(SimpleNamespace(text="txt"))
        yield SimpleNamespace(
            type="content_block_stop",
            content_block=SimpleNamespace(
                type="tool_use", id="tid", name="tname", input={"x": 1}
            ),
            index=0,
        )
        yield SimpleNamespace(type="message_stop")

    async def collect():
        stream = mod.AnthropicStream(agen())
        return [item async for item in stream]

    items = asyncio.run(collect())
    accumulator = accumulate_canonical_stream_items(items)

    assert [item.kind for item in items] == [
        StreamItemKind.STREAM_STARTED,
        StreamItemKind.REASONING_DELTA,
        StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
        StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
        StreamItemKind.ANSWER_DELTA,
        StreamItemKind.TOOL_CALL_READY,
        StreamItemKind.TOOL_CALL_DONE,
        StreamItemKind.ANSWER_DONE,
        StreamItemKind.REASONING_DONE,
        StreamItemKind.STREAM_COMPLETED,
        StreamItemKind.STREAM_CLOSED,
    ]
    assert accumulator.reasoning_text == "think"
    reasoning = next(
        item for item in items if item.kind is StreamItemKind.REASONING_DELTA
    )
    assert (
        reasoning.reasoning_representation
        is StreamReasoningRepresentation.NATIVE_TEXT
    )
    assert reasoning.segment_instance_ordinal == 0
    assert reasoning.visibility is StreamVisibility.PRIVATE
    assert reasoning.correlation.provider_output_index == 0
    assert accumulator.answer_text == "txt"
    assert accumulator.tool_call_arguments == {"tid": '{"a":1}frag'}
    ready = next(
        item for item in items if item.kind is StreamItemKind.TOOL_CALL_READY
    )
    assert ready.data == {"name": "tname"}


def test_stream_emits_stop_block_tool_input_when_no_deltas(anthropic_mod):
    mod, _ = anthropic_mod

    async def agen():
        yield SimpleNamespace(
            type="content_block_start",
            content_block=SimpleNamespace(
                type="tool_use", id="tid", name="tname"
            ),
            index=0,
        )
        yield SimpleNamespace(
            type="content_block_stop",
            content_block=SimpleNamespace(
                type="tool_use", id="tid", name="tname", input={"x": 1}
            ),
            index=0,
        )
        yield SimpleNamespace(type="message_stop")

    async def collect():
        stream = mod.AnthropicStream(agen())
        return [item async for item in stream]

    items = asyncio.run(collect())
    argument_items = [
        item
        for item in items
        if item.kind is StreamItemKind.TOOL_CALL_ARGUMENT_DELTA
    ]

    assert len(argument_items) == 1
    assert argument_items[0].text_delta == '{"x": 1}'
    accumulator = accumulate_canonical_stream_items(items)
    assert accumulator.tool_call_arguments == {"tid": '{"x": 1}'}


def test_stream_records_usage_on_message_stop(anthropic_mod):
    mod, _ = anthropic_mod
    usage_payload = {
        "type": "message_delta",
        "usage": {
            "output_tokens": 5,
            "output_tokens_details": {"thinking_tokens": 4},
        },
    }

    class ModelDumpUsageDelta:
        type = "message_delta"
        usage = usage_payload["usage"]

        def model_dump(self, *, mode: str) -> dict[str, object]:
            assert mode == "json"
            return usage_payload

    async def agen():
        yield SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=7,
                    cache_read_input_tokens=2,
                    cache_creation_input_tokens=3,
                    cache_creation=SimpleNamespace(
                        ephemeral_5m_input_tokens=11,
                        ephemeral_1h_input_tokens=13,
                    ),
                )
            ),
        )
        yield mod.RawContentBlockDeltaEvent(SimpleNamespace(thinking="think"))
        yield SimpleNamespace(type="message_delta", usage=SimpleNamespace())
        yield ModelDumpUsageDelta()
        yield SimpleNamespace(type="message_delta", usage=SimpleNamespace())
        yield SimpleNamespace(type="message_stop")

    async def collect():
        stream = mod.AnthropicStream(agen())
        return stream, [item async for item in stream]

    stream, items = asyncio.run(collect())
    accumulator = accumulate_canonical_stream_items(items)
    usage_item = next(
        item for item in items if item.kind is StreamItemKind.USAGE_COMPLETED
    )
    observation = usage_observation_from_response(stream)
    totals = usage_totals_from_response(stream)

    assert usage_item.provider_payload == usage_payload
    assert usage_item.provider_event_type == "message_delta"
    assert accumulator.reasoning_text == "think"
    assert accumulator.final_usage == {
        "input_tokens": 7,
        "cache_read_input_tokens": 2,
        "cache_creation_input_tokens": 3,
        "cache_creation": SimpleNamespace(
            ephemeral_5m_input_tokens=11,
            ephemeral_1h_input_tokens=13,
        ),
        "output_tokens": 5,
        "output_tokens_details": {"thinking_tokens": 4},
    }
    assert stream.provider_family == "anthropic"
    assert observation is not None
    assert observation.metadata == {
        "provider_family": "anthropic",
        "cache_creation_ephemeral_5m_input_tokens": 11,
        "cache_creation_ephemeral_1h_input_tokens": 13,
    }
    assert totals is not None
    assert totals.input_tokens == 7
    assert totals.cached_input_tokens == 2
    assert totals.cache_creation_input_tokens == 3
    assert totals.output_tokens == 5
    assert totals.reasoning_tokens == 4


def test_stream_records_mapping_usage_on_message_stop(anthropic_mod):
    mod, _ = anthropic_mod
    usage_payload = {
        "type": "message_delta",
        "usage": {
            "input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "output_tokens": 0,
        },
    }

    async def agen():
        yield usage_payload
        yield {"type": "message_stop"}

    async def collect():
        stream = mod.AnthropicStream(agen())
        return stream, [item async for item in stream]

    stream, items = asyncio.run(collect())
    accumulator = accumulate_canonical_stream_items(items)
    usage_item = next(
        item for item in items if item.kind is StreamItemKind.USAGE_COMPLETED
    )
    totals = usage_totals_from_response(stream)

    assert usage_item.provider_payload == usage_payload
    assert usage_item.provider_event_type == "message_delta"
    assert accumulator.final_usage == {
        "input_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "output_tokens": 0,
    }
    assert totals is not None
    assert totals.input_tokens == 0
    assert totals.cached_input_tokens == 0
    assert totals.cache_creation_input_tokens == 0
    assert totals.output_tokens == 0
    assert totals.reasoning_tokens is None
    assert totals.total_tokens is None


def test_stream_public_iterator_yields_canonical_items(anthropic_mod):
    mod, _ = anthropic_mod

    async def agen():
        yield mod.RawContentBlockDeltaEvent(SimpleNamespace(text="hi"))

    async def collect():
        stream = mod.AnthropicStream(agen())
        return [item async for item in stream]

    items = asyncio.run(collect())

    assert [item.kind for item in items] == [
        StreamItemKind.STREAM_STARTED,
        StreamItemKind.ANSWER_DELTA,
        StreamItemKind.ANSWER_DONE,
        StreamItemKind.STREAM_COMPLETED,
        StreamItemKind.STREAM_CLOSED,
    ]
    assert items[1].text_delta == "hi"
    assert {item.provider_family for item in items} == {"anthropic"}


def test_stream_direct_anext_yields_canonical_items(anthropic_mod):
    mod, _ = anthropic_mod

    async def agen():
        yield mod.RawContentBlockDeltaEvent(SimpleNamespace(text="hi"))

    async def collect():
        stream = mod.AnthropicStream(agen())
        return await stream.__anext__(), await stream.__anext__()

    started, delta = asyncio.run(collect())

    assert started.kind is StreamItemKind.STREAM_STARTED
    assert delta.kind is StreamItemKind.ANSWER_DELTA
    assert delta.text_delta == "hi"
    assert delta.provider_family == "anthropic"


def test_canonical_stream_maps_anthropic_events(anthropic_mod):
    mod, _ = anthropic_mod

    async def agen():
        yield SimpleNamespace(
            type="content_block_start",
            content_block=SimpleNamespace(
                type="tool_use", id="call-1", name="lookup"
            ),
            index=0,
        )
        yield mod.RawContentBlockDeltaEvent(SimpleNamespace(thinking="think"))
        yield mod.RawContentBlockDeltaEvent(
            SimpleNamespace(partial_json='{"q":')
        )
        yield mod.RawContentBlockDeltaEvent(SimpleNamespace(text="answer"))
        yield mod.RawContentBlockDeltaEvent(
            SimpleNamespace(partial_json='"v"}')
        )
        yield SimpleNamespace(type="content_block_stop", index=0)
        yield SimpleNamespace(
            type="message_delta",
            usage={"input_tokens": 2, "output_tokens": 3},
        )
        yield SimpleNamespace(type="message_stop")

    async def collect():
        stream = mod.AnthropicStream(agen())
        return [
            item
            async for item in stream.canonical_stream(
                stream_session_id="session",
                run_id="run",
                turn_id="turn",
            )
        ]

    items = asyncio.run(collect())

    assert [item.kind for item in items] == [
        StreamItemKind.STREAM_STARTED,
        StreamItemKind.REASONING_DELTA,
        StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
        StreamItemKind.ANSWER_DELTA,
        StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
        StreamItemKind.TOOL_CALL_READY,
        StreamItemKind.TOOL_CALL_DONE,
        StreamItemKind.ANSWER_DONE,
        StreamItemKind.REASONING_DONE,
        StreamItemKind.USAGE_COMPLETED,
        StreamItemKind.STREAM_COMPLETED,
        StreamItemKind.STREAM_CLOSED,
    ]
    accumulator = accumulate_canonical_stream_items(items)
    assert accumulator.answer_text == "answer"
    assert accumulator.reasoning_text == "think"
    assert accumulator.tool_call_arguments == {"call-1": '{"q":"v"}'}
    assert accumulator.final_usage == {
        "input_tokens": 2,
        "output_tokens": 3,
    }
    ready = next(
        item for item in items if item.kind is StreamItemKind.TOOL_CALL_READY
    )
    assert ready.data == {"name": "lookup"}
    reasoning = next(
        item for item in items if item.kind is StreamItemKind.REASONING_DELTA
    )
    assert (
        reasoning.reasoning_representation
        is StreamReasoningRepresentation.NATIVE_TEXT
    )
    assert reasoning.segment_instance_ordinal == 0
    assert reasoning.visibility is StreamVisibility.PRIVATE
    assert reasoning.correlation.provider_output_index == 0
    assert mod.AnthropicStream._reasoning_correlation({}) == (
        StreamItemCorrelation()
    )
    with pytest.raises(ValueError, match="non-negative integer"):
        mod.AnthropicStream._reasoning_correlation({"index": True})


def test_canonical_stream_uses_tool_name_policy(anthropic_mod):
    mod, _ = anthropic_mod

    async def agen():
        yield SimpleNamespace(
            type="content_block_start",
            content_block=SimpleNamespace(
                type="tool_use", id="call-1", name="math_adder"
            ),
            index=0,
        )
        yield mod.RawContentBlockDeltaEvent(
            SimpleNamespace(partial_json='{"a":1}')
        )
        yield SimpleNamespace(type="content_block_stop", index=0)
        yield SimpleNamespace(type="message_stop")

    async def collect():
        stream = mod.AnthropicStream(agen(), tool=_sanitized_policy_manager())
        return [item async for item in stream]

    items = asyncio.run(collect())

    ready = next(
        item for item in items if item.kind is StreamItemKind.TOOL_CALL_READY
    )
    assert ready.data == {"name": "math.adder"}
    assert accumulate_canonical_stream_items(items).tool_call_arguments == {
        "call-1": '{"a":1}'
    }


def test_canonical_stream_preserves_anthropic_model_dump_payloads(
    anthropic_mod,
):
    mod, _ = anthropic_mod
    delta_payload = {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"partial_json": '{"q":'},
    }
    stop_payload = {
        "type": "content_block_stop",
        "index": 0,
        "content_block": {"type": "tool_use", "id": "call-1"},
    }
    message_payload = {
        "type": "message_stop",
        "usage": {"input_tokens": 2, "output_tokens": 1},
    }
    modes: list[tuple[str, str]] = []

    class ModelDumpDelta(mod.RawContentBlockDeltaEvent):
        def model_dump(self, *, mode: str) -> dict[str, object]:
            modes.append(("delta", mode))
            return delta_payload

    class ModelDumpStop:
        type = "content_block_stop"
        content_block = SimpleNamespace(type="tool_use", id="call-1")
        index = 0

        def model_dump(self, *, mode: str) -> dict[str, object]:
            modes.append(("stop", mode))
            return stop_payload

    class ModelDumpMessageStop:
        type = "message_stop"
        usage = {"input_tokens": 2, "output_tokens": 1}

        def model_dump(self, *, mode: str) -> dict[str, object]:
            modes.append(("message_stop", mode))
            return message_payload

    async def agen():
        yield SimpleNamespace(
            type="content_block_start",
            content_block=SimpleNamespace(
                type="tool_use", id="call-1", name="lookup"
            ),
            index=0,
        )
        yield ModelDumpDelta(SimpleNamespace(partial_json='{"q":'), index=0)
        yield ModelDumpStop()
        yield ModelDumpMessageStop()

    async def collect():
        stream = mod.AnthropicStream(agen())
        return [
            item
            async for item in stream.canonical_stream(
                stream_session_id="session",
                run_id="run",
                turn_id="turn",
                close_after_terminal=False,
            )
        ]

    items = asyncio.run(collect())

    assert modes == [
        ("delta", "json"),
        ("stop", "json"),
        ("message_stop", "json"),
    ]
    assert [item.kind for item in items] == [
        StreamItemKind.STREAM_STARTED,
        StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
        StreamItemKind.TOOL_CALL_READY,
        StreamItemKind.TOOL_CALL_DONE,
        StreamItemKind.USAGE_COMPLETED,
        StreamItemKind.STREAM_COMPLETED,
    ]
    assert items[1].provider_payload == delta_payload
    assert items[2].provider_payload == stop_payload
    assert items[3].provider_payload == stop_payload
    assert items[4].provider_payload == message_payload
    assert items[5].provider_payload == message_payload
    accumulator = accumulate_canonical_stream_items(items)
    assert accumulator.final_usage == {"input_tokens": 2, "output_tokens": 1}


def test_canonical_stream_ignores_anthropic_non_object_provider_payload(
    anthropic_mod,
):
    mod, _ = anthropic_mod

    class ModelDumpDelta(mod.RawContentBlockDeltaEvent):
        def model_dump(self, *, mode: str) -> object:
            return ["not", "an", "event", mode]

    async def agen():
        yield ModelDumpDelta(SimpleNamespace(text="answer"))

    async def collect():
        stream = mod.AnthropicStream(agen())
        return [
            item
            async for item in stream.canonical_stream(
                stream_session_id="session",
                run_id="run",
                turn_id="turn",
                close_after_terminal=False,
            )
        ]

    items = asyncio.run(collect())

    assert items[1].text_delta == "answer"
    assert items[1].provider_payload is None


def test_canonical_stream_maps_malformed_anthropic_tool_delta_to_error(
    anthropic_mod,
):
    mod, _ = anthropic_mod

    async def agen():
        yield mod.RawContentBlockDeltaEvent(
            SimpleNamespace(partial_json='{"q":1}')
        )

    async def collect():
        stream = mod.AnthropicStream(agen())
        return [
            item
            async for item in stream.canonical_stream(
                stream_session_id="session",
                run_id="run",
                turn_id="turn",
            )
        ]

    items = asyncio.run(collect())

    assert [item.kind for item in items] == [
        StreamItemKind.STREAM_STARTED,
        StreamItemKind.STREAM_ERRORED,
        StreamItemKind.STREAM_CLOSED,
    ]
    assert "missing start event" in str(items[1].data)


def test_canonical_stream_maps_duplicate_anthropic_tool_stop_to_error(
    anthropic_mod,
):
    mod, _ = anthropic_mod

    async def agen():
        tool_block = SimpleNamespace(
            type="tool_use", id="call-1", name="lookup"
        )
        yield SimpleNamespace(
            type="content_block_start",
            content_block=tool_block,
            index=0,
        )
        yield SimpleNamespace(
            type="content_block_stop",
            content_block=tool_block,
            index=0,
        )
        yield SimpleNamespace(
            type="content_block_stop",
            content_block=tool_block,
            index=0,
        )

    async def collect():
        stream = mod.AnthropicStream(agen())
        return [
            item
            async for item in stream.canonical_stream(
                stream_session_id="session",
                run_id="run",
                turn_id="turn",
            )
        ]

    items = asyncio.run(collect())

    assert [item.kind for item in items] == [
        StreamItemKind.STREAM_STARTED,
        StreamItemKind.TOOL_CALL_READY,
        StreamItemKind.TOOL_CALL_DONE,
        StreamItemKind.STREAM_ERRORED,
        StreamItemKind.STREAM_CLOSED,
    ]


def test_canonical_stream_anthropic_mapping_edge_cases(anthropic_mod):
    mod, _ = anthropic_mod
    stream = mod.AnthropicStream(AsyncIter([]))

    assert (
        stream._provider_events_from_event(
            SimpleNamespace(
                type="content_block_start",
                content_block=SimpleNamespace(type="text"),
                index=0,
            )
        )
        == ()
    )
    assert (
        stream._provider_events_from_event(
            mod.RawContentBlockDeltaEvent(SimpleNamespace())
        )
        == ()
    )
    assert (
        stream._provider_events_from_event(
            SimpleNamespace(type="content_block_stop", index=0)
        )
        == ()
    )

    assert stream._mark_tool_ready("call-1", "lookup", None)
    assert stream._mark_tool_ready("call-1", "lookup", None) == ()

    invalid_events = [
        SimpleNamespace(type=1),
        SimpleNamespace(
            type="content_block_start",
            content_block=SimpleNamespace(
                type="tool_use", id="call-1", name="lookup"
            ),
            index="bad",
        ),
        SimpleNamespace(
            type="content_block_start",
            content_block=SimpleNamespace(
                type="tool_use", id="call-1", name=1
            ),
            index=0,
        ),
        SimpleNamespace(
            type="content_block_start",
            content_block=SimpleNamespace(
                type="tool_use", id=None, name="lookup"
            ),
            index=0,
        ),
        mod.RawContentBlockDeltaEvent(SimpleNamespace(partial_json=1)),
        mod.RawContentBlockDeltaEvent(
            SimpleNamespace(partial_json="{}"), index="bad"
        ),
        SimpleNamespace(type="content_block_stop", index="bad"),
        SimpleNamespace(
            type="content_block_stop",
            content_block=SimpleNamespace(
                type="tool_use", id="call-1", name=1
            ),
            index=0,
        ),
    ]
    for event in invalid_events:
        with pytest.raises(ValueError):
            stream._provider_events_from_event(event)


def test_anthropic_provider_events_stop_after_message_stop(anthropic_mod):
    mod, _ = anthropic_mod

    async def agen():
        yield {"type": "message_stop"}
        yield mod.RawContentBlockDeltaEvent(SimpleNamespace(text="late"))

    async def collect():
        stream = mod.AnthropicStream(agen())
        return [event async for event in stream._provider_events()]

    events = asyncio.run(collect())

    assert [event.kind for event in events] == [
        StreamItemKind.STREAM_COMPLETED
    ]


def test_stream_without_message_stop_drops_cumulative_usage(anthropic_mod):
    mod, _ = anthropic_mod

    async def agen():
        yield SimpleNamespace(
            type="message_delta",
            usage={"input_tokens": 1, "output_tokens": 2},
        )

    async def collect():
        stream = mod.AnthropicStream(agen())
        return stream, [item async for item in stream]

    stream, items = asyncio.run(collect())

    assert [item.kind for item in items] == [
        StreamItemKind.STREAM_STARTED,
        StreamItemKind.STREAM_COMPLETED,
        StreamItemKind.STREAM_CLOSED,
    ]
    assert stream.usage is None
    assert usage_totals_from_response(stream) is None


def test_stream_failure_before_message_stop_drops_cumulative_usage(
    anthropic_mod,
):
    mod, _ = anthropic_mod

    class FailingIter:
        def __init__(self):
            self._count = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            self._count += 1
            if self._count == 1:
                return SimpleNamespace(
                    type="message_delta",
                    usage={"input_tokens": 1, "output_tokens": 2},
                )
            raise RuntimeError("provider failure")

    async def collect():
        stream = mod.AnthropicStream(FailingIter())
        return stream, [item async for item in stream]

    stream, items = asyncio.run(collect())

    assert [item.kind for item in items] == [
        StreamItemKind.STREAM_STARTED,
        StreamItemKind.STREAM_ERRORED,
        StreamItemKind.STREAM_CLOSED,
    ]
    assert "provider failure" in str(items[1].data)
    assert stream.usage is None
    assert usage_totals_from_response(stream) is None


def test_client_call_and_model(anthropic_mod):
    mod, stub = anthropic_mod
    ctx_instance = SimpleNamespace(
        __aenter__=AsyncMock(return_value=AsyncIter([])),
        __aexit__=AsyncMock(return_value=False),
    )
    stub.AsyncAnthropic.return_value.messages.stream = MagicMock(
        return_value=ctx_instance
    )
    exit_stack = AsyncMock(spec=AsyncExitStack)
    client = mod.AnthropicClient("tok", "url", exit_stack=exit_stack)
    client._system_prompt = MagicMock(return_value="sys")
    client._template_messages = MagicMock(return_value=[{"content": "c"}])

    async def invoke():
        with patch.object(
            mod.AnthropicClient, "_tool_schemas", return_value=[{"n": 1}]
        ) as ts:
            result = await client("m", [], tool=MagicMock())
        return result, ts

    result, ts = asyncio.run(invoke())
    stub.AsyncAnthropic.assert_called_once_with(api_key="tok", base_url="url")
    exit_stack.enter_async_context.assert_awaited_once_with(ctx_instance)
    client._client.messages.stream.assert_called_once()
    ts.assert_called_once()
    assert isinstance(result, mod.AnthropicStream)

    with patch.object(mod, "AnthropicClient") as ClientMock:
        settings = TransformerEngineSettings(
            auto_load_model=False,
            auto_load_tokenizer=False,
            access_token="t",
            base_url="b",
        )
        model = mod.AnthropicModel("m", settings)
        loaded = model._load_model()
    ClientMock.assert_called_once_with(
        api_key="t", base_url="b", exit_stack=model._exit_stack
    )
    assert loaded is ClientMock.return_value


def test_client_stream_passes_tool_manager_to_stream(anthropic_mod):
    mod, stub = anthropic_mod
    manager = _sanitized_policy_manager()
    stub.AsyncAnthropic.return_value.messages.stream = MagicMock(
        return_value=object()
    )
    exit_stack = AsyncMock(spec=AsyncExitStack)
    exit_stack.enter_async_context.return_value = AsyncIter([])
    client = mod.AnthropicClient("tok", "url", exit_stack=exit_stack)
    client._template_messages = MagicMock(return_value=[{"content": "c"}])

    async def invoke():
        with patch.object(mod, "AnthropicStream") as StreamMock:
            result = await client("m", [], tool=manager)
        return result, StreamMock

    result, stream_mock = asyncio.run(invoke())

    assert result is stream_mock.return_value
    assert stream_mock.call_args.kwargs["tool"] is manager


def test_provider_instructions_are_rejected_before_api_call(anthropic_mod):
    mod, stub = anthropic_mod
    exit_stack = AsyncExitStack()
    client = mod.AnthropicClient("key", exit_stack=exit_stack)
    create_mock = stub.AsyncAnthropic.return_value.messages.create
    create_mock.reset_mock()

    async def invoke():
        with pytest.raises(AssertionError, match="provider instructions"):
            await client("model", [], instructions="private policy")

    asyncio.run(invoke())
    create_mock.assert_not_called()


def test_client_omits_unset_temperature_and_forwards_explicit(
    anthropic_mod,
):
    mod, stub = anthropic_mod
    exit_stack = AsyncExitStack()
    client = mod.AnthropicClient("key", exit_stack=exit_stack)
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")]
    )
    create_mock = AsyncMock(return_value=response)
    stub.AsyncAnthropic.return_value.messages.create = create_mock

    asyncio.run(
        client(
            "model",
            [Message(role=MessageRole.USER, content="hi")],
            settings=GenerationSettings(max_new_tokens=32, temperature=None),
            use_async_generator=False,
        )
    )

    kwargs = create_mock.await_args.kwargs
    assert kwargs["max_tokens"] == 32
    assert "temperature" not in kwargs

    create_mock.reset_mock()
    asyncio.run(
        client(
            "model",
            [Message(role=MessageRole.USER, content="hi")],
            settings=GenerationSettings(max_new_tokens=32, temperature=0.25),
            use_async_generator=False,
        )
    )

    kwargs = create_mock.await_args.kwargs
    assert kwargs["temperature"] == 0.25


def test_client_non_stream_tool_messages(anthropic_mod):
    mod, stub = anthropic_mod
    exit_stack = AsyncExitStack()
    client = mod.AnthropicClient("key", exit_stack=exit_stack)

    call = ToolCall(id="call1", name="pkg.tool", arguments={"a": 1})
    result = ToolCallResult(
        id="call1", name="pkg.tool", call=call, result={"ok": True}
    )
    messages = [
        Message(role=MessageRole.SYSTEM, content="sys"),
        Message(role=MessageRole.USER, content="hi"),
        Message(role=MessageRole.ASSISTANT, content="ack"),
        Message(role=MessageRole.TOOL, tool_call_result=result),
    ]

    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="hello"),
            SimpleNamespace(
                type="tool_use",
                id="call1",
                name="pkg__tool",
                input={"a": 1},
            ),
        ],
        usage=SimpleNamespace(input_tokens=2),
    )

    stub.AsyncAnthropic.return_value.messages.create = AsyncMock(
        return_value=response
    )

    with patch.object(
        mod.TextGenerationVendor,
        "build_tool_call_text",
        return_value="<tool_call />",
    ) as build_token:
        stream = asyncio.run(
            client(
                "model",
                messages,
                use_async_generator=False,
            )
        )

    from avalan.model.stream import TextGenerationSingleStream

    assert isinstance(stream, TextGenerationSingleStream)
    assert stream.content == "hello<tool_call />"
    assert stream.usage.input_tokens == 2
    build_token.assert_called_once_with("call1", "pkg__tool", {"a": 1})

    create_mock = stub.AsyncAnthropic.return_value.messages.create
    create_mock.assert_awaited_once()
    kwargs = create_mock.await_args.kwargs
    assert all(msg["role"] != "tool" for msg in kwargs["messages"])


def test_client_forwards_reasoning_effort(anthropic_mod):
    mod, stub = anthropic_mod
    exit_stack = AsyncExitStack()
    client = mod.AnthropicClient("key", exit_stack=exit_stack)

    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")]
    )
    stub.AsyncAnthropic.return_value.messages.create = AsyncMock(
        return_value=response
    )

    asyncio.run(
        client(
            "model",
            [Message(role=MessageRole.USER, content="hi")],
            settings=GenerationSettings(
                reasoning=ReasoningSettings(effort=ReasoningEffort.XHIGH)
            ),
            use_async_generator=False,
        )
    )

    kwargs = stub.AsyncAnthropic.return_value.messages.create.await_args.kwargs
    assert kwargs["output_config"] == {"effort": "max"}


def test_template_messages_and_exclude_roles(anthropic_mod):
    mod, _ = anthropic_mod
    exit_stack = AsyncExitStack()
    client = mod.AnthropicClient("k", exit_stack=exit_stack)

    @dataclass
    class Res:
        x: int

    call = ToolCall(id="id1", name="pkg.tool", arguments={"a": 1})
    result = ToolCallResult(
        id="id1", name="pkg.tool", call=call, result=Res(x=2)
    )
    messages = [
        Message(role=MessageRole.SYSTEM, content="s"),
        Message(role=MessageRole.USER, content="hi"),
        Message(role=MessageRole.ASSISTANT, content="ok"),
        Message(role=MessageRole.TOOL, tool_call_result=result),
    ]
    templated = client._template_messages(messages, ["system"])
    assert templated[1]["content"][1]["name"] == "avl_cGtnLnRvb2w"
    assert templated[2]["content"][0]["tool_use_id"] == "id1"

    policy_call = ToolCall(
        id="id2",
        name="math.adder",
        arguments={"a": 1, "b": 2},
    )
    policy_result = ToolCallResult(
        id="id2",
        name="math.adder",
        call=policy_call,
        result=3,
    )
    policy_messages = [
        Message(role=MessageRole.USER, content="hi"),
        Message(role=MessageRole.TOOL, tool_call_result=policy_result),
    ]
    policy_templated = client._template_messages(
        policy_messages,
        tool=_sanitized_policy_manager(),
    )
    assert policy_templated[1]["content"][0]["name"] == "math_adder"

    no_assistant = client._template_messages(
        [
            Message(role=MessageRole.SYSTEM, content="s"),
            Message(role=MessageRole.USER, content="hi"),
            Message(role=MessageRole.TOOL, tool_call_result=result),
        ],
        ["system"],
    )
    assert no_assistant[-1]["content"][0]["tool_use_id"] == "id1"
    assert no_assistant[-1]["role"] == "user"

    dup = client._template_messages(
        [
            Message(role=MessageRole.USER, content="x"),
            Message(role=MessageRole.USER, content="x"),
        ]
    )
    assert len(dup) == 1


def test_template_messages_keep_tool_results_with_matching_calls(
    anthropic_mod,
):
    mod, _ = anthropic_mod
    client = mod.AnthropicClient("k", exit_stack=AsyncExitStack())

    first_call = ToolCall(
        id="call1", name="math.calculator", arguments={"expression": "50 / 2"}
    )
    second_call = ToolCall(
        id="call2", name="math.calculator", arguments={"expression": "25 * 2"}
    )
    first_result = ToolCallResult(
        id="result1",
        name="math.calculator",
        call=first_call,
        result="25",
    )
    second_result = ToolCallResult(
        id="result2",
        name="math.calculator",
        call=second_call,
        result="50",
    )

    templated = client._template_messages(
        [
            Message(role=MessageRole.USER, content="first"),
            Message(
                role=MessageRole.ASSISTANT,
                content="calculating",
                tool_calls=[
                    MessageToolCall(
                        id="call1",
                        name="math.calculator",
                        arguments={"expression": "50 / 2"},
                    )
                ],
            ),
            Message(role=MessageRole.TOOL, tool_call_result=first_result),
            Message(role=MessageRole.ASSISTANT, content="25"),
            Message(role=MessageRole.USER, content="and that times two?"),
            Message(
                role=MessageRole.ASSISTANT,
                content="calculating again",
                tool_calls=[
                    MessageToolCall(
                        id="call2",
                        name="math.calculator",
                        arguments={"expression": "25 * 2"},
                    )
                ],
            ),
            Message(role=MessageRole.TOOL, tool_call_result=second_result),
        ],
    )

    assistant_tool_messages = [
        message
        for message in templated
        if message["role"] == str(MessageRole.ASSISTANT)
        and isinstance(message["content"], list)
        and any(
            block.get("type") == "tool_use"
            for block in message["content"]
            if isinstance(block, dict)
        )
    ]
    tool_result_messages = [
        message
        for message in templated
        if message["role"] == str(MessageRole.USER)
        and isinstance(message["content"], list)
        and message["content"]
        and message["content"][0].get("type") == "tool_result"
    ]

    assert [
        block["id"]
        for block in assistant_tool_messages[0]["content"]
        if block.get("type") == "tool_use"
    ] == ["call1"]
    assert [
        block["id"]
        for block in assistant_tool_messages[1]["content"]
        if block.get("type") == "tool_use"
    ] == ["call2"]
    assert tool_result_messages[0]["content"][0]["tool_use_id"] == "call1"
    assert tool_result_messages[1]["content"][0]["tool_use_id"] == "call2"


def test_template_messages_tool_error_details(anthropic_mod):
    mod, _ = anthropic_mod
    exit_stack = AsyncExitStack()
    client = mod.AnthropicClient("k", exit_stack=exit_stack)

    call = ToolCall(id="call1", name="pkg.tool", arguments={"a": 1})
    error = ToolCallError(
        id="err1",
        name="pkg.tool",
        call=call,
        error=Exception("boom"),
        message="boom",
    )
    messages = [
        Message(role=MessageRole.SYSTEM, content="s"),
        Message(role=MessageRole.USER, content="hi"),
        Message(role=MessageRole.ASSISTANT, content="ok"),
        Message(role=MessageRole.TOOL, tool_call_error=error),
    ]
    templated = client._template_messages(messages, ["system"])
    tool_result = templated[2]["content"][0]
    assert tool_result["tool_use_id"] == "call1"
    assert tool_result["is_error"] is True


def test_template_messages_tool_diagnostic_details(anthropic_mod):
    mod, _ = anthropic_mod
    exit_stack = AsyncExitStack()
    client = mod.AnthropicClient("k", exit_stack=exit_stack)

    diagnostic = ToolCallDiagnostic(
        id="diag1",
        call_id="call1",
        requested_name="missing",
        code=ToolCallDiagnosticCode.UNKNOWN_TOOL,
        stage=ToolCallDiagnosticStage.RESOLVE,
        message="Tool is unknown.",
    )
    messages = [
        Message(role=MessageRole.USER, content="hi"),
        Message(
            role=MessageRole.TOOL,
            name="missing",
            arguments={"a": 1},
            tool_call_diagnostic=diagnostic,
        ),
    ]

    templated = client._template_messages(messages)

    tool_use = templated[1]["content"][0]
    tool_result = templated[2]["content"][0]
    assert tool_use["type"] == "tool_use"
    assert tool_use["id"] == "call1"
    assert tool_use["name"] == "missing"
    assert tool_use["input"] == {"a": 1}
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "call1"
    assert tool_result["is_error"] is True
    assert loads(tool_result["content"])["code"] == "tool.unknown"


def test_template_messages_unanchored_tool_diagnostic(anthropic_mod):
    mod, _ = anthropic_mod
    exit_stack = AsyncExitStack()
    client = mod.AnthropicClient("k", exit_stack=exit_stack)
    diagnostic = ToolCallDiagnostic(
        id="diag1",
        code=ToolCallDiagnosticCode.MALFORMED_CALL,
        stage=ToolCallDiagnosticStage.PARSE,
        message="Tool call could not be parsed.",
    )

    templated = client._template_messages(
        [Message(role=MessageRole.TOOL, tool_call_diagnostic=diagnostic)]
    )

    assert templated[0]["role"] == str(MessageRole.ASSISTANT)
    text = templated[0]["content"][0]["text"]
    assert loads(text)["code"] == "tool_call.malformed"


def test_file_content_translation_and_beta_header(anthropic_mod):
    mod, stub = anthropic_mod
    exit_stack = AsyncExitStack()
    client = mod.AnthropicClient("key", exit_stack=exit_stack)
    messages = [
        Message(
            role=MessageRole.USER,
            content=[
                MessageContentText(type="text", text="inspect attachment"),
                MessageContentFile(
                    type="file",
                    file={
                        "citations": True,
                        "context": "ctx",
                        "file_id": "file_1",
                        "title": "Doc",
                    },
                ),
                MessageContentImage(
                    type="image_url", image_url={"file_id": "img_1"}
                ),
                MessageContentText(
                    type="text",
                    text=(
                        "Attached files available to tools:\n"
                        "Use these path values as tool arguments.\n"
                        '- "attachment/report.pdf"'
                    ),
                ),
            ],
        )
    ]

    stub.AsyncAnthropic.return_value.messages.create = AsyncMock(
        return_value=SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ok")]
        )
    )

    asyncio.run(client("model", messages, use_async_generator=False))

    create_mock = stub.AsyncAnthropic.return_value.messages.create
    create_mock.assert_awaited_once()
    kwargs = create_mock.await_args.kwargs
    assert kwargs["extra_headers"] == {
        "anthropic-beta": "files-api-2025-04-14"
    }
    assert kwargs["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "inspect attachment"},
                {
                    "type": "document",
                    "source": {"type": "file", "file_id": "file_1"},
                    "title": "Doc",
                    "context": "ctx",
                    "citations": {"enabled": True},
                },
                {
                    "type": "image",
                    "source": {"type": "file", "file_id": "img_1"},
                },
                {
                    "type": "text",
                    "text": (
                        "Attached files available to tools:\n"
                        "Use these path values as tool arguments.\n"
                        '- "attachment/report.pdf"'
                    ),
                },
            ],
        }
    ]


def test_document_source_variants(anthropic_mod):
    mod, _ = anthropic_mod

    assert mod.AnthropicClient._document_source(
        {"url": "https://example.com/doc.pdf"}
    ) == {"type": "url", "url": "https://example.com/doc.pdf"}
    assert mod.AnthropicClient._document_source(
        {
            "data": "hello",
            "mime_type": "text/plain",
        }
    ) == {
        "type": "text",
        "media_type": "text/plain",
        "data": "hello",
    }
    assert mod.AnthropicClient._document_source(
        {
            "file_data": b64encode(b"hello").decode("ascii"),
            "mime_type": "text/plain",
        }
    ) == {
        "type": "text",
        "media_type": "text/plain",
        "data": "hello",
    }
    assert mod.AnthropicClient._document_source({"file_data": "YWJj"}) == {
        "type": "base64",
        "media_type": "application/pdf",
        "data": "YWJj",
    }


def test_document_source_rejects_missing_source_and_invalid_utf8(
    anthropic_mod,
):
    mod, _ = anthropic_mod
    invalid_utf8 = b64encode(b"\xff\xfe").decode("ascii")

    assert mod.AnthropicClient._document_source(
        {
            "file_data": invalid_utf8,
            "mime_type": "text/plain",
        }
    ) == {
        "type": "text",
        "media_type": "text/plain",
        "data": invalid_utf8,
    }

    with pytest.raises(
        AssertionError,
        match="Anthropic file blocks require file_id, file_url, or file_data",
    ):
        mod.AnthropicClient._document_source({"title": "Doc"})


def test_output_config_and_content_block_variants(anthropic_mod):
    mod, _ = anthropic_mod

    assert mod.AnthropicClient._output_config(
        GenerationSettings(
            reasoning=ReasoningSettings(effort=ReasoningEffort.NONE)
        )
    ) == {"effort": "low"}
    assert mod.AnthropicClient._output_config(
        GenerationSettings(
            reasoning=ReasoningSettings(effort=ReasoningEffort.HIGH)
        )
    ) == {"effort": "high"}
    assert mod.AnthropicClient._content_block(
        {
            "type": "text",
            "text": "hello",
        }
    ) == {"type": "text", "text": "hello"}
    assert mod.AnthropicClient._content_block(
        {
            "type": "other",
            "value": 1,
        }
    ) == {"type": "other", "value": 1}


def test_image_source_and_files_api_variants(anthropic_mod):
    mod, _ = anthropic_mod

    assert mod.AnthropicClient._image_source(
        {"url": "https://example.com/image.png"}
    ) == {"type": "url", "url": "https://example.com/image.png"}
    assert mod.AnthropicClient._image_source(
        {
            "data": "YWJj",
            "mime_type": "image/jpeg",
        }
    ) == {
        "type": "base64",
        "media_type": "image/jpeg",
        "data": "YWJj",
    }
    with pytest.raises(
        AssertionError,
        match="Anthropic image blocks require file_id, url, or data",
    ):
        mod.AnthropicClient._image_source({})

    assert mod.AnthropicClient._uses_files_api(
        [
            Message(
                role=MessageRole.USER,
                content=MessageContentImage(
                    type="image_url", image_url={"file_id": "img_1"}
                ),
            )
        ]
    )


def test_tool_schemas_variants(anthropic_mod):
    mod, _ = anthropic_mod

    class DummyTool:
        def __init__(self, schemas):
            self._schemas = schemas

        def json_schemas(self):
            return self._schemas

    schemas = [
        {
            "type": "function",
            "function": {
                "name": "pkg.tool",
                "description": "d",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {"type": "noop"},
    ]
    out = mod.AnthropicClient._tool_schemas(DummyTool(schemas))
    assert out == [
        {
            "name": "avl_cGtnLnRvb2w",
            "description": "d",
            "input_schema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        }
    ]
    assert mod.AnthropicClient._tool_schemas(DummyTool([])) is None
    assert mod.AnthropicClient._tool_schemas(DummyTool(None)) is None

    policy_out = mod.AnthropicClient._tool_schemas(_sanitized_policy_manager())
    assert policy_out is not None
    assert policy_out[0]["name"] == "math_adder"


def test_non_stream_response_content_from_dict(anthropic_mod):
    mod, _ = anthropic_mod
    response = {
        "content": [
            {"type": "text", "text": "alpha"},
            {
                "type": "tool_use",
                "id": "call-id",
                "name": "pkg.tool",
                "input": {"foo": "bar"},
            },
        ]
    }
    with patch.object(
        mod.TextGenerationVendor,
        "build_tool_call_text",
        return_value="<tool>",
    ) as build:
        text = mod.AnthropicClient._non_stream_response_content(response)

    assert text == "alpha<tool>"
    build.assert_called_once_with("call-id", "pkg.tool", {"foo": "bar"})

    policy_response = {
        "content": [
            {
                "type": "tool_use",
                "id": "call-id",
                "name": "math_adder",
                "input": {"a": 1, "b": 2},
            },
        ]
    }
    policy_text = mod.AnthropicClient._non_stream_response_content(
        policy_response,
        tool=_sanitized_policy_manager(),
    )

    assert '"name": "math.adder"' in policy_text


def test_translate_api_error_for_retired_model(anthropic_mod):
    mod, stub = anthropic_mod
    exit_stack = AsyncExitStack()
    client = mod.AnthropicClient("key", exit_stack=exit_stack)

    error = stub.NotFoundError(
        "missing model",
        response=SimpleNamespace(status_code=404),
        body={
            "error": {
                "type": "not_found_error",
                "message": "model: claude-3-5-sonnet-latest",
            }
        },
    )
    client._client.messages.stream = MagicMock(side_effect=error)

    with pytest.raises(ValueError, match="Use 'claude-sonnet-4-6' instead"):
        asyncio.run(client("claude-3-5-sonnet-latest", []))


def test_translate_api_error_for_unknown_model(anthropic_mod):
    mod, stub = anthropic_mod
    exit_stack = AsyncExitStack()
    client = mod.AnthropicClient("key", exit_stack=exit_stack)

    error = stub.NotFoundError(
        "missing model",
        response=SimpleNamespace(status_code=404),
        body={
            "error": {
                "type": "not_found_error",
                "message": "model: claude-missing",
            }
        },
    )
    client._client.messages.create = AsyncMock(side_effect=error)

    with pytest.raises(
        ValueError, match="Verify the model identifier against Anthropic"
    ):
        asyncio.run(
            client(
                "claude-missing",
                [],
                use_async_generator=False,
            )
        )


def test_stream_maps_tool_events_with_missing_indexes_to_error(
    anthropic_mod,
):
    mod, _ = anthropic_mod

    async def agen():
        yield SimpleNamespace(
            type="content_block_start",
            content_block=SimpleNamespace(type="tool_use", id="x", name="y"),
            index=None,
        )
        yield mod.RawContentBlockDeltaEvent(
            SimpleNamespace(partial_json='{"x":1}'),
            index=None,
        )
        yield SimpleNamespace(type="message_stop")

    async def collect():
        stream = mod.AnthropicStream(agen())
        return [item async for item in stream]

    items = asyncio.run(collect())

    assert [item.kind for item in items] == [
        StreamItemKind.STREAM_STARTED,
        StreamItemKind.STREAM_ERRORED,
        StreamItemKind.STREAM_CLOSED,
    ]
    assert "index must be an integer" in str(items[1].data)


def test_call_reraises_when_translator_does_not_raise(anthropic_mod):
    mod, _ = anthropic_mod
    exit_stack = AsyncMock(spec=AsyncExitStack)
    client = mod.AnthropicClient("key", exit_stack=exit_stack)

    err = RuntimeError("boom")
    client._client.messages.stream = MagicMock(side_effect=err)

    with patch.object(
        mod.AnthropicClient, "_translate_api_error"
    ) as translate:
        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(client("model", []))

    translate.assert_called_once_with("model", err)


def test_error_helpers_cover_false_paths(anthropic_mod):
    mod, stub = anthropic_mod

    error = ValueError("plain")
    assert mod.AnthropicClient._error_message(error) == "plain"
    assert mod.AnthropicClient._is_missing_model_error(error) is False

    not_found_wrong_status = stub.NotFoundError(
        "bad",
        response=SimpleNamespace(status_code=500),
        body={"error": {"message": "model missing"}},
    )
    assert (
        mod.AnthropicClient._is_missing_model_error(not_found_wrong_status)
        is False
    )

    mod.AnthropicClient._translate_api_error("model", RuntimeError("x"))


def test_template_messages_skips_dynamic_none_results(anthropic_mod):
    mod, _ = anthropic_mod
    client = mod.AnthropicClient("k", exit_stack=AsyncExitStack())

    class DynamicToolMessage:
        role = MessageRole.TOOL
        tool_call_error = None

        def __init__(self) -> None:
            self._calls = 0

        @property
        def tool_call_result(self):
            self._calls += 1
            if self._calls == 1:
                return object()
            return None

    with patch.object(
        mod.TextGenerationVendor,
        "_template_messages",
        return_value=[{"role": "assistant", "content": "ok"}],
    ):
        output = client._template_messages([DynamicToolMessage()])

    assert output == [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
        }
    ]


def test_non_stream_response_content_ignores_non_list_content(anthropic_mod):
    mod, _ = anthropic_mod
    response = {"content": "not-a-list"}

    assert mod.AnthropicClient._non_stream_response_content(response) == ""


def test_content_blocks_variants(anthropic_mod):
    mod, _ = anthropic_mod
    assert mod.AnthropicClient._content_blocks([{"a": 1}, "x"]) == [{"a": 1}]
    assert mod.AnthropicClient._content_blocks("v", empty_when_none=True) == []
    assert mod.AnthropicClient._content_blocks(None) == []
