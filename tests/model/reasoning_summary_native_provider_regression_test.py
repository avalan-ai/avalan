import ast
import importlib
import sys
from asyncio import CancelledError, run
from collections.abc import AsyncIterable, AsyncIterator, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import torch

from avalan.entities import GenerationSettings, TransformerEngineSettings
from avalan.model.nlp.text.ds4 import (
    Ds4Worker,
    _Ds4GeneratedChunk,
)
from avalan.model.nlp.text.generation import TextGenerationModel
from avalan.model.nlp.text.mlxlm import MlxLmStream
from avalan.model.nlp.text.vendor.litellm import LiteLLMStream
from avalan.model.nlp.text.vendor.openai import OpenAIStream
from avalan.model.nlp.text.vllm import VllmStream
from avalan.model.stream import (
    CanonicalStreamItem,
    LocalTextStreamEventParser,
    StreamItemCorrelation,
    StreamItemKind,
    StreamProducerBackend,
    StreamProviderCapabilities,
    StreamProviderEvent,
    StreamReasoningRepresentation,
    StreamTerminalOutcome,
    StreamVisibility,
    accumulate_canonical_stream_items,
    normalize_provider_stream,
    validate_canonical_stream_items,
)

_LOCAL_SPLIT_TAG_CHUNKS = (
    "lead",
    "<thi",
    "nk>hidden",
    "</thi",
    "nk>tail",
    "<",
)
_FRAGMENT_PROVENANCE_CASES = {
    "immediate": (("plain",), (("plain", 1),)),
    "incomplete_marker": (("<", "t"), (("<", 1), ("t", 2))),
    "invalidated_marker": (("<", "x"), (("<", 1), ("x", 2))),
    "eof": (("lead<",), (("lead", 1), ("<", 1))),
}
_ABNORMAL_FRAGMENT_CASES = {
    "answer_start_marker": (
        ("lead", "<", "t"),
        (
            (StreamItemKind.ANSWER_DELTA, "lead"),
            (StreamItemKind.ANSWER_DELTA, "<"),
            (StreamItemKind.ANSWER_DELTA, "t"),
        ),
    ),
    "reasoning_end_marker": (
        ("<think>hidden", "<", "/"),
        (
            (StreamItemKind.REASONING_DELTA, "hidden"),
            (StreamItemKind.REASONING_DELTA, "<"),
            (StreamItemKind.REASONING_DELTA, "/"),
        ),
    ),
    "tool_end_marker": (
        ('<tool_call name="lookup">{"q":1}', "<", "/"),
        (
            (StreamItemKind.TOOL_CALL_ARGUMENT_DELTA, '{"q":1}'),
            (StreamItemKind.TOOL_CALL_ARGUMENT_DELTA, "<"),
            (StreamItemKind.TOOL_CALL_ARGUMENT_DELTA, "/"),
        ),
    ),
}
_CANONICAL_TERMINAL_KINDS = {
    StreamItemKind.STREAM_COMPLETED,
    StreamItemKind.STREAM_ERRORED,
    StreamItemKind.STREAM_CANCELLED,
}


async def _collect_items(
    items: AsyncIterable[CanonicalStreamItem],
) -> list[CanonicalStreamItem]:
    return [item async for item in items]


async def _provider_events(
    *events: StreamProviderEvent,
) -> AsyncIterator[StreamProviderEvent]:
    for event in events:
        yield event


async def _raw_events(*events: object) -> AsyncIterator[object]:
    for event in events:
        yield event


class _CancelledAsyncIterator:
    def __aiter__(self) -> "_CancelledAsyncIterator":
        return self

    async def __anext__(self) -> object:
        raise CancelledError

    async def aclose(self) -> None:
        return None


class _FailingIterator:
    def __init__(
        self,
        values: tuple[object, ...],
        failure_type: type[BaseException],
    ) -> None:
        self._values = values
        self._failure_type = failure_type
        self._index = 0
        self.pulls = 0

    def __iter__(self) -> "_FailingIterator":
        return self

    def __next__(self) -> object:
        self.pulls += 1
        if self._index < len(self._values):
            value = self._values[self._index]
            self._index += 1
            return value
        raise self._failure_type("upstream stream failure")

    def __len__(self) -> int:
        return len(self._values)


class _FailingAsyncTextStreamer:
    def __init__(
        self,
        chunks: tuple[str, ...],
        failure_type: type[BaseException],
    ) -> None:
        self._chunks = chunks
        self._failure_type = failure_type
        self._index = 0
        self.pulls = 0
        self.stop_signal = object()

    def __aiter__(self) -> "_FailingAsyncTextStreamer":
        return self

    async def __anext__(self) -> str:
        self.pulls += 1
        if self._index < len(self._chunks):
            chunk = self._chunks[self._index]
            self._index += 1
            return chunk
        raise self._failure_type("upstream stream failure")


class _IdleThread:
    def __init__(
        self,
        target: Any,
        name: str | None = None,
        daemon: bool | None = None,
    ) -> None:
        self.target = target
        self.name = name or "idle-thread"
        self.daemon = daemon
        self.ident = 1
        self.started = False

    def start(self) -> None:
        self.started = True

    def is_alive(self) -> bool:
        return False


class _GeneratedSequenceRow:
    def __init__(self, generated: _FailingIterator) -> None:
        self._generated = generated

    def __getitem__(self, key: slice) -> _FailingIterator:
        assert key == slice(1, None, None)
        return self._generated


@contextmanager
def _isolated_vendor_module(
    module_name: str,
    sdk_modules: dict[str, ModuleType],
) -> Iterator[ModuleType]:
    parent_name, attribute = module_name.rsplit(".", 1)
    parent = importlib.import_module(parent_name)
    missing = object()
    previous_module = sys.modules.pop(module_name, None)
    previous_attribute = getattr(parent, attribute, missing)
    try:
        with patch.dict(sys.modules, sdk_modules):
            module = importlib.import_module(module_name)
            yield module
    finally:
        sys.modules.pop(module_name, None)
        if previous_module is not None:
            sys.modules[module_name] = previous_module
        if previous_attribute is missing:
            try:
                delattr(parent, attribute)
            except AttributeError:
                pass
        else:
            setattr(parent, attribute, previous_attribute)


async def _normalized_provider_items(
    provider_family: str,
    *,
    representation: StreamReasoningRepresentation,
    terminal_kind: StreamItemKind = StreamItemKind.STREAM_COMPLETED,
    provider_event_type: str = "provider.reasoning.delta",
    provider_payload: object | None = None,
) -> list[CanonicalStreamItem]:
    reasoning_correlation = StreamItemCorrelation(
        protocol_item_id="reasoning-item-1",
        provider_output_index=0,
        provider_summary_index=1,
    )
    tool_correlation = StreamItemCorrelation(tool_call_id="call-1")
    terminal_data = (
        {"message": "provider failure"}
        if terminal_kind is StreamItemKind.STREAM_ERRORED
        else None
    )
    terminal_usage = (
        {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}
        if terminal_kind is StreamItemKind.STREAM_COMPLETED
        else None
    )
    events = _provider_events(
        StreamProviderEvent(
            kind=StreamItemKind.REASONING_DELTA,
            text_delta="reasoning",
            correlation=reasoning_correlation,
            visibility=StreamVisibility.PRIVATE,
            reasoning_representation=representation,
            segment_instance_ordinal=0,
            metadata={"safe": "metadata"},
            provider_payload=cast(Any, provider_payload),
            provider_event_type=provider_event_type,
        ),
        StreamProviderEvent(
            kind=StreamItemKind.ANSWER_DELTA,
            text_delta="answer",
            provider_event_type="provider.answer.delta",
        ),
        StreamProviderEvent(
            kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
            text_delta='{"value":1}',
            correlation=tool_correlation,
        ),
        StreamProviderEvent(
            kind=StreamItemKind.TOOL_CALL_READY,
            data={"name": "lookup"},
            correlation=tool_correlation,
        ),
        StreamProviderEvent(
            kind=StreamItemKind.TOOL_CALL_DONE,
            correlation=tool_correlation,
        ),
        StreamProviderEvent(kind=StreamItemKind.REASONING_DONE),
        StreamProviderEvent(
            kind=terminal_kind,
            data=cast(Any, terminal_data),
            usage=cast(Any, terminal_usage),
        ),
    )
    return await _collect_items(
        normalize_provider_stream(
            events,
            stream_session_id=f"{provider_family}-stream",
            run_id="run-1",
            turn_id="turn-1",
            provider_family=provider_family,
            capabilities=StreamProviderCapabilities(
                backend=StreamProducerBackend.HOSTED,
                provider_family=provider_family,
                supports_reasoning=True,
                supports_tool_calls=True,
                supports_usage=True,
                supports_terminal_events=True,
                supports_cancellation=True,
            ),
        )
    )


def _provider_semantics(item: CanonicalStreamItem) -> tuple[object, ...]:
    return (
        item.kind,
        item.channel,
        item.text_delta,
        item.data,
        item.usage,
        item.terminal_outcome,
        item.correlation,
        item.visibility,
        item.metadata,
        item.provider_family,
        item.provider_event_type,
    )


def _delta_semantics(items: list[object]) -> list[tuple[object, ...]]:
    return [
        (
            item.kind,
            item.text_delta,
            item.reasoning_representation,
            item.segment_instance_ordinal,
            item.visibility,
        )
        for item in items
        if isinstance(item, (CanonicalStreamItem, StreamProviderEvent))
        and item.kind
        in {StreamItemKind.ANSWER_DELTA, StreamItemKind.REASONING_DELTA}
    ]


def test_representation_does_not_change_provider_semantics() -> None:
    native = run(
        _normalized_provider_items(
            "openai",
            representation=StreamReasoningRepresentation.NATIVE_TEXT,
        )
    )
    summary = run(
        _normalized_provider_items(
            "openai",
            representation=StreamReasoningRepresentation.SUMMARY,
        )
    )

    assert [_provider_semantics(item) for item in native] == [
        _provider_semantics(item) for item in summary
    ]
    native_delta = next(
        item for item in native if item.kind is StreamItemKind.REASONING_DELTA
    )
    summary_delta = next(
        item for item in summary if item.kind is StreamItemKind.REASONING_DELTA
    )
    assert (
        native_delta.reasoning_representation
        is StreamReasoningRepresentation.NATIVE_TEXT
    )
    assert (
        summary_delta.reasoning_representation
        is StreamReasoningRepresentation.SUMMARY
    )


def test_native_reasoning_is_not_summary_fallback() -> None:
    items = run(
        _normalized_provider_items(
            "openai",
            representation=StreamReasoningRepresentation.NATIVE_TEXT,
            provider_event_type="response.reasoning_summary_text.delta",
            provider_payload={"type": "summary-shaped-provider-payload"},
        )
    )

    reasoning = [
        item for item in items if item.kind is StreamItemKind.REASONING_DELTA
    ]
    assert len(reasoning) == 1
    assert (
        reasoning[0].reasoning_representation
        is StreamReasoningRepresentation.NATIVE_TEXT
    )
    assert (
        reasoning[0].provider_event_type
        == "response.reasoning_summary_text.delta"
    )
    assert all(
        item.reasoning_representation
        is not StreamReasoningRepresentation.SUMMARY
        for item in reasoning
    )


def test_every_native_delta_is_typed() -> None:
    parser = LocalTextStreamEventParser()
    events = list(parser.push("<think>first</think><think>second</think>"))
    events.extend(parser.flush())
    reasoning = [
        event
        for event in events
        if event.kind is StreamItemKind.REASONING_DELTA
    ]

    assert [event.text_delta for event in reasoning] == ["first", "second"]
    assert [event.segment_instance_ordinal for event in reasoning] == [0, 1]
    assert all(
        event.reasoning_representation
        is StreamReasoningRepresentation.NATIVE_TEXT
        for event in reasoning
    )
    assert all(
        event.visibility is StreamVisibility.PRIVATE for event in reasoning
    )
    assert all(event.text_delta for event in reasoning)


async def _canonical_adapter_items(stream: Any) -> list[CanonicalStreamItem]:
    return await _collect_items(
        stream.canonical_stream(
            stream_session_id="native-stream",
            run_id="run-1",
            turn_id="turn-1",
        )
    )


async def _openai_compatibility_traces() -> tuple[
    list[CanonicalStreamItem],
    list[CanonicalStreamItem],
    list[CanonicalStreamItem],
]:
    usage = {"input_tokens": 2, "output_tokens": 3}
    happy = await _canonical_adapter_items(
        OpenAIStream(
            _raw_events(
                SimpleNamespace(
                    type="response.reasoning_text.delta",
                    delta="think",
                    item_id="reasoning-1",
                    output_index=0,
                ),
                SimpleNamespace(type="response.reasoning_text.done"),
                SimpleNamespace(
                    type="response.reasoning_text.delta",
                    delta="again",
                    item_id="reasoning-1",
                    output_index=0,
                ),
                SimpleNamespace(
                    type="response.output_item.added",
                    item=SimpleNamespace(
                        id="call-1",
                        custom_tool_call=SimpleNamespace(
                            id="call-1", name="lookup"
                        ),
                    ),
                ),
                SimpleNamespace(
                    type="response.function_call_arguments.delta",
                    item_id="call-1",
                    delta='{"q":',
                ),
                SimpleNamespace(
                    type="response.function_call_arguments.delta",
                    item_id="call-1",
                    delta="1}",
                ),
                SimpleNamespace(
                    type="response.function_call_arguments.done",
                    item_id="call-1",
                ),
                SimpleNamespace(
                    type="response.output_item.done",
                    item=SimpleNamespace(id="call-1"),
                ),
                SimpleNamespace(
                    type="response.output_text.delta", delta="answer"
                ),
                SimpleNamespace(type="response.output_text.done"),
                SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(usage=usage),
                ),
            )
        )
    )
    errored = await _canonical_adapter_items(
        OpenAIStream(
            _raw_events(
                SimpleNamespace(
                    type="response.error",
                    error=SimpleNamespace(message="provider failure"),
                )
            )
        )
    )
    cancelled = await _canonical_adapter_items(
        OpenAIStream(cast(Any, _CancelledAsyncIterator()))
    )
    return happy, errored, cancelled


async def _litellm_compatibility_traces() -> tuple[
    list[CanonicalStreamItem],
    list[CanonicalStreamItem],
    list[CanonicalStreamItem],
]:
    usage = {"input_tokens": 2, "output_tokens": 3}
    happy = await _canonical_adapter_items(
        LiteLLMStream(
            cast(
                Any,
                _raw_events(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"reasoning_content": "think"},
                            }
                        ]
                    },
                    {"choices": [{"delta": {"content": "answer"}}]},
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"reasoning_content": "again"},
                            }
                        ]
                    },
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call-1",
                                            "function": {
                                                "name": "lookup",
                                                "arguments": '{"q":',
                                            },
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {"arguments": "1}"},
                                        }
                                    ]
                                },
                                "finish_reason": "tool_calls",
                            }
                        ]
                    },
                    {"usage": usage},
                ),
            )
        )
    )
    errored = await _canonical_adapter_items(
        LiteLLMStream(
            cast(
                Any,
                _raw_events({"error": {"message": "provider failure"}}),
            )
        )
    )
    cancelled = await _canonical_adapter_items(
        LiteLLMStream(cast(Any, _CancelledAsyncIterator()))
    )
    return happy, errored, cancelled


def _anthropic_sdk_modules() -> (
    tuple[dict[str, ModuleType], type[object], type[object]]
):
    class APIStatusError(Exception):
        pass

    class DeltaEvent:
        def __init__(self, delta: object, index: object = 0) -> None:
            self.type = "content_block_delta"
            self.delta = delta
            self.index = index

    class StopEvent:
        type = "message_stop"

    anthropic = ModuleType("anthropic")
    anthropic_types = ModuleType("anthropic.types")
    setattr(anthropic, "APIStatusError", APIStatusError)
    setattr(anthropic, "AsyncAnthropic", MagicMock())
    setattr(anthropic, "types", anthropic_types)
    setattr(anthropic_types, "RawContentBlockDeltaEvent", DeltaEvent)
    setattr(anthropic_types, "RawMessageStopEvent", StopEvent)
    return (
        {"anthropic": anthropic, "anthropic.types": anthropic_types},
        DeltaEvent,
        StopEvent,
    )


async def _anthropic_compatibility_traces() -> tuple[
    list[CanonicalStreamItem],
    list[CanonicalStreamItem],
    list[CanonicalStreamItem],
]:
    sdk_modules, delta_event, stop_event = _anthropic_sdk_modules()
    with _isolated_vendor_module(
        "avalan.model.nlp.text.vendor.anthropic", sdk_modules
    ) as module:
        stream_class = cast(Any, module).AnthropicStream
        happy = await _canonical_adapter_items(
            stream_class(
                _raw_events(
                    SimpleNamespace(
                        type="content_block_start",
                        content_block=SimpleNamespace(
                            type="tool_use", id="call-1", name="lookup"
                        ),
                        index=1,
                    ),
                    delta_event(SimpleNamespace(thinking="think"), 0),
                    delta_event(SimpleNamespace(partial_json='{"q":'), 1),
                    delta_event(SimpleNamespace(text="answer"), 2),
                    delta_event(SimpleNamespace(thinking="again"), 0),
                    delta_event(SimpleNamespace(partial_json="1}"), 1),
                    SimpleNamespace(type="content_block_stop", index=1),
                    SimpleNamespace(
                        type="message_delta",
                        usage=cast(
                            Any,
                            {
                                "input_tokens": 2,
                                "output_tokens": 3,
                            },
                        ),
                    ),
                    stop_event(),
                )
            )
        )
        errored = await _canonical_adapter_items(
            stream_class(
                _raw_events(delta_event(SimpleNamespace(thinking="bad"), True))
            )
        )
        cancelled = await _canonical_adapter_items(
            stream_class(_CancelledAsyncIterator())
        )
    return happy, errored, cancelled


def _bedrock_sdk_modules() -> dict[str, ModuleType]:
    aioboto3 = ModuleType("aioboto3")
    setattr(aioboto3, "Session", MagicMock())
    return {"aioboto3": aioboto3}


async def _bedrock_compatibility_traces() -> tuple[
    list[CanonicalStreamItem],
    list[CanonicalStreamItem],
    list[CanonicalStreamItem],
]:
    with _isolated_vendor_module(
        "avalan.model.nlp.text.vendor.bedrock", _bedrock_sdk_modules()
    ) as module:
        stream_class = cast(Any, module).BedrockStream
        happy = await _canonical_adapter_items(
            stream_class(
                _raw_events(
                    {
                        "contentBlockDelta": {
                            "contentBlockIndex": 0,
                            "delta": {"reasoning": {"text": "think"}},
                        }
                    },
                    {
                        "contentBlockStart": {
                            "contentBlockIndex": 1,
                            "contentBlock": {
                                "toolUse": {
                                    "toolUseId": "call-1",
                                    "name": "lookup",
                                    "input": '{"q":',
                                }
                            },
                        }
                    },
                    {
                        "contentBlockDelta": {
                            "contentBlockIndex": 0,
                            "delta": {"text": {"text": "answer"}},
                        }
                    },
                    {
                        "contentBlockDelta": {
                            "contentBlockIndex": 0,
                            "delta": {"reasoning": {"text": "again"}},
                        }
                    },
                    {
                        "contentBlockStop": {
                            "contentBlockIndex": 1,
                            "contentBlock": {
                                "toolUse": {
                                    "toolUseId": "call-1",
                                    "name": "lookup",
                                    "input": "1}",
                                }
                            },
                        }
                    },
                    {"messageStop": {"reason": "finished"}},
                    {
                        "metadata": {
                            "usage": {
                                "input_tokens": 2,
                                "output_tokens": 3,
                            }
                        }
                    },
                )
            )
        )
        errored = await _canonical_adapter_items(
            stream_class(
                _raw_events(
                    {
                        "contentBlockDelta": {
                            "contentBlockIndex": "bad",
                            "delta": {"text": {"text": "answer"}},
                        }
                    }
                )
            )
        )
        cancelled = await _canonical_adapter_items(
            stream_class(_CancelledAsyncIterator())
        )
    return happy, errored, cancelled


async def _native_adapter_compatibility_traces() -> dict[
    str,
    tuple[
        list[CanonicalStreamItem],
        list[CanonicalStreamItem],
        list[CanonicalStreamItem],
    ],
]:
    return {
        "anthropic": await _anthropic_compatibility_traces(),
        "bedrock": await _bedrock_compatibility_traces(),
        "openai": await _openai_compatibility_traces(),
        "openai_compatible": await _litellm_compatibility_traces(),
    }


def _assert_happy_native_adapter_trace(
    provider_family: str,
    items: list[CanonicalStreamItem],
) -> None:
    validate_canonical_stream_items(items)
    assert [item.sequence for item in items] == list(range(len(items)))
    assert {item.provider_family for item in items} == {provider_family}
    assert items[0].kind is StreamItemKind.STREAM_STARTED
    assert items[-2].kind is StreamItemKind.STREAM_COMPLETED
    assert items[-2].terminal_outcome is StreamTerminalOutcome.COMPLETED
    assert items[-1].kind is StreamItemKind.STREAM_CLOSED
    for kind in (
        StreamItemKind.ANSWER_DELTA,
        StreamItemKind.ANSWER_DONE,
        StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
        StreamItemKind.TOOL_CALL_READY,
        StreamItemKind.TOOL_CALL_DONE,
        StreamItemKind.USAGE_COMPLETED,
    ):
        assert kind in [item.kind for item in items]
    assert (
        sum(item.kind is StreamItemKind.REASONING_DONE for item in items) == 1
    )

    accumulator = accumulate_canonical_stream_items(items)
    assert accumulator.answer_text == "answer"
    assert [
        item.text_delta
        for item in items
        if item.kind is StreamItemKind.REASONING_DELTA
    ] == ["think", "again"]
    assert accumulator.reasoning_text == "think\n\nagain"
    assert accumulator.tool_call_arguments == {"call-1": '{"q":1}'}
    assert accumulator.final_usage == {
        "input_tokens": 2,
        "output_tokens": 3,
    }
    assert accumulator.terminal_outcome is StreamTerminalOutcome.COMPLETED
    ready = next(
        item for item in items if item.kind is StreamItemKind.TOOL_CALL_READY
    )
    assert ready.data == {"name": "lookup"}
    reasoning = [
        item for item in items if item.kind is StreamItemKind.REASONING_DELTA
    ]
    assert [item.segment_instance_ordinal for item in reasoning] == [0, 1]
    assert all(item.text_delta for item in reasoning)
    assert all(
        item.reasoning_representation
        is StreamReasoningRepresentation.NATIVE_TEXT
        for item in reasoning
    )
    assert all(
        item.visibility is StreamVisibility.PRIVATE for item in reasoning
    )


def _assert_terminal_adapter_trace(
    items: list[CanonicalStreamItem],
    terminal_kind: StreamItemKind,
    terminal_outcome: StreamTerminalOutcome,
) -> None:
    assert [item.kind for item in items] == [
        StreamItemKind.STREAM_STARTED,
        terminal_kind,
        StreamItemKind.STREAM_CLOSED,
    ]
    assert items[1].terminal_outcome is terminal_outcome
    validate_canonical_stream_items(items)


def test_native_providers_remain_compatible() -> None:
    traces = run(_native_adapter_compatibility_traces())
    assert set(traces) == {
        "anthropic",
        "bedrock",
        "openai",
        "openai_compatible",
    }
    for provider_family, (happy, errored, cancelled) in traces.items():
        _assert_happy_native_adapter_trace(provider_family, happy)
        _assert_terminal_adapter_trace(
            errored,
            StreamItemKind.STREAM_ERRORED,
            StreamTerminalOutcome.ERRORED,
        )
        _assert_terminal_adapter_trace(
            cancelled,
            StreamItemKind.STREAM_CANCELLED,
            StreamTerminalOutcome.CANCELLED,
        )

    invalid_correlation_cases = (
        (LiteLLMStream._reasoning_correlation, {"index": -1}),
        (OpenAIStream._reasoning_correlation, {"item_id": ""}),
        (OpenAIStream._reasoning_correlation, {"output_index": True}),
    )
    for correlate, event in invalid_correlation_cases:
        try:
            correlate(event)
        except ValueError:
            continue
        raise AssertionError("invalid native reasoning correlation accepted")

    assert OpenAIStream._reasoning_correlation(
        {
            "item_id": "reasoning-1",
            "output_index": 2,
            "summary_index": 3,
        }
    ) == StreamItemCorrelation(
        protocol_item_id="reasoning-1",
        provider_output_index=2,
        provider_summary_index=3,
    )
    openai_stream = OpenAIStream(_provider_events())
    assert (
        openai_stream._provider_events_from_event(
            {
                "type": "response.reasoning_text.delta",
                "delta": "",
            }
        )
        == ()
    )
    run(openai_stream.aclose())


async def _transformers_events_for_chunks(
    chunks: tuple[str, ...],
) -> list[StreamProviderEvent]:
    model = TextGenerationModel(
        "model",
        TransformerEngineSettings(
            auto_load_model=False,
            auto_load_tokenizer=False,
        ),
    )
    model._log = MagicMock()
    model._tokenizer = MagicMock()
    decoded = {index: chunk for index, chunk in enumerate(chunks, start=1)}
    model._tokenizer.decode.side_effect = lambda token_id, **_: decoded[
        int(token_id)
    ]
    output = SimpleNamespace(
        sequences=torch.tensor([[99, *range(1, len(chunks) + 1)]]),
        scores=[torch.zeros((1, len(chunks) + 1)) for _ in chunks],
    )
    with (
        patch.object(model, "_generate_output", return_value=output),
        patch(
            "avalan.model.nlp.text.generation.softmax",
            return_value=torch.tensor(
                [index / 10 for index in range(len(chunks) + 1)]
            ),
        ),
    ):
        return [
            event
            async for event in model._token_provider_events(
                {"input_ids": torch.tensor([[99]])},
                GenerationSettings(
                    max_new_tokens=len(chunks),
                    temperature=1.0,
                ),
                None,
                False,
                pick=0,
                probability_distribution="softmax",
            )
        ]


@contextmanager
def _transformers_failure_harness(
    chunks: tuple[str, ...],
    failure_type: type[BaseException],
) -> Iterator[tuple[TextGenerationModel, _FailingIterator]]:
    model = TextGenerationModel(
        "model",
        TransformerEngineSettings(
            auto_load_model=False,
            auto_load_tokenizer=False,
        ),
    )
    model._log = MagicMock()
    model._tokenizer = MagicMock()
    decoded = {index: chunk for index, chunk in enumerate(chunks, start=1)}
    model._tokenizer.decode.side_effect = lambda token_id, **_: decoded[
        int(token_id)
    ]
    generated = _FailingIterator(
        tuple(range(1, len(chunks) + 1)),
        failure_type,
    )
    output = SimpleNamespace(
        sequences=[_GeneratedSequenceRow(generated)],
        scores=[torch.zeros((1, len(chunks) + 1)) for _ in chunks],
    )
    with (
        patch.object(model, "_generate_output", return_value=output),
        patch(
            "avalan.model.nlp.text.generation.softmax",
            return_value=torch.tensor(
                [index / 10 for index in range(len(chunks) + 1)]
            ),
        ),
    ):
        yield model, generated


async def _transformers_abnormal_items(
    chunks: tuple[str, ...],
    failure_type: type[BaseException],
) -> tuple[list[CanonicalStreamItem], int]:
    with _transformers_failure_harness(chunks, failure_type) as (
        model,
        generated,
    ):
        stream = model._token_generator(
            {"input_ids": torch.tensor([[99]])},
            GenerationSettings(
                max_new_tokens=len(chunks),
                temperature=1.0,
            ),
            None,
            False,
            pick=0,
            probability_distribution="softmax",
        )
        items = await _collect_items(stream)
        pulls = generated.pulls
        await stream.aclose()
        await stream.aclose()
        assert generated.pulls == pulls
        return items, pulls


async def _transformers_stream_abnormal_items(
    chunks: tuple[str, ...],
    failure_type: type[BaseException],
) -> tuple[list[CanonicalStreamItem], int]:
    model = TextGenerationModel(
        "model",
        TransformerEngineSettings(
            auto_load_model=False,
            auto_load_tokenizer=False,
        ),
    )
    model._log = MagicMock()
    model._tokenizer = MagicMock()
    streamer = _FailingAsyncTextStreamer(chunks, failure_type)
    with (
        patch(
            "avalan.model.nlp.text.generation.AsyncTextIteratorStreamer",
            return_value=streamer,
        ),
        patch(
            "avalan.model.nlp.text.generation.Thread",
            side_effect=_IdleThread,
        ),
    ):
        stream = model._stream_generator(
            {"input_ids": torch.tensor([[99]])},
            GenerationSettings(max_new_tokens=len(chunks)),
            None,
            False,
        )
        items = await _collect_items(stream)
        pulls = streamer.pulls
        await stream.aclose()
        await stream.aclose()
        assert streamer.pulls == pulls
        return items, pulls


async def _mlx_abnormal_items(
    chunks: tuple[str, ...],
    failure_type: type[BaseException],
) -> tuple[list[CanonicalStreamItem], int]:
    generated = _FailingIterator(
        tuple(
            SimpleNamespace(
                token=chunk,
                id=index,
                probability=index / 10,
                probability_distribution=f"distribution-{index}",
                step=index - 1,
            )
            for index, chunk in enumerate(chunks, start=1)
        ),
        failure_type,
    )
    mlx_stream = MlxLmStream(generated, use_executor=False)
    stream = mlx_stream.canonical_stream(
        stream_session_id="mlx-abnormal-stream",
        run_id="run-1",
        turn_id="turn-1",
    )
    items = await _collect_items(stream)
    pulls = generated.pulls
    await cast(Any, stream).aclose()
    await cast(Any, stream).aclose()
    await mlx_stream.aclose()
    await mlx_stream.aclose()
    assert generated.pulls == pulls
    assert mlx_stream._closed
    return items, pulls


async def _transformers_split_tag_events() -> list[StreamProviderEvent]:
    return await _transformers_events_for_chunks(_LOCAL_SPLIT_TAG_CHUNKS)


class _Ds4ParserHarness:
    async def _generate_text_chunks(
        self,
        _session: object,
        _generation_plan: object,
        _usage: object,
    ) -> AsyncIterator[_Ds4GeneratedChunk]:
        for index, chunk in enumerate(_LOCAL_SPLIT_TAG_CHUNKS):
            yield _Ds4GeneratedChunk(chunk, {"step": index})

    _event_with_metadata = staticmethod(Ds4Worker._event_with_metadata)


async def _ds4_split_tag_events() -> list[StreamProviderEvent]:
    generate = cast(Any, Ds4Worker._generate_text_events)
    return [
        event
        async for event in generate(
            _Ds4ParserHarness(),
            object(),
            object(),
            object(),
        )
    ]


def _mlx_stream_for_chunks(
    chunks: tuple[str, ...],
) -> AsyncIterable[CanonicalStreamItem]:
    return MlxLmStream(
        iter(
            SimpleNamespace(
                token=chunk,
                id=index,
                probability=index / 10,
                probability_distribution=f"distribution-{index}",
                step=index - 1,
            )
            for index, chunk in enumerate(chunks, start=1)
        ),
        use_executor=False,
    ).canonical_stream(
        stream_session_id="mlx-stream",
        run_id="run-1",
        turn_id="turn-1",
    )


async def _local_backend_split_tag_outputs() -> dict[str, list[object]]:
    transformers = await _transformers_split_tag_events()
    ds4 = await _ds4_split_tag_events()
    vllm = await _collect_items(
        VllmStream(iter(_LOCAL_SPLIT_TAG_CHUNKS)).canonical_stream(
            stream_session_id="vllm-stream",
            run_id="run-1",
            turn_id="turn-1",
        )
    )
    mlx = await _collect_items(_mlx_stream_for_chunks(_LOCAL_SPLIT_TAG_CHUNKS))
    return {
        "transformers": cast(list[object], transformers),
        "ds4": cast(list[object], ds4),
        "vllm": cast(list[object], vllm),
        "mlx": cast(list[object], mlx),
    }


async def _fragment_provenance_outputs() -> (
    dict[tuple[str, str], list[object]]
):
    outputs: dict[tuple[str, str], list[object]] = {}
    for case_name, (chunks, _) in _FRAGMENT_PROVENANCE_CASES.items():
        outputs[("transformers", case_name)] = cast(
            list[object], await _transformers_events_for_chunks(chunks)
        )
        outputs[("mlx", case_name)] = cast(
            list[object], await _collect_items(_mlx_stream_for_chunks(chunks))
        )
    return outputs


async def _abnormal_fragment_outputs() -> dict[
    tuple[str, str, str],
    tuple[list[CanonicalStreamItem], int],
]:
    outputs: dict[
        tuple[str, str, str],
        tuple[list[CanonicalStreamItem], int],
    ] = {}
    failures = (
        ("error", RuntimeError),
        ("cancelled", CancelledError),
    )
    for case_name, (chunks, _) in _ABNORMAL_FRAGMENT_CASES.items():
        for failure_name, failure_type in failures:
            outputs[("transformers", case_name, failure_name)] = (
                await _transformers_abnormal_items(chunks, failure_type)
            )
            outputs[("transformers_stream", case_name, failure_name)] = (
                await _transformers_stream_abnormal_items(
                    chunks,
                    failure_type,
                )
            )
            outputs[("mlx", case_name, failure_name)] = (
                await _mlx_abnormal_items(chunks, failure_type)
            )
    return outputs


async def _assert_buffered_consumer_close_is_silent() -> None:
    chunks = ("lead<", "must-not-be-pulled")
    with _transformers_failure_harness(chunks, RuntimeError) as (
        model,
        generated,
    ):
        events = model._token_provider_events(
            {"input_ids": torch.tensor([[99]])},
            GenerationSettings(
                max_new_tokens=len(chunks),
                temperature=1.0,
            ),
            None,
            False,
            pick=0,
            probability_distribution="softmax",
        )
        first = await events.__anext__()
        assert first.kind is StreamItemKind.ANSWER_DELTA
        assert first.text_delta == "lead"
        assert generated.pulls == 1
        await events.aclose()
        await events.aclose()
        assert generated.pulls == 1

    model = TextGenerationModel(
        "model",
        TransformerEngineSettings(
            auto_load_model=False,
            auto_load_tokenizer=False,
        ),
    )
    model._log = MagicMock()
    model._tokenizer = MagicMock()
    streamer = _FailingAsyncTextStreamer(chunks, RuntimeError)
    with (
        patch(
            "avalan.model.nlp.text.generation.AsyncTextIteratorStreamer",
            return_value=streamer,
        ),
        patch(
            "avalan.model.nlp.text.generation.Thread",
            side_effect=_IdleThread,
        ),
    ):
        stream = model._stream_generator(
            {"input_ids": torch.tensor([[99]])},
            GenerationSettings(max_new_tokens=len(chunks)),
            None,
            False,
        )
        started = await stream.__anext__()
        first = await stream.__anext__()
        assert started.kind is StreamItemKind.STREAM_STARTED
        assert first.kind is StreamItemKind.ANSWER_DELTA
        assert first.text_delta == "lead"
        assert streamer.pulls == 1
        await stream.aclose()
        await stream.aclose()
        assert streamer.pulls == 1

    mlx_generated = _FailingIterator(
        tuple(
            SimpleNamespace(
                token=chunk,
                id=index,
                probability=index / 10,
                step=index - 1,
            )
            for index, chunk in enumerate(chunks, start=1)
        ),
        RuntimeError,
    )
    mlx_stream = MlxLmStream(mlx_generated, use_executor=False)
    mlx_events = mlx_stream._provider_events()
    first = await mlx_events.__anext__()
    assert first.kind is StreamItemKind.ANSWER_DELTA
    assert first.text_delta == "lead"
    assert mlx_generated.pulls == 1
    await mlx_events.aclose()
    await mlx_events.aclose()
    await mlx_stream.aclose()
    await mlx_stream.aclose()
    assert mlx_generated.pulls == 1
    assert mlx_stream._closed


def test_shared_local_reasoning_parser_backend_parity() -> None:
    outputs = run(_local_backend_split_tag_outputs())
    expected = [
        (
            StreamItemKind.ANSWER_DELTA,
            "lead",
            None,
            None,
            StreamVisibility.PUBLIC,
        ),
        (
            StreamItemKind.REASONING_DELTA,
            "hidden",
            StreamReasoningRepresentation.NATIVE_TEXT,
            0,
            StreamVisibility.PRIVATE,
        ),
        (
            StreamItemKind.ANSWER_DELTA,
            "tail",
            None,
            None,
            StreamVisibility.PUBLIC,
        ),
        (
            StreamItemKind.ANSWER_DELTA,
            "<",
            None,
            None,
            StreamVisibility.PUBLIC,
        ),
    ]

    assert set(outputs) == {"transformers", "ds4", "vllm", "mlx"}
    for backend, items in outputs.items():
        assert _delta_semantics(items) == expected, backend

    for backend, provider_event_type in (
        ("transformers", "transformers.token"),
        ("mlx", "mlx_lm.delta"),
    ):
        deltas = [
            item
            for item in outputs[backend]
            if isinstance(item, (CanonicalStreamItem, StreamProviderEvent))
            and item.kind
            in {StreamItemKind.ANSWER_DELTA, StreamItemKind.REASONING_DELTA}
        ]
        assert [item.metadata["token_id"] for item in deltas] == [1, 3, 5, 6]
        assert {item.provider_event_type for item in deltas} == {
            provider_event_type
        }

    provenance_outputs = run(_fragment_provenance_outputs())
    assert set(provenance_outputs) == {
        (backend, case_name)
        for backend in ("transformers", "mlx")
        for case_name in _FRAGMENT_PROVENANCE_CASES
    }
    for (backend, case_name), items in provenance_outputs.items():
        deltas = [
            item
            for item in items
            if isinstance(item, (CanonicalStreamItem, StreamProviderEvent))
            and item.kind
            in {StreamItemKind.ANSWER_DELTA, StreamItemKind.REASONING_DELTA}
        ]
        expected = _FRAGMENT_PROVENANCE_CASES[case_name][1]
        assert [item.text_delta for item in deltas] == [
            text for text, _ in expected
        ]
        source_indexes = [source_index for _, source_index in expected]
        assert [item.metadata["token_id"] for item in deltas] == source_indexes
        assert [item.metadata["step"] for item in deltas] == [
            source_index - 1 for source_index in source_indexes
        ]
        assert [
            round(cast(float, item.metadata["probability"]), 6)
            for item in deltas
        ] == [source_index / 10 for source_index in source_indexes]
        if backend == "transformers":
            assert {item.provider_event_type for item in deltas} == {
                "transformers.token"
            }
            assert {
                item.metadata["probability_distribution"] for item in deltas
            } == {"softmax"}
        else:
            assert {item.provider_event_type for item in deltas} == {
                "mlx_lm.delta"
            }
            assert [
                item.metadata["probability_distribution"] for item in deltas
            ] == [
                f"distribution-{source_index}"
                for source_index in source_indexes
            ]

    abnormal_outputs = run(_abnormal_fragment_outputs())
    assert set(abnormal_outputs) == {
        (backend, case_name, failure_name)
        for backend in ("transformers", "transformers_stream", "mlx")
        for case_name in _ABNORMAL_FRAGMENT_CASES
        for failure_name in ("error", "cancelled")
    }
    for (
        backend,
        case_name,
        failure_name,
    ), (items, pulls) in abnormal_outputs.items():
        chunks, expected_deltas = _ABNORMAL_FRAGMENT_CASES[case_name]
        deltas = [
            item
            for item in items
            if item.kind
            in {
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.REASONING_DELTA,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
            }
        ]
        expected_abnormal_deltas = list(expected_deltas)
        if backend == "transformers_stream":
            buffered_kind = expected_deltas[1][0]
            assert all(
                kind is buffered_kind for kind, _ in expected_deltas[1:]
            )
            expected_abnormal_deltas = [
                expected_deltas[0],
                (
                    buffered_kind,
                    "".join(text for _, text in expected_deltas[1:]),
                ),
            ]
        assert [(item.kind, item.text_delta) for item in deltas] == list(
            expected_abnormal_deltas
        )
        assert pulls == len(chunks) + 1
        if backend == "transformers_stream":
            assert [item.metadata for item in deltas] == [{}, {}]
            assert {item.provider_event_type for item in deltas} == {None}
        else:
            expected_provider_event_type = (
                "transformers.token"
                if backend == "transformers"
                else "mlx_lm.delta"
            )
            assert {item.provider_event_type for item in deltas} == {
                expected_provider_event_type
            }
            if backend == "transformers":
                expected_metadata = [
                    {
                        "token_id": index,
                        "probability": float(torch.tensor(index / 10).item()),
                        "step": index - 1,
                        "probability_distribution": "softmax",
                    }
                    for index in range(1, 4)
                ]
            else:
                expected_metadata = [
                    {
                        "token_id": index,
                        "probability": index / 10,
                        "step": index - 1,
                        "probability_distribution": f"distribution-{index}",
                    }
                    for index in range(1, 4)
                ]
            assert [item.metadata for item in deltas] == expected_metadata

        if case_name == "reasoning_end_marker":
            assert all(
                item.reasoning_representation
                is StreamReasoningRepresentation.NATIVE_TEXT
                for item in deltas
            )
            assert [item.segment_instance_ordinal for item in deltas] == [
                0
            ] * len(deltas)
            assert all(
                item.visibility is StreamVisibility.PRIVATE for item in deltas
            )
        else:
            assert all(
                item.reasoning_representation is None for item in deltas
            )
            assert all(
                item.segment_instance_ordinal is None for item in deltas
            )
            assert all(
                item.visibility is StreamVisibility.PUBLIC for item in deltas
            )

        if case_name == "tool_end_marker":
            assert {item.correlation.tool_call_id for item in deltas} == {
                "local-tool-call-1"
            }
            diagnostics = [
                item
                for item in items
                if item.kind is StreamItemKind.STREAM_DIAGNOSTIC
            ]
            assert len(diagnostics) == 1
            assert diagnostics[0].data == {
                "code": "tool_call.malformed",
                "message": "unterminated tool call",
                "tool_call_id": "local-tool-call-1",
            }
        else:
            assert all(
                item.kind is not StreamItemKind.STREAM_DIAGNOSTIC
                for item in items
            )

        terminals = [
            item for item in items if item.kind in _CANONICAL_TERMINAL_KINDS
        ]
        expected_terminal_kind = (
            StreamItemKind.STREAM_CANCELLED
            if failure_name == "cancelled"
            else StreamItemKind.STREAM_ERRORED
        )
        expected_terminal_outcome = (
            StreamTerminalOutcome.CANCELLED
            if failure_name == "cancelled"
            else StreamTerminalOutcome.ERRORED
        )
        assert [(item.kind, item.terminal_outcome) for item in terminals] == [
            (expected_terminal_kind, expected_terminal_outcome)
        ]
        if failure_name == "error":
            assert terminals[0].data == {
                "error_type": "RuntimeError",
                "message": "upstream stream failure",
            }
        else:
            assert terminals[0].data is None
        terminal_index = items.index(terminals[0])
        delta_indexes = [
            index for index, item in enumerate(items) if item in deltas
        ]
        assert delta_indexes == sorted(delta_indexes)
        assert delta_indexes[-1] < terminal_index
        assert items[-1].kind is StreamItemKind.STREAM_CLOSED
        validate_canonical_stream_items(items)

    run(_assert_buffered_consumer_close_is_silent())


async def _mlx_untagged_cancel_and_close_cases() -> None:
    stream = MlxLmStream(
        iter(
            [
                SimpleNamespace(
                    token="plain output",
                    id=7,
                    probability=0.25,
                    step=3,
                )
            ]
        ),
        use_executor=False,
    )
    items = await _collect_items(stream)
    assert [item.kind for item in items] == [
        StreamItemKind.STREAM_STARTED,
        StreamItemKind.ANSWER_DELTA,
        StreamItemKind.ANSWER_DONE,
        StreamItemKind.STREAM_COMPLETED,
        StreamItemKind.STREAM_CLOSED,
    ]
    assert items[1].text_delta == "plain output"
    assert items[1].metadata == {
        "token_id": 7,
        "probability": 0.25,
        "step": 3,
    }
    capabilities = cast(dict[str, object], items[0].metadata["capabilities"])
    assert capabilities["supports_reasoning"] is True
    assert capabilities["supports_reasoning_summary"] is False

    pulls = 0

    def late_factory() -> AsyncIterator[str] | Any:
        nonlocal pulls
        pulls += 1
        return iter(["late"])

    cancelled = MlxLmStream(late_factory, use_executor=False)
    await cancelled.cancel()
    assert cancelled._closed
    assert pulls == 0

    closed = MlxLmStream(late_factory, use_executor=False)
    await closed.aclose()
    assert closed._closed
    assert pulls == 0


def test_mlx_reasoning_parser_preserves_untagged_cancel_and_close() -> None:
    run(_mlx_untagged_cancel_and_close_cases())


def test_public_stream_capabilities_keep_summary_support_disabled() -> None:
    default = StreamProviderCapabilities(backend=StreamProducerBackend.LOCAL)
    assert default.supports_reasoning_summary is False
    assert default.to_metadata()["supports_reasoning_summary"] is False

    source_root = Path(__file__).parents[2] / "src" / "avalan"
    capability_calls: list[tuple[Path, ast.Call]] = []
    for source_path in source_root.rglob("*.py"):
        tree = ast.parse(source_path.read_text(), filename=str(source_path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "StreamProviderCapabilities"
            ):
                capability_calls.append((source_path, node))

    assert len(capability_calls) == 12
    local_native_paths = {
        source_root / "model" / "nlp" / "text" / "mlxlm.py",
        source_root / "model" / "nlp" / "text" / "vllm.py",
    }
    for source_path, call in capability_calls:
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        summary_support = keywords.get("supports_reasoning_summary")
        assert summary_support is None or (
            isinstance(summary_support, ast.Constant)
            and summary_support.value is False
        )
        if source_path in local_native_paths:
            native_support = keywords.get("supports_reasoning")
            assert isinstance(native_support, ast.Constant)
            assert native_support.value is True
