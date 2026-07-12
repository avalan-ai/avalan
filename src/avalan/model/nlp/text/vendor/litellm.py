from .....entities import GenerationSettings, Message
from .....model.provider import ProviderFamily
from .....model.stream import (
    CanonicalStreamItem,
    StreamItemCorrelation,
    StreamItemKind,
    StreamProducerBackend,
    StreamProviderCapabilities,
    StreamProviderEvent,
    StreamReasoningRepresentation,
    StreamReasoningSegmentState,
    StreamVisibility,
    TextGenerationSingleStream,
    TextGenerationStream,
)
from .....tool.manager import ToolManager
from .....types import LooseJsonValue
from ....vendor import TextGenerationVendor, TextGenerationVendorStream
from . import (
    DiffusionPipeline,
    PreTrainedModel,
    TextGenerationVendorModel,
)

from collections.abc import Mapping
from typing import Any, AsyncIterator, cast

import litellm


class LiteLLMStream(TextGenerationVendorStream):
    _stream: AsyncIterator[Any]
    _tool_call_ids_by_index: dict[int, str]
    _tool_call_names_by_id: dict[str, str]
    _reasoning_segments: StreamReasoningSegmentState

    def __init__(self, stream: AsyncIterator[Any]) -> None:
        self._stream = stream
        self._tool_call_ids_by_index = {}
        self._tool_call_names_by_id = {}
        self._reasoning_segments = StreamReasoningSegmentState()

        async def generator() -> AsyncIterator[CanonicalStreamItem]:
            async for item in self.canonical_stream(
                stream_session_id=self._DEFAULT_STREAM_SESSION_ID,
                run_id=self._DEFAULT_RUN_ID,
                turn_id=self._DEFAULT_TURN_ID,
            ):
                yield item

        super().__init__(
            generator(),
            provider_family=ProviderFamily.OPENAI_COMPATIBLE,
            sources=(stream,),
        )

    def __aiter__(self) -> AsyncIterator[CanonicalStreamItem]:
        assert self._generator
        return self._generator

    async def __anext__(self) -> CanonicalStreamItem:
        return await super().__anext__()

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
        self._tool_call_ids_by_index = {}
        self._tool_call_names_by_id = {}
        self._reasoning_segments = StreamReasoningSegmentState()
        return self._provider_canonical_stream(
            self._provider_events(),
            stream_session_id=stream_session_id,
            run_id=run_id,
            turn_id=turn_id,
            provider_family=provider_family,
            capabilities=capabilities
            or StreamProviderCapabilities(
                backend=StreamProducerBackend.HOSTED,
                provider_family=ProviderFamily.OPENAI_COMPATIBLE,
                supports_reasoning=True,
                supports_tool_calls=True,
                supports_usage=True,
                supports_terminal_events=False,
                supports_cancellation=True,
            ),
            close_after_terminal=close_after_terminal,
        )

    async def _provider_events(self) -> AsyncIterator[StreamProviderEvent]:
        try:
            async for chunk in self._stream:
                async for event in self._provider_events_from_chunk(chunk):
                    yield event
                    if event.kind is not StreamItemKind.REASONING_DELTA:
                        self._reasoning_segments.complete_segment()
        finally:
            await self.aclose()

    async def _provider_events_from_chunk(
        self, chunk: object
    ) -> AsyncIterator[StreamProviderEvent]:
        provider_payload = self._provider_payload(chunk)
        error = LiteLLMClient._field(chunk, "error")
        if error is not None:
            yield StreamProviderEvent(
                kind=StreamItemKind.STREAM_ERRORED,
                data={"error": error},
                provider_payload=provider_payload,
                provider_event_type="chat.completion.error",
            )
            return

        choices = LiteLLMClient._field(chunk, "choices")
        if choices is None:
            for event in self._usage_events(chunk, provider_payload):
                yield event
            return
        if not isinstance(choices, list):
            raise ValueError("chat chunk choices must be a list")
        if not choices:
            for event in self._usage_events(chunk, provider_payload):
                yield event
            return

        choice = choices[0]
        delta = LiteLLMClient._field(choice, "delta")
        if delta is not None:
            reasoning = LiteLLMClient._reasoning_text(delta)
            if reasoning:
                representation = StreamReasoningRepresentation.NATIVE_TEXT
                correlation = self._reasoning_correlation(choice)
                yield StreamProviderEvent(
                    kind=StreamItemKind.REASONING_DELTA,
                    text_delta=reasoning,
                    correlation=correlation,
                    visibility=StreamVisibility.PRIVATE,
                    reasoning_representation=representation,
                    segment_instance_ordinal=(
                        self._reasoning_segments.allocate(
                            representation, correlation
                        )
                    ),
                    provider_payload=provider_payload,
                    provider_event_type="chat.completion.reasoning.delta",
                )

            content = LiteLLMClient._field(delta, "content")
            if content is not None:
                if not isinstance(content, str):
                    raise ValueError(
                        "chat chunk delta content must be a string"
                    )
                yield StreamProviderEvent(
                    kind=StreamItemKind.ANSWER_DELTA,
                    text_delta=content,
                    provider_payload=provider_payload,
                    provider_event_type="chat.completion.delta",
                )

            tool_call_ids = []
            for call_id, name, arguments in self._tool_call_deltas(delta):
                tool_call_ids.append(call_id)
                if name is not None:
                    self._tool_call_names_by_id[call_id] = name
                if arguments is not None:
                    yield StreamProviderEvent(
                        kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                        correlation=StreamItemCorrelation(
                            tool_call_id=call_id
                        ),
                        text_delta=arguments,
                        provider_payload=provider_payload,
                        provider_event_type="chat.completion.tool_call.delta",
                    )
        else:
            tool_call_ids = []

        if LiteLLMClient._field(choice, "finish_reason") == "tool_calls":
            for call_id in tool_call_ids or list(self._tool_call_names_by_id):
                correlation = StreamItemCorrelation(tool_call_id=call_id)
                yield StreamProviderEvent(
                    kind=StreamItemKind.TOOL_CALL_READY,
                    correlation=correlation,
                    data={"name": self._tool_call_names_by_id.get(call_id)},
                    provider_payload=provider_payload,
                    provider_event_type="chat.completion.tool_call.done",
                )
                yield StreamProviderEvent(
                    kind=StreamItemKind.TOOL_CALL_DONE,
                    correlation=correlation,
                    provider_payload=provider_payload,
                    provider_event_type="chat.completion.tool_call.done",
                )

        for event in self._usage_events(chunk, provider_payload):
            yield event

    @staticmethod
    def _reasoning_correlation(choice: object) -> StreamItemCorrelation:
        index = LiteLLMClient._field(choice, "index")
        if index is None:
            return StreamItemCorrelation()
        if type(index) is not int or index < 0:
            raise ValueError(
                "chat reasoning choice index must be a non-negative integer"
            )
        return StreamItemCorrelation(provider_output_index=index)

    def _usage_events(
        self,
        chunk: object,
        provider_payload: LooseJsonValue | None,
    ) -> tuple[StreamProviderEvent, ...]:
        usage = LiteLLMClient._field(chunk, "usage")
        if usage is None:
            return ()
        self._usage = usage
        return (
            StreamProviderEvent(
                kind=StreamItemKind.USAGE_COMPLETED,
                usage=cast(LooseJsonValue, usage),
                provider_payload=provider_payload,
                provider_event_type="chat.completion.usage",
            ),
        )

    def _tool_call_deltas(
        self, delta: object
    ) -> tuple[tuple[str, str | None, str | None], ...]:
        tool_calls = LiteLLMClient._field(delta, "tool_calls")
        if tool_calls is None:
            return ()
        if not isinstance(tool_calls, list):
            raise ValueError("chat chunk tool_calls must be a list")

        result: list[tuple[str, str | None, str | None]] = []
        for tool_call in tool_calls:
            index = LiteLLMClient._field(tool_call, "index")
            if index is not None and not isinstance(index, int):
                raise ValueError("chat chunk tool call index must be an int")
            call_id = LiteLLMClient._field(tool_call, "id")
            if call_id is not None:
                if not isinstance(call_id, str) or not call_id.strip():
                    raise ValueError("chat chunk tool call id is invalid")
                if index is not None:
                    self._tool_call_ids_by_index[index] = call_id
            elif index is not None:
                call_id = self._tool_call_ids_by_index.get(index)
            if not isinstance(call_id, str):
                raise ValueError("chat chunk tool call id is missing")

            function = LiteLLMClient._field(tool_call, "function")
            name = LiteLLMClient._field(function, "name")
            if name is not None and not isinstance(name, str):
                raise ValueError("chat chunk tool call name must be a string")
            arguments = LiteLLMClient._field(function, "arguments")
            if arguments is not None and not isinstance(arguments, str):
                raise ValueError(
                    "chat chunk tool call arguments must be a string"
                )
            result.append((call_id, name, arguments))

        return tuple(result)

    @staticmethod
    def _provider_payload(chunk: object) -> LooseJsonValue | None:
        if isinstance(chunk, Mapping):
            return dict(chunk)
        model_dump = getattr(chunk, "model_dump", None)
        if callable(model_dump):
            payload = model_dump(mode="json")
            if isinstance(payload, Mapping):
                return dict(payload)
        return None


class LiteLLMClient(TextGenerationVendor):
    _reasoning_summary_provider = "litellm"
    _api_key: str | None
    _base_url: str | None

    def __init__(
        self, api_key: str | None = None, base_url: str | None = None
    ):
        self._api_key = api_key
        self._base_url = base_url or "http://localhost:4000"

    async def __call__(
        self,
        model_id: str,
        messages: list[Message],
        settings: GenerationSettings | None = None,
        *,
        instructions: str | None = None,
        tool: ToolManager | None = None,
        use_async_generator: bool = True,
    ) -> TextGenerationStream:
        self._validate_reasoning_summary_request(settings)
        assert (
            instructions is None
        ), "LiteLLM does not support provider instructions"
        template_messages = self._template_messages(messages)
        kwargs: dict[str, Any] = dict(
            model=model_id,
            messages=template_messages,
            api_key=self._api_key,
            stream=use_async_generator,
        )
        if self._base_url:
            kwargs["api_base"] = self._base_url
        result = await litellm.acompletion(**kwargs)
        if use_async_generator:
            return LiteLLMStream(result)

        return TextGenerationSingleStream(
            LiteLLMClient._message_text(result) or "",
            provider_family=ProviderFamily.OPENAI_COMPATIBLE,
            usage=LiteLLMClient._field(result, "usage"),
        )

    @staticmethod
    def _field(value: object, name: str) -> object | None:
        if isinstance(value, Mapping):
            return value.get(name)
        return getattr(value, name, None)

    @staticmethod
    def _delta_text(chunk: object) -> str | None:
        choices = LiteLLMClient._field(chunk, "choices")
        if not isinstance(choices, list) or not choices:
            return None
        delta = LiteLLMClient._field(choices[0], "delta")
        content = LiteLLMClient._field(delta, "content")
        return content if isinstance(content, str) else None

    @staticmethod
    def _message_text(response: object) -> str | None:
        choices = LiteLLMClient._field(response, "choices")
        if not isinstance(choices, list) or not choices:
            return None
        message = LiteLLMClient._field(choices[0], "message")
        content = LiteLLMClient._field(message, "content")
        return content if isinstance(content, str) else None

    @staticmethod
    def _reasoning_text(delta: object) -> str | None:
        for field_name in ("reasoning_content", "reasoning"):
            value = LiteLLMClient._field(delta, field_name)
            if value is not None:
                if not isinstance(value, str):
                    raise ValueError(
                        "chat chunk reasoning delta must be a string"
                    )
                return value
        return None


class LiteLLMModel(TextGenerationVendorModel):
    def _load_model(
        self,
    ) -> PreTrainedModel | TextGenerationVendor | DiffusionPipeline:
        return LiteLLMClient(
            api_key=self._settings.access_token,
            base_url=self._settings.base_url,
        )
