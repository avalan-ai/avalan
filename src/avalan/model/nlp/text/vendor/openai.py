from .....compat import override
from .....entities import (
    GenerationSettings,
    Input,
    Message,
    MessageRole,
    ReasoningToken,
    Token,
    TokenDetail,
    ToolCallError,
    ToolCallResult,
    ToolCallToken,
)
from .....model.response.text import TextGenerationResponse
from .....model.stream import TextGenerationSingleStream
from .....tool.manager import ToolManager
from .....utils import to_json
from ....message import TemplateMessage, TemplateMessageRole
from ....vendor import TextGenerationVendor, TextGenerationVendorStream
from . import TextGenerationVendorModel

from json import dumps
from typing import Any, AsyncIterator

from diffusers import DiffusionPipeline
from openai import AsyncOpenAI
from transformers import PreTrainedModel


class OpenAIStream(TextGenerationVendorStream):
    _TEXT_DELTA_EVENTS: set[str] = {
        "response.text.delta",
        "response.output_text.delta",
    }
    _REASONING_DELTA_EVENTS: set[str] = {"response.reasoning_text.delta"}

    def __init__(self, stream: AsyncIterator[Any]) -> None:
        async def generator() -> AsyncIterator[Token | TokenDetail | str]:
            tool_calls: dict[str, dict[str, Any]] = {}

            async for event in stream:
                etype = getattr(event, "type", None)

                if etype == "response.output_item.added":
                    item = getattr(event, "item", None)
                    if item:
                        custom = getattr(item, "custom_tool_call", None)
                        if custom:
                            call_id: str | None = getattr(
                                custom, "id", getattr(item, "id", None)
                            )
                            if call_id is not None:
                                tool_calls[call_id] = {
                                    "name": getattr(custom, "name", None),
                                    "args_fragments": [],
                                }
                    continue

                if (
                    etype == "response.custom_tool_call_input.delta"
                    or etype == "response.function_call_arguments.delta"
                ):
                    call_id = getattr(event, "id", None)
                    delta = getattr(event, "delta", None)
                    if call_id is not None and delta:
                        tc = tool_calls.setdefault(
                            call_id, {"name": None, "args_fragments": []}
                        )
                        tc["args_fragments"].append(delta)
                        yield ToolCallToken(token=delta)
                    continue

                if etype in self._REASONING_DELTA_EVENTS:
                    delta = getattr(event, "delta", None)
                    if isinstance(delta, str):
                        yield ReasoningToken(token=delta)
                    continue

                if etype in self._TEXT_DELTA_EVENTS:
                    delta = getattr(event, "delta", None)
                    if isinstance(delta, str):
                        yield Token(token=delta)
                    continue

                if etype == "response.output_item.done":
                    item = getattr(event, "item", None)
                    done_call_id: str | None = (
                        getattr(item, "id", None) if item else None
                    )
                    cached = (
                        tool_calls.pop(done_call_id, None)
                        if done_call_id
                        else None
                    )
                    if cached:
                        yield TextGenerationVendor.build_tool_call_token(
                            done_call_id,
                            cached.get("name"),
                            "".join(cached["args_fragments"]) or None,
                        )
                    elif (
                        item is not None
                        and getattr(item, "type", None) == "function_call"
                    ):
                        tool_name = getattr(item, "name", None)
                        tool_id = getattr(item, "id", None)

                        if tool_id and tool_name:
                            token = TextGenerationVendor.build_tool_call_token(
                                tool_id,
                                tool_name,
                                getattr(item, "arguments", None),
                            )
                            yield token

                    continue

        super().__init__(generator())  # type: ignore[arg-type]

    async def __anext__(self) -> Token | TokenDetail | str:
        return await self._generator.__anext__()  # type: ignore[no-any-return]


class OpenAIClient(TextGenerationVendor):
    _client: AsyncOpenAI

    def __init__(self, api_key: str | None, base_url: str | None) -> None:
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    @override
    async def __call__(  # type: ignore[override]
        self,
        model_id: str | None,
        messages: list[Message],
        settings: GenerationSettings | None = None,
        *,
        timeout: int | None = None,
        tool: ToolManager | None = None,
        use_async_generator: bool = True,
    ) -> AsyncIterator[Token | TokenDetail | str] | TextGenerationSingleStream:
        template_messages = self._template_messages(messages)
        kwargs: dict[str, Any] = {
            "extra_headers": {
                "X-Title": "Avalan",
                "HTTP-Referer": "https://github.com/avalan-ai/avalan",
            },
            "model": model_id,
            "input": template_messages,
            "stream": use_async_generator,
            "timeout": timeout,
        }
        if settings:
            if settings.max_new_tokens is not None:
                kwargs["max_output_tokens"] = settings.max_new_tokens
            if settings.temperature is not None:
                kwargs["temperature"] = settings.temperature
            if settings.top_p is not None:
                kwargs["top_p"] = settings.top_p
            if settings.stop_strings is not None:
                kwargs["text"] = {"stop": settings.stop_strings}
            if settings.response_format is not None:
                kwargs["response_format"] = settings.response_format
        if tool:
            schemas = OpenAIClient._tool_schemas(tool)
            if schemas:
                kwargs["tools"] = schemas
        client_stream = await self._client.responses.create(**kwargs)

        if use_async_generator:
            return OpenAIStream(stream=client_stream)

        content = OpenAIClient._non_stream_response_content(client_stream)
        return TextGenerationSingleStream(content)

    def _template_messages(  # type: ignore[override]
        self,
        messages: list[Message],
        exclude_roles: list[TemplateMessageRole] | None = None,
    ) -> list[TemplateMessage | dict[str, Any]]:
        tool_results: list[ToolCallResult | ToolCallError] = [
            message.tool_call_result or message.tool_call_error  # type: ignore[misc]
            for message in messages
            if message.role == MessageRole.TOOL
            and (message.tool_call_result or message.tool_call_error)
        ]
        do_exclude_roles: list[TemplateMessageRole] = [
            *(exclude_roles or []),
            "tool",
        ]
        template_messages: list[TemplateMessage | dict[str, Any]] = list(
            super()._template_messages(messages, do_exclude_roles)
        )
        for result in tool_results:
            assert result.call is not None
            call_message: dict[str, Any] = {
                "type": "function_call",
                "name": TextGenerationVendor.encode_tool_name(
                    result.call.name
                ),
                "call_id": result.call.id,
                "arguments": dumps(result.call.arguments),
            }
            template_messages.append(call_message)

            result_message: dict[str, Any] = {
                "type": "function_call_output",
                "call_id": result.call.id,
                "output": to_json(
                    result.result
                    if isinstance(result, ToolCallResult)
                    else {"error": result.message}
                ),
            }
            template_messages.append(result_message)
        return template_messages

    @staticmethod
    def _tool_schemas(tool: ToolManager) -> list[dict[str, Any]] | None:
        schemas = tool.json_schemas()
        if not schemas:
            return None
        return [
            {
                "type": t["type"],
                **t["function"],
                **{
                    "name": TextGenerationVendor.encode_tool_name(
                        t["function"]["name"]
                    )
                },
            }
            for t in schemas
            if t["type"] == "function"
        ]

    @staticmethod
    def _non_stream_response_content(response: object) -> str:
        def _get(value: object, attribute: str) -> Any:
            if isinstance(value, dict):
                return value.get(attribute)
            return getattr(value, attribute, None)

        parts: list[str] = []
        output_items = _get(response, "output") or []
        for item in output_items:
            item_type = _get(item, "type")
            contents = _get(item, "content") or []

            if item_type in {None, "message", "output_text"}:
                for content in contents:
                    text = _get(content, "text")
                    if isinstance(text, str):
                        parts.append(text)
                continue

            if item_type in {"tool_call", "function_call"}:
                call = _get(item, "call") or item
                function = _get(call, "function") or call
                call_id: str | None = _get(call, "id")
                func_name: str | None = _get(function, "name")
                func_args: str | dict[str, Any] | None = _get(
                    function, "arguments"
                )
                token = TextGenerationVendor.build_tool_call_token(
                    call_id,
                    func_name,
                    func_args,
                )
                parts.append(token.token)

        return "".join(parts)


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
    def _load_model(
        self,
    ) -> PreTrainedModel | TextGenerationVendor | DiffusionPipeline:
        assert self._settings.base_url or self._settings.access_token
        return OpenAIClient(
            base_url=self._settings.base_url,
            api_key=self._settings.access_token,
        )

    @override
    async def __call__(  # type: ignore[override]
        self,
        input: Input,
        system_prompt: str | None = None,
        developer_prompt: str | None = None,
        settings: GenerationSettings | None = None,
        *,
        tool: ToolManager | None = None,
    ) -> TextGenerationResponse:
        generation_settings = settings or GenerationSettings()
        messages = self._messages(input, system_prompt, developer_prompt, tool)
        assert self._model is not None
        assert self._model_id is not None
        streamer = await self._model(  # type: ignore[operator]
            self._model_id,
            messages,
            generation_settings,
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
