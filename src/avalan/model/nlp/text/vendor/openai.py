from .....entities import (
    GenerationSettings,
    Input,
    Message,
    MessageRole,
    ReasoningEffort,
    ReasoningToken,
    Token,
    TokenDetail,
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
from typing import Any, AsyncIterator, cast

from diffusers import DiffusionPipeline
from openai import AsyncOpenAI, Omit
from transformers import PreTrainedModel


class OpenAIStream(TextGenerationVendorStream):
    _TEXT_DELTA_EVENTS = {"response.text.delta", "response.output_text.delta"}
    _REASONING_DELTA_EVENTS = {"response.reasoning_text.delta"}

    def __init__(self, stream: AsyncIterator[Any]) -> None:
        async def generator() -> AsyncIterator[Token | TokenDetail | str]:
            tool_calls: dict[str, dict[str, str | list[str] | None]] = {}

            async for event in stream:
                etype = getattr(event, "type", None)

                if etype == "response.output_item.added":
                    item = getattr(event, "item", None)
                    if item:
                        custom = getattr(item, "custom_tool_call", None)
                        if custom:
                            call_id = getattr(
                                custom, "id", getattr(item, "id", None)
                            )
                            if not isinstance(call_id, str):
                                continue
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
                    if isinstance(call_id, str) and isinstance(delta, str):
                        tc = tool_calls.setdefault(
                            call_id, {"name": None, "args_fragments": []}
                        )
                        args_fragments = tc["args_fragments"]
                        assert isinstance(args_fragments, list)
                        args_fragments.append(delta)
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
                    call_id_value = getattr(item, "id", None) if item else None
                    call_id = (
                        call_id_value
                        if isinstance(call_id_value, str)
                        else None
                    )
                    cached = (
                        tool_calls.pop(call_id, None)
                        if isinstance(call_id, str)
                        else None
                    )
                    if cached:
                        args_fragments = cached["args_fragments"]
                        assert isinstance(args_fragments, list)
                        yield TextGenerationVendor.build_tool_call_token(
                            call_id,
                            cached.get("name"),
                            "".join(args_fragments) or None,
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

        super().__init__(generator())

    async def __anext__(self) -> Token | TokenDetail | str:
        return await self._generator.__anext__()


class OpenAIClient(TextGenerationVendor):
    _DEFAULT_MODEL_ID = "default"
    _client: AsyncOpenAI

    def __init__(self, api_key: str | None, base_url: str | None):
        client_kwargs: dict[str, Any] = {"base_url": base_url}
        if api_key is None:
            assert base_url
            client_kwargs.update(
                api_key="",
                default_headers=cast(Any, {"Authorization": Omit()}),
            )
        else:
            client_kwargs["api_key"] = api_key
        self._client = AsyncOpenAI(**client_kwargs)

    async def __call__(
        self,
        model_id: str,
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
            "model": model_id or self._DEFAULT_MODEL_ID,
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
            reasoning = OpenAIClient._reasoning_config(settings)
            if reasoning:
                kwargs["reasoning"] = reasoning
        if tool:
            schemas = OpenAIClient._tool_schemas(tool)
            if schemas:
                kwargs["tools"] = schemas
        client_stream = await self._client.responses.create(**kwargs)

        if use_async_generator:
            return OpenAIStream(stream=client_stream)

        content = OpenAIClient._non_stream_response_content(client_stream)
        return TextGenerationSingleStream(content)

    def _template_messages(
        self,
        messages: list[Message],
        exclude_roles: list[TemplateMessageRole] | None = None,
    ) -> list[TemplateMessage] | list[dict[str, Any]]:
        tool_results = [
            message.tool_call_result or message.tool_call_error
            for message in messages
            if message.role == MessageRole.TOOL
            and (message.tool_call_result or message.tool_call_error)
        ]
        do_exclude_roles = [*(exclude_roles or []), "tool"]
        template_messages = super()._template_messages(
            messages, do_exclude_roles
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
        for result in tool_results:
            assert result is not None
            call_message = {
                "type": "function_call",
                "name": TextGenerationVendor.encode_tool_name(
                    result.call.name
                ),
                "call_id": result.call.id,
                "arguments": dumps(result.call.arguments),
            }
            messages_out.append(call_message)

            result_message = {
                "type": "function_call_output",
                "call_id": result.call.id,
                "output": to_json(
                    result.result
                    if isinstance(result, ToolCallResult)
                    else {"error": result.message}
                ),
            }
            messages_out.append(result_message)
        return messages_out

    @staticmethod
    def _reasoning_config(
        settings: GenerationSettings,
    ) -> dict[str, str] | None:
        effort = settings.reasoning.effort
        if effort is None:
            return None
        if effort == ReasoningEffort.MAX:
            effort = ReasoningEffort.XHIGH
        return {"effort": effort.value}

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

        if isinstance(file_id, str):
            payload["file_id"] = file_id
        elif isinstance(file_url, str):
            payload["file_url"] = file_url
        elif isinstance(file_data, str):
            payload["file_data"] = file_data
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
            payload["image_url"] = f"data:{mime_type};base64,{image_data}"
        else:
            raise AssertionError(
                "OpenAI image blocks require file_id, url, or data"
            )

        if isinstance(detail, str):
            payload["detail"] = detail
        return payload

    @staticmethod
    def _tool_schemas(tool: ToolManager) -> list[dict[str, Any]] | None:
        schemas = tool.json_schemas()
        return (
            [
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
            if schemas
            else None
        )

    @staticmethod
    def _non_stream_response_content(response: object) -> str:
        def _get(value: object, attribute: str) -> object | None:
            if isinstance(value, dict):
                return value.get(attribute)
            return getattr(value, attribute, None)

        parts: list[str] = []
        output = _get(response, "output")
        if not isinstance(output, list):
            return "".join(parts)

        for item in output:
            item_type = _get(item, "type")
            contents = _get(item, "content")
            if not isinstance(contents, list):
                contents = []

            if item_type in {None, "message", "output_text"}:
                for content in contents:
                    text = _get(content, "text")
                    if isinstance(text, str):
                        parts.append(text)
                continue

            if item_type in {"tool_call", "function_call"}:
                call = _get(item, "call") or item
                function = _get(call, "function") or call
                token = TextGenerationVendor.build_tool_call_token(
                    _get(call, "id"),
                    _get(function, "name"),
                    _get(function, "arguments"),
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

    async def __call__(
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
        streamer = await self._model(
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
