from .....model.stream import TextGenerationSingleStream
from . import TextGenerationVendorModel
from ....message import TemplateMessage, TemplateMessageRole
from ....vendor import TextGenerationVendor, TextGenerationVendorStream
from .....compat import override
from .....entities import (
    GenerationSettings,
    Message,
    MessageRole,
    ReasoningToken,
    Token,
    TokenDetail,
    ToolCall,
    ToolCallToken,
)
from .....tool.manager import ToolManager
from dataclasses import asdict, is_dataclass
from diffusers import DiffusionPipeline
from json import dumps, loads
from openai import AsyncOpenAI
from transformers import PreTrainedModel
from typing import AsyncIterator


class OpenAIStream(TextGenerationVendorStream):
    _TEXT_DELTA_EVENTS = {"response.text.delta", "response.output_text.delta"}
    _REASONING_DELTA_EVENTS = {"response.reasoning_text.delta"}

    def __init__(self, stream: AsyncIterator):
        async def generator() -> AsyncIterator[Token | TokenDetail | str]:
            tool_calls: dict[str, dict] = {}

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
                    call_id = getattr(item, "id", None) if item else None
                    cached = tool_calls.pop(call_id, None)
                    if cached:
                        tool_call = ToolCall(
                            id=call_id,
                            name=TextGenerationVendor.decode_tool_name(
                                cached.get("name")
                            ),
                            arguments="".join(cached["args_fragments"])
                            or None,
                        )
                        yield ToolCallToken(token="<tool_use>", call=tool_call)
                    elif (
                        item is not None
                        and getattr(item, "type", None) == "function_call"
                    ):
                        tool_name = getattr(item, "name", None)
                        tool_id = getattr(item, "id", None)

                        if tool_id and tool_name:
                            arguments = getattr(item, "arguments", None)
                            if arguments:
                                arguments = loads(arguments)
                            tool_call = ToolCall(
                                id=tool_id,
                                name=TextGenerationVendor.decode_tool_name(
                                    tool_name
                                ),
                                arguments=arguments or None,
                            )

                            token = ToolCallToken(
                                token="<tool_use>", call=tool_call
                            )
                            yield token

                    continue

        super().__init__(generator())

    async def __anext__(self) -> Token | TokenDetail | str:
        return await self._generator.__anext__()


class OpenAIClient(TextGenerationVendor):
    _client: AsyncOpenAI

    def __init__(self, api_key: str, base_url: str | None):
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    @override
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
        kwargs: dict = {
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

        stream = (
            OpenAIStream(stream=client_stream)
            if use_async_generator
            else TextGenerationSingleStream(
                client_stream.output[0].content[0].text
            )
        )

        return stream

    def _template_messages(
        self,
        messages: list[Message],
        exclude_roles: list[TemplateMessageRole] | None = None,
    ) -> list[TemplateMessage]:
        tool_results = [
            message.tool_call_result
            for message in messages
            if message.role == MessageRole.TOOL and message.tool_call_result
        ]
        do_exclude_roles = [*(exclude_roles or []), "tool"]
        messages = super()._template_messages(messages, do_exclude_roles)
        for result in tool_results:
            call_message = {
                "type": "function_call",
                "name": TextGenerationVendor.encode_tool_name(
                    result.call.name
                ),
                "call_id": result.call.id,
                "arguments": dumps(result.call.arguments)
            }
            messages.append(call_message)

            result_message = {
                "type": "function_call_output",
                "call_id": result.call.id,
                "output": dumps(
                    asdict(result.result)
                    if is_dataclass(result.result)
                    else result.result
                ),
            }
            messages.append(result_message)
        return messages

    @staticmethod
    def _tool_schemas(tool: ToolManager) -> list[dict] | None:
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
                for t in tool.json_schemas()
                if t["type"] == "function"
            ]
            if schemas
            else None
        )


class OpenAIModel(TextGenerationVendorModel):
    def _load_model(
        self,
    ) -> PreTrainedModel | TextGenerationVendor | DiffusionPipeline:
        assert self._settings.base_url or self._settings.access_token
        return OpenAIClient(
            base_url=self._settings.base_url,
            api_key=self._settings.access_token,
        )
