from .....compat import override
from .....entities import (
    GenerationSettings,
    Message,
    MessageContent,
    MessageContentImage,
    MessageContentText,
    MessageRole,
    ReasoningToken,
    Token,
    TokenDetail,
    ToolCallError,
    ToolCallResult,
    ToolCallToken,
)
from .....model.stream import TextGenerationSingleStream
from .....tool.manager import ToolManager
from .....utils import to_json
from ....message import TemplateMessageRole
from ....vendor import TextGenerationVendor, TextGenerationVendorStream
from . import TextGenerationVendorModel

from contextlib import AsyncExitStack
from json import dumps
from typing import Any, AsyncIterator

from aioboto3 import Session as Boto3Session
from diffusers import DiffusionPipeline
from transformers import PreTrainedModel


def _get(event: Any, key: str) -> Any:
    if isinstance(event, dict):
        return event.get(key)
    return getattr(event, key, None)


def _string(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if "text" in value:
            return _string(value["text"])
        if "reasoningText" in value:
            return _string(value["reasoningText"])
        if "string" in value:
            return _string(value["string"])
    return None


class BedrockStream(TextGenerationVendorStream):
    def __init__(self, events: AsyncIterator):
        async def generator() -> AsyncIterator[Token | TokenDetail | str]:
            tool_blocks: dict[int, dict[str, Any]] = {}

            async for event in events:
                content_start = _get(event, "contentBlockStart")
                if content_start:
                    block_index = content_start.get("contentBlockIndex")
                    block = content_start.get("contentBlock") or {}
                    tool = (
                        block.get("toolUse")
                        if isinstance(block, dict)
                        else None
                    )
                    if tool:
                        tool_blocks[block_index] = {
                            "id": tool.get("toolUseId"),
                            "name": tool.get("name"),
                            "fragments": [],
                        }
                        initial = tool.get("input")
                        if initial not in (None, ""):
                            fragment = (
                                initial
                                if isinstance(initial, str)
                                else dumps(initial)
                            )
                            tool_blocks[block_index]["fragments"].append(
                                fragment
                            )
                            yield ToolCallToken(token=fragment)
                    continue

                content_delta = _get(event, "contentBlockDelta")
                if content_delta:
                    block_index = content_delta.get("contentBlockIndex")
                    delta = content_delta.get("delta") or {}
                    text_block = delta.get("text")
                    text_value = _string(text_block)
                    if text_value:
                        yield Token(token=text_value)
                        continue
                    reasoning_block = delta.get("reasoning")
                    reasoning_value = _string(reasoning_block)
                    if reasoning_value:
                        yield ReasoningToken(token=reasoning_value)
                        continue
                    tool_delta = delta.get("toolUse")
                    if tool_delta:
                        fragment_value = tool_delta.get("input")
                        if fragment_value not in (None, ""):
                            fragment = (
                                fragment_value
                                if isinstance(fragment_value, str)
                                else dumps(fragment_value)
                            )
                            tool_block = tool_blocks.setdefault(
                                block_index,
                                {
                                    "id": tool_delta.get("toolUseId"),
                                    "name": tool_delta.get("name"),
                                    "fragments": [],
                                },
                            )
                            tool_block["fragments"].append(fragment)
                            yield ToolCallToken(token=fragment)
                    continue

                content_stop = _get(event, "contentBlockStop")
                if content_stop:
                    block_index = content_stop.get("contentBlockIndex")
                    block = content_stop.get("contentBlock") or {}
                    tool = (
                        block.get("toolUse")
                        if isinstance(block, dict)
                        else None
                    )
                    cached = tool_blocks.pop(block_index, None)
                    if tool:
                        cached = cached or {
                            "id": tool.get("toolUseId"),
                            "name": tool.get("name"),
                            "fragments": [],
                        }
                        final_input = tool.get("input")
                        if final_input not in (None, ""):
                            fragment = (
                                final_input
                                if isinstance(final_input, str)
                                else dumps(final_input)
                            )
                            cached["fragments"].append(fragment)
                    if cached:
                        token = TextGenerationVendor.build_tool_call_token(
                            cached.get("id"),
                            cached.get("name"),
                            "".join(cached.get("fragments", [])) or None,
                        )
                        yield token
                    continue

                if _get(event, "messageStop"):
                    break

        super().__init__(generator())

    async def __anext__(self) -> Token | TokenDetail | str:
        return await self._generator.__anext__()


class BedrockClient(TextGenerationVendor):
    _client: Any | None
    _endpoint_url: str | None
    _exit_stack: AsyncExitStack
    _region_name: str | None
    _session: Boto3Session

    def __init__(
        self,
        *,
        exit_stack: AsyncExitStack,
        region_name: str | None = None,
        endpoint_url: str | None = None,
    ) -> None:
        self._session = Boto3Session()
        self._region_name = region_name
        self._endpoint_url = endpoint_url
        self._exit_stack = exit_stack
        self._client = None

    async def _client_instance(self) -> Any:
        if self._client is None:
            kwargs: dict[str, Any] = {}
            if self._region_name:
                kwargs["region_name"] = self._region_name
            if self._endpoint_url:
                kwargs["endpoint_url"] = self._endpoint_url
            self._client = await self._exit_stack.enter_async_context(
                self._session.client("bedrock-runtime", **kwargs)
            )
        return self._client

    @override
    async def __call__(
        self,
        model_id: str,
        messages: list[Message],
        settings: GenerationSettings | None = None,
        *,
        tool: ToolManager | None = None,
        use_async_generator: bool = True,
    ) -> AsyncIterator[Token | TokenDetail | str] | TextGenerationSingleStream:
        client = await self._client_instance()
        system_prompt = self._system_prompt(messages)
        template_messages = self._template_messages(messages, ["system"])
        payload: dict[str, Any] = {
            "modelId": model_id,
            "messages": template_messages,
        }
        if system_prompt:
            payload["system"] = [{"text": system_prompt}]
        inference = self._inference_config(settings)
        if inference:
            payload["inferenceConfig"] = inference
        tool_config = self._tool_config(tool)
        if tool_config:
            payload["toolConfig"] = tool_config

        if use_async_generator:
            response = await client.converse_stream(**payload)
            stream = (
                response.get("stream") if isinstance(response, dict) else None
            )
            assert stream is not None, "Missing stream in Converse response"
            events = (
                await self._exit_stack.enter_async_context(stream)
                if hasattr(stream, "__aenter__")
                else stream
            )
            return BedrockStream(events=events)

        response = await client.converse(**payload)
        return TextGenerationSingleStream(self._response_text(response))

    def _inference_config(
        self, settings: GenerationSettings | None
    ) -> dict[str, Any] | None:
        if settings is None:
            return None
        config: dict[str, Any] = {}
        if settings.max_new_tokens is not None:
            config["maxTokens"] = settings.max_new_tokens
        if settings.temperature is not None:
            config["temperature"] = settings.temperature
        if settings.top_p is not None:
            config["topP"] = settings.top_p
        if settings.top_k is not None:
            config["topK"] = settings.top_k
        if settings.stop_strings is not None:
            stop = (
                [settings.stop_strings]
                if isinstance(settings.stop_strings, str)
                else settings.stop_strings
            )
            config["stopSequences"] = stop
        return config or None

    def _tool_config(self, tool: ToolManager | None) -> dict[str, Any] | None:
        schemas = self._tool_schemas(tool) if tool else None
        if not schemas:
            return None
        return {"tools": schemas, "toolChoice": {"auto": {}}}

    def _response_text(self, response: dict[str, Any]) -> str:
        output = response.get("output") if isinstance(response, dict) else None
        message = output.get("message") if isinstance(output, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            return ""
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            text_block = block.get("text")
            text_value = _string(text_block)
            if text_value:
                parts.append(text_value)
        return "".join(parts)

    def _template_messages(
        self,
        messages: list[Message],
        exclude_roles: list[TemplateMessageRole] | None = None,
    ) -> list[dict[str, Any]]:
        templated: list[dict[str, Any]] = []
        for message in messages:
            if exclude_roles and str(message.role) in exclude_roles:
                continue
            if message.role == MessageRole.TOOL:
                result = message.tool_call_result or message.tool_call_error
                if result:
                    templated.append(self._tool_result_message(result))
                continue
            templated.append(self._format_message(message))
        return templated

    def _format_message(self, message: Message) -> dict[str, Any]:
        role = str(message.role)
        if role == str(MessageRole.DEVELOPER):
            role = str(MessageRole.USER)
        content_blocks = self._format_content(message.content)
        if message.tool_calls:
            for tool_call in message.tool_calls:
                encoded_name = TextGenerationVendor.encode_tool_name(
                    tool_call.name
                )
                content_blocks.append(
                    {
                        "toolUse": {
                            "toolUseId": tool_call.id,
                            "name": encoded_name,
                            "input": tool_call.arguments or [],
                        }
                    }
                )
        return {"role": role, "content": content_blocks}

    def _format_content(
        self, content: str | MessageContent | list[MessageContent] | None
    ) -> list[dict[str, Any]]:
        if content is None:
            return []
        if isinstance(content, str):
            return [{"text": {"text": content}}]
        if isinstance(content, MessageContentText):
            return [{"text": {"text": content.text}}]
        if isinstance(content, MessageContentImage):
            return [
                {"image": {"source": self._image_source(content.image_url)}}
            ]
        if isinstance(content, list):
            blocks: list[dict[str, Any]] = []
            for block in content:
                if isinstance(block, MessageContentText):
                    blocks.append({"text": {"text": block.text}})
                elif isinstance(block, MessageContentImage):
                    blocks.append(
                        {
                            "image": {
                                "source": self._image_source(block.image_url)
                            }
                        }
                    )
            return blocks
        return [{"text": {"text": str(content)}}]

    def _image_source(self, image_url: dict[str, Any]) -> dict[str, Any]:
        if "url" in image_url:
            return {"type": "url", "url": image_url["url"]}
        if "data" in image_url:
            media_type = image_url.get("mime_type", "image/png")
            return {
                "type": "base64",
                "mediaType": media_type,
                "data": image_url["data"],
            }
        return {"type": "url", "url": image_url.get("uri", "")}

    def _tool_result_message(
        self, result: ToolCallResult | ToolCallError
    ) -> dict[str, Any]:
        content: dict[str, Any] = {
            "toolUseId": result.call.id,
            "content": [
                {
                    "text": {
                        "text": to_json(
                            result.result
                            if isinstance(result, ToolCallResult)
                            else result.message
                        )
                    }
                }
            ],
            "status": (
                "success" if isinstance(result, ToolCallResult) else "error"
            ),
        }
        if isinstance(result, ToolCallError):
            content["error"] = {
                "name": result.error.__class__.__name__,
                "message": result.message,
            }
        return {
            "role": str(MessageRole.USER),
            "content": [{"toolResult": content}],
        }

    @staticmethod
    def _tool_schemas(tool: ToolManager) -> list[dict[str, Any]] | None:
        schemas = tool.json_schemas()
        if not schemas:
            return None
        tools: list[dict[str, Any]] = []
        for schema in schemas:
            if schema.get("type") != "function":
                continue
            function = schema.get("function") or {}
            encoded_name = TextGenerationVendor.encode_tool_name(
                function.get("name", "")
            )
            tools.append(
                {
                    "toolSpec": {
                        "name": encoded_name,
                        "description": function.get("description", ""),
                        "inputSchema": {
                            "json": function.get("parameters", {})
                        },
                    }
                }
            )
        return tools or None


class BedrockModel(TextGenerationVendorModel):
    def _load_model(
        self,
    ) -> PreTrainedModel | TextGenerationVendor | DiffusionPipeline:
        return BedrockClient(
            exit_stack=self._exit_stack,
            region_name=self._settings.base_url,
            endpoint_url=self._settings.access_token,
        )
