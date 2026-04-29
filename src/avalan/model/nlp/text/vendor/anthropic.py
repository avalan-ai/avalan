from .....entities import (
    GenerationSettings,
    Message,
    MessageContentFile,
    MessageContentImage,
    MessageRole,
    ReasoningEffort,
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
from ....message import TemplateMessage, TemplateMessageRole
from ....vendor import TextGenerationVendor, TextGenerationVendorStream
from . import TextGenerationVendorModel, _decode_text_file_data

from contextlib import AsyncExitStack
from typing import Any, AsyncIterator, cast

from anthropic import APIStatusError, AsyncAnthropic
from anthropic.types import RawContentBlockDeltaEvent, RawMessageStopEvent
from diffusers import DiffusionPipeline
from transformers import PreTrainedModel


class AnthropicStream(TextGenerationVendorStream):
    def __init__(self, events: AsyncIterator[object]):
        async def generator() -> AsyncIterator[Token | TokenDetail | str]:
            tool_blocks: dict[int, dict[str, Any]] = {}

            async for event in events:
                etype = getattr(event, "type", None)

                if etype == "content_block_start":
                    cb = getattr(event, "content_block", None)
                    if (
                        cb is not None
                        and getattr(cb, "type", None) == "tool_use"
                    ):
                        index = cast(int | None, getattr(event, "index", None))
                        if index is None:
                            continue
                        tool_blocks[index] = {
                            "id": getattr(cb, "id", None),
                            "name": getattr(cb, "name", None),
                            "args_fragments": [],
                        }
                    continue

                if isinstance(event, RawContentBlockDeltaEvent):
                    delta = event.delta

                    if hasattr(delta, "thinking") and delta.thinking:
                        yield ReasoningToken(token=delta.thinking)
                        continue

                    if (
                        hasattr(delta, "partial_json")
                        and delta.partial_json is not None
                    ):
                        index = cast(int | None, getattr(event, "index", None))
                        if index is None:
                            continue
                        tb = tool_blocks.setdefault(
                            index,
                            {"id": None, "name": None, "args_fragments": []},
                        )
                        tb["args_fragments"].append(delta.partial_json)
                        yield ToolCallToken(token=delta.partial_json)
                        continue

                    if hasattr(delta, "text") and delta.text:
                        yield Token(token=delta.text)
                        continue

                    continue

                if etype == "content_block_stop":
                    cb = getattr(event, "content_block", None)
                    if (
                        cb is not None
                        and getattr(cb, "type", None) == "tool_use"
                    ):
                        tool_name = getattr(cb, "name", None)
                        tool_id = getattr(cb, "id", None)

                        if tool_id and tool_name:
                            token = TextGenerationVendor.build_tool_call_token(
                                tool_id,
                                tool_name,
                                getattr(cb, "input", None),
                            )
                            yield token

                        index = cast(int | None, getattr(event, "index", None))
                        if index is not None and index in tool_blocks:
                            del tool_blocks[index]

                    continue

                if (
                    isinstance(event, RawMessageStopEvent)
                    or etype == "message_stop"
                ):
                    break

        super().__init__(cast(AsyncIterator[str | ToolCallToken], generator()))

    async def __anext__(self) -> str | ToolCallToken:
        return cast(str | ToolCallToken, await self._generator.__anext__())


class AnthropicClient(TextGenerationVendor):
    _RETIRED_MODEL_REPLACEMENTS = {
        "claude-3-5-sonnet-20240620": "claude-sonnet-4-6",
        "claude-3-5-sonnet-20241022": "claude-sonnet-4-6",
        "claude-3-5-sonnet-latest": "claude-sonnet-4-6",
    }

    _client: AsyncAnthropic
    _exit_stack: AsyncExitStack

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        *,
        exit_stack: AsyncExitStack,
    ):
        self._client = AsyncAnthropic(api_key=api_key, base_url=base_url)
        self._exit_stack = exit_stack

    async def __call__(
        self,
        model_id: str,
        messages: list[Message],
        settings: GenerationSettings | None = None,
        *,
        tool: ToolManager | None = None,
        use_async_generator: bool = True,
    ) -> TextGenerationVendorStream:
        settings = settings or GenerationSettings()
        system_prompt = self._system_prompt(messages)
        template_messages = self._template_messages(messages, ["system"])
        kwargs: dict[str, Any] = {
            "model": model_id,
            "system": system_prompt,
            "messages": template_messages,
            "max_tokens": settings.max_new_tokens,
            "tools": AnthropicClient._tool_schemas(tool) if tool else [],
            "tool_choice": {"type": "auto"},
        }
        if settings.temperature is not None:
            kwargs["temperature"] = settings.temperature
        extra_headers = AnthropicClient._extra_headers(messages)
        if extra_headers:
            kwargs["extra_headers"] = extra_headers
        output_config = AnthropicClient._output_config(settings)
        if output_config:
            kwargs["output_config"] = output_config

        try:
            if use_async_generator:
                stream = self._client.messages.stream(**kwargs)
                events = await self._exit_stack.enter_async_context(stream)
                return AnthropicStream(events=events)

            response = await self._client.messages.create(**kwargs)
            content = self._non_stream_response_content(response)
            return cast(
                TextGenerationVendorStream,
                TextGenerationSingleStream(content),
            )
        except Exception as error:
            AnthropicClient._translate_api_error(model_id, error)
            raise

    def _template_messages(
        self,
        messages: list[Message],
        exclude_roles: list[TemplateMessageRole] | None = None,
    ) -> list[TemplateMessage]:
        tool_results = [
            message.tool_call_result or message.tool_call_error
            for message in messages
            if message.role == MessageRole.TOOL
            and (message.tool_call_result or message.tool_call_error)
        ]
        do_exclude_roles = [*(exclude_roles or []), "tool"]
        template_messages = cast(
            list[dict[str, Any]],
            super()._template_messages(messages, do_exclude_roles),
        )
        for message in template_messages:
            content = message.get("content")
            if isinstance(content, list):
                message["content"] = [
                    AnthropicClient._content_block(block)
                    for block in content
                    if isinstance(block, dict)
                ]
        last_message = next(
            (
                m
                for m in reversed(template_messages)
                if m["role"] == str(MessageRole.ASSISTANT)
            ),
            None,
        )
        last_message_index = (
            template_messages.index(last_message) if last_message else None
        )
        if last_message_index is not None and last_message:
            template_messages[last_message_index] = {
                "role": last_message["role"],
                "content": [
                    (
                        {"type": "text", "text": last_message["content"]}
                        if isinstance(last_message["content"], str)
                        else last_message["content"]
                    ),
                    *[
                        {
                            "type": "tool_use",
                            "id": r.call.id,
                            "name": TextGenerationVendor.encode_tool_name(
                                r.call.name
                            ),
                            "input": r.call.arguments,
                        }
                        for r in tool_results
                        if r is not None
                    ],
                ],
            }

        for result in tool_results:
            if result is None:
                continue
            tool_result_content: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": result.call.id,
                "content": to_json(
                    result.result
                    if isinstance(result, ToolCallResult)
                    else result.message
                ),
            }
            if isinstance(result, ToolCallError):
                tool_result_content["is_error"] = True
            result_message: dict[str, Any] = {
                "role": "user",
                "content": [tool_result_content],
            }
            if last_message_index is not None:
                template_messages.insert(
                    last_message_index + 1, result_message
                )
            else:
                template_messages.append(result_message)

        # @TODO Ensure this doesn't happen from upstream
        if len(template_messages) > 1 and (
            template_messages[0] == template_messages[-1]
        ):
            template_messages.pop()

        return cast(list[TemplateMessage], template_messages)

    @staticmethod
    def _output_config(
        settings: GenerationSettings,
    ) -> dict[str, str] | None:
        effort = settings.reasoning.effort
        if effort is None:
            return None
        match effort:
            case ReasoningEffort.NONE | ReasoningEffort.MINIMAL:
                resolved_effort = ReasoningEffort.LOW
            case ReasoningEffort.XHIGH:
                resolved_effort = ReasoningEffort.MAX
            case _:
                resolved_effort = effort
        return {"effort": resolved_effort.value}

    @staticmethod
    def _content_block(block: dict[str, Any]) -> dict[str, Any]:
        block_type = block.get("type")
        match block_type:
            case "file":
                file = block.get("file")
                assert isinstance(file, dict), "File blocks require file data"
                return AnthropicClient._document_block(file)
            case "image_url":
                image = block.get("image_url")
                assert isinstance(
                    image, dict
                ), "Image blocks require image data"
                return {
                    "type": "image",
                    "source": AnthropicClient._image_source(image),
                }
            case "text":
                return block
            case _:
                return block

    @staticmethod
    def _document_block(file: dict[str, Any]) -> dict[str, Any]:
        source = AnthropicClient._document_source(file)
        block: dict[str, Any] = {"type": "document", "source": source}
        title = file.get("title")
        context = file.get("context")
        citations = file.get("citations")
        if isinstance(title, str):
            block["title"] = title
        if isinstance(context, str):
            block["context"] = context
        if isinstance(citations, bool):
            block["citations"] = {"enabled": citations}
        return block

    @staticmethod
    def _document_source(file: dict[str, Any]) -> dict[str, Any]:
        file_id = file.get("file_id")
        file_url = file.get("file_url") or file.get("url")
        file_data = file.get("file_data") or file.get("data")
        mime_type = file.get("mime_type")
        if isinstance(file_id, str):
            return {"type": "file", "file_id": file_id}
        if isinstance(file_url, str):
            return {"type": "url", "url": file_url}
        if isinstance(file_data, str):
            resolved_mime_type = (
                mime_type if isinstance(mime_type, str) else "application/pdf"
            )
            if resolved_mime_type == "text/plain":
                return {
                    "type": "text",
                    "media_type": resolved_mime_type,
                    "data": _decode_text_file_data(file_data),
                }
            return {
                "type": "base64",
                "media_type": resolved_mime_type,
                "data": file_data,
            }
        raise AssertionError(
            "Anthropic file blocks require file_id, file_url, or file_data"
        )

    @staticmethod
    def _image_source(image: dict[str, Any]) -> dict[str, Any]:
        file_id = image.get("file_id")
        image_url = image.get("url") or image.get("uri")
        image_data = image.get("data")
        mime_type = image.get("mime_type") or "image/png"
        if isinstance(file_id, str):
            return {"type": "file", "file_id": file_id}
        if isinstance(image_url, str):
            return {"type": "url", "url": image_url}
        if isinstance(image_data, str):
            return {
                "type": "base64",
                "media_type": mime_type,
                "data": image_data,
            }
        raise AssertionError(
            "Anthropic image blocks require file_id, url, or data"
        )

    @staticmethod
    def _extra_headers(messages: list[Message]) -> dict[str, str] | None:
        if AnthropicClient._uses_files_api(messages):
            return {"anthropic-beta": "files-api-2025-04-14"}
        return None

    @staticmethod
    def _uses_files_api(messages: list[Message]) -> bool:
        def _content_uses_file_api(
            content: object,
        ) -> bool:
            if isinstance(content, MessageContentFile):
                return isinstance(content.file.get("file_id"), str)
            if isinstance(content, MessageContentImage):
                return isinstance(content.image_url.get("file_id"), str)
            if isinstance(content, list):
                return any(_content_uses_file_api(block) for block in content)
            return False

        return any(
            _content_uses_file_api(message.content) for message in messages
        )

    @staticmethod
    def _tool_schemas(tool: ToolManager) -> list[dict[str, Any]] | None:
        schemas = tool.json_schemas()
        return (
            [
                {
                    "name": TextGenerationVendor.encode_tool_name(
                        t["function"]["name"]
                    ),
                    "description": t["function"]["description"],
                    "input_schema": {
                        **t["function"]["parameters"],
                        "additionalProperties": False,
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
        content_blocks = _get(response, "content")
        if not isinstance(content_blocks, list):
            return "".join(parts)
        for block in content_blocks:
            block_type = _get(block, "type")
            if block_type == "text":
                text = _get(block, "text")
                if isinstance(text, str):
                    parts.append(text)
                continue

            if block_type == "tool_use":
                token = TextGenerationVendor.build_tool_call_token(
                    _get(block, "id"),
                    _get(block, "name"),
                    _get(block, "input"),
                )
                parts.append(token.token)

        return "".join(parts)

    @staticmethod
    def _error_message(error: Exception) -> str:
        body = getattr(error, "body", None)
        if isinstance(body, dict):
            nested = body.get("error")
            if isinstance(nested, dict):
                message = nested.get("message")
                if isinstance(message, str):
                    return message
        return str(error)

    @staticmethod
    def _is_missing_model_error(error: Exception) -> bool:
        if not isinstance(error, APIStatusError):
            return False
        if error.status_code != 404:
            return False
        return "model" in AnthropicClient._error_message(error).lower()

    @classmethod
    def _translate_api_error(cls, model_id: str, error: Exception) -> None:
        if not cls._is_missing_model_error(error):
            return

        message = (
            f"Anthropic model identifier {model_id!r} was not found. "
            f"Anthropic replied: {cls._error_message(error)}."
        )
        replacement = cls._RETIRED_MODEL_REPLACEMENTS.get(model_id)
        if replacement:
            message += (
                " This model has been retired by Anthropic."
                f" Use {replacement!r} instead."
            )
        else:
            message += (
                " Verify the model identifier against Anthropic's current "
                "models list."
            )
        raise ValueError(message) from error


class AnthropicModel(TextGenerationVendorModel):
    def _load_model(
        self,
    ) -> TextGenerationVendor | PreTrainedModel | DiffusionPipeline:
        assert self._settings.access_token
        return AnthropicClient(
            api_key=self._settings.access_token,
            base_url=self._settings.base_url,
            exit_stack=self._exit_stack,
        )
