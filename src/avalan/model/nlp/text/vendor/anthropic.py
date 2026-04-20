from .....entities import (
    GenerationSettings,
    Message,
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
from ....message import TemplateMessage, TemplateMessageRole
from ....vendor import TextGenerationVendor, TextGenerationVendorStream
from . import TextGenerationVendorModel

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
            "temperature": settings.temperature,
            "tools": AnthropicClient._tool_schemas(tool) if tool else [],
            "tool_choice": {"type": "auto"},
        }

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
            content: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": result.call.id,
                "content": to_json(
                    result.result
                    if isinstance(result, ToolCallResult)
                    else result.message
                ),
            }
            if isinstance(result, ToolCallError):
                content["is_error"] = True
            result_message: dict[str, Any] = {
                "role": "user",
                "content": [content],
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
