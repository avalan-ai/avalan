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
from anthropic import AsyncAnthropic
from anthropic.types import RawContentBlockDeltaEvent, RawMessageStopEvent
from contextlib import AsyncExitStack
from dataclasses import asdict, is_dataclass
from diffusers import DiffusionPipeline
from json import dumps
from transformers import PreTrainedModel
from typing import AsyncIterator


class AnthropicStream(TextGenerationVendorStream):
    def __init__(self, events: AsyncIterator):
        async def generator() -> AsyncIterator[Token | TokenDetail | str]:
            tool_blocks: dict[int, dict] = {}

            async for event in events:
                etype = getattr(event, "type", None)

                if etype == "content_block_start":
                    cb = getattr(event, "content_block", None)
                    if (
                        cb is not None
                        and getattr(cb, "type", None) == "tool_use"
                    ):
                        tool_blocks[event.index] = {
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
                        tb = tool_blocks.setdefault(
                            event.index,
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
                            tool_call = ToolCall(
                                id=tool_id,
                                name=TextGenerationVendor.decode_tool_name(
                                    tool_name
                                ),
                                arguments=getattr(cb, "input", None),
                            )

                            token = ToolCallToken(
                                token="<tool_use>", call=tool_call
                            )
                            yield token

                        if event.index in tool_blocks:
                            del tool_blocks[event.index]

                    continue

                if (
                    isinstance(event, RawMessageStopEvent)
                    or etype == "message_stop"
                ):
                    break

        super().__init__(generator())

    async def __anext__(self) -> Token | TokenDetail | str:
        return await self._generator.__anext__()


class AnthropicClient(TextGenerationVendor):
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

    @override
    async def __call__(
        self,
        model_id: str,
        messages: list[Message],
        settings: GenerationSettings | None = None,
        *,
        tool: ToolManager | None = None,
        use_async_generator: bool = True,
    ) -> AsyncIterator[Token | TokenDetail | str]:
        settings = settings or GenerationSettings()
        system_prompt = self._system_prompt(messages)
        template_messages = self._template_messages(messages, ["system"])
        stream = self._client.messages.stream(
            model=model_id,
            system=system_prompt,
            messages=template_messages,
            max_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            tools=AnthropicClient._tool_schemas(tool) if tool else None,
            tool_choice={"type": "auto"},
        )
        events = await self._exit_stack.enter_async_context(stream)
        return AnthropicStream(events=events)

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
        last_message = next(
            (
                m
                for m in reversed(messages)
                if m["role"] == str(MessageRole.ASSISTANT)
            ),
            None,
        )
        last_message_index = (
            messages.index(last_message) if last_message else None
        )
        if last_message_index:
            messages[last_message_index] = {
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
                    ],
                ],
            }

        for result in tool_results:
            result_message = TemplateMessage(
                role="user",
                content=[
                    {
                        "type": "tool_result",
                        "tool_use_id": result.call.id,
                        "content": dumps(
                            asdict(result.result)
                            if is_dataclass(result.result)
                            else result.result
                        ),
                    }
                ],
            )
            if last_message_index:
                messages.insert(last_message_index + 1, result_message)
            else:
                messages.append(result_message)

        # @TODO Ensure this doesn't happen from upstream
        if len(messages) > 1 and (messages[0] == messages[-1]):
            messages.pop()

        return messages

    @staticmethod
    def _tool_schemas(tool: ToolManager) -> list[dict] | None:
        schemas = tool.json_schemas()
        return [
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
            for t in tool.json_schemas()
            if t["type"] == "function"
        ] if schemas else None


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
