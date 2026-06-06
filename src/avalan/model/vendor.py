from ..entities import (
    GenerationSettings,
    Message,
    MessageContent,
    MessageContentFile,
    MessageContentImage,
    MessageContentText,
    Token,
    TokenDetail,
    ToolCall,
    ToolCallToken,
    ToolValue,
)
from ..tool.manager import ToolManager
from .message import (
    TemplateMessage,
    TemplateMessageContent,
    TemplateMessageRole,
)
from .provider import ProviderFamily, provider_family_value
from .stream import TextGenerationStream

from abc import ABC
from base64 import urlsafe_b64decode, urlsafe_b64encode
from binascii import Error as BinasciiError
from json import JSONDecodeError, dumps, loads
from re import compile as compile_regex
from typing import Any, AsyncIterator, cast


class TextGenerationVendor(ABC):
    _PROVIDER_TOOL_NAME_PATTERN = compile_regex(r"^[A-Za-z0-9_-]+$")
    _PROVIDER_TOOL_NAME_PREFIX = "avl_"

    async def __call__(
        self,
        model_id: str,
        messages: list[Message],
        settings: GenerationSettings | None = None,
        *,
        instructions: str | None = None,
        tool: ToolManager | None = None,
        use_async_generator: bool = True,
    ) -> TextGenerationStream | AsyncIterator[Token | TokenDetail | str]:
        raise NotImplementedError()

    def _system_prompt(self, messages: list[Message]) -> str | None:
        for message in messages:
            if message.role != "system":
                continue
            content = message.content
            if isinstance(content, str):
                return content
            if isinstance(content, MessageContentText):
                return content.text
            return None
        return None

    def _template_messages(
        self,
        messages: list[Message],
        exclude_roles: list[TemplateMessageRole] | None = None,
    ) -> list[TemplateMessage] | list[dict[str, Any]]:
        def _block(c: MessageContent) -> dict[str, Any]:
            if isinstance(c, MessageContentFile):
                return {"type": "file", "file": dict(c.file)}
            if isinstance(c, MessageContentImage):
                return {"type": "image_url", "image_url": c.image_url}
            return {"type": "text", "text": c.text}

        def _wrap(
            content: str | MessageContent | list[MessageContent] | None,
        ) -> str | list[dict[str, Any]]:
            if isinstance(content, str):
                return content

            if isinstance(content, list):
                return [_block(c) for c in content]

            if isinstance(content, MessageContentText):
                return content.text

            if isinstance(content, MessageContentImage):
                return [_block(content)]

            if isinstance(content, MessageContentFile):
                return [_block(content)]

            return str(content)

        out: list[TemplateMessage] = []
        for msg in messages:
            if exclude_roles and msg.role in exclude_roles:
                continue

            out.append(
                {
                    "role": cast(TemplateMessageRole, str(msg.role)),
                    "content": cast(
                        str
                        | TemplateMessageContent
                        | list[TemplateMessageContent],
                        _wrap(msg.content),
                    ),
                }
            )

        return out

    @staticmethod
    def encode_tool_name(tool_name: str) -> str:
        assert isinstance(tool_name, str)
        assert tool_name.strip(), "tool name must not be empty"
        if TextGenerationVendor._PROVIDER_TOOL_NAME_PATTERN.fullmatch(
            tool_name
        ) and not tool_name.startswith(
            TextGenerationVendor._PROVIDER_TOOL_NAME_PREFIX
        ):
            return tool_name
        encoded = urlsafe_b64encode(tool_name.encode()).decode().rstrip("=")
        return f"{TextGenerationVendor._PROVIDER_TOOL_NAME_PREFIX}{encoded}"

    @staticmethod
    def decode_tool_name(tool_name: str) -> str:
        assert isinstance(tool_name, str)
        assert tool_name.strip(), "tool name must not be empty"
        assert TextGenerationVendor._PROVIDER_TOOL_NAME_PATTERN.fullmatch(
            tool_name
        ), "provider tool name is invalid"

        prefix = TextGenerationVendor._PROVIDER_TOOL_NAME_PREFIX
        if not tool_name.startswith(prefix):
            return tool_name

        payload = tool_name[len(prefix) :]
        assert payload, "provider tool name is missing encoded content"
        padding = "=" * (-len(payload) % 4)
        try:
            decoded = urlsafe_b64decode(f"{payload}{padding}").decode()
        except (BinasciiError, UnicodeDecodeError) as exc:
            raise AssertionError("provider tool name is malformed") from exc
        assert decoded.strip(), "decoded tool name must not be empty"
        assert (
            TextGenerationVendor.encode_tool_name(decoded) == tool_name
        ), "provider tool name is malformed"
        return decoded

    @staticmethod
    def build_tool_call_token(
        call_id: str | object | None,
        tool_name: str | object | None,
        arguments: str | dict[str, Any] | object | None,
    ) -> ToolCallToken:
        tool_name_text = (
            tool_name if isinstance(tool_name, str) else str(tool_name or "")
        )
        name = (
            TextGenerationVendor.decode_tool_name(tool_name_text)
            if tool_name_text
            else ""
        )
        if isinstance(arguments, str):
            try:
                args = cast(dict[str, Any], loads(arguments))
            except JSONDecodeError:
                args = {}
        else:
            args = (
                arguments
                if isinstance(arguments, dict)
                else cast(dict[str, Any], {})
            )
        call_id_value = (
            call_id
            if isinstance(call_id, str) or call_id is None
            else str(call_id)
        )
        call = ToolCall(
            id=cast(Any, call_id_value),
            name=name,
            arguments=cast(dict[str, ToolValue], args),
        )
        token_payload: dict[str, Any] = {
            "name": name,
            "arguments": args,
        }
        if call_id_value is not None:
            token_payload["id"] = call_id_value
        token_json = dumps(token_payload)
        return ToolCallToken(
            token=f"<tool_call>{token_json}</tool_call>",
            call=call,
            provider_name=tool_name_text or None,
        )


class TextGenerationVendorStream(TextGenerationStream):
    _generator: AsyncIterator[Token | TokenDetail | str]
    _provider_family: str | None
    _usage: object | None

    def __init__(
        self,
        generator: AsyncIterator[Token | TokenDetail | str],
        *,
        provider_family: ProviderFamily | str | None = None,
        usage: object | None = None,
    ) -> None:
        self._generator = generator
        self._provider_family = provider_family_value(provider_family)
        self._usage = usage

    @property
    def provider_family(self) -> str | None:
        return self._provider_family

    @property
    def usage(self) -> object | None:
        return self._usage

    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[Token | TokenDetail | str]:
        return self.__aiter__()

    def __aiter__(self) -> AsyncIterator[Token | TokenDetail | str]:
        assert self._generator
        return self

    async def __anext__(self) -> Token | TokenDetail | str:
        return await self._generator.__anext__()
