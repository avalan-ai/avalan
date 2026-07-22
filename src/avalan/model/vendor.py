from ..entities import (
    GenerationSettings,
    Message,
    MessageContent,
    MessageContentFile,
    MessageContentImage,
    MessageContentText,
)
from ..tool.name_policy import ToolNamePolicy
from .capability import CorrelatedCapabilityResult, ModelCapabilityCatalog
from .message import (
    TemplateMessage,
    TemplateMessageContent,
    TemplateMessageRole,
)
from .provider import ProviderFamily, provider_family_value
from .reasoning import (
    ReasoningSummaryRequestCapability,
    validate_reasoning_summary_request,
)
from .stream import (
    CanonicalStreamItem,
    StreamItemKind,
    StreamProviderCapabilities,
    StreamProviderEvent,
    StreamValidationError,
    TextGenerationNonStreamToolCall,
    TextGenerationStream,
    _close_async_iterable,
    normalize_provider_stream,
)

from abc import ABC
from collections.abc import AsyncIterable, Mapping
from dataclasses import replace
from inspect import isawaitable
from json import JSONDecodeError, dumps, loads
from typing import Any, AsyncIterator, Awaitable, Iterable, TypeVar, cast

_StreamItemT = TypeVar("_StreamItemT")
_UNSUPPORTED_REASONING_SUMMARY = ReasoningSummaryRequestCapability()


class TextGenerationVendor(ABC):
    _reasoning_summary_provider = "vendor"

    @property
    def reasoning_summary_request_capability(
        self,
    ) -> ReasoningSummaryRequestCapability:
        """Return the client's request-time summary capability."""
        return _UNSUPPORTED_REASONING_SUMMARY

    @property
    def reasoning_summary_provider(self) -> str:
        """Return the provider family used in capability errors."""
        return self._reasoning_summary_provider

    def _validate_reasoning_summary_request(
        self,
        settings: GenerationSettings | None,
    ) -> None:
        """Reject unsupported summary requests before provider dispatch."""
        if settings is not None:
            validate_reasoning_summary_request(self, settings)

    async def __call__(
        self,
        model_id: str,
        messages: list[Message],
        settings: GenerationSettings | None = None,
        *,
        instructions: str | None = None,
        capability: ModelCapabilityCatalog | None = None,
        use_async_generator: bool = True,
    ) -> TextGenerationStream | AsyncIterator[CanonicalStreamItem] | str:
        raise NotImplementedError()

    @staticmethod
    def capability_result_message(
        result: CorrelatedCapabilityResult,
    ) -> object:
        """Return one provider-native correlated continuation payload."""
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
        return ToolNamePolicy.encode_encoded(tool_name)

    @staticmethod
    def decode_tool_name(tool_name: str) -> str:
        return ToolNamePolicy.decode_encoded(tool_name)

    @staticmethod
    def provider_tool_name(
        tool_name: str,
        *,
        capability: ModelCapabilityCatalog | None = None,
        provider_family: ProviderFamily | str | None = None,
    ) -> str:
        if capability is not None:
            return capability.provider_name(
                tool_name,
                provider_family=provider_family_value(provider_family),
            )
        return TextGenerationVendor.encode_tool_name(tool_name)

    @staticmethod
    def canonical_tool_name(
        tool_name: str,
        *,
        capability: ModelCapabilityCatalog | None = None,
        provider_family: ProviderFamily | str | None = None,
    ) -> str:
        if capability is not None:
            return capability.canonical_name(
                tool_name,
                provider_family=provider_family_value(provider_family),
            )
        try:
            return TextGenerationVendor.decode_tool_name(tool_name)
        except AssertionError:
            if not tool_name.startswith("avl_"):
                return tool_name
            raise

    @staticmethod
    def non_stream_tool_call(
        *,
        call_id: object,
        provider_name: object,
        arguments: object,
        capability: ModelCapabilityCatalog | None,
        provider_family: ProviderFamily | str,
        provider_event_type: str,
    ) -> TextGenerationNonStreamToolCall:
        """Return one validated provider-native non-stream call."""
        if type(call_id) is not str or not call_id.strip():
            raise ValueError(
                "provider tool call id must be a non-empty string"
            )
        if type(provider_name) is not str or not provider_name.strip():
            raise ValueError(
                "provider tool call name must be a non-empty string"
            )
        if arguments is None:
            serialized_arguments = "{}"
        elif type(arguments) is str:
            serialized_arguments = arguments
        elif isinstance(arguments, Mapping):
            serialized_arguments = dumps(
                arguments,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        else:
            raise ValueError(
                "provider tool call arguments must be an object or string"
            )
        canonical_name = TextGenerationVendor.canonical_tool_name(
            provider_name,
            capability=capability,
            provider_family=provider_family,
        )
        return TextGenerationNonStreamToolCall(
            call_id=call_id,
            name=canonical_name,
            arguments=serialized_arguments,
            provider_event_type=provider_event_type,
        )

    @staticmethod
    def build_tool_call_text(
        call_id: str | object | None,
        tool_name: str | object | None,
        arguments: str | dict[str, Any] | object | None,
        *,
        tool_name_is_canonical: bool = False,
    ) -> str:
        tool_name_text = (
            tool_name if isinstance(tool_name, str) else str(tool_name or "")
        )
        provider_name_encoded = tool_name_text.startswith("avl_")
        if tool_name_text and not tool_name_is_canonical:
            try:
                name = TextGenerationVendor.decode_tool_name(tool_name_text)
            except AssertionError:
                assert provider_name_encoded
                name = tool_name_text
        elif tool_name_text:
            name = tool_name_text
        else:
            name = ""
        if isinstance(arguments, str):
            try:
                parsed_arguments = loads(arguments)
            except JSONDecodeError:
                args = {}
            else:
                if isinstance(parsed_arguments, dict):
                    args = parsed_arguments
                else:
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
        token_payload: dict[str, Any] = {
            "name": name,
            "arguments": args,
        }
        if call_id_value is not None:
            token_payload["id"] = call_id_value
        token_json = dumps(token_payload)
        return f"<tool_call>{token_json}</tool_call>"


class TextGenerationVendorStream(TextGenerationStream):
    _DEFAULT_STREAM_SESSION_ID = "vendor-stream"
    _DEFAULT_RUN_ID = "vendor-run"
    _DEFAULT_TURN_ID = "vendor-turn"

    _generator: AsyncIterator[CanonicalStreamItem]
    _provider_family: str | None
    _stream_sources: tuple[object, ...]
    _stream_closed: bool
    _stream_cancelled: bool
    _usage: object | None
    _direct_iterator: AsyncIterator[CanonicalStreamItem] | None

    def __init__(
        self,
        generator: AsyncIterator[CanonicalStreamItem],
        *,
        provider_family: ProviderFamily | str | None = None,
        sources: Iterable[object] = (),
        usage: object | None = None,
    ) -> None:
        self._generator = generator
        self._provider_family = provider_family_value(provider_family)
        self._stream_sources = tuple(sources)
        self._stream_closed = False
        self._stream_cancelled = False
        self._usage = usage
        self._direct_iterator = None

    @property
    def provider_family(self) -> str | None:
        return self._provider_family

    @property
    def usage(self) -> object | None:
        return self._usage

    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[CanonicalStreamItem]:
        return self.__aiter__()

    def __aiter__(self) -> AsyncIterator[CanonicalStreamItem]:
        assert self._generator
        return self.canonical_stream(
            stream_session_id=self._DEFAULT_STREAM_SESSION_ID,
            run_id=self._DEFAULT_RUN_ID,
            turn_id=self._DEFAULT_TURN_ID,
        )

    async def __anext__(self) -> CanonicalStreamItem:
        if self._direct_iterator is None:
            self._direct_iterator = self.__aiter__()
        return await self._direct_iterator.__anext__()

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
        provider_family_value = self._effective_provider_family(
            provider_family,
            capabilities,
        )
        return self._close_stream_on_exit(
            self._canonical_stream(
                stream_session_id=stream_session_id,
                run_id=run_id,
                turn_id=turn_id,
                provider_family=provider_family_value,
                capabilities=capabilities,
                close_after_terminal=close_after_terminal,
            )
        )

    async def _canonical_stream(
        self,
        *,
        stream_session_id: str,
        run_id: str,
        turn_id: str,
        provider_family: str | None,
        capabilities: StreamProviderCapabilities | None,
        close_after_terminal: bool,
    ) -> AsyncIterator[CanonicalStreamItem]:
        _ = stream_session_id
        _ = run_id
        _ = turn_id
        _ = close_after_terminal
        iterator = cast(AsyncIterator[Any], self._generator.__aiter__())
        try:
            first = await iterator.__anext__()
        except StopAsyncIteration:
            first = None

        if first is not None and not isinstance(first, CanonicalStreamItem):
            raise StreamValidationError(
                "unsupported legacy vendor stream item"
            )

        if first is None:
            return

        async for item in self._canonical_items_from_first(first, iterator):
            yield self._vendor_item(
                item,
                stream_session_id=stream_session_id,
                run_id=run_id,
                turn_id=turn_id,
                provider_family=provider_family,
                capabilities=capabilities,
            )

    async def _canonical_items_from_first(
        self,
        first: CanonicalStreamItem,
        iterator: AsyncIterator[Any],
    ) -> AsyncIterator[CanonicalStreamItem]:
        yield first
        async for item in iterator:
            if not isinstance(item, CanonicalStreamItem):
                raise StreamValidationError(
                    "unsupported legacy vendor stream item"
                )
            yield item

    def _provider_canonical_stream(
        self,
        events: AsyncIterable[StreamProviderEvent],
        *,
        stream_session_id: str,
        run_id: str,
        turn_id: str,
        provider_family: ProviderFamily | str | None = None,
        capabilities: StreamProviderCapabilities | None = None,
        close_after_terminal: bool = True,
    ) -> AsyncIterator[CanonicalStreamItem]:
        provider_family_value = self._effective_provider_family(
            provider_family,
            capabilities,
        )
        return self._close_stream_on_exit(
            normalize_provider_stream(
                events,
                stream_session_id=stream_session_id,
                run_id=run_id,
                turn_id=turn_id,
                provider_family=provider_family_value,
                capabilities=capabilities,
                close_after_terminal=close_after_terminal,
            )
        )

    def _vendor_item(
        self,
        item: CanonicalStreamItem,
        *,
        stream_session_id: str,
        run_id: str,
        turn_id: str,
        provider_family: str | None,
        capabilities: StreamProviderCapabilities | None,
    ) -> CanonicalStreamItem:
        assert isinstance(item, CanonicalStreamItem)
        usage = self._usage
        if (
            item.stream_session_id != stream_session_id
            or item.run_id != run_id
            or item.turn_id != turn_id
        ):
            item = replace(
                item,
                stream_session_id=stream_session_id,
                run_id=run_id,
                turn_id=turn_id,
            )
        if provider_family is not None and item.provider_family is None:
            item = replace(item, provider_family=provider_family)
        if (
            capabilities is not None
            and item.kind is StreamItemKind.STREAM_STARTED
            and "capabilities" not in item.metadata
        ):
            item = replace(
                item,
                metadata={
                    **item.metadata,
                    "capabilities": cast(
                        Any,
                        capabilities.to_metadata(),
                    ),
                },
            )
        if (
            usage is not None
            and item.kind is StreamItemKind.STREAM_COMPLETED
            and not item.usage
        ):
            item = replace(item, usage=cast(Any, usage))
        return item

    def _effective_provider_family(
        self,
        provider_family: ProviderFamily | str | None,
        capabilities: StreamProviderCapabilities | None,
    ) -> str | None:
        return (
            provider_family_value(provider_family)
            or self._provider_family
            or (
                capabilities.normalized_provider_family
                if capabilities is not None
                else None
            )
        )

    async def _close_stream_on_exit(
        self,
        items: AsyncIterator[_StreamItemT],
    ) -> AsyncIterator[_StreamItemT]:
        try:
            async for item in items:
                yield item
        finally:
            errors: list[BaseException] = []
            try:
                await _close_async_iterable(items)
            except BaseException as error:
                errors.append(error)
            try:
                await self.aclose()
            except BaseException as error:
                errors.append(error)
            if len(errors) == 1:
                raise errors[0]
            if errors:
                raise BaseExceptionGroup(
                    "vendor stream close failed",
                    errors,
                )

    async def cancel(self) -> None:
        if self._stream_cancelled:
            return
        self._stream_cancelled = True
        await self._call_stream_cleanup("cancel")

    async def aclose(self) -> None:
        if self._stream_closed:
            return
        self._stream_closed = True
        await self._call_stream_cleanup("aclose")

    async def _call_stream_cleanup(self, method_name: str) -> None:
        assert method_name in ("cancel", "aclose")
        errors: list[BaseException] = []
        for source in self._cleanup_sources():
            method = getattr(source, method_name, None)
            if method is None:
                continue
            assert callable(method)
            try:
                result = method()
                if isawaitable(result):
                    awaited_result = await cast(Awaitable[object], result)
                    assert awaited_result is None
                else:
                    assert result is None
            except BaseException as exc:
                errors.append(exc)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("vendor stream cleanup failed", errors)

    def _cleanup_sources(self) -> tuple[object, ...]:
        sources = (self._generator, *self._stream_sources)
        unique_sources: list[object] = []
        seen: set[int] = set()
        for source in sources:
            source_id = id(source)
            if source_id in seen:
                continue
            seen.add(source_id)
            unique_sources.append(source)
        return tuple(unique_sources)
