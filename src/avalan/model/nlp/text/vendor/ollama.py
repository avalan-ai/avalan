from .....entities import (
    GenerationSettings,
    Message,
    TransformerEngineSettings,
)
from .....model.nlp.text.generation import TextGenerationModel
from .....model.provider import ProviderFamily
from .....model.stream import (
    CanonicalStreamItem,
    StreamItemKind,
    StreamProducerBackend,
    StreamProviderCapabilities,
    StreamProviderEvent,
    TextGenerationSingleStream,
    TextGenerationStream,
)
from .....tool.manager import ToolManager
from .....types import LooseJsonValue
from ....vendor import TextGenerationVendor, TextGenerationVendorStream
from . import TextGenerationVendorModel

from contextlib import AsyncExitStack
from dataclasses import replace
from logging import Logger, getLogger
from typing import Any, AsyncIterator, cast

try:
    from ollama import AsyncClient
except ImportError:  # pragma: no cover - ollama may not be installed
    AsyncClient = None


class OllamaStream(TextGenerationVendorStream):
    _stream: AsyncIterator[Any]

    def __init__(self, stream: AsyncIterator[Any]) -> None:
        self._stream = stream

        async def generator() -> AsyncIterator[CanonicalStreamItem]:
            async for item in self.canonical_stream(
                stream_session_id=self._DEFAULT_STREAM_SESSION_ID,
                run_id=self._DEFAULT_RUN_ID,
                turn_id=self._DEFAULT_TURN_ID,
            ):
                yield item

        super().__init__(
            generator(),
            provider_family=ProviderFamily.OLLAMA,
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
        return self._provider_canonical_stream(
            self._provider_events(),
            stream_session_id=stream_session_id,
            run_id=run_id,
            turn_id=turn_id,
            provider_family=provider_family,
            capabilities=capabilities
            or StreamProviderCapabilities(
                backend=StreamProducerBackend.HOSTED,
                provider_family=ProviderFamily.OLLAMA,
                supports_usage=True,
                supports_cancellation=True,
            ),
            close_after_terminal=close_after_terminal,
        )

    async def _provider_events(self) -> AsyncIterator[StreamProviderEvent]:
        terminal_usage: LooseJsonValue | None = None
        terminal_usage_payload: LooseJsonValue | None = None
        async for chunk in self._stream:
            provider_payload = dict(chunk) if isinstance(chunk, dict) else None
            usage = self._usage_from_chunk(chunk)
            if usage is not None:
                terminal_usage = usage
                terminal_usage_payload = provider_payload
            content = self._content_from_chunk(chunk)
            yield StreamProviderEvent(
                kind=StreamItemKind.ANSWER_DELTA,
                text_delta=content,
                provider_payload=provider_payload,
                provider_event_type="chat.message.delta",
            )
        if terminal_usage is not None:
            self._usage = terminal_usage
            yield StreamProviderEvent(
                kind=StreamItemKind.USAGE_COMPLETED,
                usage=terminal_usage,
                provider_payload=terminal_usage_payload,
                provider_event_type="chat.usage",
            )

    @staticmethod
    def _content_from_chunk(chunk: object) -> str:
        message = chunk.get("message", {}) if isinstance(chunk, dict) else {}
        message = message if isinstance(message, dict) else {}
        content = message.get("content", "")
        return content if isinstance(content, str) else str(content)

    @staticmethod
    def _usage_from_chunk(chunk: object) -> LooseJsonValue | None:
        if not isinstance(chunk, dict):
            return None
        usage = {
            key: value
            for key, value in chunk.items()
            if key.endswith("_count") and isinstance(value, int)
        }
        return cast(LooseJsonValue, usage) if usage else None


class OllamaClient(TextGenerationVendor):
    _reasoning_summary_provider = "ollama"
    _client: Any

    def __init__(self, base_url: str | None = None):
        assert AsyncClient, "ollama is not available"
        self._client = (
            AsyncClient(host=base_url) if base_url else AsyncClient()
        )

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
        ), "Ollama does not support provider instructions"
        template_messages = self._template_messages(messages)
        if use_async_generator:
            stream = await self._client.chat(
                model=model_id,
                messages=template_messages,
                stream=True,
            )
            return OllamaStream(cast(AsyncIterator[dict[str, Any]], stream))
        else:
            response = await self._client.chat(
                model=model_id,
                messages=template_messages,
                stream=False,
            )
            content = response["message"]["content"]
            return TextGenerationSingleStream(
                content if isinstance(content, str) else str(content),
                provider_family=ProviderFamily.OLLAMA,
            )


class OllamaModel(TextGenerationVendorModel):
    def __init__(
        self,
        model_id: str,
        settings: TransformerEngineSettings | None = None,
        logger: Logger = getLogger(__name__),
        exit_stack: AsyncExitStack | None = None,
    ) -> None:
        _ = exit_stack
        settings = settings or TransformerEngineSettings()
        settings = replace(settings, enable_eval=False)
        TextGenerationModel.__init__(self, model_id, settings, logger)

    def _load_model(self) -> TextGenerationVendor:
        return OllamaClient(base_url=self._settings.base_url)
