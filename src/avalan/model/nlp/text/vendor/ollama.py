from .....entities import (
    GenerationSettings,
    Message,
    Token,
    TokenDetail,
    TransformerEngineSettings,
)
from .....model.nlp.text.generation import TextGenerationModel
from .....tool.manager import ToolManager
from ....vendor import TextGenerationVendor, TextGenerationVendorStream
from . import TextGenerationVendorModel

from contextlib import AsyncExitStack
from dataclasses import replace
from logging import Logger, getLogger
from typing import Any, AsyncGenerator, AsyncIterator, cast

try:
    from ollama import AsyncClient
except ImportError:  # pragma: no cover - ollama may not be installed
    AsyncClient = None


class OllamaStream(TextGenerationVendorStream):
    def __init__(self, stream: AsyncIterator[dict[str, Any]]) -> None:
        super().__init__(
            cast(AsyncIterator[Token | TokenDetail | str], stream)
        )

    async def __anext__(self) -> Token | TokenDetail | str:
        chunk = await self._generator.__anext__()
        message: dict[str, Any]
        if isinstance(chunk, dict):
            chunk_message = chunk.get("message", {})
            message = chunk_message if isinstance(chunk_message, dict) else {}
        else:
            message = {}
        content = message.get("content", "")
        return content if isinstance(content, str) else str(content)


class OllamaClient(TextGenerationVendor):
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
        tool: ToolManager | None = None,
        use_async_generator: bool = True,
    ) -> AsyncIterator[Token | TokenDetail | str]:
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

            async def single_gen() -> (
                AsyncGenerator[Token | TokenDetail | str, None]
            ):
                content = response["message"]["content"]
                yield content if isinstance(content, str) else str(content)

            return single_gen()


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
