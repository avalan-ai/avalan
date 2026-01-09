from .....compat import override
from .....entities import (
    GenerationSettings,
    Message,
    Token,
    TokenDetail,
    TransformerEngineSettings,
)
from .....tool.manager import ToolManager
from ....vendor import TextGenerationVendor, TextGenerationVendorStream
from . import TextGenerationVendorModel

from contextlib import AsyncExitStack
from dataclasses import replace
from logging import Logger, getLogger
from typing import AsyncIterator

try:
    from ollama import AsyncClient  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - ollama may not be installed
    AsyncClient = None  # type: ignore[misc, assignment]


class OllamaStream(TextGenerationVendorStream):
    def __init__(self, stream: AsyncIterator[dict]) -> None:  # type: ignore[type-arg]
        super().__init__(stream)  # type: ignore[arg-type]

    async def __anext__(self) -> Token | TokenDetail | str:
        chunk = await self._generator.__anext__()
        message = chunk.get("message", {}) if isinstance(chunk, dict) else {}
        content: str = message.get("content", "")
        return content


class OllamaClient(TextGenerationVendor):
    _client: AsyncClient

    def __init__(self, base_url: str | None = None):
        assert AsyncClient, "ollama is not available"
        self._client = (
            AsyncClient(host=base_url) if base_url else AsyncClient()
        )

    @override
    async def __call__(  # type: ignore[override]
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
            return OllamaStream(stream)
        else:
            response = await self._client.chat(
                model=model_id,
                messages=template_messages,
                stream=False,
            )

            async def single_gen() -> AsyncIterator[Token | TokenDetail | str]:
                yield response["message"]["content"]

            return single_gen()


class OllamaModel(TextGenerationVendorModel):
    def __init__(
        self,
        model_id: str,
        settings: TransformerEngineSettings | None = None,
        logger: Logger = getLogger(__name__),
        *,
        exit_stack: AsyncExitStack | None = None,
    ) -> None:
        settings = settings or TransformerEngineSettings()
        settings = replace(settings, enable_eval=False)
        super().__init__(model_id, settings, logger, exit_stack=exit_stack)

    def _load_model(self):
        return OllamaClient(base_url=self._settings.base_url)
