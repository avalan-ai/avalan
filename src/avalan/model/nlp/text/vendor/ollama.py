from .....compat import override
from .....entities import (
    GenerationSettings,
    Message,
    Token,
    TokenDetail,
    TransformerEngineSettings,
)
from .....model import TextGenerationVendor, TextGenerationVendorStream
from .....model.nlp.text.vendor import TextGenerationVendorModel
from .....model.nlp.text.generation import TextGenerationModel
from .....tool.manager import ToolManager
from dataclasses import replace
from logging import Logger
from typing import AsyncIterator

try:
    from ollama import AsyncClient
except Exception:  # pragma: no cover - ollama may not be installed
    AsyncClient = None


class OllamaStream(TextGenerationVendorStream):
    def __init__(self, stream: AsyncIterator[dict]):
        super().__init__(stream)

    async def __anext__(self) -> Token | TokenDetail | str:
        chunk = await self._generator.__anext__()
        message = chunk.get("message", {}) if isinstance(chunk, dict) else {}
        return message.get("content", "")


class OllamaClient(TextGenerationVendor):
    _client: AsyncClient

    def __init__(self, base_url: str | None = None):
        assert AsyncClient, "ollama is not available"
        self._client = AsyncClient(host=base_url) if base_url else AsyncClient()

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

            async def single_gen():
                yield response["message"]["content"]

            return single_gen()


class OllamaModel(TextGenerationVendorModel):
    def __init__(
        self,
        model_id: str,
        settings: TransformerEngineSettings | None = None,
        logger: Logger | None = None,
    ) -> None:
        settings = settings or TransformerEngineSettings()
        settings = replace(settings, enable_eval=False)
        TextGenerationModel.__init__(self, model_id, settings, logger)

    def _load_model(self):
        return OllamaClient(base_url=self._settings.base_url)
