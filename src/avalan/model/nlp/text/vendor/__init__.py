from .....compat import override as override
from .....entities import (
    GenerationSettings,
    Input,
    TransformerEngineSettings,
)
from .....model.nlp.text.generation import TextGenerationModel
from .....model.response.text import TextGenerationResponse
from .....model.vendor import TextGenerationVendor
from .....tool.manager import ToolManager

from abc import ABC, abstractmethod
from contextlib import AsyncExitStack
from dataclasses import replace
from importlib import import_module
from logging import Logger, getLogger
from typing import Any, Literal, Protocol, cast

from diffusers import DiffusionPipeline
from torch import Tensor
from transformers import PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding


class _TiktokenEncoding(Protocol):
    def encode(self, text: str) -> list[int]:
        """Encode the provided text into token IDs."""


class TextGenerationVendorModel(TextGenerationModel, ABC):
    _TIKTOKEN_DEFAULT_MODEL = "cl100k_base"

    def __init__(
        self,
        model_id: str,
        settings: TransformerEngineSettings | None = None,
        logger: Logger = getLogger(__name__),
        *,
        exit_stack: AsyncExitStack | None = None,
    ) -> None:
        settings = settings or TransformerEngineSettings()
        assert (
            settings.base_url or settings.access_token
        ), "API key needed for vendor"
        settings = replace(settings, enable_eval=False)
        super().__init__(model_id, settings, logger)
        self._exit_stack = exit_stack or AsyncExitStack()

    @abstractmethod
    def _load_model(
        self,
    ) -> PreTrainedModel | TextGenerationVendor | DiffusionPipeline:
        raise NotImplementedError()

    @property
    def supports_sample_generation(self) -> bool:
        return False

    @property
    def supports_token_streaming(self) -> bool:
        return True

    @property
    def uses_tokenizer(self) -> bool:
        return False

    def _tokenize_input(
        self,
        input: Input,
        context: str | None = None,
        tensor_format: Literal["pt"] = "pt",
        **kwargs: object,
    ) -> dict[str, Tensor] | BatchEncoding | Tensor:
        raise NotImplementedError()

    @staticmethod
    def _resolve_encoding(
        model_id: str, default_model: str
    ) -> _TiktokenEncoding:
        tiktoken = import_module("tiktoken")
        encoding_for_model = cast(Any, getattr(tiktoken, "encoding_for_model"))
        get_encoding = cast(Any, getattr(tiktoken, "get_encoding"))
        try:
            return cast(_TiktokenEncoding, encoding_for_model(model_id))
        except KeyError:
            return cast(_TiktokenEncoding, get_encoding(default_model))

    def input_token_count(
        self,
        input: Input,
        system_prompt: str | None = None,
        developer_prompt: str | None = None,
    ) -> int:
        assert self._model_id
        encoding = self._resolve_encoding(
            self._model_id, self._TIKTOKEN_DEFAULT_MODEL
        )

        messages = self._messages(
            input, system_prompt, developer_prompt, tool=None
        )

        total_tokens = 0
        for message in messages:
            if isinstance(message.content, str):
                content = message.content
            else:
                content = str(message.content or "")
            total_tokens += len(encoding.encode(content))
        return total_tokens

    async def __call__(
        self,
        input: Input,
        system_prompt: str | None = None,
        developer_prompt: str | None = None,
        settings: GenerationSettings | None = None,
        *,
        tool: ToolManager | None = None,
    ) -> TextGenerationResponse:
        messages = self._messages(input, system_prompt, developer_prompt, tool)
        gen_settings = settings or GenerationSettings()
        streamer = await self._model(
            self._model_id,
            messages,
            gen_settings,
            tool=tool,
            use_async_generator=gen_settings.use_async_generator,
        )
        return TextGenerationResponse(
            streamer,
            logger=self._logger,
            generation_settings=gen_settings,
            settings=gen_settings,
            use_async_generator=gen_settings.use_async_generator,
        )
