from ...vendor import TextGenerationVendorStream
from ....model.response.text import TextGenerationResponse
from .generation import TextGenerationModel
from ....compat import override
from ....entities import (
    GenerationSettings,
    Input,
    TransformerEngineSettings,
)
from ....tool.manager import ToolManager
from asyncio import to_thread
from dataclasses import asdict, replace
from logging import Logger, getLogger
from mlx_lm import generate, load, stream_generate
from mlx_lm.sample_utils import make_sampler
from torch import Tensor
from typing import AsyncGenerator, Callable, Literal


class MlxLmStream(TextGenerationVendorStream):
    """Async wrapper around a synchronous token generator."""

    _SENTINEL = object()

    def __init__(self, generator):
        super().__init__(generator)
        self._iterator = generator

    async def __anext__(self) -> str:
        sentinel = type(self)._SENTINEL
        chunk = await to_thread(next, self._iterator, sentinel)
        if chunk is sentinel:
            raise StopAsyncIteration
        return chunk


class MlxLmModel(TextGenerationModel):
    def __init__(
        self,
        model_id: str,
        settings: TransformerEngineSettings | None = None,
        logger: Logger = getLogger(__name__),
    ) -> None:
        settings = settings or TransformerEngineSettings()
        if settings.auto_load_tokenizer:
            settings = replace(settings, auto_load_tokenizer=False)
        super().__init__(model_id, settings, logger)

    @property
    def supports_sample_generation(self) -> bool:
        return False

    def _load_model(self):
        model, tokenizer = load(self._model_id)
        self._tokenizer = tokenizer
        self._loaded_tokenizer = True
        return model

    async def _stream_generator(
        self,
        inputs: dict[str, Tensor] | Tensor,
        settings: GenerationSettings,
        skip_special_tokens: bool,
    ) -> AsyncGenerator[str, None]:
        sampler, prompt = self._get_sampler_and_prompt(
            inputs, settings, skip_special_tokens
        )
        iterator = stream_generate(
            self._model,
            self._tokenizer,
            prompt,
            sampler=sampler,
            max_tokens=settings.max_new_tokens,
        )
        stream = MlxLmStream(iter(iterator))
        async for chunk in stream:
            yield chunk.text

    def _string_output(
        self,
        inputs: dict[str, Tensor] | Tensor,
        settings: GenerationSettings,
        skip_special_tokens: bool,
    ) -> str:
        sampler, prompt = self._get_sampler_and_prompt(
            inputs, settings, skip_special_tokens
        )
        return generate(
            self._model,
            self._tokenizer,
            prompt,
            sampler=sampler,
            max_tokens=settings.max_new_tokens,
        )

    @override
    async def __call__(
        self,
        input: Input,
        system_prompt: str | None = None,
        settings: GenerationSettings | None = None,
        *,
        skip_special_tokens: bool = False,
        tensor_format: Literal["pt"] = "pt",
        tool: ToolManager | None = None,
    ) -> TextGenerationResponse:
        settings = settings or GenerationSettings()
        inputs = super()._tokenize_input(
            input,
            system_prompt,
            context=None,
            tensor_format=tensor_format,
            tool=tool,
            chat_template_settings=asdict(settings.chat_settings),
        )
        generation_settings = replace(settings, do_sample=False)
        output_fn = (
            self._stream_generator
            if settings.use_async_generator
            else self._string_output
        )

        return TextGenerationResponse(
            output_fn,
            inputs=inputs,
            logger=self._logger,
            generation_settings=generation_settings,
            settings=generation_settings,
            skip_special_tokens=skip_special_tokens,
            use_async_generator=settings.use_async_generator,
            bos_token=self._tokenizer.bos_token,
        )

    def _get_sampler_and_prompt(
        self,
        inputs: dict[str, Tensor] | Tensor,
        settings: GenerationSettings,
        skip_special_tokens: bool,
    ) -> tuple[Callable, str]:
        sampler_settings = {
            "temp": settings.temperature,
            "top_p": settings.top_p,
            "top_k": settings.top_k,
        }
        sampler_settings = {
            k: v for k, v in sampler_settings.items() if v is not None
        }
        sampler = make_sampler(**sampler_settings)
        prompt = self._tokenizer.decode(
            inputs["input_ids"][0],
            skip_special_tokens=skip_special_tokens,
        )
        return sampler, prompt
