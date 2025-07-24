from ...vendor import TextGenerationVendorStream
from .generation import TextGenerationModel
from ....compat import override
from ....entities import (
    GenerationSettings,
    Input,
    TransformerEngineSettings,
)
from ....tool.manager import ToolManager
from asyncio import to_thread
from dataclasses import replace
from logging import Logger
from mlx_lm import generate, load, stream_generate
from mlx_lm.sample_utils import make_sampler
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
        logger: Logger | None = None,
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
        prompt: str,
        settings: GenerationSettings,
    ) -> AsyncGenerator[str, None]:
        sampler = MlxLmModel._build_sampler(settings)
        iterator = stream_generate(
            self._model,
            self._tokenizer,
            prompt,
            sampler=sampler,
            max_tokens=settings.max_new_tokens
        )
        stream = MlxLmStream(iter(iterator))
        async for chunk in stream:
            yield chunk.text

    def _string_output(
        self,
        prompt: str,
        settings: GenerationSettings,
    ) -> str:
        sampler = MlxLmModel._build_sampler(settings)
        return generate(
            self._model,
            self._tokenizer,
            prompt,
            sampler=sampler,
            max_tokens=settings.max_new_tokens
        )

    @override
    async def __call__(
        self,
        input: Input,
        system_prompt: str | None = None,
        settings: GenerationSettings | None = None,
        *,
        tensor_format: Literal["pt"] = "pt",
        tool: ToolManager | None = None,
    ) -> TextGenerationVendorStream | str:
        settings = settings or GenerationSettings()
        inputs = super()._tokenize_input(
            input,
            system_prompt,
            context=None,
            tensor_format=tensor_format,
            tool=tool,
            chat_template_settings=settings.chat_template_settings,
        )
        prompt = self._tokenizer.decode(
            inputs["input_ids"][0],
            skip_special_tokens=False,
        )
        generation_settings = replace(settings, do_sample=False)
        if settings.use_async_generator:
            return self._stream_generator(prompt, generation_settings)
        return self._string_output(prompt, generation_settings)

    @staticmethod
    def _build_sampler(settings: GenerationSettings) -> Callable:
        sampler_settings = {
            "temp": settings.temperature,
            "top_p": settings.top_p,
            "top_k": settings.top_k,
        }
        return make_sampler(**sampler_settings)
