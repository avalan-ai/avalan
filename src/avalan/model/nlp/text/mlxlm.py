from ....compat import override
from ....entities import (
    GenerationSettings,
    Input,
    TransformerEngineSettings,
)
from ....model.response.text import TextGenerationResponse
from ....model.vendor import TextGenerationVendor
from ....tool.manager import ToolManager
from ...vendor import TextGenerationVendorStream
from .generation import TextGenerationModel

from asyncio import to_thread
from dataclasses import asdict, replace
from logging import Logger, getLogger
from typing import Any, AsyncGenerator, Callable, Literal

from diffusers import DiffusionPipeline
from mlx_lm import generate, load, stream_generate
from mlx_lm.sample_utils import make_sampler
from torch import Tensor
from transformers import PreTrainedModel


class MlxLmStream(TextGenerationVendorStream):
    """Async wrapper around a synchronous token generator."""

    _SENTINEL: object = object()
    _iterator: Any

    def __init__(self, generator: Any) -> None:
        super().__init__(generator)
        self._iterator = generator

    async def __anext__(self) -> Any:
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

    def _load_model(
        self,
    ) -> PreTrainedModel | TextGenerationVendor | DiffusionPipeline:
        assert self._model_id is not None
        model, tokenizer = load(self._model_id)
        self._tokenizer = tokenizer  # type: ignore[assignment]
        self._loaded_tokenizer = True
        return model  # type: ignore[return-value]

    async def _stream_generator(  # type: ignore[override]
        self,
        inputs: dict[str, Tensor] | Tensor,
        settings: GenerationSettings,
        skip_special_tokens: bool,
    ) -> AsyncGenerator[str, None]:
        sampler, prompt = self._get_sampler_and_prompt(
            inputs, settings, skip_special_tokens
        )
        iterator = stream_generate(
            self._model,  # type: ignore[arg-type]
            self._tokenizer,  # type: ignore[arg-type]
            prompt,
            sampler=sampler,
            max_tokens=settings.max_new_tokens,
        )
        stream = MlxLmStream(iter(iterator))
        async for chunk in stream:
            yield chunk.text

    def _string_output(  # type: ignore[override]
        self,
        inputs: dict[str, Tensor] | Tensor,
        settings: GenerationSettings,
        skip_special_tokens: bool,
    ) -> str:
        sampler, prompt = self._get_sampler_and_prompt(
            inputs, settings, skip_special_tokens
        )
        return generate(
            self._model,  # type: ignore[arg-type]
            self._tokenizer,  # type: ignore[arg-type]
            prompt,
            sampler=sampler,
            max_tokens=settings.max_new_tokens,
        )

    @override
    async def __call__(  # type: ignore[override]
        self,
        input: Input,
        system_prompt: str | None = None,
        developer_prompt: str | None = None,
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
            developer_prompt,
            context=None,
            tensor_format=tensor_format,
            tool=tool,
            chat_template_settings=asdict(settings.chat_settings),
        )
        generation_settings = replace(settings, do_sample=False)
        output_fn: Callable[..., Any] = (
            self._stream_generator
            if settings.use_async_generator
            else self._string_output
        )
        assert self._tokenizer is not None

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
    ) -> tuple[Callable[..., Any], str]:
        sampler_settings: dict[str, Any] = {
            "temp": settings.temperature,
            "top_p": settings.top_p,
            "top_k": settings.top_k,
        }
        sampler_settings = {
            k: v for k, v in sampler_settings.items() if v is not None
        }
        sampler = make_sampler(**sampler_settings)
        assert self._tokenizer is not None
        assert isinstance(inputs, dict), "inputs must be a dict"
        prompt: str = self._tokenizer.decode(
            inputs["input_ids"][0],
            skip_special_tokens=skip_special_tokens,
        )
        return sampler, prompt
