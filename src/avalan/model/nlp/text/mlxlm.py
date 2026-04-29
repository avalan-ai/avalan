from ....entities import (
    GenerationSettings,
    Input,
    Token,
    TokenDetail,
    TransformerEngineSettings,
)
from ....model.response.text import TextGenerationResponse
from ....tool.manager import ToolManager
from ...vendor import TextGenerationVendorStream
from .generation import TextGenerationModel

from asyncio import to_thread
from collections.abc import Mapping
from dataclasses import asdict, replace
from functools import lru_cache
from importlib import import_module
from importlib.util import find_spec
from logging import Logger, getLogger
from subprocess import DEVNULL, TimeoutExpired, run
from sys import executable, modules
from typing import Any, AsyncGenerator, Callable, Iterator, Literal, cast

from torch import Tensor
from transformers.tokenization_utils_base import BatchEncoding

_MLX_IMPORT_CHECK_TIMEOUT_SECONDS = 10


def _mlx_unavailable_message() -> str:
    return (
        "The mlx-lm dependency is not installed or cannot be imported safely. "
        "Install or repair avalan[mlx] to enable the MLX backend."
    )


@lru_cache(maxsize=1)
def _mlx_lm_import_is_safe() -> bool:
    if modules.get("mlx_lm") is not None:
        return True
    try:
        if not find_spec("mlx_lm"):
            return False
    except (ImportError, ValueError):
        return False

    try:
        completed = run(
            [executable, "-c", "import mlx_lm"],
            check=False,
            stdout=DEVNULL,
            stderr=DEVNULL,
            timeout=_MLX_IMPORT_CHECK_TIMEOUT_SECONDS,
        )
    except (OSError, TimeoutExpired):
        return False

    return completed.returncode == 0


def _require_mlx_lm() -> Any:
    if not _mlx_lm_import_is_safe():
        raise ModuleNotFoundError(_mlx_unavailable_message())
    return import_module("mlx_lm")


def make_sampler(*args: Any, **kwargs: Any) -> Any:
    if not _mlx_lm_import_is_safe():
        raise ModuleNotFoundError(_mlx_unavailable_message())
    sample_utils = import_module("mlx_lm.sample_utils")
    make_sampler_fn = cast(
        Callable[..., Any], getattr(sample_utils, "make_sampler")
    )
    return make_sampler_fn(*args, **kwargs)


class MlxLmStream(TextGenerationVendorStream):
    """Async wrapper around a synchronous token generator."""

    _SENTINEL = object()

    def __init__(self, generator: Iterator[object]) -> None:
        async def _generator() -> (
            AsyncGenerator[Token | TokenDetail | str, None]
        ):
            for item in generator:
                if isinstance(item, (Token, TokenDetail, str)):
                    yield item
                    continue
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    yield text

        super().__init__(_generator())
        self._iterator = generator

    async def __anext__(self) -> str:
        sentinel = type(self)._SENTINEL
        chunk = await to_thread(next, self._iterator, sentinel)
        if chunk is sentinel:
            raise StopAsyncIteration
        if isinstance(chunk, str):
            return chunk
        if isinstance(chunk, (Token, TokenDetail)):
            return chunk.token
        text = getattr(chunk, "text", None)
        return text if isinstance(text, str) else ""


class MlxLmModel(TextGenerationModel):
    @classmethod
    def is_available(cls) -> bool:
        return _mlx_lm_import_is_safe()

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

    def _load_model(self) -> Any:
        assert self._model_id, "A model id is required."
        mlx_lm = _require_mlx_lm()
        load_fn = cast(
            Callable[[str], tuple[Any, Any]], getattr(mlx_lm, "load")
        )
        model, tokenizer = load_fn(self._model_id)
        self._tokenizer = tokenizer
        self._loaded_tokenizer = True
        return model

    async def _stream_generator(
        self,
        inputs: dict[str, Tensor] | BatchEncoding | Tensor,
        settings: GenerationSettings,
        skip_special_tokens: bool,
    ) -> AsyncGenerator[str, None]:
        sampler, prompt = self._get_sampler_and_prompt(
            inputs, settings, skip_special_tokens
        )
        mlx_lm = _require_mlx_lm()
        stream_generate_fn = cast(
            Callable[..., Iterator[object]], getattr(mlx_lm, "stream_generate")
        )
        iterator = stream_generate_fn(
            self._model,
            self._tokenizer,
            prompt,
            sampler=sampler,
            max_tokens=settings.max_new_tokens,
        )
        stream = MlxLmStream(iter(iterator))
        async for chunk in stream:
            if isinstance(chunk, str):
                yield chunk

    def _string_output(
        self,
        inputs: dict[str, Tensor] | BatchEncoding | Tensor,
        settings: GenerationSettings,
        skip_special_tokens: bool,
    ) -> str:
        sampler, prompt = self._get_sampler_and_prompt(
            inputs, settings, skip_special_tokens
        )
        mlx_lm = _require_mlx_lm()
        generate_fn = cast(Callable[..., str], getattr(mlx_lm, "generate"))
        return generate_fn(
            self._model,
            self._tokenizer,
            prompt,
            sampler=sampler,
            max_tokens=settings.max_new_tokens,
        )

    async def __call__(
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
        inputs: dict[str, Tensor] | BatchEncoding | Tensor,
        settings: GenerationSettings,
        skip_special_tokens: bool,
    ) -> tuple[Callable[..., Any], str]:
        sampler = make_sampler(
            temp=(
                settings.temperature
                if settings.temperature is not None
                else 0.0
            ),
            top_p=settings.top_p if settings.top_p is not None else 1.0,
            top_k=settings.top_k if settings.top_k is not None else 0,
        )
        input_ids = self._input_ids_from_inputs(inputs)
        prompt = self._tokenizer.decode(
            self._first_prompt_sequence(input_ids),
            skip_special_tokens=skip_special_tokens,
        )
        return sampler, prompt

    @staticmethod
    def _input_ids_from_inputs(
        inputs: dict[str, Tensor] | BatchEncoding | Tensor,
    ) -> Any:
        if isinstance(inputs, Mapping):
            input_ids = inputs.get("input_ids")
            if input_ids is None:
                raise ValueError(
                    "Expected tokenized inputs to include input_ids."
                )
            return input_ids
        if isinstance(inputs, Tensor):
            return inputs
        raise ValueError(
            "Expected tokenized inputs to be a mapping or tensor."
        )

    @staticmethod
    def _first_prompt_sequence(input_ids: Any) -> Any:
        shape = getattr(input_ids, "shape", None)
        if shape is not None:
            try:
                if len(shape) > 1:
                    return input_ids[0]
                return input_ids
            except (IndexError, TypeError):
                return input_ids
        if isinstance(input_ids, (list, tuple)):
            if input_ids and isinstance(input_ids[0], (list, tuple)):
                return input_ids[0]
            return input_ids
        try:
            return input_ids[0]
        except (IndexError, KeyError, TypeError):
            return input_ids
