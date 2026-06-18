from ....entities import (
    GenerationSettings,
    Input,
    Token,
    TransformerEngineSettings,
)
from ....model.response.text import TextGenerationResponse
from ....model.stream import (
    CanonicalStreamItem,
    StreamItemKind,
    StreamProducerBackend,
    StreamProviderCapabilities,
    StreamProviderEvent,
    StreamValidationError,
)
from ....tool.manager import ToolManager
from ....types import LooseJsonValue
from ...vendor import TextGenerationVendorStream
from .generation import TextGenerationModel

from asyncio import get_running_loop
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from functools import lru_cache
from importlib import import_module
from importlib.util import find_spec
from logging import Logger, getLogger
from subprocess import DEVNULL, TimeoutExpired, run
from sys import executable, modules
from typing import Any, AsyncGenerator, AsyncIterator, Literal, cast

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
    """Bridge synchronous MLX generation into an async stream."""

    _SENTINEL = object()

    def __init__(
        self,
        generator: Iterator[object] | Callable[[], Iterator[object]],
        *,
        use_executor: bool = True,
    ) -> None:
        self._closed = False
        self._executor = (
            ThreadPoolExecutor(max_workers=1) if use_executor else None
        )
        if isinstance(generator, Iterator):
            self._iterator: Iterator[object] | None = generator
            self._iterator_factory: Callable[[], Iterator[object]] | None = (
                None
            )
        else:
            self._iterator = None
            self._iterator_factory = generator

        async def _generator() -> AsyncGenerator[CanonicalStreamItem, None]:
            if False:
                yield cast(CanonicalStreamItem, None)

        super().__init__(_generator())

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        """Mark the stream as closed."""
        if self._closed:
            return
        self._closed = True
        if self._executor:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def _next_chunk(self) -> object:
        if self._iterator is None:
            iterator_factory = self._iterator_factory
            assert iterator_factory is not None
            self._iterator = iter(iterator_factory())
        return next(self._iterator, self._SENTINEL)

    async def _next_raw(self) -> object:
        if self._closed:
            return self._SENTINEL

        try:
            if self._executor:
                loop = get_running_loop()
                chunk = await loop.run_in_executor(
                    self._executor, self._next_chunk
                )
            else:
                chunk = self._next_chunk()
        except Exception:
            self.close()
            raise

        if isinstance(chunk, BaseException):
            self.close()
            raise chunk
        if chunk is self._SENTINEL:
            self.close()
        return chunk

    async def __anext__(self) -> CanonicalStreamItem:
        return await super().__anext__()

    async def cancel(self) -> None:
        try:
            await super().cancel()
        finally:
            self.close()

    async def aclose(self) -> None:
        try:
            await super().aclose()
        finally:
            self.close()

    def canonical_stream(
        self,
        *,
        stream_session_id: str,
        run_id: str,
        turn_id: str,
        provider_family: str | None = None,
        capabilities: StreamProviderCapabilities | None = None,
        close_after_terminal: bool = True,
    ) -> AsyncIterator[CanonicalStreamItem]:
        return self._provider_canonical_stream(
            self._provider_events(),
            stream_session_id=stream_session_id,
            run_id=run_id,
            turn_id=turn_id,
            provider_family=provider_family or "mlx",
            capabilities=capabilities
            or StreamProviderCapabilities(
                backend=StreamProducerBackend.LOCAL,
                provider_family="mlx",
                supports_cancellation=True,
            ),
            close_after_terminal=close_after_terminal,
        )

    async def _provider_events(
        self,
    ) -> AsyncGenerator[StreamProviderEvent, None]:
        try:
            while True:
                chunk = await self._next_raw()
                if chunk is self._SENTINEL:
                    return
                text, metadata = self._chunk_text_and_metadata(chunk)
                if text is None:
                    continue
                yield StreamProviderEvent(
                    kind=StreamItemKind.ANSWER_DELTA,
                    text_delta=text,
                    metadata=metadata,
                    provider_event_type="mlx_lm.delta",
                )
        finally:
            self.close()

    @staticmethod
    def _chunk_text_and_metadata(
        chunk: object,
    ) -> tuple[str | None, dict[str, LooseJsonValue]]:
        if isinstance(chunk, str):
            return chunk, {}
        if isinstance(chunk, Token):
            raise StreamValidationError("unsupported legacy local stream item")
        token = getattr(chunk, "token", None)
        if not isinstance(token, str):
            text = getattr(chunk, "text", None)
            return (text, {}) if isinstance(text, str) else (None, {})

        metadata: dict[str, LooseJsonValue] = {}
        token_id = getattr(chunk, "id", None)
        if isinstance(token_id, int) and token_id >= 0:
            metadata["token_id"] = token_id
        probability = getattr(chunk, "probability", None)
        if isinstance(probability, int | float):
            metadata["probability"] = float(probability)
        step = getattr(chunk, "step", None)
        if isinstance(step, int):
            metadata["step"] = step
        probability_distribution = getattr(
            chunk,
            "probability_distribution",
            None,
        )
        if probability_distribution is not None:
            metadata["probability_distribution"] = cast(
                LooseJsonValue,
                probability_distribution,
            )
        candidates = getattr(chunk, "tokens", None)
        if isinstance(candidates, list):
            metadata["tokens"] = cast(
                LooseJsonValue,
                [
                    candidate
                    for item in candidates
                    if (candidate := MlxLmStream._candidate_metadata(item))
                ],
            )
        return token, metadata

    @staticmethod
    def _candidate_metadata(
        candidate: object,
    ) -> dict[str, LooseJsonValue] | None:
        text = getattr(candidate, "token", None)
        if not isinstance(text, str):
            return None
        metadata: dict[str, LooseJsonValue] = {"token": text}
        token_id = getattr(candidate, "id", None)
        if isinstance(token_id, int) and token_id >= 0:
            metadata["token_id"] = token_id
        probability = getattr(candidate, "probability", None)
        if isinstance(probability, int | float):
            metadata["probability"] = float(probability)
        return metadata


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
    ) -> AsyncGenerator[CanonicalStreamItem, None]:
        sampler, prompt = self._get_sampler_and_prompt(
            inputs, settings, skip_special_tokens
        )
        mlx_lm = _require_mlx_lm()
        stream_generate_fn = cast(
            Callable[..., Iterator[object]], getattr(mlx_lm, "stream_generate")
        )
        # mlx_lm's generation stream is thread-local, so keep generation on
        # the same thread that loaded the model and initialized MLX.
        stream = MlxLmStream(
            lambda: stream_generate_fn(
                self._model,
                self._tokenizer,
                prompt,
                sampler=sampler,
                max_tokens=settings.max_new_tokens,
            ),
            use_executor=False,
        )
        try:
            while True:
                try:
                    chunk = await stream.__anext__()
                except StopAsyncIteration:
                    break
                yield chunk
        finally:
            stream.close()

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
        instructions: str | None = None,
        skip_special_tokens: bool = False,
        tensor_format: Literal["pt"] = "pt",
        tool: ToolManager | None = None,
    ) -> TextGenerationResponse:
        settings = settings or GenerationSettings()
        inputs = super()._tokenize_input(
            input,
            system_prompt=system_prompt,
            developer_prompt=developer_prompt,
            context=None,
            tensor_format=tensor_format,
            tool=tool,
            chat_template_settings=asdict(settings.chat_settings),
            instructions=instructions,
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
            provider_family="mlx",
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
