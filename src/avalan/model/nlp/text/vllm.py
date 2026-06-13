from ....entities import (
    GenerationSettings,
    Input,
    TransformerEngineSettings,
)
from ....model.nlp.text.generation import TextGenerationModel
from ....model.provider import provider_family_value
from ....model.vendor import TextGenerationVendorStream
from ....tool.manager import ToolManager

from asyncio import to_thread
from dataclasses import asdict, replace
from importlib import import_module
from logging import Logger, getLogger
from typing import Any, AsyncGenerator, Awaitable, Callable, Iterator, cast

_UNSET = object()
LLM: Any = _UNSET
SamplingParams: Any = _UNSET


def _vllm_attribute(name: str) -> Any:
    try:
        module = import_module("vllm")
    except ImportError:  # pragma: no cover - vllm may not be installed
        return None
    return getattr(module, name, None)


def _llm_class() -> Any:
    global LLM
    if LLM is _UNSET:
        LLM = _vllm_attribute("LLM")
    assert LLM is not None and LLM is not _UNSET, "vLLM is not available"
    return LLM


def _sampling_params_class() -> Any:
    global SamplingParams
    if SamplingParams is _UNSET:
        SamplingParams = _vllm_attribute("SamplingParams")
    assert (
        SamplingParams is not None and SamplingParams is not _UNSET
    ), "vLLM is not available"
    return SamplingParams


class VllmStream(TextGenerationVendorStream):
    def __init__(
        self,
        generator: Iterator[str] | AsyncGenerator[str, None],
        *,
        provider_family: str = "vllm",
    ) -> None:
        if hasattr(generator, "__anext__"):
            super().__init__(
                cast(AsyncGenerator[str, None], generator),
                provider_family=provider_family,
            )
            self._iterator = None
            return

        self._iterator = generator
        self._provider_family = provider_family_value(provider_family)
        self._usage = None
        self._generator = cast(
            AsyncGenerator[str, None], cast(Any, self._iterator)
        )

    async def __anext__(self) -> str:
        if self._iterator is None:
            chunk = await super().__anext__()
            if isinstance(chunk, str):
                return chunk
            return chunk.token

        iterator = self._iterator
        chunk = await to_thread(
            cast(Callable[[], Any], lambda: next(iterator, None))
        )
        if chunk is None:
            raise StopAsyncIteration
        return str(chunk)


class VllmModel(TextGenerationModel):
    def __init__(
        self,
        model_id: str,
        settings: TransformerEngineSettings | None = None,
        logger: Logger = getLogger(__name__),
    ) -> None:
        super().__init__(model_id, settings, logger)

    @property
    def supports_sample_generation(self) -> bool:
        return False

    def _load_model(self) -> Any:
        llm_class = _llm_class()
        assert self._model_id, "A model id is required."
        return llm_class(
            model=self._model_id,
            tokenizer=self._settings.tokenizer_name_or_path or self._model_id,
            trust_remote_code=self._settings.trust_remote_code,
        )

    def _build_sampling_params(self, settings: GenerationSettings) -> Any:
        sampling_params_class = _sampling_params_class()
        return sampling_params_class(
            temperature=(
                settings.temperature
                if settings.temperature is not None
                else 1.0
            ),
            top_p=settings.top_p if settings.top_p is not None else 1.0,
            top_k=settings.top_k if settings.top_k is not None else -1,
            max_tokens=settings.max_new_tokens,
            stop=settings.stop_strings,
        )

    def _prompt(
        self,
        input: Input,
        system_prompt: str | None,
        developer_prompt: str | None = None,
        tool: ToolManager | None = None,
        chat_template_settings: dict[str, object] | None = None,
        *,
        instructions: str | None = None,
    ) -> str:
        inputs = super()._tokenize_input(
            input,
            system_prompt=system_prompt,
            developer_prompt=developer_prompt,
            context=None,
            tensor_format="pt",
            tool=tool,
            chat_template_settings=chat_template_settings,
            instructions=instructions,
        )
        tokenizer = self._tokenizer
        assert tokenizer is not None
        input_ids = cast(dict[str, Any], inputs)["input_ids"]
        return cast(
            str,
            tokenizer.decode(input_ids[0], skip_special_tokens=False),
        )

    async def _stream_generator(
        self,
        prompt: str,
        settings: GenerationSettings,
    ) -> AsyncGenerator[str, None]:
        params = self._build_sampling_params(settings)
        model = cast(Any, self._model)
        iterator = iter(model.generate([prompt], params, stream=True))

        def _next(default: Any = None) -> Any:
            return next(iterator, default)

        while True:
            chunk = await to_thread(lambda: _next(None))
            if chunk is None:
                return
            if isinstance(chunk, str):
                yield chunk
            else:
                text = getattr(chunk.outputs[0], "text", "")
                if text:
                    yield str(text)

    def _string_output(
        self,
        prompt: str,
        settings: GenerationSettings,
    ) -> str:
        params = self._build_sampling_params(settings)
        model = cast(Any, self._model)
        results = list(model.generate([prompt], params))
        return results[0].outputs[0].text if results else ""

    async def __call__(
        self,
        input: Input,
        system_prompt: str | None = None,
        developer_prompt: str | None = None,
        settings: GenerationSettings | None = None,
        *,
        instructions: str | None = None,
        tool: ToolManager | None = None,
    ) -> TextGenerationVendorStream | str | AsyncGenerator[str, None]:
        settings = settings or GenerationSettings()
        prompt = self._prompt(
            input,
            system_prompt,
            developer_prompt,
            tool,
            asdict(settings.chat_settings),
            instructions=instructions,
        )
        generation_settings = replace(settings, do_sample=False)
        if settings.use_async_generator:
            stream = self._stream_generator(prompt, generation_settings)
            if isinstance(stream, Awaitable):
                resolved_stream = await stream
                if isinstance(resolved_stream, str):
                    return resolved_stream
                return cast(
                    TextGenerationVendorStream
                    | str
                    | AsyncGenerator[str, None],
                    VllmStream(resolved_stream),
                )
            return VllmStream(stream)
        return self._string_output(prompt, generation_settings)
