from ....compat import override
from ....entities import (
    GenerationSettings,
    Input,
    TransformerEngineSettings,
)
from ....model.nlp.text.generation import TextGenerationModel
from ....model.vendor import TextGenerationVendorStream
from ....tool.manager import ToolManager

from asyncio import to_thread
from dataclasses import asdict, replace
from logging import Logger, getLogger
from typing import Any, AsyncGenerator, Iterator, cast

try:
    from vllm import LLM, SamplingParams
except Exception:  # pragma: no cover - vllm may not be installed
    LLM = None  # type: ignore[misc,assignment]
    SamplingParams = None  # type: ignore[misc,assignment]


class VllmStream(TextGenerationVendorStream):
    def __init__(
        self, generator: Iterator[str] | AsyncGenerator[str, None]
    ) -> None:
        if hasattr(generator, "__anext__"):
            super().__init__(cast(AsyncGenerator[str, None], generator))
            self._iterator = None
            return

        iterator = cast(Iterator[str], generator)
        self._iterator = iterator
        self._generator = iterator  # type: ignore[assignment]

    async def __anext__(self) -> str:
        if self._iterator is None:
            return await super().__anext__()

        def _next(default: str | None = None) -> str | None:
            return next(self._iterator, default)

        chunk = await to_thread(_next)
        if chunk is None:
            raise StopAsyncIteration
        return chunk


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
        assert LLM is not None, "vLLM is not available"
        assert self._model_id, "A model id is required."
        return LLM(
            model=self._model_id,
            tokenizer=self._settings.tokenizer_name_or_path or self._model_id,
            trust_remote_code=self._settings.trust_remote_code,
        )

    def _build_sampling_params(self, settings: GenerationSettings) -> Any:
        assert SamplingParams is not None, "vLLM is not available"
        return SamplingParams(
            temperature=settings.temperature if settings.temperature else 1.0,
            top_p=settings.top_p if settings.top_p else 1.0,
            top_k=settings.top_k if settings.top_k else -1,
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
    ) -> str:
        inputs = super()._tokenize_input(
            input,
            system_prompt,
            developer_prompt,
            context=None,
            tensor_format="pt",
            tool=tool,
            chat_template_settings=chat_template_settings,
        )
        tokenizer = self._tokenizer
        assert tokenizer is not None
        input_ids = cast(dict[str, Any], inputs)["input_ids"]
        return tokenizer.decode(input_ids[0], skip_special_tokens=False)

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
            chunk = await to_thread(_next, None)
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

    @override  # type: ignore[untyped-decorator]
    async def __call__(
        self,
        input: Input,
        system_prompt: str | None = None,
        developer_prompt: str | None = None,
        settings: GenerationSettings | None = None,
        *,
        tool: ToolManager | None = None,
    ) -> TextGenerationVendorStream | str:
        settings = settings or GenerationSettings()
        prompt = self._prompt(
            input,
            system_prompt,
            developer_prompt,
            tool,
            asdict(settings.chat_settings),
        )
        generation_settings = replace(settings, do_sample=False)
        if settings.use_async_generator:
            return await self._stream_generator(prompt, generation_settings)
        return self._string_output(prompt, generation_settings)
