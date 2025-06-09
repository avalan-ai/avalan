from asyncio import to_thread
from ....compat import override
from ....entities import (
    GenerationSettings,
    Input,
    TransformerEngineSettings,
)
from ....model import TextGenerationVendorStream
from ....model.nlp.text.generation import TextGenerationModel
from ....tool.manager import ToolManager
from dataclasses import replace
from logging import Logger
from typing import AsyncGenerator

try:
    from vllm import LLM, SamplingParams
except Exception:  # pragma: no cover - vllm may not be installed
    LLM = None
    SamplingParams = None


class VllmStream(TextGenerationVendorStream):
    def __init__(self, generator):
        super().__init__(generator)
        self._iterator = generator

    async def __anext__(self) -> str:
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
        logger: Logger | None = None,
    ) -> None:
        super().__init__(model_id, settings, logger)

    @property
    def supports_sample_generation(self) -> bool:
        return False

    def _load_model(self):
        assert LLM, "vLLM is not available"
        return LLM(
            model=self._model_id,
            tokenizer=self._settings.tokenizer_name_or_path or self._model_id,
            trust_remote_code=self._settings.trust_remote_code,
        )

    def _build_sampling_params(
        self, settings: GenerationSettings
    ) -> SamplingParams:
        assert SamplingParams, "vLLM is not available"
        return SamplingParams(
            temperature=settings.temperature,
            top_p=settings.top_p,
            top_k=settings.top_k,
            max_tokens=settings.max_new_tokens,
            stop=settings.stop_strings,
        )

    def _prompt(
        self,
        input: Input,
        system_prompt: str | None,
        tool: ToolManager | None,
        chat_template_settings: dict[str, object] | None,
    ) -> str:
        inputs = super()._tokenize_input(
            input,
            system_prompt,
            context=None,
            tensor_format="pt",
            tool=tool,
            chat_template_settings=chat_template_settings,
        )
        return self._tokenizer.decode(
            inputs["input_ids"][0], skip_special_tokens=False
        )

    async def _stream_generator(
        self,
        prompt: str,
        settings: GenerationSettings,
    ) -> AsyncGenerator[str, None]:
        params = self._build_sampling_params(settings)
        iterator = self._model.generate([prompt], params, stream=True)
        stream = VllmStream(iter(iterator))
        async for chunk in stream:
            yield chunk

    def _string_output(
        self,
        prompt: str,
        settings: GenerationSettings,
    ) -> str:
        params = self._build_sampling_params(settings)
        results = list(self._model.generate([prompt], params))
        return results[0].outputs[0].text if results else ""

    @override
    async def __call__(
        self,
        input: Input,
        system_prompt: str | None = None,
        settings: GenerationSettings | None = None,
        *,
        tool: ToolManager | None = None,
    ) -> TextGenerationVendorStream | str:
        settings = settings or GenerationSettings()
        prompt = self._prompt(
            input,
            system_prompt,
            tool,
            settings.chat_template_settings,
        )
        generation_settings = replace(settings, do_sample=False)
        if settings.use_async_generator:
            return await self._stream_generator(prompt, generation_settings)
        return self._string_output(prompt, generation_settings)
