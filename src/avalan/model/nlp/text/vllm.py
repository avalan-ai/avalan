from ....entities import (
    GenerationSettings,
    Input,
    TransformerEngineSettings,
)
from ....model.nlp.text.generation import TextGenerationModel
from ....model.provider import provider_family_value
from ....model.stream import (
    CanonicalStreamItem,
    LocalTextStreamEventParser,
    StreamItemKind,
    StreamProducerBackend,
    StreamProviderCapabilities,
    StreamProviderEvent,
    stream_token_metadata,
)
from ....model.vendor import TextGenerationVendorStream
from ....tool.manager import ToolManager
from ....types import LooseJsonValue

from asyncio import to_thread
from dataclasses import asdict, replace
from importlib import import_module
from logging import Logger, getLogger
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Iterator,
    cast,
)

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
    _SENTINEL = object()

    _cumulative_text: bool
    _iterator: Iterator[Any] | None
    _stream: AsyncIterator[Any]
    _supports_cancellation: bool

    def __init__(
        self,
        generator: Iterator[Any] | AsyncIterator[Any],
        *,
        cumulative_text: bool = False,
        provider_family: str = "vllm",
    ) -> None:
        async def empty_generator() -> (
            AsyncGenerator[CanonicalStreamItem, None]
        ):
            if False:
                yield cast(CanonicalStreamItem, None)

        self._cumulative_text = cumulative_text
        if hasattr(generator, "__anext__"):
            self._supports_cancellation = True
            self._iterator = None
            self._stream = cast(AsyncIterator[Any], generator)
            super().__init__(
                empty_generator(),
                provider_family=provider_family,
                sources=(self._stream,),
            )
            return

        self._supports_cancellation = False
        self._iterator = generator
        self._stream = self._async_generator_from_iterator(generator)
        super().__init__(
            empty_generator(),
            provider_family=provider_family_value(provider_family),
            sources=(self._stream,),
        )

    async def __anext__(self) -> CanonicalStreamItem:
        return await super().__anext__()

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
            provider_family=provider_family or self._provider_family,
            capabilities=capabilities
            or StreamProviderCapabilities(
                backend=StreamProducerBackend.LOCAL,
                provider_family=self._provider_family,
                supports_cancellation=self._supports_cancellation,
            ),
            close_after_terminal=close_after_terminal,
        )

    async def _provider_events(
        self,
    ) -> AsyncGenerator[StreamProviderEvent, None]:
        parser = LocalTextStreamEventParser()
        previous_text = ""
        async for chunk in self._stream:
            if isinstance(chunk, BaseException):
                raise chunk
            text, metadata = self._chunk_text_and_metadata(chunk)
            if text is None:
                continue
            if self._cumulative_text:
                if text.startswith(previous_text):
                    delta = text[len(previous_text) :]
                else:
                    delta = text
                previous_text = text
                text = delta
            if not text:
                continue
            for event in parser.push(text):
                yield self._event_with_metadata(
                    event,
                    metadata,
                    provider_event_type="vllm.delta",
                )
        for event in parser.flush():
            yield event

    @staticmethod
    def _event_with_metadata(
        event: StreamProviderEvent,
        metadata: dict[str, LooseJsonValue],
        *,
        provider_event_type: str,
    ) -> StreamProviderEvent:
        if event.kind is StreamItemKind.ANSWER_DELTA:
            event_metadata = {**event.metadata, **metadata}
        else:
            event_metadata = event.metadata
        return replace(
            event,
            metadata=event_metadata,
            provider_event_type=event.provider_event_type
            or provider_event_type,
        )

    @classmethod
    def _chunk_text(cls, chunk: object) -> str:
        text, _ = cls._chunk_text_and_metadata(chunk)
        return text if text is not None else str(chunk)

    @staticmethod
    def _chunk_text_and_metadata(
        chunk: object,
    ) -> tuple[str | None, dict[str, LooseJsonValue]]:
        if isinstance(chunk, str):
            return chunk, {}
        token = getattr(chunk, "token", None)
        if isinstance(token, str):
            return token, VllmStream._chunk_metadata(chunk)
        outputs = getattr(chunk, "outputs", None)
        if (
            isinstance(outputs, list)
            and outputs
            and (output_text := getattr(outputs[0], "text", None)) is not None
        ):
            return (
                (
                    output_text
                    if isinstance(output_text, str)
                    else str(output_text)
                ),
                VllmStream._chunk_metadata(chunk),
            )
        text = getattr(chunk, "text", None)
        if isinstance(text, str):
            return text, VllmStream._chunk_metadata(chunk)
        return str(chunk), VllmStream._chunk_metadata(chunk)

    @staticmethod
    def _chunk_metadata(chunk: object) -> dict[str, LooseJsonValue]:
        probability = getattr(chunk, "probability", None)
        if isinstance(probability, bool) or not isinstance(
            probability, (int, float)
        ):
            probability = None

        probability_distribution = getattr(
            chunk, "probability_distribution", None
        )
        if not isinstance(probability_distribution, str):
            probability_distribution = None

        candidates = getattr(chunk, "tokens", None)
        candidate_metadata = (
            tuple(
                candidate
                for item in candidates
                if (candidate := VllmStream._candidate_metadata(item))
            )
            if isinstance(candidates, list)
            else None
        )

        return stream_token_metadata(
            token_id=VllmStream._non_negative_int(getattr(chunk, "id", None)),
            probability=probability,
            step=VllmStream._non_negative_int(getattr(chunk, "step", None)),
            probability_distribution=probability_distribution,
            candidates=candidate_metadata,
        )

    @staticmethod
    def _candidate_metadata(
        candidate: object,
    ) -> tuple[str, int | None, float | None] | None:
        text = getattr(candidate, "token", None)
        if not isinstance(text, str) or not text:
            return None
        probability = getattr(candidate, "probability", None)
        if isinstance(probability, bool) or not isinstance(
            probability, (int, float)
        ):
            probability = None
        return (
            text,
            VllmStream._non_negative_int(getattr(candidate, "id", None)),
            probability,
        )

    @staticmethod
    def _non_negative_int(value: object) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value if value >= 0 else None
        return None

    @staticmethod
    async def _async_generator_from_iterator(
        iterator: Iterator[Any],
    ) -> AsyncGenerator[Any, None]:
        while True:
            chunk = await to_thread(
                cast(
                    Callable[[], Any],
                    lambda: next(iterator, VllmStream._SENTINEL),
                )
            )
            if chunk is VllmStream._SENTINEL:
                return
            yield chunk


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

    def _stream_generator(
        self,
        prompt: str,
        settings: GenerationSettings,
    ) -> TextGenerationVendorStream:
        params = self._build_sampling_params(settings)
        model = cast(Any, self._model)
        iterator = iter(model.generate([prompt], params, stream=True))
        return VllmStream(iterator, cumulative_text=True)

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
    ) -> TextGenerationVendorStream | str:
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
            return self._stream_generator(prompt, generation_settings)
        return self._string_output(prompt, generation_settings)
