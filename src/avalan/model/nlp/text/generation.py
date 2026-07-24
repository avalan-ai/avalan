from ....entities import (
    GenerationSettings,
    Input,
    Message,
    MessageContent,
    MessageContentFile,
    MessageContentImage,
    MessageContentText,
    MessageRole,
    ProbabilityDistribution,
    TextGenerationLoaderClass,
    TransformerEngineSettings,
)
from ....model.capability import ModelCapabilityCatalog
from ....model.engine import Engine
from ....model.nlp import BaseNLPModel
from ....model.provider import ProviderFamily
from ....model.reasoning import (
    ReasoningSummaryRequestCapability,
    validate_reasoning_summary_request,
)
from ....model.response.text import (
    TextGenerationResponse,
    _TextGenerationWorkerShutdownError,
)
from ....model.stream import (
    CanonicalStreamItem,
    LocalTextStreamEventParser,
    StreamProducerBackend,
    StreamProviderCapabilities,
    StreamProviderEvent,
    TextGenerationNonStreamResult,
    normalize_provider_stream,
    stream_token_metadata,
)
from ....model.vendor import TextGenerationVendor
from ....tool.parser import ToolCallParser
from .local_protocol import (
    LOCAL_STRUCTURED_OUTPUT_PROTOCOL,
    LocalStructuredOutputProtocol,
)

from asyncio import CancelledError, Queue, run_coroutine_threadsafe, sleep
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import asdict, replace
from importlib import import_module
from importlib.util import find_spec
from logging import Logger, getLogger
from threading import Event as ThreadEvent
from threading import Thread
from time import perf_counter
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Literal,
    TypeAlias,
    cast,
)

if TYPE_CHECKING:
    from diffusers import DiffusionPipeline
    from torch import Tensor
    from transformers import (
        AsyncTextIteratorStreamer,
        AutoModelForCausalLM,
        Gemma3ForConditionalGeneration,
        GptOssForCausalLM,
        Mistral3ForConditionalGeneration,
        PreTrainedModel,
    )
    from transformers.generation.stopping_criteria import StoppingCriteria
    from transformers.tokenization_utils_base import BatchEncoding
else:
    Tensor: TypeAlias = Any

    class _LazyExternal:
        __test__ = False

        def __init__(self, module_name: str, name: str) -> None:
            self._module_name = module_name
            self._name = name

        def __getattr__(self, name: str) -> Any:
            if name == "_is_coroutine" or (
                name.startswith("__") and name.endswith("__")
            ):
                raise AttributeError(name)
            module = import_module(self._module_name)
            target = getattr(module, self._name)
            return getattr(target, name)

    class AsyncTextIteratorStreamer:  # noqa: D101
        def __new__(cls, *args: object, **kwargs: object) -> object:
            streamer_type = getattr(
                import_module("transformers"), "AsyncTextIteratorStreamer"
            )
            return streamer_type(*args, **kwargs)

    class BatchEncoding:  # noqa: D101
        pass

    class DiffusionPipeline:  # noqa: D101
        pass

    class PreTrainedModel:  # noqa: D101
        pass

    class StoppingCriteria:  # noqa: D101
        pass

    AutoModelForCausalLM = _LazyExternal(
        "transformers", "AutoModelForCausalLM"
    )
    Gemma3ForConditionalGeneration = _LazyExternal(
        "transformers", "Gemma3ForConditionalGeneration"
    )
    GptOssForCausalLM = _LazyExternal("transformers", "GptOssForCausalLM")
    Mistral3ForConditionalGeneration = _LazyExternal(
        "transformers", "Mistral3ForConditionalGeneration"
    )


def log_softmax(*args: object, **kwargs: object) -> Any:
    return getattr(import_module("torch"), "log_softmax")(*args, **kwargs)


def softmax(*args: object, **kwargs: object) -> Any:
    return getattr(import_module("torch"), "softmax")(*args, **kwargs)


def topk(*args: object, **kwargs: object) -> Any:
    return getattr(import_module("torch"), "topk")(*args, **kwargs)


def gumbel_softmax(*args: object, **kwargs: object) -> Any:
    module = import_module("torch.nn.functional")
    return getattr(module, "gumbel_softmax")(*args, **kwargs)


_TOOL_MESSAGE_PARSER = ToolCallParser()
_STREAMER_TIMEOUT_SECONDS = 0.1
_STREAM_THREAD_JOIN_TIMEOUT_SECONDS = 2.0
_STREAM_HANDOFF_MAX_QUEUE_SIZE = 64
_TRANSFORMERS_PROVIDER_FAMILY = "transformers"
_TRANSFORMERS_STREAM_SESSION_ID = "transformers-stream"
_TRANSFORMERS_RUN_ID = "transformers-run"
_TRANSFORMERS_TURN_ID = "transformers-turn"
_UNSUPPORTED_REASONING_SUMMARY = ReasoningSummaryRequestCapability()


class _StopOnEventCriteria(StoppingCriteria):
    def __init__(self, event: ThreadEvent) -> None:
        self._event = event

    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
        return self._event.is_set()


def _is_event_loop_closed_error(exc: RuntimeError) -> bool:
    return str(exc) == "Event loop is closed"


def _configure_lossless_streamer_handoff(
    streamer: object,
    stop_event: ThreadEvent,
    max_queue_size: int = _STREAM_HANDOFF_MAX_QUEUE_SIZE,
) -> int | None:
    assert max_queue_size > 0
    text_queue = getattr(streamer, "text_queue", None)
    loop = getattr(streamer, "loop", None)
    if text_queue is None or loop is None:
        return None

    queue: Queue[object] = Queue(maxsize=max_queue_size)
    setattr(streamer, "text_queue", queue)

    def put_item(value: object) -> None:
        future = run_coroutine_threadsafe(queue.put(value), loop)
        while True:
            if stop_event.is_set():
                future.cancel()
                raise RuntimeError("Event loop is closed")
            try:
                future.result(timeout=_STREAMER_TIMEOUT_SECONDS)
                return
            except FutureTimeoutError:
                continue

    def on_finalized_text(text: str, stream_end: bool = False) -> None:
        put_item(text)
        if stream_end:
            put_item(getattr(streamer, "stop_signal"))

    setattr(streamer, "on_finalized_text", on_finalized_text)
    return max_queue_size


def _streamer_has_pending_items(streamer: object) -> bool:
    for attribute in ("text_queue", "queue"):
        queue = getattr(streamer, attribute, None)
        empty = getattr(queue, "empty", None)
        if callable(empty):
            return not bool(empty())
    return False


def _local_stream_capabilities(
    provider_family: str,
    *,
    max_queue_depth: int | None = None,
    supports_tool_calls: bool = True,
) -> StreamProviderCapabilities:
    return StreamProviderCapabilities(
        backend=StreamProducerBackend.LOCAL,
        provider_family=provider_family,
        supports_reasoning=True,
        supports_tool_calls=supports_tool_calls,
        supports_cancellation=True,
        max_queue_depth=max_queue_depth,
    )


async def _canonical_transformers_stream(
    events: AsyncIterable[StreamProviderEvent],
    *,
    max_queue_depth: int | None = None,
    supports_tool_calls: bool = True,
) -> AsyncIterator[CanonicalStreamItem]:
    stream = normalize_provider_stream(
        events,
        stream_session_id=_TRANSFORMERS_STREAM_SESSION_ID,
        run_id=_TRANSFORMERS_RUN_ID,
        turn_id=_TRANSFORMERS_TURN_ID,
        provider_family=_TRANSFORMERS_PROVIDER_FAMILY,
        capabilities=_local_stream_capabilities(
            _TRANSFORMERS_PROVIDER_FAMILY,
            max_queue_depth=max_queue_depth,
            supports_tool_calls=supports_tool_calls,
        ),
    )
    try:
        async for item in stream:
            yield item
    except (CancelledError, GeneratorExit):
        await cast(Any, stream).aclose()
        raise


def _non_negative_token_id(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    item = getattr(value, "item", None)
    if callable(item):
        return _non_negative_token_id(item())
    return None


def _probability_value(value: object) -> float:
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    assert isinstance(value, (int, float))
    assert not isinstance(value, bool)
    return float(value)


class TextGenerationModel(BaseNLPModel):
    _loaders: dict[TextGenerationLoaderClass, Any] = {
        "auto": AutoModelForCausalLM,
        "gemma3": Gemma3ForConditionalGeneration,
        "gpt-oss": GptOssForCausalLM,
        "mistral3": Mistral3ForConditionalGeneration,
    }

    @classmethod
    def _loader(cls, loader_class: TextGenerationLoaderClass) -> Any:
        return cls._loaders[loader_class]

    def __init__(
        self,
        model_id: str,
        settings: TransformerEngineSettings | None = None,
        logger: Logger = getLogger(__name__),
    ) -> None:
        super().__init__(
            model_id, settings or TransformerEngineSettings(), logger
        )

    @property
    def reasoning_summary_request_capability(
        self,
    ) -> ReasoningSummaryRequestCapability:
        """Return the adapter's request-time summary capability."""
        return _UNSUPPORTED_REASONING_SUMMARY

    @property
    def reasoning_summary_provider(self) -> str:
        """Return the provider family used in summary capability errors."""
        provider_modules = {
            "avalan.model.nlp.text.ds4": "ds4",
            "avalan.model.nlp.text.generation": "transformers",
            "avalan.model.nlp.text.mlxlm": "mlx",
            "avalan.model.nlp.text.vllm": "vllm",
            "avalan.model.nlp.text.vendor": "vendor",
            "avalan.model.nlp.text.vendor.anyscale": "anyscale",
            "avalan.model.nlp.text.vendor.anthropic": "anthropic",
            "avalan.model.nlp.text.vendor.bedrock": "bedrock",
            "avalan.model.nlp.text.vendor.deepinfra": "deepinfra",
            "avalan.model.nlp.text.vendor.deepseek": "deepseek",
            "avalan.model.nlp.text.vendor.google": "google",
            "avalan.model.nlp.text.vendor.groq": "groq",
            "avalan.model.nlp.text.vendor.huggingface": "huggingface",
            "avalan.model.nlp.text.vendor.hyperbolic": "hyperbolic",
            "avalan.model.nlp.text.vendor.litellm": "litellm",
            "avalan.model.nlp.text.vendor.ollama": "ollama",
            "avalan.model.nlp.text.vendor.openai": "openai",
            "avalan.model.nlp.text.vendor.openrouter": "openrouter",
            "avalan.model.nlp.text.vendor.together": "together",
        }
        resolved_provider = "local"
        for model_type in type(self).__mro__:
            module_name = model_type.__module__
            if module_name in provider_modules:
                resolved_provider = provider_modules[module_name]
                break
        return resolved_provider

    @property
    def supports_sample_generation(self) -> bool:
        return True

    @property
    def supports_token_streaming(self) -> bool:
        return True

    def _load_model(
        self,
    ) -> PreTrainedModel | TextGenerationVendor | DiffusionPipeline:
        assert (
            cast(TransformerEngineSettings, self._settings).loader_class
            in self._loaders
        ), (
            "Unrecognized loader "
            + f"{cast(TransformerEngineSettings, self._settings).loader_class}"
        )
        settings = cast(TransformerEngineSettings, self._settings)

        if settings.quantization and find_spec("bitsandbytes"):
            from transformers import BitsAndBytesConfig

            quantization = settings.quantization
            bnb_config = cast(
                object,
                cast(Any, BitsAndBytesConfig)(
                    load_in_4bit=quantization.load_in_4bit,
                    bnb_4bit_quant_type=quantization.bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=quantization.bnb_4bit_use_double_quant,
                    bnb_4bit_compute_dtype=quantization.bnb_4bit_compute_dtype,
                ),
            )
        else:
            bnb_config = None

        loader = self._loader(settings.loader_class or "auto")
        model_args = dict(
            cache_dir=settings.cache_dir,
            subfolder=settings.subfolder or "",
            attn_implementation=settings.attention,
            output_hidden_states=settings.output_hidden_states,
            trust_remote_code=settings.trust_remote_code,
            state_dict=settings.state_dict,
            local_files_only=settings.local_files_only,
            low_cpu_mem_usage=(
                True if self._device else settings.low_cpu_mem_usage
            ),
            dtype=Engine.weight(settings.weight_type),
            device_map=self._device,
            token=settings.access_token,
            quantization_config=bnb_config,
            revision=settings.revision,
            tp_plan=Engine._get_tp_plan(settings.parallel),
            distributed_config=Engine._get_distributed_config(
                settings.distributed_config
            ),
        )
        if model_args["quantization_config"] is None:
            model_args.pop("quantization_config", None)

        model = cast(
            PreTrainedModel,
            cast(Any, loader).from_pretrained(self._model_id, **model_args),
        )
        return model

    async def __call__(
        self,
        input: Input,
        system_prompt: str | None = None,
        developer_prompt: str | None = None,
        settings: GenerationSettings | None = None,
        stopping_criterias: list[StoppingCriteria] | None = None,
        *,
        instructions: str | None = None,
        manual_sampling: bool = False,
        pick: int | None = None,
        skip_special_tokens: bool = False,
        capability: ModelCapabilityCatalog | None = None,
        **kwargs: object,
    ) -> TextGenerationResponse:
        settings = settings or GenerationSettings()
        validate_reasoning_summary_request(self, settings)
        assert self._tokenizer, (
            f"Model {self._model} can't be executed "
            + "without a tokenizer loaded first"
        )
        assert self._model, (
            f"Model {self._model} can't be executed, it "
            + "needs to be loaded first"
        )
        effective_capability = self._effective_local_capability(capability)
        structured_output_protocol = (
            LOCAL_STRUCTURED_OUTPUT_PROTOCOL
            if effective_capability is not None
            else None
        )

        assert settings.temperature is None or (
            settings.temperature > 0 or settings.temperature == 0.0
        ), (
            "Temperature has to be a strictly positive float or zero, "
            + "otherwise your next token scores will be invalid"
        )

        do_sample = (
            settings.do_sample if self.supports_sample_generation else False
        )
        if self.supports_sample_generation and settings.temperature:
            do_sample = True

        assert (not do_sample and not settings.temperature) or (
            do_sample and settings.temperature
        ), "Sample-based generation can only be set with temperature"

        output_fn = (
            self._string_output
            if not settings.use_async_generator
            else (
                self._token_generator
                if manual_sampling
                else self._stream_generator
            )
        )
        generation_settings = replace(
            settings,
            do_sample=do_sample,
            pad_token_id=(
                settings.pad_token_id
                if settings.pad_token_id is not None
                else self._tokenizer.eos_token_id
            ),
        )
        inputs = self._tokenize_input(
            input,
            system_prompt=system_prompt,
            developer_prompt=developer_prompt,
            context=None,
            capability=effective_capability,
            chat_template_settings=asdict(settings.chat_settings),
            instructions=instructions,
        )
        return TextGenerationResponse(
            output_fn,
            inputs=inputs,
            logger=self._logger,
            generation_settings=generation_settings,
            pick=pick,
            settings=generation_settings,
            stopping_criterias=stopping_criterias,
            skip_special_tokens=skip_special_tokens,
            local_structured_output_protocol=structured_output_protocol,
            use_async_generator=settings.use_async_generator,
            bos_token=self._tokenizer.bos_token,
            provider_family="transformers",
        )

    async def _stream_generator(
        self,
        inputs: dict[str, Tensor] | Tensor,
        settings: GenerationSettings,
        stopping_criterias: list[StoppingCriteria] | None,
        skip_special_tokens: bool,
        **kwargs: object,
    ) -> AsyncGenerator[CanonicalStreamItem, None]:
        _l = self._log
        structured_output_protocol = kwargs.get(
            "local_structured_output_protocol"
        )
        assert structured_output_protocol is None or (
            structured_output_protocol is LOCAL_STRUCTURED_OUTPUT_PROTOCOL
        )
        stop_event = ThreadEvent()
        thread_errors: list[BaseException] = []
        stream_stopping_criterias = list(stopping_criterias or [])
        stream_stopping_criterias.append(_StopOnEventCriteria(stop_event))

        streamer = AsyncTextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            timeout=_STREAMER_TIMEOUT_SECONDS,
            decode_kwargs={"skip_special_tokens": skip_special_tokens},
        )
        queue_size = _configure_lossless_streamer_handoff(streamer, stop_event)

        if queue_size is None:
            _l("Created generator async text token streamer")
        else:
            _l(
                "Created generator async text token streamer "
                f"with {queue_size} queued chunks"
            )

        def finish_stream() -> None:
            try:
                streamer.on_finalized_text("", stream_end=True)
            except RuntimeError as exc:
                if not (
                    stop_event.is_set() and _is_event_loop_closed_error(exc)
                ):
                    raise

        def generate_stream() -> None:
            _l(
                f"Streaming up to {settings.max_new_tokens} tokens "
                f"{'with' if settings.do_sample else 'without'} sample "
                f"and {settings.temperature} temperature"
            )
            try:
                self._generate_output(
                    inputs,
                    settings,
                    stream_stopping_criterias,
                    streamer=streamer,
                )
            except BaseException as exc:
                if (
                    isinstance(exc, RuntimeError)
                    and stop_event.is_set()
                    and _is_event_loop_closed_error(exc)
                ):
                    return
                thread_errors.append(exc)
                # Thread targets cannot raise back into the consumer, so
                # record every terminal outcome before waking it. Preserve
                # process-level interrupts in the worker after recording
                # them; the async side still observes an abnormal outcome.
                if isinstance(exc, KeyboardInterrupt | SystemExit):
                    try:
                        finish_stream()
                    finally:
                        raise
                finish_stream()

        thread = Thread(
            target=generate_stream,
            name=f"{self._model_id}/generate_stream",
            daemon=True,
        )
        thread.start()

        _l(f"Generation thread #{thread.ident} ({thread.name}) started")

        shutdown_error: _TextGenerationWorkerShutdownError | None = None

        async def require_stream_thread_stopped(
            *, repeat_error: bool = True
        ) -> None:
            nonlocal shutdown_error
            if shutdown_error is not None:
                if repeat_error:
                    raise shutdown_error
                return
            await self._wait_for_stream_thread(thread)
            if not thread.is_alive():
                return
            shutdown_error = _TextGenerationWorkerShutdownError(
                "generation worker did not terminate during stream close"
            )
            raise shutdown_error

        async def events() -> AsyncGenerator[StreamProviderEvent, None]:
            parser = (
                structured_output_protocol.parser()
                if isinstance(
                    structured_output_protocol,
                    LocalStructuredOutputProtocol,
                )
                else LocalTextStreamEventParser(parse_tool_calls=False)
            )
            stream = cast(AsyncIterator[str], streamer)
            parser_flushed = False
            try:
                while True:
                    try:
                        chunk = await stream.__anext__()
                        if chunk:
                            for event in parser.push(chunk):
                                yield event
                        if (
                            thread_errors
                            and not thread.is_alive()
                            and not _streamer_has_pending_items(streamer)
                        ):
                            break
                    except TimeoutError:
                        if not thread.is_alive():
                            break
                    except StopAsyncIteration:
                        break

                await require_stream_thread_stopped()
                if thread_errors:
                    worker_error = thread_errors[0]
                    if isinstance(worker_error, CancelledError | Exception):
                        raise worker_error
                    raise RuntimeError(
                        "generation worker terminated abnormally"
                    ) from worker_error

                flushed_events = parser.flush(completed=True)
                parser_flushed = True
                for event in flushed_events:
                    yield event
            except GeneratorExit:
                stop_event.set()
                if not parser_flushed:
                    parser.flush(completed=False)
                    parser_flushed = True
                raise
            except (CancelledError, Exception):
                stop_event.set()
                if not parser_flushed:
                    flushed_events = parser.flush(completed=False)
                    parser_flushed = True
                    for event in flushed_events:
                        yield event
                raise
            finally:
                stop_event.set()
                await require_stream_thread_stopped()

            _l(f"Generation thread #{thread.ident} ({thread.name}) finished")

        canonical_stream = _canonical_transformers_stream(
            events(),
            max_queue_depth=queue_size,
            supports_tool_calls=(structured_output_protocol is not None),
        )
        try:
            async for item in canonical_stream:
                yield item
        except (CancelledError, GeneratorExit):
            await cast(Any, canonical_stream).aclose()
            raise
        finally:
            stop_event.set()
            await require_stream_thread_stopped(repeat_error=False)

    @staticmethod
    async def _wait_for_stream_thread(thread: Thread) -> None:
        deadline = perf_counter() + _STREAM_THREAD_JOIN_TIMEOUT_SECONDS
        while thread.is_alive():
            remaining = deadline - perf_counter()
            if remaining <= 0:
                return
            await sleep(min(_STREAMER_TIMEOUT_SECONDS, remaining))

    def _string_output(
        self,
        inputs: dict[str, Tensor] | Tensor,
        settings: GenerationSettings,
        stopping_criterias: list[StoppingCriteria] | None,
        skip_special_tokens: bool,
        local_structured_output_protocol: (
            LocalStructuredOutputProtocol | None
        ) = None,
        **kwargs: object,
    ) -> str | TextGenerationNonStreamResult:
        assert isinstance(inputs, dict)
        input_length = inputs["input_ids"].shape[1]
        outputs = self._generate_output(inputs, settings, stopping_criterias)
        text = cast(
            str,
            self._tokenizer.decode(
                outputs[0][input_length:],
                skip_special_tokens=skip_special_tokens,
            ),
        )
        if local_structured_output_protocol is None:
            return text
        assert (
            local_structured_output_protocol
            is LOCAL_STRUCTURED_OUTPUT_PROTOCOL
        )
        return local_structured_output_protocol.non_stream_result(
            text,
            provider_family="transformers",
            provider_event_type="transformers.generate",
        )

    async def _token_generator(
        self,
        inputs: dict[str, Tensor] | Tensor,
        settings: GenerationSettings,
        stopping_criterias: list[StoppingCriteria] | None,
        skip_special_tokens: bool,
        pick: int | None,
        probability_distribution: ProbabilityDistribution = "softmax",
        local_structured_output_protocol: (
            LocalStructuredOutputProtocol | None
        ) = None,
        **kwargs: object,
    ) -> AsyncGenerator[CanonicalStreamItem, None]:
        async for item in _canonical_transformers_stream(
            self._token_provider_events(
                inputs,
                settings,
                stopping_criterias,
                skip_special_tokens,
                pick,
                probability_distribution,
                local_structured_output_protocol,
            ),
            supports_tool_calls=(local_structured_output_protocol is not None),
        ):
            yield item

    async def _token_provider_events(
        self,
        inputs: dict[str, Tensor] | Tensor,
        settings: GenerationSettings,
        stopping_criterias: list[StoppingCriteria] | None,
        skip_special_tokens: bool,
        pick: int | None,
        probability_distribution: ProbabilityDistribution = "softmax",
        local_structured_output_protocol: (
            LocalStructuredOutputProtocol | None
        ) = None,
    ) -> AsyncGenerator[StreamProviderEvent, None]:
        assert isinstance(inputs, dict)
        assert not settings.temperature or (
            settings.temperature >= 0 and settings.temperature <= 1
        ), "temperature should be [0, 1]"
        assert not pick or pick >= 0
        temperature = (
            settings.temperature
            if settings.temperature is not None and settings.temperature > 0
            else 1.0
        )

        _l = self._log

        enable_entmax = find_spec("entmax") and probability_distribution in [
            "entmax",
            "sparsemax",
        ]
        if enable_entmax:
            import entmax

        _l(
            f"Generating up to {settings.max_new_tokens} tokens "
            f"{'with' if settings.do_sample else 'without'} sample "
            f"and {settings.temperature} temperature"
        )

        generation_settings = replace(
            settings,
            return_dict_in_generate=True,
            output_scores=True,
        )
        outputs = self._generate_output(
            inputs, generation_settings, stopping_criterias
        )
        sequences = outputs.sequences[0]
        scores = outputs.scores  # list of logits for each generated token
        start = inputs["input_ids"].shape[1]  # where generation began
        generated_sequences = sequences[start:]

        _l(f"Generated {len(generated_sequences)} sequences")

        assert local_structured_output_protocol is None or (
            local_structured_output_protocol
            is LOCAL_STRUCTURED_OUTPUT_PROTOCOL
        )
        parser = (
            local_structured_output_protocol.parser()
            if local_structured_output_protocol is not None
            else LocalTextStreamEventParser(parse_tool_calls=False)
        )
        total_tokens = 0
        try:
            for step, token_id in enumerate(generated_sequences):
                _l(f"Got step {step} token {token_id}")

                # logits are the raw-unnormalized scores output by the final
                # linear layer
                tensor = scores[step]  # scores is (batch_size, vocab_size)
                logits = tensor[0]  # first element in batch dimension

                # Normalize the final logits into token probabilities.
                logits_probs = (
                    log_softmax(logits, dim=-1)
                    if probability_distribution == "log_softmax"
                    else (
                        gumbel_softmax(
                            logits,
                            tau=temperature,
                            hard=False,
                            dim=-1,
                        )
                        if probability_distribution == "gumbel_softmax"
                        else (
                            entmax.sparsemax(logits, dim=-1)
                            if enable_entmax
                            and probability_distribution == "sparsemax"
                            else (
                                entmax.entmax15(logits, dim=-1)
                                if enable_entmax
                                and probability_distribution == "entmax"
                                else softmax(logits / temperature, dim=-1)
                            )
                        )
                    )
                )

                token_id_value = _non_negative_token_id(token_id)
                token_decode_id = (
                    token_id_value if token_id_value is not None else token_id
                )
                token_text = self._tokenizer.decode(
                    token_decode_id,
                    skip_special_tokens=skip_special_tokens,
                )

                candidate_metadata: (
                    list[tuple[str, int | None, float | None]] | None
                ) = None
                pick_count = pick or 0
                if pick_count > 0:
                    picked_logits = topk(logits_probs, pick_count)
                    picked_logits_ids = picked_logits.indices.tolist()
                    picked_logits_probs = picked_logits.values.tolist()
                    candidate_metadata = []
                    for i, candidate_id in enumerate(picked_logits_ids):
                        candidate_token_id = _non_negative_token_id(
                            candidate_id
                        )
                        candidate_decode_id = (
                            candidate_token_id
                            if candidate_token_id is not None
                            else candidate_id
                        )
                        candidate_text = self._tokenizer.decode(
                            candidate_decode_id,
                            skip_special_tokens=skip_special_tokens,
                        )
                        if not candidate_text:
                            continue
                        candidate_metadata.append(
                            (
                                candidate_text,
                                candidate_token_id,
                                _probability_value(picked_logits_probs[i]),
                            )
                        )

                metadata = stream_token_metadata(
                    token_id=token_id_value,
                    probability=_probability_value(logits_probs[token_id]),
                    step=step,
                    probability_distribution=probability_distribution,
                    candidates=(
                        tuple(candidate_metadata)
                        if candidate_metadata is not None
                        else None
                    ),
                )

                _l(
                    f"Yielding step {step} token metadata "
                    f"{metadata.__repr__()}"
                )

                if token_text:
                    for event in parser.push(token_text, metadata):
                        yield self._local_event_with_provider_type(
                            event,
                            provider_event_type="transformers.token",
                        )

                total_tokens = total_tokens + 1
        except (CancelledError, Exception):
            for event in parser.flush(completed=False):
                yield self._local_event_with_provider_type(
                    event,
                    provider_event_type="transformers.token",
                )
            raise

        for event in parser.flush():
            yield self._local_event_with_provider_type(
                event,
                provider_event_type="transformers.token",
            )

        _l(f"Yielded {total_tokens}")

        await sleep(0)  # and just like that, a generator is an async generator

    @staticmethod
    def _local_event_with_provider_type(
        event: StreamProviderEvent,
        *,
        provider_event_type: str,
    ) -> StreamProviderEvent:
        return replace(
            event,
            provider_event_type=event.provider_event_type
            or provider_event_type,
        )

    @staticmethod
    def _provider_tool_schemas(
        capability: ModelCapabilityCatalog | None,
    ) -> object | None:
        if capability is None or not capability.structured_parser_enabled:
            return None
        projection = capability.project(ProviderFamily.LOCAL.value)
        return None if projection.is_empty else projection.schemas

    def _effective_local_capability(
        self,
        capability: ModelCapabilityCatalog | None,
        *,
        chat_template: str | None = None,
    ) -> ModelCapabilityCatalog | None:
        if capability is None or not capability.structured_parser_enabled:
            return None
        if not self._tokenizer_supports_structured_capabilities(chat_template):
            return None
        return capability

    def _tokenizer_supports_structured_capabilities(
        self, chat_template: str | None = None
    ) -> bool:
        return (
            LOCAL_STRUCTURED_OUTPUT_PROTOCOL.tokenizer_template(
                self._tokenizer,
                chat_template,
            )
            is not None
        )

    def _tokenize_input(
        self,
        input: Input,
        system_prompt: str | None = None,
        developer_prompt: str | None = None,
        context: str | None = None,
        tensor_format: Literal["pt"] = "pt",
        chat_template: str | None = None,
        chat_template_settings: dict[str, object] | None = None,
        capability: ModelCapabilityCatalog | None = None,
        instructions: str | None = None,
    ) -> dict[str, Tensor] | BatchEncoding | Tensor:
        _l = self._log
        capability = self._effective_local_capability(
            capability,
            chat_template=chat_template,
        )
        structured_output_protocol = (
            LOCAL_STRUCTURED_OUTPUT_PROTOCOL
            if capability is not None
            else None
        )
        structured_output_template = (
            structured_output_protocol.tokenizer_template(
                self._tokenizer,
                chat_template,
            )
            if structured_output_protocol is not None
            else None
        )
        messages = self._messages(
            input, system_prompt, developer_prompt, capability
        )
        has_chat_template = self._tokenizer_has_chat_template()

        def _format_content(
            content: str | MessageContent | list[MessageContent] | None,
        ) -> str | list[dict[str, object]]:
            if content is None:
                return ""
            if isinstance(content, str):
                return content

            if isinstance(content, MessageContentText):
                return content.text

            if isinstance(content, MessageContentImage):
                if has_chat_template:
                    return [
                        {"type": "image_url", "image_url": content.image_url}
                    ]
                return ""

            if isinstance(content, MessageContentFile):
                return ""

            if isinstance(content, list):
                if has_chat_template:
                    blocks: list[dict[str, object]] = []
                    for c in content:
                        if isinstance(c, MessageContentImage):
                            blocks.append(
                                {
                                    "type": "image_url",
                                    "image_url": c.image_url,
                                }
                            )
                        elif isinstance(c, MessageContentFile):
                            continue
                        else:
                            assert isinstance(c, MessageContentText)
                            blocks.append({"type": "text", "text": c.text})
                    return blocks

                texts = [
                    c.text
                    for c in content
                    if isinstance(c, MessageContentText)
                ]
                return "\n".join(texts)

            return str(content)

        template_messages = []
        for message in messages:
            message_dict = asdict(message)
            if message_dict.get("tool_call_diagnostic") is None:
                message_dict.pop("tool_call_diagnostic", None)
            prepared = _TOOL_MESSAGE_PARSER.prepare_message_for_template(
                message, message_dict
            )
            message_dict = prepared.message_dict
            template_content = prepared.template_content
            template_messages.append(
                {
                    **message_dict,
                    **{"content": _format_content(template_content)},
                }
            )

        schemas = self._provider_tool_schemas(capability)
        if structured_output_protocol is not None:
            assert structured_output_template is not None
            assert isinstance(schemas, tuple)
            template_messages = structured_output_protocol.prepare_messages(
                template_messages,
                schemas,
            )

        if not has_chat_template:
            _l("Model does not support template messages, so staying plain")

            prompt = f"{instructions}\n\n" if instructions else ""
            prompt += f"{system_prompt}\n\n" or ""
            use_prefix = not any(
                tm["role"] == MessageRole.USER for tm in template_messages
            )

            for template_message in template_messages:
                if use_prefix:
                    prompt += (
                        "User: "
                        if template_message["role"] == MessageRole.USER
                        else (
                            "Assistant: "
                            if template_message["role"]
                            == MessageRole.ASSISTANT
                            else ""
                        )
                    )
                content_text = template_message["content"]
                prompt += (
                    content_text.strip()
                    if isinstance(content_text, str)
                    else str(content_text)
                ) + "\n"

            inputs = self._tokenize_prompt(prompt, tensor_format)
        else:
            _l(f"Got {len(template_messages)} template messages")

            _l(f"Applying chat template to {len(template_messages)} messages")
            template_kwargs = {
                **(chat_template_settings or {}),
                "chat_template": (
                    structured_output_template
                    if structured_output_template is not None
                    else chat_template
                ),
                "return_tensors": tensor_format,
            }
            if structured_output_protocol is not None:
                assert schemas is not None
                template_kwargs["tools"] = schemas
            inputs = self._tokenizer.apply_chat_template(
                template_messages,
                **template_kwargs,
            )
            inputs = self._normalize_tokenized_inputs(inputs, tensor_format)

        if hasattr(self._model, "device"):
            _l(f"Translating inputs to {self._model.device}")
            inputs = self._move_inputs_to_device(inputs, self._model.device)
        return inputs

    def _tokenizer_has_chat_template(self) -> bool:
        chat_template = getattr(self._tokenizer, "chat_template", None)
        has_chat_template = getattr(
            self._tokenizer, "has_chat_template", False
        )
        return bool(chat_template) or has_chat_template is True

    def _tokenize_prompt(
        self,
        prompt: str,
        tensor_format: Literal["pt"] = "pt",
    ) -> dict[str, Tensor] | BatchEncoding | Tensor:
        if callable(self._tokenizer):
            return cast(
                dict[str, Tensor] | BatchEncoding | Tensor,
                self._tokenizer(
                    prompt,
                    add_special_tokens=True,
                    return_tensors=tensor_format,
                ),
            )

        encode = getattr(self._tokenizer, "encode", None)
        if callable(encode):
            return self._inputs_from_token_ids(
                encode(prompt, add_special_tokens=True), tensor_format
            )

        raise TypeError(
            f"Tokenizer {self._tokenizer.__class__.__name__} is not callable "
            + "and does not provide encode()."
        )

    @classmethod
    def _normalize_tokenized_inputs(
        cls,
        inputs: Any,
        tensor_format: Literal["pt"] = "pt",
    ) -> dict[str, Tensor] | BatchEncoding | Tensor:
        if isinstance(inputs, (list, tuple)):
            return cls._inputs_from_token_ids(inputs, tensor_format)
        return cast(dict[str, Tensor] | BatchEncoding | Tensor, inputs)

    @staticmethod
    def _inputs_from_token_ids(
        token_ids: Any,
        tensor_format: Literal["pt"] = "pt",
    ) -> dict[str, Tensor]:
        input_ids = token_ids
        if not (
            isinstance(input_ids, (list, tuple))
            and input_ids
            and isinstance(input_ids[0], (list, tuple))
        ):
            input_ids = [input_ids]

        if tensor_format == "pt":
            tensor = getattr(import_module("torch"), "tensor")
            input_ids = tensor(input_ids)

        return {"input_ids": cast(Tensor, input_ids)}

    @staticmethod
    def _move_inputs_to_device(
        inputs: dict[str, Tensor] | BatchEncoding | Tensor,
        device: Any,
    ) -> dict[str, Tensor] | BatchEncoding | Tensor:
        if hasattr(inputs, "to"):
            return cast(
                dict[str, Tensor] | BatchEncoding | Tensor, inputs.to(device)
            )

        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    def _messages(
        self,
        input: Input,
        system_prompt: str | None,
        developer_prompt: str | None = None,
        capability: ModelCapabilityCatalog | None = None,
    ) -> list[Message]:
        _ = capability
        if isinstance(input, str):
            input = Message(role=MessageRole.USER, content=input)
        elif isinstance(input, list):
            if input and isinstance(input[0], str):
                input = [
                    Message(role=MessageRole.USER, content=m) for m in input
                ]
            else:
                for m in input:
                    assert isinstance(m, Message)
        elif not isinstance(input, Message):
            raise ValueError(input)

        messages: list[Message]
        if isinstance(input, list):
            messages = cast(list[Message], input)
        else:
            messages = [input]

        if developer_prompt:
            messages = [
                Message(role=MessageRole.DEVELOPER, content=developer_prompt)
            ] + messages
        if system_prompt:
            messages = [
                Message(role=MessageRole.SYSTEM, content=system_prompt)
            ] + messages

        assert isinstance(messages, list)
        return messages
