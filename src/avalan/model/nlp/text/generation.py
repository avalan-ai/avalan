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
from ....model.engine import Engine
from ....model.nlp import BaseNLPModel
from ....model.response.text import TextGenerationResponse
from ....model.stream import (
    CanonicalStreamItem,
    LocalTextStreamEventParser,
    StreamItemKind,
    StreamProducerBackend,
    StreamProviderCapabilities,
    StreamProviderEvent,
    normalize_provider_stream,
    stream_token_metadata,
)
from ....model.vendor import TextGenerationVendor
from ....tool.manager import ToolManager
from ....tool.parser import ToolCallParser

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


def _local_stream_capabilities(
    provider_family: str,
    *,
    max_queue_depth: int | None = None,
) -> StreamProviderCapabilities:
    return StreamProviderCapabilities(
        backend=StreamProducerBackend.LOCAL,
        provider_family=provider_family,
        supports_reasoning=True,
        supports_tool_calls=True,
        supports_cancellation=True,
        max_queue_depth=max_queue_depth,
    )


async def _canonical_transformers_stream(
    events: AsyncIterable[StreamProviderEvent],
    *,
    max_queue_depth: int | None = None,
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
        tool: ToolManager | None = None,
        **kwargs: object,
    ) -> TextGenerationResponse:
        assert self._tokenizer, (
            f"Model {self._model} can't be executed "
            + "without a tokenizer loaded first"
        )
        assert self._model, (
            f"Model {self._model} can't be executed, it "
            + "needs to be loaded first"
        )

        if not settings:
            settings = GenerationSettings()
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
            tool=tool,
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
            except RuntimeError as exc:
                if stop_event.is_set() and _is_event_loop_closed_error(exc):
                    return
                thread_errors.append(exc)
                finish_stream()
            except Exception as exc:
                # Thread targets cannot raise back into the consumer, so
                # capture worker failures and wake the async iterator.
                thread_errors.append(exc)
                finish_stream()

        thread = Thread(
            target=generate_stream,
            name=f"{self._model_id}/generate_stream",
            daemon=True,
        )
        thread.start()

        _l(f"Generation thread #{thread.ident} ({thread.name}) started")

        async def events() -> AsyncGenerator[StreamProviderEvent, None]:
            parser = LocalTextStreamEventParser()
            stream = cast(AsyncIterator[str], streamer)
            try:
                while True:
                    if thread_errors:
                        raise thread_errors[0]
                    try:
                        chunk = await stream.__anext__()
                        if thread_errors:
                            raise thread_errors[0]
                        if chunk:
                            for event in parser.push(chunk):
                                yield event
                    except TimeoutError:
                        if thread_errors:
                            raise thread_errors[0]
                        if not thread.is_alive():
                            break
                    except StopAsyncIteration:
                        break

                if thread_errors:
                    raise thread_errors[0]

                for event in parser.flush():
                    yield event
            except (CancelledError, GeneratorExit):
                stop_event.set()
                raise
            finally:
                stop_event.set()
                await self._wait_for_stream_thread(thread)

            _l(f"Generation thread #{thread.ident} ({thread.name}) finished")

        canonical_stream = _canonical_transformers_stream(
            events(),
            max_queue_depth=queue_size,
        )
        try:
            async for item in canonical_stream:
                yield item
        except (CancelledError, GeneratorExit):
            await cast(Any, canonical_stream).aclose()
            raise
        finally:
            stop_event.set()
            await self._wait_for_stream_thread(thread)

    @staticmethod
    async def _wait_for_stream_thread(thread: Thread) -> None:
        deadline = perf_counter() + _STREAM_THREAD_JOIN_TIMEOUT_SECONDS
        while thread.is_alive() and perf_counter() < deadline:
            await sleep(_STREAMER_TIMEOUT_SECONDS)

    def _string_output(
        self,
        inputs: dict[str, Tensor] | Tensor,
        settings: GenerationSettings,
        stopping_criterias: list[StoppingCriteria] | None,
        skip_special_tokens: bool,
        **kwargs: object,
    ) -> str:
        assert isinstance(inputs, dict)
        input_length = inputs["input_ids"].shape[1]
        outputs = self._generate_output(inputs, settings, stopping_criterias)
        return cast(
            str,
            self._tokenizer.decode(
                outputs[0][input_length:],
                skip_special_tokens=skip_special_tokens,
            ),
        )

    async def _token_generator(
        self,
        inputs: dict[str, Tensor] | Tensor,
        settings: GenerationSettings,
        stopping_criterias: list[StoppingCriteria] | None,
        skip_special_tokens: bool,
        pick: int | None,
        probability_distribution: ProbabilityDistribution = "softmax",
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
            ),
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

        total_tokens = 0
        for step, token_id in enumerate(generated_sequences):
            _l(f"Got step {step} token {token_id}")

            # logits are the raw-unnormalized scores output by the final
            # linear layer
            tensor = scores[step]  # scores is (batch_size, vocab_size)
            logits = tensor[0]  # first element in batch dimension

            # apply probabilty  distribution over last tensor layer, vocab_size
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
                    candidate_token_id = _non_negative_token_id(candidate_id)
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

            _l(f"Yielding step {step} token metadata {metadata.__repr__()}")

            if token_text:
                yield StreamProviderEvent(
                    kind=StreamItemKind.ANSWER_DELTA,
                    text_delta=token_text,
                    metadata=metadata,
                    provider_event_type="transformers.token",
                )

            total_tokens = total_tokens + 1

        _l(f"Yielded {total_tokens}")

        await sleep(0)  # and just like that, a generator is an async generator

    def _tokenize_input(
        self,
        input: Input,
        system_prompt: str | None = None,
        developer_prompt: str | None = None,
        context: str | None = None,
        tensor_format: Literal["pt"] = "pt",
        chat_template: str | None = None,
        chat_template_settings: dict[str, object] | None = None,
        tool: ToolManager | None = None,
        instructions: str | None = None,
    ) -> dict[str, Tensor] | BatchEncoding | Tensor:
        _l = self._log
        messages = self._messages(input, system_prompt, developer_prompt, tool)
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
            inputs = self._tokenizer.apply_chat_template(
                template_messages,
                chat_template=chat_template,
                tools=tool.json_schemas() if tool else None,
                **(chat_template_settings or {}),
                return_tensors=tensor_format,
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
        tool: ToolManager | None = None,
    ) -> list[Message]:
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
