from asyncio import sleep
from ....compat import override
from ....entities import (
    GenerationSettings,
    Input,
    Message,
    MessageRole,
    ProbabilityDistribution,
    TextGenerationLoaderClass,
    Token,
    TokenDetail,
    TransformerEngineSettings,
)
from ....model import TextGenerationResponse, TextGenerationVendor
from ....model.nlp import BaseNLPModel
from ....model.engine import Engine
from ....tool.manager import ToolManager
from dataclasses import replace
from importlib.util import find_spec
from logging import Logger
from threading import Thread
from torch import log_softmax, softmax, topk, Tensor
from torch.nn.functional import gumbel_softmax
from transformers import (
    AsyncTextIteratorStreamer,
    AutoModelForCausalLM,
    Gemma3ForConditionalGeneration,
    Mistral3ForConditionalGeneration,
    PreTrainedModel,
)
from transformers.generation import StoppingCriteria
from transformers.tokenization_utils_base import BatchEncoding
from typing import AsyncGenerator, Literal


class TextGenerationModel(BaseNLPModel):
    _loaders: dict[TextGenerationLoaderClass, type[PreTrainedModel]] = {
        "auto": AutoModelForCausalLM,
        "gemma3": Gemma3ForConditionalGeneration,
        "mistral3": Mistral3ForConditionalGeneration,
    }

    def __init__(
        self,
        model_id: str,
        settings: TransformerEngineSettings | None = None,
        logger: Logger | None = None,
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

    def _load_model(self) -> PreTrainedModel | TextGenerationVendor:
        assert (
            self._settings.loader_class in self._loaders
        ), f"Unrecognized loader {self._settings.loader_class}"

        if self._settings.quantization and find_spec("bitsandbytes"):
            from transformers import BitsAndBytesConfig

            quantization = self._settings.quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=quantization.load_in_4bit,
                bnb_4bit_quant_type=quantization.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=quantization.bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype=quantization.bnb_4bit_compute_dtype,
            )
        else:
            bnb_config = None

        loader = self._loaders[self._settings.loader_class]
        model = loader.from_pretrained(
            self._model_id,
            cache_dir=self._settings.cache_dir,
            attn_implementation=self._settings.attention,
            trust_remote_code=self._settings.trust_remote_code,
            state_dict=self._settings.state_dict,
            local_files_only=self._settings.local_files_only,
            low_cpu_mem_usage=(
                True if self._device else self._settings.low_cpu_mem_usage
            ),
            torch_dtype=BaseNLPModel._get_weight_type(
                self._settings.weight_type
            ),
            device_map=self._device,
            token=self._settings.access_token,
            quantization_config=bnb_config,
            revision=self._settings.revision,
            tp_plan=Engine._get_tp_plan(self._settings.parallel),
        )
        return model

    @override
    async def __call__(
        self,
        input: Input,
        system_prompt: str | None = None,
        settings: GenerationSettings | None = None,
        stopping_criterias: list[StoppingCriteria] | None = None,
        *,
        manual_sampling: bool = False,
        pick: int | None = None,
        skip_special_tokens: bool = False,
        tool: ToolManager | None = None,
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
            system_prompt,
            context=None,
            tool=tool,
            chat_template_settings=settings.chat_template_settings,
        )
        return TextGenerationResponse(
            output_fn,
            inputs=inputs,
            pick=pick,
            settings=generation_settings,
            stopping_criterias=stopping_criterias,
            skip_special_tokens=skip_special_tokens,
            use_async_generator=settings.use_async_generator,
        )

    async def _stream_generator(
        self,
        inputs: dict[str, Tensor] | Tensor,
        settings: GenerationSettings,
        stopping_criterias: list[StoppingCriteria] | None,
        skip_special_tokens: bool,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        _l = self._log

        streamer = AsyncTextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            decode_kwargs={"skip_special_tokens": skip_special_tokens},
        )

        _l("Created generator async text token streamer")

        def generate_stream() -> None:
            _l(
                f"Streaming up to {settings.max_new_tokens} tokens "
                f"{'with' if settings.do_sample else 'without'} sample "
                f"and {settings.temperature} temperature"
            )
            self._generate_output(
                inputs,
                settings,
                stopping_criterias,
                streamer=streamer,
            )

        thread = Thread(
            target=generate_stream, name=f"{self._model_id}/generate_stream"
        )
        thread.start()

        _l(f"Generation thread #{thread.ident} ({thread.name}) started")

        async for token in streamer:
            yield token

        thread.join()

        _l(f"Generation thread #{thread.ident} ({thread.name}) finished")

    def _string_output(
        self,
        inputs: dict[str, Tensor] | Tensor,
        settings: GenerationSettings,
        stopping_criterias: list[StoppingCriteria] | None,
        skip_special_tokens: bool,
        **kwargs,
    ) -> str:
        _l = self._log
        input_length = inputs["input_ids"].shape[1]
        outputs = self._generate_output(inputs, settings, stopping_criterias)
        response = self._tokenizer.decode(
            outputs[0][input_length:], skip_special_tokens=False
        )
        return response

    async def _token_generator(
        self,
        inputs: dict[str, Tensor] | Tensor,
        settings: GenerationSettings,
        stopping_criterias: list[StoppingCriteria] | None,
        skip_special_tokens: bool,
        pick: int | None,
        probability_distribution: ProbabilityDistribution = "softmax",
    ) -> AsyncGenerator[Token | TokenDetail, None]:
        assert not settings.temperature or (
            settings.temperature >= 0 and settings.temperature <= 1
        ), "temperature should be [0, 1]"
        assert not pick or pick >= 0

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
                        logits, tau=settings.temperature, hard=False, dim=-1
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
                            else softmax(logits / settings.temperature, dim=-1)
                        )
                    )
                )
            )

            tokens: list[Token] | None = None
            if pick > 0:
                picked_logits = topk(logits_probs, pick)
                picked_logits_ids = picked_logits.indices.tolist()
                picked_logits_probs = picked_logits.values.tolist()
                tokens = [
                    Token(
                        id=token_id,
                        token=self._tokenizer.decode(
                            token_id, skip_special_tokens=skip_special_tokens
                        ),
                        probability=picked_logits_probs[i],
                    )
                    for i, token_id in enumerate(picked_logits_ids)
                ]

            raw_token = TokenDetail(
                id=token_id,
                token=self._tokenizer.decode(
                    token_id, skip_special_tokens=skip_special_tokens
                ),
                probability=logits_probs[token_id].item(),
                step=step,
                probability_distribution=probability_distribution,
                tokens=tokens,
            )

            _l(f"Yielding step {step} token detail {raw_token.__repr__()}")

            yield raw_token

            total_tokens = total_tokens + 1

        _l(f"Yielded {total_tokens}")

        await sleep(0)  # and just like that, a generator is an async generator

    def _tokenize_input(
        self,
        input: Input,
        system_prompt: str | None,
        context: str | None,
        tensor_format: Literal["pt"] = "pt",
        chat_template: str | None = None,
        chat_template_settings: dict[str, object] | None = None,
        tool: ToolManager | None = None,
    ) -> dict[str, Tensor] | BatchEncoding | Tensor:
        _l = self._log
        messages = self._messages(input, system_prompt, tool)
        template_messages = [
            {"role": message.role, "content": message.content}
            for message in messages
        ]

        if not self._tokenizer.chat_template:
            _l("Model does not support template messages, so staying plain")

            prompt = f"{system_prompt}\n\n" or ""
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
                prompt += template_message["content"].strip() + "\n"

            inputs = self._tokenizer(
                prompt, add_special_tokens=True, return_tensors=tensor_format
            )
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

        _l(f"Translating inputs to {self._model.device}")
        inputs = inputs.to(self._model.device)
        return inputs

    def _messages(
        self,
        input: Input,
        system_prompt: str | None,
        tool: ToolManager | None = None,
    ) -> list[Message]:
        if isinstance(input, str):
            input = Message(role=MessageRole.USER, content=input)
        elif isinstance(input, list):
            for m in input:
                assert isinstance(m, Message)
        elif not isinstance(input, Message):
            raise ValueError(input)

        messages = [input] if not isinstance(input, list) else input

        if system_prompt:
            messages = [
                Message(role=MessageRole.SYSTEM, content=system_prompt)
            ] + messages

        assert isinstance(messages, list)
        return messages
