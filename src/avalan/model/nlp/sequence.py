from ...entities import GenerationSettings, Input
from ...model.engine import Engine
from ...model.nlp import BaseNLPModel
from ...model.vendor import TextGenerationVendor

from dataclasses import replace
from typing import Any, Literal, cast

from diffusers import DiffusionPipeline
from torch import Tensor, argmax, inference_mode, softmax
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    StoppingCriteria,
)
from transformers.tokenization_utils_base import BatchEncoding


class SequenceClassificationModel(BaseNLPModel):
    @property
    def supports_sample_generation(self) -> bool:
        return False

    @property
    def supports_token_streaming(self) -> bool:
        return False

    def _load_model(
        self,
    ) -> PreTrainedModel | TextGenerationVendor | DiffusionPipeline:
        assert self._model_id, "A model id is required."
        settings = cast(Any, self._settings)
        model = AutoModelForSequenceClassification.from_pretrained(
            self._model_id,
            cache_dir=settings.cache_dir,
            subfolder=settings.subfolder or "",
            attn_implementation=settings.attention,
            trust_remote_code=settings.trust_remote_code,
            torch_dtype=Engine.weight(settings.weight_type),
            state_dict=settings.state_dict,
            local_files_only=settings.local_files_only,
            token=settings.access_token,
            device_map=self._device,
            tp_plan=Engine._get_tp_plan(settings.parallel),
            distributed_config=Engine._get_distributed_config(
                settings.distributed_config
            ),
        )
        return cast(PreTrainedModel, model)

    def _tokenize_input(
        self,
        input: Input,
        system_prompt: str | None,
        developer_prompt: str | None = None,
        context: str | None = None,
        tensor_format: Literal["pt"] = "pt",
        chat_template_settings: dict[str, object] | None = None,
        **kwargs: object,
    ) -> BatchEncoding:
        assert not system_prompt and not developer_prompt, (
            "Sequence classification model "
            + f"{self._model_id} does not support chat "
            + "templates"
        )
        _l = self._log
        _l(f"Tokenizing input {input}")
        tokenizer = cast(
            PreTrainedTokenizer | PreTrainedTokenizerFast, self._tokenizer
        )
        model = cast(PreTrainedModel, self._model)
        inputs = tokenizer(input, return_tensors=tensor_format)
        return cast(BatchEncoding, inputs.to(model.device))

    async def __call__(self, input: Input) -> str:
        assert self._tokenizer, (
            f"Model {self._model} can't be executed "
            + "without a tokenizer loaded first"
        )
        assert self._model, (
            f"Model {self._model} can't be executed, it "
            + "needs to be loaded first"
        )
        model = cast(PreTrainedModel, self._model)
        inputs = self._tokenize_input(input, system_prompt=None, context=None)
        with inference_mode():
            outputs = model(**inputs)
            # logits shape (batch_size, num_labels)
            label_probs = softmax(outputs.logits, dim=-1)
            label_id = int(argmax(label_probs, dim=-1).item())
            id_to_label = cast(dict[int, str], model.config.id2label or {})
            label = id_to_label.get(label_id, str(label_id))
            return label


class SequenceToSequenceModel(BaseNLPModel):
    @property
    def supports_sample_generation(self) -> bool:
        return False

    @property
    def supports_token_streaming(self) -> bool:
        return False

    def _load_model(
        self,
    ) -> PreTrainedModel | TextGenerationVendor | DiffusionPipeline:
        assert self._model_id, "A model id is required."
        settings = cast(Any, self._settings)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self._model_id,
            cache_dir=settings.cache_dir,
            subfolder=settings.subfolder or "",
            attn_implementation=settings.attention,
            trust_remote_code=settings.trust_remote_code,
            torch_dtype=Engine.weight(settings.weight_type),
            state_dict=settings.state_dict,
            local_files_only=settings.local_files_only,
            token=settings.access_token,
            device_map=self._device,
            tp_plan=Engine._get_tp_plan(settings.parallel),
            distributed_config=Engine._get_distributed_config(
                settings.distributed_config
            ),
        )
        return cast(PreTrainedModel, model)

    def _tokenize_input(
        self,
        input: Input,
        system_prompt: str | None,
        developer_prompt: str | None = None,
        context: str | None = None,
        tensor_format: Literal["pt"] = "pt",
        chat_template_settings: dict[str, object] | None = None,
        **kwargs: object,
    ) -> Tensor:
        assert not system_prompt and not developer_prompt, (
            "SequenceToSequence model "
            + f"{self._model_id} does not support chat "
            + "templates"
        )
        _l = self._log
        _l(f"Tokenizing input {input}")
        tokenizer = cast(
            PreTrainedTokenizer | PreTrainedTokenizerFast, self._tokenizer
        )
        model = cast(PreTrainedModel, self._model)
        inputs = tokenizer(input, return_tensors=tensor_format)
        model_inputs = cast(BatchEncoding, inputs.to(model.device))
        return model_inputs["input_ids"]

    async def __call__(
        self,
        input: Input,
        settings: GenerationSettings,
        stopping_criterias: list[StoppingCriteria] | None = None,
    ) -> str:
        assert self._tokenizer, (
            f"Model {self._model} can't be executed "
            + "without a tokenizer loaded first"
        )
        assert self._model, (
            f"Model {self._model} can't be executed, it "
            + "needs to be loaded first"
        )
        assert settings.temperature is None or (
            settings.temperature > 0 and settings.temperature != 0.0
        ), (
            "Temperature has to be a strictly positive float, otherwise "
            + "your next token scores will be invalid"
        )

        inputs = self._tokenize_input(input, system_prompt=None, context=None)
        output_ids = self._generate_output(
            inputs,
            settings,
            stopping_criterias,
        )
        return cast(
            str,
            self._tokenizer.decode(output_ids[0], skip_special_tokens=True),
        )


class TranslationModel(SequenceToSequenceModel):
    async def __call__(
        self,
        input: Input,
        source_language: str,
        destination_language: str,
        settings: GenerationSettings,
        stopping_criterias: list[StoppingCriteria] | None = None,
        skip_special_tokens: bool = True,
    ) -> str:
        assert self._tokenizer, (
            f"Model {self._model} can't be executed "
            + "without a tokenizer loaded first"
        )
        assert self._model, (
            f"Model {self._model} can't be executed, it "
            + "needs to be loaded first"
        )
        assert settings.temperature is None or (
            settings.temperature > 0 and settings.temperature != 0.0
        ), (
            "Temperature has to be a strictly positive float, otherwise "
            + "your next token scores will be invalid"
        )
        assert hasattr(self._tokenizer, "src_lang") and hasattr(
            self._tokenizer, "lang_code_to_id"
        )

        previous_language = self._tokenizer.src_lang
        self._tokenizer.src_lang = source_language
        inputs = self._tokenize_input(input, system_prompt=None, context=None)
        generation_settings = replace(
            settings,
            early_stopping=True,
            repetition_penalty=1.0,
            use_cache=True,
            temperature=None,
            forced_bos_token_id=self._tokenizer.lang_code_to_id[
                destination_language
            ],
        )
        output_ids = self._generate_output(
            inputs,
            generation_settings,
            stopping_criterias,
        )
        text = cast(
            str,
            self._tokenizer.decode(
                output_ids[0], skip_special_tokens=skip_special_tokens
            ),
        )
        self._tokenizer.src_lang = previous_language
        return text
