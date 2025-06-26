from ...compat import override
from ...entities import GenerationSettings, Input
from ...model import TextGenerationVendor
from ...model.nlp import BaseNLPModel
from ...model.engine import Engine
from dataclasses import replace
from torch import argmax, inference_mode, softmax, Tensor
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    PreTrainedModel,
)
from transformers.generation import StoppingCriteria
from transformers.tokenization_utils_base import BatchEncoding
from typing import Literal


class SequenceClassificationModel(BaseNLPModel):
    @property
    def supports_sample_generation(self) -> bool:
        return False

    @property
    def supports_token_streaming(self) -> bool:
        return False

    def _load_model(self) -> PreTrainedModel | TextGenerationVendor:
        model = AutoModelForSequenceClassification.from_pretrained(
            self._model_id,
            cache_dir=self._settings.cache_dir,
            attn_implementation=self._settings.attention,
            trust_remote_code=self._settings.trust_remote_code,
            torch_dtype=BaseNLPModel._get_weight_type(
                self._settings.weight_type
            ),
            state_dict=self._settings.state_dict,
            local_files_only=self._settings.local_files_only,
            token=self._settings.access_token,
            device_map=self._device,
            tp_plan=Engine._get_tp_plan(self._settings.parallel),
        )
        return model

    def _tokenize_input(
        self,
        input: Input,
        system_prompt: str | None,
        context: str | None,
        tensor_format: Literal["pt"] = "pt",
        chat_template_settings: dict[str, object] | None = None,
    ) -> BatchEncoding:
        assert not system_prompt, (
            "Sequence classification model "
            + f"{self._model_id} does not support chat "
            + "templates"
        )
        _l = self._log
        _l(f"Tokenizing input {input}")
        inputs = self._tokenizer(input, return_tensors=tensor_format)
        inputs = inputs.to(self._model.device)
        return inputs

    @override
    async def __call__(self, input: Input) -> str:
        assert self._tokenizer, (
            f"Model {self._model} can't be executed "
            + "without a tokenizer loaded first"
        )
        assert self._model, (
            f"Model {self._model} can't be executed, it "
            + "needs to be loaded first"
        )
        inputs = self._tokenize_input(input, system_prompt=None, context=None)
        with inference_mode():
            outputs = self._model(**inputs)
            # logits shape (batch_size, num_labels)
            label_probs = softmax(outputs.logits, dim=-1)
            label_id = argmax(label_probs, dim=-1).item()
            label = self._model.config.id2label[label_id]
            return label


class SequenceToSequenceModel(BaseNLPModel):
    @property
    def supports_sample_generation(self) -> bool:
        return False

    @property
    def supports_token_streaming(self) -> bool:
        return False

    def _load_model(self) -> PreTrainedModel | TextGenerationVendor:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self._model_id,
            cache_dir=self._settings.cache_dir,
            attn_implementation=self._settings.attention,
            trust_remote_code=self._settings.trust_remote_code,
            torch_dtype=BaseNLPModel._get_weight_type(
                self._settings.weight_type
            ),
            state_dict=self._settings.state_dict,
            local_files_only=self._settings.local_files_only,
            token=self._settings.access_token,
            device_map=self._device,
            tp_plan=Engine._get_tp_plan(self._settings.parallel),
        )
        return model

    def _tokenize_input(
        self,
        input: Input,
        system_prompt: str | None,
        context: str | None,
        tensor_format: Literal["pt"] = "pt",
        chat_template_settings: dict[str, object] | None = None,
    ) -> Tensor:
        assert not system_prompt, (
            "SequenceToSequence model "
            + f"{self._model_id} does not support chat "
            + "templates"
        )
        _l = self._log
        _l(f"Tokenizing input {input}")
        inputs = self._tokenizer(input, return_tensors=tensor_format)
        inputs = inputs.to(self._model.device)
        return inputs["input_ids"]

    @override
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
        return self._tokenizer.decode(output_ids[0], skip_special_tokens=True)


class TranslationModel(SequenceToSequenceModel):
    @override
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
        text = self._tokenizer.decode(
            output_ids[0], skip_special_tokens=skip_special_tokens
        )
        self._tokenizer.src_lang = previous_language
        return text
