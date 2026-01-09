from ...compat import override
from ...entities import GenerationSettings, Input
from ...model.engine import Engine
from ...model.nlp import BaseNLPModel
from ...model.vendor import TextGenerationVendor

from dataclasses import replace
from typing import Any, Literal

from diffusers import DiffusionPipeline
from torch import Tensor, argmax, inference_mode, softmax
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    PreTrainedModel,
)
from transformers.generation import StoppingCriteria
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
        assert self._model_id is not None, "Model ID must be set"
        model: PreTrainedModel = (
            AutoModelForSequenceClassification.from_pretrained(
                self._model_id,
                cache_dir=self._settings.cache_dir,
                subfolder=self._settings.subfolder or "",
                attn_implementation=self._settings.attention,
                trust_remote_code=self._settings.trust_remote_code,
                torch_dtype=Engine.weight(self._settings.weight_type),
                state_dict=self._settings.state_dict,
                local_files_only=self._settings.local_files_only,
                token=self._settings.access_token,
                device_map=self._device,
                tp_plan=Engine._get_tp_plan(self._settings.parallel),
                distributed_config=Engine._get_distributed_config(
                    self._settings.distributed_config
                ),
            )
        )
        return model

    def _tokenize_input(  # type: ignore[override]
        self,
        input: Input,
        system_prompt: str | None,
        developer_prompt: str | None = None,
        context: str | None = None,
        tensor_format: Literal["pt"] = "pt",
        chat_template_settings: dict[str, object] | None = None,
    ) -> BatchEncoding:
        assert not system_prompt and not developer_prompt, (
            "Sequence classification model "
            + f"{self._model_id} does not support chat "
            + "templates"
        )
        assert self._tokenizer is not None, "Tokenizer must be loaded"
        assert self._model is not None, "Model must be loaded"
        _l = self._log
        _l(f"Tokenizing input {input}")
        inputs: BatchEncoding = self._tokenizer(
            input, return_tensors=tensor_format
        )
        inputs = inputs.to(self._model.device)  # type: ignore[union-attr]
        return inputs

    @override
    async def __call__(  # type: ignore[override]
        self, input: Input, **kwargs: Any
    ) -> str:
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
            outputs = self._model(**inputs)  # type: ignore[operator]
            # logits shape (batch_size, num_labels)
            label_probs = softmax(outputs.logits, dim=-1)  # type: ignore[union-attr]
            label_id = int(argmax(label_probs, dim=-1).item())
            id2label = self._model.config.id2label  # type: ignore[union-attr]
            assert id2label is not None, "Model config must have id2label"
            label: str = id2label[label_id]
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
        assert self._model_id is not None, "Model ID must be set"
        model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(
            self._model_id,
            cache_dir=self._settings.cache_dir,
            subfolder=self._settings.subfolder or "",
            attn_implementation=self._settings.attention,
            trust_remote_code=self._settings.trust_remote_code,
            torch_dtype=Engine.weight(self._settings.weight_type),
            state_dict=self._settings.state_dict,
            local_files_only=self._settings.local_files_only,
            token=self._settings.access_token,
            device_map=self._device,
            tp_plan=Engine._get_tp_plan(self._settings.parallel),
            distributed_config=Engine._get_distributed_config(
                self._settings.distributed_config
            ),
        )
        return model

    def _tokenize_input(  # type: ignore[override]
        self,
        input: Input,
        system_prompt: str | None,
        developer_prompt: str | None = None,
        context: str | None = None,
        tensor_format: Literal["pt"] = "pt",
        chat_template_settings: dict[str, object] | None = None,
    ) -> Tensor:
        assert not system_prompt and not developer_prompt, (
            "SequenceToSequence model "
            + f"{self._model_id} does not support chat "
            + "templates"
        )
        assert self._tokenizer is not None, "Tokenizer must be loaded"
        assert self._model is not None, "Model must be loaded"
        _l = self._log
        _l(f"Tokenizing input {input}")
        inputs: BatchEncoding = self._tokenizer(
            input, return_tensors=tensor_format
        )
        inputs = inputs.to(self._model.device)  # type: ignore[union-attr]
        input_ids: Tensor = inputs["input_ids"]
        return input_ids

    @override
    async def __call__(  # type: ignore[override]
        self,
        input: Input,
        settings: GenerationSettings,
        stopping_criterias: list[StoppingCriteria] | None = None,
        **kwargs: Any,
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
        return self._tokenizer.decode(output_ids[0], skip_special_tokens=True)  # type: ignore[return-value]


class TranslationModel(SequenceToSequenceModel):
    @override
    async def __call__(  # type: ignore[override]
        self,
        input: Input,
        source_language: str,
        destination_language: str,
        settings: GenerationSettings,
        stopping_criterias: list[StoppingCriteria] | None = None,
        skip_special_tokens: bool = True,
        **kwargs: Any,
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

        previous_language: str = self._tokenizer.src_lang  # type: ignore[attr-defined]
        self._tokenizer.src_lang = source_language  # type: ignore[attr-defined]
        inputs = self._tokenize_input(input, system_prompt=None, context=None)
        generation_settings = replace(
            settings,
            early_stopping=True,
            repetition_penalty=1.0,
            use_cache=True,
            temperature=None,
            forced_bos_token_id=self._tokenizer.lang_code_to_id[  # type: ignore[attr-defined]
                destination_language
            ],
        )
        output_ids = self._generate_output(
            inputs,
            generation_settings,
            stopping_criterias,
        )
        text: str = self._tokenizer.decode(
            output_ids[0], skip_special_tokens=skip_special_tokens
        )
        self._tokenizer.src_lang = previous_language  # type: ignore[attr-defined]
        return text
