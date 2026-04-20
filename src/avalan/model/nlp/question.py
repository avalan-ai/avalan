from ...entities import Input
from ...model.engine import Engine
from ...model.nlp import BaseNLPModel
from ...model.vendor import TextGenerationVendor

from typing import Any, Literal, cast

from diffusers import DiffusionPipeline
from torch import argmax, inference_mode
from transformers import (
    AutoModelForQuestionAnswering,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.tokenization_utils_base import BatchEncoding


class QuestionAnsweringModel(BaseNLPModel):
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
        model = AutoModelForQuestionAnswering.from_pretrained(
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
            "Token classification model "
            + f"{self._model_id} does not support chat "
            + "templates"
        )
        _l = self._log
        _l(f"Tokenizing input {input}")
        tokenizer = cast(
            PreTrainedTokenizer | PreTrainedTokenizerFast, self._tokenizer
        )
        model = cast(PreTrainedModel, self._model)
        inputs = tokenizer(input, context, return_tensors=tensor_format)
        return cast(BatchEncoding, inputs.to(model.device))

    async def __call__(
        self,
        input: Input,
        *,
        context: str,
        system_prompt: str | None = None,
        developer_prompt: str | None = None,
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
        model = cast(PreTrainedModel, self._model)
        inputs = self._tokenize_input(
            input,
            system_prompt=system_prompt,
            developer_prompt=developer_prompt,
            context=context,
        )
        with inference_mode():
            outputs = model(**inputs)
        start_answer_logits = outputs.start_logits
        end_answer_logits = outputs.end_logits
        start = argmax(start_answer_logits)
        end = argmax(end_answer_logits)
        answer_ids = inputs["input_ids"][0, start : end + 1]
        answer = cast(
            str,
            self._tokenizer.decode(
                answer_ids, skip_special_tokens=skip_special_tokens
            ),
        )
        return answer
