from ...compat import override
from ...model import TextGenerationVendor
from ...model.nlp import BaseNLPModel
from ...model.engine import Engine
from torch import argmax, inference_mode
from transformers import AutoModelForTokenClassification, PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding
from typing import Literal


class TokenClassificationModel(BaseNLPModel):
    @property
    def supports_sample_generation(self) -> bool:
        return False

    @property
    def supports_token_streaming(self) -> bool:
        return False

    def _load_model(self) -> PreTrainedModel | TextGenerationVendor:
        model = AutoModelForTokenClassification.from_pretrained(
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
        input: input,
        system_prompt: str | None,
        context: str | None,
        tensor_format: Literal["pt"] = "pt",
        chat_template_settings: dict[str, object] | None = None,
    ) -> BatchEncoding:
        assert not system_prompt, (
            "Token classification model "
            + f"{self._model_id} does not support chat "
            + "templates"
        )
        _l = self._log
        _l(f"Tokenizing input {input}")
        inputs = self._tokenizer(input, return_tensors=tensor_format)
        inputs = inputs.to(self._model.device)
        return inputs

    @override
    async def __call__(self, input: str) -> dict[str, str]:
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
            # logits shape (1, seq_len, num_labels)
            label_ids = argmax(outputs.logits, dim=2)
            tokens = self._tokenizer.convert_ids_to_tokens(
                inputs["input_ids"][0]
            )
            labels = [
                self._model.config.id2label[label_id.item()]
                for label_id in label_ids[0]
            ]
            tokens_to_labels = dict(zip(tokens, labels))
            return tokens_to_labels
