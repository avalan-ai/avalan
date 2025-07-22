from ...compat import override
from ...entities import Input
from ...model.vendor import TextGenerationVendor
from ...model.nlp import BaseNLPModel
from ...model.engine import Engine
from diffusers import DiffusionPipeline
from torch import argmax, inference_mode
from transformers import AutoModelForTokenClassification, PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding
from typing import Literal


class TokenClassificationModel(BaseNLPModel):
    _default_label_id: int | None = None

    @property
    def supports_sample_generation(self) -> bool:
        return False

    @property
    def supports_token_streaming(self) -> bool:
        return False

    def _load_model(
        self,
    ) -> PreTrainedModel | TextGenerationVendor | DiffusionPipeline:
        model = AutoModelForTokenClassification.from_pretrained(
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
        )
        labels = (
            getattr(model.config, "id2label", None)
            if hasattr(model, "config")
            else None
        )
        if labels:
            default_label_ids = {
                lbl_id for lbl_id, lbl in labels.items() if "-" not in lbl
            }
            self._default_label_id = (
                next(iter(default_label_ids)) if default_label_ids else None
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
    async def __call__(
        self,
        input: Input,
        *,
        labeled_only: bool = False,
        system_prompt: str | None = None,
    ) -> dict[str, str]:
        assert self._tokenizer, (
            f"Model {self._model} can't be executed "
            + "without a tokenizer loaded first"
        )
        assert self._model, (
            f"Model {self._model} can't be executed, it "
            + "needs to be loaded first"
        )
        inputs = self._tokenize_input(
            input, system_prompt=system_prompt, context=None
        )
        with inference_mode():
            outputs = self._model(**inputs)
            # logits shape (1, seq_len, num_labels)
            input_ids = inputs["input_ids"][0]
            label_ids = argmax(outputs.logits, dim=2)[0]

            if labeled_only and self._default_label_id is not None:
                mask = label_ids != self._default_label_id
                input_ids = input_ids[mask]
                label_ids = label_ids[mask]

            assert input_ids.numel() == label_ids.numel()
            tokens = self._tokenizer.convert_ids_to_tokens(input_ids)
            labels = [
                self._model.config.id2label[label_id.item()]
                for label_id in label_ids
            ]
            tokens_to_labels = dict(zip(tokens, labels))
            return tokens_to_labels
