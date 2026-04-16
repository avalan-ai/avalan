from ...compat import override
from ...entities import Input
from ...model.engine import Engine
from ...model.nlp import BaseNLPModel
from ...model.vendor import TextGenerationVendor

from typing import Any, Literal, cast

from diffusers import DiffusionPipeline
from torch import argmax, inference_mode
from transformers import (
    AutoModelForTokenClassification,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.tokenization_utils_base import BatchEncoding


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
        assert self._model_id, "A model id is required."
        settings = cast(Any, self._settings)
        model = AutoModelForTokenClassification.from_pretrained(
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
        return cast(PreTrainedModel, model)

    def _tokenize_input(  # type: ignore[override]
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
        inputs = tokenizer(input, return_tensors=tensor_format)
        return cast(BatchEncoding, inputs.to(model.device))

    @override  # type: ignore[untyped-decorator]
    async def __call__(
        self,
        input: Input,
        *,
        labeled_only: bool = False,
        system_prompt: str | None = None,
        developer_prompt: str | None = None,
    ) -> dict[str, str]:
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
            context=None,
        )
        with inference_mode():
            outputs = model(**inputs)
            # logits shape (1, seq_len, num_labels)
            input_ids = inputs["input_ids"][0]
            label_ids = argmax(outputs.logits, dim=2)[0]

            if labeled_only and self._default_label_id is not None:
                mask = label_ids != self._default_label_id
                input_ids = input_ids[mask]
                label_ids = label_ids[mask]

            assert input_ids.numel() == label_ids.numel()
            tokens = self._tokenizer.convert_ids_to_tokens(input_ids)
            id_to_label = cast(dict[int, str], model.config.id2label or {})
            labels = [
                id_to_label.get(
                    int(label_id.item()), str(int(label_id.item()))
                )
                for label_id in label_ids
            ]
            tokens_to_labels = dict(zip(tokens, labels))
            return tokens_to_labels
