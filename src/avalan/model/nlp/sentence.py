from ...entities import Input
from ...model.engine import Engine
from ...model.nlp import BaseNLPModel
from ...model.vendor import TextGenerationVendor

from contextlib import nullcontext
from typing import Any, Literal, cast

from diffusers import DiffusionPipeline
from numpy.typing import NDArray
from torch import inference_mode
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.tokenization_utils_base import BatchEncoding


class SentenceTransformerModel(BaseNLPModel):
    @property
    def supports_sample_generation(self) -> bool:
        return False

    @property
    def supports_token_streaming(self) -> bool:
        return False

    @property
    def uses_tokenizer(self) -> bool:
        return True

    def token_count(self, input: str) -> int:
        tokenizer = cast(
            PreTrainedTokenizer | PreTrainedTokenizerFast, self.tokenizer
        )
        token_ids = tokenizer.encode(input, add_special_tokens=False)
        return len(token_ids)

    def _load_model(
        self,
    ) -> PreTrainedModel | TextGenerationVendor | DiffusionPipeline:
        from sentence_transformers import SentenceTransformer

        settings = cast(Any, self._settings)
        assert self._model_id, "A model id is required."
        model = SentenceTransformer(
            self._model_id,
            cache_folder=settings.cache_dir,
            device=self._device,
            trust_remote_code=settings.trust_remote_code,
            local_files_only=settings.local_files_only,
            token=settings.access_token,
            model_kwargs={
                "attn_implementation": settings.attention,
                "torch_dtype": Engine.weight(settings.weight_type),
                "low_cpu_mem_usage": (
                    True if self._device else settings.low_cpu_mem_usage
                ),
                "device_map": self._device,
                "tp_plan": Engine._get_tp_plan(settings.parallel),
                "distributed_config": Engine._get_distributed_config(
                    settings.distributed_config
                ),
            },
            backend="torch",
            similarity_fn_name=None,
            truncate_dim=None,
        )
        return cast(PreTrainedModel | TextGenerationVendor, model)

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
        raise NotImplementedError()

    async def __call__(
        self,
        input: Input,
        *args: object,
        enable_gradient_calculation: bool = False,
    ) -> NDArray[Any]:
        assert self._model, (
            f"Model {self._model} can't be executed, it "
            + "needs to be loaded first"
        )
        assert isinstance(input, str) or isinstance(input, list)
        model = cast(Any, self._model)
        with (
            inference_mode()
            if not enable_gradient_calculation
            else nullcontext()
        ):
            embeddings = model.encode(
                input, convert_to_numpy=True, show_progress_bar=False
            )
        return cast(NDArray[Any], embeddings)
