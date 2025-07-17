from ...compat import override
from ...entities import (
    Input,
    TransformerEngineSettings,
    VisionColorModel,
    VisionImageFormat,
)
from ...model import TextGenerationVendor
from ...model.nlp import BaseNLPModel
from ...model.transformer import TransformerModel
from dataclasses import replace
from diffusers import DiffusionPipeline
from logging import Logger
from torch import inference_mode, Tensor
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.tokenization_utils_base import BatchEncoding
from typing import Literal


class TextToImageDiffusionModel(TransformerModel):
    _base: DiffusionPipeline

    def __init__(
        self,
        model_id: str,
        settings: TransformerEngineSettings | None = None,
        logger: Logger | None = None,
    ):
        settings = settings or TransformerEngineSettings()
        settings = replace(settings, enable_eval=False)
        super().__init__(model_id, settings, logger)

    def _load_model(
        self,
    ) -> PreTrainedModel | TextGenerationVendor | DiffusionPipeline:
        dtype = BaseNLPModel._get_weight_type(self._settings.weight_type)
        dtype_variant = self._settings.weight_type

        base = DiffusionPipeline.from_pretrained(
            self._model_id,
            torch_dtype=dtype,
            variant=dtype_variant,
            use_safetensors=True,
        )
        base.to(self._device)

        refiner = DiffusionPipeline.from_pretrained(
            self._settings.refiner_model_id,
            text_encoder_2=base.text_encoder_2,
            vae=base.vae,
            torch_dtype=dtype,
            use_safetensors=True,
            variant=dtype_variant,
        )
        refiner.to(self._device)

        self._base = base

        return refiner

    @override
    @property
    def uses_tokenizer(self) -> bool:
        return False

    @override
    def _load_tokenizer(
        self, tokenizer_name_or_path: str | None, use_fast: bool
    ) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        raise NotImplementedError()

    @override
    def _tokenize_input(
        self,
        input: Input,
        context: str | None = None,
        tensor_format: Literal["pt"] = "pt",
        **kwargs,
    ) -> dict[str, Tensor] | BatchEncoding | Tensor:
        raise NotImplementedError()

    @override
    async def __call__(
        self,
        prompt: str,
        path: str,
        *,
        color_model: VisionColorModel = VisionColorModel.RGB,
        high_noise_frac: float = 0.8,
        image_format: VisionImageFormat = VisionImageFormat.JPEG,
        n_steps: int = 150,
        output_type: Literal["latent"] = "latent",
    ) -> str:
        assert (
            prompt
            and path
            and color_model
            and high_noise_frac is not None
            and image_format
            and n_steps
            and output_type
        )

        with inference_mode():
            image = self._base(
                prompt=prompt,
                num_inference_steps=n_steps,
                denoising_end=high_noise_frac,
                output_type=output_type,
            ).images
            image = self._model(
                prompt=prompt,
                num_inference_steps=n_steps,
                denoising_start=high_noise_frac,
                image=image,
            ).images[0]

        image.convert(color_model)
        image.save(path, image_format)

        return path
