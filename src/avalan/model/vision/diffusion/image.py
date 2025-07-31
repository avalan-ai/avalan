from ....compat import override
from ....entities import (
    Input,
    TransformerEngineSettings,
    VisionColorModel,
    VisionImageFormat,
)
from ....model.vendor import TextGenerationVendor
from ....model.engine import Engine
from ....model.vision import BaseVisionModel
from dataclasses import replace
from diffusers import DiffusionPipeline
from logging import Logger, getLogger
from torch import inference_mode
from transformers import PreTrainedModel
from typing import Literal


class TextToImageModel(BaseVisionModel):
    _base: DiffusionPipeline

    def __init__(
        self,
        model_id: str,
        settings: TransformerEngineSettings | None = None,
        logger: Logger = getLogger(__name__),
    ):
        settings = settings or TransformerEngineSettings()
        assert settings.refiner_model_id
        settings = replace(settings, enable_eval=False)
        super().__init__(model_id, settings, logger)

    def _load_model(
        self,
    ) -> PreTrainedModel | TextGenerationVendor | DiffusionPipeline:
        dtype = Engine.weight(self._settings.weight_type)
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
    async def __call__(
        self,
        input: Input,
        path: str,
        *,
        color_model: VisionColorModel = VisionColorModel.RGB,
        high_noise_frac: float = 0.8,
        image_format: VisionImageFormat = VisionImageFormat.JPEG,
        n_steps: int = 150,
        output_type: Literal["latent"] = "latent",
    ) -> str:
        assert (
            input
            and path
            and color_model
            and high_noise_frac is not None
            and image_format
            and n_steps
            and output_type
        )

        with inference_mode():
            image = self._base(
                prompt=input if isinstance(input, str) else str(input),
                num_inference_steps=n_steps,
                denoising_end=high_noise_frac,
                output_type=output_type,
            ).images
            image = self._model(
                prompt=input if isinstance(input, str) else str(input),
                num_inference_steps=n_steps,
                denoising_start=high_noise_frac,
                image=image,
            ).images[0]

        image.convert(color_model)
        image.save(path, image_format)

        return path
