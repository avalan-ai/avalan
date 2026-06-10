from ....entities import EngineSettings, Input
from ....model.engine import DiffusionPipeline, Engine, PreTrainedModel
from ....model.vendor import TextGenerationVendor
from ....model.vision import BaseVisionModel

from dataclasses import replace
from importlib import import_module
from logging import Logger, getLogger
from typing import Any, cast

from torch import Generator, inference_mode


def _diffusion_pipeline() -> Any:
    return getattr(import_module("diffusers"), "DiffusionPipeline")


def _ltx_video_condition() -> Any:
    return getattr(
        import_module("diffusers.pipelines.ltx.pipeline_ltx_condition"),
        "LTXVideoCondition",
    )


def export_to_video(*args: object, **kwargs: object) -> Any:
    return getattr(import_module("diffusers.utils"), "export_to_video")(
        *args, **kwargs
    )


def load_image(*args: object, **kwargs: object) -> Any:
    return getattr(import_module("diffusers.utils"), "load_image")(
        *args, **kwargs
    )


def load_video(*args: object, **kwargs: object) -> Any:
    return getattr(import_module("diffusers.utils"), "load_video")(
        *args, **kwargs
    )


class TextToVideoModel(BaseVisionModel):
    _upsampler_pipe: DiffusionPipeline

    def __init__(
        self,
        model_id: str,
        settings: EngineSettings | None = None,
        logger: Logger = getLogger(__name__),
    ) -> None:
        settings = settings or EngineSettings()
        assert settings.upsampler_model_id
        settings = replace(settings, enable_eval=False)
        super().__init__(model_id, settings, logger)

    def _load_model(
        self,
    ) -> PreTrainedModel | TextGenerationVendor | DiffusionPipeline:
        assert self._model_id, "A model id is required."
        dtype = Engine.weight(self._settings.weight_type)
        pipeline = _diffusion_pipeline()
        base_pipe = cast(
            Any,
            pipeline.from_pretrained(
                self._model_id,
                torch_dtype=dtype,
            ),
        ).to(self._device)
        self._upsampler_pipe = cast(
            Any,
            pipeline.from_pretrained(
                self._settings.upsampler_model_id,
                vae=base_pipe.vae,
                torch_dtype=dtype,
            ),
        ).to(self._device)
        cast(Any, base_pipe.vae).enable_tiling()
        return cast(DiffusionPipeline, base_pipe)

    async def __call__(
        self,
        input: Input,
        negative_prompt: str,
        reference_path: str,
        path: str,
        *,
        decode_timestep: float = 0.05,
        denoise_strength: float = 0.4,
        downscale: float = 2 / 3,
        fps: int = 24,
        frames: int = 96,
        height: int = 480,
        inference_steps: int = 10,
        noise_scale: float = 0.025,
        width: int = 832,
        steps: int = 30,
    ) -> str:
        model = cast(Any, self._model)
        upsampler = cast(Any, self._upsampler_pipe)
        condition_class = _ltx_video_condition()
        ratio = int(getattr(model, "vae_spatial_compression_ratio", 1))
        image = load_image(reference_path)
        video = load_video(export_to_video([image]))
        condition = condition_class(video=video, frame_index=0)

        down_h = int(height * downscale)
        down_w = int(width * downscale)
        down_h, down_w = (
            TextToVideoModel._round_to_nearest_resolution_acceptable_by_vae(
                down_h,
                down_w,
                ratio=ratio,
            )
        )

        with inference_mode():
            latents = cast(
                Any,
                model(
                    conditions=[condition],
                    prompt=input if isinstance(input, str) else str(input),
                    negative_prompt=negative_prompt,
                    width=down_w,
                    height=down_h,
                    num_frames=frames,
                    num_inference_steps=steps,
                    generator=Generator().manual_seed(0),
                    output_type="latent",
                ),
            ).frames

            upscaled_h, upscaled_w = down_h * 2, down_w * 2
            upscaled_latents = cast(
                Any, upsampler(latents=latents, output_type="latent")
            ).frames

            video = cast(
                Any,
                model(
                    conditions=[condition],
                    prompt=input if isinstance(input, str) else str(input),
                    negative_prompt=negative_prompt,
                    width=upscaled_w,
                    height=upscaled_h,
                    num_frames=frames,
                    denoise_strength=denoise_strength,
                    num_inference_steps=inference_steps,
                    latents=upscaled_latents,
                    decode_timestep=decode_timestep,
                    image_cond_noise_scale=noise_scale,
                    generator=Generator().manual_seed(0),
                    output_type="pil",
                ),
            ).frames[0]

        video = [frame.resize((width, height)) for frame in video]
        export_to_video(video, path, fps=fps)

        return path

    @staticmethod
    def _round_to_nearest_resolution_acceptable_by_vae(
        height: int,
        width: int,
        ratio: int,
    ) -> tuple[int, int]:
        return (
            height - (height % ratio),
            width - (width % ratio),
        )
