from ....compat import override
from ....entities import EngineSettings, Input
from ....model.vendor import TextGenerationVendor
from ....model.engine import Engine
from ....model.vision import BaseVisionModel
from dataclasses import replace
from diffusers import DiffusionPipeline
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from diffusers.utils import export_to_video, load_image, load_video
from logging import Logger, getLogger
from torch import Generator, inference_mode
from transformers import PreTrainedModel


class TextToVideoModel(BaseVisionModel):
    _upsampler_pipe: DiffusionPipeline

    def __init__(
        self,
        model_id: str,
        settings: EngineSettings | None = None,
        logger: Logger = getLogger(__name__),
    ):
        settings = settings or EngineSettings()
        assert settings.upsampler_model_id
        settings = replace(settings, enable_eval=False)
        super().__init__(model_id, settings, logger)

    def _load_model(
        self,
    ) -> PreTrainedModel | TextGenerationVendor | DiffusionPipeline:
        dtype = Engine.weight(self._settings.weight_type)
        base_pipe = DiffusionPipeline.from_pretrained(
            self._model_id,
            torch_dtype=dtype,
        ).to(self._device)
        self._upsampler_pipe = DiffusionPipeline.from_pretrained(
            self._settings.upsampler_model_id,
            vae=base_pipe.vae,
            torch_dtype=dtype,
        ).to(self._device)
        base_pipe.vae.enable_tiling()
        return base_pipe

    @override
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
        image = load_image(reference_path)
        video = load_video(export_to_video([image]))
        condition = LTXVideoCondition(video=video, frame_index=0)

        down_h = int(height * downscale)
        down_w = int(width * downscale)
        down_h, down_w = (
            TextToVideoModel._round_to_nearest_resolution_acceptable_by_vae(
                down_h,
                down_w,
                ratio=self._model.vae_spatial_compression_ratio,
            )
        )

        with inference_mode():
            latents = self._model(
                conditions=[condition],
                prompt=input if isinstance(input, str) else str(input),
                negative_prompt=negative_prompt,
                width=down_w,
                height=down_h,
                num_frames=frames,
                num_inference_steps=steps,
                generator=Generator().manual_seed(0),
                output_type="latent",
            ).frames

            upscaled_h, upscaled_w = down_h * 2, down_w * 2
            upscaled_latents = self._upsampler_pipe(
                latents=latents, output_type="latent"
            ).frames

            video = self._model(
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
