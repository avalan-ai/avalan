from ....entities import BetaSchedule, EngineSettings, Input, TimestepSpacing
from ....model.engine import Engine
from ....model.vendor import TextGenerationVendor
from ....model.vision import BaseVisionModel

from dataclasses import replace
from logging import Logger, getLogger
from typing import Any, cast

import diffusers.utils as diffusers_utils
import numpy as np
from diffusers import (
    AnimateDiffPipeline,
    DiffusionPipeline,
    EulerDiscreteScheduler,
    MotionAdapter,
)
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from huggingface_hub import hf_hub_download
from numpy.typing import NDArray
from PIL import Image
from safetensors.torch import load_file
from torch import inference_mode
from transformers import PreTrainedModel

export_to_gif = cast(Any, diffusers_utils).export_to_gif


class TextToAnimationModel(BaseVisionModel):
    _schedulers: dict[tuple[TimestepSpacing, BetaSchedule], SchedulerMixin] = (
        {}
    )

    def __init__(
        self,
        model_id: str,
        settings: EngineSettings | None = None,
        logger: Logger = getLogger(__name__),
    ):
        settings = settings or EngineSettings()
        assert settings.base_model_id and settings.checkpoint
        settings = replace(settings, enable_eval=False)
        super().__init__(model_id, settings, logger)

    def _load_model(
        self,
    ) -> PreTrainedModel | TextGenerationVendor | DiffusionPipeline:
        dtype = Engine.weight(self._settings.weight_type)
        assert self._model_id and self._settings.checkpoint
        adapter = cast(Any, MotionAdapter)().to(self._device, dtype)
        adapter.load_state_dict(
            load_file(
                hf_hub_download(self._model_id, self._settings.checkpoint),
                device=self._device,
            )
        )
        pipe = cast(
            DiffusionPipeline,
            cast(Any, AnimateDiffPipeline)
            .from_pretrained(
                self._settings.base_model_id,
                feature_extractor=None,
                image_encoder=None,
                motion_adapter=adapter,
                torch_dtype=dtype,
                use_safetensors=False,
            )
            .to(self._device),
        )

        return pipe

    @staticmethod
    def _frames_to_images(frames: NDArray[Any]) -> list[Image.Image]:
        assert frames.ndim == 4, (
            "Animation frames must have shape "
            f"[frames, height, width, channels], got {frames.shape}"
        )
        sanitized_frames = np.clip(
            np.nan_to_num(
                frames,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ),
            0.0,
            1.0,
        )
        images: list[Image.Image] = []
        for frame in sanitized_frames:
            frame_uint8 = np.rint(frame * 255).astype("uint8")
            if frame_uint8.shape[-1] == 1:
                image = Image.fromarray(frame_uint8.squeeze(-1))
            else:
                image = Image.fromarray(frame_uint8)
            images.append(image)

        return images

    async def __call__(
        self,
        input: Input,
        path: str,
        *,
        beta_schedule: BetaSchedule = cast(BetaSchedule, "linear"),
        guidance_scale: float = 1.0,
        steps: int = 4,
        timestep_spacing: TimestepSpacing = cast(TimestepSpacing, "trailing"),
    ) -> str:
        assert steps and steps in [
            1,
            2,
            4,
            8,
        ], f"Invalid number of steps: {steps}, can only be 1, 2, 4, or 8"
        scheduler_settings = (timestep_spacing, beta_schedule)
        if scheduler_settings not in self._schedulers:
            scheduler = cast(
                SchedulerMixin,
                cast(Any, EulerDiscreteScheduler).from_config(
                    self._model.scheduler.config,
                    timestep_spacing=timestep_spacing,
                    beta_schedule=beta_schedule,
                ),
            )
            self._schedulers[scheduler_settings] = scheduler
        else:
            scheduler = self._schedulers[scheduler_settings]

        self._model.scheduler = scheduler

        with inference_mode():
            output = self._model(
                prompt=input if isinstance(input, str) else str(input),
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                output_type="np",
            )

        export_to_gif(
            TextToAnimationModel._frames_to_images(
                cast(NDArray[Any], output.frames[0])
            ),
            path,
        )

        return path
