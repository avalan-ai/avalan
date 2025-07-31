from ....compat import override
from ....entities import BetaSchedule, EngineSettings, Input, TimestepSpacing
from ....model.vendor import TextGenerationVendor
from ....model.engine import Engine
from ....model.vision import BaseVisionModel
from dataclasses import replace
from diffusers import (
    AnimateDiffPipeline,
    DiffusionPipeline,
    EulerDiscreteScheduler,
    MotionAdapter,
)
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import export_to_gif
from huggingface_hub import hf_hub_download
from logging import Logger, getLogger
from safetensors.torch import load_file
from torch import inference_mode
from transformers import PreTrainedModel


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
        adapter = MotionAdapter().to(self._device, dtype)
        adapter.load_state_dict(
            load_file(
                hf_hub_download(self._model_id, self._settings.checkpoint),
                device=self._device,
            )
        )
        pipe = AnimateDiffPipeline.from_pretrained(
            self._settings.base_model_id,
            motion_adapter=adapter,
            torch_dtype=dtype,
        ).to(self._device)

        return pipe

    @override
    async def __call__(
        self,
        input: Input,
        path: str,
        *,
        beta_schedule: BetaSchedule = "linear",
        guidance_scale: float = 1.0,
        steps: int = 4,
        timestep_spacing: TimestepSpacing = "trailing",
    ) -> str:
        assert steps and steps in [
            1,
            2,
            4,
            8,
        ], f"Invalid number of steps: {steps}, can only be 1, 2, 4, or 8"
        scheduler_settings = (timestep_spacing, beta_schedule)
        if scheduler_settings not in self._schedulers:
            scheduler = EulerDiscreteScheduler.from_config(
                self._model.scheduler.config,
                timestep_spacing=timestep_spacing,
                beta_schedule=beta_schedule,
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
            )

        export_to_gif(output.frames[0], path)

        return path
