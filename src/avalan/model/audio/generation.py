from ...compat import override
from ...model.audio import BaseAudioModel
from ...model.engine import Engine
from ...model.vendor import TextGenerationVendor

from typing import Any, Literal

from diffusers import DiffusionPipeline
from torch import from_numpy, inference_mode
from torchaudio import save
from transformers import (
    AutoProcessor,
    MusicgenForConditionalGeneration,
    PreTrainedModel,
)


class AudioGenerationModel(BaseAudioModel):
    _processor: Any  # AutoProcessor with Musicgen-specific methods

    def _load_model(
        self,
    ) -> PreTrainedModel | TextGenerationVendor | DiffusionPipeline:
        assert self._model_id, "model_id is required"
        self._processor = AutoProcessor.from_pretrained(self._model_id)
        model: PreTrainedModel = (
            MusicgenForConditionalGeneration.from_pretrained(
                self._model_id,
                device_map=self._device,
                tp_plan=Engine._get_tp_plan(self._settings.parallel),
                distributed_config=Engine._get_distributed_config(
                    self._settings.distributed_config
                ),
                subfolder=self._settings.subfolder or "",
            ).to(self._device)
        )
        return model

    @override
    async def __call__(  # type: ignore[override]
        self,
        prompt: str,
        path: str,
        max_new_tokens: int,
        *,
        padding: bool = True,
        tensor_format: Literal["pt"] = "pt",
    ) -> str:
        assert path
        assert self._model is not None, "Model must be loaded"
        assert isinstance(self._model, PreTrainedModel)

        inputs = self._processor(
            text=[prompt], return_tensors=tensor_format, padding=padding
        )
        inputs.to(self._device)

        with inference_mode():
            audio_tokens = self._model.generate(  # type: ignore[attr-defined,operator]
                **inputs, max_new_tokens=max_new_tokens
            )

        sampling_rate: int = self._model.config.audio_encoder.sampling_rate
        waveform = audio_tokens[0, 0].cpu().numpy()
        wave_tensor = from_numpy(waveform).unsqueeze(0)
        save(path, wave_tensor, sampling_rate)

        return path
