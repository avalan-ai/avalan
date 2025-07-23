from ...compat import override
from ...model.vendor import TextGenerationVendor
from ...model.audio import BaseAudioModel
from ...model.engine import Engine
from diffusers import DiffusionPipeline
from torch import from_numpy, inference_mode
from torchaudio import save
from transformers import (
    AutoProcessor,
    MusicgenForConditionalGeneration,
    PreTrainedModel,
)
from typing import Literal


class AudioGenerationModel(BaseAudioModel):
    _processor: AutoProcessor

    def _load_model(
        self,
    ) -> PreTrainedModel | TextGenerationVendor | DiffusionPipeline:
        self._processor = AutoProcessor.from_pretrained(self._model_id)
        model = MusicgenForConditionalGeneration.from_pretrained(
            self._model_id,
            device_map=self._device,
            tp_plan=Engine._get_tp_plan(self._settings.parallel),
            subfolder=self._settings.subfolder or "",
        ).to(self._device)
        return model

    @override
    async def __call__(
        self,
        prompt: str,
        path: str,
        max_new_tokens: int,
        *,
        padding: bool = True,
        tensor_format: Literal["pt"] = "pt",
    ) -> str:
        assert path

        inputs = self._processor(
            text=[prompt], return_tensors=tensor_format, padding=padding
        )
        inputs.to(self._device)

        with inference_mode():
            audio_tokens = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens
            )

        sampling_rate = self._model.config.audio_encoder.sampling_rate
        waveform = audio_tokens[0, 0].cpu().numpy()
        wave_tensor = from_numpy(waveform).unsqueeze(0)
        save(path, wave_tensor, sampling_rate)

        return path
