from ...model.audio import BaseAudioModel
from ...model.engine import Engine
from ...model.vendor import TextGenerationVendor

from typing import Any, Literal, cast

from diffusers import DiffusionPipeline
from torch import from_numpy, inference_mode
from torchaudio import save
from transformers import (
    AutoProcessor,
    PreTrainedModel,
)
from transformers import (
    MusicgenForConditionalGeneration as TransformersMusicgen,
)


# MusicGen checkpoints include this non-persistent sinusoidal buffer. Ignore
# only that key so unrelated load report diagnostics still surface.
class MusicgenForConditionalGeneration(TransformersMusicgen):
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.model\.decoder\.embed_positions\.weights",
    ]


class AudioGenerationModel(BaseAudioModel):
    _processor: Any

    def _load_model(
        self,
    ) -> PreTrainedModel | TextGenerationVendor | DiffusionPipeline:
        self._processor = cast(
            Any,
            cast(Any, AutoProcessor).from_pretrained(self._model_id),
        )
        model = cast(
            PreTrainedModel,
            cast(Any, MusicgenForConditionalGeneration)
            .from_pretrained(
                self._model_id,
                device_map=self._device,
                tp_plan=Engine._get_tp_plan(self._settings.parallel),
                distributed_config=Engine._get_distributed_config(
                    self._settings.distributed_config
                ),
                subfolder=self._settings.subfolder or "",
            )
            .to(self._device),
        )
        return model

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
                **inputs,
                max_new_tokens=max_new_tokens,
                remove_invalid_values=True,
            )

        sampling_rate = self._model.config.audio_encoder.sampling_rate
        waveform = audio_tokens[0, 0].cpu().numpy()
        wave_tensor = from_numpy(waveform).unsqueeze(0)
        save(path, wave_tensor, sampling_rate)

        return path
