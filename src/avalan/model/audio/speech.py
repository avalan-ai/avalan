from ...compat import override
from ...model.vendor import TextGenerationVendor
from ...model.audio import BaseAudioModel
from ...model.engine import Engine
from diffusers import DiffusionPipeline
from torch import inference_mode
from transformers import (
    AutoProcessor,
    DiaForConditionalGeneration,
    PreTrainedModel,
)
from typing import Literal


class TextToSpeechModel(BaseAudioModel):
    _processor: AutoProcessor

    def _load_model(
        self,
    ) -> PreTrainedModel | TextGenerationVendor | DiffusionPipeline:
        self._processor = AutoProcessor.from_pretrained(
            self._model_id,
            trust_remote_code=self._settings.trust_remote_code,
            subfolder=self._settings.tokenizer_subfolder or "",
        )
        model = DiaForConditionalGeneration.from_pretrained(
            self._model_id,
            trust_remote_code=self._settings.trust_remote_code,
            device_map=self._device,
            tp_plan=Engine._get_tp_plan(self._settings.parallel),
            subfolder=self._settings.subfolder or "",
        )
        return model

    @override
    async def __call__(
        self,
        prompt: str,
        path: str,
        max_new_tokens: int,
        *,
        padding: bool = True,
        reference_path: str | None = None,
        reference_text: str | None = None,
        sampling_rate: int = 44_100,
        tensor_format: Literal["pt"] = "pt",
    ) -> str:
        assert (not reference_path and not reference_text) or (
            reference_path and reference_text
        )

        reference_voice = None
        if reference_path and reference_text:
            reference_voice = self._resample(reference_path, sampling_rate)

        text = (
            f"{reference_text}\n{prompt}"
            if reference_voice is not None
            else prompt
        )

        inputs = self._processor(
            text=text,
            audio=reference_voice,
            padding=padding,
            return_tensors=tensor_format,
            sampling_rate=sampling_rate,
        ).to(self._device)

        prompt_len = (
            self._processor.get_audio_prompt_len(
                inputs["decoder_attention_mask"]
            )
            if reference_voice is not None
            else None
        )

        with inference_mode():
            outputs = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens
            )

        wave = (
            self._processor.batch_decode(outputs, audio_prompt_len=prompt_len)
            if prompt_len and outputs.shape[1] >= prompt_len
            else self._processor.batch_decode(outputs)
        )

        self._processor.save_audio(wave, path)
        return path
