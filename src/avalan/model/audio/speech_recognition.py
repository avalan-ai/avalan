from ...compat import override
from ...model.audio import BaseAudioModel
from ...model.engine import Engine
from ...model.vendor import TextGenerationVendor

from typing import Any, Literal

from diffusers import DiffusionPipeline
from torch import argmax, inference_mode
from transformers import (
    AutoModelForCTC,
    AutoProcessor,
    PreTrainedModel,
)


class SpeechRecognitionModel(BaseAudioModel):
    _processor: Any  # AutoProcessor with Wav2Vec2-specific methods

    def _load_model(
        self,
    ) -> PreTrainedModel | TextGenerationVendor | DiffusionPipeline:
        assert self._model_id, "model_id is required"
        self._processor = AutoProcessor.from_pretrained(
            self._model_id,
            trust_remote_code=self._settings.trust_remote_code,
            # default behavior in transformers v4.48
            use_fast=True,
            subfolder=self._settings.tokenizer_subfolder or "",
        )
        # AutoProcessor for CTC models has a tokenizer attribute
        pad_token_id = getattr(
            getattr(self._processor, "tokenizer", None), "pad_token_id", None
        )
        model: PreTrainedModel = AutoModelForCTC.from_pretrained(
            self._model_id,
            trust_remote_code=self._settings.trust_remote_code,
            pad_token_id=pad_token_id,
            ctc_loss_reduction="mean",
            device_map=self._device,
            tp_plan=Engine._get_tp_plan(self._settings.parallel),
            distributed_config=Engine._get_distributed_config(
                self._settings.distributed_config
            ),
            ignore_mismatched_sizes=True,
            subfolder=self._settings.subfolder or "",
        )
        return model

    @override
    async def __call__(  # type: ignore[override]
        self,
        path: str,
        sampling_rate: int = 16_000,
        tensor_format: Literal["pt"] = "pt",
    ) -> str:
        assert self._model is not None, "Model must be loaded"
        audio = self._resample(path, sampling_rate)
        inputs = self._processor(
            audio,
            sampling_rate=sampling_rate,
            return_tensors=tensor_format,
        ).to(self._device)
        with inference_mode():
            # shape (batch, time_steps, vocab_size)
            assert isinstance(self._model, PreTrainedModel)
            logits = self._model(inputs.input_values).logits
        predicted_ids = argmax(logits, dim=-1)
        transcription: str = self._processor.batch_decode(predicted_ids)[0]
        return transcription
