from abc import ABC, abstractmethod
from ..compat import override
from ..model import TextGenerationVendor, TokenizerNotSupportedException
from ..model.engine import Engine
from PIL import Image
from torch import argmax, inference_mode
from torchaudio import load
from torchaudio.functional import resample
from numpy import ndarray
from transformers import (
    AutoModelForCTC,
    AutoProcessor,
    DiaForConditionalGeneration,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from typing import Literal


class BaseAudioModel(Engine, ABC):
    @abstractmethod
    async def __call__(
        self,
        image_source: str | Image.Image,
        tensor_format: Literal["pt"] = "pt",
    ) -> str:
        raise NotImplementedError()

    def _load_tokenizer(
        self, tokenizer_name_or_path: str | None, use_fast: bool = True
    ) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        raise TokenizerNotSupportedException()

    def _load_tokenizer_with_tokens(
        self, tokenizer_name_or_path: str | None, use_fast: bool = True
    ) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        raise TokenizerNotSupportedException()

    def _resample(self, audio_source: str, sampling_rate: int) -> ndarray:
        wave, wave_sampling_rate = load(audio_source)
        if wave_sampling_rate != sampling_rate:
            wave = resample(
                wave, wave_sampling_rate, sampling_rate
            )
        wave = wave.mean(0).numpy()
        return wave


class SpeechRecognitionModel(BaseAudioModel):
    def _load_model(self) -> PreTrainedModel | TextGenerationVendor:
        self._processor = AutoProcessor.from_pretrained(
            self._model_id,
            trust_remote_code=self._settings.trust_remote_code,
            # default behavior in transformers v4.48
            use_fast=True,
        )
        model = AutoModelForCTC.from_pretrained(
            self._model_id,
            trust_remote_code=self._settings.trust_remote_code,
            pad_token_id=self._processor.tokenizer.pad_token_id,
            ctc_loss_reduction="mean",
            device_map=self._device,
            tp_plan=Engine._get_tp_plan(self._settings.parallel),
            ignore_mismatched_sizes=True
        )
        return model

    @override
    async def __call__(
        self,
        path: str,
        sampling_rate: int = 16_000,
        tensor_format: Literal["pt"] = "pt",
    ) -> str:
        audio = self._resample(path, sampling_rate)
        inputs = self._processor(
            audio,
            sampling_rate=sampling_rate,
            return_tensors=tensor_format,
        ).to(self._device)
        with inference_mode():
            # shape (batch, time_steps, vocab_size)
            logits = self._model(inputs.input_values).logits
        predicted_ids = argmax(logits, dim=-1)
        transcription = self._processor.batch_decode(predicted_ids)[0]
        return transcription


class TextToSpeechModel(BaseAudioModel):
    def _load_model(self) -> PreTrainedModel | TextGenerationVendor:
        self._processor = AutoProcessor.from_pretrained(
            self._model_id,
            trust_remote_code=self._settings.trust_remote_code,
        )
        model = DiaForConditionalGeneration.from_pretrained(
            self._model_id,
            trust_remote_code=self._settings.trust_remote_code,
            device_map=self._device,
            tp_plan=Engine._get_tp_plan(self._settings.parallel),
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
            reference_voice = self._resample(
                reference_path, sampling_rate
            )

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
