from abc import ABC, abstractmethod
from ...model import TokenizerNotSupportedException
from ...model.engine import Engine
from PIL import Image
from torchaudio import load
from torchaudio.functional import resample
from numpy import ndarray
from transformers import (
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
            wave = resample(wave, wave_sampling_rate, sampling_rate)
        wave = wave.mean(0).numpy()
        return wave
