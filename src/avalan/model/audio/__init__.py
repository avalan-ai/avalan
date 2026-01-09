from ...model import TokenizerNotSupportedException
from ...model.engine import Engine

from abc import ABC
from typing import Any

from numpy.typing import NDArray
from torch import Tensor
from torchaudio import load
from torchaudio.functional import resample
from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)


class BaseAudioModel(Engine, ABC):
    """Base class for audio models."""

    def _load_tokenizer(
        self, tokenizer_name_or_path: str | None, use_fast: bool = True
    ) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        raise TokenizerNotSupportedException()

    def _load_tokenizer_with_tokens(
        self, tokenizer_name_or_path: str | None, use_fast: bool = True
    ) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        raise TokenizerNotSupportedException()

    def _resample_mono(self, audio_source: str, sampling_rate: int) -> Tensor:
        wave_data: Tensor
        wave_data, wave_sampling_rate = load(audio_source)

        if wave_data.shape[0] > 1:
            # stereo -> mono
            wave_data = wave_data.mean(dim=0)  # type: ignore[operator]
        else:
            # already mono, just drop channel dim (samples,)
            wave_data = wave_data.squeeze(0)  # type: ignore[operator]

        if wave_sampling_rate != sampling_rate:
            wave_data = resample(
                wave_data.unsqueeze(0),
                wave_sampling_rate,
                sampling_rate,  # type: ignore[operator]
            ).squeeze(0)

        return wave_data

    def _resample(self, audio_source: str, sampling_rate: int) -> NDArray[Any]:
        wave, wave_sampling_rate = load(audio_source)
        if wave_sampling_rate != sampling_rate:
            wave = resample(wave, wave_sampling_rate, sampling_rate)
        result: NDArray[Any] = wave.mean(0).numpy()  # type: ignore[union-attr]
        return result
