from avalan.entities import EngineSettings
from avalan.model import TokenizerNotSupportedException
from avalan.model.audio import BaseAudioModel
from logging import Logger
from unittest import IsolatedAsyncioTestCase
from unittest.mock import MagicMock, patch
from typing import Literal


class DummyAudioModel(BaseAudioModel):
    def _load_model(self):
        return MagicMock()

    async def __call__(
        self, image_source: str | object, tensor_format: Literal["pt"] = "pt"
    ) -> str:
        return await super().__call__(image_source, tensor_format)


class BaseAudioModelTestCase(IsolatedAsyncioTestCase):
    async def test_base_methods_raise(self):
        model = DummyAudioModel(
            None,
            EngineSettings(auto_load_model=False),
            logger=MagicMock(spec=Logger),
        )
        with self.assertRaises(NotImplementedError):
            await model("img")
        with self.assertRaises(TokenizerNotSupportedException):
            model._load_tokenizer(None)
        with self.assertRaises(TokenizerNotSupportedException):
            model._load_tokenizer_with_tokens(None)

    def test_resample(self):
        model = DummyAudioModel(
            None,
            EngineSettings(auto_load_model=False),
            logger=MagicMock(spec=Logger),
        )
        audio_wave = MagicMock()
        mean = MagicMock()
        mean.numpy.return_value = "audio"
        audio_wave.mean.return_value = mean
        with (
            patch(
                "avalan.model.audio.load", return_value=(audio_wave, 8000)
            ) as load_patch,
            patch(
                "avalan.model.audio.resample", return_value=audio_wave
            ) as resample_patch,
        ):
            result = model._resample("a.wav", 16000)
        self.assertEqual(result, "audio")
        load_patch.assert_called_once_with("a.wav")
        resample_patch.assert_called_once_with(audio_wave, 8000, 16000)
        audio_wave.mean.assert_called_once_with(0)
        mean.numpy.assert_called_once_with()

    def test_resample_no_change(self):
        model = DummyAudioModel(
            None,
            EngineSettings(auto_load_model=False),
            logger=MagicMock(spec=Logger),
        )
        audio_wave = MagicMock()
        mean = MagicMock()
        mean.numpy.return_value = "audio"
        audio_wave.mean.return_value = mean
        with (
            patch(
                "avalan.model.audio.load", return_value=(audio_wave, 16000)
            ) as load_patch,
            patch("avalan.model.audio.resample") as resample_patch,
        ):
            result = model._resample("a.wav", 16000)
        self.assertEqual(result, "audio")
        load_patch.assert_called_once_with("a.wav")
        resample_patch.assert_not_called()
        audio_wave.mean.assert_called_once_with(0)
        mean.numpy.assert_called_once_with()
