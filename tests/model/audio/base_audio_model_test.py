from logging import Logger
from typing import Literal
from unittest import IsolatedAsyncioTestCase
from unittest.mock import MagicMock, patch

from avalan.entities import EngineSettings
from avalan.model import TokenizerNotSupportedException
from avalan.model.audio import BaseAudioModel


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

    def test_resample_mono_stereo_resample(self):
        model = DummyAudioModel(
            None,
            EngineSettings(auto_load_model=False),
            logger=MagicMock(spec=Logger),
        )

        audio_wave = MagicMock()
        audio_wave.shape = (2, 10)
        audio_wave.squeeze = MagicMock()

        mean = MagicMock()
        audio_wave.mean.return_value = mean
        unsqueezed = MagicMock()
        mean.unsqueeze.return_value = unsqueezed

        resampled = MagicMock()
        resampled.squeeze.return_value = "wave"

        with (
            patch(
                "avalan.model.audio.load", return_value=(audio_wave, 8000)
            ) as load_patch,
            patch(
                "avalan.model.audio.resample", return_value=resampled
            ) as resample_patch,
        ):
            result = model._resample_mono("a.wav", 16000)

        self.assertEqual(result, "wave")
        load_patch.assert_called_once_with("a.wav")
        audio_wave.mean.assert_called_once_with(dim=0)
        audio_wave.squeeze.assert_not_called()
        mean.unsqueeze.assert_called_once_with(0)
        resample_patch.assert_called_once_with(unsqueezed, 8000, 16000)
        resampled.squeeze.assert_called_once_with(0)

    def test_resample_mono_mono_no_change(self):
        model = DummyAudioModel(
            None,
            EngineSettings(auto_load_model=False),
            logger=MagicMock(spec=Logger),
        )

        audio_wave = MagicMock()
        audio_wave.shape = (1, 10)
        audio_wave.mean = MagicMock()
        audio_wave.squeeze.return_value = "wave"

        with (
            patch(
                "avalan.model.audio.load", return_value=(audio_wave, 16000)
            ) as load_patch,
            patch("avalan.model.audio.resample") as resample_patch,
        ):
            result = model._resample_mono("a.wav", 16000)

        self.assertEqual(result, "wave")
        load_patch.assert_called_once_with("a.wav")
        audio_wave.squeeze.assert_called_once_with(0)
        audio_wave.mean.assert_not_called()
        resample_patch.assert_not_called()
