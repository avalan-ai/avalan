from contextlib import nullcontext
from logging import Logger
from unittest import IsolatedAsyncioTestCase, TestCase, main
from unittest.mock import MagicMock, PropertyMock, patch

from transformers import PreTrainedModel

from avalan.entities import EngineSettings
from avalan.model.audio.classification import (
    AudioClassificationModel,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
)
from avalan.model.engine import Engine


class AudioClassificationModelInstantiationTestCase(TestCase):
    model_id = "dummy/model"

    def test_instantiation_no_load(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(
                AutoFeatureExtractor, "from_pretrained"
            ) as extractor_mock,
            patch.object(
                AutoModelForAudioClassification, "from_pretrained"
            ) as model_mock,
        ):
            settings = EngineSettings(auto_load_model=False)
            model = AudioClassificationModel(
                self.model_id, settings, logger=logger_mock
            )
            self.assertIsInstance(model, AudioClassificationModel)
            extractor_mock.assert_not_called()
            model_mock.assert_not_called()

    def test_instantiation_with_load_model(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(
                AutoFeatureExtractor, "from_pretrained"
            ) as extractor_mock,
            patch.object(
                AutoModelForAudioClassification, "from_pretrained"
            ) as model_mock,
        ):
            extractor_instance = MagicMock()
            extractor_mock.return_value = extractor_instance

            model_instance = MagicMock(spec=PreTrainedModel)
            model_instance.to.return_value = model_instance
            model_mock.return_value = model_instance

            settings = EngineSettings()
            model = AudioClassificationModel(
                self.model_id, settings, logger=logger_mock
            )
            self.assertIs(model.model, model_instance)
            extractor_mock.assert_called_once_with(self.model_id)
            model_mock.assert_called_once_with(
                self.model_id,
                device_map=Engine.get_default_device(),
                tp_plan=None,
                distributed_config=None,
                subfolder="",
            )
            model_instance.to.assert_called_once_with(model._device)


class AudioClassificationModelCallTestCase(IsolatedAsyncioTestCase):
    model_id = "dummy/model"

    async def test_call(self):
        logger_mock = MagicMock(spec=Logger)

        class F(float):
            def item(self):
                return float(self)

        with (
            patch.object(
                AutoFeatureExtractor, "from_pretrained"
            ) as extractor_mock,
            patch.object(
                AutoModelForAudioClassification, "from_pretrained"
            ) as model_mock,
            patch.object(
                AudioClassificationModel, "_resample_mono"
            ) as resample_mock,
            patch(
                "avalan.model.audio.classification.inference_mode",
                return_value=nullcontext(),
            ) as inf_mock,
        ):
            extractor_instance = MagicMock()
            extractor_call = MagicMock(return_value="inputs")
            extractor_call.to.return_value = {"x": "X"}
            extractor_instance.return_value = extractor_call
            extractor_mock.return_value = extractor_instance

            model_instance = MagicMock()
            model_instance.to.return_value = model_instance
            logits = MagicMock()
            logits.softmax.return_value = [[F(0.2), F(0.8)]]
            call_result = MagicMock(logits=logits)
            model_instance.return_value = call_result
            config_mock = MagicMock()
            config_mock.id2label = {0: "A", 1: "B"}
            type(model_instance).config = PropertyMock(
                return_value=config_mock
            )
            model_mock.return_value = model_instance

            resample_mock.return_value = "wave"

            settings = EngineSettings(auto_load_model=False)
            model = AudioClassificationModel(
                self.model_id, settings, logger=logger_mock
            )
            model._model = model._load_model()

            result = await model(
                "file.wav", padding=False, sampling_rate=16000
            )

            self.assertEqual(result, {"B": 0.8, "A": 0.2})
            resample_mock.assert_called_once_with("file.wav", 16000)
            extractor_instance.assert_called_with(
                "wave",
                sampling_rate=16000,
                return_tensors="pt",
                padding=False,
            )
            extractor_call.to.assert_called_once_with(model._device)
            model_instance.assert_called_once_with(**{"x": "X"})
            logits.softmax.assert_called_once_with(dim=-1)
            inf_mock.assert_called_once_with()


if __name__ == "__main__":
    main()
