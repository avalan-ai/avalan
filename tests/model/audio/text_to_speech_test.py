from avalan.entities import EngineSettings
from avalan.model.engine import Engine
from avalan.model.audio import (
    TextToSpeechModel,
    AutoProcessor,
    DiaForConditionalGeneration,
)
from contextlib import nullcontext
from logging import Logger
from transformers import PreTrainedModel
from unittest import TestCase, IsolatedAsyncioTestCase, main
from unittest.mock import MagicMock, patch


class TextToSpeechModelInstantiationTestCase(TestCase):
    model_id = "dummy/model"

    def test_instantiation_no_load(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(AutoProcessor, "from_pretrained") as processor_mock,
            patch.object(
                DiaForConditionalGeneration, "from_pretrained"
            ) as model_mock,
        ):
            settings = EngineSettings(auto_load_model=False)
            model = TextToSpeechModel(
                self.model_id, settings, logger=logger_mock
            )
            self.assertIsInstance(model, TextToSpeechModel)
            processor_mock.assert_not_called()
            model_mock.assert_not_called()

    def test_instantiation_with_load_model(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(AutoProcessor, "from_pretrained") as processor_mock,
            patch.object(
                DiaForConditionalGeneration, "from_pretrained"
            ) as model_mock,
        ):
            processor_instance = MagicMock()
            processor_mock.return_value = processor_instance

            model_instance = MagicMock(spec=PreTrainedModel)
            model_mock.return_value = model_instance

            settings = EngineSettings()
            model = TextToSpeechModel(
                self.model_id, settings, logger=logger_mock
            )
            self.assertIs(model.model, model_instance)
            processor_mock.assert_called_once_with(
                self.model_id,
                trust_remote_code=False,
            )
            model_mock.assert_called_once_with(
                self.model_id,
                trust_remote_code=False,
                device_map=Engine.get_default_device(),
                tp_plan=None,
            )


class TextToSpeechModelCallTestCase(IsolatedAsyncioTestCase):
    model_id = "dummy/model"

    async def test_call(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(AutoProcessor, "from_pretrained") as processor_mock,
            patch.object(
                DiaForConditionalGeneration, "from_pretrained"
            ) as model_mock,
            patch(
                "avalan.model.audio.inference_mode", return_value=nullcontext()
            ) as inf_mock,
        ):
            call_result = MagicMock()
            inputs = {"input_ids": [1]}
            call_result.to.return_value = inputs
            processor_instance = MagicMock(return_value=call_result)
            processor_instance.batch_decode.return_value = ["audio"]
            processor_mock.return_value = processor_instance

            model_instance = MagicMock(spec=PreTrainedModel)
            model_instance.generate = MagicMock(return_value=[2])
            model_mock.return_value = model_instance

            settings = EngineSettings()
            model = TextToSpeechModel(
                self.model_id, settings, logger=logger_mock
            )

            result = await model(["hi"], "file.wav", 3, padding=False)

            self.assertEqual(result, "file.wav")
            processor_instance.assert_called_with(
                text=["hi"], padding=False, return_tensors="pt"
            )
            call_result.to.assert_called_once_with(model._device)
            model_instance.generate.assert_called_once_with(
                **inputs, max_new_tokens=3
            )
            processor_instance.batch_decode.assert_called_once_with([2])
            processor_instance.save_audio.assert_called_once_with(
                ["audio"], "file.wav"
            )
            inf_mock.assert_called_once_with()


if __name__ == "__main__":
    main()
