from avalan.entities import EngineSettings
from avalan.model.engine import Engine
from avalan.model.audio import (
    SpeechRecognitionModel,
    AutoProcessor,
    AutoModelForCTC,
)
from contextlib import nullcontext
from logging import Logger
from transformers import PreTrainedModel
from unittest import TestCase, IsolatedAsyncioTestCase, main
from unittest.mock import MagicMock, patch, PropertyMock


class SpeechRecognitionModelInstantiationTestCase(TestCase):
    model_id = "dummy/model"

    def test_instantiation_no_load(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(AutoProcessor, "from_pretrained") as processor_mock,
            patch.object(AutoModelForCTC, "from_pretrained") as model_mock,
        ):
            settings = EngineSettings(auto_load_model=False)
            model = SpeechRecognitionModel(
                self.model_id,
                settings,
                logger=logger_mock,
            )
            self.assertIsInstance(model, SpeechRecognitionModel)
            processor_mock.assert_not_called()
            model_mock.assert_not_called()

    def test_instantiation_with_load_model(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(AutoProcessor, "from_pretrained") as processor_mock,
            patch.object(AutoModelForCTC, "from_pretrained") as model_mock,
        ):
            processor_instance = MagicMock()
            type(processor_instance.tokenizer).pad_token_id = PropertyMock(
                return_value=1
            )
            processor_mock.return_value = processor_instance

            model_instance = MagicMock(spec=PreTrainedModel)
            model_mock.return_value = model_instance

            settings = EngineSettings()
            model = SpeechRecognitionModel(
                self.model_id,
                settings,
                logger=logger_mock,
            )
            self.assertIs(model.model, model_instance)
            processor_mock.assert_called_once_with(
                self.model_id,
                trust_remote_code=False,
                use_fast=True,
            )
            model_mock.assert_called_once_with(
                self.model_id,
                trust_remote_code=False,
                pad_token_id=processor_instance.tokenizer.pad_token_id,
                ctc_loss_reduction="mean",
                device_map=Engine.get_default_device(),
                tp_plan=None,
                ignore_mismatched_sizes=True,
            )


class SpeechRecognitionModelCallTestCase(IsolatedAsyncioTestCase):
    model_id = "dummy/model"

    async def test_call_with_resampling(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(AutoProcessor, "from_pretrained") as processor_mock,
            patch.object(AutoModelForCTC, "from_pretrained") as model_mock,
            patch.object(
                SpeechRecognitionModel, "_resample"
            ) as resample_method,
            patch("avalan.model.audio.argmax") as argmax_mock,
            patch(
                "avalan.model.audio.inference_mode", return_value=nullcontext()
            ) as inference_mode_mock,
        ):
            processor_instance = MagicMock()
            processor_call = MagicMock(input_values="inputs")
            processor_call.to.return_value = processor_call
            processor_instance.return_value = processor_call
            processor_instance.batch_decode.return_value = ["ok"]
            type(processor_instance.tokenizer).pad_token_id = PropertyMock(
                return_value=1
            )
            processor_mock.return_value = processor_instance

            model_instance = MagicMock(spec=PreTrainedModel)
            call_result = MagicMock(logits="logits")
            model_instance.return_value = call_result
            model_mock.return_value = model_instance

            resample_method.return_value = "audio"

            argmax_mock.return_value = "pred"

            settings = EngineSettings()
            model = SpeechRecognitionModel(
                self.model_id,
                settings,
                logger=logger_mock,
            )

            result = await model("file.wav", sampling_rate=16000)

            self.assertEqual(result, "ok")
            resample_method.assert_called_once_with("file.wav", 16000)
            processor_instance.assert_called_with(
                "audio",
                sampling_rate=16000,
                return_tensors="pt",
            )
            model_instance.assert_called_once_with("inputs")
            argmax_mock.assert_called_once_with("logits", dim=-1)
            processor_instance.batch_decode.assert_called_once_with("pred")
            inference_mode_mock.assert_called_once_with()


if __name__ == "__main__":
    main()


class SpeechRecognitionNoResampleTestCase(IsolatedAsyncioTestCase):
    model_id = "dummy/model"

    async def test_call_without_resampling(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(AutoProcessor, "from_pretrained") as processor_mock,
            patch.object(AutoModelForCTC, "from_pretrained") as model_mock,
            patch.object(
                SpeechRecognitionModel, "_resample"
            ) as resample_method,
            patch("avalan.model.audio.argmax") as argmax_mock,
            patch(
                "avalan.model.audio.inference_mode", return_value=nullcontext()
            ) as inf_mock,
        ):
            processor_instance = MagicMock()
            processor_call = MagicMock(input_values="inputs")
            processor_call.to.return_value = processor_call
            processor_instance.return_value = processor_call
            processor_instance.batch_decode.return_value = ["ok"]
            type(processor_instance.tokenizer).pad_token_id = PropertyMock(
                return_value=1
            )
            processor_mock.return_value = processor_instance

            model_instance = MagicMock(spec=PreTrainedModel)
            call_result = MagicMock(logits="logits")
            model_instance.return_value = call_result
            model_mock.return_value = model_instance

            resample_method.return_value = "audio"

            argmax_mock.return_value = "pred"

            settings = EngineSettings()
            model = SpeechRecognitionModel(
                self.model_id,
                settings,
                logger=logger_mock,
            )

            result = await model("file.wav", sampling_rate=16000)

            self.assertEqual(result, "ok")
            resample_method.assert_called_once_with("file.wav", 16000)
            processor_instance.assert_called_with(
                "audio",
                sampling_rate=16000,
                return_tensors="pt",
            )
            model_instance.assert_called_once_with("inputs")
            argmax_mock.assert_called_once_with("logits", dim=-1)
            processor_instance.batch_decode.assert_called_once_with("pred")
            inf_mock.assert_called_once_with()
