from avalan.entities import EngineSettings
from avalan.model.engine import Engine
from avalan.model.audio.speech import (
    AutoProcessor,
    TextToSpeechModel,
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
                subfolder="",
            )
            model_mock.assert_called_once_with(
                self.model_id,
                trust_remote_code=False,
                device_map=Engine.get_default_device(),
                tp_plan=None,
                distributed_config=None,
                subfolder="",
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
                "avalan.model.audio.speech.inference_mode",
                return_value=nullcontext(),
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
                text=["hi"],
                audio=None,
                padding=False,
                return_tensors="pt",
                sampling_rate=44_100,
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


class TextToSpeechModelReferenceTestCase(IsolatedAsyncioTestCase):
    model_id = "dummy/model"

    async def test_reference_validation(self):
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

            with self.assertRaises(AssertionError):
                await model("hi", "file.wav", 3, reference_path="voice.wav")

    async def test_call_with_reference_resample(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(AutoProcessor, "from_pretrained") as processor_mock,
            patch.object(
                DiaForConditionalGeneration, "from_pretrained"
            ) as model_mock,
            patch.object(TextToSpeechModel, "_resample") as resample_method,
            patch(
                "avalan.model.audio.speech.inference_mode",
                return_value=nullcontext(),
            ) as inf_mock,
        ):
            call_result = MagicMock()
            inputs = {"input_ids": [1], "decoder_attention_mask": "mask"}
            call_result.to.return_value = inputs
            processor_instance = MagicMock(return_value=call_result)
            processor_instance.batch_decode.return_value = ["audio"]
            processor_instance.get_audio_prompt_len.return_value = 2
            processor_mock.return_value = processor_instance

            model_instance = MagicMock(spec=PreTrainedModel)
            outputs = MagicMock(shape=(1, 5))
            model_instance.generate = MagicMock(return_value=outputs)
            model_mock.return_value = model_instance

            resample_method.return_value = "voice"

            settings = EngineSettings()
            model = TextToSpeechModel(
                self.model_id, settings, logger=logger_mock
            )

            result = await model(
                "hi",
                "file.wav",
                3,
                reference_path="ref.wav",
                reference_text="ref",
                sampling_rate=16000,
            )

            self.assertEqual(result, "file.wav")
            resample_method.assert_called_once_with("ref.wav", 16000)
            processor_instance.assert_called_with(
                text="ref\nhi",
                audio="voice",
                padding=True,
                return_tensors="pt",
                sampling_rate=16000,
            )
            call_result.to.assert_called_once_with(model._device)
            processor_instance.get_audio_prompt_len.assert_called_once_with(
                inputs["decoder_attention_mask"]
            )
            model_instance.generate.assert_called_once_with(
                **inputs, max_new_tokens=3
            )
            processor_instance.batch_decode.assert_called_once_with(
                outputs, audio_prompt_len=2
            )
            processor_instance.save_audio.assert_called_once_with(
                ["audio"], "file.wav"
            )
            inf_mock.assert_called_once_with()

    async def test_call_with_reference_no_resample(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(AutoProcessor, "from_pretrained") as processor_mock,
            patch.object(
                DiaForConditionalGeneration, "from_pretrained"
            ) as model_mock,
            patch.object(TextToSpeechModel, "_resample") as resample_method,
            patch(
                "avalan.model.audio.speech.inference_mode",
                return_value=nullcontext(),
            ) as inf_mock,
        ):
            call_result = MagicMock()
            inputs = {"input_ids": [1], "decoder_attention_mask": "mask"}
            call_result.to.return_value = inputs
            processor_instance = MagicMock(return_value=call_result)
            processor_instance.batch_decode.return_value = ["audio"]
            processor_instance.get_audio_prompt_len.return_value = 2
            processor_mock.return_value = processor_instance

            model_instance = MagicMock(spec=PreTrainedModel)
            outputs = MagicMock(shape=(1, 1))
            model_instance.generate = MagicMock(return_value=outputs)
            model_mock.return_value = model_instance

            resample_method.return_value = "voice"

            settings = EngineSettings()
            model = TextToSpeechModel(
                self.model_id, settings, logger=logger_mock
            )

            result = await model(
                "hi",
                "file.wav",
                3,
                reference_path="ref.wav",
                reference_text="ref",
                sampling_rate=16000,
            )

            self.assertEqual(result, "file.wav")
            resample_method.assert_called_once_with("ref.wav", 16000)
            processor_instance.assert_called_with(
                text="ref\nhi",
                audio="voice",
                padding=True,
                return_tensors="pt",
                sampling_rate=16000,
            )
            call_result.to.assert_called_once_with(model._device)
            processor_instance.get_audio_prompt_len.assert_called_once_with(
                inputs["decoder_attention_mask"]
            )
            model_instance.generate.assert_called_once_with(
                **inputs, max_new_tokens=3
            )
            processor_instance.batch_decode.assert_called_once_with(outputs)
            processor_instance.save_audio.assert_called_once_with(
                ["audio"], "file.wav"
            )
            inf_mock.assert_called_once_with()

    async def test_call_sampling_rate(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(AutoProcessor, "from_pretrained") as processor_mock,
            patch.object(
                DiaForConditionalGeneration, "from_pretrained"
            ) as model_mock,
            patch(
                "avalan.model.audio.speech.inference_mode",
                return_value=nullcontext(),
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

            result = await model(
                ["hi"], "file.wav", 3, padding=False, sampling_rate=16000
            )

            self.assertEqual(result, "file.wav")
            processor_instance.assert_called_with(
                text=["hi"],
                audio=None,
                padding=False,
                return_tensors="pt",
                sampling_rate=16000,
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
