from avalan.entities import EngineSettings
from avalan.model.engine import Engine
from avalan.model.audio.generation import (
    AutoProcessor,
    AudioGenerationModel,
    MusicgenForConditionalGeneration,
)
from contextlib import nullcontext
from logging import Logger
from transformers import PreTrainedModel
from unittest import TestCase, IsolatedAsyncioTestCase, main
from unittest.mock import MagicMock, patch


class AudioGenerationModelInstantiationTestCase(TestCase):
    model_id = "dummy/model"

    def test_instantiation_no_load(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(AutoProcessor, "from_pretrained") as processor_mock,
            patch.object(
                MusicgenForConditionalGeneration, "from_pretrained"
            ) as model_mock,
        ):
            settings = EngineSettings(auto_load_model=False)
            model = AudioGenerationModel(
                self.model_id, settings, logger=logger_mock
            )
            self.assertIsInstance(model, AudioGenerationModel)
            processor_mock.assert_not_called()
            model_mock.assert_not_called()

    def test_instantiation_with_load_model(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(AutoProcessor, "from_pretrained") as processor_mock,
            patch.object(
                MusicgenForConditionalGeneration, "from_pretrained"
            ) as model_mock,
        ):
            processor_instance = MagicMock()
            processor_mock.return_value = processor_instance

            model_instance = MagicMock(spec=PreTrainedModel)
            model_instance.to.return_value = model_instance
            model_mock.return_value = model_instance

            settings = EngineSettings()
            model = AudioGenerationModel(
                self.model_id, settings, logger=logger_mock
            )
            self.assertIs(model.model, model_instance)
            processor_mock.assert_called_once_with(self.model_id)
            model_mock.assert_called_once_with(
                self.model_id,
                device_map=Engine.get_default_device(),
                tp_plan=None,
                distributed_config=None,
                subfolder="",
            )


class AudioGenerationModelCallTestCase(IsolatedAsyncioTestCase):
    model_id = "dummy/model"

    async def test_call(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(AutoProcessor, "from_pretrained") as processor_mock,
            patch.object(
                MusicgenForConditionalGeneration, "from_pretrained"
            ) as model_mock,
            patch(
                "avalan.model.audio.generation.inference_mode",
                return_value=nullcontext(),
            ) as inf_mock,
            patch(
                "avalan.model.audio.generation.from_numpy"
            ) as from_numpy_mock,
            patch("avalan.model.audio.generation.save") as save_mock,
        ):

            class Dummy(dict):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.to = MagicMock(return_value=self)

            inputs = Dummy({"input_ids": [1]})
            processor_instance = MagicMock(return_value=inputs)
            processor_mock.return_value = processor_instance

            model_instance = MagicMock(spec=PreTrainedModel)
            model_instance.to.return_value = model_instance
            audio_tokens = MagicMock()
            inner = MagicMock()
            cpu_obj = MagicMock()
            cpu_obj.numpy.return_value = "waveform"
            inner.cpu.return_value = cpu_obj
            audio_tokens.__getitem__.return_value = inner
            model_instance.generate = MagicMock(return_value=audio_tokens)
            type(model_instance).config = MagicMock(
                audio_encoder=MagicMock(sampling_rate=44_100)
            )
            model_mock.return_value = model_instance

            from_numpy_result = MagicMock(
                unsqueeze=MagicMock(return_value="wave")
            )
            from_numpy_mock.return_value = from_numpy_result

            settings = EngineSettings()
            model = AudioGenerationModel(
                self.model_id, settings, logger=logger_mock
            )

            result = await model("hi", "file.wav", 3, padding=False)

            self.assertEqual(result, "file.wav")
            processor_instance.assert_called_with(
                text=["hi"],
                return_tensors="pt",
                padding=False,
            )
            inputs.to.assert_called_once_with(model._device)
            model_instance.generate.assert_called_once_with(
                **inputs, max_new_tokens=3
            )
            from_numpy_mock.assert_called_once_with("waveform")
            from_numpy_result.unsqueeze.assert_called_once_with(0)
            save_mock.assert_called_once_with(
                "file.wav", from_numpy_result.unsqueeze.return_value, 44_100
            )
            inf_mock.assert_called_once_with()


if __name__ == "__main__":
    main()
