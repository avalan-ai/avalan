from avalan.entities import EngineSettings, ImageEntity
from avalan.model.engine import Engine
from avalan.model.vision.image import (
    ImageClassificationModel,
    AutoImageProcessor,
    AutoModelForImageClassification,
    BaseVisionModel,
)
from contextlib import nullcontext
from logging import Logger
from transformers import PreTrainedModel
from unittest import TestCase, IsolatedAsyncioTestCase, main
from unittest.mock import call, MagicMock, patch, PropertyMock


class DummyInputs(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.to_called_with = None

    def to(self, device):
        self.to_called_with = device
        return self


class ImageClassificationModelInstantiationTestCase(TestCase):
    model_id = "dummy/model"

    def test_instantiation_no_load(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(
                AutoImageProcessor, "from_pretrained"
            ) as processor_mock,
            patch.object(
                AutoModelForImageClassification, "from_pretrained"
            ) as model_mock,
        ):
            settings = EngineSettings(auto_load_model=False)
            model = ImageClassificationModel(
                self.model_id,
                settings,
                logger=logger_mock,
            )
            self.assertIsInstance(model, ImageClassificationModel)
            processor_mock.assert_not_called()
            model_mock.assert_not_called()

    def test_instantiation_with_load_model(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(
                AutoImageProcessor, "from_pretrained"
            ) as processor_mock,
            patch.object(
                AutoModelForImageClassification, "from_pretrained"
            ) as model_mock,
        ):
            processor_instance = MagicMock()
            processor_mock.return_value = processor_instance

            model_instance = MagicMock(spec=PreTrainedModel)
            model_mock.return_value = model_instance

            settings = EngineSettings()
            model = ImageClassificationModel(
                self.model_id,
                settings,
                logger=logger_mock,
            )
            self.assertIs(model.model, model_instance)
            self.assertIs(model._processor, processor_instance)
            processor_mock.assert_called_once_with(
                self.model_id,
                use_fast=True,
            )
            model_mock.assert_called_once_with(
                self.model_id,
                device_map=Engine.get_default_device(),
                tp_plan=None,
            )


class ImageClassificationModelCallTestCase(IsolatedAsyncioTestCase):
    model_id = "dummy/model"

    async def test_call(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(
                AutoImageProcessor, "from_pretrained"
            ) as processor_mock,
            patch.object(
                AutoModelForImageClassification, "from_pretrained"
            ) as model_mock,
            patch(
                "avalan.model.vision.image.inference_mode",
                return_value=nullcontext(),
            ) as inference_mode_mock,
            patch.object(BaseVisionModel, "_get_image") as get_image_mock,
        ):
            processor_instance = MagicMock()
            processor_instance.return_value = DummyInputs(
                pixel_values="inputs"
            )
            processor_mock.return_value = processor_instance

            logits = MagicMock()
            argmax_result = MagicMock()
            argmax_result.item.return_value = 0
            logits.argmax.return_value = argmax_result

            model_instance = MagicMock(spec=PreTrainedModel)
            model_instance.return_value = MagicMock(logits=logits)
            config_mock = MagicMock()
            config_mock.id2label = {0: "cat"}
            type(model_instance).config = PropertyMock(
                return_value=config_mock
            )
            model_mock.return_value = model_instance

            image_mock = MagicMock()
            get_image_mock.return_value = image_mock

            model = ImageClassificationModel(
                self.model_id,
                EngineSettings(),
                logger=logger_mock,
            )

            result = await model("image.jpg")

            self.assertIsInstance(result, ImageEntity)
            self.assertEqual(result.label, "cat")
            processor_instance.assert_called_once_with(
                image_mock, return_tensors="pt"
            )
            model_instance.assert_called_once()
            self.assertEqual(model_instance.call_count, 1)
            self.assertEqual(
                model_instance.call_args, call(**{"pixel_values": "inputs"})
            )
            self.assertEqual(
                processor_instance.return_value.to_called_with,
                model._device,
            )
            inference_mode_mock.assert_called_once_with()


if __name__ == "__main__":
    main()
