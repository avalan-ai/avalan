from unittest import TestCase, IsolatedAsyncioTestCase, main
from unittest.mock import MagicMock, patch, PropertyMock
from contextlib import nullcontext
from logging import Logger
from pytest import importorskip

from avalan.model.entities import EngineSettings, ImageEntity

importorskip("PIL", reason="Pillow not installed")

from avalan.model.vision.image import (
    ImageClassificationModel,
    AutoImageProcessor,
    AutoModelForImageClassification,
    BaseVisionModel,
)


class ImageClassificationModelInstantiationTestCase(TestCase):
    model_id = "dummy/model"

    def test_instantiation_no_load(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(AutoImageProcessor, "from_pretrained") as processor_mock,
            patch.object(AutoModelForImageClassification, "from_pretrained") as model_mock,
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
            patch.object(AutoImageProcessor, "from_pretrained") as processor_mock,
            patch.object(AutoModelForImageClassification, "from_pretrained") as model_mock,
        ):
            processor_instance = MagicMock()
            processor_mock.return_value = processor_instance

            model_instance = MagicMock()
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
            model_mock.assert_called_once_with(self.model_id)


class ImageClassificationModelCallTestCase(IsolatedAsyncioTestCase):
    model_id = "dummy/model"

    async def test_call(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(AutoImageProcessor, "from_pretrained") as processor_mock,
            patch.object(AutoModelForImageClassification, "from_pretrained") as model_mock,
            patch("avalan.model.vision.image.no_grad", new=lambda: nullcontext()),
            patch.object(BaseVisionModel, "_get_image") as get_image_mock,
        ):
            processor_instance = MagicMock()
            processor_instance.return_value = {"pixel_values": "inputs"}
            processor_mock.return_value = processor_instance

            logits = MagicMock()
            argmax_result = MagicMock()
            argmax_result.item.return_value = 0
            logits.argmax.return_value = argmax_result

            model_instance = MagicMock()
            model_instance.return_value = MagicMock(logits=logits)
            config_mock = MagicMock()
            config_mock.id2label = {0: "cat"}
            type(model_instance).config = PropertyMock(return_value=config_mock)
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
            processor_instance.assert_called_once_with(image_mock, return_tensors="pt")
            model_instance.assert_called_once_with(**{"pixel_values": "inputs"})


if __name__ == "__main__":
    main()
