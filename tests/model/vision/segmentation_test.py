from avalan.entities import EngineSettings
from avalan.model.engine import Engine
from avalan.model.vision.segmentation import (
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
    SemanticSegmentationModel,
)
from avalan.model.vision import BaseVisionModel
from logging import Logger
from transformers import PreTrainedModel
from unittest import IsolatedAsyncioTestCase, TestCase, main
from unittest.mock import call, MagicMock, patch


class DummyInputs(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.to_called_with = None

    def to(self, device):
        self.to_called_with = device
        return self


class SemanticSegmentationModelInstantiationTestCase(TestCase):
    model_id = "dummy/model"

    def test_instantiation_with_load_model(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(
                AutoImageProcessor, "from_pretrained"
            ) as processor_mock,
            patch.object(
                AutoModelForSemanticSegmentation, "from_pretrained"
            ) as model_mock,
        ):
            processor_instance = MagicMock()
            processor_mock.return_value = processor_instance

            model_instance = MagicMock(spec=PreTrainedModel)
            model_mock.return_value = model_instance

            model = SemanticSegmentationModel(
                self.model_id,
                EngineSettings(),
                logger=logger_mock,
            )

            self.assertIs(model.model, model_instance)
            processor_mock.assert_called_once_with(
                self.model_id, use_fast=True
            )
            model_mock.assert_called_once_with(
                self.model_id,
                device_map=Engine.get_default_device(),
                tp_plan=None,
                distributed_config=None,
            )
            # model_instance.eval.assert_called_once_with()


class SemanticSegmentationModelCallTestCase(IsolatedAsyncioTestCase):
    model_id = "dummy/model"

    async def test_call(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(
                AutoImageProcessor, "from_pretrained"
            ) as processor_mock,
            patch.object(
                AutoModelForSemanticSegmentation, "from_pretrained"
            ) as model_mock,
            patch("avalan.model.vision.segmentation.unique") as unique_mock,
            patch.object(BaseVisionModel, "_get_image") as get_image_mock,
        ):
            image_mock = MagicMock()
            get_image_mock.return_value = image_mock

            processor_instance = MagicMock()
            processor_result = DummyInputs(inputs="inputs")
            processor_instance.return_value = processor_result
            processor_mock.return_value = processor_instance

            model_instance = MagicMock(spec=PreTrainedModel)
            config = MagicMock()
            config.id2label = {0: "zero", 1: "one"}
            model_instance.config = config

            call_result = MagicMock()
            logits_mock = MagicMock()
            argmax_result = MagicMock()
            mask_tensor = MagicMock()
            argmax_result.__getitem__.return_value = mask_tensor
            logits_mock.argmax.return_value = argmax_result
            call_result.logits = logits_mock
            model_instance.return_value = call_result
            model_mock.return_value = model_instance

            idx0, idx1 = MagicMock(), MagicMock()
            idx0.item.return_value = 0
            idx1.item.return_value = 1
            unique_mock.return_value = [idx0, idx1]

            model = SemanticSegmentationModel(
                self.model_id,
                EngineSettings(),
                logger=logger_mock,
            )

            result = await model("file.png")

            self.assertEqual(result, ["zero", "one"])
            get_image_mock.assert_called_once_with("file.png")
            processor_instance.assert_called_once_with(
                images=image_mock, return_tensors="pt"
            )
            model_instance.assert_called_once()
            self.assertEqual(model_instance.call_count, 1)
            self.assertEqual(
                model_instance.call_args, call(**processor_result)
            )
            self.assertEqual(
                processor_result.to_called_with,
                model._device,
            )

            unique_mock.assert_called_once_with(mask_tensor)


if __name__ == "__main__":
    main()
