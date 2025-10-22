from logging import Logger
from unittest import IsolatedAsyncioTestCase, TestCase, main
from unittest.mock import MagicMock, PropertyMock, call, patch

from pytest import importorskip
from transformers import PreTrainedModel

from avalan.entities import EngineSettings, ImageEntity
from avalan.model.engine import Engine
from avalan.model.vision.detection import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    ObjectDetectionModel,
)

importorskip("torch", reason="torch not installed")
importorskip("PIL", reason="Pillow not installed")


class DummyInputs(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.to_called_with = None

    def to(self, device):
        self.to_called_with = device
        return self


class ObjectDetectionModelInstantiationTestCase(TestCase):
    model_id = "dummy/model"

    def test_instantiation_no_load(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(
                AutoImageProcessor, "from_pretrained"
            ) as processor_mock,
            patch.object(
                AutoModelForObjectDetection, "from_pretrained"
            ) as model_mock,
        ):
            model = ObjectDetectionModel(
                self.model_id,
                EngineSettings(auto_load_model=False),
                logger=logger_mock,
            )
            self.assertIsInstance(model, ObjectDetectionModel)
            processor_mock.assert_not_called()
            model_mock.assert_not_called()

    def test_instantiation_with_load_model(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(
                AutoImageProcessor, "from_pretrained"
            ) as processor_mock,
            patch.object(
                AutoModelForObjectDetection, "from_pretrained"
            ) as model_mock,
        ):
            processor_instance = MagicMock()
            processor_mock.return_value = processor_instance

            model_instance = MagicMock(spec=PreTrainedModel)
            model_mock.return_value = model_instance

            model = ObjectDetectionModel(
                self.model_id,
                EngineSettings(),
                logger=logger_mock,
            )
            self.assertIs(model.model, model_instance)
            processor_mock.assert_called_once_with(
                self.model_id,
                revision="no_timm",
                use_fast=True,
            )
            model_mock.assert_called_once_with(
                self.model_id,
                revision="no_timm",
                device_map=Engine.get_default_device(),
                tp_plan=None,
                distributed_config=None,
            )


class ObjectDetectionModelCallTestCase(IsolatedAsyncioTestCase):
    model_id = "dummy/model"

    async def test_call(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(
                AutoImageProcessor, "from_pretrained"
            ) as processor_mock,
            patch.object(
                AutoModelForObjectDetection, "from_pretrained"
            ) as model_mock,
            patch(
                "avalan.model.vision.detection.BaseVisionModel._get_image"
            ) as get_image_mock,
            patch("avalan.model.vision.detection.tensor") as tensor_mock,
        ):
            # mock processor
            processor_instance = MagicMock()
            processor_instance.return_value = DummyInputs(
                pixel_values="inputs"
            )
            score_tensor = MagicMock()
            score_tensor.item.return_value = 0.9
            label_tensor = MagicMock()
            label_tensor.item.return_value = 1
            box_tensor = MagicMock()
            box_tensor.tolist.return_value = [0.1, 0.2, 0.3, 0.4]
            processor_instance.post_process_object_detection.return_value = [
                {
                    "scores": [score_tensor],
                    "labels": [label_tensor],
                    "boxes": [box_tensor],
                }
            ]
            processor_mock.return_value = processor_instance

            # mock model
            model_instance = MagicMock(spec=PreTrainedModel)
            config_mock = MagicMock()
            config_mock.id2label = {1: "label"}
            type(model_instance).config = PropertyMock(
                return_value=config_mock
            )
            model_instance.return_value = "outputs"
            model_mock.return_value = model_instance

            # other mocks
            image_mock = MagicMock()
            image_mock.size = (10, 20)
            get_image_mock.return_value = image_mock
            tensor_mock.return_value = "target_sizes"

            model = ObjectDetectionModel(
                self.model_id,
                EngineSettings(),
                logger=logger_mock,
            )

            entities = await model("path.jpg", threshold=0.5)

            self.assertEqual(
                entities,
                [
                    ImageEntity(
                        label="label", score=0.9, box=[0.1, 0.2, 0.3, 0.4]
                    )
                ],
            )
            get_image_mock.assert_called_once_with("path.jpg")
            processor_instance.assert_called_once_with(
                images=image_mock,
                return_tensors="pt",
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

            tensor_mock.assert_called_once_with([image_mock.size[::-1]])
            processor_instance.post_process_object_detection.assert_called_once_with(
                "outputs",
                target_sizes="target_sizes",
                threshold=0.5,
            )


if __name__ == "__main__":
    main()
