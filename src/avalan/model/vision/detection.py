from ...compat import override
from ...entities import ImageEntity, EngineSettings
from ...model import TextGenerationVendor
from ...model.vision import BaseVisionModel
from ...model.vision.image import ImageClassificationModel
from logging import Logger
from PIL import Image
from torch import tensor
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    PreTrainedModel,
)
from typing import Literal


class ObjectDetectionModel(ImageClassificationModel):
    def __init__(
        self,
        model_id: str,
        settings: EngineSettings | None = None,
        revision: Literal["no_timm"] = "no_timm",
        logger: Logger | None = None,
    ):
        self._revision = revision
        super().__init__(model_id, settings, logger=logger)

    def _load_model(self) -> PreTrainedModel | TextGenerationVendor:
        self._processor = AutoImageProcessor.from_pretrained(
            self._model_id,
            revision=self._revision,
            # default behavior in transformers v4.48
            use_fast=True,
        )
        model = AutoModelForObjectDetection.from_pretrained(
            self._model_id,
            revision=self._revision,
            device_map=self._device,
        )
        return model

    @override
    async def __call__(
        self,
        image_source: str | Image.Image,
        threshold: float | None = 0.3,
        tensor_format: Literal["pt"] = "pt",
    ) -> list[ImageEntity]:
        image = BaseVisionModel._get_image(image_source)
        inputs = self._processor(images=image, return_tensors=tensor_format)
        outputs = self._model(**inputs)
        target_sizes = tensor([image.size[::-1]])
        results = self._processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=threshold
        )[0]

        entities = []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            box = [round(i, 2) for i in box.tolist()]
            entities.append(
                ImageEntity(
                    label=self._model.config.id2label[label.item()],
                    score=score.item(),
                    box=box,
                )
            )
        return entities
