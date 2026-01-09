from ...compat import override
from ...entities import EngineSettings, ImageEntity
from ...model.engine import Engine
from ...model.vendor import TextGenerationVendor
from ...model.vision import BaseVisionModel
from ...model.vision.classification import ImageClassificationModel

from logging import Logger, getLogger
from typing import Any, Literal

from diffusers import DiffusionPipeline
from PIL import Image
from torch import inference_mode, tensor
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    PreTrainedModel,
)


class ObjectDetectionModel(ImageClassificationModel):
    def __init__(
        self,
        model_id: str,
        settings: EngineSettings | None = None,
        revision: Literal["no_timm"] = "no_timm",
        logger: Logger = getLogger(__name__),
    ):
        self._revision = revision
        super().__init__(model_id, settings, logger=logger)

    def _load_model(
        self,
    ) -> PreTrainedModel | TextGenerationVendor | DiffusionPipeline:
        assert self._model_id is not None, "Model ID is required"
        self._processor = AutoImageProcessor.from_pretrained(
            self._model_id,
            revision=self._revision,
            # default behavior in transformers v4.48
            use_fast=True,
        )
        model: PreTrainedModel = AutoModelForObjectDetection.from_pretrained(
            self._model_id,
            revision=self._revision,
            device_map=self._device,
            tp_plan=Engine._get_tp_plan(self._settings.parallel),
            distributed_config=Engine._get_distributed_config(
                self._settings.distributed_config
            ),
        )
        return model

    @override
    async def __call__(  # type: ignore[override]
        self,
        image_source: str | Image.Image,
        threshold: float | None = 0.3,
        tensor_format: Literal["pt"] = "pt",
    ) -> list[ImageEntity]:
        assert self._model is not None, "Model must be loaded"
        assert isinstance(
            self._model, PreTrainedModel
        ), "Model must be PreTrainedModel"
        image = BaseVisionModel._get_image(image_source)
        processor: Any = self._processor
        inputs: Any = processor(images=image, return_tensors=tensor_format)
        inputs.to(self._device)
        with inference_mode():
            outputs = self._model(**inputs)  # type: ignore[operator]
        target_sizes = tensor([image.size[::-1]])
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=threshold
        )[0]

        id2label: dict[int, str] = self._model.config.id2label  # type: ignore[assignment]
        entities = []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            box_coords = [round(i, 2) for i in box.tolist()]
            entities.append(
                ImageEntity(
                    label=id2label[label.item()],
                    score=score.item(),
                    box=box_coords,
                )
            )
        return entities
