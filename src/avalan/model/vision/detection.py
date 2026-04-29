from ...entities import EngineSettings, ImageEntity
from ...model.engine import Engine
from ...model.vendor import TextGenerationVendor
from ...model.vision import BaseVisionModel
from ...model.vision.classification import ImageClassificationModel

from collections.abc import Iterator
from contextlib import contextmanager
from logging import Logger, getLogger
from os import environ
from typing import Any, Literal, cast

from diffusers import DiffusionPipeline
from PIL import Image
from torch import inference_mode, tensor
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForObjectDetection,
    PretrainedConfig,
    PreTrainedModel,
)

_DISABLE_SAFETENSORS_CONVERSION = "DISABLE_SAFETENSORS_CONVERSION"


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
        config = self._load_config()
        self._processor = cast(
            Any,
            cast(Any, AutoImageProcessor).from_pretrained(
                self._model_id,
                revision=self._revision,
                backend="torchvision",
            ),
        )
        assert self._model_id
        with self._disable_safetensors_conversion():
            model = cast(
                PreTrainedModel,
                cast(Any, AutoModelForObjectDetection).from_pretrained(
                    self._model_id,
                    config=config,
                    revision=self._revision,
                    device_map=self._device,
                    tp_plan=Engine._get_tp_plan(self._settings.parallel),
                    distributed_config=Engine._get_distributed_config(
                        self._settings.distributed_config
                    ),
                ),
            )
        return model

    @contextmanager
    def _disable_safetensors_conversion(self) -> Iterator[None]:
        previous_value = environ.get(_DISABLE_SAFETENSORS_CONVERSION)
        environ[_DISABLE_SAFETENSORS_CONVERSION] = "1"
        try:
            yield
        finally:
            if previous_value is None:
                environ.pop(_DISABLE_SAFETENSORS_CONVERSION, None)
            else:
                environ[_DISABLE_SAFETENSORS_CONVERSION] = previous_value

    def _load_config(self) -> PretrainedConfig:
        assert self._model_id
        config_dict, _ = PretrainedConfig.get_config_dict(
            self._model_id,
            revision=self._revision,
        )
        if (
            config_dict.get("model_type") == "detr"
            and config_dict.get("dilation") is None
        ):
            normalized_config = dict(config_dict)
            normalized_config["dilation"] = False
            model_type = cast(str, normalized_config.pop("model_type"))
            return AutoConfig.for_model(model_type, **normalized_config)

        return cast(
            PretrainedConfig,
            AutoConfig.from_pretrained(
                self._model_id,
                revision=self._revision,
            ),
        )

    async def __call__(
        self,
        image_source: str | Image.Image,
        threshold: float | None = 0.3,
        tensor_format: Literal["pt"] = "pt",
    ) -> list[ImageEntity]:
        image = BaseVisionModel._get_image(image_source)
        inputs = self._processor(images=image, return_tensors=tensor_format)
        inputs.to(self._device)
        with inference_mode():
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
