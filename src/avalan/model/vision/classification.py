from ...compat import override
from ...entities import ImageEntity
from ...model.engine import Engine
from ...model.vendor import TextGenerationVendor
from ...model.vision import BaseVisionModel

from typing import Any, Literal

from diffusers import DiffusionPipeline
from PIL import Image
from torch import inference_mode
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    PreTrainedModel,
)


# model predicts one of the 1000 ImageNet classes
class ImageClassificationModel(BaseVisionModel):
    _processor: AutoImageProcessor

    def _load_model(
        self,
    ) -> PreTrainedModel | TextGenerationVendor | DiffusionPipeline:
        assert self._model_id is not None, "Model ID is required"
        self._processor = AutoImageProcessor.from_pretrained(
            self._model_id,
            # default behavior in transformers v4.48
            use_fast=True,
        )
        model: PreTrainedModel = (
            AutoModelForImageClassification.from_pretrained(
                self._model_id,
                device_map=self._device,
                tp_plan=Engine._get_tp_plan(self._settings.parallel),
                distributed_config=Engine._get_distributed_config(
                    self._settings.distributed_config
                ),
            )
        )
        return model

    @override
    async def __call__(  # type: ignore[override]
        self,
        image_source: str | Image.Image,
        tensor_format: Literal["pt"] = "pt",
    ) -> ImageEntity:
        assert self._model is not None, "Model must be loaded"
        assert isinstance(
            self._model, PreTrainedModel
        ), "Model must be PreTrainedModel"
        image = BaseVisionModel._get_image(image_source)
        inputs: Any = self._processor(  # type: ignore[operator]
            image, return_tensors=tensor_format
        )
        inputs.to(self._device)

        with inference_mode():
            logits = self._model(**inputs).logits  # type: ignore[operator]

        label_index = logits.argmax(dim=1).item()
        id2label: dict[int, str] = self._model.config.id2label  # type: ignore[assignment]
        return ImageEntity(label=id2label[label_index])
