from ...compat import override
from ...entities import ImageEntity
from ...model.vendor import TextGenerationVendor
from ...model.engine import Engine
from ...model.vision import BaseVisionModel
from diffusers import DiffusionPipeline
from PIL import Image
from torch import inference_mode
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    PreTrainedModel,
)
from typing import Literal


# model predicts one of the 1000 ImageNet classes
class ImageClassificationModel(BaseVisionModel):
    def _load_model(
        self,
    ) -> PreTrainedModel | TextGenerationVendor | DiffusionPipeline:
        self._processor = AutoImageProcessor.from_pretrained(
            self._model_id,
            # default behavior in transformers v4.48
            use_fast=True,
        )
        model = AutoModelForImageClassification.from_pretrained(
            self._model_id,
            device_map=self._device,
            tp_plan=Engine._get_tp_plan(self._settings.parallel),
        )
        return model

    @override
    async def __call__(
        self,
        image_source: str | Image.Image,
        tensor_format: Literal["pt"] = "pt",
    ) -> ImageEntity:
        image = BaseVisionModel._get_image(image_source)
        inputs = self._processor(image, return_tensors=tensor_format)
        inputs.to(self._device)

        with inference_mode():
            logits = self._model(**inputs).logits

        label_index = logits.argmax(dim=1).item()
        return ImageEntity(label=self._model.config.id2label[label_index])
