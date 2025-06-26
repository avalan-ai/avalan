from ...compat import override
from ...model import TextGenerationVendor
from ...model.vision import BaseVisionModel
from ...model.engine import Engine
from PIL import Image
from torch import unique
from transformers import (
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
    PreTrainedModel,
)
from typing import Literal


class SemanticSegmentationModel(BaseVisionModel):
    def _load_model(self) -> PreTrainedModel | TextGenerationVendor:
        self._processor = AutoImageProcessor.from_pretrained(
            self._model_id,
            # default behavior in transformers v4.48
            use_fast=True,
        )
        model = AutoModelForSemanticSegmentation.from_pretrained(
            self._model_id,
            device_map=self._device,
            tp_plan=Engine._get_tp_plan(self._settings.parallel),
        )
        model.eval()
        return model

    @override
    async def __call__(
        self,
        image_source: str | Image.Image,
        tensor_format: Literal["pt"] = "pt",
    ) -> list[str]:
        image = BaseVisionModel._get_image(image_source)
        inputs = self._processor(images=image, return_tensors=tensor_format)
        logits = self._model(**inputs).logits
        # shape (height, width) with class indices
        mask = logits.argmax(dim=1)[0]
        labels_tensor = unique(mask)
        labels = [
            self._model.config.id2label[idx.item()] for idx in labels_tensor
        ]
        return labels
