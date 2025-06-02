from abc import ABC, abstractmethod
from ...entities import ImageEntity
from ...model.engine import Engine
from PIL import Image
from typing import Literal


class BaseVisionModel(Engine, ABC):
    @abstractmethod
    async def __call__(
        self,
        image_source: str | Image.Image,
        tensor_format: Literal["pt"] = "pt",
    ) -> ImageEntity | list[ImageEntity] | list[str]:
        raise NotImplementedError()

    @staticmethod
    def _get_image(image_source: str | Image.Image) -> Image.Image:
        return (
            image_source
            if isinstance(image_source, Image.Image)
            else Image.open(image_source)
        )
