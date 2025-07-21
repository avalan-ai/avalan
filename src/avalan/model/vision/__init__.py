from abc import ABC
from ...model.engine import Engine
from PIL import Image


class BaseVisionModel(Engine, ABC):
    @staticmethod
    def _get_image(image_source: str | Image.Image) -> Image.Image:
        return (
            image_source
            if isinstance(image_source, Image.Image)
            else Image.open(image_source).convert("RGB")
        )
