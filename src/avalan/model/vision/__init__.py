from abc import ABC, abstractmethod
from avalan.model.engine import Engine
from avalan.model.entities import ImageEntity
from PIL import Image
from typing import Literal, Union

class BaseVisionModel(Engine,ABC):
    @abstractmethod
    async def __call__(
        self,
        image_source: Union[str,Image.Image],
        tensor_format: Literal["pt"]="pt"
    ) -> Union[
        ImageEntity,
        list[ImageEntity],
        list[str]
    ]:
        raise NotImplementedError()

    @staticmethod
    def _get_image(image_source: Union[str,Image.Image]) -> Image:
        return image_source if isinstance(image_source, Image.Image) else \
               Image.open(image_source)

