from .response.text import TextGenerationResponse
from .vendor import TextGenerationVendorStream
from ..entities import (
    ImageEntity,
    Token,
    TokenDetail,
)
from numpy import ndarray
from typing import AsyncGenerator, Callable, Generator

OutputGenerator = AsyncGenerator[Token | TokenDetail | str, None]
OutputFunction = Callable[..., OutputGenerator | str]

EngineResponse = (
    TextGenerationResponse
    | TextGenerationVendorStream
    | Generator[str, None, None]
    | Generator[Token | TokenDetail, None, None]
    | ImageEntity
    | list[ImageEntity]
    | list[str]
    | dict[str, str]
    | ndarray
    | str
)


class ModelAlreadyLoadedException(Exception):
    pass


class TokenizerAlreadyLoadedException(Exception):
    pass


class TokenizerNotSupportedException(Exception):
    pass
