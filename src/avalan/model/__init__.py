from ..entities import (
    ImageEntity,
    Token,
    TokenDetail,
)
from .call import ModelCall as ModelCall
from .call import ModelCallContext as ModelCallContext
from .input import input_files as input_files
from .response.text import TextGenerationResponse
from .vendor import TextGenerationVendorStream

from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Generator,
    TypeAlias,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray
else:

    class NDArray:  # noqa: D101
        def __class_getitem__(cls, _: Any) -> Any:
            return Any


OutputGenerator = AsyncGenerator[Token | TokenDetail | str, None]
OutputFunction = Callable[..., OutputGenerator | str]

EngineResponse: TypeAlias = (
    TextGenerationResponse
    | TextGenerationVendorStream
    | Generator[str, None, None]
    | Generator[Token | TokenDetail, None, None]
    | ImageEntity
    | list[ImageEntity]
    | list[str]
    | dict[str, str]
    | NDArray[Any]
    | str
)


class ModelAlreadyLoadedException(Exception):
    pass


class TokenizerAlreadyLoadedException(Exception):
    pass


class TokenizerNotSupportedException(Exception):
    pass
