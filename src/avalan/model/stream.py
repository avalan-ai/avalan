from abc import ABC, abstractmethod
from ..entities import (
    Token,
    TokenDetail,
)
from typing import (
    AsyncGenerator,
    AsyncIterator,
)


class TextGenerationStream(AsyncIterator[Token | TokenDetail | str], ABC):
    _generator: AsyncGenerator | None = None

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    async def __anext__(self) -> Token | TokenDetail | str:
        raise NotImplementedError()

    def __aiter__(self):
        assert self._generator
        return self
