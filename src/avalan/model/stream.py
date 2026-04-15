from ..entities import (
    Token,
    TokenDetail,
)

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, AsyncIterator


class TextGenerationStream(AsyncIterator[Token | TokenDetail | str], ABC):
    _generator: AsyncGenerator[Token | TokenDetail | str, None] | None = None

    @abstractmethod
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[Token | TokenDetail | str]:
        raise NotImplementedError()

    @abstractmethod
    async def __anext__(self) -> Token | TokenDetail | str:
        raise NotImplementedError()

    def __aiter__(self) -> AsyncIterator[Token | TokenDetail | str]:
        assert self._generator
        return self


class TextGenerationSingleStream(TextGenerationStream):
    _content: str | Token | TokenDetail
    _consumed: bool = False

    def __init__(self, content: str | Token | TokenDetail) -> None:
        self._content = content

    @property
    def content(self) -> str | Token | TokenDetail:
        return self._content

    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[str | Token | TokenDetail]:
        self._consumed = False
        return self

    def __aiter__(self) -> AsyncIterator[str | Token | TokenDetail]:
        self._consumed = False
        return self

    async def __anext__(self) -> str | Token | TokenDetail:
        if self._consumed:
            raise StopAsyncIteration
        self._consumed = True
        return self._content
