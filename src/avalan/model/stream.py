from ..entities import (
    Token,
    TokenDetail,
)
from .provider import ProviderFamily, provider_family_value

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator


class TextGenerationStream(AsyncIterator[Token | TokenDetail | str], ABC):
    _generator: AsyncIterator[Token | TokenDetail | str] | None = None

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
    _provider_family: str | None = None
    _usage: object | None = None

    def __init__(
        self,
        content: str | Token | TokenDetail,
        *,
        provider_family: ProviderFamily | str | None = None,
        usage: object | None = None,
    ) -> None:
        self._content = content
        self._provider_family = provider_family_value(provider_family)
        self._usage = usage

    @property
    def content(self) -> str | Token | TokenDetail:
        return self._content

    @property
    def provider_family(self) -> str | None:
        return self._provider_family

    @property
    def usage(self) -> object | None:
        return self._usage

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
