from abc import ABC
from asyncio import to_thread as _to_thread
from inspect import iscoroutinefunction
from logging import Logger
from typing import Any, Callable


async def to_thread(func: Callable[..., Any], **kwargs: Any) -> Any:
    """Run *func* in a thread."""
    return await _to_thread(func, **kwargs)


class ElasticsearchMemory(ABC):
    """Common logic for Elasticsearch based memories."""

    _index: str
    _client: Any
    _logger: Logger

    def __init__(
        self,
        *,
        index: str,
        client: Any,
        logger: Logger,
        **kwargs: Any,
    ) -> None:
        self._index = index
        self._client = client
        self._logger = logger

    async def _call_client(
        self, func: Callable[..., Any], **kwargs: Any
    ) -> Any:
        if iscoroutinefunction(func):
            return await func(**kwargs)
        return await to_thread(func, **kwargs)

    async def _index_document(self, **kwargs: Any) -> Any:
        return await self._call_client(self._client.index, **kwargs)

    async def _get_document(self, **kwargs: Any) -> Any:
        return await self._call_client(self._client.get, **kwargs)

    async def _index_vector(self, **kwargs: Any) -> Any:
        return await self._call_client(self._client.index_vector, **kwargs)

    async def _query_vector(self, **kwargs: Any) -> Any:
        return await self._call_client(self._client.query_vector, **kwargs)
