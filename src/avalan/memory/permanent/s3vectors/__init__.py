from abc import ABC
from asyncio import to_thread as _to_thread
from inspect import iscoroutinefunction
from logging import Logger
from typing import Any, Callable


async def to_thread(func: Callable[..., Any], **kwargs: Any) -> Any:
    """Run *func* in a thread."""
    return await _to_thread(func, **kwargs)


class S3VectorsMemory(ABC):
    """Common logic for S3Vectors based memories."""

    _bucket: str
    _collection: str
    _client: Any
    _logger: Logger

    def __init__(
        self,
        *,
        bucket: str,
        collection: str,
        client: Any,
        logger: Logger,
        **kwargs: Any,
    ) -> None:
        self._bucket = bucket
        self._collection = collection
        self._client = client
        self._logger = logger

    async def _call_client(
        self, func: Callable[..., Any], **kwargs: Any
    ) -> Any:
        if iscoroutinefunction(func):
            return await func(**kwargs)
        return await to_thread(func, **kwargs)

    async def _put_object(self, **kwargs: Any) -> Any:
        return await self._call_client(self._client.put_object, **kwargs)

    async def _get_object(self, **kwargs: Any) -> Any:
        return await self._call_client(self._client.get_object, **kwargs)

    async def _put_vector(self, **kwargs: Any) -> Any:
        return await self._call_client(self._client.put_vector, **kwargs)

    async def _query_vector(self, **kwargs: Any) -> Any:
        return await self._call_client(self._client.query_vector, **kwargs)
