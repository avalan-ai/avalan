from ..entities import EngineMessage, EngineMessageIdempotencyKey

from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import Lock
from typing import Generic, TypeVar
from uuid import UUID

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class MemoryChunk(Generic[T]):
    repository_key: str
    key: str | None
    data: T


class MemoryStore(ABC, Generic[T]):
    @abstractmethod
    async def append(self, agent_id: UUID, data: T) -> None:
        raise NotImplementedError()

    @abstractmethod
    async def reset(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    async def search(self, query: str) -> list[T] | None:
        raise NotImplementedError()


class MessageMemory(MemoryStore[EngineMessage], ABC):
    async def search(self, query: str) -> list[EngineMessage] | None:
        raise NotImplementedError()


class RecentMessageMemory(MessageMemory):
    _lock: Lock
    _data: list[EngineMessage]
    _keyed_data: dict[EngineMessageIdempotencyKey, EngineMessage]

    def __init__(self) -> None:
        self._lock = Lock()
        self._data = []
        self._keyed_data = {}
        super().__init__()

    async def append(self, agent_id: UUID, data: EngineMessage) -> None:
        del agent_id
        with self._lock:
            key = data.idempotency_key
            if key is not None:
                existing = self._keyed_data.get(key)
                if existing is not None:
                    if existing != data:
                        raise ValueError(
                            "idempotency key identifies a different message"
                        )
                    return
                self._keyed_data[key] = data
            self._data.append(data)

    async def reset(self) -> None:
        with self._lock:
            self._data = []
            self._keyed_data = {}

    async def search(self, query: str) -> list[EngineMessage] | None:
        del query
        with self._lock:
            return list(self._data)

    @property
    def size(self) -> int:
        return len(self._data)

    @property
    def is_empty(self) -> bool:
        return not bool(self._data)

    @property
    def data(self) -> list[EngineMessage]:
        return self._data
