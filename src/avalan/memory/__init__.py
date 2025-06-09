from abc import ABC, abstractmethod
from ..entities import EngineMessage
from dataclasses import dataclass
from threading import Lock
from typing import Generic, TypeVar
from ..compat import override
from uuid import UUID

T = TypeVar("T")


@dataclass(frozen=True)
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
    def search(self, query: str) -> list[EngineMessage] | None:
        raise NotImplementedError()


class RecentMessageMemory(MessageMemory):
    _lock: Lock
    _data: list[EngineMessage]

    def __init__(self, **kwargs) -> None:
        self._lock = Lock()
        self.reset()
        super().__init__(**kwargs)

    @override
    def append(self, data: EngineMessage) -> None:
        with self._lock:
            self._data.append(data)

    def reset(self) -> None:
        with self._lock:
            self._data = []

    @property
    def size(self) -> int:
        return len(self._data)

    @property
    def is_empty(self) -> bool:
        return not bool(self._data)

    @property
    def data(self) -> list[EngineMessage]:
        return self._data
