from .entities import TextPartition

from abc import ABC, abstractmethod
from typing import Any, Callable


class Partitioner(ABC):
    @property
    @abstractmethod
    def sentence_model(self) -> Callable[..., Any]:
        raise NotImplementedError()

    @abstractmethod
    async def __call__(
        self,
        text: str,
    ) -> list[TextPartition]:
        raise NotImplementedError()
