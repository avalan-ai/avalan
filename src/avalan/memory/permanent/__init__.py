from abc import abstractmethod
from avalan.memory import MessageMemory
from avalan.memory.partitioner.text import TextPartition
from avalan.model.entities import EngineMessage, MessageRole
from avalan.model.nlp.sentence import SentenceTransformerModel
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from numpy import ndarray
from typing import Literal, Optional
from avalan.compat import override
from uuid import UUID

Order = Literal["asc", "desc"]

class VectorFunction(StrEnum):
    # cosine distance: 1 – cosine similarity (angle between vectors). smaller
    # means  more similar
    COSINE_DISTANCE = "cosine_distance"
    # negative inner product: returns –(x·y) so that ORDER BY finds top
    # dot-product hits
    INNER_PRODUCT = "inner_product"
    # L1 (Manhattan) distance: sum of absolute differences; often more robust
    # in high-D
    L1_DISTANCE = "l1_distance"
    # L2 (Euclidean) distance: straight-line distance; smaller means more
    # similar
    L2_DISTANCE = "l2_distance"
    # Hamming distance for binary vectors
    VECTOR_DIMS = "vector_dims"
    # Jaccard distance for binary vectors
    VECTOR_NORMS = "vector_norms"

@dataclass(frozen=True, kw_only=True)
class Session:
    id: UUID
    agent_id: UUID
    participant_id: UUID
    messages: int
    created_at: datetime

@dataclass(frozen=True, kw_only=True)
class PermanentMessage:
    id: UUID
    agent_id: UUID
    model_id: str
    session_id: Optional[UUID]
    author: MessageRole
    data: str
    partitions: int
    created_at: datetime

@dataclass(frozen=True, kw_only=True)
class PermanentMessageScored(PermanentMessage):
    score: float

@dataclass(frozen=True, kw_only=True)
class PermanentMessagePartition:
    agent_id: UUID
    session_id: Optional[UUID]
    message_id: UUID
    partition: int
    data: str
    embedding: ndarray
    created_at: datetime

class PermanentMessageMemory(MessageMemory):
    _session_id: Optional[UUID]=None
    _sentence_model: SentenceTransformerModel

    def __init__(
        self,
        sentence_model: SentenceTransformerModel,
        **kwargs,
    ):
        self._sentence_model = sentence_model
        super().__init__(**kwargs)

    @property
    def has_session(self) -> bool:
        return bool(self._session_id)

    @property
    def session_id(self) -> Optional[UUID]:
        return self._session_id

    def reset(self) -> None:
        raise NotImplementedError()

    async def reset_session(
        self,
        agent_id: UUID,
        participant_id: UUID
    ) -> None:
        self._session_id = await self.create_session(
            agent_id=agent_id,
            participant_id=participant_id
        )

    async def continue_session(
        self,
        agent_id: UUID,
        participant_id: UUID,
        session_id: UUID,
    ) -> None:
        self._session_id = await self.continue_session_and_get_id(
            agent_id=agent_id,
            participant_id=participant_id,
            session_id=session_id,
        )

    @override
    def append(self, data: EngineMessage) -> None:
        raise NotImplementedError()

    @abstractmethod
    async def create_session(
        self,
        agent_id: UUID,
        participant_id: UUID
    ) -> UUID:
        raise NotImplementedError()

    @abstractmethod
    async def continue_session_and_get_id(
        self,
        *args,
        agent_id: UUID,
        participant_id: UUID,
        session_id: UUID,
    ) -> UUID:
       raise NotImplementedError()

    @abstractmethod
    async def append_with_partitions(
        self,
        engine_message: EngineMessage,
        *args,
        partitions: list[TextPartition]
    ) -> None:
        raise NotImplementedError()

    @abstractmethod
    async def get_recent_messages(
        self,
        session_id: UUID,
        participant_id: UUID,
        *args,
        limit: Optional[int]=None
    ) -> list[EngineMessage]:
        raise NotImplementedError()

    @abstractmethod
    async def search_messages(
        self,
        *args,
        search_partitions: list[TextPartition],
        agent_id: UUID,
        session_id: UUID,
        participant_id: UUID,
        function: VectorFunction,
        limit: Optional[int]=None
    ) -> list[EngineMessage]:
        raise NotImplementedError()

class RecordNotFoundException(Exception):
    def __init__(self):
        super(RecordNotFoundException, self).__init__("record_not_found")

class RecordNotSavedException(Exception):
    def __init__(self):
        super(RecordNotSavedException, self).__init__("record_not_saved")

