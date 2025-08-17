from abc import abstractmethod
from ...entities import (
    EngineMessage,
    MessageContentImage,
    MessageContentText,
    MessageRole,
)
from ...memory import MessageMemory, MemoryStore
from ...memory.partitioner.text import TextPartition
from ...model.nlp.sentence import SentenceTransformerModel
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from numpy import ndarray
from typing import Literal
from ...compat import override
from uuid import UUID, uuid4

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


class MemoryType(StrEnum):
    CODE = "code"
    FILE = "file"
    URL = "url"
    RAW = "raw"


@dataclass(frozen=True, kw_only=True, slots=True)
class Session:
    id: UUID
    agent_id: UUID
    participant_id: UUID
    messages: int
    created_at: datetime


@dataclass(frozen=True, kw_only=True, slots=True)
class Memory:
    id: UUID
    model_id: str
    type: MemoryType
    participant_id: UUID
    namespace: str
    identifier: str
    data: str
    partitions: int
    symbols: dict
    created_at: datetime


@dataclass(frozen=True, kw_only=True, slots=True)
class PermanentMessage:
    id: UUID
    agent_id: UUID
    model_id: str
    session_id: UUID | None
    author: MessageRole
    data: str
    partitions: int
    created_at: datetime


@dataclass(frozen=True, kw_only=True, slots=True)
class PermanentMessageScored(PermanentMessage):
    score: float


@dataclass(frozen=True, kw_only=True, slots=True)
class PermanentMessagePartition:
    agent_id: UUID
    session_id: UUID | None
    message_id: UUID
    partition: int
    data: str
    embedding: ndarray
    created_at: datetime


@dataclass(frozen=True, kw_only=True, slots=True)
class PermanentMemoryPartition:
    participant_id: UUID
    memory_id: UUID
    partition: int
    data: str
    embedding: ndarray
    created_at: datetime


@dataclass(frozen=True, kw_only=True, slots=True)
class Hyperedge:
    id: UUID
    relation: str
    surface_text: str
    embedding: ndarray
    symbols: dict | None = None
    created_at: datetime


@dataclass(frozen=True, kw_only=True, slots=True)
class HyperedgeMemory:
    hyperedge_id: UUID
    memory_id: UUID
    char_start: int | None = None
    char_end: int | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class Entity:
    id: UUID
    name: str
    type: str | None = None
    embedding: ndarray
    symbols: dict | None = None
    participant_id: UUID | None = None
    namespace: str | None = None
    created_at: datetime


@dataclass(frozen=True, kw_only=True, slots=True)
class HyperedgeEntity:
    hyperedge_id: UUID
    entity_id: UUID
    role_idx: int
    role_label: str | None = None


class PermanentMessageMemory(MessageMemory):
    _session_id: UUID | None = None
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
    def session_id(self) -> UUID | None:
        return self._session_id

    def reset(self) -> None:
        raise NotImplementedError()

    async def reset_session(
        self, agent_id: UUID, participant_id: UUID
    ) -> None:
        self._session_id = await self.create_session(
            agent_id=agent_id, participant_id=participant_id
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
        self, agent_id: UUID, participant_id: UUID
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
        partitions: list[TextPartition],
    ) -> None:
        raise NotImplementedError()

    @abstractmethod
    async def get_recent_messages(
        self,
        session_id: UUID,
        participant_id: UUID,
        *args,
        limit: int | None = None,
    ) -> list[EngineMessage]:
        raise NotImplementedError()

    @abstractmethod
    async def search_messages(
        self,
        *args,
        agent_id: UUID,
        function: VectorFunction,
        limit: int | None = None,
        participant_id: UUID,
        search_partitions: list[TextPartition],
        search_user_messages: bool,
        session_id: UUID | None,
        exclude_session_id: UUID | None,
    ) -> list[EngineMessage]:
        raise NotImplementedError()

    @staticmethod
    def _build_message_with_partitions(
        engine_message: EngineMessage,
        session_id: UUID | None,
        partitions: list[TextPartition],
        *,
        created_at: datetime,
        message_id: UUID | None = None,
    ) -> tuple[PermanentMessage, list[PermanentMessagePartition]]:
        if message_id is None:
            message_id = uuid4()
        content = engine_message.message.content
        data = (
            content.text
            if isinstance(content, MessageContentText)
            else (
                content.image_url
                if isinstance(content, MessageContentImage)
                else str(content)
            )
        )

        message = PermanentMessage(
            id=message_id,
            agent_id=engine_message.agent_id,
            model_id=engine_message.model_id,
            session_id=session_id,
            author=engine_message.message.role,
            data=data,
            partitions=len(partitions),
            created_at=created_at,
        )
        message_partitions = [
            PermanentMessagePartition(
                agent_id=message.agent_id,
                session_id=message.session_id,
                message_id=message.id,
                partition=i + 1,
                data=p.data,
                embedding=p.embeddings,
                created_at=created_at,
            )
            for i, p in enumerate(partitions)
        ]
        return message, message_partitions

    @staticmethod
    def _build_session(
        agent_id: UUID,
        participant_id: UUID,
        *,
        created_at: datetime,
        session_id: UUID | None = None,
    ) -> Session:
        if session_id is None:
            session_id = uuid4()
        return Session(
            id=session_id,
            agent_id=agent_id,
            participant_id=participant_id,
            messages=0,
            created_at=created_at,
        )


class PermanentMemory(MemoryStore[Memory]):
    _sentence_model: SentenceTransformerModel

    def __init__(
        self, sentence_model: SentenceTransformerModel, **kwargs
    ) -> None:
        self._sentence_model = sentence_model
        super().__init__(**kwargs)

    @override
    def append(self, data: EngineMessage) -> None:
        raise NotImplementedError()

    def reset(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    async def append_with_partitions(
        self,
        namespace: str,
        participant_id: UUID,
        *args,
        memory_type: MemoryType,
        data: str,
        identifier: str,
        partitions: list[TextPartition],
        symbols: dict | None = None,
        model_id: str | None = None,
    ) -> None:
        raise NotImplementedError()

    @abstractmethod
    async def search_memories(
        self,
        *args,
        search_partitions: list[TextPartition],
        participant_id: UUID,
        namespace: str,
        function: VectorFunction,
        limit: int | None = None,
    ) -> list[Memory]:
        raise NotImplementedError()

    @staticmethod
    def _build_memory_with_partitions(
        namespace: str,
        participant_id: UUID,
        memory_type: MemoryType,
        data: str,
        identifier: str,
        partitions: list[TextPartition],
        *,
        created_at: datetime,
        symbols: dict | None = None,
        model_id: str | None = None,
        memory_id: UUID | None = None,
    ) -> tuple[Memory, list[PermanentMemoryPartition]]:
        if memory_id is None:
            memory_id = uuid4()
        entry = Memory(
            id=memory_id,
            model_id=model_id,
            type=memory_type,
            participant_id=participant_id,
            namespace=namespace,
            identifier=identifier,
            data=data,
            partitions=len(partitions),
            symbols=symbols,
            created_at=created_at,
        )
        partition_rows = [
            PermanentMemoryPartition(
                participant_id=participant_id,
                memory_id=memory_id,
                partition=i + 1,
                data=p.data,
                embedding=p.embeddings,
                created_at=created_at,
            )
            for i, p in enumerate(partitions)
        ]
        return entry, partition_rows


class RecordNotFoundException(Exception):
    def __init__(self):
        super(RecordNotFoundException, self).__init__("record_not_found")


class RecordNotSavedException(Exception):
    def __init__(self):
        super(RecordNotSavedException, self).__init__("record_not_saved")
