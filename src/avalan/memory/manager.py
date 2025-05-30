from ..entities import EngineMessage
from ..memory import RecentMessageMemory
from ..memory.partitioner.text import TextPartitioner
from ..memory.permanent import PermanentMessageMemory, VectorFunction
from typing import Any, Optional, Type
from uuid import UUID


class MemoryManager:
    _agent_id: UUID
    _participant_id: UUID
    _permanent_message_memory: Optional[PermanentMessageMemory] = None
    _recent_message_memory: Optional[RecentMessageMemory] = None
    _text_partitioner: TextPartitioner

    @classmethod
    async def create_instance(
        cls,
        *args,
        agent_id: UUID,
        participant_id: UUID,
        text_partitioner: TextPartitioner,
        with_permanent_message_memory: Optional[str] = None,
        with_recent_message_memory: bool = True,
    ):
        permanent_memory: Optional[PermanentMessageMemory] = None
        if with_permanent_message_memory:
            from .permanent.pgsql import PgsqlMessageMemory

            permanent_memory = await PgsqlMessageMemory.create_instance(
                dsn=with_permanent_message_memory
            )
        recent_memory = (
            RecentMessageMemory() if with_recent_message_memory else None
        )

        manager = cls(
            agent_id=agent_id,
            participant_id=participant_id,
            permanent_message_memory=permanent_memory,
            recent_message_memory=recent_memory,
            text_partitioner=text_partitioner,
        )
        return manager

    def __init__(
        self,
        *args,
        agent_id: UUID,
        participant_id: UUID,
        permanent_message_memory: Optional[PermanentMessageMemory],
        recent_message_memory: Optional[RecentMessageMemory],
        text_partitioner: TextPartitioner,
    ):
        assert agent_id and participant_id
        self._agent_id = agent_id
        self._participant_id = participant_id
        self._text_partitioner = text_partitioner
        if permanent_message_memory:
            self.add_permanent_message_memory(permanent_message_memory)
        if recent_message_memory:
            self.add_recent_message_memory(recent_message_memory)

    @property
    def has_permanent_message(self) -> bool:
        return bool(self._permanent_message_memory)

    @property
    def has_recent_message(self) -> bool:
        return bool(self._recent_message_memory)

    @property
    def permanent_message(self) -> Optional[PermanentMessageMemory]:
        return self._permanent_message_memory

    @property
    def recent_message(self) -> Optional[RecentMessageMemory]:
        return self._recent_message_memory

    @property
    def recent_messages(self) -> Optional[list[EngineMessage]]:
        return (
            self._recent_message_memory.data
            if self._recent_message_memory
            else None
        )

    def add_recent_message_memory(self, memory: RecentMessageMemory):
        self._recent_message_memory = memory

    def add_permanent_message_memory(self, memory: PermanentMessageMemory):
        self._permanent_message_memory = memory

    async def append_message(self, engine_message: EngineMessage) -> None:
        assert (
            isinstance(engine_message, EngineMessage)
            and engine_message.agent_id
            and engine_message.message
            and engine_message.message.content
        )

        if self._permanent_message_memory:
            partitions = await self._text_partitioner(
                engine_message.message.content
            )
            await self._permanent_message_memory.append_with_partitions(
                engine_message, partitions=partitions
            )

        if self._recent_message_memory:
            self._recent_message_memory.append(engine_message)

    async def continue_session(
        self,
        session_id: UUID,
        *args,
        load_recent_messages: bool = True,
        load_recent_messages_limit: Optional[int] = None,
    ) -> None:
        if self._permanent_message_memory:
            await self._permanent_message_memory.continue_session(
                agent_id=self._agent_id,
                participant_id=self._participant_id,
                session_id=session_id,
            )

        if (
            load_recent_messages
            and self._permanent_message_memory
            and self._recent_message_memory
        ):
            messages = await self._permanent_message_memory.get_recent_messages(
                participant_id=self._participant_id,
                session_id=session_id,
                limit=load_recent_messages_limit,
            )
            self._recent_message_memory.reset()
            for message in messages:
                self._recent_message_memory.append(message)

    async def start_session(self) -> None:
        if self._permanent_message_memory:
            await self._permanent_message_memory.reset_session(
                agent_id=self._agent_id, participant_id=self._participant_id
            )

        if self._recent_message_memory:
            self._recent_message_memory.reset()

    async def search_messages(
        self,
        search: str,
        agent_id: UUID,
        session_id: UUID,
        participant_id: UUID,
        *args,
        function: VectorFunction,
        limit: Optional[int] = None,
    ) -> list[EngineMessage]:
        assert self._permanent_message_memory
        search_partitions = await self._text_partitioner(search)
        messages = await self._permanent_message_memory.search_messages(
            search_partitions=search_partitions,
            agent_id=agent_id,
            session_id=session_id,
            participant_id=participant_id,
            function=function,
            limit=limit,
        )
        return messages

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[Any],
    ):
        pass
