from ..entities import EngineMessage
from ..event import Event, EventType
from ..event.manager import EventManager
from ..memory import RecentMessageMemory
from ..memory.partitioner.text import TextPartitioner
from ..memory.permanent import (
    PermanentMessageMemory,
    PermanentMemory,
    VectorFunction,
)
from time import perf_counter
from logging import Logger
from typing import Any
from uuid import UUID


class MemoryManager:
    _agent_id: UUID
    _participant_id: UUID
    _permanent_message_memory: PermanentMessageMemory | None = None
    _permanent_memories: dict[str, PermanentMemory]
    _recent_message_memory: RecentMessageMemory | None = None
    _text_partitioner: TextPartitioner
    _logger: Logger
    _event_manager: EventManager | None = None

    @classmethod
    async def create_instance(
        cls,
        *args,
        agent_id: UUID,
        participant_id: UUID,
        text_partitioner: TextPartitioner,
        logger: Logger,
        with_permanent_message_memory: str | None = None,
        with_recent_message_memory: bool = True,
        event_manager: EventManager | None = None,
    ):
        permanent_memory: PermanentMessageMemory | None = None
        if with_permanent_message_memory:
            from .permanent.pgsql.message import PgsqlMessageMemory

            permanent_memory = await PgsqlMessageMemory.create_instance(
                dsn=with_permanent_message_memory,
                logger=logger,
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
            logger=logger,
            event_manager=event_manager,
        )
        return manager

    def __init__(
        self,
        *args,
        agent_id: UUID,
        participant_id: UUID,
        permanent_message_memory: PermanentMessageMemory | None,
        recent_message_memory: RecentMessageMemory | None,
        text_partitioner: TextPartitioner,
        logger: Logger,
        event_manager: EventManager | None = None,
        permanent_memories: dict[str, PermanentMemory] | None = None,
    ):
        assert agent_id and participant_id
        self._logger = logger
        self._agent_id = agent_id
        self._participant_id = participant_id
        self._text_partitioner = text_partitioner
        self._permanent_memories = {}
        self._event_manager = event_manager
        if permanent_message_memory:
            self.add_permanent_message_memory(permanent_message_memory)
        if recent_message_memory:
            self.add_recent_message_memory(recent_message_memory)
        if permanent_memories:
            for namespace, memory in permanent_memories.items():
                self.add_permanent_memory(namespace, memory)

    @property
    def participant_id(self) -> UUID:
        """Return the participant identifier associated with this memory."""
        return self._participant_id

    @property
    def has_permanent_message(self) -> bool:
        return bool(self._permanent_message_memory)

    @property
    def has_recent_message(self) -> bool:
        return bool(self._recent_message_memory)

    @property
    def permanent_message(self) -> PermanentMessageMemory | None:
        return self._permanent_message_memory

    @property
    def recent_message(self) -> RecentMessageMemory | None:
        return self._recent_message_memory

    @property
    def recent_messages(self) -> list[EngineMessage] | None:
        return (
            self._recent_message_memory.data
            if self._recent_message_memory
            else None
        )

    def add_recent_message_memory(self, memory: RecentMessageMemory):
        self._recent_message_memory = memory

    def add_permanent_message_memory(self, memory: PermanentMessageMemory):
        self._permanent_message_memory = memory

    def add_permanent_memory(
        self, namespace: str, memory: PermanentMemory
    ) -> None:
        assert namespace and memory
        self._permanent_memories[namespace] = memory

    def delete_permanent_memory(self, namespace: str) -> None:
        self._permanent_memories.pop(namespace, None)

    async def append_message(self, engine_message: EngineMessage) -> None:
        assert (
            isinstance(engine_message, EngineMessage)
            and engine_message.agent_id
            and engine_message.message
            and engine_message.message.content
        )

        self._logger.debug("<Memory> Appending message")

        if self._permanent_message_memory:
            start = perf_counter()
            if self._event_manager:
                await self._event_manager.trigger(
                    Event(
                        type=EventType.MEMORY_PERMANENT_MESSAGE_ADD,
                        payload={
                            "message": engine_message,
                            "participant_id": self._participant_id,
                            "session_id": (
                                self._permanent_message_memory.session_id
                                if self._permanent_message_memory
                                else None
                            ),
                        },
                        started=start,
                    )
                )
            partitions = await self._text_partitioner(
                engine_message.message.content
            )
            await self._permanent_message_memory.append_with_partitions(
                engine_message, partitions=partitions
            )
            if self._event_manager:
                end = perf_counter()
                await self._event_manager.trigger(
                    Event(
                        type=EventType.MEMORY_PERMANENT_MESSAGE_ADDED,
                        payload={
                            "message": engine_message,
                            "participant_id": self._participant_id,
                            "session_id": (
                                self._permanent_message_memory.session_id
                                if self._permanent_message_memory
                                else None
                            ),
                        },
                        started=start,
                        finished=end,
                        ellapsed=end - start,
                    )
                )

        if self._recent_message_memory:
            self._recent_message_memory.append(engine_message)

        self._logger.debug("<Memory> Message appended")

    async def continue_session(
        self,
        session_id: UUID,
        *args,
        load_recent_messages: bool = True,
        load_recent_messages_limit: int | None = None,
    ) -> None:
        self._logger.debug("Continuing session %s", session_id)
        if self._permanent_message_memory:
            start = perf_counter()
            if self._event_manager:
                await self._event_manager.trigger(
                    Event(
                        type=EventType.MEMORY_PERMANENT_MESSAGE_SESSION_CONTINUE,
                        payload={
                            "session_id": session_id,
                            "participant_id": self._participant_id,
                        },
                        started=start,
                    )
                )
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
            messages = (
                await self._permanent_message_memory.get_recent_messages(
                    participant_id=self._participant_id,
                    session_id=session_id,
                    limit=load_recent_messages_limit,
                )
            )
            self._recent_message_memory.reset()
            for message in messages:
                self._recent_message_memory.append(message)

        self._logger.debug("Session %s continued", session_id)
        if self._permanent_message_memory and self._event_manager:
            end = perf_counter()
            await self._event_manager.trigger(
                Event(
                    type=EventType.MEMORY_PERMANENT_MESSAGE_SESSION_CONTINUED,
                    payload={
                        "session_id": session_id,
                        "participant_id": self._participant_id,
                    },
                    started=start,
                    finished=end,
                    ellapsed=end - start,
                )
            )

    async def start_session(self) -> None:
        self._logger.debug("Starting session")
        if self._permanent_message_memory:
            start = perf_counter()
            if self._event_manager:
                await self._event_manager.trigger(
                    Event(
                        type=EventType.MEMORY_PERMANENT_MESSAGE_SESSION_START,
                        payload={"participant_id": self._participant_id},
                        started=start,
                    )
                )
            await self._permanent_message_memory.reset_session(
                agent_id=self._agent_id, participant_id=self._participant_id
            )

        if self._recent_message_memory:
            self._recent_message_memory.reset()

        self._logger.debug("Session started")
        if self._permanent_message_memory and self._event_manager:
            end = perf_counter()
            await self._event_manager.trigger(
                Event(
                    type=EventType.MEMORY_PERMANENT_MESSAGE_SESSION_STARTED,
                    payload={
                        "session_id": (
                            self._permanent_message_memory.session_id
                        ),
                        "participant_id": self._participant_id,
                    },
                    started=start,
                    finished=end,
                    ellapsed=end - start,
                )
            )

    async def search_messages(
        self,
        search: str,
        agent_id: UUID,
        participant_id: UUID,
        *args,
        function: VectorFunction,
        limit: int | None = None,
        search_user_messages: bool = False,
        session_id: UUID | None = None,
        exclude_session_id: UUID | None = None,
    ) -> list[EngineMessage]:
        assert self._permanent_message_memory
        search_partitions = await self._text_partitioner(search)
        messages = await self._permanent_message_memory.search_messages(
            search_partitions=search_partitions,
            search_user_messages=search_user_messages,
            agent_id=agent_id,
            participant_id=participant_id,
            function=function,
            limit=limit,
            session_id=session_id,
            exclude_session_id=exclude_session_id,
        )
        return messages

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any | None,
    ):
        pass
