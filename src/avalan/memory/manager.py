from ..entities import EngineMessage, EngineMessageIdempotencyKey
from ..event import Event, EventType
from ..event.manager import EventManager
from ..filters import Partitioner
from ..memory import RecentMessageMemory
from ..memory.permanent import (
    Memory,
    PermanentMemory,
    PermanentMemoryPartition,
    PermanentMemoryStore,
    PermanentMessageMemory,
    VectorFunction,
)
from ..memory.permanent.codec import message_partition_text

from asyncio import Task, create_task, shield
from collections.abc import Awaitable, Callable
from enum import StrEnum
from logging import Logger
from time import perf_counter
from types import TracebackType
from uuid import UUID


class _MessageComponent(StrEnum):
    PERMANENT = "permanent"
    RECENT = "recent"


class MemoryManager:
    _agent_id: UUID
    _participant_id: UUID
    _permanent_message_memory: PermanentMessageMemory | None = None
    _permanent_memory_stores: dict[str, tuple[PermanentMemory, str | None]]
    _recent_message_memory: RecentMessageMemory | None = None
    _text_partitioner: Partitioner | None
    _text_partitioner_factory: Callable[[], Partitioner] | None
    _logger: Logger
    _event_manager: EventManager | None = None
    _message_identities: dict[EngineMessageIdempotencyKey, EngineMessage]
    _message_component_acks: dict[
        EngineMessageIdempotencyKey,
        set[_MessageComponent],
    ]
    _message_component_tasks: dict[
        tuple[EngineMessageIdempotencyKey, _MessageComponent],
        Task[None],
    ]

    @classmethod
    async def create_instance(
        cls,
        *args: object,
        agent_id: UUID,
        participant_id: UUID,
        text_partitioner: Partitioner | None,
        logger: Logger,
        text_partitioner_factory: Callable[[], Partitioner] | None = None,
        with_permanent_message_memory: str | None = None,
        with_recent_message_memory: bool = True,
        event_manager: EventManager | None = None,
    ) -> "MemoryManager":
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
            text_partitioner_factory=text_partitioner_factory,
            logger=logger,
            event_manager=event_manager,
        )
        return manager

    def __init__(
        self,
        *args: object,
        agent_id: UUID,
        participant_id: UUID,
        permanent_message_memory: PermanentMessageMemory | None,
        recent_message_memory: RecentMessageMemory | None,
        text_partitioner: Partitioner | None,
        logger: Logger,
        text_partitioner_factory: Callable[[], Partitioner] | None = None,
        event_manager: EventManager | None = None,
        permanent_memory_stores: (
            dict[str, tuple[PermanentMemory, str | None]] | None
        ) = None,
    ):
        assert agent_id and participant_id
        self._logger = logger
        self._agent_id = agent_id
        self._participant_id = participant_id
        self._text_partitioner = text_partitioner
        self._text_partitioner_factory = text_partitioner_factory
        self._permanent_memory_stores = {}
        self._event_manager = event_manager
        self._message_identities = {}
        self._message_component_acks = {}
        self._message_component_tasks = {}
        if permanent_message_memory:
            self.add_permanent_message_memory(permanent_message_memory)
        if recent_message_memory:
            self.add_recent_message_memory(recent_message_memory)
        if permanent_memory_stores:
            for namespace, store in permanent_memory_stores.items():
                memory, description = store
                self.add_permanent_memory(
                    namespace,
                    memory,
                    description=description,
                )

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

    def add_recent_message_memory(self, memory: RecentMessageMemory) -> None:
        self._recent_message_memory = memory

    def add_permanent_message_memory(
        self, memory: PermanentMessageMemory
    ) -> None:
        self._permanent_message_memory = memory

    def add_permanent_memory(
        self,
        namespace: str,
        memory: PermanentMemory,
        *,
        description: str | None = None,
    ) -> None:
        assert namespace and memory
        self._permanent_memory_stores[namespace] = (memory, description)

    def delete_permanent_memory(self, namespace: str) -> None:
        self._permanent_memory_stores.pop(namespace, None)

    def list_permanent_memory_stores(self) -> list[PermanentMemoryStore]:
        return [
            PermanentMemoryStore(namespace=namespace, description=description)
            for namespace, (
                _,
                description,
            ) in self._permanent_memory_stores.items()
        ]

    async def append_message(self, engine_message: EngineMessage) -> None:
        if not isinstance(engine_message, EngineMessage):
            self._logger.info("Skipping non engine message %s", engine_message)
            return
        content_text = message_partition_text(engine_message.message)
        if content_text is None:
            self._logger.info("Skipping non engine message %s", engine_message)
            return
        self._register_message_identity(engine_message)

        self._logger.debug("<Memory> Appending message")

        permanent = self._permanent_message_memory
        if permanent and not self._component_acknowledged(
            engine_message,
            _MessageComponent.PERMANENT,
        ):
            start = perf_counter()
            if self._event_manager:
                await self._event_manager.trigger(
                    Event(
                        type=EventType.MEMORY_PERMANENT_MESSAGE_ADD,
                        payload={
                            "message": engine_message,
                            "participant_id": self._participant_id,
                            "session_id": (
                                permanent.session_id if permanent else None
                            ),
                        },
                        started=start,
                    )
                )
            text_partitioner = self._ensure_text_partitioner(
                "Text partitioner is required for permanent message memory"
            )
            partitions = await text_partitioner(content_text)

            async def append_permanent() -> None:
                await permanent.append_with_partitions(
                    engine_message,
                    partitions=partitions,
                )

            await self._append_message_component(
                engine_message,
                _MessageComponent.PERMANENT,
                append_permanent,
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
                                permanent.session_id if permanent else None
                            ),
                        },
                        started=start,
                        finished=end,
                        elapsed=end - start,
                    )
                )

        recent = self._recent_message_memory
        if recent and not self._component_acknowledged(
            engine_message,
            _MessageComponent.RECENT,
        ):

            async def append_recent() -> None:
                await recent.append(self._agent_id, engine_message)

            await self._append_message_component(
                engine_message,
                _MessageComponent.RECENT,
                append_recent,
            )

        self._logger.debug("<Memory> Message appended")

    def _register_message_identity(self, message: EngineMessage) -> None:
        key = message.idempotency_key
        if key is None:
            return
        existing = self._message_identities.setdefault(key, message)
        if existing != message:
            raise ValueError("idempotency key identifies a different message")

    def _component_acknowledged(
        self,
        message: EngineMessage,
        component: _MessageComponent,
    ) -> bool:
        key = message.idempotency_key
        return (
            key is not None
            and component
            in self._message_component_acks.get(
                key,
                set(),
            )
        )

    async def _append_message_component(
        self,
        message: EngineMessage,
        component: _MessageComponent,
        operation: Callable[[], Awaitable[None]],
    ) -> None:
        key = message.idempotency_key
        if key is None:
            await operation()
            return
        if component in self._message_component_acks.get(key, set()):
            return
        task_key = (key, component)
        task = self._message_component_tasks.get(task_key)
        if task is None:

            async def execute() -> None:
                await operation()
                self._message_component_acks.setdefault(key, set()).add(
                    component
                )

            task = create_task(execute())
            self._message_component_tasks[task_key] = task
            task.add_done_callback(
                lambda completed: self._message_component_done(
                    task_key,
                    completed,
                )
            )
        await shield(task)

    def _message_component_done(
        self,
        task_key: tuple[EngineMessageIdempotencyKey, _MessageComponent],
        task: Task[None],
    ) -> None:
        if self._message_component_tasks.get(task_key) is task:
            self._message_component_tasks.pop(task_key, None)
        if not task.cancelled():
            task.exception()

    async def continue_session(
        self,
        session_id: UUID,
        *args: object,
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
            await self._recent_message_memory.reset()
            for message in messages:
                await self._recent_message_memory.append(
                    self._agent_id, message
                )

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
                    elapsed=end - start,
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
            await self._recent_message_memory.reset()

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
                    elapsed=end - start,
                )
            )

    async def search_messages(
        self,
        search: str,
        agent_id: UUID,
        participant_id: UUID,
        *args: object,
        function: VectorFunction,
        limit: int | None = None,
        search_user_messages: bool = False,
        session_id: UUID | None = None,
        exclude_session_id: UUID | None = None,
    ) -> list[EngineMessage]:
        assert self._permanent_message_memory
        text_partitioner = self._ensure_text_partitioner(
            "Text partitioner is required to search message memory"
        )
        search_partitions = await text_partitioner(search)
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

    async def search_partitions(
        self,
        search: str,
        *,
        participant_id: UUID,
        namespace: str,
        function: VectorFunction,
        limit: int | None = None,
    ) -> list[PermanentMemoryPartition]:
        if namespace not in self._permanent_memory_stores:
            raise KeyError(f"Memory namespace {namespace} not defined")

        text_partitioner = self._ensure_text_partitioner(
            "Text partitioner is required to search permanent memory"
        )
        search_partitions = await text_partitioner(search)
        store, _ = self._permanent_memory_stores[namespace]
        memories = await store.search_memories(
            search_partitions=search_partitions,
            participant_id=participant_id,
            namespace=namespace,
            function=function,
            limit=limit,
        )
        return memories

    def _ensure_text_partitioner(self, message: str) -> Partitioner:
        if self._text_partitioner is None and self._text_partitioner_factory:
            self._text_partitioner = self._text_partitioner_factory()
        assert self._text_partitioner, message
        return self._text_partitioner

    async def list_memories(
        self,
        *,
        participant_id: UUID,
        namespace: str,
    ) -> list[Memory]:
        if namespace not in self._permanent_memory_stores:
            raise KeyError(f"Memory namespace {namespace} not defined")

        store, _ = self._permanent_memory_stores[namespace]
        memories = await store.list_memories(
            participant_id=participant_id,
            namespace=namespace,
        )
        return memories

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        pass
