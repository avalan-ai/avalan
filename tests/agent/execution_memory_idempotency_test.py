"""Pin execution-ledger memory identity and retry-safe component storage."""

from asyncio import Event, create_task
from datetime import UTC, datetime
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

from avalan.agent import execution as execution_module
from avalan.agent.engine import EngineAgent
from avalan.agent.execution import AgentExecution, UuidExecutionIdFactory
from avalan.entities import (
    ChatSettings,
    EngineMessage,
    EngineMessageIdempotencyKey,
    EngineUri,
    Message,
    MessageRole,
    MessageToolCall,
    ReasoningSettings,
    ReasoningSummaryMode,
    TextPartition,
    merge_generation_settings_options,
)
from avalan.event.manager import EventManager
from avalan.filters import Partitioner
from avalan.interaction.entities import (
    AgentId,
    BranchId,
    ExecutionDefinitionRef,
    ExecutionOrigin,
    ModelCallId,
    PrincipalScope,
    RunId,
    StreamSessionId,
    TaskId,
    TurnId,
    UserId,
)
from avalan.memory import RecentMessageMemory
from avalan.memory.manager import MemoryManager
from avalan.memory.permanent import PermanentMessageMemory
from avalan.model.call import ModelCallContext
from avalan.model.engine import Engine
from avalan.model.manager import ModelManager
from avalan.model.response.text import TextGenerationResponse
from avalan.tool.manager import ToolManager

_NOW = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)


def _origin() -> ExecutionOrigin:
    return ExecutionOrigin(
        run_id=RunId("memory-run"),
        turn_id=TurnId("memory-turn"),
        task_id=TaskId("memory-task"),
        agent_id=AgentId("memory-agent"),
        branch_id=BranchId("memory-branch"),
        model_call_id=ModelCallId("memory-model-call"),
        stream_session_id=StreamSessionId("memory-stream"),
        definition=ExecutionDefinitionRef(
            agent_definition_locator="agent://memory",
            agent_definition_revision="agent-r1",
            operation_id="operation-memory",
            operation_index=0,
            model_config_reference="model-r1",
            tool_revision="tools-r1",
            capability_revision="capabilities-r1",
        ),
        principal=PrincipalScope(user_id=UserId("memory-user")),
    )


async def _partition(_text: str) -> list[TextPartition]:
    return []


class _PermanentStore:
    """Model atomic keyed storage with selectable first-attempt failure."""

    def __init__(self, failure: str | None = None) -> None:
        self.failure = failure
        self.attempts = 0
        self.keys: list[EngineMessageIdempotencyKey] = []
        self.stored: dict[UUID, EngineMessage] = {}
        self.session_id: UUID | None = None

    async def append_with_partitions(
        self,
        engine_message: EngineMessage,
        partitions: list[TextPartition],
    ) -> None:
        del partitions
        self.attempts += 1
        key = engine_message.idempotency_key
        assert key is not None
        self.keys.append(key)
        if self.failure == "pre_commit" and self.attempts == 1:
            raise RuntimeError("pre-commit failure")
        self.stored.setdefault(key.value, engine_message)
        if self.failure == "commit_then_raise" and self.attempts == 1:
            raise RuntimeError("commit-then-raise failure")


class _BlockingPermanentStore(_PermanentStore):
    """Block the first storage attempt at a deterministic boundary."""

    def __init__(self) -> None:
        super().__init__()
        self.started = Event()
        self.release = Event()

    async def append_with_partitions(
        self,
        engine_message: EngineMessage,
        partitions: list[TextPartition],
    ) -> None:
        self.started.set()
        await self.release.wait()
        await super().append_with_partitions(engine_message, partitions)


class _StaggeredPartitioner:
    """Hold the second partition request until the first append commits."""

    def __init__(self) -> None:
        self.calls = 0
        self.second_started = Event()
        self.release_second = Event()

    async def __call__(self, _text: str) -> list[TextPartition]:
        self.calls += 1
        if self.calls == 2:
            self.second_started.set()
            await self.release_second.wait()
        return []


class _CommitThenRaiseRecent(RecentMessageMemory):
    """Fail once after the recent store has accepted the key."""

    def __init__(self) -> None:
        super().__init__()
        self.attempts = 0

    async def append(self, agent_id: UUID, data: EngineMessage) -> None:
        self.attempts += 1
        await super().append(agent_id, data)
        if self.attempts == 1:
            raise RuntimeError("recent commit-then-raise failure")


class _Agent(EngineAgent):
    """Expose the production execution-memory synchronization bridge."""

    def _prepare_call(self, context: ModelCallContext) -> dict[str, object]:
        del context
        return {}


def _agent(
    manager: MemoryManager,
    *,
    agent_id: UUID,
) -> _Agent:
    model = MagicMock(spec=Engine)
    model.model_id = "memory-model"
    model.model_type = "fake"
    event_manager = MagicMock(spec=EventManager)
    event_manager.trigger = AsyncMock()
    return _Agent(
        model=model,
        memory=manager,
        tool=MagicMock(spec=ToolManager),
        event_manager=event_manager,
        model_manager=MagicMock(spec=ModelManager),
        engine_uri=MagicMock(spec=EngineUri),
        id=agent_id,
    )


def _manager(
    permanent: _PermanentStore,
    recent: RecentMessageMemory,
    *,
    agent_id: UUID,
) -> MemoryManager:
    return MemoryManager(
        agent_id=agent_id,
        participant_id=uuid4(),
        permanent_message_memory=cast(PermanentMessageMemory, permanent),
        recent_message_memory=recent,
        text_partitioner=cast(Partitioner, _partition),
        logger=MagicMock(),
    )


def _manager_with_partitioner(
    permanent: _PermanentStore,
    recent: RecentMessageMemory,
    partitioner: Partitioner,
    *,
    agent_id: UUID,
) -> MemoryManager:
    return MemoryManager(
        agent_id=agent_id,
        participant_id=uuid4(),
        permanent_message_memory=cast(PermanentMessageMemory, permanent),
        recent_message_memory=recent,
        text_partitioner=partitioner,
        logger=MagicMock(),
    )


def _execution(*messages: Message) -> AgentExecution:
    return AgentExecution(
        origin=_origin(),
        id_factory=UuidExecutionIdFactory(),
        initial_messages=messages,
    )


class ExecutionMemoryRetryTest(IsolatedAsyncioTestCase):
    """Acknowledge a ledger cursor only after all keyed components commit."""

    async def test_permanent_commit_then_raise_retries_same_key_once(
        self,
    ) -> None:
        agent_id = uuid4()
        permanent = _PermanentStore("commit_then_raise")
        recent = RecentMessageMemory()
        manager = _manager(permanent, recent, agent_id=agent_id)
        agent = _agent(manager, agent_id=agent_id)
        execution = _execution(
            Message(role=MessageRole.USER, content="retry exactly")
        )

        with self.assertRaises(RuntimeError):
            await agent.sync_messages(execution)
        self.assertEqual(execution.snapshot.memory_sync_cursor, 0)
        await agent.sync_messages(execution)

        self.assertEqual(permanent.attempts, 2)
        self.assertEqual(len(permanent.stored), 1)
        self.assertEqual(permanent.keys[0], permanent.keys[1])
        self.assertEqual(recent.size, 1)
        self.assertEqual(execution.snapshot.memory_sync_cursor, 1)

    async def test_pre_commit_failure_remains_retryable(self) -> None:
        agent_id = uuid4()
        permanent = _PermanentStore("pre_commit")
        recent = RecentMessageMemory()
        agent = _agent(
            _manager(permanent, recent, agent_id=agent_id),
            agent_id=agent_id,
        )
        execution = _execution(
            Message(role=MessageRole.USER, content="retry before commit")
        )

        with self.assertRaises(RuntimeError):
            await agent.sync_messages(execution)
        self.assertEqual(permanent.stored, {})
        await agent.sync_messages(execution)

        self.assertEqual(permanent.attempts, 2)
        self.assertEqual(len(permanent.stored), 1)
        self.assertEqual(recent.size, 1)

    async def test_later_recent_failure_does_not_repeat_permanent(
        self,
    ) -> None:
        agent_id = uuid4()
        permanent = _PermanentStore()
        recent = _CommitThenRaiseRecent()
        agent = _agent(
            _manager(permanent, recent, agent_id=agent_id),
            agent_id=agent_id,
        )
        execution = _execution(
            Message(role=MessageRole.USER, content="component retry")
        )

        with self.assertRaises(RuntimeError):
            await agent.sync_messages(execution)
        await agent.sync_messages(execution)

        self.assertEqual(permanent.attempts, 1)
        self.assertEqual(len(permanent.stored), 1)
        self.assertEqual(recent.attempts, 2)
        self.assertEqual(recent.size, 1)

    async def test_acknowledged_component_wins_staggered_retry(self) -> None:
        agent_id = uuid4()
        permanent = _BlockingPermanentStore()
        partitioner = _StaggeredPartitioner()
        manager = _manager_with_partitioner(
            permanent,
            RecentMessageMemory(),
            cast(Partitioner, partitioner),
            agent_id=agent_id,
        )
        message = EngineMessage(
            agent_id=agent_id,
            model_id="model",
            message=Message(role=MessageRole.USER, content="concurrent"),
            idempotency_key=EngineMessageIdempotencyKey(value=uuid4()),
        )

        first = create_task(manager.append_message(message))
        await permanent.started.wait()
        second = create_task(manager.append_message(message))
        await partitioner.second_started.wait()
        permanent.release.set()
        await first
        partitioner.release_second.set()
        await second

        self.assertEqual(permanent.attempts, 1)
        self.assertEqual(partitioner.calls, 2)

    async def test_payloadless_and_conflicting_keyed_messages_fail_closed(
        self,
    ) -> None:
        agent_id = uuid4()
        recent = RecentMessageMemory()
        manager = _manager(_PermanentStore(), recent, agent_id=agent_id)
        await manager.append_message(
            EngineMessage(
                agent_id=agent_id,
                model_id="model",
                message=Message(role=MessageRole.USER),
            )
        )
        self.assertTrue(recent.is_empty)

        key = EngineMessageIdempotencyKey(value=uuid4())
        first = EngineMessage(
            agent_id=agent_id,
            model_id="model",
            message=Message(role=MessageRole.USER, content="first"),
            idempotency_key=key,
        )
        conflicting = EngineMessage(
            agent_id=agent_id,
            model_id="model",
            message=Message(role=MessageRole.USER, content="conflict"),
            idempotency_key=key,
        )
        await manager.append_message(first)
        with self.assertRaisesRegex(ValueError, "different message"):
            await manager.append_message(conflicting)
        with self.assertRaisesRegex(ValueError, "different message"):
            await recent.append(agent_id, conflicting)

    async def test_engine_execution_context_properties_and_guards(
        self,
    ) -> None:
        agent_id = uuid4()
        manager = MemoryManager(
            agent_id=agent_id,
            participant_id=uuid4(),
            permanent_message_memory=None,
            recent_message_memory=None,
            text_partitioner=None,
            logger=MagicMock(),
        )
        agent = _agent(manager, agent_id=agent_id)
        execution = _execution()
        prompt = Message(role=MessageRole.USER, content="prompt")
        await execution.record_prompt(
            execution_module.ModelPromptRecord(
                input=[prompt],
                instructions="instructions",
                system_prompt=None,
                developer_prompt="developer",
            )
        )
        token = agent._execution_context.set(execution)
        try:
            self.assertEqual(
                agent.last_prompt,
                ([prompt], "instructions", None, "developer"),
            )
            await agent.sync_messages()
        finally:
            agent._execution_context.reset(token)

        forged_execution = MagicMock(spec=AgentExecution)
        output = MagicMock(spec=TextGenerationResponse)
        forged_execution.last_response = output
        token = agent._execution_context.set(forged_execution)
        try:
            self.assertIs(agent.output, output)
        finally:
            agent._execution_context.reset(token)

        with self.assertRaisesRegex(TypeError, "execution-memory entry"):
            await agent.append_execution_memory_entry(
                cast(execution_module.ExecutionMemoryEntry, object())
            )


class StructuredToolMemoryTest(IsolatedAsyncioTestCase):
    """Retain a contentless assistant call beside its tool result."""

    async def test_contentless_tool_call_round_trips_both_stores(self) -> None:
        agent_id = uuid4()
        permanent = _PermanentStore()
        recent = RecentMessageMemory()
        agent = _agent(
            _manager(permanent, recent, agent_id=agent_id),
            agent_id=agent_id,
        )
        assistant = Message(
            role=MessageRole.ASSISTANT,
            tool_calls=[
                MessageToolCall(
                    id="call-1",
                    name="lookup",
                    arguments={"query": "value"},
                )
            ],
        )
        tool = Message(
            role=MessageRole.TOOL,
            name="lookup",
            arguments={"query": "value"},
            content="result",
        )
        execution = _execution(assistant, tool)

        await agent.sync_messages(execution)

        self.assertEqual(
            [message.message for message in permanent.stored.values()],
            [assistant, tool],
        )
        self.assertEqual(
            [message.message for message in recent.data],
            [assistant, tool],
        )
        self.assertEqual(execution.snapshot.memory_sync_cursor, 2)


class EngineMessageContractTest(TestCase):
    """Reject invalid stable-key and message envelope values."""

    def test_generation_settings_dataclass_base_merges_nested_override(
        self,
    ) -> None:
        merged = merge_generation_settings_options(
            {"chat_settings": ChatSettings(enable_thinking=False)},
            {"chat_settings": {"enable_thinking": True}},
        )
        self.assertEqual(
            cast(dict[str, object], merged["chat_settings"])[
                "enable_thinking"
            ],
            True,
        )

    def test_engine_helpers_normalize_bootstrap_and_reasoning_summary(
        self,
    ) -> None:
        agent_id = uuid4()
        manager = MemoryManager(
            agent_id=agent_id,
            participant_id=uuid4(),
            permanent_message_memory=None,
            recent_message_memory=None,
            text_partitioner=None,
            logger=MagicMock(),
        )
        agent = _agent(manager, agent_id=agent_id)
        self.assertEqual(agent.id, agent_id)
        with self.assertRaises(NotImplementedError):
            EngineAgent._prepare_call(
                agent,
                cast(ModelCallContext, object()),
            )
        tool = MagicMock()
        tool.bootstrap_prompt = "not callable"
        agent._tool = cast(ToolManager, tool)
        self.assertIsNone(agent._developer_prompt_with_tool_bootstrap(None))

        tool.bootstrap_prompt = MagicMock(return_value="bootstrap")
        self.assertEqual(
            agent._developer_prompt_with_tool_bootstrap(None),
            "bootstrap",
        )
        self.assertEqual(
            agent._developer_prompt_with_tool_bootstrap("developer"),
            "developer\n\nbootstrap",
        )
        normalized = agent._normalize_generation_settings(
            {"reasoning": {"summary": "concise"}}
        )
        self.assertEqual(
            normalized["reasoning"],
            ReasoningSettings(summary=ReasoningSummaryMode.CONCISE),
        )

    def test_message_and_key_types_are_strict(self) -> None:
        with self.assertRaisesRegex(TypeError, "value must be a UUID"):
            EngineMessageIdempotencyKey(value=cast(UUID, "key"))
        for field, value in (
            ("agent_id", "agent"),
            ("model_id", object()),
            ("message", object()),
            ("idempotency_key", object()),
        ):
            values: dict[str, Any] = {
                "agent_id": uuid4(),
                "model_id": "model",
                "message": Message(role=MessageRole.USER, content="text"),
                "idempotency_key": None,
            }
            values[field] = value
            with self.subTest(field=field), self.assertRaises(TypeError):
                EngineMessage(**values)

        invalid_calls: tuple[dict[str, object], ...] = (
            {"id": 1, "name": "tool"},
            {"id": None, "name": object()},
            {"id": None, "name": "tool", "content_type": "text"},
        )
        for values in invalid_calls:
            with (
                self.subTest(values=values),
                self.assertRaises((TypeError, ValueError)),
            ):
                cast(Any, MessageToolCall)(arguments={}, **values)
