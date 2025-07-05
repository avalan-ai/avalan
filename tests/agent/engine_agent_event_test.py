from avalan.agent.engine import EngineAgent
from avalan.entities import Message, MessageRole, GenerationSettings
from avalan.event import EventType
from avalan.event.manager import EventManager
from avalan.memory.manager import MemoryManager
from avalan.tool.manager import ToolManager
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4


class DummyEngine:
    model_id = "m"
    model_type = "t"

    async def __call__(self, input, **kwargs):
        return "out"

    def input_token_count(self, *args, **kwargs):
        return 5


class DummyAgent(EngineAgent):
    def _prepare_call(self, specification, input, **kwargs):
        return {"settings": GenerationSettings()}


class EngineAgentEventTestCase(IsolatedAsyncioTestCase):
    async def test_events_triggered(self):
        memory = MagicMock(spec=MemoryManager)
        memory.has_permanent_message = False
        memory.has_recent_message = True
        memory.recent_message = Message(role=MessageRole.USER, content="last")
        memory.recent_messages = []
        memory.append_message = AsyncMock()
        memory.participant_id = uuid4()
        memory.permanent_message = None

        tool = MagicMock(spec=ToolManager)
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()

        agent = DummyAgent(DummyEngine(), memory, tool, event_manager)
        await agent(MagicMock(), Message(role=MessageRole.USER, content="hi"))

        await agent.input_token_count()

        called_types = [
            c.args[0].type for c in event_manager.trigger.await_args_list
        ]
        for t in [
            EventType.CALL_PREPARE_BEFORE,
            EventType.CALL_PREPARE_AFTER,
            EventType.MEMORY_APPEND_BEFORE,
            EventType.MEMORY_APPEND_AFTER,
            EventType.MODEL_EXECUTE_BEFORE,
            EventType.MODEL_EXECUTE_AFTER,
            EventType.INPUT_TOKEN_COUNT_BEFORE,
            EventType.INPUT_TOKEN_COUNT_AFTER,
        ]:
            self.assertIn(t, called_types)
        self.assertTrue(
            any(
                c.args[0].type == EventType.INPUT_TOKEN_COUNT_AFTER
                and c.args[0].payload["count"] == 5
                for c in event_manager.trigger.await_args_list
            )
        )
        self.assertTrue(
            any(
                c.args[0].type == EventType.MEMORY_APPEND_BEFORE
                and "participant_id" in c.args[0].payload
                and "session_id" in c.args[0].payload
                for c in event_manager.trigger.await_args_list
            )
        )
