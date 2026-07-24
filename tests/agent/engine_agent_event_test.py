from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from avalan.agent import Specification
from avalan.agent.engine import EngineAgent
from avalan.entities import (
    EngineUri,
    GenerationSettings,
    Message,
    MessageRole,
)
from avalan.event import EventType
from avalan.event.manager import EventManager
from avalan.memory.manager import MemoryManager
from avalan.model.call import ModelCallContext
from avalan.model.manager import ModelManager
from avalan.tool.manager import ToolManager


class DummyEngine:
    model_id = "m"
    model_type = "t"

    async def __call__(self, input, **kwargs):
        return "out"

    def input_token_count(self, *args, **kwargs):
        return 5


class DummyAgent(EngineAgent):
    def _prepare_call(self, context: ModelCallContext):
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
        tool.export_model_capability_seed.return_value = (
            ToolManager.create_instance().export_model_capability_seed()
        )
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        model_manager = AsyncMock(spec=ModelManager)
        model_manager.return_value = "out"
        engine_uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="m",
            params={},
        )
        agent = DummyAgent(
            DummyEngine(),
            memory,
            tool,
            event_manager,
            model_manager,
            engine_uri,
        )
        context = ModelCallContext(
            specification=Specification(role=None, goal=None),
            input=Message(role=MessageRole.USER, content="hi"),
        )
        await agent(context)

        await agent.input_token_count()

        task = model_manager.await_args.args[0]
        self.assertIsNotNone(task.capability)
        self.assertIs(task.context.capability, task.capability)
        context_events = [
            call.args[0]
            for call in event_manager.trigger.await_args_list
            if "context" in call.args[0].payload
        ]
        self.assertGreaterEqual(len(context_events), 2)
        self.assertTrue(
            all(
                event.payload["context"] is task.context
                for event in context_events
            )
        )

        called_types = [
            c.args[0].type for c in event_manager.trigger.await_args_list
        ]
        for t in [
            EventType.ENGINE_AGENT_CALL_BEFORE,
            EventType.CALL_PREPARE_BEFORE,
            EventType.CALL_PREPARE_AFTER,
            EventType.MEMORY_APPEND_BEFORE,
            EventType.MEMORY_APPEND_AFTER,
            EventType.MODEL_EXECUTE_BEFORE,
            EventType.MODEL_EXECUTE_AFTER,
            EventType.ENGINE_AGENT_CALL_AFTER,
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
