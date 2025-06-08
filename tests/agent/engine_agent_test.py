from avalan.agent.engine import EngineAgent
from avalan.entities import (
    Message,
    MessageRole,
    GenerationSettings,
    EngineMessage,
)
from avalan.event import EventType
from avalan.event.manager import EventManager
from avalan.tool.manager import ToolManager
from avalan.model import TextGenerationResponse
from dataclasses import replace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock


class DummyEngine:
    def __init__(self) -> None:
        self.model_id = "m"
        self.called_with = None
        self.input_token_count = MagicMock(return_value=3)

    async def __call__(self, input, **kwargs):
        self.called_with = (input, kwargs)
        return "out"


class DummyAgent(EngineAgent):
    def _prepare_call(self, specification, input, **kwargs):
        return {}


class FakeMemory:
    def __init__(self) -> None:
        self.has_permanent_message = False
        self.has_recent_message = True
        self.recent_message = object()
        self.recent_messages: list[EngineMessage] = []

    async def append_message(self, message: EngineMessage) -> None:
        self.recent_messages.append(message)


class EngineAgentPropertyTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        self.memory = MagicMock()
        self.engine = DummyEngine()
        self.tool = MagicMock(spec=ToolManager)
        self.event_manager = MagicMock(spec=EventManager)
        self.event_manager.trigger = AsyncMock()
        self.agent = DummyAgent(
            self.engine, self.memory, self.tool, self.event_manager
        )

    async def test_memory_and_engine_property(self):
        self.assertIs(self.agent.memory, self.memory)
        self.assertIs(self.agent.engine, self.engine)

    async def test_input_token_count_no_prompt(self):
        result = await self.agent.input_token_count()
        self.assertIsNone(result)
        self.engine.input_token_count.assert_not_called()
        self.event_manager.trigger.assert_not_called()

    async def test_input_token_count_with_prompt(self):
        self.agent._last_prompt = ("hi", "sys")
        result = await self.agent.input_token_count()
        self.assertEqual(result, 3)
        self.engine.input_token_count.assert_called_once_with(
            "hi", system_prompt="sys"
        )
        called_types = [
            c.args[0].type for c in self.event_manager.trigger.await_args_list
        ]
        self.assertIn(EventType.INPUT_TOKEN_COUNT_BEFORE, called_types)
        self.assertIn(EventType.INPUT_TOKEN_COUNT_AFTER, called_types)


class EngineAgentRunTestCase(IsolatedAsyncioTestCase):
    def _make_agent(self, last_output=None):
        memory = FakeMemory()
        engine = DummyEngine()
        tool = MagicMock(spec=ToolManager)
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        agent = DummyAgent(engine, memory, tool, event_manager)
        agent._last_output = last_output
        return agent, engine, memory

    async def test_run_with_settings_and_previous_response(self):
        last_response = TextGenerationResponse(
            lambda: "prev", use_async_generator=False
        )
        agent, engine, memory = self._make_agent(last_response)

        settings = GenerationSettings(max_new_tokens=1)
        await agent._run(
            Message(role=MessageRole.USER, content="hi"),
            settings=settings,
            top_p=0.7,
        )

        self.assertEqual(len(memory.recent_messages), 2)
        self.assertEqual(
            memory.recent_messages[0].message.role, MessageRole.ASSISTANT
        )
        self.assertEqual(
            memory.recent_messages[1].message.role, MessageRole.USER
        )

        args, kwargs = engine.called_with
        self.assertEqual(kwargs["settings"], replace(settings, top_p=0.7))
        self.assertEqual(agent._last_output, "out")

    async def test_run_with_settings_no_previous_response(self):
        agent, engine, memory = self._make_agent()
        settings = GenerationSettings(max_new_tokens=1)
        await agent._run(
            Message(role=MessageRole.USER, content="hi"),
            settings=settings,
            top_p=0.7,
        )

        self.assertEqual(len(memory.recent_messages), 1)
        self.assertEqual(
            memory.recent_messages[0].message.role, MessageRole.USER
        )
        args, kwargs = engine.called_with
        self.assertEqual(kwargs["settings"], replace(settings, top_p=0.7))

    async def test_run_kwargs_only_with_previous_response(self):
        last_response = TextGenerationResponse(
            lambda: "prev", use_async_generator=False
        )
        agent, engine, memory = self._make_agent(last_response)

        await agent._run(
            Message(role=MessageRole.USER, content="hi"), temperature=0.4
        )

        self.assertEqual(len(memory.recent_messages), 2)
        self.assertEqual(
            memory.recent_messages[0].message.role, MessageRole.ASSISTANT
        )
        self.assertEqual(
            memory.recent_messages[1].message.role, MessageRole.USER
        )
        args, kwargs = engine.called_with
        self.assertEqual(kwargs["settings"].temperature, 0.4)
        self.assertFalse(kwargs["settings"].do_sample)

    async def test_run_kwargs_only_no_previous_response(self):
        agent, engine, memory = self._make_agent()
        await agent._run(
            Message(role=MessageRole.USER, content="hi"), temperature=0.4
        )

        self.assertEqual(len(memory.recent_messages), 1)
        self.assertEqual(
            memory.recent_messages[0].message.role, MessageRole.USER
        )
        args, kwargs = engine.called_with
        self.assertEqual(kwargs["settings"].temperature, 0.4)
        self.assertFalse(kwargs["settings"].do_sample)
