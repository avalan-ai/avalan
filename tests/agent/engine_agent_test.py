from avalan.agent import Specification
from avalan.agent.engine import EngineAgent
from avalan.entities import (
    EngineMessage,
    EngineUri,
    GenerationSettings,
    Message,
    MessageRole,
)
from avalan.event import EventType
from avalan.event.manager import EventManager
from avalan.tool.manager import ToolManager
from avalan.model import TextGenerationResponse
from logging import getLogger
from avalan.model.manager import ModelManager
from avalan.model.call import ModelCallContext
from dataclasses import replace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock


class DummyEngine:
    def __init__(self) -> None:
        self.model_id = "m"
        self.model_type = "t"
        self.called_with = None
        self.input_token_count = MagicMock(return_value=3)

    async def __call__(self, input, **kwargs):
        self.called_with = (input, kwargs)
        return "out"


class DummyAgent(EngineAgent):
    def _prepare_call(self, context: ModelCallContext):
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
        self.model_manager = AsyncMock(spec=ModelManager)
        self.model_manager.return_value = "out"
        self.engine_uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="m",
            params={},
        )
        self.agent = DummyAgent(
            self.engine,
            self.memory,
            self.tool,
            self.event_manager,
            self.model_manager,
            self.engine_uri,
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
        self.agent._last_prompt = ("hi", "sys", None)
        result = await self.agent.input_token_count()
        self.assertEqual(result, 3)
        self.engine.input_token_count.assert_called_once_with(
            "hi", system_prompt="sys", developer_prompt=None
        )
        called_types = [
            c.args[0].type for c in self.event_manager.trigger.await_args_list
        ]
        self.assertIn(EventType.INPUT_TOKEN_COUNT_BEFORE, called_types)
        self.assertIn(EventType.INPUT_TOKEN_COUNT_AFTER, called_types)


class EngineAgentRunTestCase(IsolatedAsyncioTestCase):
    def _make_agent(self, last_output=None, params=None):
        memory = FakeMemory()
        engine = DummyEngine()
        tool = MagicMock(spec=ToolManager)
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
            params=params or {},
        )
        agent = DummyAgent(
            engine,
            memory,
            tool,
            event_manager,
            model_manager,
            engine_uri,
        )
        agent._last_output = last_output
        return agent, engine, memory, model_manager

    def _make_context(self, input_value: Message | str):
        return ModelCallContext(
            specification=Specification(role=None, goal=None),
            input=input_value,
        )

    async def test_run_with_settings_and_previous_response(self):
        last_response = TextGenerationResponse(
            lambda: "prev", logger=getLogger(), use_async_generator=False
        )
        agent, engine, memory, manager = self._make_agent(last_response)

        settings = GenerationSettings(max_new_tokens=1)
        message = Message(role=MessageRole.USER, content="hi")
        context = self._make_context(message)
        await agent._run(
            context,
            message,
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

        manager.assert_awaited_once()
        task = manager.await_args.args[0]
        self.assertEqual(task.engine_uri, agent.engine_uri)
        self.assertIs(task.model, engine)
        self.assertEqual(
            task.operation.generation_settings,
            replace(settings, top_p=0.7),
        )
        self.assertIs(task.context, context)
        self.assertEqual(agent._last_output, "out")

    async def test_run_with_settings_no_previous_response(self):
        agent, engine, memory, manager = self._make_agent()
        settings = GenerationSettings(max_new_tokens=1)
        message = Message(role=MessageRole.USER, content="hi")
        context = self._make_context(message)
        await agent._run(
            context,
            message,
            settings=settings,
            top_p=0.7,
        )

        self.assertEqual(len(memory.recent_messages), 1)
        self.assertEqual(
            memory.recent_messages[0].message.role, MessageRole.USER
        )
        manager.assert_awaited_once()
        task = manager.await_args.args[0]
        self.assertEqual(
            task.operation.generation_settings, replace(settings, top_p=0.7)
        )
        self.assertIs(task.context, context)

    async def test_run_kwargs_only_with_previous_response(self):
        last_response = TextGenerationResponse(
            lambda: "prev", logger=getLogger(), use_async_generator=False
        )
        agent, engine, memory, manager = self._make_agent(last_response)

        message = Message(role=MessageRole.USER, content="hi")
        context = self._make_context(message)
        await agent._run(
            context, message, temperature=0.4
        )

        self.assertEqual(len(memory.recent_messages), 2)
        self.assertEqual(
            memory.recent_messages[0].message.role, MessageRole.ASSISTANT
        )
        self.assertEqual(
            memory.recent_messages[1].message.role, MessageRole.USER
        )
        manager.assert_awaited_once()
        task = manager.await_args.args[0]
        self.assertEqual(task.operation.generation_settings.temperature, 0.4)
        self.assertFalse(task.operation.generation_settings.do_sample)
        self.assertIs(task.context, context)

    async def test_run_kwargs_only_no_previous_response(self):
        agent, engine, memory, manager = self._make_agent()
        message = Message(role=MessageRole.USER, content="hi")
        context = self._make_context(message)
        await agent._run(
            context, message, temperature=0.4
        )

        self.assertEqual(len(memory.recent_messages), 1)
        self.assertEqual(
            memory.recent_messages[0].message.role, MessageRole.USER
        )
        manager.assert_awaited_once()
        task = manager.await_args.args[0]
        self.assertEqual(task.operation.generation_settings.temperature, 0.4)
        self.assertFalse(task.operation.generation_settings.do_sample)
        self.assertIs(task.context, context)

    async def test_run_defaults_from_uri(self):
        agent, engine, _, manager = self._make_agent(
            params={"temperature": 0.6, "max_new_tokens": 5}
        )
        message = Message(role=MessageRole.USER, content="hi")
        context = self._make_context(message)
        await agent._run(context, message)
        manager.assert_awaited_once()
        settings = manager.await_args.args[0].operation.generation_settings
        self.assertEqual(settings.temperature, 0.6)
        self.assertEqual(settings.max_new_tokens, 5)

    async def test_run_settings_overridden_by_uri(self):
        agent, engine, _, manager = self._make_agent(params={"top_p": 0.5})
        message = Message(role=MessageRole.USER, content="hi")
        context = self._make_context(message)
        await agent._run(
            context,
            message,
            settings=GenerationSettings(),
        )
        manager.assert_awaited_once()
        used_settings = (
            manager.await_args.args[0].operation.generation_settings
        )
        self.assertEqual(used_settings.top_p, 0.5)


class EngineAgentCallTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        self.memory = MagicMock()
        self.engine = DummyEngine()
        self.tool = MagicMock(spec=ToolManager)
        self.event_manager = MagicMock(spec=EventManager)
        self.event_manager.trigger = AsyncMock()
        self.model_manager = AsyncMock(spec=ModelManager)
        self.engine_uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="m",
            params={},
        )
        self.agent = DummyAgent(
            self.engine,
            self.memory,
            self.tool,
            self.event_manager,
            self.model_manager,
            self.engine_uri,
        )

    async def test_call_sets_root_parent_when_missing(self):
        specification = Specification(role=None, goal=None)
        root_parent = ModelCallContext(
            specification=specification,
            input="root",
        )
        parent_context = ModelCallContext(
            specification=specification,
            input="parent",
            root_parent=root_parent,
        )
        child_context = ModelCallContext(
            specification=specification,
            input="child",
            parent=parent_context,
        )

        run_args = {"temperature": 0.25}
        self.agent._prepare_call = MagicMock(return_value=run_args)
        self.agent._run = AsyncMock(return_value="call-result")

        result = await self.agent(child_context)

        self.assertEqual(result, "call-result")
        self.agent._prepare_call.assert_called_once()
        prepared_context = self.agent._prepare_call.call_args.args[0]
        self.assertIs(prepared_context.root_parent, root_parent)

        self.agent._run.assert_awaited_once()
        run_context = self.agent._run.await_args.args[0]
        self.assertIs(run_context.root_parent, root_parent)
        self.assertIs(run_context.parent, parent_context)
        self.assertEqual(self.agent._run.await_args.kwargs, run_args)
        self.assertIsNone(child_context.root_parent)
        self.assertEqual(self.event_manager.trigger.await_count, 4)
