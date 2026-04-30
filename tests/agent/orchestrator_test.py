import unittest
from asyncio import CancelledError
from dataclasses import asdict
from json import dumps
from os.path import join
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from avalan.agent import (
    AgentOperation,
    EngineEnvironment,
    InputType,
    NoOperationAvailableException,
    Specification,
)
from avalan.agent.orchestrator import Orchestrator
from avalan.agent.renderer import Renderer
from avalan.entities import (
    EngineUri,
    GenerationSettings,
    Message,
    MessageContentText,
    MessageRole,
    TransformerEngineSettings,
)
from avalan.event import EventType
from avalan.event.manager import EventManager
from avalan.memory.manager import MemoryManager
from avalan.model.manager import ModelManager
from avalan.tool.manager import ToolManager


class OrchestratorCallTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        Orchestrator._engine_agents = {}
        engine_uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="m",
            params={},
        )
        self.environment = EngineEnvironment(
            engine_uri=engine_uri, settings=TransformerEngineSettings()
        )
        self.spec = Specification(
            role=None, goal=None, input_type=InputType.TEXT
        )
        self.operation = AgentOperation(
            specification=self.spec, environment=self.environment
        )
        self.logger = MagicMock()
        self.model_manager = MagicMock(spec=ModelManager)
        self.memory = MagicMock(spec=MemoryManager)
        self.memory.participant_id = uuid4()
        self.tool = MagicMock(spec=ToolManager)
        self.event_manager = MagicMock(spec=EventManager)
        self.orch = Orchestrator(
            self.logger,
            self.model_manager,
            self.memory,
            self.tool,
            self.event_manager,
            [self.operation],
        )
        self.engine_agent = AsyncMock()
        self.engine_agent.engine = MagicMock(
            model_id="m", tokenizer=MagicMock(eos_token="<eos>")
        )
        self.engine_agent.input_token_count = 5
        env_hash = dumps(asdict(self.environment))
        self.orch._engine_agents[env_hash] = self.engine_agent
        patcher = patch(
            "avalan.agent.orchestrator.OrchestratorResponse",
            lambda *a, **k: "resp",
        )
        self.addCleanup(patcher.stop)
        patcher.start()

        self.event_manager.trigger = AsyncMock()

    async def test_call_executes_operation(self):
        resp = await self.orch("hi")
        self.engine_agent.assert_awaited_once()
        self.tool.set_eos_token.assert_called_once_with("<eos>")
        self.assertEqual(resp, "resp")
        self.assertIs(self.orch.engine_agent, self.engine_agent)
        self.assertIs(self.orch.engine, self.engine_agent.engine)
        self.assertEqual(self.orch.input_token_count, 5)
        self.assertIs(self.orch.memory, self.memory)
        self.assertIs(self.orch.tool, self.tool)
        self.assertIs(self.orch.event_manager, self.event_manager)

    async def test_call_response_uses_effective_agent_prompt(self):
        captured = {}
        effective_messages = [
            Message(role=MessageRole.USER, content="previous"),
            Message(role=MessageRole.ASSISTANT, content="25"),
            Message(role=MessageRole.USER, content="and that times two?"),
        ]
        self.engine_agent.last_prompt = (effective_messages, None, None)

        def response_factory(input_value, *args, **kwargs):
            del args, kwargs
            captured["input"] = input_value
            return "resp"

        with patch(
            "avalan.agent.orchestrator.OrchestratorResponse",
            response_factory,
        ):
            resp = await self.orch("and that times two?")

        self.assertEqual(resp, "resp")
        self.assertIs(captured["input"], effective_messages)

    async def test_call_triggers_events(self):
        self.engine_agent.return_value = "ok"
        resp = await self.orch("hi")
        self.assertEqual(resp, "resp")

        called_types = [
            c.args[0].type for c in self.event_manager.trigger.await_args_list
        ]
        self.assertIn(EventType.ENGINE_RUN_BEFORE, called_types)
        self.assertIn(EventType.ENGINE_RUN_AFTER, called_types)

        before = next(
            c.args[0]
            for c in self.event_manager.trigger.await_args_list
            if c.args[0].type == EventType.ENGINE_RUN_BEFORE
        )
        after = next(
            c.args[0]
            for c in self.event_manager.trigger.await_args_list
            if c.args[0].type == EventType.ENGINE_RUN_AFTER
        )

        self.assertIsInstance(before.payload["input"], Message)
        self.assertEqual(before.payload["input"].content, "hi")
        self.assertIs(before.payload["specification"], self.spec)
        self.assertIsNone(before.finished)
        self.assertIsNone(before.elapsed)

        self.assertEqual(after.payload["result"], "ok")
        self.assertIs(after.payload["specification"], self.spec)
        self.assertEqual(after.started, before.started)
        self.assertIsNotNone(after.finished)
        self.assertIsNotNone(after.elapsed)

    async def test_call_no_operation_available(self):
        self.orch._operation_step = self.orch._total_operations
        with self.assertRaises(NoOperationAvailableException):
            await self.orch("hi")

    async def test_aexit_saves_message(self):
        msg = AsyncMock(to_str=AsyncMock(return_value="text"))
        agent = MagicMock(engine=MagicMock(model_id="m"), output=msg)
        agent.sync_messages = AsyncMock()
        self.orch._last_engine_agent = agent
        self.memory.has_permanent_message = True
        self.memory.has_recent_message = False
        self.orch._engines_stack.__exit__ = MagicMock(return_value="done")
        engine = MagicMock()
        engine.wait_closed = AsyncMock()
        self.orch._engines = [engine]
        result = await self.orch.__aexit__(None, None, None)
        agent.sync_messages.assert_awaited_once()
        self.memory.__exit__.assert_called_once_with(None, None, None)
        engine.wait_closed.assert_awaited_once()
        self.assertEqual(result, "done")

    async def test_aexit_skips_message_sync_on_keyboard_interrupt(self):
        agent = MagicMock(engine=MagicMock(model_id="m"))
        agent.sync_messages = AsyncMock()
        self.orch._last_engine_agent = agent
        self.orch._engines_stack.__exit__ = MagicMock(return_value=False)
        engine = MagicMock()
        engine.wait_closed = AsyncMock()
        self.orch._engines = [engine]

        await self.orch.__aexit__(KeyboardInterrupt, KeyboardInterrupt(), None)

        agent.sync_messages.assert_not_awaited()
        engine.wait_closed.assert_not_awaited()
        self.memory.__exit__.assert_called_once()
        self.orch._engines_stack.__exit__.assert_called_once()

    async def test_aexit_skips_message_sync_on_cancelled_error(self):
        agent = MagicMock(engine=MagicMock(model_id="m"))
        agent.sync_messages = AsyncMock()
        self.orch._last_engine_agent = agent
        self.orch._engines_stack.__exit__ = MagicMock(return_value=False)

        await self.orch.__aexit__(CancelledError, CancelledError(), None)

        agent.sync_messages.assert_not_awaited()


class OrchestratorInputTokenCountTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        engine_uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="m",
            params={},
        )
        environment = EngineEnvironment(
            engine_uri=engine_uri, settings=TransformerEngineSettings()
        )
        operation = AgentOperation(
            specification=Specification(role=None, goal=None),
            environment=environment,
        )
        memory = MagicMock(spec=MemoryManager)
        memory.participant_id = uuid4()
        self.orchestrator = Orchestrator(
            MagicMock(),
            MagicMock(spec=ModelManager),
            memory,
            MagicMock(spec=ToolManager),
            MagicMock(spec=EventManager),
            [operation],
        )

    def test_input_token_count_without_engine_agent(self):
        self.assertIsNone(self.orchestrator.input_token_count)

    def test_input_token_count_runs_callable_without_running_loop(self):
        engine_agent = MagicMock()
        engine_agent.input_token_count = AsyncMock(return_value=11)
        engine_agent.output = None
        self.orchestrator._last_engine_agent = engine_agent

        self.assertEqual(self.orchestrator.input_token_count, 11)

    async def test_input_token_count_returns_output_count_inside_loop(self):
        engine_agent = MagicMock()
        engine_agent.input_token_count = AsyncMock(return_value=5)
        engine_agent.output = MagicMock(input_token_count=7)
        self.orchestrator._last_engine_agent = engine_agent

        self.assertEqual(self.orchestrator.input_token_count, 7)

    async def test_input_token_count_returns_none_inside_loop_without_output(
        self,
    ):
        engine_agent = MagicMock()
        engine_agent.input_token_count = AsyncMock(return_value=5)
        engine_agent.output = None
        self.orchestrator._last_engine_agent = engine_agent

        self.assertIsNone(self.orchestrator.input_token_count)


class OrchestratorAenterTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_aenter_loads_engine_and_agent(self):
        env = EngineEnvironment(
            engine_uri=EngineUri(
                host=None,
                port=None,
                user=None,
                password=None,
                vendor=None,
                model_id="m",
                params={},
            ),
            settings=TransformerEngineSettings(),
        )
        op = AgentOperation(
            specification=Specification(role=None, goal=None), environment=env
        )
        model_manager = MagicMock(spec=ModelManager)
        fake_engine = MagicMock(
            __enter__=MagicMock(return_value=None),
            __exit__=MagicMock(return_value=False),
            model_id="m",
            tokenizer=None,
        )
        model_manager.load_engine.return_value = fake_engine
        memory = MagicMock(spec=MemoryManager)
        memory.has_permanent_message = False
        memory.has_recent_message = False
        tool = MagicMock(spec=ToolManager)
        event_manager = MagicMock(spec=EventManager)
        logger = MagicMock()
        Orchestrator._engine_agents = {}
        orch = Orchestrator(
            logger, model_manager, memory, tool, event_manager, [op]
        )
        with patch(
            "avalan.agent.orchestrator.TemplateEngineAgent",
            return_value=MagicMock(output=None, sync_messages=AsyncMock()),
        ) as tpatch:
            async with orch:
                pass
        tpatch.assert_called_once()
        model_manager.load_engine.assert_called_once()
        self.assertEqual(orch.model_ids, {"m"})


class OrchestratorUserTransformationOptionsTestCase(unittest.TestCase):
    def _create_orchestrator(self):
        engine_uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="m",
            params={},
        )
        environment = EngineEnvironment(
            engine_uri=engine_uri, settings=TransformerEngineSettings()
        )
        specification = Specification(template_vars={"name": "Bob"})
        operation = AgentOperation(
            specification=specification, environment=environment
        )
        orchestrator = Orchestrator(
            MagicMock(),
            MagicMock(spec=ModelManager),
            MagicMock(spec=MemoryManager),
            MagicMock(spec=ToolManager),
            MagicMock(spec=EventManager),
            [operation],
            user="hello {{input}} {{name}}",
        )
        orchestrator._renderer = Renderer()
        return orchestrator, specification

    def _message(self, text: str) -> Message:
        return Message(
            role=MessageRole.USER,
            content=MessageContentText(type="text", text=text),
        )

    def test_user_string_transformation(self):
        orchestrator, specification = self._create_orchestrator()
        message = orchestrator._input_messages(specification, "world")
        self.assertEqual(message.content, b"hello world Bob")

    def test_user_list_strings_transformation(self):
        orchestrator, specification = self._create_orchestrator()
        result = orchestrator._input_messages(specification, ["world"])
        self.assertEqual(result, ["world"])

    def test_user_message_transformation(self):
        orchestrator, specification = self._create_orchestrator()
        msg = self._message("earth")
        message = orchestrator._input_messages(specification, msg)
        self.assertEqual(message.content, b"hello earth Bob")

    def test_user_list_messages_transformation(self):
        orchestrator, specification = self._create_orchestrator()
        msg = self._message("moon")
        messages = orchestrator._input_messages(specification, [msg])
        self.assertEqual(messages[0].content, b"hello moon Bob")


class OrchestratorUserTemplateTransformationOptionsTestCase(unittest.TestCase):
    def _create_orchestrator(self):
        engine_uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="m",
            params={},
        )
        environment = EngineEnvironment(
            engine_uri=engine_uri, settings=TransformerEngineSettings()
        )
        specification = Specification(template_vars={"name": "Ann"})
        operation = AgentOperation(
            specification=specification, environment=environment
        )
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with open(join(tmp.name, "user.md"), "w", encoding="utf-8") as fh:
            fh.write("hi {{input}} {{name}}")
        orchestrator = Orchestrator(
            MagicMock(),
            MagicMock(spec=ModelManager),
            MagicMock(spec=MemoryManager),
            MagicMock(spec=ToolManager),
            MagicMock(spec=EventManager),
            [operation],
            user_template="user.md",
        )
        orchestrator._renderer = Renderer(templates_path=tmp.name)
        return orchestrator, specification

    def _message(self, text: str) -> Message:
        return Message(
            role=MessageRole.USER,
            content=MessageContentText(type="text", text=text),
        )

    def test_user_template_string_transformation(self):
        orchestrator, specification = self._create_orchestrator()
        message = orchestrator._input_messages(specification, "earth")
        self.assertEqual(message.content, "hi earth Ann")

    def test_user_template_list_strings_transformation(self):
        orchestrator, specification = self._create_orchestrator()
        result = orchestrator._input_messages(specification, ["earth"])
        self.assertEqual(result, ["earth"])

    def test_user_template_message_transformation(self):
        orchestrator, specification = self._create_orchestrator()
        msg = self._message("earth")
        message = orchestrator._input_messages(specification, msg)
        self.assertEqual(message.content, "hi earth Ann")

    def test_user_template_list_messages_transformation(self):
        orchestrator, specification = self._create_orchestrator()
        msg = self._message("earth")
        messages = orchestrator._input_messages(specification, [msg])
        self.assertEqual(messages[0].content, "hi earth Ann")


class OrchestratorSettingsTemplateVarsUserTestCase(unittest.TestCase):
    def _create_orchestrator(self):
        engine_uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="m",
            params={},
        )
        environment = EngineEnvironment(
            engine_uri=engine_uri, settings=TransformerEngineSettings()
        )
        specification = Specification(
            settings=GenerationSettings(template_vars={"name": "Bob"})
        )
        operation = AgentOperation(
            specification=specification, environment=environment
        )
        orchestrator = Orchestrator(
            MagicMock(),
            MagicMock(spec=ModelManager),
            MagicMock(spec=MemoryManager),
            MagicMock(spec=ToolManager),
            MagicMock(spec=EventManager),
            [operation],
            user="hello {{input}} {{name}}",
        )
        orchestrator._renderer = Renderer()
        return orchestrator, specification

    def test_user_string_transformation(self):
        orchestrator, specification = self._create_orchestrator()
        message = orchestrator._input_messages(specification, "world")
        self.assertEqual(message.content, b"hello world Bob")


class OrchestratorSettingsTemplateVarsUserTemplateTestCase(unittest.TestCase):
    def _create_orchestrator(self):
        engine_uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="m",
            params={},
        )
        environment = EngineEnvironment(
            engine_uri=engine_uri, settings=TransformerEngineSettings()
        )
        specification = Specification(
            settings=GenerationSettings(template_vars={"name": "Ann"})
        )
        operation = AgentOperation(
            specification=specification, environment=environment
        )
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with open(join(tmp.name, "user.md"), "w", encoding="utf-8") as fh:
            fh.write("hi {{input}} {{name}}")
        orchestrator = Orchestrator(
            MagicMock(),
            MagicMock(spec=ModelManager),
            MagicMock(spec=MemoryManager),
            MagicMock(spec=ToolManager),
            MagicMock(spec=EventManager),
            [operation],
            user_template="user.md",
        )
        orchestrator._renderer = Renderer(templates_path=tmp.name)
        return orchestrator, specification

    def test_user_template_string_transformation(self):
        orchestrator, specification = self._create_orchestrator()
        message = orchestrator._input_messages(specification, "earth")
        self.assertEqual(message.content, "hi earth Ann")
