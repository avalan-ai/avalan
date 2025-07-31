from avalan.agent import InputType, OutputType, Role
from avalan.agent.orchestrator.orchestrators.json import (
    JsonOrchestrator,
    JsonSpecification,
)
from avalan.agent.renderer import TemplateEngineAgent
from avalan.entities import (
    EngineUri,
    MessageRole,
    TransformerEngineSettings,
)
from avalan.event.manager import EventManager
from avalan.event import EventType
from avalan.model import TextGenerationResponse
from logging import getLogger
from avalan.model.manager import ModelManager
from avalan.memory.manager import MemoryManager
from avalan.tool.manager import ToolManager
from dataclasses import dataclass
from typing import Annotated
from logging import Logger
from unittest import TestCase, IsolatedAsyncioTestCase
from unittest.mock import MagicMock, AsyncMock, patch


@dataclass
class ExampleOutput:
    value: Annotated[str, "desc"]
    count: Annotated[int, "desc2"]


class JsonOrchestratorInitTestCase(TestCase):
    def test_initialization(self):
        engine_uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="m_json",
            params={},
        )
        logger = MagicMock(spec=Logger)
        model_manager = MagicMock(spec=ModelManager)
        memory = MagicMock(spec=MemoryManager)
        tool = MagicMock(spec=ToolManager)
        event_manager = MagicMock(spec=EventManager)
        settings = TransformerEngineSettings()

        orch = JsonOrchestrator(
            engine_uri,
            logger,
            model_manager,
            memory,
            tool,
            event_manager,
            ExampleOutput,
            name="Agent",
            role="assistant",
            task="do",
            instructions="something",
            rules=["a", "b"],
            template_id="tmpl",
            settings=settings,
            call_options={"x": 1},
            template_vars={"y": 2},
        )

        self.assertEqual(len(orch.operations), 1)
        op = orch.operations[0]
        self.assertIs(op.environment.engine_uri, engine_uri)
        self.assertIs(op.environment.settings, settings)
        self.assertIsInstance(op.specification, JsonSpecification)
        self.assertEqual(op.specification.role, Role(persona=["assistant"]))
        self.assertEqual(op.specification.goal.task, "do")
        self.assertEqual(op.specification.goal.instructions, ["something"])
        self.assertEqual(op.specification.rules, ["a", "b"])
        self.assertEqual(op.specification.template_id, "tmpl")
        self.assertEqual(op.specification.template_vars, {"y": 2})
        self.assertEqual(op.specification.output_type, OutputType.JSON)
        self.assertEqual(op.specification.input_type, InputType.TEXT)


class JsonOrchestratorExecutionTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        super().setUp()
        self.addCleanup(patch.stopall)

    @patch("avalan.agent.orchestrator.TemplateEngineAgent")
    async def test_json_return(self, Agent):
        engine_uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="m_json",
            params={},
        )
        logger = MagicMock(spec=Logger)
        model_manager = MagicMock(spec=ModelManager)
        memory = MagicMock(spec=MemoryManager)
        memory.has_permanent_message = False
        memory.has_recent_message = False
        memory.__exit__ = MagicMock()
        tool = MagicMock(spec=ToolManager)
        tool.is_empty = True
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        settings = TransformerEngineSettings()

        engine = MagicMock()
        engine.__enter__.return_value = engine
        engine.__exit__.return_value = False
        engine.model_id = "m"
        model_manager.load_engine.return_value = engine

        def output_fn(*args, **kwargs):
            return '{"value": "ok"}'

        response = TextGenerationResponse(
            output_fn, logger=getLogger(), use_async_generator=False
        )

        agent_mock = AsyncMock(spec=TemplateEngineAgent)
        agent_mock.engine = engine
        agent_mock.return_value = response

        Agent.return_value = agent_mock

        async with JsonOrchestrator(
            engine_uri,
            logger,
            model_manager,
            memory,
            tool,
            event_manager,
            ExampleOutput,
            name="Agent",
            role="assistant",
            task="do",
            instructions="something",
            rules=None,
            settings=settings,
        ) as orch:
            result = await orch("hi")

        agent_mock.assert_awaited_once()
        spec_arg, msg_arg = agent_mock.await_args.args
        self.assertEqual(msg_arg.content, "hi")
        self.assertEqual(msg_arg.role, MessageRole.USER)
        self.assertEqual(spec_arg.role, Role(persona=["assistant"]))

        self.assertEqual(result, '{"value": "ok"}')
        self.assertTrue(
            any(
                c.args[0].type == EventType.STREAM_END
                for c in event_manager.trigger.await_args_list
            )
        )
        memory.__exit__.assert_called_once()
