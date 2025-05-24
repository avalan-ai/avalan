from avalan.agent import Goal, InputType, OutputType, Specification, Role
from avalan.agent.orchestrators.json import JsonOrchestrator, Property, JsonSpecification
from avalan.agent.orchestrator import TemplateEngineAgent
from avalan.event.manager import EventManager
from avalan.event import EventType
from avalan.model.entities import EngineUri, MessageRole, TransformerEngineSettings
from avalan.model.nlp.text import TextGenerationResponse
from avalan.model.manager import ModelManager
from avalan.memory.manager import MemoryManager
from avalan.tool.manager import ToolManager
from dataclasses import dataclass
from typing import Annotated
from logging import Logger
from unittest import TestCase, IsolatedAsyncioTestCase
from unittest.mock import MagicMock, AsyncMock, patch

class JsonOrchestratorInitTestCase(TestCase):
    def test_init_from_class(self):
        @dataclass
        class Output:
            value: Annotated[int, "desc"]

        engine_uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="m",
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
            Output,
            name="Agent",
            role="assistant",
            task="do",
            instructions="something",
            rules=["a"],
            template_id="tmpl",
            settings=settings,
            call_options={"x": 1},
            template_vars={"y": 2},
        )

        self.assertEqual(orch.name, "Agent")
        self.assertEqual(len(orch.operations), 1)
        op = orch.operations[0]
        self.assertIs(op.environment.engine_uri, engine_uri)
        self.assertIs(op.environment.settings, settings)
        self.assertEqual(op.specification.role.persona, ["assistant"])
        self.assertEqual(op.specification.goal.task, "do")
        self.assertEqual(op.specification.goal.instructions, ["something"])
        self.assertEqual(op.specification.rules, ["a"])
        self.assertEqual(op.specification.output_type, OutputType.JSON)
        self.assertEqual(op.specification.template_id, "tmpl")
        self.assertEqual(op.specification.template_vars["y"], 2)
        self.assertEqual(
            op.specification.template_vars["output_properties"],
            [Property(name="value", data_type="int", description="desc")],
        )

    def test_init_from_properties(self):
        properties = [Property(name="value", data_type="string", description="d")]
        engine_uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="m",
            params={},
        )
        logger = MagicMock(spec=Logger)
        model_manager = MagicMock(spec=ModelManager)
        memory = MagicMock(spec=MemoryManager)
        tool = MagicMock(spec=ToolManager)
        event_manager = MagicMock(spec=EventManager)

        orch = JsonOrchestrator(
            engine_uri,
            logger,
            model_manager,
            memory,
            tool,
            event_manager,
            properties,
            role="assistant",
            task="do",
            instructions="something",
        )

        op = orch.operations[0]
        self.assertEqual(
            op.specification.template_vars["output_properties"],
            properties,
        )

class JsonOrchestratorExecutionTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        super().setUp()
        self.addCleanup(patch.stopall)

    @patch("avalan.agent.orchestrator.TemplateEngineAgent")
    async def test_json_output(self, Agent):
        engine_uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="m",
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

        async def output_gen():
            yield "{\"value\": 1}"

        def output_fn(*args, **kwargs):
            return output_gen()

        response = TextGenerationResponse(output_fn, use_async_generator=True)

        agent_mock = AsyncMock(spec=TemplateEngineAgent)
        agent_mock.engine = engine
        agent_mock.return_value = response

        Agent.return_value = agent_mock

        properties = [Property(name="value", data_type="int", description="d")]

        async with JsonOrchestrator(
            engine_uri,
            logger,
            model_manager,
            memory,
            tool,
            event_manager,
            properties,
            role="assistant",
            task="do",
            instructions="something",
            settings=settings,
        ) as orch:
            model_manager.load_engine.assert_called_once_with(
                engine_uri,
                settings,
            )
            Agent.assert_called_once()
            self.assertIs(orch.engine_agent, agent_mock)
            self.assertEqual(orch.model_ids, {"m"})

            result = await orch("hi", use_async_generator=True)

        agent_mock.assert_awaited_once()
        spec_arg, msg_arg = agent_mock.await_args.args
        self.assertEqual(msg_arg.content, "hi")
        self.assertEqual(msg_arg.role, MessageRole.USER)
        self.assertIsNone(msg_arg.name)
        self.assertIsNone(msg_arg.arguments)
        self.assertEqual(spec_arg.output_type, OutputType.JSON)
        self.assertEqual(spec_arg.goal.task, "do")
        self.assertEqual(spec_arg.role.persona, ["assistant"])
        self.assertEqual(spec_arg.template_id, "agent_json.md")
        self.assertEqual(
            spec_arg.template_vars,
            {"output_properties": properties},
        )

        self.assertEqual(result, "{\"value\": 1}")
        self.assertTrue(
            any(
                c.args[0].type == EventType.STREAM_END
                for c in event_manager.trigger.await_args_list
            )
        )

        memory.__exit__.assert_called_once()
