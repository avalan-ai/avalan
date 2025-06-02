from avalan.agent import Goal, InputType, OutputType, Specification
from avalan.agent.orchestrator.orchestrators.default import DefaultOrchestrator
from avalan.agent.renderer import TemplateEngineAgent
from avalan.event.manager import EventManager
from avalan.event import EventType
from avalan.entities import (
    EngineUri,
    MessageRole,
    TransformerEngineSettings,
)
from avalan.agent.orchestrator.response.orchestrator_execution_response import (
    OrchestratorExecutionResponse,
)
from avalan.model import TextGenerationResponse
from avalan.model.manager import ModelManager
from avalan.memory.manager import MemoryManager
from avalan.tool.manager import ToolManager
from logging import Logger
from unittest import TestCase, IsolatedAsyncioTestCase
from unittest.mock import MagicMock, AsyncMock, patch
from uuid import uuid4


class DefaultOrchestratorInitTestCase(TestCase):
    def test_initialization(self):
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

        orch = DefaultOrchestrator(
            engine_uri,
            logger,
            model_manager,
            memory,
            tool,
            event_manager,
            name="Agent",
            role="assistant",
            task="do",
            instructions="something",
            rules=["a", "b"],
            template_id="tmpl",
            settings=settings,
            call_options={"x": 1},
            template_vars={"y": 2},
            id=uuid4(),
        )

        self.assertEqual(orch.name, "Agent")
        self.assertEqual(len(orch.operations), 1)
        op = orch.operations[0]
        self.assertIs(op.environment.engine_uri, engine_uri)
        self.assertIs(op.environment.settings, settings)
        self.assertEqual(op.specification.role, "assistant")
        self.assertEqual(op.specification.goal.task, "do")
        self.assertEqual(op.specification.goal.instructions, ["something"])
        self.assertEqual(op.specification.rules, ["a", "b"])
        self.assertEqual(op.specification.template_id, "tmpl")
        self.assertEqual(op.specification.template_vars, {"y": 2})


class DefaultOrchestratorExecutionTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        super().setUp()
        self.addCleanup(patch.stopall)

    @patch("avalan.agent.orchestrator.TemplateEngineAgent")
    async def test_stream_end_event(self, Agent):
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
        engine.tokenizer = MagicMock()
        engine.tokenizer.encode.side_effect = [[1], [2]]
        model_manager.load_engine.return_value = engine

        async def output_gen():
            yield "a"
            yield "b"

        def output_fn(*args, **kwargs):
            return output_gen()

        response = TextGenerationResponse(output_fn, use_async_generator=True)

        agent_mock = AsyncMock(spec=TemplateEngineAgent)
        agent_mock.engine = engine
        agent_mock.return_value = response

        Agent.return_value = agent_mock

        async with DefaultOrchestrator(
            engine_uri,
            logger,
            model_manager,
            memory,
            tool,
            event_manager,
            name="Agent",
            role="assistant",
            task="do",
            instructions="something",
            rules=None,
            settings=settings,
        ) as orch:
            model_manager.load_engine.assert_called_once_with(
                engine_uri, settings
            )
            Agent.assert_called_once()
            self.assertIs(orch.engine_agent, agent_mock)
            self.assertEqual(orch.model_ids, {"m"})

            result = await orch("hi", use_async_generator=True)

            tokens = []
            async for t in result:
                tokens.append(t)

        agent_mock.assert_awaited_once()
        spec_arg, msg_arg = agent_mock.await_args.args
        self.assertEqual(msg_arg.content, "hi")
        self.assertEqual(msg_arg.role, MessageRole.USER)
        self.assertIsNone(msg_arg.name)
        self.assertIsNone(msg_arg.arguments)
        self.assertEqual(
            spec_arg,
            Specification(
                role="assistant",
                goal=Goal(task="do", instructions=["something"]),
                rules=None,
                input_type=InputType.TEXT,
                output_type=OutputType.TEXT,
                settings=None,
                template_id="agent.md",
                template_vars=None,
            ),
        )

        self.assertIsInstance(result, OrchestratorExecutionResponse)
        self.assertEqual(tokens, ["a", "b"])

        calls = event_manager.trigger.await_args_list
        self.assertTrue(
            any(c.args[0].type == EventType.STREAM_END for c in calls)
        )

        token_events = [
            c.args[0]
            for c in calls
            if c.args[0].type == EventType.TOKEN_GENERATED
        ]
        self.assertEqual(len(token_events), 2)
        self.assertEqual(
            token_events[0].payload,
            {"token_id": 1, "model_id": "m", "token": "a", "step": 0},
        )
        self.assertEqual(
            token_events[1].payload,
            {"token_id": 2, "model_id": "m", "token": "b", "step": 1},
        )

        memory.__exit__.assert_called_once()
