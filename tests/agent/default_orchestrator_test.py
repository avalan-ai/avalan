from avalan.agent import (
    Goal,
    InputType, 
    OutputType,
    Specification
)
from avalan.agent.orchestrators.default import DefaultOrchestrator
from avalan.agent.orchestrator import TemplateEngineAgent
from avalan.event.manager import EventManager
from avalan.model.entities import (
    EngineUri, 
    Message, 
    MessageRole,
    TransformerEngineSettings
)
from avalan.model.manager import ModelManager
from avalan.memory.manager import MemoryManager
from avalan.tool.manager import ToolManager
from logging import Logger
from unittest import TestCase, IsolatedAsyncioTestCase
from unittest.mock import MagicMock, AsyncMock, patch
from uuid import uuid4

class DefaultOrchestratorInitTestCase(TestCase):
    def test_initialization(self):
        engine_uri = EngineUri(host=None, port=None, user=None, password=None, vendor=None, model_id="m", params={})
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
    async def test_aenter_call_aexit(self):
        engine_uri = EngineUri(host=None, port=None, user=None, password=None, vendor=None, model_id="m", params={})
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
            rules=None,
            settings=settings,
        )

        engine = MagicMock()
        engine.__enter__.return_value = engine
        engine.__exit__.return_value = False
        engine.model_id = "m"
        model_manager.load_engine.return_value = engine

        agent_mock = AsyncMock(spec=TemplateEngineAgent)
        agent_mock.engine = engine
        agent_mock.return_value = "ok"

        with patch("avalan.agent.orchestrator.TemplateEngineAgent", return_value=agent_mock) as Agent:
            await orch.__aenter__()
            model_manager.load_engine.assert_called_once_with(engine_uri, settings)
            Agent.assert_called_once()
            self.assertIs(orch.engine_agent, agent_mock)
            self.assertEqual(orch.model_ids, {"m"})

            result = await orch("hi")
            agent_mock.assert_awaited_once_with(
                Specification(
                    role='assistant', 
                    goal=Goal(
                        task='do', 
                        instructions=['something']
                    ), 
                    rules=None, 
                    input_type=InputType.TEXT, 
                    output_type=OutputType.TEXT, 
                    settings=None, 
                    template_id='agent.md', 
                    template_vars=None
                ), 
                Message(
                    role=MessageRole.USER, 
                    content='hi', 
                    name=None, 
                    arguments=None
                )
            )
            self.assertEqual(result, "ok")
            
            await orch.__aexit__(None, None, None)
            memory.__exit__.assert_called_once()
