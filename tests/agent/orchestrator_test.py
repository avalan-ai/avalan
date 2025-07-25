import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from avalan.agent.orchestrator import Orchestrator
from avalan.agent import (
    AgentOperation,
    Specification,
    EngineEnvironment,
    InputType,
    NoOperationAvailableException,
)
from avalan.entities import (
    EngineUri,
    TransformerEngineSettings,
    Message,
)
from avalan.memory.manager import MemoryManager
from avalan.model.manager import ModelManager
from avalan.tool.manager import ToolManager
from avalan.event.manager import EventManager
from avalan.event import EventType
from uuid import uuid4
from json import dumps
from dataclasses import asdict


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
        self.orch._last_engine_agent = agent
        self.memory.has_permanent_message = True
        self.memory.has_recent_message = False
        self.orch._engines_stack.__exit__ = MagicMock(return_value="done")
        result = await self.orch.__aexit__(None, None, None)
        self.memory.append_message.assert_awaited_once()
        self.memory.__exit__.assert_called_once_with(None, None, None)
        self.assertEqual(result, "done")


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
            return_value=MagicMock(output=None),
        ) as tpatch:
            async with orch:
                pass
        tpatch.assert_called_once()
        model_manager.load_engine.assert_called_once()
        self.assertEqual(orch.model_ids, {"m"})
