import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from avalan.agent.engine import EngineAgent
from avalan.agent.orchestrator import Orchestrator
from avalan.agent.orchestrator.orchestrators.json import (
    JsonSpecification,
    Property,
)
from avalan.agent import (
    EngineEnvironment,
    EngineOperation,
    Specification,
    Role,
    OutputType,
    Goal,
)
from avalan.entities import EngineUri, TransformerEngineSettings
from avalan.memory.manager import MemoryManager
from avalan.model.manager import ModelManager
from avalan.tool.manager import ToolManager
from avalan.event.manager import EventManager


class DummyEngine:
    def __init__(self) -> None:
        self.model_id = "m"
        self.tokenizer = MagicMock(eos_token="<eos>")

    async def __call__(self, *a, **k):
        return "ok"

    def input_token_count(self, *a, **k):
        return 1


class _MissingPrepareAgent(EngineAgent):
    def _prepare_call(self, specification, input, **kwargs):
        return super()._prepare_call(specification, input, **kwargs)


class EngineAgentCoverageTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.engine = DummyEngine()
        self.memory = MagicMock()
        self.tool = MagicMock(spec=ToolManager)
        self.event_manager = MagicMock(spec=EventManager)
        self.event_manager.trigger = AsyncMock()

    async def test_prepare_call_not_implemented(self):
        agent = _MissingPrepareAgent(
            self.engine,
            self.memory,
            self.tool,
            self.event_manager,
            MagicMock(spec=ModelManager),
            EngineUri(
                host=None,
                port=None,
                user=None,
                password=None,
                vendor=None,
                model_id="m",
                params={},
            ),
        )
        with self.assertRaises(NotImplementedError):
            agent._prepare_call(Specification(role=None, goal=None), "hi")

    async def test_output_property(self):
        agent = _MissingPrepareAgent(
            self.engine,
            self.memory,
            self.tool,
            self.event_manager,
            MagicMock(spec=ModelManager),
            EngineUri(
                host=None,
                port=None,
                user=None,
                password=None,
                vendor=None,
                model_id="m",
                params={},
            ),
        )
        agent._last_output = "value"
        self.assertEqual(agent.output, "value")


class OrchestratorCoverageTestCase(unittest.IsolatedAsyncioTestCase):
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
        self.environment = EngineEnvironment(
            engine_uri=engine_uri,
            settings=TransformerEngineSettings(),
        )
        self.operation = EngineOperation(
            specification=Specification(role=None, goal=None),
            environment=self.environment,
        )
        self.logger = MagicMock()
        self.model_manager = MagicMock(spec=ModelManager)
        self.memory = MagicMock(spec=MemoryManager)
        self.memory.participant_id = "pid"
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
        self.agent = AsyncMock()
        self.agent.engine = DummyEngine()
        env_patch = patch(
            "avalan.agent.orchestrator.dumps", lambda *a, **k: "hash"
        )
        self.addCleanup(env_patch.stop)
        env_patch.start()
        Orchestrator._engine_agents = {}

    async def test_id_and_reset(self):
        self.orch._operation_step = 0
        Orchestrator._engine_agents["hash"] = self.agent
        await self.orch("text")
        self.agent.assert_awaited_once()
        self.assertEqual(self.orch.id, self.orch._id)

    async def test_aenter_no_engine(self):
        self.model_manager.load_engine.return_value = None
        with self.assertRaises(NotImplementedError):
            await self.orch.__aenter__()


class JsonSpecificationCoverageTestCase(unittest.TestCase):
    def test_list_properties(self):
        props = [
            Property(name="v", data_type="string"),
            Property(name="n", data_type="int"),
        ]
        spec = JsonSpecification(
            output=props,
            role="r",
            goal=Goal(task="t", instructions=[]),
        )
        self.assertEqual(spec.template_vars["output_properties"], props)
        self.assertEqual(spec.role, Role(persona=["r"]))
        self.assertEqual(spec.output_type, OutputType.JSON)


if __name__ == "__main__":
    unittest.main()
