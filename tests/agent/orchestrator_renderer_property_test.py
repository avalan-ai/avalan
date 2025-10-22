from dataclasses import dataclass
from logging import Logger
from typing import Annotated
from unittest import TestCase
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from avalan.agent import (
    AgentOperation,
    EngineEnvironment,
    EngineUri,
    Specification,
)
from avalan.agent.orchestrator import Orchestrator
from avalan.agent.orchestrator.orchestrators.default import DefaultOrchestrator
from avalan.agent.orchestrator.orchestrators.json import JsonOrchestrator
from avalan.agent.orchestrator.orchestrators.reasoning.cot import (
    ReasoningOrchestrator,
)
from avalan.agent.renderer import Renderer
from avalan.entities import TransformerEngineSettings
from avalan.event.manager import EventManager
from avalan.memory.manager import MemoryManager
from avalan.model.manager import ModelManager
from avalan.tool.manager import ToolManager


@dataclass
class DummyOutput:
    value: Annotated[str, "desc"]


class RendererPropertyTestCase(TestCase):
    def setUp(self):
        self.engine_uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="m",
            params={},
        )
        self.logger = MagicMock(spec=Logger)
        self.model_manager = MagicMock(spec=ModelManager)
        self.memory = MagicMock(spec=MemoryManager)
        self.tool = MagicMock(spec=ToolManager)
        self.event_manager = MagicMock(spec=EventManager)
        self.settings = TransformerEngineSettings()

    def test_renderer_on_orchestrator(self):
        op = AgentOperation(
            specification=Specification(role=None, goal=None),
            environment=EngineEnvironment(
                engine_uri=self.engine_uri, settings=self.settings
            ),
        )
        orch = Orchestrator(
            self.logger,
            self.model_manager,
            self.memory,
            self.tool,
            self.event_manager,
            [op],
        )
        self.assertIsInstance(orch.renderer, Renderer)

    def test_renderer_on_default_orchestrator(self):
        orch = DefaultOrchestrator(
            self.engine_uri,
            self.logger,
            self.model_manager,
            self.memory,
            self.tool,
            self.event_manager,
            name="Agent",
            role="assistant",
            task="do",
            instructions="it",
            rules=None,
            settings=self.settings,
        )
        self.assertIsInstance(orch.renderer, Renderer)

    def test_renderer_on_json_orchestrator(self):
        orch = JsonOrchestrator(
            self.engine_uri,
            self.logger,
            self.model_manager,
            self.memory,
            self.tool,
            self.event_manager,
            DummyOutput,
            role="assistant",
            task="do",
            instructions="it",
            settings=self.settings,
        )
        self.assertIsInstance(orch.renderer, Renderer)

    def test_renderer_on_reasoning_orchestrator(self):
        base = AsyncMock(
            _logger=self.logger,
            _model_manager=self.model_manager,
            _memory=self.memory,
            _tool=self.tool,
            _event_manager=self.event_manager,
            _call_options=None,
            _exit_memory=True,
            id=uuid4(),
            name="Agent",
            renderer=Renderer(),
            operations=[],
        )
        orch = ReasoningOrchestrator(base)
        self.assertIs(orch.renderer, base.renderer)
