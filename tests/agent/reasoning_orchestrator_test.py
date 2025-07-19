from avalan.agent.orchestrator.orchestrators.reasoning.cot import (
    ReasoningOrchestrator,
)
from avalan.entities import ReasoningOrchestratorResponse
from avalan.agent.renderer import Renderer
from avalan.entities import Message, MessageRole
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4


class ReasoningOrchestratorTestCase(IsolatedAsyncioTestCase):
    async def test_call_renders_and_parses(self):
        renderer = Renderer()
        orchestrator = AsyncMock()
        orchestrator._logger = MagicMock()
        orchestrator._model_manager = MagicMock()
        orchestrator._memory = MagicMock()
        orchestrator._tool = MagicMock()
        orchestrator._event_manager = MagicMock()
        orchestrator._call_options = None
        orchestrator._exit_memory = True
        orchestrator.id = uuid4()
        orchestrator.name = "agent"
        orchestrator.renderer = renderer
        spec = MagicMock()
        spec.template_vars = {"x": "1"}
        operation = MagicMock()
        operation.specification = spec
        orchestrator.operations = [operation]
        resp = MagicMock()
        resp.to_str = AsyncMock(return_value="<think>t</think> Answer: 2")
        orchestrator.return_value = resp

        cot = ReasoningOrchestrator(orchestrator)
        result = await cot("Add {{x}}")

        expected_prompt = renderer(
            "reasoning/cot.md",
            prompt=renderer.from_string("Add {{x}}", {"x": "1"}),
            x="1",
        )
        orchestrator.assert_awaited_once_with(expected_prompt)
        self.assertEqual(
            result, ReasoningOrchestratorResponse(answer="2", reasoning="t")
        )

    async def test_parse_without_thinking(self):
        renderer = Renderer()
        orchestrator = AsyncMock(
            _logger=MagicMock(),
            _model_manager=MagicMock(),
            _memory=MagicMock(),
            _tool=MagicMock(),
            _event_manager=MagicMock(),
            _call_options=None,
            _exit_memory=True,
            id=uuid4(),
            name=None,
            renderer=renderer,
            operations=[
                MagicMock(specification=MagicMock(template_vars=None))
            ],
        )
        resp = MagicMock()
        resp.to_str = AsyncMock(return_value="Answer: 42")
        orchestrator.return_value = resp
        cot = ReasoningOrchestrator(orchestrator)
        result = await cot(Message(role=MessageRole.USER, content="Q"))
        orchestrator.assert_awaited()
        self.assertEqual(
            result, ReasoningOrchestratorResponse(answer="42", reasoning=None)
        )

    async def test_parse_prefix_reasoning(self):
        renderer = Renderer()
        orchestrator = AsyncMock(
            _logger=MagicMock(),
            _model_manager=MagicMock(),
            _memory=MagicMock(),
            _tool=MagicMock(),
            _event_manager=MagicMock(),
            _call_options=None,
            _exit_memory=True,
            id=uuid4(),
            name=None,
            renderer=renderer,
            operations=[
                MagicMock(specification=MagicMock(template_vars=None))
            ],
        )
        resp = MagicMock()
        resp.to_str = AsyncMock(return_value="Reasoning: step\nAnswer: yes")
        orchestrator.return_value = resp
        cot = ReasoningOrchestrator(orchestrator)
        result = await cot("q")
        self.assertEqual(
            result,
            ReasoningOrchestratorResponse(answer="yes", reasoning="step"),
        )
