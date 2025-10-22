from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from avalan.agent.orchestrator import Orchestrator
from avalan.agent.orchestrator.orchestrators.reasoning.cot import (
    ReasoningOrchestrator,
)
from avalan.agent.orchestrator.orchestrators.reasoning.parser import (
    ReasoningOutputParser,
)
from avalan.agent.renderer import Renderer
from avalan.entities import (
    Message,
    MessageRole,
    ReasoningOrchestratorResponse,
)


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


class ReasoningOrchestratorAdditionalTestCase(IsolatedAsyncioTestCase):
    async def _make_orchestrator(self, text: str):
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
            renderer=renderer,
            operations=[
                MagicMock(specification=MagicMock(template_vars=None))
            ],
        )
        orchestrator.name = "Agent"
        resp = MagicMock()
        resp.to_str = AsyncMock(return_value=text)
        orchestrator.return_value = resp
        return ReasoningOrchestrator(orchestrator), orchestrator

    async def test_context_manager(self):
        renderer = Renderer()
        orch = AsyncMock(
            __aenter__=AsyncMock(return_value=None),
            __aexit__=AsyncMock(return_value=True),
            _logger=MagicMock(),
            _model_manager=MagicMock(),
            _memory=MagicMock(),
            _tool=MagicMock(),
            _event_manager=MagicMock(),
            _call_options=None,
            _exit_memory=True,
            id=uuid4(),
            name="ctx",
            renderer=renderer,
            operations=[],
        )
        cot = ReasoningOrchestrator(orch)
        async with cot as res:
            self.assertIs(res, cot)
        orch.__aenter__.assert_awaited_once()
        orch.__aexit__.assert_awaited_once()

    def test_name_property(self):
        renderer = Renderer()
        orch = AsyncMock(
            _logger=MagicMock(),
            _model_manager=MagicMock(),
            _memory=MagicMock(),
            _tool=MagicMock(),
            _event_manager=MagicMock(),
            _call_options=None,
            _exit_memory=True,
            id=uuid4(),
            renderer=renderer,
            operations=[],
        )
        orch.name = "named"
        cot = ReasoningOrchestrator(orch)
        self.assertEqual(cot.name, "named")

    def test_name_direct_getattr(self):
        renderer = Renderer()
        orch = AsyncMock(
            _logger=MagicMock(),
            _model_manager=MagicMock(),
            _memory=MagicMock(),
            _tool=MagicMock(),
            _event_manager=MagicMock(),
            _call_options=None,
            _exit_memory=True,
            id=uuid4(),
            renderer=renderer,
            operations=[],
        )
        orch.name = "named"
        cot = ReasoningOrchestrator(orch)
        self.assertEqual(cot.__getattr__("name"), "named")

    def test_name_indirect_getattr(self):
        renderer = Renderer()
        orig = Orchestrator.name
        delattr(Orchestrator, "name")
        try:
            orch = AsyncMock(
                _logger=MagicMock(),
                _model_manager=MagicMock(),
                _memory=MagicMock(),
                _tool=MagicMock(),
                _event_manager=MagicMock(),
                _call_options=None,
                _exit_memory=True,
                id=uuid4(),
                renderer=renderer,
                operations=[],
            )
            orch.name = "named"
            cot = ReasoningOrchestrator(orch)
            self.assertEqual(getattr(cot, "name"), "named")
        finally:
            setattr(Orchestrator, "name", orig)

    async def test_think_without_answer(self):
        cot, orchestrator = await self._make_orchestrator(
            "<think>r</think> done"
        )
        result = await cot("q")
        self.assertEqual(
            result, ReasoningOrchestratorResponse(answer="done", reasoning="r")
        )
        orchestrator.assert_awaited()

    async def test_reasoning_without_answer(self):
        cot, orchestrator = await self._make_orchestrator("Reasoning: steps")
        result = await cot("q")
        self.assertEqual(
            result,
            ReasoningOrchestratorResponse(answer="steps", reasoning="steps"),
        )
        orchestrator.assert_awaited()

    async def test_thought_with_answer(self):
        cot, orchestrator = await self._make_orchestrator(
            "Thought: r\nAnswer: a"
        )
        result = await cot("q")
        self.assertEqual(
            result, ReasoningOrchestratorResponse(answer="a", reasoning="r")
        )
        orchestrator.assert_awaited()

    async def test_thought_without_answer(self):
        cot, orchestrator = await self._make_orchestrator("Thought: r")
        result = await cot("q")
        self.assertEqual(
            result, ReasoningOrchestratorResponse(answer="r", reasoning="r")
        )
        orchestrator.assert_awaited()

    async def test_no_prefix(self):
        cot, orchestrator = await self._make_orchestrator("just text")
        result = await cot("q")
        self.assertEqual(
            result,
            ReasoningOrchestratorResponse(answer="just text", reasoning=None),
        )
        orchestrator.assert_awaited()


class ReasoningOrchestratorConfigTestCase(IsolatedAsyncioTestCase):
    async def _make(self, text: str):
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
        resp.to_str = AsyncMock(return_value=text)
        orchestrator.return_value = resp
        return orchestrator

    async def test_custom_parser_matches(self):
        orch = await self._make("<analysis>step</analysis> Final: done")
        parser = ReasoningOutputParser(
            reasoning_tag="analysis", answer_prefix="final"
        )
        cot = ReasoningOrchestrator(orch, parser=parser)
        result = await cot("q")
        self.assertEqual(
            result,
            ReasoningOrchestratorResponse(answer="done", reasoning="step"),
        )
        orch.assert_awaited()

    async def test_custom_parser_no_matches(self):
        orch = await self._make("Answer: 1")
        parser = ReasoningOutputParser(
            reasoning_tag="analysis", answer_prefix="final"
        )
        cot = ReasoningOrchestrator(orch, parser=parser)
        result = await cot("q")
        self.assertEqual(
            result,
            ReasoningOrchestratorResponse(answer="Answer: 1", reasoning=None),
        )
        orch.assert_awaited()
