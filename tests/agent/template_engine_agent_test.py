from avalan.agent.renderer import TemplateEngineAgent, Renderer
from avalan.agent import Specification, Goal, Role
from avalan.event import EventType
from avalan.memory.manager import MemoryManager
from avalan.tool.manager import ToolManager
from avalan.event.manager import EventManager
from unittest import TestCase, IsolatedAsyncioTestCase
from unittest.mock import MagicMock, AsyncMock


class TemplateEngineAgentPrepareTestCase(TestCase):
    def setUp(self):
        self.renderer = Renderer()
        self.agent = TemplateEngineAgent(
            model=MagicMock(),
            memory=MagicMock(spec=MemoryManager),
            tool=MagicMock(spec=ToolManager),
            event_manager=MagicMock(spec=EventManager),
            renderer=self.renderer,
            name="Bob",
        )

    def test_prepare_call_no_template_vars(self):
        spec = Specification(
            role="assistant",
            goal=Goal(task="do", instructions=["instr"]),
            rules=["rule"],
        )
        result = self.agent._prepare_call(spec, "hi")
        expected_prompt = self.renderer(
            "agent.md",
            name="Bob",
            roles=["assistant"],
            task=b"do",
            instructions=[b"instr"],
            rules=[b"rule"],
        )
        self.assertEqual(result["settings"], spec.settings)
        self.assertEqual(result["system_prompt"], expected_prompt)

    def test_prepare_call_with_template_vars(self):
        spec = Specification(
            role=Role(persona=["role {{verb}}"]),
            goal=Goal(task="do {{verb}}", instructions=["inst {{verb}}"]),
            rules=["rule {{verb}}"],
            template_vars={"verb": "run"},
        )
        result = self.agent._prepare_call(spec, "hi")
        expected_prompt = self.renderer(
            "agent.md",
            name="Bob",
            roles=[b"role run"],
            task=b"do run",
            instructions=[b"inst run"],
            rules=[b"rule run"],
        )
        self.assertEqual(result["system_prompt"], expected_prompt)

    def test_prepare_call_goal_none(self):
        spec = Specification(
            role="assistant",
            goal=None,
            rules=[],
            template_vars={"verb": "x"},
        )
        result = self.agent._prepare_call(spec, "hi")
        expected_prompt = self.renderer(
            "agent.md",
            name="Bob",
            roles=["assistant"],
            task=None,
            instructions=None,
            rules=[],
        )
        self.assertEqual(result["system_prompt"], expected_prompt)


class TemplateEngineAgentCallTestCase(IsolatedAsyncioTestCase):
    async def test_call_invokes_run_with_prepared_arguments(self):
        renderer = Renderer()
        memory = MagicMock(spec=MemoryManager)
        memory.has_permanent_message = False
        memory.has_recent_message = False
        tool = MagicMock(spec=ToolManager)
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        model = MagicMock()

        agent = TemplateEngineAgent(
            model=model,
            memory=memory,
            tool=tool,
            event_manager=event_manager,
            renderer=renderer,
            name="Bob",
        )

        spec = Specification(
            role="assistant",
            goal=Goal(task="do", instructions=["ins"]),
            rules=["r"],
        )

        expected_prompt = renderer(
            "agent.md",
            name="Bob",
            roles=["assistant"],
            task=b"do",
            instructions=[b"ins"],
            rules=[b"r"],
        )

        agent._run = AsyncMock(return_value="out")

        result = await agent(spec, "hello")

        agent._run.assert_awaited_once()
        self.assertEqual(result, "out")
        self.assertEqual(
            agent._run.await_args.kwargs["system_prompt"],
            expected_prompt,
        )
        event_types = [
            c.args[0].type for c in event_manager.trigger.await_args_list[:2]
        ]
        self.assertEqual(
            event_types,
            [EventType.CALL_PREPARE_BEFORE, EventType.CALL_PREPARE_AFTER],
        )
