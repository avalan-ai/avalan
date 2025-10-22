from logging import getLogger
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

from avalan.agent import Goal, Role, Specification
from avalan.agent.renderer import Renderer, TemplateEngineAgent
from avalan.entities import (
    EngineMessage,
    EngineUri,
    GenerationSettings,
    MessageRole,
)
from avalan.event import EventType
from avalan.event.manager import EventManager
from avalan.memory.manager import MemoryManager
from avalan.model import TextGenerationResponse
from avalan.model.call import ModelCallContext
from avalan.model.manager import ModelManager
from avalan.tool.manager import ToolManager


class TemplateEngineAgentPropertyTestCase(TestCase):
    def test_id_property(self) -> None:
        renderer = Renderer()
        memory = MagicMock(spec=MemoryManager)
        tool = MagicMock(spec=ToolManager)
        event_manager = MagicMock(spec=EventManager)
        model_manager = MagicMock(spec=ModelManager)
        model = MagicMock()
        model.model_id = "m"
        model.model_type = "t"
        engine_uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="m",
            params={},
        )
        agent_id: UUID = uuid4()
        agent = TemplateEngineAgent(
            model=model,
            memory=memory,
            tool=tool,
            event_manager=event_manager,
            model_manager=model_manager,
            renderer=renderer,
            engine_uri=engine_uri,
            id=agent_id,
        )
        self.assertEqual(agent.id, agent_id)


class TemplateEngineAgentPrepareTestCase(TestCase):
    def setUp(self):
        self.renderer = Renderer()
        self.model_manager = MagicMock(spec=ModelManager)
        self.engine_uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="m",
            params={},
        )
        self.agent = TemplateEngineAgent(
            model=MagicMock(),
            memory=MagicMock(spec=MemoryManager),
            tool=MagicMock(spec=ToolManager),
            event_manager=MagicMock(spec=EventManager),
            model_manager=self.model_manager,
            renderer=self.renderer,
            engine_uri=self.engine_uri,
            name="Bob",
        )

    def test_prepare_call_no_template_vars(self):
        spec = Specification(
            role="assistant",
            goal=Goal(task="do", instructions=["instr"]),
            rules=["rule"],
        )
        context = ModelCallContext(specification=spec, input="hi")
        result = self.agent._prepare_call(context)
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
        context = ModelCallContext(specification=spec, input="hi")
        result = self.agent._prepare_call(context)
        expected_prompt = self.renderer(
            "agent.md",
            name="Bob",
            roles=[b"role run"],
            task=b"do run",
            instructions=[b"inst run"],
            rules=[b"rule run"],
        )
        self.assertEqual(result["system_prompt"], expected_prompt)

    def test_prepare_call_with_settings_template_vars(self):
        spec = Specification(
            role=Role(persona=["role {{verb}}"]),
            goal=Goal(task="do {{verb}}", instructions=["inst {{verb}}"]),
            rules=["rule {{verb}}"],
            settings=GenerationSettings(template_vars={"verb": "run"}),
        )
        context = ModelCallContext(specification=spec, input="hi")
        result = self.agent._prepare_call(context)
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
        context = ModelCallContext(specification=spec, input="hi")
        result = self.agent._prepare_call(context)
        expected_prompt = self.renderer(
            "agent.md",
            name="Bob",
            roles=["assistant"],
            task=None,
            instructions=None,
            rules=[],
        )
        self.assertEqual(result["system_prompt"], expected_prompt)

    def test_prepare_call_system_prompt(self):
        spec = Specification(role=None, goal=None, system_prompt="sys")
        context = ModelCallContext(specification=spec, input="hi")
        result = self.agent._prepare_call(context)
        self.assertEqual(result["system_prompt"], "sys")

    def test_prepare_call_system_prompt_with_developer_prompt(self) -> None:
        spec = Specification(
            role=None,
            goal=None,
            system_prompt="sys",
            developer_prompt="dev",
        )

        context = ModelCallContext(specification=spec, input="hi")
        result = self.agent._prepare_call(context)

        self.assertEqual(result["system_prompt"], "sys")
        self.assertEqual(result["developer_prompt"], "dev")

    def test_prepare_call_template_developer_prompt(self) -> None:
        spec = Specification(
            role="assistant",
            goal=Goal(task="task", instructions=["inst"]),
            developer_prompt="dev",
            rules=[],
        )

        context = ModelCallContext(specification=spec, input="hi")
        result = self.agent._prepare_call(context)

        self.assertEqual(result["developer_prompt"], "dev")


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

        engine_uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="m",
            params={},
        )
        agent = TemplateEngineAgent(
            model=model,
            memory=memory,
            tool=tool,
            event_manager=event_manager,
            model_manager=MagicMock(spec=ModelManager),
            renderer=renderer,
            engine_uri=engine_uri,
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

        context = ModelCallContext(specification=spec, input="hello")
        result = await agent(context)

        agent._run.assert_awaited_once()
        self.assertIs(agent._run.await_args.args[0], context)
        self.assertEqual(result, "out")
        self.assertEqual(
            agent._run.await_args.kwargs["system_prompt"],
            expected_prompt,
        )
        self.assertEqual(len(event_manager.trigger.await_args_list), 4)
        event_types = [
            c.args[0].type for c in event_manager.trigger.await_args_list[:3]
        ]
        self.assertEqual(
            event_types,
            [
                EventType.ENGINE_AGENT_CALL_BEFORE,
                EventType.CALL_PREPARE_BEFORE,
                EventType.CALL_PREPARE_AFTER,
            ],
        )
        self.assertEqual(
            event_manager.trigger.await_args_list[3].args[0].type,
            EventType.ENGINE_AGENT_CALL_AFTER,
        )


class FakeMemory:
    def __init__(self) -> None:
        self.has_permanent_message = True
        self.has_recent_message = False
        self.messages: list[EngineMessage] = []

    async def append_message(self, message: EngineMessage) -> None:
        self.messages.append(message)


class TemplateEngineAgentSyncMessagesTestCase(IsolatedAsyncioTestCase):
    async def test_sync_messages_stores_previous_output(self) -> None:
        renderer = Renderer()
        memory = FakeMemory()
        tool = MagicMock(spec=ToolManager)
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        model = MagicMock()
        model.model_id = "m"
        model.model_type = "t"
        engine_uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="m",
            params={},
        )
        agent = TemplateEngineAgent(
            model=model,
            memory=memory,
            tool=tool,
            event_manager=event_manager,
            model_manager=MagicMock(spec=ModelManager),
            renderer=renderer,
            engine_uri=engine_uri,
        )
        agent._last_output = TextGenerationResponse(
            lambda: "prev", logger=getLogger(), use_async_generator=False
        )

        await agent.sync_messages()

        self.assertEqual(len(memory.messages), 1)
        stored = memory.messages[0]
        self.assertEqual(stored.agent_id, agent.id)
        self.assertEqual(stored.model_id, model.model_id)
        self.assertEqual(stored.message.role, MessageRole.ASSISTANT)
        self.assertEqual(stored.message.content, "prev")
        event_types = [
            c.args[0].type for c in event_manager.trigger.await_args_list
        ]
        self.assertEqual(
            event_types,
            [EventType.MEMORY_APPEND_BEFORE, EventType.MEMORY_APPEND_AFTER],
        )
