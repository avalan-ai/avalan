from dataclasses import asdict
from json import dumps
from logging import Logger, getLogger
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from avalan.agent import Goal, InputType, OutputType, Specification
from avalan.agent.orchestrator.orchestrators.default import DefaultOrchestrator
from avalan.agent.orchestrator.response.orchestrator_response import (
    OrchestratorResponse,
)
from avalan.agent.renderer import Renderer, TemplateEngineAgent
from avalan.entities import (
    EngineUri,
    Message,
    MessageRole,
    Modality,
    TransformerEngineSettings,
)
from avalan.event import EventPayloadKind, EventType
from avalan.event.manager import EventManager
from avalan.memory.manager import MemoryManager
from avalan.model import TextGenerationResponse
from avalan.model.capability import ProviderCapabilitySupport
from avalan.model.manager import ModelManager
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemKind,
    StreamTerminalOutcome,
)
from avalan.tool.manager import ToolManager


def _canonical_response(*text_deltas: str) -> TextGenerationResponse:
    async def output_gen():
        sequence = 0
        yield CanonicalStreamItem(
            stream_session_id="default-test-stream",
            run_id="default-test-run",
            turn_id="default-test-turn",
            sequence=sequence,
            kind=StreamItemKind.STREAM_STARTED,
            channel=StreamChannel.CONTROL,
        )
        sequence += 1
        for text_delta in text_deltas:
            yield CanonicalStreamItem(
                stream_session_id="default-test-stream",
                run_id="default-test-run",
                turn_id="default-test-turn",
                sequence=sequence,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta=text_delta,
            )
            sequence += 1
        yield CanonicalStreamItem(
            stream_session_id="default-test-stream",
            run_id="default-test-run",
            turn_id="default-test-turn",
            sequence=sequence,
            kind=StreamItemKind.ANSWER_DONE,
            channel=StreamChannel.ANSWER,
        )
        sequence += 1
        yield CanonicalStreamItem(
            stream_session_id="default-test-stream",
            run_id="default-test-run",
            turn_id="default-test-turn",
            sequence=sequence,
            kind=StreamItemKind.STREAM_COMPLETED,
            channel=StreamChannel.CONTROL,
            usage={},
            terminal_outcome=StreamTerminalOutcome.COMPLETED,
        )

    return TextGenerationResponse(
        lambda **_: output_gen(),
        logger=getLogger(),
        use_async_generator=True,
    )


def _single_model_input_message(input_value: object) -> Message:
    """Return the sole normalized message passed to the model."""
    assert isinstance(input_value, list)
    assert len(input_value) == 1
    message = input_value[0]
    assert isinstance(message, Message)
    return message


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
            instructions="provider",
            goal_instructions="something",
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
        self.assertEqual(op.specification.instructions, "provider")
        self.assertEqual(op.specification.goal.task, "do")
        self.assertEqual(
            op.specification.goal.goal_instructions, ["something"]
        )
        self.assertEqual(op.specification.rules, ["a", "b"])
        self.assertEqual(op.specification.template_id, "tmpl")
        self.assertEqual(op.specification.template_vars, {"y": 2})

    def test_initialization_with_system(self):
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

        orch = DefaultOrchestrator(
            engine_uri,
            logger,
            model_manager,
            memory,
            tool,
            event_manager,
            name="Agent",
            role=None,
            task=None,
            instructions="provider",
            rules=None,
            system="sys",
        )

        op = orch.operations[0]
        self.assertEqual(op.specification.instructions, "provider")
        self.assertEqual(op.specification.system_prompt, "sys")
        self.assertIsNone(op.specification.role)
        self.assertIsNone(op.specification.goal)

    def test_initialization_with_developer(self):
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

        orch = DefaultOrchestrator(
            engine_uri,
            logger,
            model_manager,
            memory,
            tool,
            event_manager,
            name="Agent",
            role="ignored",
            task="ignored",
            goal_instructions="ignored",
            rules=None,
            developer="dev",
        )

        op = orch.operations[0]
        self.assertEqual(op.specification.developer_prompt, "dev")
        self.assertIsNone(op.specification.system_prompt)
        self.assertIsNone(op.specification.role)
        self.assertIsNone(op.specification.goal)

    def test_initialization_with_system_and_developer_only(self):
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

        orch = DefaultOrchestrator(
            engine_uri,
            logger,
            model_manager,
            memory,
            tool,
            event_manager,
            name="Agent",
            role=None,
            task=None,
            instructions=None,
            rules=None,
            system="sys",
            developer="dev",
        )

        op = orch.operations[0]
        self.assertEqual(op.specification.system_prompt, "sys")
        self.assertEqual(op.specification.developer_prompt, "dev")
        self.assertIsNone(op.specification.role)
        self.assertIsNone(op.specification.goal)


class DefaultOrchestratorTestCase(IsolatedAsyncioTestCase):
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
        tool.export_model_capability_seed.return_value = (
            ToolManager.create_instance().export_model_capability_seed()
        )
        tool.is_empty = True
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        event_manager.should_emit.return_value = True
        event_manager.enrich_token_ids = True
        settings = TransformerEngineSettings()

        engine = MagicMock()
        engine.__enter__.return_value = engine
        engine.__exit__.return_value = False
        engine.model_id = "m"
        engine.tokenizer = MagicMock()
        engine.tokenizer.eos_token = None
        engine.tokenizer.encode.side_effect = [[1], [2]]
        engine.provider_capability_support = ProviderCapabilitySupport()
        model_manager.load_engine.return_value = engine

        response = _canonical_response("a", "b")

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
            goal_instructions="something",
            rules=None,
            settings=settings,
        ) as orch:
            model_manager.load_engine.assert_called_once_with(
                engine_uri,
                settings,
                Modality.TEXT_GENERATION,
            )
            Agent.assert_called_once()
            self.assertIs(orch.engine_agent, agent_mock)
            self.assertEqual(orch.model_ids, {"m"})

            result = await orch("hi", use_async_generator=True)

            tokens = []
            async for t in result:
                tokens.append(t)

        agent_mock.assert_awaited_once()
        context = agent_mock.await_args.args[0]
        message = _single_model_input_message(context.input)
        self.assertEqual(message.content, "hi")
        self.assertEqual(message.role, MessageRole.USER)
        self.assertIsNone(message.name)
        self.assertIsNone(message.arguments)
        self.assertEqual(
            context.specification,
            Specification(
                role="assistant",
                goal=Goal(task="do", goal_instructions=["something"]),
                rules=None,
                input_type=InputType.TEXT,
                output_type=OutputType.TEXT,
                settings=None,
                template_id="agent.md",
                template_vars=None,
            ),
        )

        self.assertIsInstance(result, OrchestratorResponse)
        self.assertTrue(
            all(isinstance(token, CanonicalStreamItem) for token in tokens)
        )
        self.assertEqual(
            [
                token.text_delta
                for token in tokens
                if token.kind is StreamItemKind.ANSWER_DELTA
            ],
            ["a", "b"],
        )

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
        self.assertIs(
            token_events[0].observability.kind,
            EventPayloadKind.CANONICAL_STREAM,
        )
        self.assertEqual(
            token_events[0].payload,
            token_events[0].observability.data,
        )
        self.assertEqual(
            token_events[0].observability.data["kind"],
            "answer.delta",
        )
        summary = token_events[0].observability.data["summary"]
        self.assertIsInstance(summary, dict)
        self.assertEqual(summary["text_delta_length"], 1)
        self.assertEqual(summary["model_id"], "m")
        self.assertEqual(summary["step"], 0)
        self.assertEqual(summary["token_id"], 1)
        self.assertNotIn("token", token_events[0].payload)
        self.assertEqual(
            token_events[1].payload,
            token_events[1].observability.data,
        )
        self.assertEqual(
            token_events[1].observability.data["kind"],
            "answer.delta",
        )
        summary = token_events[1].observability.data["summary"]
        self.assertIsInstance(summary, dict)
        self.assertEqual(summary["text_delta_length"], 1)
        self.assertEqual(summary["model_id"], "m")
        self.assertEqual(summary["step"], 1)
        self.assertEqual(summary["token_id"], 2)

        memory.__exit__.assert_called_once()

    async def test_user_string_rendering(self):
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
        tool.export_model_capability_seed.return_value = (
            ToolManager.create_instance().export_model_capability_seed()
        )
        tool.is_empty = True
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        settings = TransformerEngineSettings()

        engine = MagicMock()
        engine.__enter__.return_value = engine
        engine.__exit__.return_value = False
        engine.model_id = "m"
        engine.tokenizer = MagicMock()
        engine.tokenizer.eos_token = None
        engine.provider_capability_support = ProviderCapabilitySupport()
        model_manager.load_engine.return_value = engine

        agent_mock = AsyncMock()
        agent_mock.acknowledge_provider_handoff = MagicMock()
        agent_mock.engine = engine
        agent_mock.return_value = MagicMock(spec=TextGenerationResponse)

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
            goal_instructions="something",
            rules=None,
            user="hello {{input}} {{name}}",
            template_vars={"name": "Bob"},
            settings=settings,
        )
        environment_hash = dumps(asdict(orch.operations[0].environment))
        orch._engine_agents = {environment_hash: agent_mock}
        await orch("world")

        context = agent_mock.await_args.args[0]
        message = _single_model_input_message(context.input)
        self.assertEqual(message.content, "hello world Bob")

    async def test_user_template_rendering(self):
        with TemporaryDirectory() as tmp:
            template_path = f"{tmp}/user.md"
            with open(template_path, "w", encoding="utf-8") as fh:
                fh.write("hi {{input}} {{name}}")

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
            tool.export_model_capability_seed.return_value = (
                ToolManager.create_instance().export_model_capability_seed()
            )
            tool.is_empty = True
            event_manager = MagicMock(spec=EventManager)
            event_manager.trigger = AsyncMock()
            settings = TransformerEngineSettings()

            engine = MagicMock()
            engine.__enter__.return_value = engine
            engine.__exit__.return_value = False
            engine.model_id = "m"
            engine.tokenizer = MagicMock()
            engine.tokenizer.eos_token = None
            engine.provider_capability_support = ProviderCapabilitySupport()
            model_manager.load_engine.return_value = engine

            agent_mock = AsyncMock()
            agent_mock.acknowledge_provider_handoff = MagicMock()
            agent_mock.engine = engine
            agent_mock.return_value = MagicMock(spec=TextGenerationResponse)

            renderer = Renderer(templates_path=tmp)

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
                goal_instructions="something",
                rules=None,
                user_template="user.md",
                template_vars={"name": "Ann"},
                settings=settings,
            )
            orch._renderer = renderer
            environment_hash = dumps(asdict(orch.operations[0].environment))
            orch._engine_agents = {environment_hash: agent_mock}
            await orch("earth")

        context = agent_mock.await_args.args[0]
        message = _single_model_input_message(context.input)
        self.assertEqual(message.content, "hi earth Ann")

    async def test_user_prefix_rendering(self):
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
        memory.permanent_message = None
        tool = MagicMock(spec=ToolManager)
        tool.export_model_capability_seed.return_value = (
            ToolManager.create_instance().export_model_capability_seed()
        )
        tool.is_empty = True
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        settings = TransformerEngineSettings()

        engine = MagicMock()
        engine.model_id = "m"
        engine.tokenizer = MagicMock()
        engine.tokenizer.eos_token = None
        engine.provider_capability_support = ProviderCapabilitySupport()

        agent_mock = AsyncMock()
        agent_mock.acknowledge_provider_handoff = MagicMock()
        agent_mock.engine = engine
        agent_mock.return_value = MagicMock(spec=TextGenerationResponse)

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
            goal_instructions="something",
            rules=None,
            user="Answer briefly.",
            settings=settings,
        )
        environment_hash = dumps(asdict(orch.operations[0].environment))
        orch._engine_agents = {environment_hash: agent_mock}

        await orch("world")

        context = agent_mock.await_args.args[0]
        message = _single_model_input_message(context.input)
        self.assertEqual(message.content, "Answer briefly.\n\nworld")
