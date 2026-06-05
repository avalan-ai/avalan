from dataclasses import dataclass
from logging import Logger, getLogger
from typing import Annotated
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock, patch

from avalan.agent import InputType, OutputType, Role
from avalan.agent.orchestrator.orchestrators.json import (
    JsonOrchestrator,
    JsonOrchestratorOutput,
    JsonSpecification,
    _usage_responses_from,
)
from avalan.agent.renderer import TemplateEngineAgent
from avalan.entities import (
    EngineUri,
    Message,
    MessageRole,
    TransformerEngineSettings,
)
from avalan.event import EventType
from avalan.event.manager import EventManager
from avalan.memory.manager import MemoryManager
from avalan.model import TextGenerationResponse
from avalan.model.manager import ModelManager
from avalan.task.usage import (
    UsageSource,
    usage_observations_from_response,
)
from avalan.tool.manager import ToolManager


@dataclass
class ExampleOutput:
    value: Annotated[str, "desc"]
    count: Annotated[int, "desc2"]


class ProviderUsageOutput:
    def __init__(self, usage: object) -> None:
        self.usage = usage

    def __call__(self, *args: object, **kwargs: object) -> str:
        return '{"value": "ok"}'


class CallableUsageResponses:
    def __init__(self, responses: list[object]) -> None:
        self._responses = responses

    def usage_responses(self) -> list[object]:
        return self._responses


class ListUsageResponses:
    def __init__(self, responses: list[object]) -> None:
        self.usage_responses = responses


class InvalidUsageResponses:
    usage_responses = "private invalid shape"


class JsonUsageResponseHelperTestCase(TestCase):
    def test_usage_responses_from_accepts_callable_list(self) -> None:
        first = object()
        second = object()

        self.assertEqual(
            _usage_responses_from(CallableUsageResponses([first, second])),
            (first, second),
        )

    def test_usage_responses_from_accepts_list_property(self) -> None:
        first = object()
        second = object()

        self.assertEqual(
            _usage_responses_from(ListUsageResponses([first, second])),
            (first, second),
        )

    def test_usage_responses_from_ignores_invalid_shape(self) -> None:
        self.assertEqual(_usage_responses_from(InvalidUsageResponses()), ())


class JsonOrchestratorInitTestCase(TestCase):
    def test_initialization(self):
        engine_uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="m_json",
            params={},
        )
        logger = MagicMock(spec=Logger)
        model_manager = MagicMock(spec=ModelManager)
        memory = MagicMock(spec=MemoryManager)
        tool = MagicMock(spec=ToolManager)
        event_manager = MagicMock(spec=EventManager)
        settings = TransformerEngineSettings()

        orch = JsonOrchestrator(
            engine_uri,
            logger,
            model_manager,
            memory,
            tool,
            event_manager,
            ExampleOutput,
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
        )

        self.assertEqual(len(orch.operations), 1)
        op = orch.operations[0]
        self.assertIs(op.environment.engine_uri, engine_uri)
        self.assertIs(op.environment.settings, settings)
        self.assertIsInstance(op.specification, JsonSpecification)
        self.assertEqual(op.specification.role, Role(persona=["assistant"]))
        self.assertEqual(op.specification.instructions, "provider")
        self.assertEqual(op.specification.goal.task, "do")
        self.assertEqual(
            op.specification.goal.goal_instructions, ["something"]
        )
        self.assertEqual(op.specification.rules, ["a", "b"])
        self.assertEqual(op.specification.template_id, "tmpl")
        self.assertEqual(op.specification.template_vars, {"y": 2})
        self.assertEqual(op.specification.output_type, OutputType.JSON)
        self.assertEqual(op.specification.input_type, InputType.TEXT)

    def test_initialization_with_system(self):
        engine_uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="m_json",
            params={},
        )
        logger = MagicMock(spec=Logger)
        model_manager = MagicMock(spec=ModelManager)
        memory = MagicMock(spec=MemoryManager)
        tool = MagicMock(spec=ToolManager)
        event_manager = MagicMock(spec=EventManager)
        settings = TransformerEngineSettings()

        orch = JsonOrchestrator(
            engine_uri,
            logger,
            model_manager,
            memory,
            tool,
            event_manager,
            ExampleOutput,
            name="Agent",
            instructions="provider",
            system="sys",
            settings=settings,
        )

        op = orch.operations[0]
        self.assertEqual(op.specification.instructions, "provider")
        self.assertEqual(op.specification.system_prompt, "sys")
        self.assertIsNone(op.specification.goal)

    def test_initialization_with_developer_only(self):
        engine_uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="m_json",
            params={},
        )
        logger = MagicMock(spec=Logger)
        model_manager = MagicMock(spec=ModelManager)
        memory = MagicMock(spec=MemoryManager)
        tool = MagicMock(spec=ToolManager)
        event_manager = MagicMock(spec=EventManager)
        settings = TransformerEngineSettings()

        orch = JsonOrchestrator(
            engine_uri,
            logger,
            model_manager,
            memory,
            tool,
            event_manager,
            ExampleOutput,
            name="Agent",
            developer="dev",
            settings=settings,
        )

        op = orch.operations[0]
        self.assertEqual(op.specification.developer_prompt, "dev")
        self.assertIsNone(op.specification.system_prompt)
        self.assertIsNone(op.specification.goal)

    def test_initialization_with_system_and_developer_only(self):
        engine_uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="m_json",
            params={},
        )
        logger = MagicMock(spec=Logger)
        model_manager = MagicMock(spec=ModelManager)
        memory = MagicMock(spec=MemoryManager)
        tool = MagicMock(spec=ToolManager)
        event_manager = MagicMock(spec=EventManager)
        settings = TransformerEngineSettings()

        orch = JsonOrchestrator(
            engine_uri,
            logger,
            model_manager,
            memory,
            tool,
            event_manager,
            ExampleOutput,
            name="Agent",
            system="sys",
            developer="dev",
            settings=settings,
        )

        op = orch.operations[0]
        self.assertEqual(op.specification.system_prompt, "sys")
        self.assertEqual(op.specification.developer_prompt, "dev")
        self.assertIsNone(op.specification.goal)


class JsonOrchestratorExecutionTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        super().setUp()
        self.addCleanup(patch.stopall)

    @patch("avalan.agent.orchestrator.TemplateEngineAgent")
    async def test_json_return(self, Agent):
        engine_uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="m_json",
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
        model_manager.load_engine.return_value = engine

        def output_fn(*args, **kwargs):
            return '{"value": "ok"}'

        response = TextGenerationResponse(
            output_fn, logger=getLogger(), use_async_generator=False
        )

        agent_mock = AsyncMock(spec=TemplateEngineAgent)
        agent_mock.engine = engine
        agent_mock.return_value = response

        Agent.return_value = agent_mock

        async with JsonOrchestrator(
            engine_uri,
            logger,
            model_manager,
            memory,
            tool,
            event_manager,
            ExampleOutput,
            name="Agent",
            role="assistant",
            task="do",
            goal_instructions="something",
            rules=None,
            settings=settings,
        ) as orch:
            result = await orch("hi")

        agent_mock.assert_awaited_once()
        context = agent_mock.await_args.args[0]
        self.assertIsInstance(context.input, Message)
        self.assertEqual(context.input.content, "hi")
        self.assertEqual(context.input.role, MessageRole.USER)
        self.assertEqual(
            context.specification.role, Role(persona=["assistant"])
        )

        self.assertEqual(result, '{"value": "ok"}')
        self.assertTrue(
            any(
                c.args[0].type == EventType.STREAM_END
                for c in event_manager.trigger.await_args_list
            )
        )
        memory.__exit__.assert_called_once()

    @patch("avalan.agent.orchestrator.TemplateEngineAgent")
    async def test_json_return_preserves_exact_usage(self, Agent) -> None:
        engine_uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="m_json",
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
        usage = {
            "input_tokens": 4,
            "cached_input_tokens": 1,
            "cache_creation_input_tokens": 2,
            "output_tokens": 3,
            "reasoning_tokens": 1,
            "total_tokens": 7,
            "provider_family": "openai",
            "raw_response_id": "private-response-id",
        }

        engine = MagicMock()
        engine.__enter__.return_value = engine
        engine.__exit__.return_value = False
        engine.model_id = "m"
        model_manager.load_engine.return_value = engine

        response = TextGenerationResponse(
            ProviderUsageOutput(usage),
            logger=getLogger(),
            use_async_generator=False,
        )
        agent_mock = AsyncMock(spec=TemplateEngineAgent)
        agent_mock.engine = engine
        agent_mock.return_value = response
        Agent.return_value = agent_mock

        async with JsonOrchestrator(
            engine_uri,
            logger,
            model_manager,
            memory,
            tool,
            event_manager,
            ExampleOutput,
            name="Agent",
            role="assistant",
            settings=settings,
        ) as orch:
            result = await orch("hi")

        observations = usage_observations_from_response(result)

        self.assertEqual(result, '{"value": "ok"}')
        self.assertIsInstance(result, JsonOrchestratorOutput)
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].source, UsageSource.EXACT)
        self.assertEqual(observations[0].totals.input_tokens, 4)
        self.assertEqual(observations[0].totals.cached_input_tokens, 1)
        self.assertEqual(
            observations[0].totals.cache_creation_input_tokens,
            2,
        )
        self.assertEqual(observations[0].totals.output_tokens, 3)
        self.assertEqual(observations[0].totals.reasoning_tokens, 1)
        self.assertEqual(observations[0].totals.total_tokens, 7)
        self.assertEqual(
            observations[0].metadata,
            {"provider_family": "openai"},
        )

    @patch("avalan.agent.orchestrator.TemplateEngineAgent")
    async def test_json_return_drops_malformed_usage(self, Agent) -> None:
        engine_uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="m_json",
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
        usage = {
            "input_tokens": "private prompt",
            "output_tokens": -1,
            "total_tokens": True,
            "provider_family": "private-provider",
        }

        engine = MagicMock()
        engine.__enter__.return_value = engine
        engine.__exit__.return_value = False
        engine.model_id = "m"
        model_manager.load_engine.return_value = engine

        response = TextGenerationResponse(
            ProviderUsageOutput(usage),
            logger=getLogger(),
            use_async_generator=False,
        )
        agent_mock = AsyncMock(spec=TemplateEngineAgent)
        agent_mock.engine = engine
        agent_mock.return_value = response
        Agent.return_value = agent_mock

        async with JsonOrchestrator(
            engine_uri,
            logger,
            model_manager,
            memory,
            tool,
            event_manager,
            ExampleOutput,
            name="Agent",
            role="assistant",
            settings=settings,
        ) as orch:
            result = await orch("hi")

        self.assertEqual(result, '{"value": "ok"}')
        self.assertEqual(usage_observations_from_response(result), ())
