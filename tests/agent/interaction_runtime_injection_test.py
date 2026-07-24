from contextlib import AsyncExitStack
from dataclasses import asdict
from json import dumps
from logging import Logger
from tempfile import NamedTemporaryFile
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from avalan.agent import AgentOperation, EngineEnvironment, Specification
from avalan.agent.execution import (
    AgentExecutionStatus,
    AttachedInteractionRuntime,
    InteractionRuntime,
)
from avalan.agent.loader import OrchestratorLoader
from avalan.agent.orchestrator import Orchestrator
from avalan.agent.orchestrator.orchestrators.reasoning.cot import (
    ReasoningOrchestrator,
)
from avalan.entities import (
    EngineUri,
    OrchestratorSettings,
    TransformerEngineSettings,
)
from avalan.event.manager import EventManager
from avalan.interaction.broker import (
    InteractionBroker,
    InteractionBrokerRequest,
    InteractionBrokerResult,
    InteractionRequestResult,
)
from avalan.interaction.entities import PrincipalScope
from avalan.interaction.handler import (
    InputHandlerContext,
    InputHandlerDetached,
    InputHandlerOutcome,
)
from avalan.interaction.policy import InteractionActor
from avalan.interaction.store import TerminalizeInteractionScopeCommand
from avalan.memory.manager import MemoryManager
from avalan.model.capability import (
    ProviderCapabilitySupport,
    TaskInputCapabilityAdvertisement,
)
from avalan.model.hubs.huggingface import HuggingfaceHub
from avalan.model.manager import ModelManager
from avalan.tool.manager import ToolManager


class _Broker:
    async def request(
        self,
        request: InteractionBrokerRequest,
    ) -> InteractionRequestResult:
        raise AssertionError(request)

    async def cancel_scope(
        self,
        command: TerminalizeInteractionScopeCommand,
    ) -> InteractionBrokerResult:
        raise AssertionError(command)


async def _handler(context: InputHandlerContext) -> InputHandlerOutcome:
    del context
    return InputHandlerDetached()


def _runtime() -> AttachedInteractionRuntime:
    return AttachedInteractionRuntime(
        broker=cast(InteractionBroker, _Broker()),
        actor=InteractionActor(principal=PrincipalScope()),
        handler=_handler,
    )


def _settings(
    *,
    orchestrator_type: str | None = None,
) -> OrchestratorSettings:
    return OrchestratorSettings(
        agent_id=uuid4(),
        orchestrator_type=orchestrator_type,
        agent_config={"role": "assistant"},
        uri="ai://local/model",
        engine_config={},
        tools=None,
        call_options=None,
        template_vars=None,
        memory_permanent_message=None,
        permanent_memory=None,
        memory_recent=False,
        sentence_model_id=OrchestratorLoader.DEFAULT_SENTENCE_MODEL_ID,
        sentence_model_engine_config=None,
        sentence_model_max_tokens=500,
        sentence_model_overlap_size=125,
        sentence_model_window_size=250,
        json_config=(
            {"value": {"type": "string", "description": "value"}}
            if orchestrator_type == "json"
            else None
        ),
        log_events=False,
    )


async def _constructor_kwargs(
    *,
    orchestrator_type: str | None,
    interaction_runtime: InteractionRuntime | None,
) -> dict[str, Any]:
    stack = AsyncExitStack()
    memory = MagicMock(spec=MemoryManager)
    tool = MagicMock(spec=ToolManager)
    tool.__aenter__ = AsyncMock(return_value=tool)
    model_manager = MagicMock(spec=ModelManager)
    model_manager.__enter__.return_value = model_manager
    model_manager.parse_uri.return_value = MagicMock()
    model_manager.get_engine_settings.return_value = MagicMock()
    constructor_name = (
        "JsonOrchestrator"
        if orchestrator_type == "json"
        else "DefaultOrchestrator"
    )

    with (
        patch(
            "avalan.agent.loader.MemoryManager.create_instance",
            new=AsyncMock(return_value=memory),
        ),
        patch(
            "avalan.agent.loader.ModelManager",
            return_value=model_manager,
        ),
        patch(
            f"avalan.agent.loader.{constructor_name}",
            return_value=MagicMock(),
        ) as constructor,
        patch(
            "avalan.agent.loader.ToolManager.create_instance",
            return_value=tool,
        ),
        patch("avalan.agent.loader.EventManager", return_value=MagicMock()),
        patch("avalan.agent.loader.HAS_GRAPH_DEPENDENCIES", False),
        patch("avalan.agent.loader.HAS_CODE_DEPENDENCIES", False),
        patch("avalan.agent.loader.HAS_BROWSER_DEPENDENCIES", False),
        patch("avalan.agent.loader.MathToolSet", return_value=MagicMock()),
        patch("avalan.agent.loader.MemoryToolSet", return_value=MagicMock()),
    ):
        loader = OrchestratorLoader(
            hub=MagicMock(spec=HuggingfaceHub),
            logger=MagicMock(spec=Logger),
            participant_id=uuid4(),
            stack=stack,
        )
        await loader.from_settings(
            _settings(orchestrator_type=orchestrator_type),
            interaction_runtime=interaction_runtime,
        )

    await stack.aclose()
    return dict(constructor.call_args.kwargs)


class _ResponseFixture(str):
    execution: MagicMock
    ownership_cleanup_complete: bool
    sync_messages: AsyncMock

    def __new__(cls) -> "_ResponseFixture":
        return super().__new__(cls, "response")

    def __init__(self) -> None:
        self.execution = MagicMock(status=AgentExecutionStatus.COMPLETED)
        self.ownership_cleanup_complete = True
        self.aclose = AsyncMock()
        self.sync_messages = AsyncMock()


def _orchestrator(
    interaction_runtime: InteractionRuntime | None = None,
) -> tuple[Orchestrator, AsyncMock]:
    environment = EngineEnvironment(
        engine_uri=EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="model",
            params={},
        ),
        settings=TransformerEngineSettings(),
    )
    operation = AgentOperation(
        specification=Specification(),
        environment=environment,
    )
    memory = MagicMock(spec=MemoryManager)
    memory.participant_id = uuid4()
    memory.permanent_message = None
    tool = MagicMock(spec=ToolManager)
    tool.export_model_capability_seed.return_value = (
        ToolManager.create_instance().export_model_capability_seed()
    )
    event_manager = MagicMock(spec=EventManager)
    event_manager.trigger = AsyncMock()
    orchestrator = Orchestrator(
        MagicMock(spec=Logger),
        MagicMock(spec=ModelManager),
        memory,
        tool,
        event_manager,
        operation,
        interaction_runtime=interaction_runtime,
    )
    engine_agent = AsyncMock()
    engine_agent.acknowledge_provider_handoff = MagicMock()
    engine_agent.drain_pending_provider_cleanups = AsyncMock(return_value=())
    engine_agent.engine = MagicMock(
        model_id="model",
        provider_capability_support=ProviderCapabilitySupport(
            structured_invocation=True,
            stable_call_ids=True,
            correlated_results=True,
        ),
        tokenizer=MagicMock(eos_token="<eos>"),
    )
    environment_hash = dumps(asdict(environment))
    orchestrator._engine_agents[environment_hash] = engine_agent
    orchestrator._last_engine_agent = engine_agent
    return orchestrator, engine_agent


class OrchestratorLoaderInteractionRuntimeTestCase(IsolatedAsyncioTestCase):
    async def test_from_file_forwards_only_explicit_host_runtime(self) -> None:
        runtime = _runtime()
        stack = AsyncExitStack()
        loader = OrchestratorLoader(
            hub=MagicMock(spec=HuggingfaceHub),
            logger=MagicMock(spec=Logger),
            participant_id=uuid4(),
            stack=stack,
        )

        with NamedTemporaryFile("w+", suffix=".toml") as agent_file:
            agent_file.write(
                '[agent]\nrole = "assistant"\n'
                '[engine]\nuri = "ai://local/model"\n'
            )
            agent_file.flush()
            with patch.object(
                loader,
                "from_settings",
                new=AsyncMock(return_value=MagicMock()),
            ) as from_settings:
                await loader.from_file(
                    agent_file.name,
                    agent_id=uuid4(),
                    interaction_runtime=runtime,
                )
                assert from_settings.await_args is not None
                self.assertIs(
                    from_settings.await_args.kwargs["interaction_runtime"],
                    runtime,
                )

                await loader.from_file(
                    agent_file.name,
                    agent_id=uuid4(),
                )
                self.assertNotIn(
                    "interaction_runtime",
                    from_settings.await_args_list[1].kwargs,
                )

        await stack.aclose()

    async def test_from_file_forwards_runtime_to_trusted_envelope(
        self,
    ) -> None:
        runtime = _runtime()
        image = "ghcr.io/example/agent@sha256:" + "4" * 64

        class EnvelopeLoader:
            trusted_runtime_envelope_runner = True

            def __init__(self) -> None:
                self.kwargs: dict[str, object] | None = None

            async def load_agent_runtime_envelope(
                self,
                plan: object,
                **kwargs: object,
            ) -> MagicMock:
                del plan
                self.kwargs = kwargs
                return MagicMock()

        envelope_loader = EnvelopeLoader()
        stack = AsyncExitStack()
        loader = OrchestratorLoader(
            hub=MagicMock(spec=HuggingfaceHub),
            logger=MagicMock(spec=Logger),
            participant_id=uuid4(),
            stack=stack,
            runtime_envelope_loader=envelope_loader,
        )

        with NamedTemporaryFile("w+", suffix=".toml") as agent_file:
            agent_file.write(
                '[agent]\nrole = "assistant"\n'
                '[engine]\nuri = "ai://local/model"\n'
                "[tool.container]\n"
                'backend = "docker"\n'
                'default_profile = "runtime"\n'
                "[tool.container.profiles.runtime]\n"
                f'image = "{image}"\n'
                "[runtime.container]\n"
                'profile = "runtime"\n'
            )
            agent_file.flush()

            await loader.from_file(
                agent_file.name,
                agent_id=uuid4(),
                interaction_runtime=runtime,
            )

        assert envelope_loader.kwargs is not None
        self.assertIs(envelope_loader.kwargs["interaction_runtime"], runtime)
        await stack.aclose()

    async def test_from_settings_injects_runtime_into_both_constructors(
        self,
    ) -> None:
        for orchestrator_type in (None, "json"):
            with self.subTest(orchestrator_type=orchestrator_type):
                runtime = _runtime()
                kwargs = await _constructor_kwargs(
                    orchestrator_type=orchestrator_type,
                    interaction_runtime=runtime,
                )
                self.assertIs(kwargs["interaction_runtime"], runtime)

    async def test_from_settings_keeps_runtime_off_by_default(self) -> None:
        kwargs = await _constructor_kwargs(
            orchestrator_type=None,
            interaction_runtime=None,
        )

        self.assertIsNone(kwargs["interaction_runtime"])

    async def test_loader_rejects_untyped_runtime(self) -> None:
        loader = OrchestratorLoader(
            hub=MagicMock(spec=HuggingfaceHub),
            logger=MagicMock(spec=Logger),
            participant_id=uuid4(),
            stack=AsyncExitStack(),
        )
        invalid = cast(InteractionRuntime, object())

        with self.assertRaisesRegex(
            TypeError,
            "interaction_runtime must be an interaction runtime or None",
        ):
            await loader.from_file(
                "missing.toml",
                agent_id=uuid4(),
                interaction_runtime=invalid,
            )
        with self.assertRaisesRegex(
            TypeError,
            "interaction_runtime must be an interaction runtime or None",
        ):
            await loader.from_settings(
                _settings(),
                interaction_runtime=invalid,
            )


class OrchestratorInteractionRuntimeTestCase(IsolatedAsyncioTestCase):
    async def test_host_runtime_reaches_fresh_run_creation(self) -> None:
        runtime = _runtime()
        orchestrator, engine_agent = _orchestrator(runtime)

        with patch(
            "avalan.agent.orchestrator.OrchestratorResponse",
            return_value=_ResponseFixture(),
        ):
            await orchestrator("hello")

        assert engine_agent.await_args is not None
        context = engine_agent.await_args.args[0]
        self.assertIs(context.execution.interaction_runtime, runtime)
        self.assertIsNotNone(context.execution.interaction_broker)
        self.assertIs(
            context.capability.task_input_advertisement,
            TaskInputCapabilityAdvertisement.ATTACHED,
        )

    async def test_per_run_runtime_overrides_host_scope_without_mutation(
        self,
    ) -> None:
        host_runtime = _runtime()
        run_runtime = _runtime()
        orchestrator, engine_agent = _orchestrator(host_runtime)

        with patch(
            "avalan.agent.orchestrator.OrchestratorResponse",
            return_value=_ResponseFixture(),
        ):
            await orchestrator(
                "hello",
                interaction_runtime=run_runtime,
            )

        assert engine_agent.await_args is not None
        context = engine_agent.await_args.args[0]
        self.assertIs(context.execution.interaction_runtime, run_runtime)
        self.assertIs(orchestrator._interaction_runtime, host_runtime)

    async def test_runtime_capability_remains_off_when_omitted(self) -> None:
        orchestrator, engine_agent = _orchestrator()

        with patch(
            "avalan.agent.orchestrator.OrchestratorResponse",
            return_value=_ResponseFixture(),
        ):
            await orchestrator("hello")

        assert engine_agent.await_args is not None
        context = engine_agent.await_args.args[0]
        self.assertIsNone(context.execution.interaction_runtime)
        self.assertIsNone(context.execution.interaction_broker)
        self.assertIs(
            context.capability.task_input_advertisement,
            TaskInputCapabilityAdvertisement.INCAPABLE,
        )

    async def test_run_rejects_untyped_runtime_before_side_effects(
        self,
    ) -> None:
        orchestrator, engine_agent = _orchestrator()

        with self.assertRaisesRegex(
            TypeError,
            "interaction_runtime must be an interaction runtime or None",
        ):
            await orchestrator(
                "hello",
                interaction_runtime=cast(InteractionRuntime, object()),
            )

        engine_agent.assert_not_awaited()
        event_trigger = cast(AsyncMock, orchestrator._event_manager.trigger)
        event_trigger.assert_not_awaited()


class OrchestratorConstructionInteractionRuntimeTestCase(TestCase):
    def test_constructor_rejects_untyped_runtime(self) -> None:
        invalid = cast(InteractionRuntime, object())

        with self.assertRaisesRegex(
            TypeError,
            "interaction_runtime must be an interaction runtime or None",
        ):
            _orchestrator(invalid)

    def test_reasoning_wrapper_preserves_trusted_host_runtime(self) -> None:
        runtime = _runtime()
        orchestrator, _engine_agent = _orchestrator(runtime)

        reasoning = ReasoningOrchestrator(orchestrator)

        self.assertIs(reasoning._interaction_runtime, runtime)
