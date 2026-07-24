"""Exercise fail-closed orchestrator execution-contract boundaries."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import Mock, patch

from avalan.agent import NoOperationAvailableException
from avalan.agent.execution import AgentExecution, AgentExecutionStatus
from avalan.agent.orchestrator import Orchestrator
from avalan.interaction import InputErrorCode, InputSnapshotError
from avalan.model.capability import (
    ModelCapabilityCatalog,
    ProviderCapabilitySupport,
    TaskInputCapabilityAdvertisement,
)
from avalan.tool.manager import ToolManager


@dataclass(frozen=True)
class _Environment:
    """Provide one hashable dataclass environment for operation lookup."""

    name: str = "environment"


def _orchestrator() -> Orchestrator:
    """Return a minimally initialized orchestrator contract owner."""
    orchestrator = object.__new__(Orchestrator)
    orchestrator._total_operations = 1
    orchestrator._operations = [SimpleNamespace(environment=_Environment())]
    orchestrator._engine_agents = {}
    orchestrator._pending_responses = {}
    orchestrator._execution_definition_locator = None
    orchestrator._tool = SimpleNamespace(
        export_model_capability_seed=Mock(
            return_value=(
                ToolManager.create_instance().export_model_capability_seed()
            )
        )
    )
    return orchestrator


def _engine_agent(
    *,
    register_codec: object | None = None,
) -> object:
    """Return one engine-agent contract double."""
    engine = SimpleNamespace(
        tokenizer=None,
        provider_capability_support=ProviderCapabilitySupport(),
    )
    if register_codec is not None:
        engine.register_continuation_snapshot_codec = register_codec
    return SimpleNamespace(engine=engine)


class OrchestratorContractCoverageTest(TestCase):
    """Exercise synchronous durable contract validation."""

    def test_definition_locator_rejects_invalid_rebinding_and_startup(
        self,
    ) -> None:
        orchestrator = _orchestrator()
        with self.assertRaises(ValueError):
            orchestrator.bind_execution_definition_locator("agent.toml")

        orchestrator._execution_definition_locator = "file://first.toml"
        with self.assertRaises(RuntimeError):
            orchestrator.bind_execution_definition_locator(
                "file://second.toml"
            )

        orchestrator._execution_definition_locator = None
        orchestrator._pending_responses[1] = cast(Any, object())
        with self.assertRaises(RuntimeError):
            orchestrator.bind_execution_definition_locator("file://agent.toml")

    def test_operation_and_continuation_contracts_require_loaded_codec(
        self,
    ) -> None:
        orchestrator = _orchestrator()
        for operation_index in (-1, True, 1):
            with (
                self.subTest(operation_index=operation_index),
                self.assertRaises(NoOperationAvailableException),
            ):
                orchestrator.engine_agent_for_operation(operation_index)
        with self.assertRaises(RuntimeError):
            orchestrator.engine_agent_for_operation(0)

        with self.assertRaises(RuntimeError):
            orchestrator.continuation_execution_contract(0)

        orchestrator._execution_definition_locator = "file://agent.toml"
        capability = SimpleNamespace(
            revision_binding=None,
            task_input_advertisement=(
                TaskInputCapabilityAdvertisement.INCAPABLE
            ),
        )
        with (
            patch.object(
                Orchestrator,
                "_execution_contract",
                return_value=(object(), capability),
            ),
            self.assertRaises(RuntimeError),
        ):
            orchestrator.continuation_execution_contract(0)

    def test_execution_contract_rejects_invalid_advertisement_and_locator(
        self,
    ) -> None:
        orchestrator = _orchestrator()
        with self.assertRaises(TypeError):
            orchestrator._execution_contract(
                0,
                cast(TaskInputCapabilityAdvertisement, object()),
            )

        with (
            patch.object(
                Orchestrator,
                "engine_agent_for_operation",
                return_value=_engine_agent(),
            ),
            self.assertRaises(RuntimeError),
        ):
            orchestrator._execution_contract(
                0,
                TaskInputCapabilityAdvertisement.DURABLE,
            )

    def test_codec_registration_contains_snapshot_and_type_failures(
        self,
    ) -> None:
        orchestrator = _orchestrator()
        orchestrator._execution_definition_locator = "file://agent.toml"
        definition = object()
        binding = object()
        snapshot_error = InputSnapshotError(
            InputErrorCode.SNAPSHOT_INVALID,
            "continuation_snapshot",
            "invalid snapshot",
        )

        with (
            patch.object(
                Orchestrator,
                "engine_agent_for_operation",
                return_value=_engine_agent(
                    register_codec=Mock(side_effect=snapshot_error)
                ),
            ),
            patch.object(
                Orchestrator,
                "_execution_definition",
                return_value=definition,
            ),
            patch.object(
                Orchestrator,
                "_continuation_revision_binding",
                return_value=binding,
            ),
        ):
            returned_definition, capability = orchestrator._execution_contract(
                0,
                TaskInputCapabilityAdvertisement.DURABLE,
            )
        self.assertIs(returned_definition, definition)
        self.assertIs(
            capability.task_input_advertisement,
            TaskInputCapabilityAdvertisement.INCAPABLE,
        )

        with (
            patch.object(
                Orchestrator,
                "engine_agent_for_operation",
                return_value=_engine_agent(
                    register_codec=Mock(return_value=object())
                ),
            ),
            patch.object(
                Orchestrator,
                "_execution_definition",
                return_value=definition,
            ),
            patch.object(
                Orchestrator,
                "_continuation_revision_binding",
                return_value=binding,
            ),
            self.assertRaises(TypeError),
        ):
            orchestrator._execution_contract(
                0,
                TaskInputCapabilityAdvertisement.DURABLE,
            )

    def test_codec_registration_cannot_change_execution_identity(self) -> None:
        orchestrator = _orchestrator()
        orchestrator._execution_definition_locator = "file://agent.toml"
        first_definition = object()
        second_definition = object()
        capability = SimpleNamespace(
            task_input_advertisement=TaskInputCapabilityAdvertisement.DURABLE
        )

        with (
            patch.object(
                Orchestrator,
                "engine_agent_for_operation",
                return_value=_engine_agent(
                    register_codec=Mock(return_value=None)
                ),
            ),
            patch.object(
                Orchestrator,
                "_execution_definition",
                side_effect=(first_definition, second_definition),
            ),
            patch.object(
                Orchestrator,
                "_continuation_revision_binding",
                return_value=object(),
            ),
            patch.object(
                ModelCapabilityCatalog,
                "create",
                return_value=capability,
            ),
            self.assertRaises(RuntimeError),
        ):
            orchestrator._execution_contract(
                0,
                TaskInputCapabilityAdvertisement.DURABLE,
            )

    def test_revision_binding_requires_versioned_provider_model(self) -> None:
        operation = SimpleNamespace(
            environment=SimpleNamespace(
                engine_uri=SimpleNamespace(
                    vendor="",
                    model_id="",
                )
            )
        )
        with self.assertRaises(InputSnapshotError):
            Orchestrator._continuation_revision_binding(
                cast(Any, operation),
                cast(Any, object()),
            )


class OrchestratorResumeContractCoverageTest(IsolatedAsyncioTestCase):
    """Exercise asynchronous resumed-execution contract guards."""

    async def test_resume_requires_resuming_execution_and_exact_contract(
        self,
    ) -> None:
        orchestrator = _orchestrator()
        incapable = SimpleNamespace(
            revision_binding=None,
            task_input_advertisement=(
                TaskInputCapabilityAdvertisement.INCAPABLE
            ),
        )
        with self.assertRaises(RuntimeError):
            await orchestrator.resume_agent_execution(
                cast(
                    AgentExecution,
                    SimpleNamespace(status=AgentExecutionStatus.RUNNING),
                ),
                operation_index=0,
                capability=cast(Any, incapable),
                generation_settings={},
                initial_tool_cycle_count=0,
            )

        execution = SimpleNamespace(
            status=AgentExecutionStatus.RESUMING,
            definition=object(),
        )
        expected_capability = SimpleNamespace(
            revision_binding=object(),
            task_input_advertisement=(
                TaskInputCapabilityAdvertisement.DURABLE
            ),
        )
        with (
            patch.object(
                Orchestrator,
                "continuation_execution_contract",
                return_value=(
                    object(),
                    object(),
                    expected_capability,
                ),
            ),
            self.assertRaises(RuntimeError),
        ):
            await orchestrator.resume_agent_execution(
                cast(AgentExecution, execution),
                operation_index=0,
                capability=cast(Any, expected_capability),
                generation_settings={},
                initial_tool_cycle_count=0,
            )
