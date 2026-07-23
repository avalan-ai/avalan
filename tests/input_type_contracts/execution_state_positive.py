"""Lock invocation-scoped execution state and identity factory types."""

from typing import assert_type

from avalan.agent.execution import (
    AgentExecution,
    AgentExecutionSnapshot,
    BranchInteractionBroker,
    ExecutionIdFactory,
    ExecutionLedgerEntry,
)
from avalan.agent.orchestrator.response.orchestrator_response import (
    OrchestratorResponse,
)
from avalan.entities import ToolCallContext
from avalan.interaction import ExecutionOrigin, RunId
from avalan.model import ModelCallContext


async def inspect_execution(
    execution: AgentExecution,
    id_factory: ExecutionIdFactory,
    model_context: ModelCallContext,
    tool_context: ToolCallContext,
    response: OrchestratorResponse,
) -> None:
    """Exercise immutable execution views and async identity minting."""
    assert_type(execution.snapshot, AgentExecutionSnapshot)
    assert_type(execution.origin, ExecutionOrigin)
    assert_type(execution.ledger, tuple[ExecutionLedgerEntry, ...])
    assert_type(
        model_context.interaction_broker,
        BranchInteractionBroker | None,
    )
    assert_type(
        tool_context.interaction_broker,
        BranchInteractionBroker | None,
    )
    assert_type(response.execution, AgentExecution | None)
    assert_type(await id_factory.new_run_id(), RunId)
    assert_type(await response.to_str(), str)
    assert_type(await response.to_json(), str)
