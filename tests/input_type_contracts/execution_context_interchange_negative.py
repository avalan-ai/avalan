"""Reject interchange among execution, origin, and broker context fields."""

from avalan.agent import Specification
from avalan.agent.execution import AgentExecution, BranchInteractionBroker
from avalan.entities import ToolCallContext
from avalan.interaction import ExecutionOrigin
from avalan.model import ModelCallContext


def mix_execution_context(
    execution: AgentExecution,
    origin: ExecutionOrigin,
    branch_broker: BranchInteractionBroker,
) -> None:
    """Pass each typed execution dependency through the wrong field."""
    ModelCallContext(
        specification=Specification(),
        input=None,
        execution=origin,
        execution_origin=execution,
        interaction_broker=origin,
    )
    ToolCallContext(
        execution=origin,
        execution_origin=execution,
        interaction_broker=origin,
    )
    print(branch_broker)
