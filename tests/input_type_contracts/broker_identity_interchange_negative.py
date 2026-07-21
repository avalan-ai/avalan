"""Reject interchangeable opaque identities in broker correlation."""

from avalan.interaction import (
    AgentId,
    BranchId,
    ContinuationId,
    InputRequestId,
    InteractionCorrelation,
    ModelCallId,
    RunId,
    TurnId,
)

request_id = InputRequestId("request")
continuation_id = ContinuationId("continuation")
run_id = RunId("run")

InteractionCorrelation(
    request_id=run_id,
    continuation_id=request_id,
    run_id=continuation_id,
    turn_id=TurnId("turn"),
    agent_id=AgentId("agent"),
    branch_id=BranchId("branch"),
    model_call_id=ModelCallId("call"),
)
