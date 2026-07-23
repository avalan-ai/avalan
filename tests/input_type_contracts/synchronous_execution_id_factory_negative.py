"""Reject a synchronous implementation of the execution identity factory."""

from avalan.agent.execution import ExecutionIdFactory
from avalan.interaction import (
    BranchId,
    ModelCallId,
    RunId,
    StreamSessionId,
    TurnId,
)


class SynchronousExecutionIdFactory:
    """Return identities synchronously instead of through awaitables."""

    def new_run_id(self) -> RunId:
        return RunId("run")

    def new_turn_id(self) -> TurnId:
        return TurnId("turn")

    def new_model_call_id(self) -> ModelCallId:
        return ModelCallId("model-call")

    def new_branch_id(self) -> BranchId:
        return BranchId("branch")

    def new_stream_session_id(self) -> StreamSessionId:
        return StreamSessionId("stream")


factory: ExecutionIdFactory = SynchronousExecutionIdFactory()
