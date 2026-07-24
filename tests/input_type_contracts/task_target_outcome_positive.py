from avalan.interaction import (
    ContinuationId,
    InputRequestId,
    InputRequiredResult,
)
from avalan.task import (
    TaskTargetCompleted,
    TaskTargetOutcome,
    TaskTargetSuspended,
    completed_task_target_outcome,
    suspended_task_target_outcome,
)


def completed() -> TaskTargetOutcome:
    return completed_task_target_outcome({"status": "done"})


def suspended() -> TaskTargetOutcome:
    return suspended_task_target_outcome(
        InputRequiredResult(
            request_id=InputRequestId("request-1"),
            continuation_id=ContinuationId("continuation-1"),
            detached_resumption_available=True,
        ),
        checkpoint_id="checkpoint-1",
    )


def consume(outcome: TaskTargetOutcome) -> object:
    if isinstance(outcome, TaskTargetCompleted):
        return outcome.output
    assert isinstance(outcome, TaskTargetSuspended)
    return outcome.input_required
