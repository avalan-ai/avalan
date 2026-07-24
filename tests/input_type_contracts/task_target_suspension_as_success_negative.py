from avalan.interaction import (
    ContinuationId,
    InputRequestId,
    InputRequiredResult,
)
from avalan.task import (
    TaskTargetCompleted,
    TaskTargetSuspended,
)

suspended = TaskTargetSuspended(
    input_required=InputRequiredResult(
        request_id=InputRequestId("request-1"),
        continuation_id=ContinuationId("continuation-1"),
        detached_resumption_available=True,
    )
)
completed: TaskTargetCompleted = suspended
successful_output: object = suspended.output
