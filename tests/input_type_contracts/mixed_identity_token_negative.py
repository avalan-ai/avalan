"""Reject mixed opaque identity tokens in public result types."""

from avalan.interaction import (
    InputRequestId,
    InputRequiredResult,
    RunId,
)

run_id = RunId("run")
request_id = InputRequestId("request")

InputRequiredResult(
    request_id=run_id,
    continuation_id=request_id,
    detached_resumption_available=True,
)
