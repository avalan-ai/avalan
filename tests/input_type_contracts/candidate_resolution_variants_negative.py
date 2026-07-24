"""Reject non-candidate outcomes at the external resolution boundary."""

from datetime import UTC, datetime

from avalan.interaction import (
    AnswerProvenance,
    CancellationScope,
    CancelledResolution,
    ExpiredResolution,
    InputCandidateResolution,
    InputRequestId,
    SupersededResolution,
    TimedOutResolution,
    UnavailableResolution,
)

_NOW = datetime(2026, 7, 21, tzinfo=UTC)
_REQUEST_ID = InputRequestId("request-1")

cancelled: InputCandidateResolution = CancelledResolution(
    request_id=_REQUEST_ID,
    provenance=AnswerProvenance.HUMAN,
    resolved_at=_NOW,
    scope=CancellationScope.REQUEST,
)
timed_out: InputCandidateResolution = TimedOutResolution(
    request_id=_REQUEST_ID,
    provenance=AnswerProvenance.POLICY,
    resolved_at=_NOW,
)
unavailable: InputCandidateResolution = UnavailableResolution(
    request_id=_REQUEST_ID,
    provenance=AnswerProvenance.POLICY,
    resolved_at=_NOW,
)
expired: InputCandidateResolution = ExpiredResolution(
    request_id=_REQUEST_ID,
    provenance=AnswerProvenance.POLICY,
    resolved_at=_NOW,
)
superseded: InputCandidateResolution = SupersededResolution(
    request_id=_REQUEST_ID,
    provenance=AnswerProvenance.POLICY,
    resolved_at=_NOW,
)
