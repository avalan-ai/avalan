"""Encode closed internal domain values without dynamic JSON decoding."""

from avalan.patch.domain import (
    ArtifactState,
    CommitTruth,
    ErrorStage,
    LifecyclePhase,
    LineageState,
    MutationState,
    PatchDiagnostic,
    PatchErrorCode,
    PatchEventId,
    PatchLifecycleEvent,
    PatchObserverCorrelationId,
    PatchObserverId,
    PatchPending,
    PatchPendingOperationId,
    PatchPlanId,
    PatchRequestId,
    PatchResult,
    PatchStatus,
    PostconditionState,
    PublicPendingProjection,
    RequestedEffectOccurrence,
    Retryability,
    SequenceNumber,
    WorkspaceChange,
)

_SEPARATOR = "\x1f"
_RESULT_TAG = "patch-result-v1"
_PENDING_TAG = "patch-pending-v1"
_PUBLIC_PENDING_TAG = "patch-public-pending-v1"
_EVENT_TAG = "patch-event-v1"
_DIAGNOSTIC_TAG = "patch-diagnostic-v1"


def encode_result(value: PatchResult) -> bytes:
    """Encode one terminal result with stable field order."""
    truth = value.truth
    diagnostic = value.diagnostic
    stage, code, retryability = (
        ("", "", "")
        if diagnostic is None
        else (
            diagnostic.stage.value,
            diagnostic.code.value,
            diagnostic.retryability.value,
        )
    )
    return _encode(
        (
            _RESULT_TAG,
            str(value.schema_version),
            value.request_id.value,
            value.plan_id.value,
            value.lifecycle.value,
            value.status.value,
            truth.mutation_state.value,
            truth.lineage_state.value,
            truth.requested_effect_occurred.value,
            truth.artifact_state.value,
            truth.workspace_change.value,
            _boolean(truth.commit_set_exact),
            truth.postcondition.value,
            stage,
            code,
            retryability,
        )
    )


def decode_result(payload: bytes) -> PatchResult:
    """Decode one exact version-one terminal result value."""
    fields = _decode(payload, _RESULT_TAG, 16)
    truth = CommitTruth(
        mutation_state=MutationState(fields[6]),
        lineage_state=_lineage_state(fields[7]),
        requested_effect_occurred=RequestedEffectOccurrence(fields[8]),
        artifact_state=ArtifactState(fields[9]),
        workspace_change=WorkspaceChange(fields[10]),
        commit_set_exact=_decode_boolean(fields[11]),
        postcondition=PostconditionState(fields[12]),
    )
    return PatchResult(
        schema_version=_schema_version(fields[1]),
        request_id=PatchRequestId(fields[2]),
        plan_id=PatchPlanId(fields[3]),
        lifecycle=LifecyclePhase(fields[4]),
        status=PatchStatus(fields[5]),
        truth=truth,
        diagnostic=_result_diagnostic(fields[13:]),
    )


def encode_pending(value: PatchPending) -> bytes:
    """Encode one public-safe pending envelope with stable field order."""
    return _encode(
        (
            _PENDING_TAG,
            str(value.schema_version),
            value.pending_operation_id.value,
            value.request_id.value,
            value.correlation_id.value,
            value.lifecycle.value,
        )
    )


def decode_pending(payload: bytes) -> PatchPending:
    """Decode one exact version-one pending envelope."""
    fields = _decode(payload, _PENDING_TAG, 6)
    return PatchPending(
        schema_version=_schema_version(fields[1]),
        pending_operation_id=PatchPendingOperationId(fields[2]),
        request_id=PatchRequestId(fields[3]),
        correlation_id=PatchObserverCorrelationId(fields[4]),
        lifecycle=LifecyclePhase(fields[5]),
    )


def encode_public_pending(value: PublicPendingProjection) -> bytes:
    """Encode one audience-safe continuation handle with stable field order."""
    return _encode(
        (
            _PUBLIC_PENDING_TAG,
            str(value.schema_version),
            value.pending_operation_id.value,
            value.correlation_id.value,
            value.lifecycle.value,
        )
    )


def decode_public_pending(payload: bytes) -> PublicPendingProjection:
    """Decode one audience-safe continuation handle without request ID."""
    fields = _decode(payload, _PUBLIC_PENDING_TAG, 5)
    return PublicPendingProjection(
        schema_version=_schema_version(fields[1]),
        pending_operation_id=PatchPendingOperationId(fields[2]),
        correlation_id=PatchObserverCorrelationId(fields[3]),
        lifecycle=LifecyclePhase(fields[4]),
    )


def encode_event(value: PatchLifecycleEvent) -> bytes:
    """Encode one content-free lifecycle event with stable field order."""
    return _encode(
        (
            _EVENT_TAG,
            str(value.schema_version),
            value.event_id.value,
            value.observer_id.value,
            value.correlation_id.value,
            value.request_id.value,
            str(value.sequence.value),
            value.lifecycle.value,
        )
    )


def decode_event(payload: bytes) -> PatchLifecycleEvent:
    """Decode one exact version-one lifecycle event."""
    fields = _decode(payload, _EVENT_TAG, 8)
    return PatchLifecycleEvent(
        schema_version=_schema_version(fields[1]),
        event_id=PatchEventId(fields[2]),
        observer_id=PatchObserverId(fields[3]),
        correlation_id=PatchObserverCorrelationId(fields[4]),
        request_id=PatchRequestId(fields[5]),
        sequence=SequenceNumber(_positive_integer(fields[6])),
        lifecycle=LifecyclePhase(fields[7]),
    )


def encode_diagnostic(value: PatchDiagnostic) -> bytes:
    """Encode one stable diagnostic without protected bytes."""
    return _encode(
        (
            _DIAGNOSTIC_TAG,
            value.stage.value,
            value.code.value,
            value.retryability.value,
        )
    )


def decode_diagnostic(payload: bytes) -> PatchDiagnostic:
    """Decode one stable diagnostic value."""
    fields = _decode(payload, _DIAGNOSTIC_TAG, 4)
    return PatchDiagnostic(
        stage=ErrorStage(fields[1]),
        code=PatchErrorCode(fields[2]),
        retryability=Retryability(fields[3]),
    )


def _result_diagnostic(fields: tuple[str, ...]) -> PatchDiagnostic | None:
    """Decode the canonical absent-or-complete terminal diagnostic tuple."""
    if fields == ("", "", ""):
        return None
    if len(fields) != 3 or any(not field for field in fields):
        raise ValueError("terminal diagnostic fields are invalid")
    return PatchDiagnostic(
        stage=ErrorStage(fields[0]),
        code=PatchErrorCode(fields[1]),
        retryability=Retryability(fields[2]),
    )


def _encode(fields: tuple[str, ...]) -> bytes:
    """Encode closed ASCII-safe fields with a reserved separator."""
    if any(
        _SEPARATOR in field or "\n" in field or "\r" in field
        for field in fields
    ):
        raise ValueError("internal codec field contains a reserved delimiter")
    return _SEPARATOR.join(fields).encode("ascii")


def _decode(payload: bytes, tag: str, expected_size: int) -> tuple[str, ...]:
    """Decode and validate one closed ASCII internal envelope."""
    try:
        fields = tuple(payload.decode("ascii").split(_SEPARATOR))
    except UnicodeDecodeError as exc:
        raise ValueError("internal codec payload is not ASCII") from exc
    if len(fields) != expected_size or fields[0] != tag:
        raise ValueError("internal codec envelope is invalid")
    return fields


def _schema_version(value: str) -> int:
    """Decode the sole supported schema version."""
    if value != "1":
        raise ValueError("internal codec schema version is invalid")
    return 1


def _positive_integer(value: str) -> int:
    """Decode one canonical nonnegative decimal sequence number."""
    if not value.isdecimal() or (len(value) > 1 and value.startswith("0")):
        raise ValueError("internal codec integer is invalid")
    return int(value)


def _boolean(value: bool) -> str:
    """Encode one Boolean using canonical lower-case text."""
    return "true" if value else "false"


def _decode_boolean(value: str) -> bool:
    """Decode one exact canonical Boolean value."""
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError("internal codec Boolean is invalid")


def _lineage_state(value: str) -> LineageState:
    """Decode one lineage state without accepting unknown enum values."""
    return LineageState(value)
