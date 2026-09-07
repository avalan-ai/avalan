"""Encode closed queued execution and attempt context envelopes."""

from .deployment import execution_deployment_from_payload
from .provenance import (
    trigger_provenance_from_payload,
    trigger_provenance_payload,
)
from .store import (
    TaskClaim,
    TaskExecutionContext,
    TaskExecutionPayload,
    TaskExecutionRequest,
    TaskSnapshotMetadata,
    freeze_snapshot_metadata,
    freeze_snapshot_value,
)

from collections.abc import Mapping
from datetime import datetime
from typing import cast


class TaskExecutionCodecError(ValueError):
    """Reject superseded execution encodings with a safe field diagnostic."""

    def __init__(self, path: str) -> None:
        super().__init__("task.execution.invalid_encoding: " + path)


def _object(
    value: object, fields: set[str], path: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise TaskExecutionCodecError(path)
    return cast(Mapping[str, object], value)


def _envelope(value: object, tag: str) -> object:
    envelope = _object(value, {"format", "version", "payload"}, "envelope")
    if (
        envelope["format"] != tag
        or type(envelope["version"]) is not int
        or envelope["version"] != 1
    ):
        raise TaskExecutionCodecError("version")
    return envelope["payload"]


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_plain(item) for item in value]
    return value


def request_to_payload(request: TaskExecutionRequest) -> dict[str, object]:
    assert isinstance(request, TaskExecutionRequest)
    return {
        "format": "avalan.task.execution_request",
        "version": 1,
        "payload": {
            "definition_id": request.definition_id,
            "input_summary": _plain(request.input_summary),
            "file_summaries": _plain(request.file_summaries),
            "input_payload": (
                {
                    "input_value": _plain(request.input_payload.input_value),
                    "file_values": _plain(request.input_payload.file_values),
                }
                if request.input_payload is not None
                else None
            ),
            "idempotency_key": request.idempotency_key,
            "queue": request.queue,
            "metadata": _plain(request.metadata),
            "deployment": (
                request.deployment.payload()
                if request.deployment is not None
                else None
            ),
            "trigger": (
                trigger_provenance_payload(request.trigger)
                if request.trigger is not None
                else None
            ),
        },
    }


def request_from_payload(value: object) -> TaskExecutionRequest:
    payload = _object(
        _envelope(value, "avalan.task.execution_request"),
        {
            "definition_id",
            "input_summary",
            "file_summaries",
            "input_payload",
            "idempotency_key",
            "queue",
            "metadata",
            "trigger",
            "deployment",
        },
        "request",
    )
    files = payload["file_summaries"]
    if not isinstance(files, list):
        raise TaskExecutionCodecError("file_summaries")
    execution = None
    if payload["input_payload"] is not None:
        raw = _object(
            payload["input_payload"],
            {"input_value", "file_values"},
            "input_payload",
        )
        values = raw["file_values"]
        if not isinstance(values, list):
            raise TaskExecutionCodecError("file_values")
        execution = TaskExecutionPayload(
            input_value=freeze_snapshot_value(raw["input_value"]),
            file_values=tuple(freeze_snapshot_value(item) for item in values),
        )
    metadata = payload["metadata"]
    if not isinstance(metadata, Mapping):
        raise TaskExecutionCodecError("metadata")
    return TaskExecutionRequest(
        definition_id=cast(str, payload["definition_id"]),
        input_summary=freeze_snapshot_value(payload["input_summary"]),
        file_summaries=tuple(freeze_snapshot_value(item) for item in files),
        input_payload=execution,
        idempotency_key=cast(str | None, payload["idempotency_key"]),
        queue=cast(str | None, payload["queue"]),
        metadata=freeze_snapshot_metadata(metadata),
        deployment=(
            execution_deployment_from_payload(payload["deployment"])
            if payload["deployment"] is not None
            else None
        ),
        trigger=(
            trigger_provenance_from_payload(payload["trigger"])
            if payload["trigger"] is not None
            else None
        ),
    )


def context_to_payload(context: TaskExecutionContext) -> dict[str, object]:
    assert isinstance(context, TaskExecutionContext)
    claim = context.claim
    return {
        "format": "avalan.task.execution_context",
        "version": 1,
        "payload": {
            "run_id": context.run_id,
            "attempt_id": context.attempt_id,
            "attempt_number": context.attempt_number,
            "metadata": _plain(context.metadata),
            "deployment": (
                context.deployment.payload()
                if context.deployment is not None
                else None
            ),
            "trigger": (
                trigger_provenance_payload(context.trigger)
                if context.trigger is not None
                else None
            ),
            "claim": (
                {
                    "worker_id": claim.worker_id,
                    "claim_token": claim.claim_token,
                    "claimed_at": claim.claimed_at.isoformat(),
                    "lease_expires_at": claim.lease_expires_at.isoformat(),
                    "heartbeat_at": (
                        claim.heartbeat_at.isoformat()
                        if claim.heartbeat_at
                        else None
                    ),
                    "metadata": _plain(claim.metadata),
                }
                if claim
                else None
            ),
        },
    }


def context_from_payload(value: object) -> TaskExecutionContext:
    payload = _object(
        _envelope(value, "avalan.task.execution_context"),
        {
            "run_id",
            "attempt_id",
            "attempt_number",
            "metadata",
            "trigger",
            "deployment",
            "claim",
        },
        "context",
    )
    claim = None
    if payload["claim"] is not None:
        raw = _object(
            payload["claim"],
            {
                "worker_id",
                "claim_token",
                "claimed_at",
                "lease_expires_at",
                "heartbeat_at",
                "metadata",
            },
            "claim",
        )
        claim = TaskClaim(
            worker_id=cast(str, raw["worker_id"]),
            claim_token=cast(str, raw["claim_token"]),
            claimed_at=_datetime(raw["claimed_at"]),
            lease_expires_at=_datetime(raw["lease_expires_at"]),
            heartbeat_at=(
                _datetime(raw["heartbeat_at"])
                if raw["heartbeat_at"] is not None
                else None
            ),
            metadata=_metadata(raw["metadata"]),
        )
    return TaskExecutionContext(
        run_id=cast(str, payload["run_id"]),
        attempt_id=cast(str, payload["attempt_id"]),
        attempt_number=cast(int, payload["attempt_number"]),
        claim=claim,
        metadata=_metadata(payload["metadata"]),
        deployment=(
            execution_deployment_from_payload(payload["deployment"])
            if payload["deployment"] is not None
            else None
        ),
        trigger=(
            trigger_provenance_from_payload(payload["trigger"])
            if payload["trigger"] is not None
            else None
        ),
    )


def _metadata(value: object) -> TaskSnapshotMetadata:
    if not isinstance(value, Mapping):
        raise TaskExecutionCodecError("metadata")
    return freeze_snapshot_metadata(value)


def _datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise TaskExecutionCodecError("timestamp")
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        raise TaskExecutionCodecError("timestamp") from None
