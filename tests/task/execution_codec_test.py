"""Prove execution codecs are closed and retain immutable provenance."""

from dataclasses import replace
from datetime import timedelta

from deployment_test import deployment
from provenance_test import NOW, invocation
from pytest import raises

from avalan.task.execution_codec import (
    TaskExecutionCodecError,
    context_from_payload,
    context_to_payload,
    request_from_payload,
    request_to_payload,
)
from avalan.task.store import (
    TaskClaim,
    TaskExecutionContext,
    TaskExecutionPayload,
    TaskExecutionRequest,
)


def request() -> TaskExecutionRequest:
    return TaskExecutionRequest(
        definition_id=deployment().task_hash,
        deployment=deployment(),
        queue="work",
        trigger=invocation(),
        input_summary={"redacted": True},
        file_summaries=({"artifact": "ref"},),
        input_payload=TaskExecutionPayload(
            input_value={"nested": ("input",)}, file_values=({"id": "file"},)
        ),
        metadata={"safe": True},
    )


def context() -> TaskExecutionContext:
    return TaskExecutionContext(
        run_id="run",
        attempt_id="attempt",
        attempt_number=2,
        trigger=invocation(),
        deployment=deployment(),
        claim=TaskClaim(
            worker_id="worker",
            claim_token="token",
            claimed_at=NOW,
            lease_expires_at=NOW + timedelta(seconds=30),
            heartbeat_at=NOW,
            metadata={"claim": "private"},
        ),
        metadata={"nested": (1, 2)},
    )


def test_request_round_trip_keeps_provenance_outside_frozen_input() -> None:
    original = request()
    encoded = request_to_payload(original)
    assert request_from_payload(encoded) == original
    assert original.input_payload is not None
    assert original.input_payload.input_value == {"nested": ("input",)}
    ordinary = replace(
        original, trigger=None, deployment=None, input_payload=None
    )
    assert request_from_payload(request_to_payload(ordinary)) == ordinary
    payload = encoded["payload"]
    assert isinstance(payload, dict)
    summary = payload["input_summary"]
    assert isinstance(summary, dict)
    summary["redacted"] = False
    assert original.input_summary == {"redacted": True}


def test_context_round_trip_keeps_attempt_claim_and_explicit_nulls() -> None:
    original = context()
    assert context_from_payload(context_to_payload(original)) == original
    assert original.claim is not None
    without_heartbeat = replace(
        original, claim=replace(original.claim, heartbeat_at=None)
    )
    assert (
        context_from_payload(context_to_payload(without_heartbeat))
        == without_heartbeat
    )
    ordinary = replace(original, trigger=None, deployment=None, claim=None)
    assert context_from_payload(context_to_payload(ordinary)) == ordinary


def test_execution_envelopes_reject_invalid_records() -> None:
    for kind in ("request", "context"):
        envelope = (
            request_to_payload(request())
            if kind == "request"
            else context_to_payload(context())
        )
        decode = (
            request_from_payload if kind == "request" else context_from_payload
        )
        for invalid in (
            None,
            {},
            envelope["payload"],
            dict(envelope, extra="secret"),
        ):
            with raises(TaskExecutionCodecError, match="envelope"):
                decode(invalid)
        for field, value in (
            ("format", "old"),
            ("version", True),
            ("version", 0),
        ):
            with raises(TaskExecutionCodecError, match="version"):
                decode(dict(envelope, **{field: value}))
        with raises(TaskExecutionCodecError):
            decode(dict(envelope, payload={}))


def test_request_rejects_incomplete_payload_and_non_array_files() -> None:
    for key, value in (
        ("file_summaries", None),
        ("metadata", None),
        ("input_payload", {}),
        ("input_payload", {"input_value": None, "file_values": {}}),
    ):
        envelope = request_to_payload(request())
        payload = envelope["payload"]
        assert isinstance(payload, dict)
        payload[key] = value
        with raises(TaskExecutionCodecError):
            request_from_payload(envelope)


def test_context_rejects_invalid_claim_metadata_and_timestamps() -> None:
    for key, value in (
        ("metadata", []),
        ("claimed_at", None),
        ("claimed_at", "not a date"),
    ):
        envelope = context_to_payload(context())
        payload = envelope["payload"]
        assert isinstance(payload, dict)
        claim = payload["claim"]
        assert isinstance(claim, dict)
        claim[key] = value
        with raises(TaskExecutionCodecError):
            context_from_payload(envelope)
