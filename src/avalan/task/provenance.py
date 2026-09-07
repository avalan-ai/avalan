"""Preserve immutable scheduling provenance outside application input."""

from ..types import assert_non_empty_string, assert_positive_int

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from re import fullmatch


class TaskProvenanceError(ValueError):
    """Reject invalid or unsupported task provenance without private values."""

    def __init__(self, path: str, *, unsupported: bool = False) -> None:
        self.path = path
        self.code = (
            "task.trigger_provenance.unsupported_version"
            if unsupported
            else "task.trigger_provenance.invalid"
        )
        super().__init__(self.code + ": " + path)


@dataclass(frozen=True, slots=True, kw_only=True)
class TriggerInvocationContext:
    """Retain occurrence identity and its committed admission clock."""

    trigger_id: str
    trigger_revision: int
    occurrence_id: str
    scheduled_at: datetime
    dispatched_at: datetime

    def __post_init__(self) -> None:
        assert_non_empty_string(self.trigger_id, "trigger_id")
        assert_positive_int(self.trigger_revision, "trigger_revision")
        assert_non_empty_string(self.occurrence_id, "occurrence_id")
        for name in ("scheduled_at", "dispatched_at"):
            value = getattr(self, name)
            assert isinstance(value, datetime)
            assert value.tzinfo is not None and value.utcoffset() is not None
            object.__setattr__(self, name, value.astimezone(UTC))
        assert self.dispatched_at >= self.scheduled_at


def trigger_provenance_payload(
    context: TriggerInvocationContext,
) -> dict[str, object]:
    """Encode the sole supported closed provenance envelope."""
    assert isinstance(context, TriggerInvocationContext)
    return {
        "format": "avalan.task.trigger_provenance",
        "version": 1,
        "payload": {
            "trigger_id": context.trigger_id,
            "trigger_revision": context.trigger_revision,
            "occurrence_id": context.occurrence_id,
            "scheduled_at": (
                context.scheduled_at.isoformat(
                    timespec="microseconds"
                ).replace("+00:00", "Z")
            ),
            "dispatched_at": (
                context.dispatched_at.isoformat(
                    timespec="microseconds"
                ).replace("+00:00", "Z")
            ),
        },
    }


def trigger_provenance_from_payload(value: object) -> TriggerInvocationContext:
    """Reject superseded or incomplete representations explicitly."""
    if not isinstance(value, Mapping) or set(value) != {
        "format",
        "version",
        "payload",
    }:
        raise TaskProvenanceError("envelope")
    if (
        value["format"] != "avalan.task.trigger_provenance"
        or type(value["version"]) is not int
        or value["version"] != 1
    ):
        raise TaskProvenanceError("version", unsupported=True)
    payload = value["payload"]
    if not isinstance(payload, Mapping) or set(payload) != {
        "trigger_id",
        "trigger_revision",
        "occurrence_id",
        "scheduled_at",
        "dispatched_at",
    }:
        raise TaskProvenanceError("payload")
    trigger_id, occurrence_id = payload["trigger_id"], payload["occurrence_id"]
    revision = payload["trigger_revision"]
    if (
        not isinstance(trigger_id, str)
        or not trigger_id.strip()
        or not isinstance(occurrence_id, str)
        or not occurrence_id.strip()
        or type(revision) is not int
        or revision < 1
    ):
        raise TaskProvenanceError("identity")
    scheduled, dispatched = (
        _timestamp(payload["scheduled_at"]),
        _timestamp(payload["dispatched_at"]),
    )
    if dispatched < scheduled:
        raise TaskProvenanceError("dispatched_at")
    return TriggerInvocationContext(
        trigger_id=trigger_id,
        trigger_revision=revision,
        occurrence_id=occurrence_id,
        scheduled_at=scheduled,
        dispatched_at=dispatched,
    )


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str) or not fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z",
        value,
    ):
        raise TaskProvenanceError("timestamp")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise TaskProvenanceError("timestamp") from None
