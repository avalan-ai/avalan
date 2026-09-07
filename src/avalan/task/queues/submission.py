"""Persist submission recovery evidence inside a supplied transaction."""

from ...pgsql import PgsqlUnitOfWork
from ...types import assert_non_empty_string
from ..submission import PreparedTaskSubmission

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from json import dumps


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskSubmissionEvidence:
    """Retain a submission association independently of reservation TTL."""

    owner_scope: str
    submission_id: str
    run_id: str
    definition_id: str
    fingerprint: str
    created: bool

    def __post_init__(self) -> None:
        for name in (
            "owner_scope",
            "submission_id",
            "run_id",
            "definition_id",
        ):
            assert_non_empty_string(getattr(self, name), name)
        assert isinstance(self.fingerprint, str)
        assert len(self.fingerprint) == 64 and all(
            character in "0123456789abcdef" for character in self.fingerprint
        )
        assert isinstance(self.created, bool)

    def envelope(self) -> dict[str, object]:
        """Return the closed versioned recovery record."""
        return {
            "format": "avalan.task.submission",
            "version": 1,
            "payload": {
                "owner_scope": self.owner_scope,
                "submission_id": self.submission_id,
                "run_id": self.run_id,
                "definition_id": self.definition_id,
                "fingerprint": self.fingerprint,
                "created": self.created,
            },
        }


def submission_evidence_from_envelope(value: object) -> TaskSubmissionEvidence:
    """Reject unsupported versions and malformed persisted associations."""
    assert isinstance(value, Mapping)
    assert set(value) == {"format", "version", "payload"}
    assert value["format"] == "avalan.task.submission"
    assert type(value["version"]) is int and value["version"] == 1
    payload = value["payload"]
    assert isinstance(payload, Mapping)
    assert set(payload) == {
        "owner_scope",
        "submission_id",
        "run_id",
        "definition_id",
        "fingerprint",
        "created",
    }
    for key in (
        "owner_scope",
        "submission_id",
        "run_id",
        "definition_id",
        "fingerprint",
    ):
        assert isinstance(payload[key], str)
    assert isinstance(payload["created"], bool)
    return TaskSubmissionEvidence(
        owner_scope=payload["owner_scope"],
        submission_id=payload["submission_id"],
        run_id=payload["run_id"],
        definition_id=payload["definition_id"],
        fingerprint=payload["fingerprint"],
        created=payload["created"],
    )


async def lock_task_submission(
    unit: PgsqlUnitOfWork, prepared: PreparedTaskSubmission
) -> None:
    """Wait for prior writers of this submission before reading evidence.

    Use the same transaction-scoped lock for admission and reconciliation.
    Fresh-connection absence is conclusive only after this barrier; an
    older transaction may otherwise commit after the read. Hash collisions
    merely serialize unrelated submissions and never identify evidence.
    """
    assert isinstance(unit, PgsqlUnitOfWork)
    assert isinstance(prepared, PreparedTaskSubmission)
    await unit.cursor.execute(
        "SELECT current_setting('transaction_isolation') AS isolation"
    )
    row = await unit.cursor.fetchone()
    assert (
        row is not None and row["isolation"] == "read committed"
    ), "task submission requires read committed isolation"
    digest = sha256(
        dumps(
            (
                "avalan.task.submission",
                prepared.owner_scope,
                prepared.submission_id,
            ),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).digest()
    lock_id = int.from_bytes(digest[:8], "big", signed=True)
    await unit.cursor.execute("SELECT pg_advisory_xact_lock(%s)", (lock_id,))


async def read_task_submission(
    unit: PgsqlUnitOfWork,
    prepared: PreparedTaskSubmission,
    *,
    fingerprint: str,
) -> TaskSubmissionEvidence | None:
    """Read an association after waiting for the writer completion fence."""
    await lock_task_submission(unit, prepared)
    await unit.cursor.execute(
        'SELECT "run_id", "payload" FROM "task_submissions" '
        'WHERE "owner_scope" = %s AND "submission_id" = %s',
        (prepared.owner_scope, prepared.submission_id),
    )
    row = await unit.cursor.fetchone()
    if row is None:
        return None
    evidence = submission_evidence_from_envelope(row["payload"])
    assert evidence.owner_scope == prepared.owner_scope
    assert evidence.submission_id == prepared.submission_id
    assert evidence.run_id == row["run_id"]
    assert evidence.definition_id == prepared.execution.definition_id
    assert evidence.fingerprint == fingerprint
    return evidence


async def insert_task_submission(
    unit: PgsqlUnitOfWork, evidence: TaskSubmissionEvidence
) -> None:
    """Write immutable association evidence in the run's transaction."""
    assert isinstance(unit, PgsqlUnitOfWork)
    assert isinstance(evidence, TaskSubmissionEvidence)
    await unit.cursor.execute(
        'INSERT INTO "task_submissions" '
        '("owner_scope", "submission_id", "run_id", "payload") '
        "VALUES (%s, %s, %s, %s::jsonb)",
        (
            evidence.owner_scope,
            evidence.submission_id,
            evidence.run_id,
            dumps(
                evidence.envelope(), ensure_ascii=False, separators=(",", ":")
            ),
        ),
    )
