"""Own manual submission through a backend's actual transaction type."""

from .submission import PreparedTaskSubmission, TaskSubmissionWrite

from typing import AsyncContextManager, Protocol, TypeVar

_Unit = TypeVar("_Unit")


class TaskSubmissionParticipant(Protocol[_Unit]):
    def submission_transaction(self) -> AsyncContextManager[_Unit]: ...

    async def submit_prepared(
        self, prepared: PreparedTaskSubmission, *, unit_of_work: _Unit
    ) -> TaskSubmissionWrite: ...


async def commit_task_submission(
    participant: TaskSubmissionParticipant[_Unit],
    prepared: PreparedTaskSubmission,
) -> TaskSubmissionWrite:
    """Return only after the owning transaction acknowledges its commit."""
    async with participant.submission_transaction() as unit:
        write = await participant.submit_prepared(prepared, unit_of_work=unit)
    return write
