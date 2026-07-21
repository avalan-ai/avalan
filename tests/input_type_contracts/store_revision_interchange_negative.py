"""Reject interchangeable lifecycle, store, and schedule revisions."""

from avalan.interaction import (
    DeadlineScheduleRevision,
    InteractionStoreRevision,
    StateRevision,
)

state_revision = StateRevision(1)
store_revision = InteractionStoreRevision(1)
schedule_revision = DeadlineScheduleRevision(1)

expects_state: StateRevision = store_revision
expects_store: InteractionStoreRevision = schedule_revision
expects_schedule: DeadlineScheduleRevision = state_revision
