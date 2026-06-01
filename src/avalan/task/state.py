from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType


class TaskRunState(StrEnum):
    CREATED = "created"
    VALIDATED = "validated"
    QUEUED = "queued"
    CLAIMED = "claimed"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class TaskAttemptState(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABANDONED = "abandoned"


TASK_RUN_TERMINAL_STATES = frozenset(
    {
        TaskRunState.SUCCEEDED,
        TaskRunState.FAILED,
        TaskRunState.CANCELLED,
        TaskRunState.EXPIRED,
    }
)
TASK_ATTEMPT_TERMINAL_STATES = frozenset(
    {
        TaskAttemptState.SUCCEEDED,
        TaskAttemptState.FAILED,
        TaskAttemptState.ABANDONED,
    }
)

VALID_TASK_RUN_TRANSITIONS: Mapping[TaskRunState, frozenset[TaskRunState]] = (
    MappingProxyType(
        {
            TaskRunState.CREATED: frozenset(
                {
                    TaskRunState.VALIDATED,
                    TaskRunState.FAILED,
                }
            ),
            TaskRunState.VALIDATED: frozenset(
                {
                    TaskRunState.RUNNING,
                    TaskRunState.QUEUED,
                    TaskRunState.FAILED,
                }
            ),
            TaskRunState.QUEUED: frozenset(
                {
                    TaskRunState.CLAIMED,
                    TaskRunState.EXPIRED,
                    TaskRunState.CANCEL_REQUESTED,
                    TaskRunState.FAILED,
                }
            ),
            TaskRunState.CLAIMED: frozenset(
                {
                    TaskRunState.RUNNING,
                    TaskRunState.QUEUED,
                    TaskRunState.FAILED,
                }
            ),
            TaskRunState.RUNNING: frozenset(
                {
                    TaskRunState.SUCCEEDED,
                    TaskRunState.FAILED,
                    TaskRunState.QUEUED,
                    TaskRunState.CANCEL_REQUESTED,
                    TaskRunState.EXPIRED,
                }
            ),
            TaskRunState.CANCEL_REQUESTED: frozenset(
                {
                    TaskRunState.CANCELLED,
                    TaskRunState.FAILED,
                }
            ),
            TaskRunState.SUCCEEDED: frozenset(),
            TaskRunState.FAILED: frozenset(),
            TaskRunState.CANCELLED: frozenset(),
            TaskRunState.EXPIRED: frozenset(),
        }
    )
)
VALID_TASK_ATTEMPT_TRANSITIONS: Mapping[
    TaskAttemptState, frozenset[TaskAttemptState]
] = MappingProxyType(
    {
        TaskAttemptState.CREATED: frozenset(
            {
                TaskAttemptState.RUNNING,
                TaskAttemptState.ABANDONED,
            }
        ),
        TaskAttemptState.RUNNING: frozenset(
            {
                TaskAttemptState.SUCCEEDED,
                TaskAttemptState.FAILED,
                TaskAttemptState.ABANDONED,
            }
        ),
        TaskAttemptState.SUCCEEDED: frozenset(),
        TaskAttemptState.FAILED: frozenset(),
        TaskAttemptState.ABANDONED: frozenset(),
    }
)


def _assert_run_state(state: TaskRunState, field_name: str) -> None:
    assert isinstance(
        state, TaskRunState
    ), f"{field_name} must be a TaskRunState"


def _assert_attempt_state(state: TaskAttemptState, field_name: str) -> None:
    assert isinstance(
        state, TaskAttemptState
    ), f"{field_name} must be a TaskAttemptState"


def is_terminal_run_state(state: TaskRunState) -> bool:
    _assert_run_state(state, "state")
    return state in TASK_RUN_TERMINAL_STATES


def is_terminal_attempt_state(state: TaskAttemptState) -> bool:
    _assert_attempt_state(state, "state")
    return state in TASK_ATTEMPT_TERMINAL_STATES


def valid_run_transitions(
    from_state: TaskRunState,
) -> frozenset[TaskRunState]:
    _assert_run_state(from_state, "from_state")
    return VALID_TASK_RUN_TRANSITIONS[from_state]


def valid_attempt_transitions(
    from_state: TaskAttemptState,
) -> frozenset[TaskAttemptState]:
    _assert_attempt_state(from_state, "from_state")
    return VALID_TASK_ATTEMPT_TRANSITIONS[from_state]


def is_valid_run_transition(
    from_state: TaskRunState, to_state: TaskRunState
) -> bool:
    _assert_run_state(from_state, "from_state")
    _assert_run_state(to_state, "to_state")
    return to_state in valid_run_transitions(from_state)


def is_valid_attempt_transition(
    from_state: TaskAttemptState, to_state: TaskAttemptState
) -> bool:
    _assert_attempt_state(from_state, "from_state")
    _assert_attempt_state(to_state, "to_state")
    return to_state in valid_attempt_transitions(from_state)


def validate_run_transition(
    from_state: TaskRunState, to_state: TaskRunState
) -> None:
    assert is_valid_run_transition(
        from_state, to_state
    ), f"invalid task run transition: {from_state.value} -> {to_state.value}"


def validate_attempt_transition(
    from_state: TaskAttemptState, to_state: TaskAttemptState
) -> None:
    assert is_valid_attempt_transition(from_state, to_state), (
        "invalid task attempt transition: "
        f"{from_state.value} -> {to_state.value}"
    )
