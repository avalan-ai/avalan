from types import MappingProxyType
from typing import cast
from unittest import TestCase, main

from avalan.task import (
    TASK_ATTEMPT_TERMINAL_STATES,
    TASK_RUN_TERMINAL_STATES,
    VALID_TASK_ATTEMPT_TRANSITIONS,
    VALID_TASK_RUN_TRANSITIONS,
    TaskAttemptState,
    TaskRunState,
    is_terminal_attempt_state,
    is_terminal_run_state,
    is_valid_attempt_transition,
    is_valid_run_transition,
    valid_attempt_transitions,
    valid_run_transitions,
    validate_attempt_transition,
    validate_run_transition,
)


class TaskStateTest(TestCase):
    def test_run_state_values_preserve_contract_vocabulary(self) -> None:
        self.assertEqual(TaskRunState.CREATED.value, "created")
        self.assertEqual(TaskRunState.VALIDATED.value, "validated")
        self.assertEqual(TaskRunState.QUEUED.value, "queued")
        self.assertEqual(TaskRunState.CLAIMED.value, "claimed")
        self.assertEqual(TaskRunState.RUNNING.value, "running")
        self.assertEqual(TaskRunState.SUCCEEDED.value, "succeeded")
        self.assertEqual(TaskRunState.FAILED.value, "failed")
        self.assertEqual(
            TaskRunState.CANCEL_REQUESTED.value,
            "cancel_requested",
        )
        self.assertEqual(TaskRunState.CANCELLED.value, "cancelled")
        self.assertEqual(TaskRunState.EXPIRED.value, "expired")

    def test_attempt_state_values_preserve_contract_vocabulary(self) -> None:
        self.assertEqual(TaskAttemptState.CREATED.value, "created")
        self.assertEqual(TaskAttemptState.RUNNING.value, "running")
        self.assertEqual(TaskAttemptState.SUCCEEDED.value, "succeeded")
        self.assertEqual(TaskAttemptState.FAILED.value, "failed")
        self.assertEqual(TaskAttemptState.ABANDONED.value, "abandoned")

    def test_run_transition_table_matches_source_contract(self) -> None:
        expected = {
            TaskRunState.CREATED: {
                TaskRunState.VALIDATED,
                TaskRunState.FAILED,
            },
            TaskRunState.VALIDATED: {
                TaskRunState.RUNNING,
                TaskRunState.QUEUED,
                TaskRunState.FAILED,
            },
            TaskRunState.QUEUED: {
                TaskRunState.CLAIMED,
                TaskRunState.EXPIRED,
                TaskRunState.CANCEL_REQUESTED,
                TaskRunState.FAILED,
            },
            TaskRunState.CLAIMED: {
                TaskRunState.RUNNING,
                TaskRunState.QUEUED,
                TaskRunState.FAILED,
            },
            TaskRunState.RUNNING: {
                TaskRunState.SUCCEEDED,
                TaskRunState.FAILED,
                TaskRunState.QUEUED,
                TaskRunState.CANCEL_REQUESTED,
                TaskRunState.EXPIRED,
            },
            TaskRunState.CANCEL_REQUESTED: {
                TaskRunState.CANCELLED,
                TaskRunState.FAILED,
            },
            TaskRunState.SUCCEEDED: set(),
            TaskRunState.FAILED: set(),
            TaskRunState.CANCELLED: set(),
            TaskRunState.EXPIRED: set(),
        }

        self.assertEqual(set(VALID_TASK_RUN_TRANSITIONS), set(TaskRunState))
        for from_state in TaskRunState:
            self.assertEqual(
                VALID_TASK_RUN_TRANSITIONS[from_state],
                frozenset(expected[from_state]),
            )
            for to_state in TaskRunState:
                valid = to_state in expected[from_state]
                self.assertEqual(
                    is_valid_run_transition(from_state, to_state),
                    valid,
                    f"{from_state.value} -> {to_state.value}",
                )
                if valid:
                    validate_run_transition(from_state, to_state)
                else:
                    with self.assertRaisesRegex(
                        AssertionError,
                        "invalid task run transition: "
                        f"{from_state.value} -> {to_state.value}",
                    ):
                        validate_run_transition(from_state, to_state)

    def test_attempt_transition_table_matches_source_contract(self) -> None:
        expected = {
            TaskAttemptState.CREATED: {
                TaskAttemptState.RUNNING,
                TaskAttemptState.ABANDONED,
            },
            TaskAttemptState.RUNNING: {
                TaskAttemptState.SUCCEEDED,
                TaskAttemptState.FAILED,
                TaskAttemptState.ABANDONED,
            },
            TaskAttemptState.SUCCEEDED: set(),
            TaskAttemptState.FAILED: set(),
            TaskAttemptState.ABANDONED: set(),
        }

        self.assertEqual(
            set(VALID_TASK_ATTEMPT_TRANSITIONS),
            set(TaskAttemptState),
        )
        for from_state in TaskAttemptState:
            self.assertEqual(
                VALID_TASK_ATTEMPT_TRANSITIONS[from_state],
                frozenset(expected[from_state]),
            )
            for to_state in TaskAttemptState:
                valid = to_state in expected[from_state]
                self.assertEqual(
                    is_valid_attempt_transition(from_state, to_state),
                    valid,
                    f"{from_state.value} -> {to_state.value}",
                )
                if valid:
                    validate_attempt_transition(from_state, to_state)
                else:
                    with self.assertRaisesRegex(
                        AssertionError,
                        "invalid task attempt transition: "
                        f"{from_state.value} -> {to_state.value}",
                    ):
                        validate_attempt_transition(from_state, to_state)

    def test_terminal_run_states_are_immutable(self) -> None:
        self.assertEqual(
            TASK_RUN_TERMINAL_STATES,
            frozenset(
                {
                    TaskRunState.SUCCEEDED,
                    TaskRunState.FAILED,
                    TaskRunState.CANCELLED,
                    TaskRunState.EXPIRED,
                }
            ),
        )
        for state in TaskRunState:
            self.assertEqual(
                is_terminal_run_state(state),
                state in TASK_RUN_TERMINAL_STATES,
            )
        for terminal_state in TASK_RUN_TERMINAL_STATES:
            self.assertEqual(
                valid_run_transitions(terminal_state), frozenset()
            )
            for to_state in TaskRunState:
                self.assertFalse(
                    is_valid_run_transition(terminal_state, to_state)
                )

    def test_terminal_attempt_states_are_immutable(self) -> None:
        self.assertEqual(
            TASK_ATTEMPT_TERMINAL_STATES,
            frozenset(
                {
                    TaskAttemptState.SUCCEEDED,
                    TaskAttemptState.FAILED,
                    TaskAttemptState.ABANDONED,
                }
            ),
        )
        for state in TaskAttemptState:
            self.assertEqual(
                is_terminal_attempt_state(state),
                state in TASK_ATTEMPT_TERMINAL_STATES,
            )
        for terminal_state in TASK_ATTEMPT_TERMINAL_STATES:
            self.assertEqual(
                valid_attempt_transitions(terminal_state),
                frozenset(),
            )
            for to_state in TaskAttemptState:
                self.assertFalse(
                    is_valid_attempt_transition(terminal_state, to_state)
                )

    def test_transition_tables_are_externally_immutable(self) -> None:
        self.assertIsInstance(
            VALID_TASK_RUN_TRANSITIONS,
            MappingProxyType,
        )
        self.assertIsInstance(
            VALID_TASK_ATTEMPT_TRANSITIONS,
            MappingProxyType,
        )
        with self.assertRaises(TypeError):
            cast(
                dict[TaskRunState, frozenset[TaskRunState]],
                (VALID_TASK_RUN_TRANSITIONS),
            )[TaskRunState.SUCCEEDED] = frozenset({TaskRunState.CREATED})
        with self.assertRaises(AttributeError):
            valid_run_transitions(TaskRunState.CREATED).add(
                TaskRunState.EXPIRED
            )

    def test_helpers_reject_wrong_state_types(self) -> None:
        with self.assertRaisesRegex(
            AssertionError,
            "state must be a TaskRunState",
        ):
            is_terminal_run_state(cast(TaskRunState, "created"))
        with self.assertRaisesRegex(
            AssertionError,
            "state must be a TaskAttemptState",
        ):
            is_terminal_attempt_state(cast(TaskAttemptState, "created"))
        with self.assertRaisesRegex(
            AssertionError,
            "from_state must be a TaskRunState",
        ):
            valid_run_transitions(cast(TaskRunState, "created"))
        with self.assertRaisesRegex(
            AssertionError,
            "from_state must be a TaskAttemptState",
        ):
            valid_attempt_transitions(cast(TaskAttemptState, "created"))
        with self.assertRaisesRegex(
            AssertionError,
            "to_state must be a TaskRunState",
        ):
            is_valid_run_transition(
                TaskRunState.CREATED,
                cast(TaskRunState, "validated"),
            )
        with self.assertRaisesRegex(
            AssertionError,
            "to_state must be a TaskAttemptState",
        ):
            is_valid_attempt_transition(
                TaskAttemptState.CREATED,
                cast(TaskAttemptState, "running"),
            )


if __name__ == "__main__":
    main()
