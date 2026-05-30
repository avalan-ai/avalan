from unittest import TestCase, main

from avalan.task import (
    RetryBackoff,
    TaskAttemptDecisionType,
    TaskAttemptPolicy,
    TaskError,
    TaskRetryPolicy,
)


class TaskAttemptPolicyTest(TestCase):
    def test_retryable_error_retries_until_max_attempts(self) -> None:
        policy = TaskAttemptPolicy.from_retry_policy(
            TaskRetryPolicy(
                max_attempts=3,
                backoff=RetryBackoff.EXPONENTIAL,
                max_delay_seconds=1,
            )
        )

        first = policy.decide(attempt_number=1, error=TaskError.runnable())
        third = policy.decide(attempt_number=3, error=TaskError.runnable())

        self.assertEqual(first.type, TaskAttemptDecisionType.RETRY)
        self.assertTrue(first.should_retry)
        self.assertEqual(first.retry_delay_seconds, 1)
        self.assertEqual(third.type, TaskAttemptDecisionType.FAIL)
        self.assertFalse(third.should_retry)
        self.assertIsNone(third.retry_delay_seconds)

    def test_non_retryable_error_fails_immediately(self) -> None:
        policy = TaskAttemptPolicy(
            max_attempts=3,
            backoff=RetryBackoff.LINEAR,
        )

        decision = policy.decide(
            attempt_number=1,
            error=TaskError.cancellation(),
        )

        self.assertEqual(decision.type, TaskAttemptDecisionType.FAIL)
        self.assertFalse(decision.should_retry)

    def test_retry_delay_matches_backoff_modes(self) -> None:
        self.assertEqual(
            TaskAttemptPolicy(
                max_attempts=3,
                backoff=RetryBackoff.NONE,
            ).retry_delay_seconds(2),
            0,
        )
        self.assertEqual(
            TaskAttemptPolicy(
                max_attempts=3,
                backoff=RetryBackoff.LINEAR,
            ).retry_delay_seconds(2),
            2,
        )
        self.assertEqual(
            TaskAttemptPolicy(
                max_attempts=3,
                backoff=RetryBackoff.EXPONENTIAL,
            ).retry_delay_seconds(3),
            4,
        )


if __name__ == "__main__":
    main()
