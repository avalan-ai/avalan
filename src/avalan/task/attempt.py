from ..types import assert_positive_int as _assert_positive_int
from .definition import RetryBackoff, TaskRetryPolicy
from .error import TaskError

from dataclasses import dataclass
from enum import StrEnum


class TaskAttemptDecisionType(StrEnum):
    FAIL = "fail"
    RETRY = "retry"


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskAttemptDecision:
    type: TaskAttemptDecisionType
    attempt_number: int
    max_attempts: int
    retry_delay_seconds: int | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.type, TaskAttemptDecisionType)
        _assert_positive_int(self.attempt_number, "attempt_number")
        _assert_positive_int(self.max_attempts, "max_attempts")
        assert (
            self.attempt_number <= self.max_attempts
        ), "attempt_number must not exceed max_attempts"
        if self.retry_delay_seconds is not None:
            assert isinstance(self.retry_delay_seconds, int)
            assert not isinstance(self.retry_delay_seconds, bool)
            assert self.retry_delay_seconds >= 0
        if self.type == TaskAttemptDecisionType.FAIL:
            assert (
                self.retry_delay_seconds is None
            ), "failed attempts must not include retry delay"

    @property
    def should_retry(self) -> bool:
        return self.type == TaskAttemptDecisionType.RETRY


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskAttemptPolicy:
    max_attempts: int
    backoff: RetryBackoff = RetryBackoff.NONE
    max_delay_seconds: int | None = None

    def __post_init__(self) -> None:
        _assert_positive_int(self.max_attempts, "max_attempts")
        assert isinstance(self.backoff, RetryBackoff)
        if self.max_delay_seconds is not None:
            _assert_positive_int(
                self.max_delay_seconds,
                "max_delay_seconds",
            )

    @classmethod
    def from_retry_policy(
        cls,
        retry_policy: TaskRetryPolicy,
    ) -> "TaskAttemptPolicy":
        assert isinstance(retry_policy, TaskRetryPolicy)
        return cls(
            max_attempts=retry_policy.max_attempts,
            backoff=retry_policy.backoff,
            max_delay_seconds=retry_policy.max_delay_seconds,
        )

    def decide(
        self,
        *,
        attempt_number: int,
        error: TaskError,
    ) -> TaskAttemptDecision:
        _assert_positive_int(attempt_number, "attempt_number")
        assert isinstance(error, TaskError)
        if not error.retryable or attempt_number >= self.max_attempts:
            return TaskAttemptDecision(
                type=TaskAttemptDecisionType.FAIL,
                attempt_number=attempt_number,
                max_attempts=self.max_attempts,
            )
        return TaskAttemptDecision(
            type=TaskAttemptDecisionType.RETRY,
            attempt_number=attempt_number,
            max_attempts=self.max_attempts,
            retry_delay_seconds=self.retry_delay_seconds(attempt_number),
        )

    def retry_delay_seconds(self, attempt_number: int) -> int:
        _assert_positive_int(attempt_number, "attempt_number")
        match self.backoff:
            case RetryBackoff.NONE:
                delay = 0
            case RetryBackoff.LINEAR:
                delay = attempt_number
            case RetryBackoff.EXPONENTIAL:
                delay = 2 ** (attempt_number - 1)
        if self.max_delay_seconds is None:
            return delay
        return min(delay, self.max_delay_seconds)
