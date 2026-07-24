"""Define typed terminal outcomes for resumed durable tasks."""

from ..interaction.error import InputErrorCode, InputValidationError
from .store import TaskExecutionResult

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from json import dumps
from typing import NoReturn, final


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class TaskDurableResumeSuccess:
    """Describe one privacy-safe successful resumed task settlement."""

    result: TaskExecutionResult

    def __post_init__(self) -> None:
        if not isinstance(self.result, TaskExecutionResult):
            _invalid_type(
                "task_resume.settlement.result",
                "a task execution result",
            )
        if self.result.error is not None:
            _illegal_transition(
                "task_resume.settlement.result.error",
                "successful settlement cannot retain an error",
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class TaskDurableResumeFailure:
    """Describe one privacy-safe deterministic resumed task failure."""

    result: TaskExecutionResult

    def __post_init__(self) -> None:
        if not isinstance(self.result, TaskExecutionResult):
            _invalid_type(
                "task_resume.settlement.result",
                "a task execution result",
            )
        if self.result.error is None:
            _illegal_transition(
                "task_resume.settlement.result.error",
                "failed settlement requires a sanitized error",
            )
        if self.result.output_summary is not None:
            _illegal_transition(
                "task_resume.settlement.result.output_summary",
                "failed settlement cannot retain successful output",
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class TaskDurableResumeCancellation:
    """Describe one privacy-safe cancelled resumed task settlement."""

    result: TaskExecutionResult

    def __post_init__(self) -> None:
        if not isinstance(self.result, TaskExecutionResult):
            _invalid_type(
                "task_resume.settlement.result",
                "a task execution result",
            )
        if self.result.error is None:
            _illegal_transition(
                "task_resume.settlement.result.error",
                "cancelled settlement requires a sanitized error",
            )
        if self.result.output_summary is not None:
            _illegal_transition(
                "task_resume.settlement.result.output_summary",
                "cancelled settlement cannot retain successful output",
            )


TaskDurableResumeSettlement = (
    TaskDurableResumeSuccess
    | TaskDurableResumeFailure
    | TaskDurableResumeCancellation
)


def task_durable_resume_settlement_digest(
    settlement: TaskDurableResumeSettlement,
) -> str:
    """Return a canonical digest for one typed terminal settlement."""
    if type(settlement) is TaskDurableResumeSuccess:
        kind = "succeeded"
    elif type(settlement) is TaskDurableResumeFailure:
        kind = "failed"
    elif type(settlement) is TaskDurableResumeCancellation:
        kind = "cancelled"
    else:
        _invalid_type(
            "task_resume.settlement",
            "a durable resume success or failure",
        )
    result = settlement.result
    encoded = dumps(
        _thaw(
            {
                "kind": kind,
                "result": {
                    "error": result.error,
                    "metadata": result.metadata,
                    "output_summary": result.output_summary,
                },
            }
        ),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(encoded.encode("utf-8")).hexdigest()


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _invalid_type(path: str, expected: str) -> NoReturn:
    raise InputValidationError(
        InputErrorCode.INVALID_TYPE,
        path,
        f"value must be {expected}",
    )


def _illegal_transition(path: str, reason: str) -> NoReturn:
    raise InputValidationError(
        InputErrorCode.ILLEGAL_TRANSITION,
        path,
        reason,
    )
