from ..types import JsonValue
from .validation import TaskValidationError, TaskValidationIssue

from asyncio import CancelledError
from asyncio import TimeoutError as AsyncTimeoutError
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite
from types import MappingProxyType
from typing import TypeAlias, cast

TaskErrorValue: TypeAlias = JsonValue


class TaskErrorCategory(StrEnum):
    RUNNABLE = "runnable"
    INPUT_CONTRACT = "input_contract"
    OUTPUT_CONTRACT = "output_contract"
    TIMEOUT = "timeout"
    CANCELLATION = "cancellation"
    BUDGET = "budget"
    INFRA = "infra"
    PRIVACY = "privacy"


class TaskErrorCode(StrEnum):
    RUNNABLE_FAILED = "runnable.failed"
    INPUT_CONTRACT_FAILED = "input_contract.failed"
    OUTPUT_CONTRACT_FAILED = "output_contract.failed"
    TIMEOUT_EXCEEDED = "timeout.exceeded"
    CANCELLATION_REQUESTED = "cancellation.requested"
    BUDGET_EXCEEDED = "budget.exceeded"
    INFRA_FAILURE = "infra.failure"
    PRIVACY_FAILURE = "privacy.failure"


def _empty_details() -> Mapping[str, TaskErrorValue]:
    return MappingProxyType({})


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskError:
    category: TaskErrorCategory
    code: TaskErrorCode
    message: str
    retryable: bool = False
    details: Mapping[str, TaskErrorValue] = field(
        default_factory=_empty_details
    )

    def __post_init__(self) -> None:
        assert isinstance(self.category, TaskErrorCategory)
        assert isinstance(self.code, TaskErrorCode)
        assert isinstance(self.message, str) and self.message.strip()
        assert isinstance(self.retryable, bool)
        object.__setattr__(self, "details", _freeze_details(self.details))

    def as_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "category": self.category.value,
            "code": self.code.value,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.details:
            value["details"] = dict(self.details)
        return value

    @classmethod
    def output_contract(
        cls,
        issues: tuple[TaskValidationIssue, ...],
    ) -> "TaskError":
        assert issues, "issues must not be empty"
        return cls(
            category=TaskErrorCategory.OUTPUT_CONTRACT,
            code=TaskErrorCode.OUTPUT_CONTRACT_FAILED,
            message="Task output did not match the declared contract.",
            details={
                "issues": tuple(
                    _validation_issue_detail(issue) for issue in issues
                )
            },
        )

    @classmethod
    def input_contract(
        cls,
        issues: tuple[TaskValidationIssue, ...],
    ) -> "TaskError":
        assert issues, "issues must not be empty"
        return cls(
            category=TaskErrorCategory.INPUT_CONTRACT,
            code=TaskErrorCode.INPUT_CONTRACT_FAILED,
            message="Task input did not match the declared contract.",
            details={
                "issues": tuple(
                    _validation_issue_detail(issue) for issue in issues
                )
            },
        )

    @classmethod
    def runnable(cls) -> "TaskError":
        return cls(
            category=TaskErrorCategory.RUNNABLE,
            code=TaskErrorCode.RUNNABLE_FAILED,
            message="Task runnable failed.",
            retryable=True,
        )

    @classmethod
    def timeout(cls) -> "TaskError":
        return cls(
            category=TaskErrorCategory.TIMEOUT,
            code=TaskErrorCode.TIMEOUT_EXCEEDED,
            message="Task attempt timed out.",
            retryable=True,
        )

    @classmethod
    def cancellation(cls) -> "TaskError":
        return cls(
            category=TaskErrorCategory.CANCELLATION,
            code=TaskErrorCode.CANCELLATION_REQUESTED,
            message="Task attempt was cancelled.",
        )

    @classmethod
    def budget(cls) -> "TaskError":
        return cls(
            category=TaskErrorCategory.BUDGET,
            code=TaskErrorCode.BUDGET_EXCEEDED,
            message="Task attempt exceeded a runtime budget.",
        )

    @classmethod
    def infra(cls) -> "TaskError":
        return cls(
            category=TaskErrorCategory.INFRA,
            code=TaskErrorCode.INFRA_FAILURE,
            message="Task infrastructure failed.",
            retryable=True,
        )

    @classmethod
    def privacy(cls) -> "TaskError":
        return cls(
            category=TaskErrorCategory.PRIVACY,
            code=TaskErrorCode.PRIVACY_FAILURE,
            message="Task privacy policy could not be applied.",
        )


def classify_task_error(error: BaseException) -> TaskError:
    assert isinstance(error, BaseException)
    if isinstance(error, CancelledError):
        return TaskError.cancellation()
    if isinstance(error, AsyncTimeoutError):
        return TaskError.timeout()
    if isinstance(error, TaskValidationError):
        return TaskError.input_contract(error.issues)
    if isinstance(error, MemoryError):
        return TaskError.budget()
    if error.__class__.__name__ == "PrivacySanitizationError":
        return TaskError.privacy()
    if isinstance(error, OSError):
        return TaskError.infra()
    return TaskError.runnable()


def _freeze_details(
    value: Mapping[str, object],
) -> Mapping[str, TaskErrorValue]:
    assert isinstance(value, Mapping), "details must be a mapping"
    return cast(Mapping[str, TaskErrorValue], _freeze_value(value))


def _freeze_value(value: object) -> TaskErrorValue:
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        assert isfinite(value), "error floats must be finite"
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, TaskErrorValue] = {}
        for key, item in value.items():
            assert isinstance(key, str), "error detail keys must be strings"
            assert key.strip(), "error detail keys must not be empty"
            frozen[key] = _freeze_value(item)
        return MappingProxyType(frozen)
    if isinstance(value, list | tuple):
        return tuple(_freeze_value(item) for item in value)
    raise AssertionError("error details must be privacy-safe")


def _validation_issue_detail(
    issue: TaskValidationIssue,
) -> Mapping[str, TaskErrorValue]:
    return MappingProxyType(
        {
            "category": issue.category.value,
            "code": issue.code,
            "path": issue.path,
            "severity": issue.severity.value,
        }
    )
