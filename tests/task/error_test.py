from asyncio import CancelledError
from asyncio import TimeoutError as AsyncTimeoutError
from math import inf
from types import MappingProxyType
from typing import cast
from unittest import TestCase, main

from avalan.task import (
    PrivacySanitizationError,
    TaskError,
    TaskErrorCategory,
    TaskErrorCode,
    TaskValidationCategory,
    TaskValidationError,
    TaskValidationIssue,
    classify_task_error,
)


class TaskErrorTest(TestCase):
    def test_task_error_serializes_low_cardinality_fields(self) -> None:
        error = TaskError(
            category=TaskErrorCategory.INFRA,
            code=TaskErrorCode.INFRA_FAILURE,
            message="Task infrastructure failed.",
            retryable=True,
            details={
                "attempt": 1,
                "duration_ms": 1.5,
                "nested": {"state": "running"},
            },
        )

        self.assertEqual(
            error.as_dict(),
            {
                "category": "infra",
                "code": "infra.failure",
                "details": {
                    "attempt": 1,
                    "duration_ms": 1.5,
                    "nested": MappingProxyType({"state": "running"}),
                },
                "message": "Task infrastructure failed.",
                "retryable": True,
            },
        )
        with self.assertRaises(TypeError):
            cast(dict[str, object], error.details)["raw"] = "leak"

    def test_output_contract_error_uses_safe_issue_details(self) -> None:
        issue = TaskValidationIssue(
            code="output.invalid_type",
            path="output",
            message="raw output should not be copied",
            hint="Return a text value.",
            category=TaskValidationCategory.VALUE,
        )

        error = TaskError.output_contract((issue,))

        self.assertEqual(error.category, TaskErrorCategory.OUTPUT_CONTRACT)
        self.assertEqual(error.code, TaskErrorCode.OUTPUT_CONTRACT_FAILED)
        self.assertNotIn("raw output", str(error.as_dict()))
        self.assertIn("output.invalid_type", str(error.as_dict()))

    def test_classifier_maps_failures_to_stable_categories(self) -> None:
        cases = (
            (RuntimeError("secret"), TaskErrorCategory.RUNNABLE),
            (AsyncTimeoutError("secret"), TaskErrorCategory.TIMEOUT),
            (CancelledError("secret"), TaskErrorCategory.CANCELLATION),
            (
                TaskValidationError(
                    (
                        TaskValidationIssue(
                            code="input.invalid_file",
                            path="input",
                            message="private input is unavailable",
                            hint="Pass a readable input.",
                            category=TaskValidationCategory.VALUE,
                        ),
                    )
                ),
                TaskErrorCategory.INPUT_CONTRACT,
            ),
            (
                TaskValidationError(
                    (
                        TaskValidationIssue(
                            code="output.invalid_type",
                            path="output",
                            message="private output is invalid",
                            hint="Return a valid output.",
                            category=TaskValidationCategory.VALUE,
                        ),
                    )
                ),
                TaskErrorCategory.OUTPUT_CONTRACT,
            ),
            (MemoryError("secret"), TaskErrorCategory.BUDGET),
            (OSError("secret"), TaskErrorCategory.INFRA),
            (
                PrivacySanitizationError("secret"),
                TaskErrorCategory.PRIVACY,
            ),
        )

        for exception, category in cases:
            with self.subTest(category=category):
                error = classify_task_error(exception)

                self.assertEqual(error.category, category)
                self.assertNotIn("secret", str(error.as_dict()))

    def test_error_details_reject_unsafe_values(self) -> None:
        with self.assertRaises(AssertionError):
            TaskError(
                category=TaskErrorCategory.RUNNABLE,
                code=TaskErrorCode.RUNNABLE_FAILED,
                message="Task runnable failed.",
                details={"bad": object()},
            )
        with self.assertRaises(AssertionError):
            TaskError(
                category=TaskErrorCategory.RUNNABLE,
                code=TaskErrorCode.RUNNABLE_FAILED,
                message="Task runnable failed.",
                details={"bad": inf},
            )


if __name__ == "__main__":
    main()
