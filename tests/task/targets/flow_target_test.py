from asyncio import run as asyncio_run
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, TestCase, main
from unittest.mock import patch

from avalan.task import (
    TaskDefinition,
    TaskExecutionTarget,
    TaskInputContract,
    TaskMetadata,
    TaskOutputContract,
    TaskTargetContext,
    TaskValidationContext,
    TaskValidationError,
    TaskValidationIssue,
)
from avalan.task.store import TaskExecutionContext
from avalan.task.targets import (
    FlowTaskTargetRunner,
    validate_flow_task_compatibility,
)


class FlowTaskTargetRunnerValidationTest(TestCase):
    def test_flow_target_reports_runtime_incompatibilities(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            flow_path = root / "flows" / "private.toml"
            flow_path.parent.mkdir()
            flow_path.write_text("secret = 'private flow'\n", encoding="utf-8")
            runner = FlowTaskTargetRunner(ref_base=root)

            issues = self._run_validate(
                runner,
                self._definition(),
                TaskValidationContext(execution_roots=(root,)),
            )

        self.assertEqual(
            [issue.code for issue in issues],
            [
                "execution.unsupported_flow",
                "execution.unsupported_flow",
                "execution.unsupported_flow",
                "execution.unsupported_flow",
            ],
        )
        self.assertEqual(
            [issue.path for issue in issues],
            [
                "execution.async",
                "execution.cancellation",
                "run.timeout_seconds",
                "observability.capture_events",
            ],
        )
        rendered = " ".join(
            value for issue in issues for value in issue.as_dict().values()
        )
        self.assertNotIn("private.toml", rendered)
        self.assertNotIn("private flow", rendered)

    def test_flow_target_reports_file_and_artifact_contract_gaps(
        self,
    ) -> None:
        runner = FlowTaskTargetRunner()

        issues = self._run_validate(
            runner,
            self._definition(
                input_contract=TaskInputContract.file(),
                output_contract=TaskOutputContract.artifact_array(),
            ),
            TaskValidationContext(),
        )

        self.assertEqual(
            [issue.path for issue in issues],
            [
                "input.type",
                "output.type",
                "execution.async",
                "execution.cancellation",
                "run.timeout_seconds",
                "observability.capture_events",
            ],
        )
        self.assertTrue(
            all(issue.code == "execution.unsupported_flow" for issue in issues)
        )

    def test_flow_target_rejects_unsafe_references_without_raw_ref(
        self,
    ) -> None:
        runner = FlowTaskTargetRunner()
        cases = (
            "../secret/private.toml",
            "/secret/private.toml",
            "flows\\private.toml",
            "https://example.test/private.toml",
        )

        for ref in cases:
            with self.subTest(ref=ref):
                issues = self._run_validate(
                    runner,
                    self._definition(
                        execution=TaskExecutionTarget.flow(ref),
                    ),
                    TaskValidationContext(execution_roots=(Path("/tmp"),)),
                )
                self.assertEqual(issues[0].code, "execution.path_escape")
                rendered = " ".join(
                    value
                    for issue in issues
                    for value in issue.as_dict().values()
                )
                self.assertNotIn("private.toml", rendered)
                self.assertNotIn(ref, rendered)

    def test_flow_target_rejects_symlink_escape(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "allowed"
            outside = Path(tmp) / "outside"
            root.mkdir()
            outside.mkdir()
            (outside / "private.toml").write_text(
                "secret = 'outside'\n",
                encoding="utf-8",
            )
            symlink = root / "flow.toml"
            symlink.symlink_to(outside / "private.toml")
            runner = FlowTaskTargetRunner(ref_base=root)

            issues = self._run_validate(
                runner,
                self._definition(
                    execution=TaskExecutionTarget.flow("flow.toml"),
                ),
                TaskValidationContext(execution_roots=(root,)),
            )

        self.assertEqual(issues[0].code, "execution.path_escape")
        self.assertEqual(issues[0].path, "execution.ref")
        rendered = " ".join(
            value for issue in issues for value in issue.as_dict().values()
        )
        self.assertNotIn("private.toml", rendered)
        self.assertNotIn("outside", rendered)

    def test_flow_target_fails_closed_when_reference_cannot_resolve(
        self,
    ) -> None:
        runner = FlowTaskTargetRunner()

        with patch(
            "avalan.task.targets.flow.Path.resolve",
            side_effect=(Path("/tmp/allowed"), OSError("resolver secret")),
        ):
            issues = self._run_validate(
                runner,
                self._definition(),
                TaskValidationContext(execution_roots=(Path("/tmp/allowed"),)),
            )

        self.assertEqual(issues[0].code, "execution.path_escape")
        rendered = " ".join(
            value for issue in issues for value in issue.as_dict().values()
        )
        self.assertNotIn("resolver secret", rendered)
        self.assertNotIn("report.toml", rendered)

    def test_non_flow_target_returns_unknown_target_issue(self) -> None:
        runner = FlowTaskTargetRunner()

        issues = self._run_validate(
            runner,
            self._definition(
                execution=TaskExecutionTarget.agent("agents/valid.toml"),
            ),
            TaskValidationContext(),
        )

        self.assertEqual(
            [issue.code for issue in issues],
            ["execution.unknown_target"],
        )
        self.assertEqual(issues[0].path, "execution.type")

    def test_compatibility_report_marks_scalar_flow_incompatible(self) -> None:
        report = validate_flow_task_compatibility(
            self._definition(),
            TaskValidationContext(),
        )

        self.assertFalse(report.compatible)
        self.assertEqual(len(report.issues), 4)

    def _run_validate(
        self,
        runner: FlowTaskTargetRunner,
        definition: TaskDefinition,
        context: TaskValidationContext,
    ) -> tuple[TaskValidationIssue, ...]:
        return asyncio_run(runner.validate_definition(definition, context))

    def _definition(
        self,
        *,
        input_contract: TaskInputContract | None = None,
        output_contract: TaskOutputContract | None = None,
        execution: TaskExecutionTarget | None = None,
    ) -> TaskDefinition:
        return TaskDefinition(
            task=TaskMetadata(name="flow-task", version="1"),
            input=input_contract or TaskInputContract.string(),
            output=output_contract or TaskOutputContract.text(),
            execution=execution
            or TaskExecutionTarget.flow("flows/report.toml"),
        )


class FlowTaskTargetRunnerExecutionTest(IsolatedAsyncioTestCase):
    async def test_run_fails_closed(self) -> None:
        runner = FlowTaskTargetRunner()

        with self.assertRaises(TaskValidationError) as error:
            await runner.run(
                TaskTargetContext(
                    definition=TaskDefinition(
                        task=TaskMetadata(name="flow-task", version="1"),
                        input=TaskInputContract.string(),
                        output=TaskOutputContract.text(),
                        execution=TaskExecutionTarget.flow(
                            "flows/report.toml"
                        ),
                    ),
                    execution=TaskExecutionContext(
                        run_id="run-1",
                        attempt_id="attempt-1",
                        attempt_number=1,
                    ),
                    input_value="private prompt",
                )
            )

        self.assertEqual(
            [issue.code for issue in error.exception.issues],
            ["execution.unsupported_flow"],
        )
        self.assertNotIn("private prompt", str(error.exception))


if __name__ == "__main__":
    main()
