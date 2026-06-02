from asyncio import CancelledError, sleep, wait_for
from asyncio import run as asyncio_run
from collections.abc import Awaitable, Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest import IsolatedAsyncioTestCase, TestCase, main
from unittest.mock import patch

from avalan.flow.flow import Flow
from avalan.flow.node import Node
from avalan.task import (
    TaskDefinition,
    TaskExecutionTarget,
    TaskInputContract,
    TaskMetadata,
    TaskObservabilityPolicy,
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
    def test_flow_target_reports_observability_gap(self) -> None:
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
            ["execution.unsupported_flow"],
        )
        self.assertEqual(
            [issue.path for issue in issues],
            ["observability.capture_events"],
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
        self.assertEqual(len(report.issues), 1)

    def test_compatibility_report_marks_noop_scalar_flow_compatible(
        self,
    ) -> None:
        report = validate_flow_task_compatibility(
            self._definition(observability=TaskObservabilityPolicy.noop()),
            TaskValidationContext(),
        )

        self.assertTrue(report.compatible)
        self.assertEqual(report.issues, ())

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
        observability: TaskObservabilityPolicy | None = None,
    ) -> TaskDefinition:
        return TaskDefinition(
            task=TaskMetadata(name="flow-task", version="1"),
            input=input_contract or TaskInputContract.string(),
            output=output_contract or TaskOutputContract.text(),
            execution=execution
            or TaskExecutionTarget.flow("flows/report.toml"),
            observability=observability or TaskObservabilityPolicy(),
        )


class FlowTaskTargetRunnerExecutionTest(IsolatedAsyncioTestCase):
    async def test_run_fails_closed_without_resolver(self) -> None:
        runner = FlowTaskTargetRunner()

        with self.assertRaises(TaskValidationError) as error:
            await runner.run(self._context(input_value="private prompt"))

        self.assertEqual(
            [issue.code for issue in error.exception.issues],
            ["execution.unsupported_flow"],
        )
        self.assertNotIn("private prompt", str(error.exception))

    async def test_run_executes_resolved_flow(self) -> None:
        flow = Flow()
        flow.add_node(Node("A", func=lambda inputs: inputs["__init__"] + "!"))
        runner = FlowTaskTargetRunner(flow_resolver=lambda _: flow)

        result = await runner.run(self._context(input_value="ready"))

        self.assertEqual(result, "ready!")

    async def test_run_awaits_async_resolver(self) -> None:
        flow = Flow()
        flow.add_node(Node("A", func=lambda inputs: inputs["__init__"] + 1))

        async def resolver(_: TaskTargetContext) -> Flow:
            await sleep(0)
            return flow

        runner = FlowTaskTargetRunner(flow_resolver=resolver)

        result = await runner.run(self._context(input_value=1))

        self.assertEqual(result, 2)

    async def test_run_rejects_invalid_resolver_result_safely(self) -> None:
        def resolver(_: TaskTargetContext) -> Flow:
            return cast(Flow, "private flow")

        runner = FlowTaskTargetRunner(flow_resolver=resolver)

        with self.assertRaises(TaskValidationError) as error:
            await runner.run(self._context(input_value="private prompt"))

        self.assertEqual(
            [issue.path for issue in error.exception.issues],
            ["execution.ref"],
        )
        self.assertNotIn("private prompt", str(error.exception))
        self.assertNotIn("private flow", str(error.exception))

    async def test_run_checks_cancellation_before_success(self) -> None:
        executed: list[str] = []

        def start(_: dict[str, object]) -> str:
            executed.append("A")
            return "done"

        async def cancel_after_node() -> None:
            if executed == ["A"]:
                raise CancelledError()

        flow = Flow()
        flow.add_node(Node("A", func=start))
        runner = FlowTaskTargetRunner(flow_resolver=lambda _: flow)

        with self.assertRaises(CancelledError):
            await runner.run(
                self._context(
                    input_value="private prompt",
                    cancellation_checker=cancel_after_node,
                )
            )

        self.assertEqual(executed, ["A"])

    async def test_run_timeout_covers_flow_work(self) -> None:
        async def slow(_: dict[str, object]) -> str:
            await sleep(0.05)
            return "done"

        flow = Flow()
        flow.add_node(Node("A", func=slow))
        runner = FlowTaskTargetRunner(flow_resolver=lambda _: flow)

        with self.assertRaises(TimeoutError):
            await wait_for(
                runner.run(self._context(input_value="private prompt")),
                timeout=0.001,
            )

    async def test_run_rejects_non_flow_definition(self) -> None:
        runner = FlowTaskTargetRunner(flow_resolver=lambda _: Flow())

        with self.assertRaises(TaskValidationError) as error:
            await runner.run(
                self._context(
                    definition=TaskDefinition(
                        task=TaskMetadata(name="flow-task", version="1"),
                        input=TaskInputContract.string(),
                        output=TaskOutputContract.text(),
                        execution=TaskExecutionTarget.agent(
                            "agents/valid.toml"
                        ),
                    )
                )
            )

        self.assertEqual(
            [issue.code for issue in error.exception.issues],
            ["execution.unknown_target"],
        )

    def _context(
        self,
        *,
        definition: TaskDefinition | None = None,
        input_value: object = None,
        cancellation_checker: Callable[[], Awaitable[None]] | None = None,
    ) -> TaskTargetContext:
        return TaskTargetContext(
            definition=definition
            or TaskDefinition(
                task=TaskMetadata(name="flow-task", version="1"),
                input=TaskInputContract.string(),
                output=TaskOutputContract.text(),
                execution=TaskExecutionTarget.flow("flows/report.toml"),
                observability=TaskObservabilityPolicy.noop(),
            ),
            execution=TaskExecutionContext(
                run_id="run-1",
                attempt_id="attempt-1",
                attempt_number=1,
            ),
            input_value=input_value,
            cancellation_checker=cancellation_checker,
        )


if __name__ == "__main__":
    main()
