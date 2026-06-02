from asyncio import CancelledError, sleep, wait_for
from asyncio import run as asyncio_run
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest import IsolatedAsyncioTestCase, TestCase, main
from unittest.mock import patch

from avalan.flow.flow import Flow
from avalan.flow.node import Node
from avalan.task import (
    STORED_MARKER,
    DirectTaskRunner,
    PrivacyAction,
    TaskArtifactPolicy,
    TaskArtifactPurpose,
    TaskArtifactRef,
    TaskArtifactState,
    TaskDefinition,
    TaskEventCategory,
    TaskExecutionTarget,
    TaskInputContract,
    TaskKeyMaterial,
    TaskKeyPurpose,
    TaskMetadata,
    TaskObservabilityPolicy,
    TaskOutputContract,
    TaskPrivacyPolicy,
    TaskRunPolicy,
    TaskRunState,
    TaskTargetContext,
    TaskValidationContext,
    TaskValidationError,
    TaskValidationIssue,
)
from avalan.task.artifacts import LocalArtifactStore
from avalan.task.store import TaskExecutionContext
from avalan.task.stores import InMemoryTaskStore
from avalan.task.targets import (
    FLOW_TASK_INPUT_KEY,
    FlowTaskTargetRunner,
    flow_task_input_binding,
    validate_flow_task_compatibility,
)


class StaticHmacProvider:
    def hmac_key(
        self,
        *,
        purpose: TaskKeyPurpose,
        key_id: str | None = None,
    ) -> TaskKeyMaterial:
        return TaskKeyMaterial(
            key_id=key_id or purpose.value,
            algorithm="hmac-sha256",
            secret=b"flow-target-secret",
        )


class FlowTaskTargetRunnerValidationTest(TestCase):
    def test_flow_target_accepts_observability_without_path_leaks(
        self,
    ) -> None:
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

        self.assertEqual(issues, ())
        rendered = " ".join(
            value for issue in issues for value in issue.as_dict().values()
        )
        self.assertNotIn("private.toml", rendered)
        self.assertNotIn("private flow", rendered)

    def test_flow_target_reports_file_input_gap_but_allows_artifact_output(
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
            ["input.type"],
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

    def test_compatibility_report_marks_scalar_flow_compatible(self) -> None:
        report = validate_flow_task_compatibility(
            self._definition(),
            TaskValidationContext(),
        )

        self.assertTrue(report.compatible)
        self.assertEqual(report.issues, ())

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
        flow.add_node(
            Node("A", func=lambda inputs: inputs[FLOW_TASK_INPUT_KEY] + "!")
        )
        runner = FlowTaskTargetRunner(flow_resolver=lambda _: flow)

        result = await runner.run(self._context(input_value="ready"))

        self.assertEqual(result, "ready!")

    async def test_run_awaits_async_resolver(self) -> None:
        flow = Flow()
        flow.add_node(
            Node("A", func=lambda inputs: inputs[FLOW_TASK_INPUT_KEY] + 1)
        )

        async def resolver(_: TaskTargetContext) -> Flow:
            await sleep(0)
            return flow

        runner = FlowTaskTargetRunner(flow_resolver=resolver)

        result = await runner.run(
            self._context(
                definition=self._context_definition(
                    input_contract=TaskInputContract.integer(),
                ),
                input_value=1,
            )
        )

        self.assertEqual(result, 2)

    async def test_run_binds_object_fields_and_full_input(self) -> None:
        flow = Flow()
        flow.add_node(
            Node(
                "A",
                func=lambda inputs: {
                    "name": inputs["name"],
                    "limit": inputs[FLOW_TASK_INPUT_KEY]["limit"],
                },
            )
        )
        runner = FlowTaskTargetRunner(flow_resolver=lambda _: flow)

        result = await runner.run(
            self._context(
                definition=self._context_definition(
                    input_contract=TaskInputContract.object(
                        {
                            "type": "object",
                            "required": ["name", "limit"],
                            "additionalProperties": False,
                            "properties": {
                                "name": {"type": "string"},
                                "limit": {"type": "integer"},
                            },
                        }
                    ),
                    output_contract=TaskOutputContract.object(),
                ),
                input_value={"name": "report", "limit": 3},
            )
        )

        self.assertEqual(result, {"name": "report", "limit": 3})

    async def test_run_does_not_trust_reserved_object_input_key(self) -> None:
        flow = Flow()
        flow.add_node(
            Node(
                "A",
                func=lambda inputs: {
                    "reserved": inputs[FLOW_TASK_INPUT_KEY][
                        FLOW_TASK_INPUT_KEY
                    ],
                    "limit": inputs[FLOW_TASK_INPUT_KEY]["limit"],
                },
            )
        )
        runner = FlowTaskTargetRunner(flow_resolver=lambda _: flow)

        result = await runner.run(
            self._context(
                definition=self._context_definition(
                    input_contract=TaskInputContract.object(
                        {
                            "type": "object",
                            "required": [FLOW_TASK_INPUT_KEY, "limit"],
                            "additionalProperties": False,
                            "properties": {
                                FLOW_TASK_INPUT_KEY: {"type": "string"},
                                "limit": {"type": "integer"},
                            },
                        }
                    ),
                    output_contract=TaskOutputContract.object(),
                ),
                input_value={
                    FLOW_TASK_INPUT_KEY: "private spoofed input",
                    "limit": 3,
                },
            )
        )

        self.assertEqual(
            result,
            {"reserved": "private spoofed input", "limit": 3},
        )

    async def test_run_binds_scalar_input_value(self) -> None:
        flow = Flow()
        flow.add_node(
            Node(
                "A",
                func=lambda inputs: {
                    "full": inputs[FLOW_TASK_INPUT_KEY],
                    "value": inputs["value"],
                },
            )
        )
        runner = FlowTaskTargetRunner(flow_resolver=lambda _: flow)

        result = await runner.run(self._context(input_value="ready"))

        self.assertEqual(result, {"full": "ready", "value": "ready"})

    async def test_run_rejects_invalid_input_contract_safely(self) -> None:
        flow = Flow()
        flow.add_node(Node("A", func=lambda _: "unused private output"))
        runner = FlowTaskTargetRunner(flow_resolver=lambda _: flow)

        with self.assertRaises(TaskValidationError) as error:
            await runner.run(
                self._context(
                    definition=self._context_definition(
                        input_contract=TaskInputContract.integer(),
                    ),
                    input_value="private prompt",
                )
            )

        self.assertEqual(error.exception.issues[0].code, "input.invalid_type")
        self.assertNotIn("private prompt", str(error.exception))
        self.assertNotIn("unused private output", str(error.exception))

    async def test_run_unwraps_stored_queued_input(self) -> None:
        flow = Flow()
        flow.add_node(
            Node(
                "A",
                func=lambda inputs: inputs[FLOW_TASK_INPUT_KEY] + "!",
            )
        )
        runner = FlowTaskTargetRunner(flow_resolver=lambda _: flow)

        result = await runner.run(
            self._context(
                definition=self._context_definition(
                    run=TaskRunPolicy.queued("default"),
                ),
                input_value={
                    "privacy": STORED_MARKER,
                    "value": "ready",
                },
            )
        )

        self.assertEqual(result, "ready!")

    async def test_run_rejects_multiple_start_nodes_safely(self) -> None:
        flow = Flow()
        flow.add_node(Node("A", func=lambda _: "private A"))
        flow.add_node(Node("B", func=lambda _: "private B"))
        runner = FlowTaskTargetRunner(flow_resolver=lambda _: flow)

        with self.assertRaises(TaskValidationError) as error:
            await runner.run(self._context(input_value="private prompt"))

        self.assertEqual(
            error.exception.issues[0].code, "execution.unsupported_flow"
        )
        self.assertNotIn("private prompt", str(error.exception))
        self.assertNotIn("private A", str(error.exception))

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
            definition=definition or self._context_definition(),
            execution=TaskExecutionContext(
                run_id="run-1",
                attempt_id="attempt-1",
                attempt_number=1,
            ),
            input_value=input_value,
            cancellation_checker=cancellation_checker,
        )

    def _context_definition(
        self,
        *,
        input_contract: TaskInputContract | None = None,
        output_contract: TaskOutputContract | None = None,
        privacy: TaskPrivacyPolicy | None = None,
        run: TaskRunPolicy | None = None,
    ) -> TaskDefinition:
        return TaskDefinition(
            task=TaskMetadata(name="flow-task", version="1"),
            input=input_contract or TaskInputContract.string(),
            output=output_contract or TaskOutputContract.text(),
            execution=TaskExecutionTarget.flow("flows/report.toml"),
            observability=TaskObservabilityPolicy.noop(),
            privacy=privacy or TaskPrivacyPolicy(),
            run=run or TaskRunPolicy.direct(),
        )


class FlowTaskTargetRunnerE2ETest(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.store = InMemoryTaskStore()

    async def test_direct_runner_commits_object_output_after_validation(
        self,
    ) -> None:
        flow = Flow()
        flow.add_node(
            Node(
                "A",
                func=lambda inputs: {
                    "status": "ready",
                    "count": inputs["limit"],
                },
            )
        )
        runner = DirectTaskRunner(
            self.store,
            target=FlowTaskTargetRunner(flow_resolver=lambda _: flow),
            hmac_provider=StaticHmacProvider(),
            definition_hash=lambda _: "flow-direct-success",
        )

        result = await runner.run(
            self._definition(
                input_contract=self._object_input_contract(),
                output_contract=self._object_output_contract(),
            ),
            input_value={"prompt": "safe summary", "limit": 2},
        )

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(result.output, {"status": "ready", "count": 2})

    async def test_direct_runner_persists_sanitized_flow_events(self) -> None:
        flow = Flow()
        flow.add_node(
            Node(
                "A",
                func=lambda inputs: (
                    inputs[FLOW_TASK_INPUT_KEY] + " private output"
                ),
            )
        )
        runner = DirectTaskRunner(
            self.store,
            target=FlowTaskTargetRunner(flow_resolver=lambda _: flow),
            hmac_provider=StaticHmacProvider(),
            definition_hash=lambda _: "flow-direct-events",
        )

        result = await runner.run(
            self._definition(
                input_contract=TaskInputContract.string(),
                output_contract=TaskOutputContract.text(),
                observability=TaskObservabilityPolicy(),
            ),
            input_value="private prompt",
        )

        events = await self.store.list_events(result.run.run_id)
        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(
            [event.event_type for event in events],
            [
                "flow_manager_call_before",
                "flow_manager_call_after",
            ],
        )
        self.assertEqual(
            [event.category for event in events],
            [TaskEventCategory.ENGINE, TaskEventCategory.ENGINE],
        )
        start_payload = cast(Mapping[str, object], events[0].payload)
        end_payload = cast(Mapping[str, object], events[1].payload)
        self.assertEqual(start_payload["status"], "started")
        self.assertEqual(end_payload["status"], "succeeded")
        self.assertIn("duration_ms", end_payload)
        self.assertNotIn("private prompt", str(events))
        self.assertNotIn("private output", str(events))

    async def test_direct_runner_persists_failed_flow_event_safely(
        self,
    ) -> None:
        def fail(_: dict[str, object]) -> str:
            raise RuntimeError("private node failure")

        flow = Flow()
        flow.add_node(Node("A", func=fail))
        runner = DirectTaskRunner(
            self.store,
            target=FlowTaskTargetRunner(flow_resolver=lambda _: flow),
            hmac_provider=StaticHmacProvider(),
            definition_hash=lambda _: "flow-direct-failed-event",
        )

        result = await runner.run(
            self._definition(
                input_contract=TaskInputContract.string(),
                output_contract=TaskOutputContract.text(),
                observability=TaskObservabilityPolicy(),
            ),
            input_value="private prompt",
        )

        events = await self.store.list_events(result.run.run_id)
        self.assertEqual(result.run.state, TaskRunState.FAILED)
        self.assertEqual(events[1].event_type, "flow_manager_call_after")
        end_payload = cast(Mapping[str, object], events[1].payload)
        self.assertEqual(end_payload["status"], "failed")
        self.assertNotIn("private node failure", str(events))
        self.assertNotIn("private prompt", str(events))

    async def test_direct_runner_records_flow_file_output_artifact(
        self,
    ) -> None:
        with TemporaryDirectory() as artifacts:
            artifact_store = LocalArtifactStore(
                artifacts,
                raw_storage_allowed=True,
                id_factory=lambda: "flow-output-1",
            )

            async def produce_artifact(
                _: dict[str, object],
            ) -> TaskArtifactRef:
                return await artifact_store.put(
                    b"private flow bytes",
                    media_type="text/plain",
                    metadata={"filename": "private-flow.txt"},
                )

            flow = Flow()
            flow.add_node(Node("A", func=produce_artifact))
            runner = DirectTaskRunner(
                self.store,
                target=FlowTaskTargetRunner(flow_resolver=lambda _: flow),
                hmac_provider=StaticHmacProvider(),
                artifact_store=artifact_store,
                definition_hash=lambda _: "flow-direct-artifact-output",
            )

            result = await runner.run(
                self._definition(
                    input_contract=TaskInputContract.string(),
                    output_contract=TaskOutputContract.file(),
                    artifact=TaskArtifactPolicy.references_only(
                        retention_days=6,
                    ),
                    privacy=TaskPrivacyPolicy(output=PrivacyAction.REDACT),
                ),
                input_value="private prompt",
            )

        records = await self.store.list_artifacts(
            result.run.run_id,
            purpose=TaskArtifactPurpose.OUTPUT,
        )
        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].state, TaskArtifactState.READY)
        self.assertEqual(records[0].retention.delete_after_days, 6)
        self.assertEqual(records[0].ref.metadata, {"privacy": "<redacted>"})
        self.assertNotIn("private-flow.txt", str(records))
        self.assertNotIn("private flow bytes", str(records))
        self.assertNotIn("private prompt", str(records))

    async def test_direct_runner_rejects_invalid_flow_artifact_output(
        self,
    ) -> None:
        flow = Flow()
        flow.add_node(Node("A", func=lambda _: object()))
        runner = DirectTaskRunner(
            self.store,
            target=FlowTaskTargetRunner(flow_resolver=lambda _: flow),
            hmac_provider=StaticHmacProvider(),
            definition_hash=lambda _: "flow-direct-invalid-artifact-output",
        )

        result = await runner.run(
            self._definition(
                input_contract=TaskInputContract.string(),
                output_contract=TaskOutputContract.file(),
            ),
            input_value="private prompt",
        )

        self.assertEqual(result.run.state, TaskRunState.FAILED)
        self.assertEqual(
            await self.store.list_artifacts(
                result.run.run_id,
                purpose=TaskArtifactPurpose.OUTPUT,
            ),
            (),
        )
        self.assertNotIn("private prompt", str(result.run.result))

    async def test_direct_runner_rejects_invalid_flow_output(self) -> None:
        flow = Flow()
        flow.add_node(
            Node(
                "A",
                func=lambda inputs: {
                    "status": "ready",
                    "count": "private invalid count",
                },
            )
        )
        runner = DirectTaskRunner(
            self.store,
            target=FlowTaskTargetRunner(flow_resolver=lambda _: flow),
            hmac_provider=StaticHmacProvider(),
            definition_hash=lambda _: "flow-direct-invalid-output",
        )

        result = await runner.run(
            self._definition(
                input_contract=self._object_input_contract(),
                output_contract=self._object_output_contract(),
            ),
            input_value={"prompt": "private prompt", "limit": 1},
        )

        self.assertEqual(result.run.state, TaskRunState.FAILED)
        self.assertIsNone(result.output)
        self.assertNotIn("private invalid count", str(result.run.result))
        self.assertNotIn("private prompt", str(result.run.result))

    def test_input_binding_exposes_scalar_array_and_object_shapes(
        self,
    ) -> None:
        self.assertEqual(
            flow_task_input_binding("ready"),
            {FLOW_TASK_INPUT_KEY: "ready", "value": "ready"},
        )
        self.assertEqual(
            flow_task_input_binding([1, 2]),
            {FLOW_TASK_INPUT_KEY: (1, 2), "items": (1, 2)},
        )
        self.assertEqual(
            flow_task_input_binding({"limit": 2}),
            {FLOW_TASK_INPUT_KEY: {"limit": 2}, "limit": 2},
        )
        self.assertEqual(
            flow_task_input_binding(
                {
                    FLOW_TASK_INPUT_KEY: "spoofed",
                    "limit": 2,
                }
            ),
            {
                FLOW_TASK_INPUT_KEY: {
                    FLOW_TASK_INPUT_KEY: "spoofed",
                    "limit": 2,
                },
                "limit": 2,
            },
        )

    def test_flow_validator_requires_structured_output_schema(self) -> None:
        report = validate_flow_task_compatibility(
            self._definition(
                input_contract=self._object_input_contract(),
                output_contract=TaskOutputContract.object(),
            ),
            TaskValidationContext(),
        )

        self.assertFalse(report.compatible)
        self.assertEqual(
            [issue.path for issue in report.issues],
            ["output.schema"],
        )
        self.assertNotIn("private", str(report.issues))

    def _definition(
        self,
        *,
        input_contract: TaskInputContract,
        output_contract: TaskOutputContract,
        artifact: TaskArtifactPolicy | None = None,
        observability: TaskObservabilityPolicy | None = None,
        privacy: TaskPrivacyPolicy | None = None,
    ) -> TaskDefinition:
        return TaskDefinition(
            task=TaskMetadata(name="flow-task", version="1"),
            input=input_contract,
            output=output_contract,
            execution=TaskExecutionTarget.flow("flows/report.toml"),
            artifact=artifact or TaskArtifactPolicy.references_only(),
            observability=observability or TaskObservabilityPolicy.noop(),
            privacy=privacy or TaskPrivacyPolicy(),
        )

    def _object_input_contract(self) -> TaskInputContract:
        return TaskInputContract.object(
            {
                "type": "object",
                "required": ["prompt", "limit"],
                "additionalProperties": False,
                "properties": {
                    "prompt": {"type": "string", "minLength": 1},
                    "limit": {"type": "integer", "minimum": 1},
                },
            }
        )

    def _object_output_contract(self) -> TaskOutputContract:
        return TaskOutputContract.object(
            {
                "type": "object",
                "required": ["status", "count"],
                "additionalProperties": False,
                "properties": {
                    "status": {"type": "string", "enum": ["ready"]},
                    "count": {"type": "integer", "minimum": 1},
                },
            }
        )


if __name__ == "__main__":
    main()
