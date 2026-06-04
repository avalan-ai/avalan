from asyncio import CancelledError, sleep, wait_for
from asyncio import run as asyncio_run
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase, TestCase, main
from unittest.mock import patch

from avalan.entities import (
    Message,
    MessageContentFile,
    MessageContentText,
    MessageRole,
)
from avalan.event import Event, EventType
from avalan.flow import FlowNodeDefinition
from avalan.flow.flow import Flow
from avalan.flow.loader import FlowDefinitionLoader
from avalan.flow.node import Node
from avalan.task import (
    DROPPED_MARKER,
    ENCRYPTED_MARKER,
    HASHED_MARKER,
    REDACTED_MARKER,
    STORED_ENVELOPE_MARKER,
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
    TaskFileDescriptor,
    TaskInputContract,
    TaskInputFile,
    TaskKeyMaterial,
    TaskKeyPurpose,
    TaskMetadata,
    TaskObservabilityPolicy,
    TaskOutputContract,
    TaskPrivacyPolicy,
    TaskProviderReference,
    TaskProviderReferenceKind,
    TaskRunPolicy,
    TaskRunState,
    TaskTargetContext,
    TaskValidationCategory,
    TaskValidationContext,
    TaskValidationError,
    TaskValidationIssue,
)
from avalan.task.artifacts import LocalArtifactStore
from avalan.task.store import TaskExecutionContext
from avalan.task.stores import InMemoryTaskStore
from avalan.task.targets import (
    FLOW_TASK_FILES_KEY,
    FLOW_TASK_INPUT_KEY,
    AgentTaskTargetRunner,
    FlowTaskTargetRunner,
    flow_task_input_binding,
    task_flow_node_registry,
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


class FlowAgentResponse:
    input_token_count = 5
    output_token_count = 3
    total_token_count = 8

    def __init__(self, text: str) -> None:
        self.text = text

    async def to_str(self) -> str:
        return self.text


class FlowAgentEventManager:
    def __init__(self) -> None:
        self.listeners: list[Callable[[Event], Awaitable[None] | None]] = []

    def add_listener(
        self,
        listener: Callable[[Event], Awaitable[None] | None],
    ) -> None:
        self.listeners.append(listener)

    def remove_listener(
        self,
        listener: Callable[[Event], Awaitable[None] | None],
    ) -> None:
        self.listeners.remove(listener)

    async def emit(self) -> None:
        for listener in tuple(self.listeners):
            result = listener(
                Event(
                    type=EventType.TOKEN_GENERATED,
                    payload={"token": "private-token", "file_id": "file-123"},
                )
            )
            if result is not None:
                await result


class FlowAgentOrchestrator:
    def __init__(self, loader: "FlowAgentLoader") -> None:
        self._loader = loader
        self.event_manager = loader.event_manager

    async def __aenter__(self) -> "FlowAgentOrchestrator":
        self._loader.entered += 1
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> bool | None:
        self._loader.exited += 1
        return None

    async def __call__(self, input: object) -> FlowAgentResponse:
        self._loader.inputs.append(input)
        await self.event_manager.emit()
        if self._loader.error is not None:
            raise self._loader.error
        return FlowAgentResponse(self._loader.text)


class FlowAgentLoader:
    def __init__(
        self,
        *,
        text: str = "flow agent summary",
        error: BaseException | None = None,
    ) -> None:
        self.text = text
        self.error = error
        self.event_manager = FlowAgentEventManager()
        self.paths: list[str] = []
        self.inputs: list[object] = []
        self.entered = 0
        self.exited = 0

    async def from_file(
        self,
        path: str,
        *,
        agent_id: object | None,
        disable_memory: bool = False,
        uri: str | None = None,
        tool_settings: object | None = None,
    ) -> FlowAgentOrchestrator:
        _ = agent_id, disable_memory, uri, tool_settings
        self.paths.append(path)
        return FlowAgentOrchestrator(self)


class CapturingTaskTargetRunner:
    def __init__(
        self,
        *,
        issues: tuple[TaskValidationIssue, ...] = (),
        output: object = "agent output",
    ) -> None:
        self.issues = issues
        self.output = output
        self.contexts: list[TaskTargetContext] = []

    async def validate_definition(
        self,
        definition: TaskDefinition,
        context: TaskValidationContext,
    ) -> tuple[TaskValidationIssue, ...]:
        _ = definition, context
        return self.issues

    async def run(self, context: TaskTargetContext) -> object:
        self.contexts.append(context)
        return self.output


class FailingArtifactStore:
    async def stat(self, ref: TaskArtifactRef) -> object:
        _ = ref
        raise AssertionError("private artifact metadata was read")

    async def open_stream(
        self,
        ref: TaskArtifactRef,
        *,
        max_bytes: int | None = None,
    ) -> object:
        _ = ref, max_bytes
        raise AssertionError("private bytes were read")


def _write_agent_flow_workspace(
    root: Path,
    *,
    uri: str = "ai://env:KEY@openai/gpt-4o-mini",
) -> Path:
    agents = root / "agents"
    agents.mkdir()
    (agents / "review.toml").write_text(
        f"""
[agent]
name = "Flow reviewer"
task = "Review the supplied file."
user = "Review."

[engine]
uri = "{uri}"
""",
        encoding="utf-8",
    )
    flow_path = root / "flow.toml"
    flow_path.write_text(
        """
[flow]
name = "file_review"
entrypoint = "review"
output_node = "review"

[flow.input]
name = "input"
type = "file"
mime_types = ["application/pdf"]

[flow.output]
name = "summary"
type = "text"

[nodes.review]
type = "agent"
ref = "agents/review.toml"
input = "__task_input__"
""",
        encoding="utf-8",
    )
    return flow_path


def _flow_loader_resolver(
    path: Path,
    *,
    agent_runner: AgentTaskTargetRunner,
    root: Path,
) -> Callable[[TaskTargetContext], Flow]:
    def resolve(context: TaskTargetContext) -> Flow:
        result = FlowDefinitionLoader(
            registry=task_flow_node_registry(
                context,
                agent_runner=agent_runner,
                execution_roots=(root,),
            )
        ).load_result(path)
        assert result.flow is not None, result.issues
        return result.flow

    return resolve


def _agent_node_flow(
    context: TaskTargetContext,
    *,
    agent_runner: CapturingTaskTargetRunner,
) -> Flow:
    registry = task_flow_node_registry(context, agent_runner=agent_runner)
    flow = Flow()
    flow.add_node(
        registry.build(
            FlowNodeDefinition(
                name="review",
                type="agent",
                ref="agents/review.toml",
            )
        )
    )
    return flow


def _single_incoming_agent_flow(
    context: TaskTargetContext,
    *,
    agent_runner: CapturingTaskTargetRunner,
) -> Flow:
    registry = task_flow_node_registry(context, agent_runner=agent_runner)
    flow = Flow()
    flow.add_node(Node("start", func=lambda _: "ready"))
    flow.add_node(
        registry.build(
            FlowNodeDefinition(
                name="review",
                type="agent",
                ref="agents/review.toml",
            )
        )
    )
    flow.add_connection("start", "review")
    return flow


def _multi_incoming_agent_flow(
    context: TaskTargetContext,
    *,
    agent_runner: CapturingTaskTargetRunner,
) -> Flow:
    registry = task_flow_node_registry(context, agent_runner=agent_runner)
    flow = Flow()
    flow.add_node(Node("start", func=lambda _: "seed"))
    flow.add_node(Node("left", func=lambda _: "left"))
    flow.add_node(Node("right", func=lambda _: "right"))
    flow.add_node(
        registry.build(
            FlowNodeDefinition(
                name="review",
                type="agent",
                ref="agents/review.toml",
            )
        )
    )
    flow.add_connection("start", "left")
    flow.add_connection("start", "right")
    flow.add_connection("left", "review")
    flow.add_connection("right", "review")
    return flow


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

    def test_flow_target_accepts_file_input_and_artifact_output(
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

        self.assertEqual(issues, ())

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

    async def test_run_rejects_flow_agent_node_validation_issues(
        self,
    ) -> None:
        agent_runner = CapturingTaskTargetRunner(
            issues=(
                TaskValidationIssue(
                    code="execution.unsupported_flow",
                    path="nodes.review.ref",
                    message="Flow agent node is invalid.",
                    hint="Use a valid agent node reference.",
                    category=TaskValidationCategory.UNSUPPORTED,
                ),
            )
        )
        runner = FlowTaskTargetRunner(
            flow_resolver=lambda context: _agent_node_flow(
                context,
                agent_runner=agent_runner,
            )
        )

        with self.assertRaises(TaskValidationError) as error:
            await runner.run(self._context(input_value="private prompt"))

        self.assertEqual(
            error.exception.issues[0].path,
            "nodes.review.ref",
        )
        self.assertEqual(agent_runner.contexts, [])
        self.assertNotIn("private prompt", str(error.exception))

    async def test_run_agent_node_uses_flow_input_fallbacks(self) -> None:
        cases = (
            ("task_input", _agent_node_flow, "ready"),
            ("single_incoming", _single_incoming_agent_flow, "ready"),
            (
                "multi_incoming",
                _multi_incoming_agent_flow,
                {"left": "left", "right": "right"},
            ),
        )

        for name, factory, expected_input in cases:
            with self.subTest(name=name):
                agent_runner = CapturingTaskTargetRunner()
                runner = FlowTaskTargetRunner(
                    flow_resolver=lambda context: factory(
                        context,
                        agent_runner=agent_runner,
                    )
                )

                result = await runner.run(self._context(input_value="ready"))

                self.assertEqual(result, "agent output")
                self.assertEqual(
                    agent_runner.contexts[0].input_value,
                    expected_input,
                )

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
                    output_contract=TaskOutputContract.json(
                        {"type": "integer"}
                    ),
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
                    output_contract=self._object_output_contract(),
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
                    output_contract=self._object_output_contract(),
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

    async def test_run_isolates_object_input_from_node_mutation(self) -> None:
        def mutate(inputs: dict[str, object]) -> dict[str, object]:
            nested = cast(dict[str, object], inputs["nested"])
            items = cast(list[str], nested["items"])
            items.append("mutated")
            full_input = cast(dict[str, object], inputs[FLOW_TASK_INPUT_KEY])
            full_nested = cast(dict[str, object], full_input["nested"])
            return {
                "field": tuple(items),
                "full": tuple(cast(list[str], full_nested["items"])),
            }

        flow = Flow()
        flow.add_node(Node("A", func=mutate))
        runner = FlowTaskTargetRunner(flow_resolver=lambda _: flow)
        input_value = {"nested": {"items": ["original"]}}

        result = await runner.run(
            self._context(
                definition=self._context_definition(
                    input_contract=TaskInputContract.object(
                        {
                            "type": "object",
                            "required": ["nested"],
                            "additionalProperties": False,
                            "properties": {
                                "nested": {
                                    "type": "object",
                                    "required": ["items"],
                                    "additionalProperties": False,
                                    "properties": {
                                        "items": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                    },
                                },
                            },
                        }
                    ),
                    output_contract=self._object_output_contract(),
                ),
                input_value=input_value,
            )
        )

        self.assertEqual(
            result,
            {
                "field": ("original", "mutated"),
                "full": ("original",),
            },
        )
        self.assertEqual(input_value, {"nested": {"items": ["original"]}})

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

        result = await runner.run(
            self._context(
                definition=self._context_definition(
                    output_contract=self._object_output_contract(),
                ),
                input_value="ready",
            )
        )

        self.assertEqual(result, {"full": "ready", "value": "ready"})

    async def test_run_binds_array_input_as_json_lists(self) -> None:
        flow = Flow()
        flow.add_node(
            Node(
                "A",
                func=lambda inputs: {
                    "full": inputs[FLOW_TASK_INPUT_KEY],
                    "items": inputs["items"],
                },
            )
        )
        runner = FlowTaskTargetRunner(flow_resolver=lambda _: flow)

        result = await runner.run(
            self._context(
                definition=self._context_definition(
                    input_contract=TaskInputContract.array(
                        {
                            "type": "array",
                            "items": {"type": "array"},
                        }
                    ),
                    output_contract=self._object_output_contract(),
                ),
                input_value=[("first", "second")],
            )
        )

        self.assertEqual(
            result,
            {
                "full": [["first", "second"]],
                "items": [["first", "second"]],
            },
        )
        assert isinstance(result, Mapping)
        self.assertIsInstance(result["full"], list)
        self.assertIsInstance(result["items"], list)

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

    async def test_run_rejects_invalid_output_before_success_event(
        self,
    ) -> None:
        flow = Flow()
        flow.add_node(
            Node(
                "A",
                func=lambda _: {
                    "status": "ready",
                    "count": "private invalid count",
                },
            )
        )
        events: list[Event] = []
        runner = FlowTaskTargetRunner(flow_resolver=lambda _: flow)

        with self.assertRaises(TaskValidationError) as error:
            await runner.run(
                self._context(
                    definition=self._context_definition(
                        output_contract=TaskOutputContract.object(
                            {
                                "type": "object",
                                "required": ["status", "count"],
                                "additionalProperties": False,
                                "properties": {
                                    "status": {"type": "string"},
                                    "count": {"type": "integer"},
                                },
                            }
                        )
                    ),
                    input_value="private prompt",
                    event_listener=events.append,
                )
            )

        self.assertEqual(
            [issue.code for issue in error.exception.issues],
            ["output.invalid_type"],
        )
        self.assertEqual(
            [event.type for event in events],
            [
                EventType.FLOW_MANAGER_CALL_BEFORE,
                EventType.FLOW_MANAGER_CALL_AFTER,
            ],
        )
        failed_payload = cast(Mapping[str, object], events[1].payload)
        self.assertEqual(failed_payload["status"], "failed")
        self.assertNotIn("private prompt", str(events))
        self.assertNotIn("private invalid count", str(events))
        self.assertNotIn("private invalid count", str(error.exception))

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
                    "format": STORED_ENVELOPE_MARKER,
                    "privacy": STORED_MARKER,
                    "value": "ready",
                },
            )
        )

        self.assertEqual(result, "ready!")

    async def test_run_unwraps_legacy_stored_queued_input(self) -> None:
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

    async def test_run_keeps_legacy_object_input_envelope_collision(
        self,
    ) -> None:
        flow = Flow()
        flow.add_node(
            Node(
                "A",
                func=lambda inputs: {
                    "privacy": inputs["privacy"],
                    "value": inputs["value"],
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
                            "required": ["privacy", "value"],
                            "additionalProperties": False,
                            "properties": {
                                "privacy": {
                                    "type": "string",
                                    "enum": [STORED_MARKER],
                                },
                                "value": {"type": "string"},
                            },
                        }
                    ),
                    output_contract=self._object_output_contract(),
                    run=TaskRunPolicy.queued("default"),
                ),
                input_value={
                    "privacy": STORED_MARKER,
                    "value": "ready",
                },
            )
        )

        self.assertEqual(
            result,
            {"privacy": STORED_MARKER, "value": "ready"},
        )

    async def test_run_keeps_declared_object_input_with_privacy_marker(
        self,
    ) -> None:
        flow = Flow()
        flow.add_node(
            Node(
                "A",
                func=lambda inputs: {
                    "privacy": inputs["privacy"],
                    "value": inputs["value"],
                },
            )
        )
        runner = FlowTaskTargetRunner(flow_resolver=lambda _: flow)

        for marker in (
            DROPPED_MARKER,
            ENCRYPTED_MARKER,
            HASHED_MARKER,
            REDACTED_MARKER,
        ):
            with self.subTest(marker=marker):
                result = await runner.run(
                    self._context(
                        definition=self._context_definition(
                            input_contract=TaskInputContract.object(
                                {
                                    "type": "object",
                                    "required": ["privacy", "value"],
                                    "additionalProperties": False,
                                    "properties": {
                                        "privacy": {
                                            "type": "string",
                                            "enum": [marker],
                                        },
                                        "value": {"type": "string"},
                                    },
                                }
                            ),
                            output_contract=self._object_output_contract(),
                            run=TaskRunPolicy.queued("default"),
                        ),
                        input_value={
                            "privacy": marker,
                            "value": "ready",
                        },
                    )
                )

                self.assertEqual(
                    result,
                    {"privacy": marker, "value": "ready"},
                )

    async def test_run_accepts_plain_queued_input(self) -> None:
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
                input_value="ready",
            )
        )

        self.assertEqual(result, "ready!")

    async def test_run_accepts_plain_queued_object_input(self) -> None:
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
                    output_contract=self._object_output_contract(),
                    run=TaskRunPolicy.queued("default"),
                ),
                input_value={"name": "ready", "limit": 2},
            )
        )

        self.assertEqual(result, {"name": "ready", "limit": 2})

    async def test_run_rejects_unavailable_queued_input_safely(self) -> None:
        flow = Flow()
        flow.add_node(Node("A", func=lambda _: "unused output"))
        runner = FlowTaskTargetRunner(flow_resolver=lambda _: flow)

        for marker in (
            DROPPED_MARKER,
            ENCRYPTED_MARKER,
            HASHED_MARKER,
            REDACTED_MARKER,
        ):
            with self.subTest(marker=marker):
                with self.assertRaises(TaskValidationError) as error:
                    await runner.run(
                        self._context(
                            definition=self._context_definition(
                                run=TaskRunPolicy.queued("default"),
                            ),
                            input_value={
                                "privacy": marker,
                                "raw": "private prompt",
                            },
                        )
                    )

                self.assertEqual(
                    error.exception.issues[0].code,
                    "execution.unsupported_flow",
                )
                self.assertNotIn("private prompt", str(error.exception))
                self.assertNotIn("unused output", str(error.exception))

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
        event_listener: (
            Callable[[Event], Awaitable[None] | None] | None
        ) = None,
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
            event_listener=event_listener,
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

    def _object_output_contract(self) -> TaskOutputContract:
        return TaskOutputContract.object({"type": "object"})


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

        self.assertEqual(
            result.run.state,
            TaskRunState.SUCCEEDED,
        )
        self.assertEqual(result.output, {"status": "ready", "count": 2})

    async def test_direct_runner_keeps_flow_input_mutation_local(
        self,
    ) -> None:
        def mutate(inputs: dict[str, object]) -> dict[str, object]:
            nested = cast(dict[str, object], inputs["nested"])
            items = cast(list[str], nested["items"])
            items.append("mutated")
            full_input = cast(dict[str, object], inputs[FLOW_TASK_INPUT_KEY])
            full_nested = cast(dict[str, object], full_input["nested"])
            return {
                "field": items,
                "full": cast(list[str], full_nested["items"]),
            }

        flow = Flow()
        flow.add_node(Node("A", func=mutate))
        runner = DirectTaskRunner(
            self.store,
            target=FlowTaskTargetRunner(flow_resolver=lambda _: flow),
            hmac_provider=StaticHmacProvider(),
            definition_hash=lambda _: "flow-direct-input-isolation",
        )
        input_value = {"nested": {"items": ["original"]}}

        result = await runner.run(
            self._definition(
                input_contract=TaskInputContract.object(
                    {
                        "type": "object",
                        "required": ["nested"],
                        "additionalProperties": False,
                        "properties": {
                            "nested": {
                                "type": "object",
                                "required": ["items"],
                                "additionalProperties": False,
                                "properties": {
                                    "items": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                            },
                        },
                    }
                ),
                output_contract=TaskOutputContract.object(
                    {
                        "type": "object",
                        "required": ["field", "full"],
                        "additionalProperties": False,
                        "properties": {
                            "field": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "full": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                    }
                ),
            ),
            input_value=input_value,
        )

        self.assertEqual(
            result.run.state,
            TaskRunState.SUCCEEDED,
        )
        self.assertEqual(
            result.output,
            {
                "field": ["original", "mutated"],
                "full": ["original"],
            },
        )
        self.assertEqual(input_value, {"nested": {"items": ["original"]}})

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

    async def test_direct_runner_classifies_flow_validation_failure(
        self,
    ) -> None:
        def resolver(_: TaskTargetContext) -> Flow:
            return cast(Flow, "private invalid flow")

        runner = DirectTaskRunner(
            self.store,
            target=FlowTaskTargetRunner(flow_resolver=resolver),
            hmac_provider=StaticHmacProvider(),
            definition_hash=lambda _: "flow-direct-runtime-validation",
        )

        result = await runner.run(
            self._definition(
                input_contract=TaskInputContract.string(),
                output_contract=TaskOutputContract.text(),
            ),
            input_value="private prompt",
        )

        self.assertEqual(result.run.state, TaskRunState.FAILED)
        self.assertIsNone(result.output)
        assert result.run.result is not None
        error = cast(Mapping[str, object], result.run.result.error)
        self.assertEqual(error["category"], "runnable")
        self.assertEqual(error["code"], "runnable.failed")
        self.assertNotIn("private invalid flow", str(error))
        self.assertNotIn("private prompt", str(error))

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
                observability=TaskObservabilityPolicy(),
            ),
            input_value={"prompt": "private prompt", "limit": 1},
        )

        events = await self.store.list_events(result.run.run_id)
        self.assertEqual(result.run.state, TaskRunState.FAILED)
        self.assertIsNone(result.output)
        self.assertEqual(events[1].event_type, "flow_manager_call_after")
        end_payload = cast(Mapping[str, object], events[1].payload)
        self.assertEqual(end_payload["status"], "failed")
        self.assertNotIn("private invalid count", str(events))
        self.assertNotIn("private prompt", str(events))
        self.assertNotIn("private invalid count", str(result.run.result))
        self.assertNotIn("private prompt", str(result.run.result))

    async def test_direct_runner_commits_array_output_from_flow_input(
        self,
    ) -> None:
        flow = Flow()
        flow.add_node(
            Node("A", func=lambda inputs: inputs[FLOW_TASK_INPUT_KEY])
        )
        runner = DirectTaskRunner(
            self.store,
            target=FlowTaskTargetRunner(flow_resolver=lambda _: flow),
            hmac_provider=StaticHmacProvider(),
            definition_hash=lambda _: "flow-direct-array-output",
        )

        result = await runner.run(
            self._definition(
                input_contract=self._array_input_contract(),
                output_contract=self._array_output_contract(),
            ),
            input_value=["safe", "done"],
        )

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(result.output, ["safe", "done"])
        self.assertIsInstance(result.output, list)

    async def test_direct_runner_sends_file_input_to_flow_agent_node_output(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            flow_path = _write_agent_flow_workspace(root)
            loader = FlowAgentLoader(text='{"status": "ready", "count": 2}')
            agent_runner = AgentTaskTargetRunner(loader, ref_base=root)
            runner = DirectTaskRunner(
                self.store,
                target=FlowTaskTargetRunner(
                    ref_base=root,
                    flow_resolver=_flow_loader_resolver(
                        flow_path,
                        agent_runner=agent_runner,
                        root=root,
                    ),
                ),
                hmac_provider=StaticHmacProvider(),
                artifact_store=cast(Any, FailingArtifactStore()),
                execution_roots=(root,),
                definition_hash=lambda _: "flow-agent-file-success",
            )

            result = await runner.run(
                self._definition(
                    input_contract=TaskInputContract.file(
                        mime_types=("application/pdf",)
                    ),
                    output_contract=self._object_output_contract(),
                    execution=TaskExecutionTarget.flow("flow.toml"),
                    observability=TaskObservabilityPolicy(),
                ),
                input_value=TaskFileDescriptor.provider_reference_descriptor(
                    "file-123",
                    kind=TaskProviderReferenceKind.PROVIDER_FILE_ID,
                    provider="openai",
                    mime_type="application/pdf",
                    owner_scope="private-tenant",
                ),
            )

            expected_path = str(root / "agents" / "review.toml")

        self.assertEqual(
            result.run.state,
            TaskRunState.SUCCEEDED,
        )
        self.assertEqual(result.output, {"status": "ready", "count": 2})
        self.assertEqual(loader.paths, [expected_path])
        self.assertEqual(loader.entered, 1)
        self.assertEqual(loader.exited, 1)
        self.assertEqual(len(loader.inputs), 1)
        message = loader.inputs[0]
        assert isinstance(message, Message)
        self.assertEqual(message.role, MessageRole.USER)
        content = cast(list[object], message.content)
        text_blocks = [
            block for block in content if isinstance(block, MessageContentText)
        ]
        file_blocks = [
            block for block in content if isinstance(block, MessageContentFile)
        ]
        self.assertEqual([block.text for block in text_blocks], ["Review."])
        self.assertEqual(len(file_blocks), 1)
        self.assertEqual(file_blocks[0].file["file_id"], "file-123")
        self.assertEqual(
            file_blocks[0].file["mime_type"],
            "application/pdf",
        )
        self.assertNotIn("private-tenant", str(result.run.result))

    async def test_direct_runner_rejects_flow_agent_provider_mismatch(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            flow_path = _write_agent_flow_workspace(
                root,
                uri="ai://env:KEY@anthropic/claude-3-5-sonnet",
            )
            loader = FlowAgentLoader()
            agent_runner = AgentTaskTargetRunner(loader, ref_base=root)
            runner = DirectTaskRunner(
                self.store,
                target=FlowTaskTargetRunner(
                    ref_base=root,
                    flow_resolver=_flow_loader_resolver(
                        flow_path,
                        agent_runner=agent_runner,
                        root=root,
                    ),
                ),
                hmac_provider=StaticHmacProvider(),
                execution_roots=(root,),
                definition_hash=lambda _: "flow-agent-provider-mismatch",
            )

            result = await runner.run(
                self._definition(
                    input_contract=TaskInputContract.file_array(
                        mime_types=("application/pdf",)
                    ),
                    output_contract=TaskOutputContract.text(),
                    execution=TaskExecutionTarget.flow("flow.toml"),
                ),
                input_value=[
                    TaskFileDescriptor.provider_reference_descriptor(
                        "file-private",
                        kind=TaskProviderReferenceKind.PROVIDER_FILE_ID,
                        provider="openai",
                        mime_type="application/pdf",
                        owner_scope="private-tenant",
                    )
                ],
            )

        self.assertEqual(result.run.state, TaskRunState.FAILED)
        self.assertEqual(loader.inputs, [])
        self.assertEqual(loader.paths, [])
        self.assertNotIn("file-private", str(result.run.result))
        self.assertNotIn("private-tenant", str(result.run.result))

    async def test_direct_runner_rejects_invalid_flow_array_output(
        self,
    ) -> None:
        flow = Flow()
        flow.add_node(
            Node(
                "A",
                func=lambda inputs: [
                    cast(list[object], inputs[FLOW_TASK_INPUT_KEY])[0],
                    "private invalid item",
                ],
            )
        )
        runner = DirectTaskRunner(
            self.store,
            target=FlowTaskTargetRunner(flow_resolver=lambda _: flow),
            hmac_provider=StaticHmacProvider(),
            definition_hash=lambda _: "flow-direct-invalid-array-output",
        )

        result = await runner.run(
            self._definition(
                input_contract=TaskInputContract.array(
                    {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 1,
                    }
                ),
                output_contract=TaskOutputContract.array(
                    {
                        "type": "array",
                        "items": {"type": "integer"},
                    }
                ),
            ),
            input_value=[1],
        )

        self.assertEqual(result.run.state, TaskRunState.FAILED)
        self.assertIsNone(result.output)
        self.assertNotIn("private invalid item", str(result.run.result))

    def test_input_binding_exposes_scalar_array_and_object_shapes(
        self,
    ) -> None:
        self.assertEqual(
            flow_task_input_binding("ready"),
            {FLOW_TASK_INPUT_KEY: "ready", "value": "ready"},
        )
        self.assertEqual(
            flow_task_input_binding([1, 2]),
            {FLOW_TASK_INPUT_KEY: [1, 2], "items": [1, 2]},
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

    def test_input_binding_exposes_task_file_descriptors(self) -> None:
        file = TaskInputFile(
            logical_path="provider:file-123",
            media_type="application/pdf",
            provider_reference=TaskProviderReference(
                kind=TaskProviderReferenceKind.PROVIDER_FILE_ID,
                provider="openai",
                reference="file-123",
            ),
        )

        binding = flow_task_input_binding(
            {"source_kind": "provider_reference"},
            files=(file,),
        )

        self.assertIs(binding["file"], file)
        self.assertEqual(binding["files"], [file])
        self.assertEqual(binding[FLOW_TASK_FILES_KEY], [file])

    def test_input_binding_isolates_nested_object_mutation(self) -> None:
        value = {"nested": {"items": ["original"]}}
        binding = flow_task_input_binding(value)
        nested = cast(dict[str, object], binding["nested"])
        cast(list[str], nested["items"]).append("mutated")
        full_input = cast(dict[str, object], binding[FLOW_TASK_INPUT_KEY])
        full_nested = cast(dict[str, object], full_input["nested"])

        self.assertEqual(
            cast(list[str], nested["items"]),
            ["original", "mutated"],
        )
        self.assertEqual(cast(list[str], full_nested["items"]), ["original"])
        self.assertEqual(value, {"nested": {"items": ["original"]}})

    def test_input_binding_isolates_nested_array_mutation(self) -> None:
        value = [{"items": ["original"]}]
        binding = flow_task_input_binding(value)
        field_items = cast(list[object], binding["items"])
        field_item = cast(dict[str, object], field_items[0])
        cast(list[str], field_item["items"]).append("mutated")
        full_items = cast(list[object], binding[FLOW_TASK_INPUT_KEY])
        full_item = cast(dict[str, object], full_items[0])

        self.assertEqual(
            cast(list[str], field_item["items"]),
            ["original", "mutated"],
        )
        self.assertEqual(cast(list[str], full_item["items"]), ["original"])
        self.assertEqual(value, [{"items": ["original"]}])

    def test_input_binding_normalizes_nested_tuple_values(self) -> None:
        self.assertEqual(
            flow_task_input_binding([("first", "second")]),
            {
                FLOW_TASK_INPUT_KEY: [["first", "second"]],
                "items": [["first", "second"]],
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
        execution: TaskExecutionTarget | None = None,
        observability: TaskObservabilityPolicy | None = None,
        privacy: TaskPrivacyPolicy | None = None,
    ) -> TaskDefinition:
        return TaskDefinition(
            task=TaskMetadata(name="flow-task", version="1"),
            input=input_contract,
            output=output_contract,
            execution=execution
            or TaskExecutionTarget.flow("flows/report.toml"),
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

    def _array_input_contract(self) -> TaskInputContract:
        return TaskInputContract.array(
            {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            }
        )

    def _array_output_contract(self) -> TaskOutputContract:
        return TaskOutputContract.array(
            {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            }
        )


if __name__ == "__main__":
    main()
