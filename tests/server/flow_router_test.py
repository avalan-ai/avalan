from datetime import UTC, datetime
from types import SimpleNamespace
from unittest import TestCase

from fastapi import FastAPI
from fastapi.testclient import TestClient

from avalan.flow import (
    FlowDiagnostic,
    FlowDiagnosticCategory,
    FlowDiagnosticSeverity,
    FlowExecutor,
    FlowExecutorRunResult,
    FlowNodePlan,
    InMemoryFlowStateStore,
    default_flow_node_registry,
)
from avalan.server.routers.flow import router
from avalan.task import TaskClientUnsupportedOperationError, TaskRunState
from avalan.task.event import SanitizedTaskEvent, TaskEventCategory
from avalan.task.store import TaskStoreNotFoundError


class FlowRouterTestCase(TestCase):
    def setUp(self) -> None:
        self.app = FastAPI()
        self.app.include_router(router, prefix="/flows")
        self.client = TestClient(self.app)

    def test_validate_success_and_malformed_source_are_safe(self) -> None:
        success = self.client.post(
            "/flows/validate",
            json={"source": _strict_flow_source()},
        )
        failure = self.client.post(
            "/flows/validate",
            json={"source": "[flow\nsecret = 'private customer prompt'"},
        )

        self.assertEqual(success.status_code, 200)
        self.assertTrue(success.json()["ok"])
        self.assertEqual(success.json()["definition"]["name"], "server-flow")
        self.assertEqual(failure.status_code, 200)
        self.assertFalse(failure.json()["ok"])
        self.assertEqual(
            failure.json()["diagnostics"][0]["code"],
            "flow.malformed_toml",
        )
        self.assertNotIn("private customer prompt", failure.text)

    def test_mermaid_parse_render_and_compare(self) -> None:
        parsed = self.client.post(
            "/flows/mermaid/parse",
            json={"source": "graph TD\nA[Start] --> B[Done]"},
        )
        rendered = self.client.post(
            "/flows/mermaid/render",
            json={"source": "graph TD\nA[Start] --> B[Done]"},
        )
        compared = self.client.post(
            "/flows/mermaid/compare",
            json={
                "diagram_source": "graph TD\nA --> B",
                "definition_source": _strict_flow_source(),
            },
        )
        rejected = self.client.post(
            "/flows/mermaid/parse",
            json={
                "source": "graph TD\nA & B --> C\n%% private prompt",
                "mode": "executable",
            },
        )

        self.assertTrue(parsed.json()["ok"])
        self.assertEqual(parsed.json()["view"]["nodes"][0]["label"], "Start")
        self.assertTrue(rendered.json()["ok"])
        self.assertIn('A["Start"]', rendered.json()["source"])
        self.assertFalse(compared.json()["ok"])
        self.assertIn("flow.view.binding", compared.text)
        self.assertFalse(rejected.json()["ok"])
        self.assertNotIn("private prompt", rejected.text)

    def test_mermaid_render_and_compare_negative_inputs_are_safe(self) -> None:
        rendered = self.client.post(
            "/flows/mermaid/render",
            json={
                "source": "graph TD\nA & B --> C\n%% private prompt",
                "mode": "executable",
            },
        )
        compared = self.client.post(
            "/flows/mermaid/compare",
            json={
                "diagram_source": "graph TD\nA & B --> C",
                "definition_source": "[flow",
                "mode": "executable",
            },
        )

        self.assertEqual(rendered.status_code, 200)
        self.assertFalse(rendered.json()["ok"])
        self.assertNotIn("private prompt", rendered.text)
        self.assertEqual(compared.status_code, 200)
        self.assertFalse(compared.json()["ok"])
        self.assertIn("flow.malformed_toml", compared.text)

    def test_run_persists_inspect_trace_and_rejects_duplicate_run_id(
        self,
    ) -> None:
        self.app.state.flow_state_store = InMemoryFlowStateStore()

        run = self.client.post(
            "/flows/run",
            json={
                "source": _strict_flow_source(),
                "inputs": {"payload": "private input"},
                "run_id": "run-1",
            },
        )
        inspect = self.client.get("/flows/runs/run-1/inspect")
        trace = self.client.get("/flows/runs/run-1/trace")
        duplicate = self.client.post(
            "/flows/run",
            json={
                "source": _strict_flow_source(),
                "inputs": {"payload": "private input"},
                "run_id": "run-1",
            },
        )

        self.assertEqual(run.status_code, 200)
        self.assertTrue(run.json()["ok"], run.text)
        self.assertEqual(run.json()["outputs"], {"answer": "private input"})
        self.assertEqual(run.json()["record_revision"], 1)
        self.assertEqual(inspect.status_code, 200)
        self.assertEqual(
            inspect.json()["inspection"]["state"],
            "succeeded",
        )
        self.assertEqual(trace.status_code, 200)
        self.assertNotIn("private input", trace.text)
        self.assertEqual(duplicate.status_code, 409)
        self.assertEqual(
            duplicate.json()["detail"]["diagnostics"][0]["code"],
            "flow.task.run_conflict",
        )

    def test_run_rejects_invalid_source_and_supports_configured_runner(
        self,
    ) -> None:
        calls: list[str] = []

        async def runner(
            node: FlowNodePlan,
            inputs: dict[str, object],
        ) -> object:
            calls.append(node.name)
            return dict(inputs)

        invalid = self.client.post(
            "/flows/run",
            json={"source": "[flow", "inputs": {"payload": "safe"}},
        )
        no_run_id = self.client.post(
            "/flows/run",
            json={
                "source": _strict_flow_source(),
                "inputs": {"payload": "safe"},
            },
        )
        self.app.state.flow_node_runner = runner
        custom_runner = self.client.post(
            "/flows/run",
            json={
                "source": _strict_flow_source(),
                "inputs": {"payload": "safe"},
            },
        )
        self.app.state.flow_node_registry = default_flow_node_registry()
        with_registry = self.client.post(
            "/flows/validate",
            json={"source": _strict_flow_source()},
        )

        self.assertEqual(invalid.status_code, 200)
        self.assertFalse(invalid.json()["ok"])
        self.assertEqual(no_run_id.status_code, 200)
        self.assertNotIn("run_id", no_run_id.json())
        self.assertEqual(custom_runner.status_code, 200)
        self.assertEqual(calls, ["echo"])
        self.assertTrue(with_registry.json()["ok"])

    def test_run_with_executor_diagnostics_does_not_persist(self) -> None:
        self.app.state.flow_state_store = InMemoryFlowStateStore()
        self.app.state.flow_executor = _DiagnosticOnlyExecutor()

        response = self.client.post(
            "/flows/run",
            json={
                "source": _strict_flow_source(),
                "inputs": {"payload": "safe"},
                "run_id": "run-1",
            },
        )
        missing = self.client.get("/flows/runs/run-1/inspect")

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["ok"])
        self.assertEqual(
            response.json()["diagnostics"][0]["code"],
            "flow.execution.test_diagnostic",
        )
        self.assertEqual(missing.status_code, 404)

    def test_run_without_store_for_run_id_and_missing_run_fail_safely(
        self,
    ) -> None:
        run = self.client.post(
            "/flows/run",
            json={
                "source": _strict_flow_source(),
                "inputs": {"payload": "safe"},
                "run_id": "run-1",
            },
        )
        self.app.state.flow_state_store = InMemoryFlowStateStore()
        missing = self.client.get("/flows/runs/missing/inspect")

        self.assertEqual(run.status_code, 503)
        self.assertEqual(
            run.json()["detail"]["diagnostics"][0]["code"],
            "flow.task.store_unavailable",
        )
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(
            missing.json()["detail"]["diagnostics"][0]["code"],
            "flow.task.run_not_found",
        )

    def test_inspect_without_store_fails_safely(self) -> None:
        response = self.client.get("/flows/runs/run-1/inspect")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["detail"]["diagnostics"][0]["code"],
            "flow.task.store_unavailable",
        )

    def test_resume_reports_conflicts_and_can_update_record(self) -> None:
        self.app.state.flow_state_store = InMemoryFlowStateStore()
        run = self.client.post(
            "/flows/run",
            json={
                "source": _strict_flow_source(),
                "inputs": {"payload": "safe"},
                "run_id": "run-1",
            },
        )
        resumed = self.client.post(
            "/flows/runs/run-1/resume",
            json={
                "source": _strict_flow_source(),
                "decisions": {"review": {"decision": "approved"}},
                "expected_revision": 1,
            },
        )
        invalid_source = self.client.post(
            "/flows/runs/run-1/resume",
            json={"source": "[flow", "decisions": {"review": {}}},
        )
        conflict = self.client.post(
            "/flows/runs/run-1/resume",
            json={
                "source": _strict_flow_source(),
                "decisions": {"review": {"decision": "approved"}},
                "expected_revision": 1,
            },
        )

        self.assertTrue(run.json()["ok"], run.text)
        self.assertEqual(resumed.status_code, 200)
        self.assertFalse(resumed.json()["ok"], resumed.text)
        self.assertEqual(
            resumed.json()["diagnostics"][0]["code"],
            "flow.execution.unknown_resume_node",
        )
        self.assertEqual(resumed.json()["record_revision"], 2)
        self.assertEqual(invalid_source.status_code, 200)
        self.assertFalse(invalid_source.json()["ok"])
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(
            conflict.json()["detail"]["diagnostics"][0]["code"],
            "flow.task.revision_conflict",
        )

    def test_resume_without_store_and_missing_run_fail_safely(self) -> None:
        no_store = self.client.post(
            "/flows/runs/run-1/resume",
            json={
                "source": _strict_flow_source(),
                "decisions": {"review": {"decision": "approved"}},
            },
        )
        self.app.state.flow_state_store = InMemoryFlowStateStore()
        missing = self.client.post(
            "/flows/runs/missing/resume",
            json={
                "source": _strict_flow_source(),
                "decisions": {"review": {"decision": "approved"}},
            },
        )

        self.assertEqual(no_store.status_code, 503)
        self.assertEqual(
            no_store.json()["detail"]["diagnostics"][0]["code"],
            "flow.task.store_unavailable",
        )
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(
            missing.json()["detail"]["diagnostics"][0]["code"],
            "flow.task.run_not_found",
        )

    def test_resume_with_executor_diagnostics_skips_state_update(self) -> None:
        self.app.state.flow_state_store = InMemoryFlowStateStore()
        run = self.client.post(
            "/flows/run",
            json={
                "source": _strict_flow_source(),
                "inputs": {"payload": "safe"},
                "run_id": "run-1",
            },
        )
        self.app.state.flow_executor = _DiagnosticOnlyExecutor()

        response = self.client.post(
            "/flows/runs/run-1/resume",
            json={
                "source": _strict_flow_source(),
                "decisions": {"review": {"decision": "approved"}},
            },
        )

        self.assertTrue(run.json()["ok"], run.text)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["ok"])
        self.assertEqual(response.json()["record_revision"], 1)

    def test_events_and_cancel_delegate_to_task_client(self) -> None:
        task_client = _FakeFlowTaskClient()
        self.app.state.flow_task_client = task_client

        events = self.client.get(
            "/flows/runs/run-1/events",
            params={"after_sequence": 1},
        )
        with self.client.stream(
            "GET",
            "/flows/runs/run-1/events",
            params={"stream": "true"},
        ) as streamed:
            stream_lines = list(streamed.iter_lines())
        cancelled = self.client.post("/flows/runs/run-1/cancel")
        cancel_conflict = self.client.post("/flows/runs/conflict/cancel")

        self.assertEqual(events.status_code, 200)
        self.assertEqual(events.json()["events"][0]["event_type"], "flow_end")
        self.assertEqual(task_client.after_sequences[0], 1)
        self.assertEqual(task_client.after_sequences[1], None)
        self.assertEqual(streamed.status_code, 200)
        self.assertIn("event: flow.event", stream_lines)
        self.assertIn("event: flow.events.completed", stream_lines)
        self.assertNotIn("private", "\n".join(stream_lines))
        self.assertEqual(cancelled.status_code, 200)
        self.assertEqual(cancelled.json()["run"]["state"], "cancel_requested")
        self.assertEqual(cancel_conflict.status_code, 409)
        self.assertEqual(
            cancel_conflict.json()["detail"]["diagnostics"][0]["code"],
            "task.cancel_unavailable",
        )

    def test_events_and_cancel_missing_runs_are_safe(self) -> None:
        self.app.state.flow_task_client = _MissingFlowTaskClient()

        events = self.client.get("/flows/runs/missing/events")
        cancel = self.client.post("/flows/runs/missing/cancel")

        self.assertEqual(events.status_code, 404)
        self.assertEqual(
            events.json()["detail"]["diagnostics"][0]["code"],
            "flow.task.events_not_found",
        )
        self.assertEqual(cancel.status_code, 404)
        self.assertEqual(
            cancel.json()["detail"]["diagnostics"][0]["code"],
            "flow.task.run_not_found",
        )

    def test_events_and_cancel_require_task_client(self) -> None:
        events = self.client.get("/flows/runs/run-1/events")
        cancel = self.client.post("/flows/runs/run-1/cancel")

        self.assertEqual(events.status_code, 503)
        self.assertEqual(
            events.json()["detail"]["diagnostics"][0]["code"],
            "flow.task.events_unavailable",
        )
        self.assertEqual(cancel.status_code, 503)
        self.assertEqual(
            cancel.json()["detail"]["diagnostics"][0]["code"],
            "flow.task.cancel_unavailable",
        )


class _FakeFlowTaskClient:
    def __init__(self) -> None:
        self.after_sequences: list[int | None] = []

    async def events(
        self,
        run_id: str,
        *,
        after_sequence: int | None = None,
    ) -> tuple[SanitizedTaskEvent, ...]:
        self.after_sequences.append(after_sequence)
        return (
            SanitizedTaskEvent(
                event_id="event-1",
                run_id=run_id,
                sequence=2,
                event_type="flow_end",
                category=TaskEventCategory.ENGINE,
                created_at=datetime(2026, 6, 8, tzinfo=UTC),
                payload={"status": "succeeded"},
            ),
        )

    async def cancel(self, run_id: str) -> object:
        if run_id == "conflict":
            raise TaskClientUnsupportedOperationError(
                code="task.cancel_unavailable",
                operation="cancel",
                message="terminal",
            )
        return SimpleNamespace(
            run_id=run_id,
            state=TaskRunState.CANCEL_REQUESTED,
        )


class _MissingFlowTaskClient:
    async def events(
        self,
        run_id: str,
        *,
        after_sequence: int | None = None,
    ) -> tuple[SanitizedTaskEvent, ...]:
        _ = run_id, after_sequence
        raise TaskStoreNotFoundError("missing")

    async def cancel(self, run_id: str) -> object:
        _ = run_id
        raise TaskStoreNotFoundError("missing")


class _DiagnosticOnlyExecutor(FlowExecutor):
    async def run(
        self, *args: object, **kwargs: object
    ) -> FlowExecutorRunResult:
        _ = args, kwargs
        return FlowExecutorRunResult(diagnostics=(_diagnostic(),))

    async def resume(
        self,
        *args: object,
        **kwargs: object,
    ) -> FlowExecutorRunResult:
        _ = args, kwargs
        return FlowExecutorRunResult(diagnostics=(_diagnostic(),))


def _diagnostic() -> FlowDiagnostic:
    return FlowDiagnostic(
        code="flow.execution.test_diagnostic",
        path="flow",
        category=FlowDiagnosticCategory.EXECUTION,
        severity=FlowDiagnosticSeverity.ERROR,
        message="Flow execution returned a diagnostic.",
        hint="Inspect the flow run.",
    )


def _strict_flow_source() -> str:
    return """
[flow]
name = "server-flow"
version = "1"

[[inputs]]
name = "payload"
type = "string"

[[outputs]]
name = "answer"
type = "text"

[entry]
type = "node"
node = "echo"

[output_behavior]
type = "map"

[output_behavior.outputs]
answer = "echo.value"

[nodes.echo]
type = "pass-through"

[nodes.echo.mapping]
value = "input.payload"
"""
