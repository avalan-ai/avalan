from asyncio import CancelledError
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from sys import path as sys_path
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase, TestCase, main

sys_path.append(str(Path(__file__).parents[1] / "task" / "stores"))

from pgsql_contract_test import (  # type: ignore[import-not-found]
    FakeCursor,
    FakePgsqlTaskDatabase,
)
from store_contract_test import (  # type: ignore[import-not-found]
    SequenceClock,
    SequenceIds,
    definition,
)

from avalan.flow import (
    FlowDiagnostic,
    FlowDiagnosticCategory,
    FlowEdgeState,
    FlowEdgeTrace,
    FlowExecutionRecord,
    FlowExecutionTrace,
    FlowExecutionUpdate,
    FlowNodeAttemptRecord,
    FlowNodeState,
    FlowNodeTrace,
    FlowSourceSpan,
    InMemoryFlowStateStore,
    PgsqlFlowStateStore,
    flow_execution_record_from_snapshot,
    flow_node_attempt_from_snapshot,
    flow_trace_from_snapshot,
    flow_trace_to_snapshot,
)
from avalan.flow.store import _plain
from avalan.pgsql import (
    PgsqlFailure,
    PgsqlFailureCategory,
    PgsqlOperationError,
)
from avalan.task import (
    TaskExecutionRequest,
    TaskStoreConflictError,
    TaskStoreError,
    TaskStoreNotFoundError,
)
from avalan.task.stores import PgsqlTaskStore


class ExistingTaskRunLookup:
    def __init__(self) -> None:
        self.run_ids: list[str] = []

    async def get_run(self, run_id: str) -> object:
        self.run_ids.append(run_id)
        return {"run_id": run_id}


class MissingTaskRunLookup:
    async def get_run(self, run_id: str) -> object:
        raise TaskStoreNotFoundError("task run was not found")


class ErrorConnectionContext:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    async def __aenter__(self) -> object:
        raise self.error

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool:
        return False


class ErrorDatabase:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    def connection(self) -> ErrorConnectionContext:
        return ErrorConnectionContext(self.error)


class UniqueSqlStateError(RuntimeError):
    sqlstate = "23505"


class UnknownSqlStateError(RuntimeError):
    pass


class FlowStoreSnapshotTest(TestCase):
    def test_record_snapshot_round_trips_without_private_source_text(
        self,
    ) -> None:
        diagnostic = _diagnostic()
        trace = _trace().with_node_state(
            "start",
            FlowNodeState.FAILED,
            attempts=2,
            duration_ms=5,
            diagnostics=(diagnostic,),
        )
        trace = trace.with_edge_state(
            0,
            FlowEdgeState.FAILED,
            duration_ms=1,
            diagnostics=(diagnostic,),
        )
        record = FlowExecutionRecord(
            task_run_id="run-1",
            revision=1,
            trace=trace,
            node_attempts=(
                FlowNodeAttemptRecord(
                    node="start",
                    attempt=1,
                    state=FlowNodeState.FAILED,
                    duration_ms=5,
                    diagnostics=(diagnostic,),
                    artifact_refs=(
                        {"artifact_id": "artifact-1", "store": "memory"},
                    ),
                    metadata={"safe": "node"},
                ),
            ),
            node_outputs={"start": {"answer": {"value": 42}}},
            selected_outputs={"answer": {"value": 42}},
            loop_counters={"repair": 2},
            pause_tokens={"review": "pause-token"},
            diagnostics=(diagnostic,),
            artifact_refs=(
                {"artifact_id": "artifact-root", "store": "memory"},
            ),
            metadata={"safe": "record"},
            created_at=datetime(2026, 1, 1, 0, 0),
            updated_at=datetime(2026, 1, 1, 0, 1),
        )

        snapshot = record.as_snapshot()
        restored = flow_execution_record_from_snapshot(snapshot)

        self.assertEqual(restored.task_run_id, "run-1")
        self.assertEqual(restored.revision, 1)
        self.assertEqual(dict(restored.node_outputs), record.node_outputs)
        self.assertEqual(
            dict(restored.selected_outputs), record.selected_outputs
        )
        self.assertEqual(dict(restored.loop_counters), {"repair": 2})
        self.assertEqual(
            dict(restored.pause_tokens), {"review": "pause-token"}
        )
        self.assertEqual(
            restored.node_attempts[0].artifact_refs[0]["store"], "memory"
        )
        self.assertIsNotNone(restored.diagnostics[0].source_span)
        assert restored.diagnostics[0].source_span is not None
        self.assertIsNone(restored.diagnostics[0].source_span.source)
        self.assertNotIn("private-source-token", str(snapshot))
        self.assertNotIn("private-source-token", str(restored.as_snapshot()))

    def test_trace_snapshot_helpers_round_trip_public_state(self) -> None:
        trace = _trace().with_node_state(
            "start",
            FlowNodeState.SUCCEEDED,
            attempts=1,
            duration_ms=3,
        )

        snapshot = flow_trace_to_snapshot(trace)
        restored = flow_trace_from_snapshot(snapshot)
        from_text = flow_trace_from_snapshot('{"nodes":[],"edges":[]}')

        self.assertEqual(restored.nodes[0].state, FlowNodeState.SUCCEEDED)
        self.assertEqual(restored.nodes[0].duration_ms, 3)
        self.assertEqual(from_text.nodes, ())
        self.assertEqual(
            _plain(["created", datetime(2026, 1, 1, tzinfo=UTC)]),
            ["created", "2026-01-01T00:00:00+00:00"],
        )

    def test_records_and_snapshots_reject_invalid_values(self) -> None:
        with self.assertRaises(AssertionError):
            FlowNodeAttemptRecord(
                node="start",
                attempt=0,
                state=FlowNodeState.RUNNING,
            )
        with self.assertRaises(AssertionError):
            FlowNodeAttemptRecord(
                node="start",
                attempt=1,
                state=FlowNodeState.RUNNING,
                duration_ms=-1,
            )
        with self.assertRaises(AssertionError):
            FlowExecutionRecord(
                task_run_id="run-1",
                revision=1,
                trace=_trace(),
                node_attempts=(
                    FlowNodeAttemptRecord(
                        node="start",
                        attempt=1,
                        state=FlowNodeState.RUNNING,
                    ),
                    FlowNodeAttemptRecord(
                        node="start",
                        attempt=1,
                        state=FlowNodeState.SUCCEEDED,
                    ),
                ),
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
                updated_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
        with self.assertRaises(AssertionError):
            FlowExecutionUpdate(node_outputs={"bad": object()})
        with self.assertRaises(AssertionError):
            FlowExecutionUpdate(selected_outputs={"bad": object()})
        with self.assertRaises(AssertionError):
            FlowExecutionUpdate(loop_counters={"repair": -1})
        with self.assertRaises(AssertionError):
            FlowExecutionUpdate(pause_tokens={"review": ""})
        with self.assertRaises(AssertionError):
            FlowExecutionUpdate(
                artifact_refs=cast(
                    tuple[Mapping[str, object], ...],
                    (object(),),
                )
            )
        with self.assertRaises(AssertionError):
            flow_node_attempt_from_snapshot(
                {
                    "node": "start",
                    "attempt": True,
                    "state": "running",
                }
            )
        with self.assertRaises(AssertionError):
            flow_trace_from_snapshot({"nodes": {"bad": "shape"}, "edges": []})


class InMemoryFlowStateStoreTest(IsolatedAsyncioTestCase):
    async def test_memory_store_creates_updates_and_detects_conflicts(
        self,
    ) -> None:
        lookup = ExistingTaskRunLookup()
        clock = SequenceClock()
        store = InMemoryFlowStateStore(task_store=lookup, clock=clock)
        selected_outputs = {"answer": {"items": ["one"]}}

        created = await store.create_flow_execution(
            "run-1",
            trace=_trace(),
            selected_outputs=selected_outputs,
            metadata={"created": True},
        )
        selected_outputs["answer"] = {"items": ["changed"]}
        attempt = FlowNodeAttemptRecord(
            node="start",
            attempt=1,
            state=FlowNodeState.SUCCEEDED,
            artifact_refs=({"artifact_id": "artifact-1", "store": "memory"},),
        )
        updated_trace = created.trace.with_node_state(
            "start",
            FlowNodeState.SUCCEEDED,
            attempts=1,
        )

        updated = await store.update_flow_execution(
            "run-1",
            FlowExecutionUpdate(
                trace=updated_trace,
                node_attempts=(attempt,),
                node_outputs={"start": {"result": "stored"}},
                selected_outputs={"answer": ["stored", "list"]},
                loop_counters={"loop": 1},
                pause_tokens={"review": "pause-1"},
                diagnostics=(_diagnostic(),),
                artifact_refs=(
                    {"artifact_id": "artifact-root", "store": "memory"},
                ),
                metadata={"updated": True},
            ),
            expected_revision=created.revision,
        )

        self.assertEqual(lookup.run_ids, ["run-1"])
        self.assertEqual(updated.revision, 2)
        self.assertGreater(updated.updated_at, created.updated_at)
        self.assertEqual(updated.trace.nodes[0].state, FlowNodeState.SUCCEEDED)
        self.assertEqual(updated.node_attempts, (attempt,))
        self.assertEqual(
            dict(updated.node_outputs),
            {"start": {"result": "stored"}},
        )
        self.assertEqual(
            dict(updated.selected_outputs), {"answer": ("stored", "list")}
        )
        self.assertEqual(dict(updated.loop_counters), {"loop": 1})
        self.assertEqual(dict(updated.pause_tokens), {"review": "pause-1"})
        self.assertEqual(dict(updated.metadata), {"updated": True})
        self.assertEqual(await store.get_flow_execution("run-1"), updated)
        with self.assertRaises(TaskStoreConflictError):
            await store.create_flow_execution("run-1", trace=_trace())
        with self.assertRaises(TaskStoreConflictError):
            await store.update_flow_execution(
                "run-1",
                FlowExecutionUpdate(metadata={"stale": True}),
                expected_revision=created.revision,
            )
        with self.assertRaises(TaskStoreNotFoundError):
            await store.get_flow_execution("missing")
        with self.assertRaises(TaskStoreNotFoundError):
            await store.update_flow_execution(
                "missing",
                FlowExecutionUpdate(metadata={"missing": True}),
                expected_revision=1,
            )

    async def test_memory_store_can_skip_task_run_lookup(self) -> None:
        store = InMemoryFlowStateStore()

        created = await store.create_flow_execution("run-1", trace=_trace())

        self.assertEqual(created.task_run_id, "run-1")
        self.assertEqual(created.created_at.tzinfo, UTC)

    async def test_memory_store_rejects_missing_task_run(self) -> None:
        store = InMemoryFlowStateStore(task_store=MissingTaskRunLookup())

        with self.assertRaises(TaskStoreNotFoundError):
            await store.create_flow_execution("run-1", trace=_trace())


class PgsqlFlowStateStoreTest(IsolatedAsyncioTestCase):
    async def test_pgsql_store_persists_flow_execution_state(self) -> None:
        database = FakePgsqlTaskDatabase()
        clock = SequenceClock()
        task_store = PgsqlTaskStore(
            database,
            clock=clock,
            id_factory=SequenceIds(),
        )
        await task_store.register_definition(
            definition(),
            definition_hash="hash-a",
        )
        run = await task_store.create_run(
            TaskExecutionRequest(definition_id="hash-a")
        )
        flow_store = PgsqlFlowStateStore(database, clock=clock)

        created = await flow_store.create_flow_execution(
            run.run_id,
            trace=_trace(),
            diagnostics=(_diagnostic(),),
        )
        fetched = await flow_store.get_flow_execution(run.run_id)
        updated = await flow_store.update_flow_execution(
            run.run_id,
            FlowExecutionUpdate(
                selected_outputs={"answer": {"ok": True}},
                node_outputs={"start": {"answer": {"ok": True}}},
                loop_counters={"repair": 3},
                pause_tokens={"review": "pause-1"},
                artifact_refs=(
                    {"artifact_id": "artifact-1", "store": "pgsql"},
                ),
                metadata={"safe": "metadata"},
            ),
            expected_revision=created.revision,
        )

        self.assertEqual(fetched.task_run_id, run.run_id)
        self.assertEqual(updated.revision, 2)
        self.assertEqual(
            dict(updated.node_outputs),
            {"start": {"answer": {"ok": True}}},
        )
        self.assertEqual(
            dict(updated.selected_outputs), {"answer": {"ok": True}}
        )
        self.assertEqual(dict(updated.loop_counters), {"repair": 3})
        self.assertEqual(dict(updated.pause_tokens), {"review": "pause-1"})
        self.assertEqual(updated.artifact_refs[0]["store"], "pgsql")
        self.assertNotIn(
            "private-source-token",
            str(database.flow_executions[run.run_id]),
        )
        with self.assertRaises(TaskStoreConflictError):
            await flow_store.create_flow_execution(run.run_id, trace=_trace())
        with self.assertRaises(TaskStoreConflictError):
            await flow_store.update_flow_execution(
                run.run_id,
                FlowExecutionUpdate(metadata={"stale": True}),
                expected_revision=created.revision,
            )
        with self.assertRaises(TaskStoreNotFoundError):
            await flow_store.get_flow_execution("missing")
        with self.assertRaises(TaskStoreNotFoundError):
            await flow_store.create_flow_execution("missing", trace=_trace())
        with self.assertRaises(TaskStoreNotFoundError):
            await flow_store.update_flow_execution(
                "missing",
                FlowExecutionUpdate(metadata={"missing": True}),
                expected_revision=1,
            )

    async def test_pgsql_store_rolls_back_unmatched_insert(self) -> None:
        database = FakePgsqlTaskDatabase()
        clock = SequenceClock()
        task_store = PgsqlTaskStore(
            database,
            clock=clock,
            id_factory=SequenceIds(),
        )
        await task_store.register_definition(
            definition(),
            definition_hash="hash-a",
        )
        run = await task_store.create_run(
            TaskExecutionRequest(definition_id="hash-a")
        )
        flow_store = PgsqlFlowStateStore(database, clock=clock)
        original = FakeCursor._insert_flow_execution

        def unmatched_insert(
            self: FakeCursor,
            params: tuple[object, ...],
        ) -> dict[str, object] | None:
            _ = self, params
            return None

        FakeCursor._insert_flow_execution = unmatched_insert
        try:
            with self.assertRaises(TaskStoreConflictError):
                await flow_store.create_flow_execution(
                    run.run_id, trace=_trace()
                )
        finally:
            FakeCursor._insert_flow_execution = original

        self.assertEqual(database.flow_executions, {})

    async def test_pgsql_store_maps_transaction_errors(self) -> None:
        unique_failure = PgsqlFailure(
            category=PgsqlFailureCategory.UNIQUE_CONFLICT,
            code="23505",
            retryable=False,
        )
        unknown_failure = PgsqlFailure(
            category=PgsqlFailureCategory.UNKNOWN,
            code="99999",
            retryable=False,
        )

        cases: tuple[tuple[BaseException, type[BaseException]], ...] = (
            (PgsqlOperationError(unique_failure), TaskStoreConflictError),
            (PgsqlOperationError(unknown_failure), TaskStoreError),
            (UniqueSqlStateError("duplicate"), TaskStoreConflictError),
            (UnknownSqlStateError("private backend detail"), TaskStoreError),
            (AssertionError("bad assertion"), AssertionError),
            (CancelledError(), CancelledError),
        )
        for error, expected in cases:
            with self.subTest(error=type(error).__name__):
                store = PgsqlFlowStateStore(cast(Any, ErrorDatabase(error)))
                with self.assertRaises(expected):
                    await store.get_flow_execution("run-1")


def _trace() -> FlowExecutionTrace:
    return FlowExecutionTrace(
        nodes=(FlowNodeTrace(node="start"),),
        edges=(FlowEdgeTrace(index=0, source="start", target="finish"),),
    )


def _diagnostic() -> FlowDiagnostic:
    return FlowDiagnostic(
        code="flow.execution.node_failed",
        category=FlowDiagnosticCategory.EXECUTION,
        path="nodes.start",
        source_span=FlowSourceSpan(
            start_line=1,
            start_column=1,
            end_line=1,
            end_column=6,
            source="private-source-token",
        ),
        message="Flow node failed.",
        hint="Inspect the safe node state.",
        related_spans=(
            FlowSourceSpan(
                start_line=2,
                start_column=1,
                end_line=2,
                end_column=4,
                source="private-source-token",
            ),
        ),
    )


if __name__ == "__main__":
    main()
