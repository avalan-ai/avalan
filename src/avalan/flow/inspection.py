from ..task.artifact import TaskArtifactRef
from ..task.context import TaskInputFile
from ..task.store import freeze_snapshot_metadata
from .definition import FlowEdgeKind, FlowNodeKind
from .diagnostics import FlowDiagnostic
from .plan import FlowEdgePlan, FlowExecutionPlan, FlowNodePlan
from .runtime import FlowPlanExecutionResult
from .state import (
    FlowEdgeState,
    FlowExecutionTrace,
    FlowNodeState,
    FlowNodeTrace,
)
from .store import FlowExecutionRecord, FlowSnapshotMetadata

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import cast


class FlowInspectionRunState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class FlowReviewState(StrEnum):
    PENDING = "pending"
    PAUSED = "paused"
    RESUMED = "resumed"
    SUCCEEDED = "succeeded"
    SKIPPED = "skipped"
    FAILED = "failed"
    CANCELLED = "cancelled"


_FLOW_REVIEW_STATE_VALUES = frozenset(state.value for state in FlowReviewState)


def _assert_non_empty_string(value: str, field_name: str) -> None:
    assert (
        isinstance(value, str) and value.strip()
    ), f"{field_name} must be a non-empty string"


def _assert_optional_duration(
    value: int | float | None,
    field_name: str,
) -> None:
    if value is not None:
        assert isinstance(value, int | float) and not isinstance(value, bool)
        assert value >= 0, f"{field_name} must be non-negative"


def _assert_diagnostics(value: tuple[FlowDiagnostic, ...]) -> None:
    assert isinstance(value, tuple)
    for diagnostic in value:
        assert isinstance(diagnostic, FlowDiagnostic)


def _public_diagnostics(
    diagnostics: tuple[FlowDiagnostic, ...],
) -> tuple[dict[str, object], ...]:
    _assert_diagnostics(diagnostics)
    return tuple(diagnostic.as_public_dict() for diagnostic in diagnostics)


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowNodeInspection:
    node: str
    state: FlowNodeState
    attempts: int
    kind: FlowNodeKind | None = None
    duration_ms: int | float | None = None
    diagnostics: tuple[FlowDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.node, "node")
        assert isinstance(self.state, FlowNodeState)
        assert isinstance(self.attempts, int) and not isinstance(
            self.attempts,
            bool,
        )
        assert self.attempts >= 0
        if self.kind is not None:
            assert isinstance(self.kind, FlowNodeKind)
        _assert_optional_duration(self.duration_ms, "duration_ms")
        _assert_diagnostics(self.diagnostics)

    def as_public_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "node": self.node,
            "state": self.state.value,
            "attempts": self.attempts,
        }
        if self.kind is not None:
            value["kind"] = self.kind.value
        if self.duration_ms is not None:
            value["duration_ms"] = self.duration_ms
        if self.diagnostics:
            value["diagnostics"] = _public_diagnostics(self.diagnostics)
        return value


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowEdgeInspection:
    index: int
    source: str
    target: str
    state: FlowEdgeState
    kind: FlowEdgeKind | None = None
    duration_ms: int | float | None = None
    diagnostics: tuple[FlowDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        assert isinstance(self.index, int) and not isinstance(
            self.index,
            bool,
        )
        assert self.index >= 0
        _assert_non_empty_string(self.source, "source")
        _assert_non_empty_string(self.target, "target")
        assert isinstance(self.state, FlowEdgeState)
        if self.kind is not None:
            assert isinstance(self.kind, FlowEdgeKind)
        _assert_optional_duration(self.duration_ms, "duration_ms")
        _assert_diagnostics(self.diagnostics)

    def as_public_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "index": self.index,
            "source": self.source,
            "target": self.target,
            "state": self.state.value,
        }
        if self.kind is not None:
            value["kind"] = self.kind.value
        if self.duration_ms is not None:
            value["duration_ms"] = self.duration_ms
        if self.diagnostics:
            value["diagnostics"] = _public_diagnostics(self.diagnostics)
        return value


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowRetryInspection:
    node: str
    attempts: int
    exhausted: bool
    max_attempts: int | None = None
    retryable_categories: tuple[str, ...] = ()
    non_retryable_categories: tuple[str, ...] = ()
    exhausted_route: str | None = None

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.node, "node")
        assert isinstance(self.attempts, int) and not isinstance(
            self.attempts,
            bool,
        )
        assert self.attempts >= 0
        assert isinstance(self.exhausted, bool)
        if self.max_attempts is not None:
            assert isinstance(self.max_attempts, int)
            assert not isinstance(self.max_attempts, bool)
            assert self.max_attempts > 0
        for category in self.retryable_categories:
            _assert_non_empty_string(category, "retryable_categories")
        for category in self.non_retryable_categories:
            _assert_non_empty_string(category, "non_retryable_categories")
        if self.exhausted_route is not None:
            _assert_non_empty_string(self.exhausted_route, "exhausted_route")

    def as_public_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "node": self.node,
            "attempts": self.attempts,
            "exhausted": self.exhausted,
        }
        if self.max_attempts is not None:
            value["max_attempts"] = self.max_attempts
        if self.retryable_categories:
            value["retryable_categories"] = self.retryable_categories
        if self.non_retryable_categories:
            value["non_retryable_categories"] = self.non_retryable_categories
        if self.exhausted_route is not None:
            value["exhausted_route"] = self.exhausted_route
        return value


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowLoopInspection:
    node: str
    iterations: int
    state: FlowNodeState
    max_iterations: int | None = None
    max_elapsed_seconds: int | float | None = None
    limit_route: str | None = None

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.node, "node")
        assert isinstance(self.iterations, int) and not isinstance(
            self.iterations,
            bool,
        )
        assert self.iterations >= 0
        assert isinstance(self.state, FlowNodeState)
        if self.max_iterations is not None:
            assert isinstance(self.max_iterations, int)
            assert not isinstance(self.max_iterations, bool)
            assert self.max_iterations > 0
        _assert_optional_duration(
            self.max_elapsed_seconds,
            "max_elapsed_seconds",
        )
        if self.limit_route is not None:
            _assert_non_empty_string(self.limit_route, "limit_route")

    def as_public_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "node": self.node,
            "iterations": self.iterations,
            "state": self.state.value,
        }
        if self.max_iterations is not None:
            value["max_iterations"] = self.max_iterations
        if self.max_elapsed_seconds is not None:
            value["max_elapsed_seconds"] = self.max_elapsed_seconds
        if self.limit_route is not None:
            value["limit_route"] = self.limit_route
        return value


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowArtifactInspection:
    artifact_id: str
    store: str
    media_type: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.artifact_id, "artifact_id")
        _assert_non_empty_string(self.store, "store")
        if self.media_type is not None:
            _assert_non_empty_string(self.media_type, "media_type")
        if self.size_bytes is not None:
            assert isinstance(self.size_bytes, int)
            assert not isinstance(self.size_bytes, bool)
            assert self.size_bytes >= 0
        if self.sha256 is not None:
            _assert_non_empty_string(self.sha256, "sha256")

    def as_public_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "artifact_id": self.artifact_id,
            "store": self.store,
        }
        if self.media_type is not None:
            value["media_type"] = self.media_type
        if self.size_bytes is not None:
            value["size_bytes"] = self.size_bytes
        if self.sha256 is not None:
            value["sha256"] = self.sha256
        return value


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowReviewInspection:
    node: str
    state: FlowReviewState
    has_pause_token: bool = False
    allowed_decisions: tuple[str, ...] = ()
    decision: str | None = None
    timeout_seconds: int | float | None = None

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.node, "node")
        assert isinstance(self.state, FlowReviewState)
        assert isinstance(self.has_pause_token, bool)
        for decision in self.allowed_decisions:
            _assert_non_empty_string(decision, "allowed_decisions")
        if self.decision is not None:
            _assert_non_empty_string(self.decision, "decision")
        _assert_optional_duration(self.timeout_seconds, "timeout_seconds")

    def as_public_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "node": self.node,
            "state": self.state.value,
            "has_pause_token": self.has_pause_token,
        }
        if self.allowed_decisions:
            value["allowed_decisions"] = self.allowed_decisions
        if self.decision is not None:
            value["decision"] = self.decision
        if self.timeout_seconds is not None:
            value["timeout_seconds"] = self.timeout_seconds
        return value


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowInspection:
    state: FlowInspectionRunState
    nodes: tuple[FlowNodeInspection, ...]
    edges: tuple[FlowEdgeInspection, ...] = ()
    retries: tuple[FlowRetryInspection, ...] = ()
    loops: tuple[FlowLoopInspection, ...] = ()
    artifacts: tuple[FlowArtifactInspection, ...] = ()
    reviews: tuple[FlowReviewInspection, ...] = ()
    selected_outputs: tuple[str, ...] = ()
    diagnostics: tuple[FlowDiagnostic, ...] = ()
    flow_name: str | None = None
    flow_version: str | None = None
    flow_revision: str | None = None
    task_run_id: str | None = None
    record_revision: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.state, FlowInspectionRunState)
        for node in self.nodes:
            assert isinstance(node, FlowNodeInspection)
        for edge in self.edges:
            assert isinstance(edge, FlowEdgeInspection)
        for retry in self.retries:
            assert isinstance(retry, FlowRetryInspection)
        for loop in self.loops:
            assert isinstance(loop, FlowLoopInspection)
        for artifact in self.artifacts:
            assert isinstance(artifact, FlowArtifactInspection)
        for review in self.reviews:
            assert isinstance(review, FlowReviewInspection)
        for output in self.selected_outputs:
            _assert_non_empty_string(output, "selected_outputs")
        _assert_diagnostics(self.diagnostics)
        for field_name in ("flow_name", "flow_version", "flow_revision"):
            value = getattr(self, field_name)
            if value is not None:
                _assert_non_empty_string(value, field_name)
        if self.task_run_id is not None:
            _assert_non_empty_string(self.task_run_id, "task_run_id")
        if self.record_revision is not None:
            assert isinstance(self.record_revision, int)
            assert not isinstance(self.record_revision, bool)
            assert self.record_revision > 0
        if self.created_at is not None:
            assert isinstance(self.created_at, datetime)
        if self.updated_at is not None:
            assert isinstance(self.updated_at, datetime)

    def as_public_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "state": self.state.value,
            "nodes": tuple(node.as_public_dict() for node in self.nodes),
            "edges": tuple(edge.as_public_dict() for edge in self.edges),
        }
        if self.flow_name is not None:
            value["flow_name"] = self.flow_name
        if self.flow_version is not None:
            value["flow_version"] = self.flow_version
        if self.flow_revision is not None:
            value["flow_revision"] = self.flow_revision
        if self.task_run_id is not None:
            value["task_run_id"] = self.task_run_id
        if self.record_revision is not None:
            value["record_revision"] = self.record_revision
        if self.retries:
            value["retries"] = tuple(
                retry.as_public_dict() for retry in self.retries
            )
        if self.loops:
            value["loops"] = tuple(
                loop.as_public_dict() for loop in self.loops
            )
        if self.artifacts:
            value["artifacts"] = tuple(
                artifact.as_public_dict() for artifact in self.artifacts
            )
        if self.reviews:
            value["reviews"] = tuple(
                review.as_public_dict() for review in self.reviews
            )
        if self.selected_outputs:
            value["selected_outputs"] = self.selected_outputs
        if self.diagnostics:
            value["diagnostics"] = _public_diagnostics(self.diagnostics)
        if self.created_at is not None:
            value["created_at"] = self.created_at.isoformat()
        if self.updated_at is not None:
            value["updated_at"] = self.updated_at.isoformat()
        return value

    def export_sanitized_trace(self) -> FlowSnapshotMetadata:
        return freeze_snapshot_metadata(self.as_public_dict())


def inspect_flow_result(
    result: FlowPlanExecutionResult,
    *,
    plan: FlowExecutionPlan | None = None,
) -> FlowInspection:
    assert isinstance(result, FlowPlanExecutionResult)
    if plan is not None:
        assert isinstance(plan, FlowExecutionPlan)
    return _inspection_from_parts(
        trace=result.trace,
        plan=plan,
        selected_outputs=tuple(result.outputs.keys()),
        diagnostics=result.diagnostics,
        artifacts=_artifact_inspections_from_value(result.outputs),
        pause_tokens=result.pause_tokens,
    )


def inspect_flow_record(
    record: FlowExecutionRecord,
    *,
    plan: FlowExecutionPlan | None = None,
) -> FlowInspection:
    assert isinstance(record, FlowExecutionRecord)
    if plan is not None:
        assert isinstance(plan, FlowExecutionPlan)
    flow_name, flow_version, flow_revision = _flow_identity(plan, record)
    return _inspection_from_parts(
        trace=record.trace,
        plan=plan,
        selected_outputs=tuple(record.selected_outputs.keys()),
        diagnostics=_record_diagnostics(record),
        artifacts=_record_artifacts(record),
        loop_counters=record.loop_counters,
        pause_tokens=record.pause_tokens,
        review_audit=_review_audit(record),
        flow_name=flow_name,
        flow_version=flow_version,
        flow_revision=flow_revision,
        task_run_id=record.task_run_id,
        record_revision=record.revision,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def export_sanitized_flow_trace(
    value: FlowInspection | FlowPlanExecutionResult | FlowExecutionRecord,
    *,
    plan: FlowExecutionPlan | None = None,
) -> FlowSnapshotMetadata:
    if isinstance(value, FlowInspection):
        return value.export_sanitized_trace()
    if isinstance(value, FlowPlanExecutionResult):
        return inspect_flow_result(value, plan=plan).export_sanitized_trace()
    if isinstance(value, FlowExecutionRecord):
        return inspect_flow_record(value, plan=plan).export_sanitized_trace()
    raise AssertionError("value must be a flow inspection, result, or record")


def _inspection_from_parts(
    *,
    trace: FlowExecutionTrace,
    plan: FlowExecutionPlan | None,
    selected_outputs: tuple[str, ...],
    diagnostics: tuple[FlowDiagnostic, ...],
    artifacts: tuple[FlowArtifactInspection, ...],
    loop_counters: Mapping[str, int] | None = None,
    pause_tokens: Mapping[str, str] | None = None,
    review_audit: Mapping[str, Mapping[str, object]] | None = None,
    flow_name: str | None = None,
    flow_version: str | None = None,
    flow_revision: str | None = None,
    task_run_id: str | None = None,
    record_revision: int | None = None,
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
) -> FlowInspection:
    assert isinstance(trace, FlowExecutionTrace)
    if plan is not None:
        assert isinstance(plan, FlowExecutionPlan)
        flow_name = flow_name or plan.name
        flow_version = flow_version or plan.version
        flow_revision = flow_revision or plan.revision
    return FlowInspection(
        state=_run_state(trace, pause_tokens or {}, diagnostics),
        nodes=_node_inspections(trace, plan),
        edges=_edge_inspections(trace, plan),
        retries=_retry_inspections(trace, plan, loop_counters or {}),
        loops=_loop_inspections(trace, plan, loop_counters or {}),
        artifacts=artifacts,
        reviews=_review_inspections(
            trace,
            plan,
            pause_tokens or {},
            review_audit or {},
        ),
        selected_outputs=selected_outputs,
        diagnostics=diagnostics,
        flow_name=flow_name,
        flow_version=flow_version,
        flow_revision=flow_revision,
        task_run_id=task_run_id,
        record_revision=record_revision,
        created_at=created_at,
        updated_at=updated_at,
    )


def _node_inspections(
    trace: FlowExecutionTrace,
    plan: FlowExecutionPlan | None,
) -> tuple[FlowNodeInspection, ...]:
    plan_nodes = _plan_node_map(plan)
    return tuple(
        FlowNodeInspection(
            node=node.node,
            state=node.state,
            attempts=node.attempts,
            kind=(
                plan_nodes[node.node].kind if node.node in plan_nodes else None
            ),
            duration_ms=node.duration_ms,
            diagnostics=node.diagnostics,
        )
        for node in trace.nodes
    )


def _edge_inspections(
    trace: FlowExecutionTrace,
    plan: FlowExecutionPlan | None,
) -> tuple[FlowEdgeInspection, ...]:
    plan_edges = _plan_edge_map(plan)
    return tuple(
        FlowEdgeInspection(
            index=edge.index,
            source=edge.source,
            target=edge.target,
            state=edge.state,
            kind=(
                plan_edges[edge.index].kind
                if edge.index in plan_edges
                else None
            ),
            duration_ms=edge.duration_ms,
            diagnostics=edge.diagnostics,
        )
        for edge in trace.edges
    )


def _retry_inspections(
    trace: FlowExecutionTrace,
    plan: FlowExecutionPlan | None,
    loop_counters: Mapping[str, int],
) -> tuple[FlowRetryInspection, ...]:
    plan_nodes = _plan_node_map(plan)
    retries: list[FlowRetryInspection] = []
    for node in trace.nodes:
        if node.node in loop_counters:
            continue
        retry = (
            plan_nodes[node.node].retry if node.node in plan_nodes else None
        )
        if retry is None and node.attempts <= 1:
            continue
        retries.append(
            FlowRetryInspection(
                node=node.node,
                attempts=node.attempts,
                max_attempts=retry.max_attempts if retry is not None else None,
                retryable_categories=(
                    retry.retryable_categories if retry is not None else ()
                ),
                non_retryable_categories=(
                    retry.non_retryable_categories if retry is not None else ()
                ),
                exhausted=(
                    retry is not None
                    and node.state == FlowNodeState.FAILED
                    and node.attempts >= retry.max_attempts
                ),
                exhausted_route=(
                    retry.exhausted_route if retry is not None else None
                ),
            )
        )
    return tuple(retries)


def _loop_inspections(
    trace: FlowExecutionTrace,
    plan: FlowExecutionPlan | None,
    loop_counters: Mapping[str, int],
) -> tuple[FlowLoopInspection, ...]:
    trace_nodes = _trace_node_map(trace)
    plan_nodes = _plan_node_map(plan)
    loop_node_names = set(loop_counters)
    loop_node_names.update(
        node.name for node in plan_nodes.values() if node.loop is not None
    )
    loops: list[FlowLoopInspection] = []
    for node_name in sorted(loop_node_names):
        trace_node = trace_nodes.get(node_name)
        plan_node = plan_nodes.get(node_name)
        loop = plan_node.loop if plan_node is not None else None
        loops.append(
            FlowLoopInspection(
                node=node_name,
                iterations=loop_counters.get(
                    node_name,
                    trace_node.attempts if trace_node is not None else 0,
                ),
                state=(
                    trace_node.state
                    if trace_node is not None
                    else FlowNodeState.PENDING
                ),
                max_iterations=(
                    loop.max_iterations if loop is not None else None
                ),
                max_elapsed_seconds=(
                    loop.max_elapsed_seconds if loop is not None else None
                ),
                limit_route=loop.limit_route if loop is not None else None,
            )
        )
    return tuple(loops)


def _review_inspections(
    trace: FlowExecutionTrace,
    plan: FlowExecutionPlan | None,
    pause_tokens: Mapping[str, str],
    review_audit: Mapping[str, Mapping[str, object]],
) -> tuple[FlowReviewInspection, ...]:
    trace_nodes = _trace_node_map(trace)
    plan_nodes = _plan_node_map(plan)
    review_node_names = set(pause_tokens)
    review_node_names.update(review_audit)
    review_node_names.update(
        node.name
        for node in plan_nodes.values()
        if node.kind == FlowNodeKind.HUMAN_REVIEW
    )
    reviews: list[FlowReviewInspection] = []
    for node_name in sorted(review_node_names):
        plan_node = plan_nodes.get(node_name)
        trace_node = trace_nodes.get(node_name)
        audit_entry = review_audit.get(node_name, {})
        allowed_decisions, timeout_seconds = _review_request(
            plan_node,
            audit_entry,
        )
        reviews.append(
            FlowReviewInspection(
                node=node_name,
                state=_review_state(trace_node, audit_entry),
                has_pause_token=node_name in pause_tokens,
                allowed_decisions=allowed_decisions,
                decision=_review_decision(audit_entry),
                timeout_seconds=timeout_seconds,
            )
        )
    return tuple(reviews)


def _review_request(
    plan_node: FlowNodePlan | None,
    audit_entry: Mapping[str, object],
) -> tuple[tuple[str, ...], int | float | None]:
    if plan_node is not None:
        decisions = plan_node.config.get("allowed_decisions")
        timeout_seconds = plan_node.config.get("timeout_seconds")
        return (
            _safe_string_tuple(decisions),
            (
                timeout_seconds
                if isinstance(timeout_seconds, int | float)
                and not isinstance(timeout_seconds, bool)
                else None
            ),
        )
    request = audit_entry.get("request")
    if not isinstance(request, Mapping):
        return (), None
    timeout = request.get("timeout_seconds")
    return (
        _safe_string_tuple(request.get("allowed_decisions")),
        (
            timeout
            if isinstance(timeout, int | float)
            and not isinstance(timeout, bool)
            else None
        ),
    )


def _review_state(
    trace_node: FlowNodeTrace | None,
    audit_entry: Mapping[str, object],
) -> FlowReviewState:
    audit_state = audit_entry.get("state")
    if (
        isinstance(audit_state, str)
        and audit_state in _FLOW_REVIEW_STATE_VALUES
    ):
        return FlowReviewState(audit_state)
    if trace_node is None:
        return FlowReviewState.PENDING
    match trace_node.state:
        case FlowNodeState.PAUSED:
            return FlowReviewState.PAUSED
        case FlowNodeState.SUCCEEDED:
            return FlowReviewState.SUCCEEDED
        case FlowNodeState.SKIPPED:
            return FlowReviewState.SKIPPED
        case FlowNodeState.FAILED:
            return FlowReviewState.FAILED
        case FlowNodeState.CANCELLED:
            return FlowReviewState.CANCELLED
        case _:
            return FlowReviewState.PENDING


def _review_decision(
    audit_entry: Mapping[str, object],
) -> str | None:
    decision = audit_entry.get("decision")
    return decision if isinstance(decision, str) and decision.strip() else None


def _run_state(
    trace: FlowExecutionTrace,
    pause_tokens: Mapping[str, str],
    diagnostics: tuple[FlowDiagnostic, ...],
) -> FlowInspectionRunState:
    node_states = tuple(node.state for node in trace.nodes)
    edge_states = tuple(edge.state for edge in trace.edges)
    if any(state == FlowNodeState.CANCELLED for state in node_states):
        return FlowInspectionRunState.CANCELLED
    if pause_tokens or any(
        state == FlowNodeState.PAUSED for state in node_states
    ):
        return FlowInspectionRunState.PAUSED
    if (
        diagnostics
        or any(state == FlowNodeState.FAILED for state in node_states)
        or any(state == FlowEdgeState.FAILED for state in edge_states)
    ):
        return FlowInspectionRunState.FAILED
    if any(
        state
        in {
            FlowNodeState.READY,
            FlowNodeState.RETRYING,
            FlowNodeState.RUNNING,
        }
        for state in node_states
    ):
        return FlowInspectionRunState.RUNNING
    if node_states and all(
        state in {FlowNodeState.SKIPPED, FlowNodeState.SUCCEEDED}
        for state in node_states
    ):
        return FlowInspectionRunState.SUCCEEDED
    return FlowInspectionRunState.PENDING


def _record_diagnostics(
    record: FlowExecutionRecord,
) -> tuple[FlowDiagnostic, ...]:
    return record.diagnostics + _trace_diagnostics(record.trace)


def _trace_diagnostics(
    trace: FlowExecutionTrace,
) -> tuple[FlowDiagnostic, ...]:
    diagnostics: list[FlowDiagnostic] = []
    for node in trace.nodes:
        diagnostics.extend(node.diagnostics)
    for edge in trace.edges:
        diagnostics.extend(edge.diagnostics)
    return tuple(diagnostics)


def _record_artifacts(
    record: FlowExecutionRecord,
) -> tuple[FlowArtifactInspection, ...]:
    artifacts: list[FlowArtifactInspection] = []
    seen: set[tuple[str, str]] = set()
    for artifact in record.artifact_refs:
        _append_artifact_from_mapping(artifact, artifacts, seen)
    for attempt in record.node_attempts:
        for artifact in attempt.artifact_refs:
            _append_artifact_from_mapping(artifact, artifacts, seen)
    return tuple(artifacts)


def _artifact_inspections_from_value(
    value: object,
) -> tuple[FlowArtifactInspection, ...]:
    artifacts: list[FlowArtifactInspection] = []
    seen: set[tuple[str, str]] = set()
    _append_artifacts_from_value(value, artifacts, seen)
    return tuple(artifacts)


def _append_artifacts_from_value(
    value: object,
    artifacts: list[FlowArtifactInspection],
    seen: set[tuple[str, str]],
) -> None:
    if isinstance(value, TaskInputFile):
        if value.artifact_ref is not None:
            _append_artifact_ref(value.artifact_ref, artifacts, seen)
        return
    if isinstance(value, TaskArtifactRef):
        _append_artifact_ref(value, artifacts, seen)
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _append_artifacts_from_value(item, artifacts, seen)
        return
    if isinstance(value, list | tuple):
        for item in value:
            _append_artifacts_from_value(item, artifacts, seen)


def _append_artifact_ref(
    ref: TaskArtifactRef,
    artifacts: list[FlowArtifactInspection],
    seen: set[tuple[str, str]],
) -> None:
    summary = ref.summary(include_metadata=False, include_sha256=True)
    assert isinstance(summary, Mapping)
    _append_artifact_from_mapping(
        cast(Mapping[str, object], summary),
        artifacts,
        seen,
    )


def _append_artifact_from_mapping(
    value: Mapping[str, object],
    artifacts: list[FlowArtifactInspection],
    seen: set[tuple[str, str]],
) -> None:
    artifact = _artifact_from_mapping(value)
    if artifact is None:
        return
    key = (artifact.store, artifact.artifact_id)
    if key in seen:
        return
    seen.add(key)
    artifacts.append(artifact)


def _artifact_from_mapping(
    value: Mapping[str, object],
) -> FlowArtifactInspection | None:
    artifact_id = value.get("artifact_id")
    store = value.get("store")
    if not isinstance(artifact_id, str) or not isinstance(store, str):
        return None
    media_type = value.get("media_type")
    size_bytes = value.get("size_bytes")
    sha256 = value.get("sha256")
    return FlowArtifactInspection(
        artifact_id=artifact_id,
        store=store,
        media_type=media_type if isinstance(media_type, str) else None,
        size_bytes=size_bytes if isinstance(size_bytes, int) else None,
        sha256=sha256 if isinstance(sha256, str) else None,
    )


def _review_audit(
    record: FlowExecutionRecord,
) -> Mapping[str, Mapping[str, object]]:
    value = record.metadata.get("human_review_audit")
    if not isinstance(value, Mapping):
        return {}
    audit: dict[str, Mapping[str, object]] = {}
    for node, entry in value.items():
        if (
            isinstance(node, str)
            and node.strip()
            and isinstance(
                entry,
                Mapping,
            )
        ):
            audit[node] = entry
    return audit


def _flow_identity(
    plan: FlowExecutionPlan | None,
    record: FlowExecutionRecord,
) -> tuple[str | None, str | None, str | None]:
    if plan is not None:
        return plan.name, plan.version, plan.revision
    signature = record.metadata.get("strict_flow")
    if not isinstance(signature, Mapping):
        return None, None, None
    return (
        _optional_string(signature.get("name")),
        _optional_string(signature.get("version")),
        _optional_string(signature.get("revision")),
    )


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _safe_string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)


def _trace_node_map(
    trace: FlowExecutionTrace,
) -> Mapping[str, FlowNodeTrace]:
    return {node.node: node for node in trace.nodes}


def _plan_node_map(
    plan: FlowExecutionPlan | None,
) -> Mapping[str, FlowNodePlan]:
    if plan is None:
        return {}
    return plan.node_map


def _plan_edge_map(
    plan: FlowExecutionPlan | None,
) -> Mapping[int, FlowEdgePlan]:
    if plan is None:
        return {}
    return {edge.index: edge for edge in plan.edges}
