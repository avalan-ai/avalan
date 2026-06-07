from .diagnostics import FlowDiagnostic
from .plan import FlowExecutionPlan

from dataclasses import dataclass, replace
from enum import StrEnum


class FlowNodeState(StrEnum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    SKIPPED = "skipped"
    FAILED = "failed"
    RETRYING = "retrying"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class FlowEdgeState(StrEnum):
    PENDING = "pending"
    ELIGIBLE = "eligible"
    TAKEN = "taken"
    SUPPRESSED = "suppressed"
    FAILED = "failed"


def _assert_string(value: str, field_name: str) -> None:
    assert (
        isinstance(value, str) and value.strip()
    ), f"{field_name} must be a non-empty string"


def _assert_duration(value: int | float | None, field_name: str) -> None:
    if value is not None:
        assert isinstance(value, int | float) and not isinstance(
            value,
            bool,
        ), f"{field_name} must be a number"
        assert value >= 0, f"{field_name} must be non-negative"


def _assert_diagnostics(value: tuple[FlowDiagnostic, ...]) -> None:
    assert isinstance(value, tuple), "diagnostics must be a tuple"
    for diagnostic in value:
        assert isinstance(diagnostic, FlowDiagnostic)


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowNodeTrace:
    node: str
    state: FlowNodeState = FlowNodeState.PENDING
    attempts: int = 0
    duration_ms: int | float | None = None
    diagnostics: tuple[FlowDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        _assert_string(self.node, "node")
        assert isinstance(self.state, FlowNodeState)
        assert isinstance(self.attempts, int) and not isinstance(
            self.attempts,
            bool,
        )
        assert self.attempts >= 0
        _assert_duration(self.duration_ms, "duration_ms")
        _assert_diagnostics(self.diagnostics)

    @property
    def public_diagnostics(self) -> tuple[dict[str, object], ...]:
        return tuple(
            diagnostic.as_public_dict() for diagnostic in self.diagnostics
        )

    def as_public_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "node": self.node,
            "state": self.state.value,
            "attempts": self.attempts,
        }
        if self.duration_ms is not None:
            value["duration_ms"] = self.duration_ms
        if self.diagnostics:
            value["diagnostics"] = self.public_diagnostics
        return value


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowEdgeTrace:
    index: int
    source: str
    target: str
    state: FlowEdgeState = FlowEdgeState.PENDING
    duration_ms: int | float | None = None
    diagnostics: tuple[FlowDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        assert isinstance(self.index, int) and not isinstance(
            self.index,
            bool,
        )
        assert self.index >= 0
        _assert_string(self.source, "source")
        _assert_string(self.target, "target")
        assert isinstance(self.state, FlowEdgeState)
        _assert_duration(self.duration_ms, "duration_ms")
        _assert_diagnostics(self.diagnostics)

    @property
    def public_diagnostics(self) -> tuple[dict[str, object], ...]:
        return tuple(
            diagnostic.as_public_dict() for diagnostic in self.diagnostics
        )

    def as_public_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "index": self.index,
            "source": self.source,
            "target": self.target,
            "state": self.state.value,
        }
        if self.duration_ms is not None:
            value["duration_ms"] = self.duration_ms
        if self.diagnostics:
            value["diagnostics"] = self.public_diagnostics
        return value


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowExecutionTrace:
    nodes: tuple[FlowNodeTrace, ...]
    edges: tuple[FlowEdgeTrace, ...] = ()

    def __post_init__(self) -> None:
        assert isinstance(self.nodes, tuple)
        node_names: set[str] = set()
        for node in self.nodes:
            assert isinstance(node, FlowNodeTrace)
            assert node.node not in node_names, "node traces must be unique"
            node_names.add(node.node)
        assert isinstance(self.edges, tuple)
        edge_indexes: set[int] = set()
        for edge in self.edges:
            assert isinstance(edge, FlowEdgeTrace)
            assert edge.index not in edge_indexes, "edge traces must be unique"
            edge_indexes.add(edge.index)

    @classmethod
    def from_plan(cls, plan: FlowExecutionPlan) -> "FlowExecutionTrace":
        assert isinstance(plan, FlowExecutionPlan)
        return cls(
            nodes=tuple(FlowNodeTrace(node=node.name) for node in plan.nodes),
            edges=tuple(
                FlowEdgeTrace(
                    index=edge.index,
                    source=edge.source,
                    target=edge.target,
                )
                for edge in plan.edges
            ),
        )

    def with_node_state(
        self,
        node: str,
        state: FlowNodeState,
        *,
        attempts: int | None = None,
        duration_ms: int | float | None = None,
        diagnostics: tuple[FlowDiagnostic, ...] = (),
    ) -> "FlowExecutionTrace":
        _assert_string(node, "node")
        assert isinstance(state, FlowNodeState)
        replacements = []
        found = False
        for trace in self.nodes:
            if trace.node != node:
                replacements.append(trace)
                continue
            found = True
            replacements.append(
                replace(
                    trace,
                    state=state,
                    attempts=trace.attempts if attempts is None else attempts,
                    duration_ms=duration_ms,
                    diagnostics=diagnostics,
                )
            )
        assert found, f"unknown node trace: {node}"
        return replace(self, nodes=tuple(replacements))

    def with_edge_state(
        self,
        index: int,
        state: FlowEdgeState,
        *,
        duration_ms: int | float | None = None,
        diagnostics: tuple[FlowDiagnostic, ...] = (),
    ) -> "FlowExecutionTrace":
        assert isinstance(index, int) and not isinstance(index, bool)
        assert isinstance(state, FlowEdgeState)
        replacements = []
        found = False
        for trace in self.edges:
            if trace.index != index:
                replacements.append(trace)
                continue
            found = True
            replacements.append(
                replace(
                    trace,
                    state=state,
                    duration_ms=duration_ms,
                    diagnostics=diagnostics,
                )
            )
        assert found, f"unknown edge trace: {index}"
        return replace(self, edges=tuple(replacements))

    def as_public_dict(self) -> dict[str, object]:
        return {
            "nodes": tuple(node.as_public_dict() for node in self.nodes),
            "edges": tuple(edge.as_public_dict() for edge in self.edges),
        }
