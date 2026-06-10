from ...flow import (
    FlowDefinition,
    FlowDefinitionLoader,
    FlowDiagnostic,
    FlowDiagnosticCategory,
    FlowDiagnosticSeverity,
    FlowExecutionRecord,
    FlowExecutionUpdate,
    FlowExecutor,
    FlowExecutorRunResult,
    FlowNodeRegistry,
    FlowStateStore,
    FlowView,
    FlowViewImportMode,
    compare_flow_topology,
    compile_flow_source,
    inspect_flow_graph_source,
    inspect_flow_record,
    parse_mermaid_view,
    render_flow_view,
)
from ...server.sse import sse_bytes, sse_headers
from ...task import TaskClientUnsupportedOperationError
from ...task.event import SanitizedTaskEvent
from ...task.store import TaskStoreConflictError, TaskStoreNotFoundError

from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from json import dumps
from typing import Any, Protocol, cast

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field
from starlette.responses import Response


class FlowTaskProtocolClient(Protocol):
    async def cancel(self, run_id: str) -> object: ...

    async def events(
        self,
        run_id: str,
        *,
        after_sequence: int | None = None,
    ) -> tuple[SanitizedTaskEvent, ...]: ...


class FlowDefinitionSourceRequest(BaseModel):
    source: str = Field(..., min_length=1)


class FlowMermaidRequest(BaseModel):
    source: str = Field(..., min_length=1)
    mode: FlowViewImportMode = FlowViewImportMode.PRESENTATION


class FlowMermaidCompareRequest(BaseModel):
    diagram_source: str = Field(..., min_length=1)
    definition_source: str = Field(..., min_length=1)
    mode: FlowViewImportMode = FlowViewImportMode.PRESENTATION


class FlowRunRequest(BaseModel):
    source: str = Field(..., min_length=1)
    inputs: dict[str, object] | None = None
    run_id: str | None = None
    concurrency_limit: int | None = Field(None, ge=1)


class FlowResumeRequest(BaseModel):
    source: str = Field(..., min_length=1)
    decisions: dict[str, dict[str, object]]
    inputs: dict[str, object] | None = None
    concurrency_limit: int | None = Field(None, ge=1)
    expected_revision: int | None = Field(None, ge=1)


class FlowRoute(APIRoute):
    def get_route_handler(self) -> Callable[[Request], Awaitable[Response]]:
        handler = super().get_route_handler()

        async def flow_route_handler(request: Request) -> Response:
            try:
                return await handler(request)
            except RequestValidationError as exc:
                return _request_validation_response(exc)

        return flow_route_handler


router = APIRouter(tags=["flows"], route_class=FlowRoute)


@router.post("/validate")
async def validate_flow(
    payload: FlowDefinitionSourceRequest,
    request: Request,
) -> dict[str, object]:
    """Validate a flow definition without executing nodes."""
    result = await _loader(request).loads_validation_result(payload.source)
    return _flow_result(
        ok=result.ok,
        diagnostics=result.diagnostics,
        definition=(
            _definition_public_dict(result.definition)
            if result.definition is not None
            else None
        ),
    )


@router.post("/compile")
async def compile_flow(
    payload: FlowDefinitionSourceRequest,
    request: Request,
) -> dict[str, object]:
    """Compile an authoring flow into strict public metadata."""
    result = await compile_flow_source(
        payload.source,
        registry=_flow_registry(request),
    )
    return cast(dict[str, object], _public_value(result.as_public_dict()))


@router.post("/graph/inspect")
async def inspect_graph(
    payload: FlowDefinitionSourceRequest,
    request: Request,
) -> dict[str, object]:
    """Inspect a static authoring graph without running nodes."""
    result = await inspect_flow_graph_source(
        payload.source,
        registry=_flow_registry(request),
    )
    return cast(dict[str, object], _public_value(result.as_public_dict()))


@router.post("/mermaid/parse")
async def parse_mermaid(
    payload: FlowMermaidRequest,
) -> dict[str, object]:
    """Parse Mermaid source into an inert Flow View."""
    result = parse_mermaid_view(payload.source, import_mode=payload.mode)
    return _flow_result(
        ok=result.ok,
        diagnostics=result.diagnostics,
        view=_view_public_dict(result.view),
    )


@router.post("/mermaid/render")
async def render_mermaid(
    payload: FlowMermaidRequest,
) -> dict[str, object]:
    """Render Mermaid source through a safe Flow View round trip."""
    parsed = parse_mermaid_view(payload.source, import_mode=payload.mode)
    if not parsed.ok:
        return _flow_result(ok=False, diagnostics=parsed.diagnostics)
    rendered = render_flow_view(parsed.view)
    return _flow_result(
        ok=rendered.ok,
        diagnostics=parsed.diagnostics + rendered.diagnostics,
        source=rendered.source,
    )


@router.post("/mermaid/compare")
async def compare_mermaid(
    payload: FlowMermaidCompareRequest,
    request: Request,
) -> dict[str, object]:
    """Compare inert Mermaid topology with a structured definition."""
    parsed = parse_mermaid_view(
        payload.diagram_source,
        import_mode=payload.mode,
    )
    loaded = await _loader(request).loads_validation_result(
        payload.definition_source
    )
    if parsed.ok and loaded.ok and loaded.definition is not None:
        comparison = compare_flow_topology(parsed.view, loaded.definition)
        return _flow_result(
            ok=comparison.ok,
            diagnostics=comparison.diagnostics,
        )
    diagnostics = parsed.diagnostics + loaded.diagnostics
    return _flow_result(
        ok=_diagnostics_ok(diagnostics), diagnostics=diagnostics
    )


@router.post("/run")
async def run_flow(
    payload: FlowRunRequest,
    request: Request,
) -> dict[str, object]:
    """Run a strict flow through the SDK executor."""
    loaded = await _loader(request).loads_validation_result(payload.source)
    if loaded.definition is None or not loaded.ok:
        return _flow_result(ok=False, diagnostics=loaded.diagnostics)
    result = await _executor(request).run(
        loaded.definition,
        inputs=payload.inputs,
        concurrency_limit=payload.concurrency_limit,
    )
    record = await _persist_run(request, payload.run_id, result)
    value = _run_result_public_dict(result, run_id=payload.run_id)
    if record is not None:
        value["record_revision"] = record.revision
    return value


@router.get("/runs/{run_id}/inspect")
async def inspect_run(
    run_id: str,
    request: Request,
) -> dict[str, object]:
    """Inspect a durable flow run."""
    record = await _flow_record(request, run_id)
    return {
        "ok": True,
        "inspection": _public_value(
            inspect_flow_record(record).as_public_dict()
        ),
    }


@router.get("/runs/{run_id}/trace")
async def export_trace(
    run_id: str,
    request: Request,
) -> dict[str, object]:
    """Export a sanitized durable flow trace."""
    record = await _flow_record(request, run_id)
    return {
        "ok": True,
        "trace": _public_value(
            inspect_flow_record(record).export_sanitized_trace()
        ),
    }


@router.get("/runs/{run_id}/events", response_model=None)
async def run_events(
    run_id: str,
    request: Request,
    after_sequence: int | None = Query(None, ge=0),
    stream: bool = Query(False),
) -> dict[str, object] | StreamingResponse:
    """Return sanitized task events for a flow-backed run."""
    client = _task_client(request)
    if client is None or not hasattr(client, "events"):
        raise _http_error(
            status_code=503,
            code="flow.task.events_unavailable",
            message="Flow task events are not configured.",
            hint="Configure a task client that supports event inspection.",
        )
    try:
        events = await client.events(run_id, after_sequence=after_sequence)
    except (AssertionError, TaskStoreNotFoundError) as exc:
        raise _not_found_error("flow.task.events_not_found") from exc
    if stream:
        return StreamingResponse(
            _event_stream(run_id, events),
            headers=sse_headers(),
            media_type="text/event-stream",
        )
    return {
        "ok": True,
        "events": tuple(_event_public_dict(event) for event in events),
    }


@router.post("/runs/{run_id}/cancel")
async def cancel_run(
    run_id: str,
    request: Request,
) -> dict[str, object]:
    """Request cancellation through the task client."""
    client = _task_client(request)
    if client is None or not hasattr(client, "cancel"):
        raise _http_error(
            status_code=503,
            code="flow.task.cancel_unavailable",
            message="Flow cancellation is not configured.",
            hint="Configure a task client that supports cancellation.",
        )
    try:
        run = await client.cancel(run_id)
    except TaskClientUnsupportedOperationError as exc:
        raise _http_error(
            status_code=409,
            code=exc.code,
            message="Flow run cannot be cancelled from its current state.",
            hint="Inspect the run state before cancelling.",
        ) from exc
    except (AssertionError, TaskStoreNotFoundError) as exc:
        raise _not_found_error("flow.task.run_not_found") from exc
    return {
        "ok": True,
        "run": _public_value(
            {
                "run_id": getattr(run, "run_id", run_id),
                "state": getattr(getattr(run, "state", None), "value", None),
            }
        ),
    }


@router.post("/runs/{run_id}/resume")
async def resume_run(
    run_id: str,
    payload: FlowResumeRequest,
    request: Request,
) -> dict[str, object]:
    """Resume a paused flow from durable state."""
    loaded = await _loader(request).loads_validation_result(payload.source)
    if loaded.definition is None or not loaded.ok:
        return _flow_result(ok=False, diagnostics=loaded.diagnostics)
    store = _flow_state_store(request)
    if store is None:
        raise _store_unavailable_error()
    try:
        record = await store.get_flow_execution(run_id)
        result = await _executor(request).resume(
            loaded.definition,
            record,
            decisions=payload.decisions,
            inputs=payload.inputs,
            concurrency_limit=payload.concurrency_limit,
        )
        if result.result is not None:
            updated = await store.update_flow_execution(
                run_id,
                FlowExecutionUpdate(
                    trace=result.result.trace,
                    node_outputs=result.result.node_outputs,
                    selected_outputs=result.result.outputs,
                    diagnostics=result.result.diagnostics,
                    pause_tokens=result.result.pause_tokens,
                ),
                expected_revision=payload.expected_revision or record.revision,
            )
        else:
            updated = record
    except TaskStoreConflictError as exc:
        raise _http_error(
            status_code=409,
            code="flow.task.revision_conflict",
            message="Flow run state changed before resume completed.",
            hint="Inspect the latest run state and retry with that revision.",
        ) from exc
    except (AssertionError, TaskStoreNotFoundError) as exc:
        raise _not_found_error("flow.task.run_not_found") from exc
    value = _run_result_public_dict(result, run_id=run_id)
    value["record_revision"] = updated.revision
    return value


def _loader(request: Request) -> FlowDefinitionLoader:
    registry = _flow_registry(request)
    return FlowDefinitionLoader(registry=registry)


def _flow_registry(request: Request) -> FlowNodeRegistry | None:
    registry = getattr(request.app.state, "flow_node_registry", None)
    if registry is None:
        return None
    assert isinstance(registry, FlowNodeRegistry)
    return registry


def _executor(request: Request) -> FlowExecutor:
    executor = getattr(request.app.state, "flow_executor", None)
    if executor is not None:
        assert isinstance(executor, FlowExecutor)
        return executor
    runner = getattr(request.app.state, "flow_node_runner", None)
    if runner is not None:
        assert callable(runner)
    return FlowExecutor(
        registry=_flow_registry(request),
        runner=runner,
    )


def _flow_state_store(request: Request) -> FlowStateStore | None:
    store = getattr(request.app.state, "flow_state_store", None)
    if store is None:
        return None
    assert hasattr(store, "get_flow_execution")
    return cast(FlowStateStore, store)


def _task_client(request: Request) -> FlowTaskProtocolClient | None:
    client = getattr(request.app.state, "flow_task_client", None)
    if client is None:
        return None
    return cast(FlowTaskProtocolClient, client)


async def _flow_record(
    request: Request,
    run_id: str,
) -> FlowExecutionRecord:
    store = _flow_state_store(request)
    if store is None:
        raise _store_unavailable_error()
    try:
        return await store.get_flow_execution(run_id)
    except (AssertionError, TaskStoreNotFoundError) as exc:
        raise _not_found_error("flow.task.run_not_found") from exc


async def _persist_run(
    request: Request,
    run_id: str | None,
    result: FlowExecutorRunResult,
) -> FlowExecutionRecord | None:
    if run_id is None:
        return None
    if result.result is None:
        return None
    store = _flow_state_store(request)
    if store is None:
        raise _store_unavailable_error()
    try:
        return await store.create_flow_execution(
            run_id,
            trace=result.result.trace,
            node_outputs=result.result.node_outputs,
            selected_outputs=result.result.outputs,
            diagnostics=result.result.diagnostics,
            pause_tokens=result.result.pause_tokens,
            metadata={
                "strict_flow": {
                    "name": result.plan.name if result.plan else None,
                    "version": result.plan.version if result.plan else None,
                    "revision": result.plan.revision if result.plan else None,
                }
            },
        )
    except TaskStoreConflictError as exc:
        raise _http_error(
            status_code=409,
            code="flow.task.run_conflict",
            message="Flow run state already exists.",
            hint="Use a unique run id or inspect the existing run.",
        ) from exc


def _run_result_public_dict(
    result: FlowExecutorRunResult,
    *,
    run_id: str | None,
) -> dict[str, object]:
    value = _flow_result(
        ok=result.ok,
        diagnostics=_execution_diagnostics(result),
        outputs=result.outputs,
    )
    if result.result is not None:
        value["inspection"] = result.inspect().as_public_dict()
        value["trace"] = result.export_sanitized_trace()
    if run_id is not None:
        value["run_id"] = run_id
    return cast(dict[str, object], _public_value(value))


def _execution_diagnostics(
    result: FlowExecutorRunResult,
) -> tuple[FlowDiagnostic, ...]:
    diagnostics = result.diagnostics
    if result.result is not None:
        diagnostics = diagnostics + result.result.diagnostics
    return diagnostics


def _flow_result(
    *,
    ok: bool,
    diagnostics: tuple[FlowDiagnostic, ...],
    **values: object,
) -> dict[str, object]:
    result: dict[str, object] = {
        "ok": ok,
        "diagnostics": tuple(
            diagnostic.as_public_dict() for diagnostic in diagnostics
        ),
    }
    for key, value in values.items():
        if value is not None:
            result[key] = value
    return cast(dict[str, object], _public_value(result))


def _request_validation_response(
    exc: RequestValidationError,
) -> JSONResponse:
    diagnostics = tuple(
        _request_validation_diagnostic(_request_validation_path(error))
        for error in exc.errors()
    )
    if not diagnostics:
        diagnostics = (_request_validation_diagnostic("request"),)
    return JSONResponse(
        status_code=422,
        content=_flow_result(ok=False, diagnostics=diagnostics),
    )


def _request_validation_diagnostic(path: str) -> FlowDiagnostic:
    return FlowDiagnostic(
        code="flow.definition.request_invalid",
        path=path,
        category=FlowDiagnosticCategory.FLOW_DEFINITION_VALIDATION,
        severity=FlowDiagnosticSeverity.ERROR,
        message="Flow request payload is invalid.",
        hint="Check request field types and numeric bounds.",
    )


def _request_validation_path(error: Mapping[str, object]) -> str:
    loc = error.get("loc")
    if not isinstance(loc, list | tuple):
        return "request"
    parts = tuple(str(part) for part in loc if isinstance(part, str | int))
    return ".".join(parts) or "request"


def _diagnostics_ok(
    diagnostics: tuple[FlowDiagnostic, ...],
) -> bool:
    return not any(
        diagnostic.severity == FlowDiagnosticSeverity.ERROR
        for diagnostic in diagnostics
    )


def _definition_public_dict(definition: FlowDefinition) -> dict[str, object]:
    return {
        "name": definition.name,
        "version": definition.version,
        "revision": definition.revision,
        "inputs": tuple(
            {"name": item.name, "type": item.type.value}
            for item in definition.inputs
        ),
        "outputs": tuple(
            {"name": item.name, "type": item.type.value}
            for item in definition.outputs
        ),
        "nodes": tuple(
            {"name": item.name, "type": item.type} for item in definition.nodes
        ),
    }


def _view_public_dict(view: FlowView) -> dict[str, object]:
    return {
        "import_mode": view.import_mode.value,
        "direction": (
            view.direction.value if view.direction is not None else None
        ),
        "nodes": tuple(
            {
                "id": node.id,
                "label": node.label,
                "shape": node.shape.value,
                "implicit": node.implicit,
            }
            for node in view.nodes
        ),
        "edges": tuple(
            {
                "id": edge.id,
                "source": edge.source,
                "target": edge.target,
                "label": edge.label,
                "style": edge.style.value,
                "bidirectional": edge.bidirectional,
            }
            for edge in view.edges
        ),
    }


def _event_public_dict(event: SanitizedTaskEvent) -> dict[str, object]:
    return cast(
        dict[str, object],
        _public_value(
            {
                "event_id": event.event_id,
                "run_id": event.run_id,
                "sequence": event.sequence,
                "event_type": event.event_type,
                "category": event.category.value,
                "created_at": event.created_at.isoformat(),
                "payload": event.payload,
                "attempt_id": event.attempt_id,
            }
        ),
    )


async def _event_stream(
    run_id: str,
    events: tuple[SanitizedTaskEvent, ...],
) -> AsyncIterator[bytes]:
    for event in events:
        yield sse_bytes(
            dumps(
                _event_public_dict(event),
                sort_keys=True,
                separators=(",", ":"),
            ),
            event="flow.event",
        )
    yield sse_bytes(
        dumps(
            {"run_id": run_id, "event_count": len(events)},
            sort_keys=True,
            separators=(",", ":"),
        ),
        event="flow.events.completed",
    )


def _public_value(value: object) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _public_value(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list | tuple):
        return [_public_value(item) for item in value]
    return value


def _store_unavailable_error() -> HTTPException:
    return _http_error(
        status_code=503,
        code="flow.task.store_unavailable",
        message="Flow state store is not configured.",
        hint="Configure a flow state store for durable flow operations.",
    )


def _not_found_error(code: str) -> HTTPException:
    return _http_error(
        status_code=404,
        code=code,
        message="Flow run was not found.",
        hint="Use an existing durable flow run id.",
    )


def _http_error(
    *,
    status_code: int,
    code: str,
    message: str,
    hint: str,
) -> HTTPException:
    diagnostic = FlowDiagnostic(
        code=code,
        path="flow",
        category=FlowDiagnosticCategory.TASK_DURABILITY,
        severity=FlowDiagnosticSeverity.ERROR,
        message=message,
        hint=hint,
    )
    return HTTPException(
        status_code=status_code,
        detail=_flow_result(ok=False, diagnostics=(diagnostic,)),
    )
