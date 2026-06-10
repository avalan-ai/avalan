from ...event import Event, EventType
from ...flow.definition import (
    FlowDefinition,
    FlowInputDefinition,
    FlowInputMapping,
    FlowInputType,
    FlowMappingKind,
    FlowNodeCapability,
    FlowNodeContract,
    FlowNodeDefinition,
    FlowNodeKind,
    FlowNodeMetadata,
    FlowOutputDefinition,
    FlowOutputType,
)
from ...flow.diagnostics import FlowDiagnostic, FlowDiagnosticCategory
from ...flow.flow import Flow
from ...flow.node import Node
from ...flow.plan import (
    FlowConditionPlan,
    FlowEdgePlan,
    FlowExecutionPlan,
    FlowJoinPlan,
    FlowLoopPlan,
    FlowMappingPlan,
    FlowNodePlan,
    FlowRetryPlan,
    FlowTimeoutPlan,
    compile_flow_definition,
)
from ...flow.registry import (
    FlowNodeConfigurationError,
    FlowNodeFactory,
    FlowNodeRegistry,
    FlowToolResolver,
    default_flow_node_registry,
    tool_flow_node_registry,
)
from ...flow.runtime import (
    execute_flow_plan,
    flow_node_registry_runner,
)
from ...flow.selector import (
    FlowSelector,
    FlowSelectorError,
    FlowSelectorRoot,
    parse_flow_selector,
)
from ...flow.state import (
    FlowEdgeState,
    FlowExecutionTrace,
    FlowNodeState,
    FlowNodeTrace,
)
from ...flow.store import (
    FlowExecutionRecord,
    FlowExecutionUpdate,
    FlowNodeAttemptRecord,
    FlowStateStore,
)
from ...flow.validator import validate_flow_definition
from ..artifact import TaskArtifactRef, TaskArtifactRetention
from ..context import TaskInputFile, TaskTargetContext
from ..converters import (
    FileConverter,
    TaskFileConversionDependencyError,
    TaskFileConversionError,
    TaskFileConversionPageCollection,
    TaskFileConversionResult,
    TaskFileConverterCapability,
    convert_task_artifact_pages,
)
from ..definition import (
    RunMode,
    TaskDefinition,
    TaskInputType,
    TaskOutputType,
    TaskTargetType,
)
from ..feature_gate import TaskFeature, feature_available, feature_diagnostic
from ..input import TaskFileConversionRequest, TaskFileDescriptor
from ..privacy import (
    DROPPED_MARKER,
    ENCRYPTED_MARKER,
    HASHED_MARKER,
    REDACTED_MARKER,
    STORED_ENVELOPE_MARKER,
    STORED_MARKER,
)
from ..store import TaskExecutionContext, TaskStoreNotFoundError
from ..target import TaskTargetRunner, TaskValidationContext
from ..usage import tag_usage_response
from ..validation import (
    TaskValidationCategory,
    TaskValidationError,
    TaskValidationIssue,
    validate_task_input,
    validate_task_output,
)
from .flow_constants import FLOW_RESUME_DECISIONS_METADATA_KEY

from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from inspect import isawaitable
from pathlib import Path, PurePosixPath, PureWindowsPath
from time import perf_counter
from types import MappingProxyType
from typing import Any, cast

FlowResolver = Callable[[TaskTargetContext], Flow | Awaitable[Flow]]
StrictFlowResolver = Callable[
    [TaskTargetContext],
    FlowDefinition
    | FlowExecutionPlan
    | Awaitable[FlowDefinition | FlowExecutionPlan],
]
FLOW_TASK_INPUT_KEY = "__task_input__"
FLOW_TASK_FILES_KEY = "__task_files__"
_FLOW_RESERVED_BINDINGS = frozenset(
    {
        FLOW_TASK_FILES_KEY,
        FLOW_TASK_INPUT_KEY,
        "file",
        "files",
    }
)
_FILE_CONVERT_CONFIG_KEYS = frozenset(
    {
        "converter",
        "dpi",
        "format",
        "max_pages",
        "max_pixels_per_page",
        "max_total_pixels",
        "pages",
        "quality",
    }
)
_AGENT_CONFIG_KEYS = frozenset({"file_policy", "files_input"})
_AGENT_FILE_POLICIES = frozenset({"append", "replace"})
_UNAVAILABLE_PRIVACY_MARKERS = frozenset(
    {
        DROPPED_MARKER,
        ENCRYPTED_MARKER,
        HASHED_MARKER,
        REDACTED_MARKER,
    }
)
_NO_STRICT_RESUME = object()
_FLOW_REVIEW_AUDIT_METADATA_KEY = "human_review_audit"
_INVALID_RESUME_DIAGNOSTIC_CODES = frozenset(
    {
        "flow.execution.invalid_resume_decision",
        "flow.execution.invalid_resume_node",
        "flow.execution.invalid_resume_payload",
        "flow.execution.invalid_resume_schema",
        "flow.execution.invalid_resume_state",
        "flow.execution.unknown_resume_decision",
        "flow.execution.unknown_resume_node",
    }
)


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowCompatibility:
    issues: tuple[TaskValidationIssue, ...]

    @property
    def compatible(self) -> bool:
        return not self.issues


class FlowTaskTargetRunner(TaskTargetRunner):
    def __init__(
        self,
        *,
        ref_base: str | Path | None = None,
        flow_resolver: FlowResolver | None = None,
        strict_resolver: StrictFlowResolver | None = None,
        flow_state_store: FlowStateStore | None = None,
        agent_runner: TaskTargetRunner | None = None,
        execution_roots: Iterable[str | Path] = (),
        tool_resolver: FlowToolResolver | None = None,
        concurrency_limit: int = 1,
    ) -> None:
        self._ref_base = Path(ref_base) if ref_base is not None else None
        self._flow_resolver = flow_resolver
        self._strict_resolver = strict_resolver
        self._flow_state_store = flow_state_store
        self._agent_runner = agent_runner
        self._execution_roots = tuple(Path(root) for root in execution_roots)
        self._tool_resolver = tool_resolver
        assert isinstance(concurrency_limit, int)
        assert not isinstance(concurrency_limit, bool)
        assert concurrency_limit > 0
        self._concurrency_limit = concurrency_limit

    async def validate_definition(
        self,
        definition: TaskDefinition,
        context: TaskValidationContext,
    ) -> tuple[TaskValidationIssue, ...]:
        assert isinstance(definition, TaskDefinition)
        assert isinstance(context, TaskValidationContext)
        issues = list(
            validate_flow_task_compatibility(
                definition,
                context,
                ref_base=self._ref_base,
            ).issues
        )
        if (
            issues
            or self._strict_resolver is None
            or definition.execution.type != TaskTargetType.FLOW
        ):
            return tuple(issues)
        target_context = self._validation_target_context(
            definition,
            context,
        )
        try:
            resolved = await self._resolve_strict_flow_for_validation(
                target_context,
            )
        except TaskValidationError as error:
            issues.extend(error.issues)
        else:
            if isinstance(resolved, FlowDefinition):
                result = validate_flow_definition(
                    resolved,
                    task_flow_node_registry(
                        target_context,
                        agent_runner=self._agent_runner,
                        execution_roots=self._execution_roots,
                        tool_resolver=self._tool_resolver,
                    ),
                )
                if result.diagnostics:
                    issues.extend(
                        _flow_diagnostics_to_issues(result.diagnostics)
                    )
        return tuple(issues)

    def _validation_target_context(
        self,
        definition: TaskDefinition,
        context: TaskValidationContext,
    ) -> TaskTargetContext:
        assert isinstance(definition, TaskDefinition)
        assert isinstance(context, TaskValidationContext)
        return TaskTargetContext(
            definition=definition,
            execution=TaskExecutionContext(
                run_id="validation-run",
                attempt_id="validation-attempt",
                attempt_number=1,
            ),
            artifact_store=context.artifact_store,
            task_store=context.task_store,
            file_converters=context.file_converters,
        )

    async def _resolve_strict_flow_for_validation(
        self,
        context: TaskTargetContext,
    ) -> FlowDefinition | FlowExecutionPlan:
        assert isinstance(context, TaskTargetContext)
        assert self._strict_resolver is not None
        resolved = self._strict_resolver(context)
        if isawaitable(resolved):
            resolved = await resolved
        if not isinstance(resolved, FlowDefinition | FlowExecutionPlan):
            raise TaskValidationError(
                (
                    _unsupported_flow_issue(
                        path="execution.ref",
                        message=(
                            "Flow resolver did not return a flow "
                            "definition or execution plan."
                        ),
                        hint=(
                            "Return a validated flow definition or compiled "
                            "execution plan."
                        ),
                    ),
                )
            )
        return resolved

    async def run(self, context: TaskTargetContext) -> object:
        assert isinstance(context, TaskTargetContext)
        if context.definition.execution.type != TaskTargetType.FLOW:
            raise TaskValidationError((_unknown_target_issue(),))
        if self._strict_resolver is not None:
            return await self._run_strict(context)
        if self._flow_resolver is None:
            raise TaskValidationError(
                (
                    _unsupported_flow_issue(
                        path="execution.ref",
                        message="Flow task target cannot resolve the flow.",
                        hint="Configure a flow resolver for execution.",
                    ),
                )
            )
        await context.check_cancelled()
        resolved = self._flow_resolver(context)
        if isawaitable(resolved):
            resolved = await resolved
        if not isinstance(resolved, Flow):
            raise TaskValidationError(
                (
                    _unsupported_flow_issue(
                        path="execution.ref",
                        message="Flow resolver did not return a flow.",
                        hint="Return an avalan Flow instance.",
                    ),
                )
            )
        await context.check_cancelled()
        task_input = _task_input_value(context)
        input_issues = validate_task_input(context.definition, task_input)
        if input_issues:
            raise TaskValidationError(input_issues)
        start_node = _single_start_node_name(resolved)
        if start_node is None:
            raise TaskValidationError(
                (
                    _unsupported_flow_issue(
                        path="execution.ref",
                        message=(
                            "Flow task target requires exactly one start node."
                        ),
                        hint="Use a compatible flow with one entry point.",
                    ),
                )
            )
        started = perf_counter()
        await _emit_flow_event(
            context,
            EventType.FLOW_MANAGER_CALL_BEFORE,
            status="started",
            started=started,
        )
        try:
            result = await resolved.execute_async(
                initial_node=start_node,
                initial_inputs=flow_task_input_binding(
                    task_input,
                    files=context.files,
                ),
                cancellation_checker=context.check_cancelled,
            )
            output_issues = validate_task_output(context.definition, result)
            if output_issues:
                raise TaskValidationError(output_issues)
        except BaseException:
            finished = perf_counter()
            await _emit_flow_event(
                context,
                EventType.FLOW_MANAGER_CALL_AFTER,
                status="failed",
                started=started,
                finished=finished,
            )
            raise
        finished = perf_counter()
        await _emit_flow_event(
            context,
            EventType.FLOW_MANAGER_CALL_AFTER,
            status="succeeded",
            started=started,
            finished=finished,
        )
        return result

    async def _run_strict(self, context: TaskTargetContext) -> object:
        await context.check_cancelled()
        plan = await self._strict_plan(context)
        await context.check_cancelled()
        if self._flow_state_store is None and _strict_plan_has_human_review(
            plan
        ):
            raise TaskValidationError(
                (_unsupported_human_review_state_issue(),)
            )
        task_input = _task_input_value(context)
        input_issues = validate_task_input(context.definition, task_input)
        if input_issues:
            raise TaskValidationError(input_issues)
        started = perf_counter()
        await _emit_flow_event(
            context,
            EventType.FLOW_MANAGER_CALL_BEFORE,
            status="started",
            started=started,
        )
        try:
            record = await self._strict_flow_record(context, plan)
            resume_decisions = _strict_resume_decisions(context)
            resumed_output = _strict_resumed_output(plan, record)
            if resumed_output is _NO_STRICT_RESUME:
                resume_node_outputs = _strict_resume_node_outputs(
                    plan,
                    record,
                )
                await self._ensure_strict_flow_state_started(
                    context,
                    plan=plan,
                    record=record,
                )
                result = await execute_flow_plan(
                    plan,
                    flow_node_registry_runner(
                        task_flow_node_registry(
                            context,
                            agent_runner=self._agent_runner,
                            execution_roots=self._execution_roots,
                            tool_resolver=self._tool_resolver,
                        )
                    ),
                    inputs=_strict_flow_input_binding(
                        plan,
                        task_input,
                        files=context.files,
                    ),
                    cancellation_checker=context.check_cancelled,
                    event_listener=context.event_listener,
                    concurrency_limit=self._concurrency_limit,
                    resume_trace=(
                        record.trace
                        if record
                        and (
                            resume_node_outputs is not None
                            or resume_decisions is not None
                        )
                        else None
                    ),
                    resume_node_outputs=resume_node_outputs,
                    resume_decisions=resume_decisions,
                )
                if resume_decisions and _has_invalid_resume_diagnostics(
                    result.diagnostics
                ):
                    raise TaskValidationError(
                        _flow_diagnostics_to_issues(result.diagnostics)
                    )
                await self._record_strict_flow_state(
                    context,
                    plan=plan,
                    trace=result.trace,
                    outputs=result.outputs,
                    node_outputs=result.node_outputs,
                    diagnostics=result.diagnostics,
                    pause_tokens=result.pause_tokens,
                    resume_decisions=resume_decisions,
                )
                if result.pause_tokens:
                    raise TaskValidationError(
                        (_flow_paused_issue(result.pause_tokens),)
                    )
                if not result.ok:
                    raise TaskValidationError(
                        _flow_diagnostics_to_issues(result.diagnostics)
                    )
                output = _strict_task_output(plan, result.outputs)
            else:
                output = resumed_output
            output_issues = validate_task_output(context.definition, output)
            if output_issues:
                raise TaskValidationError(output_issues)
        except BaseException:
            finished = perf_counter()
            await _emit_flow_event(
                context,
                EventType.FLOW_MANAGER_CALL_AFTER,
                status="failed",
                started=started,
                finished=finished,
            )
            raise
        finished = perf_counter()
        await _emit_flow_event(
            context,
            EventType.FLOW_MANAGER_CALL_AFTER,
            status="succeeded",
            started=started,
            finished=finished,
        )
        return output

    async def _strict_plan(
        self,
        context: TaskTargetContext,
    ) -> FlowExecutionPlan:
        assert self._strict_resolver is not None
        resolved = self._strict_resolver(context)
        if isawaitable(resolved):
            resolved = await resolved
        if isinstance(resolved, FlowExecutionPlan):
            return resolved
        if not isinstance(resolved, FlowDefinition):
            raise TaskValidationError(
                (
                    _unsupported_flow_issue(
                        path="execution.ref",
                        message=(
                            "Flow resolver did not return a flow "
                            "definition or execution plan."
                        ),
                        hint=(
                            "Return a validated flow definition or compiled "
                            "execution plan."
                        ),
                    ),
                )
            )
        registry = task_flow_node_registry(
            context,
            agent_runner=self._agent_runner,
            execution_roots=self._execution_roots,
            tool_resolver=self._tool_resolver,
        )
        result = await compile_flow_definition(resolved, registry)
        if not result.ok:
            raise TaskValidationError(
                _flow_diagnostics_to_issues(result.diagnostics)
            )
        assert result.plan is not None
        return result.plan

    async def _strict_flow_record(
        self,
        context: TaskTargetContext,
        plan: FlowExecutionPlan,
    ) -> FlowExecutionRecord | None:
        if self._flow_state_store is None:
            return None
        try:
            record = await self._flow_state_store.get_flow_execution(
                context.execution.run_id
            )
        except TaskStoreNotFoundError:
            return None
        if _strict_flow_record_mismatches_plan(plan, record):
            raise TaskValidationError((_flow_state_mismatch_issue(),))
        return record

    async def _ensure_strict_flow_state_started(
        self,
        context: TaskTargetContext,
        *,
        plan: FlowExecutionPlan,
        record: FlowExecutionRecord | None,
    ) -> None:
        if self._flow_state_store is None or record is not None:
            return
        await self._flow_state_store.create_flow_execution(
            context.execution.run_id,
            trace=FlowExecutionTrace.from_plan(plan),
            metadata=_strict_flow_record_metadata(plan),
        )

    async def _record_strict_flow_state(
        self,
        context: TaskTargetContext,
        *,
        plan: FlowExecutionPlan,
        trace: FlowExecutionTrace,
        outputs: Mapping[str, object],
        node_outputs: Mapping[str, Mapping[str, object]],
        diagnostics: tuple[FlowDiagnostic, ...] = (),
        pause_tokens: Mapping[str, str] | None = None,
        resume_decisions: Mapping[str, Mapping[str, object]] | None = None,
    ) -> None:
        if self._flow_state_store is None:
            return
        try:
            current = await self._flow_state_store.get_flow_execution(
                context.execution.run_id
            )
        except TaskStoreNotFoundError:
            current = None
        metadata = _strict_flow_record_metadata(
            plan,
            record=current,
            trace=trace,
            pause_tokens=pause_tokens,
            resume_decisions=resume_decisions,
        )
        update = FlowExecutionUpdate(
            trace=trace,
            node_attempts=_node_attempt_records(trace),
            node_outputs=_flow_node_outputs_snapshot(trace, node_outputs),
            selected_outputs=_flow_snapshot_mapping(outputs),
            loop_counters=_loop_counters(plan, trace),
            pause_tokens=pause_tokens,
            diagnostics=diagnostics,
            artifact_refs=_artifact_refs(outputs),
            metadata=metadata,
        )
        if current is None:
            await self._flow_state_store.create_flow_execution(
                context.execution.run_id,
                trace=trace,
                node_attempts=update.node_attempts or (),
                node_outputs=update.node_outputs,
                selected_outputs=update.selected_outputs,
                loop_counters=update.loop_counters,
                pause_tokens=update.pause_tokens,
                diagnostics=diagnostics,
                artifact_refs=update.artifact_refs or (),
                metadata=metadata,
            )
            return
        await self._flow_state_store.update_flow_execution(
            context.execution.run_id,
            update,
            expected_revision=current.revision,
        )


def validate_flow_task_compatibility(
    definition: TaskDefinition,
    context: TaskValidationContext,
    *,
    ref_base: str | Path | None = None,
) -> FlowCompatibility:
    assert isinstance(definition, TaskDefinition)
    assert isinstance(context, TaskValidationContext)
    issues: list[TaskValidationIssue] = []
    if definition.execution.type != TaskTargetType.FLOW:
        return FlowCompatibility(issues=(_unknown_target_issue(),))

    path_issue = _validate_flow_reference(
        definition.execution.ref,
        context=context,
        ref_base=ref_base,
    )
    if path_issue is not None:
        issues.append(path_issue)
    issues.extend(_validate_flow_contracts(definition))
    return FlowCompatibility(issues=tuple(issues))


def task_flow_node_registry(
    context: TaskTargetContext,
    *,
    agent_runner: TaskTargetRunner | None = None,
    execution_roots: Iterable[str | Path] = (),
    tool_resolver: FlowToolResolver | None = None,
) -> FlowNodeRegistry:
    assert isinstance(context, TaskTargetContext)
    registry = default_flow_node_registry()
    registry.register(
        "file_convert",
        _file_convert_node_factory(context),
        metadata=FlowNodeMetadata(
            kind=FlowNodeKind.FILE_CONVERSION,
            async_only=True,
            input_contract=FlowNodeContract(
                name="files",
                type=FlowInputType.FILE_ARRAY,
            ),
            output_contract=FlowNodeContract(
                name="files",
                type=FlowOutputType.FILE_ARRAY,
            ),
            capabilities=(
                FlowNodeCapability.ASYNC_ONLY,
                FlowNodeCapability.TASK_BACKED,
            ),
        ),
        validator=_file_convert_definition_validator(context),
    )
    registry.register(
        "pdf_to_images",
        _file_convert_node_factory(context, default_converter="pdf_image"),
        metadata=FlowNodeMetadata(
            kind=FlowNodeKind.FILE_CONVERSION,
            async_only=True,
            input_contract=FlowNodeContract(
                name="files",
                type=FlowInputType.FILE_ARRAY,
            ),
            output_contract=FlowNodeContract(
                name="files",
                type=FlowOutputType.FILE_ARRAY,
            ),
            capabilities=(
                FlowNodeCapability.ASYNC_ONLY,
                FlowNodeCapability.TASK_BACKED,
            ),
        ),
        validator=_file_convert_definition_validator(
            context,
            default_converter="pdf_image",
        ),
    )
    if agent_runner is not None:
        registry.register(
            "agent",
            _agent_node_factory(
                context,
                agent_runner=agent_runner,
                execution_roots=execution_roots,
            ),
            metadata=FlowNodeMetadata(
                kind=FlowNodeKind.AGENT,
                supports_ref=True,
                async_only=True,
                capabilities=(
                    FlowNodeCapability.ASYNC_ONLY,
                    FlowNodeCapability.TASK_BACKED,
                ),
                requires_ref=True,
                input_contracts=(
                    FlowNodeContract(
                        name="input",
                        type="any",
                    ),
                    FlowNodeContract(
                        name=None,
                        type="object",
                        metadata={"dynamic": True},
                    ),
                ),
                output_contract=FlowNodeContract(
                    name="result",
                    type=FlowOutputType.JSON,
                    metadata={"dynamic": True},
                ),
            ),
            validator=_agent_definition_validator(
                context,
                execution_roots=execution_roots,
            ),
        )
    if tool_resolver is not None:
        registry = tool_flow_node_registry(
            tool_resolver,
            base_registry=registry,
        )
    return registry


@dataclass(frozen=True, slots=True, kw_only=True)
class _FlowConversionLimits:
    max_pages: int | None = None
    max_pixels_per_page: int | None = None
    max_total_pixels: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class _AgentFilePlan:
    files_input: str | None
    file_policy: str | None


class _FlowFileConverter:
    def __init__(
        self,
        converter: FileConverter,
        *,
        limits: _FlowConversionLimits,
    ) -> None:
        assert isinstance(limits, _FlowConversionLimits)
        self._converter = converter
        self._limits = limits

    @property
    def name(self) -> str:
        return self._converter.name

    @property
    def version(self) -> str:
        return self._converter.version

    @property
    def capability(self) -> TaskFileConverterCapability:
        capability = self._converter.capability
        return TaskFileConverterCapability(
            source_mime_types=capability.source_mime_types,
            output_mime_types=capability.output_mime_types,
            supports_streaming=capability.supports_streaming,
            max_input_bytes=capability.max_input_bytes,
            max_output_bytes=capability.max_output_bytes,
            max_pages=_lower_limit(
                capability.max_pages,
                self._limits.max_pages,
            ),
            min_dpi=capability.min_dpi,
            max_dpi=capability.max_dpi,
            min_quality=capability.min_quality,
            max_quality=capability.max_quality,
            max_pixels=_lower_limit(
                capability.max_pixels,
                self._limits.max_pixels_per_page,
            ),
            estimated_memory_bytes=capability.estimated_memory_bytes,
            timeout_seconds=capability.timeout_seconds,
            options_schema=capability.options_schema,
            dependency_gates=capability.dependency_gates,
        )

    def validate_options(self, options: Mapping[str, object]) -> None:
        validate_options = getattr(self._converter, "validate_options", None)
        if callable(validate_options):
            validate_options(options)

    async def convert(
        self,
        content: bytes,
        *,
        source_media_type: str | None = None,
        options: Mapping[str, object] | None = None,
    ) -> TaskFileConversionResult:
        return await self._converter.convert(
            content,
            source_media_type=source_media_type,
            options=options,
        )

    async def convert_pages(
        self,
        content: bytes,
        *,
        source_media_type: str | None = None,
        options: Mapping[str, object] | None = None,
    ) -> TaskFileConversionPageCollection:
        convert_pages = getattr(self._converter, "convert_pages", None)
        if not callable(convert_pages):
            raise TaskFileConversionError(
                "file converter does not produce page artifacts"
            )
        collection = await convert_pages(
            content,
            source_media_type=source_media_type,
            options=options,
        )
        assert isinstance(collection, TaskFileConversionPageCollection)
        if (
            self._limits.max_pages is not None
            and len(collection.pages) > self._limits.max_pages
        ):
            raise TaskFileConversionError(
                "file conversion exceeded the page limit"
            )
        if self._limits.max_total_pixels is not None:
            total_pixels = sum(
                page.width_pixels * page.height_pixels
                for page in collection.pages
            )
            if total_pixels > self._limits.max_total_pixels:
                raise TaskFileConversionError(
                    "file conversion exceeded the pixel limit"
                )
        return collection


def _file_convert_definition_validator(
    context: TaskTargetContext,
    *,
    default_converter: str | None = None,
) -> Callable[
    [FlowDefinition, FlowNodeDefinition],
    tuple[FlowNodeConfigurationError, ...],
]:
    def validate(
        definition: FlowDefinition,
        node: FlowNodeDefinition,
    ) -> tuple[FlowNodeConfigurationError, ...]:
        errors: list[FlowNodeConfigurationError] = []
        try:
            request = _file_conversion_request(
                node,
                default_converter=default_converter,
            )
            limits = _file_conversion_limits(node.config)
        except ValueError:
            return (
                _flow_node_config_error(
                    node.name,
                    message="Flow file conversion node is invalid.",
                    hint="Use supported file conversion configuration.",
                ),
            )
        converter = context.file_converters.get(request.name)
        if converter is None:
            errors.append(
                FlowNodeConfigurationError(
                    code="flow.converter_unsupported",
                    path=f"nodes.{node.name}.config.converter",
                    message="Flow file conversion is not supported.",
                    hint="Use a registered task file converter.",
                )
            )
            return tuple(errors)
        adapted = _FlowFileConverter(converter, limits=limits)
        try:
            _validate_file_conversion_preflight(adapted, request)
        except TaskFileConversionDependencyError as error:
            diagnostic = feature_diagnostic(
                error.feature,
                path=f"nodes.{node.name}.config.converter",
            )
            errors.append(
                FlowNodeConfigurationError(
                    code=diagnostic.code,
                    path=diagnostic.path,
                    message=diagnostic.message,
                    hint=diagnostic.hint,
                )
            )
        except TaskFileConversionError:
            errors.append(
                _flow_node_config_error(
                    node.name,
                    message="Flow file conversion options are invalid.",
                    hint="Use supported file conversion options.",
                )
            )
        if definition.is_strict and context.artifact_store is None:
            errors.append(
                FlowNodeConfigurationError(
                    code="flow.missing_artifact_store",
                    path=f"nodes.{node.name}.config",
                    message="Flow file conversion requires artifact storage.",
                    hint="Configure artifact storage for task-backed flows.",
                )
            )
        if definition.is_strict and context.task_store is None:
            errors.append(
                FlowNodeConfigurationError(
                    code="flow.missing_task_store",
                    path=f"nodes.{node.name}.config",
                    message="Flow file conversion requires a task store.",
                    hint="Run file conversion inside task execution.",
                )
            )
        errors.extend(
            _validate_file_conversion_mime(definition, node, adapted)
        )
        return tuple(errors)

    return validate


def _agent_definition_validator(
    context: TaskTargetContext,
    *,
    execution_roots: Iterable[str | Path],
) -> Callable[
    [FlowDefinition, FlowNodeDefinition],
    tuple[FlowNodeConfigurationError, ...],
]:
    roots = tuple(Path(root) for root in execution_roots)

    def validate(
        definition: FlowDefinition,
        node: FlowNodeDefinition,
    ) -> tuple[FlowNodeConfigurationError, ...]:
        errors: list[FlowNodeConfigurationError] = []
        try:
            _agent_file_plan(node)
        except FlowNodeConfigurationError as error:
            errors.append(error)
        ref_issue = _validate_flow_reference(
            node.ref,
            context=TaskValidationContext(execution_roots=roots),
            ref_base=None,
        )
        if ref_issue is not None:
            errors.append(
                FlowNodeConfigurationError(
                    code=ref_issue.code,
                    path=f"nodes.{node.name}.ref",
                    message="Flow agent node ref is invalid.",
                    hint="Reference an agent under an allowed execution root.",
                )
            )
        if definition.is_strict and context.task_store is None:
            errors.append(
                FlowNodeConfigurationError(
                    code="flow.missing_task_store",
                    path=f"nodes.{node.name}.config",
                    message="Flow agent node requires a task store.",
                    hint="Run agent nodes inside task execution.",
                )
            )
        return tuple(errors)

    return validate


def _validate_file_conversion_mime(
    definition: FlowDefinition,
    node: FlowNodeDefinition,
    converter: _FlowFileConverter,
) -> tuple[FlowNodeConfigurationError, ...]:
    source_mime_types = frozenset(converter.capability.source_mime_types)
    if not source_mime_types:
        return ()
    errors: list[FlowNodeConfigurationError] = []
    input_mime_types = {
        input_definition.name: frozenset(input_definition.mime_types)
        for input_definition in definition.inputs
        if input_definition.type
        in {FlowInputType.FILE, FlowInputType.FILE_ARRAY}
    }
    for mapping in node.mappings:
        if mapping.target != "files":
            continue
        declared_mime_types = _file_mapping_input_mime_types(
            mapping,
            input_mime_types,
        )
        if not declared_mime_types:
            continue
        if declared_mime_types.isdisjoint(source_mime_types):
            errors.append(
                FlowNodeConfigurationError(
                    code="flow.incompatible_file_mime",
                    path=f"nodes.{node.name}.mapping.{mapping.target}",
                    message="Flow file conversion input MIME is incompatible.",
                    hint=(
                        "Use a converter that accepts the declared file input."
                    ),
                )
            )
    return tuple(errors)


def _file_mapping_input_mime_types(
    mapping: FlowInputMapping,
    input_mime_types: Mapping[str, frozenset[str]],
) -> frozenset[str]:
    sources: tuple[str, ...]
    if mapping.kind in {FlowMappingKind.FILE, FlowMappingKind.FILE_ARRAY}:
        sources = (mapping.source,) if mapping.source is not None else ()
    elif mapping.kind == FlowMappingKind.ARRAY:
        sources = mapping.items
    else:
        sources = ()
    found: set[str] = set()
    for source in sources:
        try:
            selector = parse_flow_selector(
                source,
                allowed_roots=frozenset({FlowSelectorRoot.FLOW_INPUT}),
            )
        except FlowSelectorError:
            continue
        found.update(input_mime_types.get(selector.source, ()))
    return frozenset(found)


def _flow_node_config_error(
    node_name: str,
    *,
    message: str,
    hint: str,
) -> FlowNodeConfigurationError:
    return FlowNodeConfigurationError(
        code="flow.invalid_node",
        path=f"nodes.{node_name}.config",
        message=message,
        hint=hint,
    )


def _file_convert_node_factory(
    context: TaskTargetContext,
    *,
    default_converter: str | None = None,
) -> FlowNodeFactory:
    def build(definition: FlowNodeDefinition) -> Node:
        request = _file_conversion_request(
            definition,
            default_converter=default_converter,
        )
        limits = _file_conversion_limits(definition.config)
        converter = context.file_converters.get(request.name)
        if converter is None:
            raise FlowNodeConfigurationError(
                code="flow.converter_unsupported",
                path=f"nodes.{definition.name}.config.converter",
                message="Flow file conversion is not supported.",
                hint="Use a registered task file converter.",
            )
        adapted = _FlowFileConverter(converter, limits=limits)
        try:
            _validate_file_conversion_preflight(adapted, request)
        except TaskFileConversionDependencyError as error:
            diagnostic = feature_diagnostic(
                error.feature,
                path=f"nodes.{definition.name}.config.converter",
            )
            raise FlowNodeConfigurationError(
                code=diagnostic.code,
                path=diagnostic.path,
                message=diagnostic.message,
                hint=diagnostic.hint,
            ) from error
        except TaskFileConversionError as error:
            raise FlowNodeConfigurationError(
                code="flow.invalid_node",
                path=f"nodes.{definition.name}.config",
                message="Flow file conversion options are invalid.",
                hint="Use supported file conversion options.",
            ) from error

        async def run(inputs: dict[str, object]) -> object:
            await context.check_cancelled()
            files = _flow_node_input_files(
                definition,
                inputs,
                context=context,
            )
            if context.artifact_store is None:
                raise TaskValidationError(
                    (
                        _unsupported_flow_issue(
                            path=f"nodes.{definition.name}.config",
                            message=(
                                "Flow file conversion requires an artifact"
                                " store."
                            ),
                            hint="Configure artifact storage before running.",
                        ),
                    )
                )
            if context.task_store is None:
                raise TaskValidationError(
                    (
                        _unsupported_flow_issue(
                            path=f"nodes.{definition.name}.config",
                            message=(
                                "Flow file conversion requires a task store."
                            ),
                            hint="Run the node inside task execution.",
                        ),
                    )
                )
            converted_files: list[object] = []
            for index, file in enumerate(files):
                await context.check_cancelled()
                if file.artifact_ref is None:
                    raise TaskValidationError(
                        (
                            _invalid_flow_file_issue(
                                definition.name,
                                index,
                                "Flow file conversion requires artifact "
                                "backed files.",
                                "Materialize the file into an artifact store "
                                "before conversion.",
                            ),
                        )
                    )
                try:
                    collection = await convert_task_artifact_pages(
                        file.artifact_ref,
                        request,
                        converter=adapted,
                        artifact_store=context.artifact_store,
                        task_store=context.task_store,
                        run_id=context.execution.run_id,
                        attempt_id=context.execution.attempt_id,
                        retention=TaskArtifactRetention(
                            delete_after_days=(
                                context.definition.artifact.retention_days
                            ),
                        ),
                    )
                except TaskFileConversionDependencyError as error:
                    raise TaskValidationError(
                        (
                            _feature_issue(
                                error.feature,
                                path=(
                                    f"nodes.{definition.name}.config.converter"
                                ),
                            ),
                        )
                    ) from error
                except TaskFileConversionError as error:
                    raise TaskValidationError(
                        (
                            _invalid_flow_file_issue(
                                definition.name,
                                index,
                                "Flow file conversion could not be completed.",
                                "Use a supported file, converter, and "
                                "conversion options.",
                            ),
                        )
                    ) from error
                converted_files.extend(
                    _converted_flow_file(page.ref) for page in collection.pages
                )
            await context.check_cancelled()
            if definition.output is None:
                return converted_files
            return {definition.output: converted_files}

        return Node(definition.name, func=run)

    return build


def _validate_file_conversion_preflight(
    converter: _FlowFileConverter,
    request: TaskFileConversionRequest,
) -> None:
    assert isinstance(converter, _FlowFileConverter)
    assert isinstance(request, TaskFileConversionRequest)
    for feature in converter.capability.dependency_gates:
        if not feature_available(feature):
            raise TaskFileConversionDependencyError(feature)
    if not callable(getattr(converter._converter, "convert_pages", None)):
        raise TaskFileConversionError(
            "file converter does not produce page artifacts"
        )
    converter.validate_options(request.options)


def _validate_flow_reference(
    ref: object,
    *,
    context: TaskValidationContext,
    ref_base: str | Path | None,
) -> TaskValidationIssue | None:
    if not isinstance(ref, str) or not ref.strip() or _is_path_escape(ref):
        return _path_escape_issue()
    roots = context.execution_roots
    if not roots:
        return None
    base = Path(ref_base) if ref_base is not None else None
    for root in roots:
        resolved_root = root.resolve(strict=False)
        candidate_base = base or resolved_root
        try:
            candidate = (candidate_base / ref).resolve(strict=False)
        except (OSError, RuntimeError, ValueError):
            continue
        if _is_relative_to(candidate, resolved_root):
            return None
    return _path_escape_issue()


def _file_conversion_request(
    definition: FlowNodeDefinition,
    *,
    default_converter: str | None,
) -> TaskFileConversionRequest:
    config = definition.config
    _validate_file_conversion_config_keys(config)
    converter = _file_conversion_name(config, default_converter)
    if definition.type == "pdf_to_images" and converter != "pdf_image":
        raise ValueError("pdf_to_images requires pdf_image conversion")
    options: dict[str, object] = {}
    for key in ("dpi", "format", "quality"):
        if key in config:
            options[key] = config[key]
    if "pages" in config:
        options["pages"] = _page_range_option(config["pages"])
    return TaskFileConversionRequest(name=converter, options=options)


def _file_conversion_limits(
    config: Mapping[str, object],
) -> _FlowConversionLimits:
    return _FlowConversionLimits(
        max_pages=_positive_config_int(config, "max_pages"),
        max_pixels_per_page=_positive_config_int(
            config,
            "max_pixels_per_page",
        ),
        max_total_pixels=_positive_config_int(config, "max_total_pixels"),
    )


def _validate_file_conversion_config_keys(
    config: Mapping[str, object],
) -> None:
    for key in config:
        if key not in _FILE_CONVERT_CONFIG_KEYS:
            raise ValueError("file conversion node option is unsupported")


def _file_conversion_name(
    config: Mapping[str, object],
    default_converter: str | None,
) -> str:
    value = config.get("converter", default_converter)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("file conversion node requires a converter")
    return value


def _page_range_option(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return MappingProxyType(dict(value))
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return MappingProxyType({"start": value, "end": value})
    if not isinstance(value, str):
        raise ValueError("file conversion page range is invalid")
    stripped = value.strip()
    if not stripped:
        raise ValueError("file conversion page range is invalid")
    if stripped.isdecimal():
        page = int(stripped)
        if page <= 0:
            raise ValueError("file conversion page range is invalid")
        return MappingProxyType({"start": page, "end": page})
    if ".." not in stripped:
        raise ValueError("file conversion page range is invalid")
    start_text, end_text = stripped.split("..", 1)
    if not start_text.isdecimal() or (end_text and not end_text.isdecimal()):
        raise ValueError("file conversion page range is invalid")
    start = int(start_text)
    if start <= 0:
        raise ValueError("file conversion page range is invalid")
    pages: dict[str, object] = {"start": start}
    if end_text:
        end = int(end_text)
        if end < start:
            raise ValueError("file conversion page range is invalid")
        pages["end"] = end
    return MappingProxyType(pages)


def _positive_config_int(
    config: Mapping[str, object],
    key: str,
) -> int | None:
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("file conversion limit is invalid")
    return value


def _flow_node_input_files(
    definition: FlowNodeDefinition,
    inputs: Mapping[str, object],
    *,
    context: TaskTargetContext,
) -> tuple[TaskInputFile, ...]:
    value = _flow_node_file_value(definition, inputs, context=context)
    values = _flow_files_from_value(value)
    if not values:
        raise TaskValidationError(
            (
                _unsupported_flow_issue(
                    path=f"nodes.{definition.name}.input",
                    message="Flow file conversion requires file input.",
                    hint="Pass one or more task input files to the node.",
                ),
            )
        )
    files: list[TaskInputFile] = []
    for index, item in enumerate(values):
        if not isinstance(item, TaskInputFile):
            raise TaskValidationError(
                (
                    _invalid_flow_file_issue(
                        definition.name,
                        index,
                        "Flow file conversion input is invalid.",
                        "Pass task input file values to the conversion node.",
                    ),
                )
            )
        files.append(item)
    return tuple(files)


def _flow_node_file_value(
    definition: FlowNodeDefinition,
    inputs: Mapping[str, object],
    *,
    context: TaskTargetContext,
) -> object:
    if definition.input is not None and definition.input in inputs:
        return inputs[definition.input]
    if definition.input is not None:
        if context.files and len(context.files) == 1:
            return context.files[0]
        raise TaskValidationError(
            (
                _unsupported_flow_issue(
                    path=f"nodes.{definition.name}.input",
                    message="Flow file input selector is unavailable.",
                    hint="Use a selector that resolves to one file value.",
                ),
            )
        )
    if FLOW_TASK_FILES_KEY in inputs:
        return inputs[FLOW_TASK_FILES_KEY]
    if "files" in inputs:
        return inputs["files"]
    if "file" in inputs:
        return inputs["file"]
    return _flow_node_input_value(definition, inputs)


def _flow_files_from_value(value: object) -> tuple[object, ...]:
    if isinstance(value, Mapping):
        if "files" in value:
            return _flow_files_from_value(value["files"])
        if "file" in value:
            return _flow_files_from_value(value["file"])
    if isinstance(value, list | tuple):
        return tuple(value)
    return (value,)


def _converted_flow_file(ref: TaskArtifactRef) -> TaskInputFile:
    return TaskInputFile(
        logical_path=f"artifact:{ref.artifact_id}",
        artifact_ref=ref,
        media_type=ref.media_type,
        size_bytes=ref.size_bytes,
        metadata=ref.metadata,
    )


def _lower_limit(first: int | None, second: int | None) -> int | None:
    if first is None:
        return second
    if second is None:
        return first
    return min(first, second)


def _validate_flow_contracts(
    definition: TaskDefinition,
) -> tuple[TaskValidationIssue, ...]:
    issues: list[TaskValidationIssue] = []
    if (
        definition.output.type in {TaskOutputType.OBJECT, TaskOutputType.ARRAY}
        and definition.output.schema is None
        and definition.output.schema_ref is None
    ):
        issues.append(
            _unsupported_flow_issue(
                path="output.schema",
                message=(
                    "Flow task targets require a structured output schema."
                ),
                hint="Declare the expected flow output schema.",
            )
        )
    return tuple(issues)


def flow_task_input_binding(
    value: object,
    *,
    files: tuple[object, ...] = (),
) -> dict[str, object]:
    assert isinstance(files, tuple)
    if isinstance(value, Mapping):
        binding = _copy_mapping(value)
        binding[FLOW_TASK_INPUT_KEY] = _copy_mapping(value)
    elif isinstance(value, list | tuple):
        binding = {
            FLOW_TASK_INPUT_KEY: _copy_sequence(value),
            "items": _copy_sequence(value),
        }
    else:
        binding = {
            FLOW_TASK_INPUT_KEY: _copy_task_input_value(value),
            "value": _copy_task_input_value(value),
        }
    if files:
        binding[FLOW_TASK_FILES_KEY] = list(files)
        binding["files"] = list(files)
        if len(files) == 1:
            binding["file"] = files[0]
    return binding


def _agent_node_factory(
    context: TaskTargetContext,
    *,
    agent_runner: TaskTargetRunner,
    execution_roots: Iterable[str | Path],
) -> FlowNodeFactory:
    roots = tuple(Path(root) for root in execution_roots)

    def build(definition: FlowNodeDefinition) -> Node:
        assert isinstance(definition, FlowNodeDefinition)
        assert definition.ref is not None
        file_plan = _agent_file_plan(definition)

        async def run(inputs: dict[str, object]) -> object:
            await context.check_cancelled()
            files = _agent_node_files(
                definition,
                inputs,
                context=context,
                file_plan=file_plan,
            )
            agent_definition = _agent_node_task_definition(
                context.definition,
                definition,
            )
            issues = await agent_runner.validate_definition(
                agent_definition,
                TaskValidationContext(execution_roots=roots),
            )
            if issues:
                raise TaskValidationError(issues)
            return await agent_runner.run(
                TaskTargetContext(
                    definition=agent_definition,
                    execution=context.execution,
                    input_value=_flow_node_input_value(definition, inputs),
                    files=files,
                    metadata=context.metadata,
                    cancellation_checker=context.cancellation_checker,
                    event_listener=_flow_node_event_listener(
                        definition.name,
                        context.event_listener,
                    ),
                    usage_observer=_flow_node_usage_observer(
                        definition.name,
                        context.usage_observer,
                    ),
                    artifact_store=context.artifact_store,
                    task_store=context.task_store,
                    file_converters=context.file_converters,
                )
            )

        return Node(definition.name, func=run)

    return build


def _flow_node_event_listener(
    node_name: str,
    listener: Callable[[Event], Awaitable[None] | None] | None,
) -> Callable[[Event], Awaitable[None] | None] | None:
    assert isinstance(node_name, str) and node_name.strip()
    if listener is None:
        return None
    assert callable(listener)

    def observe(event: Event) -> Awaitable[None] | None:
        assert isinstance(event, Event)
        payload: dict[str, Any] = {}
        if isinstance(event.payload, Mapping):
            payload.update(event.payload)
        payload["flow_node"] = node_name
        return listener(
            Event(
                type=event.type,
                payload=payload,
                started=event.started,
                finished=event.finished,
                elapsed=event.elapsed,
            )
        )

    return observe


def _flow_node_usage_observer(
    node_name: str,
    observer: Callable[[object], Awaitable[None] | None] | None,
) -> Callable[[object], Awaitable[None] | None] | None:
    assert isinstance(node_name, str) and node_name.strip()
    if observer is None:
        return None
    assert callable(observer)

    def observe(response: object) -> Awaitable[None] | None:
        return observer(tag_usage_response(response, flow_node=node_name))

    return observe


def _agent_file_plan(definition: FlowNodeDefinition) -> _AgentFilePlan:
    config = definition.config
    for key in config:
        if key not in _AGENT_CONFIG_KEYS:
            raise FlowNodeConfigurationError(
                code="flow.invalid_node",
                path=f"nodes.{definition.name}.config.{key}",
                message="Flow agent node option is unsupported.",
                hint="Use supported agent node configuration keys.",
            )
    files_input = config.get("files_input")
    file_policy = config.get("file_policy")
    if files_input is None and file_policy is None:
        return _AgentFilePlan(files_input=None, file_policy=None)
    if not isinstance(files_input, str) or not files_input.strip():
        raise FlowNodeConfigurationError(
            code="flow.invalid_node",
            path=f"nodes.{definition.name}.config.files_input",
            message="Flow agent file input selector is invalid.",
            hint="Use a dotted upstream node output selector.",
        )
    if not isinstance(file_policy, str) or file_policy not in (
        _AGENT_FILE_POLICIES
    ):
        raise FlowNodeConfigurationError(
            code="flow.invalid_node",
            path=f"nodes.{definition.name}.config.file_policy",
            message="Flow agent file policy is invalid.",
            hint="Use replace or append.",
        )
    _validate_agent_files_input_selector(definition.name, files_input)
    return _AgentFilePlan(files_input=files_input, file_policy=file_policy)


def _validate_agent_files_input_selector(
    node_name: str,
    selector: str,
) -> None:
    parts = selector.split(".")
    if len(parts) != 2 or any(not part.strip() for part in parts):
        raise FlowNodeConfigurationError(
            code="flow.invalid_node",
            path=f"nodes.{node_name}.config.files_input",
            message="Flow agent file input selector is invalid.",
            hint="Use a dotted upstream node output selector.",
        )
    if parts[0] in _FLOW_RESERVED_BINDINGS:
        raise FlowNodeConfigurationError(
            code="flow.invalid_node",
            path=f"nodes.{node_name}.config.files_input",
            message="Flow agent file input selector is reserved.",
            hint="Reference a named upstream node output instead.",
        )


def _agent_node_files(
    definition: FlowNodeDefinition,
    inputs: Mapping[str, object],
    *,
    context: TaskTargetContext,
    file_plan: _AgentFilePlan,
) -> tuple[TaskInputFile, ...]:
    if file_plan.files_input is None:
        return context.files
    selected = _flow_file_selector_value(
        definition,
        inputs,
        file_plan.files_input,
    )
    files = _flow_file_array_from_value(definition, selected)
    if file_plan.file_policy == "replace":
        return files
    if file_plan.file_policy == "append":
        return context.files + _append_policy_files(files)
    raise AssertionError("validated agent file policy is unsupported")


def _append_policy_files(
    files: tuple[TaskInputFile, ...],
) -> tuple[TaskInputFile, ...]:
    return tuple(
        replace(
            file,
            metadata={
                **file.metadata,
                "file_policy": "append",
            },
        )
        for file in files
    )


def _flow_file_selector_value(
    definition: FlowNodeDefinition,
    inputs: Mapping[str, object],
    selector: str,
) -> object:
    node_name, output_key = selector.split(".", 1)
    if node_name not in inputs:
        raise TaskValidationError(
            (
                _unsupported_flow_issue(
                    path=f"nodes.{definition.name}.config.files_input",
                    message="Flow agent file input selector is unavailable.",
                    hint=(
                        "Connect the selected upstream node to the agent node."
                    ),
                ),
            )
        )
    value = inputs[node_name]
    if not isinstance(value, Mapping) or output_key not in value:
        raise TaskValidationError(
            (
                _unsupported_flow_issue(
                    path=f"nodes.{definition.name}.config.files_input",
                    message="Flow agent file input selector has no output.",
                    hint="Select an upstream file array output key.",
                ),
            )
        )
    return value[output_key]


def _flow_file_array_from_value(
    definition: FlowNodeDefinition,
    value: object,
) -> tuple[TaskInputFile, ...]:
    if not isinstance(value, list | tuple):
        raise TaskValidationError(
            (
                _unsupported_flow_issue(
                    path=f"nodes.{definition.name}.config.files_input",
                    message="Flow agent file input is not a file array.",
                    hint="Select an upstream node output containing files.",
                ),
            )
        )
    if not value:
        raise TaskValidationError(
            (
                _unsupported_flow_issue(
                    path=f"nodes.{definition.name}.config.files_input",
                    message="Flow agent file input is empty.",
                    hint="Pass at least one generated file to the agent node.",
                ),
            )
        )
    files: list[TaskInputFile] = []
    for index, item in enumerate(value):
        if not isinstance(item, TaskInputFile):
            raise TaskValidationError(
                (
                    _invalid_flow_file_issue(
                        definition.name,
                        index,
                        "Flow agent file input item is invalid.",
                        "Pass task input file values to the agent node.",
                    ),
                )
            )
        files.append(item)
    return tuple(files)


def _agent_node_task_definition(
    parent: TaskDefinition,
    node: FlowNodeDefinition,
) -> TaskDefinition:
    assert node.ref is not None
    return replace(
        parent,
        execution=parent.execution.agent(node.ref),
    )


def _flow_node_input_value(
    definition: FlowNodeDefinition,
    inputs: Mapping[str, object],
) -> object:
    if definition.input is not None and definition.input in inputs:
        return _copy_task_input_value(inputs[definition.input])
    if FLOW_TASK_INPUT_KEY in inputs:
        return _copy_task_input_value(inputs[FLOW_TASK_INPUT_KEY])
    if len(inputs) == 1:
        return _copy_task_input_value(next(iter(inputs.values())))
    return _copy_task_input_value(cast(dict[str, object], dict(inputs)))


def _copy_mapping(value: Mapping[object, object]) -> dict[str, object]:
    copied: dict[str, object] = {}
    for key, item in value.items():
        assert isinstance(key, str), "task input keys must be strings"
        copied[key] = _copy_task_input_value(item)
    return copied


def _copy_task_input_value(value: object) -> object:
    if isinstance(value, Mapping):
        return _copy_mapping(value)
    if isinstance(value, list):
        return _copy_sequence(value)
    if isinstance(value, tuple):
        return _copy_sequence(value)
    return value


def _copy_sequence(value: list[object] | tuple[object, ...]) -> list[object]:
    return [_copy_task_input_value(item) for item in value]


def _task_input_value(context: TaskTargetContext) -> object:
    value = context.input_value
    if context.definition.run.mode != RunMode.QUEUE:
        return value
    queued_file_value = _queued_file_input_value(context)
    if queued_file_value is not None:
        return queued_file_value
    if not isinstance(value, Mapping):
        return value
    if _is_stored_privacy_envelope(value):
        return value["value"]
    if _is_legacy_stored_privacy_envelope(value):
        if _can_be_declared_object_input(context.definition, value):
            return value
        return value["value"]
    if value.get("privacy") in _UNAVAILABLE_PRIVACY_MARKERS:
        if _can_be_declared_object_input(context.definition, value):
            return value
        raise TaskValidationError(
            (
                _unsupported_flow_issue(
                    path="input",
                    message="Queued flow task input is not available.",
                    hint=(
                        "Persist a JSON-compatible input value or run the "
                        "task directly."
                    ),
                ),
            )
        )
    return value


def _queued_file_input_value(context: TaskTargetContext) -> object | None:
    input_type = context.definition.input.type
    if input_type == TaskInputType.FILE:
        if len(context.files) != 1:
            return None
        return _queued_file_descriptor(context.files[0])
    if input_type == TaskInputType.FILE_ARRAY:
        descriptors: list[TaskFileDescriptor] = []
        for file in context.files:
            descriptor = _queued_file_descriptor(file)
            if descriptor is None:
                return None
            descriptors.append(descriptor)
        return descriptors
    return None


def _queued_file_descriptor(
    file: TaskInputFile,
) -> TaskFileDescriptor | None:
    if file.artifact_ref is not None:
        ref = file.artifact_ref
        return TaskFileDescriptor.artifact(
            ref.artifact_id,
            mime_type=file.media_type or ref.media_type,
            size_bytes=(
                file.size_bytes
                if file.size_bytes is not None
                else ref.size_bytes
            ),
            sha256=ref.sha256,
        )
    if file.provider_reference is not None:
        provider_ref = file.provider_reference
        return TaskFileDescriptor.provider_reference_descriptor(
            provider_ref.reference,
            kind=provider_ref.kind,
            provider=provider_ref.provider,
            owner_scope=provider_ref.owner_scope,
            expires_at=provider_ref.expires_at,
            mime_type=file.media_type or provider_ref.mime_type,
            size_bytes=file.size_bytes,
            size_bucket=provider_ref.size_bucket,
            identity_hmac=provider_ref.identity_hmac,
            durable=provider_ref.durable,
            metadata=provider_ref.metadata,
        )
    return None


def _is_stored_privacy_envelope(value: Mapping[object, object]) -> bool:
    return (
        value.get("privacy") == STORED_MARKER
        and value.get("format") == STORED_ENVELOPE_MARKER
        and "value" in value
    )


def _is_legacy_stored_privacy_envelope(
    value: Mapping[object, object],
) -> bool:
    return (
        value.get("privacy") == STORED_MARKER
        and "format" not in value
        and "value" in value
    )


def _can_be_declared_object_input(
    definition: TaskDefinition,
    value: Mapping[object, object],
) -> bool:
    if definition.input.type != TaskInputType.OBJECT:
        return False
    issues = validate_task_input(definition, value)
    return not issues or all(
        issue.code == "dependency.jsonschema_missing" for issue in issues
    )


def _strict_flow_input_binding(
    plan: FlowExecutionPlan,
    task_input: object,
    *,
    files: tuple[TaskInputFile, ...],
) -> Mapping[str, object]:
    assert isinstance(plan, FlowExecutionPlan)
    assert isinstance(files, tuple)
    binding: dict[str, object] = {}
    for input_definition in plan.inputs:
        if input_definition.type == FlowInputType.FILE:
            if len(files) == 1:
                binding[input_definition.name] = files[0]
            else:
                binding[input_definition.name] = _mapped_task_input(
                    task_input,
                    input_definition.name,
                    single_input=len(plan.inputs) == 1,
                )
            continue
        if input_definition.type == FlowInputType.FILE_ARRAY:
            if files:
                binding[input_definition.name] = list(files)
            else:
                binding[input_definition.name] = _mapped_task_input(
                    task_input,
                    input_definition.name,
                    single_input=len(plan.inputs) == 1,
                )
            continue
        binding[input_definition.name] = _mapped_task_input(
            task_input,
            input_definition.name,
            single_input=len(plan.inputs) == 1,
        )
    return binding


def _mapped_task_input(
    value: object,
    name: str,
    *,
    single_input: bool,
) -> object:
    if isinstance(value, Mapping) and name in value:
        return _copy_task_input_value(value[name])
    if single_input:
        return _copy_task_input_value(value)
    return None


def _strict_task_output(
    plan: FlowExecutionPlan,
    outputs: Mapping[str, object],
) -> object:
    assert isinstance(plan, FlowExecutionPlan)
    assert isinstance(outputs, Mapping)
    if len(plan.outputs) == 1:
        name = plan.outputs[0].name
        if name in outputs:
            return outputs[name]
    return _copy_task_input_value(outputs)


def _flow_diagnostics_to_issues(
    diagnostics: tuple[FlowDiagnostic, ...],
) -> tuple[TaskValidationIssue, ...]:
    if not diagnostics:
        return (
            _unsupported_flow_issue(
                path="execution.ref",
                message="Flow execution failed.",
                hint="Inspect the flow diagnostics.",
            ),
        )
    issues: list[TaskValidationIssue] = []
    for diagnostic in diagnostics:
        assert isinstance(diagnostic, FlowDiagnostic)
        path = getattr(diagnostic, "path", None)
        hint = getattr(diagnostic, "hint", None)
        issues.append(
            TaskValidationIssue(
                code=diagnostic.code,
                path=path if isinstance(path, str) and path else "execution",
                message=diagnostic.message,
                hint=(
                    hint
                    if isinstance(hint, str) and hint
                    else ("Inspect the flow diagnostics.")
                ),
                category=_task_validation_category(diagnostic.category),
            )
        )
    return tuple(issues)


def _task_validation_category(
    category: FlowDiagnosticCategory,
) -> TaskValidationCategory:
    assert isinstance(category, FlowDiagnosticCategory)
    match category:
        case FlowDiagnosticCategory.PRIVACY:
            return TaskValidationCategory.PRIVACY
        case FlowDiagnosticCategory.FLOW_DEFINITION_VALIDATION:
            return TaskValidationCategory.STRUCTURE
        case FlowDiagnosticCategory.TASK_DURABILITY:
            return TaskValidationCategory.DEPENDENCY
        case _:
            return TaskValidationCategory.UNSUPPORTED


def _flow_state_mismatch_issue() -> TaskValidationIssue:
    return TaskValidationIssue(
        code="flow.execution_state_mismatch",
        path="execution.run_id",
        message="Flow execution state does not match the resolved flow.",
        hint="Use a fresh task run id or the matching flow definition.",
        category=TaskValidationCategory.DEPENDENCY,
    )


def _flow_paused_issue(
    pause_tokens: Mapping[str, str],
) -> TaskValidationIssue:
    assert isinstance(pause_tokens, Mapping)
    return TaskValidationIssue(
        code="flow.execution.paused",
        path="execution.run_id",
        message="Flow execution paused for human review.",
        hint="Resume the paused review with a decision payload.",
        category=TaskValidationCategory.DEPENDENCY,
    )


def _invalid_flow_resume_decisions_issue() -> TaskValidationIssue:
    return TaskValidationIssue(
        code="flow.execution.invalid_resume_payload",
        path=f"metadata.{FLOW_RESUME_DECISIONS_METADATA_KEY}",
        message="Flow resume decisions are invalid.",
        hint="Provide a mapping of paused review nodes to decision payloads.",
        category=TaskValidationCategory.STRUCTURE,
    )


def _unsupported_human_review_state_issue() -> TaskValidationIssue:
    return TaskValidationIssue(
        code="flow.unsupported_human_review_direct_mode",
        path="execution.ref",
        message="Human review flow execution requires durable state.",
        hint="Configure a flow state store before running review nodes.",
        category=TaskValidationCategory.DEPENDENCY,
    )


def _strict_plan_has_human_review(plan: FlowExecutionPlan) -> bool:
    assert isinstance(plan, FlowExecutionPlan)
    return any(
        node.kind == FlowNodeKind.HUMAN_REVIEW
        or _strict_subflow_has_human_review(node)
        for node in plan.nodes
    )


def _strict_subflow_has_human_review(node: FlowNodePlan) -> bool:
    assert isinstance(node, FlowNodePlan)
    metadata = node.metadata.get("subflow")
    subflow_plan = (
        metadata.get("plan") if isinstance(metadata, Mapping) else None
    )
    return isinstance(
        subflow_plan,
        FlowExecutionPlan,
    ) and _strict_plan_has_human_review(subflow_plan)


def _strict_resume_decisions(
    context: TaskTargetContext,
) -> Mapping[str, Mapping[str, object]] | None:
    value = context.metadata.get(FLOW_RESUME_DECISIONS_METADATA_KEY)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TaskValidationError((_invalid_flow_resume_decisions_issue(),))
    decisions: dict[str, Mapping[str, object]] = {}
    for node, payload in value.items():
        if (
            not isinstance(node, str)
            or not node.strip()
            or not isinstance(payload, Mapping)
        ):
            raise TaskValidationError(
                (_invalid_flow_resume_decisions_issue(),)
            )
        decisions[node] = _flow_snapshot_mapping(payload)
    return MappingProxyType(decisions)


def _has_invalid_resume_diagnostics(
    diagnostics: tuple[FlowDiagnostic, ...],
) -> bool:
    return any(
        diagnostic.code in _INVALID_RESUME_DIAGNOSTIC_CODES
        for diagnostic in diagnostics
    )


def _strict_resumed_output(
    plan: FlowExecutionPlan,
    record: FlowExecutionRecord | None,
) -> object:
    assert isinstance(plan, FlowExecutionPlan)
    if record is None:
        return _NO_STRICT_RESUME
    assert isinstance(record, FlowExecutionRecord)
    if not _strict_flow_record_is_complete(plan, record):
        return _NO_STRICT_RESUME
    return _strict_task_output(plan, record.selected_outputs)


def _strict_resume_node_outputs(
    plan: FlowExecutionPlan,
    record: FlowExecutionRecord | None,
) -> Mapping[str, Mapping[str, object]] | None:
    assert isinstance(plan, FlowExecutionPlan)
    if record is None:
        return None
    assert isinstance(record, FlowExecutionRecord)
    if record.metadata.get("strict_flow") != _strict_flow_record_signature(
        plan
    ):
        return None
    if record.diagnostics:
        return None
    node_outputs = _strict_record_node_outputs(record)
    if not node_outputs:
        return None
    succeeded = {
        node.node
        for node in record.trace.nodes
        if node.state == FlowNodeState.SUCCEEDED
    }
    if not succeeded.issubset(node_outputs.keys()):
        return None
    return node_outputs


def _strict_flow_record_is_complete(
    plan: FlowExecutionPlan,
    record: FlowExecutionRecord,
) -> bool:
    assert isinstance(plan, FlowExecutionPlan)
    assert isinstance(record, FlowExecutionRecord)
    if record.metadata.get("strict_flow") != _strict_flow_record_signature(
        plan
    ):
        return False
    if record.diagnostics:
        return False
    node_states = {node.node: node.state for node in record.trace.nodes}
    plan_nodes = {node.name for node in plan.nodes}
    if set(node_states) != plan_nodes:
        return False
    complete_states = {FlowNodeState.SKIPPED, FlowNodeState.SUCCEEDED}
    if any(state not in complete_states for state in node_states.values()):
        return False
    edge_states = {edge.index: edge.state for edge in record.trace.edges}
    plan_edges = {edge.index for edge in plan.edges}
    if set(edge_states) != plan_edges:
        return False
    if any(state == FlowEdgeState.FAILED for state in edge_states.values()):
        return False
    output_names = {output.name for output in plan.outputs}
    return output_names.issubset(record.selected_outputs.keys())


def _strict_flow_record_mismatches_plan(
    plan: FlowExecutionPlan,
    record: FlowExecutionRecord,
) -> bool:
    assert isinstance(plan, FlowExecutionPlan)
    assert isinstance(record, FlowExecutionRecord)
    signature = record.metadata.get("strict_flow")
    if signature is None:
        return False
    return signature != _strict_flow_record_signature(plan)


def _strict_flow_record_metadata(
    plan: FlowExecutionPlan,
    *,
    record: FlowExecutionRecord | None = None,
    trace: FlowExecutionTrace | None = None,
    pause_tokens: Mapping[str, str] | None = None,
    resume_decisions: Mapping[str, Mapping[str, object]] | None = None,
) -> Mapping[str, object]:
    assert isinstance(plan, FlowExecutionPlan)
    if record is not None:
        assert isinstance(record, FlowExecutionRecord)
    if trace is not None:
        assert isinstance(trace, FlowExecutionTrace)
    metadata: dict[str, object] = {
        "strict_flow": _strict_flow_record_signature(plan)
    }
    review_audit = _strict_human_review_audit(
        plan,
        record=record,
        trace=trace,
        pause_tokens=pause_tokens,
        resume_decisions=resume_decisions,
    )
    if review_audit:
        metadata[_FLOW_REVIEW_AUDIT_METADATA_KEY] = review_audit
    return metadata


def _strict_human_review_audit(
    plan: FlowExecutionPlan,
    *,
    record: FlowExecutionRecord | None,
    trace: FlowExecutionTrace | None,
    pause_tokens: Mapping[str, str] | None,
    resume_decisions: Mapping[str, Mapping[str, object]] | None,
) -> Mapping[str, object]:
    audit = _strict_human_review_audit_from_record(record)
    if trace is not None and pause_tokens:
        for node_name in pause_tokens:
            node = plan.node_map.get(node_name)
            if node is not None and node.kind == FlowNodeKind.HUMAN_REVIEW:
                entry = dict(audit.get(node_name, {}))
                entry["state"] = FlowNodeState.PAUSED.value
                entry["request"] = _strict_human_review_request_metadata(node)
                audit[node_name] = _flow_snapshot_mapping(entry)
    if trace is not None and resume_decisions:
        node_states = {node.node: node.state for node in trace.nodes}
        for node_name, payload in resume_decisions.items():
            if node_states.get(node_name) != FlowNodeState.SUCCEEDED:
                continue
            decision = payload.get("decision")
            if not isinstance(decision, str) or not decision.strip():
                continue
            node = plan.node_map.get(node_name)
            if node is None or node.kind != FlowNodeKind.HUMAN_REVIEW:
                continue
            entry = dict(audit.get(node_name, {}))
            entry.setdefault(
                "request",
                _strict_human_review_request_metadata(node),
            )
            entry["state"] = "resumed"
            entry["decision"] = decision
            audit[node_name] = _flow_snapshot_mapping(entry)
    return MappingProxyType(audit)


def _strict_human_review_audit_from_record(
    record: FlowExecutionRecord | None,
) -> dict[str, Mapping[str, object]]:
    if record is None:
        return {}
    value = record.metadata.get(_FLOW_REVIEW_AUDIT_METADATA_KEY)
    if not isinstance(value, Mapping):
        return {}
    audit: dict[str, Mapping[str, object]] = {}
    for node_name, entry in value.items():
        if (
            isinstance(node_name, str)
            and node_name.strip()
            and isinstance(entry, Mapping)
        ):
            audit[node_name] = _flow_snapshot_mapping(entry)
    return audit


def _strict_human_review_request_metadata(
    node: FlowNodePlan,
) -> Mapping[str, object]:
    assert isinstance(node, FlowNodePlan)
    value: dict[str, object] = {
        "node": node.name,
        "allowed_decisions": _strict_human_review_decisions(node),
    }
    timeout_seconds = node.config.get("timeout_seconds")
    if isinstance(timeout_seconds, int | float) and not isinstance(
        timeout_seconds,
        bool,
    ):
        value["timeout_seconds"] = timeout_seconds
    audit_metadata = node.config.get("audit_metadata")
    if isinstance(audit_metadata, Mapping):
        value["audit_metadata"] = _flow_snapshot_mapping(audit_metadata)
    return _flow_snapshot_mapping(value)


def _strict_human_review_decisions(
    node: FlowNodePlan,
) -> tuple[str, ...]:
    value = node.config.get("allowed_decisions")
    if not isinstance(value, list | tuple) or isinstance(value, str | bytes):
        return ()
    return tuple(
        decision
        for decision in value
        if isinstance(decision, str) and decision.strip()
    )


def _strict_flow_record_signature(
    plan: FlowExecutionPlan,
) -> Mapping[str, object]:
    assert isinstance(plan, FlowExecutionPlan)
    return {
        "name": plan.name,
        "version": plan.version,
        "revision": plan.revision,
        "entry_node": plan.entry_node,
        "inputs": tuple(
            _flow_input_signature(input_) for input_ in plan.inputs
        ),
        "outputs": tuple(output.name for output in plan.outputs),
        "output_contracts": tuple(
            _flow_output_signature(output) for output in plan.outputs
        ),
        "output_selectors": {
            name: _flow_selector_signature(selector)
            for name, selector in plan.output_selectors.items()
        },
        "nodes": tuple(_flow_node_signature(node) for node in plan.nodes),
        "edges": tuple(_flow_edge_signature(edge) for edge in plan.edges),
    }


def _flow_input_signature(
    input_: FlowInputDefinition,
) -> Mapping[str, object]:
    assert isinstance(input_, FlowInputDefinition)
    return {
        "name": input_.name,
        "type": input_.type.value,
        "mime_types": input_.mime_types,
        "schema": _flow_signature_value(input_.schema),
        "schema_ref": input_.schema_ref,
    }


def _flow_output_signature(
    output: FlowOutputDefinition,
) -> Mapping[str, object]:
    assert isinstance(output, FlowOutputDefinition)
    return {
        "name": output.name,
        "type": output.type.value,
        "schema": _flow_signature_value(output.schema),
        "schema_ref": output.schema_ref,
    }


def _flow_node_signature(node: FlowNodePlan) -> Mapping[str, object]:
    assert isinstance(node, FlowNodePlan)
    return {
        "name": node.name,
        "type": node.type,
        "kind": node.kind.value,
        "ref": node.ref,
        "input_contracts": tuple(
            _flow_contract_signature(contract)
            for contract in node.input_contracts
        ),
        "output_contracts": tuple(
            _flow_contract_signature(contract)
            for contract in node.output_contracts
        ),
        "capabilities": tuple(
            capability.value for capability in node.capabilities
        ),
        "mappings": tuple(
            _flow_mapping_signature(mapping) for mapping in node.mappings
        ),
        "join": _flow_join_signature(node.join),
        "retry": _flow_retry_signature(node.retry),
        "timeout": _flow_timeout_signature(node.timeout),
        "loop": _flow_loop_signature(node.loop),
        "config": _flow_signature_value(node.config),
    }


def _flow_edge_signature(edge: FlowEdgePlan) -> Mapping[str, object]:
    assert isinstance(edge, FlowEdgePlan)
    return {
        "index": edge.index,
        "source": edge.source,
        "target": edge.target,
        "kind": edge.kind.value,
        "label": edge.label,
        "condition": _flow_condition_signature(edge.condition),
        "priority": edge.priority,
        "default": edge.default,
        "routing_policy": edge.routing_policy.value,
    }


def _flow_contract_signature(
    contract: FlowNodeContract,
) -> Mapping[str, object]:
    assert isinstance(contract, FlowNodeContract)
    type_ = contract.type
    return {
        "name": contract.name,
        "type": type_.value if isinstance(type_, Enum) else type_,
        "schema": _flow_signature_value(contract.schema),
        "schema_ref": contract.schema_ref,
        "metadata": _flow_signature_value(contract.metadata),
    }


def _flow_mapping_signature(
    mapping: FlowMappingPlan,
) -> Mapping[str, object]:
    assert isinstance(mapping, FlowMappingPlan)
    return {
        "target": mapping.target,
        "kind": mapping.kind.value,
        "source": (
            _flow_selector_signature(mapping.source)
            if mapping.source is not None
            else None
        ),
        "sources": tuple(
            _flow_selector_signature(source) for source in mapping.sources
        ),
        "fields": {
            name: _flow_selector_signature(selector)
            for name, selector in mapping.fields.items()
        },
        "items": tuple(
            _flow_selector_signature(selector) for selector in mapping.items
        ),
    }


def _flow_join_signature(
    join: FlowJoinPlan | None,
) -> Mapping[str, object] | None:
    if join is None:
        return None
    assert isinstance(join, FlowJoinPlan)
    return {
        "type": join.type.value,
        "quorum": join.quorum,
        "optional_inputs": join.optional_inputs,
    }


def _flow_retry_signature(
    retry: FlowRetryPlan | None,
) -> Mapping[str, object] | None:
    if retry is None:
        return None
    assert isinstance(retry, FlowRetryPlan)
    return {
        "max_attempts": retry.max_attempts,
        "backoff": retry.backoff.value,
        "initial_delay_seconds": retry.initial_delay_seconds,
        "max_delay_seconds": retry.max_delay_seconds,
        "retryable_categories": retry.retryable_categories,
        "non_retryable_categories": retry.non_retryable_categories,
        "exhausted_route": retry.exhausted_route,
    }


def _flow_timeout_signature(
    timeout: FlowTimeoutPlan | None,
) -> Mapping[str, object] | None:
    if timeout is None:
        return None
    assert isinstance(timeout, FlowTimeoutPlan)
    return {"per_attempt_seconds": timeout.per_attempt_seconds}


def _flow_loop_signature(
    loop: FlowLoopPlan | None,
) -> Mapping[str, object] | None:
    if loop is None:
        return None
    assert isinstance(loop, FlowLoopPlan)
    return {
        "max_iterations": loop.max_iterations,
        "max_elapsed_seconds": loop.max_elapsed_seconds,
        "continue_condition": _flow_condition_signature(
            loop.continue_condition
        ),
        "exit_condition": _flow_condition_signature(loop.exit_condition),
        "output_selector": _flow_selector_signature(loop.output_selector),
        "limit_route": loop.limit_route,
    }


def _flow_condition_signature(
    condition: FlowConditionPlan | None,
) -> Mapping[str, object] | None:
    if condition is None:
        return None
    assert isinstance(condition, FlowConditionPlan)
    return {
        "operator": condition.operator.value,
        "selector": (
            _flow_selector_signature(condition.selector)
            if condition.selector is not None
            else None
        ),
        "value": _flow_signature_value(condition.value),
        "value_selector": (
            _flow_selector_signature(condition.value_selector)
            if condition.value_selector is not None
            else None
        ),
        "values": tuple(
            _flow_signature_value(value) for value in condition.values
        ),
        "value_type": (
            condition.value_type.value
            if condition.value_type is not None
            else None
        ),
        "conditions": tuple(
            _flow_condition_signature(child) for child in condition.conditions
        ),
        "condition": _flow_condition_signature(condition.condition),
    }


def _flow_selector_signature(selector: FlowSelector) -> Mapping[str, object]:
    assert isinstance(selector, FlowSelector)
    return {
        "root": selector.root.value,
        "source": selector.source,
        "output": selector.output,
        "path": tuple(
            {"kind": step.kind.value, "value": step.value}
            for step in selector.path
        ),
    }


def _flow_signature_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            key: _flow_signature_value(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, list | tuple):
        return tuple(_flow_signature_value(item) for item in value)
    if isinstance(value, Enum):
        return value.value
    return value


def _node_attempt_records(
    trace: FlowExecutionTrace,
) -> tuple[FlowNodeAttemptRecord, ...]:
    assert isinstance(trace, FlowExecutionTrace)
    records: list[FlowNodeAttemptRecord] = []
    for node in trace.nodes:
        records.extend(_node_attempt_records_for_trace(node))
    return tuple(records)


def _flow_node_outputs_snapshot(
    trace: FlowExecutionTrace,
    node_outputs: Mapping[str, Mapping[str, object]],
) -> Mapping[str, object]:
    assert isinstance(trace, FlowExecutionTrace)
    assert isinstance(node_outputs, Mapping)
    states = {node.node: node.state for node in trace.nodes}
    return {
        node: _flow_snapshot_mapping(outputs)
        for node, outputs in node_outputs.items()
        if states.get(node) == FlowNodeState.SUCCEEDED
    }


def _strict_record_node_outputs(
    record: FlowExecutionRecord,
) -> Mapping[str, Mapping[str, object]]:
    assert isinstance(record, FlowExecutionRecord)
    outputs: dict[str, Mapping[str, object]] = {}
    for node, node_output in record.node_outputs.items():
        if not isinstance(node, str) or not node.strip():
            continue
        if isinstance(node_output, Mapping):
            outputs[node] = _flow_snapshot_mapping(node_output)
    return MappingProxyType(outputs)


def _node_attempt_records_for_trace(
    node: FlowNodeTrace,
) -> tuple[FlowNodeAttemptRecord, ...]:
    assert isinstance(node, FlowNodeTrace)
    records: list[FlowNodeAttemptRecord] = []
    for attempt in range(1, node.attempts + 1):
        final_attempt = attempt == node.attempts
        records.append(
            FlowNodeAttemptRecord(
                node=node.node,
                attempt=attempt,
                state=node.state if final_attempt else FlowNodeState.FAILED,
                duration_ms=node.duration_ms if final_attempt else None,
                diagnostics=node.diagnostics if final_attempt else (),
            )
        )
    return tuple(records)


def _loop_counters(
    plan: FlowExecutionPlan,
    trace: FlowExecutionTrace,
) -> Mapping[str, int]:
    trace_by_node = {node.node: node for node in trace.nodes}
    return {
        node.name: trace_by_node[node.name].attempts
        for node in plan.nodes
        if node.loop is not None
        and node.name in trace_by_node
        and trace_by_node[node.name].attempts > 0
    }


def _flow_snapshot_mapping(
    value: Mapping[str, object],
) -> Mapping[str, object]:
    return {
        key: _flow_snapshot_value(item)
        for key, item in value.items()
        if isinstance(key, str) and key.strip()
    }


def _flow_snapshot_value(value: object) -> object:
    if value is None or isinstance(value, bool | str | int | float):
        return value
    if isinstance(value, TaskInputFile):
        return value.summary()
    if isinstance(value, TaskArtifactRef):
        return value.summary(include_metadata=False, include_sha256=True)
    if isinstance(value, Mapping):
        return _flow_snapshot_mapping(value)
    if isinstance(value, list | tuple):
        return tuple(_flow_snapshot_value(item) for item in value)
    return {"type": type(value).__name__}


def _artifact_refs(
    value: object,
) -> tuple[Mapping[str, object], ...]:
    refs: list[Mapping[str, object]] = []
    seen: set[str] = set()
    _append_artifact_refs(value, refs, seen)
    return tuple(refs)


def _append_artifact_refs(
    value: object,
    refs: list[Mapping[str, object]],
    seen: set[str],
) -> None:
    if isinstance(value, TaskInputFile):
        if value.artifact_ref is not None:
            _append_artifact_ref(value.artifact_ref, refs, seen)
        return
    if isinstance(value, TaskArtifactRef):
        _append_artifact_ref(value, refs, seen)
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _append_artifact_refs(item, refs, seen)
        return
    if isinstance(value, list | tuple):
        for item in value:
            _append_artifact_refs(item, refs, seen)


def _append_artifact_ref(
    value: TaskArtifactRef,
    refs: list[Mapping[str, object]],
    seen: set[str],
) -> None:
    key = f"{value.store}:{value.artifact_id}"
    if key in seen:
        return
    seen.add(key)
    summary = value.summary(include_metadata=False, include_sha256=True)
    assert isinstance(summary, Mapping)
    refs.append(summary)


async def _emit_flow_event(
    context: TaskTargetContext,
    event_type: EventType,
    *,
    status: str,
    started: float,
    finished: float | None = None,
) -> None:
    if context.event_listener is None:
        return
    result = context.event_listener(
        Event(
            type=event_type,
            payload={
                "name": "flow",
                "status": status,
            },
            started=started,
            finished=finished,
            elapsed=finished - started if finished is not None else None,
        )
    )
    if result is not None:
        await result


def _single_start_node_name(flow: Flow) -> str | None:
    start_nodes = [
        name for name, inbound in flow.incoming.items() if not inbound
    ]
    if len(start_nodes) != 1:
        return None
    return start_nodes[0]


def _is_path_escape(ref: str) -> bool:
    if "://" in ref or "\\" in ref:
        return True
    posix_path = PurePosixPath(ref)
    windows_path = PureWindowsPath(ref)
    if posix_path.is_absolute() or windows_path.is_absolute():
        return True
    return ".." in posix_path.parts or ".." in windows_path.parts


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _path_escape_issue() -> TaskValidationIssue:
    return TaskValidationIssue(
        code="execution.path_escape",
        path="execution.ref",
        message="Task execution reference escapes allowed roots.",
        hint="Use a logical reference inside an allowed execution root.",
        category=TaskValidationCategory.PRIVACY,
    )


def _unknown_target_issue() -> TaskValidationIssue:
    return TaskValidationIssue(
        code="execution.unknown_target",
        path="execution.type",
        message="Task execution target is not supported.",
        hint="Use a flow execution target.",
        category=TaskValidationCategory.UNSUPPORTED,
    )


def _unsupported_flow_issue(
    *,
    path: str,
    message: str,
    hint: str,
) -> TaskValidationIssue:
    return TaskValidationIssue(
        code="execution.unsupported_flow",
        path=path,
        message=message,
        hint=hint,
        category=TaskValidationCategory.UNSUPPORTED,
    )


def _invalid_flow_file_issue(
    node_name: str,
    index: int,
    message: str,
    hint: str,
) -> TaskValidationIssue:
    return TaskValidationIssue(
        code="execution.unsupported_flow",
        path=f"nodes.{node_name}.input[{index}]",
        message=message,
        hint=hint,
        category=TaskValidationCategory.UNSUPPORTED,
    )


def _feature_issue(
    feature: TaskFeature,
    *,
    path: str,
) -> TaskValidationIssue:
    diagnostic = feature_diagnostic(feature, path=path)
    return TaskValidationIssue(
        code=diagnostic.code,
        path=diagnostic.path,
        message=diagnostic.message,
        hint=diagnostic.hint,
        category=TaskValidationCategory.UNSUPPORTED,
    )
