from ...flow.flow import Flow
from ..context import TaskTargetContext
from ..definition import (
    RunMode,
    TaskDefinition,
    TaskInputType,
    TaskOutputType,
    TaskTargetType,
)
from ..privacy import (
    DROPPED_MARKER,
    ENCRYPTED_MARKER,
    HASHED_MARKER,
    REDACTED_MARKER,
    STORED_MARKER,
)
from ..target import TaskTargetRunner, TaskValidationContext
from ..validation import (
    TaskValidationCategory,
    TaskValidationError,
    TaskValidationIssue,
    validate_task_input,
)

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from inspect import isawaitable
from pathlib import Path, PurePosixPath, PureWindowsPath

FlowResolver = Callable[[TaskTargetContext], Flow | Awaitable[Flow]]
FLOW_TASK_INPUT_KEY = "__task_input__"
_UNAVAILABLE_PRIVACY_MARKERS = frozenset(
    {
        DROPPED_MARKER,
        ENCRYPTED_MARKER,
        HASHED_MARKER,
        REDACTED_MARKER,
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
    ) -> None:
        self._ref_base = Path(ref_base) if ref_base is not None else None
        self._flow_resolver = flow_resolver

    async def validate_definition(
        self,
        definition: TaskDefinition,
        context: TaskValidationContext,
    ) -> tuple[TaskValidationIssue, ...]:
        assert isinstance(definition, TaskDefinition)
        assert isinstance(context, TaskValidationContext)
        return validate_flow_task_compatibility(
            definition,
            context,
            ref_base=self._ref_base,
        ).issues

    async def run(self, context: TaskTargetContext) -> object:
        assert isinstance(context, TaskTargetContext)
        if context.definition.execution.type != TaskTargetType.FLOW:
            raise TaskValidationError((_unknown_target_issue(),))
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
        return await resolved.execute_async(
            initial_node=start_node,
            initial_inputs=flow_task_input_binding(task_input),
            cancellation_checker=context.check_cancelled,
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
    issues.extend(_flow_observability_issues(definition))
    return FlowCompatibility(issues=tuple(issues))


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


def _validate_flow_contracts(
    definition: TaskDefinition,
) -> tuple[TaskValidationIssue, ...]:
    issues: list[TaskValidationIssue] = []
    if definition.input.type in {
        TaskInputType.FILE,
        TaskInputType.FILE_ARRAY,
    }:
        issues.append(
            _unsupported_flow_issue(
                path="input.type",
                message="Flow task targets do not support file inputs.",
                hint="Use an agent target for file-backed tasks.",
            )
        )
    if definition.output.type in {
        TaskOutputType.FILE,
        TaskOutputType.FILE_ARRAY,
        TaskOutputType.ARTIFACT_ARRAY,
    }:
        issues.append(
            _unsupported_flow_issue(
                path="output.type",
                message="Flow task targets do not support artifact outputs.",
                hint="Use an agent target for artifact-producing tasks.",
            )
        )
    return tuple(issues)


def flow_task_input_binding(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        binding = dict(value)
        binding.setdefault(FLOW_TASK_INPUT_KEY, dict(value))
        return binding
    if isinstance(value, list | tuple):
        return {
            FLOW_TASK_INPUT_KEY: tuple(value),
            "items": tuple(value),
        }
    return {
        FLOW_TASK_INPUT_KEY: value,
        "value": value,
    }


def _task_input_value(context: TaskTargetContext) -> object:
    value = context.input_value
    if context.definition.run.mode != RunMode.QUEUE:
        return value
    if (
        isinstance(value, Mapping)
        and value.get("privacy") == STORED_MARKER
        and "value" in value
    ):
        return value["value"]
    if (
        isinstance(value, Mapping)
        and value.get("privacy") in _UNAVAILABLE_PRIVACY_MARKERS
    ):
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


def _single_start_node_name(flow: Flow) -> str | None:
    start_nodes = [
        name for name, inbound in flow.incoming.items() if not inbound
    ]
    if len(start_nodes) != 1:
        return None
    return start_nodes[0]


def _flow_observability_issues(
    definition: TaskDefinition,
) -> tuple[TaskValidationIssue, ...]:
    if (
        not definition.observability.capture_events
        and not definition.observability.metrics
        and not definition.observability.trace
    ):
        return ()
    return (
        _unsupported_flow_issue(
            path="observability.capture_events",
            message="Flow task targets do not expose sanitized task events.",
            hint="Disable flow observability until flow events are available.",
        ),
    )


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
