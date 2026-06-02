from ..context import TaskTargetContext
from ..definition import (
    TaskDefinition,
    TaskInputType,
    TaskOutputType,
    TaskTargetType,
)
from ..target import TaskTargetRunner, TaskValidationContext
from ..validation import (
    TaskValidationCategory,
    TaskValidationError,
    TaskValidationIssue,
)

from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowCompatibility:
    issues: tuple[TaskValidationIssue, ...]

    @property
    def compatible(self) -> bool:
        return not self.issues


class FlowTaskTargetRunner(TaskTargetRunner):
    def __init__(self, *, ref_base: str | Path | None = None) -> None:
        self._ref_base = Path(ref_base) if ref_base is not None else None

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
        raise TaskValidationError(
            (
                _unsupported_flow_issue(
                    path="execution.type",
                    message="Flow-backed task execution is not available.",
                    hint="Use an agent execution target.",
                ),
            )
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
    issues.extend(_flow_runtime_issues())
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


def _flow_runtime_issues() -> tuple[TaskValidationIssue, ...]:
    return (
        _unsupported_flow_issue(
            path="execution.async",
            message="Flow task targets do not provide async execution.",
            hint=(
                "Use an agent target until flow execution is task-compatible."
            ),
        ),
        _unsupported_flow_issue(
            path="execution.cancellation",
            message=(
                "Flow task targets do not provide cancellation checkpoints."
            ),
            hint="Use an agent target until flow cancellation is available.",
        ),
        _unsupported_flow_issue(
            path="run.timeout_seconds",
            message="Flow task targets do not enforce task timeouts.",
            hint=(
                "Use an agent target until flow timeout boundaries are ready."
            ),
        ),
        _unsupported_flow_issue(
            path="observability.capture_events",
            message="Flow task targets do not expose sanitized task events.",
            hint="Use an agent target until flow events are task-compatible.",
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
