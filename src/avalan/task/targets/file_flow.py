"""Resolve strict Flow definitions from one host-retained execution root."""

from ...flow.definition import FlowDefinition
from ...flow.loader import FlowDefinitionLoader
from ..context import TaskTargetContext
from ..deployment import ExecutionDeploymentError, deployment_path
from ..validation import (
    TaskValidationCategory,
    TaskValidationError,
    TaskValidationIssue,
)
from .flow import task_flow_node_registry

from pathlib import Path


class FileFlowTaskResolver:
    """Load a strict Flow graph from its retained root."""

    def __init__(self, root: Path) -> None:
        self._root = root.resolve(strict=True)

    @property
    def execution_deployment_root(self) -> Path:
        return self._root

    async def __call__(self, context: TaskTargetContext) -> FlowDefinition:
        reference = deployment_path(context.definition.execution.ref)
        candidate = self._root / reference
        if candidate.resolve(strict=True) != candidate:
            raise ExecutionDeploymentError("flow.root")
        result = await FlowDefinitionLoader(
            registry=task_flow_node_registry(context)
        ).load_validation_result(candidate)
        if result.definition is None:
            raise TaskValidationError(
                tuple(
                    TaskValidationIssue(
                        code=issue.code,
                        path=issue.path,
                        message=issue.message,
                        hint=issue.hint,
                        category=TaskValidationCategory.UNSUPPORTED,
                    )
                    for issue in result.issues
                )
            )
        return result.definition
