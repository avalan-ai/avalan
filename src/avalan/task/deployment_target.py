"""Declare the explicit host capability required to bind execution files."""

from .deployment import DeploymentRuntimeOption
from .resume import TaskDurableResumeCoordinator
from .target import TaskTargetRunner

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class ExecutionDeploymentTarget(TaskTargetRunner, Protocol):
    def execution_deployment_options(
        self, application_base: Path
    ) -> tuple[DeploymentRuntimeOption, ...]:
        """Verify the actual resolver root and return non-secret settings."""
        ...


@runtime_checkable
class ExecutionDeploymentResolver(Protocol):
    @property
    def execution_deployment_root(self) -> Path:
        """Return the immutable base actually used to resolve definitions."""
        ...


@runtime_checkable
class ExecutionDeploymentResumeTarget(Protocol):
    def execution_deployment_resume_coordinator(
        self, application_base: Path
    ) -> TaskDurableResumeCoordinator | None:
        """Return the cold loader owned by this retained target binding."""
        ...
