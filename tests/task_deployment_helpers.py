"""Provide typed deployment fixtures with explicit proof boundaries."""

from collections.abc import Mapping
from importlib.metadata import version
from json import dumps
from pathlib import Path
from uuid import UUID

from avalan.agent.execution import InteractionRuntime
from avalan.entities import Input
from avalan.task.canonical import spec_hash
from avalan.task.client import TaskClient
from avalan.task.context import TaskTargetContext
from avalan.task.definition import TaskDefinition
from avalan.task.deployment import DeploymentRuntimeOption, ExecutionDeployment
from avalan.task.deployment_catalog import (
    ExecutionDeploymentBinding,
    ExecutionDeploymentCatalog,
)
from avalan.task.deployment_closure import ExecutionClosure
from avalan.task.stores.pgsql import _definition_to_payload
from avalan.task.target import (
    TaskTargetOutcome,
    TaskTargetRunner,
    TaskValidationContext,
)
from avalan.task.validation import TaskValidationIssue


class DeploymentProviderResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    async def to_str(self) -> str:
        return self.text

    async def to_json(self) -> str:
        return self.text


class DeploymentProvider:
    def __init__(self, loader: "DeploymentProviderLoader") -> None:
        self.loader = loader

    async def __aenter__(self) -> "DeploymentProvider":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        return None

    async def __call__(
        self,
        input: Input,
        *,
        interaction_runtime: InteractionRuntime | None = None,
    ) -> DeploymentProviderResponse:
        self.loader.inputs.append(input)
        result = (
            self.loader.responses.pop(0)
            if self.loader.responses
            else DeploymentProviderResponse(self.loader.response_text)
        )
        if isinstance(result, BaseException):
            raise result
        return result


class DeploymentProviderLoader:
    def __init__(
        self,
        *,
        response_text: str = "result",
        responses: tuple[DeploymentProviderResponse | BaseException, ...] = (),
    ) -> None:
        self.response_text = response_text
        self.responses = list(responses)
        self.paths: list[str] = []
        self.configurations: list[str] = []
        self.inputs: list[Input] = []

    async def from_file(
        self,
        path: str,
        *,
        agent_id: UUID | None,
        disable_memory: bool = False,
        uri: str | None = None,
        tool_settings: object | None = None,
    ) -> DeploymentProvider:
        self.paths.append(path)
        self.configurations.append(Path(path).read_text())
        return DeploymentProvider(self)


class FixtureDeploymentTarget:
    """Bind callback fixtures for protocol tests."""

    def __init__(self, target: TaskTargetRunner, root: Path) -> None:
        self.target = target
        self.root = root.resolve(strict=True)

    def execution_deployment_options(
        self, application_base: Path
    ) -> tuple[DeploymentRuntimeOption, ...]:
        assert application_base.resolve(strict=True) == self.root
        return (
            DeploymentRuntimeOption(name="fixture.binding", value="protocol"),
        )

    async def validate_definition(
        self, definition: TaskDefinition, context: TaskValidationContext
    ) -> tuple[TaskValidationIssue, ...]:
        return await self.target.validate_definition(definition, context)

    async def run(self, context: TaskTargetContext) -> TaskTargetOutcome:
        return await self.target.run(context)


async def configure_fixture_deployment(
    client: TaskClient,
    definition: TaskDefinition,
    root: Path,
    *,
    task_ref: str = "task.toml",
) -> str:
    """Declare real fixture files while retaining the test's typed target."""
    assert definition.skills is None and definition.skills_config is None
    assert not definition.container.enabled
    root = root.resolve(strict=True)
    payload = _definition_to_payload(definition)
    payload.pop("container")
    source = "\n".join(
        f"[{name}]\n"
        + "\n".join(
            f"{dumps(key)} = {_fixture_toml(value)}"
            for key, value in section.items()
            if value is not None
        )
        for name, section in payload.items()
        if isinstance(section, Mapping)
    )
    (root / task_ref).write_text(source)
    agent_path = root / definition.execution.ref
    if not agent_path.exists():
        agent_path.parent.mkdir(parents=True, exist_ok=True)
        agent_path.write_text(
            '[agent]\nname="fixture"\ninstructions="fixture"\n[engine]\nuri="ai://env:KEY@openai/gpt-4o-mini"\n[tool]\nenable=[]'
        )
    target = FixtureDeploymentTarget(client._target, root)
    runtime = Path(__file__).resolve().parents[1] / "src/avalan"
    manifest = ExecutionDeployment(
        task_ref=task_ref,
        task_hash=await spec_hash(definition, schema_base_path=root),
        files=ExecutionClosure(root, runtime).collect(task_ref),
        runtime_version=version("avalan"),
        runtime_options=target.execution_deployment_options(root),
    )
    catalog = ExecutionDeploymentCatalog(
        allowed_runtime_options=("fixture.binding",)
    )
    identity = await catalog.register(
        ExecutionDeploymentBinding(
            manifest=manifest,
            application_root=root,
            runtime_root=runtime,
            definition=definition,
            target=target,
        )
    )
    client._execution_deployments = catalog
    client._execution_deployment_id = identity
    client._target = target
    return identity


def _fixture_toml(value: object) -> str:
    if isinstance(value, Mapping):
        return (
            "{"
            + ", ".join(
                f"{dumps(key)} = {_fixture_toml(item)}"
                for key, item in value.items()
                if item is not None
            )
            + "}"
        )
    if isinstance(value, tuple | list):
        return "[" + ", ".join(_fixture_toml(item) for item in value) + "]"
    assert isinstance(value, str | int | float | bool)
    return dumps(value)
