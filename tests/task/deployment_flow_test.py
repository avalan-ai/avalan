"""Verify retained file bindings through the existing strict Flow engine."""

from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase

from avalan.task.canonical import spec_hash
from avalan.task.context import TaskTargetContext
from avalan.task.deployment import (
    ExecutionDeployment,
    ExecutionDeploymentError,
)
from avalan.task.deployment_catalog import (
    ExecutionDeploymentBinding,
    ExecutionDeploymentCatalog,
)
from avalan.task.deployment_closure import ExecutionClosure
from avalan.task.loader import TaskDefinitionLoader
from avalan.task.store import TaskExecutionContext
from avalan.task.target import TaskTargetCompleted, TaskValidationContext
from avalan.task.targets.file_flow import FileFlowTaskResolver
from avalan.task.targets.flow import FlowTaskTargetRunner
from avalan.task.validation import TaskValidationError


class DeploymentFlowTest(IsolatedAsyncioTestCase):
    async def test_strict_graph_executes_and_modified_dependency_is_rejected(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = Path(__file__).resolve().parents[2] / "src/avalan"
            (root / "task.toml").write_text("""[task]
name="retained-flow"
version="1"
[input]
type="string"
[output]
type="text"
[execution]
type="flow"
ref="flow.toml"
""")
            (root / "graph.mmd").write_text(
                "flowchart LR\nstart route_1@--> finish\n"
            )
            (root / "flow.toml").write_text("""[flow]
name="retained"
version="1"
[[inputs]]
name="prompt"
type="string"
[[outputs]]
name="answer"
type="text"
[entry]
type="node"
node="start"
[output_behavior]
type="map"
[output_behavior.outputs]
answer="finish.value"
[graph]
format="mermaid"
source="file"
mode="executable"
path="graph.mmd"
[nodes.start]
type="pass-through"
[nodes.start.mapping.value]
type="select"
source="input.prompt"
[nodes.finish]
type="pass-through"
[nodes.finish.mapping.value]
type="select"
source="start.value"
""")
            definition = await TaskDefinitionLoader().load(root / "task.toml")
            resolver = FileFlowTaskResolver(root)
            target = FlowTaskTargetRunner(
                ref_base=root, strict_resolver=resolver
            )
            options = target.execution_deployment_options(root)
            manifest = ExecutionDeployment(
                task_ref="task.toml",
                task_hash=await spec_hash(definition, schema_base_path=root),
                files=ExecutionClosure(root, runtime).collect("task.toml"),
                runtime_version=version("avalan"),
                runtime_options=options,
            )
            catalog = ExecutionDeploymentCatalog(
                allowed_runtime_options=tuple(
                    option.name for option in options
                )
            )
            binding = ExecutionDeploymentBinding(
                manifest=manifest,
                application_root=root,
                runtime_root=runtime,
                definition=definition,
                target=target,
            )
            identity = await catalog.register(binding)
            self.assertEqual(
                await target.validate_definition(
                    definition, TaskValidationContext(execution_roots=(root,))
                ),
                (),
            )
            context = TaskTargetContext(
                definition=definition,
                execution=TaskExecutionContext(
                    run_id="run",
                    attempt_id="attempt",
                    attempt_number=1,
                    deployment=manifest,
                ),
                input_value="retained input",
            )
            result = await (await catalog.resolve(identity)).target.run(
                context
            )
            self.assertIsInstance(result, TaskTargetCompleted)
            assert isinstance(result, TaskTargetCompleted)
            self.assertEqual(result.output, "retained input")
            (root / "graph.mmd").write_text(
                "flowchart LR\nstart -->|not executable| finish\n"
            )
            with self.assertRaises(ExecutionDeploymentError):
                await catalog.resolve(identity)
            (root / "flow.toml").write_text("invalid = [")
            with self.assertRaises(TaskValidationError):
                await resolver(context)
            (root / "linked.toml").symlink_to(root / "flow.toml")
            with self.assertRaises(ExecutionDeploymentError):
                await resolver(
                    replace(
                        context,
                        definition=replace(
                            definition,
                            execution=replace(
                                definition.execution, ref="linked.toml"
                            ),
                        ),
                    )
                )
