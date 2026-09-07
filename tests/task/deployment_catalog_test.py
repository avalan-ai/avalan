from dataclasses import replace
from datetime import UTC, datetime
from functools import partial
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase
from unittest.mock import MagicMock
from uuid import UUID, uuid4

from avalan.model.file_delivery import resolve_file_delivery_profile
from avalan.task.canonical import spec_hash
from avalan.task.client import TaskClient
from avalan.task.context import TaskTargetContext
from avalan.task.deployment import (
    DeploymentFile,
    DeploymentFileRoot,
    DeploymentRuntimeOption,
    ExecutionDeployment,
    ExecutionDeploymentError,
)
from avalan.task.deployment_catalog import (
    ExecutionDeploymentBinding,
    ExecutionDeploymentCatalog,
)
from avalan.task.deployment_closure import ExecutionClosure
from avalan.task.loader import TaskDefinitionLoader
from avalan.task.provenance import TriggerInvocationContext
from avalan.task.queue import TaskQueue
from avalan.task.state import TaskRunState
from avalan.task.store import TaskExecutionRequest, TaskRun
from avalan.task.stores.memory import InMemoryTaskStore
from avalan.task.target import CallableTaskTargetRunner
from avalan.task.targets.agent import (
    AgentOrchestrator,
    AgentTaskTargetRunner,
)
from avalan.task.targets.file_flow import FileFlowTaskResolver
from avalan.task.targets.flow import FlowTaskTargetRunner
from avalan.task.worker import TaskWorker
from avalan.tool.context import ToolSettingsContext


async def target(context: TaskTargetContext) -> object:
    return context.input_value


class FixtureLoader:
    async def from_file(
        self,
        path: str,
        *,
        agent_id: UUID | None,
        disable_memory: bool = False,
        uri: str | None = None,
        tool_settings: object | None = None,
    ) -> AgentOrchestrator:
        raise NotImplementedError("catalog tests do not dispatch a provider")


OPTIONS = (
    DeploymentRuntimeOption(name="agent.disable_memory", value=False),
    DeploymentRuntimeOption(
        name="agent.require_shell_pipeline_opt_in", value=False
    ),
)


class ExecutionDeploymentCatalogTest(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.runtime = Path(__file__).resolve().parents[2] / "src/avalan"
        self.catalog = ExecutionDeploymentCatalog(
            allowed_runtime_options=tuple(option.name for option in OPTIONS)
        )

    async def binding(self, name: str) -> ExecutionDeploymentBinding:
        root = self.root / name
        root.mkdir()
        (root / "task.toml").write_text("""[task]
name="deployment"
version="1"
[input]
type="string"
[output]
type="text"
[execution]
type="agent"
ref="agent.toml"
""")
        (root / "agent.toml").write_text(
            '[agent]\ninstructions="' + name + '"\n[tool]\nenable=[]'
        )
        definition = await TaskDefinitionLoader().load(root / "task.toml")
        manifest = ExecutionDeployment(
            task_ref="task.toml",
            task_hash=await spec_hash(definition, schema_base_path=root),
            files=ExecutionClosure(root, self.runtime).collect("task.toml"),
            runtime_version=version("avalan"),
            runtime_options=OPTIONS,
        )
        return ExecutionDeploymentBinding(
            manifest=manifest,
            application_root=root,
            runtime_root=self.runtime,
            definition=definition,
            target=AgentTaskTargetRunner(FixtureLoader(), ref_base=root),
        )

    async def test_retains_old_binding_and_reconstructs_fresh_catalog(
        self,
    ) -> None:
        old = await self.binding("old")
        old_id = await self.catalog.register(old)
        new = await self.binding("new")
        new_id = await self.catalog.register(new)
        self.assertNotEqual(old_id, new_id)
        self.assertIs(await self.catalog.resolve(old_id), old)
        self.assertIs(await self.catalog.resolve(new_id), new)
        self.assertEqual(await self.catalog.register(old), old_id)
        with self.assertRaisesRegex(
            ExecutionDeploymentError, "binding.replacement"
        ):
            await self.catalog.register(replace(old))
        fresh = ExecutionDeploymentCatalog(
            allowed_runtime_options=tuple(option.name for option in OPTIONS)
        )
        reconstructed = replace(
            old,
            target=AgentTaskTargetRunner(
                FixtureLoader(), ref_base=old.application_root
            ),
        )
        self.assertEqual(await fresh.register(reconstructed), old_id)
        self.assertIs(await fresh.resolve(old_id), reconstructed)
        with self.assertRaisesRegex(
            ExecutionDeploymentError, "binding.unavailable"
        ):
            await fresh.resolve(new_id)
        (new.application_root / "agent.toml").write_text(
            '[agent]\ninstructions="tampered"'
        )
        with self.assertRaisesRegex(ExecutionDeploymentError, "closure"):
            await self.catalog.resolve(new_id)
        self.assertIs(await self.catalog.resolve(old_id), old)

    async def test_rejects_manifest_runtime_task_and_capability_mismatches(
        self,
    ) -> None:
        binding = await self.binding("deployment")
        manifest = binding.manifest
        for changed, diagnostic in (
            (replace(manifest, runtime_version="wrong"), "runtime_version"),
            (replace(manifest, task_hash="0" * 64), "task_hash"),
            (replace(manifest, required_tools=("missing",)), "required_tools"),
            (
                replace(
                    manifest,
                    runtime_options=(
                        DeploymentRuntimeOption(
                            name="concurrency_limit", value=2
                        ),
                    ),
                ),
                "runtime_options",
            ),
        ):
            with (
                self.subTest(diagnostic=diagnostic),
                self.assertRaisesRegex(ExecutionDeploymentError, diagnostic),
            ):
                await self.catalog.register(replace(binding, manifest=changed))
        (binding.application_root / "extra").write_text("extra")
        extra = DeploymentFile(
            root=DeploymentFileRoot.APPLICATION, path="extra", sha256="0" * 64
        )
        changed = replace(
            manifest,
            files=tuple(
                sorted(
                    (*manifest.files, extra),
                    key=lambda value: (value.root.value, value.path),
                )
            ),
        )
        with self.assertRaisesRegex(ExecutionDeploymentError, "file.sha256"):
            await self.catalog.register(replace(binding, manifest=changed))

    async def test_binding_rejects_unsealed_runtime_and_capabilities(
        self,
    ) -> None:
        binding = await self.binding("boundaries")
        for changed, diagnostic in (
            (replace(binding, runtime_root=self.root), "runtime.root"),
            (
                replace(binding, target=CallableTaskTargetRunner(target)),
                "target.binding_capability",
            ),
            (
                replace(
                    binding,
                    manifest=replace(binding.manifest, runtime_options=()),
                ),
                "runtime_options",
            ),
            (
                replace(
                    binding,
                    manifest=replace(
                        binding.manifest,
                        container_image_digests=("repo@sha256:" + "a" * 64,),
                    ),
                ),
                "container_image_digests",
            ),
            (
                replace(
                    binding,
                    manifest=replace(
                        binding.manifest, skill_manifest_id="a" * 64
                    ),
                ),
                "skill_manifest_id",
            ),
        ):
            with (
                self.subTest(diagnostic=diagnostic),
                self.assertRaisesRegex(ExecutionDeploymentError, diagnostic),
            ):
                await changed.verify()
        agent = binding.application_root / "agent.toml"
        agent.write_text('[agent]\ninstructions="boundaries"')
        unsealed_tools = replace(
            binding,
            manifest=replace(
                binding.manifest,
                files=ExecutionClosure(
                    binding.application_root, self.runtime
                ).collect("task.toml"),
            ),
        )
        with self.assertRaisesRegex(
            ExecutionDeploymentError, "agent.tool_allowlist"
        ):
            await unsealed_tools.verify()

    async def test_tool_patterns_match_exact_canonical_host_allowlist(
        self,
    ) -> None:
        binding = await self.binding("tools")
        agent = binding.application_root / "agent.toml"
        agent.write_text(
            '[agent]\ninstructions="tools"\n[tool]\nenable="search.*"'
        )
        manifest = replace(
            binding.manifest,
            files=ExecutionClosure(
                binding.application_root, self.runtime
            ).collect("task.toml"),
            required_tools=("search.web",),
            task_hash=await spec_hash(
                binding.definition, schema_base_path=binding.application_root
            ),
        )
        binding = replace(binding, manifest=manifest)
        catalog = ExecutionDeploymentCatalog(
            available_tools=("search.web",),
            allowed_runtime_options=tuple(option.name for option in OPTIONS),
        )
        self.assertEqual(
            await catalog.register(binding), manifest.execution_deployment_id
        )
        with self.assertRaisesRegex(
            ExecutionDeploymentError, "required_tools"
        ):
            await catalog.register(
                replace(binding, manifest=replace(manifest, required_tools=()))
            )
        agent.write_text(
            '[agent]\ninstructions="tools"\n[tool]\nenable=["missing.*"]'
        )
        unmatched = replace(
            binding,
            manifest=replace(
                manifest,
                files=ExecutionClosure(
                    binding.application_root, self.runtime
                ).collect("task.toml"),
                task_hash=await spec_hash(
                    binding.definition,
                    schema_base_path=binding.application_root,
                ),
            ),
        )
        with self.assertRaisesRegex(
            ExecutionDeploymentError, "required_tools"
        ):
            await catalog.register(unmatched)

    async def test_flow_tool_dependency_must_be_declared_before_binding(
        self,
    ) -> None:
        binding = await self.binding("flowtools")
        task = binding.application_root / "task.toml"
        task.write_text(
            task.read_text()
            .replace('type="agent"', 'type="flow"')
            .replace('ref="agent.toml"', 'ref="flow.toml"')
        )
        (binding.application_root / "flow.toml").write_text(
            '[nodes.tool]\ntype="tool"\nref="search.web"'
        )
        definition = await TaskDefinitionLoader().load(task)
        changed = replace(
            binding,
            definition=definition,
            manifest=replace(
                binding.manifest,
                files=ExecutionClosure(
                    binding.application_root, self.runtime
                ).collect("task.toml"),
                task_hash=await spec_hash(
                    definition, schema_base_path=binding.application_root
                ),
            ),
        )
        with self.assertRaisesRegex(
            ExecutionDeploymentError, "required_tools"
        ):
            await changed.verify()

    async def test_native_override_requires_a_concrete_sealed_binding(
        self,
    ) -> None:
        binding = await self.binding("overrides")
        for runner, diagnostic in (
            (
                AgentTaskTargetRunner(
                    FixtureLoader(),
                    ref_base=binding.application_root,
                    token_counter=lambda text: 1,
                ),
                "agent.token_counter_binding",
            ),
            (
                AgentTaskTargetRunner(
                    FixtureLoader(),
                    ref_base=binding.application_root,
                    file_delivery_resolver=partial(
                        resolve_file_delivery_profile
                    ),
                ),
                "agent.file_delivery_binding",
            ),
        ):
            with (
                self.subTest(diagnostic=diagnostic),
                self.assertRaisesRegex(ExecutionDeploymentError, diagnostic),
            ):
                await replace(binding, target=runner).verify()

    async def test_auxiliary_file_delivery_requires_its_application_template(
        self,
    ) -> None:
        binding = await self.binding("auxiliary")
        agent = binding.application_root / "agent.toml"
        agent.write_text(
            agent.read_text().replace(
                "[agent]", '[agent]\nuser_template="agent.md"'
            )
        )
        manifest = replace(
            binding.manifest,
            files=ExecutionClosure(
                binding.application_root, self.runtime
            ).collect("task.toml"),
            task_hash=await spec_hash(
                binding.definition, schema_base_path=binding.application_root
            ),
        )
        binding = replace(binding, manifest=manifest)
        await binding.verify()
        with self.assertRaisesRegex(
            ExecutionDeploymentError, "file.unavailable"
        ):
            await binding.verify(file_delivery=True)
        (binding.application_root / "agent.md").write_text(
            "Application file prompt {{ files }}"
        )
        with self.assertRaisesRegex(ExecutionDeploymentError, "closure"):
            await binding.verify(file_delivery=True)
        complete = replace(
            binding,
            manifest=replace(
                manifest,
                files=ExecutionClosure(
                    binding.application_root, self.runtime
                ).collect("task.toml", file_delivery=True),
            ),
        )
        await complete.verify(file_delivery=True)

    async def test_native_options_and_flow_delegate_capabilities(self) -> None:
        binding = await self.binding("options")
        root = binding.application_root
        loader = FixtureLoader()
        for runner, diagnostic in (
            (AgentTaskTargetRunner(loader), "agent.root"),
            (
                AgentTaskTargetRunner(
                    loader, ref_base=root, tool_settings=ToolSettingsContext()
                ),
                "agent.tool_settings_binding",
            ),
            (
                AgentTaskTargetRunner(
                    loader, ref_base=root, uri="ai://secret@openai/model"
                ),
                "agent.uri_credentials",
            ),
            (FlowTaskTargetRunner(ref_base=root), "flow.resolver_binding"),
            (
                FlowTaskTargetRunner(
                    ref_base=root,
                    strict_resolver=FileFlowTaskResolver(root),
                    agent_runner=CallableTaskTargetRunner(target),
                ),
                "flow.agent_binding",
            ),
        ):
            with (
                self.subTest(diagnostic=diagnostic),
                self.assertRaisesRegex(ExecutionDeploymentError, diagnostic),
            ):
                runner.execution_deployment_options(root)
        native = AgentTaskTargetRunner(
            loader,
            ref_base=root,
            agent_id=uuid4(),
            uri="ai://env:KEY@openai/model",
        )
        options = native.execution_deployment_options(root)
        self.assertIn("agent.id", {item.name for item in options})
        flow = FlowTaskTargetRunner(
            ref_base=root,
            strict_resolver=FileFlowTaskResolver(root),
            agent_runner=native,
        )
        self.assertTrue(
            set(options).issubset(set(flow.execution_deployment_options(root)))
        )
        self.assertIsNone(flow.execution_deployment_resume_coordinator(root))
        empty = FlowTaskTargetRunner(
            ref_base=root, strict_resolver=FileFlowTaskResolver(root)
        )
        self.assertIsNone(empty.execution_deployment_resume_coordinator(root))

    async def test_worker_checks_catalog_identity_and_file_closure(
        self,
    ) -> None:
        binding = await self.binding("worker")
        identity = await self.catalog.register(binding)
        store = InMemoryTaskStore()
        queue = MagicMock(spec=TaskQueue)
        worker = TaskWorker(store, queue, target=binding.target)
        now = datetime(2026, 9, 7, tzinfo=UTC)
        invocation = TriggerInvocationContext(
            trigger_id="trigger",
            trigger_revision=1,
            occurrence_id="occurrence",
            scheduled_at=now,
            dispatched_at=now,
        )
        run = TaskRun(
            run_id="run",
            definition_id=binding.manifest.task_hash,
            state=TaskRunState.QUEUED,
            request=TaskExecutionRequest(
                definition_id=binding.manifest.task_hash, trigger=invocation
            ),
            created_at=now,
            updated_at=now,
        )
        with self.assertRaisesRegex(
            ExecutionDeploymentError, "request.deployment"
        ):
            await worker._verify_execution_deployment(run, binding.definition)
        run = replace(
            run,
            request=replace(
                run.request,
                deployment=binding.manifest,
                file_summaries=("auxiliary",),
            ),
        )
        with self.assertRaisesRegex(
            ExecutionDeploymentError, "catalog.unavailable"
        ):
            await worker._verify_execution_deployment(run, binding.definition)
        worker = TaskWorker(
            store,
            queue,
            target=binding.target,
            execution_deployments=self.catalog,
        )
        wrong = replace(
            binding.definition,
            task=replace(binding.definition.task, version="2"),
        )
        with self.assertRaisesRegex(ExecutionDeploymentError, "task_hash"):
            await worker._verify_execution_deployment(run, wrong)
        await worker._verify_execution_deployment(run, binding.definition)
        client = TaskClient(store, target=binding.target)
        with self.assertRaisesRegex(
            ExecutionDeploymentError, "catalog.unavailable"
        ):
            await client._deployment_client(identity)


def catalog_fixture() -> ExecutionDeploymentCatalogTest:
    """Reuse the typed catalog fixture without re-exporting collected tests."""
    return ExecutionDeploymentCatalogTest()
