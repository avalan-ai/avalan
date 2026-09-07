"""Resume an actual native Agent against its retained deployment files."""

from contextlib import AsyncExitStack
from dataclasses import replace
from importlib.metadata import version
from logging import Logger
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase
from unittest.mock import MagicMock, patch
from uuid import uuid4

from agent.durable_runtime_test import (
    _NOW,
    _agent_file,
    _claimed_continuation,
    _context,
    _definition,
    _ManagerFactory,
    _resume_command,
    _ResumeHandle,
    _terminal_request,
    _UnusedDurableStore,
)

from avalan.agent.durable_runtime import TrustedAgentContinuationExecutor
from avalan.agent.loader import OrchestratorLoader
from avalan.model.hubs.huggingface import HuggingfaceHub
from avalan.task.canonical import spec_hash
from avalan.task.deployment import (
    ExecutionDeployment,
    ExecutionDeploymentError,
)
from avalan.task.deployment_catalog import (
    ExecutionDeploymentBinding,
    ExecutionDeploymentCatalog,
)
from avalan.task.deployment_closure import ExecutionClosure
from avalan.task.durable_agent import DurableAgentTaskHost
from avalan.task.provenance import TriggerInvocationContext
from avalan.task.target import TaskTargetSuspended
from avalan.task.targets.agent import AgentTaskTargetRunner


class DeploymentResumeHandle(_ResumeHandle):
    @property
    def request_id(self) -> str:
        return str(self._command.request.request_id)

    @property
    def continuation_id(self) -> str:
        return str(self._command.continuation.continuation_id)

    @property
    def checkpoint_id(self) -> str:
        raise AssertionError(
            "native target fixture does not admit task checkpoints"
        )


class DeploymentResumeTest(IsolatedAsyncioTestCase):
    async def test_native_cold_resume_retains_old_root_after_replacement(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            old, new = root / "old", root / "new"
            old.mkdir()
            new.mkdir()
            source = Path(__file__).resolve().parents[2] / "src/avalan"
            bindings: list[ExecutionDeploymentBinding] = []
            hosts: list[DurableAgentTaskHost] = []
            async with AsyncExitStack() as stack:
                for base in (old, new):
                    agent = _agent_file(base)
                    agent.write_text(
                        agent.read_text() + "\n[tool]\nenable=[]\n"
                    )
                    (base / "task.toml").write_text(
                        '[task]\nname="cold-runtime"\nversion="1"\n[input]\ntype="string"\n[output]\ntype="text"\n[execution]\ntype="agent"\nref="agent.toml"'
                    )
                    loader = OrchestratorLoader(
                        hub=MagicMock(spec=HuggingfaceHub),
                        logger=MagicMock(spec=Logger),
                        participant_id=uuid4(),
                        stack=stack,
                    )
                    host = DurableAgentTaskHost(
                        orchestrator_loader=loader,
                        stack=stack,
                        allowed_roots=(base,),
                        continuation_store=_UnusedDurableStore(),
                        clock=lambda: _NOW,
                    )
                    with self.assertRaisesRegex(
                        ExecutionDeploymentError,
                        "agent.durable_loader_binding",
                    ):
                        host.execution_deployment_options(
                            base,
                            loader=object(),
                            disable_memory=False,
                            uri=None,
                        )
                    with self.assertRaisesRegex(
                        ExecutionDeploymentError, "agent.durable_host_binding"
                    ):
                        AgentTaskTargetRunner(
                            loader,
                            ref_base=base,
                            durable_interaction_runtime_factory=host.interaction_runtime,
                        ).execution_deployment_options(base)
                    target = AgentTaskTargetRunner(
                        loader, ref_base=base, durable_host=host
                    )
                    definition = _definition()
                    manifest = ExecutionDeployment(
                        task_ref="task.toml",
                        task_hash=await spec_hash(
                            definition, schema_base_path=base
                        ),
                        files=ExecutionClosure(base, source).collect(
                            "task.toml"
                        ),
                        runtime_version=version("avalan"),
                        runtime_options=target.execution_deployment_options(
                            base
                        ),
                    )
                    bindings.append(
                        ExecutionDeploymentBinding(
                            manifest=manifest,
                            application_root=base,
                            runtime_root=source,
                            definition=definition,
                            target=target,
                        )
                    )
                    hosts.append(host)
                # A different immutable file closure, not merely another path.
                (new / "agent.toml").write_text(
                    (new / "agent.toml")
                    .read_text()
                    .replace(
                        "Request confirmation",
                        "Replacement must request confirmation",
                    )
                )
                changed = bindings[1]
                changed_manifest = replace(
                    changed.manifest,
                    task_hash=await spec_hash(
                        changed.definition, schema_base_path=new
                    ),
                    files=ExecutionClosure(new, source).collect("task.toml"),
                )
                bindings[1] = replace(changed, manifest=changed_manifest)
                options = tuple(
                    option.name
                    for option in bindings[0].manifest.runtime_options
                )
                catalog = ExecutionDeploymentCatalog(
                    allowed_runtime_options=options
                )
                old_id = await catalog.register(bindings[0])
                new_id = await catalog.register(bindings[1])
                self.assertNotEqual(old_id, new_id)
                trigger = TriggerInvocationContext(
                    trigger_id="trigger",
                    trigger_revision=1,
                    occurrence_id="occurrence",
                    scheduled_at=_NOW,
                    dispatched_at=_NOW,
                )
                initial = _context(_definition(), input_value="frozen input")
                initial = replace(
                    initial,
                    execution=replace(
                        initial.execution,
                        trigger=trigger,
                        deployment=bindings[0].manifest,
                    ),
                )
                first_factory = _ManagerFactory([True])
                with patch(
                    "avalan.agent.loader.ModelManager",
                    side_effect=first_factory,
                ):
                    suspension = await bindings[0].target.run(initial)
                assert (
                    isinstance(suspension, TaskTargetSuspended)
                    and suspension.durable is not None
                )
                portable = suspension.durable.continuation
                retained = await catalog.resolve(old_id)
                assert isinstance(retained.target, AgentTaskTargetRunner)
                self.assertIs(
                    retained.target.execution_deployment_resume_coordinator(
                        old
                    ),
                    hosts[0].resume_coordinator,
                )
                self.assertIsNot(
                    retained.target.execution_deployment_resume_coordinator(
                        old
                    ),
                    hosts[1].resume_coordinator,
                )
                fresh_factory = _ManagerFactory([True], call_id_start=2)
                with patch(
                    "avalan.agent.loader.ModelManager",
                    side_effect=fresh_factory,
                ):
                    runtime = await hosts[
                        0
                    ].continuation_runtime_loader.load_continuation_runtime(
                        portable.definition, portable.revision_binding
                    )
                assert isinstance(
                    runtime.runtime, TrustedAgentContinuationExecutor
                )
                command = _resume_command(
                    _claimed_continuation(portable),
                    _terminal_request(suspension.durable.command.request),
                    runtime,
                )
                handle = DeploymentResumeHandle(runtime.runtime, command)
                resumed_context = replace(initial, durable_resume=handle)
                resumed = await retained.target.resume(resumed_context, handle)
                self.assertIsInstance(resumed, TaskTargetSuspended)
                self.assertEqual(handle.dispatch_count, 1)
                self.assertEqual(resumed_context.execution.trigger, trigger)
                self.assertEqual(
                    portable.definition.agent_definition_locator,
                    (old / "agent.toml").as_uri(),
                )
                (old / "agent.toml").write_text("tampered = true")
                with self.assertRaises(ExecutionDeploymentError):
                    await catalog.resolve(old_id)
                self.assertEqual(handle.dispatch_count, 1)
