from asyncio import run as run_async
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from os import environ
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest import IsolatedAsyncioTestCase, TestCase, main

from avalan.container import (
    CONFORMANCE_PLAN,
    ContainerAuditCorrelation,
    ContainerBackend,
    ContainerBackendCapabilities,
    ContainerBackendDiagnosticCode,
    ContainerBackendOperation,
    ContainerBackendStream,
    ContainerBackendStreamChunk,
    ContainerBuildCachePolicy,
    ContainerBuildContextPolicy,
    ContainerBuildPolicy,
    ContainerCacheMode,
    ContainerCommandMode,
    ContainerDeviceClass,
    ContainerDevicePolicy,
    ContainerDurablePlanKind,
    ContainerDurablePlanMetadata,
    ContainerEffectiveSettings,
    ContainerEnvironmentPolicy,
    ContainerExecutionScope,
    ContainerFakeBackend,
    ContainerFakeBackendScript,
    ContainerFakeRuntimeEnvelopeBackend,
    ContainerFakeRuntimeEnvelopeScript,
    ContainerImageCachePolicy,
    ContainerImagePolicy,
    ContainerMountAccess,
    ContainerMountDeclaration,
    ContainerMountType,
    ContainerNetworkMode,
    ContainerNetworkPolicy,
    ContainerOutputPolicy,
    ContainerPlanRequest,
    ContainerPlanRequestKind,
    ContainerPoolingMode,
    ContainerPoolingPolicy,
    ContainerPoolTeardownMode,
    ContainerProfile,
    ContainerProfileSelection,
    ContainerResourceLimits,
    ContainerResultStatus,
    ContainerRuntimeEnvelope,
    ContainerRuntimeEnvelopeKind,
    ContainerRuntimeEnvelopeOperation,
    ContainerSecretReference,
    ContainerSettings,
    ContainerSettingsSource,
    ContainerSurface,
    ContainerTrustLevel,
    container_backend_capability_profile,
    container_backend_capability_profiles,
    normalize_container_run_plan,
    normalize_runtime_envelope_plan,
    trusted_container_source,
)
from avalan.entities import ToolExecutionStreamEvent, ToolExecutionStreamKind
from avalan.flow import (
    FlowDefinition,
    FlowEntryBehavior,
    FlowInputDefinition,
    FlowInputType,
    FlowNodeDefinition,
    FlowOutputBehavior,
    FlowOutputDefinition,
    FlowOutputType,
    compile_flow_definition,
)
from avalan.task import (
    TaskContainerExecutionSettings,
    TaskDefinition,
    TaskExecutionRequest,
    TaskExecutionTarget,
    TaskInputContract,
    TaskInputType,
    TaskMetadata,
    TaskObservabilityPolicy,
    TaskOutputContract,
    TaskRetryPolicy,
    TaskRunPolicy,
)
from avalan.task.canonical import spec_hash
from avalan.task.container import (
    TASK_CONTAINER_ATTEMPT_KEY,
    TASK_CONTAINER_METADATA_KEY,
    TASK_CONTAINER_WORKER_ENVELOPE_KEY,
    task_container_run_metadata,
    verify_task_container_request,
)
from avalan.task.stores import InMemoryTaskStore
from avalan.tool.shell import (
    CommandExecutor,
    ExecutionPolicy,
    ExecutionResult,
    ExecutionSpec,
    PathOperand,
    ShellCommandRequest,
    ShellContainerCommandExecutor,
    ShellExecutionMode,
    ShellExecutionStatus,
    ShellToolSettings,
    normalize_shell_execution_request,
)

_DIGEST = "c" * 64
_BUILD_DIGEST = "d" * 64
_IMAGE = f"ghcr.io/example/release-tools@sha256:{_DIGEST}"
_BUILD_IMAGE = f"ghcr.io/example/build-tools@sha256:{_BUILD_DIGEST}"


class ContainerReleaseConformanceTest(TestCase):
    def test_core_contract_canonicalizes_plans_without_secret_values(
        self,
    ) -> None:
        profile = ContainerProfile(
            name="core-contract",
            image=ContainerImagePolicy(reference=_IMAGE),
            mounts=(
                ContainerMountDeclaration(
                    source=".",
                    target="/workspace",
                    mount_type=ContainerMountType.WORKSPACE,
                    access=ContainerMountAccess.READ,
                ),
                ContainerMountDeclaration(
                    source="./out",
                    target="/outputs",
                    mount_type=ContainerMountType.OUTPUT,
                    access=ContainerMountAccess.WRITE,
                ),
            ),
            environment=ContainerEnvironmentPolicy(
                variables={"VISIBLE_FLAG": "do-not-fingerprint-value"},
                allowlist=("LANG",),
            ),
            secrets=(
                ContainerSecretReference(
                    name="api-token",
                    env_name="API_TOKEN",
                ),
            ),
            network=ContainerNetworkPolicy(
                mode=ContainerNetworkMode.ALLOWLIST,
                egress_allowlist=("api.example.test",),
            ),
            devices=ContainerDevicePolicy(devices=(ContainerDeviceClass.CPU,)),
            resources=ContainerResourceLimits(
                cpu_count=2,
                memory_bytes=1048576,
                pids=32,
                timeout_seconds=20,
            ),
            command_mode=ContainerCommandMode.FIXED_EXECUTABLE,
        )
        settings = _settings(
            profile,
            scope=ContainerExecutionScope.CORE_CONTRACT,
        )
        request = ContainerPlanRequest(
            request_kind=ContainerPlanRequestKind.TYPED_TOOL,
            logical_name="core-readiness",
            command="avalan-check",
            argv=("avalan-check", "--json"),
            scope=ContainerExecutionScope.CORE_CONTRACT,
            request_id="tool-call-1",
        )

        plan = normalize_container_run_plan(settings, request)
        canonical = plan.canonical_policy_input()
        durable = ContainerDurablePlanMetadata.from_dict(
            plan.to_metadata().to_dict()
        )
        reloaded_settings = ContainerEffectiveSettings.from_dict(
            settings.to_dict()
        )

        self.assertEqual(reloaded_settings.to_dict(), settings.to_dict())
        self.assertEqual(
            canonical["environment_names"],
            ["LANG", "VISIBLE_FLAG"],
        )
        self.assertEqual(canonical["secret_names"], ["api-token"])
        self.assertNotIn("do-not-fingerprint-value", str(canonical))
        self.assertEqual(durable.plan_kind, ContainerDurablePlanKind.RUN)
        durable.assert_matches(plan)

        higher_risk = normalize_container_run_plan(
            _settings(
                replace(
                    profile,
                    network=ContainerNetworkPolicy(
                        mode=ContainerNetworkMode.FULL
                    ),
                ),
                scope=ContainerExecutionScope.CORE_CONTRACT,
            ),
            request,
        )
        self.assertNotEqual(
            plan.plan_fingerprint,
            higher_risk.plan_fingerprint,
        )

    def test_backend_breadth_catalog_and_runtime_gates_are_release_ready(
        self,
    ) -> None:
        expected = {
            ContainerBackend.DOCKER,
            ContainerBackend.PODMAN,
            ContainerBackend.NERDCTL,
            ContainerBackend.APPLE_CONTAINER,
            ContainerBackend.WINDOWS_DOCKER,
        }
        profiles = container_backend_capability_profiles()
        by_backend = {profile.backend for profile in profiles}

        self.assertEqual(expected, by_backend)
        self.assertEqual(
            CONFORMANCE_PLAN.promoted_integration_backends,
            (ContainerBackend.DOCKER,),
        )
        for profile in profiles:
            with self.subTest(profile_id=profile.profile_id):
                probe = profile.probe()
                requirements = probe.runtime_requirements

                self.assertFalse(probe.ok)
                self.assertFalse(probe.available)
                self.assertEqual(
                    probe.diagnostics[0].code,
                    ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
                )
                self.assertEqual(requirements.marker, profile.profile_id)
                self.assertTrue(requirements.environment_variables)
                self.assertEqual(
                    profile.capabilities.to_dict()["backend"],
                    profile.backend.value,
                )

    def test_docker_runtime_e2e_gate_documents_skip_diagnostic(self) -> None:
        profile = container_backend_capability_profile("docker-engine-linux")
        marker = profile.runtime_requirements.environment_variables[0]
        if not environ.get(marker):
            self.skipTest(f"set {marker}=1 to run Docker runtime conformance")

        probe = profile.probe(available=True)

        self.assertTrue(probe.ok)
        self.assertEqual(probe.backend, ContainerBackend.DOCKER)

    def test_runtime_envelope_lifecycle_handoff_uses_durable_metadata(
        self,
    ) -> None:
        profile = ContainerProfile(
            name="agent-envelope",
            image=ContainerImagePolicy(reference=_IMAGE),
        )
        settings = _settings(
            profile,
            scope=ContainerExecutionScope.RUNTIME_ENVELOPE,
        )
        command_plan = normalize_container_run_plan(
            _settings(profile),
            ContainerPlanRequest(
                request_kind=ContainerPlanRequestKind.TYPED_TOOL,
                logical_name="runtime.command",
                command="python",
                argv=("python", "-c", "print('ok')"),
                scope=ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
                request_id="tool-call-1",
            ),
        )
        plan = normalize_runtime_envelope_plan(
            settings,
            ContainerPlanRequest(
                request_kind=ContainerPlanRequestKind.AGENT_SESSION,
                logical_name="agent-session",
                command="avalan",
                argv=("avalan", "agent", "run"),
                scope=ContainerExecutionScope.RUNTIME_ENVELOPE,
                request_id="session-1",
            ),
            envelope_kind=ContainerRuntimeEnvelopeKind.WHOLE_AGENT,
            readiness_timeout_seconds=5,
        )
        durable = ContainerDurablePlanMetadata.from_dict(
            plan.to_metadata().to_dict()
        )
        backend = ContainerFakeRuntimeEnvelopeBackend(
            ContainerFakeRuntimeEnvelopeScript(state={"cursor": "node-1"})
        )
        envelope = ContainerRuntimeEnvelope(
            backend,
            plan,
            correlation=_correlation("agent-envelope"),
        )

        start = _run(envelope.start())
        execution = _run(envelope.execute(command_plan))
        handoff = _run(envelope.handoff(durable))
        cleanup = _run(envelope.cleanup())

        durable.assert_matches(plan)
        self.assertEqual(
            start.execution.status,
            ContainerResultStatus.COMPLETED,
        )
        self.assertEqual(
            execution.execution.status,
            ContainerResultStatus.COMPLETED,
        )
        self.assertIsNotNone(handoff.handoff)
        assert handoff.handoff is not None
        self.assertEqual(handoff.handoff.state["cursor"], "node-1")
        handoff.handoff.metadata.assert_matches(plan)
        self.assertTrue(cleanup.cleanup.ok if cleanup.cleanup else False)
        self.assertIn(
            ContainerRuntimeEnvelopeOperation.SCOPED_EXECUTION,
            backend.operations,
        )

    def test_advanced_runtime_features_are_policy_owned_and_bounded(
        self,
    ) -> None:
        profile = ContainerProfile(
            name="service-build",
            image=ContainerImagePolicy(
                reference=_BUILD_IMAGE,
                build_policy=ContainerBuildPolicy.TRUSTED_ONLY,
                build_context=ContainerBuildContextPolicy(
                    context_path="containers/tooling",
                    context_digest=f"sha256:{_BUILD_DIGEST}",
                    context_size_bytes=2048,
                    max_context_bytes=4096,
                ),
                image_cache=ContainerImageCachePolicy(
                    mode=ContainerCacheMode.READ_ONLY,
                    ttl_seconds=60,
                ),
                build_cache=ContainerBuildCachePolicy(
                    mode=ContainerCacheMode.READ_WRITE,
                    ttl_seconds=60,
                ),
            ),
            mounts=(
                ContainerMountDeclaration(
                    source=".",
                    target="/workspace",
                    mount_type=ContainerMountType.WORKSPACE,
                ),
            ),
            output=ContainerOutputPolicy(
                allow_artifacts=True,
                max_artifact_bytes=4096,
            ),
            pooling=ContainerPoolingPolicy(
                mode=ContainerPoolingMode.SERVICE,
                max_age_seconds=120,
                max_uses=8,
                idle_ttl_seconds=20,
                health_check_command=("avalan-health", "--ready"),
                teardown=ContainerPoolTeardownMode.RESET,
                audit_labels={"surface": "server", "profile": "service"},
            ),
        )
        plan = normalize_container_run_plan(
            _settings(
                profile,
                scope=ContainerExecutionScope.ADVANCED_RUNTIME_FEATURES,
            ),
            ContainerPlanRequest(
                request_kind=ContainerPlanRequestKind.SERVER,
                logical_name="service-profile",
                command="serve",
                argv=("serve", "--once"),
                scope=ContainerExecutionScope.ADVANCED_RUNTIME_FEATURES,
                request_id="request-1",
            ),
        )
        canonical = plan.canonical_policy_input()
        image = cast(Mapping[str, object], canonical["image"])
        pooling = cast(Mapping[str, object], canonical["pooling"])

        self.assertEqual(image["build_policy"], "trusted_only")
        self.assertEqual(
            cast(Mapping[str, object], image["build_context"])[
                "context_digest"
            ],
            f"sha256:{_BUILD_DIGEST}",
        )
        self.assertEqual(
            cast(Mapping[str, object], image["image_cache"])["mode"],
            "read_only",
        )
        self.assertEqual(pooling["mode"], "service")
        self.assertEqual(
            pooling["health_check_command"],
            ["avalan-health", "--ready"],
        )

        with self.assertRaisesRegex(AssertionError, "service pools"):
            ContainerPoolingPolicy(mode=ContainerPoolingMode.SERVICE)
        with self.assertRaisesRegex(AssertionError, "build secrets"):
            ContainerBuildContextPolicy(
                context_path="containers/tooling",
                context_digest=f"sha256:{_BUILD_DIGEST}",
                context_size_bytes=2048,
                secret_names=("registry-token",),
            )


class ContainerReleaseAsyncConformanceTest(IsolatedAsyncioTestCase):
    async def test_shell_container_execution_uses_fake_backend_no_fallback(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "visible.txt").write_text("hello\n", encoding="utf-8")
            settings = ShellToolSettings(workspace_root=str(root))
            policy = ExecutionPolicy(settings=settings)
            container_settings = _settings(
                ContainerProfile.minimal_readonly(
                    name="shell-readonly",
                    image_reference=_IMAGE,
                ),
                required=True,
            )
            plan = await normalize_shell_execution_request(
                _cat_request("visible.txt"),
                policy,
                container_settings=container_settings,
            )
            backend = ContainerFakeBackend(
                ContainerFakeBackendScript(
                    capabilities=_capabilities(),
                    stream_chunks=(
                        ContainerBackendStreamChunk(
                            stream=ContainerBackendStream.STDOUT,
                            content=b"container\n",
                            sequence=0,
                        ),
                    ),
                )
            )
            events: list[ToolExecutionStreamEvent] = []

            async def record(event: ToolExecutionStreamEvent) -> None:
                events.append(event)

            result = await ShellContainerCommandExecutor(
                container_settings=container_settings,
                container_backend=backend,
                local_executor=_NoFallbackExecutor(),
                rootful_authorized=True,
            ).execute(plan.local_spec, stream=record)

        self.assertEqual(plan.mode, ShellExecutionMode.CONTAINER)
        self.assertIsNotNone(plan.container_plan)
        assert plan.container_plan is not None
        self.assertEqual(
            plan.container_plan.run_plan.command.cwd,
            "/workspace",
        )
        self.assertEqual(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(result.backend, "container")
        self.assertEqual(result.stdout, "container\n")
        self.assertIn(
            "container_plan_fingerprint",
            result.metadata,
        )
        self.assertIn(ContainerBackendOperation.CREATE, backend.operations)
        self.assertEqual(
            [event.kind for event in events],
            [ToolExecutionStreamKind.STDOUT],
        )

    async def test_durable_flow_and_task_metadata_survives_restart(
        self,
    ) -> None:
        flow_profile = ContainerProfile.minimal_readonly(
            name="flow-readonly",
            image_reference=_IMAGE,
        )
        flow_result = await compile_flow_definition(
            _flow_definition(
                container=_container_settings(
                    flow_profile,
                    source=trusted_container_source(
                        ContainerSurface.FLOW_TOML
                    ),
                    registry_id="flow-registry",
                    policy_version="flow-policy",
                )
            )
        )

        self.assertTrue(flow_result.ok, flow_result.public_diagnostics)
        assert flow_result.plan is not None
        flow_node = flow_result.plan.node_map["work"]
        self.assertIsNotNone(flow_node.container)
        assert flow_node.container is not None
        self.assertEqual(
            flow_node.container.scope,
            ContainerExecutionScope.DURABLE_WORKFLOW,
        )
        self.assertEqual(flow_node.container.profile_name, "flow-readonly")

        task_definition = _task_definition(
            container=TaskContainerExecutionSettings(
                attempt=_settings(
                    ContainerProfile.minimal_readonly(
                        name="task-attempt",
                        image_reference=_IMAGE,
                    ),
                    required=True,
                    scope=ContainerExecutionScope.DURABLE_WORKFLOW,
                    registry_id="task-registry",
                    policy_version="task-policy",
                ),
                worker_envelope=_settings(
                    ContainerProfile.minimal_readonly(
                        name="task-worker",
                        image_reference=_IMAGE,
                    ),
                    required=True,
                    scope=ContainerExecutionScope.RUNTIME_ENVELOPE,
                    registry_id="worker-registry",
                    policy_version="worker-policy",
                ),
            )
        )
        store = InMemoryTaskStore()
        definition_id = await spec_hash(task_definition)
        await store.register_definition(
            task_definition,
            definition_hash=definition_id,
        )
        run = await store.create_run(
            TaskExecutionRequest(
                definition_id=definition_id,
                metadata=task_container_run_metadata(task_definition, None),
            )
        )
        attempt = await store.create_attempt(run.run_id)
        plans = verify_task_container_request(
            task_definition,
            run=run,
            attempt=attempt,
        )
        metadata = cast(
            Mapping[str, object],
            run.request.metadata[TASK_CONTAINER_METADATA_KEY],
        )
        attempt_metadata = cast(
            Mapping[str, object],
            metadata[TASK_CONTAINER_ATTEMPT_KEY],
        )
        worker_metadata = cast(
            Mapping[str, object],
            metadata[TASK_CONTAINER_WORKER_ENVELOPE_KEY],
        )

        assert plans.attempt is not None
        assert plans.worker_envelope is not None
        self.assertEqual(
            attempt_metadata["plan_fingerprint"],
            plans.attempt.plan_fingerprint,
        )
        self.assertEqual(
            worker_metadata["plan_fingerprint"],
            plans.worker_envelope.plan_fingerprint,
        )
        ContainerDurablePlanMetadata.from_dict(
            plans.to_metadata()[TASK_CONTAINER_ATTEMPT_KEY]
        ).assert_matches(plans.attempt)
        ContainerDurablePlanMetadata.from_dict(
            plans.to_metadata()[TASK_CONTAINER_WORKER_ENVELOPE_KEY]
        ).assert_matches(plans.worker_envelope)


def _settings(
    profile: ContainerProfile,
    *,
    required: bool = False,
    scope: ContainerExecutionScope = (
        ContainerExecutionScope.SHELL_CONTAINER_EXECUTION
    ),
    source: ContainerSettingsSource | None = None,
    registry_id: str = "release-registry",
    policy_version: str = "release-policy",
) -> ContainerEffectiveSettings:
    settings = _container_settings(
        profile,
        source=source or _trusted_source(),
        registry_id=registry_id,
        policy_version=policy_version,
    )
    return settings.select_profile(
        ContainerProfileSelection(
            required=required,
            profile=profile.name,
            scope=scope,
        )
    )


def _container_settings(
    profile: ContainerProfile,
    *,
    source: ContainerSettingsSource,
    registry_id: str,
    policy_version: str,
) -> ContainerSettings:
    return ContainerSettings(
        source=source,
        backend=ContainerBackend.DOCKER,
        default_profile=profile.name,
        allowed_profiles=(profile.name,),
        profiles={profile.name: profile},
        profile_registry_id=registry_id,
        policy_version=policy_version,
    )


def _trusted_source() -> ContainerSettingsSource:
    return ContainerSettingsSource(
        surface=ContainerSurface.SDK,
        trust_level=ContainerTrustLevel.TRUSTED_DEPLOYMENT,
    )


def _cat_request(path: str) -> ShellCommandRequest:
    return ShellCommandRequest(
        tool_name="shell.cat",
        command="cat",
        options={},
        paths=(
            PathOperand(
                name="path",
                path=path,
                kind="text_file",
                access="read",
            ),
        ),
        cwd=None,
    )


def _capabilities() -> ContainerBackendCapabilities:
    return ContainerBackendCapabilities(
        backend=ContainerBackend.DOCKER,
        host_os="linux",
        guest_os="linux",
        architecture="amd64",
        rootless=False,
        user_namespace=True,
        pull=True,
        mount_types=(ContainerMountType.WORKSPACE,),
        resource_limits=True,
        streaming_attach=True,
        stats=True,
    )


def _flow_definition(*, container: ContainerSettings) -> FlowDefinition:
    return FlowDefinition(
        name="release-container-flow",
        version="2026-06-24",
        inputs=(
            FlowInputDefinition(name="payload", type=FlowInputType.OBJECT),
        ),
        outputs=(
            FlowOutputDefinition(name="answer", type=FlowOutputType.OBJECT),
        ),
        entry_behavior=FlowEntryBehavior(node="work"),
        output_behavior=FlowOutputBehavior(outputs={"answer": "work.value"}),
        container=container,
        nodes=(FlowNodeDefinition(name="work", type="pass-through"),),
    )


def _task_definition(
    *,
    container: TaskContainerExecutionSettings,
) -> TaskDefinition:
    return TaskDefinition(
        task=TaskMetadata(name="release-container-task", version="1"),
        input=TaskInputContract(type=TaskInputType.STRING, required=False),
        output=TaskOutputContract.text(),
        execution=TaskExecutionTarget.agent("agents/release.toml"),
        run=TaskRunPolicy.direct(timeout_seconds=30),
        retry=TaskRetryPolicy(),
        observability=TaskObservabilityPolicy(),
        container=container,
    )


def _run(awaitable: Awaitable[object]) -> object:
    return run_async(awaitable)


def _correlation(profile_name: str) -> ContainerAuditCorrelation:
    return ContainerAuditCorrelation(
        profile_name=profile_name,
        policy_version="release-policy",
        session_id="session-1",
        image_digest=f"sha256:{_DIGEST}",
    )


class _NoFallbackExecutor(CommandExecutor):
    async def execute(
        self,
        spec: ExecutionSpec,
        *,
        stream: (
            Callable[[ToolExecutionStreamEvent], Awaitable[None]] | None
        ) = None,
    ) -> ExecutionResult:
        _ = spec
        _ = stream
        raise AssertionError("local fallback must not be used")


if __name__ == "__main__":
    main()
