from asyncio import run as run_async
from collections.abc import Sequence
from unittest import TestCase, main

from avalan.container import (
    ContainerAuditCorrelation,
    ContainerAuditEventType,
    ContainerAuthorityCaps,
    ContainerBackend,
    ContainerBuildPolicy,
    ContainerCommandMode,
    ContainerDeviceClass,
    ContainerDevicePolicy,
    ContainerDurablePlanMetadata,
    ContainerEffectiveSettings,
    ContainerEnvironmentPolicy,
    ContainerExecutionResult,
    ContainerExecutionScope,
    ContainerFakeRuntimeEnvelopeBackend,
    ContainerFakeRuntimeEnvelopeScript,
    ContainerImagePolicy,
    ContainerMappedDiagnostic,
    ContainerMountAccess,
    ContainerMountDeclaration,
    ContainerMountType,
    ContainerNetworkMode,
    ContainerNetworkPolicy,
    ContainerNormalizedRunPlan,
    ContainerNormalizedRuntimeEnvelopePlan,
    ContainerPlanRequest,
    ContainerPlanRequestKind,
    ContainerProfile,
    ContainerResourceLimits,
    ContainerResultStatus,
    ContainerRuntimeEnvelope,
    ContainerRuntimeEnvelopeCleanupResult,
    ContainerRuntimeEnvelopeCompositionResult,
    ContainerRuntimeEnvelopeHandle,
    ContainerRuntimeEnvelopeHandoff,
    ContainerRuntimeEnvelopeHealth,
    ContainerRuntimeEnvelopeKind,
    ContainerRuntimeEnvelopeOperation,
    ContainerRuntimeEnvelopeOperationResult,
    ContainerRuntimeEnvelopeReadiness,
    ContainerRuntimeEnvelopeShutdownResult,
    ContainerRuntimeEnvelopeState,
    ContainerSecretReference,
    ContainerSettings,
    ContainerSettingsSource,
    ContainerStableDiagnosticCode,
    ContainerSurface,
    ContainerTrustLevel,
    normalize_container_run_plan,
    normalize_runtime_envelope_plan,
    validate_runtime_envelope_composition,
)
from avalan.container.runtime import (
    _audit_records_for_operation,
    _enum_value,
    _execution_result,
    _mapped_runtime_diagnostic,
)

_DIGEST = "a" * 64
_IMAGE = f"ghcr.io/example/runtime-tools@sha256:{_DIGEST}"


class ContainerRuntimeEnvelopeTest(TestCase):
    def test_fake_e2e_runtime_envelope_lifecycle_and_handoff(self) -> None:
        plan = _envelope_plan(
            profile=_profile(
                telemetry_resources=ContainerResourceLimits(
                    cpu_count=4,
                    memory_bytes=1024,
                    pids=64,
                    timeout_seconds=60,
                )
            )
        )
        command_plan = _command_plan(
            resources=ContainerResourceLimits(
                cpu_count=2,
                memory_bytes=512,
                pids=16,
                timeout_seconds=30,
            )
        )
        backend = ContainerFakeRuntimeEnvelopeBackend(
            ContainerFakeRuntimeEnvelopeScript(
                readiness_delay_seconds=0.001,
                telemetry={"cpu_ms": "42"},
                state={"cursor": "node-7"},
                artifacts={"snapshot": "sha256:artifact"},
            )
        )
        envelope = ContainerRuntimeEnvelope(
            backend,
            plan,
            correlation=_correlation(),
        )

        start = run_async(envelope.start())
        execution = run_async(envelope.execute(command_plan))
        health = run_async(envelope.health())
        telemetry = run_async(envelope.telemetry())
        handoff = run_async(envelope.handoff(plan.to_metadata()))
        shutdown = run_async(envelope.shutdown(timeout_seconds=1))
        cleanup = run_async(envelope.cleanup())

        self.assertEqual(
            start.execution.status,
            ContainerResultStatus.COMPLETED,
        )
        self.assertIsNotNone(start.handle)
        assert start.handle is not None
        self.assertEqual(
            start.handle.state, ContainerRuntimeEnvelopeState.READY
        )
        self.assertTrue(start.readiness.ok if start.readiness else False)
        assert envelope.handle is not None
        self.assertEqual(
            envelope.handle.state, ContainerRuntimeEnvelopeState.CLEANED
        )
        self.assertEqual(
            execution.execution.status,
            ContainerResultStatus.COMPLETED,
        )
        self.assertTrue(execution.composition.ok)
        self.assertEqual(execution.telemetry["executions"], "1")
        self.assertTrue(health.ok)
        self.assertEqual(telemetry["cpu_ms"], "42")
        self.assertEqual(telemetry["executions"], "1")
        self.assertEqual(start.to_dict()["readiness"]["ready"], True)
        self.assertEqual(execution.to_dict()["health"]["healthy"], True)
        self.assertIsNotNone(handoff.handoff)
        assert handoff.handoff is not None
        self.assertEqual(
            handoff.to_dict()["handoff"]["state"]["cursor"],
            "node-7",
        )
        self.assertEqual(handoff.handoff.telemetry["executions"], "1")
        self.assertEqual(handoff.handoff.state["cursor"], "node-7")
        self.assertEqual(
            handoff.handoff.artifacts["snapshot"], "sha256:artifact"
        )
        handoff.handoff.metadata.assert_matches(plan)
        self.assertEqual(
            shutdown.execution.status,
            ContainerResultStatus.COMPLETED,
        )
        self.assertEqual(shutdown.to_dict()["shutdown"]["graceful"], True)
        self.assertTrue(shutdown.shutdown.ok if shutdown.shutdown else False)
        self.assertEqual(
            cleanup.execution.status,
            ContainerResultStatus.COMPLETED,
        )
        self.assertTrue(cleanup.cleanup.ok if cleanup.cleanup else False)
        self.assertEqual(
            cleanup.to_dict()["cleanup"],
            {
                "cleanup_completed": True,
                "cleanup_uncertain": False,
                "diagnostics": [],
                "metadata": {"envelope_id": "fake-runtime-1"},
            },
        )
        self.assertEqual(
            backend.operations,
            (
                ContainerRuntimeEnvelopeOperation.START,
                ContainerRuntimeEnvelopeOperation.READINESS,
                ContainerRuntimeEnvelopeOperation.HEALTH,
                ContainerRuntimeEnvelopeOperation.SCOPED_EXECUTION,
                ContainerRuntimeEnvelopeOperation.TELEMETRY,
                ContainerRuntimeEnvelopeOperation.HEALTH,
                ContainerRuntimeEnvelopeOperation.TELEMETRY,
                ContainerRuntimeEnvelopeOperation.HANDOFF,
                ContainerRuntimeEnvelopeOperation.SHUTDOWN,
                ContainerRuntimeEnvelopeOperation.CLEANUP,
            ),
        )

    def test_runtime_value_objects_validate_and_serialize(self) -> None:
        diagnostic = _mapped_runtime_diagnostic(
            ContainerStableDiagnosticCode.CANCELLED,
            "cancelled",
            source_code="container.runtime_envelope.cancelled",
        )
        handle = ContainerRuntimeEnvelopeHandle(
            envelope_id="runtime-1",
            envelope_kind="whole_agent",
            backend="docker",
            plan_fingerprint="fp",
            state="ready",
        )
        readiness = ContainerRuntimeEnvelopeReadiness(ready=True)
        health = ContainerRuntimeEnvelopeHealth(healthy=True)
        handoff = ContainerRuntimeEnvelopeHandoff(
            metadata=_envelope_plan().to_metadata(),
            telemetry={"t": "1"},
            state={"s": "2"},
            artifacts={"a": "3"},
        )
        operation = ContainerRuntimeEnvelopeOperationResult(
            execution=_execution_result((diagnostic,), _correlation()),
            handle=handle,
            readiness=readiness,
            health=health,
            handoff=handoff,
            shutdown=ContainerRuntimeEnvelopeShutdownResult(graceful=True),
            cleanup=ContainerRuntimeEnvelopeCleanupResult(
                cleanup_completed=True
            ),
        )

        self.assertEqual(handle.to_dict()["state"], "ready")
        self.assertEqual(operation.to_dict()["health"]["healthy"], True)
        self.assertEqual(
            operation.execution.status,
            ContainerResultStatus.CANCELLED,
        )
        self.assertEqual(
            _execution_result((), _correlation()).status,
            ContainerResultStatus.COMPLETED,
        )
        self.assertEqual(
            _audit_records_for_operation(
                ContainerRuntimeEnvelopeOperation.TELEMETRY,
                (diagnostic,),
                _correlation(),
            )[0].event_type,
            ContainerAuditEventType.STATS,
        )
        self.assertEqual(
            _enum_value("ready", ContainerRuntimeEnvelopeState, "state"),
            ContainerRuntimeEnvelopeState.READY,
        )
        with self.assertRaises(AssertionError):
            ContainerRuntimeEnvelopeCompositionResult(allowed=False)

    def test_whole_agent_envelope_allows_surface_request_kind(self) -> None:
        plan = _envelope_plan(
            envelope_kind=ContainerRuntimeEnvelopeKind.WHOLE_AGENT,
            request_kind=ContainerPlanRequestKind.AGENT_SESSION,
        )
        envelope = ContainerRuntimeEnvelope(
            ContainerFakeRuntimeEnvelopeBackend(
                ContainerFakeRuntimeEnvelopeScript()
            ),
            plan,
            correlation=_correlation(),
        )

        result = run_async(envelope.start())

        self.assertEqual(
            result.execution.status,
            ContainerResultStatus.COMPLETED,
        )
        assert result.handle is not None
        self.assertEqual(
            result.handle.envelope_kind,
            ContainerRuntimeEnvelopeKind.WHOLE_AGENT,
        )

    def test_readiness_timeout_fails_start_without_real_runtime(self) -> None:
        backend = ContainerFakeRuntimeEnvelopeBackend(
            ContainerFakeRuntimeEnvelopeScript(readiness_timeout=True)
        )
        envelope = ContainerRuntimeEnvelope(
            backend,
            _envelope_plan(),
            correlation=_correlation(),
        )

        result = run_async(envelope.start())

        self.assertEqual(result.execution.status, ContainerResultStatus.FAILED)
        self.assertTrue(result.cleanup.ok if result.cleanup else False)
        self.assertEqual(
            _diagnostic_codes(result),
            {ContainerStableDiagnosticCode.TIMEOUT.value},
        )
        self.assertIn(
            ContainerRuntimeEnvelopeOperation.READINESS,
            backend.operations,
        )
        self.assertIn(
            ContainerRuntimeEnvelopeOperation.CLEANUP,
            backend.operations,
        )
        with self.assertRaisesRegex(AssertionError, "already cleaned"):
            run_async(envelope.execute(_command_plan()))

    def test_start_timeout_before_handle_fails_closed(self) -> None:
        class StartTimeoutBackend(ContainerFakeRuntimeEnvelopeBackend):
            async def start(self, plan):
                raise TimeoutError

        envelope = ContainerRuntimeEnvelope(
            StartTimeoutBackend(ContainerFakeRuntimeEnvelopeScript()),
            _envelope_plan(),
            correlation=_correlation(),
        )

        result = run_async(envelope.start())

        self.assertEqual(result.execution.status, ContainerResultStatus.FAILED)
        self.assertIsNone(result.handle)
        self.assertIsNone(result.cleanup)
        with self.assertRaisesRegex(AssertionError, "not started"):
            run_async(envelope.execute(_command_plan()))

    def test_not_ready_start_fails_closed_and_cleans_up(self) -> None:
        backend = ContainerFakeRuntimeEnvelopeBackend(
            ContainerFakeRuntimeEnvelopeScript(ready=False)
        )
        envelope = ContainerRuntimeEnvelope(
            backend,
            _envelope_plan(),
            correlation=_correlation(),
        )

        result = run_async(envelope.start())

        self.assertEqual(result.execution.status, ContainerResultStatus.FAILED)
        self.assertTrue(result.cleanup.ok if result.cleanup else False)
        self.assertIn(
            ContainerRuntimeEnvelopeOperation.CLEANUP,
            backend.operations,
        )
        with self.assertRaisesRegex(AssertionError, "already cleaned"):
            run_async(envelope.execute(_command_plan()))

    def test_health_failure_blocks_scoped_execution(self) -> None:
        backend = ContainerFakeRuntimeEnvelopeBackend(
            ContainerFakeRuntimeEnvelopeScript(healthy=False)
        )
        envelope = ContainerRuntimeEnvelope(
            backend,
            _envelope_plan(),
            correlation=_correlation(),
        )
        run_async(envelope.start())

        result = run_async(envelope.execute(_command_plan()))

        self.assertEqual(result.execution.status, ContainerResultStatus.FAILED)
        self.assertFalse(result.health.ok if result.health else True)
        self.assertNotIn(
            ContainerRuntimeEnvelopeOperation.SCOPED_EXECUTION,
            backend.operations,
        )

    def test_direct_health_failure_marks_envelope_degraded(self) -> None:
        envelope = ContainerRuntimeEnvelope(
            ContainerFakeRuntimeEnvelopeBackend(
                ContainerFakeRuntimeEnvelopeScript(healthy=False)
            ),
            _envelope_plan(),
            correlation=_correlation(),
        )
        run_async(envelope.start())

        health = run_async(envelope.health())

        self.assertFalse(health.ok)
        assert envelope.handle is not None
        self.assertEqual(
            envelope.handle.state,
            ContainerRuntimeEnvelopeState.DEGRADED,
        )
        with self.assertRaisesRegex(AssertionError, "not ready"):
            run_async(envelope.execute(_command_plan()))

    def test_nested_runtime_envelope_composition_is_rejected(self) -> None:
        plan = _envelope_plan()
        nested = _envelope_plan(request_id="nested-runtime").run_plan

        result = validate_runtime_envelope_composition(plan, nested)

        self.assertFalse(result.ok)
        self.assertIn(
            "container.runtime_envelope.nested_runtime",
            _source_codes(result.diagnostics),
        )

    def test_double_mount_composition_is_rejected(self) -> None:
        profile = _profile(
            mounts=(
                ContainerMountDeclaration(
                    source=".",
                    target="/workspace",
                    mount_type=ContainerMountType.WORKSPACE,
                ),
            )
        )
        plan = _envelope_plan(profile=profile)
        command = _command_plan(profile=profile)

        result = validate_runtime_envelope_composition(plan, command)

        self.assertFalse(result.ok)
        self.assertIn(
            "container.runtime_envelope.double_mount",
            _source_codes(result.diagnostics),
        )

    def test_policy_widening_composition_is_rejected(self) -> None:
        plan = _envelope_plan()
        command = _command_plan(
            profile=_profile(
                network=ContainerNetworkPolicy(
                    mode=ContainerNetworkMode.ALLOWLIST,
                    egress_allowlist=("api.example.test",),
                ),
                telemetry_resources=ContainerResourceLimits(cpu_count=1),
            )
        )

        result = validate_runtime_envelope_composition(plan, command)

        self.assertFalse(result.ok)
        self.assertIn(
            "container.runtime_envelope.policy_widening",
            _source_codes(result.diagnostics),
        )

    def test_all_policy_widening_dimensions_are_rejected(self) -> None:
        plan = _envelope_plan(
            profile=_profile(
                mounts=(
                    ContainerMountDeclaration(
                        source=".",
                        target="/scratch",
                        mount_type=ContainerMountType.SCRATCH,
                    ),
                )
            )
        )
        command = _command_plan(
            backend=ContainerBackend.APPLE_CONTAINER,
            registry="other-registry",
            policy_version="phase16",
            profile=_profile(
                name="other-profile",
                environment=ContainerEnvironmentPolicy(
                    variables={"EXTRA": "1"}
                ),
                mounts=(
                    ContainerMountDeclaration(
                        source="./other",
                        target="/other",
                        mount_type=ContainerMountType.SCRATCH,
                    ),
                    ContainerMountDeclaration(
                        source=".",
                        target="/scratch",
                        mount_type=ContainerMountType.SCRATCH,
                        access=ContainerMountAccess.WRITE,
                    ),
                ),
                network=ContainerNetworkPolicy(
                    mode=ContainerNetworkMode.ALLOWLIST,
                    egress_allowlist=("api.example.test",),
                ),
                secrets=(
                    ContainerSecretReference(
                        name="api-token",
                        env_name="API_TOKEN",
                    ),
                ),
            ),
        )

        result = validate_runtime_envelope_composition(plan, command)

        self.assertFalse(result.ok)
        messages = {diagnostic.message for diagnostic in result.diagnostics}
        self.assertIn(
            "nested command container policy would widen: "
            "profile registry changed",
            messages,
        )
        self.assertIn(
            "nested command container policy would widen: "
            "policy version changed",
            messages,
        )
        self.assertIn(
            "nested command container policy would widen: backend changed",
            messages,
        )
        self.assertIn(
            "nested command container policy would widen: profile changed",
            messages,
        )
        self.assertIn(
            "nested command container policy would widen: secrets widened",
            messages,
        )
        self.assertIn(
            "nested command container policy would widen: environment widened",
            messages,
        )
        self.assertIn(
            "nested command container policy would widen: mounts widened",
            messages,
        )
        self.assertIn(
            "nested command container policy would widen: "
            "mount access widened",
            messages,
        )

    def test_resource_cap_composition_respects_unbounded_envelope(
        self,
    ) -> None:
        plan = _envelope_plan()
        command = _command_plan(resources=ContainerResourceLimits(cpu_count=1))

        result = validate_runtime_envelope_composition(plan, command)

        self.assertTrue(result.ok)

    def test_resource_cap_removal_is_policy_widening(self) -> None:
        plan = _envelope_plan(
            profile=_profile(
                telemetry_resources=ContainerResourceLimits(cpu_count=1)
            )
        )
        command = _command_plan()

        result = validate_runtime_envelope_composition(plan, command)

        self.assertFalse(result.ok)
        self.assertIn(
            "container.runtime_envelope.policy_widening",
            _source_codes(result.diagnostics),
        )

    def test_execute_returns_denied_result_for_policy_widening(self) -> None:
        backend = ContainerFakeRuntimeEnvelopeBackend(
            ContainerFakeRuntimeEnvelopeScript()
        )
        envelope = ContainerRuntimeEnvelope(
            backend,
            _envelope_plan(),
            correlation=_correlation(),
        )
        run_async(envelope.start())

        result = run_async(
            envelope.execute(
                _command_plan(
                    profile=_profile(
                        devices=ContainerDevicePolicy(
                            devices=(ContainerDeviceClass.NVIDIA_CDI,)
                        )
                    )
                )
            )
        )

        self.assertEqual(result.execution.status, ContainerResultStatus.DENIED)
        self.assertNotIn(
            ContainerRuntimeEnvelopeOperation.SCOPED_EXECUTION,
            backend.operations,
        )

    def test_backend_execution_diagnostics_are_audited(self) -> None:
        envelope = ContainerRuntimeEnvelope(
            ContainerFakeRuntimeEnvelopeBackend(
                ContainerFakeRuntimeEnvelopeScript(
                    execution_result=ContainerExecutionResult(
                        status=ContainerResultStatus.FAILED,
                        diagnostics=("boom",),
                    )
                )
            ),
            _envelope_plan(),
            correlation=_correlation(),
        )
        run_async(envelope.start())

        result = run_async(envelope.execute(_command_plan()))

        self.assertEqual(result.execution.status, ContainerResultStatus.FAILED)
        self.assertEqual(
            result.audit_records[1].diagnostics[0].message,
            "boom",
        )

    def test_shutdown_timeout_is_reported(self) -> None:
        backend = ContainerFakeRuntimeEnvelopeBackend(
            ContainerFakeRuntimeEnvelopeScript(shutdown_timeout=True)
        )
        envelope = ContainerRuntimeEnvelope(
            backend,
            _envelope_plan(),
            correlation=_correlation(),
        )
        run_async(envelope.start())

        result = run_async(envelope.shutdown(timeout_seconds=1))

        self.assertEqual(result.execution.status, ContainerResultStatus.FAILED)
        self.assertFalse(result.shutdown.ok if result.shutdown else True)
        self.assertEqual(
            _diagnostic_codes(result),
            {ContainerStableDiagnosticCode.TIMEOUT.value},
        )

    def test_execute_after_shutdown_or_cleanup_is_rejected(self) -> None:
        envelope = ContainerRuntimeEnvelope(
            ContainerFakeRuntimeEnvelopeBackend(
                ContainerFakeRuntimeEnvelopeScript()
            ),
            _envelope_plan(),
            correlation=_correlation(),
        )
        run_async(envelope.start())
        run_async(envelope.shutdown(timeout_seconds=1))

        with self.assertRaisesRegex(AssertionError, "not ready"):
            run_async(envelope.execute(_command_plan()))

        run_async(envelope.cleanup())

        with self.assertRaisesRegex(AssertionError, "already cleaned"):
            run_async(envelope.execute(_command_plan()))

    def test_cleanup_failure_is_reported(self) -> None:
        backend = ContainerFakeRuntimeEnvelopeBackend(
            ContainerFakeRuntimeEnvelopeScript(cleanup_failure=True)
        )
        envelope = ContainerRuntimeEnvelope(
            backend,
            _envelope_plan(),
            correlation=_correlation(),
        )
        run_async(envelope.start())

        result = run_async(envelope.cleanup())

        self.assertEqual(result.execution.status, ContainerResultStatus.FAILED)
        self.assertFalse(result.cleanup.ok if result.cleanup else True)
        self.assertEqual(
            _diagnostic_codes(result),
            {ContainerStableDiagnosticCode.CLEANUP_FAILED.value},
        )

    def test_durable_handoff_revalidates_metadata(self) -> None:
        plan = _envelope_plan(request_id="runtime-1")
        stale_plan = _envelope_plan(request_id="runtime-2")
        stale_metadata = ContainerDurablePlanMetadata.from_dict(
            stale_plan.to_metadata().to_dict()
        )
        envelope = ContainerRuntimeEnvelope(
            ContainerFakeRuntimeEnvelopeBackend(
                ContainerFakeRuntimeEnvelopeScript()
            ),
            plan,
            correlation=_correlation(),
        )
        run_async(envelope.start())

        with self.assertRaises(AssertionError):
            run_async(envelope.handoff(stale_metadata))


def _envelope_plan(
    *,
    envelope_kind: ContainerRuntimeEnvelopeKind = (
        ContainerRuntimeEnvelopeKind.FLOW_RUNTIME
    ),
    profile: ContainerProfile | None = None,
    request_kind: ContainerPlanRequestKind = (
        ContainerPlanRequestKind.RUNTIME_ENVELOPE
    ),
    request_id: str = "runtime-1",
) -> ContainerNormalizedRuntimeEnvelopePlan:
    return normalize_runtime_envelope_plan(
        _effective_settings(profile=profile),
        _request(
            request_kind=request_kind,
            request_id=request_id,
            scope=ContainerExecutionScope.RUNTIME_ENVELOPE,
        ),
        envelope_kind=envelope_kind,
        readiness_timeout_seconds=1,
    )


def _command_plan(
    *,
    backend: ContainerBackend = ContainerBackend.DOCKER,
    profile: ContainerProfile | None = None,
    registry: str = "runtime-registry",
    policy_version: str = "phase15",
    resources: ContainerResourceLimits | None = None,
) -> ContainerNormalizedRunPlan:
    resolved_profile = profile or _profile(telemetry_resources=resources)
    if resources is not None and profile is not None:
        resolved_profile = _profile(
            name=profile.name,
            environment=profile.environment,
            network=profile.network,
            devices=profile.devices,
            mounts=tuple(profile.mounts),
            secrets=tuple(profile.secrets),
            telemetry_resources=resources,
        )
    return normalize_container_run_plan(
        _effective_settings(
            backend=backend,
            profile=resolved_profile,
            registry=registry,
            policy_version=policy_version,
        ),
        _request(),
    )


def _request(
    *,
    request_kind: ContainerPlanRequestKind = (
        ContainerPlanRequestKind.TYPED_TOOL
    ),
    request_id: str | None = None,
    scope: ContainerExecutionScope = (
        ContainerExecutionScope.SHELL_CONTAINER_EXECUTION
    ),
) -> ContainerPlanRequest:
    return ContainerPlanRequest(
        request_kind=request_kind,
        logical_name="runtime.command",
        command="python",
        argv=("python", "-c", "print('ok')"),
        cwd="/workspace",
        scope=scope,
        request_id=request_id,
    )


def _effective_settings(
    *,
    backend: ContainerBackend = ContainerBackend.DOCKER,
    profile: ContainerProfile | None = None,
    registry: str = "runtime-registry",
    policy_version: str = "phase15",
) -> ContainerEffectiveSettings:
    resolved_profile = profile or _profile()
    settings = ContainerSettings(
        source=ContainerSettingsSource(
            surface=ContainerSurface.RUNTIME_ENVELOPE,
            trust_level=ContainerTrustLevel.TRUSTED_OPERATOR,
        ),
        backend=backend,
        default_profile=resolved_profile.name,
        allowed_profiles=(resolved_profile.name,),
        profiles={resolved_profile.name: resolved_profile},
        profile_registry_id=registry,
        policy_version=policy_version,
    )
    return ContainerAuthorityCaps(settings=settings).merge(())


def _profile(
    *,
    name: str = "runtime-profile",
    environment: ContainerEnvironmentPolicy | None = None,
    network: ContainerNetworkPolicy | None = None,
    devices: ContainerDevicePolicy | None = None,
    mounts: tuple[ContainerMountDeclaration, ...] = (),
    secrets: tuple[ContainerSecretReference, ...] = (),
    telemetry_resources: ContainerResourceLimits | None = None,
) -> ContainerProfile:
    return ContainerProfile(
        name=name,
        image=ContainerImagePolicy(
            reference=_IMAGE,
            build_policy=ContainerBuildPolicy.DISABLED,
        ),
        command_mode=ContainerCommandMode.FIXED_EXECUTABLE,
        environment=environment or ContainerEnvironmentPolicy(),
        mounts=mounts,
        network=network or ContainerNetworkPolicy(),
        devices=devices or ContainerDevicePolicy(),
        secrets=secrets,
        resources=telemetry_resources or ContainerResourceLimits(),
    )


def _correlation() -> ContainerAuditCorrelation:
    return ContainerAuditCorrelation(
        profile_name="runtime-profile",
        policy_version="phase15",
        flow_node_id="flow-1",
        image_digest=f"sha256:{_DIGEST}",
    )


def _diagnostic_codes(
    result: ContainerRuntimeEnvelopeOperationResult,
) -> set[str]:
    codes = result.execution.metadata["diagnostic_codes"]
    return set(codes.split(",")) if codes else set()


def _source_codes(
    diagnostics: Sequence[ContainerMappedDiagnostic],
) -> set[str]:
    return {
        diagnostic.source_code
        for diagnostic in diagnostics
        if diagnostic.source_code is not None
    }


if __name__ == "__main__":
    main()
