from asyncio import run as run_async
from asyncio import sleep
from collections.abc import Mapping
from unittest import TestCase, main

from avalan.container import (
    ContainerAuditCorrelation,
    ContainerAuthorityCaps,
    ContainerBackend,
    ContainerBackendCapabilities,
    ContainerBackendContainer,
    ContainerBackendDiagnosticCode,
    ContainerBackendOperation,
    ContainerBuildPolicy,
    ContainerCommandMode,
    ContainerCommandPlan,
    ContainerDurablePlanMetadata,
    ContainerEffectiveSettings,
    ContainerEnvironmentPolicy,
    ContainerExecutionResult,
    ContainerExecutionScope,
    ContainerFakeBackend,
    ContainerFakeBackendScript,
    ContainerImagePolicy,
    ContainerLifecycleCleanup,
    ContainerLifecycleDeadlines,
    ContainerLifecyclePhase,
    ContainerMountType,
    ContainerNetworkMode,
    ContainerNetworkPolicy,
    ContainerNormalizedRunPlan,
    ContainerNormalizedRuntimeEnvelopePlan,
    ContainerOutputContract,
    ContainerOutputContractType,
    ContainerPlanRequest,
    ContainerPlanRequestKind,
    ContainerProfile,
    ContainerPullPolicy,
    ContainerResultStatus,
    ContainerRunPlan,
    ContainerRuntimeEnvelope,
    ContainerRuntimeEnvelopeBackend,
    ContainerRuntimeEnvelopeCleanupResult,
    ContainerRuntimeEnvelopeHandle,
    ContainerRuntimeEnvelopeHandoff,
    ContainerRuntimeEnvelopeHealth,
    ContainerRuntimeEnvelopeKind,
    ContainerRuntimeEnvelopeOperation,
    ContainerRuntimeEnvelopeReadiness,
    ContainerRuntimeEnvelopeShutdownResult,
    ContainerRuntimeEnvelopeState,
    ContainerSettings,
    ContainerSettingsSource,
    ContainerStableDiagnosticCode,
    ContainerSurface,
    ContainerTrustLevel,
    normalize_container_run_plan,
    normalize_runtime_envelope_plan,
    run_container_managed_lifecycle,
)

_DIGEST = "b" * 64
_IMAGE = f"ghcr.io/example/watchdog-tools@sha256:{_DIGEST}"
_WATCHDOG_SECONDS = 0.001
_SLOW_SECONDS = 0.05
_TIMEOUT_CODE = ContainerStableDiagnosticCode.TIMEOUT.value


class ContainerWatchdogConformanceTest(TestCase):
    def test_managed_lifecycle_watchdogs_bound_runtime_phases(self) -> None:
        cases = (
            (
                ContainerBackendOperation.IMAGE_RESOLUTION,
                ContainerLifecyclePhase.IMAGE_RESOLUTION,
                "image_resolution_seconds",
                _run_plan(),
                None,
            ),
            (
                ContainerBackendOperation.IMAGE_PULL,
                ContainerLifecyclePhase.IMAGE_PULL,
                "pull_seconds",
                _run_plan(pull_policy=ContainerPullPolicy.IF_MISSING),
                None,
            ),
            (
                ContainerBackendOperation.IMAGE_BUILD,
                ContainerLifecyclePhase.IMAGE_BUILD,
                "build_seconds",
                _run_plan(build_policy=ContainerBuildPolicy.TRUSTED_ONLY),
                None,
            ),
            (
                ContainerBackendOperation.CREATE,
                ContainerLifecyclePhase.CREATE,
                "create_seconds",
                _run_plan(),
                None,
            ),
            (
                ContainerBackendOperation.ATTACH,
                ContainerLifecyclePhase.ATTACH,
                "start_seconds",
                _run_plan(),
                None,
            ),
            (
                ContainerBackendOperation.START,
                ContainerLifecyclePhase.START,
                "start_seconds",
                _run_plan(),
                None,
            ),
            (
                ContainerBackendOperation.STREAM,
                ContainerLifecyclePhase.STREAM,
                "execution_seconds",
                _run_plan(),
                None,
            ),
            (
                ContainerBackendOperation.STATS,
                ContainerLifecyclePhase.STATS,
                "stats_seconds",
                _run_plan(),
                None,
            ),
            (
                ContainerBackendOperation.WAIT,
                ContainerLifecyclePhase.WAIT,
                "execution_seconds",
                _run_plan(),
                None,
            ),
            (
                ContainerBackendOperation.INSPECT,
                ContainerLifecyclePhase.INSPECT,
                "inspect_seconds",
                _run_plan(),
                None,
            ),
            (
                ContainerBackendOperation.COPY_OUTPUTS,
                ContainerLifecyclePhase.COPY_OUTPUTS,
                "copy_seconds",
                _run_plan(),
                _output_contract(),
            ),
        )

        for operation, phase, deadline_field, plan, contract in cases:
            with self.subTest(operation=operation.value):
                backend = ContainerFakeBackend(
                    ContainerFakeBackendScript(
                        capabilities=_capabilities(),
                        operation_delay_seconds={
                            operation: _SLOW_SECONDS,
                        },
                    )
                )

                result = run_async(
                    run_container_managed_lifecycle(
                        backend,
                        plan,
                        output_contract=contract,
                        deadlines=ContainerLifecycleDeadlines(
                            **{deadline_field: _WATCHDOG_SECONDS}
                        ),
                    )
                )

                self.assertEqual(
                    result.execution.status,
                    ContainerResultStatus.FAILED,
                )
                self.assertEqual(result.timed_out_phase, phase)
                self.assertIn(
                    ContainerBackendDiagnosticCode.TIMEOUT,
                    {diagnostic.code for diagnostic in result.diagnostics},
                )
                self.assertIn(
                    operation,
                    {
                        diagnostic.operation
                        for diagnostic in result.diagnostics
                    },
                )

    def test_cleanup_watchdogs_report_uncertainty(self) -> None:
        container = ContainerBackendContainer(
            container_id="cleanup-1",
            backend=ContainerBackend.DOCKER,
            plan_fingerprint="cleanup-plan",
        )
        cases = (
            ContainerBackendOperation.STOP,
            ContainerBackendOperation.KILL,
            ContainerBackendOperation.REMOVE,
            ContainerBackendOperation.CLEANUP,
        )

        for operation in cases:
            with self.subTest(operation=operation.value):
                backend = ContainerFakeBackend(
                    ContainerFakeBackendScript(
                        capabilities=_capabilities(),
                        operation_delay_seconds={
                            operation: _SLOW_SECONDS,
                        },
                    )
                )

                result = run_async(
                    ContainerLifecycleCleanup().cleanup(
                        backend,
                        container,
                        deadlines=ContainerLifecycleDeadlines(
                            cleanup_seconds=_WATCHDOG_SECONDS,
                        ),
                        force_kill=operation
                        in {
                            ContainerBackendOperation.STOP,
                            ContainerBackendOperation.KILL,
                        },
                    )
                )

                self.assertTrue(result.cleanup_uncertain)
                self.assertIn(
                    ContainerBackendDiagnosticCode.TIMEOUT,
                    {diagnostic.code for diagnostic in result.diagnostics},
                )
                self.assertIn(
                    operation,
                    {
                        diagnostic.operation
                        for diagnostic in result.diagnostics
                    },
                )

    def test_runtime_envelope_watchdogs_cover_deployed_surfaces(self) -> None:
        cases = (
            (
                "server_handler_start",
                ContainerRuntimeEnvelopeKind.SERVER,
                ContainerPlanRequestKind.SERVER,
                ContainerRuntimeEnvelopeOperation.START,
            ),
            (
                "task_worker_execution",
                ContainerRuntimeEnvelopeKind.TASK_WORKER,
                ContainerPlanRequestKind.RUNTIME_ENVELOPE,
                ContainerRuntimeEnvelopeOperation.SCOPED_EXECUTION,
            ),
            (
                "flow_health_before_execution",
                ContainerRuntimeEnvelopeKind.FLOW_RUNTIME,
                ContainerPlanRequestKind.RUNTIME_ENVELOPE,
                ContainerRuntimeEnvelopeOperation.HEALTH,
            ),
            (
                "model_backend_stream_telemetry",
                ContainerRuntimeEnvelopeKind.MODEL_BACKEND,
                ContainerPlanRequestKind.RUNTIME_ENVELOPE,
                ContainerRuntimeEnvelopeOperation.TELEMETRY,
            ),
        )

        for name, envelope_kind, request_kind, operation in cases:
            with self.subTest(name=name):
                backend = _SlowRuntimeEnvelopeBackend(operation)
                envelope = ContainerRuntimeEnvelope(
                    backend,
                    _envelope_plan(
                        envelope_kind=envelope_kind,
                        request_kind=request_kind,
                    ),
                    correlation=_correlation(),
                )

                if operation is ContainerRuntimeEnvelopeOperation.START:
                    result = run_async(envelope.start())

                    self.assertEqual(
                        result.execution.status,
                        ContainerResultStatus.FAILED,
                    )
                    self.assertIsNone(result.handle)
                    self.assertEqual(
                        _diagnostic_codes(result.execution),
                        {_TIMEOUT_CODE},
                    )
                else:
                    start = run_async(envelope.start())
                    self.assertEqual(
                        start.execution.status,
                        ContainerResultStatus.COMPLETED,
                    )

                    result = run_async(envelope.execute(_command_plan()))

                    self.assertEqual(
                        result.execution.status,
                        ContainerResultStatus.FAILED,
                    )
                    self.assertEqual(
                        _diagnostic_codes(result.execution),
                        {_TIMEOUT_CODE},
                    )
                    self.assertEqual(
                        result.audit_records[0].metadata["operation"],
                        operation.value,
                    )
                self.assertIn(operation, backend.operations)

    def test_runtime_envelope_direct_watchdogs_report_diagnostics(
        self,
    ) -> None:
        direct_cases = (
            (
                ContainerRuntimeEnvelopeOperation.READINESS,
                "start",
            ),
            (
                ContainerRuntimeEnvelopeOperation.HEALTH,
                "health",
            ),
            (
                ContainerRuntimeEnvelopeOperation.TELEMETRY,
                "telemetry",
            ),
            (
                ContainerRuntimeEnvelopeOperation.HANDOFF,
                "handoff",
            ),
            (
                ContainerRuntimeEnvelopeOperation.SHUTDOWN,
                "shutdown",
            ),
            (
                ContainerRuntimeEnvelopeOperation.CLEANUP,
                "cleanup",
            ),
        )

        for operation, method_name in direct_cases:
            with self.subTest(operation=operation.value):
                backend = _SlowRuntimeEnvelopeBackend(operation)
                envelope = ContainerRuntimeEnvelope(
                    backend,
                    _envelope_plan(),
                    correlation=_correlation(),
                )

                if method_name == "start":
                    result = run_async(envelope.start())
                    execution = result.execution
                else:
                    start = run_async(envelope.start())
                    self.assertEqual(
                        start.execution.status,
                        ContainerResultStatus.COMPLETED,
                    )
                    if method_name == "health":
                        health = run_async(envelope.health())
                        self.assertFalse(health.ok)
                        self.assertEqual(
                            {
                                diagnostic.code.value
                                for diagnostic in health.diagnostics
                            },
                            {_TIMEOUT_CODE},
                        )
                        continue
                    if method_name == "telemetry":
                        telemetry = run_async(envelope.telemetry())
                        self.assertEqual(
                            telemetry["operation"],
                            ContainerRuntimeEnvelopeOperation.TELEMETRY.value,
                        )
                        self.assertEqual(
                            telemetry["status"],
                            ContainerResultStatus.FAILED.value,
                        )
                        self.assertEqual(
                            telemetry["diagnostic_codes"],
                            _TIMEOUT_CODE,
                        )
                        self.assertEqual(
                            telemetry["diagnostic_source_codes"],
                            "container.runtime_envelope.telemetry_timeout",
                        )
                        self.assertEqual(
                            telemetry["diagnostic_retryable"],
                            "true",
                        )
                        self.assertIn(
                            "runtime envelope telemetry timed out",
                            telemetry["diagnostic_messages"],
                        )
                        self.assertNotIn(
                            "_RuntimeEnvelopeOperationTimeout",
                            telemetry["diagnostic_messages"],
                        )
                        self.assertNotIn(
                            "TimeoutError",
                            telemetry["diagnostic_messages"],
                        )
                        self.assertIn(operation, backend.operations)
                        continue
                    if method_name == "handoff":
                        result = run_async(envelope.handoff())
                    elif method_name == "shutdown":
                        result = run_async(
                            envelope.shutdown(
                                timeout_seconds=_WATCHDOG_SECONDS,
                            )
                        )
                    else:
                        result = run_async(envelope.cleanup())
                    execution = result.execution

                self.assertEqual(
                    execution.status,
                    ContainerResultStatus.FAILED,
                )
                self.assertEqual(
                    _diagnostic_codes(execution),
                    {_TIMEOUT_CODE},
                )

    def test_runtime_envelope_failed_start_cleanup_is_bounded(self) -> None:
        backend = _SlowRuntimeEnvelopeBackend(
            (
                ContainerRuntimeEnvelopeOperation.READINESS,
                ContainerRuntimeEnvelopeOperation.CLEANUP,
            )
        )
        envelope = ContainerRuntimeEnvelope(
            backend,
            _envelope_plan(),
            correlation=_correlation(),
        )

        result = run_async(envelope.start())

        self.assertEqual(
            result.execution.status,
            ContainerResultStatus.FAILED,
        )
        self.assertIsNotNone(result.cleanup)
        assert result.cleanup is not None
        self.assertTrue(result.cleanup.cleanup_uncertain)
        self.assertEqual(
            {
                diagnostic.code.value
                for diagnostic in result.cleanup.diagnostics
            },
            {_TIMEOUT_CODE},
        )
        self.assertIn(
            ContainerRuntimeEnvelopeOperation.CLEANUP,
            backend.operations,
        )


class _SlowRuntimeEnvelopeBackend(ContainerRuntimeEnvelopeBackend):
    def __init__(
        self,
        slow_operation: (
            ContainerRuntimeEnvelopeOperation
            | tuple[ContainerRuntimeEnvelopeOperation, ...]
        ),
    ) -> None:
        self._slow_operations = (
            slow_operation
            if isinstance(slow_operation, tuple)
            else (slow_operation,)
        )
        self._operations: list[ContainerRuntimeEnvelopeOperation] = []
        self._executions = 0

    @property
    def operations(self) -> tuple[ContainerRuntimeEnvelopeOperation, ...]:
        return tuple(self._operations)

    async def start(
        self,
        plan: ContainerNormalizedRuntimeEnvelopePlan,
    ) -> ContainerRuntimeEnvelopeHandle:
        await self._enter(ContainerRuntimeEnvelopeOperation.START)
        return ContainerRuntimeEnvelopeHandle(
            envelope_id="slow-runtime-1",
            envelope_kind=plan.envelope_kind,
            backend=plan.run_plan.run_plan.backend,
            plan_fingerprint=plan.plan_fingerprint,
            state=ContainerRuntimeEnvelopeState.STARTING,
        )

    async def readiness(
        self,
        handle: ContainerRuntimeEnvelopeHandle,
    ) -> ContainerRuntimeEnvelopeReadiness:
        assert isinstance(handle, ContainerRuntimeEnvelopeHandle)
        await self._enter(ContainerRuntimeEnvelopeOperation.READINESS)
        return ContainerRuntimeEnvelopeReadiness(ready=True)

    async def execute(
        self,
        handle: ContainerRuntimeEnvelopeHandle,
        plan: ContainerNormalizedRunPlan,
    ) -> ContainerExecutionResult:
        assert isinstance(handle, ContainerRuntimeEnvelopeHandle)
        assert isinstance(plan, ContainerNormalizedRunPlan)
        await self._enter(ContainerRuntimeEnvelopeOperation.SCOPED_EXECUTION)
        self._executions += 1
        return ContainerExecutionResult(
            status=ContainerResultStatus.COMPLETED,
            exit_code=0,
            metadata={"executions": str(self._executions)},
        )

    async def health(
        self,
        handle: ContainerRuntimeEnvelopeHandle,
    ) -> ContainerRuntimeEnvelopeHealth:
        assert isinstance(handle, ContainerRuntimeEnvelopeHandle)
        await self._enter(ContainerRuntimeEnvelopeOperation.HEALTH)
        return ContainerRuntimeEnvelopeHealth(healthy=True)

    async def telemetry(
        self,
        handle: ContainerRuntimeEnvelopeHandle,
    ) -> Mapping[str, str]:
        assert isinstance(handle, ContainerRuntimeEnvelopeHandle)
        await self._enter(ContainerRuntimeEnvelopeOperation.TELEMETRY)
        return {"executions": str(self._executions)}

    async def handoff(
        self,
        handle: ContainerRuntimeEnvelopeHandle,
        metadata: ContainerDurablePlanMetadata,
    ) -> ContainerRuntimeEnvelopeHandoff:
        assert isinstance(handle, ContainerRuntimeEnvelopeHandle)
        await self._enter(ContainerRuntimeEnvelopeOperation.HANDOFF)
        return ContainerRuntimeEnvelopeHandoff(
            metadata=metadata,
            telemetry={"executions": str(self._executions)},
            state={"envelope_id": handle.envelope_id},
        )

    async def shutdown(
        self,
        handle: ContainerRuntimeEnvelopeHandle,
    ) -> ContainerRuntimeEnvelopeShutdownResult:
        assert isinstance(handle, ContainerRuntimeEnvelopeHandle)
        await self._enter(ContainerRuntimeEnvelopeOperation.SHUTDOWN)
        return ContainerRuntimeEnvelopeShutdownResult(graceful=True)

    async def cleanup(
        self,
        handle: ContainerRuntimeEnvelopeHandle,
    ) -> ContainerRuntimeEnvelopeCleanupResult:
        assert isinstance(handle, ContainerRuntimeEnvelopeHandle)
        await self._enter(ContainerRuntimeEnvelopeOperation.CLEANUP)
        return ContainerRuntimeEnvelopeCleanupResult(cleanup_completed=True)

    async def _enter(
        self,
        operation: ContainerRuntimeEnvelopeOperation,
    ) -> None:
        self._operations.append(operation)
        if operation in self._slow_operations:
            await sleep(_SLOW_SECONDS)


def _capabilities() -> ContainerBackendCapabilities:
    return ContainerBackendCapabilities(
        backend=ContainerBackend.DOCKER,
        host_os="linux",
        guest_os="linux",
        architecture="amd64",
        rootless=True,
        build=True,
        pull=True,
        network_modes=(ContainerNetworkMode.NONE,),
        mount_types=(ContainerMountType.WORKSPACE,),
        device_classes=(),
        resource_limits=True,
        streaming_attach=True,
        stats=True,
    )


def _run_plan(
    *,
    pull_policy: ContainerPullPolicy = ContainerPullPolicy.NEVER,
    build_policy: ContainerBuildPolicy = ContainerBuildPolicy.DISABLED,
) -> ContainerRunPlan:
    return ContainerRunPlan(
        backend=ContainerBackend.DOCKER,
        profile_name="watchdog-profile",
        image=ContainerImagePolicy(
            reference=_IMAGE,
            pull_policy=pull_policy,
            build_policy=build_policy,
        ),
        command=ContainerCommandPlan(
            tool_name="shell.echo",
            command="echo",
            argv=("echo", "ok"),
            cwd="/workspace",
            scope=ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
        ),
        policy_version="phase20",
    )


def _output_contract() -> ContainerOutputContract:
    return ContainerOutputContract(
        contract_type=ContainerOutputContractType.TASK_ARTIFACT,
        max_bytes=64,
    )


def _envelope_plan(
    *,
    envelope_kind: ContainerRuntimeEnvelopeKind = (
        ContainerRuntimeEnvelopeKind.FLOW_RUNTIME
    ),
    request_kind: ContainerPlanRequestKind = (
        ContainerPlanRequestKind.RUNTIME_ENVELOPE
    ),
) -> ContainerNormalizedRuntimeEnvelopePlan:
    return normalize_runtime_envelope_plan(
        _effective_settings(),
        _request(
            request_kind=request_kind,
            scope=ContainerExecutionScope.RUNTIME_ENVELOPE,
        ),
        envelope_kind=envelope_kind,
        readiness_timeout_seconds=_WATCHDOG_SECONDS,
    )


def _command_plan() -> ContainerNormalizedRunPlan:
    return normalize_container_run_plan(
        _effective_settings(),
        _request(),
    )


def _request(
    *,
    request_kind: ContainerPlanRequestKind = (
        ContainerPlanRequestKind.TYPED_TOOL
    ),
    scope: ContainerExecutionScope = (
        ContainerExecutionScope.SHELL_CONTAINER_EXECUTION
    ),
) -> ContainerPlanRequest:
    return ContainerPlanRequest(
        request_kind=request_kind,
        logical_name="watchdog.command",
        command="python",
        argv=("python", "-c", "print('ok')"),
        cwd="/workspace",
        scope=scope,
    )


def _effective_settings() -> ContainerEffectiveSettings:
    profile = _profile()
    settings = ContainerSettings(
        source=ContainerSettingsSource(
            surface=ContainerSurface.RUNTIME_ENVELOPE,
            trust_level=ContainerTrustLevel.TRUSTED_OPERATOR,
        ),
        backend=ContainerBackend.DOCKER,
        default_profile=profile.name,
        allowed_profiles=(profile.name,),
        profiles={profile.name: profile},
        profile_registry_id="watchdog-registry",
        policy_version="phase20",
    )
    return ContainerAuthorityCaps(settings=settings).merge(())


def _profile() -> ContainerProfile:
    return ContainerProfile(
        name="watchdog-profile",
        image=ContainerImagePolicy(
            reference=_IMAGE,
            build_policy=ContainerBuildPolicy.DISABLED,
        ),
        command_mode=ContainerCommandMode.FIXED_EXECUTABLE,
        environment=ContainerEnvironmentPolicy(),
        network=ContainerNetworkPolicy(),
    )


def _correlation() -> ContainerAuditCorrelation:
    return ContainerAuditCorrelation(
        profile_name="watchdog-profile",
        policy_version="phase20",
        flow_node_id="watchdog-node",
        image_digest=f"sha256:{_DIGEST}",
    )


def _diagnostic_codes(execution: ContainerExecutionResult) -> set[str]:
    codes = execution.metadata["diagnostic_codes"]
    return set(codes.split(",")) if codes else set()


if __name__ == "__main__":
    main()
