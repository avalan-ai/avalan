from asyncio import run as run_async
from typing import cast
from unittest import TestCase, main

from avalan.container import (
    ContainerAuditCorrelation,
    ContainerAuditEventType,
    ContainerAuditRecord,
    ContainerAuthorizationDecision,
    ContainerAuthorizationDecisionType,
    ContainerBackend,
    ContainerBackendCapabilities,
    ContainerBackendDiagnostic,
    ContainerBackendDiagnosticCode,
    ContainerBackendOperation,
    ContainerBackendStats,
    ContainerBackendStream,
    ContainerBackendStreamChunk,
    ContainerBuildPolicy,
    ContainerCommandPlan,
    ContainerDeviceClass,
    ContainerDiagnostic,
    ContainerDiagnosticCategory,
    ContainerDiagnosticCode,
    ContainerExecutionScope,
    ContainerFakeBackend,
    ContainerFakeBackendScript,
    ContainerFormattedOutput,
    ContainerImagePolicy,
    ContainerMappedDiagnostic,
    ContainerMountType,
    ContainerNetworkMode,
    ContainerOutputContract,
    ContainerOutputContractType,
    ContainerOutputDecisionType,
    ContainerOutputDiagnostic,
    ContainerOutputDiagnosticCode,
    ContainerOutputValidationResult,
    ContainerPullPolicy,
    ContainerResultStatus,
    ContainerRunPlan,
    ContainerStableDiagnosticCode,
    container_diagnostic_audit_event,
    container_execution_result_from_diagnostics,
    container_lifecycle_audit_events,
    format_container_diagnostics_for_model,
    normalize_container_diagnostic,
    redact_container_audit_value,
    run_container_managed_lifecycle,
    sanitize_container_audit_metadata,
)

_DIGEST = "8" * 64
_IMAGE = f"ghcr.io/example/audit-tools@sha256:{_DIGEST}"


class ContainerAuditTest(TestCase):
    def test_successful_lifecycle_emits_privacy_safe_event_sequence(
        self,
    ) -> None:
        contract = _output_contract()
        backend = ContainerFakeBackend(
            ContainerFakeBackendScript(
                capabilities=_capabilities(pull=True, build=True),
                stream_chunks=(
                    ContainerBackendStreamChunk(
                        stream=ContainerBackendStream.STDOUT,
                        content=b"stdout secret bytes",
                        sequence=0,
                    ),
                    ContainerBackendStreamChunk(
                        stream=ContainerBackendStream.STDERR,
                        content=b"stderr private bytes",
                        sequence=1,
                    ),
                    ContainerBackendStreamChunk(
                        stream=ContainerBackendStream.PROGRESS,
                        content=b"progress",
                        sequence=2,
                    ),
                ),
                stats_samples=(
                    ContainerBackendStats(
                        cpu_nanos=10,
                        memory_bytes=2048,
                        pids=3,
                    ),
                ),
                output_result=ContainerOutputValidationResult(
                    decision=ContainerOutputDecisionType.ACCEPT,
                    contract=contract,
                ),
            )
        )
        result = run_async(
            run_container_managed_lifecycle(
                backend,
                _run_plan(
                    pull_policy=ContainerPullPolicy.IF_MISSING,
                    build_policy=ContainerBuildPolicy.TRUSTED_ONLY,
                ),
                output_contract=contract,
            )
        )

        records = container_lifecycle_audit_events(
            result,
            _correlation(),
            scope=ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
            include_review=True,
        )
        event_types = {record.event_type for record in records}
        serialized = str([record.to_dict() for record in records])
        legacy = records[0].to_event().to_dict()
        legacy_metadata = cast(dict[str, str], legacy["metadata"])

        self.assertEqual(
            result.execution.status,
            ContainerResultStatus.COMPLETED,
        )
        self.assertTrue(
            {
                ContainerAuditEventType.POLICY_EVALUATION,
                ContainerAuditEventType.REVIEW_REQUEST,
                ContainerAuditEventType.REVIEW_DECISION,
                ContainerAuditEventType.BACKEND_SELECTION,
                ContainerAuditEventType.IMAGE_RESOLUTION,
                ContainerAuditEventType.IMAGE_PULL,
                ContainerAuditEventType.BUILD_PROGRESS,
                ContainerAuditEventType.MOUNT_PREPARATION,
                ContainerAuditEventType.CONTAINER_CREATE,
                ContainerAuditEventType.CONTAINER_START,
                ContainerAuditEventType.STDOUT_CHUNK,
                ContainerAuditEventType.STDERR_CHUNK,
                ContainerAuditEventType.PROGRESS,
                ContainerAuditEventType.STATS,
                ContainerAuditEventType.EXIT,
                ContainerAuditEventType.OUTPUT_COPY,
                ContainerAuditEventType.CLEANUP,
                ContainerAuditEventType.RESULT_RECORDED,
            }.issubset(event_types)
        )
        self.assertEqual(legacy_metadata["agent_id"], "agent-1")
        self.assertNotIn("stdout secret bytes", serialized)
        self.assertNotIn("stderr private bytes", serialized)
        self.assertIn("<redacted-stream>", serialized)

    def test_metadata_redaction_removes_sensitive_values(self) -> None:
        metadata = sanitize_container_audit_metadata(
            {
                "safe": "visible",
                "stdout": "raw stdout should not leak",
                "api_token": "super-secret-token",
                "provider": "Bearer sk-test-token",
                "prompt": "private prompt text",
                "path": "/Users/mariano/.ssh/id_rsa",
                "bytes": b"\x00raw-file-bytes",
                "long": "x" * 300,
            }
        )
        record = ContainerAuditRecord(
            event_type="failure",
            scope="shell_container_execution",
            correlation=_correlation(),
            metadata=metadata,
        )
        text = str(record.to_dict()) + str(record.to_event().to_dict())

        self.assertEqual(metadata["safe"], "visible")
        self.assertEqual(metadata["stdout"], "<redacted-stream>")
        self.assertEqual(metadata["api_token"], "<redacted>")
        self.assertEqual(metadata["provider"], "<redacted>")
        self.assertEqual(metadata["prompt"], "<redacted>")
        self.assertEqual(metadata["bytes"], "<redacted-bytes>")
        self.assertIn("<host-path>/id_rsa", metadata["path"])
        self.assertTrue(metadata["long"].endswith("...<truncated>"))
        self.assertNotIn("super-secret-token", text)
        self.assertNotIn("raw stdout should not leak", text)
        self.assertNotIn("/Users/mariano", text)
        self.assertNotIn("private prompt text", text)
        self.assertEqual(
            redact_container_audit_value("message", "plain value"),
            "plain value",
        )
        self.assertEqual(
            redact_container_audit_value("message", "raw\x00bytes"),
            "<redacted-bytes>",
        )
        with self.assertRaises(AssertionError):
            sanitize_container_audit_metadata({"bad/key": "value"})

    def test_diagnostics_map_to_results_and_formatted_output_safely(
        self,
    ) -> None:
        diagnostic = ContainerBackendDiagnostic(
            code=ContainerBackendDiagnosticCode.PULL_FAILED,
            operation=ContainerBackendOperation.IMAGE_PULL,
            message=(
                "pull failed for /Users/mariano/.docker/config.json "
                "with Bearer sk-provider-token"
            ),
            retryable=True,
        )

        mapped = normalize_container_diagnostic(diagnostic)
        result = container_execution_result_from_diagnostics(
            (diagnostic,),
            _correlation(),
            exit_code=12,
        )
        formatted = format_container_diagnostics_for_model(
            (diagnostic,),
            _correlation(),
        )
        event = container_diagnostic_audit_event(
            diagnostic,
            _correlation(),
            scope=ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
        )
        serialized = (
            str(mapped.to_dict())
            + str(result.to_dict())
            + str(formatted.to_dict())
            + str(event.to_dict())
        )

        self.assertEqual(
            mapped.code,
            ContainerStableDiagnosticCode.PULL_FAILED,
        )
        self.assertEqual(result.status, ContainerResultStatus.FAILED)
        self.assertEqual(result.exit_code, 12)
        self.assertIn(
            ContainerStableDiagnosticCode.PULL_FAILED.value,
            result.metadata["diagnostic_codes"],
        )
        self.assertIn("container status: failed", formatted.text)
        self.assertEqual(event.event_type, ContainerAuditEventType.FAILURE)
        self.assertNotIn("/Users/mariano", serialized)
        self.assertNotIn("sk-provider-token", serialized)

    def test_negative_outputs_do_not_leak_sensitive_content(self) -> None:
        diagnostics = (
            ContainerOutputDiagnostic(
                code=ContainerOutputDiagnosticCode.TOO_LARGE,
                path="artifact.txt",
                message=(
                    "raw bytes abc\x00def from /private/var/root/secret "
                    "prompt=private prompt content"
                ),
            ),
            "stderr contained provider credential Bearer sk-live-token",
        )

        result = container_execution_result_from_diagnostics(
            diagnostics,
            _correlation(),
        )
        formatted = format_container_diagnostics_for_model(
            diagnostics,
            _correlation(),
        )
        serialized = str(result.to_dict()) + str(formatted.to_dict())

        self.assertEqual(result.status, ContainerResultStatus.DENIED)
        self.assertNotIn("abc", serialized)
        self.assertNotIn("/private/var/root", serialized)
        self.assertNotIn("private prompt content", serialized)
        self.assertNotIn("sk-live-token", serialized)

    def test_output_shaped_diagnostics_do_not_leak_payloads(self) -> None:
        messages = (
            "stdout: customer-private-stdout-line",
            "stderr=customer-private-stderr-line",
            "log: customer-private-log-line",
            "output=customer-private-output-line",
            "content: customer-private-content-line",
            "stderr contained customer-private-stderr-line",
            "stdout customer-private-stdout-line",
            "container log customer-private-log-line",
        )

        for message in messages:
            with self.subTest(message=message):
                diagnostic = ContainerBackendDiagnostic(
                    code=ContainerBackendDiagnosticCode.WAIT_FAILED,
                    operation=ContainerBackendOperation.WAIT,
                    message=message,
                )
                raw_result = container_execution_result_from_diagnostics(
                    (message,),
                    _correlation(),
                )
                structured_result = (
                    container_execution_result_from_diagnostics(
                        (diagnostic,),
                        _correlation(),
                    )
                )
                formatted = format_container_diagnostics_for_model(
                    (diagnostic, message),
                    _correlation(),
                )
                serialized = (
                    str(raw_result.to_dict())
                    + str(structured_result.to_dict())
                    + str(formatted.to_dict())
                )

                self.assertIn(
                    ContainerStableDiagnosticCode.WAIT_FAILED.value,
                    structured_result.metadata["diagnostic_codes"],
                )
                self.assertIn("<redacted-stream>", serialized)
                self.assertNotIn(message, serialized)
                self.assertNotIn("customer-private", serialized)

    def test_fake_e2e_all_stable_diagnostics_map_consistently(self) -> None:
        correlation = _correlation()
        for code in ContainerStableDiagnosticCode:
            with self.subTest(code=code.value):
                mapped = normalize_container_diagnostic(code)
                result = container_execution_result_from_diagnostics(
                    (code,),
                    correlation,
                )
                event = container_diagnostic_audit_event(
                    code,
                    correlation,
                    scope=ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
                )
                formatted = format_container_diagnostics_for_model(
                    (code,),
                    correlation,
                )

                self.assertIn(code.value, result.metadata["diagnostic_codes"])
                self.assertIn(code.value, event.to_event().metadata["code"])
                self.assertIn(
                    code.value,
                    formatted.metadata["diagnostic_codes"],
                )
                self.assertIn(code.value, mapped.result_message())

    def test_existing_diagnostic_vocabulary_normalizes(self) -> None:
        for code in ContainerBackendDiagnosticCode:
            with self.subTest(backend_code=code.value):
                mapped = normalize_container_diagnostic(
                    ContainerBackendDiagnostic(
                        code=code,
                        operation=ContainerBackendOperation.PROBE,
                        message=f"{code.value} failed",
                    )
                )
                self.assertEqual(mapped.source_code, code.value)
        for output_code in ContainerOutputDiagnosticCode:
            with self.subTest(output_code=output_code.value):
                mapped = normalize_container_diagnostic(
                    ContainerOutputDiagnostic(
                        code=output_code,
                        path="artifact",
                        message=f"{output_code.value} failed",
                    )
                )
                self.assertEqual(mapped.source_code, output_code.value)

        conformance = normalize_container_diagnostic(
            ContainerDiagnostic(
                code=ContainerDiagnosticCode.UNSUPPORTED_SYNTAX,
                path="agent.container",
                message="unsupported syntax",
                hint="remove container section",
                category=ContainerDiagnosticCategory.UNSUPPORTED,
            )
        )
        review = normalize_container_diagnostic(
            ContainerAuthorizationDecision(
                decision=ContainerAuthorizationDecisionType.REQUIRES_REVIEW,
                code="requires-review",
                explanation="review is required",
                policy_version="phase9",
            )
        )
        denied = normalize_container_diagnostic(
            ContainerAuthorizationDecision(
                decision=ContainerAuthorizationDecisionType.DENY,
                code="policy-denied",
                explanation="denied",
                policy_version="phase9",
            )
        )
        allowed = normalize_container_diagnostic(
            ContainerAuthorizationDecision(
                decision=ContainerAuthorizationDecisionType.ALLOW,
                code="allowed",
                explanation="allowed",
                policy_version="phase9",
            )
        )
        mapped = ContainerMappedDiagnostic(
            code=ContainerStableDiagnosticCode.UNKNOWN,
            message="already mapped",
            status=ContainerResultStatus.FAILED,
        )

        self.assertEqual(
            conformance.code,
            ContainerStableDiagnosticCode.UNSUPPORTED_SYNTAX,
        )
        self.assertEqual(
            review.code,
            ContainerStableDiagnosticCode.REVIEW_REQUIRED,
        )
        self.assertEqual(
            denied.code,
            ContainerStableDiagnosticCode.POLICY_DENIED,
        )
        self.assertEqual(allowed.code, ContainerStableDiagnosticCode.UNKNOWN)
        self.assertIs(normalize_container_diagnostic(mapped), mapped)
        self.assertEqual(
            container_execution_result_from_diagnostics(
                (),
                _correlation(),
            ).status,
            ContainerResultStatus.COMPLETED,
        )
        with self.assertRaises(AssertionError):
            normalize_container_diagnostic(object())

    def test_value_objects_validate_inputs(self) -> None:
        formatted = ContainerFormattedOutput(
            text="container status: completed",
            metadata={"status": "completed"},
        )
        compact_events = container_lifecycle_audit_events(
            run_async(
                run_container_managed_lifecycle(
                    ContainerFakeBackend(
                        ContainerFakeBackendScript(
                            capabilities=_capabilities()
                        )
                    ),
                    _run_plan(),
                )
            ),
            _correlation(),
            scope="shell_container_execution",
            include_mount_preparation=False,
        )
        timeout_events = container_lifecycle_audit_events(
            run_async(
                run_container_managed_lifecycle(
                    ContainerFakeBackend(
                        ContainerFakeBackendScript(
                            capabilities=_capabilities(),
                            timeout_operations=(
                                ContainerBackendOperation.WAIT,
                            ),
                        )
                    ),
                    _run_plan(),
                )
            ),
            _correlation(),
            scope="shell_container_execution",
        )
        cancellation_events = container_lifecycle_audit_events(
            run_async(
                run_container_managed_lifecycle(
                    ContainerFakeBackend(
                        ContainerFakeBackendScript(
                            capabilities=_capabilities(),
                            cancel_operations=(
                                ContainerBackendOperation.WAIT,
                            ),
                        )
                    ),
                    _run_plan(),
                )
            ),
            _correlation(),
            scope="shell_container_execution",
        )
        denied_events = container_lifecycle_audit_events(
            run_async(
                run_container_managed_lifecycle(
                    ContainerFakeBackend(
                        ContainerFakeBackendScript(
                            capabilities=_capabilities(),
                            soft_operation_diagnostics={
                                ContainerBackendOperation.IMAGE_RESOLUTION: (
                                    ContainerBackendDiagnosticCode.IMAGE_DENIED
                                ),
                            },
                        )
                    ),
                    _run_plan(),
                )
            ),
            _correlation(),
            scope="shell_container_execution",
        )
        failed_events = container_lifecycle_audit_events(
            run_async(
                run_container_managed_lifecycle(
                    ContainerFakeBackend(
                        ContainerFakeBackendScript(
                            capabilities=_capabilities(),
                            wait_exit_code=1,
                        )
                    ),
                    _run_plan(),
                )
            ),
            _correlation(),
            scope="shell_container_execution",
        )
        empty_formatted = format_container_diagnostics_for_model(
            (),
            _correlation(),
        )

        formatted_metadata = cast(
            dict[str, str],
            formatted.to_dict()["metadata"],
        )

        self.assertEqual(formatted_metadata["status"], "completed")
        self.assertEqual(empty_formatted.text, "container status: completed")
        self.assertNotIn(
            ContainerAuditEventType.MOUNT_PREPARATION,
            {record.event_type for record in compact_events},
        )
        self.assertIn(
            ContainerAuditEventType.TIMEOUT,
            {record.event_type for record in timeout_events},
        )
        self.assertIn(
            ContainerAuditEventType.CANCELLATION,
            {record.event_type for record in cancellation_events},
        )
        self.assertIn(
            ContainerAuditEventType.DENIAL,
            {record.event_type for record in denied_events},
        )
        self.assertIn(
            ContainerAuditEventType.FAILURE,
            {record.event_type for record in failed_events},
        )
        with self.assertRaises(AssertionError):
            ContainerAuditCorrelation(
                profile_name="bad/profile",
                policy_version="phase9",
            )
        with self.assertRaises(AssertionError):
            ContainerAuditCorrelation(
                profile_name="phase9",
                policy_version="phase9",
                image_digest="sha256:bad",
            )
        with self.assertRaises(AssertionError):
            ContainerAuditRecord(
                event_type="bad",
                scope="shell_container_execution",
                correlation=_correlation(),
            )
        with self.assertRaises(AssertionError):
            ContainerFormattedOutput(text="")


def _correlation() -> ContainerAuditCorrelation:
    return ContainerAuditCorrelation(
        agent_id="agent-1",
        session_id="session-1",
        tool_call_id="tool-1",
        flow_node_id="flow-1",
        task_run_id="task-1",
        attempt_id="attempt-1",
        profile_name="phase9",
        image_digest=f"sha256:{_DIGEST}",
        policy_version="phase9",
    )


def _capabilities(
    *,
    pull: bool = True,
    build: bool = True,
) -> ContainerBackendCapabilities:
    return ContainerBackendCapabilities(
        backend=ContainerBackend.DOCKER,
        host_os="linux",
        guest_os="linux",
        architecture="amd64",
        rootless=True,
        build=build,
        pull=pull,
        network_modes=(ContainerNetworkMode.NONE,),
        mount_types=(ContainerMountType.WORKSPACE,),
        device_classes=(ContainerDeviceClass.CPU,),
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
        profile_name="phase9",
        image=ContainerImagePolicy(
            reference=_IMAGE,
            pull_policy=pull_policy,
            build_policy=build_policy,
        ),
        command=ContainerCommandPlan(
            tool_name="shell.audit",
            command="echo",
            argv=("echo", "audit"),
            cwd="/workspace",
            scope=ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
        ),
        policy_version="phase9",
    )


def _output_contract() -> ContainerOutputContract:
    return ContainerOutputContract(
        contract_type=ContainerOutputContractType.TASK_ARTIFACT,
        max_bytes=64,
    )


if __name__ == "__main__":
    main()
