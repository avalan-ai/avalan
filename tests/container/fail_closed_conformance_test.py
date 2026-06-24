from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase, main

from fastapi import HTTPException
from pydantic import ValidationError

from avalan.container import (
    CONFORMANCE_PLAN,
    ContainerAuditCorrelation,
    ContainerAuthorizationDecisionType,
    ContainerBackend,
    ContainerBackendCapabilities,
    ContainerBackendDiagnosticCode,
    ContainerBackendProbeResult,
    ContainerCommandPlan,
    ContainerDeviceClass,
    ContainerDevicePolicy,
    ContainerDiagnostic,
    ContainerDiagnosticCategory,
    ContainerDiagnosticCode,
    ContainerEffectiveSettings,
    ContainerEscalationMode,
    ContainerEscalationPolicy,
    ContainerEscalationTrigger,
    ContainerExecutionScope,
    ContainerExecutionSettings,
    ContainerFormattedOutput,
    ContainerImagePolicy,
    ContainerMountType,
    ContainerNetworkMode,
    ContainerNetworkPolicy,
    ContainerPolicy,
    ContainerPolicyContext,
    ContainerPolicyPlan,
    ContainerProcessSecurityPlan,
    ContainerProfile,
    ContainerProfileSelection,
    ContainerResourceLimits,
    ContainerResultStatus,
    ContainerReviewSurface,
    ContainerRunPlan,
    ContainerSettings,
    ContainerSettingsOverride,
    ContainerSettingsSource,
    ContainerStableDiagnosticCode,
    ContainerSurface,
    ContainerToolRuntimeSettings,
    ContainerTrustLevel,
    assert_container_syntax_supported,
    container_execution_result_from_diagnostics,
    container_syntax_diagnostics,
    format_container_diagnostics_for_model,
    normalize_container_diagnostic,
    resolve_container_backend,
    select_container_backend,
    validate_container_process_security,
)
from avalan.server.container_policy import (
    RemoteContainerRequestError,
    RemoteContainerRequestPolicy,
    server_runtime_envelope_status_from_runtime_settings,
    validate_remote_container_arguments,
)
from avalan.server.entities import ChatCompletionRequest, ResponsesRequest
from avalan.server.remote_container import (
    validate_remote_container_profile_selection,
)

_DIGEST = "9" * 64
_IMAGE = f"ghcr.io/example/fail-closed@sha256:{_DIGEST}"


class ContainerFailClosedConformanceTest(TestCase):
    def test_unsupported_surface_paths_have_actionable_diagnostics(
        self,
    ) -> None:
        for (
            surface,
            paths,
        ) in CONFORMANCE_PLAN.unsupported_surface_paths.items():
            for path in paths:
                with self.subTest(surface=surface.value, path=path):
                    raw = _raw_with_path(path, _container_policy_value(path))

                    diagnostics = container_syntax_diagnostics(surface, raw)

                    self.assertEqual(len(diagnostics), 1)
                    diagnostic = diagnostics[0]
                    self.assertEqual(
                        diagnostic.code,
                        ContainerDiagnosticCode.UNSUPPORTED_SYNTAX,
                    )
                    self.assertEqual(
                        diagnostic.category,
                        ContainerDiagnosticCategory.UNSUPPORTED,
                    )
                    self.assertEqual(
                        diagnostic.path,
                        path.replace("*", "selected"),
                    )
                    _assert_actionable_conformance_diagnostic(
                        self,
                        diagnostic,
                    )
                    formatted = _format_diagnostics((diagnostic,))
                    self.assertIn(
                        ContainerStableDiagnosticCode.UNSUPPORTED_SYNTAX.value,
                        formatted.text,
                    )
                    with self.assertRaisesRegex(
                        AssertionError,
                        diagnostic.path,
                    ):
                        assert_container_syntax_supported(surface, raw)

    def test_required_backends_fail_closed_without_host_fallback(
        self,
    ) -> None:
        cases = (
            (
                "no selected backend",
                resolve_container_backend(
                    ContainerExecutionSettings(required=True)
                ),
                ContainerDiagnosticCode.BACKEND_REQUIRED,
                ContainerStableDiagnosticCode.BACKEND_REQUIRED,
            ),
            (
                "unavailable selected backend",
                resolve_container_backend(
                    ContainerExecutionSettings(
                        backend=ContainerBackend.DOCKER,
                        required=True,
                        profile="workspace-readonly",
                    ),
                    available_backends=(),
                ),
                ContainerDiagnosticCode.BACKEND_UNAVAILABLE,
                ContainerStableDiagnosticCode.CONFORMANCE_BACKEND_UNAVAILABLE,
            ),
        )

        for name, resolution, expected_code, stable_code in cases:
            with self.subTest(name=name):
                self.assertFalse(resolution.ok)
                self.assertIsNone(resolution.backend)
                self.assertFalse(resolution.direct_execution_allowed)
                self.assertEqual(len(resolution.diagnostics), 1)
                diagnostic = resolution.diagnostics[0]

                self.assertEqual(diagnostic.code, expected_code)
                _assert_actionable_conformance_diagnostic(self, diagnostic)
                mapped = normalize_container_diagnostic(diagnostic)
                self.assertEqual(mapped.code, stable_code)
                result = container_execution_result_from_diagnostics(
                    (diagnostic,),
                    _correlation(),
                )
                self.assertEqual(result.status, ContainerResultStatus.DENIED)
                self.assertIn(
                    stable_code.value,
                    result.metadata["diagnostic_codes"],
                )

    def test_backend_capability_mismatches_deny_selection(
        self,
    ) -> None:
        cases = (
            (
                "network",
                _run_plan(
                    network=ContainerNetworkPolicy(
                        mode=ContainerNetworkMode.ALLOWLIST,
                        egress_allowlist=("api.example.test",),
                    )
                ),
                _capabilities(network_modes=(ContainerNetworkMode.NONE,)),
                "network mode allowlist is not supported",
            ),
            (
                "device",
                _run_plan(
                    devices=ContainerDevicePolicy(
                        devices=(ContainerDeviceClass.NVIDIA_CDI,),
                    )
                ),
                _capabilities(device_classes=(ContainerDeviceClass.CPU,)),
                "device class nvidia_cdi is not supported",
            ),
            (
                "resources",
                _run_plan(
                    resources=ContainerResourceLimits(memory_bytes=1024)
                ),
                _capabilities(resource_limits=False),
                "resource limits are not supported",
            ),
        )

        for name, plan, capabilities, message in cases:
            with self.subTest(name=name):
                selection = select_container_backend(
                    plan,
                    (
                        ContainerBackendProbeResult(
                            backend=ContainerBackend.DOCKER,
                            available=True,
                            capabilities=capabilities,
                        ),
                    ),
                    auto_enabled=False,
                )

                self.assertFalse(selection.ok)
                self.assertIsNone(selection.backend)
                self.assertIn(
                    ContainerBackendDiagnosticCode.CAPABILITY_MISMATCH,
                    {diagnostic.code for diagnostic in selection.diagnostics},
                )
                self.assertIn(
                    message,
                    {
                        diagnostic.message
                        for diagnostic in selection.diagnostics
                    },
                )
                mapped = tuple(
                    normalize_container_diagnostic(diagnostic)
                    for diagnostic in selection.diagnostics
                )
                self.assertIn(
                    ContainerStableDiagnosticCode.CAPABILITY_MISMATCH,
                    {diagnostic.code for diagnostic in mapped},
                )
                formatted = _format_diagnostics(selection.diagnostics)
                self.assertIn(
                    ContainerStableDiagnosticCode.CAPABILITY_MISMATCH.value,
                    formatted.text,
                )
                self.assertIn(message, formatted.text)

    def test_fail_closed_review_surfaces_map_to_policy_denial(self) -> None:
        for surface in (
            ContainerReviewSurface.DIRECT_TASK,
            ContainerReviewSurface.SERVER,
            ContainerReviewSurface.MCP,
            ContainerReviewSurface.A2A,
        ):
            with self.subTest(surface=surface.value):
                plan = _policy_plan(surface)

                decision = ContainerPolicy(
                    policy_version="phase20",
                ).authorize(plan)

                self.assertEqual(
                    decision.decision,
                    ContainerAuthorizationDecisionType.DENY,
                )
                self.assertEqual(
                    decision.code,
                    "container.deny.review_unavailable",
                )
                self.assertFalse(decision.retryable)
                self.assertFalse(decision.cacheable)
                mapped = normalize_container_diagnostic(decision)
                self.assertEqual(
                    mapped.code,
                    ContainerStableDiagnosticCode.POLICY_DENIED,
                )
                formatted = _format_diagnostics((decision,))
                self.assertIn(
                    ContainerStableDiagnosticCode.POLICY_DENIED.value,
                    formatted.text,
                )
                self.assertIn(
                    "cannot perform required review",
                    formatted.text,
                )

    def test_untrusted_settings_authority_attempts_are_rejected(
        self,
    ) -> None:
        untrusted_agent = ContainerSettingsSource(
            surface=ContainerSurface.AGENT_TOML,
            trust_level=ContainerTrustLevel.UNTRUSTED_AGENT,
        )
        model_source = ContainerSettingsSource(
            surface=ContainerSurface.SERVER,
            trust_level=ContainerTrustLevel.MODEL,
        )

        with self.assertRaisesRegex(
            AssertionError,
            "untrusted sources cannot define container runtime authority",
        ):
            ContainerSettings.from_dict(
                {
                    "backend": "docker",
                    "profiles": {
                        "workspace-readonly": _readonly_profile().to_dict()
                    },
                },
                source=untrusted_agent,
            )
        with self.assertRaisesRegex(
            AssertionError,
            "untrusted sources can only select or narrow profiles",
        ):
            ContainerSettingsOverride.from_dict(
                {"backend": "docker"},
                source=untrusted_agent,
            )
        with self.assertRaisesRegex(
            AssertionError,
            "model output cannot select container runtime profiles",
        ):
            ContainerProfileSelection.from_dict(
                {"profile": "workspace-readonly"},
                source=model_source,
            )

    def test_model_visible_runtime_authority_attempts_are_rejected(
        self,
    ) -> None:
        policy = RemoteContainerRequestPolicy(
            exposed_profiles=("workspace-readonly",)
        )
        allowed = validate_remote_container_arguments(
            {
                "prompt": "hi",
                "containerProfile": "workspace-readonly",
            },
            policy=policy,
        )

        self.assertEqual(allowed.profile, "workspace-readonly")
        self.assertEqual(allowed.arguments, {"prompt": "hi"})
        for arguments, expected in (
            (
                {
                    "messages": [
                        {
                            "metadata": {
                                "runtimeEnvelope": {
                                    "profile": "workspace-readonly"
                                }
                            }
                        }
                    ]
                },
                "runtimeEnvelope",
            ),
            (
                {
                    "container": {
                        "profile": "workspace-readonly",
                        "image": "registry.example/untrusted:latest",
                    }
                },
                "can only select a profile",
            ),
            (
                {"containerProfile": "private-profile"},
                'profile "private-profile" is not exposed',
            ),
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(
                    RemoteContainerRequestError,
                    expected,
                ):
                    validate_remote_container_arguments(
                        arguments,
                        policy=policy,
                    )

    def test_server_models_reject_unsupported_container_policy(
        self,
    ) -> None:
        cases = (
            (
                ChatCompletionRequest,
                {
                    "model": "m",
                    "messages": [{"role": "user", "content": "hi"}],
                    "container": {
                        "profiles": {
                            "unsafe": {
                                "image": "registry.example/untrusted:latest"
                            }
                        }
                    },
                },
            ),
            (
                ResponsesRequest,
                {
                    "input": "hi",
                    "container": {
                        "profiles": {
                            "unsafe": {
                                "image": "registry.example/untrusted:latest"
                            }
                        }
                    },
                },
            ),
        )

        for request_type, payload in cases:
            with self.subTest(request_type=request_type.__name__):
                with self.assertRaisesRegex(
                    ValidationError, "runtime authority"
                ):
                    request_type.model_validate(payload)

    def test_unsafe_profile_and_process_security_rejections_are_actionable(
        self,
    ) -> None:
        for kwargs, expected in (
            ({"read_only_rootfs": False}, "read-only rootfs"),
            ({"user": "0:0"}, "root user is unsafe"),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(AssertionError, expected):
                    ContainerProfile(
                        name="unsafe-profile",
                        image=ContainerImagePolicy(reference=_IMAGE),
                        **kwargs,
                    )
        for plan, expected in (
            (
                ContainerProcessSecurityPlan(user="0:0"),
                "root user is denied",
            ),
            (
                ContainerProcessSecurityPlan(
                    user="1000:1000",
                    privileged=True,
                ),
                "privileged containers are denied",
            ),
            (
                ContainerProcessSecurityPlan(
                    user="1000:1000",
                    capabilities=("NET_ADMIN",),
                ),
                "extra capabilities are denied",
            ),
        ):
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(AssertionError, expected):
                    validate_container_process_security(plan)

    def test_runtime_envelope_unavailability_has_actionable_status(
        self,
    ) -> None:
        status = server_runtime_envelope_status_from_runtime_settings(
            _runtime_settings(ContainerExecutionScope.RUNTIME_ENVELOPE),
            server_name="api",
            request_id="server-1",
        )

        self.assertFalse(status.ok)
        self.assertIsNotNone(status.plan)
        self.assertEqual(len(status.diagnostics), 1)
        diagnostic = status.diagnostics[0]
        self.assertEqual(
            diagnostic["code"],
            "server.runtime_envelope_unavailable",
        )
        self.assertEqual(diagnostic["path"], "runtime.container.envelope")
        self.assertIn("not available", diagnostic["message"])
        self.assertIn("trusted envelope-aware", diagnostic["hint"])


class RemoteContainerDependencyFailClosedTest(IsolatedAsyncioTestCase):
    async def test_rejects_malformed_container_envelope(self) -> None:
        payload = {
            "container": {
                "profile": "workspace-readonly",
                "image": "registry.example/untrusted:latest",
            },
        }
        request = _Request(payload)
        request.app.state.remote_container_policy = (
            RemoteContainerRequestPolicy(
                exposed_profiles=("workspace-readonly",)
            )
        )

        with self.assertRaises(HTTPException) as exc:
            await validate_remote_container_profile_selection(request)

        self.assertEqual(exc.exception.status_code, 400)
        self.assertIn("can only select a profile", str(exc.exception.detail))
        self.assertFalse(hasattr(request.state, "remote_container_profile"))


class _Request:
    def __init__(self, payload: object) -> None:
        self._payload = payload
        self.app = SimpleNamespace(state=SimpleNamespace())
        self.state = SimpleNamespace()

    async def json(self) -> object:
        return self._payload


def _assert_actionable_conformance_diagnostic(
    test: TestCase,
    diagnostic: ContainerDiagnostic,
) -> None:
    serialized = diagnostic.as_dict()
    for key in ("code", "path", "category", "message", "hint"):
        test.assertIsInstance(serialized[key], str)
        test.assertTrue(serialized[key])


def _capabilities(
    *,
    backend: ContainerBackend = ContainerBackend.DOCKER,
    network_modes: tuple[ContainerNetworkMode, ...] = (
        ContainerNetworkMode.NONE,
        ContainerNetworkMode.ALLOWLIST,
    ),
    device_classes: tuple[ContainerDeviceClass, ...] = (
        ContainerDeviceClass.CPU,
    ),
    resource_limits: bool = True,
) -> ContainerBackendCapabilities:
    return ContainerBackendCapabilities(
        backend=backend,
        host_os="linux",
        guest_os="linux",
        architecture="amd64",
        platform_emulation=True,
        rootless=True,
        pull=True,
        network_modes=network_modes,
        mount_types=(ContainerMountType.WORKSPACE,),
        device_classes=device_classes,
        resource_limits=resource_limits,
        streaming_attach=True,
        stats=True,
    )


def _container_policy_value(path: str) -> object:
    if path == "tool.shell.backend":
        return "container"
    return {"profile": "workspace-readonly"}


def _correlation() -> ContainerAuditCorrelation:
    return ContainerAuditCorrelation(
        profile_name="workspace-readonly",
        policy_version="phase20",
    )


def _format_diagnostics(
    diagnostics: tuple[object, ...],
) -> ContainerFormattedOutput:
    return format_container_diagnostics_for_model(
        diagnostics,
        _correlation(),
    )


def _policy_plan(surface: ContainerReviewSurface) -> ContainerPolicyPlan:
    profile = ContainerProfile(
        name="workspace-review",
        image=ContainerImagePolicy(reference=_IMAGE),
        escalation=ContainerEscalationPolicy(
            mode=ContainerEscalationMode.REQUIRE_REVIEW,
        ),
    )
    return ContainerPolicyPlan(
        effective_settings=_effective_settings(
            profile=profile,
            source=ContainerSettingsSource(
                surface=ContainerSurface.SERVER,
                trust_level=ContainerTrustLevel.TRUSTED_OPERATOR,
            ),
            scope=ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
        ),
        context=ContainerPolicyContext(
            surface=surface,
            scope_id=f"scope.{surface.value}",
            attempt_id="attempt.1",
        ),
        command_fingerprint="shell.echo:ok",
        escalation_triggers=(ContainerEscalationTrigger.SECRET,),
    )


def _raw_with_path(path: str, value: object) -> dict[str, object]:
    raw: dict[str, object] = {}
    current = raw
    for part in path.replace("*", "selected").split(".")[:-1]:
        child: dict[str, object] = {}
        current[part] = child
        current = child
    current[path.replace("*", "selected").split(".")[-1]] = value
    return raw


def _readonly_profile() -> ContainerProfile:
    return ContainerProfile.minimal_readonly(
        name="workspace-readonly",
        image_reference=_IMAGE,
    )


def _run_plan(
    *,
    network: ContainerNetworkPolicy | None = None,
    devices: ContainerDevicePolicy | None = None,
    resources: ContainerResourceLimits | None = None,
) -> ContainerRunPlan:
    return ContainerRunPlan(
        backend=ContainerBackend.DOCKER,
        profile_name="backend-profile",
        image=ContainerImagePolicy(reference=_IMAGE),
        command=ContainerCommandPlan(
            tool_name="shell.echo",
            command="echo",
            argv=("echo", "ok"),
            cwd="/workspace",
            scope=ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
        ),
        network=network or ContainerNetworkPolicy(),
        devices=devices or ContainerDevicePolicy(),
        resources=resources or ContainerResourceLimits(),
        policy_version="phase20",
    )


def _runtime_settings(
    scope: ContainerExecutionScope,
) -> ContainerToolRuntimeSettings:
    profile = _readonly_profile()
    return ContainerToolRuntimeSettings(
        effective_settings=_effective_settings(
            profile=profile,
            source=ContainerSettingsSource(
                surface=ContainerSurface.SERVER,
                trust_level=ContainerTrustLevel.TRUSTED_OPERATOR,
            ),
            scope=scope,
        )
    )


def _effective_settings(
    *,
    profile: ContainerProfile,
    source: ContainerSettingsSource,
    scope: ContainerExecutionScope,
) -> ContainerEffectiveSettings:
    return ContainerEffectiveSettings(
        backend=ContainerBackend.DOCKER,
        required=True,
        scope=scope,
        source=source,
        policy_version="phase20",
        profile_registry_id="fail-closed",
        profile_name=profile.name,
        profile=profile,
        allowed_profiles=(profile.name,),
    )


if __name__ == "__main__":
    main()
