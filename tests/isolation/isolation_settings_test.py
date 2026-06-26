from typing import cast
from unittest import TestCase, main

from avalan.container import (
    ContainerBackend,
    ContainerBackendCapabilities,
    ContainerBackendDiagnostic,
    ContainerBackendDiagnosticCode,
    ContainerBackendOperation,
    ContainerFakeBackend,
    ContainerFakeBackendScript,
    ContainerMountType,
    ContainerOutputDiagnostic,
    ContainerOutputDiagnosticCode,
)
from avalan.isolation import (
    IsolationDiagnosticCategory,
    IsolationDiagnosticCode,
    IsolationDiagnosticInventoryItem,
    IsolationDiagnosticSeverity,
    IsolationEffectiveSettings,
    IsolationMappedDiagnostic,
    IsolationMode,
    IsolationProfileSelection,
    IsolationSettings,
    IsolationSettingsSource,
    IsolationSettingsSurface,
    IsolationToolRuntimeSettings,
    IsolationTrustLevel,
    LocalIsolationPolicy,
    SandboxBackend,
    SandboxEffectiveSettings,
    SandboxNetworkMode,
    SandboxProfile,
    SandboxProfileSelection,
    SandboxSettings,
    deserialize_isolation_effective_settings,
    format_isolation_diagnostics_for_model,
    format_isolation_diagnostics_output_for_model,
    isolation_diagnostic,
    isolation_diagnostic_audit_metadata,
    isolation_diagnostic_codes,
    isolation_diagnostics_metadata,
    isolation_public_diagnostics,
    isolation_selection_from_mapping,
    normalize_isolation_diagnostic,
    redact_isolation_value,
    sanitize_isolation_metadata,
    serialize_isolation_effective_settings,
    stable_isolation_diagnostic_codes,
    stable_isolation_diagnostic_inventory,
    trusted_isolation_runtime_from_mapping,
    trusted_isolation_settings_from_mapping,
    trusted_isolation_source,
)
from avalan.sandbox import (
    SandboxBackendDiagnostic,
    SandboxBackendDiagnosticCode,
    SandboxBackendOperation,
)

_DIGEST = "9" * 64
_IMAGE = f"ghcr.io/example/isolation-tools@sha256:{_DIGEST}"


class IsolationSettingsTest(TestCase):
    def test_positive_minimal_local_settings_serialize(self) -> None:
        source = trusted_isolation_source(IsolationSettingsSurface.SDK)
        settings = IsolationSettings.from_dict(
            {"mode": "local"}, source=source
        )
        self.assertEqual(
            IsolationSettings.from_dict(
                settings.to_dict(),
                source=source,
            ).to_dict(),
            settings.to_dict(),
        )

        effective = settings.select_profile(
            IsolationProfileSelection(mode="local")
        )
        self.assertEqual(effective.mode, IsolationMode.LOCAL)
        self.assertIsInstance(effective.local, LocalIsolationPolicy)
        self.assertIsNone(effective.sandbox)
        self.assertIsNone(effective.container)
        self.assertEqual(
            effective.canonical_policy_input()["mode"],
            "local",
        )

        serialized = serialize_isolation_effective_settings(effective)
        self.assertEqual(
            deserialize_isolation_effective_settings(serialized).to_dict(),
            effective.to_dict(),
        )

        def authorize(value: object) -> object:
            return value

        def audit(value: object) -> object:
            return value

        runtime = IsolationToolRuntimeSettings(
            effective_settings=effective,
            authorization_provider=authorize,
            audit_listeners=(audit,),
        )
        self.assertEqual(runtime.mode, IsolationMode.LOCAL)
        self.assertIs(runtime.local, effective.local)
        assert runtime.authorization_provider is not None
        self.assertEqual(runtime.authorization_provider("plan"), "plan")
        self.assertEqual(runtime.audit_listeners[0]("event"), "event")
        self.assertEqual(
            runtime.to_dict()["effective_settings"],
            effective.to_dict(),
        )

        with self.assertRaises(AssertionError):
            IsolationToolRuntimeSettings(
                effective_settings=effective,
                audit_listeners=(object(),),  # type: ignore[arg-type]
            )

    def test_positive_sandbox_seatbelt_settings(self) -> None:
        effective = _sandbox_effective("seatbelt")

        self.assertEqual(effective.mode, IsolationMode.SANDBOX)
        self.assertIsInstance(effective.sandbox, SandboxEffectiveSettings)
        assert effective.sandbox is not None
        self.assertEqual(effective.sandbox.backend, SandboxBackend.SEATBELT)
        self.assertTrue(effective.sandbox.required)
        self.assertEqual(effective.sandbox.profile_name, "host-tools")
        self.assertEqual(
            effective.sandbox.profile.trusted_executables,
            ("/bin/sh",),
        )
        self.assertEqual(
            effective.sandbox.profile.network.mode,
            SandboxNetworkMode.LOOPBACK,
        )

    def test_positive_sandbox_bubblewrap_settings(self) -> None:
        effective = _sandbox_effective("bubblewrap")

        self.assertIsInstance(effective.sandbox, SandboxEffectiveSettings)
        assert effective.sandbox is not None
        self.assertEqual(effective.sandbox.backend, SandboxBackend.BUBBLEWRAP)
        self.assertEqual(
            effective.sandbox.canonical_policy_input()["backend"],
            "bubblewrap",
        )

    def test_positive_container_docker_settings(self) -> None:
        effective = _container_effective("docker")

        self.assertEqual(effective.mode, IsolationMode.CONTAINER)
        self.assertIsNone(effective.local)
        self.assertIsNone(effective.sandbox)
        assert effective.container is not None
        self.assertEqual(effective.container.backend, ContainerBackend.DOCKER)
        self.assertEqual(effective.container.profile_name, "tools")

    def test_positive_container_apple_container_settings(self) -> None:
        effective = _container_effective("apple-container")

        assert effective.container is not None
        self.assertEqual(
            effective.container.backend,
            ContainerBackend.APPLE_CONTAINER,
        )
        self.assertEqual(
            effective.canonical_policy_input()["mode"],
            "container",
        )

    def test_diagnostic_inventory_is_stable(self) -> None:
        self.assertEqual(
            {code.value for code in IsolationDiagnosticCode},
            {
                "isolation.mode_conflict",
                "isolation.unsupported_mode",
                "isolation.unsupported_backend",
                "isolation.mode_unavailable",
                "isolation.capability_mismatch",
                "isolation.elevation_required",
                "isolation.elevation_denied",
                "isolation.fallback_denied",
                "isolation.approval_stale",
                "isolation.policy_drift",
                "isolation.audit_unavailable",
                "sandbox.provider_unavailable",
                "sandbox.profile_generation_failed",
                "sandbox.path_denied",
                "sandbox.network_unenforceable",
                "container.backend.unavailable",
                "container.backend.capability_mismatch",
                "isolation.unsupported_syntax",
            },
        )
        self.assertEqual(
            {
                spec.to_dict()["code"]
                for spec in stable_isolation_diagnostic_inventory()
            },
            {
                "isolation.mode_conflict",
                "isolation.unsupported_mode",
                "isolation.unsupported_backend",
                "isolation.mode_unavailable",
                "isolation.capability_mismatch",
                "isolation.elevation_required",
                "isolation.elevation_denied",
                "isolation.fallback_denied",
                "isolation.approval_stale",
                "isolation.policy_drift",
                "isolation.audit_unavailable",
                "sandbox.provider_unavailable",
                "sandbox.profile_generation_failed",
                "sandbox.path_denied",
                "sandbox.network_unenforceable",
                "container.backend.unavailable",
                "container.backend.capability_mismatch",
            },
        )
        self.assertEqual(
            stable_isolation_diagnostic_codes(),
            (
                "isolation.mode_conflict",
                "isolation.unsupported_mode",
                "isolation.unsupported_backend",
                "isolation.mode_unavailable",
                "isolation.capability_mismatch",
                "isolation.elevation_required",
                "isolation.elevation_denied",
                "isolation.fallback_denied",
                "isolation.approval_stale",
                "isolation.policy_drift",
                "isolation.audit_unavailable",
                "sandbox.provider_unavailable",
                "sandbox.profile_generation_failed",
                "sandbox.path_denied",
                "sandbox.network_unenforceable",
                "container.backend.unavailable",
                "container.backend.capability_mismatch",
            ),
        )
        for spec in stable_isolation_diagnostic_inventory():
            with self.subTest(code=spec.to_dict()["code"]):
                self.assertIn(
                    "diagnostic_codes",
                    spec.to_dict()["metadata_fields"],
                )
                self.assertIn("audit_event", spec.to_dict())
                self.assertIn("model_status", spec.to_dict())
        diagnostic = isolation_diagnostic(
            IsolationDiagnosticCode.UNSUPPORTED_BACKEND,
            path="runtime.isolation.sandbox.backend",
            message="Unsupported sandbox backend.",
            hint="Use seatbelt or bubblewrap.",
            category=IsolationDiagnosticCategory.UNSUPPORTED,
            severity=IsolationDiagnosticSeverity.ERROR,
        )
        self.assertEqual(
            diagnostic.to_dict(),
            {
                "code": "isolation.unsupported_backend",
                "path": "runtime.isolation.sandbox.backend",
                "category": "unsupported",
                "severity": "error",
                "message": "Unsupported sandbox backend.",
                "hint": "Use seatbelt or bubblewrap.",
                "retryable": False,
            },
        )
        self.assertEqual(
            diagnostic.to_audit_metadata(),
            {
                "diagnostic_code": "isolation.unsupported_backend",
                "diagnostic_category": "unsupported",
                "diagnostic_severity": "error",
                "diagnostic_retryable": "false",
            },
        )
        self.assertEqual(
            format_isolation_diagnostics_for_model((diagnostic,)),
            "isolation.unsupported_backend: Unsupported sandbox backend. "
            "Use seatbelt or bubblewrap.",
        )
        self.assertEqual(
            diagnostic.model_message(),
            "isolation.unsupported_backend: Unsupported sandbox backend. "
            "Use seatbelt or bubblewrap.",
        )

    def test_stable_diagnostics_are_public_audited_and_model_visible(
        self,
    ) -> None:
        diagnostics = tuple(
            isolation_diagnostic(
                spec.to_dict()["code"],
                path=f"runtime.isolation.{index}",
                message=f"{spec.to_dict()['code']} occurred.",
                hint="Inspect the trusted isolation policy.",
                category=spec.to_dict()["category"],
                severity=spec.to_dict()["severity"],
            )
            for index, spec in enumerate(
                stable_isolation_diagnostic_inventory()
            )
        )
        public = isolation_public_diagnostics(diagnostics)
        metadata = isolation_diagnostics_metadata(diagnostics)
        output = format_isolation_diagnostics_output_for_model(diagnostics)

        self.assertEqual(
            tuple(item["code"] for item in public),
            stable_isolation_diagnostic_codes(),
        )
        self.assertEqual(
            metadata["diagnostic_codes"],
            ",".join(stable_isolation_diagnostic_codes()),
        )
        self.assertEqual(
            output.metadata["diagnostic_count"],
            str(len(stable_isolation_diagnostic_codes())),
        )
        for diagnostic in diagnostics:
            with self.subTest(code=diagnostic.to_dict()["code"]):
                audit = isolation_diagnostic_audit_metadata(diagnostic)
                code = cast(str, diagnostic.to_dict()["code"])
                self.assertEqual(audit["code"], code)
                self.assertIn(code, output.text)
                self.assertIn(code, metadata["diagnostic_codes"])

    def test_diagnostic_public_metadata_and_model_output_are_sanitized(
        self,
    ) -> None:
        diagnostic = isolation_diagnostic(
            IsolationDiagnosticCode.SANDBOX_PATH_DENIED,
            path="/Users/mariano/.ssh/id_rsa",
            message=(
                "Denied /Users/mariano/.ssh/id_rsa with Bearer "
                "sk-provider-token"
            ),
            hint="Remove token=private-token from the request.",
            category=IsolationDiagnosticCategory.SECURITY,
        )
        public = isolation_public_diagnostics((diagnostic,))
        metadata = isolation_diagnostics_metadata(
            (diagnostic,),
            {
                "request_id": "request-1",
                "api_token": "super-secret-token",
                "path": "/Users/mariano/.ssh/id_rsa",
            },
        )
        audit = isolation_diagnostic_audit_metadata(
            diagnostic,
            {"session_id": "session-1"},
        )
        model_output = format_isolation_diagnostics_for_model((diagnostic,))
        serialized = str(public) + str(metadata) + str(audit) + model_output

        self.assertEqual(public[0]["code"], "sandbox.path_denied")
        self.assertEqual(metadata["diagnostic_codes"], "sandbox.path_denied")
        self.assertEqual(audit["audit_event"], "sandbox.filesystem")
        self.assertEqual(metadata["api_token"], "<redacted>")
        self.assertNotIn("/Users/mariano", serialized)
        self.assertNotIn("sk-provider-token", serialized)
        self.assertNotIn("private-token", serialized)
        self.assertIn("<redacted>", serialized)

    def test_backend_diagnostics_map_to_stable_isolation_codes(self) -> None:
        sandbox_timeout = SandboxBackendDiagnostic(
            code=SandboxBackendDiagnosticCode.TIMEOUT,
            operation=SandboxBackendOperation.WAIT,
            message="timed out under /tmp/avalan-run",
            backend=SandboxBackend.SEATBELT,
        )
        container_image_denied = ContainerBackendDiagnostic(
            code=ContainerBackendDiagnosticCode.IMAGE_DENIED,
            operation=ContainerBackendOperation.IMAGE_RESOLUTION,
            message="image denied",
            backend=ContainerBackend.DOCKER,
        )
        container_output = ContainerOutputDiagnostic(
            code=ContainerOutputDiagnosticCode.SYMLINK_ESCAPE,
            path="/Volumes/secrets/out",
            message="symlink escaped /Volumes/secrets/out",
        )

        public = isolation_public_diagnostics(
            (
                sandbox_timeout,
                container_image_denied,
                container_output,
            )
        )
        metadata = isolation_diagnostics_metadata(
            (
                sandbox_timeout,
                container_image_denied,
                container_output,
            )
        )
        output = format_isolation_diagnostics_output_for_model(
            (
                sandbox_timeout,
                container_image_denied,
                container_output,
            )
        )
        serialized = str(public) + str(metadata) + output.text

        self.assertEqual(
            tuple(item["code"] for item in public),
            (
                "isolation.mode_unavailable",
                "container.backend.capability_mismatch",
                "container.backend.capability_mismatch",
            ),
        )
        self.assertEqual(
            metadata["diagnostic_source_codes"],
            "sandbox.backend.timeout,"
            "container.backend.image_denied,"
            "container.output.symlink_escape",
        )
        self.assertIn("container.backend.capability_mismatch", output.text)
        self.assertNotIn("/tmp/avalan-run", serialized)
        self.assertNotIn("/Volumes/secrets", serialized)
        for code in SandboxBackendDiagnosticCode:
            with self.subTest(source_code=code.value):
                self.assertTrue(
                    isolation_public_diagnostics(
                        (
                            SandboxBackendDiagnostic(
                                code=code,
                                operation=SandboxBackendOperation.WAIT,
                                message=f"{code.value} from /tmp/sandbox",
                                backend=SandboxBackend.SEATBELT,
                            ),
                        )
                    )
                )
        for code in ContainerBackendDiagnosticCode:
            with self.subTest(source_code=code.value):
                self.assertTrue(
                    isolation_public_diagnostics(
                        (
                            ContainerBackendDiagnostic(
                                code=code,
                                operation=ContainerBackendOperation.WAIT,
                                message=f"{code.value} from /tmp/container",
                                backend=ContainerBackend.DOCKER,
                            ),
                        )
                    )
                )
        for code in ContainerOutputDiagnosticCode:
            with self.subTest(source_code=code.value):
                self.assertTrue(
                    isolation_public_diagnostics(
                        (
                            ContainerOutputDiagnostic(
                                code=code,
                                path="/tmp/output",
                                message=f"{code.value} from /tmp/output",
                            ),
                        )
                    )
                )

    def test_diagnostic_helpers_cover_defensive_paths(self) -> None:
        self.assertEqual(
            isolation_diagnostic_codes(("unknown.source",)),
            (),
        )
        self.assertEqual(
            format_isolation_diagnostics_for_model(
                (),
                metadata={"request_id": "request-1"},
            ),
            "isolation: ok",
        )
        empty_output = format_isolation_diagnostics_output_for_model(())
        self.assertEqual(
            empty_output.to_dict(),
            {
                "text": "isolation status: ok",
                "metadata": {
                    "diagnostic_count": "0",
                    "diagnostic_codes": "",
                },
            },
        )
        self.assertEqual(
            redact_isolation_value("stdout", "visible"), "<redacted-stream>"
        )
        self.assertEqual(
            redact_isolation_value("blob", b"secret"), "<redacted-bytes>"
        )
        self.assertEqual(
            redact_isolation_value("diagnostic_message", "stdout: secret"),
            "<redacted-stream>",
        )
        self.assertEqual(
            redact_isolation_value("value", "bad\x00value"),
            "<redacted-bytes>",
        )
        self.assertTrue(
            redact_isolation_value("value", "x" * 300).endswith(
                "...<truncated>"
            )
        )
        self.assertTrue(
            redact_isolation_value("text", "x" * 5000).endswith(
                "...<truncated>"
            )
        )

        with self.assertRaises(AssertionError):
            sanitize_isolation_metadata({"bad key": "value"})
        with self.assertRaises(AssertionError):
            IsolationDiagnosticInventoryItem(
                code=IsolationDiagnosticCode.MODE_CONFLICT,
                category=IsolationDiagnosticCategory.VALUE,
                severity=IsolationDiagnosticSeverity.ERROR,
                message="message",
                hint="hint",
                audit_event="audit",
                model_status="denied",
                metadata_fields="diagnostic_codes",  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            IsolationDiagnosticInventoryItem(
                code=IsolationDiagnosticCode.MODE_CONFLICT,
                category=IsolationDiagnosticCategory.VALUE,
                severity=IsolationDiagnosticSeverity.ERROR,
                message="message",
                hint="hint",
                audit_event="audit",
                model_status="denied",
                metadata_fields=(1,),  # type: ignore[list-item]
            )
        with self.assertRaises(AssertionError):
            IsolationDiagnosticInventoryItem(
                code=IsolationDiagnosticCode.MODE_CONFLICT,
                category=IsolationDiagnosticCategory.VALUE,
                severity=IsolationDiagnosticSeverity.ERROR,
                message="message",
                hint="hint",
                audit_event="audit",
                model_status="denied",
                metadata_fields=("bad key",),
            )
        with self.assertRaises(AssertionError):
            IsolationMappedDiagnostic(
                code="not-a-code",
                category=IsolationDiagnosticCategory.VALUE,
                severity=IsolationDiagnosticSeverity.ERROR,
                message="message",
                hint="hint",
                audit_event="audit",
                model_status="denied",
            )
        with self.assertRaises(AssertionError):
            IsolationMappedDiagnostic(
                code=IsolationDiagnosticCode.MODE_CONFLICT,
                category=1,  # type: ignore[arg-type]
                severity=IsolationDiagnosticSeverity.ERROR,
                message="message",
                hint="hint",
                audit_event="audit",
                model_status="denied",
            )
        with self.assertRaises(AssertionError):
            normalize_isolation_diagnostic(object())
        with self.assertRaises(AssertionError):
            normalize_isolation_diagnostic(_DiagnosticStub(code=""))

        mapped = normalize_isolation_diagnostic(
            _DiagnosticStub(
                code=IsolationDiagnosticCode.MODE_CONFLICT,
                operation="custom",
                message="message",
            )
        )
        self.assertEqual(mapped.source_code, None)
        self.assertEqual(
            normalize_isolation_diagnostic(
                IsolationDiagnosticCode.MODE_CONFLICT
            ).code,
            IsolationDiagnosticCode.MODE_CONFLICT,
        )
        self.assertEqual(
            isolation_diagnostic_audit_metadata(
                ContainerBackendDiagnostic(
                    code=ContainerBackendDiagnosticCode.IMAGE_DENIED,
                    operation=ContainerBackendOperation.IMAGE_RESOLUTION,
                    message="image denied",
                    backend=ContainerBackend.DOCKER,
                )
            )["source_code"],
            "container.backend.image_denied",
        )
        self.assertEqual(
            normalize_isolation_diagnostic(
                _DiagnosticStub(code="isolation.unsupported_syntax")
            ).code,
            IsolationDiagnosticCode.UNSUPPORTED_MODE,
        )
        self.assertEqual(
            normalize_isolation_diagnostic(
                _DiagnosticStub(code="isolation.policy_widening")
            ).code,
            IsolationDiagnosticCode.POLICY_DRIFT,
        )
        self.assertEqual(
            normalize_isolation_diagnostic(
                _DiagnosticStub(code="isolation.review.local")
            ).code,
            IsolationDiagnosticCode.ELEVATION_REQUIRED,
        )
        self.assertEqual(
            normalize_isolation_diagnostic(
                _DiagnosticStub(code="isolation.deny.local")
            ).code,
            IsolationDiagnosticCode.ELEVATION_DENIED,
        )
        self.assertEqual(
            normalize_isolation_diagnostic(
                _DiagnosticStub(code="isolation.approval.stale")
            ).code,
            IsolationDiagnosticCode.APPROVAL_STALE,
        )
        self.assertEqual(
            normalize_isolation_diagnostic(
                _DiagnosticStub(
                    code="sandbox.backend.capability_mismatch",
                    message="network mode unsupported",
                )
            ).code,
            IsolationDiagnosticCode.SANDBOX_NETWORK_UNENFORCEABLE,
        )
        self.assertEqual(
            normalize_isolation_diagnostic(
                _DiagnosticStub(
                    code="sandbox.backend.execution_failed",
                    operation="prepare_profile",
                )
            ).code,
            IsolationDiagnosticCode.SANDBOX_PROFILE_GENERATION_FAILED,
        )
        self.assertEqual(
            normalize_isolation_diagnostic(
                _DiagnosticStub(code="container.backend_required")
            ).code,
            IsolationDiagnosticCode.UNSUPPORTED_BACKEND,
        )
        self.assertEqual(
            normalize_isolation_diagnostic(
                _DiagnosticStub(code="container.backend_unavailable")
            ).code,
            IsolationDiagnosticCode.CONTAINER_BACKEND_UNAVAILABLE,
        )
        self.assertEqual(
            normalize_isolation_diagnostic(
                _DiagnosticStub(code="container.unsupported_syntax")
            ).code,
            IsolationDiagnosticCode.UNSUPPORTED_MODE,
        )

        self.assertIn(
            "isolation status: unavailable",
            format_isolation_diagnostics_output_for_model(
                (
                    isolation_diagnostic(
                        IsolationDiagnosticCode.MODE_UNAVAILABLE,
                        path="runtime.isolation",
                        message="unavailable",
                        hint="retry later",
                        category=IsolationDiagnosticCategory.AVAILABILITY,
                    ),
                )
            ).text,
        )
        self.assertIn(
            "isolation status: requires_review",
            format_isolation_diagnostics_output_for_model(
                (
                    isolation_diagnostic(
                        IsolationDiagnosticCode.ELEVATION_REQUIRED,
                        path="runtime.isolation",
                        message="review required",
                        hint="request approval",
                        category=IsolationDiagnosticCategory.APPROVAL,
                    ),
                )
            ).text,
        )
        self.assertIn(
            "isolation status: informational",
            format_isolation_diagnostics_output_for_model(
                (
                    IsolationMappedDiagnostic(
                        code=IsolationDiagnosticCode.MODE_CONFLICT,
                        category=IsolationDiagnosticCategory.VALUE,
                        severity=IsolationDiagnosticSeverity.INFO,
                        message="message",
                        hint="hint",
                        audit_event="audit",
                        model_status="informational",
                    ),
                )
            ).text,
        )

    def test_negative_multiple_active_modes_are_rejected(self) -> None:
        source = trusted_isolation_source("sdk")

        with self.assertRaises(AssertionError):
            IsolationEffectiveSettings(
                mode="sandbox",
                source=source,
                local=LocalIsolationPolicy(),
                sandbox=_sandbox_effective("seatbelt").sandbox,
            )

    def test_negative_mixed_sandbox_and_container_fields_are_rejected(
        self,
    ) -> None:
        source = trusted_isolation_source("sdk")

        with self.assertRaises(AssertionError):
            IsolationSettings.from_dict(
                {
                    "mode": "sandbox",
                    "sandbox": _sandbox_settings_raw("seatbelt"),
                    "container": _container_settings_raw("docker"),
                },
                source=source,
            )

    def test_negative_image_fields_under_sandbox_are_rejected(self) -> None:
        raw = _sandbox_profile_raw()
        raw["image"] = _IMAGE

        with self.assertRaises(AssertionError):
            SandboxProfile.from_dict(raw)

    def test_negative_sandbox_roots_under_container_are_rejected(self) -> None:
        source = trusted_isolation_source("cli")
        raw = _container_settings_raw("docker")
        profiles = cast(dict[str, object], raw["profiles"])
        profile = profiles["tools"]
        assert isinstance(profile, dict)
        profile["read_roots"] = ["/tmp"]

        with self.assertRaises(AssertionError):
            IsolationSettings.from_dict(
                {"mode": "container", "container": raw},
                source=source,
            )

    def test_negative_local_marked_isolated_is_rejected(self) -> None:
        with self.assertRaises(AssertionError):
            LocalIsolationPolicy(isolated=True)

        with self.assertRaises(AssertionError):
            IsolationSettings.from_dict(
                {"mode": "local", "local": {"isolated": True}},
                source=trusted_isolation_source("sdk"),
            )

    def test_negative_untrusted_authority_definitions_are_rejected(
        self,
    ) -> None:
        source = IsolationSettingsSource(
            surface=IsolationSettingsSurface.TASK_TOML,
            trust_level=IsolationTrustLevel.UNTRUSTED_TASK,
        )

        with self.assertRaises(AssertionError):
            SandboxSettings.from_dict(
                _sandbox_settings_raw("seatbelt"),
                source=source,
            )

        with self.assertRaises(AssertionError):
            IsolationSettings.from_dict(
                {
                    "mode": "sandbox",
                    "sandbox": _sandbox_settings_raw("seatbelt"),
                },
                source=source,
            )

    def test_negative_unsupported_mode_backend_and_syntax(self) -> None:
        source = trusted_isolation_source("sdk")

        with self.assertRaises(AssertionError):
            IsolationSettings.from_dict({"mode": "jail"}, source=source)

        with self.assertRaises(AssertionError):
            SandboxSettings.from_dict(
                _sandbox_settings_raw("firejail"),
                source=source,
            )

        with self.assertRaises(AssertionError):
            isolation_selection_from_mapping(
                {"profile": "host-tools", "backend": "seatbelt"},
                source=source,
            )

    def test_untrusted_selection_can_only_narrow_profiles(self) -> None:
        trusted_source = trusted_isolation_source("sdk")
        untrusted_source = IsolationSettingsSource(
            surface="task_toml",
            trust_level="untrusted_task",
        )
        settings = trusted_isolation_settings_from_mapping(
            {
                "mode": "sandbox",
                "sandbox": _sandbox_settings_raw("seatbelt"),
            },
            source=trusted_source,
        )
        selection = isolation_selection_from_mapping(
            {"mode": "sandbox", "profile": "host-tools"},
            source=untrusted_source,
        )

        effective = settings.select_profile(selection)
        self.assertEqual(effective.mode, IsolationMode.SANDBOX)

        with self.assertRaises(AssertionError):
            settings.select_profile(
                IsolationProfileSelection(mode="container", profile="tools")
            )
        with self.assertRaises(AssertionError):
            settings.select_profile(
                IsolationProfileSelection(
                    mode="sandbox",
                    profile="missing",
                )
            )

    def test_untrusted_selection_cannot_widen_container_scope(self) -> None:
        untrusted_source = IsolationSettingsSource(
            surface="task_toml",
            trust_level="untrusted_task",
        )

        with self.assertRaises(AssertionError):
            isolation_selection_from_mapping(
                {
                    "mode": "container",
                    "profile": "tools",
                    "scope": "runtime_envelope",
                },
                source=untrusted_source,
            )

    def test_fake_e2e_equivalent_trusted_sources_normalize_equally(
        self,
    ) -> None:
        raw = {
            "mode": "sandbox",
            "sandbox": _sandbox_settings_raw("seatbelt"),
        }
        selection_raw = {
            "mode": "sandbox",
            "profile": "host-tools",
            "required": True,
        }
        sources = (
            trusted_isolation_source("sdk"),
            trusted_isolation_source("cli"),
            IsolationSettingsSource(
                surface="agent_toml",
                trust_level="trusted_deployment",
            ),
        )

        canonical_inputs = []
        for source in sources:
            runtime = trusted_isolation_runtime_from_mapping(
                raw,
                source=source,
                selection=isolation_selection_from_mapping(
                    selection_raw,
                    source=source,
                ),
            )
            canonical_inputs.append(
                runtime.effective_settings.canonical_policy_input()
            )

        self.assertEqual(canonical_inputs[0], canonical_inputs[1])
        self.assertEqual(canonical_inputs[1], canonical_inputs[2])

    def test_serialization_is_deterministic_and_validation_is_linear_shape(
        self,
    ) -> None:
        source = trusted_isolation_source("sdk")
        profiles = {
            f"profile-{index:02d}": _sandbox_profile_raw(
                f"profile-{index:02d}",
                executable=f"/bin/tool-{index:02d}",
            )
            for index in range(64)
        }
        raw = {
            "mode": "sandbox",
            "sandbox": {
                "backend": "seatbelt",
                "default_profile": "profile-63",
                "allowed_profiles": list(reversed(tuple(profiles))),
                "profiles": profiles,
            },
        }

        settings = IsolationSettings.from_dict(raw, source=source)
        effective = settings.select_profile(
            IsolationProfileSelection(mode="sandbox", required=True)
        )

        self.assertEqual(effective.to_json(), effective.to_json())
        assert effective.sandbox is not None
        self.assertEqual(effective.sandbox.profile_name, "profile-63")
        self.assertEqual(len(effective.sandbox.allowed_profiles), 64)
        self.assertEqual(
            IsolationEffectiveSettings.from_json(
                effective.to_json()
            ).to_dict(),
            effective.to_dict(),
        )

    def test_selection_serialization_helpers_reject_model_source(self) -> None:
        source = trusted_isolation_source("sdk")
        sandbox_selection = SandboxProfileSelection.from_dict(
            {"profile": "host-tools", "required": True},
            source=source,
        )
        self.assertEqual(
            sandbox_selection.to_dict(),
            {"profile": "host-tools", "required": True},
        )

        selection = IsolationProfileSelection.from_dict(
            {"profile": "host-tools"},
            source=source,
        )
        self.assertEqual(
            selection.to_dict(),
            {
                "mode": None,
                "profile": "host-tools",
                "required": False,
                "scope": "shell_container_execution",
            },
        )

        model_source = IsolationSettingsSource(
            surface="mcp",
            trust_level="model",
        )
        with self.assertRaises(AssertionError):
            SandboxProfileSelection.from_dict(
                {"profile": "host-tools"},
                source=model_source,
            )
        with self.assertRaises(AssertionError):
            IsolationProfileSelection.from_dict(
                {"profile": "host-tools"},
                source=model_source,
            )

    def test_runtime_branch_accessors_and_canonical_json(self) -> None:
        sandbox_effective = _sandbox_effective("seatbelt")
        sandbox_backend = _FakeSandboxBackend()
        sandbox_runtime = IsolationToolRuntimeSettings(
            effective_settings=sandbox_effective,
            sandbox_backend=sandbox_backend,
        )

        self.assertIs(sandbox_runtime.sandbox, sandbox_effective.sandbox)
        self.assertIs(sandbox_runtime.sandbox_backend, sandbox_backend)
        self.assertIsNone(sandbox_runtime.local)
        self.assertIsNone(sandbox_runtime.container)
        self.assertEqual(
            sandbox_effective.canonical_json(),
            sandbox_effective.canonical_json(),
        )

        container_effective = _container_effective("docker")
        container_backend = _fake_container_backend()

        def resolve_secret(name: str) -> object:
            return f"resolved:{name}"

        container_runtime = IsolationToolRuntimeSettings(
            effective_settings=container_effective,
            container_backend=container_backend,
            secret_resolver=resolve_secret,
        )

        self.assertIs(
            container_runtime.container, container_effective.container
        )
        self.assertIs(container_runtime.container_backend, container_backend)
        assert container_runtime.secret_resolver is not None
        self.assertEqual(
            container_runtime.secret_resolver("TOKEN"),
            "resolved:TOKEN",
        )
        self.assertIsNone(container_runtime.local)
        self.assertIsNone(container_runtime.sandbox)

        with self.assertRaises(AssertionError):
            IsolationToolRuntimeSettings(
                effective_settings=sandbox_effective,
                container_backend=container_backend,
            )
        with self.assertRaises(AssertionError):
            IsolationToolRuntimeSettings(
                effective_settings=container_effective,
                sandbox_backend=sandbox_backend,
            )

    def test_settings_to_dict_and_optional_sequence_defaults(self) -> None:
        source = trusted_isolation_source("sdk")
        settings = SandboxSettings.from_dict(
            _sandbox_settings_raw("seatbelt"),
            source=source,
        )
        self.assertEqual(settings.to_dict()["backend"], "seatbelt")

        local = LocalIsolationPolicy.from_dict(
            {
                "allowed_roots": None,
                "executable_allowlist": None,
            }
        )
        self.assertEqual(local.allowed_roots, ())
        self.assertEqual(local.executable_allowlist, ())

    def test_effective_from_dict_rejects_missing_branches(self) -> None:
        effective = _container_effective("docker")

        self.assertEqual(
            IsolationEffectiveSettings.from_dict(
                {
                    "mode": "container",
                    "source": effective.source.to_dict(),
                    "local": None,
                    "sandbox": None,
                    "container": (
                        effective.container.to_dict()
                        if effective.container
                        else None
                    ),
                }
            ).to_dict(),
            effective.to_dict(),
        )

        with self.assertRaises(AssertionError):
            IsolationEffectiveSettings.from_dict(
                {
                    "mode": "local",
                    "source": trusted_isolation_source("sdk").to_dict(),
                    "local": None,
                    "sandbox": None,
                    "container": None,
                }
            )

    def test_select_profile_rejects_unsupported_mutated_mode(self) -> None:
        settings = IsolationSettings.from_dict(
            {"mode": "local"},
            source=trusted_isolation_source("sdk"),
        )
        object.__setattr__(settings, "mode", "jail")

        with self.assertRaisesRegex(AssertionError, "unsupported"):
            settings.select_profile(IsolationProfileSelection())


def _sandbox_effective(backend: str) -> IsolationEffectiveSettings:
    source = trusted_isolation_source("sdk")
    settings = IsolationSettings.from_dict(
        {
            "mode": "sandbox",
            "sandbox": _sandbox_settings_raw(backend),
        },
        source=source,
    )
    return settings.select_profile(
        IsolationProfileSelection(
            mode="sandbox",
            profile="host-tools",
            required=True,
        )
    )


def _container_effective(backend: str) -> IsolationEffectiveSettings:
    source = trusted_isolation_source("cli")
    settings = IsolationSettings.from_dict(
        {
            "mode": "container",
            "container": _container_settings_raw(backend),
        },
        source=source,
    )
    return settings.select_profile(
        IsolationProfileSelection(
            mode="container",
            profile="tools",
            required=True,
        )
    )


def _sandbox_settings_raw(backend: str) -> dict[str, object]:
    return {
        "backend": backend,
        "default_profile": "host-tools",
        "allowed_profiles": ["host-tools"],
        "profiles": {"host-tools": _sandbox_profile_raw()},
        "profile_registry_id": "default",
        "policy_version": "phase2",
    }


def _sandbox_profile_raw(
    name: str = "host-tools",
    *,
    executable: str = "/bin/sh",
) -> dict[str, object]:
    return {
        "name": name,
        "trusted_executables": [executable],
        "executable_search_roots": ["/bin"],
        "read_roots": ["/workspace"],
        "write_roots": ["/workspace/out"],
        "deny_roots": ["/etc/ssh"],
        "scratch_roots": ["/tmp/avalan"],
        "output_roots": ["/workspace/out"],
        "environment": {
            "variables": {"LC_ALL": "C.UTF-8"},
            "allowlist": ["PATH"],
        },
        "network": {"mode": "loopback"},
        "resources": {"timeout_seconds": 10, "pids": 32},
        "output": {"allow_artifacts": True, "max_artifact_bytes": 1024},
        "child_processes": "deny",
    }


def _container_settings_raw(backend: str) -> dict[str, object]:
    return {
        "backend": backend,
        "default_profile": "tools",
        "allowed_profiles": ["tools"],
        "profiles": {
            "tools": {
                "name": "tools",
                "image": {"reference": _IMAGE},
            },
        },
        "profile_registry_id": "default",
        "policy_version": "phase2",
    }


class _FakeSandboxBackend:
    async def probe(self) -> object:
        return object()

    async def execute(self, plan: object) -> object:
        return plan


class _DiagnosticStub:
    def __init__(
        self,
        *,
        code: object,
        operation: object | None = None,
        message: object = "",
        hint: object = "",
        path: object = "",
        retryable: object = False,
    ) -> None:
        self.code = code
        self.operation = operation
        self.message = message
        self.hint = hint
        self.path = path
        self.retryable = retryable


def _fake_container_backend() -> ContainerFakeBackend:
    return ContainerFakeBackend(
        ContainerFakeBackendScript(
            capabilities=ContainerBackendCapabilities(
                backend=ContainerBackend.DOCKER,
                host_os="linux",
                guest_os="linux",
                architecture="amd64",
                rootless=True,
                mount_types=(ContainerMountType.WORKSPACE,),
                streaming_attach=True,
            )
        )
    )


if __name__ == "__main__":
    main()
