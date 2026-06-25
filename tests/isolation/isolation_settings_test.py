from typing import cast
from unittest import TestCase, main

from avalan.container import (
    ContainerBackend,
)
from avalan.isolation import (
    IsolationDiagnosticCategory,
    IsolationDiagnosticCode,
    IsolationEffectiveSettings,
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
    isolation_diagnostic,
    isolation_selection_from_mapping,
    serialize_isolation_effective_settings,
    trusted_isolation_runtime_from_mapping,
    trusted_isolation_settings_from_mapping,
    trusted_isolation_source,
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
                "isolation.policy_widening",
                "isolation.unsupported_syntax",
            },
        )
        diagnostic = isolation_diagnostic(
            IsolationDiagnosticCode.UNSUPPORTED_BACKEND,
            path="runtime.isolation.sandbox.backend",
            message="Unsupported sandbox backend.",
            hint="Use seatbelt or bubblewrap.",
            category=IsolationDiagnosticCategory.UNSUPPORTED,
        )
        self.assertEqual(
            diagnostic.to_dict(),
            {
                "code": "isolation.unsupported_backend",
                "path": "runtime.isolation.sandbox.backend",
                "category": "unsupported",
                "message": "Unsupported sandbox backend.",
                "hint": "Use seatbelt or bubblewrap.",
            },
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
        sandbox_runtime = IsolationToolRuntimeSettings(
            effective_settings=sandbox_effective,
        )

        self.assertIs(sandbox_runtime.sandbox, sandbox_effective.sandbox)
        self.assertIsNone(sandbox_runtime.local)
        self.assertIsNone(sandbox_runtime.container)
        self.assertEqual(
            sandbox_effective.canonical_json(),
            sandbox_effective.canonical_json(),
        )

        container_effective = _container_effective("docker")
        container_runtime = IsolationToolRuntimeSettings(
            effective_settings=container_effective,
        )

        self.assertIs(
            container_runtime.container, container_effective.container
        )
        self.assertIsNone(container_runtime.local)
        self.assertIsNone(container_runtime.sandbox)

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


if __name__ == "__main__":
    main()
