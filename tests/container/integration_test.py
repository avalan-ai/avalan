from unittest import TestCase, main

from avalan.container import (
    ContainerBackend,
    ContainerBackendCapabilities,
    ContainerEffectiveSettings,
    ContainerExecutionScope,
    ContainerFakeBackend,
    ContainerFakeBackendScript,
    ContainerMountAccess,
    ContainerMountType,
    ContainerNetworkMode,
    ContainerSettingsSource,
    ContainerSurface,
    ContainerToolRuntimeSettings,
    ContainerTrustLevel,
    container_selection_from_mapping,
    disabled_required_container_settings,
    trusted_container_runtime_from_mapping,
    trusted_container_settings_from_mapping,
    trusted_container_source,
)

_DIGEST = "8" * 64
_IMAGE = f"ghcr.io/example/sdk-tools@sha256:{_DIGEST}"


class ContainerIntegrationTest(TestCase):
    def test_sdk_runtime_settings_accept_custom_hooks(self) -> None:
        backend = ContainerFakeBackend(
            ContainerFakeBackendScript(capabilities=_capabilities())
        )

        def authorize(value: object) -> object:
            return value

        def resolve_secret(name: str) -> object:
            return f"resolved:{name}"

        def audit(value: object) -> object:
            return value

        runtime = ContainerToolRuntimeSettings(
            effective_settings=disabled_required_container_settings("sdk"),
            backend=backend,
            authorization_provider=authorize,
            secret_resolver=resolve_secret,
            audit_listeners=(audit,),
        )

        self.assertIs(runtime.backend, backend)
        assert runtime.effective_settings is not None
        self.assertFalse(runtime.effective_settings.enabled)
        self.assertTrue(runtime.effective_settings.required)
        assert runtime.authorization_provider is not None
        assert runtime.secret_resolver is not None
        self.assertEqual(runtime.authorization_provider("plan"), "plan")
        self.assertEqual(runtime.secret_resolver("TOKEN"), "resolved:TOKEN")
        self.assertEqual(runtime.audit_listeners[0]("event"), "event")

        with self.assertRaises(AssertionError):
            ContainerToolRuntimeSettings(
                audit_listeners=(object(),),  # type: ignore[arg-type]
            )

    def test_trusted_mapping_normalizes_toml_friendly_profile(self) -> None:
        source = trusted_container_source(ContainerSurface.AGENT_TOML)
        selection = container_selection_from_mapping(
            {"profile": "workspace-readonly", "required": True},
            source=source,
        )

        runtime = trusted_container_runtime_from_mapping(
            {
                "backend": "docker",
                "default_profile": "workspace-readonly",
                "profiles": {
                    "workspace-readonly": {
                        "image": _IMAGE,
                        "workspace": "/workspace",
                        "workspace_root": ".",
                        "mounts": [
                            {
                                "source": ".",
                                "target": "/workspace",
                                "access": "read",
                            },
                            {
                                "source": "input",
                                "target": "/input",
                                "type": "input",
                            },
                        ],
                        "environment": {
                            "variables": {"LC_ALL": "C.UTF-8"},
                            "allowlist": ["PATH"],
                        },
                        "secrets": [{"name": "token", "env_name": "TOKEN"}],
                        "network": "loopback",
                        "devices": {"devices": []},
                        "resources": {"cpu_count": 1},
                        "output": {
                            "allow_artifacts": True,
                            "max_artifact_bytes": 1024,
                        },
                        "cleanup": {"grace_seconds": 3},
                        "audit": {"mode": "full"},
                        "review_mode": "require_review",
                    }
                },
            },
            source=source,
            selection=selection,
        )
        effective = runtime.effective_settings

        self.assertIsInstance(effective, ContainerEffectiveSettings)
        assert effective is not None
        self.assertEqual(effective.backend, ContainerBackend.DOCKER)
        self.assertTrue(effective.required)
        self.assertEqual(effective.profile_name, "workspace-readonly")
        assert effective.profile is not None
        self.assertEqual(
            effective.profile.workspace.container_path,
            "/workspace",
        )
        self.assertEqual(
            effective.profile.mounts[0].access,
            ContainerMountAccess.READ,
        )
        self.assertEqual(
            effective.profile.network.mode,
            ContainerNetworkMode.LOOPBACK,
        )
        self.assertEqual(effective.profile.resources.cpu_count, 1)
        self.assertTrue(effective.profile.output.allow_artifacts)
        self.assertEqual(effective.profile.audit.to_dict()["mode"], "full")
        self.assertEqual(
            effective.profile.escalation.to_dict()["mode"],
            "require_review",
        )

    def test_full_mapping_shape_and_runtime_scope_selection(self) -> None:
        source = trusted_container_source(ContainerSurface.AGENT_TOML)
        selection = container_selection_from_mapping(
            {"profile": "sdk-profile"},
            source=source,
            scope=ContainerExecutionScope.RUNTIME_ENVELOPE,
        )
        settings = trusted_container_settings_from_mapping(
            {
                "backend": "podman",
                "default_profile": "sdk-profile",
                "profiles": {
                    "sdk-profile": {
                        "image": {
                            "reference": _IMAGE,
                            "platform": "linux/amd64",
                        },
                        "workspace": {
                            "host_root": ".",
                            "container_path": "/workspace",
                            "working_directory": "/workspace",
                        },
                        "network": {"mode": "none"},
                        "escalation": {"mode": "deny"},
                    }
                },
            },
            source=source,
        )
        effective = settings.select_profile(selection)

        self.assertEqual(effective.backend, ContainerBackend.PODMAN)
        self.assertEqual(
            effective.scope,
            ContainerExecutionScope.RUNTIME_ENVELOPE,
        )
        self.assertEqual(effective.profile_name, "sdk-profile")

    def test_unknown_unsafe_and_untrusted_values_reject(self) -> None:
        source = trusted_container_source(ContainerSurface.AGENT_TOML)
        untrusted = ContainerSettingsSource(
            surface=ContainerSurface.AGENT_TOML,
            trust_level=ContainerTrustLevel.UNTRUSTED_AGENT,
        )
        cases = (
            {"unexpected": True},
            {
                "backend": "docker",
                "profiles": {
                    "bad": {
                        "name": "other",
                        "image": _IMAGE,
                    }
                },
            },
            {
                "backend": "docker",
                "profiles": {"bad": {"image": 123}},
            },
            {
                "backend": "docker",
                "profiles": {
                    "bad": {
                        "image": _IMAGE,
                        "workspace": 123,
                    }
                },
            },
            {
                "backend": "docker",
                "profiles": {
                    "bad": {
                        "image": _IMAGE,
                        "mounts": [{"target": "/workspace", "bad": True}],
                    }
                },
            },
        )

        for raw in cases:
            with self.subTest(raw=raw):
                with self.assertRaises(AssertionError):
                    trusted_container_settings_from_mapping(raw, source=source)
        with self.assertRaises(AssertionError):
            trusted_container_settings_from_mapping(
                {
                    "backend": "docker",
                    "profiles": {"bad": {"image": _IMAGE}},
                },
                source=untrusted,
            )

    def test_empty_sequence_and_string_scope_helpers_are_covered(self) -> None:
        source = trusted_container_source(ContainerSurface.AGENT_TOML)
        settings = trusted_container_settings_from_mapping(
            {
                "backend": "docker",
                "default_profile": "workspace-readonly",
                "profiles": {
                    "workspace-readonly": {
                        "image": _IMAGE,
                        "mounts": None,
                    }
                },
            },
            source=source,
        )
        selection = container_selection_from_mapping(
            {"profile": "workspace-readonly"},
            source=source,
            scope="shell_container_execution",
        )

        effective = settings.select_profile(selection)

        assert effective.profile is not None
        self.assertEqual(len(effective.profile.mounts), 1)
        mount = effective.profile.mounts[0]
        self.assertEqual(mount.target, "/workspace")
        self.assertEqual(mount.source, ".")
        self.assertEqual(mount.mount_type, ContainerMountType.WORKSPACE)
        self.assertEqual(mount.access, ContainerMountAccess.READ)
        self.assertEqual(
            effective.scope,
            ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
        )


def _capabilities() -> ContainerBackendCapabilities:
    return ContainerBackendCapabilities(
        backend=ContainerBackend.DOCKER,
        host_os="linux",
        guest_os="linux",
        architecture="amd64",
    )


if __name__ == "__main__":
    main()
