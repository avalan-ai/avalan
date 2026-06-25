from unittest import TestCase

from avalan.container import (
    ContainerBackend,
    ContainerEffectiveSettings,
    ContainerExecutionScope,
    ContainerProfile,
    ContainerRuntimeEnvelopeKind,
    ContainerSettingsSource,
    ContainerSurface,
    ContainerToolRuntimeSettings,
    ContainerTrustLevel,
    disabled_required_container_settings,
)
from avalan.server.container_policy import (
    RemoteContainerRequestError,
    RemoteContainerRequestPolicy,
    remote_container_policy_from_runtime_settings,
    server_runtime_envelope_status_from_runtime_settings,
    validate_remote_container_arguments,
)


class RemoteContainerRequestPolicyTestCase(TestCase):
    def test_policy_from_runtime_settings_exposes_trusted_profiles(
        self,
    ) -> None:
        policy = remote_container_policy_from_runtime_settings(
            _runtime_settings(
                "workspace-readonly",
            )
        )

        assert policy is not None
        self.assertEqual(
            policy.exposed_profiles,
            ("workspace-readonly",),
        )

    def test_policy_exposes_only_effective_profile(self) -> None:
        policy = remote_container_policy_from_runtime_settings(
            _runtime_settings("workspace-readonly", "workspace-rich")
        )

        assert policy is not None
        self.assertEqual(policy.exposed_profiles, ("workspace-readonly",))

    def test_policy_from_runtime_settings_fails_closed_without_trust(
        self,
    ) -> None:
        cases = (
            None,
            ContainerToolRuntimeSettings(),
            ContainerToolRuntimeSettings(
                effective_settings=disabled_required_container_settings(
                    ContainerSurface.SERVER
                ),
            ),
            _runtime_settings("workspace-readonly", trusted=False),
            _runtime_settings(),
            _runtime_settings("workspace-readonly", selected=False),
        )

        for runtime_settings in cases:
            with self.subTest(runtime_settings=runtime_settings):
                self.assertIsNone(
                    remote_container_policy_from_runtime_settings(
                        runtime_settings
                    )
                )

    def test_rejects_multiple_profile_selectors(self) -> None:
        policy = RemoteContainerRequestPolicy(
            exposed_profiles=("workspace-readonly",)
        )

        with self.assertRaisesRegex(
            RemoteContainerRequestError,
            "one container profile selector",
        ):
            validate_remote_container_arguments(
                {
                    "container": {"profile": "workspace-readonly"},
                    "container_profile": "workspace-readonly",
                },
                policy=policy,
            )

    def test_rejects_non_object_container_selector(self) -> None:
        with self.assertRaisesRegex(
            RemoteContainerRequestError,
            "selector must be an object",
        ):
            validate_remote_container_arguments({"container": "profile"})

    def test_rejects_empty_profile_selector(self) -> None:
        with self.assertRaisesRegex(
            RemoteContainerRequestError,
            "non-empty string",
        ):
            validate_remote_container_arguments({"containerProfile": " "})

    def test_ignores_non_string_argument_keys(self) -> None:
        request = validate_remote_container_arguments({1: {"name": "safe"}})

        self.assertEqual(request.arguments, {1: {"name": "safe"}})
        self.assertIsNone(request.profile)

    def test_server_runtime_envelope_status_uses_trusted_settings(
        self,
    ) -> None:
        status = server_runtime_envelope_status_from_runtime_settings(
            _runtime_settings(
                "workspace-readonly",
                scope=ContainerExecutionScope.RUNTIME_ENVELOPE,
            ),
            server_name="api",
            request_id="server-1",
        )

        assert status.plan is not None
        self.assertFalse(status.ok)
        self.assertEqual(
            status.plan.envelope_kind,
            ContainerRuntimeEnvelopeKind.SERVER,
        )
        self.assertEqual(
            status.plan.envelope_plan.profile_name,
            "workspace-readonly",
        )
        self.assertEqual(
            status.plan.run_plan.request.request_id,
            "server-1",
        )
        self.assertEqual(
            status.diagnostics[0]["code"],
            "server.runtime_envelope_unavailable",
        )

    def test_server_runtime_envelope_status_accepts_available_runtime(
        self,
    ) -> None:
        status = server_runtime_envelope_status_from_runtime_settings(
            _runtime_settings(
                "workspace-readonly",
                scope=ContainerExecutionScope.RUNTIME_ENVELOPE,
            ),
            envelope_runtime_available=True,
        )

        assert status.plan is not None
        self.assertTrue(status.ok)
        self.assertEqual(status.diagnostics, ())

    def test_server_runtime_envelope_status_ignores_shell_settings(
        self,
    ) -> None:
        status = server_runtime_envelope_status_from_runtime_settings(
            _runtime_settings("workspace-readonly")
        )

        self.assertIsNone(status.plan)
        self.assertTrue(status.ok)

    def test_remote_runtime_envelope_authority_still_rejected(self) -> None:
        with self.assertRaisesRegex(
            RemoteContainerRequestError,
            "runtime authority",
        ):
            validate_remote_container_arguments(
                {"runtime_envelope": {"profile": "workspace-readonly"}}
            )

    def test_remote_isolation_and_sandbox_authority_rejected(self) -> None:
        for arguments in (
            {"isolation": {"mode": "sandbox"}},
            {"isolationPolicy": {"profile": "workspace-readonly"}},
            {"sandboxProfile": "workspace-readonly"},
            {"sandbox": {"read_roots": ["/workspace"]}},
            {"approval": {"review_mode": "manual"}},
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(
                    RemoteContainerRequestError,
                    "runtime authority",
                ):
                    validate_remote_container_arguments(arguments)


def _runtime_settings(
    *profiles: str,
    trusted: bool = True,
    selected: bool = True,
    scope: ContainerExecutionScope = (
        ContainerExecutionScope.SHELL_CONTAINER_EXECUTION
    ),
) -> ContainerToolRuntimeSettings:
    trust_level = (
        ContainerTrustLevel.TRUSTED_OPERATOR
        if trusted
        else ContainerTrustLevel.UNTRUSTED_REQUEST
    )
    source = ContainerSettingsSource(
        surface=ContainerSurface.SERVER,
        trust_level=trust_level,
    )
    return ContainerToolRuntimeSettings(
        effective_settings=ContainerEffectiveSettings(
            backend=ContainerBackend.DOCKER,
            required=False,
            scope=scope,
            source=source,
            policy_version="phase14",
            profile_registry_id="server",
            profile_name=profiles[0] if profiles and selected else None,
            profile=(
                _readonly_profile(profiles[0])
                if profiles and selected
                else None
            ),
            allowed_profiles=profiles,
        )
    )


def _readonly_profile(name: str) -> ContainerProfile:
    return ContainerProfile.minimal_readonly(
        name=name,
        image_reference=(
            "registry.example/workspace@sha256:"
            "1111111111111111111111111111111111111111111111111111111111111111"
        ),
    )
