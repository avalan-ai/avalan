from unittest import TestCase, main

from avalan.container import (
    ContainerAuditEvent,
    ContainerAuditEventType,
    ContainerAuditMode,
    ContainerAuditPolicy,
    ContainerAuthorizationDecision,
    ContainerAuthorizationDecisionType,
    ContainerBackend,
    ContainerBackendCapabilities,
    ContainerCleanupMode,
    ContainerCleanupPolicy,
    ContainerCommandMode,
    ContainerCommandPlan,
    ContainerDeviceClass,
    ContainerDevicePolicy,
    ContainerEffectiveSettings,
    ContainerEnvironmentPolicy,
    ContainerEscalationMode,
    ContainerEscalationPolicy,
    ContainerExecutionResult,
    ContainerExecutionScope,
    ContainerImagePolicy,
    ContainerMountAccess,
    ContainerMountDeclaration,
    ContainerMountType,
    ContainerNetworkMode,
    ContainerNetworkPolicy,
    ContainerOutputPolicy,
    ContainerPoolingMode,
    ContainerPoolingPolicy,
    ContainerProfile,
    ContainerProfileSelection,
    ContainerPullPolicy,
    ContainerResourceLimits,
    ContainerResultStatus,
    ContainerRunPlan,
    ContainerRuntimeEnvelopePlan,
    ContainerSecretReference,
    ContainerSettings,
    ContainerSettingsSource,
    ContainerSurface,
    ContainerTrustLevel,
    ContainerWorkspaceMapping,
)

_DIGEST = "0" * 64
_IMAGE = f"ghcr.io/example/avalan-tools@sha256:{_DIGEST}"


class ContainerSettingsTest(TestCase):
    def test_valid_disabled_settings_serialize(self) -> None:
        source = ContainerSettingsSource(
            surface=ContainerSurface.CLI,
            trust_level=ContainerTrustLevel.TRUSTED_OPERATOR,
        )
        settings = ContainerSettings(source=source)

        self.assertFalse(settings.enabled)
        self.assertEqual(settings.backend, ContainerBackend.NONE)
        self.assertEqual(settings.to_dict()["profiles"], {})

        loaded = ContainerSettings.from_dict(
            settings.to_dict(),
            source=source,
        )
        self.assertEqual(loaded.to_dict(), settings.to_dict())

        selection = ContainerProfileSelection(required=False)
        self.assertEqual(
            selection.to_dict(),
            {
                "profile": None,
                "required": False,
                "scope": "shell_container_execution",
            },
        )
        loaded_selection = ContainerProfileSelection.from_dict(
            {},
            source=source,
        )
        self.assertEqual(loaded_selection.to_dict(), selection.to_dict())
        effective = settings.select_profile(selection)
        self.assertFalse(effective.enabled)
        self.assertEqual(
            ContainerEffectiveSettings.from_dict(
                effective.to_dict()
            ).to_dict(),
            effective.to_dict(),
        )

    def test_minimal_readonly_profile_selection_is_serializable(self) -> None:
        settings = _trusted_settings(ContainerSurface.CLI)
        source = ContainerSettingsSource(
            surface=ContainerSurface.TASK_TOML,
            trust_level=ContainerTrustLevel.UNTRUSTED_TASK,
        )
        selection = ContainerProfileSelection.from_dict(
            {"profile": "workspace-readonly", "required": True},
            source=source,
        )

        effective = settings.select_profile(selection)
        serialized = effective.to_dict()

        self.assertTrue(effective.enabled)
        self.assertEqual(effective.profile_name, "workspace-readonly")
        self.assertTrue(serialized["required"])
        self.assertEqual(
            ContainerEffectiveSettings.from_dict(serialized).to_dict(),
            serialized,
        )
        serialized["profile"]["name"] = "changed"
        self.assertEqual(effective.profile.name, "workspace-readonly")

    def test_full_profile_and_vocabulary_serialize(self) -> None:
        image = ContainerImagePolicy(
            reference="ghcr.io/example/tools:ignored",
            digest=f"sha256:{_DIGEST}",
            pull_policy=ContainerPullPolicy.IF_MISSING,
            platform="linux/arm64",
        )
        workspace = ContainerWorkspaceMapping(
            host_root=".",
            container_path="/workspace",
            working_directory="/workspace/project",
        )
        mounts = (
            ContainerMountDeclaration(
                source="out",
                target="/out",
                mount_type=ContainerMountType.OUTPUT,
                access=ContainerMountAccess.WRITE,
            ),
        )
        environment = ContainerEnvironmentPolicy(
            variables={"LC_ALL": "C.UTF-8"},
            allowlist=("PATH",),
        )
        secrets = (
            ContainerSecretReference(
                name="api-token",
                env_name="API_TOKEN",
            ),
            ContainerSecretReference(
                name="mounted-token",
                mount_path="/run/secrets/token",
            ),
        )
        network = ContainerNetworkPolicy(
            mode=ContainerNetworkMode.ALLOWLIST,
            egress_allowlist=("api.example.test",),
        )
        devices = ContainerDevicePolicy(devices=(ContainerDeviceClass.CPU,))
        resources = ContainerResourceLimits(
            cpu_count=2,
            memory_bytes=536870912,
            pids=128,
            timeout_seconds=30,
        )
        output = ContainerOutputPolicy(
            max_artifact_bytes=1024,
            allow_artifacts=True,
        )
        cleanup = ContainerCleanupPolicy(
            mode=ContainerCleanupMode.QUARANTINE,
            grace_seconds=7,
        )
        pooling = ContainerPoolingPolicy(mode=ContainerPoolingMode.SHORT_LIVED)
        audit = ContainerAuditPolicy(mode=ContainerAuditMode.FULL)
        escalation = ContainerEscalationPolicy(
            mode=ContainerEscalationMode.REQUIRE_REVIEW,
        )
        profile = ContainerProfile(
            name="full-profile",
            image=image,
            workspace=workspace,
            mounts=mounts,
            environment=environment,
            secrets=secrets,
            network=network,
            devices=devices,
            resources=resources,
            output=output,
            cleanup=cleanup,
            pooling=pooling,
            audit=audit,
            escalation=escalation,
            command_mode=ContainerCommandMode.FIXED_ENTRYPOINT,
        )
        profile_dict = profile.to_dict()

        self.assertEqual(
            ContainerProfile.from_dict(profile_dict).to_dict(),
            profile_dict,
        )

        capabilities = ContainerBackendCapabilities(
            backend="docker",
            host_os="darwin",
            guest_os="linux",
            architecture="arm64",
            platform_emulation=True,
            rootless=True,
            user_namespace=True,
            build=True,
            pull=True,
            network_modes=("none", "allowlist"),
            mount_types=("workspace", "output"),
            resource_limits=True,
            device_classes=("cpu",),
            per_container_vm_isolation=True,
            windows_process_isolation=True,
            windows_hyperv_isolation=True,
            streaming_attach=True,
            stats=True,
        )
        command = ContainerCommandPlan(
            tool_name="shell.rg",
            command="rg",
            argv=("rg", "needle"),
            cwd="/workspace",
            scope="shell_container_execution",
        )
        run_plan = ContainerRunPlan(
            backend=ContainerBackend.DOCKER,
            profile_name="full-profile",
            image=image,
            command=command,
            mounts=mounts,
            environment_names=("LC_ALL",),
            secret_names=("api-token",),
            network=network,
            devices=devices,
            resources=resources,
            policy_version="phase1",
        )
        envelope = ContainerRuntimeEnvelopePlan(
            scope=ContainerExecutionScope.RUNTIME_ENVELOPE,
            profile_name="full-profile",
            command=command,
        )
        decision = ContainerAuthorizationDecision(
            decision=ContainerAuthorizationDecisionType.ALLOW,
            code="allow.readonly",
            explanation="Allowed by read-only profile.",
            policy_version="phase1",
            profile_name="full-profile",
            cacheable=True,
        )
        result = ContainerRunResultFactory.completed()
        audit_event = ContainerAuditEvent(
            event_type=ContainerAuditEventType.RESULT_RECORDED,
            scope=ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
            profile_name="full-profile",
            policy_version="phase1",
            metadata={"status": "completed"},
        )

        self.assertEqual(capabilities.to_dict()["backend"], "docker")
        self.assertTrue(capabilities.to_dict()["platform_emulation"])
        self.assertTrue(
            capabilities.to_dict()["per_container_vm_isolation"],
        )
        self.assertTrue(capabilities.to_dict()["windows_process_isolation"])
        self.assertTrue(capabilities.to_dict()["windows_hyperv_isolation"])
        self.assertEqual(run_plan.to_dict()["command"], command.to_dict())
        self.assertEqual(envelope.to_dict()["scope"], "runtime_envelope")
        self.assertEqual(decision.to_dict()["decision"], "allow")
        self.assertEqual(result.to_dict()["status"], "completed")
        self.assertEqual(
            audit_event.to_dict()["event_type"],
            "result_recorded",
        )

    def test_audit_event_vocabulary_covers_core_phases(self) -> None:
        expected_events = {
            "policy_evaluation",
            "review_request",
            "review_decision",
            "backend_selection",
            "image_resolution",
            "image_pull",
            "build_progress",
            "mount_preparation",
            "container_create",
            "container_start",
            "stream_chunk",
            "output_copy",
            "cleanup",
            "denial",
            "failure",
        }
        actual_events = {
            event_type.value for event_type in ContainerAuditEventType
        }

        self.assertTrue(expected_events.issubset(actual_events))

    def test_equivalent_trusted_surfaces_normalize_consistently(self) -> None:
        cli = _trusted_settings(ContainerSurface.CLI)
        agent = _trusted_settings(ContainerSurface.AGENT_TOML)
        selection = ContainerProfileSelection(profile="workspace-readonly")

        cli_effective = cli.select_profile(selection).to_dict()
        agent_effective = agent.select_profile(selection).to_dict()
        cli_effective.pop("source")
        agent_effective.pop("source")

        self.assertEqual(cli_effective, agent_effective)

    def test_unknown_fields_are_rejected(self) -> None:
        source = _trusted_source(ContainerSurface.CLI)
        with self.assertRaises(AssertionError):
            ContainerSettings.from_dict(
                {"backend": "none", "unexpected": True},
                source=source,
            )
        with self.assertRaises(AssertionError):
            ContainerProfile.from_dict(
                {
                    "name": "bad",
                    "image": {"reference": _IMAGE},
                    "extra": True,
                }
            )

    def test_serialized_values_reject_wrong_scalar_types(self) -> None:
        with self.assertRaises(AssertionError):
            ContainerImagePolicy.from_dict(
                {
                    "reference": 123,
                    "digest": f"sha256:{_DIGEST}",
                }
            )
        with self.assertRaises(AssertionError):
            ContainerProfileSelection.from_dict(
                {"required": "yes"},
                source=_trusted_source(ContainerSurface.CLI),
            )
        with self.assertRaises(AssertionError):
            ContainerOutputPolicy.from_dict({"max_stdout_bytes": "100"})
        with self.assertRaises(AssertionError):
            ContainerProfile.from_dict(
                {
                    "name": 123,
                    "image": {"reference": _IMAGE},
                }
            )

    def test_unsafe_values_are_rejected(self) -> None:
        with self.assertRaises(AssertionError):
            ContainerImagePolicy(reference="alpine:latest")
        with self.assertRaises(AssertionError):
            ContainerMountDeclaration(
                source=".",
                target="/workspace",
                mount_type=ContainerMountType.WORKSPACE,
                access=ContainerMountAccess.WRITE,
            )
        with self.assertRaises(AssertionError):
            ContainerEnvironmentPolicy(inherit_host=True)
        with self.assertRaises(AssertionError):
            ContainerNetworkPolicy(
                mode=ContainerNetworkMode.NONE,
                egress_allowlist=("example.test",),
            )
        with self.assertRaises(AssertionError):
            ContainerOutputPolicy(max_artifact_bytes=1)
        with self.assertRaises(AssertionError):
            ContainerOutputPolicy.from_dict({"max_stdout_bytes": 0})
        with self.assertRaises(AssertionError):
            ContainerOutputPolicy.from_dict({"max_stderr_bytes": 0})
        with self.assertRaises(AssertionError):
            ContainerCleanupPolicy.from_dict({"grace_seconds": 0})
        with self.assertRaises(AssertionError):
            ContainerProfile(
                name="root-profile",
                image=ContainerImagePolicy(reference=_IMAGE),
                user="0:0",
            )

    def test_untrusted_sources_cannot_define_runtime_authority(self) -> None:
        source = ContainerSettingsSource(
            surface=ContainerSurface.AGENT_TOML,
            trust_level=ContainerTrustLevel.UNTRUSTED_AGENT,
        )

        with self.assertRaises(AssertionError):
            ContainerSettings.from_dict(
                {
                    "backend": "docker",
                    "profiles": {
                        "workspace-readonly": _readonly_profile().to_dict()
                    },
                },
                source=source,
            )

    def test_raw_settings_source_cannot_self_attest_authority(self) -> None:
        trusted_settings = _trusted_settings(ContainerSurface.CLI)
        raw = trusted_settings.to_dict()

        with self.assertRaises(AssertionError):
            ContainerSettings.from_dict(raw)

        raw["source"] = ContainerSettingsSource(
            surface=ContainerSurface.SERVER,
            trust_level=ContainerTrustLevel.MODEL,
        ).to_dict()
        loaded = ContainerSettings.from_dict(
            raw,
            source=_trusted_source(ContainerSurface.CLI),
        )

        self.assertEqual(
            loaded.source.trust_level,
            ContainerTrustLevel.TRUSTED_OPERATOR,
        )
        self.assertEqual(loaded.backend, ContainerBackend.DOCKER)

    def test_profile_mapping_rejects_non_string_keys(self) -> None:
        profile = _readonly_profile()

        with self.assertRaises(AssertionError):
            ContainerSettings.from_dict(
                {
                    "backend": "docker",
                    "profiles": {123: profile.to_dict()},
                },
                source=_trusted_source(ContainerSurface.CLI),
            )

    def test_model_visible_selection_rejects_runtime_authority(self) -> None:
        source = ContainerSettingsSource(
            surface=ContainerSurface.SERVER,
            trust_level=ContainerTrustLevel.MODEL,
        )

        with self.assertRaises(AssertionError):
            ContainerProfileSelection.from_dict(
                {
                    "profile": "workspace-readonly",
                    "required": True,
                    "scope": "runtime_envelope",
                },
                source=source,
            )
        with self.assertRaises(AssertionError):
            ContainerProfileSelection.from_dict(
                {},
                source=source,
            )


class ContainerRunResultFactory:
    @staticmethod
    def completed():
        return ContainerExecutionResult(
            status=ContainerResultStatus.COMPLETED,
            exit_code=0,
            diagnostics=("ok",),
            metadata={"phase": "test"},
        )


def _trusted_source(surface: ContainerSurface) -> ContainerSettingsSource:
    return ContainerSettingsSource(
        surface=surface,
        trust_level=ContainerTrustLevel.TRUSTED_OPERATOR,
    )


def _readonly_profile() -> ContainerProfile:
    return ContainerProfile.minimal_readonly(
        name="workspace-readonly",
        image_reference=_IMAGE,
    )


def _trusted_settings(surface: ContainerSurface) -> ContainerSettings:
    profile = _readonly_profile()
    return ContainerSettings(
        source=_trusted_source(surface),
        backend=ContainerBackend.DOCKER,
        default_profile=profile.name,
        allowed_profiles=(profile.name,),
        profiles={profile.name: profile},
        profile_registry_id="unit-registry",
        policy_version="phase1",
    )


if __name__ == "__main__":
    main()
