from unittest import TestCase, main

from avalan.container import (
    CONTAINER_SETTINGS_PRECEDENCE,
    ContainerAuditEvent,
    ContainerAuditEventType,
    ContainerAuditMode,
    ContainerAuditPolicy,
    ContainerAuthorityCaps,
    ContainerAuthorizationDecision,
    ContainerAuthorizationDecisionType,
    ContainerBackend,
    ContainerBackendCapabilities,
    ContainerBackendSupportLevel,
    ContainerCleanupMode,
    ContainerCleanupPolicy,
    ContainerCleanupPolicyOverride,
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
    ContainerOutputPolicyOverride,
    ContainerPlatformBehavior,
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
    ContainerSettingsOverride,
    ContainerSettingsPrecedence,
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
            runtime_name="Docker Desktop Linux VM",
            support_level="supported",
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
            vm_backed=True,
            remote_engine=True,
            windows_process_isolation=True,
            windows_hyperv_isolation=True,
            streaming_attach=True,
            stats=True,
            lifecycle_normalization=True,
            platform_behavior=ContainerPlatformBehavior(
                file_io="shared VM file I/O",
                networking="VM networking",
                architecture_emulation="amd64 emulation available",
                resources="VM resource ceiling",
                signals="signals cross VM boundary",
                path_syntax="POSIX host paths",
                drive_letters="not applicable",
                case_behavior="host dependent",
            ),
            shared_mount_prefixes=("/Users/",),
            parity_requirements=("none",),
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
        self.assertEqual(
            capabilities.support_level,
            ContainerBackendSupportLevel.SUPPORTED,
        )
        self.assertEqual(
            capabilities.to_dict()["runtime_name"],
            "Docker Desktop Linux VM",
        )
        self.assertEqual(
            capabilities.to_dict()["support_level"],
            "supported",
        )
        self.assertTrue(capabilities.to_dict()["platform_emulation"])
        self.assertTrue(
            capabilities.to_dict()["per_container_vm_isolation"],
        )
        self.assertTrue(capabilities.to_dict()["vm_backed"])
        self.assertTrue(capabilities.to_dict()["remote_engine"])
        self.assertTrue(capabilities.to_dict()["windows_process_isolation"])
        self.assertTrue(capabilities.to_dict()["windows_hyperv_isolation"])
        behavior_dict = capabilities.to_dict()["platform_behavior"]
        assert isinstance(behavior_dict, dict)
        self.assertEqual(
            behavior_dict,
            ContainerPlatformBehavior.from_dict(behavior_dict).to_dict(),
        )
        self.assertEqual(
            capabilities.to_dict()["shared_mount_prefixes"],
            ["/Users/"],
        )
        self.assertEqual(
            capabilities.to_dict()["parity_requirements"], ["none"]
        )
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
            ContainerImagePolicy(
                reference=f"ghcr.io/example/tools@sha256:{'5' * 64}",
                digest=f"sha256:{'6' * 64}",
            )
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
            ContainerPlatformBehavior(
                file_io="",
                networking="networking",
                architecture_emulation="emulation",
                resources="resources",
                signals="signals",
                path_syntax="paths",
                drive_letters="drives",
                case_behavior="case",
            )
        with self.assertRaises(AssertionError):
            ContainerPlatformBehavior.from_dict({"file_io": "only"})
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

    def test_authority_caps_merge_narrows_each_policy_field(self) -> None:
        settings = _authority_settings(ContainerSurface.SERVER)
        caps = ContainerAuthorityCaps(settings=settings)
        profile = settings.profiles["workspace-rich"]
        sdk = ContainerSettingsOverride(
            source=_trusted_source(ContainerSurface.SDK),
            layer=ContainerSettingsPrecedence.SDK,
            profile="workspace-rich",
            backend=ContainerBackend.DOCKER,
            image=profile.image,
            workspace=profile.workspace,
            resources=ContainerResourceLimits(
                cpu_count=2,
                memory_bytes=268435456,
                pids=64,
                timeout_seconds=20,
            ),
            output=ContainerOutputPolicyOverride(
                max_stdout_bytes=4096,
                max_artifact_bytes=512,
                allow_artifacts=True,
            ),
            cleanup=ContainerCleanupPolicyOverride(grace_seconds=4),
            command_mode=ContainerCommandMode.FIXED_EXECUTABLE,
            read_only_rootfs=True,
            user="1000:1000",
        )
        cli = ContainerSettingsOverride.from_dict(
            {
                "layer": "cli",
                "network": {
                    "mode": "allowlist",
                    "egress_allowlist": ["api.example.test"],
                },
                "devices": {"devices": ["cpu"]},
                "output": {"max_stdout_bytes": 2048},
                "cleanup": {
                    "mode": "quarantine",
                    "grace_seconds": 4,
                },
                "audit": {"mode": "full"},
            },
            source=_trusted_source(ContainerSurface.CLI),
        )
        agent = ContainerSettingsOverride.from_dict(
            {
                "mounts": [
                    {
                        "source": ".",
                        "target": "/workspace",
                        "mount_type": "workspace",
                    }
                ],
                "environment": {
                    "variables": {"LC_ALL": "C.UTF-8"},
                    "allowlist": ["PATH"],
                },
                "secrets": [
                    {
                        "name": "api-token",
                        "env_name": "API_TOKEN",
                    }
                ],
            },
            source=_untrusted_source(
                ContainerSurface.AGENT_TOML,
                ContainerTrustLevel.UNTRUSTED_AGENT,
            ),
        )
        flow = ContainerSettingsOverride.from_dict(
            {
                "network": {"mode": "loopback"},
                "escalation": {"mode": "require_review"},
            },
            source=_untrusted_source(
                ContainerSurface.FLOW_TOML,
                ContainerTrustLevel.UNTRUSTED_FLOW,
            ),
        )
        task = ContainerSettingsOverride.from_dict(
            {
                "profile": "workspace-rich",
                "required": True,
                "scope": "shell_container_execution",
                "network": {"mode": "none"},
                "output": {"allow_artifacts": False},
                "escalation": {"mode": "deny"},
            },
            source=_untrusted_source(
                ContainerSurface.TASK_TOML,
                ContainerTrustLevel.UNTRUSTED_TASK,
            ),
        )

        effective = caps.merge((task, flow, cli, agent, sdk))
        narrowed = effective.profile

        self.assertTrue(effective.required)
        self.assertEqual(effective.profile_registry_id, "unit-registry")
        self.assertEqual(effective.policy_version, "phase1")
        self.assertEqual(effective.profile_name, "workspace-rich")
        self.assertEqual(
            effective.canonical_policy_input()["profile_name"],
            "workspace-rich",
        )
        self.assertEqual(sdk.to_dict()["backend"], "docker")
        self.assertEqual(cli.to_dict()["network"]["mode"], "allowlist")
        self.assertEqual(narrowed.mounts[0].target, "/workspace")
        self.assertEqual(len(narrowed.mounts), 1)
        self.assertEqual(
            dict(narrowed.environment.variables),
            {
                "LC_ALL": "C.UTF-8",
            },
        )
        self.assertEqual(narrowed.environment.allowlist, ("PATH",))
        self.assertEqual(narrowed.secrets[0].name, "api-token")
        self.assertEqual(narrowed.network.mode, ContainerNetworkMode.NONE)
        self.assertEqual(narrowed.devices.devices, (ContainerDeviceClass.CPU,))
        self.assertEqual(narrowed.resources.cpu_count, 2)
        self.assertEqual(narrowed.resources.timeout_seconds, 20)
        self.assertEqual(narrowed.output.max_stdout_bytes, 2048)
        self.assertFalse(narrowed.output.allow_artifacts)
        self.assertEqual(narrowed.output.max_artifact_bytes, 0)
        self.assertEqual(narrowed.cleanup.grace_seconds, 4)
        self.assertEqual(narrowed.audit.mode, ContainerAuditMode.FULL)
        self.assertEqual(
            narrowed.escalation.mode,
            ContainerEscalationMode.DENY,
        )

    def test_network_full_override_requires_full_network_cap(self) -> None:
        rich = _rich_profile()
        settings = ContainerSettings(
            source=_trusted_source(ContainerSurface.SERVER),
            backend=ContainerBackend.DOCKER,
            default_profile=rich.name,
            allowed_profiles=(rich.name,),
            profiles={rich.name: rich},
            profile_registry_id="unit-registry",
            policy_version="phase1",
        )
        request_source = _untrusted_source(
            ContainerSurface.SERVER,
            ContainerTrustLevel.UNTRUSTED_REQUEST,
        )
        full_override = ContainerSettingsOverride.from_dict(
            {"network": {"mode": "full"}},
            source=request_source,
        )

        with self.assertRaises(AssertionError):
            ContainerAuthorityCaps(settings=settings).merge((full_override,))

        full_profile_dict = rich.to_dict()
        full_profile_dict["network"] = {
            "mode": "full",
            "egress_allowlist": [],
        }
        full_profile = ContainerProfile.from_dict(full_profile_dict)
        full_settings = ContainerSettings(
            source=_trusted_source(ContainerSurface.SERVER),
            backend=ContainerBackend.DOCKER,
            default_profile=full_profile.name,
            allowed_profiles=(full_profile.name,),
            profiles={full_profile.name: full_profile},
            profile_registry_id="unit-registry",
            policy_version="phase1",
        )
        full_caps = ContainerAuthorityCaps(settings=full_settings)
        allowlist_override = ContainerSettingsOverride.from_dict(
            {
                "network": {
                    "mode": "allowlist",
                    "egress_allowlist": ["api.example.test"],
                },
            },
            source=request_source,
        )

        self.assertEqual(
            full_caps.merge((full_override,)).profile.network.mode,
            ContainerNetworkMode.FULL,
        )
        self.assertEqual(
            full_caps.merge((allowlist_override,)).profile.network.mode,
            ContainerNetworkMode.ALLOWLIST,
        )

    def test_precedence_and_finite_resource_narrowing_are_deterministic(
        self,
    ) -> None:
        settings = _authority_settings(ContainerSurface.CLI)
        caps = ContainerAuthorityCaps(settings=settings)
        sdk = ContainerSettingsOverride.from_dict(
            {"profile": "workspace-rich"},
            source=_trusted_source(ContainerSurface.SDK),
        )
        worker = ContainerSettingsOverride.from_dict(
            {},
            source=_trusted_source(ContainerSurface.SERVER),
            layer=ContainerSettingsPrecedence.WORKER,
        )
        task = ContainerSettingsOverride.from_dict(
            {
                "resources": {"cpu_count": 1},
            },
            source=_untrusted_source(
                ContainerSurface.TASK_TOML,
                ContainerTrustLevel.UNTRUSTED_TASK,
            ),
        )

        effective = caps.merge((task, sdk, worker))

        self.assertEqual(
            CONTAINER_SETTINGS_PRECEDENCE,
            (
                ContainerSettingsPrecedence.SERVER_OPERATOR,
                ContainerSettingsPrecedence.WORKER,
                ContainerSettingsPrecedence.SDK,
                ContainerSettingsPrecedence.CLI,
                ContainerSettingsPrecedence.AGENT_TOML,
                ContainerSettingsPrecedence.FLOW_TOML,
                ContainerSettingsPrecedence.TASK_TOML,
                ContainerSettingsPrecedence.REQUEST,
            ),
        )
        self.assertEqual(effective.profile_name, "workspace-rich")
        self.assertEqual(effective.profile.resources.cpu_count, 1)
        self.assertEqual(worker.layer, ContainerSettingsPrecedence.WORKER)

    def test_disabled_authority_caps_merge_without_runtime_profile(
        self,
    ) -> None:
        source = _trusted_source(ContainerSurface.SERVER)
        caps = ContainerAuthorityCaps(
            settings=ContainerSettings(source=source),
        )
        override = ContainerSettingsOverride.from_dict(
            {"required": True},
            source=_trusted_source(ContainerSurface.CLI),
        )

        effective = caps.merge((override,))

        self.assertFalse(effective.enabled)
        self.assertTrue(effective.required)
        self.assertIsNone(effective.profile)

    def test_fake_e2e_equivalent_merges_have_same_canonical_policy(
        self,
    ) -> None:
        cli = ContainerAuthorityCaps(
            settings=_authority_settings(
                ContainerSurface.CLI,
                allowed_profiles=("workspace-rich", "workspace-readonly"),
                rich_profile=_rich_profile(),
            ),
        )
        server = ContainerAuthorityCaps(
            settings=_authority_settings(
                ContainerSurface.SERVER,
                allowed_profiles=("workspace-readonly", "workspace-rich"),
                rich_profile=_rich_profile(reverse_order=True),
            ),
        )
        overrides = (
            ContainerSettingsOverride.from_dict(
                {
                    "profile": "workspace-rich",
                    "network": {
                        "mode": "allowlist",
                        "egress_allowlist": ["api.example.test"],
                    },
                    "resources": {
                        "cpu_count": 2,
                        "timeout_seconds": 30,
                    },
                },
                source=_trusted_source(ContainerSurface.CLI),
            ),
        )

        self.assertEqual(
            cli.merge(overrides).canonical_policy_input(),
            server.merge(overrides).canonical_policy_input(),
        )
        self.assertNotEqual(
            cli.merge(overrides).to_dict()["profile"],
            server.merge(overrides).to_dict()["profile"],
        )

    def test_untrusted_overrides_cannot_define_authority_fields(self) -> None:
        source = _untrusted_source(
            ContainerSurface.AGENT_TOML,
            ContainerTrustLevel.UNTRUSTED_AGENT,
        )
        forbidden = (
            {"backend": "docker"},
            {"image": {"reference": _IMAGE}},
            {"workspace": {"host_root": ".", "container_path": "/workspace"}},
            {"command_mode": "fixed_entrypoint"},
            {"read_only_rootfs": True},
            {"user": "1000:1000"},
        )

        for raw in forbidden:
            with self.subTest(raw=raw):
                with self.assertRaises(AssertionError):
                    ContainerSettingsOverride.from_dict(raw, source=source)
        with self.assertRaises(AssertionError):
            ContainerSettingsOverride.from_dict(
                {"privileged": True},
                source=source,
            )
        with self.assertRaises(AssertionError):
            ContainerSettingsOverride.from_dict(
                {"capabilities": ["SYS_ADMIN"]},
                source=source,
            )
        with self.assertRaises(AssertionError):
            ContainerSettingsOverride.from_dict(
                {"layer": "task_toml"},
                source=source,
            )
        with self.assertRaises(AssertionError):
            ContainerSettingsOverride.from_dict(
                {"scope": "runtime_envelope"},
                source=source,
            )
        with self.assertRaises(AssertionError):
            ContainerSettingsOverride(
                source=source,
                layer=ContainerSettingsPrecedence.TASK_TOML,
            )
        with self.assertRaises(AssertionError):
            ContainerSettingsOverride(
                source=source,
                layer=ContainerSettingsPrecedence.AGENT_TOML,
                scope=ContainerExecutionScope.RUNTIME_ENVELOPE,
            )
        with self.assertRaises(AssertionError):
            ContainerSettingsOverride.from_dict(
                {},
                source=_untrusted_source(
                    ContainerSurface.SERVER,
                    ContainerTrustLevel.MODEL,
                ),
            )

    def test_untrusted_selection_cannot_raise_scope(self) -> None:
        source = _untrusted_source(
            ContainerSurface.TASK_TOML,
            ContainerTrustLevel.UNTRUSTED_TASK,
        )

        with self.assertRaises(AssertionError):
            ContainerProfileSelection.from_dict(
                {"scope": "runtime_envelope"},
                source=source,
            )

    def test_untrusted_server_request_uses_request_precedence(self) -> None:
        source = _untrusted_source(
            ContainerSurface.SERVER,
            ContainerTrustLevel.UNTRUSTED_REQUEST,
        )

        override = ContainerSettingsOverride.from_dict({}, source=source)

        self.assertEqual(override.layer, ContainerSettingsPrecedence.REQUEST)
        with self.assertRaises(AssertionError):
            ContainerSettingsOverride.from_dict(
                {"layer": "server_operator"},
                source=source,
            )
        with self.assertRaises(AssertionError):
            ContainerSettingsOverride(
                source=source,
                layer=ContainerSettingsPrecedence.SERVER_OPERATOR,
            )

    def test_merge_rejects_untrusted_widening_attempts(self) -> None:
        caps = ContainerAuthorityCaps(
            settings=_authority_settings(ContainerSurface.CLI),
        )
        source = _untrusted_source(
            ContainerSurface.TASK_TOML,
            ContainerTrustLevel.UNTRUSTED_TASK,
        )
        attempts = (
            ContainerSettingsOverride(
                source=source,
                layer=ContainerSettingsPrecedence.TASK_TOML,
                profile="workspace-rich",
            ),
            ContainerSettingsOverride(
                source=source,
                layer=ContainerSettingsPrecedence.TASK_TOML,
                profile="workspace-rich",
                mounts=(
                    ContainerMountDeclaration(
                        source="cache",
                        target="/cache",
                        mount_type=ContainerMountType.CACHE,
                    ),
                ),
            ),
            ContainerSettingsOverride(
                source=source,
                layer=ContainerSettingsPrecedence.TASK_TOML,
                profile="workspace-rich",
                secrets=(
                    ContainerSecretReference(
                        name="new-token",
                        env_name="NEW_TOKEN",
                    ),
                ),
            ),
            ContainerSettingsOverride(
                source=source,
                layer=ContainerSettingsPrecedence.TASK_TOML,
                profile="workspace-readonly",
                network=ContainerNetworkPolicy(
                    mode=ContainerNetworkMode.ALLOWLIST,
                    egress_allowlist=("api.example.test",),
                ),
            ),
            ContainerSettingsOverride(
                source=source,
                layer=ContainerSettingsPrecedence.TASK_TOML,
                profile="workspace-rich",
                devices=ContainerDevicePolicy(
                    devices=(ContainerDeviceClass.VULKAN_FORWARDED,),
                ),
            ),
            ContainerSettingsOverride(
                source=source,
                layer=ContainerSettingsPrecedence.TASK_TOML,
                profile="workspace-rich",
                resources=ContainerResourceLimits(cpu_count=9),
            ),
            ContainerSettingsOverride(
                source=source,
                layer=ContainerSettingsPrecedence.TASK_TOML,
                profile="workspace-rich",
                output=ContainerOutputPolicyOverride(
                    max_stdout_bytes=90000,
                ),
            ),
            ContainerSettingsOverride(
                source=source,
                layer=ContainerSettingsPrecedence.TASK_TOML,
                profile="workspace-rich",
                cleanup=ContainerCleanupPolicyOverride(grace_seconds=99),
            ),
            ContainerSettingsOverride(
                source=source,
                layer=ContainerSettingsPrecedence.TASK_TOML,
                profile="workspace-rich",
                audit=ContainerAuditPolicy(mode=ContainerAuditMode.MINIMAL),
            ),
            ContainerSettingsOverride(
                source=source,
                layer=ContainerSettingsPrecedence.TASK_TOML,
                profile="workspace-readonly",
                escalation=ContainerEscalationPolicy(
                    mode=ContainerEscalationMode.REQUIRE_REVIEW,
                ),
            ),
        )

        for attempt in attempts:
            with self.subTest(attempt=attempt.to_dict()):
                with self.assertRaises(AssertionError):
                    caps.merge((attempt,))

    def test_merge_rejects_trusted_specificity_widening(self) -> None:
        caps = ContainerAuthorityCaps(
            settings=_authority_settings(ContainerSurface.CLI),
        )
        source = _trusted_source(ContainerSurface.SDK)
        attempts = (
            ContainerSettingsOverride(
                source=source,
                layer=ContainerSettingsPrecedence.SDK,
                profile="workspace-rich",
                image=ContainerImagePolicy(
                    reference=f"ghcr.io/example/other@sha256:{_DIGEST}",
                ),
            ),
            ContainerSettingsOverride(
                source=source,
                layer=ContainerSettingsPrecedence.SDK,
                profile="workspace-rich",
                workspace=ContainerWorkspaceMapping(
                    host_root="other",
                    container_path="/workspace",
                ),
            ),
            ContainerSettingsOverride(
                source=source,
                layer=ContainerSettingsPrecedence.SDK,
                profile="workspace-rich",
                command_mode=ContainerCommandMode.FIXED_ENTRYPOINT,
            ),
            ContainerSettingsOverride(
                source=source,
                layer=ContainerSettingsPrecedence.SDK,
                profile="workspace-rich",
                read_only_rootfs=False,
            ),
            ContainerSettingsOverride(
                source=source,
                layer=ContainerSettingsPrecedence.SDK,
                profile="workspace-rich",
                user="2000:2000",
            ),
            ContainerSettingsOverride(
                source=source,
                layer=ContainerSettingsPrecedence.SDK,
                profile="workspace-rich",
                backend=ContainerBackend.PODMAN,
            ),
        )

        for attempt in attempts:
            with self.subTest(attempt=attempt.to_dict()):
                with self.assertRaises(AssertionError):
                    caps.merge((attempt,))


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


def _untrusted_source(
    surface: ContainerSurface,
    trust_level: ContainerTrustLevel,
) -> ContainerSettingsSource:
    return ContainerSettingsSource(
        surface=surface,
        trust_level=trust_level,
    )


def _readonly_profile() -> ContainerProfile:
    return ContainerProfile.minimal_readonly(
        name="workspace-readonly",
        image_reference=_IMAGE,
    )


def _rich_profile(*, reverse_order: bool = False) -> ContainerProfile:
    mounts = (
        ContainerMountDeclaration(
            source=".",
            target="/workspace",
            mount_type=ContainerMountType.WORKSPACE,
        ),
        ContainerMountDeclaration(
            source="out",
            target="/out",
            mount_type=ContainerMountType.OUTPUT,
            access=ContainerMountAccess.WRITE,
        ),
    )
    env_variables = {
        "LC_ALL": "C.UTF-8",
        "TOOL_MODE": "safe",
    }
    env_allowlist = ("PATH", "PYTHONPATH")
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
    egress_allowlist = ("api.example.test", "cdn.example.test")
    devices = (
        ContainerDeviceClass.CPU,
        ContainerDeviceClass.NVIDIA_CDI,
    )
    if reverse_order:
        mounts = tuple(reversed(mounts))
        env_variables = {
            "TOOL_MODE": "safe",
            "LC_ALL": "C.UTF-8",
        }
        env_allowlist = tuple(reversed(env_allowlist))
        secrets = tuple(reversed(secrets))
        egress_allowlist = tuple(reversed(egress_allowlist))
        devices = tuple(reversed(devices))
    return ContainerProfile(
        name="workspace-rich",
        image=ContainerImagePolicy(reference=_IMAGE),
        workspace=ContainerWorkspaceMapping(
            host_root=".",
            container_path="/workspace",
            working_directory="/workspace",
        ),
        mounts=mounts,
        environment=ContainerEnvironmentPolicy(
            variables=env_variables,
            allowlist=env_allowlist,
        ),
        secrets=secrets,
        network=ContainerNetworkPolicy(
            mode=ContainerNetworkMode.ALLOWLIST,
            egress_allowlist=egress_allowlist,
        ),
        devices=ContainerDevicePolicy(devices=devices),
        resources=ContainerResourceLimits(
            cpu_count=4,
            memory_bytes=536870912,
            pids=128,
            timeout_seconds=60,
        ),
        output=ContainerOutputPolicy(
            max_stdout_bytes=8192,
            max_stderr_bytes=4096,
            max_artifact_bytes=1024,
            allow_artifacts=True,
        ),
        cleanup=ContainerCleanupPolicy(
            mode=ContainerCleanupMode.QUARANTINE,
            grace_seconds=8,
        ),
        audit=ContainerAuditPolicy(mode=ContainerAuditMode.FULL),
        escalation=ContainerEscalationPolicy(
            mode=ContainerEscalationMode.PREAUTHORIZED,
        ),
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


def _authority_settings(
    surface: ContainerSurface,
    allowed_profiles: tuple[str, ...] | None = None,
    rich_profile: ContainerProfile | None = None,
) -> ContainerSettings:
    readonly = _readonly_profile()
    rich = rich_profile or _rich_profile()
    profile_names = allowed_profiles or (readonly.name, rich.name)
    return ContainerSettings(
        source=_trusted_source(surface),
        backend=ContainerBackend.DOCKER,
        default_profile=readonly.name,
        allowed_profiles=profile_names,
        profiles={
            readonly.name: readonly,
            rich.name: rich,
        },
        profile_registry_id="unit-registry",
        policy_version="phase1",
    )


if __name__ == "__main__":
    main()
