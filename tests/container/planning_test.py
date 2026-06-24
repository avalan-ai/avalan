from unittest import TestCase, main

from avalan.container import (
    ContainerAuthorityCaps,
    ContainerBackend,
    ContainerCommandMode,
    ContainerDeviceClass,
    ContainerDevicePolicy,
    ContainerDurablePlanKind,
    ContainerDurablePlanMetadata,
    ContainerEffectiveSettings,
    ContainerEnvironmentPolicy,
    ContainerExecutionScope,
    ContainerImagePolicy,
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
    ContainerRuntimeEnvelopeKind,
    ContainerSecretReference,
    ContainerSettings,
    ContainerSettingsSource,
    ContainerSurface,
    ContainerTrustLevel,
    normalize_container_run_plan,
    normalize_runtime_envelope_plan,
)

_DIGEST = "2" * 64
_DIGEST_ALT = "3" * 64
_IMAGE = f"ghcr.io/example/plan-tools@sha256:{_DIGEST}"


class ContainerPlanningTest(TestCase):
    def test_all_request_kinds_normalize_to_command_plans(self) -> None:
        effective = _effective_settings()

        for request_kind in ContainerPlanRequestKind:
            with self.subTest(request_kind=request_kind.value):
                request = _request(
                    request_kind=request_kind,
                    attempt_id=(
                        "attempt.1"
                        if (
                            request_kind
                            is ContainerPlanRequestKind.TASK_ATTEMPT
                        )
                        else None
                    ),
                    scope=(
                        ContainerExecutionScope.RUNTIME_ENVELOPE
                        if (
                            request_kind
                            is ContainerPlanRequestKind.RUNTIME_ENVELOPE
                        )
                        else ContainerExecutionScope.SHELL_CONTAINER_EXECUTION
                    ),
                )

                plan = normalize_container_run_plan(effective, request)

                self.assertEqual(
                    plan.run_plan.command.tool_name,
                    f"{request_kind.value}:shell.rg",
                )
                self.assertEqual(plan.run_plan.command.cwd, "/workspace/app")
                self.assertEqual(
                    plan.canonical_policy_input()["request_kind"],
                    request_kind.value,
                )

    def test_equivalent_run_plans_fingerprint_and_reload(self) -> None:
        first = normalize_container_run_plan(
            _effective_settings(profile=_rich_profile(reverse=False)),
            _request(request_id="tool.call.1"),
        )
        second = normalize_container_run_plan(
            _effective_settings(profile=_rich_profile(reverse=True)),
            _request(
                request_id="tool.call.2",
                cwd="/workspace/app/../app",
            ),
        )

        loaded = ContainerNormalizedRunPlan.from_dict(first.to_dict())
        metadata = ContainerDurablePlanMetadata.from_dict(
            first.to_metadata().to_dict()
        )

        self.assertEqual(first.plan_fingerprint, second.plan_fingerprint)
        self.assertEqual(loaded.plan_fingerprint, first.plan_fingerprint)
        self.assertEqual(
            first.canonical_policy_input()["environment_names"],
            ["HOME", "LC_ALL", "PATH"],
        )
        metadata.assert_matches(first)

    def test_higher_risk_plan_changes_require_fresh_decisions(self) -> None:
        base = normalize_container_run_plan(
            _effective_settings(),
            _request(request_id="tool.call.1"),
        )
        higher_risk = normalize_container_run_plan(
            _effective_settings(
                profile=_profile(
                    network=ContainerNetworkPolicy(
                        mode=ContainerNetworkMode.ALLOWLIST,
                        egress_allowlist=("api.example.test",),
                    ),
                    resources=ContainerResourceLimits(cpu_count=4),
                )
            ),
            _request(request_id="tool.call.1"),
        )
        metadata = base.to_metadata()

        self.assertNotEqual(
            base.plan_fingerprint,
            higher_risk.plan_fingerprint,
        )
        with self.assertRaises(AssertionError):
            metadata.assert_matches(higher_risk)

    def test_mutable_image_requires_resolved_digest(self) -> None:
        profile = _profile(
            image=ContainerImagePolicy(
                reference="ghcr.io/example/plan-tools:latest",
                digest=f"sha256:{_DIGEST}",
            )
        )
        effective = _effective_settings(profile=profile)

        with self.assertRaises(AssertionError):
            normalize_container_run_plan(effective, _request())

        plan = normalize_container_run_plan(
            effective,
            _request(),
            resolved_image_digest=f"sha256:{_DIGEST_ALT}",
        )

        self.assertEqual(plan.run_plan.image.digest, f"sha256:{_DIGEST_ALT}")
        self.assertEqual(
            plan.canonical_policy_input()["image"],
            {
                "build_cache": {
                    "allow_stale": False,
                    "mode": "disabled",
                    "require_context_digest_key": True,
                    "ttl_seconds": 0,
                },
                "build_context": None,
                "build_policy": "disabled",
                "digest": f"sha256:{_DIGEST_ALT}",
                "image_cache": {
                    "allow_stale": False,
                    "mode": "disabled",
                    "require_digest_key": True,
                    "ttl_seconds": 0,
                },
                "platform": "linux/amd64",
                "pull_policy": "never",
                "reference": "ghcr.io/example/plan-tools:latest",
                "reference_is_mutable": True,
            },
        )

    def test_mutable_and_pinned_image_identity_do_not_collide(self) -> None:
        pinned = normalize_container_run_plan(
            _effective_settings(),
            _request(request_id="tool.call.1"),
        )
        mutable = normalize_container_run_plan(
            _effective_settings(
                profile=_profile(
                    image=ContainerImagePolicy(
                        reference="ghcr.io/example/plan-tools:latest",
                        digest=f"sha256:{_DIGEST}",
                    )
                )
            ),
            _request(request_id="tool.call.1"),
            resolved_image_digest=f"sha256:{_DIGEST}",
        )
        metadata = pinned.to_metadata()

        self.assertNotEqual(pinned.plan_fingerprint, mutable.plan_fingerprint)
        self.assertFalse(
            pinned.canonical_policy_input()["image"]["reference_is_mutable"]
        )
        self.assertTrue(
            mutable.canonical_policy_input()["image"]["reference_is_mutable"]
        )
        with self.assertRaises(AssertionError):
            metadata.assert_matches(mutable)

    def test_runtime_envelope_kinds_fingerprint_and_reload(self) -> None:
        effective = _effective_settings()

        for envelope_kind in ContainerRuntimeEnvelopeKind:
            with self.subTest(envelope_kind=envelope_kind.value):
                request = _request(
                    request_kind=ContainerPlanRequestKind.RUNTIME_ENVELOPE,
                    scope=ContainerExecutionScope.RUNTIME_ENVELOPE,
                    request_id=f"{envelope_kind.value}.1",
                )
                plan = normalize_runtime_envelope_plan(
                    effective,
                    request,
                    envelope_kind=envelope_kind,
                    readiness_timeout_seconds=45,
                )
                loaded = ContainerNormalizedRuntimeEnvelopePlan.from_dict(
                    plan.to_dict()
                )

                self.assertEqual(
                    plan.plan_fingerprint,
                    loaded.plan_fingerprint,
                )
                self.assertEqual(
                    plan.canonical_policy_input()["envelope_kind"],
                    envelope_kind.value,
                )
                plan.to_metadata().assert_matches(plan)

    def test_fake_e2e_worker_metadata_revalidates_before_execution(
        self,
    ) -> None:
        worker_plan = normalize_container_run_plan(
            _effective_settings(),
            _request(
                request_kind=ContainerPlanRequestKind.TASK_ATTEMPT,
                request_id="task.1",
                attempt_id="attempt.1",
            ),
        )
        reloaded_metadata = ContainerDurablePlanMetadata.from_dict(
            worker_plan.to_metadata().to_dict()
        )
        changed_attempt_plan = normalize_container_run_plan(
            _effective_settings(),
            _request(
                request_kind=ContainerPlanRequestKind.TASK_ATTEMPT,
                request_id="task.1",
                attempt_id="attempt.2",
            ),
        )

        reloaded_metadata.assert_matches(worker_plan)
        with self.assertRaises(AssertionError):
            reloaded_metadata.assert_matches(changed_attempt_plan)

    def test_fake_e2e_flow_resume_metadata_revalidates_envelope(self) -> None:
        envelope_plan = normalize_runtime_envelope_plan(
            _effective_settings(),
            _request(
                request_kind=ContainerPlanRequestKind.RUNTIME_ENVELOPE,
                request_id="flow.1",
                scope=ContainerExecutionScope.RUNTIME_ENVELOPE,
            ),
            envelope_kind=ContainerRuntimeEnvelopeKind.FLOW_RUNTIME,
        )
        reloaded_metadata = ContainerDurablePlanMetadata.from_dict(
            envelope_plan.to_metadata().to_dict()
        )
        changed_envelope = normalize_runtime_envelope_plan(
            _effective_settings(),
            _request(
                request_kind=ContainerPlanRequestKind.RUNTIME_ENVELOPE,
                request_id="flow.1",
                scope=ContainerExecutionScope.RUNTIME_ENVELOPE,
            ),
            envelope_kind=ContainerRuntimeEnvelopeKind.FLOW_RUNTIME,
            readiness_timeout_seconds=90,
        )

        self.assertEqual(
            reloaded_metadata.plan_kind,
            ContainerDurablePlanKind.RUNTIME_ENVELOPE,
        )
        reloaded_metadata.assert_matches(envelope_plan)
        with self.assertRaises(AssertionError):
            reloaded_metadata.assert_matches(changed_envelope)
        with self.assertRaises(AssertionError):
            envelope_plan.run_plan.to_metadata().assert_matches(envelope_plan)

    def test_invalid_planning_inputs_are_rejected(self) -> None:
        minimal = ContainerPlanRequest.from_dict(
            {
                "request_kind": "typed_tool",
                "logical_name": "shell.rg",
                "command": "rg",
                "argv": ("rg", "needle"),
            }
        )

        self.assertEqual(minimal.cwd, "/workspace")
        with self.assertRaises(AssertionError):
            ContainerPlanRequest.from_dict(
                {
                    "request_kind": "typed_tool",
                    "logical_name": "shell.rg",
                    "command": "rg",
                    "argv": ("rg", "needle"),
                    "unknown": True,
                }
            )
        with self.assertRaises(AssertionError):
            ContainerPlanRequest(
                request_kind=ContainerPlanRequestKind.TASK_ATTEMPT,
                logical_name="shell.rg",
                command="rg",
                argv=("rg", "needle"),
            )
        with self.assertRaises(AssertionError):
            normalize_container_run_plan(
                ContainerAuthorityCaps(
                    settings=ContainerSettings(
                        source=ContainerSettingsSource(
                            surface=ContainerSurface.CLI,
                            trust_level=ContainerTrustLevel.TRUSTED_OPERATOR,
                        )
                    )
                ).merge(()),
                _request(),
            )


def _request(
    *,
    request_kind: ContainerPlanRequestKind = (
        ContainerPlanRequestKind.TYPED_TOOL
    ),
    request_id: str | None = None,
    attempt_id: str | None = None,
    cwd: str = "/workspace/./app",
    scope: ContainerExecutionScope = (
        ContainerExecutionScope.SHELL_CONTAINER_EXECUTION
    ),
) -> ContainerPlanRequest:
    return ContainerPlanRequest(
        request_kind=request_kind,
        logical_name="shell.rg",
        command="rg",
        argv=("rg", "needle"),
        cwd=cwd,
        scope=scope,
        request_id=request_id,
        attempt_id=attempt_id,
    )


def _effective_settings(
    *,
    profile: ContainerProfile | None = None,
) -> ContainerEffectiveSettings:
    profile = profile or _profile()
    settings = ContainerSettings(
        source=ContainerSettingsSource(
            surface=ContainerSurface.CLI,
            trust_level=ContainerTrustLevel.TRUSTED_OPERATOR,
        ),
        backend=ContainerBackend.DOCKER,
        default_profile=profile.name,
        allowed_profiles=(profile.name,),
        profiles={profile.name: profile},
        profile_registry_id="planning-registry",
        policy_version="phase4",
    )
    return ContainerAuthorityCaps(settings=settings).merge(())


def _profile(
    *,
    image: ContainerImagePolicy | None = None,
    network: ContainerNetworkPolicy | None = None,
    resources: ContainerResourceLimits | None = None,
) -> ContainerProfile:
    return ContainerProfile(
        name="planning-profile",
        image=image or ContainerImagePolicy(reference=_IMAGE),
        command_mode=ContainerCommandMode.FIXED_EXECUTABLE,
        network=network or ContainerNetworkPolicy(),
        resources=resources or ContainerResourceLimits(),
    )


def _rich_profile(*, reverse: bool) -> ContainerProfile:
    mounts = (
        ContainerMountDeclaration(
            source="out",
            target="/out",
            mount_type=ContainerMountType.OUTPUT,
            access=ContainerMountAccess.WRITE,
        ),
        ContainerMountDeclaration(
            source="cache",
            target="/cache",
            mount_type=ContainerMountType.CACHE,
        ),
    )
    secrets = (
        ContainerSecretReference(name="api-token", env_name="API_TOKEN"),
        ContainerSecretReference(
            name="mounted-token",
            mount_path="/run/secrets/token",
        ),
    )
    egress = ("z.example.test", "a.example.test")
    devices = (ContainerDeviceClass.CPU, ContainerDeviceClass.NVIDIA_CDI)
    if reverse:
        mounts = tuple(reversed(mounts))
        secrets = tuple(reversed(secrets))
        egress = tuple(reversed(egress))
        devices = tuple(reversed(devices))
    return ContainerProfile(
        name="planning-profile",
        image=ContainerImagePolicy(reference=_IMAGE),
        command_mode=ContainerCommandMode.FIXED_EXECUTABLE,
        mounts=mounts,
        environment=ContainerEnvironmentPolicy(
            variables={"LC_ALL": "C.UTF-8"},
            allowlist=("PATH", "HOME"),
        ),
        secrets=secrets,
        network=ContainerNetworkPolicy(
            mode=ContainerNetworkMode.ALLOWLIST,
            egress_allowlist=egress,
        ),
        devices=ContainerDevicePolicy(devices=devices),
        resources=ContainerResourceLimits(
            cpu_count=2,
            memory_bytes=536870912,
            pids=128,
            timeout_seconds=30,
        ),
    )


if __name__ == "__main__":
    main()
