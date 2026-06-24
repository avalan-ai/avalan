from unittest import TestCase, main

from avalan.container import (
    ContainerApprovalRecord,
    ContainerAuthorityCaps,
    ContainerAuthorizationDecisionType,
    ContainerBackend,
    ContainerBuildPolicy,
    ContainerDeviceClass,
    ContainerDevicePolicy,
    ContainerEffectiveSettings,
    ContainerEscalationMode,
    ContainerEscalationPolicy,
    ContainerEscalationTrigger,
    ContainerImagePolicy,
    ContainerMountAccess,
    ContainerMountDeclaration,
    ContainerMountType,
    ContainerNetworkMode,
    ContainerNetworkPolicy,
    ContainerPolicy,
    ContainerPolicyContext,
    ContainerPolicyPlan,
    ContainerProfile,
    ContainerResourceLimits,
    ContainerReviewMode,
    ContainerReviewSurface,
    ContainerSecretReference,
    ContainerSettings,
    ContainerSettingsSource,
    ContainerSurface,
    ContainerTrustLevel,
)

_DIGEST = "1" * 64
_IMAGE = f"ghcr.io/example/policy-tools@sha256:{_DIGEST}"


class ContainerPolicyTest(TestCase):
    def test_preauthorized_profile_allows_low_risk_plan(self) -> None:
        policy = _policy()
        plan = _plan(escalation=ContainerEscalationMode.DENY)

        decision = policy.authorize(plan)

        self.assertEqual(
            decision.decision,
            ContainerAuthorizationDecisionType.ALLOW,
        )
        self.assertEqual(
            decision.code,
            "container.allow.preauthorized_profile",
        )
        self.assertEqual(decision.policy_version, "phase3")
        self.assertEqual(decision.profile_name, "policy-profile")
        self.assertTrue(decision.cacheable)
        self.assertFalse(decision.retryable)

    def test_preauthorized_escalation_allows_triggered_plan(self) -> None:
        plan = _plan(
            escalation=ContainerEscalationMode.PREAUTHORIZED,
            triggers=(ContainerEscalationTrigger.NETWORK,),
        )

        decision = _policy().authorize(plan)

        self.assertEqual(
            decision.decision,
            ContainerAuthorizationDecisionType.ALLOW,
        )
        self.assertEqual(
            decision.code,
            "container.allow.preauthorized_escalation",
        )
        self.assertTrue(decision.cacheable)

    def test_review_required_decisions_are_surface_specific(self) -> None:
        expected = {
            ContainerReviewSurface.INTERACTIVE_CLI: (
                ContainerReviewMode.INTERACTIVE,
                "container.review.interactive",
                ContainerAuthorizationDecisionType.REQUIRES_REVIEW,
            ),
            ContainerReviewSurface.STRICT_FLOW: (
                ContainerReviewMode.DURABLE,
                "container.review.durable",
                ContainerAuthorizationDecisionType.REQUIRES_REVIEW,
            ),
            ContainerReviewSurface.QUEUED_TASK: (
                ContainerReviewMode.DURABLE,
                "container.review.durable",
                ContainerAuthorizationDecisionType.REQUIRES_REVIEW,
            ),
            ContainerReviewSurface.DIRECT_TASK: (
                ContainerReviewMode.FAIL_CLOSED,
                "container.deny.review_unavailable",
                ContainerAuthorizationDecisionType.DENY,
            ),
            ContainerReviewSurface.SERVER: (
                ContainerReviewMode.FAIL_CLOSED,
                "container.deny.review_unavailable",
                ContainerAuthorizationDecisionType.DENY,
            ),
            ContainerReviewSurface.MCP: (
                ContainerReviewMode.FAIL_CLOSED,
                "container.deny.review_unavailable",
                ContainerAuthorizationDecisionType.DENY,
            ),
            ContainerReviewSurface.A2A: (
                ContainerReviewMode.FAIL_CLOSED,
                "container.deny.review_unavailable",
                ContainerAuthorizationDecisionType.DENY,
            ),
        }

        for surface, (mode, code, decision_type) in expected.items():
            with self.subTest(surface=surface.value):
                context = ContainerPolicyContext(
                    surface=surface.value,
                    scope_id=f"scope.{surface.value}",
                    attempt_id="attempt.1",
                )
                plan = _plan(
                    escalation=ContainerEscalationMode.REQUIRE_REVIEW,
                    surface=surface,
                    triggers=(ContainerEscalationTrigger.SECRET,),
                )
                decision = _policy().authorize(plan)

                self.assertEqual(context.review_mode, mode)
                self.assertEqual(context.to_dict()["surface"], surface.value)
                self.assertEqual(decision.code, code)
                self.assertEqual(decision.decision, decision_type)

    def test_review_decision_uses_privacy_safe_override_explanation(
        self,
    ) -> None:
        policy = ContainerPolicy(
            policy_version="phase3",
            explanations={
                "container.review.durable": "Durable review required.",
            },
        )
        plan = _plan(
            escalation=ContainerEscalationMode.REQUIRE_REVIEW,
            surface=ContainerReviewSurface.STRICT_FLOW,
            triggers=(ContainerEscalationTrigger.WRITE_MOUNT,),
        )

        decision = policy.authorize(plan)

        self.assertEqual(decision.explanation, "Durable review required.")
        self.assertTrue(decision.retryable)
        self.assertFalse(decision.cacheable)

    def test_scoped_approval_allows_equivalent_cached_plan(self) -> None:
        plan = _plan(
            escalation=ContainerEscalationMode.REQUIRE_REVIEW,
            surface=ContainerReviewSurface.STRICT_FLOW,
            triggers=(
                ContainerEscalationTrigger.SECRET,
                ContainerEscalationTrigger.NETWORK,
            ),
        )
        equivalent = _plan(
            escalation=ContainerEscalationMode.REQUIRE_REVIEW,
            surface=ContainerReviewSurface.STRICT_FLOW,
            triggers=(
                ContainerEscalationTrigger.NETWORK,
                ContainerEscalationTrigger.SECRET,
            ),
        )
        approval = ContainerApprovalRecord.for_plan(
            plan,
            reviewer_identity="operator@example.test",
            expires_at_seconds=100,
        )

        decision = _policy().authorize(
            equivalent,
            approval=approval,
            now_seconds=10,
        )

        self.assertEqual(plan.plan_fingerprint, equivalent.plan_fingerprint)
        self.assertEqual(
            decision.decision,
            ContainerAuthorizationDecisionType.ALLOW,
        )
        self.assertEqual(decision.code, "container.allow.cached_approval")
        self.assertEqual(
            approval.to_dict()["approved_triggers"],
            ["combined_risk", "network", "secret"],
        )

    def test_plan_fingerprint_includes_review_surface_category(
        self,
    ) -> None:
        cli = _plan(
            surface=ContainerReviewSurface.INTERACTIVE_CLI,
            triggers=(
                ContainerEscalationTrigger.SECRET,
                ContainerEscalationTrigger.NETWORK,
            ),
            scope_id="cli.scope.1",
        )
        equivalent_cli = _plan(
            surface=ContainerReviewSurface.INTERACTIVE_CLI,
            triggers=(
                ContainerEscalationTrigger.NETWORK,
                ContainerEscalationTrigger.SECRET,
            ),
            attempt_id="attempt.2",
            scope_id="cli.scope.2",
        )
        server = _plan(
            surface=ContainerReviewSurface.SERVER,
            triggers=(
                ContainerEscalationTrigger.NETWORK,
                ContainerEscalationTrigger.SECRET,
            ),
            attempt_id="attempt.2",
            scope_id="server.scope.1",
        )

        self.assertEqual(cli.plan_fingerprint, equivalent_cli.plan_fingerprint)
        self.assertNotEqual(cli.plan_fingerprint, server.plan_fingerprint)
        self.assertEqual(
            cli.canonical_policy_input()["review_surface"],
            "interactive_cli",
        )
        self.assertEqual(
            server.canonical_policy_input()["review_mode"],
            "fail_closed",
        )

    def test_profile_risks_are_derived_when_triggers_are_omitted(
        self,
    ) -> None:
        cases = (
            (
                _profile(
                    mounts=(
                        ContainerMountDeclaration(
                            target="/scratch",
                            mount_type=ContainerMountType.SCRATCH,
                            access=ContainerMountAccess.WRITE,
                        ),
                    )
                ),
                ContainerEscalationTrigger.WRITE_MOUNT,
            ),
            (
                _profile(
                    secrets=(
                        ContainerSecretReference(
                            name="api-token",
                            env_name="API_TOKEN",
                        ),
                    )
                ),
                ContainerEscalationTrigger.SECRET,
            ),
            (
                _profile(
                    mounts=(
                        ContainerMountDeclaration(
                            target="/run/secrets/api-token",
                            mount_type=ContainerMountType.SECRET,
                        ),
                    )
                ),
                ContainerEscalationTrigger.SECRET,
            ),
            (
                _profile(
                    network=ContainerNetworkPolicy(
                        mode=ContainerNetworkMode.ALLOWLIST,
                        egress_allowlist=("api.example.test",),
                    )
                ),
                ContainerEscalationTrigger.NETWORK,
            ),
            (
                _profile(
                    devices=ContainerDevicePolicy(
                        devices=(ContainerDeviceClass.CPU,),
                    )
                ),
                ContainerEscalationTrigger.DEVICE,
            ),
            (
                _profile(
                    mounts=(
                        ContainerMountDeclaration(
                            source="/var/run/docker.sock",
                            target="/docker.sock",
                            mount_type=ContainerMountType.INPUT,
                        ),
                    )
                ),
                ContainerEscalationTrigger.RUNTIME_SOCKET,
            ),
            (
                _profile(
                    mounts=(
                        ContainerMountDeclaration(
                            source="/var/run/../run/docker.sock",
                            target="/docker.sock",
                            mount_type=ContainerMountType.INPUT,
                        ),
                    )
                ),
                ContainerEscalationTrigger.RUNTIME_SOCKET,
            ),
            (
                _profile(
                    mounts=(
                        ContainerMountDeclaration(
                            source="/run/../var/run/podman/podman.sock",
                            target="/podman.sock",
                            mount_type=ContainerMountType.INPUT,
                        ),
                    )
                ),
                ContainerEscalationTrigger.RUNTIME_SOCKET,
            ),
            (
                _profile(
                    mounts=(
                        ContainerMountDeclaration(
                            source="/private/var/run/docker.sock",
                            target="/docker.sock",
                            mount_type=ContainerMountType.INPUT,
                        ),
                    )
                ),
                ContainerEscalationTrigger.RUNTIME_SOCKET,
            ),
            (
                _profile(
                    mounts=(
                        ContainerMountDeclaration(
                            source=(
                                "/private/var/run/../run/podman/podman.sock"
                            ),
                            target="/podman.sock",
                            mount_type=ContainerMountType.INPUT,
                        ),
                    )
                ),
                ContainerEscalationTrigger.RUNTIME_SOCKET,
            ),
            (
                _profile(
                    image=ContainerImagePolicy(
                        reference="ghcr.io/example/policy-tools:latest",
                        digest=f"sha256:{_DIGEST}",
                    )
                ),
                ContainerEscalationTrigger.MUTABLE_IMAGE,
            ),
            (
                _profile(
                    image=ContainerImagePolicy(
                        reference=_IMAGE,
                        build_policy=ContainerBuildPolicy.TRUSTED_ONLY,
                    )
                ),
                ContainerEscalationTrigger.BUILD,
            ),
            (
                _profile(resources=ContainerResourceLimits(cpu_count=2)),
                ContainerEscalationTrigger.RESOURCE_INCREASE,
            ),
        )

        for profile, trigger in cases:
            with self.subTest(trigger=trigger.value):
                plan = _plan(
                    escalation=ContainerEscalationMode.DENY,
                    profile=profile,
                )
                decision = _policy().authorize(plan)

                self.assertIn(trigger, plan.escalation_triggers)
                self.assertEqual(
                    decision.decision,
                    ContainerAuthorizationDecisionType.DENY,
                )
                self.assertEqual(decision.code, "container.deny.escalation")

    def test_malformed_effective_root_user_derives_root_trigger(self) -> None:
        profile = _profile()
        object.__setattr__(profile, "user", "0")
        plan = _plan(
            escalation=ContainerEscalationMode.DENY,
            profile=profile,
            preserve_profile=True,
        )

        decision = _policy().authorize(plan)

        self.assertIn(
            ContainerEscalationTrigger.ROOT_USER,
            plan.escalation_triggers,
        )
        self.assertEqual(
            decision.decision,
            ContainerAuthorizationDecisionType.DENY,
        )

    def test_combined_risk_is_derived_when_multiple_risks_are_omitted(
        self,
    ) -> None:
        plan = _plan(
            escalation=ContainerEscalationMode.DENY,
            profile=_profile(
                secrets=(
                    ContainerSecretReference(
                        name="api-token",
                        env_name="API_TOKEN",
                    ),
                ),
                network=ContainerNetworkPolicy(
                    mode=ContainerNetworkMode.ALLOWLIST,
                    egress_allowlist=("api.example.test",),
                ),
            ),
        )

        decision = _policy().authorize(plan)

        self.assertEqual(
            plan.escalation_triggers,
            (
                ContainerEscalationTrigger.COMBINED_RISK,
                ContainerEscalationTrigger.NETWORK,
                ContainerEscalationTrigger.SECRET,
            ),
        )
        self.assertEqual(
            decision.decision,
            ContainerAuthorizationDecisionType.DENY,
        )

    def test_mutable_image_preauthorized_durable_plans_fail_image_trust(
        self,
    ) -> None:
        for surface in (
            ContainerReviewSurface.STRICT_FLOW,
            ContainerReviewSurface.QUEUED_TASK,
        ):
            with self.subTest(surface=surface.value):
                plan = _plan(
                    escalation=ContainerEscalationMode.PREAUTHORIZED,
                    surface=surface,
                    profile=_profile(
                        image=ContainerImagePolicy(
                            reference="ghcr.io/example/policy-tools:latest",
                            digest=f"sha256:{_DIGEST}",
                        )
                    ),
                )

                decision = _policy().authorize(plan)

                self.assertEqual(
                    plan.escalation_triggers,
                    (
                        ContainerEscalationTrigger.COMBINED_RISK,
                        ContainerEscalationTrigger.MUTABLE_IMAGE,
                    ),
                )
                self.assertEqual(
                    decision.decision,
                    ContainerAuthorizationDecisionType.DENY,
                )
                self.assertEqual(decision.code, "container.deny.image_trust")

    def test_mutable_image_preauthorized_server_plan_fails_image_trust(
        self,
    ) -> None:
        plan = _plan(
            escalation=ContainerEscalationMode.PREAUTHORIZED,
            surface=ContainerReviewSurface.SERVER,
            profile=_profile(
                image=ContainerImagePolicy(
                    reference="ghcr.io/example/policy-tools:latest",
                    digest=f"sha256:{_DIGEST}",
                )
            ),
        )

        decision = _policy().authorize(plan)

        self.assertEqual(
            plan.escalation_triggers,
            (
                ContainerEscalationTrigger.COMBINED_RISK,
                ContainerEscalationTrigger.MUTABLE_IMAGE,
            ),
        )
        self.assertEqual(
            decision.decision,
            ContainerAuthorizationDecisionType.DENY,
        )
        self.assertEqual(decision.code, "container.deny.image_trust")

    def test_all_escalation_triggers_deny_without_review_authority(
        self,
    ) -> None:
        for trigger in ContainerEscalationTrigger:
            with self.subTest(trigger=trigger.value):
                plan = _plan(
                    escalation=ContainerEscalationMode.DENY,
                    triggers=(trigger,),
                )
                decision = _policy().authorize(plan)

                self.assertEqual(
                    decision.decision,
                    ContainerAuthorizationDecisionType.DENY,
                )
                self.assertEqual(decision.code, "container.deny.escalation")

    def test_stale_and_mismatched_approvals_are_rejected(self) -> None:
        plan = _plan(
            escalation=ContainerEscalationMode.REQUIRE_REVIEW,
            surface=ContainerReviewSurface.STRICT_FLOW,
            triggers=(ContainerEscalationTrigger.SECRET,),
        )
        approval = ContainerApprovalRecord.for_plan(
            plan,
            reviewer_identity="operator@example.test",
            expires_at_seconds=10,
        )
        mismatches = (
            (approval, plan, 10),
            (
                ContainerApprovalRecord(
                    reviewer_identity="operator@example.test",
                    review_mode=ContainerReviewMode.INTERACTIVE,
                    plan_fingerprint=plan.plan_fingerprint,
                    scope_id=plan.context.scope_id,
                    attempt_id=plan.context.attempt_id,
                    profile_name=plan.profile_name,
                    policy_version=plan.policy_version,
                    expires_at_seconds=100,
                    approved_triggers=plan.escalation_triggers,
                ),
                plan,
                1,
            ),
            (
                ContainerApprovalRecord(
                    reviewer_identity="operator@example.test",
                    review_mode=plan.context.review_mode,
                    plan_fingerprint="0" * 64,
                    scope_id=plan.context.scope_id,
                    attempt_id=plan.context.attempt_id,
                    profile_name=plan.profile_name,
                    policy_version=plan.policy_version,
                    expires_at_seconds=100,
                    approved_triggers=plan.escalation_triggers,
                ),
                plan,
                1,
            ),
            (
                ContainerApprovalRecord(
                    reviewer_identity="operator@example.test",
                    review_mode=plan.context.review_mode,
                    plan_fingerprint=plan.plan_fingerprint,
                    scope_id="flow",
                    attempt_id=plan.context.attempt_id,
                    profile_name=plan.profile_name,
                    policy_version=plan.policy_version,
                    expires_at_seconds=100,
                    approved_triggers=plan.escalation_triggers,
                ),
                plan,
                1,
            ),
            (
                ContainerApprovalRecord(
                    reviewer_identity="operator@example.test",
                    review_mode=plan.context.review_mode,
                    plan_fingerprint=plan.plan_fingerprint,
                    scope_id=plan.context.scope_id,
                    attempt_id="attempt.2",
                    profile_name=plan.profile_name,
                    policy_version=plan.policy_version,
                    expires_at_seconds=100,
                    approved_triggers=plan.escalation_triggers,
                ),
                plan,
                1,
            ),
            (
                ContainerApprovalRecord(
                    reviewer_identity="operator@example.test",
                    review_mode=plan.context.review_mode,
                    plan_fingerprint=plan.plan_fingerprint,
                    scope_id=plan.context.scope_id,
                    attempt_id=plan.context.attempt_id,
                    profile_name="other-profile",
                    policy_version=plan.policy_version,
                    expires_at_seconds=100,
                    approved_triggers=plan.escalation_triggers,
                ),
                plan,
                1,
            ),
            (
                ContainerApprovalRecord(
                    reviewer_identity="operator@example.test",
                    review_mode=plan.context.review_mode,
                    plan_fingerprint=plan.plan_fingerprint,
                    scope_id=plan.context.scope_id,
                    attempt_id=plan.context.attempt_id,
                    profile_name=plan.profile_name,
                    policy_version="phase2",
                    expires_at_seconds=100,
                    approved_triggers=plan.escalation_triggers,
                ),
                plan,
                1,
            ),
            (
                ContainerApprovalRecord(
                    reviewer_identity="operator@example.test",
                    review_mode=plan.context.review_mode,
                    plan_fingerprint=plan.plan_fingerprint,
                    scope_id=plan.context.scope_id,
                    attempt_id=plan.context.attempt_id,
                    profile_name=plan.profile_name,
                    policy_version=plan.policy_version,
                    expires_at_seconds=100,
                    approved_triggers=(),
                ),
                plan,
                1,
            ),
        )

        for record, target_plan, now_seconds in mismatches:
            with self.subTest(record=record.to_dict()):
                with self.assertRaises(AssertionError):
                    _policy().authorize(
                        target_plan,
                        approval=record,
                        now_seconds=now_seconds,
                    )

    def test_invalid_policy_values_are_rejected(self) -> None:
        with self.assertRaises(AssertionError):
            ContainerPolicyContext(surface="unknown", scope_id="scope")
        with self.assertRaises(AssertionError):
            ContainerPolicyContext(
                surface=ContainerReviewSurface.SERVER,
                scope_id="",
            )
        with self.assertRaises(AssertionError):
            ContainerPolicyPlan(
                effective_settings=_effective(),
                context=_context(),
                command_fingerprint="cmd",
                escalation_triggers=(
                    ContainerEscalationTrigger.SECRET,
                    ContainerEscalationTrigger.SECRET,
                ),
            )
        with self.assertRaises(AssertionError):
            ContainerApprovalRecord(
                reviewer_identity="operator@example.test",
                review_mode=ContainerReviewMode.DURABLE,
                plan_fingerprint="fp",
                scope_id="scope",
                attempt_id=None,
                profile_name="policy-profile",
                policy_version="phase3",
                expires_at_seconds=1,
                approved_triggers=(
                    ContainerEscalationTrigger.SECRET,
                    ContainerEscalationTrigger.SECRET,
                ),
            )
        with self.assertRaises(AssertionError):
            ContainerPolicy(
                policy_version="phase3",
                explanations={"container.review.durable": ""},
            )
        with self.assertRaises(AssertionError):
            _policy().authorize(
                _plan(),
                approval="not approval",
                now_seconds=0,
            )
        with self.assertRaises(AssertionError):
            ContainerPolicy(policy_version="other").authorize(_plan())

    def test_queued_task_review_requires_attempt_identity(self) -> None:
        with self.assertRaises(AssertionError):
            ContainerPolicyContext(
                surface=ContainerReviewSurface.QUEUED_TASK,
                scope_id="queue.task.1",
            )

    def test_queued_task_approvals_do_not_cross_attempts(self) -> None:
        plan = _plan(
            escalation=ContainerEscalationMode.REQUIRE_REVIEW,
            surface=ContainerReviewSurface.QUEUED_TASK,
            triggers=(ContainerEscalationTrigger.SECRET,),
        )
        next_attempt = _plan(
            escalation=ContainerEscalationMode.REQUIRE_REVIEW,
            surface=ContainerReviewSurface.QUEUED_TASK,
            triggers=(ContainerEscalationTrigger.SECRET,),
            attempt_id="attempt.2",
        )
        approval = ContainerApprovalRecord.for_plan(
            plan,
            reviewer_identity="operator@example.test",
            expires_at_seconds=100,
        )

        with self.assertRaises(AssertionError):
            _policy().authorize(
                next_attempt,
                approval=approval,
                now_seconds=1,
            )

    def test_noninteractive_approval_records_are_rejected(self) -> None:
        plan = _plan(
            escalation=ContainerEscalationMode.REQUIRE_REVIEW,
            surface=ContainerReviewSurface.SERVER,
            triggers=(ContainerEscalationTrigger.NETWORK,),
        )
        approval = ContainerApprovalRecord(
            reviewer_identity="operator@example.test",
            review_mode=ContainerReviewMode.FAIL_CLOSED,
            plan_fingerprint=plan.plan_fingerprint,
            scope_id=plan.context.scope_id,
            attempt_id=plan.context.attempt_id,
            profile_name=plan.profile_name,
            policy_version=plan.policy_version,
            expires_at_seconds=100,
            approved_triggers=plan.escalation_triggers,
        )

        with self.assertRaises(AssertionError):
            _policy().authorize(plan, approval=approval, now_seconds=1)

    def test_fake_e2e_review_modes_are_distinct_and_deterministic(
        self,
    ) -> None:
        policy = _policy()
        surfaces = (
            ContainerReviewSurface.INTERACTIVE_CLI,
            ContainerReviewSurface.STRICT_FLOW,
            ContainerReviewSurface.SERVER,
        )
        decisions = [
            policy.authorize(
                _plan(
                    escalation=ContainerEscalationMode.REQUIRE_REVIEW,
                    surface=surface,
                    triggers=(ContainerEscalationTrigger.COMBINED_RISK,),
                )
            ).to_dict()
            for surface in surfaces
        ]

        self.assertEqual(
            [decision["code"] for decision in decisions],
            [
                "container.review.interactive",
                "container.review.durable",
                "container.deny.review_unavailable",
            ],
        )
        self.assertEqual(
            [decision["decision"] for decision in decisions],
            ["requires_review", "requires_review", "deny"],
        )


def _policy() -> ContainerPolicy:
    return ContainerPolicy(policy_version="phase3")


def _context(
    surface: ContainerReviewSurface = ContainerReviewSurface.INTERACTIVE_CLI,
    attempt_id: str | None = "attempt.1",
    scope_id: str = "tool.shell.rg",
) -> ContainerPolicyContext:
    return ContainerPolicyContext(
        surface=surface,
        scope_id=scope_id,
        attempt_id=attempt_id,
    )


def _plan(
    *,
    escalation: ContainerEscalationMode = ContainerEscalationMode.DENY,
    surface: ContainerReviewSurface = ContainerReviewSurface.INTERACTIVE_CLI,
    triggers: tuple[ContainerEscalationTrigger, ...] = (),
    profile: ContainerProfile | None = None,
    attempt_id: str | None = "attempt.1",
    scope_id: str = "tool.shell.rg",
    preserve_profile: bool = False,
) -> ContainerPolicyPlan:
    return ContainerPolicyPlan(
        effective_settings=_effective(
            escalation=escalation,
            profile=profile,
            preserve_profile=preserve_profile,
        ),
        context=_context(
            surface,
            attempt_id=attempt_id,
            scope_id=scope_id,
        ),
        command_fingerprint="shell.rg:needle",
        escalation_triggers=triggers,
    )


def _effective(
    *,
    escalation: ContainerEscalationMode = ContainerEscalationMode.DENY,
    profile: ContainerProfile | None = None,
    preserve_profile: bool = False,
) -> ContainerEffectiveSettings:
    profile = profile or _profile()
    if preserve_profile:
        object.__setattr__(
            profile,
            "escalation",
            ContainerEscalationPolicy(mode=escalation),
        )
    else:
        profile = ContainerProfile(
            name=profile.name,
            image=profile.image,
            workspace=profile.workspace,
            mounts=profile.mounts,
            environment=profile.environment,
            secrets=profile.secrets,
            network=profile.network,
            devices=profile.devices,
            resources=profile.resources,
            output=profile.output,
            cleanup=profile.cleanup,
            pooling=profile.pooling,
            audit=profile.audit,
            escalation=ContainerEscalationPolicy(mode=escalation),
            command_mode=profile.command_mode,
            read_only_rootfs=profile.read_only_rootfs,
            user=profile.user,
        )
    settings = ContainerSettings(
        source=ContainerSettingsSource(
            surface=ContainerSurface.CLI,
            trust_level=ContainerTrustLevel.TRUSTED_OPERATOR,
        ),
        backend=ContainerBackend.DOCKER,
        default_profile=profile.name,
        allowed_profiles=(profile.name,),
        profiles={profile.name: profile},
        profile_registry_id="policy-registry",
        policy_version="phase3",
    )
    return ContainerAuthorityCaps(settings=settings).merge(())


def _profile(
    *,
    image: ContainerImagePolicy | None = None,
    mounts: tuple[ContainerMountDeclaration, ...] = (),
    secrets: tuple[ContainerSecretReference, ...] = (),
    network: ContainerNetworkPolicy | None = None,
    devices: ContainerDevicePolicy | None = None,
    resources: ContainerResourceLimits | None = None,
) -> ContainerProfile:
    return ContainerProfile(
        name="policy-profile",
        image=image or ContainerImagePolicy(reference=_IMAGE),
        mounts=mounts,
        secrets=secrets,
        network=network or ContainerNetworkPolicy(),
        devices=devices or ContainerDevicePolicy(),
        resources=resources or ContainerResourceLimits(),
    )


if __name__ == "__main__":
    main()
