from typing import cast
from unittest import TestCase, main

from avalan.container import (
    ContainerBackend,
    ContainerEffectiveSettings,
    ContainerMountAccess,
    ContainerMountType,
    ContainerNetworkMode,
    ContainerResourceLimits,
)
from avalan.isolation import (
    IsolationApprovalRecord,
    IsolationAuditRecord,
    IsolationCleanupStatus,
    IsolationDecisionType,
    IsolationDurablePlanMetadata,
    IsolationEffectiveSettings,
    IsolationMode,
    IsolationPlan,
    IsolationPlanRequestKind,
    IsolationPolicy,
    IsolationPolicyContext,
    IsolationPolicyEvaluationCache,
    IsolationProfileSelection,
    IsolationReviewMode,
    IsolationReviewSurface,
    IsolationSettings,
    IsolationShellRequest,
    LocalIsolationPolicy,
    SandboxEffectiveSettings,
    SandboxIsolationSubplan,
    SandboxNetworkMode,
    elevate_isolation_plan,
    lower_isolation_plan,
    normalize_shell_request,
    trusted_isolation_source,
)

_DIGEST = "4" * 64
_DIGEST_ALT = "5" * 64
_IMAGE = f"ghcr.io/example/isolation-phase4@sha256:{_DIGEST}"
_IMAGE_ALT = f"ghcr.io/example/isolation-phase4@sha256:{_DIGEST_ALT}"


class IsolationPlanningTest(TestCase):
    def test_stable_fingerprint_and_durable_reload(self) -> None:
        first = lower_isolation_plan(
            _container_effective(),
            _request(
                environment={
                    "PATH": "/request/bin",
                    "LC_ALL": "request-locale",
                }
            ),
        )
        second = lower_isolation_plan(
            _container_effective(reverse_mounts=True),
            _request(
                cwd="/workspace/project/../project",
                environment={"LC_ALL": "ignored", "PATH": "ignored"},
            ),
        )
        loaded = IsolationPlan.from_json(first.to_json())
        metadata = IsolationDurablePlanMetadata.from_dict(
            first.to_metadata().to_dict()
        )

        self.assertEqual(first.plan_fingerprint, second.plan_fingerprint)
        self.assertEqual(loaded.plan_fingerprint, first.plan_fingerprint)
        self.assertEqual(metadata.to_dict()["effective_mode"], "container")
        metadata.assert_matches(first)
        self.assertNotIn(
            "/request/bin",
            str(first.canonical_policy_input()),
        )
        self.assertNotIn(
            "request-locale",
            str(first.canonical_policy_input()),
        )
        self.assertEqual(
            normalize_shell_request(_request()).executable_path,
            "/bin/sh",
        )

    def test_fingerprint_drift_and_fail_closed_reload(self) -> None:
        base = lower_isolation_plan(_container_effective(), _request())
        backend_changed = lower_isolation_plan(
            _container_effective(backend="apple-container"),
            _request(),
        )
        path_changed = lower_isolation_plan(
            _container_effective(),
            _request(cwd="/workspace/other"),
        )
        image_changed = lower_isolation_plan(
            _container_effective(image=_IMAGE_ALT),
            _request(),
        )
        resources_changed = lower_isolation_plan(
            _container_effective(resources={"cpu_count": 2}),
            _request(),
        )
        sandbox_base = lower_isolation_plan(
            _sandbox_effective(),
            _request(),
        )
        sandbox_root_changed = lower_isolation_plan(
            _sandbox_effective(read_roots=("/workspace", "/opt/tools")),
            _request(),
        )
        local = lower_isolation_plan(
            _local_policy(approval_required=True),
            _request(cwd="/workspace/project"),
            approval_identity="operator-a",
        )
        local_reapproved = lower_isolation_plan(
            _local_policy(approval_required=True),
            _request(cwd="/workspace/project"),
            approval_identity="operator-b",
        )
        serialized = base.to_dict()
        serialized["plan_fingerprint"] = "0" * 64

        self.assertNotEqual(
            base.plan_fingerprint,
            sandbox_base.plan_fingerprint,
        )
        self.assertNotEqual(
            base.plan_fingerprint,
            backend_changed.plan_fingerprint,
        )
        self.assertNotEqual(
            base.plan_fingerprint,
            path_changed.plan_fingerprint,
        )
        self.assertNotEqual(
            base.plan_fingerprint,
            image_changed.plan_fingerprint,
        )
        self.assertNotEqual(
            base.plan_fingerprint,
            resources_changed.plan_fingerprint,
        )
        self.assertNotEqual(
            sandbox_base.plan_fingerprint,
            sandbox_root_changed.plan_fingerprint,
        )
        self.assertNotEqual(
            local.plan_fingerprint,
            local_reapproved.plan_fingerprint,
        )
        with self.assertRaises(AssertionError):
            IsolationPlan.from_dict(serialized)
        with self.assertRaises(AssertionError):
            lower_isolation_plan(
                _local_policy(),
                _request(cwd="/private"),
            )

    def test_fingerprint_includes_required_profile_and_local_policy_fields(
        self,
    ) -> None:
        container = lower_isolation_plan(
            _container_effective(required=False),
            _request(),
        )
        container_required = lower_isolation_plan(
            _container_effective(required=True),
            _request(),
        )
        container_allowed_profiles = lower_isolation_plan(
            _container_effective(
                allowed_profiles=("tools", "debug-tools"),
            ),
            _request(),
        )
        sandbox = lower_isolation_plan(
            _sandbox_effective(required=False),
            _request(),
        )
        sandbox_required = lower_isolation_plan(
            _sandbox_effective(required=True),
            _request(),
        )
        sandbox_allowed_profiles = lower_isolation_plan(
            _sandbox_effective(
                allowed_profiles=("host-tools", "debug-tools"),
            ),
            _request(),
        )
        local = lower_isolation_plan(
            _local_policy(
                approval_required=False,
                timeout_seconds=5,
                max_stdout_bytes=1024,
            ),
            _request(),
        )
        local_timeout = lower_isolation_plan(
            _local_policy(
                approval_required=False,
                timeout_seconds=6,
                max_stdout_bytes=1024,
            ),
            _request(),
        )
        local_output = lower_isolation_plan(
            _local_policy(
                approval_required=False,
                timeout_seconds=5,
                max_stdout_bytes=2048,
            ),
            _request(),
        )
        local_allowlist = lower_isolation_plan(
            _local_policy(
                approval_required=False,
                executable_allowlist=("/bin/sh", "/usr/bin/env"),
                timeout_seconds=5,
                max_stdout_bytes=1024,
            ),
            _request(),
        )
        local_approval_policy = lower_isolation_plan(
            _local_policy(
                approval_required=True,
                timeout_seconds=5,
                max_stdout_bytes=1024,
            ),
            _request(),
        )
        serialized = local.to_dict()
        local_raw = cast(dict[str, object], serialized["local"])
        policy_raw = cast(dict[str, object], local_raw["policy"])
        policy_raw["timeout_seconds"] = 6

        self.assertNotEqual(
            container.plan_fingerprint,
            container_required.plan_fingerprint,
        )
        self.assertNotEqual(
            container.plan_fingerprint,
            container_allowed_profiles.plan_fingerprint,
        )
        self.assertNotEqual(
            sandbox.plan_fingerprint,
            sandbox_required.plan_fingerprint,
        )
        self.assertNotEqual(
            sandbox.plan_fingerprint,
            sandbox_allowed_profiles.plan_fingerprint,
        )
        self.assertNotEqual(
            local.plan_fingerprint,
            local_timeout.plan_fingerprint,
        )
        self.assertNotEqual(
            local.plan_fingerprint,
            local_output.plan_fingerprint,
        )
        self.assertNotEqual(
            local.plan_fingerprint,
            local_allowlist.plan_fingerprint,
        )
        self.assertNotEqual(
            local.plan_fingerprint,
            local_approval_policy.plan_fingerprint,
        )
        with self.assertRaises(AssertionError):
            IsolationPlan.from_dict(serialized)

    def test_review_allow_deny_requires_review_and_scoped_approval(
        self,
    ) -> None:
        policy = IsolationPolicy(policy_version="phase4")
        container_plan = lower_isolation_plan(
            _container_effective(required=True),
            _request(),
        )
        sandbox_plan = elevate_isolation_plan(
            container_plan,
            _sandbox_effective(required=True),
            temp_dir="/tmp/avalan/session",
            output_dir="/workspace/out/session",
            collect_outputs=True,
        )
        context = IsolationPolicyContext(
            surface=IsolationReviewSurface.STRICT_FLOW,
            scope_id="flow-1",
        )
        allow = policy.authorize(container_plan, context)
        review = policy.authorize(sandbox_plan, context)
        approval = IsolationApprovalRecord.for_plan(
            IsolationPlan.from_dict(sandbox_plan.to_dict()),
            context,
            reviewer_identity="operator@example.test",
            expires_at_seconds=100,
        )
        approved = policy.authorize(
            sandbox_plan,
            context,
            approval=IsolationApprovalRecord.from_dict(approval.to_dict()),
            now_seconds=1,
        )
        denied = IsolationPolicy(
            policy_version="phase4",
            denied_modes=("container",),
        ).authorize(container_plan, context)
        audit = IsolationAuditRecord.from_plan_decision(
            sandbox_plan,
            approved,
            cleanup_status=IsolationCleanupStatus.SUCCEEDED,
        )

        self.assertEqual(allow.decision, IsolationDecisionType.ALLOW)
        self.assertEqual(
            review.decision,
            IsolationDecisionType.REQUIRES_REVIEW,
        )
        self.assertEqual(approved.decision, IsolationDecisionType.ALLOW)
        self.assertEqual(approved.reviewer_identity, "operator@example.test")
        self.assertEqual(denied.decision, IsolationDecisionType.DENY)
        self.assertEqual(approval.review_mode, IsolationReviewMode.DURABLE)
        self.assertEqual(
            audit.to_dict(),
            IsolationAuditRecord.from_dict(audit.to_dict()).to_dict(),
        )
        self.assertEqual(audit.to_dict()["requested_mode"], "container")
        self.assertEqual(audit.to_dict()["effective_mode"], "sandbox")
        self.assertEqual(audit.to_dict()["backend"], "seatbelt")
        self.assertEqual(audit.to_dict()["profile"], "host-tools")
        self.assertEqual(
            audit.to_dict()["diagnostic_code"],
            "isolation.allow.cached_approval",
        )

    def test_stale_broader_cross_mode_and_noninteractive_approvals_fail(
        self,
    ) -> None:
        policy = IsolationPolicy(policy_version="phase4")
        sandbox_plan = lower_isolation_plan(
            _sandbox_effective(required=True),
            _request(),
        )
        local_plan = elevate_isolation_plan(
            sandbox_plan,
            _local_policy(approval_required=True),
            approval_identity="operator@example.test",
        )
        durable_context = IsolationPolicyContext(
            surface="queued_task",
            scope_id="task-1",
            attempt_id="attempt-1",
        )
        approval = IsolationApprovalRecord.for_plan(
            sandbox_plan,
            durable_context,
            reviewer_identity="operator@example.test",
            expires_at_seconds=10,
        )
        local_approval = IsolationApprovalRecord.for_plan(
            local_plan,
            durable_context,
            reviewer_identity="operator@example.test",
            expires_at_seconds=10,
        )
        broader_context = IsolationPolicyContext(
            surface="queued_task",
            scope_id="task",
            attempt_id="attempt-1",
        )
        fail_closed = IsolationPolicyContext(
            surface=IsolationReviewSurface.DIRECT_TASK,
            scope_id="task-1",
        )

        with self.assertRaises(AssertionError):
            policy.authorize(
                sandbox_plan,
                durable_context,
                approval=approval,
                now_seconds=10,
            )
        with self.assertRaises(AssertionError):
            policy.authorize(
                sandbox_plan,
                broader_context,
                approval=approval,
                now_seconds=1,
            )
        with self.assertRaises(AssertionError):
            policy.authorize(
                local_plan,
                durable_context,
                approval=approval,
                now_seconds=1,
            )
        with self.assertRaises(AssertionError):
            policy.authorize(
                local_plan,
                fail_closed,
                approval=local_approval,
                now_seconds=1,
            )
        self.assertEqual(
            policy.authorize(local_plan, fail_closed).decision,
            IsolationDecisionType.DENY,
        )
        with self.assertRaises(AssertionError):
            IsolationPolicyContext(surface="queued_task", scope_id="task")

    def test_local_host_execution_requires_explicit_approval(self) -> None:
        local_plan = lower_isolation_plan(
            _local_policy(approval_required=False),
            _request(),
        )
        policy = IsolationPolicy(
            policy_version="phase4",
            preapproved_modes=("local",),
        )
        interactive = IsolationPolicyContext(
            surface="interactive_cli",
            scope_id="cli",
        )
        fail_closed = IsolationPolicyContext(
            surface="direct_task",
            scope_id="task-1",
        )
        durable = IsolationPolicyContext(
            surface="strict_flow",
            scope_id="flow-1",
        )
        approval = IsolationApprovalRecord.for_plan(
            local_plan,
            durable,
            reviewer_identity="operator@example.test",
            expires_at_seconds=10,
        )

        self.assertEqual(
            policy.authorize(local_plan, interactive).decision,
            IsolationDecisionType.REQUIRES_REVIEW,
        )
        self.assertEqual(
            policy.authorize(local_plan, fail_closed).decision,
            IsolationDecisionType.DENY,
        )
        self.assertEqual(
            policy.authorize(local_plan, durable).decision,
            IsolationDecisionType.REQUIRES_REVIEW,
        )
        self.assertEqual(
            policy.authorize(
                local_plan,
                durable,
                approval=approval,
                now_seconds=1,
            ).decision,
            IsolationDecisionType.ALLOW,
        )

    def test_fake_e2e_required_modes_do_not_fallback_without_review(
        self,
    ) -> None:
        policy = IsolationPolicy(policy_version="phase4")
        fail_closed = IsolationPolicyContext(
            surface="server",
            scope_id="request-1",
        )
        required_container = lower_isolation_plan(
            _container_effective(required=True),
            _request(),
        )
        container_to_sandbox = elevate_isolation_plan(
            required_container,
            _sandbox_effective(required=True),
        )
        required_sandbox = lower_isolation_plan(
            _sandbox_effective(required=True),
            _request(),
        )
        sandbox_to_local = elevate_isolation_plan(
            required_sandbox,
            _local_policy(approval_required=True),
            approval_identity="operator@example.test",
        )
        container_originated_local = elevate_isolation_plan(
            container_to_sandbox,
            _local_policy(approval_required=True),
            approval_identity="operator@example.test",
        )
        chained_decision = policy.authorize(
            container_originated_local,
            fail_closed,
        )
        chained_audit = IsolationAuditRecord.from_plan_decision(
            container_originated_local,
            chained_decision,
        )
        chained_audit_dict = chained_audit.to_dict()
        chained_lost_controls = cast(
            list[str],
            chained_audit_dict["lost_controls"],
        )

        self.assertEqual(
            container_to_sandbox.requested_mode,
            IsolationMode.CONTAINER,
        )
        self.assertEqual(
            container_to_sandbox.effective_mode,
            IsolationMode.SANDBOX,
        )
        self.assertEqual(
            sandbox_to_local.requested_mode,
            IsolationMode.SANDBOX,
        )
        self.assertEqual(sandbox_to_local.effective_mode, IsolationMode.LOCAL)
        self.assertEqual(
            container_originated_local.requested_mode,
            IsolationMode.SANDBOX,
        )
        self.assertEqual(
            container_originated_local.effective_mode,
            IsolationMode.LOCAL,
        )
        self.assertNotEqual(
            container_to_sandbox.plan_fingerprint,
            container_originated_local.plan_fingerprint,
        )
        self.assertEqual(
            chained_audit_dict["requested_mode"],
            "sandbox",
        )
        self.assertEqual(chained_audit_dict["effective_mode"], "local")
        self.assertIn(
            "sandbox_filesystem_policy",
            chained_lost_controls,
        )
        self.assertEqual(
            policy.authorize(container_to_sandbox, fail_closed).to_dict()[
                "decision"
            ],
            "deny",
        )
        self.assertEqual(
            policy.authorize(sandbox_to_local, fail_closed).to_dict()["code"],
            "isolation.deny.review_unavailable",
        )
        with self.assertRaises(AssertionError):
            lower_isolation_plan(
                _local_policy(),
                _request(),
                requested_mode="container",
            )
        with self.assertRaises(AssertionError):
            elevate_isolation_plan(required_container, _local_policy())
        with self.assertRaises(AssertionError):
            elevate_isolation_plan(sandbox_to_local, _local_policy())

    def test_policy_cache_is_bounded_and_uses_cacheable_decisions(
        self,
    ) -> None:
        policy = IsolationPolicy(
            policy_version="phase4",
            explanations={"isolation.allow.preapproved": "Allowed."},
        )
        context = IsolationPolicyContext(
            surface="strict_flow",
            scope_id="flow-1",
        )
        cache = IsolationPolicyEvaluationCache(max_entries=1)
        first = lower_isolation_plan(_container_effective(), _request())
        second = lower_isolation_plan(
            _container_effective(backend="apple-container"),
            _request(),
        )

        first_decision = cache.authorize(policy, first, context)
        first_cached = cache.authorize(policy, first, context)
        second_decision = cache.authorize(policy, second, context)
        first_after_evict = cache.authorize(policy, first, context)

        self.assertIs(first_decision, first_cached)
        self.assertIsNot(first_decision, first_after_evict)
        self.assertEqual(second_decision.explanation, "Allowed.")
        self.assertEqual(cache.size, 1)
        self.assertEqual(
            policy.cache_key(first, context),
            policy.cache_key(first, context),
        )
        self.assertNotEqual(
            policy.cache_key(first, context),
            IsolationPolicy(
                policy_version="phase4",
                denied_modes=("container",),
            ).cache_key(first, context),
        )
        self.assertEqual(
            cache.authorize(
                IsolationPolicy(
                    policy_version="phase4",
                    denied_modes=("container",),
                ),
                first,
                context,
            ).decision,
            IsolationDecisionType.DENY,
        )
        with self.assertRaises(AssertionError):
            IsolationPolicyEvaluationCache(max_entries=0)

    def test_invalid_command_and_subplan_shapes_are_rejected(self) -> None:
        command = normalize_shell_request(_request())

        with self.assertRaises(AssertionError):
            IsolationShellRequest(
                request_kind="missing",
                logical_name="shell",
                command="/bin/sh",
                argv=("/bin/sh",),
                cwd="/workspace",
            )
        with self.assertRaises(AssertionError):
            IsolationShellRequest(
                request_kind="task_attempt",
                logical_name="shell",
                command="/bin/sh",
                argv=("/bin/sh",),
                cwd="/workspace",
            )
        with self.assertRaises(AssertionError):
            lower_isolation_plan(
                _sandbox_effective(),
                _request(command="/usr/bin/env"),
            )
        with self.assertRaises(AssertionError):
            lower_isolation_plan(
                _sandbox_effective(),
                _request(environment={"SECRET_TOKEN": "value"}),
            )
        with self.assertRaises(AssertionError):
            IsolationPlan(requested_mode="container", local=None)
        with self.assertRaises(AssertionError):
            IsolationPlan(
                requested_mode="container",
                container=cast(object, command),  # type: ignore[arg-type]
            )

    def test_effective_settings_wrapper_lowers_to_active_branch(self) -> None:
        container = IsolationEffectiveSettings(
            mode="container",
            source=trusted_isolation_source("cli"),
            container=_container_effective(),
        )
        sandbox = IsolationEffectiveSettings(
            mode="sandbox",
            source=trusted_isolation_source("sdk"),
            sandbox=_sandbox_effective(),
        )
        local = IsolationEffectiveSettings(
            mode="local",
            source=trusted_isolation_source("sdk"),
            local=_local_policy(approval_required=False),
        )

        self.assertEqual(
            lower_isolation_plan(container, _request()).effective_mode,
            IsolationMode.CONTAINER,
        )
        self.assertEqual(
            lower_isolation_plan(sandbox, _request()).effective_mode,
            IsolationMode.SANDBOX,
        )
        self.assertEqual(
            lower_isolation_plan(local, _request()).effective_mode,
            IsolationMode.LOCAL,
        )

    def test_local_task_attempt_and_default_reload_branches(self) -> None:
        task_request = _request(
            request_kind=IsolationPlanRequestKind.TASK_ATTEMPT,
        )
        local_plan = lower_isolation_plan(
            LocalIsolationPolicy(
                approval_required=False,
                executable_allowlist=("/bin/sh",),
            ),
            task_request,
        )
        assert local_plan.local is not None
        approved_local = IsolationPlan(
            requested_mode="local",
            local=local_plan.local.with_approval_identity("operator"),
        )
        local_loaded = IsolationPlan.from_dict(approved_local.to_dict())
        sandbox_defaults = SandboxIsolationSubplan.from_dict(
            {
                "command": normalize_shell_request(_request()).to_dict(),
                "settings": _sandbox_effective().to_dict(),
            }
        )
        minimal_local = approved_local.to_dict()
        local_raw = cast(dict[str, object], minimal_local["local"])
        local_raw.pop("policy_version")
        local_raw.pop("profile_name")
        local_defaulted = IsolationPlan.from_dict(minimal_local)
        no_preapproval = IsolationPolicy(
            policy_version="phase4",
            preapproved_modes=(),
        ).authorize(
            lower_isolation_plan(_sandbox_effective(), _request()),
            IsolationPolicyContext(surface="interactive_cli", scope_id="cli"),
        )
        local_without_required_approval = IsolationPolicy(
            policy_version="phase4",
        ).authorize(
            local_plan,
            IsolationPolicyContext(surface="interactive_cli", scope_id="cli"),
        )

        self.assertIsNone(local_loaded.backend)
        self.assertEqual(local_loaded.profile_name, "local")
        self.assertEqual(local_loaded.command.attempt_id, "attempt-1")
        self.assertEqual(
            local_loaded.to_metadata().to_dict()["attempt_id"],
            "attempt-1",
        )
        self.assertEqual(
            sandbox_defaults.stream_buffer_bytes,
            65536,
        )
        self.assertEqual(local_defaulted.profile_name, "local")
        self.assertEqual(
            no_preapproval.to_dict()["decision"],
            "requires_review",
        )
        self.assertEqual(
            local_without_required_approval.to_dict()["decision"],
            "requires_review",
        )
        with self.assertRaises(AssertionError):
            lower_isolation_plan(
                _sandbox_effective(),
                _request(cwd="/etc/ssh"),
            )


def _request(
    *,
    request_kind: str | IsolationPlanRequestKind = (
        IsolationPlanRequestKind.TYPED_TOOL
    ),
    command: str = "/bin/sh",
    cwd: str = "/workspace/project",
    environment: dict[str, str] | None = None,
) -> IsolationShellRequest:
    return IsolationShellRequest(
        request_kind=request_kind,
        logical_name="shell",
        command=command,
        argv=(command, "-lc", "echo ok"),
        cwd=cwd,
        environment=environment or {"PATH": "/bin", "LC_ALL": "C.UTF-8"},
        request_id="request-1",
        attempt_id=(
            "attempt-1"
            if request_kind == IsolationPlanRequestKind.TASK_ATTEMPT
            or request_kind == "task_attempt"
            else None
        ),
    )


def _local_policy(
    *,
    approval_required: bool = True,
    allowed_roots: tuple[str, ...] = ("/workspace",),
    executable_allowlist: tuple[str, ...] = ("/bin/sh",),
    timeout_seconds: int | None = None,
    max_stdout_bytes: int = 65536,
    max_stderr_bytes: int = 32768,
) -> LocalIsolationPolicy:
    return LocalIsolationPolicy(
        approval_required=approval_required,
        allowed_roots=allowed_roots,
        executable_allowlist=executable_allowlist,
        timeout_seconds=timeout_seconds,
        max_stdout_bytes=max_stdout_bytes,
        max_stderr_bytes=max_stderr_bytes,
    )


def _sandbox_effective(
    *,
    backend: str = "seatbelt",
    read_roots: tuple[str, ...] = ("/workspace",),
    profile: str = "host-tools",
    allowed_profiles: tuple[str, ...] | None = None,
    profile_registry_id: str = "default",
    required: bool = False,
) -> SandboxEffectiveSettings:
    source = trusted_isolation_source("sdk")
    selected_allowed_profiles = allowed_profiles or (profile,)
    profile_names = tuple(sorted(set(selected_allowed_profiles) | {profile}))
    profile_template = {
        "trusted_executables": ["/bin/sh"],
        "executable_search_roots": ["/bin"],
        "read_roots": list(read_roots),
        "write_roots": ["/workspace/out"],
        "deny_roots": ["/etc/ssh"],
        "scratch_roots": ["/tmp/avalan"],
        "output_roots": ["/workspace/out"],
        "environment": {
            "variables": {"LC_ALL": "C.UTF-8"},
            "allowlist": ["PATH"],
        },
        "network": {"mode": SandboxNetworkMode.LOOPBACK.value},
        "resources": {"timeout_seconds": 10, "pids": 32},
        "output": {
            "allow_artifacts": True,
            "max_artifact_bytes": 1024,
        },
    }
    settings = IsolationSettings.from_dict(
        {
            "mode": "sandbox",
            "sandbox": {
                "backend": backend,
                "default_profile": profile,
                "allowed_profiles": list(selected_allowed_profiles),
                "profiles": {
                    name: {"name": name, **profile_template}
                    for name in profile_names
                },
                "profile_registry_id": profile_registry_id,
                "policy_version": "phase4",
            },
        },
        source=source,
    )
    effective = settings.select_profile(
        selection=IsolationProfileSelection(
            mode="sandbox",
            profile=profile,
            required=required,
        )
    )
    assert effective.sandbox is not None
    return effective.sandbox


def _container_effective(
    *,
    backend: str = "docker",
    image: str = _IMAGE,
    resources: dict[str, int] | None = None,
    reverse_mounts: bool = False,
    profile: str = "tools",
    allowed_profiles: tuple[str, ...] | None = None,
    profile_registry_id: str = "default",
    required: bool = False,
) -> ContainerEffectiveSettings:
    source = trusted_isolation_source("cli")
    mounts = [
        {
            "source": ".",
            "target": "/workspace",
            "mount_type": ContainerMountType.WORKSPACE.value,
            "access": ContainerMountAccess.READ.value,
        },
        {
            "source": None,
            "target": "/workspace/out",
            "mount_type": ContainerMountType.OUTPUT.value,
            "access": ContainerMountAccess.WRITE.value,
        },
    ]
    if reverse_mounts:
        mounts.reverse()
    selected_allowed_profiles = allowed_profiles or (profile,)
    profile_names = tuple(sorted(set(selected_allowed_profiles) | {profile}))
    profile_template = {
        "image": {"reference": image},
        "mounts": mounts,
        "environment": {
            "variables": {"LC_ALL": "C.UTF-8"},
            "allowlist": ["PATH"],
        },
        "secrets": [{"name": "api-key", "env_name": "API_KEY"}],
        "network": {
            "mode": ContainerNetworkMode.ALLOWLIST.value,
            "egress_allowlist": ["api.example.test"],
        },
        "resources": resources or ContainerResourceLimits().to_dict(),
        "output": {
            "allow_artifacts": True,
            "max_artifact_bytes": 2048,
        },
    }
    settings = IsolationSettings.from_dict(
        {
            "mode": "container",
            "container": {
                "backend": backend,
                "default_profile": profile,
                "allowed_profiles": list(selected_allowed_profiles),
                "profiles": {
                    name: {"name": name, **profile_template}
                    for name in profile_names
                },
                "profile_registry_id": profile_registry_id,
                "policy_version": "phase4",
            },
        },
        source=source,
    )
    effective = settings.select_profile(
        selection=IsolationProfileSelection(
            mode="container",
            profile=profile,
            required=required,
        )
    )
    assert effective.container is not None
    if backend == "apple-container":
        self_backend = effective.container.backend
        assert self_backend == ContainerBackend.APPLE_CONTAINER
    return effective.container


if __name__ == "__main__":
    main()
