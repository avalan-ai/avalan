from asyncio import CancelledError, sleep
from collections.abc import Mapping
from dataclasses import replace
from time import time
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import patch

from avalan.container import (
    ContainerApprovalRecord,
    ContainerAuditEventType,
    ContainerAuditMode,
    ContainerAuditPolicy,
    ContainerAuthorityCaps,
    ContainerBackend,
    ContainerDeviceClass,
    ContainerDevicePolicy,
    ContainerDiagnostic,
    ContainerDiagnosticCategory,
    ContainerDiagnosticCode,
    ContainerEffectiveSettings,
    ContainerEnvironmentPolicy,
    ContainerEscalationMode,
    ContainerEscalationPolicy,
    ContainerExecutionScope,
    ContainerImagePolicy,
    ContainerMountAccess,
    ContainerMountDeclaration,
    ContainerMountType,
    ContainerNetworkMode,
    ContainerNetworkPolicy,
    ContainerPolicyContext,
    ContainerPolicyPlan,
    ContainerPoolingMode,
    ContainerPoolingPolicy,
    ContainerProfile,
    ContainerProfileSelection,
    ContainerResourceLimits,
    ContainerReviewSurface,
    ContainerRuntimeEnvelopeKind,
    ContainerSecretReference,
    ContainerSettings,
    ContainerSettingsOverride,
    ContainerSettingsPrecedence,
    ContainerSettingsSource,
    ContainerSurface,
    ContainerTrustLevel,
    ContainerWorkspaceMapping,
    trusted_container_source,
)
from avalan.entities import ToolManagerSettings
from avalan.event import EventType
from avalan.flow import (
    FlowDefinition,
    FlowDefinitionLoader,
    FlowDiagnostic,
    FlowEntryBehavior,
    FlowExecutionPlan,
    FlowExecutionTrace,
    FlowExecutor,
    FlowInputDefinition,
    FlowInputType,
    FlowIsolationMetadata,
    FlowNodeContract,
    FlowNodeDefinition,
    FlowNodeExecutionError,
    FlowNodeKind,
    FlowNodeMetadata,
    FlowNodePlan,
    FlowNodeRegistry,
    FlowNodeState,
    FlowOutputBehavior,
    FlowOutputDefinition,
    FlowOutputType,
    FlowPlanCompileResult,
    FlowPlanExecutionResult,
    FlowPlanNodeRunner,
    FlowRetryBackoffStrategy,
    FlowRetryPlan,
    FlowRuntimeEnvelopeDefinition,
    FlowTimeoutPlan,
    FlowTimeoutPolicy,
    Node,
    build_flow,
    compile_flow_definition,
    execute_flow_plan,
    flow_node_container_fingerprint,
    flow_resume_isolation_metadata,
    loads_flow_definition_result,
    parse_flow_selector,
    tool_flow_node_registry,
)
from avalan.flow.plan import (
    _assert_effective_no_wider,
    _validate_node_container_deadline,
)
from avalan.flow.registry import FlowNodeConfigurationError
from avalan.flow.runtime import (
    _append_container_event_draft,
    _container_approval_from_resume,
    _emit_container_event,
    _FlowEventDraft,
    _resume_isolation_metadata_diagnostics,
    _resume_isolation_node_names,
)
from avalan.flow.serializer import serialize_flow_definition
from avalan.flow.stream import flow_stream_session
from avalan.isolation import IsolationMode
from avalan.model.stream import CanonicalStreamItem
from avalan.tool import ToolSet
from avalan.tool.manager import ToolManager

_DIGEST = "a" * 64
_IMAGE = f"ghcr.io/example/flow-tools@sha256:{_DIGEST}"
_OTHER_DIGEST = "b" * 64
_OTHER_IMAGE = f"ghcr.io/example/other-tools@sha256:{_OTHER_DIGEST}"


async def container_echo(value: str = "ok") -> str:
    return value


container_echo.aliases = ["container-echo"]  # type: ignore[attr-defined]


class FlowContainerPlanTestCase(IsolatedAsyncioTestCase):
    async def test_compile_returns_validation_diagnostics(self) -> None:
        result = await compile_flow_definition(
            _definition(
                node=FlowNodeDefinition(name="work", type="missing-type")
            )
        )

        self.assertFalse(result.ok)
        self.assertIn(
            "flow.unknown_node_type",
            [diagnostic.code for diagnostic in result.diagnostics],
        )

    async def test_compile_requires_strict_definition(self) -> None:
        result = await compile_flow_definition(
            FlowDefinition(
                name="legacy",
                entrypoint="work",
                output_node="work",
                input=FlowInputDefinition(
                    name="payload",
                    type=FlowInputType.OBJECT,
                ),
                output=FlowOutputDefinition(
                    name="answer",
                    type=FlowOutputType.OBJECT,
                ),
                nodes=(FlowNodeDefinition(name="work", type="pass-through"),),
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.execution.plan_requires_strict_definition",
        )

    async def test_default_container_settings_are_inherited_by_nodes(
        self,
    ) -> None:
        defaults = _settings(
            profiles={"wide": _profile("wide")},
            default_profile="wide",
        )

        result = await compile_flow_definition(
            _definition(container=defaults),
        )

        self.assertTrue(result.ok, result.public_diagnostics)
        assert result.plan is not None
        self.assertIsNotNone(result.plan.container)
        assert result.plan.container is not None
        self.assertEqual(
            result.plan.container.scope,
            ContainerExecutionScope.DURABLE_WORKFLOW,
        )
        assert result.plan.isolation is not None
        self.assertEqual(result.plan.isolation.mode, IsolationMode.CONTAINER)
        self.assertEqual(result.plan.isolation.profile_name, "wide")
        self.assertEqual(result.plan.isolation.policy_version, "phase12")
        self.assertEqual(result.plan.isolation.container_fingerprint, None)
        node = result.plan.node_map["work"]
        assert node.container is not None
        assert node.isolation is not None
        self.assertEqual(node.container.profile_name, "wide")
        self.assertEqual(node.container.policy_version, "phase12")
        self.assertEqual(
            node.container.scope,
            ContainerExecutionScope.DURABLE_WORKFLOW,
        )
        self.assertEqual(node.isolation.mode, IsolationMode.CONTAINER)
        self.assertEqual(node.isolation.profile_name, "wide")
        self.assertEqual(
            node.isolation.container_fingerprint,
            flow_node_container_fingerprint(result.plan, node),
        )
        metadata = flow_resume_isolation_metadata(result.plan)
        isolation = metadata["isolation"]
        assert isinstance(isolation, Mapping)
        nodes = isolation["nodes"]
        assert isinstance(nodes, Mapping)
        self.assertEqual(
            nodes["work"],
            node.isolation.to_dict(),
        )

        repeated = await compile_flow_definition(
            _definition(container=defaults),
        )
        self.assertTrue(repeated.ok, repeated.public_diagnostics)
        assert repeated.plan is not None
        assert repeated.plan.isolation is not None
        repeated_node = repeated.plan.node_map["work"]
        assert repeated_node.isolation is not None
        self.assertEqual(
            repeated.plan.isolation.plan_fingerprint,
            result.plan.isolation.plan_fingerprint,
        )
        self.assertEqual(
            repeated_node.isolation.plan_fingerprint,
            node.isolation.plan_fingerprint,
        )

    async def test_node_container_override_can_only_narrow_defaults(
        self,
    ) -> None:
        defaults = _settings(
            profiles={"wide": _profile("wide", cpu_count=4, timeout=20)},
            default_profile="wide",
        )
        override = _override(
            resources=ContainerResourceLimits(cpu_count=2, timeout_seconds=5),
            mounts=(
                ContainerMountDeclaration(
                    source=".",
                    target="/workspace",
                    mount_type=ContainerMountType.WORKSPACE,
                ),
            ),
        )

        result = await compile_flow_definition(
            _definition(
                container=defaults,
                node=FlowNodeDefinition(
                    name="work",
                    type="pass-through",
                    container=override,
                ),
            ),
        )

        self.assertTrue(result.ok, result.public_diagnostics)
        assert result.plan is not None
        node = result.plan.node_map["work"]
        assert node.container is not None
        assert node.container.profile is not None
        self.assertEqual(node.container.profile.resources.cpu_count, 2)
        self.assertEqual(node.container.profile.resources.timeout_seconds, 5)
        self.assertEqual(len(node.container.profile.mounts), 1)
        self.assertEqual(
            node.container.scope,
            ContainerExecutionScope.DURABLE_WORKFLOW,
        )
        assert result.plan.isolation is not None
        assert node.isolation is not None
        self.assertNotEqual(
            node.isolation.plan_fingerprint,
            result.plan.isolation.plan_fingerprint,
        )

    async def test_node_can_select_allowed_narrower_profile(self) -> None:
        defaults = _settings(
            profiles={
                "wide": _profile("wide", cpu_count=4, timeout=20),
                "narrow": _profile("narrow", cpu_count=1, timeout=5),
            },
            default_profile="wide",
            allowed_profiles=("wide", "narrow"),
        )

        result = await compile_flow_definition(
            _definition(
                container=defaults,
                node=FlowNodeDefinition(
                    name="work",
                    type="pass-through",
                    container=_override(profile="narrow"),
                ),
            ),
        )

        self.assertTrue(result.ok, result.public_diagnostics)
        assert result.plan is not None
        node = result.plan.node_map["work"]
        assert node.container is not None
        self.assertEqual(node.container.profile_name, "narrow")

    async def test_tool_nodes_inherit_registered_tool_container_policy(
        self,
    ) -> None:
        settings = _settings(
            profiles={"tool": _profile("tool")},
            default_profile="tool",
        )
        effective = ContainerAuthorityCaps(settings=settings).merge()
        registry = _tool_registry(effective)

        result = await compile_flow_definition(
            _definition(
                node=FlowNodeDefinition(
                    name="work",
                    type="tool",
                    ref="container_echo",
                ),
                output_selector="work.result",
            ),
            registry,
        )

        self.assertTrue(result.ok, result.public_diagnostics)
        assert result.plan is not None
        node = result.plan.node_map["work"]
        assert node.container is not None
        self.assertEqual(node.container.profile_name, "tool")
        self.assertEqual(
            node.container.scope,
            ContainerExecutionScope.DURABLE_WORKFLOW,
        )

    async def test_tool_policy_cannot_widen_flow_defaults(self) -> None:
        defaults = _settings(
            profiles={"flow": _profile("flow", cpu_count=1)},
            default_profile="flow",
        )
        tool_settings = _settings(
            profiles={"tool": _profile("tool", cpu_count=2)},
            default_profile="tool",
        )
        registry = _tool_registry(
            ContainerAuthorityCaps(settings=tool_settings).merge()
        )

        result = await compile_flow_definition(
            _definition(
                container=defaults,
                node=FlowNodeDefinition(
                    name="work",
                    type="tool",
                    ref="container_echo",
                ),
                output_selector="work.result",
            ),
            registry,
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.container_policy_invalid",
        )
        self.assertIn(
            "tool cpu_count",
            _diagnostic_hint(result.diagnostics[0]),
        )

    async def test_tool_policy_cannot_change_flow_image(self) -> None:
        result = await _compile_tool_policy_against_flow_default(
            flow_profile=_profile("flow"),
            tool_profile=_profile("tool", image_reference=_OTHER_IMAGE),
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.container_policy_invalid",
        )
        self.assertIn("tool image", _diagnostic_hint(result.diagnostics[0]))

    async def test_tool_policy_cannot_change_flow_workspace(self) -> None:
        result = await _compile_tool_policy_against_flow_default(
            flow_profile=_profile("flow"),
            tool_profile=_profile(
                "tool",
                workspace=ContainerWorkspaceMapping(
                    host_root="tools",
                    container_path="/workspace",
                    working_directory="/workspace",
                ),
            ),
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.container_policy_invalid",
        )
        self.assertIn(
            "tool workspace",
            _diagnostic_hint(result.diagnostics[0]),
        )

    async def test_tool_policy_cannot_change_flow_pooling(self) -> None:
        result = await _compile_tool_policy_against_flow_default(
            flow_profile=_profile("flow"),
            tool_profile=_profile(
                "tool",
                pooling=ContainerPoolingPolicy(
                    mode=ContainerPoolingMode.SHORT_LIVED,
                ),
            ),
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.container_policy_invalid",
        )
        self.assertIn("tool pooling", _diagnostic_hint(result.diagnostics[0]))

    async def test_tool_policy_cannot_change_flow_audit(self) -> None:
        result = await _compile_tool_policy_against_flow_default(
            flow_profile=_profile("flow"),
            tool_profile=_profile(
                "tool",
                audit=ContainerAuditPolicy(mode=ContainerAuditMode.FULL),
            ),
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.container_policy_invalid",
        )
        self.assertIn("tool audit", _diagnostic_hint(result.diagnostics[0]))

    async def test_tool_policy_exact_identity_fields_can_narrow_resources(
        self,
    ) -> None:
        result = await _compile_tool_policy_against_flow_default(
            flow_profile=_profile("flow", cpu_count=4),
            tool_profile=_profile("tool", cpu_count=1),
        )

        self.assertTrue(result.ok, result.public_diagnostics)
        assert result.plan is not None
        node = result.plan.node_map["work"]
        assert node.container is not None
        assert node.container.profile is not None
        self.assertEqual(node.container.profile.resources.cpu_count, 1)

    async def test_tool_node_override_can_narrow_inherited_policy(
        self,
    ) -> None:
        tool_settings = _settings(
            profiles={"tool": _profile("tool", cpu_count=4)},
            default_profile="tool",
        )
        registry = _tool_registry(
            ContainerAuthorityCaps(settings=tool_settings).merge()
        )

        result = await compile_flow_definition(
            _definition(
                node=FlowNodeDefinition(
                    name="work",
                    type="tool",
                    ref="container_echo",
                    container=_override(
                        resources=ContainerResourceLimits(cpu_count=1),
                    ),
                ),
                output_selector="work.result",
            ),
            registry,
        )

        self.assertTrue(result.ok, result.public_diagnostics)
        assert result.plan is not None
        node = result.plan.node_map["work"]
        assert node.container is not None
        assert node.container.profile is not None
        self.assertEqual(node.container.profile.resources.cpu_count, 1)

    async def test_tool_node_override_cannot_widen_inherited_policy(
        self,
    ) -> None:
        tool_settings = _settings(
            profiles={"tool": _profile("tool", cpu_count=1)},
            default_profile="tool",
        )
        registry = _tool_registry(
            ContainerAuthorityCaps(settings=tool_settings).merge()
        )

        result = await compile_flow_definition(
            _definition(
                node=FlowNodeDefinition(
                    name="work",
                    type="tool",
                    ref="container_echo",
                    container=_override(
                        resources=ContainerResourceLimits(cpu_count=2),
                    ),
                ),
                output_selector="work.result",
            ),
            registry,
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.container_policy_invalid",
        )
        self.assertIn(
            "resource limit cannot widen",
            _diagnostic_hint(result.diagnostics[0]),
        )

    async def test_node_container_override_requires_trusted_defaults(
        self,
    ) -> None:
        result = await compile_flow_definition(
            _definition(
                node=FlowNodeDefinition(
                    name="work",
                    type="pass-through",
                    container=_override(resources=ContainerResourceLimits()),
                )
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.container_policy_invalid",
        )
        self.assertIn("trusted flow defaults", result.diagnostics[0].message)

    async def test_tool_policy_cannot_enable_disabled_flow_defaults(
        self,
    ) -> None:
        defaults = _disabled_settings()
        tool_settings = _settings(
            profiles={"tool": _profile("tool")},
            default_profile="tool",
        )
        registry = _tool_registry(
            ContainerAuthorityCaps(settings=tool_settings).merge()
        )

        result = await compile_flow_definition(
            _definition(
                container=defaults,
                node=FlowNodeDefinition(
                    name="work",
                    type="tool",
                    ref="container_echo",
                ),
                output_selector="work.result",
            ),
            registry,
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.container_policy_invalid",
        )
        self.assertIn("tool backend", _diagnostic_hint(result.diagnostics[0]))

    async def test_no_wider_rejects_profile_when_caps_have_none(self) -> None:
        caps = ContainerEffectiveSettings(
            backend=ContainerBackend.DOCKER,
            required=False,
            scope=ContainerExecutionScope.DURABLE_WORKFLOW,
            source=trusted_container_source(ContainerSurface.FLOW_TOML),
            policy_version="phase12",
            profile_registry_id="phase12",
        )

        with self.assertRaisesRegex(AssertionError, "disabled flow"):
            _assert_effective_no_wider(
                caps,
                _effective_settings(_profile("tool")),
            )

        _assert_effective_no_wider(caps, caps)

    async def test_tool_policy_comparators_reject_extra_fields(
        self,
    ) -> None:
        cases = (
            (
                _profile(
                    "flow",
                    environment=ContainerEnvironmentPolicy(
                        variables={"FLOW_ENV": "1"}
                    ),
                ),
                _profile(
                    "tool",
                    environment=ContainerEnvironmentPolicy(
                        variables={"TOOL_ENV": "1"}
                    ),
                ),
                "tool environment",
            ),
            (
                _profile(
                    "flow",
                    environment=ContainerEnvironmentPolicy(
                        allowlist=("FLOW_ENV",)
                    ),
                ),
                _profile(
                    "tool",
                    environment=ContainerEnvironmentPolicy(
                        allowlist=("TOOL_ENV",)
                    ),
                ),
                "environment allowlist",
            ),
            (
                _profile(
                    "flow",
                    network=ContainerNetworkPolicy(
                        mode=ContainerNetworkMode.ALLOWLIST,
                        egress_allowlist=("api.example.test",),
                    ),
                ),
                _profile(
                    "tool",
                    network=ContainerNetworkPolicy(
                        mode=ContainerNetworkMode.ALLOWLIST,
                        egress_allowlist=("other.example.test",),
                    ),
                ),
                "network allowlist",
            ),
            (
                _profile(
                    "flow",
                    devices=ContainerDevicePolicy(
                        devices=(ContainerDeviceClass.CPU,)
                    ),
                ),
                _profile(
                    "tool",
                    devices=ContainerDevicePolicy(
                        devices=(ContainerDeviceClass.NVIDIA_CDI,)
                    ),
                ),
                "tool devices",
            ),
            (
                _profile(
                    "flow",
                    secrets=(
                        ContainerSecretReference(
                            name="flow-secret",
                            env_name="FLOW_SECRET",
                        ),
                    ),
                ),
                _profile(
                    "tool",
                    secrets=(
                        ContainerSecretReference(
                            name="tool-secret",
                            env_name="TOOL_SECRET",
                        ),
                    ),
                ),
                "tool secrets",
            ),
        )

        for flow_profile, tool_profile, hint in cases:
            with self.subTest(hint=hint):
                result = await _compile_tool_policy_against_flow_default(
                    flow_profile=flow_profile,
                    tool_profile=tool_profile,
                )

                self.assertFalse(result.ok)
                self.assertEqual(
                    result.diagnostics[0].code,
                    "flow.container_policy_invalid",
                )
                self.assertIn(hint, _diagnostic_hint(result.diagnostics[0]))

    async def test_trusted_runtime_envelope_selection_is_compiled(
        self,
    ) -> None:
        settings = _settings(
            profiles={"flow": _profile("flow")},
            default_profile="flow",
        )

        result = await compile_flow_definition(
            _definition(
                container=settings,
                runtime_envelope=FlowRuntimeEnvelopeDefinition(
                    container=ContainerProfileSelection(
                        profile="flow",
                        required=True,
                        scope=ContainerExecutionScope.RUNTIME_ENVELOPE,
                    ),
                    readiness_timeout_seconds=11,
                ),
            ),
        )

        self.assertTrue(result.ok, result.public_diagnostics)
        assert result.plan is not None
        assert result.plan.runtime_envelope is not None
        self.assertEqual(
            result.plan.runtime_envelope.envelope_kind,
            ContainerRuntimeEnvelopeKind.FLOW_RUNTIME,
        )
        self.assertEqual(
            result.plan.runtime_envelope.envelope_plan.profile_name,
            "flow",
        )
        self.assertEqual(
            result.plan.runtime_envelope.envelope_plan.readiness_timeout_seconds,
            11,
        )

    async def test_runtime_envelope_requires_trusted_defaults(self) -> None:
        result = await compile_flow_definition(
            _definition(
                runtime_envelope=FlowRuntimeEnvelopeDefinition(
                    container=ContainerProfileSelection(
                        profile="flow",
                        required=True,
                        scope=ContainerExecutionScope.RUNTIME_ENVELOPE,
                    ),
                )
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.container_policy_invalid",
        )
        self.assertEqual(
            result.diagnostics[0].path,
            "runtime.container.envelope",
        )

    async def test_runtime_envelope_unknown_profile_is_rejected(self) -> None:
        settings = _settings(
            profiles={"flow": _profile("flow")},
            default_profile="flow",
        )

        result = await compile_flow_definition(
            _definition(
                container=settings,
                runtime_envelope=FlowRuntimeEnvelopeDefinition(
                    container=ContainerProfileSelection(
                        profile="missing",
                        required=True,
                        scope=ContainerExecutionScope.RUNTIME_ENVELOPE,
                    ),
                ),
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.container_policy_invalid",
        )
        self.assertIn(
            "selected profile",
            _diagnostic_hint(result.diagnostics[0]),
        )

    async def test_invalid_container_defaults_report_diagnostic(self) -> None:
        defaults = _settings(
            profiles={"flow": _profile("flow")},
            default_profile="flow",
        )

        with patch(
            "avalan.flow.plan.ContainerAuthorityCaps.merge",
            side_effect=AssertionError(),
        ):
            result = await compile_flow_definition(
                _definition(container=defaults)
            )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.container_policy_invalid",
        )
        self.assertEqual(
            _diagnostic_hint(result.diagnostics[0]),
            "Use only trusted container defaults and narrower node settings.",
        )

    async def test_runtime_envelope_fails_closed_in_direct_executor(
        self,
    ) -> None:
        settings = _settings(
            profiles={"flow": _profile("flow")},
            default_profile="flow",
        )
        flow = _definition(
            container=settings,
            runtime_envelope=FlowRuntimeEnvelopeDefinition(
                container=ContainerProfileSelection(
                    profile="flow",
                    required=True,
                    scope=ContainerExecutionScope.RUNTIME_ENVELOPE,
                ),
            ),
        )

        result = await FlowExecutor().run(flow)

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.execution.container_runtime_envelope_unavailable",
        )

    async def test_runtime_envelope_fails_closed_in_strict_runner(
        self,
    ) -> None:
        settings = _settings(
            profiles={"flow": _profile("flow")},
            default_profile="flow",
        )
        compiled = await compile_flow_definition(
            _definition(
                container=settings,
                runtime_envelope=FlowRuntimeEnvelopeDefinition(
                    container=ContainerProfileSelection(
                        profile="flow",
                        required=True,
                        scope=ContainerExecutionScope.RUNTIME_ENVELOPE,
                    ),
                ),
            )
        )
        assert compiled.plan is not None

        result = await execute_flow_plan(
            compiled.plan,
            _constant_runner("should not run"),
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.execution.container_runtime_envelope_unavailable",
        )

    async def test_runtime_envelope_runner_receives_resume_state(
        self,
    ) -> None:
        settings = _settings(
            profiles={"flow": _profile("flow")},
            default_profile="flow",
        )
        compiled = await compile_flow_definition(
            _definition(
                container=settings,
                runtime_envelope=FlowRuntimeEnvelopeDefinition(
                    container=ContainerProfileSelection(
                        profile="flow",
                        required=True,
                        scope=ContainerExecutionScope.RUNTIME_ENVELOPE,
                    ),
                ),
            )
        )
        assert compiled.plan is not None
        resume_trace = FlowExecutionTrace.from_plan(compiled.plan)
        resume_outputs = {"work": {"value": "previous"}}
        resume_decisions = {"work": {"approved": True}}

        class FakeRuntimeEnvelopeRunner:
            trusted_runtime_envelope_runner = True

            def __init__(self) -> None:
                self.plan: FlowExecutionPlan | None = None
                self.resume_trace: FlowExecutionTrace | None = None
                self.resume_node_outputs: (
                    Mapping[str, Mapping[str, object]] | None
                ) = None
                self.resume_decisions: (
                    Mapping[str, Mapping[str, object]] | None
                ) = None

            async def run_flow_runtime_envelope(
                self,
                plan: FlowExecutionPlan,
                *,
                inputs: Mapping[str, object],
                cancellation_checker: object | None,
                event_listener: object | None,
                stream_session: object | None,
                concurrency_limit: int,
                resume_trace: FlowExecutionTrace | None,
                resume_node_outputs: Mapping[str, Mapping[str, object]],
                resume_decisions: Mapping[str, Mapping[str, object]],
            ) -> FlowPlanExecutionResult:
                self.plan = plan
                self.resume_trace = resume_trace
                self.resume_node_outputs = resume_node_outputs
                self.resume_decisions = resume_decisions
                return FlowPlanExecutionResult(
                    trace=resume_trace or FlowExecutionTrace.from_plan(plan),
                    node_outputs=resume_node_outputs,
                )

        runner = FakeRuntimeEnvelopeRunner()

        result = await execute_flow_plan(
            compiled.plan,
            _constant_runner("should not run"),
            resume_trace=resume_trace,
            resume_node_outputs=resume_outputs,
            resume_decisions=resume_decisions,
            runtime_envelope_runner=runner,
        )

        self.assertTrue(result.ok, result.public_diagnostics)
        self.assertIs(runner.plan, compiled.plan)
        self.assertIs(runner.resume_trace, resume_trace)
        self.assertEqual(runner.resume_node_outputs, resume_outputs)
        self.assertEqual(runner.resume_decisions, resume_decisions)
        self.assertEqual(result.node_outputs, resume_outputs)

    async def test_runtime_envelope_fingerprint_is_resume_metadata(
        self,
    ) -> None:
        settings = _settings(
            profiles={"flow": _profile("flow")},
            default_profile="flow",
        )
        compiled = await compile_flow_definition(
            _definition(
                container=settings,
                runtime_envelope=FlowRuntimeEnvelopeDefinition(
                    container=ContainerProfileSelection(
                        profile="flow",
                        required=True,
                        scope=ContainerExecutionScope.RUNTIME_ENVELOPE,
                    ),
                    readiness_timeout_seconds=30,
                ),
            )
        )

        self.assertTrue(compiled.ok, compiled.public_diagnostics)
        assert compiled.plan is not None
        assert compiled.plan.runtime_envelope is not None
        metadata = flow_resume_isolation_metadata(compiled.plan)
        isolation = metadata["isolation"]
        assert isinstance(isolation, Mapping)
        runtime_envelope = isolation["runtime_envelope"]
        assert isinstance(runtime_envelope, Mapping)
        self.assertEqual(
            runtime_envelope["plan_fingerprint"],
            compiled.plan.runtime_envelope.plan_fingerprint,
        )

        repeated = await compile_flow_definition(
            _definition(
                container=settings,
                runtime_envelope=FlowRuntimeEnvelopeDefinition(
                    container=ContainerProfileSelection(
                        profile="flow",
                        required=True,
                        scope=ContainerExecutionScope.RUNTIME_ENVELOPE,
                    ),
                    readiness_timeout_seconds=30,
                ),
            )
        )

        self.assertTrue(repeated.ok, repeated.public_diagnostics)
        assert repeated.plan is not None
        assert repeated.plan.runtime_envelope is not None
        self.assertEqual(
            repeated.plan.runtime_envelope.plan_fingerprint,
            compiled.plan.runtime_envelope.plan_fingerprint,
        )

    async def test_runtime_envelope_runner_must_be_trusted(self) -> None:
        settings = _settings(
            profiles={"flow": _profile("flow")},
            default_profile="flow",
        )
        compiled = await compile_flow_definition(
            _definition(
                container=settings,
                runtime_envelope=FlowRuntimeEnvelopeDefinition(
                    container=ContainerProfileSelection(
                        profile="flow",
                        required=True,
                        scope=ContainerExecutionScope.RUNTIME_ENVELOPE,
                    ),
                ),
            )
        )
        assert compiled.plan is not None
        plan = compiled.plan

        class UntrustedRuntimeEnvelopeRunner:
            async def run_flow_runtime_envelope(
                self,
                *args: object,
                **kwargs: object,
            ) -> FlowPlanExecutionResult:
                return FlowPlanExecutionResult(
                    trace=FlowExecutionTrace.from_plan(plan),
                )

        with self.assertRaisesRegex(
            AssertionError,
            "runtime envelope runner must be trusted",
        ):
            await execute_flow_plan(
                plan,
                _constant_runner("should not run"),
                runtime_envelope_runner=cast(
                    Any,
                    UntrustedRuntimeEnvelopeRunner(),
                ),
            )

        with self.assertRaisesRegex(
            AssertionError,
            "runtime envelope runner must be trusted",
        ):
            FlowExecutor(
                runtime_envelope_runner=cast(
                    Any,
                    UntrustedRuntimeEnvelopeRunner(),
                )
            )

    async def test_node_widening_is_rejected(self) -> None:
        defaults = _settings(
            profiles={"wide": _profile("wide", cpu_count=1)},
            default_profile="wide",
        )

        result = await compile_flow_definition(
            _definition(
                container=defaults,
                node=FlowNodeDefinition(
                    name="work",
                    type="pass-through",
                    container=_override(
                        resources=ContainerResourceLimits(cpu_count=2),
                    ),
                ),
            ),
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.container_policy_invalid",
        )
        self.assertIn(
            "resource limit cannot widen",
            _diagnostic_hint(result.diagnostics[0]),
        )

    async def test_unknown_node_profile_is_rejected(self) -> None:
        defaults = _settings(
            profiles={"wide": _profile("wide")},
            default_profile="wide",
        )

        result = await compile_flow_definition(
            _definition(
                container=defaults,
                node=FlowNodeDefinition(
                    name="work",
                    type="pass-through",
                    container=_override(profile="missing"),
                ),
            ),
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.container_policy_invalid",
        )
        self.assertIn(
            "selected profile must be allowed",
            _diagnostic_hint(result.diagnostics[0]),
        )

    async def test_parent_deadline_cannot_be_violated_by_container_timeout(
        self,
    ) -> None:
        defaults = _settings(
            profiles={"wide": _profile("wide", timeout=10)},
            default_profile="wide",
        )

        result = await compile_flow_definition(
            _definition(
                container=defaults,
                node=FlowNodeDefinition(
                    name="work",
                    type="pass-through",
                    timeout_policy=_timeout_policy(5),
                ),
            ),
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.container_policy_invalid",
        )
        self.assertIn(
            "container timeout cannot exceed node timeout",
            _diagnostic_hint(result.diagnostics[0]),
        )

    async def test_container_deadline_validation_allows_unbounded_paths(
        self,
    ) -> None:
        effective_without_profile = ContainerEffectiveSettings(
            backend=ContainerBackend.DOCKER,
            required=False,
            scope=ContainerExecutionScope.DURABLE_WORKFLOW,
            source=trusted_container_source(ContainerSurface.FLOW_TOML),
            policy_version="phase12",
            profile_registry_id="phase12",
        )
        cases = (
            (
                FlowNodeDefinition(
                    name="work",
                    type="pass-through",
                    timeout_policy=FlowTimeoutPolicy(),
                ),
                _effective_settings(_profile("wide", timeout=10)),
            ),
            (
                FlowNodeDefinition(
                    name="work",
                    type="pass-through",
                    timeout_policy=_timeout_policy(5),
                ),
                effective_without_profile,
            ),
            (
                FlowNodeDefinition(
                    name="work",
                    type="pass-through",
                    timeout_policy=_timeout_policy(5),
                ),
                _effective_settings(_profile("wide")),
            ),
        )

        for node, effective in cases:
            with self.subTest(node_timeout=node.timeout_policy):
                _validate_node_container_deadline(
                    node,
                    effective,
                )

    async def test_container_fingerprint_handles_json_config_shapes(
        self,
    ) -> None:
        settings = _effective_settings(_profile("fingerprint"))
        node = FlowNodePlan(
            name="work",
            type="pass-through",
            kind=FlowNodeKind.PASS_THROUGH,
            output_contracts=(FlowNodeContract(name="value"),),
            container=settings,
            config={
                "scalar": 1,
                "nested": {"enabled": True, "none": None},
                "items": ["a", object()],
            },
        )
        plan = FlowExecutionPlan(
            name="fingerprint-flow",
            version="1",
            revision="rev",
            inputs=(
                FlowInputDefinition(name="payload", type=FlowInputType.OBJECT),
            ),
            outputs=(
                FlowOutputDefinition(
                    name="answer",
                    type=FlowOutputType.OBJECT,
                ),
            ),
            entry_node="work",
            output_selectors={"answer": parse_flow_selector("work.value")},
            nodes=(node,),
        )

        fingerprint = flow_node_container_fingerprint(plan, node)

        self.assertEqual(len(fingerprint), 64)
        self.assertEqual(
            fingerprint,
            flow_node_container_fingerprint(plan, node),
        )

    async def test_stale_manual_isolation_metadata_is_rejected(
        self,
    ) -> None:
        stale = FlowIsolationMetadata(
            mode=IsolationMode.LOCAL,
            backend=None,
            profile_registry_id=None,
            profile_name=None,
            policy_version=None,
            scope=None,
            plan_fingerprint="f" * 64,
        )
        plain_node = FlowNodePlan(
            name="work",
            type="pass-through",
            kind=FlowNodeKind.PASS_THROUGH,
        )
        settings = _effective_settings(_profile("manual"))
        isolated_node = FlowNodePlan(
            name="work",
            type="pass-through",
            kind=FlowNodeKind.PASS_THROUGH,
            container=settings,
            isolation=stale,
        )
        baseline = _manual_plan(
            FlowNodePlan(
                name="work",
                type="pass-through",
                kind=FlowNodeKind.PASS_THROUGH,
                container=settings,
            ),
            container=settings,
        )
        baseline_node = baseline.node_map["work"]
        accepted = _manual_plan(
            baseline_node,
            container=settings,
            isolation=baseline.isolation,
        )
        self.assertEqual(
            accepted.node_map["work"].isolation,
            baseline_node.isolation,
        )

        with self.assertRaisesRegex(
            AssertionError,
            "flow plan cannot carry stale isolation metadata",
        ):
            _manual_plan(plain_node, isolation=stale)

        with self.assertRaisesRegex(
            AssertionError,
            "flow node cannot carry stale isolation metadata",
        ):
            _manual_plan(
                FlowNodePlan(
                    name="work",
                    type="pass-through",
                    kind=FlowNodeKind.PASS_THROUGH,
                    isolation=stale,
                )
            )

        with self.assertRaisesRegex(
            AssertionError,
            "flow plan isolation metadata changed",
        ):
            _manual_plan(plain_node, container=settings, isolation=stale)

        with self.assertRaisesRegex(
            AssertionError,
            "flow node isolation metadata changed",
        ):
            _manual_plan(isolated_node, container=settings)

    async def test_subflow_node_profile_selection_uses_allowlist(
        self,
    ) -> None:
        registry = FlowNodeRegistry(
            {"subflow": lambda definition: Node(definition.name)},
            {
                "subflow": FlowNodeMetadata(
                    kind=FlowNodeKind.SUBFLOW,
                    output_contract=FlowNodeContract(name="value"),
                )
            },
        )
        defaults = _settings(
            profiles={"wide": _profile("wide")},
            default_profile="wide",
        )

        result = await compile_flow_definition(
            _definition(
                container=defaults,
                node=FlowNodeDefinition(
                    name="work",
                    type="subflow",
                    container=_override(profile="missing"),
                ),
            ),
            registry,
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.container_policy_invalid",
        )

    async def test_subflow_node_metadata_is_compiled(self) -> None:
        registry = FlowNodeRegistry(
            {"subflow": lambda definition: Node(definition.name)},
            {
                "subflow": FlowNodeMetadata(
                    kind=FlowNodeKind.SUBFLOW,
                    supports_ref=True,
                    output_contract=FlowNodeContract(name="value"),
                )
            },
        ).register_subflow_resolver("subflow", _SubflowResolver())

        result = await compile_flow_definition(
            _definition(
                node=FlowNodeDefinition(
                    name="work",
                    type="subflow",
                    ref="child.flow.toml",
                )
            ),
            registry,
        )

        self.assertTrue(result.ok, result.public_diagnostics)
        assert result.plan is not None
        metadata = result.plan.node_map["work"].metadata
        subflow = metadata["subflow"]
        self.assertIsInstance(subflow, Mapping)
        assert isinstance(subflow, Mapping)
        self.assertEqual(subflow["ref"], "child.flow.toml")


class FlowContainerLoaderTestCase(IsolatedAsyncioTestCase):
    async def test_container_syntax_diagnostic_short_circuits_loading(
        self,
    ) -> None:
        diagnostic = ContainerDiagnostic(
            code=ContainerDiagnosticCode.UNSUPPORTED_SYNTAX,
            path="runtime.container",
            message="Unsupported.",
            hint="Remove it.",
            category=ContainerDiagnosticCategory.UNSUPPORTED,
        )

        with patch(
            "avalan.flow.loader.container_syntax_diagnostics",
            return_value=(diagnostic,),
        ):
            result = await FlowDefinitionLoader().loads_result(
                _strict_source()
            )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.issues[0].code,
            "container.unsupported_syntax",
        )
        self.assertIsNone(result.definition)

    async def test_flow_toml_container_authority_fails_closed(self) -> None:
        source = serialize_flow_definition(
            _definition(
                container=_settings(
                    profiles={"wide": _profile("wide")},
                    default_profile="wide",
                )
            )
        )

        result = await loads_flow_definition_result(source)

        self.assertFalse(result.ok)
        self.assertEqual(
            result.issues[0].code,
            "flow.untrusted_container_authority",
        )

        validation = await FlowDefinitionLoader().loads_validation_result(
            source
        )
        self.assertEqual(
            validation.issues[0].code,
            "flow.untrusted_container_authority",
        )

    async def test_runtime_section_without_container_is_accepted(
        self,
    ) -> None:
        result = await FlowDefinitionLoader().loads_validation_result(
            _strict_source("[runtime]\n")
        )

        self.assertTrue(result.ok, result.public_diagnostics)
        assert result.definition is not None
        self.assertIsNone(result.definition.container)
        self.assertIsNone(result.definition.runtime_envelope)

    async def test_unsupported_isolation_syntax_fails_closed(self) -> None:
        cases = (
            ('[isolation]\nmode = "sandbox"\n', "isolation"),
            ('[runtime.isolation]\nprofile = "wide"\n', "runtime.isolation"),
            ('[runtime.sandbox]\nprofile = "wide"\n', "runtime.sandbox"),
            (
                '[nodes.work.runtime.isolation]\nprofile = "wide"\n',
                "nodes.work.runtime.isolation",
            ),
            (
                '[nodes.work.runtime.sandboxProfile]\nname = "wide"\n',
                "nodes.work.runtime.sandboxProfile",
            ),
        )

        for section, path in cases:
            with self.subTest(path=path):
                result = await FlowDefinitionLoader().loads_validation_result(
                    _strict_source(section)
                )

                self.assertFalse(result.ok)
                self.assertEqual(
                    result.issues[0].code,
                    "isolation.unsupported_syntax",
                )
                self.assertEqual(result.issues[0].path, path)

    async def test_runtime_envelope_must_be_table(self) -> None:
        result = await FlowDefinitionLoader().loads_validation_result(
            _strict_source('[runtime.container]\nenvelope = "bad"\n')
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.issues[0].code,
            "flow.invalid_section",
        )
        self.assertEqual(result.issues[0].path, "runtime.container.envelope")

    async def test_runtime_envelope_defaults_readiness_timeout(self) -> None:
        with patch(
            "avalan.flow.loader.container_selection_from_mapping",
            return_value=ContainerProfileSelection(
                profile="flow",
                scope=ContainerExecutionScope.RUNTIME_ENVELOPE,
            ),
        ):
            result = await FlowDefinitionLoader().loads_validation_result(
                _strict_source(
                    '[runtime.container.envelope]\nprofile = "flow"\n'
                )
            )

        self.assertTrue(result.ok, result.public_diagnostics)
        assert result.definition is not None
        assert result.definition.runtime_envelope is not None
        self.assertEqual(
            result.definition.runtime_envelope.readiness_timeout_seconds,
            30,
        )

    async def test_runtime_envelope_invalid_selection_reports_issue(
        self,
    ) -> None:
        result = await FlowDefinitionLoader().loads_validation_result(
            _strict_source('[runtime.container.envelope]\nprofile = ""\n')
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.issues[0].code,
            "flow.invalid_container_runtime_envelope",
        )
        self.assertEqual(result.issues[0].path, "runtime.container.envelope")

    async def test_legacy_node_container_syntax_reports_policy_boundary(
        self,
    ) -> None:
        source = """
[flow]
name = "legacy-node-container"
entrypoint = "work"
output_node = "work"

[nodes.work]
type = "pass-through"

[nodes.work.runtime.container]
profile = "wide"
"""

        result = await FlowDefinitionLoader().loads_result(source)

        self.assertFalse(result.ok)
        self.assertEqual(
            result.issues[0].code,
            "flow.container_legacy_runtime_unsupported",
        )

    async def test_build_flow_rejects_container_runtime_definition(
        self,
    ) -> None:
        with self.assertRaises(FlowNodeConfigurationError) as raised:
            build_flow(
                _definition(
                    container=_settings(
                        profiles={"wide": _profile("wide")},
                        default_profile="wide",
                    )
                )
            )
        self.assertEqual(
            raised.exception.code,
            "flow.container_legacy_runtime_unsupported",
        )
        self.assertIn(
            "Legacy flow runtime cannot enforce container policy",
            raised.exception.message,
        )

    async def test_node_runtime_must_be_table(self) -> None:
        result = await FlowDefinitionLoader().loads_validation_result(
            _strict_source(node_extra='runtime = "bad"\n')
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.issues[0].code,
            "flow.invalid_section",
        )
        self.assertEqual(result.issues[0].path, "nodes.work.runtime")

    async def test_node_container_invalid_override_reports_issue(self) -> None:
        result = await FlowDefinitionLoader().loads_validation_result(
            _strict_source("""
[nodes.work.runtime.container]
required = "yes"
""")
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.issues[0].code, "flow.invalid_node_container")
        self.assertEqual(result.issues[0].path, "nodes.work.runtime.container")
        self.assertIn("required", result.issues[0].hint)

    async def test_node_container_blank_assertion_uses_generic_hint(
        self,
    ) -> None:
        with patch(
            "avalan.flow.loader.ContainerSettingsOverride.from_dict",
            side_effect=AssertionError(),
        ):
            result = await FlowDefinitionLoader().loads_validation_result(
                _strict_source("""
[nodes.work.runtime.container]
profile = "wide"
""")
            )

        self.assertFalse(result.ok)
        self.assertEqual(result.issues[0].code, "flow.invalid_node_container")
        self.assertEqual(
            result.issues[0].hint,
            "Use only supported, trusted flow configuration values.",
        )

    async def test_runtime_envelope_readiness_timeout_must_be_positive(
        self,
    ) -> None:
        source = """
[flow]
name = "bad-envelope"
version = "2026-06-24"

[[inputs]]
name = "payload"
type = "object"

[[outputs]]
name = "answer"
type = "object"

[entry]
type = "node"
node = "work"

[output_behavior]
type = "map"

[output_behavior.outputs]
answer = "work.value"

[runtime.container.envelope]
profile = "flow"
readiness_timeout_seconds = 0

[nodes.work]
type = "pass-through"
"""

        result = await FlowDefinitionLoader().loads_validation_result(source)

        self.assertFalse(result.ok)
        self.assertEqual(
            result.issues[0].code,
            "flow.invalid_container_runtime_envelope",
        )
        self.assertEqual(
            result.issues[0].path,
            "runtime.container.envelope.readiness_timeout_seconds",
        )

    async def test_unsupported_node_runtime_sections_are_rejected(
        self,
    ) -> None:
        source = """
[flow]
name = "bad"

[[inputs]]
name = "payload"
type = "object"

[[outputs]]
name = "answer"
type = "object"

[entry]
type = "node"
node = "work"

[output_behavior]
type = "map"

[output_behavior.outputs]
answer = "work.value"

[nodes.work]
type = "pass-through"

[nodes.work.runtime.permissions]
network = "full"
"""

        result = await FlowDefinitionLoader().loads_validation_result(source)

        self.assertFalse(result.ok)
        self.assertEqual(result.issues[0].code, "flow.unsupported_field")
        self.assertEqual(
            result.issues[0].path,
            "nodes.work.runtime.permissions",
        )

    async def test_graph_authoring_cannot_grant_container_permissions(
        self,
    ) -> None:
        source = """
[flow]
name = "graph-bad"

[[inputs]]
name = "payload"
type = "object"

[[outputs]]
name = "answer"
type = "object"

[entry]
type = "node"
node = "work"

[output_behavior]
type = "map"

[output_behavior.outputs]
answer = "work.value"

[nodes.work]
type = "pass-through"

[graph]
format = "mermaid"
source = "inline"
mode = "strict"
diagram = "flowchart TD\\n  work"

[graph.edges.synthetic.runtime.container]
profile = "wide"
"""

        result = await FlowDefinitionLoader().loads_validation_result(source)

        self.assertFalse(result.ok)
        self.assertEqual(result.issues[0].code, "flow.unsupported_field")
        self.assertEqual(result.issues[0].path, "graph.edges.metadata.runtime")


class FlowContainerSerializerTestCase(IsolatedAsyncioTestCase):
    async def test_serializes_runtime_envelope_and_node_container(
        self,
    ) -> None:
        source = serialize_flow_definition(
            _definition(
                container=_settings(
                    profiles={"flow": _profile("flow")},
                    default_profile="flow",
                ),
                runtime_envelope=FlowRuntimeEnvelopeDefinition(
                    container=ContainerProfileSelection(
                        profile="flow",
                        required=True,
                        scope=ContainerExecutionScope.RUNTIME_ENVELOPE,
                    ),
                    readiness_timeout_seconds=45,
                ),
                node=FlowNodeDefinition(
                    name="work",
                    type="pass-through",
                    container=_override(
                        resources=ContainerResourceLimits(cpu_count=1),
                    ),
                ),
            )
        )

        self.assertIn("[runtime.container.envelope]", source)
        self.assertIn("readiness_timeout_seconds = 45", source)
        self.assertIn("[nodes.work.runtime.container]", source)
        self.assertIn("cpu_count = 1", source)


class FlowContainerRuntimeTestCase(IsolatedAsyncioTestCase):
    async def test_strict_container_flow_run_streams_lifecycle_events(
        self,
    ) -> None:
        plan = _runtime_plan(_effective_settings(_profile("run")))
        events: list[CanonicalStreamItem] = []

        result = await execute_flow_plan(
            plan,
            _constant_runner("ok"),
            event_listener=events.append,
        )

        self.assertTrue(result.ok, result.public_diagnostics)
        self.assertEqual(result.outputs["answer"], "ok")
        self.assertEqual(
            _container_event_names(events),
            (
                "policy_evaluation",
                "backend_selection",
                "container_create",
                "container_start",
                "result_recorded",
                "cleanup",
            ),
        )

    async def test_container_review_can_pause_and_resume_flow(
        self,
    ) -> None:
        plan = _runtime_plan(
            _effective_settings(
                _profile(
                    "review",
                    escalation=ContainerEscalationMode.REQUIRE_REVIEW,
                    cpu_count=1,
                )
            )
        )
        events: list[CanonicalStreamItem] = []

        paused = await execute_flow_plan(
            plan,
            _constant_runner("approved"),
            event_listener=events.append,
        )

        self.assertTrue(paused.ok, paused.public_diagnostics)
        self.assertEqual(paused.pause_tokens.keys(), {"work"})
        self.assertEqual(
            _container_event_names(events),
            ("policy_evaluation", "review_request"),
        )

        resumed = await execute_flow_plan(
            plan,
            _constant_runner("approved"),
            event_listener=events.append,
            resume_trace=paused.trace,
            resume_node_outputs=paused.node_outputs,
            resume_decisions={
                "work": {
                    "decision": "approved",
                    "approval": _approval_payload(plan),
                }
            },
        )

        self.assertTrue(resumed.ok, resumed.public_diagnostics)
        self.assertEqual(resumed.outputs["answer"], "approved")
        self.assertIn("review_decision", _container_event_names(events))
        self.assertIn("result_recorded", _container_event_names(events))

    async def test_container_resume_rejects_stale_isolation_metadata(
        self,
    ) -> None:
        paused_plan = _runtime_plan(
            _effective_settings(
                _profile(
                    "review",
                    escalation=ContainerEscalationMode.REQUIRE_REVIEW,
                    cpu_count=1,
                )
            )
        )
        resumed_plan = _runtime_plan(
            _effective_settings(
                _profile(
                    "review",
                    escalation=ContainerEscalationMode.REQUIRE_REVIEW,
                    cpu_count=2,
                )
            )
        )
        paused = await execute_flow_plan(
            paused_plan,
            _constant_runner("ignored"),
        )
        called = False

        async def runner(
            node: FlowNodePlan,
            inputs: Mapping[str, object],
        ) -> object:
            nonlocal called
            called = True
            return "should-not-run"

        result = await execute_flow_plan(
            resumed_plan,
            runner,
            resume_trace=paused.trace,
            resume_node_outputs=paused.node_outputs,
            resume_decisions={
                "work": {
                    "decision": "approved",
                    "approval": _approval_payload(resumed_plan),
                }
            },
        )

        self.assertFalse(result.ok)
        self.assertFalse(called)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.execution.isolation_resume_metadata_stale",
        )

    async def test_runtime_envelope_resume_rejects_stale_metadata(
        self,
    ) -> None:
        settings = _settings(
            profiles={"flow": _profile("flow")},
            default_profile="flow",
        )
        paused = await compile_flow_definition(
            _definition(
                container=settings,
                runtime_envelope=FlowRuntimeEnvelopeDefinition(
                    container=ContainerProfileSelection(
                        profile="flow",
                        required=True,
                        scope=ContainerExecutionScope.RUNTIME_ENVELOPE,
                    ),
                    readiness_timeout_seconds=30,
                ),
            )
        )
        resumed = await compile_flow_definition(
            _definition(
                container=settings,
                runtime_envelope=FlowRuntimeEnvelopeDefinition(
                    container=ContainerProfileSelection(
                        profile="flow",
                        required=True,
                        scope=ContainerExecutionScope.RUNTIME_ENVELOPE,
                    ),
                    readiness_timeout_seconds=45,
                ),
            )
        )
        self.assertTrue(paused.ok, paused.public_diagnostics)
        self.assertTrue(resumed.ok, resumed.public_diagnostics)
        assert paused.plan is not None
        assert resumed.plan is not None
        paused_plan = paused.plan
        resumed_plan = resumed.plan
        assert paused_plan.runtime_envelope is not None
        assert resumed_plan.runtime_envelope is not None
        self.assertNotEqual(
            paused_plan.runtime_envelope.plan_fingerprint,
            resumed_plan.runtime_envelope.plan_fingerprint,
        )
        called = False

        class FakeRuntimeEnvelopeRunner:
            trusted_runtime_envelope_runner = True

            async def run_flow_runtime_envelope(
                self,
                *args: object,
                **kwargs: object,
            ) -> FlowPlanExecutionResult:
                nonlocal called
                called = True
                return FlowPlanExecutionResult(
                    trace=FlowExecutionTrace.from_plan(resumed_plan),
                )

        result = await execute_flow_plan(
            resumed_plan,
            _constant_runner("should not run"),
            resume_trace=FlowExecutionTrace.from_plan(paused_plan),
            runtime_envelope_runner=FakeRuntimeEnvelopeRunner(),
        )

        self.assertFalse(result.ok)
        self.assertFalse(called)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.execution.isolation_resume_metadata_stale",
        )

    async def test_container_resume_rejects_missing_isolation_metadata(
        self,
    ) -> None:
        plan = _runtime_plan(
            _effective_settings(
                _profile(
                    "review",
                    escalation=ContainerEscalationMode.REQUIRE_REVIEW,
                    cpu_count=1,
                )
            )
        )
        paused = await execute_flow_plan(plan, _constant_runner("ignored"))
        trace = replace(paused.trace, metadata={})
        called = False

        async def runner(
            node: FlowNodePlan,
            inputs: Mapping[str, object],
        ) -> object:
            nonlocal called
            called = True
            return "should-not-run"

        result = await execute_flow_plan(
            plan,
            runner,
            resume_trace=trace,
            resume_node_outputs=paused.node_outputs,
            resume_decisions={
                "work": {
                    "decision": "approved",
                    "approval": _approval_payload(plan),
                }
            },
        )

        self.assertFalse(result.ok)
        self.assertFalse(called)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.execution.isolation_resume_metadata_widened",
        )

    async def test_isolation_resume_metadata_helpers_cover_defensive_paths(
        self,
    ) -> None:
        isolated_plan = _runtime_plan(
            _effective_settings(
                _profile(
                    "review",
                    escalation=ContainerEscalationMode.REQUIRE_REVIEW,
                    cpu_count=1,
                )
            )
        )
        plain_plan = _manual_plan(
            FlowNodePlan(
                name="work",
                type="pass-through",
                kind=FlowNodeKind.PASS_THROUGH,
            )
        )
        isolated_trace = FlowExecutionTrace.from_plan(isolated_plan)
        plain_trace = replace(
            FlowExecutionTrace.from_plan(plain_plan),
            metadata=isolated_trace.metadata,
        )
        missing_node_trace = replace(
            isolated_trace,
            metadata={
                "isolation": {
                    "version": "phase9",
                    "flow": None,
                    "nodes": {},
                }
            },
        )
        malformed_trace = replace(
            isolated_trace,
            metadata={
                "isolation": {
                    "version": "phase9",
                    "flow": None,
                    "nodes": (),
                }
            },
        )

        self.assertEqual(
            _resume_isolation_metadata_diagnostics(
                plain_plan,
                FlowExecutionTrace.from_plan(plain_plan),
            ),
            (),
        )
        self.assertEqual(
            _resume_isolation_metadata_diagnostics(
                plain_plan,
                plain_trace,
            )[0].code,
            "flow.execution.isolation_resume_metadata_stale",
        )
        self.assertEqual(
            _resume_isolation_metadata_diagnostics(
                isolated_plan,
                missing_node_trace,
            )[0].code,
            "flow.execution.isolation_resume_metadata_widened",
        )
        self.assertEqual(
            _resume_isolation_metadata_diagnostics(
                isolated_plan,
                malformed_trace,
            )[0].code,
            "flow.execution.isolation_resume_metadata_stale",
        )
        self.assertIsNone(_resume_isolation_node_names({"nodes": ()}))
        self.assertIsNone(_resume_isolation_node_names({"nodes": {"": {}}}))

        with patch(
            "avalan.flow.runtime.flow_resume_isolation_metadata",
            return_value={"isolation": "bad"},
        ):
            self.assertEqual(
                _resume_isolation_metadata_diagnostics(
                    isolated_plan,
                    isolated_trace,
                ),
                (),
            )

    async def test_container_resume_requeues_after_processed_nodes(
        self,
    ) -> None:
        settings = _effective_settings(
            _profile(
                "review",
                escalation=ContainerEscalationMode.REQUIRE_REVIEW,
                cpu_count=1,
            )
        )
        work = FlowNodePlan(
            name="work",
            type="pass-through",
            kind=FlowNodeKind.PASS_THROUGH,
            output_contracts=(FlowNodeContract(name="value"),),
            container=settings,
        )
        plan = FlowExecutionPlan(
            name="resume-tail",
            version=None,
            revision=None,
            inputs=(
                FlowInputDefinition(name="payload", type=FlowInputType.OBJECT),
            ),
            outputs=(
                FlowOutputDefinition(
                    name="answer",
                    type=FlowOutputType.OBJECT,
                ),
            ),
            entry_node="start",
            output_selectors={"answer": parse_flow_selector("work.value")},
            nodes=(
                FlowNodePlan(
                    name="start",
                    type="pass-through",
                    kind=FlowNodeKind.PASS_THROUGH,
                    output_contracts=(FlowNodeContract(name="value"),),
                ),
                work,
            ),
        )
        trace = (
            FlowExecutionTrace.from_plan(plan)
            .with_node_state("start", FlowNodeState.SUCCEEDED, attempts=1)
            .with_node_state("work", FlowNodeState.PAUSED, attempts=1)
        )

        result = await execute_flow_plan(
            plan,
            _constant_runner("resumed"),
            resume_trace=trace,
            resume_node_outputs={"start": {"value": "done"}},
            resume_decisions={
                "work": {
                    "decision": "approved",
                    "approval": _approval_payload(plan),
                }
            },
        )

        self.assertTrue(result.ok, result.public_diagnostics)
        self.assertEqual(result.outputs["answer"], "resumed")

    async def test_container_review_can_resume_with_denial(
        self,
    ) -> None:
        plan = _runtime_plan(
            _effective_settings(
                _profile(
                    "review",
                    escalation=ContainerEscalationMode.REQUIRE_REVIEW,
                    cpu_count=1,
                )
            )
        )
        paused = await execute_flow_plan(plan, _constant_runner("ignored"))
        events: list[CanonicalStreamItem] = []

        result = await execute_flow_plan(
            plan,
            _constant_runner("ignored"),
            event_listener=events.append,
            resume_trace=paused.trace,
            resume_node_outputs=paused.node_outputs,
            resume_decisions={"work": {"decision": "denied"}},
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.execution.container_review_denied",
        )
        self.assertEqual(
            result.trace.nodes[0].state,
            FlowNodeState.FAILED,
        )
        self.assertIn("review_decision", _container_event_names(events))

    async def test_container_review_requires_durable_reviewer_identity(
        self,
    ) -> None:
        plan = _runtime_plan(
            _effective_settings(
                _profile(
                    "review",
                    escalation=ContainerEscalationMode.REQUIRE_REVIEW,
                    cpu_count=1,
                )
            )
        )
        paused = await execute_flow_plan(plan, _constant_runner("ignored"))

        result = await execute_flow_plan(
            plan,
            _constant_runner("ignored"),
            resume_trace=paused.trace,
            resume_node_outputs=paused.node_outputs,
            resume_decisions={"work": {"decision": "approved"}},
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.execution.missing_container_approval",
        )

    async def test_container_approval_from_resume_ignores_non_approvals(
        self,
    ) -> None:
        plan = _runtime_plan(
            _effective_settings(
                _profile(
                    "review",
                    escalation=ContainerEscalationMode.REQUIRE_REVIEW,
                    cpu_count=1,
                )
            )
        )
        node = plan.node_map["work"]

        self.assertIsNone(
            _container_approval_from_resume(
                plan,
                node,
                {"decision": "denied"},
            )
        )
        self.assertIsNone(
            _container_approval_from_resume(
                plan,
                node,
                {"decision": "approved", "approval": "not-a-record"},
            )
        )

    async def test_container_review_rejects_malformed_approval_mapping(
        self,
    ) -> None:
        plan = _runtime_plan(
            _effective_settings(
                _profile(
                    "review",
                    escalation=ContainerEscalationMode.REQUIRE_REVIEW,
                    cpu_count=1,
                )
            )
        )
        paused = await execute_flow_plan(plan, _constant_runner("ignored"))
        events: list[CanonicalStreamItem] = []

        result = await execute_flow_plan(
            plan,
            _constant_runner("ignored"),
            event_listener=events.append,
            resume_trace=paused.trace,
            resume_node_outputs=paused.node_outputs,
            resume_decisions={
                "work": {
                    "decision": "approved",
                    "approval": {"approved_triggers": ["network"]},
                }
            },
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.execution.invalid_container_approval",
        )
        self.assertIn("review_decision", _container_event_names(events))

    async def test_container_review_rejects_invalid_resume_targets(
        self,
    ) -> None:
        plan = _runtime_plan(
            _effective_settings(
                _profile(
                    "review",
                    escalation=ContainerEscalationMode.REQUIRE_REVIEW,
                    cpu_count=1,
                )
            )
        )
        paused = await execute_flow_plan(plan, _constant_runner("ignored"))
        cases = (
            (
                FlowExecutionTrace.from_plan(plan),
                {"decision": "denied"},
                "flow.execution.invalid_resume_state",
            ),
            (
                paused.trace,
                {"decision": "maybe"},
                "flow.execution.invalid_container_review_decision",
            ),
        )

        for trace, decision, code in cases:
            with self.subTest(code=code):
                result = await execute_flow_plan(
                    plan,
                    _constant_runner("ignored"),
                    resume_trace=trace,
                    resume_node_outputs=paused.node_outputs,
                    resume_decisions={"work": decision},
                )

                self.assertFalse(result.ok)
                self.assertEqual(result.diagnostics[0].code, code)

    async def test_container_review_rejects_mismatched_approval_record(
        self,
    ) -> None:
        plan = _runtime_plan(
            _effective_settings(
                _profile(
                    "review",
                    escalation=ContainerEscalationMode.REQUIRE_REVIEW,
                    cpu_count=1,
                )
            )
        )
        paused = await execute_flow_plan(plan, _constant_runner("ignored"))
        approval = dict(_approval_payload(plan))
        approval["plan_fingerprint"] = "different"

        result = await execute_flow_plan(
            plan,
            _constant_runner("ignored"),
            resume_trace=paused.trace,
            resume_node_outputs=paused.node_outputs,
            resume_decisions={
                "work": {"decision": "approved", "approval": approval}
            },
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.execution.container_review_invalid",
        )

    async def test_container_review_rejects_mismatched_attempt_id(
        self,
    ) -> None:
        plan = _runtime_plan(
            _effective_settings(
                _profile(
                    "review",
                    escalation=ContainerEscalationMode.REQUIRE_REVIEW,
                    cpu_count=1,
                )
            )
        )
        paused = await execute_flow_plan(plan, _constant_runner("ignored"))
        approval = dict(_approval_payload(plan))
        approval["attempt_id"] = "attempt-1"

        result = await execute_flow_plan(
            plan,
            _constant_runner("ignored"),
            resume_trace=paused.trace,
            resume_node_outputs=paused.node_outputs,
            resume_decisions={
                "work": {"decision": "approved", "approval": approval}
            },
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.execution.container_review_invalid",
        )

    async def test_container_review_rejects_stale_approval_record(
        self,
    ) -> None:
        plan = _runtime_plan(
            _effective_settings(
                _profile(
                    "review",
                    escalation=ContainerEscalationMode.REQUIRE_REVIEW,
                    cpu_count=1,
                )
            )
        )
        paused = await execute_flow_plan(plan, _constant_runner("ignored"))
        approval = dict(_approval_payload(plan))
        approval["expires_at_seconds"] = 1

        result = await execute_flow_plan(
            plan,
            _constant_runner("ignored"),
            resume_trace=paused.trace,
            resume_node_outputs=paused.node_outputs,
            resume_decisions={
                "work": {"decision": "approved", "approval": approval}
            },
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.execution.container_review_invalid",
        )

    async def test_container_flow_cancel_retry_and_timeout_paths_emit_cleanup(
        self,
    ) -> None:
        cancel_events: list[CanonicalStreamItem] = []
        cancel_result = await execute_flow_plan(
            _runtime_plan(_effective_settings(_profile("cancel"))),
            _cancel_runner,
            event_listener=cancel_events.append,
        )
        self.assertFalse(cancel_result.ok)
        self.assertEqual(
            cancel_result.trace.nodes[0].state,
            FlowNodeState.CANCELLED,
        )
        self.assertIn("cleanup", _container_event_names(cancel_events))

        retry_events: list[CanonicalStreamItem] = []
        retry_plan = _runtime_plan(
            _effective_settings(_profile("retry")),
            retry=FlowRetryPlan(
                max_attempts=2,
                backoff=FlowRetryBackoffStrategy.NONE,
                retryable_categories=("error",),
            ),
        )
        retry_result = await execute_flow_plan(
            retry_plan,
            _flaky_runner(),
            event_listener=retry_events.append,
        )
        self.assertTrue(retry_result.ok, retry_result.public_diagnostics)
        self.assertEqual(retry_result.trace.nodes[0].attempts, 2)
        self.assertIn("result_recorded", _container_event_names(retry_events))

        timeout_events: list[CanonicalStreamItem] = []
        timeout_result = await execute_flow_plan(
            _runtime_plan(
                _effective_settings(_profile("timeout")),
                timeout=FlowTimeoutPlan(per_attempt_seconds=0.01),
            ),
            _slow_runner,
            event_listener=timeout_events.append,
        )
        self.assertFalse(timeout_result.ok)
        self.assertEqual(
            timeout_result.diagnostics[0].code,
            "flow.execution.node_timeout",
        )
        self.assertIn("cleanup", _container_event_names(timeout_events))

    async def test_container_policy_denial_streams_denial_event(self) -> None:
        events: list[CanonicalStreamItem] = []

        result = await execute_flow_plan(
            _runtime_plan(
                _effective_settings(
                    _profile(
                        "denied",
                        network=ContainerNetworkPolicy(
                            mode=ContainerNetworkMode.FULL,
                        ),
                        escalation=ContainerEscalationMode.DENY,
                    )
                )
            ),
            _constant_runner("ignored"),
            event_listener=events.append,
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.execution.container_policy_denied",
        )
        self.assertIn("denial", _container_event_names(events))

    async def test_container_event_helpers_ignore_nodes_without_container(
        self,
    ) -> None:
        plan = _runtime_plan(_effective_settings(_profile("helpers")))
        node = FlowNodePlan(
            name="plain",
            type="pass-through",
            kind=FlowNodeKind.PASS_THROUGH,
        )
        drafts: list[_FlowEventDraft] = []

        _append_container_event_draft(
            drafts,
            node=node,
            event_type=ContainerAuditEventType.REVIEW_DECISION,
        )
        _append_container_event_draft(
            None,
            node=plan.node_map["work"],
            event_type=ContainerAuditEventType.REVIEW_DECISION,
        )
        await _emit_container_event(
            None,
            flow_stream_session(
                stream_session_id="session",
                run_id="run",
                turn_id="turn",
            ),
            plan,
            node,
            ContainerAuditEventType.REVIEW_DECISION,
        )

        self.assertEqual(drafts, [])


def _definition(
    *,
    container: ContainerSettings | None = None,
    runtime_envelope: FlowRuntimeEnvelopeDefinition | None = None,
    node: FlowNodeDefinition | None = None,
    output_selector: str = "work.value",
) -> FlowDefinition:
    return FlowDefinition(
        name="container-flow",
        version="2026-06-24",
        inputs=(
            FlowInputDefinition(name="payload", type=FlowInputType.OBJECT),
        ),
        outputs=(
            FlowOutputDefinition(name="answer", type=FlowOutputType.OBJECT),
        ),
        entry_behavior=FlowEntryBehavior(node="work"),
        output_behavior=FlowOutputBehavior(
            outputs={"answer": output_selector},
        ),
        container=container,
        runtime_envelope=runtime_envelope,
        nodes=(node or FlowNodeDefinition(name="work", type="pass-through"),),
    )


def _strict_source(
    *extra_sections: str,
    node_extra: str = "",
) -> str:
    return """
[flow]
name = "strict-container-loader"
version = "2026-06-24"

[[inputs]]
name = "payload"
type = "object"

[[outputs]]
name = "answer"
type = "object"

[entry]
type = "node"
node = "work"

[output_behavior]
type = "map"

[output_behavior.outputs]
answer = "work.value"

[nodes.work]
type = "pass-through"
""" + node_extra + "\n".join(extra_sections)


def _settings(
    *,
    profiles: Mapping[str, ContainerProfile],
    default_profile: str,
    allowed_profiles: tuple[str, ...] | None = None,
) -> ContainerSettings:
    return ContainerSettings(
        source=trusted_container_source(ContainerSurface.FLOW_TOML),
        backend=ContainerBackend.DOCKER,
        default_profile=default_profile,
        allowed_profiles=allowed_profiles or tuple(profiles),
        profiles=profiles,
        profile_registry_id="phase12",
        policy_version="phase12",
    )


def _disabled_settings() -> ContainerSettings:
    return ContainerSettings(
        source=trusted_container_source(ContainerSurface.FLOW_TOML),
        backend=ContainerBackend.NONE,
        profile_registry_id="phase12",
        policy_version="phase12",
    )


def _profile(
    name: str,
    *,
    image_reference: str = _IMAGE,
    workspace: ContainerWorkspaceMapping | None = None,
    environment: ContainerEnvironmentPolicy | None = None,
    secrets: tuple[ContainerSecretReference, ...] = (),
    network: ContainerNetworkPolicy | None = None,
    devices: ContainerDevicePolicy | None = None,
    pooling: ContainerPoolingPolicy | None = None,
    audit: ContainerAuditPolicy | None = None,
    cpu_count: int | None = None,
    timeout: int | None = None,
    escalation: ContainerEscalationMode = ContainerEscalationMode.DENY,
) -> ContainerProfile:
    return ContainerProfile(
        name=name,
        image=ContainerImagePolicy(reference=image_reference),
        workspace=workspace or ContainerWorkspaceMapping(),
        mounts=(
            ContainerMountDeclaration(
                source=".",
                target="/workspace",
                mount_type=ContainerMountType.WORKSPACE,
                access=ContainerMountAccess.READ,
            ),
            ContainerMountDeclaration(
                source=".",
                target="/input",
                mount_type=ContainerMountType.INPUT,
                access=ContainerMountAccess.READ,
            ),
        ),
        environment=environment or ContainerEnvironmentPolicy(),
        secrets=secrets,
        network=network
        or ContainerNetworkPolicy(mode=ContainerNetworkMode.NONE),
        devices=devices or ContainerDevicePolicy(),
        resources=ContainerResourceLimits(
            cpu_count=cpu_count,
            timeout_seconds=timeout,
        ),
        pooling=pooling or ContainerPoolingPolicy(),
        audit=audit or ContainerAuditPolicy(),
        escalation=ContainerEscalationPolicy(mode=escalation),
    )


def _override(
    *,
    profile: str | None = None,
    resources: ContainerResourceLimits | None = None,
    mounts: tuple[ContainerMountDeclaration, ...] | None = None,
) -> ContainerSettingsOverride:
    return ContainerSettingsOverride(
        source=ContainerSettingsSource(
            surface=ContainerSurface.FLOW_TOML,
            trust_level=ContainerTrustLevel.UNTRUSTED_FLOW,
        ),
        layer=ContainerSettingsPrecedence.FLOW_TOML,
        profile=profile,
        resources=resources,
        mounts=mounts,
    )


def _effective_settings(
    profile: ContainerProfile,
) -> ContainerEffectiveSettings:
    settings = _settings(
        profiles={profile.name: profile},
        default_profile=profile.name,
    )
    return ContainerAuthorityCaps(settings=settings).merge()


async def _compile_tool_policy_against_flow_default(
    *,
    flow_profile: ContainerProfile,
    tool_profile: ContainerProfile,
) -> FlowPlanCompileResult:
    defaults = _settings(
        profiles={flow_profile.name: flow_profile},
        default_profile=flow_profile.name,
    )
    tool_settings = _settings(
        profiles={tool_profile.name: tool_profile},
        default_profile=tool_profile.name,
    )
    registry = _tool_registry(
        ContainerAuthorityCaps(settings=tool_settings).merge()
    )
    return await compile_flow_definition(
        _definition(
            container=defaults,
            node=FlowNodeDefinition(
                name="work",
                type="tool",
                ref="container_echo",
            ),
            output_selector="work.result",
        ),
        registry,
    )


def _runtime_plan(
    settings: ContainerEffectiveSettings,
    *,
    retry: FlowRetryPlan | None = None,
    timeout: FlowTimeoutPlan | None = None,
) -> FlowExecutionPlan:
    return FlowExecutionPlan(
        name="runtime-container",
        version=None,
        revision=None,
        inputs=(
            FlowInputDefinition(name="payload", type=FlowInputType.OBJECT),
        ),
        outputs=(
            FlowOutputDefinition(name="answer", type=FlowOutputType.OBJECT),
        ),
        entry_node="work",
        output_selectors={"answer": parse_flow_selector("work.value")},
        nodes=(
            FlowNodePlan(
                name="work",
                type="pass-through",
                kind=FlowNodeKind.PASS_THROUGH,
                output_contracts=(FlowNodeContract(name="value"),),
                retry=retry,
                timeout=timeout,
                container=settings,
            ),
        ),
    )


def _manual_plan(
    node: FlowNodePlan,
    *,
    container: ContainerEffectiveSettings | None = None,
    isolation: FlowIsolationMetadata | None = None,
) -> FlowExecutionPlan:
    return FlowExecutionPlan(
        name="manual-isolation",
        version=None,
        revision=None,
        inputs=(
            FlowInputDefinition(name="payload", type=FlowInputType.OBJECT),
        ),
        outputs=(
            FlowOutputDefinition(name="answer", type=FlowOutputType.OBJECT),
        ),
        entry_node="work",
        output_selectors={"answer": parse_flow_selector("work.value")},
        nodes=(node,),
        container=container,
        isolation=isolation,
    )


def _diagnostic_hint(diagnostic: FlowDiagnostic) -> str:
    assert diagnostic.hint is not None
    return diagnostic.hint


def _approval_payload(plan: FlowExecutionPlan) -> Mapping[str, object]:
    node = plan.node_map["work"]
    assert node.container is not None
    policy_plan = ContainerPolicyPlan(
        effective_settings=node.container,
        context=ContainerPolicyContext(
            surface=ContainerReviewSurface.STRICT_FLOW,
            scope_id=f"{plan.name}:work",
        ),
        command_fingerprint=flow_node_container_fingerprint(plan, node),
    )
    return ContainerApprovalRecord.for_plan(
        policy_plan,
        reviewer_identity="operator@example.test",
        expires_at_seconds=int(time()) + 60,
    ).to_dict()


def _timeout_policy(seconds: int) -> FlowTimeoutPolicy:
    return FlowTimeoutPolicy(per_attempt_seconds=seconds)


def _tool_registry(settings: ContainerEffectiveSettings) -> FlowNodeRegistry:
    manager = ToolManager.create_instance(
        enable_tools=["container_echo"],
        available_toolsets=[ToolSet(tools=[container_echo])],
        settings=ToolManagerSettings(),
    )
    return tool_flow_node_registry(manager, container_settings=settings)


def _constant_runner(value: object) -> FlowPlanNodeRunner:
    async def run(
        node: FlowNodePlan,
        inputs: Mapping[str, object],
    ) -> object:
        return value

    return run


async def _cancel_runner(
    node: FlowNodePlan,
    inputs: Mapping[str, object],
) -> object:
    raise CancelledError()


def _flaky_runner() -> FlowPlanNodeRunner:
    attempts = 0

    async def run(
        node: FlowNodePlan,
        inputs: Mapping[str, object],
    ) -> object:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise FlowNodeExecutionError()
        return "retried"

    return run


async def _slow_runner(
    node: FlowNodePlan,
    inputs: Mapping[str, object],
) -> object:
    await sleep(0.05)
    return "late"


class _SubflowResolver:
    async def compile_subflow(
        self,
        ref: str,
        *,
        parent_definition: FlowDefinition,
        node: FlowNodeDefinition,
        registry: FlowNodeRegistry,
    ) -> Mapping[str, object]:
        return {
            "ref": ref,
            "parent": parent_definition.name,
            "node": node.name,
            "registry": registry.supports(node.type),
        }


def _container_event_names(
    events: list[CanonicalStreamItem],
) -> tuple[str, ...]:
    names: list[str] = []
    for event in events:
        if event.metadata.get("event_type") != EventType.FLOW_CONTAINER_EVENT:
            continue
        if not isinstance(event.data, Mapping):
            continue
        value = event.data.get("container_event")
        if isinstance(value, str):
            names.append(value)
    return tuple(names)


if __name__ == "__main__":
    main()
