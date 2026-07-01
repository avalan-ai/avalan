from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import patch

from avalan.flow import (
    FlowDefinition,
    FlowEntryBehavior,
    FlowExecutionPlan,
    FlowExecutionTrace,
    FlowInputDefinition,
    FlowInputType,
    FlowNodeDefinition,
    FlowNodeKind,
    FlowNodeMetadata,
    FlowNodePlan,
    FlowNodeRegistry,
    FlowOutputBehavior,
    FlowOutputDefinition,
    FlowOutputType,
    compile_flow_definition,
    execute_flow_plan,
    flow_resume_skills_metadata,
    tool_flow_node_registry,
)
from avalan.flow import (
    loader as loader_module,
)
from avalan.flow import (
    plan as plan_module,
)
from avalan.flow import (
    registry as registry_module,
)
from avalan.flow import (
    runtime as runtime_module,
)
from avalan.flow.node import Node
from avalan.skill import (
    SkillDiagnosticCode,
    SkillDiagnosticInfo,
    SkillIndexLimits,
    SkillPrivacySettings,
    SkillReadLimits,
    SkillRegistry,
    SkillRegistrySource,
    SkillRegistryVersion,
    SkillSettingsSurface,
    SkillSourceAuthorityKind,
    SkillSourceConfig,
    SkillStatus,
    TrustedSkillSettings,
    UntrustedSkillSettings,
    WorkspaceSkillSourceAuthority,
)
from avalan.tool.manager import ToolManager
from avalan.tool.skills import SkillsToolSet


class FlowSkillsRuntimeTestCase(IsolatedAsyncioTestCase):
    async def test_node_narrowing_participates_in_plan_identity(self) -> None:
        with TemporaryDirectory() as directory:
            trusted = _trusted_settings(Path(directory))
            pdf = await compile_flow_definition(
                _agent_definition(
                    trusted,
                    UntrustedSkillSettings(
                        surface=SkillSettingsSurface.FLOW,
                        skill_ids=("pdf",),
                    ),
                ),
                _agent_registry(),
            )
            doc = await compile_flow_definition(
                _agent_definition(
                    trusted,
                    UntrustedSkillSettings(
                        surface=SkillSettingsSurface.FLOW,
                        skill_ids=("doc",),
                    ),
                ),
                _agent_registry(),
            )

        self.assertTrue(pdf.ok, pdf.public_diagnostics)
        self.assertTrue(doc.ok, doc.public_diagnostics)
        assert pdf.plan is not None
        assert doc.plan is not None
        pdf_node = pdf.plan.node_map["agent"]
        doc_node = doc.plan.node_map["agent"]
        assert pdf_node.skills is not None
        assert doc_node.skills is not None
        self.assertEqual(pdf_node.skills.allowed_skill_ids, ("pdf",))
        self.assertEqual(doc_node.skills.allowed_skill_ids, ("doc",))
        assert pdf_node.skills_identity is not None
        assert doc_node.skills_identity is not None
        self.assertNotEqual(
            pdf_node.skills_identity.settings_fingerprint,
            doc_node.skills_identity.settings_fingerprint,
        )

    async def test_resume_fails_when_skills_metadata_is_missing(self) -> None:
        plan = await _compiled_agent_plan()
        trace = replace(FlowExecutionTrace.from_plan(plan), metadata={})

        result = await execute_flow_plan(
            plan,
            _runner,
            resume_trace=trace,
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.execution.skills_resume_metadata_widened",
        )

    async def test_resume_fails_when_unskilled_plan_has_skills_metadata(
        self,
    ) -> None:
        plan = await _compiled_plain_agent_plan()
        metadata = _mutable_metadata(FlowExecutionTrace.from_plan(plan))
        metadata["skills"] = {"status": "ok", "nodes": {}}
        trace = replace(FlowExecutionTrace.from_plan(plan), metadata=metadata)

        result = await execute_flow_plan(
            plan,
            _runner,
            resume_trace=trace,
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.execution.skills_resume_metadata_stale",
        )

    async def test_resume_fails_when_skills_metadata_is_stale(self) -> None:
        plan = await _compiled_agent_plan()
        metadata = _mutable_metadata(FlowExecutionTrace.from_plan(plan))
        skills = cast(dict[str, object], metadata["skills"])
        skills["version"] = "old"
        trace = replace(FlowExecutionTrace.from_plan(plan), metadata=metadata)

        result = await execute_flow_plan(
            plan,
            _runner,
            resume_trace=trace,
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.execution.skills_resume_metadata_stale",
        )

    async def test_resume_fails_on_malformed_or_narrowed_node_metadata(
        self,
    ) -> None:
        cases: tuple[tuple[object, str], ...] = (
            (
                {},
                "flow.execution.skills_resume_metadata_widened",
            ),
            (
                "invalid",
                "flow.execution.skills_resume_metadata_stale",
            ),
        )
        for nodes, expected_code in cases:
            with self.subTest(nodes=nodes):
                plan = await _compiled_agent_plan()
                metadata = _mutable_metadata(
                    FlowExecutionTrace.from_plan(plan)
                )
                skills = cast(dict[str, object], metadata["skills"])
                skills["nodes"] = nodes
                trace = replace(
                    FlowExecutionTrace.from_plan(plan),
                    metadata=metadata,
                )

                result = await execute_flow_plan(
                    plan,
                    _runner,
                    resume_trace=trace,
                )

                self.assertFalse(result.ok)
                self.assertEqual(result.diagnostics[0].code, expected_code)

    async def test_resume_fails_on_policy_denied_skills_metadata(self) -> None:
        plan = await _compiled_agent_plan()
        metadata = _mutable_metadata(FlowExecutionTrace.from_plan(plan))
        skills = cast(dict[str, object], metadata["skills"])
        skills["status"] = "policy_denied"
        trace = replace(FlowExecutionTrace.from_plan(plan), metadata=metadata)

        result = await execute_flow_plan(
            plan,
            _runner,
            resume_trace=trace,
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.execution.skills_resume_metadata_policy_denied",
        )

    async def test_resume_accepts_matching_skills_metadata(self) -> None:
        plan = await _compiled_agent_plan()
        trace = FlowExecutionTrace.from_plan(plan)

        result = await execute_flow_plan(
            plan,
            _runner,
            resume_trace=trace,
        )

        self.assertTrue(result.ok, result.public_diagnostics)

    async def test_resume_metadata_does_not_include_host_paths(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            result = await compile_flow_definition(
                _agent_definition(_trusted_settings(root), None),
                _agent_registry(),
            )

            self.assertTrue(result.ok, result.public_diagnostics)
            assert result.plan is not None
            metadata = flow_resume_skills_metadata(result.plan)

        self.assertNotIn(str(root), str(metadata))

    async def test_public_trace_redacts_skills_source_policy_fields(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            result = await compile_flow_definition(
                _agent_definition(
                    TrustedSkillSettings(
                        sources=(
                            SkillSourceConfig(
                                label="workspace-main",
                                authority=WorkspaceSkillSourceAuthority(),
                                root_path=root,
                            ),
                        ),
                        privacy=SkillPrivacySettings(
                            include_source_labels=False,
                            include_authority=False,
                        ),
                    ),
                    None,
                ),
                _agent_registry(),
            )

            self.assertTrue(result.ok, result.public_diagnostics)
            assert result.plan is not None
            trace = FlowExecutionTrace.from_plan(result.plan)
            private = str(trace.metadata)
            public = trace.as_public_dict()
            public_metadata = cast(Mapping[str, object], public["metadata"])
            skills = cast(Mapping[str, object], public_metadata["skills"])
            flow = cast(Mapping[str, object], skills["flow"])
            nodes = cast(Mapping[str, object], skills["nodes"])
            agent = cast(Mapping[str, object], nodes["agent"])

        self.assertIn("source_labels", private)
        self.assertIn("authority_kinds", private)
        self.assertNotIn("source_labels", flow)
        self.assertNotIn("authority_kinds", flow)
        self.assertNotIn("source_labels", agent)
        self.assertNotIn("authority_kinds", agent)
        self.assertNotIn("workspace-main", str(public))

    async def test_compile_accepts_explicit_skills_registry_metadata(
        self,
    ) -> None:
        settings = TrustedSkillSettings()
        registry = _empty_skill_registry(settings)

        result = await compile_flow_definition(
            _agent_definition(settings, None),
            _agent_registry(),
            skills_settings=settings,
            skills_registry=registry,
        )

        self.assertTrue(result.ok, result.public_diagnostics)
        assert result.plan is not None
        self.assertIs(result.plan.skills_registry, registry)
        assert result.plan.skills_identity is not None
        self.assertEqual(
            result.plan.skills_identity.registry_version,
            registry.registry_version.value,
        )

    async def test_flow_plan_rejects_stale_skills_metadata(self) -> None:
        plan = await _compiled_agent_plan()
        assert plan.skills_identity is not None
        node = plan.node_map["agent"]
        assert node.skills_identity is not None

        same_node = replace(node, skills_identity=node.skills_identity)
        same_plan = replace(
            plan,
            skills_identity=plan.skills_identity,
            nodes=(same_node,),
        )

        self.assertEqual(
            same_plan.skills_identity.to_dict(),
            plan.skills_identity.to_dict(),
        )
        with self.assertRaisesRegex(
            AssertionError,
            "flow plan cannot carry stale skills metadata",
        ):
            replace(plan, skills=None, skills_identity=plan.skills_identity)
        with self.assertRaisesRegex(
            AssertionError,
            "flow node cannot carry stale skills metadata",
        ):
            replace(
                plan,
                nodes=(
                    replace(
                        node,
                        skills=None,
                        skills_identity=node.skills_identity,
                    ),
                ),
            )
        with self.assertRaisesRegex(
            AssertionError,
            "flow node skills metadata changed",
        ):
            replace(
                plan,
                nodes=(
                    replace(
                        node,
                        skills_identity=replace(
                            node.skills_identity,
                            settings_fingerprint="changed",
                        ),
                    ),
                ),
            )

    async def test_compile_rejects_invalid_node_skill_policies(self) -> None:
        cases = (
            (
                FlowNodeDefinition(
                    name="input",
                    type="input",
                    skills=UntrustedSkillSettings(
                        surface=SkillSettingsSurface.FLOW,
                    ),
                ),
                FlowNodeKind.INPUT,
                TrustedSkillSettings(),
                None,
                "flow.skills_unsupported_node",
            ),
            (
                FlowNodeDefinition(
                    name="agent",
                    type="agent",
                    skills=UntrustedSkillSettings(
                        surface=SkillSettingsSurface.FLOW,
                        skill_ids=("pdf",),
                    ),
                ),
                FlowNodeKind.AGENT,
                None,
                None,
                "flow.skills_trusted_settings_required",
            ),
            (
                FlowNodeDefinition(
                    name="agent",
                    type="agent",
                    skills=UntrustedSkillSettings(
                        surface=SkillSettingsSurface.FLOW,
                        skill_ids=("doc",),
                    ),
                ),
                FlowNodeKind.AGENT,
                TrustedSkillSettings(allowed_skill_ids=("pdf",)),
                None,
                "skills.policy_denied",
            ),
        )

        for node, kind, flow_skills, skills_registry, expected_code in cases:
            with self.subTest(expected_code=expected_code):
                with self.assertRaises(ValueError) as raised:
                    plan_module._compile_node_skills(
                        node,
                        kind,
                        _agent_registry(),
                        {},
                        flow_skills=flow_skills,
                        skills_registry=skills_registry,
                    )

                self.assertEqual(raised.exception.args[0], expected_code)

    async def test_tool_registry_defaults_can_supply_node_skills(self) -> None:
        registry = _empty_skill_registry(TrustedSkillSettings())
        manager = ToolManager.create_instance(
            available_toolsets=[SkillsToolSet(registry)],
            enable_tools=["skills.read"],
        )
        node_registry = tool_flow_node_registry(manager)
        node = FlowNodeDefinition(name="read", type="tool", ref="skills.read")

        inherited = plan_module._registry_node_skills(node_registry, node)

        self.assertIs(inherited, registry.settings)
        plan_module._assert_node_registry_settings_allow(
            TrustedSkillSettings(),
            node_registry,
            node,
            path="nodes.read.skills",
        )

    def test_skills_policy_helpers_cover_widening_rejections(self) -> None:
        workspace_source = SkillSourceConfig(
            label="workspace-main",
            authority=WorkspaceSkillSourceAuthority(),
        )
        other_workspace_source = SkillSourceConfig(
            label="workspace-main",
            authority=WorkspaceSkillSourceAuthority(workspace_id="other"),
        )

        self.assertFalse(
            plan_module._trusted_settings_allow(
                TrustedSkillSettings(enabled=False),
                TrustedSkillSettings(enabled=True),
            )
        )
        self.assertFalse(
            plan_module._trusted_settings_allow(
                TrustedSkillSettings(bootstrap_enabled=False),
                TrustedSkillSettings(bootstrap_enabled=True),
            )
        )
        self.assertFalse(
            plan_module._trusted_settings_allow(
                TrustedSkillSettings(
                    authority_kinds=(SkillSourceAuthorityKind.WORKSPACE,),
                ),
                TrustedSkillSettings(
                    authority_kinds=(SkillSourceAuthorityKind.USER_LOCAL,),
                ),
            )
        )
        self.assertFalse(
            plan_module._trusted_settings_allow(
                TrustedSkillSettings(sources=(workspace_source,)),
                TrustedSkillSettings(),
            )
        )
        self.assertFalse(
            plan_module._trusted_sources_allow(
                TrustedSkillSettings(sources=(workspace_source,)),
                TrustedSkillSettings(sources=(other_workspace_source,)),
            )
        )
        self.assertFalse(
            plan_module._trusted_settings_allow(
                TrustedSkillSettings(allowed_skill_ids=("pdf",)),
                TrustedSkillSettings(allowed_skill_ids=()),
            )
        )
        self.assertTrue(
            plan_module._trusted_skill_ids_allow(("pdf",), ("pdf",))
        )

    def test_registry_source_identity_falls_back_to_public_fields(
        self,
    ) -> None:
        source = SkillRegistrySource(
            label="workspace-main",
            authority=WorkspaceSkillSourceAuthority(),
        )

        identity = plan_module._registry_source_identity(source)

        self.assertEqual(identity["label"], "workspace-main")
        self.assertEqual(
            identity["authority"],
            {
                "kind": "workspace",
                "workspace_id": "workspace",
            },
        )
        self.assertEqual(identity["status"], "ok")
        self.assertEqual(plan_module._registry_source_identity(object()), {})

    def test_registry_policy_checks_reject_wider_settings(self) -> None:
        settings = TrustedSkillSettings(allowed_skill_ids=("pdf",))
        registry = _empty_skill_registry(settings)

        with self.assertRaises(ValueError) as raised:
            plan_module._assert_registry_settings_allow(
                TrustedSkillSettings(allowed_skill_ids=("doc",)),
                registry,
                path="skills",
            )

        self.assertEqual(
            raised.exception.args[0],
            "flow.skills_policy_invalid",
        )
        plan_module._assert_registry_settings_allow(
            TrustedSkillSettings(),
            _empty_skill_registry(None),
            path="skills",
        )

    def test_node_metadata_helpers_find_skills_tools(self) -> None:
        node = FlowNodePlan(
            name="read",
            type="tool",
            kind=FlowNodeKind.TOOL,
            skills=TrustedSkillSettings(),
            metadata={"tool": {"canonical_name": "skills.read"}},
        )

        self.assertEqual(plan_module._node_tool_name(node), "skills.read")
        self.assertFalse(plan_module._node_uses_skills(FlowNodeKind.TOOL, {}))
        missing_name = FlowNodePlan(
            name="read",
            type="tool",
            kind=FlowNodeKind.TOOL,
            metadata={"tool": {"canonical_name": ""}},
        )
        self.assertIsNone(plan_module._node_tool_name(missing_name))

    def test_skills_configuration_error_maps_paths(self) -> None:
        diagnostic = SkillDiagnosticInfo(
            code=SkillDiagnosticCode.POLICY_DENIED,
            status=SkillStatus.POLICY_DENIED,
            message="Denied.",
            path="settings.skill_ids",
            hint="Use allowed skills.",
        )

        error = plan_module._skills_configuration_error(
            diagnostic,
            path="nodes.agent.skills",
        )

        self.assertEqual(error.args[0], "skills.policy_denied")
        self.assertEqual(error.path, "nodes.agent.skills.skill_ids")

    def test_loader_support_rejects_missing_metadata(self) -> None:
        self.assertFalse(
            loader_module._node_supports_skills(
                FlowNodeDefinition(name="agent", type="agent"),
                object(),
                _agent_registry(),
            )
        )

    def test_tool_registry_without_skills_registry_returns_none(self) -> None:
        manager = ToolManager.create_instance()
        node_registry = tool_flow_node_registry(manager)
        node = FlowNodeDefinition(name="read", type="tool", ref="skills.read")

        self.assertEqual(node_registry.tool_descriptors("tool"), {})
        self.assertIsNone(
            plan_module._registry_node_skills(node_registry, node)
        )
        plan_module._assert_node_registry_settings_allow(
            TrustedSkillSettings(),
            tool_flow_node_registry(
                ToolManager.create_instance(
                    available_toolsets=[
                        SkillsToolSet(_empty_skill_registry(None))
                    ],
                    enable_tools=["skills.read"],
                )
            ),
            node,
            path="nodes.read.skills",
        )

    def test_flow_registry_narrows_only_when_tool_settings_require_it(
        self,
    ) -> None:
        self.assertIsNone(
            registry_module._flow_skill_registry(
                FlowNodeDefinition(name="read", type="tool"),
                "skills.read",
            )
        )

        settings = TrustedSkillSettings()
        registry = _empty_skill_registry(settings)
        definition = FlowNodeDefinition(
            name="read",
            type="tool",
            config={
                "__flow_skills_settings": settings,
                "__flow_skills_registry": registry,
            },
        )
        self.assertIsNone(
            registry_module._flow_skill_registry(definition, "skills.read")
        )

        requested = TrustedSkillSettings(
            sources=(
                SkillSourceConfig(
                    label="workspace-main",
                    authority=WorkspaceSkillSourceAuthority(
                        workspace_id="trusted",
                    ),
                ),
            )
        )
        mismatched_registry = SkillRegistry(
            registry_version=SkillRegistryVersion(
                value="skills-registry:44444444444444444444444444444444"
            ),
            read_limits=SkillReadLimits(),
            index_limits=SkillIndexLimits(),
            settings=requested,
            sources=(
                SkillRegistrySource(
                    label="workspace-main",
                    authority=WorkspaceSkillSourceAuthority(
                        workspace_id="actual",
                    ),
                ),
            ),
            skills=(),
        )
        narrowed = registry_module._flow_skill_registry(
            FlowNodeDefinition(
                name="read",
                type="tool",
                config={
                    "__flow_skills_settings": requested,
                    "__flow_skills_registry": mismatched_registry,
                },
            ),
            "skills.read",
        )

        self.assertIsNotNone(narrowed)
        assert narrowed is not None
        self.assertEqual(narrowed.sources, ())

    def test_flow_registry_skill_source_policy_handles_source_sets(
        self,
    ) -> None:
        self.assertTrue(
            registry_module._skill_source_allowed(
                "workspace-main",
                registry_has_sources=False,
                allowed_source_labels=set(),
                configured_source_labels={"workspace-main"},
                sources_explicit=True,
            )
        )
        self.assertFalse(
            registry_module._skill_source_allowed(
                "other",
                registry_has_sources=False,
                allowed_source_labels=set(),
                configured_source_labels={"workspace-main"},
                sources_explicit=True,
            )
        )
        self.assertTrue(
            registry_module._skill_source_allowed(
                "other",
                registry_has_sources=False,
                allowed_source_labels=set(),
                configured_source_labels=set(),
                sources_explicit=False,
            )
        )
        self.assertEqual(
            registry_module._registry_source_identity(object()),
            {},
        )

    def test_resume_skill_helpers_handle_internal_malformed_metadata(
        self,
    ) -> None:
        plan = FlowExecutionPlan(
            name="plain",
            version="1",
            revision=None,
            inputs=(),
            outputs=(),
            entry_node="start",
            output_selectors={},
            nodes=(),
        )
        trace = FlowExecutionTrace.from_plan(plan)

        with patch.object(
            runtime_module,
            "flow_resume_skills_metadata",
            return_value={"skills": "invalid"},
        ):
            diagnostics = runtime_module._resume_skills_metadata_diagnostics(
                plan,
                trace,
            )

        self.assertEqual(diagnostics, ())
        self.assertIsNone(
            runtime_module._resume_skills_node_names({"nodes": {"": {}}})
        )


async def _runner(
    node: FlowNodePlan,
    inputs: Mapping[str, object],
) -> dict[str, object]:
    del node, inputs
    return {"value": {"ok": True}}


async def _compiled_agent_plan() -> FlowExecutionPlan:
    with TemporaryDirectory() as directory:
        result = await compile_flow_definition(
            _agent_definition(_trusted_settings(Path(directory)), None),
            _agent_registry(),
        )
    assert result.plan is not None
    return result.plan


async def _compiled_plain_agent_plan() -> FlowExecutionPlan:
    result = await compile_flow_definition(
        _agent_definition(None, None),
        _agent_registry(),
    )
    assert result.plan is not None
    return result.plan


def _agent_definition(
    skills: TrustedSkillSettings | None,
    node_skills: UntrustedSkillSettings | None,
) -> FlowDefinition:
    return FlowDefinition(
        name="skills_runtime",
        version="1",
        inputs=(
            FlowInputDefinition(name="payload", type=FlowInputType.OBJECT),
        ),
        outputs=(
            FlowOutputDefinition(name="answer", type=FlowOutputType.OBJECT),
        ),
        entry_behavior=FlowEntryBehavior(node="agent"),
        output_behavior=FlowOutputBehavior(outputs={"answer": "agent.value"}),
        skills=skills,
        nodes=(
            FlowNodeDefinition(
                name="agent",
                type="agent",
                skills=node_skills,
            ),
        ),
    )


def _agent_registry() -> FlowNodeRegistry:
    return FlowNodeRegistry(
        {"agent": lambda definition: Node(definition.name)},
        {"agent": FlowNodeMetadata(kind=FlowNodeKind.AGENT)},
    )


def _trusted_settings(root: Path) -> TrustedSkillSettings:
    return TrustedSkillSettings(
        sources=(
            SkillSourceConfig(
                label="workspace-main",
                authority=WorkspaceSkillSourceAuthority(),
                root_path=root,
            ),
        )
    )


def _empty_skill_registry(
    settings: TrustedSkillSettings | None,
) -> SkillRegistry:
    return SkillRegistry(
        registry_version=SkillRegistryVersion(
            value="skills-registry:33333333333333333333333333333333"
        ),
        read_limits=SkillReadLimits(),
        index_limits=SkillIndexLimits(),
        settings=settings,
        sources=(),
        skills=(),
    )


def _mutable_metadata(trace: FlowExecutionTrace) -> dict[str, object]:
    return {
        key: _mutable_value(value) for key, value in trace.metadata.items()
    }


def _mutable_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _mutable_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_mutable_value(item) for item in value)
    return value


if __name__ == "__main__":
    main()
