from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest import IsolatedAsyncioTestCase, main

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
)
from avalan.flow.node import Node
from avalan.skill import (
    SkillPrivacySettings,
    SkillSettingsSurface,
    SkillSourceConfig,
    TrustedSkillSettings,
    UntrustedSkillSettings,
    WorkspaceSkillSourceAuthority,
)


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


def _agent_definition(
    skills: TrustedSkillSettings,
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
