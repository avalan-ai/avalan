from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest import IsolatedAsyncioTestCase, main

from avalan.entities import (
    ToolCall,
    ToolCallContext,
    ToolCallOutcome,
    ToolFilterResult,
    ToolFilterResultStatus,
    ToolManagerSettings,
)
from avalan.flow import (
    FlowDefinitionLoader,
    FlowExecutor,
    tool_flow_node_registry,
)
from avalan.skill import (
    SkillConfiguredSource,
    SkillIndexLimits,
    SkillMetadata,
    SkillReadLimits,
    SkillRegistry,
    SkillRegistrySkill,
    SkillRegistrySource,
    SkillRegistryVersion,
    SkillSourceAuthorityKind,
    SkillSourceConfig,
    SkillStatus,
    TrustedSkillSettings,
    UserLocalSkillSourceAuthority,
    WorkspaceSkillSourceAuthority,
    build_skill_registry,
    resolve_skill_sources,
)
from avalan.tool.manager import ToolManager
from avalan.tool.skills import SkillsToolSet


class _RecordingToolManager(ToolManager):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.calls: list[str] = []

    async def execute_call(
        self,
        call: ToolCall,
        context: ToolCallContext,
        *,
        confirm: (
            Callable[
                [ToolCall],
                Awaitable[str | bool | None] | str | bool | None,
            ]
            | None
        ) = None,
    ) -> ToolCallOutcome:
        self.calls.append(call.name)
        return await super().execute_call(call, context, confirm=confirm)


class FlowSkillsE2ETestCase(IsolatedAsyncioTestCase):
    async def test_tool_node_uses_match_then_read(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "pdf" / "SKILL.md",
                body="# PDF Body\nFOLLOW_THE_FLOW_STEPS\n",
            )
            settings = _trusted_settings(root)
            registry = await _registry(root, settings)
            manager = _RecordingToolManager.create_instance(
                available_toolsets=[SkillsToolSet(registry)],
                enable_tools=["skills.*"],
            )
            assert isinstance(manager, _RecordingToolManager)
            node_registry = tool_flow_node_registry(manager)
            loaded = await FlowDefinitionLoader(
                node_registry,
                skills_settings=settings,
            ).loads_validation_result(_flow_source())

            self.assertTrue(loaded.ok, loaded.public_diagnostics)
            assert loaded.definition is not None
            result = await FlowExecutor(registry=node_registry).run(
                loaded.definition,
                inputs={"payload": {"task": "render a pdf", "skill": "pdf"}},
            )

        self.assertTrue(result.ok, result.public_diagnostics)
        answer = result.outputs["answer"]
        self.assertIsInstance(answer, Mapping)
        content = answer["content"]
        self.assertIsInstance(content, Mapping)
        text = content["text"]
        self.assertIsInstance(text, str)
        self.assertIn("FOLLOW_THE_FLOW_STEPS", text)
        self.assertEqual(manager.calls, ["skills.match", "skills.read"])

    async def test_flow_level_narrowing_uses_wider_skills_registry(
        self,
    ) -> None:
        result = await _run_tool_flow(
            _flow_source(
                flow_skills="""
                    [skills]
                    skill_ids = ["pdf"]
                """,
            )
        )

        self.assertTrue(result.ok, result.public_diagnostics)
        assert result.plan is not None
        match = result.plan.node_map["match"]
        read = result.plan.node_map["read"]
        assert match.skills is not None
        assert read.skills is not None
        self.assertEqual(match.skills.allowed_skill_ids, ("pdf",))
        self.assertEqual(read.skills.allowed_skill_ids, ("pdf",))

    async def test_node_level_narrowing_uses_wider_skills_registry(
        self,
    ) -> None:
        result = await _run_tool_flow(
            _flow_source(
                read_node_skills="""
                    [nodes.read.skills]
                    skill_ids = ["pdf"]
                """,
            )
        )

        self.assertTrue(result.ok, result.public_diagnostics)
        assert result.plan is not None
        match = result.plan.node_map["match"]
        read = result.plan.node_map["read"]
        assert match.skills is not None
        assert read.skills is not None
        self.assertEqual(match.skills.allowed_skill_ids, ())
        self.assertEqual(read.skills.allowed_skill_ids, ("pdf",))

    async def test_filter_applies_to_narrowed_skill_tool_node(self) -> None:
        filtered_registries: list[object] = []

        def suppress_read(
            call: ToolCall,
            context: ToolCallContext,
        ) -> ToolFilterResult | None:
            if call.name != "skills.read":
                return None
            filtered_registries.append(context.skills_registry)
            return ToolFilterResult(
                status=ToolFilterResultStatus.SUPPRESS,
                message="blocked by test filter",
            )

        result = await _run_tool_flow(
            _flow_source(
                flow_skills="""
                    [skills]
                    skill_ids = ["pdf"]
                """,
            ),
            manager_settings=ToolManagerSettings(filters=[suppress_read]),
        )

        self.assertFalse(result.ok)
        self.assertEqual(len(filtered_registries), 1)
        self.assertIsInstance(filtered_registries[0], SkillRegistry)

    async def test_cursor_continues_across_narrowed_skill_nodes(self) -> None:
        result = await _run_tool_flow(
            _cursor_flow_source(),
            pdf_body="one\ntwo\nthree\nfour\n",
        )

        self.assertTrue(result.ok, result.public_diagnostics)
        answer = result.outputs["answer"]
        self.assertIsInstance(answer, Mapping)
        content = answer["content"]
        self.assertIsInstance(content, Mapping)
        text = content["text"]
        self.assertIsInstance(text, str)
        self.assertIn("three", text)

    async def test_disabled_flow_skills_block_all_skill_tools(self) -> None:
        cases = (
            ("skills.list", {"query": "pdf"}),
            ("skills.match", {"query": "render pdf"}),
            ("skills.read", {"skill": "pdf"}),
            ("skills.check", {"skill": "pdf"}),
        )
        for tool_name, arguments in cases:
            with self.subTest(tool=tool_name):
                result = await _run_tool_flow(
                    _single_skill_tool_source(
                        tool_name,
                        flow_skills="""
                            [skills]
                            enabled = false
                        """,
                    ),
                    payload=arguments,
                )

                self.assertTrue(result.ok, result.public_diagnostics)
                self.assertEqual(result.outputs["answer"], "disabled")

    async def test_disabled_node_blocks_cursor_continuation(self) -> None:
        result = await _run_tool_flow(
            _disabled_cursor_flow_source(),
            pdf_body="one\ntwo\nthree\nfour\n",
        )

        self.assertTrue(result.ok, result.public_diagnostics)
        self.assertEqual(result.outputs["answer"], "disabled")

    async def test_authority_narrowing_filters_registry_sources(self) -> None:
        registry = _mixed_authority_registry()
        manager = _RecordingToolManager.create_instance(
            available_toolsets=[SkillsToolSet(registry)],
            enable_tools=["skills.list"],
        )
        assert isinstance(manager, _RecordingToolManager)
        node_registry = tool_flow_node_registry(manager)
        loaded = await FlowDefinitionLoader(
            node_registry,
        ).loads_validation_result(
            _single_skill_tool_source(
                "skills.list",
                flow_skills="""
                    [skills]
                    authority_kinds = ["workspace"]
                """,
                output_selector="tool.result.items",
            )
        )

        self.assertTrue(loaded.ok, loaded.public_diagnostics)
        assert loaded.definition is not None
        result = await FlowExecutor(registry=node_registry).run(
            loaded.definition,
            inputs={"payload": {"query": ""}},
        )

        self.assertTrue(result.ok, result.public_diagnostics)
        items = result.outputs["answer"]
        self.assertIsInstance(items, tuple)
        self.assertEqual(len(items), 1)
        first = items[0]
        self.assertIsInstance(first, Mapping)
        self.assertEqual(first["skill_id"], "workspace-pdf")

    async def test_same_label_different_source_identity_fails_closed(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(root / "pdf" / "SKILL.md", body="# PDF\n")
            registry_settings = TrustedSkillSettings(
                sources=(
                    SkillSourceConfig(
                        label="workspace-main",
                        authority=WorkspaceSkillSourceAuthority(
                            workspace_id="workspace-a",
                        ),
                        root_path=root,
                    ),
                ),
            )
            flow_settings = TrustedSkillSettings(
                sources=(
                    SkillSourceConfig(
                        label="workspace-main",
                        authority=WorkspaceSkillSourceAuthority(
                            workspace_id="workspace-b",
                        ),
                        root_path=root,
                    ),
                ),
            )
            registry = await _registry(root, registry_settings)
            manager = _RecordingToolManager.create_instance(
                available_toolsets=[SkillsToolSet(registry)],
                enable_tools=["skills.read"],
            )
            assert isinstance(manager, _RecordingToolManager)
            node_registry = tool_flow_node_registry(manager)
            loaded = await FlowDefinitionLoader(
                node_registry,
                skills_settings=flow_settings,
            ).loads_validation_result(_single_skill_tool_source("skills.read"))

            self.assertTrue(loaded.ok, loaded.public_diagnostics)
            assert loaded.definition is not None
            result = await FlowExecutor(registry=node_registry).run(
                loaded.definition,
                inputs={"payload": {"skill": "pdf"}},
            )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.skills_policy_invalid",
        )

    async def test_same_label_same_authority_different_root_fails_closed(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory)
            root_a = base / "a"
            root_b = base / "b"
            _write_skill(root_a / "pdf" / "SKILL.md", body="# PDF\n")
            _write_skill(root_b / "pdf" / "SKILL.md", body="# PDF\n")
            registry_settings = TrustedSkillSettings(
                sources=(
                    SkillSourceConfig(
                        label="workspace-main",
                        authority=WorkspaceSkillSourceAuthority(
                            workspace_id="workspace",
                        ),
                        root_path=root_a,
                    ),
                ),
            )
            flow_settings = TrustedSkillSettings(
                sources=(
                    SkillSourceConfig(
                        label="workspace-main",
                        authority=WorkspaceSkillSourceAuthority(
                            workspace_id="workspace",
                        ),
                        root_path=root_b,
                    ),
                ),
            )
            registry = await _registry(root_a, registry_settings)
            manager = _RecordingToolManager.create_instance(
                available_toolsets=[SkillsToolSet(registry)],
                enable_tools=["skills.read"],
            )
            assert isinstance(manager, _RecordingToolManager)
            node_registry = tool_flow_node_registry(manager)
            loaded = await FlowDefinitionLoader(
                node_registry,
                skills_settings=flow_settings,
            ).loads_validation_result(_single_skill_tool_source("skills.read"))

            self.assertTrue(loaded.ok, loaded.public_diagnostics)
            assert loaded.definition is not None
            result = await FlowExecutor(registry=node_registry).run(
                loaded.definition,
                inputs={"payload": {"skill": "pdf"}},
            )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.skills_policy_invalid",
        )

    async def test_registry_actual_root_identity_cannot_be_spoofed(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory)
            trusted_root = base / "trusted"
            actual_root = base / "actual"
            trusted_root.mkdir()
            _write_skill(actual_root / "pdf" / "SKILL.md", body="# PDF\n")
            registry_settings = TrustedSkillSettings(
                sources=(
                    SkillSourceConfig(
                        label="workspace-main",
                        authority=WorkspaceSkillSourceAuthority(),
                        root_path=trusted_root,
                    ),
                ),
            )
            actual_sources = await resolve_skill_sources(
                (
                    SkillConfiguredSource(
                        label="workspace-main",
                        authority=WorkspaceSkillSourceAuthority(),
                        root_path=actual_root,
                    ),
                ),
            )
            registry = await build_skill_registry(
                actual_sources,
                settings=registry_settings,
            )
            manager = _RecordingToolManager.create_instance(
                available_toolsets=[SkillsToolSet(registry)],
                enable_tools=["skills.read"],
            )
            assert isinstance(manager, _RecordingToolManager)
            node_registry = tool_flow_node_registry(manager)
            loaded = await FlowDefinitionLoader(
                node_registry,
                skills_settings=registry_settings,
            ).loads_validation_result(_single_skill_tool_source("skills.read"))

            self.assertTrue(loaded.ok, loaded.public_diagnostics)
            assert loaded.definition is not None
            result = await FlowExecutor(registry=node_registry).run(
                loaded.definition,
                inputs={"payload": {"skill": "pdf"}},
            )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.skills_policy_invalid",
        )

    async def test_authority_filter_after_explicit_sources_exposes_no_skills(
        self,
    ) -> None:
        registry = _explicit_user_settings_mixed_registry()
        manager = _RecordingToolManager.create_instance(
            available_toolsets=[SkillsToolSet(registry)],
            enable_tools=["skills.list"],
        )
        assert isinstance(manager, _RecordingToolManager)
        node_registry = tool_flow_node_registry(manager)
        loaded = await FlowDefinitionLoader(
            node_registry,
        ).loads_validation_result(
            _single_skill_tool_source(
                "skills.list",
                flow_skills="""
                    [skills]
                    authority_kinds = ["workspace"]
                """,
            )
        )

        self.assertTrue(loaded.ok, loaded.public_diagnostics)
        assert loaded.definition is not None
        result = await FlowExecutor(registry=node_registry).run(
            loaded.definition,
            inputs={"payload": {"query": ""}},
        )

        self.assertTrue(result.ok, result.public_diagnostics)
        self.assertEqual(result.outputs["answer"], "empty")


async def _run_tool_flow(
    source: str,
    *,
    manager_settings: ToolManagerSettings | None = None,
    pdf_body: str = "# PDF Body\nFOLLOW_THE_FLOW_STEPS\n",
    payload: Mapping[str, object] | None = None,
) -> object:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        _write_skill(
            root / "pdf" / "SKILL.md",
            body=pdf_body,
        )
        _write_skill(
            root / "doc" / "SKILL.md",
            body="# Doc Body\nDO_NOT_USE_FOR_PDF\n",
        )
        settings = _trusted_settings(root)
        registry = await _registry(root, settings)
        manager = _RecordingToolManager.create_instance(
            available_toolsets=[SkillsToolSet(registry)],
            enable_tools=["skills.*"],
            settings=manager_settings,
        )
        assert isinstance(manager, _RecordingToolManager)
        node_registry = tool_flow_node_registry(manager)
        loaded = await FlowDefinitionLoader(
            node_registry,
            skills_settings=settings,
        ).loads_validation_result(source)

        assert loaded.ok, loaded.public_diagnostics
        assert loaded.definition is not None
        return await FlowExecutor(registry=node_registry).run(
            loaded.definition,
            inputs={
                "payload": payload or {"task": "render a pdf", "skill": "pdf"}
            },
        )


def _flow_source(
    *,
    flow_skills: str = "",
    read_node_skills: str = "",
) -> str:
    return f"""
        [flow]
        name = "skills_tool_flow"
        version = "1"

        [[inputs]]
        name = "payload"
        type = "object"

        [[outputs]]
        name = "answer"
        type = "object"

        {flow_skills}

        [entry]
        type = "node"
        node = "match"

        [output_behavior]
        type = "map"

        [output_behavior.outputs]
        answer = "read.result"

        [nodes.match]
        type = "tool"
        ref = "skills.match"

        [nodes.match.mapping.arguments]
        type = "object"

        [nodes.match.mapping.arguments.fields]
        query = "input.payload.task"

        [nodes.read]
        type = "tool"
        ref = "skills.read"

        {read_node_skills}

        [nodes.read.mapping.arguments]
        type = "object"

        [nodes.read.mapping.arguments.fields]
        skill = "input.payload.skill"

        [[edges]]
        source = "match"
        target = "read"
        kind = "success"
    """


def _single_skill_tool_source(
    tool_name: str,
    *,
    flow_skills: str = "",
    output_selector: str = "tool.result.status",
) -> str:
    if tool_name in ("skills.list", "skills.match"):
        argument_fields = 'query = "input.payload.query"'
        config_arguments = 'query = "query"'
    else:
        argument_fields = 'skill = "input.payload.skill"'
        config_arguments = 'skill = "skill"'
    return f"""
        [flow]
        name = "single_skill_tool"
        version = "1"

        [[inputs]]
        name = "payload"
        type = "object"

        [[outputs]]
        name = "answer"
        type = "object"

        {flow_skills}

        [entry]
        type = "node"
        node = "tool"

        [output_behavior]
        type = "map"

        [output_behavior.outputs]
        answer = "{output_selector}"

        [nodes.tool]
        type = "tool"
        ref = "{tool_name}"

        [nodes.tool.mapping.arguments]
        type = "object"

        [nodes.tool.mapping.arguments.fields]
        {argument_fields}

        [nodes.tool.config.arguments]
        {config_arguments}
    """


def _cursor_flow_source() -> str:
    return """
        [flow]
        name = "skills_cursor_flow"
        version = "1"

        [[inputs]]
        name = "payload"
        type = "object"

        [[outputs]]
        name = "answer"
        type = "object"

        [skills]
        skill_ids = ["pdf"]

        [skills.read_limits]
        max_lines_per_read = 6

        [entry]
        type = "node"
        node = "first"

        [output_behavior]
        type = "map"

        [output_behavior.outputs]
        answer = "second.result"

        [nodes.first]
        type = "tool"
        ref = "skills.read"

        [nodes.first.mapping.arguments]
        type = "object"

        [nodes.first.mapping.arguments.fields]
        skill = "input.payload.skill"

        [nodes.first.config.arguments]
        skill = "skill"

        [nodes.second]
        type = "tool"
        ref = "skills.read"

        [nodes.second.mapping.arguments]
        type = "object"

        [nodes.second.mapping.arguments.fields]
        cursor_id = "first.result.next_cursor"

        [[edges]]
        source = "first"
        target = "second"
        kind = "success"
    """


def _disabled_cursor_flow_source() -> str:
    return """
        [flow]
        name = "skills_disabled_cursor_flow"
        version = "1"

        [[inputs]]
        name = "payload"
        type = "object"

        [[outputs]]
        name = "answer"
        type = "object"

        [skills]
        skill_ids = ["pdf"]

        [skills.read_limits]
        max_lines_per_read = 6

        [entry]
        type = "node"
        node = "first"

        [output_behavior]
        type = "map"

        [output_behavior.outputs]
        answer = "second.result.status"

        [nodes.first]
        type = "tool"
        ref = "skills.read"

        [nodes.first.mapping.arguments]
        type = "object"

        [nodes.first.mapping.arguments.fields]
        skill = "input.payload.skill"

        [nodes.second]
        type = "tool"
        ref = "skills.read"

        [nodes.second.skills]
        enabled = false

        [nodes.second.mapping.arguments]
        type = "object"

        [nodes.second.mapping.arguments.fields]
        cursor_id = "first.result.next_cursor"

        [[edges]]
        source = "first"
        target = "second"
        kind = "success"
    """


async def _registry(
    root: Path,
    settings: TrustedSkillSettings,
) -> SkillRegistry:
    source_result = await resolve_skill_sources(
        (
            SkillConfiguredSource(
                label="workspace-main",
                authority=WorkspaceSkillSourceAuthority(),
                root_path=root,
            ),
        ),
        settings=settings,
    )
    return await build_skill_registry(source_result, settings=settings)


def _mixed_authority_registry() -> SkillRegistry:
    settings = TrustedSkillSettings(
        sources=(),
        authority_kinds=(
            SkillSourceAuthorityKind.WORKSPACE,
            SkillSourceAuthorityKind.USER_LOCAL,
        ),
    )
    return SkillRegistry(
        registry_version=SkillRegistryVersion(
            value="skills-registry:11111111111111111111111111111111"
        ),
        read_limits=SkillReadLimits(),
        index_limits=SkillIndexLimits(),
        settings=settings,
        sources=(
            SkillRegistrySource(
                label="workspace-main",
                authority=WorkspaceSkillSourceAuthority(),
            ),
            SkillRegistrySource(
                label="user-main",
                authority=UserLocalSkillSourceAuthority(),
            ),
        ),
        skills=(
            _registry_skill("workspace-pdf", "workspace-main"),
            _registry_skill("user-pdf", "user-main"),
        ),
    )


def _explicit_user_settings_mixed_registry() -> SkillRegistry:
    settings = TrustedSkillSettings(
        sources=(
            SkillSourceConfig(
                label="user-main",
                authority=UserLocalSkillSourceAuthority(),
            ),
        ),
    )
    return SkillRegistry(
        registry_version=SkillRegistryVersion(
            value="skills-registry:22222222222222222222222222222222"
        ),
        read_limits=SkillReadLimits(),
        index_limits=SkillIndexLimits(),
        settings=settings,
        sources=(
            SkillRegistrySource(
                label="workspace-main",
                authority=WorkspaceSkillSourceAuthority(),
            ),
            SkillRegistrySource(
                label="user-main",
                authority=UserLocalSkillSourceAuthority(),
            ),
        ),
        skills=(
            _registry_skill("workspace-pdf", "workspace-main"),
            _registry_skill("user-pdf", "user-main"),
        ),
    )


def _registry_skill(
    skill_id: str,
    source_label: str,
) -> SkillRegistrySkill:
    metadata = SkillMetadata(
        skill_id=skill_id,
        name=skill_id,
        description=f"{skill_id} guidance.",
        source_label=source_label,
    )
    return SkillRegistrySkill(
        source_label=source_label,
        manifest_resource_id=f"{skill_id}/SKILL.md",
        package_resource_id=skill_id,
        status=SkillStatus.OK,
        readable=True,
        usable=True,
        skill_id=skill_id,
        metadata=metadata,
    )


def _trusted_settings(
    root: Path,
    *,
    read_limits: SkillReadLimits | None = None,
) -> TrustedSkillSettings:
    return TrustedSkillSettings(
        sources=(
            SkillSourceConfig(
                label="workspace-main",
                authority=WorkspaceSkillSourceAuthority(),
                root_path=root,
            ),
        ),
        read_limits=read_limits or SkillReadLimits(),
    )


def _write_skill(path: Path, *, body: str) -> None:
    skill_id = path.parent.name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        f"name: {skill_id}\n"
        f"description: {skill_id} guidance.\n"
        f'tags: ["{skill_id}"]\n'
        "resources: []\n"
        "---\n"
        f"{body}",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
