from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, main

from avalan.flow import (
    FlowDefinitionLoader,
    FlowNodeKind,
    FlowNodeMetadata,
    FlowNodeRegistry,
    serialize_flow_definition,
    tool_flow_node_registry,
)
from avalan.flow.node import Node
from avalan.skill import (
    SkillConfiguredSource,
    SkillReadLimits,
    SkillRegistry,
    SkillSourceConfig,
    TrustedSkillSettings,
    WorkspaceSkillSourceAuthority,
    build_skill_registry,
    resolve_skill_sources,
)
from avalan.tool import ToolSet
from avalan.tool.manager import ToolManager
from avalan.tool.skills import SkillsToolSet


class FlowSkillsLoaderTestCase(IsolatedAsyncioTestCase):
    async def test_flow_skills_toml_narrows_trusted_settings(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            settings = _trusted_settings(root)
            source = _flow_source("""
                [skills]
                source_labels = ["workspace-main"]
                skill_ids = ["pdf"]
                bootstrap = "off"

                [skills.read_limits]
                max_bytes_per_read = 1024
                """)

            result = await FlowDefinitionLoader(
                skills_settings=settings,
            ).loads_validation_result(source)

            self.assertTrue(result.ok, result.public_diagnostics)
            assert result.definition is not None
            assert result.definition.skills is not None
            self.assertEqual(
                result.definition.skills.allowed_skill_ids,
                ("pdf",),
            )
            self.assertFalse(result.definition.skills.bootstrap_enabled)
            self.assertEqual(
                result.definition.skills.read_limits.max_bytes_per_read,
                1024,
            )
            self.assertIsNotNone(result.definition.skills_config)

            serialized = serialize_flow_definition(result.definition)

        self.assertIn("[skills]", serialized)
        self.assertIn('skill_ids = ["pdf"]', serialized)
        self.assertIn("max_bytes_per_read = 1024", serialized)
        self.assertNotIn("max_lines_per_read", serialized)
        self.assertNotIn(str(root), serialized)
        self.assertNotIn("root_path", serialized)

        round_trip = await FlowDefinitionLoader(
            skills_settings=settings,
        ).loads_validation_result(serialized)
        self.assertTrue(round_trip.ok, round_trip.public_diagnostics)

    async def test_explicit_scalar_skills_fields_round_trip(self) -> None:
        with TemporaryDirectory() as directory:
            settings = _trusted_settings(Path(directory))
            source = _flow_source("""
                [skills]
                enabled = true
                bootstrap = "auto"
                skill_ids = []
                source_labels = []
                """)

            result = await FlowDefinitionLoader(
                skills_settings=settings,
            ).loads_validation_result(source)

            self.assertTrue(result.ok, result.public_diagnostics)
            assert result.definition is not None
            serialized = serialize_flow_definition(result.definition)
            round_trip = await FlowDefinitionLoader(
                skills_settings=settings,
            ).loads_validation_result(serialized)

        self.assertIn("enabled = true", serialized)
        self.assertIn('bootstrap = "auto"', serialized)
        self.assertIn("skill_ids = []", serialized)
        self.assertIn("source_labels = []", serialized)
        self.assertTrue(round_trip.ok, round_trip.public_diagnostics)

    async def test_flow_skills_without_trusted_base_fails_closed(self) -> None:
        result = await FlowDefinitionLoader().loads_validation_result(
            _flow_source("""
                [skills]
                skill_ids = ["pdf"]
                """)
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.issues[0].code,
            "flow.skills_trusted_settings_required",
        )

    async def test_trusted_defaults_do_not_make_legacy_flow_strict(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            result = await FlowDefinitionLoader(
                skills_settings=_trusted_settings(Path(directory)),
            ).loads_validation_result(_legacy_flow_source())

        self.assertTrue(result.ok, result.public_diagnostics)
        assert result.definition is not None
        self.assertIsNotNone(result.definition.skills)
        self.assertIsNone(result.definition.skills_config)
        self.assertFalse(result.definition.is_strict)

    async def test_flow_skills_reject_source_definitions(self) -> None:
        with TemporaryDirectory() as directory:
            result = await FlowDefinitionLoader(
                skills_settings=_trusted_settings(Path(directory)),
            ).loads_validation_result(
                _flow_source("""
                    [skills]
                    sources = [{label = "private"}]
                    """)
            )

        self.assertFalse(result.ok)
        self.assertEqual(result.issues[0].code, "flow.invalid_skills_settings")
        self.assertNotIn(str(directory), str(result.public_diagnostics))

    async def test_flow_skills_reject_limit_widening(self) -> None:
        with TemporaryDirectory() as directory:
            result = await FlowDefinitionLoader(
                skills_settings=_trusted_settings(
                    Path(directory),
                    read_limits=SkillReadLimits(max_bytes_per_read=2048),
                ),
            ).loads_validation_result(
                _flow_source("""
                    [skills.read_limits]
                    max_bytes_per_read = 4096
                    """)
            )

        self.assertFalse(result.ok)
        self.assertEqual(result.issues[0].code, "skills.policy_denied")
        self.assertEqual(result.issues[0].path, "skills.read_limits")

    async def test_flow_skills_reject_invalid_authority_kind(self) -> None:
        with TemporaryDirectory() as directory:
            result = await FlowDefinitionLoader(
                skills_settings=_trusted_settings(Path(directory)),
            ).loads_validation_result(
                _flow_source("""
                    [skills]
                    authority_kinds = ["host-filesystem"]
                    """)
            )

        self.assertFalse(result.ok)
        self.assertEqual(result.issues[0].code, "flow.invalid_skills_settings")

    async def test_flow_skills_use_registry_settings_as_trusted_base(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            settings = _trusted_settings(root)
            registry = await _skills_flow_node_registry(root, settings)
            result = await FlowDefinitionLoader(
                registry,
            ).loads_validation_result(
                _flow_source("""
                    [skills]
                    skill_ids = ["pdf"]
                    """)
            )

        self.assertTrue(result.ok, result.public_diagnostics)
        assert result.definition is not None
        assert result.definition.skills is not None
        self.assertEqual(result.definition.skills.allowed_skill_ids, ("pdf",))

    async def test_node_skills_use_registry_settings_as_trusted_base(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            settings = _trusted_settings(root)
            registry = await _skills_flow_node_registry(root, settings)
            result = await FlowDefinitionLoader(
                registry,
            ).loads_validation_result(_skills_read_tool_source())

        self.assertTrue(result.ok, result.public_diagnostics)
        assert result.definition is not None
        node = result.definition.node_map["start"]
        self.assertIsNotNone(node.skills)

    async def test_node_skills_without_trusted_base_fails_closed(self) -> None:
        result = await FlowDefinitionLoader().loads_validation_result(
            _skills_read_tool_source()
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.issues[0].code,
            "flow.skills_trusted_settings_required",
        )
        self.assertEqual(result.issues[0].path, "nodes.start.skills")

    async def test_node_skills_reject_non_mapping_values(self) -> None:
        with TemporaryDirectory() as directory:
            result = await FlowDefinitionLoader(
                skills_settings=_trusted_settings(Path(directory)),
            ).loads_validation_result(
                _flow_source("""
                    skills = "bad"
                    """)
            )

        self.assertFalse(result.ok)
        self.assertEqual(result.issues[0].code, "flow.invalid_section")
        self.assertEqual(result.issues[0].path, "nodes.start.skills")

    async def test_agent_node_skills_are_supported(self) -> None:
        with TemporaryDirectory() as directory:
            result = await FlowDefinitionLoader(
                _agent_node_registry(),
                skills_settings=_trusted_settings(Path(directory)),
            ).loads_validation_result(
                _agent_node_source("""
                    [nodes.start.skills]
                    skill_ids = ["pdf"]
                    """)
            )

        self.assertTrue(result.ok, result.public_diagnostics)
        assert result.definition is not None
        node = result.definition.node_map["start"]
        self.assertIsNotNone(node.skills)

    async def test_tool_kind_without_resolution_rejects_node_skills(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            result = await FlowDefinitionLoader(
                _unresolved_tool_kind_registry(),
                skills_settings=_trusted_settings(Path(directory)),
            ).loads_validation_result(
                _custom_tool_node_source("""
                    [nodes.start.skills]
                    skill_ids = ["pdf"]
                    """)
            )

        self.assertFalse(result.ok)
        self.assertEqual(result.issues[0].code, "flow.skills_unsupported_node")

    async def test_malformed_tool_node_rejects_node_skills(self) -> None:
        with TemporaryDirectory() as directory:
            manager = ToolManager.create_instance()
            result = await FlowDefinitionLoader(
                tool_flow_node_registry(manager),
                skills_settings=_trusted_settings(Path(directory)),
            ).loads_validation_result(
                _malformed_tool_node_source("""
                    [nodes.start.skills]
                    skill_ids = ["pdf"]
                    """)
            )

        self.assertFalse(result.ok)
        self.assertEqual(result.issues[0].code, "flow.skills_unsupported_node")

    async def test_node_skills_round_trip_only_when_fields_are_serialized(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            settings = _trusted_settings(Path(directory))
            result = await FlowDefinitionLoader(
                await _skills_flow_node_registry(Path(directory), settings),
                skills_settings=settings,
            ).loads_validation_result(_skills_read_tool_source())

            self.assertTrue(result.ok, result.public_diagnostics)
            assert result.definition is not None
            serialized = serialize_flow_definition(result.definition)
            round_trip = await FlowDefinitionLoader(
                await _skills_flow_node_registry(Path(directory), settings),
                skills_settings=settings,
            ).loads_validation_result(serialized)

        self.assertIn("[nodes.start.skills]", serialized)
        self.assertIn('skill_ids = ["pdf"]', serialized)
        self.assertTrue(round_trip.ok, round_trip.public_diagnostics)

    async def test_empty_node_skills_config_is_not_serialized(self) -> None:
        with TemporaryDirectory() as directory:
            settings = _trusted_settings(Path(directory))
            result = await FlowDefinitionLoader(
                _agent_node_registry(),
                skills_settings=settings,
            ).loads_validation_result(
                _agent_node_source("""
                    [nodes.start.skills]
                    """)
            )

            self.assertTrue(result.ok, result.public_diagnostics)
            assert result.definition is not None
            serialized = serialize_flow_definition(result.definition)

        self.assertNotIn("[nodes.start.skills]", serialized)

    async def test_node_skills_reject_non_skills_tool_node(self) -> None:
        with TemporaryDirectory() as directory:
            settings = _trusted_settings(Path(directory))
            manager = ToolManager.create_instance(
                available_toolsets=[
                    ToolSet(namespace="misc", tools=[misc_echo])
                ],
                enable_tools=["misc.misc_echo"],
            )
            result = await FlowDefinitionLoader(
                tool_flow_node_registry(manager),
                skills_settings=settings,
            ).loads_validation_result(_non_skills_tool_source())

        self.assertFalse(result.ok)
        self.assertEqual(result.issues[0].code, "flow.skills_unsupported_node")

    async def test_node_skills_reject_unsupported_node(self) -> None:
        with TemporaryDirectory() as directory:
            result = await FlowDefinitionLoader(
                skills_settings=_trusted_settings(Path(directory)),
            ).loads_validation_result(
                _flow_source("""
                    [nodes.start.skills]
                    skill_ids = ["pdf"]
                    """)
            )

        self.assertFalse(result.ok)
        self.assertEqual(result.issues[0].code, "flow.skills_unsupported_node")


def _flow_source(extra: str) -> str:
    return f"""
        [flow]
        name = "skills_loader"
        version = "1"

        [[inputs]]
        name = "payload"
        type = "object"

        [[outputs]]
        name = "answer"
        type = "object"

        [entry]
        type = "node"
        node = "start"

        [output_behavior]
        type = "map"

        [output_behavior.outputs]
        answer = "start.value"

        [nodes.start]
        type = "input"

        {extra}
    """


def _legacy_flow_source() -> str:
    return """
        [flow]
        name = "legacy_skills_defaults"
        version = "1"
        entrypoint = "start"
        output_node = "finish"

        [nodes.start]
        type = "input"
        input = "payload"

        [nodes.finish]
        type = "echo"
        input = "start"

        [[edges]]
        source = "start"
        target = "finish"
    """


def _non_skills_tool_source() -> str:
    return """
        [flow]
        name = "non_skills_tool"
        version = "1"

        [[inputs]]
        name = "payload"
        type = "object"

        [[outputs]]
        name = "answer"
        type = "object"

        [entry]
        type = "node"
        node = "start"

        [output_behavior]
        type = "map"

        [output_behavior.outputs]
        answer = "start.result"

        [nodes.start]
        type = "tool"
        ref = "misc.misc_echo"

        [nodes.start.skills]
        skill_ids = ["pdf"]

        [nodes.start.mapping.arguments]
        type = "object"

        [nodes.start.mapping.arguments.fields]
        value = "input.payload"
    """


def _agent_node_source(extra: str) -> str:
    return f"""
        [flow]
        name = "agent_node_skills"
        version = "1"

        [[inputs]]
        name = "payload"
        type = "object"

        [[outputs]]
        name = "answer"
        type = "object"

        [entry]
        type = "node"
        node = "start"

        [output_behavior]
        type = "map"

        [output_behavior.outputs]
        answer = "start.value"

        [nodes.start]
        type = "agent"

        {extra}
    """


def _custom_tool_node_source(extra: str) -> str:
    return f"""
        [flow]
        name = "custom_tool_node_skills"
        version = "1"

        [[inputs]]
        name = "payload"
        type = "object"

        [[outputs]]
        name = "answer"
        type = "object"

        [entry]
        type = "node"
        node = "start"

        [output_behavior]
        type = "map"

        [output_behavior.outputs]
        answer = "start.result"

        [nodes.start]
        type = "custom-tool"

        {extra}
    """


def _malformed_tool_node_source(extra: str) -> str:
    return f"""
        [flow]
        name = "malformed_tool_node_skills"
        version = "1"

        [[inputs]]
        name = "payload"
        type = "object"

        [[outputs]]
        name = "answer"
        type = "object"

        [entry]
        type = "node"
        node = "start"

        [output_behavior]
        type = "map"

        [output_behavior.outputs]
        answer = "start.result"

        [nodes.start]
        type = "tool"

        {extra}
    """


def _skills_read_tool_source() -> str:
    return """
        [flow]
        name = "skills_read_tool"
        version = "1"

        [[inputs]]
        name = "payload"
        type = "object"

        [[outputs]]
        name = "answer"
        type = "object"

        [entry]
        type = "node"
        node = "start"

        [output_behavior]
        type = "map"

        [output_behavior.outputs]
        answer = "start.result"

        [nodes.start]
        type = "tool"
        ref = "skills.read"

        [nodes.start.skills]
        skill_ids = ["pdf"]

        [nodes.start.mapping.arguments]
        type = "object"

        [nodes.start.mapping.arguments.fields]
        skill = "input.payload.skill"
    """


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


def _agent_node_registry() -> FlowNodeRegistry:
    return FlowNodeRegistry(
        {"agent": lambda definition: Node(definition.name)},
        {"agent": FlowNodeMetadata(kind=FlowNodeKind.AGENT)},
    )


def _unresolved_tool_kind_registry() -> FlowNodeRegistry:
    return FlowNodeRegistry(
        {"custom-tool": lambda definition: Node(definition.name)},
        {"custom-tool": FlowNodeMetadata(kind=FlowNodeKind.TOOL)},
    )


async def _skills_flow_node_registry(
    root: Path,
    settings: TrustedSkillSettings,
) -> object:
    _write_skill(root / "pdf" / "SKILL.md")
    registry = await _skill_registry(root, settings)
    manager = ToolManager.create_instance(
        available_toolsets=[SkillsToolSet(registry)],
        enable_tools=["skills.read"],
    )
    return tool_flow_node_registry(manager)


async def _skill_registry(
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


def _write_skill(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        "name: pdf\n"
        "description: PDF rendering guidance.\n"
        'tags: ["pdf"]\n'
        "resources: []\n"
        "---\n"
        "# PDF Body\n",
        encoding="utf-8",
    )


async def misc_echo(value: object) -> object:
    return value


if __name__ == "__main__":
    main()
