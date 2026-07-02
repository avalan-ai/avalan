from json import dumps
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase, TestCase, main

from avalan.entities import (
    ToolCall,
    ToolCallContext,
    ToolCallDiagnosticCode,
    ToolCallError,
    ToolCallResult,
    ToolCapabilities,
    ToolManagerSettings,
    ToolNameResolutionStatus,
)
from avalan.skill import (
    SkillConfiguredSource,
    SkillIndexLimits,
    SkillObservabilitySettings,
    SkillPrivacySettings,
    SkillReadLimits,
    SkillRegistry,
    SkillRegistryVersion,
    SkillResourceReader,
    SkillSourceConfig,
    SkillStatus,
    TrustedSkillSettings,
    WorkspaceSkillSourceAuthority,
    build_skill_registry,
    resolve_skill_sources,
)
from avalan.tool.manager import ToolManager
from avalan.tool.skills import (
    ListSkillsTool,
    MatchSkillsTool,
    ReadSkillTool,
    SkillsToolSet,
)


class SkillsToolSetSchemaTestCase(IsolatedAsyncioTestCase):
    async def test_schemas_expose_only_canonical_skills_tools(self) -> None:
        with TemporaryDirectory() as directory:
            registry = await _registry(Path(directory))
            toolset = SkillsToolSet(registry)

            schemas = toolset.json_schemas()

        assert schemas is not None
        self.assertEqual(
            [schema["function"]["name"] for schema in schemas],
            [
                "skills.list",
                "skills.match",
                "skills.read",
                "skills.check",
            ],
        )
        self.assertNotIn("skills.load", dumps(schemas))

    async def test_tool_manager_enablement_filters_skills_namespace(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            registry = await _registry(Path(directory))
            cases: tuple[tuple[list[str], list[str]], ...] = (
                (
                    ["skills"],
                    [
                        "skills.list",
                        "skills.match",
                        "skills.read",
                        "skills.check",
                    ],
                ),
                (
                    ["skills.*"],
                    [
                        "skills.list",
                        "skills.match",
                        "skills.read",
                        "skills.check",
                    ],
                ),
                (["skills.read"], ["skills.read"]),
                (["skillsx"], []),
            )

            for enabled, expected in cases:
                with self.subTest(enabled=enabled):
                    manager = ToolManager.create_instance(
                        available_toolsets=[SkillsToolSet(registry)],
                        enable_tools=list(enabled),
                        settings=ToolManagerSettings(),
                    )

                    self.assertEqual(
                        [
                            descriptor.name
                            for descriptor in manager.list_tools()
                        ],
                        expected,
                    )

    async def test_capabilities_mark_read_as_cursoring_not_parallel_safe(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            registry = await _registry(Path(directory))
            manager = ToolManager.create_instance(
                available_toolsets=[SkillsToolSet(registry)],
                enable_tools=["skills"],
                settings=ToolManagerSettings(),
            )

        capabilities = {
            descriptor.name: descriptor.capabilities
            for descriptor in manager.list_tools()
        }
        self.assertEqual(
            capabilities["skills.list"],
            ToolCapabilities(side_effecting=False, parallel_safe=True),
        )
        self.assertEqual(
            capabilities["skills.match"],
            ToolCapabilities(side_effecting=False, parallel_safe=True),
        )
        self.assertEqual(
            capabilities["skills.check"],
            ToolCapabilities(side_effecting=False, parallel_safe=True),
        )
        self.assertEqual(
            capabilities["skills.read"],
            ToolCapabilities(side_effecting=False, parallel_safe=False),
        )
        self.assertTrue(
            manager.is_tool_call_parallel_safe(
                ToolCall(id="call-1", name="skills.list", arguments={})
            )
        )
        self.assertFalse(
            manager.is_tool_call_parallel_safe(
                ToolCall(
                    id="call-2",
                    name="skills.read",
                    arguments={"skill": "pdf"},
                )
            )
        )


class SkillsToolSetCallTestCase(IsolatedAsyncioTestCase):
    async def test_sdk_privacy_observability_filter_model_responses(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "pdf" / "SKILL.md",
                name="pdf",
                description="PDF rendering guidance.",
                tags=("pdf",),
                body="# PDF Body\nUNIQUE_BODY_TOKEN\n",
            )
            registry = await _registry(
                root,
                settings=TrustedSkillSettings(
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
                        include_diagnostic_paths=False,
                    ),
                    observability=SkillObservabilitySettings(
                        include_diagnostics=False,
                        include_byte_counts=False,
                    ),
                ),
            )
            manager = ToolManager.create_instance(
                available_toolsets=[SkillsToolSet(registry)],
                enable_tools=["skills"],
                settings=ToolManagerSettings(),
            )

            listed = await manager(
                ToolCall(id="call-1", name="skills.list", arguments={}),
                ToolCallContext(),
            )
            matched = await manager(
                ToolCall(
                    id="call-2",
                    name="skills.match",
                    arguments={"query": "render pdf"},
                ),
                ToolCallContext(),
            )
            checked = await manager(
                ToolCall(
                    id="call-3",
                    name="skills.check",
                    arguments={"skill": "pdf"},
                ),
                ToolCallContext(),
            )
            read = await manager(
                ToolCall(
                    id="call-4",
                    name="skills.read",
                    arguments={"skill": "pdf"},
                ),
                ToolCallContext(),
            )

        for outcome in (listed, matched, checked, read):
            assert isinstance(outcome, ToolCallResult)
            encoded = dumps(_result_dict(outcome), sort_keys=True)
            self.assertNotIn("source_label", encoded)
            self.assertNotIn("authority", encoded)
            self.assertNotIn("diagnostics", encoded)
            self.assertNotIn("path", encoded)
            self.assertNotIn("size_bytes", encoded)
            self.assertNotIn("start_byte", encoded)
            self.assertNotIn("end_byte", encoded)
            self.assertNotIn(str(root), encoded)

    async def test_sdk_diagnostic_paths_and_byte_counts_filter_when_kept(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "pdf" / "SKILL.md",
                name="pdf",
                description="PDF rendering guidance.",
                tags=("pdf",),
                body="# PDF Body\n" + ("READ_ME\n" * 64),
            )
            registry = await _registry(
                root,
                settings=TrustedSkillSettings(
                    sources=(
                        SkillSourceConfig(
                            label="workspace-main",
                            authority=WorkspaceSkillSourceAuthority(),
                            root_path=root,
                        ),
                    ),
                    privacy=SkillPrivacySettings(
                        include_diagnostic_paths=False,
                    ),
                    observability=SkillObservabilitySettings(
                        include_diagnostics=True,
                        include_byte_counts=False,
                    ),
                ),
            )
            tool = ReadSkillTool(
                registry,
                SkillResourceReader(
                    read_limits=SkillReadLimits(
                        max_bytes_per_read=64,
                        max_lines_per_read=16,
                    )
                ),
            )

            result = await tool(
                context=ToolCallContext(),
                skill="pdf",
            )

        self.assertEqual(result["status"], SkillStatus.TRUNCATED.value)
        self.assertIn("diagnostics", result)
        encoded = dumps(result, sort_keys=True)
        self.assertNotIn('"path":', encoded)
        self.assertNotIn('"max_bytes_per_read":', encoded)
        self.assertNotIn('"max_lines_per_read":', encoded)
        self.assertNotIn('"size_bytes":', encoded)
        self.assertNotIn('"start_byte":', encoded)
        self.assertNotIn('"end_byte":', encoded)
        self.assertNotIn(str(root), encoded)

    async def test_hidden_source_labels_do_not_affect_list_match_behavior(
        self,
    ) -> None:
        source_label = "workspace-main"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "pdf" / "SKILL.md",
                name="pdf",
                description="PDF rendering guidance.",
                tags=("pdf",),
            )
            registry = await _registry(
                root,
                settings=TrustedSkillSettings(
                    sources=(
                        SkillSourceConfig(
                            label=source_label,
                            authority=WorkspaceSkillSourceAuthority(),
                            root_path=root,
                        ),
                    ),
                    privacy=SkillPrivacySettings(
                        include_source_labels=False,
                    ),
                ),
            )
            toolset = SkillsToolSet(registry)
            list_tool = cast(ListSkillsTool, toolset.tools[0])
            match_tool = cast(MatchSkillsTool, toolset.tools[1])

            filtered = await list_tool(
                context=ToolCallContext(),
                source_label=source_label,
            )
            match_filtered = await match_tool(
                context=ToolCallContext(),
                source_label=source_label,
            )
            listed = await list_tool(
                context=ToolCallContext(),
                query=source_label,
            )
            matched = await match_tool(
                context=ToolCallContext(),
                query=source_label,
            )

        self.assertEqual(filtered["status"], SkillStatus.POLICY_DENIED.value)
        self.assertNotIn("items", filtered)
        self.assertNotIn(source_label, dumps(filtered, sort_keys=True))

        self.assertEqual(
            match_filtered["status"],
            SkillStatus.POLICY_DENIED.value,
        )
        self.assertNotIn("items", match_filtered)
        self.assertNotIn(source_label, dumps(match_filtered, sort_keys=True))

        self.assertEqual(listed["status"], SkillStatus.EMPTY.value)
        self.assertNotIn("items", listed)
        self.assertNotIn(source_label, dumps(listed, sort_keys=True))

        self.assertEqual(matched["status"], SkillStatus.EMPTY.value)
        self.assertNotIn("items", matched)
        matched_encoded = dumps(matched, sort_keys=True)
        self.assertNotIn(source_label, matched_encoded)
        self.assertNotIn("source label matched query", matched_encoded)
        self.assertNotIn("source filter matched", matched_encoded)

    async def test_hidden_source_labels_deny_read_check_filters(
        self,
    ) -> None:
        source_label = "workspace-main"
        wrong_label = "other-source"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "pdf" / "SKILL.md",
                name="pdf",
                description="PDF rendering guidance.",
                tags=("pdf",),
                body="# PDF Body\nUNIQUE_BODY_TOKEN\n",
            )
            registry = await _registry(
                root,
                settings=TrustedSkillSettings(
                    sources=(
                        SkillSourceConfig(
                            label=source_label,
                            authority=WorkspaceSkillSourceAuthority(),
                            root_path=root,
                        ),
                    ),
                    privacy=SkillPrivacySettings(
                        include_source_labels=False,
                    ),
                ),
            )
            manager = ToolManager.create_instance(
                available_toolsets=[SkillsToolSet(registry)],
                enable_tools=["skills"],
                settings=ToolManagerSettings(),
            )

            read_real = await manager(
                ToolCall(
                    id="read-real",
                    name="skills.read",
                    arguments={
                        "skill": "pdf",
                        "source_label": source_label,
                    },
                ),
                ToolCallContext(),
            )
            read_wrong = await manager(
                ToolCall(
                    id="read-wrong",
                    name="skills.read",
                    arguments={
                        "skill": "pdf",
                        "source_label": wrong_label,
                    },
                ),
                ToolCallContext(),
            )
            check_real = await manager(
                ToolCall(
                    id="check-real",
                    name="skills.check",
                    arguments={
                        "skill": "pdf",
                        "source_label": source_label,
                    },
                ),
                ToolCallContext(),
            )
            check_wrong = await manager(
                ToolCall(
                    id="check-wrong",
                    name="skills.check",
                    arguments={
                        "skill": "pdf",
                        "source_label": wrong_label,
                    },
                ),
                ToolCallContext(),
            )
            read = await manager(
                ToolCall(
                    id="read",
                    name="skills.read",
                    arguments={"skill": "pdf"},
                ),
                ToolCallContext(),
            )
            checked = await manager(
                ToolCall(
                    id="check",
                    name="skills.check",
                    arguments={"skill": "pdf"},
                ),
                ToolCallContext(),
            )

        for outcome in (read_real, read_wrong, check_real, check_wrong):
            assert isinstance(outcome, ToolCallResult)
            result = _result_dict(outcome)
            self.assertEqual(
                result["status"],
                SkillStatus.POLICY_DENIED.value,
            )
            encoded = dumps(result, sort_keys=True)
            self.assertNotIn("content", result)
            self.assertNotIn(source_label, encoded)
            self.assertNotIn(wrong_label, encoded)

        assert isinstance(read, ToolCallResult)
        read_result = _result_dict(read)
        self.assertEqual(read_result["status"], SkillStatus.OK.value)
        read_content = cast(dict[str, Any], read_result["content"])
        self.assertIn("UNIQUE_BODY_TOKEN", read_content["text"])
        self.assertNotIn("source_label", dumps(read_result, sort_keys=True))

        assert isinstance(checked, ToolCallResult)
        check_result = _result_dict(checked)
        self.assertEqual(check_result["status"], SkillStatus.OK.value)
        self.assertNotIn("source_label", dumps(check_result, sort_keys=True))

    async def test_visible_source_labels_keep_default_list_match_behavior(
        self,
    ) -> None:
        source_label = "workspace-main"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "pdf" / "SKILL.md",
                name="pdf",
                description="PDF rendering guidance.",
                tags=("pdf",),
            )
            registry = await _registry(root)
            toolset = SkillsToolSet(registry)
            list_tool = cast(ListSkillsTool, toolset.tools[0])
            match_tool = cast(MatchSkillsTool, toolset.tools[1])

            listed = await list_tool(
                context=ToolCallContext(),
                query=source_label,
            )
            matched = await match_tool(
                context=ToolCallContext(),
                query=source_label,
            )

        self.assertEqual(listed["status"], SkillStatus.OK.value)
        listed_items = cast(tuple[dict[str, Any], ...], listed["items"])
        self.assertEqual(listed_items[0]["skill_id"], "pdf")

        self.assertEqual(matched["status"], SkillStatus.OK.value)
        matched_items = cast(tuple[dict[str, Any], ...], matched["items"])
        reasons = cast(tuple[str, ...], matched_items[0]["reasons"])
        self.assertIn("source label matched query", reasons)

    async def test_list_match_check_do_not_return_skill_bodies(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "pdf" / "SKILL.md",
                name="pdf",
                description="PDF rendering guidance.",
                tags=("pdf",),
                body="# PDF Body\nUNIQUE_BODY_TOKEN\n",
            )
            _write_skill(
                root / "docx" / "SKILL.md",
                name="docx",
                description="DOCX formatting guidance.",
                tags=("documents",),
                body="# DOCX Body\nDOCX_BODY_TOKEN\n",
            )
            registry = await _registry(root)
            manager = ToolManager.create_instance(
                available_toolsets=[SkillsToolSet(registry)],
                enable_tools=["skills"],
                settings=ToolManagerSettings(),
            )

            listed = await manager(
                ToolCall(id="call-1", name="skills.list", arguments={}),
                ToolCallContext(),
            )
            matched = await manager(
                ToolCall(
                    id="call-2",
                    name="skills.match",
                    arguments={"query": "render pdf"},
                ),
                ToolCallContext(),
            )
            checked = await manager(
                ToolCall(
                    id="call-3",
                    name="skills.check",
                    arguments={"skill": "pdf"},
                ),
                ToolCallContext(),
            )
            read = await manager(
                ToolCall(
                    id="call-4",
                    name="skills.read",
                    arguments={"skill": "pdf"},
                ),
                ToolCallContext(),
            )

        for outcome in (listed, matched, checked):
            assert outcome is not None
            assert isinstance(outcome, ToolCallResult)
            result = _result_dict(outcome)
            encoded = dumps(result, sort_keys=True)
            self.assertNotIn("UNIQUE_BODY_TOKEN", encoded)
            self.assertNotIn("DOCX_BODY_TOKEN", encoded)
            self.assertNotIn("content", result)

        assert listed is not None
        assert isinstance(listed, ToolCallResult)
        listed_result = _result_dict(listed)
        self.assertEqual(listed_result["status"], SkillStatus.OK.value)
        listed_items = cast(tuple[dict[str, Any], ...], listed_result["items"])
        self.assertEqual(
            sorted(item["skill_id"] for item in listed_items),
            ["docx", "pdf"],
        )
        assert matched is not None
        assert isinstance(matched, ToolCallResult)
        matched_result = _result_dict(matched)
        matched_items = cast(
            tuple[dict[str, Any], ...],
            matched_result["items"],
        )
        self.assertEqual(
            matched_items[0]["metadata"]["skill_id"],
            "pdf",
        )
        assert checked is not None
        assert isinstance(checked, ToolCallResult)
        checked_result = _result_dict(checked)
        self.assertEqual(checked_result["status"], SkillStatus.OK.value)
        assert read is not None
        assert isinstance(read, ToolCallResult)
        read_result = _result_dict(read)
        read_content = cast(dict[str, Any], read_result["content"])
        self.assertIn("UNIQUE_BODY_TOKEN", read_content["text"])

    async def test_list_filters_and_invalid_arguments_return_envelopes(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "pdf" / "SKILL.md",
                name="pdf",
                description="PDF rendering guidance.",
                tags=("pdf", "rendering"),
            )
            _write_skill(
                root / "docx" / "SKILL.md",
                name="docx",
                description="DOCX formatting guidance.",
                tags=("documents",),
            )
            registry = await _registry(root)
            toolset = SkillsToolSet(registry)
            list_tool = cast(ListSkillsTool, toolset.tools[0])
            match_tool = cast(MatchSkillsTool, toolset.tools[1])

            result = await list_tool(
                query="render",
                tags=["pdf"],
                context=ToolCallContext(),
            )
            no_source_match = await list_tool(
                source_label="other-source",
                context=ToolCallContext(),
            )
            no_status_match = await list_tool(
                status=SkillStatus.DISABLED.value,
                usable_only=False,
                context=ToolCallContext(),
            )
            invalid_status = await list_tool(
                status="unknown",
                context=ToolCallContext(),
            )
            invalid_tags = await list_tool(
                tags=[""],
                context=ToolCallContext(),
            )
            invalid_source = await list_tool(
                source_label="Workspace Main",
                context=ToolCallContext(),
            )
            invalid_match_status = await match_tool(
                status="unknown",
                context=ToolCallContext(),
            )
            invalid_match_tags = await match_tool(
                tags=[""],
                context=ToolCallContext(),
            )
            bad_tag = await match_tool(
                tags=["bad tag"],
                context=ToolCallContext(),
            )
            bad_source = await match_tool(
                source_label="Workspace Main",
                context=ToolCallContext(),
            )
            bad_max_results = await match_tool(
                max_results=0,
                context=ToolCallContext(),
            )
            unsafe_list_status = await list_tool(
                status="/tmp/x",
                context=ToolCallContext(),
            )
            unsafe_match_status = await match_tool(
                status="/tmp/x",
                context=ToolCallContext(),
            )

        self.assertEqual(result["status"], SkillStatus.OK.value)
        result_items = cast(tuple[dict[str, Any], ...], result["items"])
        self.assertEqual(result_items[0]["skill_id"], "pdf")
        self.assertEqual(no_source_match["status"], SkillStatus.EMPTY.value)
        self.assertEqual(no_status_match["status"], SkillStatus.EMPTY.value)
        self.assertEqual(
            invalid_status["status"],
            SkillStatus.POLICY_DENIED.value,
        )
        self.assertEqual(
            invalid_tags["status"],
            SkillStatus.POLICY_DENIED.value,
        )
        self.assertEqual(
            invalid_match_status["status"],
            SkillStatus.POLICY_DENIED.value,
        )
        self.assertEqual(
            invalid_match_tags["status"],
            SkillStatus.POLICY_DENIED.value,
        )
        for invalid in (
            invalid_source,
            bad_tag,
            bad_source,
            bad_max_results,
            unsafe_list_status,
            unsafe_match_status,
        ):
            with self.subTest(invalid=invalid):
                self.assertEqual(
                    invalid["status"],
                    SkillStatus.POLICY_DENIED.value,
                )
                self.assertNotIn("/tmp/x", dumps(invalid, sort_keys=True))

    async def test_match_invalid_model_args_return_structured_envelopes(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            registry = await _registry(Path(directory))
            manager = ToolManager.create_instance(
                available_toolsets=[SkillsToolSet(registry)],
                enable_tools=["skills.match"],
                settings=ToolManagerSettings(),
            )
            calls = (
                {"max_results": 0},
                {"source_label": "Workspace Main"},
                {"tags": ["bad tag"]},
                {"status": "/tmp/x"},
            )

            for index, arguments in enumerate(calls, start=1):
                with self.subTest(arguments=arguments):
                    outcome = await manager(
                        ToolCall(
                            id=f"call-{index}",
                            name="skills.match",
                            arguments=cast(dict[str, Any], arguments),
                        ),
                        ToolCallContext(),
                    )

                    self.assertNotIsInstance(outcome, ToolCallError)
                    assert isinstance(outcome, ToolCallResult)
                    result = _result_dict(outcome)
                    self.assertEqual(
                        result["status"],
                        SkillStatus.POLICY_DENIED.value,
                    )
                    self.assertNotIn("/tmp/x", dumps(result, sort_keys=True))

    async def test_unknown_disabled_and_ambiguous_skill_calls_are_structured(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "disabled" / "SKILL.md",
                name="disabled",
                description="Disabled guidance.",
                enabled=False,
            )
            _write_skill(
                root / "dupe-one" / "SKILL.md",
                name="duplicate",
                description="Duplicate guidance.",
            )
            _write_skill(
                root / "dupe-two" / "SKILL.md",
                name="duplicate",
                description="Other duplicate guidance.",
            )
            registry = await _registry(root)
            manager = ToolManager.create_instance(
                available_toolsets=[SkillsToolSet(registry)],
                enable_tools=["skills"],
                settings=ToolManagerSettings(),
            )

            unknown = await manager(
                ToolCall(
                    id="call-1",
                    name="skills.read",
                    arguments={"skill": "missing"},
                ),
                ToolCallContext(),
            )
            disabled = await manager(
                ToolCall(
                    id="call-2",
                    name="skills.read",
                    arguments={"skill": "disabled"},
                ),
                ToolCallContext(),
            )
            ambiguous = await manager(
                ToolCall(
                    id="call-3",
                    name="skills.read",
                    arguments={"skill": "duplicate"},
                ),
                ToolCallContext(),
            )

        assert unknown is not None
        assert disabled is not None
        assert ambiguous is not None
        assert isinstance(unknown, ToolCallResult)
        assert isinstance(disabled, ToolCallResult)
        assert isinstance(ambiguous, ToolCallResult)
        unknown_result = _result_dict(unknown)
        disabled_result = _result_dict(disabled)
        ambiguous_result = _result_dict(ambiguous)
        self.assertEqual(
            unknown_result["status"],
            SkillStatus.NOT_FOUND.value,
        )
        self.assertEqual(
            disabled_result["status"],
            SkillStatus.DISABLED.value,
        )
        self.assertEqual(
            ambiguous_result["status"],
            SkillStatus.AMBIGUOUS.value,
        )

    async def test_tool_manager_resolves_disabled_unknown_and_functions_prefix(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            registry = await _registry(Path(directory))
            manager = ToolManager.create_instance(
                available_toolsets=[SkillsToolSet(registry)],
                enable_tools=["skills.list"],
                settings=ToolManagerSettings(),
            )

        exact = manager.resolve_tool_name("functions.skills.list")
        disabled = manager.resolve_tool_name("skills.read")
        unknown = manager.resolve_tool_name("skills.load")

        self.assertIs(exact.status, ToolNameResolutionStatus.EXACT)
        self.assertEqual(exact.canonical_name, "skills.list")
        self.assertIs(disabled.status, ToolNameResolutionStatus.DISABLED)
        self.assertIs(
            disabled.diagnostic_code,
            ToolCallDiagnosticCode.DISABLED_TOOL,
        )
        self.assertIs(unknown.status, ToolNameResolutionStatus.UNKNOWN)
        self.assertIs(
            unknown.diagnostic_code,
            ToolCallDiagnosticCode.UNKNOWN_TOOL,
        )


class SkillsBootstrapPromptTestCase(TestCase):
    def test_bootstrap_prompt_only_when_skills_tools_enabled(self) -> None:
        empty = ToolManager.create_instance(
            enable_tools=[],
            settings=ToolManagerSettings(),
        )
        self.assertIsNone(empty.bootstrap_prompt())

    def test_bootstrap_prompt_reflects_concrete_enabled_subset(self) -> None:
        registry = _minimal_registry()
        manager = ToolManager.create_instance(
            available_toolsets=[SkillsToolSet(registry)],
            enable_tools=["skills.read"],
            settings=ToolManagerSettings(),
        )

        prompt = manager.bootstrap_prompt()

        assert prompt is not None
        self.assertIn("skills.read", prompt)
        self.assertNotIn("skills.list", prompt)
        self.assertNotIn("skills.match", prompt)
        self.assertNotIn("skills.check", prompt)

    def test_bootstrap_prompt_uses_all_tool_wording_for_full_set(self) -> None:
        registry = _minimal_registry()
        manager = ToolManager.create_instance(
            available_toolsets=[SkillsToolSet(registry)],
            enable_tools=["skills"],
            settings=ToolManagerSettings(),
        )

        prompt = manager.bootstrap_prompt()

        assert prompt is not None
        self.assertIn("skills.list", prompt)
        self.assertIn("skills.match", prompt)
        self.assertIn("skills.read", prompt)
        self.assertIn("skills.check", prompt)
        self.assertIn("Use skills.match or skills.list", prompt)


async def _registry(
    root: Path,
    *,
    settings: TrustedSkillSettings | None = None,
) -> SkillRegistry:
    if not any(root.iterdir()):
        _write_skill(
            root / "pdf" / "SKILL.md",
            name="pdf",
            description="PDF rendering guidance.",
            tags=("pdf",),
        )
    source_result = await resolve_skill_sources(
        (_config(root),), settings=settings
    )
    return await build_skill_registry(source_result, settings=settings)


def _result_dict(outcome: ToolCallResult) -> dict[str, Any]:
    assert isinstance(outcome.result, dict)
    return cast(dict[str, Any], outcome.result)


def _minimal_registry() -> SkillRegistry:
    return SkillRegistry(
        registry_version=SkillRegistryVersion(
            value="skills-registry:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        ),
        read_limits=SkillReadLimits(),
        index_limits=SkillIndexLimits(),
    )


def _config(root: Path) -> SkillConfiguredSource:
    return SkillConfiguredSource(
        label="Workspace Main",
        authority=WorkspaceSkillSourceAuthority(),
        root_path=root,
    )


def _write_skill(
    path: Path,
    *,
    name: str,
    description: str,
    tags: tuple[str, ...] = (),
    enabled: bool = True,
    body: str = "# Body\n",
) -> None:
    tag_line = ""
    if tags:
        tag_line = "tags: [" + ", ".join(f'"{tag}"' for tag in tags) + "]\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        f"enabled: {'true' if enabled else 'false'}\n"
        f"{tag_line}"
        "resources: []\n"
        "---\n"
        f"{body}",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
