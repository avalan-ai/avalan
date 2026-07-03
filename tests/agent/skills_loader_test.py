from argparse import Namespace
from contextlib import AsyncExitStack
from json import dumps
from logging import Logger
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from avalan.agent import loader as loader_module
from avalan.agent.loader import OrchestratorLoader
from avalan.cli.commands import agent as agent_cmds
from avalan.entities import (
    EngineUri,
    OrchestratorSettings,
    ToolCall,
    ToolCallContext,
    ToolCallResult,
    TransformerEngineSettings,
)
from avalan.event.manager import EventManagerMode
from avalan.model.hubs.huggingface import HuggingfaceHub
from avalan.skill import (
    BundledSkillSourceAuthority,
    PluginProvidedSkillSourceAuthority,
    PreinstalledRemoteSkillSourceAuthority,
    SkillObservabilitySettings,
    SkillPrivacySettings,
    SkillReadLimits,
    SkillSourceConfig,
    TrustedSkillSettings,
    UserLocalSkillSourceAuthority,
    WorkspaceSkillSourceAuthority,
)
from avalan.tool.context import ToolSettingsContext
from avalan.tool.manager import ToolManager


class SkillsLoaderTestCase(IsolatedAsyncioTestCase):
    def test_effective_skills_enabled_tools_adds_manifest_filter(
        self,
    ) -> None:
        settings = TrustedSkillSettings(
            sources=(
                SkillSourceConfig(
                    label="pdf",
                    authority=WorkspaceSkillSourceAuthority(),
                    manifest_path="/tmp/SKILL.md",
                ),
            ),
        )

        self.assertEqual(
            loader_module.effective_skills_enabled_tools(
                settings,
                ["shell.pdfinfo"],
            ),
            ["shell.pdfinfo", "skills"],
        )
        self.assertIsNone(
            loader_module.effective_skills_enabled_tools(settings, None)
        )
        self.assertEqual(
            loader_module.effective_skills_enabled_tools(
                settings,
                ["skills.read"],
            ),
            ["skills.read"],
        )

    def test_manifest_skills_config_merges_with_trusted_settings(
        self,
    ) -> None:
        trusted = TrustedSkillSettings(
            sources=(
                SkillSourceConfig(
                    label="workspace-main",
                    authority=WorkspaceSkillSourceAuthority(),
                    root_path="/tmp/skills",
                ),
            ),
        )

        merged, narrowing = (
            OrchestratorLoader._trusted_manifest_skills_settings_from_config(
                {
                    "file_auto_enable": False,
                    "files": {
                        "pdf": {
                            "path": "/tmp/SKILL.md",
                            "authority": "plugin_provided:pdf-plugin",
                            "allow_hidden": True,
                        }
                    },
                },
                trusted=trusted,
            )
        )

        assert isinstance(merged, TrustedSkillSettings)
        self.assertFalse(merged.manifest_auto_enable)
        self.assertEqual(narrowing, {})
        self.assertEqual(
            tuple(source.label for source in merged.sources),
            ("workspace-main", "pdf"),
        )
        manifest_source = merged.sources[1]
        self.assertEqual(manifest_source.manifest_path, "/tmp/SKILL.md")
        self.assertTrue(manifest_source.allow_hidden_paths)
        self.assertIsInstance(
            manifest_source.authority,
            PluginProvidedSkillSourceAuthority,
        )

        with self.assertRaisesRegex(AssertionError, "labels must be unique"):
            OrchestratorLoader._trusted_manifest_skills_settings_from_config(
                {"files": {"workspace-main": "/tmp/SKILL.md"}},
                trusted=trusted,
            )

    def test_manifest_source_config_rejects_invalid_shapes(self) -> None:
        with self.assertRaisesRegex(AssertionError, "strings or mappings"):
            OrchestratorLoader._manifest_skill_source_from_config("pdf", 123)
        with self.assertRaisesRegex(AssertionError, "unknown keys"):
            OrchestratorLoader._manifest_skill_source_from_config(
                "pdf",
                {"path": "/tmp/SKILL.md", "unknown": True},
            )
        with self.assertRaisesRegex(AssertionError, "path must be"):
            OrchestratorLoader._manifest_skill_source_from_config(
                "pdf",
                {"path": 123},
            )
        with self.assertRaisesRegex(AssertionError, "authority must be"):
            OrchestratorLoader._manifest_skill_source_from_config(
                "pdf",
                {"path": "/tmp/SKILL.md", "authority": 123},
            )
        with self.assertRaisesRegex(AssertionError, "allow_hidden"):
            OrchestratorLoader._manifest_skill_source_from_config(
                "pdf",
                {"path": "/tmp/SKILL.md", "allow_hidden": "yes"},
            )

    def test_skill_source_authority_from_config_variants(self) -> None:
        cases = (
            (
                "bundled:core",
                BundledSkillSourceAuthority,
                "bundle_id",
                "core",
            ),
            (
                "workspace:docs",
                WorkspaceSkillSourceAuthority,
                "workspace_id",
                "docs",
            ),
            (
                "user_local:profile",
                UserLocalSkillSourceAuthority,
                "profile_id",
                "profile",
            ),
            (
                "plugin_provided:pdf-plugin",
                PluginProvidedSkillSourceAuthority,
                "plugin_id",
                "pdf-plugin",
            ),
            (
                "preinstalled_remote:registry",
                PreinstalledRemoteSkillSourceAuthority,
                "registry_id",
                "registry",
            ),
        )

        for value, cls, attribute, expected in cases:
            with self.subTest(value=value):
                authority = (
                    OrchestratorLoader._skill_source_authority_from_config(
                        value
                    )
                )
                self.assertIsInstance(authority, cls)
                self.assertEqual(getattr(authority, attribute), expected)

        with self.assertRaisesRegex(AssertionError, "unsupported"):
            OrchestratorLoader._skill_source_authority_from_config("network")
        with self.assertRaisesRegex(AssertionError, "requires plugin id"):
            OrchestratorLoader._skill_source_authority_from_config(
                "plugin_provided"
            )
        with self.assertRaisesRegex(AssertionError, "requires registry id"):
            OrchestratorLoader._skill_source_authority_from_config(
                "preinstalled_remote"
            )

    async def test_from_settings_exposes_enabled_skills_toolset(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(root / "pdf" / "SKILL.md")

            manager = await _loaded_tool_manager(
                _settings(tools=["skills"]),
                tool_settings=ToolSettingsContext(
                    skills=_trusted_settings(root)
                ),
            )

            self.assertEqual(
                [descriptor.name for descriptor in manager.list_tools()],
                [
                    "skills.list",
                    "skills.match",
                    "skills.read",
                    "skills.check",
                ],
            )
            self.assertIsNotNone(manager.bootstrap_prompt())

    async def test_cli_manifest_file_settings_auto_expose_skills_toolset(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "pdf" / "SKILL.md"
            _write_skill(
                manifest,
                body="# PDF Body\nFOLLOW_THE_PDF_STEPS\n",
            )
            tool_settings = agent_cmds._agent_tool_settings(
                Namespace(tool_skills_file=[f"pdf={manifest}"])
            )

            manager = await _loaded_tool_manager(
                _settings(tools=None),
                tool_settings=tool_settings,
            )
            read = await manager(
                ToolCall(
                    id="read",
                    name="skills.read",
                    arguments={"skill": "pdf"},
                ),
                ToolCallContext(),
            )

        self.assertIn(
            "skills.read",
            [descriptor.name for descriptor in manager.list_tools()],
        )
        self.assertIsInstance(read, ToolCallResult)
        assert isinstance(read, ToolCallResult)
        content = cast(dict[str, Any], _result_dict(read)["content"])
        self.assertIn("FOLLOW_THE_PDF_STEPS", content["text"])

    async def test_from_settings_fake_loop_matches_then_reads(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "pdf" / "SKILL.md",
                body="# PDF Body\nFOLLOW_THE_PDF_STEPS\n",
            )

            manager = await _loaded_tool_manager(
                _settings(tools=["skills"]),
                tool_settings=ToolSettingsContext(
                    skills=_trusted_settings(root)
                ),
            )
            loop = _FakeSkillLoop()

            answer = await loop.answer(manager, "render a pdf")

        self.assertEqual(loop.calls, ["skills.match", "skills.read"])
        self.assertEqual(answer, "answered after read")

    async def test_from_settings_reads_manifest_file_source(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "pdf" / "SKILL.md"
            _write_skill(
                manifest,
                body="# PDF Body\nFOLLOW_THE_PDF_STEPS\n",
            )

            manager = await _loaded_tool_manager(
                _settings(tools=["skills.read"]),
                tool_settings=ToolSettingsContext(
                    skills=TrustedSkillSettings(
                        sources=(
                            SkillSourceConfig(
                                label="pdf",
                                authority=WorkspaceSkillSourceAuthority(),
                                manifest_path=manifest,
                            ),
                        ),
                    )
                ),
            )
            read = await manager(
                ToolCall(
                    id="read",
                    name="skills.read",
                    arguments={"skill": "pdf"},
                ),
                ToolCallContext(),
            )

        self.assertIsInstance(read, ToolCallResult)
        assert isinstance(read, ToolCallResult)
        content = cast(dict[str, Any], _result_dict(read)["content"])
        self.assertIn("FOLLOW_THE_PDF_STEPS", content["text"])

    async def test_configured_skills_without_enablement_are_not_exposed(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(root / "pdf" / "SKILL.md")

            manager = await _loaded_tool_manager(
                _settings(tools=None),
                tool_settings=ToolSettingsContext(
                    skills=_trusted_settings(root)
                ),
            )

        self.assertNotIn(
            "skills.read",
            [descriptor.name for descriptor in manager.list_tools()],
        )
        self.assertIsNone(manager.describe_tool("skills.read"))

    async def test_enabling_skills_without_trusted_settings_fails_closed(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            AssertionError,
            "trusted skills settings",
        ):
            await _loaded_tool_manager(_settings(tools=["skills"]))

    async def test_disabled_trusted_skills_fail_closed_when_enabled(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(root / "pdf" / "SKILL.md")

            with self.assertRaisesRegex(AssertionError, "enabled trusted"):
                await _loaded_tool_manager(
                    _settings(tools=["skills"]),
                    tool_settings=ToolSettingsContext(
                        skills=_trusted_settings(root, enabled=False)
                    ),
                )

    async def test_missing_trusted_source_root_fails_closed(self) -> None:
        settings = TrustedSkillSettings(
            sources=(
                SkillSourceConfig(
                    label="workspace-main",
                    authority=WorkspaceSkillSourceAuthority(),
                ),
            )
        )

        with self.assertRaisesRegex(
            AssertionError,
            "at least one trusted source",
        ):
            await _loaded_tool_manager(
                _settings(tools=["skills"]),
                tool_settings=ToolSettingsContext(skills=settings),
            )

    async def test_toml_skills_narrow_trusted_settings(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(root / "pdf" / "SKILL.md")

            manager = await _loaded_tool_manager_from_file(
                _agent_toml("""
[tool]
enable = ["skills.read"]

[tool.skills]
source_labels = ["workspace-main"]
skill_ids = ["pdf"]
bootstrap = "off"

[tool.skills.read_limits]
max_bytes_per_read = 1024
"""),
                tool_settings=ToolSettingsContext(
                    skills=_trusted_settings(
                        root,
                        read_limits=SkillReadLimits(max_bytes_per_read=2048),
                    )
                ),
            )

            read = await manager(
                ToolCall(
                    id="read",
                    name="skills.read",
                    arguments={"skill": "pdf"},
                ),
                ToolCallContext(),
            )

        self.assertIsInstance(read, ToolCallResult)
        self.assertIsNone(manager.bootstrap_prompt())

    async def test_toml_skills_accepts_bool_bootstrap_and_authority_kinds(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(root / "pdf" / "SKILL.md")

            manager = await _loaded_tool_manager_from_file(
                _agent_toml("""
[tool]
enable = ["skills.read"]

[tool.skills]
bootstrap = false
authority_kinds = ["workspace"]
"""),
                tool_settings=ToolSettingsContext(
                    skills=_trusted_settings(root)
                ),
            )

            read = await manager(
                ToolCall(
                    id="read",
                    name="skills.read",
                    arguments={"skill": "pdf"},
                ),
                ToolCallContext(),
            )

        self.assertIsInstance(read, ToolCallResult)
        self.assertIsNone(manager.bootstrap_prompt())

    async def test_toml_skills_files_define_trusted_manifest_source(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "pdf" / "SKILL.md"
            _write_skill(
                manifest,
                body="# PDF Body\nFOLLOW_THE_PDF_STEPS\n",
            )

            manager = await _loaded_tool_manager_from_file(
                _agent_toml(f"""
[tool]
enable = ["skills.read"]

[tool.skills.files]
pdf = {dumps(str(manifest))}
"""),
            )
            read = await manager(
                ToolCall(
                    id="read",
                    name="skills.read",
                    arguments={"skill": "pdf"},
                ),
                ToolCallContext(),
            )

        self.assertIsInstance(read, ToolCallResult)
        assert isinstance(read, ToolCallResult)
        content = cast(dict[str, Any], _result_dict(read)["content"])
        self.assertIn("FOLLOW_THE_PDF_STEPS", content["text"])

    async def test_toml_skills_files_auto_expose_without_tool_enable(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "pdf" / "SKILL.md"
            _write_skill(
                manifest,
                body="# PDF Body\nFOLLOW_THE_PDF_STEPS\n",
            )

            manager = await _loaded_tool_manager_from_file(
                _agent_toml(f"""
[tool.skills.files]
pdf = {dumps(str(manifest))}
"""),
            )
            read = await manager(
                ToolCall(
                    id="read",
                    name="skills.read",
                    arguments={"skill": "pdf"},
                ),
                ToolCallContext(),
            )

        self.assertIn(
            "skills.read",
            [descriptor.name for descriptor in manager.list_tools()],
        )
        self.assertIsInstance(read, ToolCallResult)
        assert isinstance(read, ToolCallResult)
        content = cast(dict[str, Any], _result_dict(read)["content"])
        self.assertIn("FOLLOW_THE_PDF_STEPS", content["text"])

    async def test_toml_skills_files_auto_enable_opt_out_hides_tools(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "pdf" / "SKILL.md"
            _write_skill(manifest)

            manager = await _loaded_tool_manager_from_file(
                _agent_toml(f"""
[tool.skills]
manifest_auto_enable = false

[tool.skills.files]
pdf = {dumps(str(manifest))}
"""),
            )

        self.assertNotIn(
            "skills.read",
            [descriptor.name for descriptor in manager.list_tools()],
        )
        self.assertIsNone(manager.describe_tool("skills.read"))

    async def test_toml_skills_files_explicit_skill_ids_are_authoritative(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "pdf" / "SKILL.md"
            _write_skill(
                manifest,
                body="# PDF Body\nFOLLOW_THE_PDF_STEPS\n",
            )

            manager = await _loaded_tool_manager_from_file(
                _agent_toml(f"""
[tool.skills]
skill_ids = ["ocr"]

[tool.skills.files]
pdf = {dumps(str(manifest))}
"""),
            )
            read = await manager(
                ToolCall(
                    id="read",
                    name="skills.read",
                    arguments={"skill": "pdf"},
                ),
                ToolCallContext(),
            )

        self.assertIn(
            "skills.read",
            [descriptor.name for descriptor in manager.list_tools()],
        )
        self.assertIsInstance(read, ToolCallResult)
        assert isinstance(read, ToolCallResult)
        encoded = dumps(_result_dict(read), sort_keys=True)
        self.assertIn("skill_not_allowed", encoded)
        self.assertNotIn("FOLLOW_THE_PDF_STEPS", encoded)

    async def test_cli_manifest_auto_label_does_not_block_toml_skill_ids(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "pdf" / "SKILL.md"
            _write_skill(
                manifest,
                body="# PDF Body\nFOLLOW_THE_PDF_STEPS\n",
            )
            tool_settings = agent_cmds._agent_tool_settings(
                Namespace(tool_skills_file=[f"pdf={manifest}"])
            )

            manager = await _loaded_tool_manager_from_file(
                _agent_toml("""
[tool.skills]
skill_ids = ["ocr"]
"""),
                tool_settings=tool_settings,
            )
            read = await manager(
                ToolCall(
                    id="read",
                    name="skills.read",
                    arguments={"skill": "pdf"},
                ),
                ToolCallContext(),
            )

        self.assertIn(
            "skills.read",
            [descriptor.name for descriptor in manager.list_tools()],
        )
        self.assertIsInstance(read, ToolCallResult)
        assert isinstance(read, ToolCallResult)
        encoded = dumps(_result_dict(read), sort_keys=True)
        self.assertIn("skill_not_allowed", encoded)
        self.assertNotIn("FOLLOW_THE_PDF_STEPS", encoded)

    async def test_toml_partial_read_limits_inherit_trusted_values(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(root / "pdf" / "SKILL.md")

            manager = await _loaded_tool_manager_from_file(
                _agent_toml("""
[tool]
enable = ["skills.read"]

[tool.skills.read_limits]
max_bytes_per_read = 512
"""),
                tool_settings=ToolSettingsContext(
                    skills=_trusted_settings(
                        root,
                        read_limits=SkillReadLimits(
                            max_bytes_per_read=1024,
                            max_lines_per_read=100,
                        ),
                    )
                ),
            )

            read = await manager(
                ToolCall(
                    id="read",
                    name="skills.read",
                    arguments={"skill": "pdf"},
                ),
                ToolCallContext(),
            )

        self.assertIsInstance(read, ToolCallResult)

    async def test_toml_partial_privacy_observability_inherit_trusted_values(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(root / "pdf" / "SKILL.md")

            manager = await _loaded_tool_manager_from_file(
                _agent_toml("""
[tool]
enable = ["skills.list"]

[tool.skills.privacy]
include_source_labels = false

[tool.skills.observability]
emit_events = false
"""),
                tool_settings=ToolSettingsContext(
                    skills=_trusted_settings(
                        root,
                        privacy=SkillPrivacySettings(
                            include_authority=False,
                        ),
                        observability=SkillObservabilitySettings(
                            include_diagnostics=False,
                        ),
                    )
                ),
            )

            listed = await manager(
                ToolCall(
                    id="list",
                    name="skills.list",
                    arguments={},
                ),
                ToolCallContext(),
            )

        assert isinstance(listed, ToolCallResult)
        encoded = dumps(_result_dict(listed), sort_keys=True)
        self.assertNotIn("source_label", encoded)
        self.assertNotIn("diagnostics", encoded)

    async def test_cli_observability_settings_filter_model_responses(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "pdf" / "SKILL.md",
                body="# PDF Body\nFOLLOW_THE_PDF_STEPS\n",
            )
            tool_settings = agent_cmds._agent_tool_settings(
                Namespace(
                    tool_skills_source=[f"workspace-main={root}"],
                    tool_skills_diagnostics="off",
                    tool_skills_observability="off",
                )
            )
            manager = await _loaded_tool_manager(
                _settings(tools=["skills"]),
                tool_settings=tool_settings,
            )

            listed = await manager(
                ToolCall(id="list", name="skills.list", arguments={}),
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

        for outcome in (listed, read):
            assert isinstance(outcome, ToolCallResult)
            encoded = dumps(_result_dict(outcome), sort_keys=True)
            self.assertNotIn("diagnostics", encoded)
            self.assertNotIn("path", encoded)
            self.assertNotIn("size_bytes", encoded)
            self.assertNotIn("start_byte", encoded)
            self.assertNotIn("end_byte", encoded)
            self.assertNotIn(str(root), encoded)

    async def test_toml_skills_without_trusted_base_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            AssertionError,
            "requires trusted skills settings",
        ):
            await _loaded_tool_manager_from_file(_agent_toml("""
[tool.skills]
skill_ids = ["pdf"]
"""))

    async def test_toml_skills_reject_source_definitions(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(root / "pdf" / "SKILL.md")

            with self.assertRaisesRegex(
                AssertionError,
                "cannot define sources",
            ):
                await _loaded_tool_manager_from_file(
                    _agent_toml("""
[tool.skills]
sources = ["workspace-main"]
"""),
                    tool_settings=ToolSettingsContext(
                        skills=_trusted_settings(root)
                    ),
                )

    async def test_toml_skills_reject_unknown_keys(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(root / "pdf" / "SKILL.md")

            with self.assertRaisesRegex(AssertionError, "unknown keys"):
                await _loaded_tool_manager_from_file(
                    _agent_toml("""
[tool.skills]
load = true
"""),
                    tool_settings=ToolSettingsContext(
                        skills=_trusted_settings(root)
                    ),
                )

    async def test_toml_skills_cannot_widen_limits(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(root / "pdf" / "SKILL.md")

            with self.assertRaisesRegex(
                AssertionError,
                "cannot widen trusted settings",
            ):
                await _loaded_tool_manager_from_file(
                    _agent_toml("""
[tool.skills.read_limits]
max_bytes_per_read = 2048
"""),
                    tool_settings=ToolSettingsContext(
                        skills=_trusted_settings(
                            root,
                            read_limits=SkillReadLimits(
                                max_bytes_per_read=1024
                            ),
                        )
                    ),
                )


class _FakeSkillLoop:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def answer(self, manager: ToolManager, task: str) -> str:
        matched = await manager(
            ToolCall(
                id="match",
                name="skills.match",
                arguments={"query": task},
            ),
            ToolCallContext(),
        )
        self.calls.append("skills.match")
        assert isinstance(matched, ToolCallResult)
        matched_result = _result_dict(matched)
        assert "FOLLOW_THE_PDF_STEPS" not in str(matched_result)
        items = cast(tuple[dict[str, Any], ...], matched_result["items"])
        metadata = cast(dict[str, Any], items[0]["metadata"])
        skill_id = metadata["skill_id"]
        assert isinstance(skill_id, str)

        read = await manager(
            ToolCall(
                id="read",
                name="skills.read",
                arguments={"skill": skill_id},
            ),
            ToolCallContext(),
        )
        self.calls.append("skills.read")
        assert isinstance(read, ToolCallResult)
        read_result = _result_dict(read)
        content = cast(dict[str, Any], read_result["content"])
        body = content["text"]
        assert isinstance(body, str)
        if "FOLLOW_THE_PDF_STEPS" not in body:
            return "missing skill body"
        return "answered after read"


async def _loaded_tool_manager(
    settings: OrchestratorSettings,
    *,
    tool_settings: ToolSettingsContext | None = None,
) -> ToolManager:
    stack = AsyncExitStack()
    captured: list[ToolManager] = []

    class CapturingOrchestrator:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            captured.append(cast(ToolManager, _args[4]))

    try:
        with _loader_patches(CapturingOrchestrator):
            loader = OrchestratorLoader(
                hub=MagicMock(spec=HuggingfaceHub),
                logger=MagicMock(spec=Logger),
                participant_id=uuid4(),
                stack=stack,
            )
            await loader.from_settings(
                settings,
                tool_settings=tool_settings,
                event_manager_mode=EventManagerMode.CLI,
            )
            assert captured
            return captured[0]
    finally:
        await stack.aclose()


async def _loaded_tool_manager_from_file(
    config: str,
    *,
    tool_settings: ToolSettingsContext | None = None,
) -> ToolManager:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "agent.toml"
        path.write_text(config, encoding="utf-8")
        settings_stack = AsyncExitStack()
        captured: list[ToolManager] = []

        class CapturingOrchestrator:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                captured.append(cast(ToolManager, _args[4]))

        try:
            with _loader_patches(CapturingOrchestrator):
                loader = OrchestratorLoader(
                    hub=MagicMock(spec=HuggingfaceHub),
                    logger=MagicMock(spec=Logger),
                    participant_id=uuid4(),
                    stack=settings_stack,
                )
                await loader.from_file(
                    str(path),
                    agent_id=uuid4(),
                    tool_settings=tool_settings,
                    event_manager_mode=EventManagerMode.CLI,
                )
                assert captured
                return captured[0]
        finally:
            await settings_stack.aclose()


def _loader_patches(orchestrator: type[object]) -> Any:
    model_manager = MagicMock()
    model_manager.__enter__.return_value = model_manager
    model_manager.__exit__.return_value = None
    model_manager.parse_uri.return_value = EngineUri(
        host=None,
        port=None,
        user=None,
        password=None,
        vendor=None,
        model_id="model",
        params={},
    )
    model_manager.get_engine_settings.return_value = (
        TransformerEngineSettings()
    )
    memory = MagicMock()
    memory.participant_id = uuid4()
    return patch.multiple(
        "avalan.agent.loader",
        MemoryManager=MagicMock(
            create_instance=AsyncMock(return_value=memory)
        ),
        ModelManager=MagicMock(return_value=model_manager),
        DefaultOrchestrator=orchestrator,
        EventManager=MagicMock(return_value=MagicMock()),
        HAS_GRAPH_DEPENDENCIES=False,
        HAS_CODE_DEPENDENCIES=False,
        HAS_BROWSER_DEPENDENCIES=False,
    )


def _settings(*, tools: list[str] | None) -> OrchestratorSettings:
    return OrchestratorSettings(
        agent_id=uuid4(),
        orchestrator_type=None,
        agent_config={"role": "assistant"},
        uri="ai://local/model",
        engine_config={},
        call_options=None,
        template_vars=None,
        memory_permanent_message=None,
        permanent_memory=None,
        memory_recent=False,
        sentence_model_id=OrchestratorLoader.DEFAULT_SENTENCE_MODEL_ID,
        sentence_model_engine_config=None,
        sentence_model_max_tokens=500,
        sentence_model_overlap_size=125,
        sentence_model_window_size=250,
        json_config=None,
        tools=tools,
        log_events=True,
    )


def _trusted_settings(
    root: Path,
    *,
    enabled: bool = True,
    read_limits: SkillReadLimits | None = None,
    privacy: SkillPrivacySettings | None = None,
    observability: SkillObservabilitySettings | None = None,
) -> TrustedSkillSettings:
    return TrustedSkillSettings(
        enabled=enabled,
        sources=(
            SkillSourceConfig(
                label="workspace-main",
                authority=WorkspaceSkillSourceAuthority(),
                root_path=root,
            ),
        ),
        read_limits=read_limits or SkillReadLimits(),
        privacy=privacy or SkillPrivacySettings(),
        observability=observability or SkillObservabilitySettings(),
    )


def _agent_toml(extra: str = "") -> str:
    return """
[agent]
role = "assistant"

[engine]
uri = "ai://local/model"
""" + extra


def _result_dict(outcome: ToolCallResult) -> dict[str, Any]:
    assert isinstance(outcome.result, dict)
    return cast(dict[str, Any], outcome.result)


def _write_skill(path: Path, *, body: str = "# PDF Body\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        "name: pdf\n"
        "description: PDF rendering guidance.\n"
        'tags: ["pdf"]\n'
        "resources: []\n"
        "---\n"
        f"{body}",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
