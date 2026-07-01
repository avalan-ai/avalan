from argparse import ArgumentParser, Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main

from avalan.cli.__main__ import CLI
from avalan.cli.commands import agent as agent_cmds
from avalan.skill import (
    BundledSkillSourceAuthority,
    PluginProvidedSkillSourceAuthority,
    PreinstalledRemoteSkillSourceAuthority,
    SkillCursorLimits,
    SkillIndexLimits,
    SkillObservabilitySettings,
    SkillPrivacySettings,
    SkillReadLimits,
    SkillSourceAuthorityKind,
    SkillSourceConfig,
    SkillSourceLimits,
    TrustedSkillSettings,
    UserLocalSkillSourceAuthority,
    WorkspaceSkillSourceAuthority,
)


class SkillsSettingsCliTestCase(TestCase):
    def test_agent_message_search_accepts_skills_settings_flags(self) -> None:
        parser = CLI._create_parser(
            default_device="cpu",
            cache_dir="/tmp",
            default_locales_path="/tmp",
            default_locale="en_US",
        )

        args = parser.parse_args(
            [
                "agent",
                "message",
                "search",
                "agent.toml",
                "--function",
                "l2_distance",
                "--id",
                "agent-id",
                "--participant",
                "participant-id",
                "--session",
                "session-id",
                "--tool-skills-source",
                "workspace-main=/tmp/skills",
                "--tool-skills-source-authority",
                "workspace-main=workspace:project",
                "--tool-skills-bootstrap",
                "off",
                "--tool-skills-max-bytes-per-read",
                "256",
            ]
        )
        context = agent_cmds._agent_tool_settings(args)

        self.assertEqual(args.agent_command, "message")
        self.assertEqual(args.agent_message_command, "search")
        settings = context.skills
        assert isinstance(settings, TrustedSkillSettings)
        self.assertFalse(settings.bootstrap_enabled)
        self.assertEqual(settings.read_limits.max_bytes_per_read, 256)
        self.assertEqual(len(settings.sources), 1)
        source = settings.sources[0]
        self.assertEqual(source.label, "workspace-main")
        self.assertEqual(source.root_path, "/tmp/skills")
        authority = source.authority
        assert isinstance(authority, WorkspaceSkillSourceAuthority)
        self.assertEqual(authority.workspace_id, "project")

    def test_agent_tool_settings_maps_skills_flags(self) -> None:
        with TemporaryDirectory() as directory:
            root = str(Path(directory) / "skills")
            args = Namespace(
                tool_skills_source=[f"workspace-main={root}"],
                tool_skills_source_authority=[
                    "workspace-main=workspace:project"
                ],
                tool_skills_source_package=["workspace-main=package"],
                tool_skills_source_allow_hidden=["workspace-main"],
                tool_skills_authority_kind=["workspace"],
                tool_skills_skill=["pdf"],
                tool_skills_bootstrap="off",
                tool_skills_diagnostics="verbose",
                tool_skills_observability="verbose",
                tool_skills_max_bytes_per_read=123,
                tool_skills_max_lines_per_read=45,
                tool_skills_max_skills=6,
                tool_skills_max_resources_per_skill=7,
                tool_skills_max_indexed_bytes=890,
                tool_skills_max_sources=2,
                tool_skills_max_resources_per_source=11,
                tool_skills_max_source_depth=3,
                tool_skills_max_files_per_source=12,
                tool_skills_max_directory_entries_per_source=13,
                tool_skills_max_active_cursors=4,
                tool_skills_max_cursor_age_seconds=30,
            )

            context = agent_cmds._agent_tool_settings(args)

        settings = context.skills
        assert isinstance(settings, TrustedSkillSettings)
        self.assertFalse(settings.bootstrap_enabled)
        self.assertEqual(
            settings.authority_kinds,
            (SkillSourceAuthorityKind.WORKSPACE,),
        )
        self.assertEqual(settings.allowed_skill_ids, ("pdf",))
        self.assertEqual(settings.read_limits.max_bytes_per_read, 123)
        self.assertEqual(settings.read_limits.max_lines_per_read, 45)
        self.assertEqual(settings.index_limits.max_skills, 6)
        self.assertEqual(settings.index_limits.max_resources_per_skill, 7)
        self.assertEqual(settings.index_limits.max_indexed_bytes, 890)
        self.assertEqual(settings.source_limits.max_sources, 2)
        self.assertEqual(
            settings.source_limits.max_resources_per_source,
            11,
        )
        self.assertEqual(settings.source_limits.max_source_depth, 3)
        self.assertEqual(settings.source_limits.max_files_per_source, 12)
        self.assertEqual(
            settings.source_limits.max_directory_entries_per_source,
            13,
        )
        self.assertEqual(settings.cursor_limits.max_active_cursors, 4)
        self.assertEqual(settings.cursor_limits.max_cursor_age_seconds, 30)
        self.assertTrue(settings.observability.include_byte_counts)
        self.assertTrue(settings.privacy.include_diagnostic_paths)
        self.assertEqual(len(settings.sources), 1)
        source = settings.sources[0]
        self.assertEqual(source.label, "workspace-main")
        self.assertEqual(source.root_path, root)
        self.assertEqual(source.package_path, "package")
        self.assertTrue(source.allow_hidden_paths)
        authority = source.authority
        assert isinstance(authority, WorkspaceSkillSourceAuthority)
        self.assertEqual(authority.workspace_id, "project")
        self.assertNotIn(root, str(settings.as_model_dict()))

    def test_agent_tool_settings_without_skills_flags_is_none(self) -> None:
        context = agent_cmds._agent_tool_settings(Namespace())

        self.assertIsNone(context.skills)

    def test_agent_tool_settings_disabled_flag_creates_disabled_skills(
        self,
    ) -> None:
        context = agent_cmds._agent_tool_settings(
            Namespace(tool_skills_disabled=True)
        )

        settings = context.skills
        assert isinstance(settings, TrustedSkillSettings)
        self.assertFalse(settings.enabled)

    def test_agent_tool_settings_maps_all_source_authority_variants(
        self,
    ) -> None:
        args = Namespace(
            tool_skills_source=[
                "bundled-main=/tmp/bundled",
                "user-main=/tmp/user",
                "plugin-main=/tmp/plugin",
                "remote-main=/tmp/remote",
            ],
            tool_skills_source_authority=[
                "bundled-main=bundled:core",
                "user-main=user_local:profile",
                "plugin-main=plugin_provided:pdf-plugin",
                "remote-main=preinstalled_remote:registry",
            ],
        )

        context = agent_cmds._agent_tool_settings(args)

        settings = context.skills
        assert isinstance(settings, TrustedSkillSettings)
        authorities = {
            source.label: source.authority for source in settings.sources
        }
        bundled = authorities["bundled-main"]
        assert isinstance(bundled, BundledSkillSourceAuthority)
        self.assertEqual(bundled.bundle_id, "core")
        user_local = authorities["user-main"]
        assert isinstance(user_local, UserLocalSkillSourceAuthority)
        self.assertEqual(user_local.profile_id, "profile")
        plugin = authorities["plugin-main"]
        assert isinstance(plugin, PluginProvidedSkillSourceAuthority)
        self.assertEqual(plugin.plugin_id, "pdf-plugin")
        remote = authorities["remote-main"]
        assert isinstance(remote, PreinstalledRemoteSkillSourceAuthority)
        self.assertEqual(remote.registry_id, "registry")

    def test_skills_template_settings_serialize_non_default_values(
        self,
    ) -> None:
        settings = TrustedSkillSettings(
            enabled=False,
            bootstrap_enabled=False,
            authority_kinds=(SkillSourceAuthorityKind.WORKSPACE,),
            sources=(
                SkillSourceConfig(
                    label="workspace-main",
                    authority=WorkspaceSkillSourceAuthority(),
                    root_path="/tmp/skills",
                ),
            ),
            allowed_skill_ids=("pdf",),
            read_limits=SkillReadLimits(max_bytes_per_read=123),
            index_limits=SkillIndexLimits(max_skills=7),
            source_limits=SkillSourceLimits(max_sources=1),
            cursor_limits=SkillCursorLimits(max_active_cursors=2),
            privacy=SkillPrivacySettings(include_authority=False),
            observability=SkillObservabilitySettings(include_byte_counts=True),
        )

        rendered = agent_cmds._skills_tool_template_settings(settings)

        assert rendered is not None
        self.assertFalse(rendered["enabled"])
        self.assertEqual(rendered["bootstrap"], "off")
        self.assertEqual(rendered["authority_kinds"], ("workspace",))
        self.assertEqual(rendered["source_labels"], ("workspace-main",))
        self.assertEqual(rendered["skill_ids"], ("pdf",))
        read_limits = rendered["read_limits"]
        assert isinstance(read_limits, dict)
        self.assertEqual(read_limits["max_bytes_per_read"], 123)
        index_limits = rendered["index_limits"]
        assert isinstance(index_limits, dict)
        self.assertEqual(index_limits["max_skills"], 7)
        source_limits = rendered["source_limits"]
        assert isinstance(source_limits, dict)
        self.assertEqual(source_limits["max_sources"], 1)
        cursor_limits = rendered["cursor_limits"]
        assert isinstance(cursor_limits, dict)
        self.assertEqual(cursor_limits["max_active_cursors"], 2)
        privacy = rendered["privacy"]
        assert isinstance(privacy, dict)
        self.assertFalse(privacy["include_authority"])
        observability = rendered["observability"]
        assert isinstance(observability, dict)
        self.assertTrue(observability["include_byte_counts"])

    def test_agent_tool_settings_rejects_unknown_source_label(self) -> None:
        args = Namespace(
            tool_skills_source=["workspace-main=/tmp/skills"],
            tool_skills_source_package=["other=package"],
        )

        with self.assertRaisesRegex(AssertionError, "unknown labels"):
            agent_cmds._agent_tool_settings(args)

    def test_agent_tool_settings_rejects_duplicate_source_label(
        self,
    ) -> None:
        args = Namespace(
            tool_skills_source=[
                "workspace-main=/tmp/one",
                "workspace-main=/tmp/two",
            ],
        )

        with self.assertRaisesRegex(AssertionError, "labels must be unique"):
            agent_cmds._agent_tool_settings(args)

    def test_agent_tool_settings_rejects_invalid_plugin_authority(
        self,
    ) -> None:
        args = Namespace(
            tool_skills_source=["workspace-main=/tmp/skills"],
            tool_skills_source_authority=["workspace-main=plugin_provided"],
        )

        with self.assertRaisesRegex(AssertionError, "requires plugin id"):
            agent_cmds._agent_tool_settings(args)

    def test_agent_tool_settings_rejects_unknown_source_authority(
        self,
    ) -> None:
        args = Namespace(
            tool_skills_source=["workspace-main=/tmp/skills"],
            tool_skills_source_authority=["workspace-main=network"],
        )

        with self.assertRaisesRegex(AssertionError, "unsupported"):
            agent_cmds._agent_tool_settings(args)

    def test_argparse_rejects_unknown_skills_choices(self) -> None:
        parser = ArgumentParser()
        CLI._add_skills_settings_arguments(parser)

        with self.assertRaises(SystemExit):
            parser.parse_args(["--tool-skills-observability", "loud"])

    def test_argparse_accepts_skills_flags(self) -> None:
        parser = ArgumentParser()
        CLI._add_skills_settings_arguments(parser)

        args = parser.parse_args(
            [
                "--tool-skills-source",
                "workspace-main=/tmp/skills",
                "--tool-skills-authority-kind",
                "workspace",
                "--tool-skills-bootstrap",
                "off",
            ]
        )

        self.assertEqual(
            args.tool_skills_source,
            ["workspace-main=/tmp/skills"],
        )
        self.assertEqual(args.tool_skills_authority_kind, ["workspace"])
        self.assertEqual(args.tool_skills_bootstrap, "off")


if __name__ == "__main__":
    main()
