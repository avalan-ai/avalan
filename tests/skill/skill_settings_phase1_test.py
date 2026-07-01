from json import dumps
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main

from avalan.skill import (
    PluginProvidedSkillSourceAuthority,
    SkillCursorLimits,
    SkillIndexLimits,
    SkillObservabilitySettings,
    SkillPrivacySettings,
    SkillReadLimits,
    SkillSettingsSurface,
    SkillSourceAuthorityKind,
    SkillSourceConfig,
    SkillSourceLimits,
    SkillStatus,
    TrustedSkillSettings,
    UntrustedSkillSettings,
    WorkspaceSkillSourceAuthority,
    merge_skill_settings,
    trusted_skill_settings_identity_dict,
    trusted_skill_source_identity_dict,
    untrusted_skill_settings_config_dict,
)


class SkillSettingsTest(TestCase):
    def test_untrusted_settings_narrow_trusted_bounds(self) -> None:
        workspace_source = SkillSourceConfig(
            label="workspace-main",
            authority=WorkspaceSkillSourceAuthority(),
        )
        plugin_source = SkillSourceConfig(
            label="plugin-pdf",
            authority=PluginProvidedSkillSourceAuthority(
                plugin_id="pdf-plugin",
            ),
        )
        trusted = TrustedSkillSettings(
            enabled=True,
            bootstrap_enabled=True,
            authority_kinds=(
                SkillSourceAuthorityKind.WORKSPACE,
                SkillSourceAuthorityKind.PLUGIN_PROVIDED,
            ),
            sources=(workspace_source, plugin_source),
            allowed_skill_ids=("pdf", "docx"),
            read_limits=SkillReadLimits(max_bytes_per_read=4096),
            index_limits=SkillIndexLimits(max_skills=64),
            source_limits=SkillSourceLimits(max_sources=4),
            cursor_limits=SkillCursorLimits(max_active_cursors=8),
            privacy=SkillPrivacySettings(include_authority=True),
            observability=SkillObservabilitySettings(emit_events=True),
        )
        override = UntrustedSkillSettings(
            surface=SkillSettingsSurface.TASK,
            bootstrap_enabled=False,
            authority_kinds=(SkillSourceAuthorityKind.WORKSPACE,),
            source_labels=("workspace-main",),
            skill_ids=("pdf",),
            read_limits=SkillReadLimits(max_bytes_per_read=1024),
            index_limits=SkillIndexLimits(max_skills=10),
            source_limits=SkillSourceLimits(max_sources=1),
            cursor_limits=SkillCursorLimits(max_active_cursors=2),
            privacy=SkillPrivacySettings(include_authority=False),
            observability=SkillObservabilitySettings(emit_events=False),
        )

        result = merge_skill_settings(trusted, override)
        settings = result.settings

        self.assertEqual(result.status, SkillStatus.OK)
        self.assertFalse(settings.bootstrap_enabled)
        self.assertEqual(settings.sources, (workspace_source,))
        self.assertEqual(
            settings.authority_kinds,
            (SkillSourceAuthorityKind.WORKSPACE,),
        )
        self.assertEqual(settings.allowed_skill_ids, ("pdf",))
        self.assertEqual(settings.read_limits.max_bytes_per_read, 1024)
        self.assertEqual(settings.index_limits.max_skills, 10)
        self.assertEqual(settings.source_limits.max_sources, 1)
        self.assertEqual(settings.cursor_limits.max_active_cursors, 2)
        self.assertFalse(settings.privacy.include_authority)
        self.assertFalse(settings.observability.emit_events)

        encoded = dumps(settings.as_model_dict(), sort_keys=True)
        self.assertNotIn("/Users/", encoded)
        self.assertIn("workspace", encoded)

    def test_merge_without_override_and_untrusted_disable(self) -> None:
        trusted = TrustedSkillSettings(
            sources=(
                SkillSourceConfig(
                    label="workspace-main",
                    authority=WorkspaceSkillSourceAuthority(),
                ),
            )
        )
        unchanged = merge_skill_settings(trusted)
        disabled = merge_skill_settings(
            trusted,
            UntrustedSkillSettings(
                surface=SkillSettingsSurface.FLOW,
                enabled=False,
            ),
        )

        self.assertIs(unchanged.settings, trusted)
        self.assertEqual(unchanged.as_model_dict()["status"], "ok")
        self.assertFalse(disabled.settings.enabled)
        self.assertEqual(disabled.status, SkillStatus.OK)

    def test_unrestricted_trusted_skill_ids_can_be_narrowed(self) -> None:
        trusted = TrustedSkillSettings()
        result = merge_skill_settings(
            trusted,
            UntrustedSkillSettings(
                surface=SkillSettingsSurface.AGENT,
                skill_ids=("pdf",),
            ),
        )

        self.assertEqual(result.status, SkillStatus.OK)
        self.assertEqual(result.settings.allowed_skill_ids, ("pdf",))

    def test_untrusted_widening_returns_policy_diagnostics(self) -> None:
        workspace_source = SkillSourceConfig(
            label="workspace-main",
            authority=WorkspaceSkillSourceAuthority(),
        )
        trusted = TrustedSkillSettings(
            enabled=False,
            bootstrap_enabled=False,
            authority_kinds=(SkillSourceAuthorityKind.WORKSPACE,),
            sources=(workspace_source,),
            allowed_skill_ids=("pdf",),
            read_limits=SkillReadLimits(max_bytes_per_read=1024),
            source_limits=SkillSourceLimits(max_sources=1),
            privacy=SkillPrivacySettings(include_authority=False),
            observability=SkillObservabilitySettings(
                include_byte_counts=False,
            ),
        )
        cases = (
            UntrustedSkillSettings(
                surface=SkillSettingsSurface.REQUEST,
                enabled=True,
            ),
            UntrustedSkillSettings(
                surface=SkillSettingsSurface.REQUEST,
                bootstrap_enabled=True,
            ),
            UntrustedSkillSettings(
                surface=SkillSettingsSurface.REQUEST,
                authority_kinds=(SkillSourceAuthorityKind.USER_LOCAL,),
            ),
            UntrustedSkillSettings(
                surface=SkillSettingsSurface.REQUEST,
                sources=(
                    SkillSourceConfig(
                        label="user-local",
                        authority=WorkspaceSkillSourceAuthority(),
                    ),
                ),
            ),
            UntrustedSkillSettings(
                surface=SkillSettingsSurface.REQUEST,
                read_limits=SkillReadLimits(max_bytes_per_read=2048),
            ),
            UntrustedSkillSettings(
                surface=SkillSettingsSurface.REQUEST,
                skill_ids=("docx",),
            ),
            UntrustedSkillSettings(
                surface=SkillSettingsSurface.REQUEST,
                index_limits=SkillIndexLimits(max_skills=512),
            ),
            UntrustedSkillSettings(
                surface=SkillSettingsSurface.REQUEST,
                source_limits=SkillSourceLimits(max_sources=2),
            ),
            UntrustedSkillSettings(
                surface=SkillSettingsSurface.REQUEST,
                cursor_limits=SkillCursorLimits(max_active_cursors=128),
            ),
            UntrustedSkillSettings(
                surface=SkillSettingsSurface.REQUEST,
                privacy=SkillPrivacySettings(include_authority=True),
            ),
            UntrustedSkillSettings(
                surface=SkillSettingsSurface.REQUEST,
                observability=SkillObservabilitySettings(
                    include_byte_counts=True,
                ),
            ),
        )

        for override in cases:
            with self.subTest(override=override):
                result = merge_skill_settings(trusted, override)

                self.assertEqual(result.status, SkillStatus.POLICY_DENIED)
                self.assertEqual(
                    result.diagnostics[0].code.value,
                    "skills.policy_denied",
                )

    def test_missing_source_label_is_structured_not_found(self) -> None:
        trusted = TrustedSkillSettings(
            sources=(
                SkillSourceConfig(
                    label="workspace-main",
                    authority=WorkspaceSkillSourceAuthority(),
                ),
            )
        )
        result = merge_skill_settings(
            trusted,
            UntrustedSkillSettings(
                surface=SkillSettingsSurface.AGENT,
                source_labels=("missing",),
            ),
        )

        self.assertEqual(result.status, SkillStatus.NOT_FOUND)
        self.assertEqual(result.diagnostics[0].candidates, ("missing",))

    def test_internal_identity_distinguishes_same_label_root_paths(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root_a = Path(directory) / "a"
            root_b = Path(directory) / "b"
            source_a = SkillSourceConfig(
                label="workspace-main",
                authority=WorkspaceSkillSourceAuthority(),
                root_path=root_a,
            )
            source_b = SkillSourceConfig(
                label="workspace-main",
                authority=WorkspaceSkillSourceAuthority(),
                root_path=root_b,
            )
            settings_a = TrustedSkillSettings(sources=(source_a,))
            settings_b = TrustedSkillSettings(sources=(source_b,))

            identity_a = trusted_skill_settings_identity_dict(settings_a)
            identity_b = trusted_skill_settings_identity_dict(settings_b)

        self.assertEqual(
            settings_a.as_model_dict()["sources"],
            settings_b.as_model_dict()["sources"],
        )
        self.assertNotEqual(identity_a, identity_b)
        encoded = dumps(identity_a, sort_keys=True)
        self.assertNotIn(str(root_a), encoded)
        self.assertNotIn(str(root_b), encoded)

    def test_source_identity_hashes_effective_package_paths(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            nested_source = SkillSourceConfig(
                label="workspace-main",
                authority=WorkspaceSkillSourceAuthority(),
                root_path=root,
                package_path="guides/pdf",
            )
            effective_source = SkillSourceConfig(
                label="workspace-main",
                authority=WorkspaceSkillSourceAuthority(),
                root_path=root / "guides" / "pdf",
            )
            windows_package_source = SkillSourceConfig(
                label="workspace-main",
                authority=WorkspaceSkillSourceAuthority(),
                package_path="guides\\pdf ",
            )
            posix_package_source = SkillSourceConfig(
                label="workspace-main",
                authority=WorkspaceSkillSourceAuthority(),
                package_path="guides/pdf",
            )

            nested_identity = trusted_skill_source_identity_dict(nested_source)
            effective_identity = trusted_skill_source_identity_dict(
                effective_source
            )
            windows_package_identity = trusted_skill_source_identity_dict(
                windows_package_source
            )
            posix_package_identity = trusted_skill_source_identity_dict(
                posix_package_source
            )

        self.assertEqual(nested_identity, effective_identity)
        self.assertEqual(windows_package_identity, posix_package_identity)
        self.assertIn("effective_root_sha256", nested_identity)
        self.assertIn("package_path_sha256", windows_package_identity)
        encoded = dumps(nested_identity, sort_keys=True)
        self.assertNotIn(str(root), encoded)

    def test_untrusted_config_dict_serializes_complete_manual_override(
        self,
    ) -> None:
        settings = UntrustedSkillSettings(
            surface=SkillSettingsSurface.WORKER_ENVELOPE,
            authority_kinds=(SkillSourceAuthorityKind.WORKSPACE,),
            read_limits=SkillReadLimits(
                max_bytes_per_read=1024,
                max_lines_per_read=10,
            ),
            index_limits=SkillIndexLimits(
                max_skills=8,
                max_resources_per_skill=3,
                max_indexed_bytes=4096,
            ),
            source_limits=SkillSourceLimits(
                max_sources=2,
                max_resources_per_source=5,
                max_source_depth=3,
                max_files_per_source=11,
                max_directory_entries_per_source=13,
            ),
            cursor_limits=SkillCursorLimits(
                max_active_cursors=4,
                max_cursor_age_seconds=30,
            ),
            privacy=SkillPrivacySettings(
                include_source_labels=False,
                include_authority=False,
                include_diagnostic_paths=False,
                redact_host_paths=True,
            ),
            observability=SkillObservabilitySettings(
                enabled=True,
                emit_events=False,
                include_diagnostics=False,
                include_byte_counts=True,
            ),
        )

        config = untrusted_skill_settings_config_dict(settings)

        self.assertEqual(config["authority_kinds"], ("workspace",))
        self.assertEqual(
            config["read_limits"],
            {
                "max_bytes_per_read": 1024,
                "max_lines_per_read": 10,
            },
        )
        self.assertEqual(
            config["index_limits"],
            {
                "max_skills": 8,
                "max_resources_per_skill": 3,
                "max_indexed_bytes": 4096,
            },
        )
        self.assertEqual(
            config["source_limits"],
            {
                "max_sources": 2,
                "max_resources_per_source": 5,
                "max_source_depth": 3,
                "max_files_per_source": 11,
                "max_directory_entries_per_source": 13,
            },
        )
        self.assertEqual(
            config["cursor_limits"],
            {
                "max_active_cursors": 4,
                "max_cursor_age_seconds": 30,
            },
        )
        self.assertEqual(
            config["privacy"],
            {
                "include_source_labels": False,
                "include_authority": False,
                "include_diagnostic_paths": False,
                "redact_host_paths": True,
            },
        )
        self.assertEqual(
            config["observability"],
            {
                "enabled": True,
                "emit_events": False,
                "include_diagnostics": False,
                "include_byte_counts": True,
            },
        )

    def test_settings_reject_non_ascii_and_invalid_logical_ids(self) -> None:
        invalid_builders = (
            lambda: TrustedSkillSettings(allowed_skill_ids=("café",)),
            lambda: TrustedSkillSettings(allowed_skill_ids=("Bad",)),
            lambda: TrustedSkillSettings(allowed_skill_ids=("bad label",)),
            lambda: UntrustedSkillSettings(
                surface=SkillSettingsSurface.AGENT,
                source_labels=("café",),
            ),
            lambda: UntrustedSkillSettings(
                surface=SkillSettingsSurface.AGENT,
                source_labels=("bad label",),
            ),
            lambda: UntrustedSkillSettings(
                surface=SkillSettingsSurface.AGENT,
                skill_ids=("café",),
            ),
            lambda: UntrustedSkillSettings(
                surface=SkillSettingsSurface.AGENT,
                skill_ids=("Bad",),
            ),
        )

        for invalid_builder in invalid_builders:
            with self.subTest(invalid_builder=invalid_builder):
                with self.assertRaises(AssertionError):
                    invalid_builder()

    def test_settings_reject_invalid_values_and_mutable_collections(
        self,
    ) -> None:
        source = SkillSourceConfig(
            label="workspace-main",
            authority=WorkspaceSkillSourceAuthority(),
        )
        invalid_builders = (
            lambda: SkillReadLimits(max_bytes_per_read=0),
            lambda: SkillIndexLimits(max_skills=0),
            lambda: SkillSourceLimits(max_sources=0),
            lambda: SkillCursorLimits(max_active_cursors=0),
            lambda: SkillPrivacySettings(
                redact_host_paths="yes"  # type: ignore[arg-type]
            ),
            lambda: SkillObservabilitySettings(
                emit_events="yes"  # type: ignore[arg-type]
            ),
            lambda: TrustedSkillSettings(
                authority_kinds=(SkillSourceAuthorityKind.USER_LOCAL,),
                sources=(source,),
            ),
            lambda: TrustedSkillSettings(
                sources=[source],  # type: ignore[arg-type]
            ),
            lambda: TrustedSkillSettings(
                authority_kinds=[  # type: ignore[arg-type]
                    SkillSourceAuthorityKind.WORKSPACE
                ],
            ),
            lambda: UntrustedSkillSettings(
                surface=SkillSettingsSurface.OPERATOR,
            ),
            lambda: UntrustedSkillSettings(
                surface="task",  # type: ignore[arg-type]
            ),
            lambda: UntrustedSkillSettings(
                surface=SkillSettingsSurface.TASK,
                source_labels=["workspace-main"],  # type: ignore[arg-type]
            ),
            lambda: UntrustedSkillSettings(
                surface=SkillSettingsSurface.TASK,
                bootstrap_enabled="off",  # type: ignore[arg-type]
            ),
        )

        for invalid_builder in invalid_builders:
            with self.subTest(invalid_builder=invalid_builder):
                with self.assertRaises(AssertionError):
                    invalid_builder()


if __name__ == "__main__":
    main()
