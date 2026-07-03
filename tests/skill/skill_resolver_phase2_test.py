from json import dumps
from os import stat_result
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest import IsolatedAsyncioTestCase, main

from avalan.skill import (
    BundledSkillSourceAuthority,
    PluginProvidedSkillSourceAuthority,
    PreinstalledRemoteSkillSourceAuthority,
    SkillAsyncFileSystem,
    SkillAuthorizedResource,
    SkillAuthorizedSourceRoot,
    SkillConfiguredSource,
    SkillDiagnosticCode,
    SkillDiagnosticInfo,
    SkillIndexLimits,
    SkillReadLimits,
    SkillResourceAuthorizationResult,
    SkillSourceAuthorityKind,
    SkillSourceConfig,
    SkillSourceFileSystem,
    SkillSourceLimits,
    SkillSourceManifestConfig,
    SkillSourceResolutionResult,
    SkillSourceResolver,
    SkillSourceRootConfig,
    SkillStatus,
    TrustedSkillSettings,
    UserLocalSkillSourceAuthority,
    WorkspaceSkillSourceAuthority,
    authorize_skill_resource,
    resolve_skill_sources,
)
from avalan.skill import resolver as resolver_module


class SkillResolverPhase2Test(IsolatedAsyncioTestCase):
    async def test_protocol_default_methods_return_none(self) -> None:
        file_system = _protocol_default_file_system()
        path = Path("SKILL.md")

        self.assertIsNone(
            await SkillSourceFileSystem.resolve_path(file_system, path)
        )
        self.assertIsNone(
            await SkillSourceFileSystem.stat_path(file_system, path)
        )
        self.assertIsNone(
            await SkillSourceFileSystem.lstat_path(file_system, path)
        )
        self.assertIsNone(
            await SkillSourceFileSystem.list_directory(file_system, path, 1)
        )
        self.assertIsNone(
            await SkillSourceFileSystem.read_bytes(file_system, path, 1)
        )

    async def test_resolves_workspace_bundled_and_user_local_sources(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            bundled = root / "bundled"
            user_local = root / "user-local"
            _write_text(workspace / "skills" / "SKILL.md", "workspace\n")
            _write_text(bundled / "pkg" / "SKILL.md", "bundled\n")
            _write_text(user_local / "SKILL.md", "user-local\n")
            sources = (
                SkillConfiguredSource(
                    label="workspace-main",
                    authority=WorkspaceSkillSourceAuthority(),
                    root_path=workspace,
                    package_path="skills",
                ),
                SkillConfiguredSource(
                    label="bundled-main",
                    authority=BundledSkillSourceAuthority(bundle_id="avalan"),
                    root_path=bundled,
                    package_path="pkg",
                ),
                SkillConfiguredSource(
                    label="user-local",
                    authority=UserLocalSkillSourceAuthority(
                        profile_id="default"
                    ),
                    root_path=user_local,
                ),
            )
            settings = TrustedSkillSettings(
                authority_kinds=(
                    SkillSourceAuthorityKind.WORKSPACE,
                    SkillSourceAuthorityKind.BUNDLED,
                    SkillSourceAuthorityKind.USER_LOCAL,
                ),
                sources=(
                    SkillSourceConfig(
                        label="workspace-main",
                        authority=WorkspaceSkillSourceAuthority(),
                    ),
                    SkillSourceConfig(
                        label="bundled-main",
                        authority=BundledSkillSourceAuthority(
                            bundle_id="avalan"
                        ),
                    ),
                    SkillSourceConfig(
                        label="user-local",
                        authority=UserLocalSkillSourceAuthority(
                            profile_id="default"
                        ),
                    ),
                ),
            )
            configured_model = sources[0].as_model_dict()

            result = await resolve_skill_sources(sources, settings=settings)

            self.assertEqual(result.status, SkillStatus.OK)
            self.assertEqual(configured_model["package"], "skills")
            self.assertNotIn(str(workspace), dumps(configured_model))
            self.assertEqual(
                tuple(source.label for source in result.sources),
                ("workspace-main", "bundled-main", "user-local"),
            )
            self.assertEqual(len(result.resources), 3)
            self.assertEqual(
                {resource.resource_id for resource in result.resources},
                {"SKILL.md"},
            )
            encoded = dumps(result.as_model_dict(), sort_keys=True)
            self.assertNotIn(str(root), encoded)
            self.assertNotIn("/private/", encoded)
            self.assertIn("source:workspace-main", encoded)

    async def test_missing_source_root_returns_structured_diagnostic(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"
            result = await resolve_skill_sources(
                (
                    SkillConfiguredSource(
                        label="workspace-main",
                        authority=WorkspaceSkillSourceAuthority(),
                        root_path=missing,
                    ),
                ),
            )

            self.assertEqual(result.status, SkillStatus.UNAVAILABLE)
            self.assertEqual(
                result.diagnostics[0].code.value,
                "skills.source_unavailable",
            )
            encoded = dumps(result.as_model_dict(), sort_keys=True)
            self.assertNotIn(str(missing.parent), encoded)
            self.assertIn("<host-path>/missing", encoded)

            file_root = Path(directory) / "not-directory"
            file_root.write_text("not a directory", encoding="utf-8")
            not_directory = await resolve_skill_sources(
                (
                    SkillConfiguredSource(
                        label="not-directory",
                        authority=WorkspaceSkillSourceAuthority(),
                        root_path=file_root,
                    ),
                ),
            )
            self.assertEqual(not_directory.status, SkillStatus.UNAVAILABLE)
            self.assertIn("not_directory", _reasons(not_directory))

    async def test_manifest_source_authorizes_only_direct_manifest(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "SKILL-pdf.md"
            _write_text(manifest_path, "manifest\n")
            _write_text(root / "notes.md", "notes\n")
            _write_text(root / "SKILL-other.md", "other\n")

            result = await resolve_skill_sources(
                (
                    SkillConfiguredSource(
                        label="workspace-main",
                        authority=WorkspaceSkillSourceAuthority(),
                        manifest_path=manifest_path,
                    ),
                )
            )

            self.assertEqual(result.status, SkillStatus.OK)
            self.assertEqual(len(result.sources), 1)
            source = result.sources[0]
            self.assertEqual(source.root, manifest_path.parent.resolve())
            self.assertEqual(source.identity_root, manifest_path.resolve())
            self.assertEqual(source.manifest_resource_id, "SKILL-pdf.md")
            self.assertEqual(
                tuple(resource.resource_id for resource in result.resources),
                ("SKILL-pdf.md",),
            )
            self.assertEqual(
                source.as_model_dict()["source_type"],
                "manifest",
            )
            self.assertNotIn(str(root), dumps(source.as_model_dict()))

            manifest = await authorize_skill_resource(
                source,
                "SKILL-pdf.md",
            )
            sibling = await authorize_skill_resource(source, "notes.md")

            self.assertEqual(manifest.status, SkillStatus.OK)
            self.assertEqual(sibling.status, SkillStatus.POLICY_DENIED)
            self.assertIn("manifest_source_resource", _reasons(sibling))

    async def test_manifest_source_model_dicts_and_authorization_diagnostics(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "SKILL.md"
            _write_text(manifest_path, "manifest\n")
            diagnostic = SkillDiagnosticInfo(
                code=SkillDiagnosticCode.POLICY_DENIED,
                status=SkillStatus.POLICY_DENIED,
                message="Denied.",
                path="resource.policy",
                hint="Use a safe resource.",
                details={"resource_id": "notes.md"},
            )
            source = SkillAuthorizedSourceRoot(
                label="workspace-main",
                authority=WorkspaceSkillSourceAuthority(),
                root=root,
                manifest_resource_id="SKILL.md",
                diagnostics=(diagnostic,),
            )
            resolver = SkillSourceResolver()

            configured_model = SkillConfiguredSource(
                label="workspace-main",
                authority=WorkspaceSkillSourceAuthority(),
                manifest_path=manifest_path,
            ).as_model_dict()
            manifest_model = SkillSourceManifestConfig(
                label="workspace-main",
                authority=WorkspaceSkillSourceAuthority(),
                manifest_path=manifest_path,
            ).as_model_dict()
            existing_diagnostic = await resolver.authorize_resource(
                source,
                "notes.md",
            )
            policy_diagnostic = (
                await resolver.authorize_manifest_declared_resource(
                    source,
                    "../secret.md",
                )
            )
            missing_diagnostic = (
                await resolver.authorize_manifest_declared_resource(
                    source,
                    "missing.md",
                )
            )

        self.assertEqual(configured_model["source_type"], "manifest")
        self.assertEqual(manifest_model["source_type"], "manifest")
        self.assertEqual(existing_diagnostic.diagnostics, (diagnostic,))
        self.assertIn("traversal", _reasons(policy_diagnostic))
        self.assertEqual(policy_diagnostic.status, SkillStatus.POLICY_DENIED)
        self.assertEqual(missing_diagnostic.status, SkillStatus.NOT_FOUND)

    async def test_manifest_source_rejects_non_manifest_paths(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            notes_path = root / "notes.md"
            _write_text(notes_path, "notes\n")
            skill_directory = root / "SKILL.md"
            skill_directory.mkdir()

            non_manifest = await resolve_skill_sources(
                (
                    SkillConfiguredSource(
                        label="workspace-main",
                        authority=WorkspaceSkillSourceAuthority(),
                        manifest_path=notes_path,
                    ),
                )
            )
            directory = await resolve_skill_sources(
                (
                    SkillConfiguredSource(
                        label="workspace-main",
                        authority=WorkspaceSkillSourceAuthority(),
                        manifest_path=skill_directory,
                    ),
                )
            )

            self.assertEqual(non_manifest.status, SkillStatus.UNAVAILABLE)
            self.assertIn("not_manifest", _reasons(non_manifest))
            self.assertEqual(directory.status, SkillStatus.UNAVAILABLE)
            self.assertIn("not_file", _reasons(directory))

    async def test_manifest_source_rejects_symlinked_manifest(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            notes_path = root / "notes.md"
            _write_text(notes_path, "notes\n")
            symlink_path = root / "SKILL.md"
            try:
                symlink_path.symlink_to(notes_path)
            except OSError as error:
                self.skipTest(f"symlinks unavailable: {error}")

            result = await resolve_skill_sources(
                (
                    SkillConfiguredSource(
                        label="workspace-main",
                        authority=WorkspaceSkillSourceAuthority(),
                        manifest_path=symlink_path,
                    ),
                )
            )

            self.assertEqual(result.status, SkillStatus.UNAVAILABLE)
            self.assertIn("manifest_symlink", _reasons(result))
            self.assertEqual(result.resources, ())

    async def test_manifest_source_rejects_outside_symlinked_manifest(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / "outside"
            target_path = outside / "SKILL.md"
            _write_text(target_path, "manifest\n")
            symlink_path = root / "trusted" / "SKILL.md"
            symlink_path.parent.mkdir()
            try:
                symlink_path.symlink_to(target_path)
            except OSError as error:
                self.skipTest(f"symlinks unavailable: {error}")

            result = await resolve_skill_sources(
                (
                    SkillConfiguredSource(
                        label="workspace-main",
                        authority=WorkspaceSkillSourceAuthority(),
                        manifest_path=symlink_path,
                    ),
                )
            )

            self.assertEqual(result.status, SkillStatus.UNAVAILABLE)
            self.assertIn("manifest_symlink", _reasons(result))
            self.assertEqual(result.resources, ())

    async def test_manifest_source_reports_resolution_defensive_diagnostics(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            valid_path = root / "valid" / "SKILL.md"
            mismatch_config = root / "mismatch" / "SKILL.md"
            mismatch_target = root / "mismatch" / "SKILL-other.md"
            hidden_config = root / "hidden" / "SKILL.md"
            hidden_target = root / ".hidden" / "SKILL.md"
            binary_path = root / "binary" / "SKILL.md"
            _write_text(valid_path, "valid\n")
            _write_text(mismatch_config, "configured\n")
            _write_text(mismatch_target, "target\n")
            _write_text(hidden_config, "configured\n")
            _write_text(hidden_target, "target\n")
            binary_path.parent.mkdir(parents=True)
            binary_path.write_bytes(b"bad\x00content")

            nul_label = await resolve_skill_sources(
                (
                    SkillConfiguredSource(
                        label="bad\x00label",
                        authority=WorkspaceSkillSourceAuthority(),
                        manifest_path=valid_path,
                    ),
                )
            )
            missing = await resolve_skill_sources(
                (
                    SkillConfiguredSource(
                        label="workspace-main",
                        authority=WorkspaceSkillSourceAuthority(),
                        manifest_path=root / "missing" / "SKILL.md",
                    ),
                )
            )
            mismatch = await resolve_skill_sources(
                (
                    SkillConfiguredSource(
                        label="workspace-main",
                        authority=WorkspaceSkillSourceAuthority(),
                        manifest_path=mismatch_config,
                    ),
                ),
                file_system=RedirectResolveFileSystem(
                    mismatch_config,
                    mismatch_target,
                ),
            )
            hidden = await resolve_skill_sources(
                (
                    SkillConfiguredSource(
                        label="workspace-main",
                        authority=WorkspaceSkillSourceAuthority(),
                        manifest_path=hidden_config,
                    ),
                ),
                file_system=RedirectResolveFileSystem(
                    hidden_config,
                    hidden_target,
                ),
            )
            binary = await resolve_skill_sources(
                (
                    SkillConfiguredSource(
                        label="workspace-main",
                        authority=WorkspaceSkillSourceAuthority(),
                        manifest_path=binary_path,
                    ),
                )
            )

        self.assertIn("nul_byte", _reasons(nul_label))
        self.assertIn("unavailable", _reasons(missing))
        self.assertIn("manifest_resource_mismatch", _reasons(mismatch))
        self.assertIn("hidden_path", _reasons(hidden))
        self.assertIn("nul_byte", _reasons(binary))
        self.assertIsNone(
            resolver_module._manifest_resource_id_from_path("SKILL-secret.md")
        )

    async def test_duplicate_sanitized_source_labels_are_blocked(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()

            result = await resolve_skill_sources(
                (
                    SkillConfiguredSource(
                        label="workspace-main",
                        authority=WorkspaceSkillSourceAuthority(),
                        root_path=first,
                    ),
                    SkillConfiguredSource(
                        label="Workspace Main",
                        authority=WorkspaceSkillSourceAuthority(),
                        root_path=second,
                    ),
                ),
            )

            self.assertEqual(result.status, SkillStatus.BLOCKED)
            self.assertEqual(
                result.diagnostics[0].code.value,
                "skills.duplicate_id",
            )
            self.assertEqual(
                result.diagnostics[0].candidates,
                ("workspace-main",),
            )

    async def test_source_count_is_bounded_before_filesystem_access(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()

            result = await resolve_skill_sources(
                (
                    SkillConfiguredSource(
                        label="first",
                        authority=WorkspaceSkillSourceAuthority(),
                        root_path=first,
                    ),
                    SkillConfiguredSource(
                        label="second",
                        authority=WorkspaceSkillSourceAuthority(),
                        root_path=second,
                    ),
                ),
                settings=TrustedSkillSettings(
                    source_limits=SkillSourceLimits(max_sources=1)
                ),
            )

            self.assertEqual(result.status, SkillStatus.BLOCKED)
            self.assertEqual(result.diagnostics[0].path, "source.count")

    async def test_disabled_configured_source_is_skipped_without_access(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"

            result = await resolve_skill_sources(
                (
                    SkillConfiguredSource(
                        label="disabled",
                        authority=WorkspaceSkillSourceAuthority(),
                        root_path=missing,
                        enabled=False,
                    ),
                )
            )

            self.assertEqual(result.status, SkillStatus.EMPTY)
            self.assertEqual(result.sources, ())
            self.assertEqual(result.diagnostics, ())

    async def test_reports_configured_resolution_bounds(self) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory)

            file_root = base / "file-count"
            _write(file_root, "a.md", "a")
            _write(file_root, "b.md", "b")
            file_count = await resolve_skill_sources(
                (_config("file-count", file_root),),
                source_limits=SkillSourceLimits(max_files_per_source=1),
            )
            self.assertIn("file_count", _reasons(file_count))

            resource_root = base / "resource-count"
            _write(resource_root, "a.md", "a")
            _write(resource_root, "b.md", "b")
            resource_count = await resolve_skill_sources(
                (_config("resource-count", resource_root),),
                source_limits=SkillSourceLimits(max_resources_per_source=1),
            )
            self.assertIn("resource_count", _reasons(resource_count))

            indexed_root = base / "indexed-bytes"
            _write(indexed_root, "a.md", "1234")
            _write(indexed_root, "b.md", "5678")
            indexed_bytes = await resolve_skill_sources(
                (_config("indexed-bytes", indexed_root),),
                index_limits=SkillIndexLimits(max_indexed_bytes=5),
            )
            self.assertIn("indexed_bytes", _reasons(indexed_bytes))

            byte_root = base / "byte-count"
            _write(byte_root, "a.md", "12345")
            byte_count = await resolve_skill_sources(
                (_config("byte-count", byte_root),),
                read_limits=SkillReadLimits(max_bytes_per_read=4),
            )
            self.assertIn("per_resource_bytes", _reasons(byte_count))

            line_root = base / "line-count"
            _write(line_root, "a.md", "one\ntwo\nthree\n")
            line_count = await resolve_skill_sources(
                (_config("line-count", line_root),),
                read_limits=SkillReadLimits(max_lines_per_read=2),
            )
            self.assertIn("line_count", _reasons(line_count))

            depth_root = base / "depth"
            _write(depth_root, "a/b.md", "deep")
            depth = await resolve_skill_sources(
                (_config("depth", depth_root),),
                source_limits=SkillSourceLimits(max_source_depth=1),
            )
            self.assertIn("source_depth", _reasons(depth))

            directory_depth_root = base / "directory-depth"
            _write(directory_depth_root, "a/b/c.md", "deep")
            directory_depth = await resolve_skill_sources(
                (_config("directory-depth", directory_depth_root),),
                source_limits=SkillSourceLimits(max_source_depth=1),
            )
            self.assertIn("source_depth", _reasons(directory_depth))

            traversal_root = base / "traversal-work"
            _write(traversal_root, "a.md", "a")
            _write(traversal_root, "b.md", "b")
            traversal = await resolve_skill_sources(
                (_config("traversal-work", traversal_root),),
                source_limits=SkillSourceLimits(
                    max_directory_entries_per_source=1
                ),
            )
            self.assertIn("directory_traversal", _reasons(traversal))

            bounded_real_root = base / "bounded-real"
            _write(bounded_real_root, "a.md", "a")
            _write(bounded_real_root, "b.md", "b")
            _write(bounded_real_root, "c.md", "c")
            _write(bounded_real_root, "d.md", "d")
            bounded_real = await resolve_skill_sources(
                (_config("bounded-real", bounded_real_root),),
                source_limits=SkillSourceLimits(
                    max_directory_entries_per_source=2
                ),
            )
            self.assertIn("directory_traversal", _reasons(bounded_real))

    async def test_tight_directory_entry_limit_allows_empty_source(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "empty"
            root.mkdir()

            result = await resolve_skill_sources(
                (_config("empty", root),),
                source_limits=SkillSourceLimits(
                    max_directory_entries_per_source=1
                ),
            )

            self.assertEqual(result.status, SkillStatus.OK)
            self.assertEqual(result.resources, ())
            self.assertNotIn("directory_traversal", _reasons(result))

    async def test_tight_directory_entry_limit_allows_single_entry_source(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "single-entry"
            _write(root, "SKILL.md", "single\n")

            result = await resolve_skill_sources(
                (_config("single-entry", root),),
                source_limits=SkillSourceLimits(
                    max_directory_entries_per_source=1
                ),
            )

            self.assertEqual(result.status, SkillStatus.OK)
            self.assertEqual(
                tuple(resource.resource_id for resource in result.resources),
                ("SKILL.md",),
            )
            self.assertNotIn("directory_traversal", _reasons(result))

    async def test_bounded_directory_listing_stops_before_exhaustion(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            file_system = HugeDirectoryFileSystem()

            result = await resolve_skill_sources(
                (_config("huge", root),),
                source_limits=SkillSourceLimits(
                    max_directory_entries_per_source=2
                ),
                file_system=file_system,
            )

            self.assertIn("directory_traversal", _reasons(result))
            self.assertEqual(file_system.requested_limit, 3)
            self.assertEqual(file_system.returned_count, 3)
            self.assertEqual(file_system.lstat_count, 0)

    async def test_file_bound_counts_invalid_candidates_before_reading_next(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "file-bound"
            root.mkdir()
            (root / "a-binary.md").write_bytes(b"\xff")
            _write_text(root / "b-valid.md", "valid")
            file_system = CountingFileSystem()

            result = await resolve_skill_sources(
                (_config("file-bound", root),),
                source_limits=SkillSourceLimits(max_files_per_source=1),
                file_system=file_system,
            )

            self.assertIn("non_utf8_resource", _reasons(result))
            self.assertIn("file_count", _reasons(result))
            self.assertEqual(file_system.read_names, ["a-binary.md"])
            self.assertNotIn("b-valid.md", file_system.resolve_names)
            self.assertNotIn("b-valid.md", file_system.stat_names)

    async def test_resource_bound_stops_before_authorizing_next_candidate(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "resource-bound"
            _write_text(root / "a.md", "a")
            _write_text(root / "b.md", "b")
            file_system = CountingFileSystem()

            result = await resolve_skill_sources(
                (_config("resource-bound", root),),
                source_limits=SkillSourceLimits(max_resources_per_source=1),
                file_system=file_system,
            )

            self.assertIn("resource_count", _reasons(result))
            self.assertEqual(
                tuple(resource.resource_id for resource in result.resources),
                ("a.md",),
            )
            self.assertEqual(file_system.read_names, ["a.md"])
            self.assertNotIn("b.md", file_system.resolve_names)
            self.assertNotIn("b.md", file_system.stat_names)

    async def test_untrusted_label_or_authority_returns_policy_diagnostic(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            root.mkdir(exist_ok=True)
            settings = TrustedSkillSettings(
                authority_kinds=(SkillSourceAuthorityKind.WORKSPACE,),
                sources=(
                    SkillSourceConfig(
                        label="workspace-main",
                        authority=WorkspaceSkillSourceAuthority(),
                    ),
                ),
            )

            missing_label = await resolve_skill_sources(
                (
                    SkillConfiguredSource(
                        label="other",
                        authority=WorkspaceSkillSourceAuthority(),
                        root_path=root,
                    ),
                ),
                settings=settings,
            )
            bad_authority = await resolve_skill_sources(
                (
                    SkillConfiguredSource(
                        label="workspace-main",
                        authority=BundledSkillSourceAuthority(),
                        root_path=root,
                    ),
                ),
                settings=settings,
            )

            self.assertEqual(missing_label.status, SkillStatus.POLICY_DENIED)
            self.assertEqual(bad_authority.status, SkillStatus.POLICY_DENIED)
            self.assertEqual(missing_label.diagnostics[0].path, "source.label")
            self.assertEqual(
                bad_authority.diagnostics[0].path,
                "source.authority",
            )

    async def test_explicit_empty_trusted_sources_deny_all_configs(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            root.mkdir(exist_ok=True)

            result = await resolve_skill_sources(
                (_config("workspace-main", root),),
                settings=TrustedSkillSettings(
                    sources=(),
                    sources_explicit=True,
                ),
                file_system=FailOnAccessFileSystem(),
            )

        self.assertEqual(result.status, SkillStatus.POLICY_DENIED)
        self.assertEqual(result.diagnostics[0].path, "source.label")
        self.assertIn("untrusted_label", _reasons(result))

    async def test_resolved_source_root_must_match_trusted_identity(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory)
            trusted_root = base / "trusted"
            configured_root = base / "configured"
            trusted_root.mkdir()
            configured_root.mkdir()

            result = await resolve_skill_sources(
                (
                    SkillConfiguredSource(
                        label="workspace-main",
                        authority=WorkspaceSkillSourceAuthority(),
                        root_path=configured_root,
                    ),
                ),
                settings=TrustedSkillSettings(
                    sources=(
                        SkillSourceConfig(
                            label="workspace-main",
                            authority=WorkspaceSkillSourceAuthority(),
                            root_path=trusted_root,
                        ),
                    ),
                ),
            )

        self.assertEqual(result.status, SkillStatus.POLICY_DENIED)
        self.assertEqual(result.diagnostics[0].path, "source.identity")
        self.assertIn("untrusted_source_identity", _reasons(result))
        self.assertNotIn(str(trusted_root), str(result.as_model_dict()))
        self.assertNotIn(str(configured_root), str(result.as_model_dict()))

    async def test_manifest_only_trust_rejects_directory_source(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory)
            trusted_manifest = base / "trusted" / "SKILL.md"
            configured_root = base / "configured"
            _write_text(trusted_manifest, "trusted\n")
            _write_text(configured_root / "SKILL.md", "configured\n")
            _write_text(configured_root / "notes.md", "notes\n")
            denied_file_system = CountingFileSystem()

            denied = await resolve_skill_sources(
                (
                    SkillConfiguredSource(
                        label="pdf",
                        authority=WorkspaceSkillSourceAuthority(),
                        root_path=configured_root,
                    ),
                ),
                settings=TrustedSkillSettings(
                    sources=(
                        SkillSourceConfig(
                            label="pdf",
                            authority=WorkspaceSkillSourceAuthority(),
                            manifest_path=trusted_manifest,
                        ),
                    ),
                ),
                file_system=denied_file_system,
            )
            label_only = await resolve_skill_sources(
                (
                    SkillConfiguredSource(
                        label="pdf",
                        authority=WorkspaceSkillSourceAuthority(),
                        root_path=configured_root,
                    ),
                ),
                settings=TrustedSkillSettings(
                    sources=(
                        SkillSourceConfig(
                            label="pdf",
                            authority=WorkspaceSkillSourceAuthority(),
                        ),
                    ),
                ),
            )
            exact_root = await resolve_skill_sources(
                (
                    SkillConfiguredSource(
                        label="pdf",
                        authority=WorkspaceSkillSourceAuthority(),
                        root_path=configured_root,
                    ),
                ),
                settings=TrustedSkillSettings(
                    sources=(
                        SkillSourceConfig(
                            label="pdf",
                            authority=WorkspaceSkillSourceAuthority(),
                            root_path=configured_root,
                        ),
                    ),
                ),
            )
            manifest_allowed = await resolve_skill_sources(
                (
                    SkillConfiguredSource(
                        label="pdf",
                        authority=WorkspaceSkillSourceAuthority(),
                        manifest_path=trusted_manifest,
                    ),
                ),
                settings=TrustedSkillSettings(
                    sources=(
                        SkillSourceConfig(
                            label="pdf",
                            authority=WorkspaceSkillSourceAuthority(),
                            manifest_path=trusted_manifest,
                        ),
                    ),
                ),
            )

        self.assertEqual(denied.status, SkillStatus.POLICY_DENIED)
        self.assertEqual(denied.diagnostics[0].path, "source.identity")
        self.assertIn("untrusted_source_identity", _reasons(denied))
        self.assertEqual(denied.resources, ())
        self.assertEqual(denied_file_system.read_names, [])
        self.assertEqual(label_only.status, SkillStatus.OK)
        self.assertEqual(exact_root.status, SkillStatus.OK)
        self.assertEqual(manifest_allowed.status, SkillStatus.OK)

    async def test_resolved_manifest_source_must_match_trusted_identity(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory)
            trusted_path = base / "trusted" / "SKILL.md"
            configured_path = base / "configured" / "SKILL.md"
            _write_text(trusted_path, "trusted\n")
            _write_text(configured_path, "configured\n")

            allowed = await resolve_skill_sources(
                (
                    SkillConfiguredSource(
                        label="workspace-main",
                        authority=WorkspaceSkillSourceAuthority(),
                        manifest_path=trusted_path,
                    ),
                ),
                settings=TrustedSkillSettings(
                    sources=(
                        SkillSourceConfig(
                            label="workspace-main",
                            authority=WorkspaceSkillSourceAuthority(),
                            manifest_path=trusted_path,
                        ),
                    ),
                ),
            )
            denied_file_system = CountingFileSystem()
            denied = await resolve_skill_sources(
                (
                    SkillConfiguredSource(
                        label="workspace-main",
                        authority=WorkspaceSkillSourceAuthority(),
                        manifest_path=configured_path,
                    ),
                ),
                settings=TrustedSkillSettings(
                    sources=(
                        SkillSourceConfig(
                            label="workspace-main",
                            authority=WorkspaceSkillSourceAuthority(),
                            manifest_path=trusted_path,
                        ),
                    ),
                ),
                file_system=denied_file_system,
            )
            label_only = await resolve_skill_sources(
                (
                    SkillConfiguredSource(
                        label="workspace-main",
                        authority=WorkspaceSkillSourceAuthority(),
                        manifest_path=configured_path,
                    ),
                ),
                settings=TrustedSkillSettings(
                    sources=(
                        SkillSourceConfig(
                            label="workspace-main",
                            authority=WorkspaceSkillSourceAuthority(),
                        ),
                    ),
                ),
            )

        self.assertEqual(allowed.status, SkillStatus.OK)
        self.assertEqual(denied.status, SkillStatus.POLICY_DENIED)
        self.assertIn("untrusted_source_identity", _reasons(denied))
        self.assertEqual(denied_file_system.read_names, [])
        self.assertEqual(label_only.status, SkillStatus.OK)

    async def test_trusted_manifest_identity_helper_handles_label_only(
        self,
    ) -> None:
        config = SkillSourceManifestConfig(
            label="workspace-main",
            authority=WorkspaceSkillSourceAuthority(),
            manifest_path="/tmp/SKILL.md",
        )

        missing_label = resolver_module._trusted_resolved_source_diagnostic(
            config,
            TrustedSkillSettings(
                sources=(
                    SkillSourceConfig(
                        label="other",
                        authority=WorkspaceSkillSourceAuthority(),
                        manifest_path="/tmp/SKILL.md",
                    ),
                )
            ),
            Path("/tmp"),
            manifest_path=Path("/tmp/SKILL.md"),
        )
        label_only = resolver_module._trusted_resolved_source_diagnostic(
            config,
            TrustedSkillSettings(
                sources=(
                    SkillSourceConfig(
                        label="workspace-main",
                        authority=WorkspaceSkillSourceAuthority(),
                    ),
                )
            ),
            Path("/tmp"),
            manifest_path=Path("/tmp/SKILL.md"),
        )

        self.assertIsNone(missing_label)
        self.assertIsNone(label_only)

    async def test_trusted_source_authority_identity_must_match(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cases = (
                (
                    WorkspaceSkillSourceAuthority(workspace_id="trusted"),
                    WorkspaceSkillSourceAuthority(workspace_id="other"),
                ),
                (
                    BundledSkillSourceAuthority(bundle_id="trusted"),
                    BundledSkillSourceAuthority(bundle_id="other"),
                ),
                (
                    UserLocalSkillSourceAuthority(profile_id="trusted"),
                    UserLocalSkillSourceAuthority(profile_id="other"),
                ),
                (
                    PluginProvidedSkillSourceAuthority(plugin_id="trusted"),
                    PluginProvidedSkillSourceAuthority(plugin_id="other"),
                ),
                (
                    PreinstalledRemoteSkillSourceAuthority(
                        registry_id="trusted"
                    ),
                    PreinstalledRemoteSkillSourceAuthority(
                        registry_id="other"
                    ),
                ),
            )

            for trusted_authority, configured_authority in cases:
                with self.subTest(authority=trusted_authority.kind.value):
                    result = await resolve_skill_sources(
                        (
                            SkillConfiguredSource(
                                label="source",
                                authority=configured_authority,
                                root_path=root,
                            ),
                        ),
                        settings=TrustedSkillSettings(
                            authority_kinds=(trusted_authority.kind,),
                            sources=(
                                SkillSourceConfig(
                                    label="source",
                                    authority=trusted_authority,
                                ),
                            ),
                        ),
                    )

                    self.assertEqual(result.status, SkillStatus.POLICY_DENIED)
                    self.assertIn("untrusted_authority", _reasons(result))

    async def test_trusted_settings_disabled_and_status_diagnostics(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = _write(Path(directory) / "source", "SKILL.md", "ok")
            disabled = await resolve_skill_sources(
                (_config("source", root),),
                settings=TrustedSkillSettings(
                    sources=(
                        SkillSourceConfig(
                            label="source",
                            authority=WorkspaceSkillSourceAuthority(),
                            enabled=False,
                            status=SkillStatus.DISABLED,
                        ),
                    ),
                ),
            )
            status_diagnostic = SkillDiagnosticInfo(
                code=SkillDiagnosticCode.SOURCE_UNAVAILABLE,
                status=SkillStatus.UNAVAILABLE,
                message="Source unavailable.",
                path="source.availability",
                hint="Do not use it.",
            )
            unavailable = await resolve_skill_sources(
                (_config("source", root),),
                settings=TrustedSkillSettings(
                    sources=(
                        SkillSourceConfig(
                            label="source",
                            authority=WorkspaceSkillSourceAuthority(),
                            status=SkillStatus.UNAVAILABLE,
                            diagnostics=(status_diagnostic,),
                        ),
                    ),
                ),
            )
            status_without_diagnostic = await resolve_skill_sources(
                (_config("source", root),),
                settings=TrustedSkillSettings(
                    sources=(
                        SkillSourceConfig(
                            label="source",
                            authority=WorkspaceSkillSourceAuthority(),
                            status=SkillStatus.UNAVAILABLE,
                        ),
                    ),
                ),
            )
            authority_mismatch = await resolve_skill_sources(
                (_config("source", root),),
                settings=TrustedSkillSettings(
                    authority_kinds=(
                        SkillSourceAuthorityKind.WORKSPACE,
                        SkillSourceAuthorityKind.BUNDLED,
                    ),
                    sources=(
                        SkillSourceConfig(
                            label="source",
                            authority=BundledSkillSourceAuthority(),
                        ),
                    ),
                ),
            )
            unrestricted = await resolve_skill_sources(
                (_config("source", root),),
                settings=TrustedSkillSettings(),
            )
            globally_disabled = await resolve_skill_sources(
                (_config("source", root),),
                settings=TrustedSkillSettings(enabled=False),
                file_system=FailOnAccessFileSystem(),
            )

            self.assertEqual(disabled.status, SkillStatus.DISABLED)
            self.assertIs(unavailable.diagnostics[0], status_diagnostic)
            self.assertIn(
                "trusted_source_status",
                _reasons(status_without_diagnostic),
            )
            self.assertIn("untrusted_authority", _reasons(authority_mismatch))
            self.assertEqual(unrestricted.status, SkillStatus.OK)
            self.assertEqual(globally_disabled.status, SkillStatus.DISABLED)
            self.assertIn("settings_disabled", _reasons(globally_disabled))

    async def test_package_roots_are_authorized_before_traversal(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "source"
            _write(root, "pkg/SKILL.md", "ok")
            _write(root, "not-package.md", "no")
            outside = base / "outside"
            outside.mkdir()

            package = await resolve_skill_sources(
                (
                    SkillConfiguredSource(
                        label="package",
                        authority=WorkspaceSkillSourceAuthority(),
                        root_path=root,
                        package_path="pkg",
                    ),
                )
            )
            missing = await resolve_skill_sources(
                (
                    SkillConfiguredSource(
                        label="missing-package",
                        authority=WorkspaceSkillSourceAuthority(),
                        root_path=root,
                        package_path="missing",
                    ),
                )
            )
            not_directory = await resolve_skill_sources(
                (
                    SkillConfiguredSource(
                        label="file-package",
                        authority=WorkspaceSkillSourceAuthority(),
                        root_path=root,
                        package_path="not-package.md",
                    ),
                )
            )
            try:
                (root / "escape").symlink_to(outside, target_is_directory=True)
            except OSError as error:
                self.skipTest(f"directory symlinks unavailable: {error}")
            escaped = await resolve_skill_sources(
                (
                    SkillConfiguredSource(
                        label="escape-package",
                        authority=WorkspaceSkillSourceAuthority(),
                        root_path=root,
                        package_path="escape",
                    ),
                )
            )

            self.assertEqual(package.status, SkillStatus.OK)
            self.assertEqual(package.resources[0].resource_id, "SKILL.md")
            self.assertIn("unavailable_package", _reasons(missing))
            self.assertIn("package_not_directory", _reasons(not_directory))
            self.assertIn("package_escape", _reasons(escaped))

    async def test_resolved_roots_recheck_hidden_and_sensitive_policy(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory)
            hidden_root = base / ".hidden"
            _write(hidden_root, "SKILL.md", "hidden\n")
            root_link = base / "visible-root"
            try:
                root_link.symlink_to(
                    hidden_root,
                    target_is_directory=True,
                )
            except OSError as error:
                self.skipTest(f"directory symlinks unavailable: {error}")

            package_parent = base / "source"
            sensitive_package = package_parent / "secrets"
            _write(sensitive_package, "SKILL.md", "secret\n")
            package_link = package_parent / "pkg"
            package_link.symlink_to(
                sensitive_package,
                target_is_directory=True,
            )

            hidden_result = await resolve_skill_sources(
                (
                    SkillConfiguredSource(
                        label="hidden-root",
                        authority=WorkspaceSkillSourceAuthority(),
                        root_path=root_link,
                    ),
                )
            )
            sensitive_package_result = await resolve_skill_sources(
                (
                    SkillConfiguredSource(
                        label="sensitive-package",
                        authority=WorkspaceSkillSourceAuthority(),
                        root_path=package_parent,
                        package_path="pkg",
                    ),
                )
            )

        self.assertIn("hidden_path", _reasons(hidden_result))
        self.assertIn("sensitive_path", _reasons(sensitive_package_result))

    async def test_content_and_filesystem_failures_are_structured(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory)
            nul_root = base / "nul"
            nul_root.mkdir()
            (nul_root / "SKILL.md").write_bytes(b"bad\x00content")
            binary_root = base / "binary"
            binary_root.mkdir()
            (binary_root / "SKILL.md").write_bytes(b"\xff")
            empty_root = base / "empty"
            empty_root.mkdir()
            (empty_root / "SKILL.md").write_bytes(b"")

            nul = await resolve_skill_sources((_config("nul", nul_root),))
            binary = await resolve_skill_sources(
                (_config("binary", binary_root),)
            )
            empty = await resolve_skill_sources(
                (_config("empty", empty_root),)
            )
            outside_root = _write(base / "outside", "outside.md", "outside")
            outside_entry = await resolve_skill_sources(
                (_config("outside-entry", empty_root),),
                file_system=OutsideEntryFileSystem(
                    outside_root / "outside.md"
                ),
            )
            list_failure = await resolve_skill_sources(
                (_config("list-failure", empty_root),),
                file_system=ListFailureFileSystem(),
            )
            lstat_failure = await resolve_skill_sources(
                (_config("lstat-failure", empty_root),),
                file_system=LstatFailureFileSystem(),
            )

            self.assertIn("nul_byte", _reasons(nul))
            self.assertIn("non_utf8_resource", _reasons(binary))
            self.assertEqual(empty.resources[0].line_count, 0)
            self.assertIn("path_escape", _reasons(outside_entry))
            self.assertIn("unavailable", _reasons(list_failure))
            self.assertIn(
                "skills.resource_missing",
                tuple(
                    diagnostic.code.value
                    for source in lstat_failure.sources
                    for diagnostic in source.diagnostics
                ),
            )

    async def test_sensitive_resource_filenames_are_denied(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            _write(root, "token.md", "secret")
            _write(root, "secret.txt", "secret")
            _write(root, "credentials.json", "{}")
            _write(root, "tokenizer.md", "ok")

            result = await resolve_skill_sources((_config("source", root),))

            self.assertIn("sensitive_path", _reasons(result))
            self.assertEqual(
                tuple(resource.resource_id for resource in result.resources),
                ("tokenizer.md",),
            )

    async def test_authorize_resource_handles_path_races(self) -> None:
        with TemporaryDirectory() as directory:
            root = _write(Path(directory) / "source", "SKILL.md", "ok")
            _write(root, "deep/target.md", "deep")
            result = await resolve_skill_sources((_config("source", root),))
            source = result.sources[0]

            escape = await authorize_skill_resource(
                source,
                "SKILL.md",
                file_system=ResolveEscapeFileSystem(root),
            )
            stat_failure = await authorize_skill_resource(
                source,
                "SKILL.md",
                file_system=StatFailureFileSystem(),
            )
            read_failure = await authorize_skill_resource(
                source,
                "SKILL.md",
                file_system=ReadFailureFileSystem(),
            )
            read_growth = await authorize_skill_resource(
                source,
                "SKILL.md",
                read_limits=SkillReadLimits(max_bytes_per_read=8),
                file_system=ReadGrowthFileSystem(),
            )
            resolve_failure = await authorize_skill_resource(
                source,
                "SKILL.md",
                file_system=ResolveFailureFileSystem(),
            )
            depth_failure = await authorize_skill_resource(
                source,
                "deep/target.md",
                source_limits=SkillSourceLimits(max_source_depth=1),
            )
            hidden_target = await authorize_skill_resource(
                source,
                "SKILL.md",
                file_system=HiddenTargetFileSystem(source.root),
            )

            self.assertIn("path_escape", _reasons(escape))
            self.assertIn("unavailable", _reasons(stat_failure))
            self.assertIn("unavailable", _reasons(read_failure))
            self.assertIn("per_resource_bytes", _reasons(read_growth))
            self.assertIn("unavailable", _reasons(resolve_failure))
            self.assertIn("source_depth", _reasons(depth_failure))
            self.assertIn("hidden_path", _reasons(hidden_target))

    async def test_symlink_target_depth_is_enforced(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            target = root / "a" / "b" / "target.md"
            _write_text(target, "deep")
            try:
                (root / "link.md").symlink_to(target)
            except OSError as error:
                self.skipTest(f"symlinks unavailable: {error}")

            result = await resolve_skill_sources(
                (_config("source", root),),
                source_limits=SkillSourceLimits(max_source_depth=1),
            )

            self.assertIn("source_depth", _reasons(result))
            self.assertNotIn(
                "link.md",
                tuple(resource.resource_id for resource in result.resources),
            )

    def test_phase2_result_entities_serialize_edge_states(self) -> None:
        resource = SkillAuthorizedResource(
            source_label="source",
            resource_id="SKILL.md",
            path=Path("/tmp/SKILL.md"),
            size_bytes=1,
            line_count=1,
        )
        with_resource = SkillResourceAuthorizationResult(
            source_label="source",
            resource=resource,
        )
        root_config = SkillSourceRootConfig(
            label="source",
            authority=WorkspaceSkillSourceAuthority(),
            root=Path("/tmp/source"),
        )
        empty_authorization = SkillResourceAuthorizationResult(
            source_label="source"
        )
        empty_resolution = SkillSourceResolutionResult()

        self.assertEqual(with_resource.status, SkillStatus.OK)
        self.assertIn("resource", with_resource.as_model_dict())
        self.assertEqual(root_config.as_model_dict()["package_path"], ".")
        self.assertEqual(empty_authorization.status, SkillStatus.NOT_FOUND)
        self.assertEqual(empty_resolution.status, SkillStatus.EMPTY)


class ListFailureFileSystem(SkillAsyncFileSystem):
    async def list_directory(
        self,
        path: Path,
        limit: int,
    ) -> tuple[Path, ...]:
        raise OSError("list failed")


class FailOnAccessFileSystem(SkillAsyncFileSystem):
    async def resolve_path(self, path: Path) -> Path:
        raise AssertionError("filesystem should not be reached")

    async def stat_path(self, path: Path) -> stat_result:
        raise AssertionError("filesystem should not be reached")

    async def lstat_path(self, path: Path) -> stat_result:
        raise AssertionError("filesystem should not be reached")

    async def list_directory(
        self,
        path: Path,
        limit: int,
    ) -> tuple[Path, ...]:
        raise AssertionError("filesystem should not be reached")

    async def read_bytes(self, path: Path, limit: int) -> bytes:
        raise AssertionError("filesystem should not be reached")


class LstatFailureFileSystem(SkillAsyncFileSystem):
    async def lstat_path(self, path: Path) -> stat_result:
        raise OSError("lstat failed")


class HugeDirectoryFileSystem(SkillAsyncFileSystem):
    def __init__(self) -> None:
        super().__init__()
        self.requested_limit = 0
        self.returned_count = 0
        self.lstat_count = 0

    async def list_directory(
        self,
        path: Path,
        limit: int,
    ) -> tuple[Path, ...]:
        self.requested_limit = limit
        entries = tuple(path / f"entry-{index}.md" for index in range(limit))
        self.returned_count = len(entries)
        return entries

    async def lstat_path(self, path: Path) -> stat_result:
        self.lstat_count += 1
        return await super().lstat_path(path)


class CountingFileSystem(SkillAsyncFileSystem):
    def __init__(self) -> None:
        super().__init__()
        self.resolve_names: list[str] = []
        self.stat_names: list[str] = []
        self.read_names: list[str] = []

    async def resolve_path(self, path: Path) -> Path:
        self.resolve_names.append(path.name)
        return await super().resolve_path(path)

    async def stat_path(self, path: Path) -> stat_result:
        self.stat_names.append(path.name)
        return await super().stat_path(path)

    async def read_bytes(self, path: Path, limit: int) -> bytes:
        self.read_names.append(path.name)
        return await super().read_bytes(path, limit)


class OutsideEntryFileSystem(SkillAsyncFileSystem):
    def __init__(self, outside_entry: Path) -> None:
        super().__init__()
        self._outside_entry = outside_entry

    async def list_directory(
        self,
        path: Path,
        limit: int,
    ) -> tuple[Path, ...]:
        return (self._outside_entry,)


class ResolveEscapeFileSystem(SkillAsyncFileSystem):
    def __init__(self, root: Path) -> None:
        super().__init__()
        self._root = root

    async def resolve_path(self, path: Path) -> Path:
        if path.name == "SKILL.md":
            return self._root.parent / "escaped.md"
        return await super().resolve_path(path)


class RedirectResolveFileSystem(SkillAsyncFileSystem):
    def __init__(self, source: Path, target: Path) -> None:
        super().__init__()
        self._source = source
        self._target = target

    async def resolve_path(self, path: Path) -> Path:
        if path == self._source:
            return self._target.resolve(strict=True)
        return await super().resolve_path(path)


class ResolveFailureFileSystem(SkillAsyncFileSystem):
    async def resolve_path(self, path: Path) -> Path:
        if path.name == "SKILL.md":
            raise RuntimeError("resolve failed")
        return await super().resolve_path(path)


class StatFailureFileSystem(SkillAsyncFileSystem):
    async def stat_path(self, path: Path) -> stat_result:
        raise OSError("stat failed")


class ReadFailureFileSystem(SkillAsyncFileSystem):
    async def read_bytes(self, path: Path, limit: int) -> bytes:
        raise OSError("read failed")


class ReadGrowthFileSystem(SkillAsyncFileSystem):
    async def read_bytes(self, path: Path, limit: int) -> bytes:
        return b"123456789"


class HiddenTargetFileSystem(SkillAsyncFileSystem):
    def __init__(self, root: Path) -> None:
        super().__init__()
        self._root = root

    async def resolve_path(self, path: Path) -> Path:
        if path.name == "SKILL.md":
            return self._root / ".hidden" / "SKILL.md"
        return await super().resolve_path(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _protocol_default_file_system() -> SkillSourceFileSystem:
    return cast(SkillSourceFileSystem, object())


def _config(label: str, root: Path) -> SkillConfiguredSource:
    return SkillConfiguredSource(
        label=label,
        authority=WorkspaceSkillSourceAuthority(),
        root_path=root,
    )


def _write(root: Path, relative: str, text: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return root


def _reasons(result: object) -> tuple[str, ...]:
    diagnostics = []
    if hasattr(result, "diagnostics"):
        diagnostics.extend(result.diagnostics)
    if hasattr(result, "sources"):
        for source in result.sources:
            diagnostics.extend(source.diagnostics)
    reasons: list[str] = []
    for diagnostic in diagnostics:
        reason = diagnostic.details.get("reason")
        if isinstance(reason, str):
            reasons.append(reason)
    return tuple(reasons)


if __name__ == "__main__":
    main()
