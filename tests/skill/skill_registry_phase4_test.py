from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
from json import dumps
from os import stat_result
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, main

from avalan.skill import (
    BundledSkillSourceAuthority,
    SkillAuthorizedSourceRoot,
    SkillConfiguredSource,
    SkillDiagnosticCode,
    SkillDiagnosticInfo,
    SkillIndexLimits,
    SkillReadLimits,
    SkillRegistry,
    SkillRegistryResourceCheck,
    SkillRegistryVersion,
    SkillResourceFingerprint,
    SkillResourceHandle,
    SkillResourceReader,
    SkillStatus,
    TrustedSkillSettings,
    WorkspaceSkillSourceAuthority,
    build_skill_registry,
    resolve_skill_sources,
)


class SkillRegistryPhase4Test(IsolatedAsyncioTestCase):
    async def test_empty_registry_is_deterministic_and_immutable(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "empty"
            root.mkdir()
            source_result = await resolve_skill_sources((_config(root),))

            first = await build_skill_registry(source_result)
            second = await build_skill_registry(source_result.sources)

            self.assertEqual(first.status, SkillStatus.EMPTY)
            self.assertEqual(second.status, SkillStatus.EMPTY)
            self.assertEqual(
                first.diagnostics[0].code,
                SkillDiagnosticCode.EMPTY_REGISTRY,
            )
            self.assertEqual(
                first.registry_version.as_model_value(),
                second.registry_version.as_model_value(),
            )
            self.assertEqual(first.skills, ())
            self.assertEqual(first.metadata, ())
            with self.assertRaises(FrozenInstanceError):
                first.skills += ()
            with self.assertRaises(TypeError):
                first.skills_by_id["pdf"] = object()

    async def test_valid_registry_stores_metadata_sources_and_handles(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory)
            workspace = base / "workspace"
            bundled = base / "bundled"
            _write_skill(
                workspace / "pdf" / "SKILL.md",
                name="PDF",
                description="PDF guidance.",
                resources='["references/rendering.md", "empty.md"]',
            )
            _write_text(
                workspace / "pdf" / "references" / "rendering.md",
                "Render pages.\n",
            )
            _write_text(workspace / "pdf" / "empty.md", "")
            _write_skill(
                bundled / "docx" / "SKILL.md",
                name="DOCX",
                description="DOCX guidance.",
            )
            workspace_result = await resolve_skill_sources(
                (
                    SkillConfiguredSource(
                        label="Workspace Main",
                        authority=WorkspaceSkillSourceAuthority(),
                        root_path=workspace,
                    ),
                    SkillConfiguredSource(
                        label="Bundled Main",
                        authority=BundledSkillSourceAuthority(
                            bundle_id="avalan"
                        ),
                        root_path=bundled,
                    ),
                )
            )
            settings = TrustedSkillSettings(
                read_limits=SkillReadLimits(max_bytes_per_read=4096),
                index_limits=SkillIndexLimits(max_skills=16),
            )

            registry = await build_skill_registry(
                workspace_result,
                settings=settings,
            )
            repeated = await build_skill_registry(
                workspace_result,
                settings=settings,
            )
            model = registry.as_model_dict()
            encoded = dumps(model, sort_keys=True)
            rendering_sha256 = sha256(b"Render pages.\n").hexdigest()

            self.assertEqual(registry.status, SkillStatus.OK)
            self.assertEqual(
                registry.registry_version.as_model_value(),
                repeated.registry_version.as_model_value(),
            )
            self.assertEqual(
                tuple(source.label for source in registry.sources),
                ("workspace-main", "bundled-main"),
            )
            self.assertEqual(
                tuple(metadata.skill_id for metadata in registry.metadata),
                ("pdf", "docx"),
            )
            self.assertEqual(
                tuple(
                    metadata.skill_id for metadata in registry.usable_metadata
                ),
                ("pdf", "docx"),
            )
            self.assertEqual(
                tuple(
                    handle.resource_id for handle in registry.resource_handles
                ),
                (
                    "main",
                    "references/rendering.md",
                    "empty.md",
                    "main",
                ),
            )
            empty = next(
                resource
                for skill in registry.skills
                for resource in skill.resources
                if resource.handle.resource_id == "empty.md"
            )
            self.assertEqual(empty.fingerprint.line_count, 0)
            self.assertEqual(
                registry.skills_by_id["pdf"].metadata.skill_id,
                "pdf",
            )
            self.assertIn("settings", model)
            self.assertNotIn(str(base), encoded)
            self.assertNotIn("/private/", encoded)
            self.assertNotIn(rendering_sha256, encoded)
            self.assertNotIn("content_sha256", encoded)
            self.assertNotIn("function", encoded)
            self.assertNotIn("skills.load", encoded)
            with self.assertRaises(TypeError):
                registry.resources_by_key[
                    ("workspace-main", "pdf", "main")
                ] = empty
            with self.assertRaises(FrozenInstanceError):
                setattr(
                    registry.resource_handles[0], "status", SkillStatus.STALE
                )

    async def test_direct_manifest_reads_declared_resources_only(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "SKILL.md"
            _write_skill(
                manifest,
                name="pdf",
                description="PDF guidance.",
                resources='["references/rendering.md"]',
            )
            _write_text(
                root / "references" / "rendering.md",
                "Render pages.\n",
            )
            _write_skill(
                root / "SKILL-other.md",
                name="other",
                description="Other guidance.",
            )
            source_result = await resolve_skill_sources(
                (
                    SkillConfiguredSource(
                        label="pdf",
                        authority=WorkspaceSkillSourceAuthority(),
                        manifest_path=manifest,
                    ),
                )
            )

            registry = await build_skill_registry(source_result)
            read = await SkillResourceReader().read(
                registry,
                "pdf",
                resource_id="references/rendering.md",
            )

        self.assertEqual(registry.status, SkillStatus.OK)
        self.assertEqual(tuple(registry.skills_by_id), ("pdf",))
        self.assertEqual(read.status, SkillStatus.OK)
        self.assertIsNotNone(read.content)
        assert read.content is not None
        self.assertEqual(read.content.text, "Render pages.\n")
        self.assertEqual(
            tuple(handle.resource_id for handle in registry.resource_handles),
            ("main", "references/rendering.md"),
        )

    async def test_disabled_malformed_and_duplicate_skills_are_structured(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory)
            disabled_root = base / "disabled"
            malformed_root = base / "malformed"
            duplicate_root = base / "duplicate"
            _write_skill(
                disabled_root / "SKILL.md",
                name="Disabled Skill",
                description="Disabled guidance.",
                enabled=False,
            )
            _write_text(
                malformed_root / "SKILL.md",
                "---\nname: Broken\ndescription: Broken.\n",
            )
            _write_skill(
                duplicate_root / "one" / "SKILL.md",
                name="PDF Tools",
                description="PDF guidance.",
            )
            _write_skill(
                duplicate_root / "two" / "SKILL.md",
                name="pdf-tools",
                description="Other PDF guidance.",
            )

            disabled = await build_skill_registry(
                await resolve_skill_sources((_config(disabled_root),))
            )
            malformed = await build_skill_registry(
                await resolve_skill_sources((_config(malformed_root),))
            )
            duplicate = await build_skill_registry(
                await resolve_skill_sources((_config(duplicate_root),))
            )

            self.assertEqual(disabled.status, SkillStatus.DISABLED)
            self.assertEqual(disabled.metadata[0].status, SkillStatus.DISABLED)
            self.assertEqual(disabled.usable_metadata, ())
            self.assertEqual(
                disabled.skills[0].diagnostics[0].code,
                SkillDiagnosticCode.DISABLED,
            )
            disabled_check = await disabled.check_resource(
                disabled.resource_handles[0]
            )
            self.assertEqual(disabled_check.status, SkillStatus.DISABLED)
            self.assertEqual(
                disabled_check.diagnostics[0].code,
                SkillDiagnosticCode.DISABLED,
            )
            self.assertEqual(malformed.status, SkillStatus.MALFORMED)
            self.assertIsNone(malformed.skills[0].metadata)
            self.assertIsNotNone(malformed.skills[0].manifest_fingerprint)
            self.assertEqual(
                malformed.skills[0].diagnostics[0].code,
                SkillDiagnosticCode.MANIFEST_MALFORMED,
            )
            self.assertEqual(duplicate.status, SkillStatus.BLOCKED)
            self.assertEqual(duplicate.usable_metadata, ())
            self.assertEqual(duplicate.skills_by_id, {})
            self.assertEqual(
                duplicate.diagnostics[0].code,
                SkillDiagnosticCode.DUPLICATE_ID,
            )
            self.assertNotEqual(duplicate.resources_by_key, {})
            duplicate_check = await duplicate.check_resource(
                duplicate.resource_handles[0]
            )
            self.assertEqual(duplicate_check.status, SkillStatus.BLOCKED)
            self.assertEqual(
                duplicate_check.diagnostics[0].code,
                SkillDiagnosticCode.DUPLICATE_ID,
            )
            self.assertTrue(
                all(
                    skill.status == SkillStatus.BLOCKED
                    for skill in duplicate.skills
                )
            )

    async def test_unavailable_sources_and_rejected_manifest_are_deterministic(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory)
            missing = base / "missing"
            rejected = base / "rejected"
            rejected.mkdir()
            (rejected / "SKILL.md").write_bytes(b"bad\x00content")

            missing_registry = await build_skill_registry(
                await resolve_skill_sources((_config(missing),))
            )
            repeated_missing = await build_skill_registry(
                await resolve_skill_sources((_config(missing),))
            )
            rejected_registry = await build_skill_registry(
                await resolve_skill_sources((_config(rejected),))
            )

            self.assertEqual(missing_registry.status, SkillStatus.UNAVAILABLE)
            self.assertEqual(
                missing_registry.diagnostics[0].code,
                SkillDiagnosticCode.SOURCE_UNAVAILABLE,
            )
            self.assertEqual(
                missing_registry.registry_version.as_model_value(),
                repeated_missing.registry_version.as_model_value(),
            )
            self.assertEqual(
                rejected_registry.status, SkillStatus.POLICY_DENIED
            )
            self.assertIsNone(rejected_registry.skills[0].manifest_fingerprint)
            self.assertEqual(rejected_registry.skills[0].resources, ())
            self.assertEqual(
                rejected_registry.source_diagnostics[0].code,
                SkillDiagnosticCode.POLICY_DENIED,
            )

    async def test_registry_version_changes_for_changed_inputs(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            skill_path = root / "SKILL.md"
            _write_skill(
                skill_path,
                name="PDF",
                description="PDF guidance.",
            )
            source_result = await resolve_skill_sources((_config(root),))
            baseline = await build_skill_registry(source_result)
            same = await build_skill_registry(source_result)
            tight_limits = await build_skill_registry(
                source_result,
                read_limits=SkillReadLimits(max_bytes_per_read=1024),
            )
            bundled_source = SkillAuthorizedSourceRoot(
                label=source_result.sources[0].label,
                authority=BundledSkillSourceAuthority(bundle_id="avalan"),
                root=source_result.sources[0].root,
                resources=source_result.sources[0].resources,
            )
            authority_changed = await build_skill_registry((bundled_source,))
            diagnostic = SkillDiagnosticInfo(
                code=SkillDiagnosticCode.SOURCE_UNAVAILABLE,
                status=SkillStatus.UNAVAILABLE,
                message="Source unavailable.",
                path="source.availability",
                hint="Do not use this source.",
            )
            source_status_changed = await build_skill_registry(
                (
                    replace(
                        source_result.sources[0],
                        diagnostics=(diagnostic,),
                    ),
                )
            )

            _write_skill(
                skill_path,
                name="PDF",
                description="Updated PDF guidance.",
            )
            changed_result = await resolve_skill_sources((_config(root),))
            content_changed = await build_skill_registry(changed_result)

            self.assertEqual(
                baseline.registry_version.as_model_value(),
                same.registry_version.as_model_value(),
            )
            self.assertNotEqual(
                baseline.registry_version.as_model_value(),
                tight_limits.registry_version.as_model_value(),
            )
            self.assertNotEqual(
                baseline.registry_version.as_model_value(),
                authority_changed.registry_version.as_model_value(),
            )
            self.assertNotEqual(
                baseline.registry_version.as_model_value(),
                source_status_changed.registry_version.as_model_value(),
            )

            _write_text(
                skill_path,
                "---\n"
                "name: PDF\n"
                "description: PDF guidance.\n"
                "enabled: true\n"
                "resources: []\n"
                "---\n"
                "# Bady\n",
            )
            same_size_body_changed = await build_skill_registry(
                await resolve_skill_sources((_config(root),))
            )
            self.assertNotEqual(
                baseline.registry_version.as_model_value(),
                same_size_body_changed.registry_version.as_model_value(),
            )
            self.assertNotEqual(
                baseline.registry_version.as_model_value(),
                content_changed.registry_version.as_model_value(),
            )

    async def test_registry_version_changes_for_manifest_file_sources(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_path = root / "SKILL-pdf.md"
            docx_path = root / "SKILL-docx.md"
            _write_skill(
                pdf_path,
                name="PDF",
                description="PDF guidance.",
            )
            _write_skill(
                docx_path,
                name="PDF",
                description="PDF guidance.",
            )

            pdf_registry = await build_skill_registry(
                await resolve_skill_sources(
                    (
                        SkillConfiguredSource(
                            label="workspace-main",
                            authority=WorkspaceSkillSourceAuthority(),
                            manifest_path=pdf_path,
                        ),
                    )
                )
            )
            docx_registry = await build_skill_registry(
                await resolve_skill_sources(
                    (
                        SkillConfiguredSource(
                            label="workspace-main",
                            authority=WorkspaceSkillSourceAuthority(),
                            manifest_path=docx_path,
                        ),
                    )
                )
            )

        self.assertNotEqual(
            pdf_registry.registry_version.as_model_value(),
            docx_registry.registry_version.as_model_value(),
        )
        self.assertEqual(
            pdf_registry.skills[0].manifest_resource_id,
            "SKILL-pdf.md",
        )

    async def test_resource_check_detects_stale_deleted_and_unavailable(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "SKILL.md",
                name="PDF",
                description="PDF guidance.",
            )
            registry = await build_skill_registry(
                await resolve_skill_sources((_config(root),))
            )
            handle = registry.resource_handles[0]
            spoofed_handle = SkillResourceHandle(
                source_label=handle.source_label,
                skill_id=handle.skill_id,
                resource_id=handle.resource_id,
                media_type="application/json",
                size_bytes=999,
                status=SkillStatus.STALE,
                stale=True,
            )

            ok = await registry.check_resource(spoofed_handle)
            ok_model = ok.as_model_dict()
            _write_skill(
                root / "SKILL.md",
                name="PDF",
                description="Changed PDF guidance.",
            )
            stale = await registry.check_resource(handle)
            (root / "SKILL.md").unlink()
            deleted = await registry.check_resource(handle)
            unavailable = await registry.check_resource(
                handle,
                file_system=StatFailureFileSystem(),
            )
            unknown = await registry.check_resource(
                SkillResourceHandle(
                    source_label="workspace-main",
                    skill_id="unknown",
                    resource_id="main",
                )
            )

            self.assertEqual(ok.status, SkillStatus.OK)
            self.assertIs(ok.handle, handle)
            self.assertEqual(ok.handle.status, SkillStatus.OK)
            self.assertFalse(ok.handle.stale)
            self.assertEqual(ok.handle.media_type, "text/markdown")
            self.assertEqual(ok.handle.size_bytes, handle.size_bytes)
            self.assertIn("stored_fingerprint", ok_model)
            self.assertIn("current_fingerprint", ok_model)
            self.assertNotIn("content_sha256", dumps(ok_model))
            self.assertEqual(stale.status, SkillStatus.STALE)
            self.assertTrue(stale.handle.stale)
            self.assertEqual(
                stale.diagnostics[0].code,
                SkillDiagnosticCode.RESOURCE_STALE,
            )
            self.assertEqual(deleted.status, SkillStatus.NOT_FOUND)
            self.assertEqual(
                deleted.diagnostics[0].code,
                SkillDiagnosticCode.RESOURCE_MISSING,
            )
            self.assertEqual(unavailable.status, SkillStatus.UNAVAILABLE)
            self.assertEqual(
                unavailable.diagnostics[0].code,
                SkillDiagnosticCode.RUNTIME_SOURCE_UNAVAILABLE,
            )
            self.assertEqual(unknown.status, SkillStatus.NOT_FOUND)

    async def test_resource_check_detects_runtime_read_races(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "SKILL.md",
                name="PDF",
                description="PDF guidance.",
            )
            registry = await build_skill_registry(
                await resolve_skill_sources((_config(root),))
            )
            handle = registry.resource_handles[0]

            read_missing = await registry.check_resource(
                handle,
                file_system=ReadMissingFileSystem(),
            )
            read_failure = await registry.check_resource(
                handle,
                file_system=ReadFailureFileSystem(),
            )
            read_limit = await registry.check_resource(
                handle,
                read_limits=SkillReadLimits(max_bytes_per_read=4),
            )
            stat_mismatch = await registry.check_resource(
                handle,
                file_system=StatMismatchFileSystem(),
            )

            self.assertEqual(read_missing.status, SkillStatus.NOT_FOUND)
            self.assertEqual(read_failure.status, SkillStatus.UNAVAILABLE)
            self.assertEqual(read_limit.status, SkillStatus.STALE)
            self.assertEqual(
                read_limit.diagnostics[0].details["reason"],
                "read_limit_exceeded",
            )
            self.assertEqual(stat_mismatch.status, SkillStatus.STALE)
            self.assertEqual(
                stat_mismatch.diagnostics[0].details["reason"],
                "stat_content_mismatch",
            )

    async def test_build_detects_stale_and_unreadable_fingerprints(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "SKILL.md",
                name="PDF",
                description="PDF guidance.",
                resources='["references/rendering.md"]',
            )
            _write_text(root / "references" / "rendering.md", "Render.\n")
            source_result = await resolve_skill_sources((_config(root),))

            stale = await build_skill_registry(
                source_result,
                file_system=GrowthAfterManifestFileSystem(),
            )
            unreadable = await build_skill_registry(
                source_result,
                file_system=ReferenceReadFailureFileSystem(),
            )
            stale_check = await stale.check_resource(
                stale.resource_handles[0],
                file_system=FailOnRuntimeCheckFileSystem(),
            )
            unreadable_check = await unreadable.check_resource(
                unreadable.resource_handles[1],
                file_system=FailOnRuntimeCheckFileSystem(),
            )

            self.assertEqual(stale.status, SkillStatus.STALE)
            self.assertFalse(stale.skills[0].usable)
            self.assertEqual(
                stale.diagnostics[0].details["reason"],
                "resolved_metadata_changed",
            )
            self.assertEqual(unreadable.status, SkillStatus.UNAVAILABLE)
            self.assertFalse(unreadable.skills[0].usable)
            self.assertEqual(
                unreadable.diagnostics[0].code,
                SkillDiagnosticCode.SOURCE_UNAVAILABLE,
            )
            self.assertEqual(stale_check.status, SkillStatus.STALE)
            self.assertEqual(
                stale_check.diagnostics[0].code,
                SkillDiagnosticCode.RESOURCE_STALE,
            )
            self.assertEqual(
                stale_check.stored_fingerprint.status,
                SkillStatus.STALE,
            )
            self.assertEqual(unreadable_check.status, SkillStatus.UNAVAILABLE)
            self.assertEqual(
                unreadable_check.diagnostics[0].code,
                SkillDiagnosticCode.SOURCE_UNAVAILABLE,
            )
            self.assertEqual(
                unreadable_check.stored_fingerprint.status,
                SkillStatus.UNAVAILABLE,
            )

    async def test_multiple_failed_resources_return_exact_diagnostics(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "SKILL.md",
                name="PDF",
                description="PDF guidance.",
                resources='["references/first.md", "references/second.md"]',
            )
            _write_text(root / "references" / "first.md", "First.\n")
            _write_text(root / "references" / "second.md", "Second.\n")
            source_result = await resolve_skill_sources((_config(root),))

            registry = await build_skill_registry(
                source_result,
                file_system=MultipleReferenceReadFailureFileSystem(),
            )
            second = next(
                handle
                for handle in registry.resource_handles
                if handle.resource_id == "references/second.md"
            )

            checked = await registry.check_resource(
                second,
                file_system=FailOnRuntimeCheckFileSystem(),
            )

            self.assertEqual(checked.status, SkillStatus.UNAVAILABLE)
            self.assertEqual(
                checked.diagnostics[0].details["resource_id"],
                "references/second.md",
            )

    async def test_constructor_provided_indexes_are_recomputed(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "SKILL.md",
                name="PDF",
                description="PDF guidance.",
            )
            registry = await build_skill_registry(
                await resolve_skill_sources((_config(root),))
            )
            resource = registry.skills[0].resources[0]

            spoofed = SkillRegistry(
                registry_version=registry.registry_version,
                read_limits=registry.read_limits,
                index_limits=registry.index_limits,
                sources=registry.sources,
                skills=registry.skills,
                diagnostics=registry.diagnostics,
                skills_by_id={"evil": registry.skills[0]},
                resources_by_key={
                    ("workspace-main", "evil", "main"): resource
                },
            )
            checked = await spoofed.check_resource(
                SkillResourceHandle(
                    source_label="workspace-main",
                    skill_id="evil",
                    resource_id="main",
                )
            )

            self.assertNotIn("evil", spoofed.skills_by_id)
            self.assertNotIn(
                ("workspace-main", "evil", "main"),
                spoofed.resources_by_key,
            )
            self.assertEqual(checked.status, SkillStatus.NOT_FOUND)
            self.assertEqual(
                checked.diagnostics[0].code,
                SkillDiagnosticCode.NOT_FOUND,
            )

    def test_registry_entities_reject_invalid_values(self) -> None:
        valid_fingerprint = SkillResourceFingerprint(
            source_label="workspace-main",
            resource_id="SKILL.md",
            size_bytes=1,
            line_count=1,
            content_sha256="a" * 64,
        )
        check = SkillRegistryResourceCheck(
            handle=SkillResourceHandle(
                source_label="workspace-main",
                skill_id="pdf",
                resource_id="main",
            )
        )
        empty = SkillRegistry(
            registry_version=SkillRegistryVersion(
                value="skills-registry:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            ),
            read_limits=SkillReadLimits(),
            index_limits=SkillIndexLimits(),
        )

        self.assertEqual(valid_fingerprint.as_model_dict()["status"], "ok")
        self.assertNotIn("content_sha256", valid_fingerprint.as_model_dict())
        self.assertEqual(check.as_model_dict()["status"], "ok")
        self.assertEqual(empty.status, SkillStatus.EMPTY)
        with self.assertRaises(AssertionError):
            SkillResourceFingerprint(
                source_label="workspace-main",
                resource_id="SKILL.md",
                size_bytes=-1,
                line_count=1,
            )
        with self.assertRaises(AssertionError):
            SkillResourceFingerprint(
                source_label="workspace-main",
                resource_id="SKILL.md",
                size_bytes=1,
                line_count=1,
                content_sha256="not-a-digest",
            )


class StatFailureFileSystem:
    async def stat_path(self, path: Path) -> stat_result:
        raise OSError("stat failed")

    async def read_bytes(self, path: Path, limit: int) -> bytes:
        raise AssertionError("read should not happen")


class ReadMissingFileSystem:
    async def stat_path(self, path: Path) -> stat_result:
        return path.stat()

    async def read_bytes(self, path: Path, limit: int) -> bytes:
        raise FileNotFoundError("missing during read")


class ReadFailureFileSystem:
    async def stat_path(self, path: Path) -> stat_result:
        return path.stat()

    async def read_bytes(self, path: Path, limit: int) -> bytes:
        raise OSError("read failed")


class StatMismatchFileSystem:
    async def stat_path(self, path: Path) -> stat_result:
        values = list(path.stat())
        values[6] += 1
        return stat_result(values)

    async def read_bytes(self, path: Path, limit: int) -> bytes:
        with path.open("rb") as input_file:
            return input_file.read(limit)


class FailOnRuntimeCheckFileSystem:
    async def stat_path(self, path: Path) -> stat_result:
        raise AssertionError("runtime check should not stat stored failures")

    async def read_bytes(self, path: Path, limit: int) -> bytes:
        raise AssertionError("runtime check should not read stored failures")


class GrowthAfterManifestFileSystem:
    def __init__(self) -> None:
        self._reads_by_name: dict[str, int] = {}

    async def read_bytes(self, path: Path, limit: int) -> bytes:
        with path.open("rb") as input_file:
            content = input_file.read(limit)
        reads = self._reads_by_name.get(path.name, 0)
        self._reads_by_name[path.name] = reads + 1
        if path.name == "SKILL.md" and reads > 0:
            return content + b"\nchanged"
        return content


class ReferenceReadFailureFileSystem:
    async def read_bytes(self, path: Path, limit: int) -> bytes:
        if path.name == "rendering.md":
            raise OSError("reference read failed")
        with path.open("rb") as input_file:
            return input_file.read(limit)


class MultipleReferenceReadFailureFileSystem:
    async def read_bytes(self, path: Path, limit: int) -> bytes:
        if path.name in {"first.md", "second.md"}:
            raise OSError(f"{path.name} read failed")
        with path.open("rb") as input_file:
            return input_file.read(limit)


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
    resources: str = "[]",
    enabled: bool = True,
) -> None:
    _write_text(
        path,
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        f"enabled: {'true' if enabled else 'false'}\n"
        f"resources: {resources}\n"
        "---\n"
        "# Body\n",
    )


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
