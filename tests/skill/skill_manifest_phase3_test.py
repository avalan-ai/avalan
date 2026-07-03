from json import dumps
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, TestCase, main

from avalan.skill import (
    SkillAsyncFileSystem,
    SkillAuthorizedResource,
    SkillAuthorizedSourceRoot,
    SkillConfiguredSource,
    SkillDiagnosticCode,
    SkillIndexLimits,
    SkillManifestDocument,
    SkillManifestLoadResult,
    SkillReadLimits,
    SkillSourceLimits,
    SkillStatus,
    WorkspaceSkillSourceAuthority,
    normalize_skill_manifest_resource,
    normalize_skill_manifests,
    parse_skill_manifest_markdown,
    parse_skill_manifests,
    resolve_skill_sources,
)
from avalan.skill import manifest as manifest_module

FIXTURE_DIR = Path(__file__).with_name("fixtures") / "phase3"


class SkillManifestPhase3AsyncTest(IsolatedAsyncioTestCase):
    async def test_loads_pdf_style_manifest_and_declared_resources(
        self,
    ) -> None:
        source_result = await resolve_skill_sources(
            (
                SkillConfiguredSource(
                    label="Workspace Main",
                    authority=WorkspaceSkillSourceAuthority(),
                    root_path=FIXTURE_DIR,
                ),
            )
        )
        file_system = CountingReadFileSystem()

        result = await parse_skill_manifests(
            source_result.sources,
            file_system=file_system,
        )

        self.assertEqual(result.status, SkillStatus.OK)
        self.assertEqual(
            set(file_system.read_names), {"SKILL.md", "SKILL-pdf.md"}
        )
        self.assertEqual(len(result.usable_manifests), 2)
        manifest = _manifest_by_id(result, "pdf")
        self.assertTrue(manifest.readable)
        self.assertTrue(manifest.usable)
        self.assertIsNotNone(manifest.metadata)
        assert manifest.metadata is not None
        self.assertEqual(manifest.source_label, "workspace-main")
        self.assertEqual(manifest.package_resource_id, "pdf")
        self.assertEqual(manifest.skill_id, "pdf")
        self.assertEqual(manifest.metadata.tags, ("pdf", "rendering"))
        self.assertEqual(manifest.metadata.version, "1.0.0")
        self.assertIn("PDF files", manifest.metadata.description)
        self.assertEqual(
            tuple(
                resource.resource_id
                for resource in manifest.declared_resources
            ),
            ("main", "references/rendering.md"),
        )
        self.assertEqual(
            manifest.declared_resources[1].source_resource_id,
            "pdf/references/rendering.md",
        )
        encoded = dumps(result.as_model_dict(), sort_keys=True)
        self.assertNotIn(str(FIXTURE_DIR), encoded)
        self.assertNotIn("/Users/", encoded)

    async def test_loads_skill_dash_slug_manifest_resource(self) -> None:
        source_result = await resolve_skill_sources(
            (_config(FIXTURE_DIR / "standalone"),)
        )
        self.assertIn(
            "SKILL-pdf.md",
            {resource.resource_id for resource in source_result.resources},
        )
        file_system = CountingReadFileSystem()

        result = await parse_skill_manifests(
            source_result.sources,
            file_system=file_system,
        )

        self.assertEqual(result.status, SkillStatus.OK)
        self.assertEqual(file_system.read_names, ("SKILL-pdf.md",))
        self.assertEqual(len(result.usable_manifests), 1)
        manifest = result.manifests[0]
        self.assertEqual(manifest.manifest_resource_id, "SKILL-pdf.md")
        self.assertEqual(manifest.package_resource_id, ".")
        self.assertEqual(manifest.skill_id, "pdf-standalone")
        self.assertIsNotNone(manifest.metadata)
        assert manifest.metadata is not None
        self.assertEqual(manifest.metadata.name, "pdf-standalone")
        self.assertEqual(
            manifest.metadata.tags, ("pdf", "rendering", "standalone")
        )
        self.assertEqual(
            tuple(
                resource.source_resource_id
                for resource in manifest.declared_resources
            ),
            ("SKILL-pdf.md", "references/rendering.md"),
        )

    async def test_loads_direct_manifest_file_source(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            standalone_path = root / "SKILL-pdf.md"
            package_path = root / "package" / "SKILL.md"
            _write_skill(
                standalone_path,
                name="PDF",
                description="PDF guidance.",
            )
            _write_skill(
                package_path,
                name="Package PDF",
                description="Package guidance.",
            )
            _write_text(root / "SKILL-other.md", "not loaded\n")
            file_system = CountingReadFileSystem()

            standalone_source = await resolve_skill_sources(
                (
                    SkillConfiguredSource(
                        label="standalone",
                        authority=WorkspaceSkillSourceAuthority(),
                        manifest_path=standalone_path,
                    ),
                )
            )
            package_source = await resolve_skill_sources(
                (
                    SkillConfiguredSource(
                        label="package",
                        authority=WorkspaceSkillSourceAuthority(),
                        manifest_path=package_path,
                    ),
                )
            )
            result = await parse_skill_manifests(
                (*standalone_source.sources, *package_source.sources),
                file_system=file_system,
            )

        self.assertEqual(result.status, SkillStatus.OK)
        self.assertEqual(
            tuple(manifest.skill_id for manifest in result.manifests),
            ("pdf", "package-pdf"),
        )
        self.assertEqual(
            tuple(
                manifest.manifest_resource_id for manifest in result.manifests
            ),
            ("SKILL-pdf.md", "SKILL.md"),
        )
        self.assertEqual(
            tuple(
                manifest.package_resource_id for manifest in result.manifests
            ),
            (".", "."),
        )
        self.assertEqual(file_system.read_names, ("SKILL-pdf.md", "SKILL.md"))

    async def test_invalid_skill_dash_manifest_names_are_not_loaded(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "SKILL-safe.md",
                name="safe",
                description="Safe guidance.",
            )
            invalid_names = (
                "SKILL-123.md",
                "SKILL-.hidden.md",
                "SKILL-..md",
                "SKILL-bad__slug.md",
                "SKILL-secret.md",
                "SKILL-token.md",
                "SKILL-http:pdf.md",
            )
            for name in invalid_names:
                _write_skill(
                    root / name,
                    name="ignored",
                    description="Ignored guidance.",
                )
            source_result = await resolve_skill_sources((_config(root),))
            file_system = CountingReadFileSystem()

            result = await parse_skill_manifests(
                source_result.sources,
                file_system=file_system,
            )

            self.assertEqual(result.status, SkillStatus.OK)
            self.assertEqual(
                tuple(manifest.skill_id for manifest in result.manifests),
                ("safe",),
            )
            self.assertEqual(file_system.read_names, ("SKILL-safe.md",))

    async def test_resolver_rejected_manifest_resource_is_visible(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "SKILL.md").write_bytes(b"bad\x00content")
            source_result = await resolve_skill_sources((_config(root),))
            file_system = CountingReadFileSystem()

            result = await parse_skill_manifests(
                source_result.sources,
                file_system=file_system,
            )

            self.assertEqual(source_result.resources, ())
            self.assertEqual(file_system.read_names, ())
            self.assertEqual(result.status, SkillStatus.POLICY_DENIED)
            self.assertEqual(result.diagnostics, ())
            self.assertEqual(len(result.manifests), 1)
            manifest = result.manifests[0]
            self.assertEqual(manifest.manifest_resource_id, "SKILL.md")
            self.assertEqual(manifest.package_resource_id, ".")
            self.assertFalse(manifest.readable)
            self.assertFalse(manifest.usable)
            self.assertEqual(
                manifest.diagnostics[0].code,
                SkillDiagnosticCode.POLICY_DENIED,
            )
            self.assertEqual(
                manifest.diagnostics[0].details["reason"],
                "nul_byte",
            )
            self.assertEqual(
                manifest.diagnostics[0].details["resource_id"],
                "SKILL.md",
            )
            encoded = dumps(result.as_model_dict(), sort_keys=True)
            self.assertNotIn(str(root), encoded)
            self.assertNotIn("/private/", encoded)

    async def test_empty_sources_without_manifest_diagnostics_stay_empty(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            empty_root = Path(directory) / "empty"
            empty_root.mkdir()
            notes_root = Path(directory) / "notes"
            notes_root.mkdir()
            (notes_root / "notes.md").write_bytes(b"bad\x00content")

            empty_source = await resolve_skill_sources((_config(empty_root),))
            notes_source = await resolve_skill_sources((_config(notes_root),))
            empty_result = await parse_skill_manifests(empty_source.sources)
            notes_result = await parse_skill_manifests(notes_source.sources)

            self.assertEqual(empty_result.status, SkillStatus.EMPTY)
            self.assertEqual(empty_result.manifests, ())
            self.assertEqual(empty_result.diagnostics, ())
            self.assertEqual(notes_result.status, SkillStatus.EMPTY)
            self.assertEqual(notes_result.manifests, ())
            self.assertEqual(notes_result.diagnostics, ())

    async def test_duplicate_normalized_skill_ids_fail_closed(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "one" / "SKILL.md",
                name="PDF Tools",
                description="PDF guidance.",
            )
            _write_skill(
                root / "two" / "SKILL.md",
                name="pdf-tools",
                description="Other PDF guidance.",
            )

            source_result = await resolve_skill_sources((_config(root),))
            result = await parse_skill_manifests(source_result.sources)

            self.assertEqual(result.status, SkillStatus.BLOCKED)
            self.assertEqual(result.usable_manifests, ())
            self.assertEqual(
                result.diagnostics[0].code,
                SkillDiagnosticCode.DUPLICATE_ID,
            )
            self.assertEqual(result.diagnostics[0].candidates, ("pdf-tools",))
            self.assertTrue(
                all(not manifest.usable for manifest in result.manifests)
            )

    async def test_declared_resources_must_be_owned_by_skill_package(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "pdf" / "SKILL.md",
                name="pdf",
                description="PDF guidance.",
                resources='["references/rendering.md"]',
            )
            _write_text(
                root / "other" / "references" / "rendering.md",
                "wrong package\n",
            )

            source_result = await resolve_skill_sources((_config(root),))
            result = await parse_skill_manifests(source_result.sources)

            self.assertEqual(result.status, SkillStatus.NOT_FOUND)
            self.assertFalse(result.manifests[0].usable)
            self.assertEqual(
                result.manifests[0].diagnostics[0].code,
                SkillDiagnosticCode.RESOURCE_MISSING,
            )

    async def test_async_loader_reports_manifest_read_failures(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "SKILL.md",
                name="pdf",
                description="PDF guidance.",
            )
            source_result = await resolve_skill_sources((_config(root),))

            result = await parse_skill_manifests(
                source_result.sources,
                file_system=FailingReadFileSystem(),
            )

            self.assertEqual(result.status, SkillStatus.UNAVAILABLE)
            self.assertFalse(result.manifests[0].usable)
            self.assertEqual(
                result.manifests[0].diagnostics[0].code,
                SkillDiagnosticCode.SOURCE_UNAVAILABLE,
            )

    async def test_async_loader_reports_read_bounds_and_binary_content(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "SKILL.md",
                name="pdf",
                description="PDF guidance.",
            )
            source_result = await resolve_skill_sources((_config(root),))

            oversized = await parse_skill_manifests(
                source_result.sources,
                read_limits=SkillReadLimits(max_bytes_per_read=4),
            )
            nul_byte = await parse_skill_manifests(
                source_result.sources,
                file_system=StaticReadFileSystem(b"\x00markdown"),
            )
            non_utf8 = await parse_skill_manifests(
                source_result.sources,
                file_system=StaticReadFileSystem(b"\xff"),
            )

            self.assertEqual(oversized.status, SkillStatus.TRUNCATED)
            self.assertEqual(
                oversized.manifests[0].diagnostics[0].code,
                SkillDiagnosticCode.RESOURCE_OVERSIZED,
            )
            self.assertEqual(nul_byte.status, SkillStatus.UNAVAILABLE)
            self.assertEqual(
                nul_byte.manifests[0].diagnostics[0].code,
                SkillDiagnosticCode.BINARY_RESOURCE,
            )
            self.assertEqual(non_utf8.status, SkillStatus.UNAVAILABLE)
            self.assertEqual(
                non_utf8.manifests[0].diagnostics[0].code,
                SkillDiagnosticCode.BINARY_RESOURCE,
            )

    async def test_manifest_count_is_bounded_before_loading(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "one" / "SKILL.md",
                name="one",
                description="One guidance.",
            )
            _write_skill(
                root / "two" / "SKILL.md",
                name="two",
                description="Two guidance.",
            )
            source_result = await resolve_skill_sources((_config(root),))

            result = await parse_skill_manifests(
                source_result.sources,
                index_limits=SkillIndexLimits(max_skills=1),
            )

            self.assertEqual(result.status, SkillStatus.TRUNCATED)
            self.assertEqual(
                result.diagnostics[0].code,
                SkillDiagnosticCode.RESOURCE_OVERSIZED,
            )
            self.assertEqual(result.manifests, ())

    async def test_direct_manifest_declared_resource_authorization_edges(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            manifest_path = root / "SKILL.md"
            keep_path = root / "references" / "keep.md"
            huge_path = root / "references" / "huge.md"
            _write_text(manifest_path, "---\nname: pdf\n---\n")
            _write_text(keep_path, "keep\n")
            _write_text(huge_path, "0123456789\n")
            manifest_resource = SkillAuthorizedResource(
                source_label="workspace-main",
                resource_id="SKILL.md",
                path=manifest_path,
                size_bytes=16,
                line_count=3,
            )
            keep_resource = SkillAuthorizedResource(
                source_label="workspace-main",
                resource_id="references/keep.md",
                path=keep_path,
                size_bytes=5,
                line_count=1,
            )
            source = SkillAuthorizedSourceRoot(
                label="workspace-main",
                authority=WorkspaceSkillSourceAuthority(),
                root=root,
                manifest_resource_id="SKILL.md",
                resources=(manifest_resource,),
            )
            source_with_existing = SkillAuthorizedSourceRoot(
                label="workspace-main",
                authority=WorkspaceSkillSourceAuthority(),
                root=root,
                manifest_resource_id="SKILL.md",
                resources=(manifest_resource, keep_resource),
            )
            file_system = SkillAsyncFileSystem()
            kwargs = {
                "source": source,
                "resource": manifest_resource,
                "file_system": file_system,
                "read_limits": SkillReadLimits(),
                "index_limits": SkillIndexLimits(),
                "source_limits": SkillSourceLimits(),
            }

            malformed = (
                await (
                    manifest_module._manifest_resource_authorized_resources(
                        content="not front matter",
                        **kwargs,
                    )
                )
            )
            skill_bound = (
                await (
                    manifest_module._manifest_resource_authorized_resources(
                        content=_content_with_resources(
                            '"references/keep.md", "references/huge.md"'
                        ),
                        **{
                            **kwargs,
                            "index_limits": (
                                SkillIndexLimits(max_resources_per_skill=2)
                            ),
                        },
                    )
                )
            )
            invalid_then_valid = (
                await (
                    manifest_module._manifest_resource_authorized_resources(
                        content=_content_with_resources(
                            '"bad*name.md", "references/keep.md"'
                        ),
                        **kwargs,
                    )
                )
            )
            duplicate = (
                await (
                    manifest_module._manifest_resource_authorized_resources(
                        content=_content_with_resources(
                            '"references/keep.md"'
                        ),
                        **{
                            **kwargs,
                            "source": source_with_existing,
                        },
                    )
                )
            )
            source_bound = (
                await (
                    manifest_module._manifest_resource_authorized_resources(
                        content=_content_with_resources(
                            '"references/keep.md"'
                        ),
                        **{
                            **kwargs,
                            "source_limits": (
                                SkillSourceLimits(max_resources_per_source=1)
                            ),
                        },
                    )
                )
            )
            file_bound = (
                await (
                    manifest_module._manifest_resource_authorized_resources(
                        content=_content_with_resources(
                            '"references/keep.md"'
                        ),
                        **{
                            **kwargs,
                            "source_limits": (
                                SkillSourceLimits(max_files_per_source=1)
                            ),
                        },
                    )
                )
            )
            missing = (
                await (
                    manifest_module._manifest_resource_authorized_resources(
                        content=_content_with_resources(
                            '"references/missing.md"'
                        ),
                        **kwargs,
                    )
                )
            )
            indexed_bound = (
                await manifest_module._manifest_resource_authorized_resources(
                    content=_content_with_resources('"references/huge.md"'),
                    **{
                        **kwargs,
                        "index_limits": (
                            SkillIndexLimits(
                                max_indexed_bytes=manifest_resource.size_bytes
                            )
                        ),
                    },
                )
            )

        self.assertEqual(malformed, (manifest_resource,))
        self.assertEqual(skill_bound, (manifest_resource,))
        self.assertEqual(
            tuple(resource.resource_id for resource in invalid_then_valid),
            ("SKILL.md", "references/keep.md"),
        )
        self.assertEqual(duplicate, (manifest_resource, keep_resource))
        self.assertEqual(source_bound, (manifest_resource,))
        self.assertEqual(file_bound, (manifest_resource,))
        self.assertEqual(missing, (manifest_resource,))
        self.assertEqual(indexed_bound, (manifest_resource,))


class SkillManifestPhase3Test(TestCase):
    def test_empty_manifest_document_and_load_result_are_not_usable(
        self,
    ) -> None:
        manifest = SkillManifestDocument(
            source_label="workspace-main",
            manifest_resource_id="SKILL.md",
            package_resource_id=".",
        )
        result = SkillManifestLoadResult()
        wrapped = normalize_skill_manifests((manifest,))

        self.assertEqual(manifest.status, SkillStatus.MALFORMED)
        self.assertFalse(manifest.usable)
        self.assertEqual(result.status, SkillStatus.EMPTY)
        self.assertEqual(wrapped.status, SkillStatus.OK)
        self.assertEqual(
            normalize_skill_manifest_resource("references/rendering.md"),
            "references/rendering.md",
        )

    def test_disabled_manifest_is_structured_but_not_usable(self) -> None:
        manifest = _parse(
            "---\n"
            "name: PDF\n"
            "description: PDF guidance.\n"
            "enabled: false\n"
            "---\n"
            "# Body\n"
        )

        self.assertEqual(manifest.status, SkillStatus.DISABLED)
        self.assertFalse(manifest.readable)
        self.assertFalse(manifest.usable)
        self.assertIsNotNone(manifest.metadata)
        assert manifest.metadata is not None
        self.assertFalse(manifest.metadata.enabled)
        self.assertEqual(
            manifest.diagnostics[0].code, SkillDiagnosticCode.DISABLED
        )

    def test_malformed_manifests_are_not_readable_or_usable(self) -> None:
        cases = (
            (
                "not front matter",
                SkillStatus.MALFORMED,
                SkillDiagnosticCode.MANIFEST_MALFORMED,
                "manifest",
            ),
            (
                "---\ndescription: PDF guidance.\n---\n",
                SkillStatus.MALFORMED,
                SkillDiagnosticCode.MANIFEST_MALFORMED,
                "manifest.name",
            ),
            (
                "---\nname:\ndescription: PDF guidance.\n---\n",
                SkillStatus.MALFORMED,
                SkillDiagnosticCode.MANIFEST_MALFORMED,
                "manifest.name",
            ),
            (
                "---\nname: pdf\n---\n",
                SkillStatus.MALFORMED,
                SkillDiagnosticCode.MANIFEST_MALFORMED,
                "manifest.description",
            ),
            (
                "---\nname: pdf\nunknown: yes\ndescription: PDF.\n---\n",
                SkillStatus.MALFORMED,
                SkillDiagnosticCode.MANIFEST_MALFORMED,
                "manifest.unknown",
            ),
            (
                "---\nname: pdf\nbad/key: yes\ndescription: PDF.\n---\n",
                SkillStatus.MALFORMED,
                SkillDiagnosticCode.MANIFEST_MALFORMED,
                "manifest.field",
            ),
            (
                "---\nname pdf\ndescription: PDF.\n---\n",
                SkillStatus.MALFORMED,
                SkillDiagnosticCode.MANIFEST_MALFORMED,
                "manifest.line.2",
            ),
            (
                "---\nname: pdf\ndescription: PDF.\n",
                SkillStatus.MALFORMED,
                SkillDiagnosticCode.MANIFEST_MALFORMED,
                "manifest",
            ),
            (
                "---\nname: pdf\nname: other\ndescription: PDF.\n---\n",
                SkillStatus.MALFORMED,
                SkillDiagnosticCode.MANIFEST_MALFORMED,
                "manifest.name",
            ),
            (
                "---\nname: pdf\ndescription: PDF.\nenabled: yes\n---\n",
                SkillStatus.MALFORMED,
                SkillDiagnosticCode.MANIFEST_MALFORMED,
                "manifest.enabled",
            ),
            (
                "---\nname: pdf\ndescription: PDF.\ntags: [\n---\n",
                SkillStatus.MALFORMED,
                SkillDiagnosticCode.MANIFEST_MALFORMED,
                "manifest.tags",
            ),
            (
                '---\nname: "unterminated\ndescription: PDF.\n---\n',
                SkillStatus.MALFORMED,
                SkillDiagnosticCode.MANIFEST_MALFORMED,
                "manifest.name",
            ),
            (
                '---\nname: "pdf",\ndescription: PDF.\n---\n',
                SkillStatus.MALFORMED,
                SkillDiagnosticCode.MANIFEST_MALFORMED,
                "manifest.name",
            ),
            (
                '---\nname: ["pdf"]\ndescription: PDF.\n---\n',
                SkillStatus.MALFORMED,
                SkillDiagnosticCode.MANIFEST_MALFORMED,
                "manifest.name",
            ),
            (
                '---\nname: {"slug": "pdf"}\ndescription: PDF.\n---\n',
                SkillStatus.MALFORMED,
                SkillDiagnosticCode.MANIFEST_MALFORMED,
                "manifest.name",
            ),
            (
                '---\nname: pdf\ndescription: ["PDF guidance"]\n---\n',
                SkillStatus.MALFORMED,
                SkillDiagnosticCode.MANIFEST_MALFORMED,
                "manifest.description",
            ),
            (
                '---\nname: pdf\ndescription: {"text": "PDF guidance"}\n---\n',
                SkillStatus.MALFORMED,
                SkillDiagnosticCode.MANIFEST_MALFORMED,
                "manifest.description",
            ),
            (
                "---\nname: pdf\ndescription: PDF.\nversion: [1]\n---\n",
                SkillStatus.MALFORMED,
                SkillDiagnosticCode.MANIFEST_MALFORMED,
                "manifest.version",
            ),
            (
                "---\nname: pdf\ndescription: PDF.\nversion: {v: 1}\n---\n",
                SkillStatus.MALFORMED,
                SkillDiagnosticCode.MANIFEST_MALFORMED,
                "manifest.version",
            ),
        )

        for content, status, code, path in cases:
            with self.subTest(path=path):
                manifest = _parse(content)
                self.assertEqual(manifest.status, status)
                self.assertEqual(manifest.diagnostics[0].code, code)
                self.assertEqual(manifest.diagnostics[0].path, path)
                self.assertFalse(manifest.readable)
                self.assertFalse(manifest.usable)
                self.assertIsNone(manifest.metadata)

    def test_bad_tags_and_ambiguous_names_are_diagnostics(self) -> None:
        enabled_true = _parse(
            "---\nname: pdf\ndescription: PDF guidance.\nenabled: true\n---\n"
        )
        bad_tags = _parse(
            "---\n"
            "name: pdf\n"
            "description: PDF guidance.\n"
            'tags: ["bad/path"]\n'
            "---\n"
        )
        ambiguous_name = _parse(
            "---\nname: pdf/read\ndescription: PDF guidance.\n---\n"
        )

        self.assertEqual(enabled_true.status, SkillStatus.OK)
        self.assertEqual(bad_tags.status, SkillStatus.MALFORMED)
        self.assertEqual(bad_tags.diagnostics[0].path, "manifest.tags")
        self.assertEqual(ambiguous_name.status, SkillStatus.AMBIGUOUS)
        self.assertEqual(
            ambiguous_name.diagnostics[0].code,
            SkillDiagnosticCode.AMBIGUOUS_NAME,
        )
        self.assertFalse(ambiguous_name.usable)

    def test_invalid_resource_declarations_fail_closed(self) -> None:
        cases = (
            (
                '"references/rendering.md"',
                SkillStatus.MALFORMED,
                SkillDiagnosticCode.MANIFEST_MALFORMED,
            ),
            (
                '["../secret.md"]',
                SkillStatus.POLICY_DENIED,
                SkillDiagnosticCode.RESOURCE_OUTSIDE_ROOT,
            ),
            (
                '["references/"]',
                SkillStatus.MALFORMED,
                SkillDiagnosticCode.MANIFEST_MALFORMED,
            ),
            (
                '["references/*.md"]',
                SkillStatus.MALFORMED,
                SkillDiagnosticCode.MANIFEST_MALFORMED,
            ),
            (
                '["main"]',
                SkillStatus.MALFORMED,
                SkillDiagnosticCode.MANIFEST_MALFORMED,
            ),
            (
                '["notes.md", "notes.md"]',
                SkillStatus.BLOCKED,
                SkillDiagnosticCode.DUPLICATE_ID,
            ),
        )

        for resources, status, code in cases:
            with self.subTest(resources=resources):
                manifest = _parse(
                    "---\n"
                    "name: pdf\n"
                    "description: PDF guidance.\n"
                    f"resources: {resources}\n"
                    "---\n"
                )
                self.assertEqual(manifest.status, status)
                self.assertEqual(manifest.diagnostics[0].code, code)
                self.assertEqual(
                    manifest.diagnostics[0].path,
                    "manifest.resources",
                )
                self.assertFalse(manifest.usable)

    def test_resource_count_per_skill_is_bounded(self) -> None:
        manifest = _parse(
            "---\n"
            "name: pdf\n"
            "description: PDF guidance.\n"
            'resources: ["references/rendering.md"]\n'
            "---\n",
            index_limits=SkillIndexLimits(max_resources_per_skill=1),
        )

        self.assertEqual(manifest.status, SkillStatus.TRUNCATED)
        self.assertEqual(
            manifest.diagnostics[0].code,
            SkillDiagnosticCode.RESOURCE_OVERSIZED,
        )
        self.assertEqual(manifest.diagnostics[0].path, "manifest.resources")
        self.assertFalse(manifest.usable)

    def test_resources_outside_package_are_rejected_before_read_runtime(
        self,
    ) -> None:
        manifest = _parse(
            "---\n"
            "name: pdf\n"
            "description: PDF guidance.\n"
            'resources: ["../outside.md"]\n'
            "---\n"
        )

        self.assertEqual(manifest.status, SkillStatus.POLICY_DENIED)
        self.assertFalse(manifest.readable)
        self.assertEqual(
            manifest.diagnostics[0].code,
            SkillDiagnosticCode.RESOURCE_OUTSIDE_ROOT,
        )

    def test_declared_resources_require_same_source_authorization(
        self,
    ) -> None:
        missing_authorization = _parse(
            "---\n"
            "name: pdf\n"
            "description: PDF guidance.\n"
            'resources: ["references/rendering.md"]\n'
            "---\n"
        )
        cross_source_authorization = parse_skill_manifest_markdown(
            "---\n"
            "name: pdf\n"
            "description: PDF guidance.\n"
            'resources: ["references/rendering.md"]\n'
            "---\n",
            source_label="workspace-main",
            manifest_resource_id="pdf/SKILL.md",
            authorized_resources=(
                SkillAuthorizedResource(
                    source_label="other-source",
                    resource_id="pdf/references/rendering.md",
                    path=Path("/tmp/rendering.md"),
                    size_bytes=12,
                    line_count=1,
                ),
            ),
        )

        for manifest in (missing_authorization, cross_source_authorization):
            with self.subTest(
                manifest_resource_id=manifest.manifest_resource_id
            ):
                self.assertEqual(manifest.status, SkillStatus.NOT_FOUND)
                self.assertFalse(manifest.usable)
                self.assertIsNone(manifest.metadata)
                self.assertEqual(
                    manifest.diagnostics[0].code,
                    SkillDiagnosticCode.RESOURCE_MISSING,
                )
                self.assertEqual(
                    tuple(
                        resource.resource_id
                        for resource in manifest.declared_resources
                    ),
                    ("main",),
                )

    def test_unsafe_description_and_version_metadata_fail_closed(self) -> None:
        unsafe_description = _parse(
            "---\n"
            "name: pdf\n"
            "description: /Users/example/private/skills/pdf/SKILL.md\n"
            "---\n"
        )
        unsafe_version = _parse(
            "---\n"
            "name: pdf\n"
            "description: PDF guidance.\n"
            "version: $HOME/skills/pdf/SKILL.md\n"
            "---\n"
        )
        bad_name = _parse("---\nname: 123\ndescription: PDF guidance.\n---\n")
        commented = _parse(
            "---\n# comment\n\nname: pdf\ndescription: PDF guidance.\n---\n"
        )

        self.assertEqual(unsafe_description.status, SkillStatus.MALFORMED)
        self.assertEqual(
            unsafe_description.diagnostics[0].path,
            "manifest.description",
        )
        self.assertEqual(unsafe_version.status, SkillStatus.MALFORMED)
        self.assertEqual(
            unsafe_version.diagnostics[0].path,
            "manifest.version",
        )
        self.assertEqual(bad_name.status, SkillStatus.MALFORMED)
        self.assertEqual(bad_name.diagnostics[0].path, "manifest.name")
        self.assertEqual(commented.status, SkillStatus.OK)

    def test_model_dict_for_malformed_manifest_omits_body_and_host_paths(
        self,
    ) -> None:
        manifest = parse_skill_manifest_markdown(
            "---\nname: /Users/example/private/SKILL.md\n---\n# Body\n",
            source_label="/Users/example/private/source",
            manifest_resource_id="/Users/example/private/SKILL.md",
        )
        encoded = dumps(manifest.as_model_dict(), sort_keys=True)

        self.assertFalse(manifest.usable)
        self.assertNotIn("# Body", encoded)
        self.assertNotIn("/Users/", encoded)
        self.assertNotIn("private", encoded)

    def test_manifest_document_rejects_inconsistent_values(self) -> None:
        with self.assertRaises(AssertionError):
            SkillManifestDocument(
                source_label="workspace-main",
                manifest_resource_id="SKILL.md",
                package_resource_id=".",
                skill_id="Bad Name",
            )


class CountingReadFileSystem(SkillAsyncFileSystem):
    def __init__(self) -> None:
        super().__init__()
        self._read_names: list[str] = []

    @property
    def read_names(self) -> tuple[str, ...]:
        return tuple(self._read_names)

    async def read_bytes(self, path: Path, limit: int) -> bytes:
        self._read_names.append(path.name)
        return await super().read_bytes(path, limit)


class FailingReadFileSystem(SkillAsyncFileSystem):
    async def read_bytes(self, path: Path, limit: int) -> bytes:
        raise OSError("read failed")


class StaticReadFileSystem(SkillAsyncFileSystem):
    def __init__(self, content: bytes) -> None:
        super().__init__()
        self._content = content

    async def read_bytes(self, path: Path, limit: int) -> bytes:
        return self._content


def _parse(
    content: str,
    *,
    index_limits: SkillIndexLimits | None = None,
) -> SkillManifestDocument:
    return parse_skill_manifest_markdown(
        content,
        source_label="workspace-main",
        index_limits=index_limits,
    )


def _content_with_resources(resources: str) -> str:
    return (
        "---\n"
        "name: pdf\n"
        "description: PDF guidance.\n"
        f"resources: [{resources}]\n"
        "---\n"
        "# Body\n"
    )


def _manifest_by_id(
    result: SkillManifestLoadResult,
    skill_id: str,
) -> SkillManifestDocument:
    matching = tuple(
        manifest
        for manifest in result.manifests
        if manifest.skill_id == skill_id
    )
    assert len(matching) == 1
    return matching[0]


def _config(root: Path) -> SkillConfiguredSource:
    return SkillConfiguredSource(
        label="workspace-main",
        authority=WorkspaceSkillSourceAuthority(),
        root_path=root,
    )


def _write_skill(
    path: Path,
    *,
    name: str,
    description: str,
    resources: str = "[]",
) -> None:
    _write_text(
        path,
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        f"resources: {resources}\n"
        "---\n"
        "# Body\n",
    )


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
