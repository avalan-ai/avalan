from collections.abc import Callable
from json import dumps
from os import getenv, symlink
from pathlib import Path
from shutil import copytree
from tempfile import TemporaryDirectory
from typing import cast
from unittest import IsolatedAsyncioTestCase, main

from avalan.skill import (
    BundledSkillSourceAuthority,
    SkillAsyncFileSystem,
    SkillAuthorizedSourceRoot,
    SkillConfiguredSource,
    SkillDiagnosticCode,
    SkillDiagnosticInfo,
    SkillLocalRuntimeBackend,
    SkillReadLimits,
    SkillResourceReader,
    SkillRuntimeBackend,
    SkillRuntimeMappedFileSystem,
    SkillRuntimeMappingKind,
    SkillRuntimeMappingResult,
    SkillRuntimeMode,
    SkillRuntimeRegistry,
    SkillRuntimeResolution,
    SkillRuntimeSourceMapping,
    SkillSourceAuthority,
    SkillSourceConfig,
    SkillSourceResolutionResult,
    SkillStatus,
    TrustedSkillSettings,
    WorkspaceSkillSourceAuthority,
    build_skill_runtime_registry,
    resolve_skill_runtime_sources,
)


class SkillRuntimeParityPhase13Test(IsolatedAsyncioTestCase):
    async def test_runtime_protocol_defaults_and_status_models(
        self,
    ) -> None:
        backend = cast(SkillRuntimeBackend, object())
        mode_property = cast(property, SkillRuntimeBackend.__dict__["mode"])
        mode_getter = mode_property.fget
        assert mode_getter is not None
        self.assertIsNone(mode_getter(backend))
        self.assertIsNone(await SkillRuntimeBackend.map_sources(backend, ()))

        empty_config_resolution = await resolve_skill_runtime_sources(())
        self.assertEqual(empty_config_resolution.status, SkillStatus.EMPTY)
        self.assertEqual(
            empty_config_resolution.mapping.status,
            SkillStatus.EMPTY,
        )

        with TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "source"
            _write_text(root / "note.md", "note")
            file_system = SkillRuntimeMappedFileSystem(roots=(root,))
            self.assertEqual(
                await file_system.read_bytes(root / "note.md", 16),
                b"note",
            )
            source = SkillAuthorizedSourceRoot(
                label="workspace-main",
                authority=WorkspaceSkillSourceAuthority(),
                root=root,
            )
            local_backend = SkillLocalRuntimeBackend()
            local_mapping = await local_backend.map_sources((source,))
            self.assertEqual(local_mapping.status, SkillStatus.OK)

            diagnostic = _mapping_policy_diagnostic("direct_mapping_denied")
            denied_mapping = SkillRuntimeSourceMapping(
                source_label=source.label,
                authority=source.authority,
                host_root=root,
                runtime_root=root,
                diagnostics=(diagnostic,),
            )
            self.assertEqual(
                denied_mapping.status,
                SkillStatus.POLICY_DENIED,
            )
            self.assertEqual(
                denied_mapping.as_model_dict()["status"],
                SkillStatus.POLICY_DENIED.value,
            )
            unavailable_mapping = SkillRuntimeMappingResult(
                mode=SkillRuntimeMode.SANDBOX,
                file_system=SkillAsyncFileSystem(),
                mappings=(
                    SkillRuntimeSourceMapping(
                        source_label=source.label,
                        authority=source.authority,
                        host_root=root,
                        runtime_root=root,
                        available=False,
                    ),
                ),
            )
            self.assertEqual(
                unavailable_mapping.status,
                SkillStatus.UNAVAILABLE,
            )
            empty_mapping = SkillRuntimeMappingResult(
                mode=SkillRuntimeMode.LOCAL,
                file_system=SkillAsyncFileSystem(),
            )
            empty_resolution = SkillRuntimeResolution(
                mode=SkillRuntimeMode.LOCAL,
                mapping=empty_mapping,
                resolution=SkillSourceResolutionResult(),
                file_system=empty_mapping.file_system,
            )
            self.assertEqual(empty_resolution.status, SkillStatus.EMPTY)
            self.assertEqual(
                empty_resolution.as_model_dict()["status"],
                SkillStatus.EMPTY.value,
            )
            denied_resolution = SkillRuntimeResolution(
                mode=SkillRuntimeMode.LOCAL,
                mapping=empty_mapping,
                resolution=SkillSourceResolutionResult(
                    diagnostics=(diagnostic,),
                ),
                file_system=empty_mapping.file_system,
            )
            self.assertEqual(
                denied_resolution.status,
                SkillStatus.POLICY_DENIED,
            )

    async def test_fake_modes_return_identical_logical_responses(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            source_root = base / "source"
            _write_skill(
                source_root / "pdf" / "SKILL.md",
                name="pdf-tools",
                description="PDF rendering guidance.",
                resources='["references/rendering.md"]',
            )
            _write_text(
                source_root / "pdf" / "references" / "rendering.md",
                "Render pages with a bounded rasterizer.\n",
            )
            read_limits = SkillReadLimits(max_bytes_per_read=4096)
            settings = TrustedSkillSettings(
                sources=(
                    SkillSourceConfig(
                        label="workspace-main",
                        authority=WorkspaceSkillSourceAuthority(),
                        root_path=source_root,
                    ),
                ),
                read_limits=read_limits,
            )
            runtimes = []
            runtimes.append(
                await build_skill_runtime_registry(
                    (_config(source_root),),
                    settings=settings,
                )
            )
            for backend in _fake_backends(base / "runtime"):
                runtimes.append(
                    await build_skill_runtime_registry(
                        (_config(source_root),),
                        backend=backend,
                        settings=settings,
                    )
                )

            registry_models = tuple(
                runtime.registry.as_model_dict() for runtime in runtimes
            )
            read_models = []
            for runtime in runtimes:
                self.assertEqual(runtime.status, SkillStatus.OK)
                read = await SkillResourceReader().read(
                    runtime.registry,
                    "pdf-tools",
                    file_system=runtime.file_system,
                )
                self.assertEqual(read.status, SkillStatus.OK)
                read_models.append(read.as_model_dict())
                encoded = dumps(
                    (
                        runtime.mapping.as_model_dict(),
                        runtime.registry.as_model_dict(),
                        read.as_model_dict(),
                    ),
                    sort_keys=True,
                )
                self.assertNotIn(str(base), encoded)
                self.assertNotIn("/private/", encoded)

            self.assertEqual(registry_models[0], registry_models[1])
            self.assertEqual(registry_models[0], registry_models[2])
            self.assertEqual(registry_models[0], registry_models[3])
            self.assertEqual(read_models[0], read_models[1])
            self.assertEqual(read_models[0], read_models[2])
            self.assertEqual(read_models[0], read_models[3])

    async def test_fake_modes_return_matching_unsafe_denials(self) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            cases = (
                (
                    "hidden",
                    _hidden_source,
                    SkillStatus.POLICY_DENIED,
                    SkillDiagnosticCode.POLICY_DENIED,
                    "hidden_path",
                    None,
                ),
                (
                    "binary",
                    _binary_source,
                    SkillStatus.POLICY_DENIED,
                    SkillDiagnosticCode.POLICY_DENIED,
                    "nul_byte",
                    None,
                ),
                (
                    "oversized",
                    _oversized_source,
                    SkillStatus.TRUNCATED,
                    SkillDiagnosticCode.RESOURCE_OVERSIZED,
                    "per_resource_bytes",
                    SkillReadLimits(max_bytes_per_read=16),
                ),
                (
                    "symlink",
                    _symlink_escape_source,
                    SkillStatus.UNAVAILABLE,
                    SkillDiagnosticCode.SOURCE_UNAVAILABLE,
                    "dangling_symlink",
                    None,
                ),
            )
            for (
                name,
                builder,
                status,
                code,
                reason,
                read_limits,
            ) in cases:
                source_root = base / name / "source"
                builder(source_root)
                observed = []
                for backend in _fake_backends(base / name / "runtime"):
                    runtime = await build_skill_runtime_registry(
                        (_config(source_root),),
                        backend=backend,
                        read_limits=read_limits,
                    )
                    observed.append(_registry_denial(runtime))
                    encoded = dumps(runtime.as_model_dict(), sort_keys=True)
                    self.assertNotIn(str(base), encoded)

                self.assertEqual(observed, [(status, code, reason)] * 3)

    async def test_widened_runtime_mount_is_denied_before_scan(self) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            source_root = base / "source" / "authorized"
            _write_skill(
                source_root / "SKILL.md",
                name="safe-skill",
                description="Safe guidance.",
            )
            _write_skill(
                source_root.parent / "SKILL.md",
                name="widened-skill",
                description="This parent must not become visible.",
            )
            backend = FakeSkillRuntimeBackend(
                mode=SkillRuntimeMode.CONTAINER,
                runtime_base=base / "runtime",
                host_root_override=source_root.parent,
                runtime_root_override=source_root.parent,
            )

            runtime = await build_skill_runtime_registry(
                (_config(source_root),),
                backend=backend,
            )

            self.assertEqual(runtime.status, SkillStatus.POLICY_DENIED)
            self.assertEqual(
                runtime.registry.diagnostics[0].details["reason"],
                "source_root_widened",
            )
            self.assertNotIn(
                str(base),
                dumps(runtime.as_model_dict(), sort_keys=True),
            )

    async def test_duplicate_runtime_mapping_label_fails_closed(self) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            source_root = base / "source" / "authorized"
            widened_root = source_root.parent
            _write_skill(
                source_root / "SKILL.md",
                name="safe-skill",
                description="Safe guidance.",
            )
            _write_skill(
                widened_root / "SKILL.md",
                name="widened-skill",
                description="This parent must not become visible.",
            )
            backend = FakeSkillRuntimeBackend(
                mode=SkillRuntimeMode.CONTAINER,
                runtime_base=base / "runtime",
                duplicate_runtime_root=widened_root,
            )

            runtime = await build_skill_runtime_registry(
                (_config(source_root),),
                backend=backend,
            )

            self.assertEqual(runtime.status, SkillStatus.POLICY_DENIED)
            self.assertEqual(
                runtime.registry.diagnostics[0].details["reason"],
                "duplicate_mapping_label",
            )
            with self.assertRaises(PermissionError):
                await runtime.file_system.read_bytes(
                    widened_root / "SKILL.md",
                    4096,
                )
            self.assertNotIn(
                str(base),
                dumps(runtime.as_model_dict(), sort_keys=True),
            )

    async def test_unproved_or_unavailable_mappings_fail_closed(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            source_root = base / "source"
            _write_skill(
                source_root / "SKILL.md",
                name="safe-skill",
                description="Safe guidance.",
            )
            cases = (
                (
                    FakeSkillRuntimeBackend(
                        mode=SkillRuntimeMode.SANDBOX,
                        runtime_base=base / "runtime" / "readonly",
                        read_only_proven=False,
                    ),
                    "read_only_unproven",
                ),
                (
                    FakeSkillRuntimeBackend(
                        mode=SkillRuntimeMode.CONTAINER,
                        runtime_base=base / "runtime" / "sharing",
                        kind=SkillRuntimeMappingKind.DIRECT,
                        path_sharing_proven=False,
                    ),
                    "path_sharing_unproven",
                ),
                (
                    FakeSkillRuntimeBackend(
                        mode=SkillRuntimeMode.SANDBOX,
                        runtime_base=base / "runtime" / "unavailable",
                        available=False,
                    ),
                    "mapping_unavailable",
                ),
                (
                    FakeSkillRuntimeBackend(
                        mode=SkillRuntimeMode.CONTAINER,
                        runtime_base=base / "runtime" / "root",
                        source_root_proven=False,
                    ),
                    "source_root_unproven",
                ),
            )

            for backend, reason in cases:
                runtime = await build_skill_runtime_registry(
                    (_config(source_root),),
                    backend=backend,
                )

                self.assertEqual(runtime.status, SkillStatus.UNAVAILABLE)
                self.assertEqual(
                    runtime.registry.diagnostics[0].code,
                    SkillDiagnosticCode.RUNTIME_SOURCE_UNAVAILABLE,
                )
                self.assertEqual(
                    runtime.registry.diagnostics[0].details["reason"],
                    reason,
                )
                self.assertNotIn(
                    str(base),
                    dumps(runtime.as_model_dict(), sort_keys=True),
                )

    async def test_runtime_backend_failure_branches_fail_closed(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            source_root = base / "source"
            secret_path = base / "secret.md"
            _write_skill(
                source_root / "SKILL.md",
                name="safe-skill",
                description="Safe guidance.",
            )
            _write_text(secret_path, "secret")
            cases = (
                (
                    FakeSkillRuntimeBackend(
                        mode=SkillRuntimeMode.CONTAINER,
                        runtime_base=base / "runtime" / "raise",
                        raise_error=True,
                    ),
                    SkillDiagnosticCode.RUNTIME_SOURCE_UNAVAILABLE,
                    "backend_unavailable",
                ),
                (
                    FakeSkillRuntimeBackend(
                        mode=SkillRuntimeMode.CONTAINER,
                        runtime_base=base / "runtime" / "diagnostic",
                        mapping_diagnostics=(
                            _mapping_policy_diagnostic(
                                "backend_preflight_denied"
                            ),
                        ),
                    ),
                    SkillDiagnosticCode.POLICY_DENIED,
                    "backend_preflight_denied",
                ),
                (
                    FakeSkillRuntimeBackend(
                        mode=SkillRuntimeMode.CONTAINER,
                        runtime_base=base / "runtime" / "empty",
                        drop_mappings=True,
                    ),
                    SkillDiagnosticCode.POLICY_DENIED,
                    "mapping_source_set_changed",
                ),
                (
                    FakeSkillRuntimeBackend(
                        mode=SkillRuntimeMode.CONTAINER,
                        runtime_base=base / "runtime" / "source-diagnostic",
                        source_mapping_diagnostics=(
                            _mapping_policy_diagnostic(
                                "source_mapping_denied"
                            ),
                        ),
                    ),
                    SkillDiagnosticCode.POLICY_DENIED,
                    "source_mapping_denied",
                ),
                (
                    FakeSkillRuntimeBackend(
                        mode=SkillRuntimeMode.CONTAINER,
                        runtime_base=base / "runtime" / "authority",
                        authority_override=BundledSkillSourceAuthority(
                            bundle_id="avalan"
                        ),
                    ),
                    SkillDiagnosticCode.POLICY_DENIED,
                    "mapping_authority_changed",
                ),
            )

            for backend, code, reason in cases:
                runtime = await build_skill_runtime_registry(
                    (_config(source_root),),
                    backend=backend,
                )

                self.assertNotEqual(runtime.status, SkillStatus.OK)
                self.assertEqual(runtime.registry.diagnostics[0].code, code)
                self.assertEqual(
                    runtime.registry.diagnostics[0].details["reason"],
                    reason,
                )
                with self.assertRaises(PermissionError):
                    await runtime.file_system.read_bytes(secret_path, 1024)
                self.assertNotIn(
                    str(base),
                    dumps(runtime.as_model_dict(), sort_keys=True),
                )

    async def test_missing_source_returns_empty_guarded_filesystem(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            missing_root = base / "missing"
            secret_path = base / "secret.md"
            _write_text(secret_path, "secret")

            resolution = await resolve_skill_runtime_sources(
                (_config(missing_root),),
            )
            self.assertEqual(resolution.status, SkillStatus.UNAVAILABLE)
            self.assertEqual(
                resolution.mapping.status,
                SkillStatus.EMPTY,
            )
            with self.assertRaises(PermissionError):
                await resolution.file_system.read_bytes(secret_path, 1024)

            runtime = await build_skill_runtime_registry(
                (_config(missing_root),),
            )

            self.assertNotEqual(runtime.status, SkillStatus.OK)
            self.assertEqual(runtime.status, SkillStatus.UNAVAILABLE)
            self.assertEqual(
                runtime.registry.diagnostics[0].code,
                SkillDiagnosticCode.SOURCE_UNAVAILABLE,
            )
            with self.assertRaises(PermissionError):
                await runtime.file_system.read_bytes(secret_path, 1024)
            self.assertNotIn(
                str(base),
                dumps(runtime.as_model_dict(), sort_keys=True),
            )

    async def test_read_revalidates_after_mapping_before_content_return(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            source_root = base / "source"
            _write_skill(
                source_root / "SKILL.md",
                name="safe-skill",
                description="Safe guidance.",
            )
            runtime = await build_skill_runtime_registry(
                (_config(source_root),),
                backend=FakeSkillRuntimeBackend(
                    mode=SkillRuntimeMode.CONTAINER,
                    runtime_base=base / "runtime",
                ),
            )
            resource_path = runtime.registry.skills[0].resources[0].path
            _write_skill(
                resource_path,
                name="safe-skill",
                description="Changed after registry build.",
            )

            read = await SkillResourceReader().read(
                runtime.registry,
                "safe-skill",
                file_system=runtime.file_system,
            )

            self.assertEqual(read.status, SkillStatus.STALE)
            self.assertEqual(
                read.diagnostics[0].code,
                SkillDiagnosticCode.RESOURCE_STALE,
            )
            self.assertIsNone(read.content)
            self.assertNotIn(
                str(base),
                dumps(read.as_model_dict(), sort_keys=True),
            )

    async def test_materialized_same_size_tampering_fails_closed(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            source_root = base / "source"
            original = (
                "---\n"
                "name: safe-skill\n"
                "description: Alpha guidance.\n"
                "resources: []\n"
                "---\n"
                "# Safe skill\n"
            )
            changed = (
                "---\n"
                "name: safe-skill\n"
                "description: Bravo guidance.\n"
                "resources: []\n"
                "---\n"
                "# Safe skill\n"
            )
            self.assertEqual(len(original), len(changed))
            self.assertEqual(
                original.count("\n"),
                changed.count("\n"),
            )
            _write_text(source_root / "SKILL.md", original)
            tampered_path = (
                base
                / "runtime"
                / SkillRuntimeMode.CONTAINER.value
                / "workspace-main"
                / "SKILL.md"
            )
            backend = FakeSkillRuntimeBackend(
                mode=SkillRuntimeMode.CONTAINER,
                runtime_base=base / "runtime",
                tamper=lambda root: _write_text(root / "SKILL.md", changed),
            )

            runtime = await build_skill_runtime_registry(
                (_config(source_root),),
                backend=backend,
            )

            self.assertEqual(runtime.status, SkillStatus.POLICY_DENIED)
            self.assertEqual(
                runtime.registry.diagnostics[0].details["reason"],
                "runtime_resource_set_changed",
            )
            with self.assertRaises(PermissionError):
                await runtime.file_system.read_bytes(tampered_path, 4096)
            encoded = dumps(runtime.as_model_dict(), sort_keys=True)
            self.assertNotIn(str(base), encoded)
            self.assertNotIn("Alpha guidance", encoded)
            self.assertNotIn("Bravo guidance", encoded)

    async def test_mapped_filesystem_blocks_direct_symlink_read(self) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            root = base / "root"
            outside = base / "outside.md"
            _write_text(outside, "outside")
            root.mkdir(parents=True)
            symlink(outside, root / "link.md")
            file_system = SkillRuntimeMappedFileSystem(
                roots=(root,),
                file_system=SkillAsyncFileSystem(),
            )

            with self.assertRaises(PermissionError):
                await file_system.read_bytes(root / "link.md", 1024)
            with self.assertRaises(PermissionError):
                await file_system.stat_path(root / "link.md")

    async def test_mapped_filesystem_blocks_lstat_outside_roots(self) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            root = base / "root"
            outside = base / "outside.md"
            root.mkdir(parents=True)
            _write_text(outside, "outside")
            file_system = SkillRuntimeMappedFileSystem(
                roots=(root,),
                file_system=SkillAsyncFileSystem(),
            )

            with self.assertRaises(PermissionError):
                await file_system.lstat_path(outside)

    async def test_mapped_filesystem_denies_missing_configured_root(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            missing_root = base / "missing"
            file_system = SkillRuntimeMappedFileSystem(
                roots=(missing_root,),
                file_system=SkillAsyncFileSystem(),
            )

            with self.assertRaises(PermissionError):
                await file_system.read_bytes(
                    missing_root / "secret.txt",
                    1024,
                )

    async def test_mapped_filesystem_blocks_retargeted_symlink_root(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            allowed = base / "allowed"
            outside = base / "outside"
            link = base / "link"
            _write_text(allowed / "secret.txt", "allowed")
            _write_text(outside / "secret.txt", "outside")
            symlink(allowed, link)
            file_system = SkillRuntimeMappedFileSystem(
                roots=(link,),
                file_system=SkillAsyncFileSystem(),
            )

            link.unlink()
            symlink(outside, link)

            with self.assertRaises(PermissionError):
                await file_system.read_bytes(link / "secret.txt", 1024)

    async def test_mapped_filesystem_blocks_symlink_directory_listing(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            root = base / "root"
            outside = base / "outside"
            _write_text(outside / "outside-secret.md", "secret")
            root.mkdir(parents=True)
            symlink(outside, root / "linkdir")
            file_system = SkillRuntimeMappedFileSystem(
                roots=(root,),
                file_system=SkillAsyncFileSystem(),
            )

            entries: tuple[Path, ...] = ()
            try:
                entries = await file_system.list_directory(
                    root / "linkdir",
                    10,
                )
            except PermissionError:
                pass
            else:
                self.fail("symlink directory listing must fail closed")
            self.assertNotIn(
                "outside-secret.md",
                {entry.name for entry in entries},
            )

    async def test_optional_real_backend_parity_is_env_gated(self) -> None:
        if getenv("AVALAN_SKILL_REAL_RUNTIME_PARITY") != "1":
            self.skipTest(
                "Set AVALAN_SKILL_REAL_RUNTIME_PARITY=1 to run optional "
                "real skills runtime parity checks; default CI uses fakes."
            )
        self.skipTest(
            "No real skills runtime backend is wired in Phase 13; fake "
            "backend parity is the required default coverage."
        )


class FakeSkillRuntimeBackend:
    def __init__(
        self,
        *,
        mode: SkillRuntimeMode,
        runtime_base: Path,
        kind: SkillRuntimeMappingKind = (
            SkillRuntimeMappingKind.MATERIALIZED_COPY
        ),
        source_root_proven: bool = True,
        path_sharing_proven: bool = True,
        read_only_proven: bool = True,
        available: bool = True,
        host_root_override: Path | None = None,
        runtime_root_override: Path | None = None,
        duplicate_runtime_root: Path | None = None,
        tamper: Callable[[Path], None] | None = None,
        raise_error: bool = False,
        mapping_diagnostics: tuple[SkillDiagnosticInfo, ...] = (),
        source_mapping_diagnostics: tuple[SkillDiagnosticInfo, ...] = (),
        authority_override: SkillSourceAuthority | None = None,
        drop_mappings: bool = False,
    ) -> None:
        assert isinstance(mode, SkillRuntimeMode)
        assert isinstance(runtime_base, Path)
        assert isinstance(kind, SkillRuntimeMappingKind)
        for diagnostic in mapping_diagnostics:
            assert isinstance(diagnostic, SkillDiagnosticInfo)
        for diagnostic in source_mapping_diagnostics:
            assert isinstance(diagnostic, SkillDiagnosticInfo)
        self._mode = mode
        self._runtime_base = runtime_base
        self._kind = kind
        self._source_root_proven = source_root_proven
        self._path_sharing_proven = path_sharing_proven
        self._read_only_proven = read_only_proven
        self._available = available
        self._host_root_override = host_root_override
        self._runtime_root_override = runtime_root_override
        self._duplicate_runtime_root = duplicate_runtime_root
        self._tamper = tamper
        self._raise_error = raise_error
        self._mapping_diagnostics = mapping_diagnostics
        self._source_mapping_diagnostics = source_mapping_diagnostics
        self._authority_override = authority_override
        self._drop_mappings = drop_mappings

    @property
    def mode(self) -> SkillRuntimeMode:
        return self._mode

    async def map_sources(
        self,
        sources: tuple[SkillAuthorizedSourceRoot, ...],
    ) -> SkillRuntimeMappingResult:
        if self._raise_error:
            raise RuntimeError("runtime backend unavailable")
        mappings: list[SkillRuntimeSourceMapping] = []
        source_iter: tuple[SkillAuthorizedSourceRoot, ...] = (
            () if self._drop_mappings else sources
        )
        for source in source_iter:
            runtime_root = self._runtime_root(source)
            if (
                self._kind is SkillRuntimeMappingKind.MATERIALIZED_COPY
                and self._runtime_root_override is None
            ):
                copytree(
                    source.root,
                    runtime_root,
                    dirs_exist_ok=True,
                    symlinks=True,
                )
                if self._tamper is not None:
                    self._tamper(runtime_root)
            mappings.append(
                SkillRuntimeSourceMapping(
                    source_label=source.label,
                    authority=self._authority_override or source.authority,
                    host_root=self._host_root_override or source.root,
                    runtime_root=runtime_root,
                    kind=self._kind,
                    source_root_proven=self._source_root_proven,
                    path_sharing_proven=self._path_sharing_proven,
                    read_only_proven=self._read_only_proven,
                    available=self._available,
                    diagnostics=self._source_mapping_diagnostics,
                )
            )
            if self._duplicate_runtime_root is not None:
                mappings.append(
                    SkillRuntimeSourceMapping(
                        source_label=source.label,
                        authority=source.authority,
                        host_root=source.root,
                        runtime_root=self._duplicate_runtime_root,
                        kind=self._kind,
                        source_root_proven=self._source_root_proven,
                        path_sharing_proven=self._path_sharing_proven,
                        read_only_proven=self._read_only_proven,
                        available=self._available,
                    )
                )
        return SkillRuntimeMappingResult(
            mode=self._mode,
            file_system=SkillAsyncFileSystem(),
            mappings=tuple(mappings),
            diagnostics=self._mapping_diagnostics,
        )

    def _runtime_root(self, source: SkillAuthorizedSourceRoot) -> Path:
        if self._runtime_root_override is not None:
            return self._runtime_root_override
        if self._kind is SkillRuntimeMappingKind.DIRECT:
            return source.root
        return self._runtime_base / self._mode.value / source.label


def _fake_backends(base: Path) -> tuple[FakeSkillRuntimeBackend, ...]:
    return (
        FakeSkillRuntimeBackend(
            mode=SkillRuntimeMode.LOCAL,
            runtime_base=base / "local",
            kind=SkillRuntimeMappingKind.DIRECT,
        ),
        FakeSkillRuntimeBackend(
            mode=SkillRuntimeMode.SANDBOX,
            runtime_base=base / "sandbox",
        ),
        FakeSkillRuntimeBackend(
            mode=SkillRuntimeMode.CONTAINER,
            runtime_base=base / "container",
        ),
    )


def _registry_denial(
    runtime: SkillRuntimeRegistry,
) -> tuple[
    SkillStatus,
    SkillDiagnosticCode,
    object,
]:
    diagnostic = (
        runtime.registry.source_diagnostics[0]
        if runtime.registry.source_diagnostics
        else runtime.registry.diagnostics[0]
    )
    return (
        diagnostic.status,
        diagnostic.code,
        diagnostic.details.get("reason"),
    )


def _mapping_policy_diagnostic(reason: str) -> SkillDiagnosticInfo:
    return SkillDiagnosticInfo(
        code=SkillDiagnosticCode.POLICY_DENIED,
        status=SkillStatus.POLICY_DENIED,
        message="The runtime skill source mapping is not authorized.",
        path="source.runtime",
        hint="Expose only authorized read-only skill source roots.",
        details={"reason": reason},
    )


def _config(root: Path) -> SkillConfiguredSource:
    return SkillConfiguredSource(
        label="workspace-main",
        authority=WorkspaceSkillSourceAuthority(),
        root_path=root,
    )


def _hidden_source(root: Path) -> None:
    _write_skill(
        root / ".hidden" / "SKILL.md",
        name="hidden-skill",
        description="Hidden guidance.",
    )


def _binary_source(root: Path) -> None:
    _write_bytes(root / "SKILL.md", b"\x00binary")


def _oversized_source(root: Path) -> None:
    _write_text(root / "SKILL.md", "x" * 128)


def _symlink_escape_source(root: Path) -> None:
    outside = root.parent / "outside.md"
    _write_text(outside, "outside")
    root.mkdir(parents=True, exist_ok=True)
    try:
        symlink(outside, root / "SKILL.md")
    except OSError as error:
        raise AssertionError(
            "symlink support is required for this test"
        ) from error


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
        f"# {name}\n",
    )


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


if __name__ == "__main__":
    main()
