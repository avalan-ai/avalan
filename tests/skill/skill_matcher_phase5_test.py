from asyncio import gather
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from json import dumps
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase, TestCase, main

from avalan.skill import (
    BundledSkillSourceAuthority,
    SkillAsyncFileSystem,
    SkillConfiguredSource,
    SkillDiagnosticCode,
    SkillDiagnosticInfo,
    SkillIndexLimits,
    SkillMatchFilters,
    SkillMatchIndex,
    SkillMatchIndexEntry,
    SkillMatchLimits,
    SkillMatchResult,
    SkillMetadata,
    SkillReadLimits,
    SkillRegistry,
    SkillRegistryVersion,
    SkillResponseEnvelope,
    SkillStatus,
    WorkspaceSkillSourceAuthority,
    build_skill_match_index,
    build_skill_registry,
    match_skill_registry,
    resolve_skill_sources,
)


class SkillMatcherPhase5AsyncTest(IsolatedAsyncioTestCase):
    async def test_matches_exact_name_tags_source_and_description(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory)
            workspace = base / "workspace"
            bundled = base / "bundled"
            _write_skill(
                workspace / "pdf-reader" / "SKILL.md",
                name="pdf-reader",
                description="Read portable documents and extract text.",
                tags=("pdf", "rendering"),
            )
            _write_skill(
                workspace / "analytics" / "SKILL.md",
                name="analytics",
                description="Spreadsheet analytics guidance.",
            )
            _write_skill(
                bundled / "diagram" / "SKILL.md",
                name="diagram",
                description="Diagram guidance.",
            )
            registry = await _registry(
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

            exact = await match_skill_registry(
                registry,
                query="pdf-reader",
            )
            named = await match_skill_registry(
                registry,
                query="reader",
            )
            tagged = await match_skill_registry(
                registry,
                query="rendering",
            )
            sourced = await match_skill_registry(
                registry,
                source_label="bundled-main",
            )
            described = await match_skill_registry(
                registry,
                query="spreadsheet analytics",
            )

            self.assertEqual(exact.status, SkillStatus.OK)
            exact_item = _match_item(exact)
            named_item = _match_item(named)
            tagged_item = _match_item(tagged)
            sourced_item = _match_item(sourced)
            described_item = _match_item(described)

            self.assertEqual(exact_item.metadata.skill_id, "pdf-reader")
            self.assertEqual(exact_item.score, 1.0)
            self.assertIn(
                "exact skill_id matched query",
                exact_item.reasons,
            )
            self.assertIn("exact name matched query", exact_item.reasons)
            self.assertEqual(named_item.metadata.skill_id, "pdf-reader")
            self.assertEqual(
                named_item.reasons,
                ("name tokens matched query",),
            )
            self.assertEqual(tagged_item.metadata.skill_id, "pdf-reader")
            self.assertEqual(
                tagged_item.reasons,
                ("tag metadata matched query",),
            )
            self.assertEqual(sourced_item.metadata.skill_id, "diagram")
            self.assertEqual(
                sourced_item.reasons,
                ("source filter matched",),
            )
            self.assertEqual(described_item.metadata.skill_id, "analytics")
            self.assertEqual(
                described_item.reasons,
                ("description matched query",),
            )

    async def test_bounded_excerpt_matching_never_returns_body_text(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "pdf" / "SKILL.md",
                name="pdf",
                description="PDF metadata.",
                body=(
                    "# Private Instructions\n"
                    "Use the calibration vector omega for rendering.\n"
                    "Do not leak this full instruction body.\n"
                ),
            )
            registry = await _registry((_config(root),))

            metadata_only = await match_skill_registry(
                registry,
                query="omega",
            )
            index = await build_skill_match_index(
                registry,
                include_resource_excerpts=True,
                match_limits=SkillMatchLimits(max_excerpt_bytes_per_skill=512),
            )
            matched = await match_skill_registry(
                registry,
                query="omega",
                index=index,
            )
            partial = await match_skill_registry(
                registry,
                query="omega missing",
                index=index,
            )
            encoded_match = dumps(matched.as_model_dict(), sort_keys=True)
            encoded_index = dumps(index.as_model_dict(), sort_keys=True)

            self.assertEqual(metadata_only.status, SkillStatus.EMPTY)
            self.assertEqual(
                metadata_only.diagnostics[0].code,
                SkillDiagnosticCode.NO_MATCH,
            )
            self.assertEqual(matched.status, SkillStatus.OK)
            matched_item = _match_item(matched)
            partial_item = _match_item(partial)

            self.assertEqual(matched_item.metadata.skill_id, "pdf")
            self.assertEqual(
                matched_item.reasons,
                ("bounded indexed excerpt matched query",),
            )
            self.assertEqual(
                partial_item.reasons,
                ("bounded indexed excerpt partially matched query",),
            )
            self.assertGreater(index.indexed_bytes, 0)
            self.assertLessEqual(index.indexed_bytes, 512)
            self.assertNotIn(
                "Do not leak this full instruction body",
                encoded_match,
            )
            self.assertNotIn("Private Instructions", encoded_match)
            self.assertNotIn(
                "Do not leak this full instruction body",
                encoded_index,
            )
            self.assertNotIn("Private Instructions", encoded_index)
            self.assertNotIn("text", matched.as_model_dict())

    async def test_filters_limits_and_nonexact_metadata_reasons(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory)
            workspace = base / "workspace"
            bundled = base / "bundled"
            _write_skill(
                workspace / "pdf" / "SKILL.md",
                name="pdf",
                description="Read portable documents.",
                tags=("pdf",),
            )
            _write_skill(
                workspace / "docx" / "SKILL.md",
                name="docx",
                description="DOCX guidance.",
            )
            _write_skill(
                bundled / "diagram" / "SKILL.md",
                name="diagram",
                description="Diagram guidance.",
            )
            registry = await _registry(
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

            tag_filter = await match_skill_registry(registry, tags=("pdf",))
            status_filter = await match_skill_registry(
                registry,
                status=SkillStatus.OK,
                usable_only=False,
            )
            source_query = await match_skill_registry(
                registry,
                query="bundled main",
            )
            partial_metadata = await match_skill_registry(
                registry,
                query="portable missing",
            )
            limited = await match_skill_registry(registry, max_results=1)
            missing_status = await match_skill_registry(
                registry,
                status=SkillStatus.DISABLED,
                usable_only=False,
            )
            missing_tag = await match_skill_registry(
                registry,
                tags=("missing",),
            )

            self.assertEqual(
                _match_item(tag_filter).reasons,
                ("tag filter matched",),
            )
            self.assertEqual(
                _match_item(status_filter).reasons,
                ("status filter matched",),
            )
            self.assertEqual(
                _match_item(source_query).reasons,
                ("source label matched query",),
            )
            self.assertEqual(
                _match_item(partial_metadata).reasons,
                ("metadata partially matched query",),
            )
            self.assertEqual(len(limited.items), 1)
            self.assertEqual(missing_status.status, SkillStatus.EMPTY)
            self.assertEqual(missing_tag.status, SkillStatus.EMPTY)

    async def test_empty_no_match_and_unavailable_diagnostics(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory)
            empty_root = base / "empty"
            empty_root.mkdir()
            valid_root = base / "valid"
            missing_root = base / "missing"
            _write_skill(
                valid_root / "pdf" / "SKILL.md",
                name="pdf",
                description="PDF guidance.",
            )

            empty = await match_skill_registry(
                await _registry((_config(empty_root),)),
                query="pdf",
            )
            no_match = await match_skill_registry(
                await _registry((_config(valid_root),)),
                query="no-such-skill",
            )
            unavailable = await match_skill_registry(
                await _registry((_config(missing_root),)),
                query="pdf",
            )

            self.assertEqual(empty.status, SkillStatus.EMPTY)
            self.assertEqual(
                empty.diagnostics[0].code,
                SkillDiagnosticCode.EMPTY_REGISTRY,
            )
            self.assertEqual(no_match.status, SkillStatus.EMPTY)
            self.assertEqual(
                no_match.diagnostics[0].code,
                SkillDiagnosticCode.NO_MATCH,
            )
            self.assertEqual(unavailable.status, SkillStatus.UNAVAILABLE)
            self.assertEqual(
                unavailable.diagnostics[0].code,
                SkillDiagnosticCode.SOURCE_UNAVAILABLE,
            )

    async def test_ambiguous_names_return_ranked_candidates_and_diagnostic(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "pdf-basic" / "SKILL.md",
                name="pdf-basic",
                description="Basic PDF handling.",
            )
            _write_skill(
                root / "pdf-advanced" / "SKILL.md",
                name="pdf-advanced",
                description="Advanced PDF handling.",
            )
            registry = await _registry((_config(root),))

            result = await match_skill_registry(registry, query="pdf")

            self.assertEqual(result.status, SkillStatus.AMBIGUOUS)
            self.assertEqual(
                result.diagnostics[0].code,
                SkillDiagnosticCode.AMBIGUOUS_NAME,
            )
            self.assertEqual(
                result.diagnostics[0].candidates,
                ("pdf-advanced", "pdf-basic"),
            )
            self.assertEqual(
                _match_skill_ids(result),
                ("pdf-advanced", "pdf-basic"),
            )

    async def test_disabled_exclusions_can_be_overridden_explicitly(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "disabled-skill" / "SKILL.md",
                name="disabled-skill",
                description="Disabled guidance.",
                enabled=False,
            )
            registry = await _registry((_config(root),))

            excluded = await match_skill_registry(
                registry,
                query="disabled-skill",
            )
            included = await match_skill_registry(
                registry,
                query="disabled-skill",
                status=SkillStatus.DISABLED,
                usable_only=False,
            )

            self.assertEqual(excluded.status, SkillStatus.DISABLED)
            self.assertEqual(excluded.items, ())
            self.assertEqual(
                excluded.diagnostics[0].code,
                SkillDiagnosticCode.DISABLED,
            )
            self.assertEqual(included.status, SkillStatus.OK)
            self.assertEqual(
                _match_item(included).metadata.status,
                SkillStatus.DISABLED,
            )
            self.assertEqual(
                _match_item(included).metadata.skill_id,
                "disabled-skill",
            )

    async def test_blocked_duplicate_exact_match_returns_blocking_diagnostic(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "one" / "SKILL.md",
                name="PDF Tools",
                description="First PDF guidance.",
            )
            _write_skill(
                root / "two" / "SKILL.md",
                name="pdf-tools",
                description="Second PDF guidance.",
            )
            registry = await _registry((_config(root),))

            result = await match_skill_registry(
                registry,
                query="pdf-tools",
            )

            self.assertEqual(result.status, SkillStatus.BLOCKED)
            self.assertEqual(result.items, ())
            self.assertEqual(
                result.diagnostics[0].code,
                SkillDiagnosticCode.DUPLICATE_ID,
            )
            self.assertEqual(result.diagnostics[0].candidates, ("pdf-tools",))

    async def test_supplied_index_entries_must_match_registry_skills(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "pdf" / "SKILL.md",
                name="pdf",
                description="PDF guidance.",
            )
            registry = await _registry((_config(root),))
            forged_metadata = SkillMetadata(
                skill_id="evil",
                name="evil",
                description="Forged guidance.",
                source_label="workspace-main",
            )
            forged_index = SkillMatchIndex(
                registry_version=registry.registry_version,
                entries=(_entry(forged_metadata),),
            )
            stale_entry = SkillMatchIndexEntry(
                metadata=registry.metadata[0],
                status=SkillStatus.OK,
                usable=True,
                id_tokens=("evil",),
                name_tokens=("evil",),
                tag_tokens=(),
                source_tokens=("workspace-main", "workspace", "main"),
                description_tokens=("forged",),
            )
            stale_index = SkillMatchIndex(
                registry_version=registry.registry_version,
                entries=(stale_entry,),
            )
            status_mismatch_entry = SkillMatchIndexEntry(
                metadata=registry.metadata[0],
                status=SkillStatus.DISABLED,
                usable=False,
                id_tokens=("pdf",),
                name_tokens=("pdf",),
                tag_tokens=(),
                source_tokens=("workspace-main", "workspace", "main"),
                description_tokens=("pdf", "guidance"),
            )
            status_mismatch_index = SkillMatchIndex(
                registry_version=registry.registry_version,
                entries=(status_mismatch_entry,),
            )

            forged = await match_skill_registry(
                registry,
                query="evil",
                index=forged_index,
            )
            forged_real = await match_skill_registry(
                registry,
                query="pdf",
                index=forged_index,
            )
            stale = await match_skill_registry(
                registry,
                query="evil",
                index=stale_index,
            )
            status_mismatch = await match_skill_registry(
                registry,
                query="pdf",
                index=status_mismatch_index,
            )

            self.assertEqual(forged.status, SkillStatus.NOT_FOUND)
            self.assertEqual(forged.items, ())
            self.assertEqual(
                forged.diagnostics[0].code,
                SkillDiagnosticCode.NOT_FOUND,
            )
            self.assertEqual(forged.diagnostics[0].candidates, ("evil",))
            self.assertEqual(forged_real.status, SkillStatus.OK)
            self.assertEqual(_match_item(forged_real).metadata.skill_id, "pdf")
            self.assertEqual(
                forged_real.diagnostics[0].code,
                SkillDiagnosticCode.NOT_FOUND,
            )
            self.assertEqual(
                forged_real.diagnostics[0].candidates,
                ("evil",),
            )
            self.assertEqual(stale.status, SkillStatus.NOT_FOUND)
            self.assertEqual(stale.items, ())
            self.assertEqual(
                stale.diagnostics[0].details["reason"],
                "index_entry_not_in_registry",
            )
            self.assertEqual(status_mismatch.status, SkillStatus.OK)
            self.assertEqual(
                _match_item(status_mismatch).metadata.skill_id,
                "pdf",
            )
            self.assertEqual(
                status_mismatch.diagnostics[0].candidates,
                ("pdf",),
            )

    async def test_supplied_index_excerpt_tokens_are_sanitized(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "pdf" / "SKILL.md",
                name="pdf",
                description="PDF guidance.",
            )
            registry = await _registry((_config(root),))
            spoofed_entry = SkillMatchIndexEntry(
                metadata=registry.metadata[0],
                status=SkillStatus.OK,
                usable=True,
                id_tokens=("pdf",),
                name_tokens=("pdf",),
                tag_tokens=(),
                source_tokens=("workspace-main", "workspace", "main"),
                description_tokens=("pdf", "guidance"),
                excerpt_tokens=("omega",),
                indexed_excerpt_bytes=5,
            )
            spoofed_index = SkillMatchIndex(
                registry_version=registry.registry_version,
                entries=(spoofed_entry,),
                indexed_bytes=5,
            )

            spoofed = await match_skill_registry(
                registry,
                query="omega",
                index=spoofed_index,
            )
            metadata = await match_skill_registry(
                registry,
                query="pdf",
                index=spoofed_index,
            )
            encoded_spoofed = dumps(spoofed.as_model_dict(), sort_keys=True)

            self.assertEqual(spoofed.status, SkillStatus.EMPTY)
            self.assertEqual(spoofed.items, ())
            self.assertEqual(
                spoofed.diagnostics[0].code,
                SkillDiagnosticCode.NO_MATCH,
            )
            self.assertNotIn("bounded indexed excerpt", encoded_spoofed)
            self.assertEqual(metadata.status, SkillStatus.OK)
            self.assertEqual(_match_item(metadata).metadata.skill_id, "pdf")

    async def test_supplied_index_duplicate_entries_are_deduped(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "alpha" / "SKILL.md",
                name="alpha",
                description="Alpha guidance.",
            )
            _write_skill(
                root / "beta" / "SKILL.md",
                name="beta",
                description="Beta guidance.",
            )
            registry = await _registry((_config(root),))
            metadata_by_id = {
                metadata.skill_id: metadata for metadata in registry.metadata
            }
            alpha_entry = SkillMatchIndexEntry(
                metadata=metadata_by_id["alpha"],
                status=SkillStatus.OK,
                usable=True,
                id_tokens=("alpha",),
                name_tokens=("alpha",),
                tag_tokens=(),
                source_tokens=("workspace-main", "workspace", "main"),
                description_tokens=("alpha", "guidance"),
            )
            beta_entry = SkillMatchIndexEntry(
                metadata=metadata_by_id["beta"],
                status=SkillStatus.OK,
                usable=True,
                id_tokens=("beta",),
                name_tokens=("beta",),
                tag_tokens=(),
                source_tokens=("workspace-main", "workspace", "main"),
                description_tokens=("beta", "guidance"),
            )
            duplicate_index = SkillMatchIndex(
                registry_version=registry.registry_version,
                entries=(
                    alpha_entry,
                    alpha_entry,
                    alpha_entry,
                    beta_entry,
                ),
            )

            result = await match_skill_registry(
                registry,
                index=duplicate_index,
                max_results=2,
            )

            self.assertEqual(result.status, SkillStatus.OK)
            self.assertEqual(
                _match_skill_ids(result),
                ("alpha", "beta"),
            )
            self.assertEqual(
                result.diagnostics[0].code,
                SkillDiagnosticCode.DUPLICATE_ID,
            )
            self.assertEqual(
                result.diagnostics[0].details["reason"],
                "duplicate_index_entry",
            )
            self.assertEqual(result.diagnostics[0].candidates, ("alpha",))

    async def test_supplied_index_omitted_entries_are_reconciled(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "alpha" / "SKILL.md",
                name="alpha",
                description="Alpha guidance.",
            )
            _write_skill(
                root / "beta" / "SKILL.md",
                name="beta",
                description="Beta guidance.",
            )
            _write_text(
                root / "broken" / "SKILL.md",
                "---\nname: Broken\ndescription: Broken.\n",
            )
            registry = await _registry((_config(root),))
            full_index = await build_skill_match_index(registry)
            alpha_entry = next(
                entry
                for entry in full_index.entries
                if entry.metadata.skill_id == "alpha"
            )
            supplied_index = SkillMatchIndex(
                registry_version=registry.registry_version,
                entries=(alpha_entry,),
            )

            result = await match_skill_registry(
                registry,
                query="beta",
                index=supplied_index,
            )

            self.assertEqual(result.status, SkillStatus.OK)
            self.assertEqual(_match_item(result).metadata.skill_id, "beta")
            self.assertEqual(
                _match_item(result).reasons,
                (
                    "exact skill_id matched query",
                    "exact name matched query",
                ),
            )

    async def test_supplied_index_diagnostics_are_sanitized(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "pdf" / "SKILL.md",
                name="pdf",
                description="PDF guidance.",
            )
            registry = await _registry((_config(root),))
            full_index = await build_skill_match_index(registry)
            injected = SkillDiagnosticInfo(
                code=SkillDiagnosticCode.SOURCE_UNAVAILABLE,
                status=SkillStatus.UNAVAILABLE,
                message="Injected unavailable diagnostic.",
                path="source.availability",
                hint="This diagnostic did not come from registry indexing.",
                candidates=("workspace-main",),
                details={"source_label": "workspace-main"},
            )
            supplied_index = SkillMatchIndex(
                registry_version=registry.registry_version,
                entries=full_index.entries,
                diagnostics=(injected,),
            )

            no_match = await match_skill_registry(
                registry,
                query="missing",
                source_label="workspace-main",
                index=supplied_index,
            )
            matched = await match_skill_registry(
                registry,
                query="pdf",
                index=supplied_index,
            )

            self.assertEqual(no_match.status, SkillStatus.EMPTY)
            self.assertEqual(
                no_match.diagnostics[0].code,
                SkillDiagnosticCode.NO_MATCH,
            )
            self.assertEqual(matched.status, SkillStatus.OK)
            self.assertEqual(matched.diagnostics, ())

    async def test_untrusted_index_limits_do_not_hide_registry_matches(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "pdf" / "SKILL.md",
                name="pdf",
                description="PDF guidance.",
            )
            registry = await _registry((_config(root),))
            restrictive_limits = SkillMatchLimits(max_query_characters=1)
            supplied_index = SkillMatchIndex(
                registry_version=registry.registry_version,
                limits=restrictive_limits,
            )

            untrusted = await match_skill_registry(
                registry,
                query="pdf",
                index=supplied_index,
            )
            explicit = await match_skill_registry(
                registry,
                query="pdf",
                index=supplied_index,
                match_limits=restrictive_limits,
            )

            self.assertEqual(untrusted.status, SkillStatus.OK)
            self.assertEqual(_match_item(untrusted).metadata.skill_id, "pdf")
            self.assertEqual(untrusted.diagnostics, ())
            self.assertEqual(explicit.status, SkillStatus.EMPTY)
            self.assertEqual(
                explicit.diagnostics[0].code,
                SkillDiagnosticCode.NO_MATCH,
            )
            self.assertEqual(
                explicit.diagnostics[1].code,
                SkillDiagnosticCode.RESOURCE_OVERSIZED,
            )

    async def test_trusted_index_limits_remain_implicit(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "pdf" / "SKILL.md",
                name="pdf",
                description="PDF guidance.",
            )
            registry = await _registry((_config(root),))
            trusted_index = await build_skill_match_index(
                registry,
                match_limits=SkillMatchLimits(max_query_characters=1),
            )

            result = await match_skill_registry(
                registry,
                query="pdf",
                index=trusted_index,
            )

            self.assertEqual(result.status, SkillStatus.EMPTY)
            self.assertEqual(
                result.diagnostics[0].code,
                SkillDiagnosticCode.NO_MATCH,
            )
            self.assertEqual(
                result.diagnostics[1].code,
                SkillDiagnosticCode.RESOURCE_OVERSIZED,
            )

    async def test_explicit_index_token_limit_keeps_trusted_index_valid(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "pdf" / "SKILL.md",
                name="pdf",
                description="PDF guidance.",
            )
            registry = await _registry((_config(root),))
            trusted_index = await build_skill_match_index(registry)

            result = await match_skill_registry(
                registry,
                query="pdf",
                index=trusted_index,
                match_limits=SkillMatchLimits(max_index_tokens_per_skill=1),
            )

            self.assertEqual(result.status, SkillStatus.OK)
            self.assertEqual(_match_item(result).metadata.skill_id, "pdf")
            self.assertEqual(result.diagnostics, ())

    async def test_oversized_untrusted_index_is_bounded_and_reconciled(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "pdf" / "SKILL.md",
                name="pdf",
                description="PDF guidance.",
            )
            registry = await _registry((_config(root),))
            oversized_entries = tuple(
                _entry(
                    SkillMetadata(
                        skill_id=f"evil-{index}",
                        name=f"evil-{index}",
                        description="Forged guidance.",
                        source_label="workspace-main",
                    )
                )
                for index in range(registry.index_limits.max_skills + 1)
            )
            supplied_index = SkillMatchIndex(
                registry_version=registry.registry_version,
                entries=oversized_entries,
            )

            result = await match_skill_registry(
                registry,
                query="pdf",
                index=supplied_index,
            )

            self.assertEqual(result.status, SkillStatus.OK)
            self.assertEqual(_match_item(result).metadata.skill_id, "pdf")
            self.assertEqual(
                result.diagnostics[0].code,
                SkillDiagnosticCode.RESOURCE_OVERSIZED,
            )
            self.assertEqual(
                result.diagnostics[0].details["reason"],
                "max_supplied_index_entries_exceeded",
            )

    async def test_index_limits_bound_oversized_inputs(
        self,
    ) -> None:
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
            registry = await _registry((_config(root),))

            index = await build_skill_match_index(
                registry,
                index_limits=registry.index_limits.__class__(max_skills=1),
            )
            result = await match_skill_registry(
                registry,
                query="one",
                index=index,
            )

            self.assertEqual(index.status, SkillStatus.TRUNCATED)
            self.assertEqual(index.entries, ())
            self.assertEqual(
                index.diagnostics[0].code,
                SkillDiagnosticCode.RESOURCE_OVERSIZED,
            )
            self.assertEqual(
                index.diagnostics[0].details["reason"],
                "max_skills_exceeded",
            )
            self.assertEqual(result.status, SkillStatus.TRUNCATED)
            self.assertEqual(
                result.diagnostics[0].code,
                SkillDiagnosticCode.RESOURCE_OVERSIZED,
            )

    async def test_excerpt_index_bounds_and_read_failures_are_diagnostic(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "pdf" / "SKILL.md",
                name="pdf",
                description="PDF guidance.",
                resources='["references/extra.md"]',
                body="# Body\nalpha beta gamma delta epsilon\n",
            )
            _write_text(
                root / "pdf" / "references" / "extra.md",
                "zeta eta theta\n",
            )
            _write_skill(
                root / "docx" / "SKILL.md",
                name="docx",
                description="DOCX guidance.",
                body="# Body\none two three\n",
            )
            registry = await _registry((_config(root),))

            too_many_resources = await build_skill_match_index(
                registry,
                index_limits=SkillIndexLimits(max_resources_per_skill=1),
            )
            exhausted_global = await build_skill_match_index(
                registry,
                include_resource_excerpts=True,
                index_limits=SkillIndexLimits(max_indexed_bytes=1),
                match_limits=SkillMatchLimits(max_excerpt_bytes_per_skill=1),
            )
            stopped_at_skill_limit = await build_skill_match_index(
                registry,
                include_resource_excerpts=True,
                match_limits=SkillMatchLimits(max_excerpt_bytes_per_skill=1),
            )
            token_limited = await build_skill_match_index(
                registry,
                include_resource_excerpts=True,
                match_limits=SkillMatchLimits(max_index_tokens_per_skill=1),
            )
            unavailable = await build_skill_match_index(
                registry,
                include_resource_excerpts=True,
                file_system=FailingReadFileSystem(),
            )

            self.assertEqual(too_many_resources.status, SkillStatus.TRUNCATED)
            self.assertEqual(
                too_many_resources.diagnostics[0].details["reason"],
                "max_resources_per_skill_exceeded",
            )
            self.assertIn("pdf", too_many_resources.diagnostics[0].candidates)
            self.assertEqual(exhausted_global.status, SkillStatus.TRUNCATED)
            self.assertEqual(
                exhausted_global.diagnostics[0].details["reason"],
                "max_indexed_bytes_exceeded",
            )
            self.assertEqual(stopped_at_skill_limit.indexed_bytes, 2)
            self.assertEqual(token_limited.status, SkillStatus.OK)
            self.assertEqual(unavailable.status, SkillStatus.UNAVAILABLE)
            self.assertEqual(
                unavailable.diagnostics[0].code,
                SkillDiagnosticCode.SOURCE_UNAVAILABLE,
            )

    async def test_bare_empty_registry_gets_match_index_diagnostic(
        self,
    ) -> None:
        registry = SkillRegistry(
            registry_version=SkillRegistryVersion(
                value="skills-registry:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
            ),
            read_limits=SkillReadLimits(),
            index_limits=SkillIndexLimits(),
        )

        index = await build_skill_match_index(registry)

        self.assertEqual(index.status, SkillStatus.EMPTY)
        self.assertEqual(
            index.diagnostics[0].code,
            SkillDiagnosticCode.EMPTY_REGISTRY,
        )

    async def test_unavailable_source_filter_requires_tied_diagnostic(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory)
            valid = base / "valid"
            missing = base / "missing"
            _write_skill(
                valid / "pdf" / "SKILL.md",
                name="pdf",
                description="PDF guidance.",
            )
            registry = await _registry(
                (
                    SkillConfiguredSource(
                        label="Workspace Main",
                        authority=WorkspaceSkillSourceAuthority(),
                        root_path=valid,
                    ),
                    SkillConfiguredSource(
                        label="Missing Main",
                        authority=WorkspaceSkillSourceAuthority(),
                        root_path=missing,
                    ),
                )
            )

            result = await match_skill_registry(
                registry,
                query="pdf",
                source_label="missing-main",
            )
            healthy_filtered = await match_skill_registry(
                registry,
                query="docx",
                source_label="workspace-main",
            )

            self.assertEqual(result.status, SkillStatus.EMPTY)
            self.assertEqual(
                result.diagnostics[0].code,
                SkillDiagnosticCode.NO_MATCH,
            )
            self.assertEqual(healthy_filtered.status, SkillStatus.EMPTY)
            self.assertEqual(
                healthy_filtered.diagnostics[0].code,
                SkillDiagnosticCode.NO_MATCH,
            )

    async def test_large_query_scanning_is_bounded_and_diagnostic(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "pdf" / "SKILL.md",
                name="pdf",
                description="PDF guidance.",
            )
            registry = await _registry((_config(root),))
            query = ExplodingLowerString("pdf " + ("ignored " * 20_000))

            result = await match_skill_registry(
                registry,
                query=query,
                match_limits=SkillMatchLimits(max_query_characters=3),
            )

            self.assertEqual(result.status, SkillStatus.OK)
            self.assertEqual(_match_item(result).metadata.skill_id, "pdf")
            self.assertEqual(
                result.diagnostics[0].code,
                SkillDiagnosticCode.RESOURCE_OVERSIZED,
            )
            self.assertEqual(
                result.diagnostics[0].details["reason"],
                "max_query_characters_exceeded",
            )

    async def test_concurrent_matches_are_deterministic(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "pdf-basic" / "SKILL.md",
                name="pdf-basic",
                description="Basic PDF handling.",
            )
            _write_skill(
                root / "pdf-advanced" / "SKILL.md",
                name="pdf-advanced",
                description="Advanced PDF handling.",
            )
            registry = await _registry((_config(root),))
            index = await build_skill_match_index(registry)

            results = await gather(
                *(
                    match_skill_registry(
                        registry,
                        query="pdf",
                        index=index,
                    )
                    for _ in range(12)
                )
            )
            encoded = tuple(
                dumps(result.as_model_dict(), sort_keys=True)
                for result in results
            )

            self.assertEqual(len(set(encoded)), 1)


class SkillMatcherPhase5EntityTest(TestCase):
    def test_match_entities_are_model_safe_and_immutable(self) -> None:
        metadata = _metadata()
        entry = _entry(metadata)
        index = SkillMatchIndex(
            registry_version=SkillRegistryVersion(
                value="skills-registry:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            ),
            entries=(entry,),
            indexed_bytes=entry.indexed_excerpt_bytes,
        )
        filters = SkillMatchFilters(
            query="PDF reader",
            tags=("pdf",),
            source_label="workspace-main",
            status=SkillStatus.OK,
            usable_only=False,
        )
        encoded = dumps(
            {
                "filters": filters.as_model_dict(),
                "index": index.as_model_dict(),
                "limits": SkillMatchLimits().as_model_dict(),
            },
            sort_keys=True,
        )

        self.assertEqual(index.status, SkillStatus.OK)
        self.assertIn("query_tokens", filters.as_model_dict())
        self.assertIn("pdf", encoded)
        self.assertNotIn("secret body", encoded)
        with self.assertRaises(FrozenInstanceError):
            setattr(entry, "usable", False)

    def test_empty_index_status_is_empty(self) -> None:
        index = SkillMatchIndex(
            registry_version=SkillRegistryVersion(
                value="skills-registry:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
            )
        )

        self.assertEqual(index.status, SkillStatus.EMPTY)
        self.assertEqual(index.as_model_dict()["entry_count"], 0)

    def test_entities_reject_invalid_values(self) -> None:
        metadata = _metadata()
        invalid_tags: Any = ["pdf"]
        invalid_entries: Any = [_entry(metadata)]

        cases: tuple[Callable[[], object], ...] = (
            lambda: SkillMatchLimits(max_results=0),
            lambda: SkillMatchLimits(max_query_tokens=True),
            lambda: SkillMatchLimits(max_query_characters=0),
            lambda: SkillMatchLimits(max_index_tokens_per_skill=0),
            lambda: SkillMatchLimits(max_excerpt_bytes_per_skill=0),
            lambda: SkillMatchFilters(query="bad\x00query"),
            lambda: SkillMatchFilters(tags=("bad/path",)),
            lambda: SkillMatchFilters(tags=invalid_tags),
            lambda: SkillMatchFilters(source_label="bad/path"),
            lambda: SkillMatchIndexEntry(
                metadata=metadata,
                status=SkillStatus.DISABLED,
                usable=True,
                id_tokens=("pdf",),
                name_tokens=("pdf",),
                tag_tokens=(),
                source_tokens=("workspace",),
                description_tokens=("guidance",),
            ),
            lambda: SkillMatchIndexEntry(
                metadata=metadata,
                status=SkillStatus.OK,
                usable=True,
                id_tokens=("bad/path",),
                name_tokens=("pdf",),
                tag_tokens=(),
                source_tokens=("workspace",),
                description_tokens=("guidance",),
            ),
            lambda: SkillMatchIndexEntry(
                metadata=metadata,
                status=SkillStatus.OK,
                usable=True,
                id_tokens=("pdf",),
                name_tokens=("pdf",),
                tag_tokens=(),
                source_tokens=("workspace",),
                description_tokens=("guidance",),
                indexed_excerpt_bytes=-1,
            ),
            lambda: SkillMatchIndex(
                registry_version=SkillRegistryVersion(
                    value="skills-registry:cccccccccccccccccccccccccccccccc"
                ),
                entries=(_entry(metadata),),
                indexed_bytes=999,
            ),
            lambda: SkillMatchIndex(
                registry_version=SkillRegistryVersion(
                    value="skills-registry:dddddddddddddddddddddddddddddddd"
                ),
                entries=invalid_entries,
            ),
        )

        for case in cases:
            with self.subTest(case=case):
                with self.assertRaises(AssertionError):
                    case()


async def _registry(
    configs: tuple[SkillConfiguredSource, ...],
) -> SkillRegistry:
    return await build_skill_registry(await resolve_skill_sources(configs))


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
    resources: str = "[]",
    enabled: bool = True,
    body: str = "# Body\n",
) -> None:
    tag_line = ""
    if tags:
        tag_line = "tags: [" + ", ".join(f'"{tag}"' for tag in tags) + "]\n"
    _write_text(
        path,
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        f"enabled: {'true' if enabled else 'false'}\n"
        f"{tag_line}"
        f"resources: {resources}\n"
        "---\n"
        f"{body}",
    )


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _metadata() -> SkillMetadata:
    return SkillMetadata(
        skill_id="pdf",
        name="pdf",
        description="PDF guidance.",
        source_label="workspace-main",
        tags=("pdf",),
    )


def _entry(metadata: SkillMetadata) -> SkillMatchIndexEntry:
    return SkillMatchIndexEntry(
        metadata=metadata,
        status=SkillStatus.OK,
        usable=True,
        id_tokens=("pdf",),
        name_tokens=("pdf",),
        tag_tokens=("pdf",),
        source_tokens=("workspace-main", "workspace", "main"),
        description_tokens=("pdf", "guidance"),
        indexed_excerpt_bytes=0,
    )


def _match_item(
    envelope: SkillResponseEnvelope,
    index: int = 0,
) -> SkillMatchResult:
    return _match_items(envelope)[index]


def _match_items(
    envelope: SkillResponseEnvelope,
) -> tuple[SkillMatchResult, ...]:
    assert all(isinstance(item, SkillMatchResult) for item in envelope.items)
    return cast(tuple[SkillMatchResult, ...], envelope.items)


def _match_skill_ids(envelope: SkillResponseEnvelope) -> tuple[str, ...]:
    return tuple(item.metadata.skill_id for item in _match_items(envelope))


class FailingReadFileSystem(SkillAsyncFileSystem):
    async def read_bytes(self, path: Path, limit: int) -> bytes:
        raise OSError("read failed")


class ExplodingLowerString(str):
    def __getitem__(self, key: int | slice) -> str:
        value = super().__getitem__(key)
        return str(value)

    def lower(self) -> str:
        raise AssertionError("unbounded query lower should not be called")


if __name__ == "__main__":
    main()
