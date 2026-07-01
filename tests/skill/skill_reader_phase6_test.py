from asyncio import CancelledError, create_task, gather, sleep
from dataclasses import replace
from json import dumps
from os import stat_result
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import patch

from avalan.skill import (
    SkillConfiguredSource,
    SkillCursorLimits,
    SkillDiagnosticCode,
    SkillDiagnosticInfo,
    SkillReadLimits,
    SkillRegistry,
    SkillResourceReader,
    SkillStatus,
    TrustedSkillSettings,
    WorkspaceSkillSourceAuthority,
    build_skill_registry,
    check_skill_registry_read,
    read_skill_registry_resource,
    resolve_skill_sources,
)
from avalan.skill import reader as reader_module
from avalan.skill.resolver import SkillAsyncFileSystem


class SkillReaderPhase6Test(IsolatedAsyncioTestCase):
    async def test_reads_main_and_declared_secondary_with_provenance(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "pdf" / "SKILL.md",
                name="pdf",
                description="PDF guidance.",
                resources='["references/rendering.md"]',
                body="# PDF\nUse pages.\n",
            )
            _write_text(
                root / "pdf" / "references" / "rendering.md",
                "Render pages.\n",
            )
            registry = await _registry(root)
            reader = SkillResourceReader()

            main_read = await reader.read(registry, "PDF")
            labeled_read = await reader.read(
                registry,
                "pdf",
                source_label="workspace-main",
            )
            secondary_read = await read_skill_registry_resource(
                registry,
                "pdf",
                resource_id="references/rendering.md",
                reader=reader,
            )
            checked = await check_skill_registry_read(
                registry,
                "pdf",
                reader=reader,
            )

            self.assertEqual(main_read.status, SkillStatus.OK)
            self.assertIsNotNone(main_read.content)
            assert main_read.content is not None
            self.assertIn("# PDF", main_read.content.text)
            self.assertEqual(labeled_read.status, SkillStatus.OK)
            self.assertEqual(secondary_read.status, SkillStatus.OK)
            self.assertIsNotNone(secondary_read.content)
            assert secondary_read.content is not None
            self.assertEqual(secondary_read.content.text, "Render pages.\n")
            self.assertEqual(checked.status, SkillStatus.OK)
            provenance = main_read.provenance[0].as_model_dict()
            self.assertEqual(
                provenance["registry_version"],
                registry.registry_version.value,
            )
            self.assertEqual(provenance["source_label"], "workspace-main")
            self.assertEqual(provenance["skill_id"], "pdf")
            self.assertEqual(provenance["resource_id"], "main")
            self.assertEqual(provenance["authority"], "workspace")
            self.assertFalse(provenance["truncated"])
            self.assertEqual(
                provenance["declared_follow_up_resources"],
                ("references/rendering.md",),
            )
            self.assertIn("content_sha256_prefix", provenance)
            self.assertNotIn(str(root), dumps(main_read.as_model_dict()))

    async def test_cursor_continuation_reports_truncation_and_hash(
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
                root / "pdf" / "references" / "rendering.md",
                "one\ntwo\nthree\nfour\n",
            )
            registry = await _registry(root)
            reader = SkillResourceReader()

            first = await reader.read(
                registry,
                "pdf",
                resource_id="references/rendering.md",
                read_limits=SkillReadLimits(
                    max_bytes_per_read=128,
                    max_lines_per_read=2,
                ),
            )
            assert first.next_cursor is not None
            second = await reader.read(
                registry,
                cursor_id=first.next_cursor.cursor_id,
            )

            self.assertEqual(first.status, SkillStatus.TRUNCATED)
            self.assertEqual(
                first.diagnostics[0].code,
                SkillDiagnosticCode.RESOURCE_OVERSIZED,
            )
            self.assertIsNotNone(first.content)
            assert first.content is not None
            self.assertEqual(first.content.text, "one\ntwo\n")
            self.assertTrue(first.provenance[0].truncated)
            self.assertEqual(second.status, SkillStatus.OK)
            self.assertIsNotNone(second.content)
            assert second.content is not None
            self.assertEqual(second.content.text, "three\nfour\n")
            self.assertIsNone(second.next_cursor)
            self.assertIn(
                "content_sha256_prefix",
                first.provenance[0].as_model_dict(),
            )

    async def test_cursor_continuation_honors_allow_cursor_false(
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
                root / "pdf" / "references" / "rendering.md",
                "one\ntwo\nthree\n",
            )
            registry = await _registry(root)
            reader = SkillResourceReader()
            limits = SkillReadLimits(
                max_bytes_per_read=128,
                max_lines_per_read=1,
            )

            first = await reader.read(
                registry,
                "pdf",
                resource_id="references/rendering.md",
                read_limits=limits,
            )
            assert first.next_cursor is not None
            blocked = await reader.read(
                registry,
                cursor_id=first.next_cursor.cursor_id,
                allow_cursor=False,
            )

            self.assertEqual(blocked.status, SkillStatus.TRUNCATED)
            self.assertEqual(
                blocked.diagnostics[0].code,
                SkillDiagnosticCode.RESOURCE_OVERSIZED,
            )
            self.assertIsNone(blocked.content)
            self.assertIsNone(blocked.next_cursor)

    async def test_helper_owned_reader_does_not_return_unusable_cursor(
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
                root / "pdf" / "references" / "rendering.md",
                "one\ntwo\n",
            )
            registry = await _registry(root)

            result = await read_skill_registry_resource(
                registry,
                "pdf",
                resource_id="references/rendering.md",
                read_limits=SkillReadLimits(
                    max_bytes_per_read=128,
                    max_lines_per_read=1,
                ),
            )

            self.assertEqual(result.status, SkillStatus.TRUNCATED)
            self.assertEqual(
                result.diagnostics[0].code,
                SkillDiagnosticCode.RESOURCE_OVERSIZED,
            )
            self.assertEqual(
                result.diagnostics[0].hint,
                "Restart the read with cursor support or wider trusted "
                "limits.",
            )
            self.assertIsNone(result.content)
            self.assertIsNone(result.next_cursor)

    async def test_cursor_limits_evict_and_expire_deterministically(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "pdf" / "SKILL.md",
                name="pdf",
                description="PDF guidance.",
                resources='["references/one.md", "references/two.md"]',
            )
            _write_text(root / "pdf" / "references" / "one.md", "a\nb\n")
            _write_text(root / "pdf" / "references" / "two.md", "c\nd\n")
            settings = TrustedSkillSettings(
                cursor_limits=SkillCursorLimits(
                    max_active_cursors=1,
                    max_cursor_age_seconds=5,
                )
            )
            registry = await _registry(root, settings=settings)
            now = [0.0]
            reader = SkillResourceReader(clock=lambda: now[0])
            limits = SkillReadLimits(
                max_bytes_per_read=128,
                max_lines_per_read=1,
            )

            first = await reader.read(
                registry,
                "pdf",
                resource_id="references/one.md",
                read_limits=limits,
            )
            second = await reader.read(
                registry,
                "pdf",
                resource_id="references/two.md",
                read_limits=limits,
            )
            assert first.next_cursor is not None
            assert second.next_cursor is not None
            evicted = await reader.read(
                registry,
                cursor_id=first.next_cursor.cursor_id,
            )
            now[0] = 6.0
            expired = await reader.read(
                registry,
                cursor_id=second.next_cursor.cursor_id,
            )

            self.assertEqual(reader.active_cursor_count, 0)
            self.assertEqual(evicted.status, SkillStatus.NOT_FOUND)
            self.assertEqual(
                evicted.diagnostics[0].code,
                SkillDiagnosticCode.NOT_FOUND,
            )
            self.assertEqual(expired.status, SkillStatus.STALE)
            self.assertEqual(
                expired.diagnostics[0].code,
                SkillDiagnosticCode.RESOURCE_STALE,
            )

    async def test_cursor_lookup_precedes_continuation_time_limit_checks(
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
                root / "pdf" / "references" / "rendering.md",
                "one\ntwo\n",
            )
            settings = TrustedSkillSettings(
                read_limits=SkillReadLimits(max_bytes_per_read=256),
                cursor_limits=SkillCursorLimits(max_cursor_age_seconds=1),
            )
            registry = await _registry(root, settings=settings)
            now = [0.0]
            reader = SkillResourceReader(clock=lambda: now[0])
            first = await reader.read(
                registry,
                "pdf",
                resource_id="references/rendering.md",
                read_limits=SkillReadLimits(
                    max_bytes_per_read=128,
                    max_lines_per_read=1,
                ),
            )
            assert first.next_cursor is not None
            now[0] = 2.0

            expired = await reader.read(
                registry,
                cursor_id=first.next_cursor.cursor_id,
                read_limits=SkillReadLimits(max_bytes_per_read=257),
                cursor_limits=SkillCursorLimits(max_cursor_age_seconds=2),
            )

            self.assertEqual(expired.status, SkillStatus.STALE)
            self.assertEqual(
                expired.diagnostics[0].details["reason"],
                "cursor_expired",
            )
            self.assertEqual(reader.active_cursor_count, 0)

    async def test_cursor_continuations_validate_stored_state_and_limits(
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
                root / "pdf" / "references" / "rendering.md",
                "one\ntwo\nthree\n",
            )
            settings = TrustedSkillSettings(
                read_limits=SkillReadLimits(max_bytes_per_read=256),
                cursor_limits=SkillCursorLimits(max_active_cursors=1),
            )
            registry = await _registry(root, settings=settings)
            reader = SkillResourceReader()

            first = await reader.read(
                registry,
                "pdf",
                resource_id="references/rendering.md",
                read_limits=SkillReadLimits(
                    max_bytes_per_read=128,
                    max_lines_per_read=2,
                ),
            )
            assert first.next_cursor is not None
            narrowed_registry = replace(
                registry,
                read_limits=SkillReadLimits(
                    max_bytes_per_read=256,
                    max_lines_per_read=1,
                ),
            )
            narrowed = await reader.read(
                narrowed_registry,
                cursor_id=first.next_cursor.cursor_id,
            )

            second = await reader.read(
                registry,
                "pdf",
                resource_id="references/rendering.md",
                read_limits=SkillReadLimits(
                    max_bytes_per_read=128,
                    max_lines_per_read=2,
                ),
            )
            assert second.next_cursor is not None
            accepted_without_new_cursor = await reader.read(
                registry,
                cursor_id=second.next_cursor.cursor_id,
                cursor_limits=SkillCursorLimits(max_active_cursors=2),
            )

            third = await reader.read(
                registry,
                "pdf",
                resource_id="references/rendering.md",
                read_limits=SkillReadLimits(
                    max_bytes_per_read=128,
                    max_lines_per_read=1,
                ),
            )
            assert third.next_cursor is not None
            denied_when_new_cursor_needed = await reader.read(
                registry,
                cursor_id=third.next_cursor.cursor_id,
                cursor_limits=SkillCursorLimits(max_active_cursors=2),
            )

            self.assertEqual(narrowed.status, SkillStatus.POLICY_DENIED)
            self.assertEqual(
                narrowed.diagnostics[0].details["reason"],
                "read_limits_exceeded",
            )
            self.assertEqual(
                accepted_without_new_cursor.status,
                SkillStatus.OK,
            )
            self.assertIsNone(accepted_without_new_cursor.next_cursor)
            self.assertEqual(
                denied_when_new_cursor_needed.status,
                SkillStatus.POLICY_DENIED,
            )
            self.assertEqual(
                denied_when_new_cursor_needed.diagnostics[0].details["reason"],
                "cursor_limits_exceeded",
            )
            self.assertIsNone(denied_when_new_cursor_needed.content)

    async def test_cursor_target_changes_are_stale_and_dropped(
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
                root / "pdf" / "references" / "rendering.md",
                "one\ntwo\n",
            )
            registry = await _registry(root)
            limits = SkillReadLimits(
                max_bytes_per_read=128,
                max_lines_per_read=1,
            )

            missing_skill_reader = SkillResourceReader()
            missing_skill_first = await missing_skill_reader.read(
                registry,
                "pdf",
                resource_id="references/rendering.md",
                read_limits=limits,
            )
            assert missing_skill_first.next_cursor is not None
            missing_skill = await missing_skill_reader.read(
                replace(registry, skills=()),
                cursor_id=missing_skill_first.next_cursor.cursor_id,
            )

            missing_resource_reader = SkillResourceReader()
            missing_resource_first = await missing_resource_reader.read(
                registry,
                "pdf",
                resource_id="references/rendering.md",
                read_limits=limits,
            )
            assert missing_resource_first.next_cursor is not None
            skill_without_secondary = replace(
                registry.skills[0],
                resources=tuple(
                    resource
                    for resource in registry.skills[0].resources
                    if resource.handle.resource_id != "references/rendering.md"
                ),
            )
            missing_resource = await missing_resource_reader.read(
                replace(registry, skills=(skill_without_secondary,)),
                cursor_id=missing_resource_first.next_cursor.cursor_id,
            )

            unusable_reader = SkillResourceReader()
            unusable_first = await unusable_reader.read(
                registry,
                "pdf",
                resource_id="references/rendering.md",
                read_limits=limits,
            )
            assert unusable_first.next_cursor is not None
            diagnostic = SkillDiagnosticInfo(
                code=SkillDiagnosticCode.POLICY_DENIED,
                status=SkillStatus.POLICY_DENIED,
                message="The requested skill is blocked.",
                path="skills.read",
                hint="Use only usable skills.",
                details={"reason": "blocked_for_test"},
            )
            unusable_skill = replace(
                registry.skills[0],
                usable=False,
                diagnostics=(diagnostic,),
            )
            unusable = await unusable_reader.read(
                replace(registry, skills=(unusable_skill,)),
                cursor_id=unusable_first.next_cursor.cursor_id,
            )

            self.assertEqual(missing_skill.status, SkillStatus.STALE)
            self.assertEqual(
                missing_skill.diagnostics[0].details["reason"],
                "skill_removed",
            )
            self.assertEqual(missing_skill_reader.active_cursor_count, 0)
            self.assertEqual(missing_resource.status, SkillStatus.STALE)
            self.assertEqual(
                missing_resource.diagnostics[0].details["reason"],
                "resource_removed",
            )
            self.assertEqual(missing_resource_reader.active_cursor_count, 0)
            self.assertEqual(unusable.status, SkillStatus.POLICY_DENIED)
            self.assertEqual(unusable_reader.active_cursor_count, 0)

    async def test_unknown_cursor_prunes_expired_stored_state(
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
                root / "pdf" / "references" / "rendering.md",
                "one\ntwo\n",
            )
            registry = await _registry(root)
            now = [0.0]
            reader = SkillResourceReader(clock=lambda: now[0])

            first = await reader.read(
                registry,
                "pdf",
                resource_id="references/rendering.md",
                read_limits=SkillReadLimits(
                    max_bytes_per_read=128,
                    max_lines_per_read=1,
                ),
                cursor_limits=SkillCursorLimits(max_cursor_age_seconds=1),
            )
            assert first.next_cursor is not None
            self.assertEqual(reader.active_cursor_count, 1)
            now[0] = 2.0

            missing = await reader.read(
                registry,
                cursor_id="skill-cursor:unknown",
            )

            self.assertEqual(missing.status, SkillStatus.NOT_FOUND)
            self.assertEqual(reader.active_cursor_count, 0)

    async def test_cursor_ids_are_random_and_retry_collisions(
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
                root / "pdf" / "references" / "rendering.md",
                "one\ntwo\n",
            )
            registry = await _registry(root)
            reader = SkillResourceReader()
            limits = SkillReadLimits(
                max_bytes_per_read=128,
                max_lines_per_read=1,
            )

            first = await reader.read(
                registry,
                "pdf",
                resource_id="references/rendering.md",
                read_limits=limits,
            )
            assert first.next_cursor is not None
            existing_token = first.next_cursor.cursor_id.split(":", 1)[1]
            calls: list[int | None] = []

            def fake_token_hex(byte_count: int | None = None) -> str:
                calls.append(byte_count)
                if len(calls) == 1:
                    return existing_token
                return "a" * 32

            with patch.object(
                reader_module,
                "token_hex",
                side_effect=fake_token_hex,
            ):
                second = await reader.read(
                    registry,
                    "pdf",
                    resource_id="references/rendering.md",
                    read_limits=limits,
                )

            assert second.next_cursor is not None
            self.assertEqual(calls, [16, 16])
            self.assertEqual(
                second.next_cursor.cursor_id,
                f"skill-cursor:{'a' * 32}",
            )
            self.assertNotEqual(
                second.next_cursor.cursor_id,
                first.next_cursor.cursor_id,
            )

    async def test_cursor_expiry_uses_creation_deadline_not_continuation_limit(
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
                root / "pdf" / "references" / "rendering.md",
                "one\ntwo\n",
            )
            registry = await _registry(root)
            now = [0.0]
            reader = SkillResourceReader(clock=lambda: now[0])

            first = await reader.read(
                registry,
                "pdf",
                resource_id="references/rendering.md",
                read_limits=SkillReadLimits(
                    max_bytes_per_read=128,
                    max_lines_per_read=1,
                ),
                cursor_limits=SkillCursorLimits(max_cursor_age_seconds=1),
            )
            assert first.next_cursor is not None
            now[0] = 2.0

            expired = await reader.read(
                registry,
                cursor_id=first.next_cursor.cursor_id,
            )

            self.assertEqual(expired.status, SkillStatus.STALE)
            self.assertEqual(
                expired.diagnostics[0].code,
                SkillDiagnosticCode.RESOURCE_STALE,
            )
            self.assertEqual(
                expired.diagnostics[0].details["reason"],
                "cursor_expired",
            )

    async def test_rejects_skill_lookup_policy_and_resource_errors(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "pdf" / "SKILL.md",
                name="pdf",
                description="PDF guidance.",
            )
            _write_skill(
                root / "disabled" / "SKILL.md",
                name="disabled",
                description="Disabled guidance.",
                enabled=False,
            )
            _write_text(
                root / "malformed" / "SKILL.md",
                "---\n"
                "name: malformed\n"
                "description: Broken guidance.\n"
                'tags: ["bad/path"]\n'
                "---\n",
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
            reader = SkillResourceReader()

            unknown = await reader.read(registry, "missing")
            ambiguous = await reader.read(registry, "duplicate")
            disabled = await reader.read(registry, "disabled")
            malformed = await reader.read(registry, "malformed")
            traversal = await reader.read(
                registry,
                "pdf",
                resource_id="../secret.md",
            )
            hidden = await reader.read(
                registry,
                "pdf",
                resource_id=".hidden.md",
            )
            undeclared = await reader.read(
                registry,
                "pdf",
                resource_id="references/missing.md",
            )
            private_key = await reader.read(
                registry,
                "pdf",
                resource_id="private/key",
            )
            private_file = await reader.read(
                registry,
                "pdf",
                resource_id="private/file.md",
            )
            invalid_cursor = await reader.read(registry, cursor_id="../bad")
            unsafe_absolute = await reader.read(
                registry,
                "/Users/mariano/secret",
            )
            unsafe_traversal = await reader.read(registry, "../x")
            unsafe_nul = await reader.read(registry, "bad\x00skill")
            settings_disabled = await reader.read(
                replace(
                    registry,
                    settings=TrustedSkillSettings(enabled=False),
                ),
                "pdf",
            )

            policy_root = root / "policy"
            _write_skill(
                policy_root / "pdf" / "SKILL.md",
                name="pdf",
                description="PDF guidance.",
            )
            _write_skill(
                policy_root / "docx" / "SKILL.md",
                name="docx",
                description="DOCX guidance.",
            )
            policy_registry = await _registry(
                policy_root,
                settings=TrustedSkillSettings(allowed_skill_ids=("pdf",)),
            )
            policy_denied = await reader.read(policy_registry, "docx")

            self.assertEqual(unknown.status, SkillStatus.NOT_FOUND)
            self.assertEqual(ambiguous.status, SkillStatus.AMBIGUOUS)
            self.assertEqual(disabled.status, SkillStatus.DISABLED)
            self.assertEqual(malformed.status, SkillStatus.MALFORMED)
            self.assertEqual(traversal.status, SkillStatus.POLICY_DENIED)
            self.assertEqual(
                traversal.diagnostics[0].code,
                SkillDiagnosticCode.RESOURCE_OUTSIDE_ROOT,
            )
            self.assertEqual(hidden.status, SkillStatus.POLICY_DENIED)
            self.assertEqual(
                hidden.diagnostics[0].code,
                SkillDiagnosticCode.POLICY_DENIED,
            )
            self.assertEqual(undeclared.status, SkillStatus.NOT_FOUND)
            self.assertEqual(
                undeclared.diagnostics[0].code,
                SkillDiagnosticCode.RESOURCE_MISSING,
            )
            self.assertEqual(private_key.status, SkillStatus.NOT_FOUND)
            self.assertEqual(private_file.status, SkillStatus.NOT_FOUND)
            self.assertEqual(
                private_key.diagnostics[0].details["resource_id"],
                "resource/unsafe",
            )
            self.assertEqual(
                private_file.diagnostics[0].details["resource_id"],
                "resource/unsafe",
            )
            self.assertEqual(invalid_cursor.status, SkillStatus.NOT_FOUND)
            self.assertEqual(
                unsafe_absolute.status,
                SkillStatus.POLICY_DENIED,
            )
            self.assertEqual(
                unsafe_traversal.status,
                SkillStatus.POLICY_DENIED,
            )
            self.assertEqual(unsafe_nul.status, SkillStatus.POLICY_DENIED)
            self.assertEqual(settings_disabled.status, SkillStatus.DISABLED)
            self.assertEqual(policy_denied.status, SkillStatus.POLICY_DENIED)
            for result in (
                unknown,
                ambiguous,
                disabled,
                malformed,
                traversal,
                hidden,
                undeclared,
                private_key,
                private_file,
                invalid_cursor,
                unsafe_absolute,
                unsafe_traversal,
                unsafe_nul,
                settings_disabled,
                policy_denied,
            ):
                model = result.as_model_dict()
                _assert_model_safe(self, model, root)
                encoded = dumps(model, sort_keys=True)
                self.assertNotIn("/Users/mariano/secret", encoded)
                self.assertNotIn("../x", encoded)
                self.assertNotIn("\\u0000", encoded)
                self.assertNotIn("private/key", encoded)
                self.assertNotIn("private/file.md", encoded)

    async def test_skill_usability_fallback_diagnostics_are_structured(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "pdf" / "SKILL.md",
                name="pdf",
                description="PDF guidance.",
            )
            registry = await _registry(root)
            base_skill = registry.skills[0]
            assert base_skill.metadata is not None
            reader = SkillResourceReader()

            disabled_metadata_skill = replace(
                base_skill,
                metadata=replace(
                    base_skill.metadata,
                    enabled=False,
                    status=SkillStatus.DISABLED,
                ),
                usable=False,
                diagnostics=(),
            )
            disabled_metadata = await reader.read(
                replace(registry, skills=(disabled_metadata_skill,)),
                "pdf",
            )

            disabled_status_skill = replace(
                base_skill,
                status=SkillStatus.DISABLED,
                usable=False,
                diagnostics=(),
            )
            disabled_status = await reader.read(
                replace(registry, skills=(disabled_status_skill,)),
                "pdf",
            )

            malformed_status_skill = replace(
                base_skill,
                status=SkillStatus.MALFORMED,
                metadata=None,
                usable=False,
                diagnostics=(),
            )
            malformed_status = await reader.read(
                replace(registry, skills=(malformed_status_skill,)),
                "pdf",
            )

            registry_diagnostic = SkillDiagnosticInfo(
                code=SkillDiagnosticCode.POLICY_DENIED,
                status=SkillStatus.POLICY_DENIED,
                message="The registry marks this skill unusable.",
                path="skills.read",
                hint="Use a usable registry snapshot.",
                details={"reason": "registry_blocked"},
            )
            unusable_skill = replace(
                base_skill,
                usable=False,
                diagnostics=(),
            )
            registry_blocked = await reader.read(
                replace(
                    registry,
                    skills=(unusable_skill,),
                    diagnostics=(registry_diagnostic,),
                ),
                "pdf",
            )
            not_usable = await reader.read(
                replace(registry, skills=(unusable_skill,)),
                "pdf",
            )

            self.assertEqual(disabled_metadata.status, SkillStatus.DISABLED)
            self.assertEqual(disabled_status.status, SkillStatus.DISABLED)
            self.assertEqual(malformed_status.status, SkillStatus.MALFORMED)
            self.assertEqual(
                registry_blocked.status,
                SkillStatus.POLICY_DENIED,
            )
            self.assertEqual(not_usable.status, SkillStatus.POLICY_DENIED)
            self.assertEqual(
                registry_blocked.diagnostics[0].details["reason"],
                "registry_blocked",
            )
            self.assertEqual(
                not_usable.diagnostics[0].details["reason"],
                "skill_not_usable",
            )
            for result in (
                disabled_metadata,
                disabled_status,
                malformed_status,
                registry_blocked,
                not_usable,
            ):
                _assert_model_safe(self, result.as_model_dict(), root)

    async def test_rejects_stale_deleted_symlink_binary_and_oversize(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory)
            stale_root = base / "stale"
            _write_skill(
                stale_root / "pdf" / "SKILL.md",
                name="pdf",
                description="PDF guidance.",
            )
            stale_registry = await _registry(stale_root)
            _write_skill(
                stale_root / "pdf" / "SKILL.md",
                name="pdf",
                description="Changed guidance.",
            )
            reader = SkillResourceReader()

            stale = await reader.read(stale_registry, "pdf")
            (stale_root / "pdf" / "SKILL.md").unlink()
            deleted = await reader.read(stale_registry, "pdf")

            symlink_root = base / "symlink"
            outside = base / "outside.md"
            _write_skill(
                symlink_root / "pdf" / "SKILL.md",
                name="pdf",
                description="PDF guidance.",
            )
            outside.write_text("outside\n", encoding="utf-8")
            symlink_registry = await _registry(symlink_root)
            skill_path = symlink_root / "pdf" / "SKILL.md"
            skill_path.unlink()
            skill_path.symlink_to(outside)
            symlink = await reader.read(symlink_registry, "pdf")

            binary_root = base / "binary"
            _write_skill(
                binary_root / "pdf" / "SKILL.md",
                name="pdf",
                description="PDF guidance.",
            )
            binary_registry = await _registry(binary_root)
            binary = await reader.read(
                binary_registry,
                "pdf",
                file_system=MutatingReadFileSystem(b"\x00markdown"),
            )
            invalid_utf8 = await reader.read(
                binary_registry,
                "pdf",
                file_system=MutatingReadFileSystem(b"\xff"),
            )

            oversized_root = base / "oversized"
            _write_skill(
                oversized_root / "pdf" / "SKILL.md",
                name="pdf",
                description="PDF guidance.",
                resources='["references/rendering.md"]',
            )
            _write_text(
                oversized_root / "pdf" / "references" / "rendering.md",
                "one\ntwo\nthree\n",
            )
            oversized_registry = await _registry(oversized_root)
            oversized = await reader.read(
                oversized_registry,
                "pdf",
                resource_id="references/rendering.md",
                allow_cursor=False,
                read_limits=SkillReadLimits(
                    max_bytes_per_read=128,
                    max_lines_per_read=1,
                ),
            )

            self.assertEqual(stale.status, SkillStatus.STALE)
            self.assertIsNone(stale.content)
            self.assertNotIn("Changed guidance", dumps(stale.as_model_dict()))
            self.assertEqual(deleted.status, SkillStatus.NOT_FOUND)
            self.assertEqual(
                deleted.diagnostics[0].code,
                SkillDiagnosticCode.RESOURCE_MISSING,
            )
            self.assertEqual(symlink.status, SkillStatus.POLICY_DENIED)
            self.assertEqual(
                symlink.diagnostics[0].code,
                SkillDiagnosticCode.RESOURCE_OUTSIDE_ROOT,
            )
            self.assertEqual(
                symlink.diagnostics[0].details["reason"],
                "symlink_escape",
            )
            self.assertEqual(binary.status, SkillStatus.UNAVAILABLE)
            self.assertEqual(
                binary.diagnostics[0].code,
                SkillDiagnosticCode.BINARY_RESOURCE,
            )
            self.assertEqual(invalid_utf8.status, SkillStatus.UNAVAILABLE)
            self.assertEqual(
                invalid_utf8.diagnostics[0].code,
                SkillDiagnosticCode.BINARY_RESOURCE,
            )
            self.assertEqual(oversized.status, SkillStatus.TRUNCATED)
            self.assertIsNone(oversized.content)
            self.assertIsNone(oversized.next_cursor)
            self.assertEqual(
                oversized.diagnostics[0].code,
                SkillDiagnosticCode.RESOURCE_OVERSIZED,
            )
            for result in (
                stale,
                deleted,
                symlink,
                binary,
                invalid_utf8,
                oversized,
            ):
                _assert_model_safe(self, result.as_model_dict(), base)

    async def test_trusted_limits_source_labels_and_cursors_are_checked(
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
                root / "pdf" / "references" / "rendering.md",
                "one\ntwo\n",
            )
            settings = TrustedSkillSettings(
                read_limits=SkillReadLimits(max_bytes_per_read=256),
                cursor_limits=SkillCursorLimits(max_active_cursors=1),
            )
            registry = await _registry(root, settings=settings)
            reader = SkillResourceReader(
                read_limits=SkillReadLimits(max_bytes_per_read=128),
                cursor_limits=SkillCursorLimits(max_active_cursors=1),
            )

            read_limit_denied = await reader.read(
                registry,
                "pdf",
                read_limits=SkillReadLimits(max_bytes_per_read=257),
            )
            cursor_limit_denied = await reader.read(
                registry,
                "pdf",
                resource_id="references/rendering.md",
                read_limits=SkillReadLimits(
                    max_bytes_per_read=128,
                    max_lines_per_read=1,
                ),
                cursor_limits=SkillCursorLimits(max_active_cursors=2),
            )
            bad_source = await reader.read(
                registry,
                "pdf",
                source_label="bad/source",
            )
            missing_skill = await reader.read(registry)
            missing_check = await reader.check(registry, "missing")
            checked = await check_skill_registry_read(registry, "pdf")
            read = await read_skill_registry_resource(registry, "pdf")

            first = await reader.read(
                registry,
                "pdf",
                resource_id="references/rendering.md",
                read_limits=SkillReadLimits(
                    max_bytes_per_read=128,
                    max_lines_per_read=1,
                ),
            )
            assert first.next_cursor is not None
            _write_text(
                root / "pdf" / "references" / "rendering.md",
                "one\ntwo\nthree\n",
            )
            changed_registry = await _registry(root, settings=settings)
            stale_cursor = await reader.read(
                changed_registry,
                cursor_id=first.next_cursor.cursor_id,
            )
            deleted_registry = await _registry(root, settings=settings)
            (root / "pdf" / "SKILL.md").unlink()
            failed_check = await reader.check(deleted_registry, "pdf")

            self.assertEqual(
                read_limit_denied.status,
                SkillStatus.POLICY_DENIED,
            )
            self.assertEqual(
                cursor_limit_denied.status,
                SkillStatus.POLICY_DENIED,
            )
            self.assertEqual(bad_source.status, SkillStatus.NOT_FOUND)
            self.assertEqual(missing_skill.status, SkillStatus.NOT_FOUND)
            self.assertEqual(missing_check.status, SkillStatus.NOT_FOUND)
            self.assertEqual(checked.status, SkillStatus.OK)
            self.assertEqual(read.status, SkillStatus.OK)
            self.assertEqual(stale_cursor.status, SkillStatus.STALE)
            self.assertEqual(failed_check.status, SkillStatus.NOT_FOUND)

    async def test_runtime_path_and_read_races_are_structured(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory)
            path_root = base / "path"
            _write_skill(
                path_root / "pdf" / "SKILL.md",
                name="pdf",
                description="PDF guidance.",
            )
            path_registry = await _registry(path_root)
            reader = SkillResourceReader()

            lstat_unavailable = await reader.read(
                path_registry,
                "pdf",
                file_system=LstatFailureFileSystem(),
            )
            resolve_missing = await reader.read(
                path_registry,
                "pdf",
                file_system=ResolveMissingFileSystem(),
            )
            resolve_unavailable = await reader.read(
                path_registry,
                "pdf",
                file_system=ResolveFailureFileSystem(),
            )
            path_escape = await reader.read(
                path_registry,
                "pdf",
                file_system=ResolveEscapeFileSystem(),
            )
            stat_missing = await reader.read(
                path_registry,
                "pdf",
                file_system=StatMissingFileSystem(),
            )
            path_unavailable = await reader.read(
                path_registry,
                "pdf",
                file_system=StatFailureFileSystem(),
            )
            special_file = await reader.read(
                path_registry,
                "pdf",
                file_system=DirectoryStatFileSystem(),
            )
            check_unavailable = await reader.check(
                path_registry,
                "pdf",
                file_system=StatFailureFileSystem(),
            )
            read_missing = await reader.read(
                path_registry,
                "pdf",
                file_system=ReadMissingAfterCheckFileSystem(),
            )
            read_failure = await reader.read(
                path_registry,
                "pdf",
                file_system=ReadFailureAfterCheckFileSystem(),
            )
            size_changed = await reader.read(
                path_registry,
                "pdf",
                file_system=SizeRaceFileSystem(),
            )
            hash_changed = await reader.read(
                path_registry,
                "pdf",
                file_system=HashRaceFileSystem(),
            )

            unsafe_root = base / "unsafe"
            _write_skill(
                unsafe_root / "pdf" / "SKILL.md",
                name="pdf",
                description="PDF guidance.",
                body="/Users/mariano/secret\n",
            )
            unsafe_registry = await _registry(unsafe_root)
            unsafe_text = await reader.read(unsafe_registry, "pdf")

            utf8_root = base / "utf8"
            _write_skill(
                utf8_root / "pdf" / "SKILL.md",
                name="pdf",
                description="PDF guidance.",
                resources='["references/unicode.md"]',
            )
            _write_text(utf8_root / "pdf" / "references" / "unicode.md", "€")
            utf8_registry = await _registry(utf8_root)
            utf8_limit = await reader.read(
                utf8_registry,
                "pdf",
                resource_id="references/unicode.md",
                read_limits=SkillReadLimits(max_bytes_per_read=1),
            )

            boundary_root = base / "boundary"
            _write_skill(
                boundary_root / "pdf" / "SKILL.md",
                name="pdf",
                description="PDF guidance.",
                resources='["references/unsafe.md"]',
            )
            _write_text(
                boundary_root / "pdf" / "references" / "unsafe.md",
                "prefix /Users/mariano/secret\n",
            )
            boundary_registry = await _registry(boundary_root)
            boundary_unsafe = await reader.read(
                boundary_registry,
                "pdf",
                resource_id="references/unsafe.md",
                read_limits=SkillReadLimits(max_bytes_per_read=10),
            )

            self.assertEqual(lstat_unavailable.status, SkillStatus.UNAVAILABLE)
            self.assertEqual(resolve_missing.status, SkillStatus.NOT_FOUND)
            self.assertEqual(
                resolve_unavailable.status,
                SkillStatus.UNAVAILABLE,
            )
            self.assertEqual(path_escape.status, SkillStatus.POLICY_DENIED)
            self.assertEqual(
                path_escape.diagnostics[0].code,
                SkillDiagnosticCode.RESOURCE_OUTSIDE_ROOT,
            )
            self.assertEqual(stat_missing.status, SkillStatus.NOT_FOUND)
            self.assertEqual(
                path_unavailable.status,
                SkillStatus.UNAVAILABLE,
            )
            self.assertEqual(special_file.status, SkillStatus.POLICY_DENIED)
            self.assertEqual(check_unavailable.status, SkillStatus.UNAVAILABLE)
            self.assertEqual(read_missing.status, SkillStatus.NOT_FOUND)
            self.assertEqual(read_failure.status, SkillStatus.UNAVAILABLE)
            self.assertEqual(size_changed.status, SkillStatus.STALE)
            self.assertEqual(hash_changed.status, SkillStatus.STALE)
            self.assertEqual(unsafe_text.status, SkillStatus.POLICY_DENIED)
            self.assertEqual(utf8_limit.status, SkillStatus.TRUNCATED)
            self.assertIsNone(utf8_limit.content)
            self.assertEqual(
                boundary_unsafe.status,
                SkillStatus.POLICY_DENIED,
            )
            self.assertIsNone(boundary_unsafe.content)
            self.assertIsNone(boundary_unsafe.next_cursor)
            self.assertNotIn(
                "/Users/mariano/secret",
                dumps(boundary_unsafe.as_model_dict(), sort_keys=True),
            )
            for result in (
                lstat_unavailable,
                resolve_missing,
                resolve_unavailable,
                path_escape,
                stat_missing,
                path_unavailable,
                special_file,
                check_unavailable,
                read_missing,
                read_failure,
                size_changed,
                hash_changed,
                unsafe_text,
                utf8_limit,
                boundary_unsafe,
            ):
                _assert_model_safe(self, result.as_model_dict(), base)

    async def test_chunk_unsafe_window_does_not_leak_cursor_state(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "pdf" / "SKILL.md",
                name="pdf",
                description="PDF guidance.",
                resources='["references/slashes.md"]',
            )
            _write_text(
                root / "pdf" / "references" / "slashes.md",
                "a/b/c/d\n",
            )
            registry = await _registry(root)
            reader = SkillResourceReader()

            first = await reader.read(
                registry,
                "pdf",
                resource_id="references/slashes.md",
                read_limits=SkillReadLimits(max_bytes_per_read=1),
            )
            assert first.next_cursor is not None
            self.assertEqual(reader.active_cursor_count, 1)

            second = await reader.read(
                registry,
                cursor_id=first.next_cursor.cursor_id,
            )

            self.assertEqual(second.status, SkillStatus.POLICY_DENIED)
            self.assertEqual(
                second.diagnostics[0].code,
                SkillDiagnosticCode.POLICY_DENIED,
            )
            self.assertIsNone(second.content)
            self.assertIsNone(second.next_cursor)
            self.assertEqual(reader.active_cursor_count, 0)

    def test_read_window_bounds_return_structured_diagnostics(self) -> None:
        limits = SkillReadLimits(max_bytes_per_read=2)

        empty, empty_diagnostic = reader_module._read_window(
            b"abc",
            offset_bytes=3,
            read_limits=limits,
            resource_id="main",
        )
        missing, missing_diagnostic = reader_module._read_window(
            b"abc",
            offset_bytes=4,
            read_limits=limits,
            resource_id="main",
        )

        self.assertIsNone(empty_diagnostic)
        self.assertIsNotNone(empty)
        assert empty is not None
        self.assertEqual(empty.text, "")
        self.assertEqual(empty.start_byte, 3)
        self.assertFalse(empty.truncated)
        self.assertIsNone(missing)
        self.assertIsNotNone(missing_diagnostic)
        assert missing_diagnostic is not None
        self.assertEqual(missing_diagnostic.status, SkillStatus.NOT_FOUND)
        self.assertEqual(
            missing_diagnostic.details["reason"],
            "offset_out_of_bounds",
        )
        self.assertEqual(missing_diagnostic.details["resource_id"], "main")
        _assert_model_safe(
            self,
            missing_diagnostic.as_model_dict(),
            Path("/tmp/not-leaked"),
        )

    async def test_cancellation_and_concurrent_reads_are_bounded(
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
                root / "pdf" / "references" / "rendering.md",
                "Render pages.\n",
            )
            registry = await _registry(root)
            reader = SkillResourceReader()
            task = create_task(
                reader.read(
                    registry,
                    "pdf",
                    file_system=SlowReadFileSystem(),
                )
            )
            await sleep(0.01)
            task.cancel()

            with self.assertRaises(CancelledError):
                await task

            results = await gather(
                reader.read(registry, "pdf"),
                reader.read(
                    registry,
                    "pdf",
                    resource_id="references/rendering.md",
                ),
                reader.check(registry, "pdf"),
            )

            self.assertEqual(
                tuple(result.status for result in results),
                (SkillStatus.OK, SkillStatus.OK, SkillStatus.OK),
            )


class MutatingReadFileSystem(SkillAsyncFileSystem):
    def __init__(self, replacement: bytes) -> None:
        super().__init__()
        self._replacement = replacement
        self._reads = 0

    async def read_bytes(self, path: Path, limit: int) -> bytes:
        self._reads += 1
        if self._reads == 1:
            return await super().read_bytes(path, limit)
        return self._replacement[:limit]


class SlowReadFileSystem(SkillAsyncFileSystem):
    async def read_bytes(self, path: Path, limit: int) -> bytes:
        await sleep(10)
        return await super().read_bytes(path, limit)


class ResolveEscapeFileSystem(SkillAsyncFileSystem):
    async def resolve_path(self, path: Path) -> Path:
        return path.parent.parent / "escaped.md"


class LstatFailureFileSystem(SkillAsyncFileSystem):
    async def lstat_path(self, path: Path) -> stat_result:
        raise OSError("lstat failed")


class ResolveMissingFileSystem(SkillAsyncFileSystem):
    async def resolve_path(self, path: Path) -> Path:
        raise FileNotFoundError("missing")


class ResolveFailureFileSystem(SkillAsyncFileSystem):
    async def resolve_path(self, path: Path) -> Path:
        raise RuntimeError("resolve failed")


class StatMissingFileSystem(SkillAsyncFileSystem):
    async def stat_path(self, path: Path) -> stat_result:
        raise FileNotFoundError("missing")


class StatFailureFileSystem(SkillAsyncFileSystem):
    async def stat_path(self, path: Path) -> stat_result:
        raise OSError("stat failed")


class DirectoryStatFileSystem(SkillAsyncFileSystem):
    async def stat_path(self, path: Path) -> stat_result:
        return path.parent.stat()


class ReadMissingAfterCheckFileSystem(SkillAsyncFileSystem):
    def __init__(self) -> None:
        super().__init__()
        self._reads = 0

    async def read_bytes(self, path: Path, limit: int) -> bytes:
        self._reads += 1
        if self._reads == 1:
            return await super().read_bytes(path, limit)
        raise FileNotFoundError("missing")


class ReadFailureAfterCheckFileSystem(SkillAsyncFileSystem):
    def __init__(self) -> None:
        super().__init__()
        self._reads = 0

    async def read_bytes(self, path: Path, limit: int) -> bytes:
        self._reads += 1
        if self._reads == 1:
            return await super().read_bytes(path, limit)
        raise OSError("read failed")


class SizeRaceFileSystem(SkillAsyncFileSystem):
    def __init__(self) -> None:
        super().__init__()
        self._reads = 0

    async def read_bytes(self, path: Path, limit: int) -> bytes:
        self._reads += 1
        content = await super().read_bytes(path, limit)
        if self._reads == 1:
            return content
        return (content + b"x")[:limit]


class HashRaceFileSystem(SkillAsyncFileSystem):
    def __init__(self) -> None:
        super().__init__()
        self._reads = 0

    async def read_bytes(self, path: Path, limit: int) -> bytes:
        self._reads += 1
        content = await super().read_bytes(path, limit)
        if self._reads == 1:
            return content
        return content.replace(b"PDF guidance.", b"PDF changed.!")


async def _registry(
    root: Path,
    *,
    settings: TrustedSkillSettings | None = None,
) -> SkillRegistry:
    source_result = await resolve_skill_sources((_config(root),))
    return await build_skill_registry(source_result, settings=settings)


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
    body: str = "# Body\n",
) -> None:
    _write_text(
        path,
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        f"enabled: {'true' if enabled else 'false'}\n"
        f"resources: {resources}\n"
        "---\n"
        f"{body}",
    )


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _assert_model_safe(
    test_case: IsolatedAsyncioTestCase,
    value: object,
    root: Path,
) -> None:
    encoded = dumps(value, sort_keys=True)
    test_case.assertNotIn(str(root), encoded)
    test_case.assertNotIn("/private/", encoded)
    test_case.assertNotIn("Traceback", encoded)
    test_case.assertNotIn('File "', encoded)


if __name__ == "__main__":
    main()
