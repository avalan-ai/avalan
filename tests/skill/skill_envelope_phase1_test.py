from json import dumps
from unittest import TestCase, main

from avalan.skill import (
    SkillDiagnosticCode,
    SkillDiagnosticInfo,
    SkillMatchResult,
    SkillMetadata,
    SkillProvenance,
    SkillReadCursor,
    SkillRegistryVersion,
    SkillResourceContent,
    SkillResourceHandle,
    SkillResponseEnvelope,
    SkillSourceAuthorityKind,
    SkillStatus,
)


class SkillEnvelopeTest(TestCase):
    def test_list_match_and_read_envelopes_serialize(self) -> None:
        registry_version = SkillRegistryVersion(value="skills-phase1:abc123")
        handle = SkillResourceHandle(
            source_label="workspace-main",
            skill_id="pdf",
            resource_id="main",
        )
        metadata = SkillMetadata(
            skill_id="pdf",
            name="pdf",
            description="PDF guidance.",
            source_label="workspace-main",
            resources=(handle,),
        )
        match = SkillMatchResult(
            metadata=metadata,
            score=0.75,
            reasons=("description matched",),
        )
        provenance = SkillProvenance(
            registry_version=registry_version,
            source_label="workspace-main",
            skill_id="pdf",
            resource_id="main",
            authority=SkillSourceAuthorityKind.WORKSPACE,
        )
        content_text = "Use PDF rendering guidance."
        content = SkillResourceContent(
            handle=handle,
            text=content_text,
            start_byte=0,
            end_byte=len(content_text.encode("utf-8")),
            truncated=True,
        )
        cursor = SkillReadCursor(
            cursor_id="cursor:pdf.001",
            registry_version=registry_version,
            source_label="workspace-main",
            skill_id="pdf",
            resource_id="main",
            offset_bytes=content.end_byte,
            limit_bytes=512,
        )

        list_envelope = SkillResponseEnvelope(
            status=SkillStatus.OK,
            registry_version=registry_version,
            items=(metadata,),
            provenance=(provenance,),
        )
        match_envelope = SkillResponseEnvelope(
            status=SkillStatus.OK,
            registry_version=registry_version,
            items=(match,),
            provenance=(provenance,),
        )
        read_envelope = SkillResponseEnvelope(
            status=SkillStatus.OK,
            registry_version=registry_version,
            content=content,
            next_cursor=cursor,
            provenance=(provenance,),
        )

        list_model = list_envelope.as_model_dict()
        list_items = list_model["items"]
        assert isinstance(list_items, tuple)
        list_item = list_items[0]
        assert isinstance(list_item, dict)
        self.assertEqual(list_item["skill_id"], "pdf")

        match_model = match_envelope.as_model_dict()
        match_items = match_model["items"]
        assert isinstance(match_items, tuple)
        match_item = match_items[0]
        assert isinstance(match_item, dict)
        self.assertEqual(match_item["score"], 0.75)

        read_model = read_envelope.as_model_dict()
        self.assertEqual(read_model["next_cursor"], "cursor:pdf.001")
        read_content = read_model["content"]
        assert isinstance(read_content, dict)
        read_handle = read_content["handle"]
        assert isinstance(read_handle, dict)
        self.assertEqual(read_handle["resource_id"], "main")
        dumps(read_envelope.as_model_dict(), allow_nan=False, sort_keys=True)

    def test_failure_states_are_structured_envelopes(self) -> None:
        registry_version = SkillRegistryVersion(value="skills-phase1:abc123")
        cases = (
            (
                SkillStatus.MALFORMED,
                SkillDiagnosticCode.MANIFEST_MALFORMED,
                "manifest.name",
            ),
            (
                SkillStatus.DISABLED,
                SkillDiagnosticCode.DISABLED,
                "skills.disabled",
            ),
            (
                SkillStatus.UNAVAILABLE,
                SkillDiagnosticCode.SOURCE_UNAVAILABLE,
                "source.availability",
            ),
            (
                SkillStatus.POLICY_DENIED,
                SkillDiagnosticCode.POLICY_DENIED,
                "source.policy",
            ),
        )

        for status, code, path in cases:
            with self.subTest(status=status):
                diagnostic = SkillDiagnosticInfo(
                    code=code,
                    status=status,
                    message="Skill is not usable.",
                    path=path,
                    hint="Continue without this skill.",
                )
                envelope = SkillResponseEnvelope(
                    status=status,
                    registry_version=registry_version,
                    diagnostics=(diagnostic,),
                )

                self.assertEqual(envelope.as_model_dict()["status"], status)
                envelope_model = envelope.as_model_dict()
                diagnostics = envelope_model["diagnostics"]
                assert isinstance(diagnostics, tuple)
                diagnostic_model = diagnostics[0]
                assert isinstance(diagnostic_model, dict)
                self.assertEqual(diagnostic_model["code"], code.value)

    def test_invalid_envelopes_reject_malformed_public_state(self) -> None:
        registry_version = SkillRegistryVersion(value="skills-phase1:abc123")
        other_version = SkillRegistryVersion(value="skills-phase1:def456")
        handle = SkillResourceHandle(
            source_label="workspace-main",
            skill_id="pdf",
            resource_id="main",
        )
        content_text = "Use PDF guidance."
        content = SkillResourceContent(
            handle=handle,
            text=content_text,
            end_byte=len(content_text.encode("utf-8")),
        )
        truncated_content = SkillResourceContent(
            handle=handle,
            text=content_text,
            end_byte=len(content_text.encode("utf-8")),
            truncated=True,
        )
        metadata = SkillMetadata(
            skill_id="pdf",
            name="pdf",
            description="PDF guidance.",
            source_label="workspace-main",
            resources=(handle,),
        )
        diagnostic = SkillDiagnosticInfo(
            code=SkillDiagnosticCode.POLICY_DENIED,
            status=SkillStatus.POLICY_DENIED,
            message="Denied.",
            path="source.policy",
            hint="Continue without it.",
        )
        provenance = SkillProvenance(
            registry_version=registry_version,
            source_label="workspace-main",
            skill_id="pdf",
            resource_id="main",
            authority=SkillSourceAuthorityKind.WORKSPACE,
        )

        invalid_builders = (
            lambda: SkillResponseEnvelope(
                status=SkillStatus.POLICY_DENIED,
                registry_version=registry_version,
            ),
            lambda: SkillResponseEnvelope(
                status=SkillStatus.OK,
                registry_version=registry_version,
                items=(metadata,),
                content=content,
                provenance=(provenance,),
            ),
            lambda: SkillResponseEnvelope(
                status=SkillStatus.OK,
                registry_version=registry_version,
                content=content,
            ),
            lambda: SkillResponseEnvelope(
                status=SkillStatus.OK,
                registry_version=registry_version,
                content=truncated_content,
                provenance=(provenance,),
            ),
            lambda: SkillResponseEnvelope(
                status=SkillStatus.OK,
                registry_version=registry_version,
                content=content,
                next_cursor=SkillReadCursor(
                    cursor_id="cursor:pdf.001",
                    registry_version=registry_version,
                    source_label="workspace-main",
                    skill_id="pdf",
                    resource_id="main",
                    offset_bytes=content.end_byte,
                    limit_bytes=512,
                ),
                provenance=(provenance,),
            ),
            lambda: SkillResponseEnvelope(
                status=SkillStatus.OK,
                registry_version=registry_version,
                next_cursor=SkillReadCursor(
                    cursor_id="cursor:pdf.001",
                    registry_version=registry_version,
                    source_label="workspace-main",
                    skill_id="pdf",
                    resource_id="main",
                    offset_bytes=content.end_byte,
                    limit_bytes=512,
                ),
            ),
            lambda: SkillResponseEnvelope(
                status=SkillStatus.OK,
                registry_version=registry_version,
                next_cursor=SkillReadCursor(
                    cursor_id="cursor:pdf.002",
                    registry_version=other_version,
                    source_label="workspace-main",
                    skill_id="pdf",
                    resource_id="main",
                    offset_bytes=0,
                    limit_bytes=512,
                ),
            ),
            lambda: SkillResponseEnvelope(
                status=SkillStatus.OK,
                registry_version=registry_version,
                content=truncated_content,
                next_cursor=SkillReadCursor(
                    cursor_id="cursor:pdf.003",
                    registry_version=registry_version,
                    source_label="workspace-main",
                    skill_id="pdf",
                    resource_id="main",
                    offset_bytes=content.end_byte + 1,
                    limit_bytes=512,
                ),
                provenance=(provenance,),
            ),
            lambda: SkillResponseEnvelope(
                status=SkillStatus.OK,
                registry_version=registry_version,
                diagnostics=[diagnostic],  # type: ignore[arg-type]
            ),
            lambda: SkillResponseEnvelope(
                status=SkillStatus.MALFORMED,
                registry_version=registry_version,
                diagnostics=(diagnostic,),
            ),
            lambda: SkillResourceContent(
                handle=handle,
                text="/Users/example/private/SKILL.md",
            ),
        )

        for invalid_builder in invalid_builders:
            with self.subTest(invalid_builder=invalid_builder):
                with self.assertRaises(AssertionError):
                    invalid_builder()


if __name__ == "__main__":
    main()
