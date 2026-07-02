from collections.abc import Callable
from dataclasses import FrozenInstanceError
from unittest import TestCase, main

from avalan.skill import (
    SkillContractFixture,
    SkillContractMetadata,
    SkillDiagnostic,
    SkillDiagnosticCode,
    SkillFailureMode,
    SkillStatus,
    diagnostic_contract_for_failure,
)


class SkillEntitiesTestCase(TestCase):
    def test_diagnostic_serializes_model_shape(self) -> None:
        diagnostic = SkillDiagnostic(
            code=SkillDiagnosticCode.AMBIGUOUS_NAME,
            status=SkillStatus.AMBIGUOUS,
            message="Ambiguous skill.",
            path="skills.request.name",
            hint="Use an ID.",
            candidates=("pdf-basic", "pdf-review"),
        )

        self.assertEqual(
            diagnostic.as_model_dict(),
            {
                "code": "skills.ambiguous_name",
                "status": "ambiguous",
                "message": "Ambiguous skill.",
                "path": "skills.request.name",
                "hint": "Use an ID.",
                "candidates": ("pdf-basic", "pdf-review"),
            },
        )
        with self.assertRaises(FrozenInstanceError):
            diagnostic.code = SkillDiagnosticCode.NO_MATCH  # type: ignore[misc]

    def test_diagnostic_rejects_invalid_values(self) -> None:
        with self.assertRaises(AssertionError):
            SkillDiagnostic(
                code="skills.ok",  # type: ignore[arg-type]
                status=SkillStatus.OK,
                message="Message.",
                path="skills",
                hint="Hint.",
            )
        with self.assertRaises(AssertionError):
            SkillDiagnostic(
                code=SkillDiagnosticCode.NO_MATCH,
                status="empty",  # type: ignore[arg-type]
                message="Message.",
                path="skills",
                hint="Hint.",
            )
        with self.assertRaises(AssertionError):
            SkillDiagnostic(
                code=SkillDiagnosticCode.NO_MATCH,
                status=SkillStatus.EMPTY,
                message="",
                path="skills",
                hint="Hint.",
            )

    def test_metadata_serializes_model_item(self) -> None:
        metadata = SkillContractMetadata(
            skill_id="pdf",
            name="pdf",
            description="PDF guidance.",
            source_label="contract-valid",
            tags=("pdf",),
            version="1.0.0",
            resources=("references/rendering.md",),
        )

        self.assertEqual(
            metadata.as_model_dict(),
            {
                "skill_id": "pdf",
                "name": "pdf",
                "description": "PDF guidance.",
                "source_label": "contract-valid",
                "main_resource_id": "main",
                "enabled": True,
                "status": "ok",
                "tags": ("pdf",),
                "version": "1.0.0",
                "resources": ("references/rendering.md",),
            },
        )

    def test_metadata_rejects_invalid_values(self) -> None:
        invalid_builders: tuple[Callable[[], SkillContractMetadata], ...] = (
            lambda: SkillContractMetadata(
                skill_id="",
                name="pdf",
                description="PDF.",
                source_label="contract-valid",
            ),
            lambda: SkillContractMetadata(
                skill_id="pdf",
                name="other",
                description="PDF.",
                source_label="contract-valid",
            ),
            lambda: SkillContractMetadata(
                skill_id="pdf",
                name="pdf",
                description="",
                source_label="contract-valid",
            ),
            lambda: SkillContractMetadata(
                skill_id="pdf",
                name="pdf",
                description="/Users/example/private/skills/pdf/SKILL.md",
                source_label="contract-valid",
            ),
            lambda: SkillContractMetadata(
                skill_id="pdf",
                name="pdf",
                description="token\x00value",
                source_label="contract-valid",
            ),
            lambda: SkillContractMetadata(
                skill_id="pdf",
                name="pdf",
                description="PDF.",
                source_label="/private/path",
            ),
            lambda: SkillContractMetadata(
                skill_id="pdf",
                name="pdf",
                description="PDF.",
                source_label="contract-valid",
                version="$HOME/skills/pdf/SKILL.md",
            ),
            lambda: SkillContractMetadata(
                skill_id="pdf",
                name="pdf",
                description="PDF.",
                source_label="contract-valid",
                tags=("Not Safe",),
            ),
            lambda: SkillContractMetadata(
                skill_id="pdf",
                name="pdf",
                description="PDF.",
                source_label="contract-valid",
                resources=("$HOME/skills/pdf/SKILL.md",),
            ),
            lambda: SkillContractMetadata(
                skill_id="pdf",
                name="pdf",
                description="PDF.",
                source_label="contract-valid",
                resources=("references/../secret.md",),
            ),
        )

        for invalid_builder in invalid_builders:
            with self.subTest(invalid_builder=invalid_builder):
                with self.assertRaises(AssertionError):
                    invalid_builder()

    def test_fixture_rejects_invalid_values(self) -> None:
        with self.assertRaises(AssertionError):
            SkillContractFixture(
                source_label="/private/path",
                content="---\nname: pdf\ndescription: PDF.\n---\n# PDF",
            )
        with self.assertRaises(AssertionError):
            SkillContractFixture(
                source_label="contract-valid",
                content="text",
                content_bytes=b"text",
            )
        with self.assertRaises(AssertionError):
            SkillContractFixture(
                source_label="contract-valid",
                ambiguous_candidates=("pdf",),
            )
        with self.assertRaises(AssertionError):
            SkillContractFixture(
                source_label="contract-valid",
                requested_name="",
            )

    def test_failure_contract_rejects_invalid_mode(self) -> None:
        with self.assertRaises(AssertionError):
            diagnostic_contract_for_failure("missing")  # type: ignore[arg-type]

        contract = diagnostic_contract_for_failure(
            SkillFailureMode.RESOURCE_OUTSIDE_AUTHORIZED_ROOT
        )
        self.assertEqual(contract.status, SkillStatus.POLICY_DENIED)


if __name__ == "__main__":
    main()
