from unittest import TestCase, main

from avalan.skill import (
    SKILL_ID_CONVENTION,
    normalize_skill_description,
    normalize_skill_name,
    normalize_skill_resource_id,
    normalize_skill_source_label,
    normalize_skill_tag,
    normalize_skill_tags,
    skill_name_denial_reason,
    skill_resource_denial_reason,
)


class SkillNormalizerPhase3Test(TestCase):
    def test_skill_id_convention_uses_manifest_name_slug_not_path(
        self,
    ) -> None:
        self.assertEqual(SKILL_ID_CONVENTION, "manifest_name_slug")
        self.assertEqual(normalize_skill_name("PDF"), "pdf")
        self.assertEqual(normalize_skill_name("PDF Tools"), "pdf-tools")
        self.assertEqual(normalize_skill_name("pdf_tools"), "pdf-tools")
        self.assertEqual(normalize_skill_name("render.v2"), "render-v2")
        self.assertIsNone(normalize_skill_name("/skills/pdf/SKILL.md"))
        self.assertEqual(
            skill_name_denial_reason("pdf/read"), "path_like_name"
        )
        self.assertEqual(skill_name_denial_reason(""), "empty_name")
        self.assertEqual(skill_name_denial_reason("bad\x00name"), "nul_byte")
        self.assertEqual(skill_name_denial_reason(".hidden"), "path_like_name")
        self.assertEqual(
            skill_name_denial_reason("bad..name"),
            "path_like_name",
        )
        self.assertEqual(skill_name_denial_reason("???"), "empty_name")
        self.assertEqual(skill_name_denial_reason("123"), "invalid_name")

    def test_normalizes_descriptions_tags_sources_and_resources(self) -> None:
        self.assertEqual(
            normalize_skill_description("  Read\nPDF\tfiles.  "),
            "Read PDF files.",
        )
        self.assertIsNone(normalize_skill_description(" \n "))
        self.assertEqual(
            normalize_skill_source_label("Workspace Main"),
            "workspace-main",
        )
        self.assertEqual(normalize_skill_tag("PDF Rendering"), "pdf-rendering")
        self.assertEqual(
            normalize_skill_tags(("PDF", "rendering", "pdf")),
            ("pdf", "rendering"),
        )
        self.assertIsNone(normalize_skill_tags(("PDF", "bad/path")))
        self.assertEqual(
            normalize_skill_resource_id("references/rendering.md"),
            "references/rendering.md",
        )

    def test_rejects_unsafe_or_recursive_resource_ids(self) -> None:
        cases = {
            "": "empty_resource",
            "main": "reserved_resource",
            "references/": "directory_resource",
            "references/*.md": "recursive_resource",
            "../secret.md": "traversal",
            "/secret.md": "absolute_handle",
            "bad\\path": "backslash",
        }

        for resource_id, reason in cases.items():
            with self.subTest(resource_id=resource_id):
                self.assertIsNone(normalize_skill_resource_id(resource_id))
                self.assertEqual(
                    skill_resource_denial_reason(resource_id), reason
                )


if __name__ == "__main__":
    main()
