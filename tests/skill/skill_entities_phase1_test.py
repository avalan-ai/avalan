from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from json import dumps
from types import MappingProxyType
from unittest import TestCase, main

from avalan.skill import (
    BundledSkillSourceAuthority,
    PluginProvidedSkillSourceAuthority,
    PreinstalledRemoteSkillSourceAuthority,
    SkillDiagnosticCode,
    SkillDiagnosticInfo,
    SkillFailureMode,
    SkillMatchResult,
    SkillMetadata,
    SkillProvenance,
    SkillReadCursor,
    SkillRegistryVersion,
    SkillResourceContent,
    SkillResourceHandle,
    SkillSourceAuthority,
    SkillSourceAuthorityKind,
    SkillSourceConfig,
    SkillStatus,
    UserLocalSkillSourceAuthority,
    WorkspaceSkillSourceAuthority,
    diagnostic_from_failure,
    model_dict,
    to_model_value,
)


class SkillEntitiesTest(TestCase):
    def test_valid_entities_serialize_to_model_safe_values(self) -> None:
        registry_version = SkillRegistryVersion(
            value="skills-phase1:abc123",
        )
        authorities = (
            BundledSkillSourceAuthority(bundle_id="avalan"),
            WorkspaceSkillSourceAuthority(workspace_id="workspace"),
            UserLocalSkillSourceAuthority(profile_id="user-local"),
            PluginProvidedSkillSourceAuthority(plugin_id="pdf-plugin"),
            PreinstalledRemoteSkillSourceAuthority(registry_id="codex"),
        )
        source = SkillSourceConfig(
            label="workspace-main",
            authority=authorities[1],
            tags=("workspace",),
        )
        main_resource = SkillResourceHandle(
            source_label=source.label,
            skill_id="pdf",
            resource_id="main",
            size_bytes=512,
        )
        reference = SkillResourceHandle(
            source_label=source.label,
            skill_id="pdf",
            resource_id="references/rendering.md",
        )
        metadata = SkillMetadata(
            skill_id="pdf",
            name="pdf",
            description="PDF guidance.",
            source_label=source.label,
            tags=("pdf", "rendering"),
            version="1.0.0",
            resources=(main_resource, reference),
        )
        diagnostic = diagnostic_from_failure(
            SkillFailureMode.AMBIGUOUS_SKILL_NAME,
            path="skills.request",
            candidates=("pdf", "pdf-review"),
            details={"reason": "Multiple names matched."},
        )
        cursor = SkillReadCursor(
            cursor_id="cursor:pdf.001",
            registry_version=registry_version,
            source_label=source.label,
            skill_id="pdf",
            resource_id="main",
            offset_bytes=128,
            limit_bytes=256,
        )
        match = SkillMatchResult(
            metadata=metadata,
            score=0.95,
            reasons=("tag pdf matched",),
        )
        provenance = SkillProvenance(
            registry_version=registry_version,
            source_label=source.label,
            skill_id="pdf",
            resource_id="main",
            authority=SkillSourceAuthorityKind.WORKSPACE,
        )

        for authority in authorities:
            self.assertIn("kind", authority.as_model_dict())
        self.assertEqual(source.as_model_dict()["label"], "workspace-main")
        self.assertEqual(metadata.as_model_dict()["skill_id"], "pdf")
        self.assertEqual(
            diagnostic.details["reason"], "Multiple names matched."
        )
        self.assertIsInstance(diagnostic.details, MappingProxyType)
        self.assertEqual(cursor.as_model_value(), "cursor:pdf.001")
        self.assertEqual(match.as_model_dict()["score"], 0.95)
        self.assertEqual(provenance.as_model_dict()["authority"], "workspace")

        encoded = dumps(
            {
                "source": source.as_model_dict(),
                "metadata": metadata.as_model_dict(),
                "diagnostic": diagnostic.as_model_dict(),
                "cursor": cursor.as_internal_dict(),
                "match": match.as_model_dict(),
                "provenance": provenance.as_model_dict(),
            },
            allow_nan=False,
            sort_keys=True,
        )
        self.assertNotIn("/Users/", encoded)
        self.assertNotIn("/private/", encoded)

        with self.assertRaises(FrozenInstanceError):
            metadata.skill_id = "other"  # type: ignore[misc]
        with self.assertRaises(TypeError):
            diagnostic.details["new"] = "value"  # type: ignore[index]

    def test_diagnostic_details_are_deeply_immutable(self) -> None:
        nested = {"count": 1}
        item_mapping_source = {"name": "beta"}
        items: list[str | dict[str, str]] = ["alpha", item_mapping_source]
        labels = {"pdf", "rendering"}
        nested_details = {
            "nested": nested,
            "items": items,
            "labels": labels,
            "safe_none": None,
            "safe_bool": True,
            "safe_float": 1.5,
        }
        diagnostic = SkillDiagnosticInfo(
            code=SkillDiagnosticCode.NO_MATCH,
            status=SkillStatus.EMPTY,
            message="No match.",
            path="skills.match",
            hint="Continue without a skill.",
            details=nested_details,  # type: ignore[arg-type]
        )

        nested["count"] = 2
        item_mapping_source["name"] = "changed"
        items.append("changed")
        labels.add("changed")

        stored_nested = diagnostic.details["nested"]
        stored_items = diagnostic.details["items"]
        stored_labels = diagnostic.details["labels"]
        assert isinstance(stored_nested, Mapping)
        assert isinstance(stored_items, tuple)
        assert isinstance(stored_labels, tuple)
        self.assertEqual(stored_nested["count"], 1)
        self.assertEqual(len(stored_items), 2)
        self.assertEqual(stored_labels, ("pdf", "rendering"))
        self.assertIsNone(diagnostic.details["safe_none"])
        self.assertIs(diagnostic.details["safe_bool"], True)
        self.assertEqual(diagnostic.details["safe_float"], 1.5)

        with self.assertRaises(TypeError):
            stored_nested["count"] = 3  # type: ignore[index]
        item_mapping = stored_items[1]
        assert isinstance(item_mapping, Mapping)
        with self.assertRaises(TypeError):
            item_mapping["name"] = "changed"  # type: ignore[index]

    def test_source_diagnostics_and_disabled_metadata_serialize(self) -> None:
        diagnostic = SkillDiagnosticInfo(
            code=SkillDiagnosticCode.SOURCE_UNAVAILABLE,
            status=SkillStatus.UNAVAILABLE,
            message="Source unavailable.",
            path="source.availability",
            hint="Continue without this source.",
        )
        source = SkillSourceConfig(
            label="workspace-main",
            authority=WorkspaceSkillSourceAuthority(),
            status=SkillStatus.UNAVAILABLE,
            diagnostics=(diagnostic,),
        )
        disabled = SkillMetadata(
            skill_id="pdf",
            name="pdf",
            description="PDF guidance.",
            source_label="workspace-main",
            enabled=False,
            status=SkillStatus.DISABLED,
        )

        source_model = source.as_model_dict()
        source_diagnostics = source_model["diagnostics"]
        assert isinstance(source_diagnostics, tuple)
        source_diagnostic = source_diagnostics[0]
        assert isinstance(source_diagnostic, dict)
        self.assertEqual(
            source_diagnostic["code"],
            "skills.source_unavailable",
        )
        self.assertEqual(disabled.as_model_dict()["status"], "disabled")

    def test_entities_reject_invalid_values(self) -> None:
        source = SkillSourceConfig(
            label="workspace-main",
            authority=WorkspaceSkillSourceAuthority(),
        )
        main_resource = SkillResourceHandle(
            source_label=source.label,
            skill_id="pdf",
            resource_id="main",
        )

        invalid_builders = (
            lambda: SkillRegistryVersion(value=""),
            lambda: SkillSourceAuthority(
                kind="workspace"  # type: ignore[arg-type]
            ),
            lambda: PluginProvidedSkillSourceAuthority(plugin_id="Bad Plugin"),
            lambda: SkillSourceConfig(
                label="/private/path",
                authority=WorkspaceSkillSourceAuthority(),
            ),
            lambda: SkillSourceConfig(
                label="plugin-source",
                authority=SkillSourceAuthority(
                    kind=SkillSourceAuthorityKind.PLUGIN_PROVIDED
                ),
            ),
            lambda: SkillSourceConfig(
                label="remote-source",
                authority=SkillSourceAuthority(
                    kind=SkillSourceAuthorityKind.PREINSTALLED_REMOTE
                ),
            ),
            lambda: SkillSourceConfig(
                label="disabled",
                authority=WorkspaceSkillSourceAuthority(),
                enabled=False,
                status=SkillStatus.OK,
            ),
            lambda: SkillResourceHandle(
                source_label=source.label,
                skill_id="",
                resource_id="main",
            ),
            lambda: SkillResourceHandle(
                source_label=source.label,
                skill_id="pdf",
                resource_id="../secret.md",
            ),
            lambda: SkillResourceHandle(
                source_label=source.label,
                skill_id="pdf",
                resource_id="bad\\path",
            ),
            lambda: SkillResourceHandle(
                source_label=source.label,
                skill_id="pdf",
                resource_id="/secret.md",
            ),
            lambda: SkillResourceHandle(
                source_label=source.label,
                skill_id="pdf",
                resource_id="references/./rendering.md",
            ),
            lambda: SkillResourceHandle(
                source_label=source.label,
                skill_id="pdf",
                resource_id="main",
                status=SkillStatus.STALE,
            ),
            lambda: SkillResourceHandle(
                source_label=source.label,
                skill_id="pdf",
                resource_id="main",
                stale=True,
            ),
            lambda: SkillMetadata(
                skill_id="pdf",
                name="other",
                description="PDF.",
                source_label=source.label,
                resources=(main_resource,),
            ),
            lambda: SkillMetadata(
                skill_id="pdf",
                name="pdf",
                description="contains\x00nul",
                source_label=source.label,
                tags=("pdf",),
                resources=(main_resource,),
            ),
            lambda: SkillMetadata(
                skill_id="pdf",
                name="pdf",
                description="$HOME skills.",
                source_label=source.label,
                resources=(main_resource,),
            ),
            lambda: SkillMetadata(
                skill_id="pdf",
                name="pdf",
                description="../private skill.",
                source_label=source.label,
                resources=(main_resource,),
            ),
            lambda: SkillMetadata(
                skill_id="pdf",
                name="pdf",
                description="Use /opt/skills/pdf/SKILL.md",
                source_label=source.label,
                resources=(main_resource,),
            ),
            lambda: SkillMetadata(
                skill_id="pdf",
                name="pdf",
                description="Use C:\\Users\\example\\skill.md",
                source_label=source.label,
                resources=(main_resource,),
            ),
            lambda: SkillMetadata(
                skill_id="pdf",
                name="pdf",
                description="PDF.",
                source_label=source.label,
                tags=("Bad Tag",),
                resources=(main_resource,),
            ),
            lambda: SkillMetadata(
                skill_id="pdf",
                name="pdf",
                description="PDF.",
                source_label=source.label,
                tags=["pdf"],  # type: ignore[arg-type]
                resources=(main_resource,),
            ),
            lambda: SkillProvenance(
                registry_version=SkillRegistryVersion(
                    value="skills-phase1:abc123"
                ),
                source_label="/Users/example",
                skill_id="pdf",
                resource_id="main",
                authority=SkillSourceAuthorityKind.WORKSPACE,
            ),
            lambda: SkillSourceConfig(
                label="workspace-main",
                authority=WorkspaceSkillSourceAuthority(),
                diagnostics=("bad",),  # type: ignore[arg-type]
            ),
            lambda: SkillResourceContent(
                handle=main_resource,
                text="non-empty",
            ),
            lambda: SkillResourceContent(
                handle=main_resource,
                text="non-empty",
                start_byte=0,
                end_byte=99,
            ),
            lambda: SkillResourceContent(
                handle=main_resource,
                text="",
                start_byte=0,
                end_byte=1,
            ),
            lambda: SkillResourceContent(
                handle=main_resource,
                text="",
                truncated=True,
            ),
        )

        for invalid_builder in invalid_builders:
            with self.subTest(invalid_builder=invalid_builder):
                with self.assertRaises(AssertionError):
                    invalid_builder()

    def test_model_dict_rejects_non_json_safe_values(self) -> None:
        valid = model_dict(
            {
                "status": "ok",
                "items": ({"skill_id": "pdf"},),
                "labels": {"rendering", "pdf"},
                "count": 1,
            }
        )

        self.assertEqual(valid["status"], "ok")
        self.assertEqual(valid["labels"], ("pdf", "rendering"))
        self.assertIsNone(to_model_value(None))
        with self.assertRaises(AssertionError):
            model_dict({"bad": object()})
        with self.assertRaises(AssertionError):
            model_dict({"bad": float("nan")})
        with self.assertRaises(AssertionError):
            model_dict({"path": "/Users/example/private/SKILL.md"})

    def test_diagnostic_info_rejects_invalid_fields(self) -> None:
        with self.assertRaises(AssertionError):
            SkillDiagnosticInfo(
                code="bad",  # type: ignore[arg-type]
                status=SkillStatus.MALFORMED,
                message="Malformed.",
                path="skills",
                hint="Fix it.",
            )
        with self.assertRaises(AssertionError):
            SkillDiagnosticInfo(
                code=SkillDiagnosticCode.MANIFEST_MALFORMED,
                status=SkillStatus.MALFORMED,
                message="Malformed.",
                path="/private/path",
                hint="Fix it.",
            )
        with self.assertRaises(AssertionError):
            SkillDiagnosticInfo(
                code=SkillDiagnosticCode.MANIFEST_MALFORMED,
                status=SkillStatus.MALFORMED,
                message="Malformed.",
                path="skills",
                hint="Fix it.",
                details={"bad": object()},  # type: ignore[dict-item]
            )


if __name__ == "__main__":
    main()
