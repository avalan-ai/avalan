from copy import deepcopy
from json import loads
from pathlib import Path
from typing import Any
from unittest import TestCase, main

DOC_ROOT = Path(__file__).parents[2] / "docs"
INVENTORY_DOC = DOC_ROOT / "FLOW_FIXTURE_INVENTORY.md"
MANIFEST = Path(__file__).parents[1] / "fixtures" / "flow" / "manifest.json"


class FlowFixtureInventoryTest(TestCase):
    def test_inventory_document_is_linked_from_docs_index(self) -> None:
        docs = INVENTORY_DOC.read_text(encoding="utf-8")
        index = (DOC_ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("[Flow fixture inventory]", index)
        self.assertIn("## Fixture Roots", docs)
        self.assertIn("## Required Mermaid Security Buckets", docs)
        self.assertIn("## Acceptance Map", docs)
        self.assertIn("Mermaid executable-import security", docs)
        self.assertIn("tests/fixtures/flow/manifest.json", docs)

    def test_manifest_covers_fixture_categories_and_acceptance(self) -> None:
        manifest = _load_manifest()

        _validate_manifest(manifest)
        categories = {
            category["id"]: category for category in manifest["categories"]
        }

        self.assertEqual(
            set(categories),
            {
                "cli_sdk_parity",
                "definition_validation",
                "flow_view_binding",
                "mermaid_negative",
                "mermaid_positive",
                "mermaid_security_executable_import",
                "privacy_projection",
                "runtime_semantics",
            },
        )
        for case_type in manifest["required_case_types"]:
            with self.subTest(case_type=case_type):
                self.assertIn(case_type, _case_types(manifest))

        security = categories["mermaid_security_executable_import"]
        self.assertTrue(security["permanent"])
        self.assertEqual(
            set(security["unsafe_constructs"]),
            {
                "ambiguous_shorthand",
                "callback_directive",
                "click_directive",
                "frontmatter",
                "href_directive",
                "html_label",
                "init_directive",
                "link_directive",
                "malformed_directive",
                "malformed_subgraph",
                "script_like_label",
                "unsafe_external_link",
                "unsupported_diagram_type",
                "unknown_directive",
            },
        )
        self.assertEqual(
            set(security["unsafe_constructs"]),
            {
                path.stem
                for path in (
                    Path(__file__).parents[1]
                    / security["path"].removeprefix("tests/")
                ).glob("*.mmd")
            },
        )

    def test_manifest_validation_rejects_unmapped_acceptance(self) -> None:
        manifest = _load_manifest()
        broken = deepcopy(manifest)
        broken["categories"] = [
            category
            for category in broken["categories"]
            if category["id"] != "cli_sdk_parity"
        ]

        with self.assertRaisesRegex(
            AssertionError, "Missing category coverage"
        ):
            _validate_manifest(broken)

    def test_manifest_validation_rejects_nonpermanent_security(self) -> None:
        manifest = _load_manifest()
        broken = deepcopy(manifest)
        for category in broken["categories"]:
            if category["id"] == "mermaid_security_executable_import":
                category["permanent"] = False

        with self.assertRaisesRegex(AssertionError, "must be permanent"):
            _validate_manifest(broken)


def _load_manifest() -> dict[str, Any]:
    return loads(MANIFEST.read_text(encoding="utf-8"))


def _validate_manifest(manifest: dict[str, Any]) -> None:
    acceptance_ids = {item["id"] for item in manifest["acceptance"]}
    category_ids = {category["id"] for category in manifest["categories"]}
    category_coverage = {
        acceptance
        for category in manifest["categories"]
        for acceptance in category["acceptance"]
    }
    owner_coverage = {item["owner"] for item in manifest["acceptance"]}

    assert acceptance_ids == {f"AC{number:02d}" for number in range(1, 22)}
    assert owner_coverage.issubset(category_ids), "Missing category coverage."
    assert acceptance_ids.issubset(
        category_coverage
    ), "Missing acceptance coverage."
    assert set(manifest["required_case_types"]).issubset(_case_types(manifest))

    for category in manifest["categories"]:
        assert category["path"].startswith("tests/fixtures/flow/")
        assert category["future_tests"]
        assert category["acceptance"]
        if "security" in category["case_types"]:
            assert category[
                "permanent"
            ], "Security fixtures must be permanent."


def _case_types(manifest: dict[str, Any]) -> set[str]:
    return {
        case_type
        for category in manifest["categories"]
        for case_type in category["case_types"]
    }


if __name__ == "__main__":
    main()
