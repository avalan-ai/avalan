from subprocess import run
from sys import executable
from typing import Any, cast
from unittest import TestCase

from avalan.entities import ToolNamePolicyMode, ToolNamePolicySettings
from avalan.model.provider import ProviderFamily
from avalan.tool.name_policy import ToolNamePolicy


class ToolNamePolicyTestCase(TestCase):
    def test_tool_manager_and_name_policy_import_cleanly(self):
        run(
            [
                executable,
                "-B",
                "-c",
                (
                    "from avalan.tool.manager import ToolManager; "
                    "from avalan.tool.name_policy import ToolNamePolicy"
                ),
            ],
            check=True,
        )

    def test_encoded_default_preserves_safe_names_and_encodes_prefix(self):
        policy = ToolNamePolicy.default().bind(
            [
                "plain",
                "pkg.tool",
                "avl_plain",
            ]
        )

        self.assertEqual(policy.provider_name("plain"), "plain")
        self.assertEqual(policy.provider_name("pkg.tool"), "avl_cGtnLnRvb2w")
        self.assertEqual(policy.provider_name("avl_plain"), "avl_YXZsX3BsYWlu")
        self.assertEqual(policy.canonical_name("avl_cGtnLnRvb2w"), "pkg.tool")

    def test_encoded_accepts_custom_prefix(self):
        settings = ToolNamePolicySettings(prefix="tool_")
        policy = ToolNamePolicy(settings=settings).bind(["pkg.tool"])

        provider_name = policy.provider_name("pkg.tool")

        self.assertEqual(provider_name, "tool_cGtnLnRvb2w")
        self.assertEqual(policy.canonical_name(provider_name), "pkg.tool")

    def test_encoded_decodes_unbound_provider_names(self):
        policy = ToolNamePolicy.default()

        self.assertEqual(
            policy.canonical_name("avl_cGtnLnRvb2w"),
            "pkg.tool",
        )
        self.assertEqual(policy.canonical_name("plain"), "plain")
        self.assertEqual(ToolNamePolicy.decode_encoded("plain"), "plain")

    def test_mapped_uses_explicit_provider_names(self):
        settings = ToolNamePolicySettings(
            mode=ToolNamePolicyMode.MAPPED,
            map={"shell.pdfinfo": "my_pdfinfo"},
        )
        policy = ToolNamePolicy(settings=settings).bind(["shell.pdfinfo"])

        self.assertEqual(policy.provider_name("shell.pdfinfo"), "my_pdfinfo")
        self.assertEqual(policy.canonical_name("my_pdfinfo"), "shell.pdfinfo")
        self.assertEqual(policy.canonical_name("unknown_tool"), "unknown_tool")

    def test_mapped_allows_empty_prefix(self):
        settings = ToolNamePolicySettings(
            mode=ToolNamePolicyMode.MAPPED,
            prefix="",
            map={"shell.pdfinfo": "pdfinfo"},
        )
        policy = ToolNamePolicy(settings=settings).bind(["shell.pdfinfo"])

        self.assertEqual(policy.provider_name("shell.pdfinfo"), "pdfinfo")
        self.assertEqual(policy.canonical_name("pdfinfo"), "shell.pdfinfo")

    def test_unsupported_mode_rejects_before_explicit_map(self):
        settings = ToolNamePolicySettings(
            mode=ToolNamePolicyMode.MAPPED,
            map={"shell.pdfinfo": "my_pdfinfo"},
        )
        object.__setattr__(settings, "mode", "unsupported")
        policy = ToolNamePolicy(settings=settings)

        with self.assertRaisesRegex(
            AssertionError, "unsupported ToolNamePolicyMode"
        ):
            policy.provider_name("shell.pdfinfo")

    def test_sanitized_replaces_and_collapses_repeated_replacements(self):
        settings = ToolNamePolicySettings(mode=ToolNamePolicyMode.SANITIZED)
        policy = ToolNamePolicy(settings=settings).bind(
            ["graph.3d.encode-_graph"]
        )

        self.assertEqual(
            policy.provider_name("graph.3d.encode-_graph"),
            "graph_3d_encode_graph",
        )

    def test_sanitized_policy_uses_explicit_map_before_fallback(self):
        settings = ToolNamePolicySettings(
            mode=ToolNamePolicyMode.SANITIZED,
            map={
                "shell.pdfinfo": "pdfinfo",
                "shell.tesseract": "tesseract",
            },
        )
        policy = ToolNamePolicy(settings=settings).bind(
            [
                "shell.pdfinfo",
                "shell.tesseract",
                "database.query",
            ]
        )

        self.assertEqual(policy.provider_name("shell.pdfinfo"), "pdfinfo")
        self.assertEqual(policy.provider_name("shell.tesseract"), "tesseract")
        self.assertEqual(
            policy.provider_name("database.query"), "database_query"
        )
        self.assertEqual(policy.canonical_name("pdfinfo"), "shell.pdfinfo")
        self.assertEqual(policy.canonical_name("tesseract"), "shell.tesseract")
        self.assertEqual(
            policy.canonical_name("database_query"), "database.query"
        )

    def test_sanitized_can_keep_repeated_replacements(self):
        settings = ToolNamePolicySettings(
            mode=ToolNamePolicyMode.SANITIZED,
            collapse_replacement=False,
        )
        policy = ToolNamePolicy(settings=settings).bind(
            ["graph.3d.encode-_graph"]
        )

        self.assertEqual(
            policy.provider_name("graph.3d.encode-_graph"),
            "graph_3d_encode__graph",
        )

    def test_sanitized_uses_maps_and_policy_fallback_together(self):
        settings = ToolNamePolicySettings(
            mode=ToolNamePolicyMode.SANITIZED,
            map={
                "shell.pdfinfo": "pdfinfo",
                "shell.tesseract": "tesseract",
            },
        )
        policy = ToolNamePolicy(settings=settings).bind(
            [
                "shell.pdfinfo",
                "shell.pdftoppm",
                "shell.tesseract",
            ]
        )

        self.assertEqual(policy.provider_name("shell.pdfinfo"), "pdfinfo")
        self.assertEqual(
            policy.provider_name("shell.pdftoppm"), "shell_pdftoppm"
        )
        self.assertEqual(policy.provider_name("shell.tesseract"), "tesseract")
        self.assertEqual(policy.canonical_name("pdfinfo"), "shell.pdfinfo")
        self.assertEqual(
            policy.canonical_name("shell_pdftoppm"), "shell.pdftoppm"
        )
        self.assertEqual(policy.canonical_name("tesseract"), "shell.tesseract")

    def test_sanitized_collision_fails_without_explicit_map(self):
        settings = ToolNamePolicySettings(mode=ToolNamePolicyMode.SANITIZED)

        with self.assertRaisesRegex(AssertionError, "collision"):
            ToolNamePolicy(settings=settings).bind(["a.b", "a_b"])

    def test_mapped_collision_fails_without_explicit_disambiguation(self):
        settings = ToolNamePolicySettings(
            mode=ToolNamePolicyMode.MAPPED,
            map={"a.b": "tool", "a_b": "tool"},
        )

        with self.assertRaisesRegex(AssertionError, "collision"):
            ToolNamePolicy(settings=settings).bind(["a.b", "a_b"])

    def test_sanitized_collision_can_be_disambiguated_by_map(self):
        settings = ToolNamePolicySettings(
            mode=ToolNamePolicyMode.SANITIZED,
            map={"a_b": "a_b_raw"},
        )
        policy = ToolNamePolicy(settings=settings).bind(["a.b", "a_b"])

        self.assertEqual(policy.provider_name("a.b"), "a_b")
        self.assertEqual(policy.provider_name("a_b"), "a_b_raw")
        self.assertEqual(policy.canonical_name("a_b_raw"), "a_b")

    def test_mapped_name_collision_with_policy_fallback_fails(self):
        settings = ToolNamePolicySettings(
            mode=ToolNamePolicyMode.SANITIZED,
            map={"shell.pdfinfo": "shell_tesseract"},
        )

        with self.assertRaisesRegex(AssertionError, "collision"):
            ToolNamePolicy(settings=settings).bind(
                [
                    "shell.pdfinfo",
                    "shell.tesseract",
                ]
            )

    def test_raw_rejects_openai_dotted_names_and_allows_local(self):
        settings = ToolNamePolicySettings(mode=ToolNamePolicyMode.RAW)
        openai_policy = ToolNamePolicy(settings=settings).for_provider(
            ProviderFamily.OPENAI
        )

        with self.assertRaisesRegex(AssertionError, "openai"):
            openai_policy.provider_name("shell.pdfinfo")

        local_policy = ToolNamePolicy(settings=settings).for_provider(
            ProviderFamily.LOCAL
        )
        self.assertEqual(
            local_policy.provider_name("shell.pdfinfo"), "shell.pdfinfo"
        )

    def test_invalid_provider_name_policy_settings_fail_early(self):
        with self.assertRaisesRegex(AssertionError, "prefix"):
            ToolNamePolicySettings(prefix="")
        with self.assertRaisesRegex(AssertionError, "prefix"):
            ToolNamePolicy(settings=ToolNamePolicySettings(prefix="bad."))
        with self.assertRaisesRegex(AssertionError, "replacement"):
            ToolNamePolicy(settings=ToolNamePolicySettings(replacement="."))
        with self.assertRaisesRegex(AssertionError, "provider tool name"):
            ToolNamePolicy(
                settings=ToolNamePolicySettings(
                    mode=ToolNamePolicyMode.MAPPED,
                    map={"tool": "bad.name"},
                )
            )

    def test_invalid_tool_name_policy_map_settings_fail_early(self):
        cases: tuple[Any, ...] = (
            {"": "tool"},
            {" ": "tool"},
            {"tool": ""},
            {"tool": " "},
            cast(Any, {1: "tool"}),
            cast(Any, {"tool": 1}),
        )

        for name_map in cases:
            with self.subTest(name_map=name_map):
                with self.assertRaises(AssertionError):
                    ToolNamePolicySettings(map=name_map)

    def test_decode_rejects_malformed_encoded_names(self):
        for name in ("pkg.tool", "avl_notbase64", "avl_A"):
            with self.subTest(name=name):
                with self.assertRaises(AssertionError):
                    ToolNamePolicy.decode_encoded(name)

    def test_validate_provider_names_checks_bound_names(self):
        settings = ToolNamePolicySettings(
            mode=ToolNamePolicyMode.RAW,
            provider_family=ProviderFamily.LOCAL.value,
        )
        policy = ToolNamePolicy(settings=settings).bind(["shell.pdfinfo"])

        policy.validate_provider_names()
