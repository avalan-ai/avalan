import os
from json import dumps
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, main

from avalan.skill import (
    SkillDiagnosticCode,
    SkillPathPolicy,
    SkillSourceRootConfig,
    SkillStatus,
    WorkspaceSkillSourceAuthority,
    authorize_skill_resource,
    contains_skill_source_path_reference,
    contains_skill_source_resource_reference,
    could_contain_skill_source_path_reference_prefix,
    redact_host_path,
    resolve_skill_sources,
    sanitize_skill_resource_id,
    sanitize_skill_source_label,
    skill_model_handle_denial_reason,
    skill_source_root_denial_reason,
)
from avalan.skill.path_policy import sanitize_source_label


class SkillPathPolicyPhase2Test(IsolatedAsyncioTestCase):
    async def test_rejects_unsafe_configured_roots_and_duplicate_labels(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            valid = _write(base / "valid", "SKILL.md", "ok")
            cases = (
                ("nul", "bad\x00root", "nul_byte"),
                ("home", "~/skills", "home_expansion"),
                ("environment", "$HOME/skills", "environment_expansion"),
                ("traversal", "skills/../secret", "traversal"),
                ("url", "file:///tmp/skills", "url_handle"),
                ("url-short", "//server/share", "url_handle"),
            )

            for label, root, reason in cases:
                with self.subTest(reason=reason):
                    result = await resolve_skill_sources(
                        (
                            SkillSourceRootConfig(
                                label=label,
                                authority=WorkspaceSkillSourceAuthority(),
                                root=root,
                            ),
                        )
                    )
                    self.assertEqual(result.status, SkillStatus.UNAVAILABLE)
                    self.assertIn(reason, _reasons(result))
                    self.assertNotIn(root, dumps(result.as_model_dict()))

            unsafe_label = await resolve_skill_sources(
                (
                    SkillSourceRootConfig(
                        label="bad\x00label",
                        authority=WorkspaceSkillSourceAuthority(),
                        root=valid,
                    ),
                )
            )
            self.assertEqual(unsafe_label.status, SkillStatus.POLICY_DENIED)
            self.assertIn("nul_byte", _reasons(unsafe_label))
            self.assertNotIn(
                "bad\x00label",
                dumps(unsafe_label.as_model_dict()),
            )

            missing = await resolve_skill_sources(
                (
                    SkillSourceRootConfig(
                        label="missing",
                        authority=WorkspaceSkillSourceAuthority(),
                        root=base / "missing",
                    ),
                )
            )
            self.assertEqual(missing.status, SkillStatus.UNAVAILABLE)
            self.assertNotIn(str(base), dumps(missing.as_model_dict()))

            duplicate = await resolve_skill_sources(
                (
                    _config("duplicate", valid),
                    _config("Duplicate", valid),
                )
            )
            self.assertEqual(duplicate.status, SkillStatus.BLOCKED)
            self.assertIn("duplicate_source_label", _reasons(duplicate))

    async def test_denies_hidden_sensitive_symlink_and_special_paths(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            base = Path(temporary)

            hidden_root = base / "hidden"
            _write(hidden_root, ".hidden/SKILL.md", "hidden")
            hidden = await resolve_skill_sources(
                (_config("hidden", hidden_root),)
            )
            self.assertIn("hidden_path", _reasons(hidden))
            self.assertEqual(hidden.sources[0].resources, ())

            hidden_allowed = await resolve_skill_sources(
                (
                    SkillSourceRootConfig(
                        label="hidden-allowed",
                        authority=WorkspaceSkillSourceAuthority(),
                        root=hidden_root,
                        allow_hidden_paths=True,
                    ),
                )
            )
            self.assertEqual(hidden_allowed.status, SkillStatus.OK)
            resource_id = hidden_allowed.sources[0].resources[0].resource_id
            self.assertNotIn(".hidden", resource_id)
            self.assertTrue(resource_id.startswith("resource/"))

            sensitive_root = base / "sensitive"
            _write(sensitive_root, "secrets/token.md", "secret")
            sensitive = await resolve_skill_sources(
                (_config("sensitive", sensitive_root),)
            )
            self.assertIn("sensitive_path", _reasons(sensitive))
            self.assertNotIn("secrets", dumps(sensitive.as_model_dict()))

            outside = _write(base / "outside", "secret.md", "secret")
            symlink_root = base / "symlink"
            symlink_root.mkdir()
            safe_target = symlink_root / "safe.md"
            safe_target.write_text("safe", encoding="utf-8")
            hidden_target = symlink_root / ".hidden" / "target.md"
            hidden_target.parent.mkdir()
            hidden_target.write_text("hidden", encoding="utf-8")
            try:
                (symlink_root / "safe-link.md").symlink_to(safe_target)
                (symlink_root / "escape.md").symlink_to(outside / "secret.md")
                (symlink_root / "dangling.md").symlink_to(
                    symlink_root / "missing.md"
                )
                (symlink_root / "hidden-link.md").symlink_to(hidden_target)
            except OSError as error:
                self.skipTest(f"symlinks unavailable: {error}")
            symlink = await resolve_skill_sources(
                (_config("symlink", symlink_root),)
            )
            self.assertIn("symlink_escape", _reasons(symlink))
            self.assertIn("dangling_symlink", _reasons(symlink))
            self.assertIn("hidden_path", _reasons(symlink))
            self.assertIn(
                "safe-link.md",
                tuple(resource.resource_id for resource in symlink.resources),
            )

            if not hasattr(os, "mkfifo"):
                self.skipTest("mkfifo unavailable")
            special_root = base / "special"
            special_root.mkdir()
            os.mkfifo(special_root / "pipe.md")
            special = await resolve_skill_sources(
                (_config("special", special_root),)
            )
            self.assertIn("special_file", _reasons(special))

    async def test_authorizes_only_safe_model_resource_handles(self) -> None:
        with TemporaryDirectory() as temporary:
            root = _write(Path(temporary) / "source", "SKILL.md", "ok")
            result = await resolve_skill_sources((_config("source", root),))
            source = result.sources[0]

            valid = await authorize_skill_resource(source, "SKILL.md")
            self.assertEqual(valid.status, SkillStatus.OK)
            self.assertIsNotNone(valid.resource)
            assert valid.resource is not None
            self.assertEqual(valid.resource.resource_id, "SKILL.md")

            missing = await authorize_skill_resource(source, "missing.md")
            self.assertEqual(missing.status, SkillStatus.NOT_FOUND)
            self.assertEqual(
                missing.diagnostics[0].code,
                SkillDiagnosticCode.RESOURCE_MISSING,
            )

            cases = (
                ("../secret.md", "traversal"),
                ("/etc/passwd", "absolute_handle"),
                ("C:/Users/example/skill.md", "absolute_handle"),
                ("$HOME/skill.md", "environment_expansion"),
                ("~/skill.md", "home_expansion"),
                ("https://example.com/SKILL.md", "url_handle"),
                ("file:/tmp/SKILL.md", "url_handle"),
                ("data:text/plain,hello", "url_handle"),
                ("mailto:user@example.com", "url_handle"),
                ("//example.com/SKILL.md", "url_handle"),
                ("bad\x00name.md", "nul_byte"),
                ("bad\\name.md", "backslash"),
                ("references/./SKILL.md", "empty_path_segment"),
                ("references//SKILL.md", "empty_path_segment"),
                ("references/", "empty_path_segment"),
            )
            for handle, reason in cases:
                with self.subTest(reason=reason):
                    denied = await authorize_skill_resource(source, handle)

                    self.assertEqual(denied.status, SkillStatus.POLICY_DENIED)
                    self.assertIn(reason, _reasons(denied))
                    self.assertNotIn(handle, dumps(denied.as_model_dict()))

    async def test_model_facing_values_sanitize_labels_and_paths(self) -> None:
        with TemporaryDirectory() as temporary:
            root = _write(Path(temporary) / "source", "SKILL.md", "ok")
            unsafe_label = "/Users/example/.ssh/skills"
            config = SkillSourceRootConfig(
                label=unsafe_label,
                authority=WorkspaceSkillSourceAuthority(),
                root=root,
                package_path="../secrets",
            )
            plain_config = SkillSourceRootConfig(
                label="plain",
                authority=WorkspaceSkillSourceAuthority(),
                root=root,
            )

            result = await resolve_skill_sources((config,))
            encoded = dumps(
                {
                    "config": config.as_model_dict(),
                    "plain_config": plain_config.as_model_dict(),
                    "result": result.as_model_dict(),
                    "redacted": redact_host_path(root / "SKILL.md"),
                },
                sort_keys=True,
            )
            policy = SkillPathPolicy(allow_hidden_paths=True)

            self.assertEqual(result.status, SkillStatus.UNAVAILABLE)
            self.assertTrue(config.source_label.startswith("source-"))
            self.assertNotIn(unsafe_label, encoded)
            self.assertNotIn(str(root), encoded)
            self.assertNotIn(".ssh", encoded)
            self.assertIn("<host-path>/SKILL.md", encoded)
            self.assertEqual(plain_config.as_model_dict()["package_path"], ".")
            self.assertEqual(
                sanitize_skill_source_label(unsafe_label),
                config.source_label,
            )
            self.assertTrue(
                sanitize_skill_resource_id("../secret.md").startswith(
                    "resource/"
                )
            )
            self.assertEqual(
                skill_model_handle_denial_reason("/absolute.md"),
                "absolute_handle",
            )
            self.assertEqual(
                skill_source_root_denial_reason("$HOME/skills"),
                "environment_expansion",
            )
            self.assertEqual(
                policy.model_handle_denial_reason(".hidden/SKILL.md"),
                None,
            )
            self.assertEqual(
                policy.source_root_denial_reason(".hidden/skills"),
                None,
            )
            self.assertEqual(
                policy.sanitize_source_label("Workspace Main"),
                "workspace-main",
            )
            self.assertTrue(
                policy.sanitize_resource_id("/absolute.md").startswith(
                    "resource/"
                )
            )
            self.assertEqual(sanitize_source_label("PDF Tools"), "pdf-tools")
            self.assertEqual(
                skill_model_handle_denial_reason(""),
                "empty_handle",
            )
            self.assertEqual(
                skill_model_handle_denial_reason("//server/share"),
                "url_handle",
            )
            self.assertEqual(
                skill_source_root_denial_reason(""),
                "empty_root",
            )
            self.assertEqual(
                skill_source_root_denial_reason("C:/skills"),
                "windows_absolute_path",
            )
            self.assertEqual(
                skill_model_handle_denial_reason("."),
                "empty_path_segment",
            )
            self.assertEqual(
                skill_model_handle_denial_reason("token.md"),
                "sensitive_path",
            )
            self.assertEqual(
                skill_model_handle_denial_reason("secret.txt"),
                "sensitive_path",
            )
            self.assertEqual(
                skill_model_handle_denial_reason("credentials.json"),
                "sensitive_path",
            )
            self.assertIsNone(skill_model_handle_denial_reason("tokenizer.md"))
            self.assertTrue(
                sanitize_skill_source_label("").startswith("source-")
            )
            self.assertTrue(
                sanitize_skill_resource_id("token.md").startswith("resource/")
            )
            self.assertEqual(redact_host_path("bad\x00path"), "<host-path>")
            self.assertEqual(redact_host_path("/root/.ssh"), "<host-path>")

            for source_path in (
                "Source: ~/.codex/skills/demo/README.md",
                "Source: file:///skills/demo",
                "Source: /skills/demo/README.md",
                "Source: /skills",
                "Source: \\skills\\demo\\README.md",
                "Source: \\skills",
                "Source: C:/skills/demo/README.md",
                "Source: C:/skills",
                "Source: C:\\skills",
            ):
                with self.subTest(source_path=source_path):
                    self.assertTrue(
                        contains_skill_source_path_reference(source_path)
                    )

            for prefix in (
                "",
                "/skills/demo",
            ):
                with self.subTest(prefix=prefix):
                    self.assertEqual(
                        could_contain_skill_source_path_reference_prefix(
                            prefix
                        ),
                        bool(prefix),
                    )

            for prefix in (
                "~",
                "~/",
                "~/.c",
                ".codex",
                "/",
                "/ski",
                "\\",
                "\\ski",
                "C:",
                "C:/",
                "C:/ski",
                "C:\\",
                "C:\\ski",
            ):
                with self.subTest(prefix=prefix):
                    self.assertTrue(
                        could_contain_skill_source_path_reference_prefix(
                            prefix
                        )
                    )
            self.assertFalse(
                contains_skill_source_path_reference(
                    "Source note for deployment."
                )
            )
            self.assertFalse(contains_skill_source_path_reference("   "))
            self.assertFalse(contains_skill_source_path_reference("."))
            self.assertFalse(
                contains_skill_source_resource_reference("   ", "SKILL.md")
            )
            self.assertFalse(
                contains_skill_source_resource_reference(
                    "Source: /tmp/SKILL.md",
                    "",
                )
            )
            for private_resource in (
                "Source: file:///tmp/demo/SKILL.md",
                "Source: /tmp/demo/SKILL.md",
                "Source: C:/tmp/demo/SKILL.md",
            ):
                with self.subTest(private_resource=private_resource):
                    self.assertTrue(
                        contains_skill_source_resource_reference(
                            private_resource,
                            "SKILL.md",
                        )
                    )
            self.assertFalse(
                could_contain_skill_source_path_reference_prefix(
                    "Source note for deployment."
                )
            )
            for public_reference in (
                "https://docs.example.com/skills/demo",
                "https://docs.example.com/skills/demo/SKILL.md",
                "docs/skills/demo",
                "docs/skills/demo/SKILL.md",
            ):
                with self.subTest(public_reference=public_reference):
                    self.assertFalse(
                        contains_skill_source_path_reference(public_reference)
                    )
                    self.assertFalse(
                        could_contain_skill_source_path_reference_prefix(
                            public_reference
                        )
                    )
                    self.assertFalse(
                        contains_skill_source_resource_reference(
                            public_reference,
                            "SKILL.md",
                        )
                    )


def _config(label: str, root: Path) -> SkillSourceRootConfig:
    return SkillSourceRootConfig(
        label=label,
        authority=WorkspaceSkillSourceAuthority(),
        root=root,
    )


def _write(root: Path, relative: str, text: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return root


def _reasons(result: object) -> tuple[str, ...]:
    diagnostics = []
    if hasattr(result, "diagnostics"):
        diagnostics.extend(result.diagnostics)
    if hasattr(result, "sources"):
        for source in result.sources:
            diagnostics.extend(source.diagnostics)
    reasons: list[str] = []
    for diagnostic in diagnostics:
        reason = diagnostic.details.get("reason")
        if isinstance(reason, str):
            reasons.append(reason)
    return tuple(reasons)


if __name__ == "__main__":
    main()
