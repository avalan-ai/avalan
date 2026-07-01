from json import dumps, load
from pathlib import Path
from unittest import TestCase, main

from avalan.skill import (
    CANONICAL_SKILLS_TOOL_NAMES,
    DISALLOWED_MODEL_FACING_SKILLS_TOOL_NAMES,
    FIRST_RELEASE_SKILL_TARGETS,
    LATER_RELEASE_SKILL_TARGETS,
    REQUIRED_SKILL_MANIFEST_FIELDS,
    SECTION_14_FAILURE_DIAGNOSTICS,
    SKILL_BACKWARD_COMPATIBILITY_REQUIRED,
    SKILL_MAIN_RESOURCE_ID,
    SKILL_MANIFEST_FORMAT,
    SKILL_VOCABULARY,
    SKILLS_SYNTAX_REJECTING_SURFACES,
    SUPPORTED_SKILL_MANIFEST_FIELDS,
    SkillContractCompilation,
    SkillContractFixture,
    SkillDiagnosticCode,
    SkillFailureMode,
    SkillManifestField,
    SkillReleaseTarget,
    SkillStatus,
    SkillSyntaxSurface,
    SkillVocabularyTerm,
    all_failure_diagnostic_contracts,
    canonical_skills_tool_schemas,
    compile_skill_contract_fixtures,
    diagnostic_contract_for_failure,
    reject_skills_syntax,
    rejects_skills_syntax,
)

FIXTURE_DIR = Path(__file__).with_name("fixtures") / "contract"

EXPECTED_FIXTURES = {
    "valid": (SkillStatus.OK, None),
    "empty": (SkillStatus.EMPTY, SkillDiagnosticCode.EMPTY_REGISTRY),
    "malformed": (
        SkillStatus.MALFORMED,
        SkillDiagnosticCode.MANIFEST_MALFORMED,
    ),
    "duplicate": (SkillStatus.BLOCKED, SkillDiagnosticCode.DUPLICATE_ID),
    "disabled": (SkillStatus.DISABLED, SkillDiagnosticCode.DISABLED),
    "unavailable": (
        SkillStatus.UNAVAILABLE,
        SkillDiagnosticCode.SOURCE_UNAVAILABLE,
    ),
    "ambiguous": (SkillStatus.AMBIGUOUS, SkillDiagnosticCode.AMBIGUOUS_NAME),
    "stale": (SkillStatus.STALE, SkillDiagnosticCode.RESOURCE_STALE),
    "denied": (SkillStatus.POLICY_DENIED, SkillDiagnosticCode.POLICY_DENIED),
    "oversized": (
        SkillStatus.TRUNCATED,
        SkillDiagnosticCode.RESOURCE_OVERSIZED,
    ),
    "binary": (SkillStatus.UNAVAILABLE, SkillDiagnosticCode.BINARY_RESOURCE),
    "traversal": (
        SkillStatus.POLICY_DENIED,
        SkillDiagnosticCode.RESOURCE_OUTSIDE_ROOT,
    ),
}

EXPECTED_SECTION_14_STATUS_CODES = {
    SkillFailureMode.EMPTY_REGISTRY: (
        SkillStatus.EMPTY,
        SkillDiagnosticCode.EMPTY_REGISTRY,
    ),
    SkillFailureMode.NO_MATCH: (
        SkillStatus.EMPTY,
        SkillDiagnosticCode.NO_MATCH,
    ),
    SkillFailureMode.UNKNOWN_SKILL_ID: (
        SkillStatus.NOT_FOUND,
        SkillDiagnosticCode.NOT_FOUND,
    ),
    SkillFailureMode.AMBIGUOUS_SKILL_NAME: (
        SkillStatus.AMBIGUOUS,
        SkillDiagnosticCode.AMBIGUOUS_NAME,
    ),
    SkillFailureMode.DISABLED_SKILL: (
        SkillStatus.DISABLED,
        SkillDiagnosticCode.DISABLED,
    ),
    SkillFailureMode.SOURCE_UNAVAILABLE: (
        SkillStatus.UNAVAILABLE,
        SkillDiagnosticCode.SOURCE_UNAVAILABLE,
    ),
    SkillFailureMode.MALFORMED_MANIFEST: (
        SkillStatus.MALFORMED,
        SkillDiagnosticCode.MANIFEST_MALFORMED,
    ),
    SkillFailureMode.RESOURCE_MISSING: (
        SkillStatus.NOT_FOUND,
        SkillDiagnosticCode.RESOURCE_MISSING,
    ),
    SkillFailureMode.RESOURCE_OUTSIDE_AUTHORIZED_ROOT: (
        SkillStatus.POLICY_DENIED,
        SkillDiagnosticCode.RESOURCE_OUTSIDE_ROOT,
    ),
    SkillFailureMode.RESOURCE_REQUIRES_CURSOR: (
        SkillStatus.TRUNCATED,
        SkillDiagnosticCode.RESOURCE_OVERSIZED,
    ),
    SkillFailureMode.RESOURCE_STALE: (
        SkillStatus.STALE,
        SkillDiagnosticCode.RESOURCE_STALE,
    ),
    SkillFailureMode.DUPLICATE_SKILL_IDS: (
        SkillStatus.BLOCKED,
        SkillDiagnosticCode.DUPLICATE_ID,
    ),
    SkillFailureMode.RUNTIME_SOURCE_INACCESSIBLE: (
        SkillStatus.UNAVAILABLE,
        SkillDiagnosticCode.RUNTIME_SOURCE_UNAVAILABLE,
    ),
}


def load_fixture(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as file:
        value = load(file)
    assert isinstance(value, dict)
    return value


def contract_fixtures_from_json(
    fixture: dict[str, object],
) -> tuple[SkillContractFixture, ...]:
    source = fixture["source"]
    assert isinstance(source, dict)
    source_label = source["label"]
    host_path = source["root"]
    assert isinstance(source_label, str)
    assert isinstance(host_path, str)

    skills = fixture.get("skills", ())
    assert isinstance(skills, list)
    if not skills:
        return ()

    result: list[SkillContractFixture] = []
    for skill in skills:
        assert isinstance(skill, dict)
        content = skill.get("content")
        assert isinstance(content, str)
        resource_state = skill.get("resource_state")
        ambiguous_candidates = fixture.get("ambiguous_candidates", ())
        assert isinstance(ambiguous_candidates, list | tuple)
        source_available = source.get("available") is not False
        runtime_available = source.get("runtime_accessible") is not False
        policy_allowed = skill.get("policy") != "denied"
        stale = resource_state == "stale"
        candidates = tuple(
            candidate
            for candidate in ambiguous_candidates
            if isinstance(candidate, str)
        )
        assert len(candidates) == len(ambiguous_candidates)
        if resource_state == "binary":
            result.append(
                SkillContractFixture(
                    source_label=source_label,
                    content_bytes=b"\x00binary",
                    host_path=host_path,
                    source_available=source_available,
                    runtime_available=runtime_available,
                    policy_allowed=policy_allowed,
                    stale=stale,
                    ambiguous_candidates=candidates,
                )
            )
        else:
            result.append(
                SkillContractFixture(
                    source_label=source_label,
                    content=content,
                    host_path=host_path,
                    source_available=source_available,
                    runtime_available=runtime_available,
                    policy_allowed=policy_allowed,
                    stale=stale,
                    ambiguous_candidates=candidates,
                )
            )
    return tuple(result)


def compile_json_fixture(
    fixture: dict[str, object],
) -> SkillContractCompilation:
    maximum_manifest_bytes = 64 if fixture["case"] == "oversized" else 65_536
    return compile_skill_contract_fixtures(
        contract_fixtures_from_json(fixture),
        maximum_manifest_bytes=maximum_manifest_bytes,
    )


def compile_markdown_fixture(content: str) -> SkillContractCompilation:
    return compile_skill_contract_fixtures(
        (
            SkillContractFixture(
                source_label="contract-public-behavior",
                content=content,
                host_path="/Users/example/private/skills/SKILL.md",
            ),
        )
    )


class SkillContractTestCase(TestCase):
    def test_phase_zero_release_targets_are_locked(self) -> None:
        self.assertEqual(
            FIRST_RELEASE_SKILL_TARGETS,
            (
                SkillReleaseTarget.FILESYSTEM_TRUSTED_SOURCES,
                SkillReleaseTarget.READ_QUERY_ONLY_TOOLS,
            ),
        )
        self.assertEqual(
            LATER_RELEASE_SKILL_TARGETS,
            (
                SkillReleaseTarget.PLUGIN_PROVIDED_SOURCES,
                SkillReleaseTarget.BUNDLED_SOURCES,
                SkillReleaseTarget.USER_LOCAL_SOURCES,
                SkillReleaseTarget.REMOTE_PREINSTALLED_SOURCES,
                SkillReleaseTarget.RICHER_MATCHING,
                SkillReleaseTarget.BACKEND_MATERIALIZED_SOURCES,
            ),
        )
        self.assertFalse(SKILL_BACKWARD_COMPATIBILITY_REQUIRED)

    def test_canonical_vocabulary_is_complete(self) -> None:
        self.assertEqual(
            set(SKILL_VOCABULARY),
            {
                SkillVocabularyTerm.SOURCE,
                SkillVocabularyTerm.SOURCE_AUTHORITY,
                SkillVocabularyTerm.SOURCE_LABEL,
                SkillVocabularyTerm.SKILL_ID,
                SkillVocabularyTerm.RESOURCE_ID,
                SkillVocabularyTerm.MAIN_RESOURCE,
                SkillVocabularyTerm.REGISTRY_VERSION,
                SkillVocabularyTerm.READ_CURSOR,
                SkillVocabularyTerm.DIAGNOSTIC,
                SkillVocabularyTerm.STATUS,
                SkillVocabularyTerm.PROVENANCE,
            },
        )

    def test_initial_manifest_shape_matches_front_matter_contract(
        self,
    ) -> None:
        self.assertEqual(SKILL_MANIFEST_FORMAT, "markdown_front_matter")
        self.assertEqual(SKILL_MAIN_RESOURCE_ID, "main")
        self.assertEqual(
            REQUIRED_SKILL_MANIFEST_FIELDS,
            (SkillManifestField.NAME, SkillManifestField.DESCRIPTION),
        )
        self.assertIn(
            SkillManifestField.RESOURCES, SUPPORTED_SKILL_MANIFEST_FIELDS
        )

        sample = (FIXTURE_DIR / "pdf.md").read_text(encoding="utf-8")
        compilation = compile_markdown_fixture(sample)

        self.assertEqual(compilation.status, SkillStatus.OK)
        self.assertEqual(len(compilation.items), 1)
        self.assertEqual(compilation.items[0].skill_id, "pdf")
        self.assertEqual(compilation.items[0].main_resource_id, "main")
        self.assertIn("PDF files", compilation.items[0].description)

    def test_section_14_failure_modes_have_deterministic_mapping(self) -> None:
        self.assertEqual(
            set(SECTION_14_FAILURE_DIAGNOSTICS),
            set(EXPECTED_SECTION_14_STATUS_CODES),
        )
        self.assertIs(
            all_failure_diagnostic_contracts(),
            SECTION_14_FAILURE_DIAGNOSTICS,
        )
        for failure_mode, expected in EXPECTED_SECTION_14_STATUS_CODES.items():
            with self.subTest(failure_mode=failure_mode):
                contract = diagnostic_contract_for_failure(failure_mode)
                expected_status, expected_code = expected
                self.assertEqual(contract.status, expected_status)
                self.assertEqual(contract.code, expected_code)
                self.assertEqual(contract.failure_mode, failure_mode)
                self.assertTrue(contract.message)

    def test_canonical_model_facing_tool_schemas_exclude_load(self) -> None:
        schemas = canonical_skills_tool_schemas()
        names: list[str] = []
        for schema in schemas:
            function = schema["function"]
            assert isinstance(function, dict)
            name = function["name"]
            assert isinstance(name, str)
            names.append(name)

        self.assertEqual(tuple(names), CANONICAL_SKILLS_TOOL_NAMES)
        self.assertNotIn("skills.load", names)
        self.assertEqual(
            DISALLOWED_MODEL_FACING_SKILLS_TOOL_NAMES,
            ("skills.load",),
        )

    def test_contract_fixtures_compile_with_expected_diagnostics(self) -> None:
        fixture_paths = sorted(FIXTURE_DIR.glob("*.json"))
        self.assertEqual(
            {path.stem for path in fixture_paths},
            set(EXPECTED_FIXTURES),
        )

        for path in fixture_paths:
            with self.subTest(fixture=path.name):
                fixture = load_fixture(path)
                compilation = compile_json_fixture(fixture)
                repeated = compile_json_fixture(fixture)
                expected_status, expected_code = EXPECTED_FIXTURES[path.stem]

                self.assertEqual(
                    compilation.as_model_dict(),
                    repeated.as_model_dict(),
                )
                self.assertEqual(compilation.status, expected_status)
                if expected_code is None:
                    self.assertEqual(compilation.diagnostics, ())
                    self.assertEqual(len(compilation.items), 1)
                else:
                    self.assertGreaterEqual(len(compilation.diagnostics), 1)
                    self.assertEqual(
                        compilation.diagnostics[0].code,
                        expected_code,
                    )

    def test_runtime_unavailable_and_empty_content_fail_closed(self) -> None:
        runtime_unavailable = compile_skill_contract_fixtures(
            (
                SkillContractFixture(
                    source_label="contract-runtime",
                    content=(
                        "---\n"
                        "name: runtime\n"
                        "description: Runtime unavailable.\n"
                        "---\n"
                    ),
                    runtime_available=False,
                ),
            )
        )
        empty_content = compile_skill_contract_fixtures(
            (SkillContractFixture(source_label="contract-empty-content"),)
        )

        self.assertEqual(
            runtime_unavailable.diagnostics[0].code,
            SkillDiagnosticCode.RUNTIME_SOURCE_UNAVAILABLE,
        )
        self.assertEqual(empty_content.status, SkillStatus.MALFORMED)
        self.assertEqual(
            empty_content.diagnostics[0].code,
            SkillDiagnosticCode.MANIFEST_MALFORMED,
        )

    def test_contract_compiler_branch_regressions(self) -> None:
        valid = compile_markdown_fixture(
            "---\nname: pdf\ndescription: PDF guidance.\n---\n# PDF\n"
        )
        unsafe_version = compile_markdown_fixture(
            "---\n"
            "name: pdf\n"
            "description: PDF guidance.\n"
            "version: $HOME/skills/pdf/SKILL.md\n"
            "---\n"
            "# PDF\n"
        )
        invalid_tag = compile_markdown_fixture(
            "---\n"
            "name: pdf\n"
            "description: PDF guidance.\n"
            'tags: ["Bad Tag"]\n'
            "---\n"
            "# PDF\n"
        )
        nested_traversal_resource = compile_markdown_fixture(
            "---\n"
            "name: pdf\n"
            "description: PDF guidance.\n"
            'resources: ["references/../secret.md"]\n'
            "---\n"
            "# PDF\n"
        )

        self.assertEqual(valid.status, SkillStatus.OK)
        self.assertEqual(valid.diagnostics, ())
        self.assertTrue(valid.registry_version.startswith("skills-contract:"))
        self.assertEqual(unsafe_version.status, SkillStatus.MALFORMED)
        self.assertEqual(
            unsafe_version.diagnostics[0].path,
            "manifest.version",
        )
        self.assertEqual(invalid_tag.status, SkillStatus.MALFORMED)
        self.assertEqual(
            invalid_tag.diagnostics[0].path,
            "manifest.tags",
        )
        self.assertEqual(
            nested_traversal_resource.status,
            SkillStatus.POLICY_DENIED,
        )
        self.assertEqual(
            nested_traversal_resource.diagnostics[0].code,
            SkillDiagnosticCode.RESOURCE_OUTSIDE_ROOT,
        )

    def test_front_matter_edge_cases_have_deterministic_diagnostics(
        self,
    ) -> None:
        valid_with_comments = compile_skill_contract_fixtures(
            (
                SkillContractFixture(
                    source_label="contract-comments",
                    content=(
                        "---\n"
                        "# ignored comment\n"
                        "\n"
                        "name: comment-skill\n"
                        "description: Unquoted description.\n"
                        "enabled: true\n"
                        "---\n"
                        "# Body\n"
                    ),
                ),
            )
        )
        self.assertEqual(valid_with_comments.status, SkillStatus.OK)
        self.assertEqual(
            valid_with_comments.items[0].skill_id,
            "comment-skill",
        )

        malformed_cases = (
            "not front matter",
            "---\nname: pdf\n",
            "---\nname pdf\n---\n",
            "---\nname: pdf\nunknown: yes\ndescription: PDF.\n---\n",
            "---\nname: pdf\nname: other\ndescription: PDF.\n---\n",
            "---\nname: pdf\ndescription: PDF.\nenabled: yes\n---\n",
            "---\nname: pdf\ndescription: PDF.\ntags: [\n---\n",
            '---\nname: pdf\ndescription: PDF.\ntags: "pdf"\n---\n',
            "---\nname:\ndescription: PDF.\n---\n",
            '---\nname: ""\ndescription: PDF.\n---\n',
            '---\nname: "unterminated\ndescription: PDF.\n---\n',
            '---\nname: "pdf",\ndescription: PDF.\n---\n',
            "---\nname: PDF\ndescription: PDF.\n---\n",
            '---\nname: pdf\ndescription: PDF.\ntags: ["Bad Tag"]\n---\n',
            (
                "---\n"
                "name: pdf\n"
                "description: PDF.\n"
                'resources: ["bad\\\\path"]\n'
                "---\n"
            ),
            (
                "---\n"
                "name: pdf\n"
                "description: PDF.\n"
                'resources: ["bad\\x00path"]\n'
                "---\n"
            ),
            (
                "---\n"
                "name: pdf\n"
                "description: PDF.\n"
                'resources: ["/secret.md"]\n'
                "---\n"
            ),
        )

        for content in malformed_cases:
            with self.subTest(content=content):
                compilation = compile_skill_contract_fixtures(
                    (
                        SkillContractFixture(
                            source_label="contract-malformed-edge",
                            content=content,
                        ),
                    )
                )
                self.assertNotEqual(compilation.status, SkillStatus.OK)
                self.assertGreaterEqual(len(compilation.diagnostics), 1)

    def test_path_bearing_metadata_fails_closed(self) -> None:
        cases = (
            (
                "description: /Users/example/private/skills/pdf/SKILL.md",
                "manifest.description",
                ("/Users/", "private"),
            ),
            (
                "description: Use /private/tmp/skills/pdf/SKILL.md",
                "manifest.description",
                ("/private/", "/tmp/"),
            ),
            (
                "description: Use /opt/avalan/skills/pdf/SKILL.md",
                "manifest.description",
                ("/opt/", "/skills/"),
            ),
            (
                "description: Use C:\\Users\\example\\.codex\\skill",
                "manifest.description",
                ("C:\\Users", ".codex"),
            ),
            (
                "description: ../private/skill instructions",
                "manifest.description",
                ("../private",),
            ),
            (
                "description: Use private/agent.toml",
                "manifest.description",
                ("private/agent.toml",),
            ),
            (
                "description: Use secrets/api_key",
                "manifest.description",
                ("secrets/api_key",),
            ),
            (
                "description: Use ~/.codex/skills/pdf/SKILL.md",
                "manifest.description",
                ("~/.codex",),
            ),
            (
                "description: Use .aws/credentials",
                "manifest.description",
                (".aws/credentials",),
            ),
            (
                "description: Use .codex/config.toml",
                "manifest.description",
                (".codex/config.toml",),
            ),
            (
                "description: Use .config/avalan/settings.toml",
                "manifest.description",
                (".config/avalan/settings.toml",),
            ),
            (
                "description: Use .env",
                "manifest.description",
                (".env",),
            ),
            (
                "description: Use .ssh/config for setup",
                "manifest.description",
                (".ssh/config",),
            ),
            (
                "version: $HOME/skills/pdf/SKILL.md",
                "manifest.version",
                ("$HOME",),
            ),
        )

        for manifest_line, expected_path, unsafe_fragments in cases:
            with self.subTest(manifest_line=manifest_line):
                description_line = (
                    ""
                    if manifest_line.startswith("description:")
                    else "description: PDF guidance.\n"
                )
                compilation = compile_markdown_fixture(
                    "---\n"
                    "name: pdf\n"
                    f"{description_line}"
                    f"{manifest_line}\n"
                    "---\n"
                    "# PDF\n"
                )
                encoded = dumps(compilation.as_model_dict(), sort_keys=True)

                self.assertEqual(compilation.status, SkillStatus.MALFORMED)
                self.assertEqual(compilation.items, ())
                self.assertEqual(
                    compilation.diagnostics[0].code,
                    SkillDiagnosticCode.MANIFEST_MALFORMED,
                )
                self.assertEqual(
                    compilation.diagnostics[0].path,
                    expected_path,
                )
                for unsafe_fragment in unsafe_fragments:
                    self.assertNotIn(unsafe_fragment, encoded)

    def test_unsafe_resource_paths_fail_public_compilation(self) -> None:
        cases = (
            (
                '["$HOME/skills/SKILL.md"]',
                SkillStatus.POLICY_DENIED,
                SkillDiagnosticCode.RESOURCE_OUTSIDE_ROOT,
            ),
            (
                '["~/.codex/skills/SKILL.md"]',
                SkillStatus.POLICY_DENIED,
                SkillDiagnosticCode.RESOURCE_OUTSIDE_ROOT,
            ),
            (
                '["/secret.md"]',
                SkillStatus.POLICY_DENIED,
                SkillDiagnosticCode.RESOURCE_OUTSIDE_ROOT,
            ),
            (
                '["../secret.md"]',
                SkillStatus.POLICY_DENIED,
                SkillDiagnosticCode.RESOURCE_OUTSIDE_ROOT,
            ),
            (
                '["bad\\x00path"]',
                SkillStatus.POLICY_DENIED,
                SkillDiagnosticCode.RESOURCE_OUTSIDE_ROOT,
            ),
            (
                '["bad\\\\path"]',
                SkillStatus.MALFORMED,
                SkillDiagnosticCode.MANIFEST_MALFORMED,
            ),
            (
                '["."]',
                SkillStatus.MALFORMED,
                SkillDiagnosticCode.MANIFEST_MALFORMED,
            ),
            (
                '["references/"]',
                SkillStatus.MALFORMED,
                SkillDiagnosticCode.MANIFEST_MALFORMED,
            ),
            (
                '["references/./rendering.md"]',
                SkillStatus.MALFORMED,
                SkillDiagnosticCode.MANIFEST_MALFORMED,
            ),
        )

        for resources, expected_status, expected_code in cases:
            with self.subTest(resources=resources):
                compilation = compile_markdown_fixture(
                    "---\n"
                    "name: pdf\n"
                    "description: PDF guidance.\n"
                    f"resources: {resources}\n"
                    "---\n"
                    "# PDF\n"
                )

                self.assertEqual(compilation.status, expected_status)
                self.assertEqual(
                    compilation.diagnostics[0].code, expected_code
                )
                self.assertEqual(
                    compilation.diagnostics[0].path,
                    "manifest.resources",
                )

    def test_model_facing_fixture_responses_do_not_expose_host_paths(
        self,
    ) -> None:
        for path in sorted(FIXTURE_DIR.glob("*.json")):
            with self.subTest(fixture=path.name):
                response = compile_json_fixture(
                    load_fixture(path)
                ).as_model_dict()
                encoded = dumps(response, sort_keys=True)

                self.assertNotIn("/Users/", encoded)
                self.assertNotIn("/private/", encoded)
                self.assertNotIn("resource_path", encoded)
                self.assertNotIn('"root"', encoded)

    def test_unimplemented_surfaces_reject_skills_syntax(self) -> None:
        self.assertEqual(
            SKILLS_SYNTAX_REJECTING_SURFACES,
            (
                SkillSyntaxSurface.SDK_SETTINGS,
                SkillSyntaxSurface.CLI_SETTINGS,
                SkillSyntaxSurface.AGENT_TOML,
                SkillSyntaxSurface.FLOW_DEFINITION,
                SkillSyntaxSurface.TASK_DEFINITION,
                SkillSyntaxSurface.SERVER_REQUEST,
                SkillSyntaxSurface.WORKER_ENVELOPE,
            ),
        )
        for surface in SKILLS_SYNTAX_REJECTING_SURFACES:
            with self.subTest(surface=surface):
                self.assertTrue(rejects_skills_syntax(surface))
                diagnostic = reject_skills_syntax(surface)
                self.assertEqual(
                    diagnostic.code,
                    SkillDiagnosticCode.SKILLS_SYNTAX_UNSUPPORTED,
                )
                self.assertEqual(diagnostic.status, SkillStatus.BLOCKED)

    def test_model_facing_diagnostics_omit_unsafe_paths(self) -> None:
        for safe_path in (
            "skills.request.name",
            "manifest.name",
            "manifest.line.2",
            "source.policy",
            "resource.main",
        ):
            with self.subTest(safe_path=safe_path):
                diagnostic = reject_skills_syntax(
                    SkillSyntaxSurface.SDK_SETTINGS,
                    path=safe_path,
                )

                self.assertEqual(
                    diagnostic.as_model_dict()["path"],
                    safe_path,
                )

        unsafe_paths = (
            "/Users/example/private/agent.toml",
            "/private/tmp/agent.toml",
            "~/.codex/agent.toml",
            "$HOME/agent.toml",
            "../agent.toml",
            "manifest/../secret",
            "manifest\\secret",
            "manifest\x00secret",
            "manifest:secret",
            ".codex/agent.toml",
            "private/agent.toml",
            "users/mariano/.codex/agent.toml",
            "users/mariano/private/agent.toml",
            "home/mariano/.ssh/config",
            "users.mariano.private.agent",
            "codex.agent.toml",
            "skills..name",
            "source/private/agent.toml",
            "resource/main",
        )

        for unsafe_path in unsafe_paths:
            with self.subTest(unsafe_path=unsafe_path):
                diagnostic = reject_skills_syntax(
                    SkillSyntaxSurface.SDK_SETTINGS,
                    path=unsafe_path,
                )

                self.assertNotIn("path", diagnostic.as_model_dict())


if __name__ == "__main__":
    main()
