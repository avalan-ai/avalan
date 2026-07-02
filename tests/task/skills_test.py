from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, TestCase, main

from avalan.skill import (
    SkillBootstrapPromptSettings,
    SkillReadLimits,
    SkillSettingsSurface,
    SkillSourceAuthorityKind,
    SkillSourceConfig,
    SkillStatus,
    TrustedSkillSettings,
    UntrustedSkillSettings,
    UserLocalSkillSourceAuthority,
    WorkspaceSkillSourceAuthority,
)
from avalan.task import (
    TaskDefinition,
    TaskExecutionTarget,
    TaskInputContract,
    TaskMetadata,
    TaskOutputContract,
    TaskValidationError,
)
from avalan.task.skills import (
    build_task_skill_registry,
    revalidate_task_skills_for_worker,
    task_definition_with_skills_identity,
    task_effective_skills_settings,
    task_enabled_skills_tools,
    task_skill_settings_allow,
    task_skills_identity,
)


class TaskSkillsIdentityTest(IsolatedAsyncioTestCase):
    async def test_identity_rejects_stored_identity_without_settings(
        self,
    ) -> None:
        definition = replace(
            _definition(),
            skills_identity={"version": "task.skills.v1"},
        )

        with self.assertRaises(TaskValidationError) as error:
            await task_definition_with_skills_identity(definition)

        self.assertEqual(
            error.exception.issues[0].code,
            "task.skills_registry_missing",
        )

    async def test_worker_revalidation_rejects_skills_without_identity(
        self,
    ) -> None:
        definition = replace(_definition(), skills=TrustedSkillSettings())

        with self.assertRaises(TaskValidationError) as error:
            await revalidate_task_skills_for_worker(definition)

        self.assertEqual(
            error.exception.issues[0].code,
            "task.skills_registry_missing",
        )

    async def test_worker_revalidation_rejects_non_mapping_identity(
        self,
    ) -> None:
        with self.assertRaises(TaskValidationError) as error:
            await revalidate_task_skills_for_worker(
                _definition(),
                expected_identity=("not", "a", "mapping"),  # type: ignore[arg-type]
            )

        self.assertEqual(
            error.exception.issues[0].code,
            "task.skills_registry_stale",
        )

    async def test_worker_revalidation_reports_expected_policy_statuses(
        self,
    ) -> None:
        settings = TrustedSkillSettings(enabled=False)
        definition = _definition()

        for status, code in (
            (
                SkillStatus.POLICY_DENIED,
                "task.skills_registry_policy_denied",
            ),
            (SkillStatus.MALFORMED, "task.skills_registry_malformed"),
        ):
            with self.subTest(status=status.value):
                with self.assertRaises(TaskValidationError) as error:
                    await revalidate_task_skills_for_worker(
                        definition,
                        trusted_settings=settings,
                        expected_identity={"status": status.value},
                    )

                self.assertEqual(error.exception.issues[0].code, code)

    async def test_worker_revalidation_detects_widened_identity_fields(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(root / "skills-a" / "pdf" / "SKILL.md")
            _write_skill(root / "skills-b" / "ocr" / "SKILL.md")
            settings = _trusted_skills(
                root / "skills-a",
                extra_sources=(
                    SkillSourceConfig(
                        label="workspace-b",
                        authority=WorkspaceSkillSourceAuthority(),
                        root_path=root / "skills-b",
                    ),
                ),
                allowed_skill_ids=("pdf", "ocr"),
            )
            registry = await build_task_skill_registry(settings)
            actual = task_skills_identity(
                settings,
                registry=registry,
                enabled_tools=("skills.read",),
                target_type=_definition().execution.type,
            )

            widened_cases = (
                {"enabled": False},
                {"bootstrap_enabled": False},
                {"source_labels": ("workspace-main",)},
                {"allowed_skill_ids": ("pdf",)},
            )
            for patch in widened_cases:
                with self.subTest(patch=patch):
                    expected = {**actual, **patch}
                    with self.assertRaises(TaskValidationError) as error:
                        await revalidate_task_skills_for_worker(
                            _definition(),
                            trusted_settings=settings,
                            registry=registry,
                            expected_identity=expected,
                        )

                    self.assertEqual(
                        error.exception.issues[0].code,
                        "task.skills_registry_widened",
                    )

    async def test_worker_revalidation_stales_on_non_mapping_limits(
        self,
    ) -> None:
        settings = TrustedSkillSettings(enabled=False)
        actual = task_skills_identity(
            settings,
            registry=None,
            enabled_tools=("", "skills.read"),
            target_type=_definition().execution.type,
        )
        expected = {
            **actual,
            "enabled_tools": "skills.read",
            "read_limits": "wide-open",
        }

        with self.assertRaises(TaskValidationError) as error:
            await revalidate_task_skills_for_worker(
                _definition(),
                trusted_settings=settings,
                expected_identity=expected,
            )

        self.assertEqual(
            error.exception.issues[0].code,
            "task.skills_registry_stale",
        )

    async def test_registry_builder_ignores_sources_without_roots(
        self,
    ) -> None:
        settings = TrustedSkillSettings(
            sources=(
                SkillSourceConfig(
                    label="workspace-empty",
                    authority=WorkspaceSkillSourceAuthority(),
                ),
            ),
        )

        registry = await build_task_skill_registry(settings)

        self.assertIn(
            registry.status,
            {SkillStatus.EMPTY, SkillStatus.NOT_FOUND, SkillStatus.OK},
        )


class TaskSkillsSettingsTest(TestCase):
    def test_effective_settings_rejects_unsupported_targets(self) -> None:
        definition = replace(
            _definition(),
            execution=TaskExecutionTarget.callable("module:function"),
            skills_identity={"version": "task.skills.v1"},
        )

        with self.assertRaises(TaskValidationError) as error:
            task_effective_skills_settings(definition)

        self.assertEqual(
            error.exception.issues[0].code,
            "task.skills_unsupported_target",
        )

    def test_effective_settings_ignores_plain_unsupported_targets(
        self,
    ) -> None:
        definition = replace(
            _definition(),
            execution=TaskExecutionTarget.callable("module:function"),
        )

        self.assertIsNone(task_effective_skills_settings(definition))

    def test_effective_settings_requires_trusted_parent_for_config(
        self,
    ) -> None:
        definition = replace(
            _definition(),
            skills_config=UntrustedSkillSettings(
                surface=SkillSettingsSurface.TASK,
                enabled=False,
            ),
        )

        with self.assertRaises(TaskValidationError) as error:
            task_effective_skills_settings(
                definition,
                trust_definition_skills=False,
            )

        self.assertEqual(
            error.exception.issues[0].code,
            "task.skills_registry_missing",
        )

    def test_effective_settings_returns_narrowed_override(self) -> None:
        definition = replace(
            _definition(),
            skills_config=UntrustedSkillSettings(
                surface=SkillSettingsSurface.TASK,
                read_limits=SkillReadLimits(max_lines_per_read=20),
            ),
        )

        settings = task_effective_skills_settings(
            definition,
            trusted_settings=TrustedSkillSettings(
                read_limits=SkillReadLimits(max_lines_per_read=100),
            ),
            trust_definition_skills=False,
        )

        assert settings is not None
        self.assertEqual(settings.read_limits.max_lines_per_read, 20)

    def test_effective_settings_reports_policy_denied_override(self) -> None:
        definition = replace(
            _definition(),
            skills_config=UntrustedSkillSettings(
                surface=SkillSettingsSurface.TASK,
                read_limits=SkillReadLimits(max_lines_per_read=200),
            ),
        )

        with self.assertRaises(TaskValidationError) as error:
            task_effective_skills_settings(
                definition,
                trusted_settings=TrustedSkillSettings(
                    read_limits=SkillReadLimits(max_lines_per_read=20),
                ),
                trust_definition_skills=False,
            )

        self.assertEqual(
            error.exception.issues[0].code,
            "task.skills_registry_policy_denied",
        )

    def test_skill_settings_allow_rejects_widened_settings(self) -> None:
        workspace_source = SkillSourceConfig(
            label="workspace-main",
            authority=WorkspaceSkillSourceAuthority(),
            root_path="/tmp/workspace-main",
        )
        other_source = SkillSourceConfig(
            label="workspace-other",
            authority=WorkspaceSkillSourceAuthority(),
            root_path="/tmp/workspace-other",
        )
        cases = (
            (
                TrustedSkillSettings(enabled=False),
                TrustedSkillSettings(enabled=True),
            ),
            (
                TrustedSkillSettings(bootstrap_enabled=False),
                TrustedSkillSettings(bootstrap_enabled=True),
            ),
            (
                TrustedSkillSettings(
                    bootstrap_prompt=SkillBootstrapPromptSettings(
                        include_read_guidance=False
                    )
                ),
                TrustedSkillSettings(),
            ),
            (
                TrustedSkillSettings(
                    bootstrap_prompt=SkillBootstrapPromptSettings(
                        additional_instructions=("Approved.",)
                    )
                ),
                TrustedSkillSettings(
                    bootstrap_prompt=SkillBootstrapPromptSettings(
                        additional_instructions=("Unapproved.",)
                    )
                ),
            ),
            (
                TrustedSkillSettings(
                    authority_kinds=(SkillSourceAuthorityKind.WORKSPACE,),
                ),
                TrustedSkillSettings(
                    authority_kinds=(SkillSourceAuthorityKind.USER_LOCAL,),
                ),
            ),
            (
                TrustedSkillSettings(sources=(workspace_source,)),
                TrustedSkillSettings(sources=(other_source,)),
            ),
            (
                TrustedSkillSettings(allowed_skill_ids=("pdf",)),
                TrustedSkillSettings(),
            ),
            (
                TrustedSkillSettings(allowed_skill_ids=("pdf",)),
                TrustedSkillSettings(allowed_skill_ids=("pdf", "ocr")),
            ),
            (
                TrustedSkillSettings(
                    read_limits=SkillReadLimits(max_lines_per_read=20)
                ),
                TrustedSkillSettings(
                    read_limits=SkillReadLimits(max_lines_per_read=200)
                ),
            ),
        )

        for parent, child in cases:
            with self.subTest(parent=parent, child=child):
                self.assertFalse(task_skill_settings_allow(parent, child))

    def test_skill_settings_allow_matching_explicit_sources(self) -> None:
        source = SkillSourceConfig(
            label="workspace-main",
            authority=WorkspaceSkillSourceAuthority(),
            root_path="/tmp/workspace-main",
        )

        self.assertTrue(
            task_skill_settings_allow(
                TrustedSkillSettings(sources=(source,)),
                TrustedSkillSettings(sources=(source,)),
            )
        )

    def test_user_local_authority_constructor_is_available(self) -> None:
        settings = TrustedSkillSettings(
            authority_kinds=(SkillSourceAuthorityKind.USER_LOCAL,),
            sources=(
                SkillSourceConfig(
                    label="user-main",
                    authority=UserLocalSkillSourceAuthority(),
                    root_path="/tmp/user-main",
                ),
            ),
        )

        self.assertEqual(settings.sources[0].label, "user-main")


class TaskSkillsToolsDiscoveryTest(IsolatedAsyncioTestCase):
    async def test_agent_tool_discovery_handles_string_and_invalid_values(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            agent = root / "agents" / "assistant.toml"
            agent.parent.mkdir()
            agent.write_text(
                """
                [tool]
                enable = "skills.read"
                """,
                encoding="utf-8",
            )

            tools = await task_enabled_skills_tools(
                replace(
                    _definition(),
                    execution=TaskExecutionTarget.agent(
                        "agents/assistant.toml"
                    ),
                ),
                schema_base_path=root / "task.toml",
            )
            agent.write_text(
                """
                [tool]
                enable = 123
                """,
                encoding="utf-8",
            )
            empty_tools = await task_enabled_skills_tools(
                replace(
                    _definition(),
                    execution=TaskExecutionTarget.agent(
                        "agents/assistant.toml"
                    ),
                ),
                schema_base_path=root / "task.toml",
            )

        self.assertEqual(tools, ("skills.read",))
        self.assertEqual(empty_tools, ())

    async def test_flow_tool_discovery_handles_config_and_odd_nodes(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            no_nodes = root / "flows" / "no_nodes.toml"
            odd_nodes = root / "flows" / "odd.toml"
            config_node = root / "flows" / "config.toml"
            no_nodes.parent.mkdir()
            no_nodes.write_text("[flow]\nname = 'no-nodes'\n", "utf-8")
            odd_nodes.write_text("[nodes]\nanswer = 'skip-me'\n", "utf-8")
            config_node.write_text(
                """
                [nodes.answer]
                type = "tool"

                [nodes.answer.config]
                canonical_name = "skills.read"
                """,
                "utf-8",
            )

            no_node_tools = await task_enabled_skills_tools(
                _flow_definition("flows/no_nodes.toml"),
                schema_base_path=root / "task.toml",
            )
            odd_node_tools = await task_enabled_skills_tools(
                _flow_definition("flows/odd.toml"),
                schema_base_path=root / "task.toml",
            )
            config_tools = await task_enabled_skills_tools(
                _flow_definition("flows/config.toml"),
                schema_base_path=root / "task.toml",
            )

        self.assertEqual(no_node_tools, ())
        self.assertEqual(odd_node_tools, ())
        self.assertEqual(config_tools, ("skills.read",))

    async def test_ref_discovery_fails_closed_for_bad_base_and_refs(
        self,
    ) -> None:
        with self.assertRaises(TaskValidationError) as bad_base:
            await task_enabled_skills_tools(
                replace(
                    _definition(),
                    execution=TaskExecutionTarget.agent("agent.toml"),
                ),
                schema_base_path=object(),  # type: ignore[arg-type]
                require_trusted_refs=True,
            )
        with self.assertRaises(TaskValidationError) as bad_ref:
            await task_enabled_skills_tools(
                replace(
                    _definition(),
                    execution=TaskExecutionTarget.agent("../agent.toml"),
                ),
                schema_base_path="/tmp/task.toml",
                require_trusted_refs=True,
            )
        with self.assertRaises(TaskValidationError) as remote_ref:
            await task_enabled_skills_tools(
                replace(
                    _definition(),
                    execution=TaskExecutionTarget.agent(
                        "https://example.test/agent.toml"
                    ),
                ),
                schema_base_path="/tmp/task.toml",
                require_trusted_refs=True,
            )
        with self.assertRaises(TaskValidationError) as absolute_ref:
            await task_enabled_skills_tools(
                replace(
                    _definition(),
                    execution=TaskExecutionTarget.agent("/tmp/agent.toml"),
                ),
                schema_base_path="/tmp/task.toml",
                require_trusted_refs=True,
            )

        for error in (bad_base, bad_ref, remote_ref, absolute_ref):
            self.assertEqual(
                error.exception.issues[0].code,
                "task.skills_registry_unavailable",
            )

    async def test_ref_discovery_non_strict_bad_base_and_ref_are_empty(
        self,
    ) -> None:
        bad_base = await task_enabled_skills_tools(
            replace(
                _definition(),
                execution=TaskExecutionTarget.agent("agent.toml"),
            ),
            schema_base_path=object(),  # type: ignore[arg-type]
        )
        bad_ref = await task_enabled_skills_tools(
            replace(
                _definition(),
                execution=TaskExecutionTarget.agent("../agent.toml"),
            ),
            schema_base_path="/tmp/task.toml",
        )

        self.assertEqual(bad_base, ())
        self.assertEqual(bad_ref, ())

    async def test_ref_discovery_handles_malformed_and_missing_refs(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            malformed = root / "agents" / "bad.toml"
            malformed.parent.mkdir()
            malformed.write_text("[tool\n", "utf-8")
            definition = replace(
                _definition(),
                execution=TaskExecutionTarget.agent("agents/bad.toml"),
            )

            non_strict_malformed = await task_enabled_skills_tools(
                definition,
                schema_base_path=root / "task.toml",
            )
            with self.assertRaises(TaskValidationError) as strict_malformed:
                await task_enabled_skills_tools(
                    definition,
                    schema_base_path=root / "task.toml",
                    require_trusted_refs=True,
                )
            missing = await task_enabled_skills_tools(
                replace(
                    _definition(),
                    execution=TaskExecutionTarget.agent("agents/missing.toml"),
                ),
                schema_base_path=root / "task.toml",
            )

        self.assertEqual(non_strict_malformed, ())
        self.assertEqual(missing, ())
        self.assertEqual(
            strict_malformed.exception.issues[0].code,
            "task.skills_registry_malformed",
        )


def _definition() -> TaskDefinition:
    return TaskDefinition(
        task=TaskMetadata(name="skills", version="1"),
        input=TaskInputContract.string(),
        output=TaskOutputContract.text(),
        execution=TaskExecutionTarget.tool("skills.read"),
    )


def _flow_definition(ref: str) -> TaskDefinition:
    return replace(_definition(), execution=TaskExecutionTarget.flow(ref))


def _trusted_skills(
    root: Path,
    *,
    extra_sources: tuple[SkillSourceConfig, ...] = (),
    allowed_skill_ids: tuple[str, ...] = (),
) -> TrustedSkillSettings:
    return TrustedSkillSettings(
        sources=(
            SkillSourceConfig(
                label="workspace-main",
                authority=WorkspaceSkillSourceAuthority(),
                root_path=root,
            ),
        )
        + extra_sources,
        allowed_skill_ids=allowed_skill_ids,
    )


def _write_skill(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        f"name: {path.parent.name}\n"
        "description: Skill body.\n"
        "resources: []\n"
        "---\n"
        "# Body\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
