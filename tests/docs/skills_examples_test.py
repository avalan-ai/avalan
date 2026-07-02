from argparse import ArgumentParser, _SubParsersAction
from pathlib import Path
from tomllib import loads
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase, TestCase, main

from avalan.cli.__main__ import CLI
from avalan.entities import ToolCall, ToolCallContext, ToolCallResult
from avalan.flow import FlowDefinitionLoader, tool_flow_node_registry
from avalan.skill import (
    SkillConfiguredSource,
    SkillReadLimits,
    SkillRegistry,
    SkillSettingsSurface,
    SkillSourceConfig,
    SkillStatus,
    TrustedSkillSettings,
    WorkspaceSkillSourceAuthority,
    build_skill_registry,
    parse_untrusted_skill_settings_config,
    resolve_skill_sources,
)
from avalan.task import RunMode, TaskDefinitionLoader, TaskTargetType
from avalan.tool.manager import ToolManager
from avalan.tool.skills import SkillsToolSet

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
EXAMPLES = DOCS / "examples"
SKILLS_ROOT = EXAMPLES / "skills"

SKILLS_FLAGS = (
    "--tool-skills-source",
    "--tool-skills-source-authority",
    "--tool-skills-source-package",
    "--tool-skills-source-allow-hidden",
    "--tool-skills-authority-kind",
    "--tool-skills-skill",
    "--tool-skills-disable",
    "--tool-skills-bootstrap",
    "--tool-skills-diagnostics",
    "--tool-skills-observability",
    "--tool-skills-max-bytes-per-read",
    "--tool-skills-max-lines-per-read",
    "--tool-skills-max-skills",
    "--tool-skills-max-resources-per-skill",
    "--tool-skills-max-indexed-bytes",
    "--tool-skills-max-sources",
    "--tool-skills-max-resources-per-source",
    "--tool-skills-max-source-depth",
    "--tool-skills-max-files-per-source",
    "--tool-skills-max-directory-entries-per-source",
    "--tool-skills-max-active-cursors",
    "--tool-skills-max-cursor-age-seconds",
)


class SkillsDocumentationExamplesTest(IsolatedAsyncioTestCase):
    def test_public_skills_docs_avoid_internal_specs_and_install_commands(
        self,
    ) -> None:
        pdf_skill = (EXAMPLES / "skills" / "pdf" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        examples_readme = (EXAMPLES / "README.md").read_text(encoding="utf-8")
        skills_readme = (EXAMPLES / "skills" / "README.md").read_text(
            encoding="utf-8"
        )

        for forbidden in (
            "uv pip",
            "python3 -m pip",
            "brew install",
            "sudo apt-get",
            "install Poppler",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, pdf_skill)
        self.assertNotIn("specs/SKILL-pdf.md", examples_readme)
        self.assertNotIn("specs/SKILL-pdf.md", skills_readme)

    async def test_docs_skills_registry_lists_and_reads_pdf_skill(
        self,
    ) -> None:
        registry = await _docs_skill_registry()
        manager = ToolManager.create_instance(
            available_toolsets=[SkillsToolSet(registry)],
            enable_tools=["skills"],
        )

        listed = await manager(
            ToolCall(id="list", name="skills.list", arguments={}),
            ToolCallContext(),
        )
        read = await manager(
            ToolCall(
                id="read",
                name="skills.read",
                arguments={"skill": "pdf", "resource_id": "main"},
            ),
            ToolCallContext(),
        )

        listed_result = _result_dict(listed)
        read_result = _result_dict(read)
        self.assertEqual(listed_result["status"], SkillStatus.OK.value)
        self.assertEqual(read_result["status"], SkillStatus.OK.value)
        self.assertNotIn(str(SKILLS_ROOT), str(listed_result))
        self.assertNotIn(str(SKILLS_ROOT), str(read_result))
        content = cast(dict[str, object], read_result["content"])
        self.assertIn("PDF Skill", content["text"])

    async def test_agent_example_skills_section_is_valid_narrowing(
        self,
    ) -> None:
        raw = _load_toml(EXAMPLES / "agent_skills_pdf.toml")
        tool = cast(dict[str, object], raw["tool"])
        self.assertEqual(
            tuple(cast(list[str], tool["enable"])),
            ("skills.match", "skills.read"),
        )

        settings = _trusted_settings()
        skills = cast(dict[str, object], tool["skills"])
        override = parse_untrusted_skill_settings_config(
            skills,
            trusted=settings,
            surface=SkillSettingsSurface.AGENT,
            section="tool.skills",
        )

        self.assertEqual(override.source_labels, ("workspace-main",))
        self.assertEqual(override.skill_ids, ("pdf",))
        self.assertEqual(
            override.read_limits,
            SkillReadLimits(
                max_bytes_per_read=65536,
                max_lines_per_read=2000,
            ),
        )
        self.assertNotIn("sources", skills)

    async def test_task_and_flow_skills_examples_load_with_trusted_settings(
        self,
    ) -> None:
        settings = _trusted_settings()
        task = await TaskDefinitionLoader(skills_settings=settings).load(
            EXAMPLES / "tasks" / "skills_pdf.task.toml"
        )
        registry = await _docs_skill_registry()
        manager = ToolManager.create_instance(
            available_toolsets=[SkillsToolSet(registry)],
            enable_tools=["skills.read"],
        )
        flow_result = await FlowDefinitionLoader(
            tool_flow_node_registry(manager),
            skills_settings=settings,
        ).load_validation_result(EXAMPLES / "tasks" / "skills_read.flow.toml")

        self.assertEqual(task.execution.type, TaskTargetType.AGENT)
        self.assertEqual(task.execution.ref, "agents/skills_pdf_reader.toml")
        self.assertEqual(task.run.mode, RunMode.DIRECT)
        assert task.skills is not None
        self.assertEqual(task.skills.allowed_skill_ids, ("pdf",))
        self.assertTrue(flow_result.ok, flow_result.public_diagnostics)
        assert flow_result.definition is not None
        assert flow_result.definition.skills is not None
        self.assertEqual(
            flow_result.definition.skills.allowed_skill_ids,
            ("pdf",),
        )


class SkillsDocumentationCliHelpTest(TestCase):
    def test_documented_skills_flags_match_agent_help_surfaces(self) -> None:
        parser = CLI._create_parser(
            default_device="cpu",
            cache_dir="/tmp",
            default_locales_path="/tmp",
            default_locale="en_US",
        )
        agent_command_suffixes = (
            "agent message search",
            "agent run",
            "agent serve",
            "agent proxy",
            "agent init",
        )
        task_run_help = _find_parser_with_suffix(
            parser,
            "task run",
        ).format_help()
        flow_run_help = _find_parser_with_suffix(
            parser,
            "flow run",
        ).format_help()
        cli_docs = (DOCS / "CLI.md").read_text(encoding="utf-8")
        tools_docs = (DOCS / "TOOLS.md").read_text(encoding="utf-8")

        for suffix in agent_command_suffixes:
            help_text = _find_parser_with_suffix(parser, suffix).format_help()
            for flag in SKILLS_FLAGS:
                with self.subTest(command=suffix, flag=flag):
                    self.assertIn(flag, help_text)
                    self.assertIn(flag, cli_docs)
        self.assertNotIn("--tool-skills-source", task_run_help)
        self.assertNotIn("--tool-skills-source", flow_run_help)
        self.assertIn("skills.list", tools_docs)
        self.assertIn("skills.match", tools_docs)
        self.assertIn("skills.read", tools_docs)
        self.assertIn("skills.check", tools_docs)
        self.assertIn("skills.syntax_unsupported", tools_docs)


async def _docs_skill_registry() -> SkillRegistry:
    settings = _trusted_settings()
    source_result = await resolve_skill_sources(
        (
            SkillConfiguredSource(
                label="workspace-main",
                authority=WorkspaceSkillSourceAuthority(workspace_id="docs"),
                root_path=SKILLS_ROOT,
            ),
        ),
        settings=settings,
    )
    return await build_skill_registry(source_result, settings=settings)


def _trusted_settings() -> TrustedSkillSettings:
    return TrustedSkillSettings(
        sources=(
            SkillSourceConfig(
                label="workspace-main",
                authority=WorkspaceSkillSourceAuthority(workspace_id="docs"),
                root_path=SKILLS_ROOT,
            ),
        ),
        allowed_skill_ids=("pdf",),
        read_limits=SkillReadLimits(
            max_bytes_per_read=65536,
            max_lines_per_read=2000,
        ),
    )


def _load_toml(path: Path) -> dict[str, object]:
    return cast(dict[str, object], loads(path.read_text(encoding="utf-8")))


def _result_dict(outcome: object) -> dict[str, Any]:
    assert isinstance(outcome, ToolCallResult)
    assert isinstance(outcome.result, dict)
    return cast(dict[str, Any], outcome.result)


def _find_parser_with_suffix(
    parser: ArgumentParser,
    suffix: str,
) -> ArgumentParser:
    parts = suffix.split()
    current = parser
    for part in parts:
        subparsers = _subparsers(current)
        current = subparsers.choices[part]
    return current


def _subparsers(parser: ArgumentParser) -> _SubParsersAction:
    for action in parser._actions:
        if isinstance(action, _SubParsersAction):
            return action
    raise AssertionError(f"parser {parser.prog} has no subparsers")


if __name__ == "__main__":
    main()
