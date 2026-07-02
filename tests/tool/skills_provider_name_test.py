from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, main

from avalan.entities import (
    ToolManagerSettings,
    ToolNamePolicyMode,
    ToolNamePolicySettings,
    ToolNameResolutionStatus,
)
from avalan.model.provider import ProviderFamily
from avalan.skill import (
    SkillConfiguredSource,
    SkillRegistry,
    WorkspaceSkillSourceAuthority,
    build_skill_registry,
    resolve_skill_sources,
)
from avalan.tool.manager import ToolManager
from avalan.tool.skills import SkillsToolSet


class SkillsProviderNameTestCase(IsolatedAsyncioTestCase):
    async def test_provider_safe_names_round_trip_for_skills_tools(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(root / "SKILL.md")
            registry = await _registry(root)
            manager = ToolManager.create_instance(
                available_toolsets=[SkillsToolSet(registry)],
                enable_tools=["skills"],
                settings=ToolManagerSettings(
                    tool_name_policy=ToolNamePolicySettings(
                        mode=ToolNamePolicyMode.SANITIZED
                    )
                ),
            )

        provider_schemas = manager.provider_json_schemas(
            provider_family=ProviderFamily.OPENAI.value
        )
        assert provider_schemas is not None
        self.assertEqual(
            [schema["function"]["name"] for schema in provider_schemas],
            [
                "skills_list",
                "skills_match",
                "skills_read",
                "skills_check",
            ],
        )

        for canonical_name, provider_name in (
            ("skills.list", "skills_list"),
            ("skills.match", "skills_match"),
            ("skills.read", "skills_read"),
            ("skills.check", "skills_check"),
        ):
            with self.subTest(canonical_name=canonical_name):
                self.assertEqual(
                    manager.provider_tool_name(
                        canonical_name,
                        provider_family=ProviderFamily.OPENAI.value,
                    ),
                    provider_name,
                )
                self.assertEqual(
                    manager.canonical_tool_name(
                        provider_name,
                        provider_family=ProviderFamily.OPENAI.value,
                    ),
                    canonical_name,
                )
                resolution = manager.resolve_tool_name(
                    provider_name,
                    provider_originated=True,
                )
                self.assertIs(
                    resolution.status,
                    ToolNameResolutionStatus.EXACT,
                )
                self.assertEqual(resolution.canonical_name, canonical_name)


async def _registry(root: Path) -> SkillRegistry:
    source_result = await resolve_skill_sources(
        (
            SkillConfiguredSource(
                label="Workspace Main",
                authority=WorkspaceSkillSourceAuthority(),
                root_path=root,
            ),
        )
    )
    return await build_skill_registry(source_result)


def _write_skill(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        "name: pdf\n"
        "description: PDF rendering guidance.\n"
        "resources: []\n"
        "---\n"
        "# PDF Body\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
