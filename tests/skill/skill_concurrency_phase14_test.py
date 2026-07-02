from asyncio import gather
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase, main

from avalan.entities import (
    ToolCall,
    ToolCallContext,
    ToolCallResult,
    ToolManagerSettings,
)
from avalan.skill import (
    SkillConfiguredSource,
    SkillRegistry,
    SkillStatus,
    WorkspaceSkillSourceAuthority,
    build_skill_registry,
    resolve_skill_sources,
)
from avalan.tool.manager import ToolManager
from avalan.tool.skills import SkillsToolSet


class SkillConcurrencyPhase14Test(IsolatedAsyncioTestCase):
    async def test_parallel_public_calls_are_deterministic(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "pdf" / "SKILL.md",
                name="pdf",
                description="PDF rendering guidance.",
                tags=("pdf", "render"),
                body="# PDF\nRender pages.\n",
            )
            _write_skill(
                root / "docx" / "SKILL.md",
                name="docx",
                description="DOCX authoring guidance.",
                tags=("docx", "write"),
                body="# DOCX\nWrite paragraphs.\n",
            )
            registry = await _registry(root)
            manager = ToolManager.create_instance(
                available_toolsets=[SkillsToolSet(registry)],
                enable_tools=["skills"],
                settings=ToolManagerSettings(),
            )

            rounds = await gather(
                *(_public_call_round(manager) for _ in range(8))
            )

        self.assertEqual(tuple(rounds), tuple(rounds[0] for _ in rounds))
        listed, matched, checked, read = rounds[0]
        self.assertEqual(listed["status"], SkillStatus.OK.value)
        self.assertEqual(matched["status"], SkillStatus.OK.value)
        self.assertEqual(checked["status"], SkillStatus.OK.value)
        self.assertEqual(read["status"], SkillStatus.OK.value)
        self.assertIn("content", read)


async def _public_call_round(
    manager: ToolManager,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    listed, matched, checked, read = await gather(
        _execute(
            manager,
            ToolCall(id="phase14-list", name="skills.list", arguments={}),
        ),
        _execute(
            manager,
            ToolCall(
                id="phase14-match",
                name="skills.match",
                arguments={"query": "render pdf", "max_results": 1},
            ),
        ),
        _execute(
            manager,
            ToolCall(
                id="phase14-check",
                name="skills.check",
                arguments={"skill": "pdf"},
            ),
        ),
        _execute(
            manager,
            ToolCall(
                id="phase14-read",
                name="skills.read",
                arguments={"skill": "pdf"},
            ),
        ),
    )
    return listed, matched, checked, read


async def _execute(
    manager: ToolManager,
    call: ToolCall,
) -> dict[str, Any]:
    outcome = await manager(call, ToolCallContext())
    assert isinstance(outcome, ToolCallResult)
    assert isinstance(outcome.result, dict)
    return cast(dict[str, Any], outcome.result)


async def _registry(root: Path) -> SkillRegistry:
    source_result = await resolve_skill_sources((_config(root),))
    return await build_skill_registry(source_result)


def _config(root: Path) -> SkillConfiguredSource:
    return SkillConfiguredSource(
        label="Workspace Main",
        authority=WorkspaceSkillSourceAuthority(),
        root_path=root,
    )


def _write_skill(
    path: Path,
    *,
    name: str,
    description: str,
    tags: tuple[str, ...],
    body: str,
) -> None:
    tag_line = "tags: [" + ", ".join(f'"{tag}"' for tag in tags) + "]\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        f"{tag_line}"
        "---\n"
        f"{body}",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
