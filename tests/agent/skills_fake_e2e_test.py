from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import AsyncMock, MagicMock

from avalan.agent import Specification
from avalan.agent.engine import EngineAgent
from avalan.entities import (
    EngineUri,
    ToolCall,
    ToolCallContext,
    ToolCallResult,
)
from avalan.memory.manager import MemoryManager
from avalan.model.call import ModelCallContext
from avalan.model.engine import Engine
from avalan.model.manager import ModelManager
from avalan.skill import (
    SkillConfiguredSource,
    SkillRegistry,
    WorkspaceSkillSourceAuthority,
    build_skill_registry,
    resolve_skill_sources,
)
from avalan.tool.manager import ToolManager
from avalan.tool.skills import SkillsToolSet


class _Engine:
    model_id = "model"
    model_type = "text"


class _Memory:
    has_permanent_message = False
    has_recent_message = False


class _Agent(EngineAgent):
    def _prepare_call(self, context: ModelCallContext) -> dict[str, str]:
        return {"developer_prompt": "operator guidance"}


class _ToolManagerSubclass(ToolManager):
    pass


class _ToolBootstrapWrapper:
    def __init__(self, bootstrap: str) -> None:
        self._bootstrap = bootstrap

    def bootstrap_prompt(self) -> str:
        return self._bootstrap


class _NoBootstrapTool:
    pass


class _FakeSkillLoop:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def answer(self, manager: ToolManager, task: str) -> str:
        matched = await manager(
            ToolCall(
                id="match",
                name="skills.match",
                arguments={"query": task},
            ),
            ToolCallContext(),
        )
        self.calls.append("skills.match")
        assert isinstance(matched, ToolCallResult)
        matched_result = _result_dict(matched)
        assert "FOLLOW_THE_PDF_STEPS" not in str(matched_result)
        matched_items = cast(
            tuple[dict[str, Any], ...],
            matched_result["items"],
        )
        skill_id = matched_items[0]["metadata"]["skill_id"]
        assert isinstance(skill_id, str)

        read = await manager(
            ToolCall(
                id="read",
                name="skills.read",
                arguments={"skill": skill_id},
            ),
            ToolCallContext(),
        )
        self.calls.append("skills.read")
        assert isinstance(read, ToolCallResult)
        read_result = _result_dict(read)
        read_content = cast(dict[str, Any], read_result["content"])
        body = read_content["text"]
        assert isinstance(body, str)
        if "FOLLOW_THE_PDF_STEPS" not in body:
            return "missing skill body"
        return "answered after read"


class SkillsFakeE2ETestCase(IsolatedAsyncioTestCase):
    async def test_fake_tool_loop_requires_read_before_skill_body(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "pdf" / "SKILL.md",
                body="# PDF Body\nFOLLOW_THE_PDF_STEPS\n",
            )
            registry = await _registry(root)
            manager = ToolManager.create_instance(
                available_toolsets=[SkillsToolSet(registry)],
                enable_tools=["skills"],
            )
            loop = _FakeSkillLoop()

            answer = await loop.answer(manager, "render a pdf")

        self.assertEqual(loop.calls, ["skills.match", "skills.read"])
        self.assertEqual(answer, "answered after read")

    async def test_agent_prompt_gets_body_free_skills_bootstrap(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(root / "pdf" / "SKILL.md", body="SECRET_BODY\n")
            registry = await _registry(root)
            manager = ToolManager.create_instance(
                available_toolsets=[SkillsToolSet(registry)],
                enable_tools=["skills"],
            )
            model_manager = AsyncMock(spec=ModelManager)
            model_manager.return_value = "out"
            agent = _agent(manager, model_manager)

            await agent(
                ModelCallContext(
                    specification=Specification(),
                    input="hello",
                )
            )

        prompt = agent.last_prompt
        assert prompt is not None
        developer_prompt = prompt[3]
        assert developer_prompt is not None
        self.assertIn("operator guidance", developer_prompt)
        self.assertIn("skills.list", developer_prompt)
        self.assertIn("skills.match", developer_prompt)
        self.assertIn("skills.read", developer_prompt)
        self.assertIn("skills.check", developer_prompt)
        self.assertNotIn("SECRET_BODY", developer_prompt)

    async def test_agent_prompt_uses_tool_manager_subclasses(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(root / "pdf" / "SKILL.md", body="SECRET_BODY\n")
            registry = await _registry(root)
            manager = _ToolManagerSubclass.create_instance(
                available_toolsets=[SkillsToolSet(registry)],
                enable_tools=["skills.read"],
            )
            model_manager = AsyncMock(spec=ModelManager)
            model_manager.return_value = "out"
            agent = _agent(manager, model_manager)

            await agent(
                ModelCallContext(
                    specification=Specification(),
                    input="hello",
                )
            )

        prompt = agent.last_prompt
        assert prompt is not None
        developer_prompt = prompt[3]
        assert developer_prompt is not None
        self.assertIn("skills.read", developer_prompt)
        self.assertNotIn("skills.list", developer_prompt)
        self.assertNotIn("SECRET_BODY", developer_prompt)

    async def test_agent_prompt_uses_tool_manager_wrappers(self) -> None:
        model_manager = AsyncMock(spec=ModelManager)
        model_manager.return_value = "out"
        manager = cast(
            ToolManager,
            _ToolBootstrapWrapper("wrapped skills bootstrap"),
        )
        agent = _agent(manager, model_manager)

        self.assertEqual(
            agent._developer_prompt_with_tool_bootstrap(None),
            "wrapped skills bootstrap",
        )

        await agent(
            ModelCallContext(
                specification=Specification(),
                input="hello",
            )
        )

        prompt = agent.last_prompt
        assert prompt is not None
        developer_prompt = prompt[3]
        assert developer_prompt is not None
        self.assertIn("operator guidance", developer_prompt)
        self.assertIn("wrapped skills bootstrap", developer_prompt)

    async def test_agent_prompt_ignores_tools_without_bootstrap_hook(
        self,
    ) -> None:
        model_manager = AsyncMock(spec=ModelManager)
        model_manager.return_value = "out"
        manager = cast(ToolManager, _NoBootstrapTool())
        agent = _agent(manager, model_manager)

        await agent(
            ModelCallContext(
                specification=Specification(),
                input="hello",
            )
        )

        prompt = agent.last_prompt
        assert prompt is not None
        self.assertEqual(prompt[3], "operator guidance")

    async def test_agent_prompt_omits_bootstrap_without_enabled_skills(
        self,
    ) -> None:
        manager = ToolManager.create_instance(enable_tools=[])
        model_manager = AsyncMock(spec=ModelManager)
        model_manager.return_value = "out"
        agent = _agent(manager, model_manager)

        await agent(
            ModelCallContext(
                specification=Specification(),
                input="hello",
            )
        )

        prompt = agent.last_prompt
        assert prompt is not None
        self.assertEqual(prompt[3], "operator guidance")


def _agent(
    manager: ToolManager,
    model_manager: AsyncMock,
) -> _Agent:
    event_manager = MagicMock()
    event_manager.trigger = AsyncMock()
    return _Agent(
        cast(Engine, _Engine()),
        cast(MemoryManager, _Memory()),
        manager,
        event_manager,
        model_manager,
        EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="model",
            params={},
        ),
    )


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


def _result_dict(outcome: ToolCallResult) -> dict[str, Any]:
    assert isinstance(outcome.result, dict)
    return cast(dict[str, Any], outcome.result)


def _write_skill(path: Path, *, body: str = "# PDF Body\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        "name: pdf\n"
        "description: PDF rendering guidance.\n"
        'tags: ["pdf"]\n'
        "resources: []\n"
        "---\n"
        f"{body}",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
