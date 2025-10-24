import asyncio
from unittest import IsolatedAsyncioTestCase, TestCase, main, skipIf
from unittest.mock import AsyncMock, patch

from avalan.entities import ToolCallContext
from avalan.tool.code import HAS_CODE_DEPENDENCIES, AstGrepTool, CodeToolSet


class AstGrepToolTestCase(IsolatedAsyncioTestCase):
    async def test_search(self):
        tool = AstGrepTool()
        process = AsyncMock()
        process.communicate = AsyncMock(return_value=(b"out", b""))
        process.returncode = 0
        with patch(
            "avalan.tool.code.create_subprocess_exec",
            AsyncMock(return_value=process),
        ) as create:
            result = await tool("x", context=ToolCallContext(), lang="py")
        create.assert_awaited_once_with(
            "ast-grep",
            "--pattern",
            "x",
            "--lang",
            "py",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self.assertEqual(result, "out")

    async def test_search_and_rewrite_with_paths(self):
        tool = AstGrepTool()
        process = AsyncMock()
        process.communicate = AsyncMock(return_value=(b"", b""))
        process.returncode = 0
        with patch(
            "avalan.tool.code.create_subprocess_exec",
            AsyncMock(return_value=process),
        ) as create:
            await tool(
                "p",
                context=ToolCallContext(),
                lang="ts",
                rewrite="r",
                paths=["a", "b"],
            )
        create.assert_awaited_once_with(
            "ast-grep",
            "--pattern",
            "p",
            "--lang",
            "ts",
            "--rewrite",
            "r",
            "a",
            "b",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def test_error(self):
        tool = AstGrepTool()
        process = AsyncMock()
        process.communicate = AsyncMock(return_value=(b"", b"err"))
        process.returncode = 1
        with patch(
            "avalan.tool.code.create_subprocess_exec",
            AsyncMock(return_value=process),
        ):
            with self.assertRaises(RuntimeError):
                await tool("p", context=ToolCallContext(), lang="py")


@skipIf(not HAS_CODE_DEPENDENCIES, "RestrictedPython not installed")
class CodeToolSetTestCase(TestCase):
    def test_json_schema_includes_ast_grep(self):
        toolset = CodeToolSet(namespace="code")
        schemas = toolset.json_schemas()
        names = [s["function"]["name"] for s in schemas]
        self.assertIn("code.search.ast.grep", names)


if __name__ == "__main__":
    main()
