from avalan.entities import ToolCallContext
from avalan.tool.code import CodeTool
from pytest import importorskip
from unittest import IsolatedAsyncioTestCase, main

importorskip("RestrictedPython", reason="RestrictedPython not installed")


class CodeToolTestCase(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tool = CodeTool()

    async def test_exec(self):
        result = await self.tool("result = 1 + 1", context=ToolCallContext())
        self.assertEqual(result, "2")

    async def test_restricted_builtin(self):
        with self.assertRaises(NameError):
            await self.tool("open('f', 'w')", context=ToolCallContext())


if __name__ == "__main__":
    main()
