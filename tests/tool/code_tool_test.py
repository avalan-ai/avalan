from avalan.entities import ToolCallContext
from avalan.tool.code import CodeTool
from unittest import IsolatedAsyncioTestCase, main


class CodeToolTestCase(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tool = CodeTool()

    async def test_exec(self):
        result = await self.tool(
            "def run(): return 1 + 1", context=ToolCallContext()
        )
        self.assertEqual(result, "2")

    async def test_exec_with_args(self):
        result = await self.tool(
            """
def run(name: str, stars: int):
    return f"Hello {name} with {stars} stars!"
        """,
            "Leo Messi",
            context=ToolCallContext(),
            stars=3,
        )
        self.assertEqual(result, "Hello Leo Messi with 3 stars!")

    async def test_restricted_builtin(self):
        with self.assertRaises(NameError):
            await self.tool("open('f', 'w')", context=ToolCallContext())


if __name__ == "__main__":
    main()
