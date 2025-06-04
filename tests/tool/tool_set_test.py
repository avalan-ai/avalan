from avalan.tool import ToolSet
from avalan.tool.calculator import CalculatorTool
from dataclasses import FrozenInstanceError, is_dataclass
from unittest import IsolatedAsyncioTestCase, TestCase, main


class ToolSetTestCase(TestCase):
    def test_dataclass_properties(self):
        self.assertTrue(is_dataclass(ToolSet))
        calculator = CalculatorTool()
        toolset = ToolSet(tools=[calculator], namespace="math")
        self.assertEqual(toolset.namespace, "math")
        self.assertEqual(list(toolset.tools), [calculator])
        with self.assertRaises(FrozenInstanceError):
            toolset.namespace = "other"


class DummyTool:
    def __init__(self) -> None:
        self.__name__ = "dummy"

    def __call__(self, text: str) -> str:
        """Return upper-case text."""
        return text.upper()


class ContextTool(DummyTool):
    def __init__(self) -> None:
        super().__init__()
        self.entered = False
        self.exited = False

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.exited = True


class ToolSetCallableTestCase(IsolatedAsyncioTestCase):
    async def test_async_context_and_doc(self):
        tool = ContextTool()
        toolset = ToolSet(tools=[tool])
        self.assertEqual(tool.__doc__, tool.__call__.__doc__)
        async with toolset:
            self.assertTrue(tool.entered)
        self.assertTrue(tool.exited)


if __name__ == "__main__":
    main()
