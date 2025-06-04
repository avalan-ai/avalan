from avalan.tool import Tool, ToolSet
from avalan.tool.calculator import CalculatorTool
from unittest import IsolatedAsyncioTestCase, TestCase, main


class ToolSetTestCase(TestCase):
    def test_dataclass_properties(self):
        calculator = CalculatorTool()
        toolset = ToolSet(tools=[calculator], namespace="math")
        self.assertEqual(toolset.namespace, "math")
        self.assertEqual(list(toolset.tools), [calculator])
        with self.assertRaises(AttributeError):
            toolset.namespace = "other"


class DummyTool(Tool):
    """
    Return upper-case text.

    Args:
        text: Text to upper-case.

    Returns:
        Upper-cased text.
    """

    def __init__(self) -> None:
        self.__name__ = "dummy"

    async def __call__(self, text: str) -> str:
        return text.upper()


class ContextTool(DummyTool):
    """
    Return upper-case text.

    Args:
        text: Text to upper-case.

    Returns:
        Upper-cased text.
    """

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
    async def test_async_context_and_schema(self):
        tool = ContextTool()
        toolset = ToolSet(tools=[tool])
        async with toolset:
            self.assertTrue(tool.entered)
            schemas = toolset.json_schemas()
            self.assertEqual(len(schemas), 1)
            self.assertEqual(schemas[0]["type"], "function")
            self.assertEqual(schemas[0]["function"]["name"], "dummy")
            self.assertEqual(schemas[0]["function"]["description"], "Return upper-case text.")
            self.assertEqual(schemas[0]["function"]["parameters"]["type"], "object")
            self.assertEqual(schemas[0]["function"]["parameters"]["properties"]["text"]["type"], "string")
            self.assertEqual(schemas[0]["function"]["parameters"]["properties"]["text"]["description"], "Text to upper-case.")
            self.assertEqual(schemas[0]["function"]["return"]["type"], "string")
            self.assertEqual(schemas[0]["function"]["return"]["description"], "Upper-cased text.")
        self.assertTrue(tool.exited)


if __name__ == "__main__":
    main()
