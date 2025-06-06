from avalan.tool import Tool, ToolSet
from avalan.tool.math import CalculatorTool
from contextlib import AsyncExitStack
from unittest import IsolatedAsyncioTestCase, TestCase, main
from unittest.mock import AsyncMock, MagicMock


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
            self.assertEqual(
                schemas[0]["function"]["description"], "Return upper-case text."
            )
            self.assertEqual(
                schemas[0]["function"]["parameters"]["type"], "object"
            )
            self.assertEqual(
                schemas[0]["function"]["parameters"]["properties"]["text"][
                    "type"
                ],
                "string",
            )
            self.assertEqual(
                schemas[0]["function"]["parameters"]["properties"]["text"][
                    "description"
                ],
                "Text to upper-case.",
            )
            self.assertEqual(schemas[0]["function"]["return"]["type"], "string")
            self.assertEqual(
                schemas[0]["function"]["return"]["description"],
                "Upper-cased text.",
            )
        self.assertTrue(tool.exited)


class ToolJsonSchemaPrefixTestCase(TestCase):
    def test_json_schema_with_prefix(self):
        tool = CalculatorTool()
        ToolSet(tools=[tool])
        schema = tool.json_schema(prefix="pre.")
        self.assertEqual(schema["function"]["name"], "pre.calculator")


class ToolSetEnterExitTestCase(IsolatedAsyncioTestCase):
    async def test_aenter_and_aexit(self):
        async_tool = MagicMock(spec=["__aenter__", "__aexit__"])
        sync_tool = MagicMock(spec=["__enter__", "__exit__"])
        stack = AsyncMock(spec=AsyncExitStack)
        stack.enter_async_context = AsyncMock()
        stack.enter_context = MagicMock()
        stack.__aexit__ = AsyncMock(return_value=False)

        toolset = ToolSet(exit_stack=stack, tools=[async_tool, sync_tool])
        result = await toolset.__aenter__()
        self.assertIs(result, toolset)
        stack.enter_async_context.assert_awaited_once_with(async_tool)
        stack.enter_context.assert_called_once_with(sync_tool)

        exit_result = await toolset.__aexit__(None, None, None)
        stack.__aexit__.assert_awaited_once_with(None, None, None)
        self.assertFalse(exit_result)


class ToolSetDocstringFallbackTestCase(TestCase):
    def test_docstring_fallback(self):
        class DoclessTool(Tool):
            def __init__(self) -> None:
                super().__init__()
                self.__name__ = "docless"

            async def __call__(self) -> str:
                """Call doc"""
                return "ok"

        tool = DoclessTool()
        self.assertIsNone(tool.__doc__)
        ToolSet(tools=[tool])
        self.assertEqual(tool.__doc__, "Call doc")


class ToolSetJsonSchemasTestCase(TestCase):
    def test_nested_toolsets_and_function(self):
        def greet(name: str) -> str:
            """Greet.

            Args:
                name: Name.
            """
            return f"hi {name}"

        inner = ToolSet(namespace="inner", tools=[CalculatorTool()])
        outer = ToolSet(namespace="outer", tools=[CalculatorTool(), inner])
        mix = ToolSet(namespace="mix", tools=[greet])

        schemas_outer = outer.json_schemas()
        self.assertEqual(
            [s["function"]["name"] for s in schemas_outer],
            ["outer.calculator", "outer..calculator"],
        )

        schemas_mix = mix.json_schemas()
        self.assertEqual(
            [s["function"]["name"] for s in schemas_mix],
            ["mix.greet"],
        )


if __name__ == "__main__":
    main()
