from contextlib import AsyncExitStack
from typing import Literal
from unittest import IsolatedAsyncioTestCase, TestCase, main
from unittest.mock import AsyncMock, MagicMock

from avalan.tool import Tool, ToolSet
from avalan.tool.json_schema import get_json_schema
from avalan.tool.math import CalculatorTool
from avalan.tool.names import matches_tool_namespace


class ToolSetTestCase(TestCase):
    def test_dataclass_properties(self):
        calculator = CalculatorTool()
        toolset = ToolSet(tools=[calculator], namespace="math")
        self.assertEqual(toolset.namespace, "math")
        self.assertEqual(list(toolset.tools), [calculator])
        with self.assertRaises(AttributeError):
            toolset.namespace = "other"


class ToolNameMatchTestCase(TestCase):
    def test_namespace_matches_exact_name_and_child_segments(self):
        self.assertTrue(matches_tool_namespace("math", "math"))
        self.assertTrue(matches_tool_namespace("math.calculator", "math"))
        self.assertTrue(matches_tool_namespace("math.calculator", "math.*"))
        self.assertTrue(matches_tool_namespace("calculator", None))

    def test_namespace_does_not_match_unsafe_prefix(self):
        self.assertFalse(matches_tool_namespace("mathx", "math"))
        self.assertFalse(matches_tool_namespace("mathx.calculator", "math"))
        self.assertFalse(matches_tool_namespace("mathx.calculator", "math.*"))


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
                schemas[0]["function"]["description"],
                "Return upper-case text.",
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
            self.assertEqual(
                schemas[0]["function"]["return"]["type"], "string"
            )
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
            ["outer.calculator", "outer.inner.calculator"],
        )

        schemas_mix = mix.json_schemas()
        self.assertEqual(
            [s["function"]["name"] for s in schemas_mix],
            ["mix.greet"],
        )

    def test_nested_enabled_tools_filters_by_canonical_name(self):
        inner = ToolSet(namespace="inner", tools=[CalculatorTool()])
        outer = ToolSet(namespace="outer", tools=[CalculatorTool(), inner])

        outer.with_enabled_tools(["outer.inner.calculator"])

        schemas = outer.json_schemas()
        self.assertEqual(
            [schema["function"]["name"] for schema in schemas],
            ["outer.inner.calculator"],
        )

    def test_enabled_namespace_uses_segment_boundaries(self):
        math_set = ToolSet(namespace="math", tools=[CalculatorTool()])
        mathx_set = ToolSet(namespace="mathx", tools=[CalculatorTool()])

        math_set.with_enabled_tools(["math"])
        mathx_set.with_enabled_tools(["math"])

        math_schemas = math_set.json_schemas()
        mathx_schemas = mathx_set.json_schemas()

        assert math_schemas is not None
        assert mathx_schemas is not None
        self.assertEqual(
            [schema["function"]["name"] for schema in math_schemas],
            ["math.calculator"],
        )
        self.assertEqual(mathx_schemas, [])

    def test_enabled_wildcard_namespace_uses_segment_boundaries(self):
        math_set = ToolSet(namespace="math", tools=[CalculatorTool()])
        mathx_set = ToolSet(namespace="mathx", tools=[CalculatorTool()])

        math_set.with_enabled_tools(["math.*"])
        mathx_set.with_enabled_tools(["math.*"])

        math_schemas = math_set.json_schemas()
        mathx_schemas = mathx_set.json_schemas()

        assert math_schemas is not None
        assert mathx_schemas is not None
        self.assertEqual(
            [schema["function"]["name"] for schema in math_schemas],
            ["math.calculator"],
        )
        self.assertEqual(mathx_schemas, [])


class ToolJsonSchemaUtilityTestCase(TestCase):
    def test_get_json_schema_handles_optional_and_missing_doc_items(self):
        def maybe_count(value: str | None, enabled: bool = False) -> int:
            """Summarize value.

            Args:
                value: Optional value.
            """
            return len(value) if value else 0

        schema = get_json_schema(maybe_count)
        self.assertEqual(schema["function"]["name"], "maybe_count")
        self.assertEqual(
            schema["function"]["parameters"]["properties"]["value"]["type"],
            ["string", "null"],
        )
        self.assertEqual(
            schema["function"]["parameters"]["properties"]["enabled"][
                "description"
            ],
            "",
        )
        self.assertEqual(schema["function"]["return"]["description"], "")

    def test_get_json_schema_ignores_varargs(self):
        def echo(name: str, *args: object, **kwargs: object) -> str:
            """Echo.

            Args:
                name: Name.

            Returns:
                Value.
            """
            return name

        schema = get_json_schema(echo)
        self.assertEqual(
            set(schema["function"]["parameters"]["properties"].keys()),
            {"name"},
        )
        self.assertEqual(schema["function"]["return"]["description"], "Value.")

    def test_get_json_schema_maps_literal_to_scalar_enum(self):
        def draw(orientation: Literal["vertical", "horizontal"]) -> str:
            """Draw.

            Args:
                orientation: Axis orientation.
            """
            return orientation

        schema = get_json_schema(draw)
        orientation_schema = schema["function"]["parameters"]["properties"][
            "orientation"
        ]
        self.assertEqual(orientation_schema["type"], "string")
        self.assertEqual(
            orientation_schema["enum"], ["vertical", "horizontal"]
        )


if __name__ == "__main__":
    main()
