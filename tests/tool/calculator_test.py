from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import AsyncMock

from pytest import raises
from sympy.core.sympify import SympifyError

from avalan.entities import ToolCallContext
from avalan.tool.math import CalculatorTool, MathToolSet


class CalculatorToolTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        self.calc = CalculatorTool()

    async def test_addition(self):
        self.assertEqual(
            await self.calc("1 + 1", context=ToolCallContext()), "2"
        )

    async def test_parentheses(self):
        self.assertEqual(
            await self.calc("2*(3+4)", context=ToolCallContext()), "14"
        )

    async def test_production_calculator_does_not_stream_or_delay(self):
        stream_event = AsyncMock()
        context = ToolCallContext(stream_event=stream_event)

        self.assertEqual(
            await self.calc("(4 + 6) * 5 / 2", context=context),
            "25",
        )
        stream_event.assert_not_awaited()

    async def test_invalid(self):
        with raises(SympifyError) as exc:
            await self.calc("2**", context=ToolCallContext())

        assert (
            "Sympify of expression 'could not parse '2**'' failed, "
            + "because of exception being raised"
            in str(exc.value)
        )


class MathToolSetTestCase(IsolatedAsyncioTestCase):
    async def test_defaults_to_calculator_tool(self):
        toolset = MathToolSet(namespace="math")

        self.assertEqual(toolset.namespace, "math")
        self.assertEqual(len(toolset.tools), 1)
        self.assertIsInstance(toolset.tools[0], CalculatorTool)


if __name__ == "__main__":
    main()
