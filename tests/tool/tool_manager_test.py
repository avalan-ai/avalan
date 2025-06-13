from avalan.entities import (
    ToolCall,
    ToolCallContext,
    ToolCallResult,
    ToolManagerSettings,
)
from avalan.tool import ToolSet
from avalan.tool.math import CalculatorTool
from avalan.tool.manager import ToolManager
from unittest import IsolatedAsyncioTestCase, TestCase, main
from unittest.mock import AsyncMock, patch
from uuid import uuid4 as _uuid4


class ToolManagerCreationTestCase(TestCase):
    def test_default_instance_empty(self):
        manager = ToolManager.create_instance(
            enable_tools=[], settings=ToolManagerSettings()
        )
        self.assertTrue(manager.is_empty)
        self.assertIsNone(manager.tools)

    def test_instance_with_enabled_tool(self):
        calculator = CalculatorTool()
        manager = ToolManager.create_instance(
            enable_tools=[calculator.__name__],
            available_toolsets=[ToolSet(tools=[calculator])],
            settings=ToolManagerSettings(),
        )
        self.assertFalse(manager.is_empty)
        self.assertEqual(manager.tools, [calculator])


class DummyAdder:
    def __init__(self) -> None:
        self.__name__ = "adder"

    async def __call__(self, a: int, b: int) -> int:
        """Return the sum of ``a`` and ``b``."""
        return a + b


class ToolManagerCallTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        calculator = CalculatorTool()
        self.manager = ToolManager.create_instance(
            enable_tools=["calculator"],
            available_toolsets=[ToolSet(tools=[calculator])],
            settings=ToolManagerSettings(),
        )

    async def test_callable_class_tool(self):
        adder = DummyAdder()
        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[adder])],
            settings=ToolManagerSettings(),
        )
        call = ToolCall(id=_uuid4(), name="adder", arguments={"a": 1, "b": 2})
        result = await manager(call, context=ToolCallContext())
        self.assertEqual(result.result, 3)

    async def test_call_no_tool_call(self):
        calls = self.manager.get_calls("no tools here")
        self.assertIsNone(calls)

    async def test_call_with_tool(self):
        text = (
            '<tool_call>{"name": "calculator", '
            '"arguments": {"expression": "1 + 1"}}</tool_call>'
        )
        call_id = _uuid4()
        result_id = _uuid4()
        with (
            patch("avalan.tool.parser.uuid4", return_value=call_id),
            patch("avalan.tool.manager.uuid4", return_value=result_id),
        ):
            calls = self.manager.get_calls(text)
            expected_call = ToolCall(
                id=call_id,
                name="calculator",
                arguments={"expression": "1 + 1"},
            )
            self.assertEqual(calls, [expected_call])

            results = await self.manager(calls[0], context=ToolCallContext())

            expected_result = ToolCallResult(
                id=result_id,
                call=expected_call,
                name="calculator",
                arguments={"expression": "1 + 1"},
                result="2",
            )
            self.assertEqual(results, expected_result)

    async def test_avoid_repetition(self):
        adder = DummyAdder()
        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[adder])],
            settings=ToolManagerSettings(avoid_repetition=True),
        )
        call = ToolCall(id=_uuid4(), name="adder", arguments={"a": 1, "b": 2})
        result1 = await manager(call, context=ToolCallContext())
        self.assertIsNotNone(result1)
        result2 = await manager(call, context=ToolCallContext(calls=[call]))
        self.assertIsNone(result2)

    async def test_maximum_depth(self):
        adder = DummyAdder()
        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[adder])],
            settings=ToolManagerSettings(maximum_depth=1),
        )
        call = ToolCall(id=_uuid4(), name="adder", arguments={"a": 1, "b": 2})
        result1 = await manager(call, context=ToolCallContext())
        self.assertIsNotNone(result1)
        result2 = await manager(call, context=ToolCallContext(calls=[call]))
        self.assertIsNone(result2)

    async def test_async_context(self):
        calculator = CalculatorTool()
        toolset = ToolSet(tools=[calculator])
        manager = ToolManager.create_instance(
            enable_tools=["calculator"],
            available_toolsets=[toolset],
            settings=ToolManagerSettings(),
        )
        manager._stack.enter_async_context = AsyncMock()
        manager._stack.__aexit__ = AsyncMock(return_value=False)

        async with manager:
            manager._stack.enter_async_context.assert_awaited_once()

        manager._stack.__aexit__.assert_awaited_once()

    async def test_set_eos_token(self):
        self.manager.set_eos_token("<END>")
        text = (
            '<tool_call>{"name": "calculator", '
            '"arguments": {"expression": "2"}}</tool_call><END>'
        )

        call_id = _uuid4()
        result_id = _uuid4()
        with (
            patch("avalan.tool.parser.uuid4", return_value=call_id),
            patch("avalan.tool.manager.uuid4", return_value=result_id),
        ):
            calls = self.manager.get_calls(text)
            expected_call = ToolCall(
                id=call_id,
                name="calculator",
                arguments={"expression": "2"},
            )
            self.assertEqual(calls, [expected_call])

            results = await self.manager(calls[0], context=ToolCallContext())

            expected_result = ToolCallResult(
                id=result_id,
                call=expected_call,
                name="calculator",
                arguments={"expression": "2"},
                result="2",
            )
            self.assertEqual(results, expected_result)
            self.assertEqual(self.manager._parser._eos_token, "<END>")

    async def test_namespaced_tool(self):
        calculator = CalculatorTool()
        namespaced_manager = ToolManager.create_instance(
            enable_tools=["math.calculator"],
            available_toolsets=[ToolSet(namespace="math", tools=[calculator])],
            settings=ToolManagerSettings(),
        )
        text = (
            '<tool_call>{"name": "math.calculator", '
            '"arguments": {"expression": "3"}}</tool_call>'
        )

        call_id = _uuid4()
        result_id = _uuid4()
        with (
            patch("avalan.tool.parser.uuid4", return_value=call_id),
            patch("avalan.tool.manager.uuid4", return_value=result_id),
        ):
            calls = namespaced_manager.get_calls(text)
            expected_call = ToolCall(
                id=call_id,
                name="math.calculator",
                arguments={"expression": "3"},
            )
            self.assertEqual(calls, [expected_call])

            results = await namespaced_manager(
                calls[0], context=ToolCallContext()
            )

            expected_result = ToolCallResult(
                id=result_id,
                call=expected_call,
                name="math.calculator",
                arguments={"expression": "3"},
                result="3",
            )
            self.assertEqual(results, expected_result)


if __name__ == "__main__":
    main()
