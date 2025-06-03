from avalan.entities import ToolCall, ToolCallResult
from avalan.tool import ToolSet
from avalan.tool.calculator import calculator
from avalan.tool.manager import ToolManager
from unittest import IsolatedAsyncioTestCase, main, TestCase
from unittest.mock import patch
from uuid import uuid4 as _uuid4


class ToolManagerCreationTestCase(TestCase):
    def test_default_instance_empty(self):
        manager = ToolManager.create_instance()
        self.assertTrue(manager.is_empty)
        self.assertIsNone(manager.tools)

    def test_instance_with_enabled_tool(self):
        manager = ToolManager.create_instance(
            enable_tools=["calculator"],
            available_toolsets=[ToolSet(tools=[calculator])],
        )
        self.assertFalse(manager.is_empty)
        self.assertEqual(manager.tools, [calculator])


class ToolManagerCallTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        self.manager = ToolManager.create_instance(
            enable_tools=["calculator"],
            available_toolsets=[ToolSet(tools=[calculator])],
        )

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

            results = await self.manager(calls[0])

            expected_result = ToolCallResult(
                id=result_id,
                call=expected_call,
                name="calculator",
                arguments={"expression": "1 + 1"},
                result="2",
            )
            self.assertEqual(results, expected_result)

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

            results = await self.manager(calls[0])

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
        namespaced_manager = ToolManager.create_instance(
            enable_tools=["math.calculator"],
            available_toolsets=[ToolSet(namespace="math", tools=[calculator])],
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

            results = await namespaced_manager(calls[0])

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
