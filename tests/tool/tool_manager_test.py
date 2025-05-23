from avalan.model.entities import ToolCall, ToolCallResult
from avalan.tool.calculator import calculator
from avalan.tool.manager import ToolManager
from unittest import TestCase, main

class ToolManagerCreationTestCase(TestCase):
    def test_default_instance_empty(self):
        manager = ToolManager.create_instance()
        self.assertTrue(manager.is_empty)
        self.assertIsNone(manager.tools)

    def test_instance_with_enabled_tool(self):
        manager = ToolManager.create_instance(enable_tools=["calculator"])
        self.assertFalse(manager.is_empty)
        self.assertEqual(manager.tools, [calculator])

class ToolManagerCallTestCase(TestCase):
    def setUp(self):
        self.manager = ToolManager.create_instance(enable_tools=["calculator"])

    def test_call_no_tool_call(self):
        calls, results = self.manager("no tools here")
        self.assertIsNone(calls)
        self.assertIsNone(results)

    def test_call_with_tool(self):
        text = (
            '<tool_call>{"name": "calculator", '
            '"arguments": {"expression": "1 + 1"}}</tool_call>'
        )
        calls, results = self.manager(text)
        expected_call = ToolCall(
            name="calculator",
            arguments={"expression": "1 + 1"},
        )
        expected_result = ToolCallResult(
            name="calculator",
            arguments={"expression": "1 + 1"},
            result="2",
        )
        self.assertEqual(calls, [expected_call])
        self.assertEqual(results, [expected_result])

    def test_set_eos_token(self):
        self.manager.set_eos_token("<END>")
        text = (
            '<tool_call>{"name": "calculator", '
            '"arguments": {"expression": "2"}}</tool_call><END>'
        )
        calls, results = self.manager(text)
        expected_call = ToolCall(
            name="calculator",
            arguments={"expression": "2"},
        )
        expected_result = ToolCallResult(
            name="calculator",
            arguments={"expression": "2"},
            result="2",
        )
        self.assertEqual(calls, [expected_call])
        self.assertEqual(results, [expected_result])
        self.assertEqual(self.manager._parser._eos_token, "<END>")

if __name__ == "__main__":
    main()
