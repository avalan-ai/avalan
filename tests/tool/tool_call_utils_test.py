from avalan.tool import ToolSet
from avalan.tool.math import CalculatorTool
from avalan.tool.manager import ToolManager
from avalan.entities import ToolManagerSettings
from unittest import TestCase, main


class GetToolCallsTestCase(TestCase):
    def test_no_tool_call(self):
        calculator = CalculatorTool()
        manager = ToolManager.create_instance(
            enable_tools=["calculator"],
            available_toolsets=[ToolSet(tools=[calculator])],
            settings=ToolManagerSettings(),
        )
        self.assertIsNone(manager.get_calls("hello"))

    def test_partial_tool_call(self):
        calculator = CalculatorTool()
        manager = ToolManager.create_instance(
            enable_tools=["calculator"],
            available_toolsets=[ToolSet(tools=[calculator])],
            settings=ToolManagerSettings(),
        )
        self.assertIsNone(manager.get_calls("<tool_call>{"))

    def test_full_tool_call(self):
        calculator = CalculatorTool()
        text = (
            '<tool_call>{"name": "calculator", "arguments": {}} </tool_call>'
        )
        manager = ToolManager.create_instance(
            enable_tools=["calculator"],
            available_toolsets=[ToolSet(tools=[calculator])],
            settings=ToolManagerSettings(),
        )
        result = manager.get_calls(text)
        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    main()
