from unittest import TestCase, main

from avalan.entities import ToolManagerSettings
from avalan.model import ModelCapabilityCatalog
from avalan.tool import ToolSet
from avalan.tool.manager import ToolManager
from avalan.tool.math import CalculatorTool


class GetToolCallsTestCase(TestCase):
    @staticmethod
    def _catalog(manager: ToolManager) -> ModelCapabilityCatalog:
        return ModelCapabilityCatalog.create(
            manager.export_model_capability_seed()
        )

    def test_no_tool_call(self):
        calculator = CalculatorTool()
        manager = ToolManager.create_instance(
            enable_tools=["calculator"],
            available_toolsets=[ToolSet(tools=[calculator])],
            settings=ToolManagerSettings(),
        )
        self.assertIsNone(self._catalog(manager).get_calls("hello"))

    def test_partial_tool_call(self):
        calculator = CalculatorTool()
        manager = ToolManager.create_instance(
            enable_tools=["calculator"],
            available_toolsets=[ToolSet(tools=[calculator])],
            settings=ToolManagerSettings(),
        )
        self.assertIsNone(self._catalog(manager).get_calls("<tool_call>{"))

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
        result = self._catalog(manager).get_calls(text)
        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    main()
