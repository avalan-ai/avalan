from avalan.tool.manager import ToolManager
from unittest import TestCase, main


class HasToolCallTestCase(TestCase):
    def test_no_tool_call(self):
        self.assertFalse(ToolManager.has_tool_call("hello"))

    def test_partial_tool_call(self):
        self.assertFalse(ToolManager.has_tool_call("<tool_call>{"))

    def test_full_tool_call(self):
        text = '<tool_call>{"name": "calculator", "arguments": {}} </tool_call>'
        self.assertTrue(ToolManager.has_tool_call(text))


if __name__ == "__main__":
    main()
