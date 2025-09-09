from unittest import TestCase

from avalan.entities import ToolCall, ToolCallToken
from avalan.model.vendor import TextGenerationVendor


class VendorBuildToolCallTokenTestCase(TestCase):
    def test_build_tool_call_token_from_string_json(self) -> None:
        token = TextGenerationVendor.build_tool_call_token(
            call_id="1",
            tool_name="pkg__tool",
            arguments='{"a": 1}',
        )
        expected = ToolCallToken(
            token=(
                '<tool_call>{"name": "pkg.tool", "arguments": {"a":'
                " 1}}</tool_call>"
            ),
            call=ToolCall(id="1", name="pkg.tool", arguments={"a": 1}),
        )
        self.assertEqual(token, expected)

    def test_build_tool_call_token_handles_invalid_json_str(self) -> None:
        token = TextGenerationVendor.build_tool_call_token(
            call_id=None,
            tool_name="tool",
            arguments='{"a": }',
        )
        expected = ToolCallToken(
            token='<tool_call>{"name": "tool", "arguments": {}}</tool_call>',
            call=ToolCall(id=None, name="tool", arguments={}),
        )
        self.assertEqual(token, expected)

    def test_build_tool_call_token_from_dict(self) -> None:
        token = TextGenerationVendor.build_tool_call_token(
            call_id="2",
            tool_name=None,
            arguments={"b": 2},
        )
        expected = ToolCallToken(
            token='<tool_call>{"name": "", "arguments": {"b": 2}}</tool_call>',
            call=ToolCall(id="2", name="", arguments={"b": 2}),
        )
        self.assertEqual(token, expected)
