from avalan.entities import ToolCall, ToolFormat
from avalan.tool.parser import ToolCallParser
from unittest import TestCase, main
from unittest.mock import patch
from uuid import uuid4 as _uuid4


class ToolCallParserExtraTestCase(TestCase):
    def test_react_invalid_json_exception(self):
        parser = ToolCallParser(tool_format=ToolFormat.REACT)
        text = 'Action: calc\nAction Input: {"value": 1,}'
        self.assertIsNone(parser(text))

    def test_bracket_no_match(self):
        parser = ToolCallParser(tool_format=ToolFormat.BRACKET)
        self.assertIsNone(parser("nothing"))

    def test_regex_fallback(self):
        parser = ToolCallParser()
        call_id = _uuid4()
        text = (
            '<tool_call>{"name": "calculator", "arguments": {"expression":'
            ' "1"}}</tool_call></extra>'
        )
        with patch("avalan.tool.parser.uuid4", return_value=call_id):
            expected = [
                ToolCall(
                    id=call_id,
                    name="calculator",
                    arguments={"expression": "1"},
                )
            ]
            self.assertEqual(parser(text), expected)

    def test_named_tool_call_invalid_json(self):
        parser = ToolCallParser()
        text = '<tool_call name="calc">{"expr": 2,}</tool_call>'
        self.assertIsNone(parser(text))

    def test_tool_tag_invalid_json(self):
        parser = ToolCallParser()
        text = '<tool name="calc">{"expr": 2,}</tool>'
        self.assertIsNone(parser(text))

    def test_self_closing_invalid_json(self):
        parser = ToolCallParser()
        text = '<tool_call name="calc" arguments=\'{"expr": 2,}\'/>'
        self.assertIsNone(parser(text))

    def test_is_potential_tool_call(self):
        parser = ToolCallParser()
        self.assertFalse(parser.is_potential_tool_call("", ""))
        self.assertTrue(parser.is_potential_tool_call("", "a"))

    def test_tool_format_property(self):
        parser = ToolCallParser(tool_format=ToolFormat.REACT)
        self.assertIs(parser.tool_format, ToolFormat.REACT)

    def test_set_eos_token_method(self):
        parser = ToolCallParser()
        parser.set_eos_token("<END>")
        call_id = _uuid4()
        text = (
            '<tool_call>{"name": "calculator", "arguments": {"expression": '
            '"2"}}</tool_call><END>'
        )
        with patch("avalan.tool.parser.uuid4", return_value=call_id):
            expected = [
                ToolCall(
                    id=call_id,
                    name="calculator",
                    arguments={"expression": "2"},
                )
            ]
            self.assertEqual(parser(text), expected)

    def test_message_tool_calls_tuple_payload(self):
        parser = ToolCallParser(tool_format=ToolFormat.JSON)
        text = '{"tool": "calculator", "arguments": {"value": 2}}'
        calls = parser.message_tool_calls(text)
        self.assertEqual(
            calls,
            [
                {
                    "id": None,
                    "name": "calculator",
                    "arguments": {"value": 2},
                    "content_type": "json",
                }
            ],
        )

    def test_message_tool_calls_tuple_invalid_name(self):
        parser = ToolCallParser(tool_format=ToolFormat.JSON)
        text = '{"tool": 123, "arguments": {}}'
        self.assertEqual(parser.message_tool_calls(text), [])

    def test_extract_harmony_content_skips_empty_segments(self):
        parser = ToolCallParser()
        text = (
            "intro\n\n\n<|channel|>analysis<|message|>   <|call|>\n\n\noutro"
        )
        thinking, content = parser.extract_harmony_content(text)
        self.assertIsNone(thinking)
        self.assertEqual(content, "intro\n\nanalysis   \n\noutro")

    def test_resolve_text_source_prefers_serialized_payloads(self):
        parser = ToolCallParser()
        self.assertEqual(
            parser._resolve_text_source(None, "serialized"),
            "serialized",
        )
        self.assertEqual(
            parser._resolve_text_source(
                None,
                {"type": "text", "text": "dict text"},
            ),
            "dict text",
        )
        self.assertEqual(
            parser._resolve_text_source(
                None,
                [{"type": "text", "text": "list text"}],
            ),
            "list text",
        )

    def test_merge_thinking_handles_none_and_combines(self):
        parser = ToolCallParser()
        message_dict: dict[str, object] = {"thinking": ""}
        parser._merge_thinking(message_dict, None)
        self.assertIsNone(message_dict["thinking"])

        message_dict = {"thinking": "existing"}
        parser._merge_thinking(message_dict, "new")
        self.assertEqual(message_dict["thinking"], "existing\n\nnew")


if __name__ == "__main__":
    main()
