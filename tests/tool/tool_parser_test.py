from avalan.model.entities import ToolCall, ToolFormat
from avalan.tool.parser import ToolCallParser
from unittest import TestCase, main


class ToolCallParserFormatTestCase(TestCase):
    def test_json(self):
        parser = ToolCallParser(tool_format=ToolFormat.JSON)
        text = '{"tool": "calculator", "arguments": {"expression": "1 + 1"}}'
        self.assertEqual(
            parser(text),
            ("calculator", {"expression": "1 + 1"})
        )

    def test_react(self):
        parser = ToolCallParser(tool_format=ToolFormat.REACT)
        text = 'Action: calculator\nAction Input: {"expression": "2"}'
        self.assertEqual(
            parser(text),
            ("calculator", {"expression": "2"})
        )

    def test_bracket(self):
        parser = ToolCallParser(tool_format=ToolFormat.BRACKET)
        text = '[calculator](2)'
        self.assertEqual(
            parser(text),
            ("calculator", {"input": "2"})
        )

    def test_openai_json(self):
        parser = ToolCallParser(tool_format=ToolFormat.OPENAI)
        text = '{"name": "calculator", "arguments": {"expression": "3"}}'
        self.assertEqual(
            parser(text),
            ("calculator", {"expression": "3"})
        )

    def test_json_invalid(self):
        parser = ToolCallParser(tool_format=ToolFormat.JSON)
        text = '{"tool": "calculator", "arguments": {"expression": 1}'
        self.assertIsNone(parser(text))

    def test_react_invalid_json(self):
        parser = ToolCallParser(tool_format=ToolFormat.REACT)
        text = 'Action: calculator\nAction Input: {"expression": 2'
        self.assertIsNone(parser(text))

    def test_openai_json_invalid(self):
        parser = ToolCallParser(tool_format=ToolFormat.OPENAI)
        text = '{"name": "calculator", "arguments": {"expression": "3"'
        self.assertIsNone(parser(text))

    def test_json_additional_fields(self):
        parser = ToolCallParser(tool_format=ToolFormat.JSON)
        text = '{"tool": "calculator", "arguments": {"expression": "1"}, "id": 1}'
        self.assertEqual(
            parser(text),
            ("calculator", {"expression": "1"})
        )

    def test_react_additional_fields(self):
        parser = ToolCallParser(tool_format=ToolFormat.REACT)
        text = 'Action: calculator\nAction Input: {"expression": "2", "unit": "m"}'
        self.assertEqual(
            parser(text),
            ("calculator", {"expression": "2", "unit": "m"})
        )

    def test_bracket_with_spaces(self):
        parser = ToolCallParser(tool_format=ToolFormat.BRACKET)
        text = '[calculator]( 2 )'
        self.assertEqual(
            parser(text),
            ("calculator", {"input": " 2 "})
        )

    def test_openai_json_additional_fields(self):
        parser = ToolCallParser(tool_format=ToolFormat.OPENAI)
        text = '{"name": "calculator", "arguments": {"expression": "3"}, "extra": null}'
        self.assertEqual(
            parser(text),
            ("calculator", {"expression": "3"})
        )


class ToolCallParserTagTestCase(TestCase):
    def setUp(self):
        self.parser = ToolCallParser()

    def test_single(self):
        text = (
            '<tool_call>{"name": "calculator", '
            '"arguments": {"expression": "1 + 1"}}</tool_call>'
        )
        expected = [
            ToolCall(name="calculator", arguments={"expression": "1 + 1"})
        ]
        self.assertEqual(self.parser(text), expected)

    def test_single_quotes(self):
        text = (
            "<tool_call>{'name': 'calculator', 'arguments': {'expression': '2'}}"
            "</tool_call>"
        )
        expected = [
            ToolCall(name="calculator", arguments={"expression": "2"})
        ]
        self.assertEqual(self.parser(text), expected)

    def test_multiple(self):
        text = (
            '<tool_call>{"name": "calculator", "arguments": {"expression": "1"}}</tool_call>'
            '<tool_call>{"name": "calculator", "arguments": {"expression": "2"}}</tool_call>'
        )
        expected = [
            ToolCall(name="calculator", arguments={"expression": "1"}),
            ToolCall(name="calculator", arguments={"expression": "2"})
        ]
        self.assertEqual(self.parser(text), expected)

    def test_with_name_attr(self):
        text = (
            '<tool_call name="calculator">{"expression": "2"}</tool_call>'
        )
        expected = [
            ToolCall(name="calculator", arguments={"expression": "2"})
        ]
        self.assertEqual(self.parser(text), expected)

    def test_self_closing(self):
        text = (
            '<tool_call name="calculator" arguments=\'{"expression": "2"}\'/>'
        )
        expected = [
            ToolCall(name="calculator", arguments={"expression": "2"})
        ]
        self.assertEqual(self.parser(text), expected)

    def test_tool_tag(self):
        text = (
            '<tool name="calculator">{"expression": "2"}</tool>'
        )
        expected = [
            ToolCall(name="calculator", arguments={"expression": "2"})
        ]
        self.assertEqual(self.parser(text), expected)

    def test_invalid_json(self):
        text = (
            '<tool_call>{"name": "calculator", "arguments": {"expression": "2"}</tool_call>'
        )
        self.assertIsNone(self.parser(text))

    def test_eos_token(self):
        parser = ToolCallParser(eos_token="<END>")
        text = (
            '<tool_call>{"name": "calculator", "arguments": {"expression": "2"}}</tool_call><END>'
        )
        expected = [
            ToolCall(name="calculator", arguments={"expression": "2"})
        ]
        self.assertEqual(parser(text), expected)

    def test_no_tool_call(self):
        self.assertIsNone(self.parser("hello"))


if __name__ == "__main__":
    main()
