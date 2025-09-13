from avalan.entities import ToolCall, ToolFormat
from avalan.tool.parser import ToolCallParser
from unittest import TestCase, main
from unittest.mock import patch
from uuid import uuid4 as _uuid4


class ToolCallParserFormatTestCase(TestCase):
    def test_json(self):
        parser = ToolCallParser(tool_format=ToolFormat.JSON)
        text = '{"tool": "calculator", "arguments": {"expression": "1 + 1"}}'
        self.assertEqual(parser(text), ("calculator", {"expression": "1 + 1"}))

    def test_react(self):
        parser = ToolCallParser(tool_format=ToolFormat.REACT)
        text = 'Action: calculator\nAction Input: {"expression": "2"}'
        self.assertEqual(parser(text), ("calculator", {"expression": "2"}))

    def test_bracket(self):
        parser = ToolCallParser(tool_format=ToolFormat.BRACKET)
        text = "[calculator](2)"
        self.assertEqual(parser(text), ("calculator", {"input": "2"}))

    def test_openai_json(self):
        parser = ToolCallParser(tool_format=ToolFormat.OPENAI)
        text = '{"name": "calculator", "arguments": {"expression": "3"}}'
        self.assertEqual(parser(text), ("calculator", {"expression": "3"}))

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
        text = (
            '{"tool": "calculator", "arguments": {"expression": "1"}, "id": 1}'
        )
        self.assertEqual(parser(text), ("calculator", {"expression": "1"}))

    def test_react_additional_fields(self):
        parser = ToolCallParser(tool_format=ToolFormat.REACT)
        text = (
            'Action: calculator\nAction Input: {"expression": "2", "unit":'
            ' "m"}'
        )
        self.assertEqual(
            parser(text), ("calculator", {"expression": "2", "unit": "m"})
        )

    def test_bracket_with_spaces(self):
        parser = ToolCallParser(tool_format=ToolFormat.BRACKET)
        text = "[calculator]( 2 )"
        self.assertEqual(parser(text), ("calculator", {"input": " 2 "}))

    def test_openai_json_additional_fields(self):
        parser = ToolCallParser(tool_format=ToolFormat.OPENAI)
        text = (
            '{"name": "calculator", "arguments": {"expression": "3"}, "extra":'
            " null}"
        )
        self.assertEqual(parser(text), ("calculator", {"expression": "3"}))


class ToolCallParserHarmonyTestCase(TestCase):
    def test_multiple_with_commentary(self):
        parser = ToolCallParser(tool_format=ToolFormat.HARMONY)
        text = (
            "<|channel|>commentary to=functions.database.tables "
            "<|constrain|>json<|message|>{}<|call|>commentary"
            "<|channel|>commentary to=functions.database.run "
            "<|constrain|>json<|message|>"
            '{"sql":"SELECT COUNT(*) AS product_count FROM products;"}'
            "<|call|>commentary"
        )
        first_id = _uuid4()
        second_id = _uuid4()
        with patch(
            "avalan.tool.parser.uuid4", side_effect=[first_id, second_id]
        ):
            expected = [
                ToolCall(id=first_id, name="database.tables", arguments={}),
                ToolCall(
                    id=second_id,
                    name="database.run",
                    arguments={
                        "sql": (
                            "SELECT COUNT(*) AS product_count FROM products;"
                        )
                    },
                ),
            ]
            self.assertEqual(parser(text), expected)

    def test_multiple_without_commentary(self):
        parser = ToolCallParser(tool_format=ToolFormat.HARMONY)
        text = (
            "<|channel|>commentary to=functions.database.tables "
            "<|constrain|>json<|message|>{}<|call|>"
            "<|channel|>commentary to=functions.database.run "
            "<|constrain|>json<|message|>"
            '{"sql":"SELECT COUNT(*) AS product_count FROM products;"}'
            "<|call|>"
        )
        first_id = _uuid4()
        second_id = _uuid4()
        with patch(
            "avalan.tool.parser.uuid4", side_effect=[first_id, second_id]
        ):
            expected = [
                ToolCall(id=first_id, name="database.tables", arguments={}),
                ToolCall(
                    id=second_id,
                    name="database.run",
                    arguments={
                        "sql": (
                            "SELECT COUNT(*) AS product_count FROM products;"
                        )
                    },
                ),
            ]
            self.assertEqual(parser(text), expected)

    def test_without_constrain_trailing_text(self):
        parser = ToolCallParser(tool_format=ToolFormat.HARMONY)
        text = (
            "<|start|>assistant<|channel|>commentary to=functions.database.run"
            ' code<|message|>{"sql":"SELECT count(*) AS total_products FROM'
            ' product;"}<|call|>'
        )
        call_id = _uuid4()
        with patch("avalan.tool.parser.uuid4", return_value=call_id):
            expected = [
                ToolCall(
                    id=call_id,
                    name="database.run",
                    arguments={
                        "sql": (
                            "SELECT count(*) AS total_products FROM product;"
                        )
                    },
                )
            ]
            self.assertEqual(parser(text), expected)

    def test_without_constrain_without_trailing_text(self):
        parser = ToolCallParser(tool_format=ToolFormat.HARMONY)
        text = (
            "<|channel|>commentary to=functions.database.run"
            '<|message|>{"sql":"SELECT 1"}<|call|>'
        )
        call_id = _uuid4()
        with patch("avalan.tool.parser.uuid4", return_value=call_id):
            expected = [
                ToolCall(
                    id=call_id,
                    name="database.run",
                    arguments={"sql": "SELECT 1"},
                )
            ]
            self.assertEqual(parser(text), expected)

    def test_analysis_channel(self):
        parser = ToolCallParser(tool_format=ToolFormat.HARMONY)
        text = (
            "<|channel|>analysis to=functions.database.inspect "
            "<|constrain|>json<|message|>{}<|call|>"
        )
        call_id = _uuid4()
        with patch("avalan.tool.parser.uuid4", return_value=call_id):
            expected = [
                ToolCall(
                    id=call_id,
                    name="database.inspect",
                    arguments={},
                )
            ]
            self.assertEqual(parser(text), expected)

    def test_invalid_json(self):
        parser = ToolCallParser(tool_format=ToolFormat.HARMONY)
        text = (
            "<|channel|>commentary to=functions.calc "
            "<|constrain|>json<|message|>{<|call|>commentary"
        )
        self.assertIsNone(parser(text))

    def test_invalid_json_followed_by_valid(self):
        parser = ToolCallParser(tool_format=ToolFormat.HARMONY)
        text = (
            "<|channel|>commentary to=functions.bad "
            '<|constrain|>json<|message|>{"foo": }<|call|>commentary'
            "<|channel|>commentary to=functions.database.run "
            '<|constrain|>json<|message|>{"sql":"SELECT 1"}<|call|>'
        )
        call_id = _uuid4()
        with patch("avalan.tool.parser.uuid4", return_value=call_id):
            expected = [
                ToolCall(
                    id=call_id,
                    name="database.run",
                    arguments={"sql": "SELECT 1"},
                )
            ]
            self.assertEqual(parser(text), expected)


class ToolCallParserTagTestCase(TestCase):
    def setUp(self):
        self.parser = ToolCallParser()

    def test_single(self):
        text = (
            '<tool_call>{"name": "calculator", '
            '"arguments": {"expression": "1 + 1"}}</tool_call>'
        )
        call_id = _uuid4()
        with patch("avalan.tool.parser.uuid4", return_value=call_id):
            expected = [
                ToolCall(
                    id=call_id,
                    name="calculator",
                    arguments={"expression": "1 + 1"},
                )
            ]
            self.assertEqual(self.parser(text), expected)

    def test_single_quotes(self):
        text = (
            "<tool_call>{'name': 'calculator', 'arguments': {'expression':"
            " '2'}}</tool_call>"
        )
        call_id = _uuid4()
        with patch("avalan.tool.parser.uuid4", return_value=call_id):
            expected = [
                ToolCall(
                    id=call_id,
                    name="calculator",
                    arguments={"expression": "2"},
                )
            ]
            self.assertEqual(self.parser(text), expected)

    def test_multiple(self):
        text = (
            '<tool_call>{"name": "calculator", "arguments": {"expression":'
            ' "1"}}</tool_call><tool_call>{"name": "calculator", "arguments":'
            ' {"expression": "2"}}</tool_call>'
        )
        first_id = _uuid4()
        second_id = _uuid4()
        with patch(
            "avalan.tool.parser.uuid4", side_effect=[first_id, second_id]
        ):
            expected = [
                ToolCall(
                    id=first_id,
                    name="calculator",
                    arguments={"expression": "1"},
                ),
                ToolCall(
                    id=second_id,
                    name="calculator",
                    arguments={"expression": "2"},
                ),
            ]
            self.assertEqual(self.parser(text), expected)

    def test_with_name_attr(self):
        text = '<tool_call name="calculator">{"expression": "2"}</tool_call>'
        call_id = _uuid4()
        with patch("avalan.tool.parser.uuid4", return_value=call_id):
            expected = [
                ToolCall(
                    id=call_id,
                    name="calculator",
                    arguments={"expression": "2"},
                )
            ]
            self.assertEqual(self.parser(text), expected)

    def test_self_closing(self):
        text = (
            '<tool_call name="calculator" arguments=\'{"expression": "2"}\'/>'
        )
        call_id = _uuid4()
        with patch("avalan.tool.parser.uuid4", return_value=call_id):
            expected = [
                ToolCall(
                    id=call_id,
                    name="calculator",
                    arguments={"expression": "2"},
                )
            ]
            self.assertEqual(self.parser(text), expected)

    def test_tool_tag(self):
        text = '<tool name="calculator">{"expression": "2"}</tool>'
        call_id = _uuid4()
        with patch("avalan.tool.parser.uuid4", return_value=call_id):
            expected = [
                ToolCall(
                    id=call_id,
                    name="calculator",
                    arguments={"expression": "2"},
                )
            ]
            self.assertEqual(self.parser(text), expected)

    def test_invalid_json(self):
        text = (
            '<tool_call>{"name": "calculator", "arguments": {"expression":'
            ' "2"}</tool_call>'
        )
        self.assertIsNone(self.parser(text))

    def test_eos_token(self):
        parser = ToolCallParser(eos_token="<END>")
        text = (
            '<tool_call>{"name": "calculator", "arguments": {"expression":'
            ' "2"}}</tool_call><END>'
        )
        call_id = _uuid4()
        with patch("avalan.tool.parser.uuid4", return_value=call_id):
            expected = [
                ToolCall(
                    id=call_id,
                    name="calculator",
                    arguments={"expression": "2"},
                )
            ]
            self.assertEqual(parser(text), expected)

    def test_no_tool_call(self):
        self.assertIsNone(self.parser("hello"))

    def test_react_invalid_json_with_action_input(self):
        parser = ToolCallParser(tool_format=ToolFormat.REACT)
        text = 'Action: calc\nAction Input: {"expression": "1"'
        self.assertIsNone(parser(text))


if __name__ == "__main__":
    main()
