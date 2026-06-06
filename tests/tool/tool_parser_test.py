from typing import Any, cast
from unittest import TestCase, main
from unittest.mock import patch
from uuid import uuid4 as _uuid4

from avalan.entities import (
    ToolCall,
    ToolCallDiagnosticCode,
    ToolCallDiagnosticStage,
    ToolCallRecoveryFormat,
    ToolFormat,
)
from avalan.tool.parser import ToolCallParser


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

    def test_current_tuple_formats_do_not_return_tool_call_objects(self):
        cases = (
            (
                ToolFormat.JSON,
                '{"tool": "calculator", "arguments": {"expression": "1"}}',
            ),
            (
                ToolFormat.REACT,
                'Action: calculator\nAction Input: {"expression": "2"}',
            ),
            (ToolFormat.BRACKET, "[calculator](3)"),
            (
                ToolFormat.OPENAI,
                '{"name": "calculator", "arguments": {"expression": "4"}}',
            ),
        )

        for tool_format, text in cases:
            with self.subTest(tool_format=tool_format):
                parsed = ToolCallParser(tool_format=tool_format)(text)

                self.assertIsInstance(parsed, tuple)
                self.assertEqual(len(parsed), 2)
                self.assertIsInstance(parsed[0], str)
                self.assertNotIsInstance(parsed, list)

    def test_json_non_object_arguments_keep_current_tuple_shape(self):
        parser = ToolCallParser(tool_format=ToolFormat.JSON)

        parsed = parser('{"tool": "calculator", "arguments": ["1 + 1"]}')

        self.assertEqual(parsed, ("calculator", ["1 + 1"]))

    def test_openai_json_non_object_arguments_are_rejected(self):
        parser = ToolCallParser(tool_format=ToolFormat.OPENAI)

        parsed = parser('{"name": "calculator", "arguments": ["1 + 1"]}')

        self.assertIsNone(parsed)

    def test_configured_text_limit_keeps_legacy_none_shape(self):
        parser = ToolCallParser(
            tool_format=ToolFormat.JSON,
            maximum_text_size=4,
        )

        parsed = parser('{"tool": "calculator", "arguments": {}}')

        self.assertIsNone(parsed)

    def test_rejects_invalid_resource_limits(self):
        invalid_cases = (
            {"maximum_text_size": 0},
            {"maximum_payload_depth": -1},
            {"maximum_payload_size": True},
        )

        for kwargs in invalid_cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(AssertionError):
                    ToolCallParser(**kwargs)

    def test_recovery_formats_default_to_empty(self):
        parser = ToolCallParser()

        self.assertEqual(parser.recovery_formats, ())

    def test_recovery_formats_are_explicit(self):
        parser = ToolCallParser(
            recovery_formats=[
                ToolCallRecoveryFormat.FENCED,
                ToolCallRecoveryFormat.TOOL_CALL_BLOCK,
            ]
        )

        self.assertEqual(
            parser.recovery_formats,
            (
                ToolCallRecoveryFormat.FENCED,
                ToolCallRecoveryFormat.TOOL_CALL_BLOCK,
            ),
        )

    def test_rejects_invalid_recovery_formats(self):
        with self.assertRaises(AssertionError):
            ToolCallParser(recovery_formats=cast(Any, ["fenced"]))

    def test_recovery_shaped_text_does_not_parse_by_default(self):
        cases = (
            (
                "tool_call_block",
                (
                    '[TOOL_CALL]{"name": "calculator", "arguments": {}}'
                    "[/TOOL_CALL]"
                ),
            ),
            (
                "minimax_xml",
                (
                    '<invoke name="calculator"><parameter name="expression">'
                    "2 + 2</parameter></invoke>"
                ),
            ),
            (
                "tool_code",
                (
                    '<tool_code>{"name": "calculator", "arguments":'
                    " {}}</tool_code>"
                ),
            ),
            (
                "broad_xml",
                (
                    '<function name="calculator"><arguments>{}</arguments>'
                    "</function>"
                ),
            ),
            (
                "dsml_leakage",
                (
                    "<DSML:tool_calls>"
                    '<DSML:invoke name="calculator">'
                    "</DSML:invoke></DSML:tool_calls>"
                ),
            ),
            (
                "fenced",
                (
                    "```xml\n"
                    '<tool_call>{"name": "calculator", "arguments": {}}'
                    "</tool_call>\n```"
                ),
            ),
        )

        parser = ToolCallParser()
        for name, text in cases:
            with self.subTest(name=name):
                outcome = parser.parse(text)

                self.assertEqual(outcome.calls, [])
                self.assertEqual(outcome.diagnostics, [])

    def test_recovery_diagnostic_details_identify_source_format(self):
        details = {"reason": "malformed"}

        diagnostic = ToolCallParser._malformed_call_diagnostic(
            details=details,
            recovery_format=ToolCallRecoveryFormat.FENCED,
        )

        self.assertEqual(diagnostic.details["reason"], "malformed")
        self.assertEqual(diagnostic.details["source_format"], "fenced")
        self.assertEqual(details, {"reason": "malformed"})

    def test_recovery_formats_parse_explicit_payloads(self):
        cases = (
            (
                ToolCallRecoveryFormat.TOOL_CALL_BLOCK,
                (
                    '[TOOL_CALL]{"name": "calculator", "arguments": '
                    '{"expression": "1 + 1"}}[/TOOL_CALL]'
                ),
                {"expression": "1 + 1"},
            ),
            (
                ToolCallRecoveryFormat.MINIMAX_XML,
                (
                    '<invoke name="calculator"><parameter name="expression">'
                    "2 + 2</parameter></invoke>"
                ),
                {"expression": "2 + 2"},
            ),
            (
                ToolCallRecoveryFormat.MINIMAX_XML,
                (
                    '<mm:invoke name="calculator"><mm:parameter '
                    'name="expression">3 + 3</mm:parameter></mm:invoke>'
                ),
                {"expression": "3 + 3"},
            ),
            (
                ToolCallRecoveryFormat.TOOL_CODE,
                (
                    '<tool_code>{"name": "calculator", "arguments": '
                    '{"expression": "4 + 4"}}</tool_code>'
                ),
                {"expression": "4 + 4"},
            ),
            (
                ToolCallRecoveryFormat.TOOL_CODE,
                '<tool_code>calculator({"expression": "5 + 5"})</tool_code>',
                {"expression": "5 + 5"},
            ),
            (
                ToolCallRecoveryFormat.BROAD_XML,
                (
                    '<function name="calculator"><arguments>'
                    '{"expression": "6 + 6"}</arguments></function>'
                ),
                {"expression": "6 + 6"},
            ),
            (
                ToolCallRecoveryFormat.DSML_LEAKAGE,
                (
                    "<DSML:tool_calls>"
                    '<DSML:invoke name="calculator">'
                    '<DSML:parameter name="expression">7 + 7'
                    "</DSML:parameter>"
                    "</DSML:invoke></DSML:tool_calls>"
                ),
                {"expression": "7 + 7"},
            ),
            (
                ToolCallRecoveryFormat.FENCED,
                (
                    "```xml\n"
                    '<invoke name="calculator"><parameter '
                    'name="expression">8 + 8</parameter></invoke>\n'
                    "```"
                ),
                {"expression": "8 + 8"},
            ),
        )

        for recovery_format, text, arguments in cases:
            with self.subTest(recovery_format=recovery_format):
                call_id = _uuid4()
                parser = ToolCallParser(recovery_formats=[recovery_format])

                with patch("avalan.tool.parser.uuid4", return_value=call_id):
                    outcome = parser.parse(text)

                self.assertEqual(
                    outcome.calls,
                    [
                        ToolCall(
                            id=call_id,
                            name="calculator",
                            arguments=arguments,
                        )
                    ],
                )
                self.assertEqual(outcome.diagnostics, [])

    def test_recovery_formats_report_malformed_payloads(self):
        cases = (
            (
                ToolCallRecoveryFormat.TOOL_CALL_BLOCK,
                "[TOOL_CALL]{bad[/TOOL_CALL]",
            ),
            (
                ToolCallRecoveryFormat.MINIMAX_XML,
                (
                    '<invoke name="calculator"><parameter>bad</parameter>'
                    "</invoke>"
                ),
            ),
            (
                ToolCallRecoveryFormat.TOOL_CODE,
                "<tool_code>{bad</tool_code>",
            ),
            (
                ToolCallRecoveryFormat.BROAD_XML,
                (
                    '<function name="calculator"><arguments>[]</arguments>'
                    "</function>"
                ),
            ),
            (
                ToolCallRecoveryFormat.DSML_LEAKAGE,
                (
                    '<DSML:invoke name="calculator"><DSML:parameter>bad'
                    "</DSML:parameter></DSML:invoke>"
                ),
            ),
            (
                ToolCallRecoveryFormat.FENCED,
                "```json\n{bad\n```",
            ),
        )

        for recovery_format, text in cases:
            with self.subTest(recovery_format=recovery_format):
                outcome = ToolCallParser(
                    recovery_formats=[recovery_format]
                ).parse(text)

                self.assertEqual(outcome.calls, [])
                self.assertEqual(len(outcome.diagnostics), 1)
                self.assertEqual(
                    outcome.diagnostics[0].code,
                    ToolCallDiagnosticCode.MALFORMED_CALL,
                )
                self.assertEqual(
                    outcome.diagnostics[0].stage,
                    ToolCallDiagnosticStage.PARSE,
                )
                self.assertEqual(
                    outcome.diagnostics[0].details["source_format"],
                    recovery_format.value,
                )

    def test_recovery_keeps_valid_call_after_malformed_segment(self):
        first_id = _uuid4()
        second_id = _uuid4()
        parser = ToolCallParser(
            recovery_formats=[ToolCallRecoveryFormat.TOOL_CALL_BLOCK]
        )
        text = (
            "[TOOL_CALL]{bad[/TOOL_CALL]"
            '[TOOL_CALL]{"name": "calculator", "arguments": '
            '{"expression": "9 + 9"}}[/TOOL_CALL]'
        )

        with patch(
            "avalan.tool.parser.uuid4",
            side_effect=[first_id, second_id],
        ):
            outcome = parser.parse(text)

        self.assertEqual(
            outcome.calls,
            [
                ToolCall(
                    id=first_id,
                    name="calculator",
                    arguments={"expression": "9 + 9"},
                )
            ],
        )
        self.assertEqual(len(outcome.diagnostics), 1)
        self.assertEqual(
            outcome.diagnostics[0].details["source_format"],
            ToolCallRecoveryFormat.TOOL_CALL_BLOCK.value,
        )

    def test_fenced_recovery_preserves_multiple_calls_in_order(self):
        first_id = _uuid4()
        second_id = _uuid4()
        parser = ToolCallParser(
            recovery_formats=[ToolCallRecoveryFormat.FENCED]
        )
        text = (
            "```xml\n"
            '<invoke name="first"><parameter name="value">1</parameter>'
            "</invoke>"
            '<invoke name="second"><parameter name="value">2</parameter>'
            "</invoke>\n"
            "```"
        )

        with patch(
            "avalan.tool.parser.uuid4",
            side_effect=[first_id, second_id],
        ):
            outcome = parser.parse(text)

        self.assertEqual(
            outcome.calls,
            [
                ToolCall(
                    id=first_id,
                    name="first",
                    arguments={"value": "1"},
                ),
                ToolCall(
                    id=second_id,
                    name="second",
                    arguments={"value": "2"},
                ),
            ],
        )
        self.assertEqual(outcome.diagnostics, [])


class ToolCallParserParseOutcomeTestCase(TestCase):
    def test_parse_normalizes_tuple_formats(self):
        cases = (
            (
                ToolFormat.JSON,
                '{"tool": "calculator", "arguments": {"expression": "1"}}',
                "calculator",
                {"expression": "1"},
            ),
            (
                ToolFormat.REACT,
                'Action: calculator\nAction Input: {"expression": "2"}',
                "calculator",
                {"expression": "2"},
            ),
            (
                ToolFormat.BRACKET,
                "[calculator](3)",
                "calculator",
                {"input": "3"},
            ),
            (
                ToolFormat.OPENAI,
                '{"name": "calculator", "arguments": {"expression": "4"}}',
                "calculator",
                {"expression": "4"},
            ),
        )

        for tool_format, text, name, arguments in cases:
            with self.subTest(tool_format=tool_format):
                call_id = _uuid4()
                parser = ToolCallParser(tool_format=tool_format)

                with patch("avalan.tool.parser.uuid4", return_value=call_id):
                    outcome = parser.parse(text)

                self.assertEqual(
                    outcome.calls,
                    [
                        ToolCall(
                            id=call_id,
                            name=name,
                            arguments=arguments,
                        )
                    ],
                )
                self.assertEqual(outcome.diagnostics, [])

    def test_parse_preserves_call_ids_from_list_formats(self):
        tag_parser = ToolCallParser()
        tag_outcome = tag_parser.parse(
            '<tool_call>{"id": "call-1", "name": "calculator", '
            '"arguments": {"expression": "2 + 2"}}</tool_call>'
        )
        self.assertEqual(
            tag_outcome.calls,
            [
                ToolCall(
                    id="call-1",
                    name="calculator",
                    arguments={"expression": "2 + 2"},
                )
            ],
        )
        self.assertEqual(tag_outcome.diagnostics, [])

        harmony_id = _uuid4()
        harmony_parser = ToolCallParser(tool_format=ToolFormat.HARMONY)
        harmony_text = (
            "<|channel|>commentary to=functions.calculator"
            '<|message|>{"expression": "3 + 3"}<|call|>'
        )
        with patch("avalan.tool.parser.uuid4", return_value=harmony_id):
            harmony_outcome = harmony_parser.parse(harmony_text)
        self.assertEqual(
            harmony_outcome.calls,
            [
                ToolCall(
                    id=harmony_id,
                    name="calculator",
                    arguments={"expression": "3 + 3"},
                )
            ],
        )
        self.assertEqual(harmony_outcome.diagnostics, [])

        dsml_call = ToolCall(
            id="dsml-call",
            name="calculator",
            arguments={"expression": "4 + 4"},
        )
        with patch(
            "avalan.tool.parser.DsmlTools.parse_tool_calls",
            return_value=[dsml_call],
        ):
            dsml_outcome = ToolCallParser(tool_format=ToolFormat.DSML).parse(
                "dsml"
            )
        self.assertEqual(dsml_outcome.calls, [dsml_call])
        self.assertEqual(dsml_outcome.diagnostics, [])

    def test_parse_reports_non_object_argument_payloads(self):
        cases = (
            (
                ToolFormat.JSON,
                '{"tool": "calculator", "arguments": ["1 + 1"]}',
            ),
            (
                ToolFormat.REACT,
                'Action: calculator\nAction Input: ["2 + 2"]',
            ),
            (
                ToolFormat.OPENAI,
                '{"name": "calculator", "arguments": ["3 + 3"]}',
            ),
            (
                ToolFormat.HARMONY,
                (
                    "<|channel|>commentary to=functions.calculator"
                    '<|message|>["4 + 4"]<|call|>'
                ),
            ),
            (
                None,
                (
                    '<tool_call>{"name": "calculator", "arguments": '
                    '["5 + 5"]}</tool_call>'
                ),
            ),
        )

        for tool_format, text in cases:
            with self.subTest(tool_format=tool_format):
                outcome = ToolCallParser(tool_format=tool_format).parse(text)

                self.assertEqual(outcome.calls, [])
                self.assertEqual(len(outcome.diagnostics), 1)
                diagnostic = outcome.diagnostics[0]
                self.assertEqual(
                    diagnostic.code,
                    ToolCallDiagnosticCode.MALFORMED_ARGUMENTS,
                )
                self.assertEqual(
                    diagnostic.stage, ToolCallDiagnosticStage.PARSE
                )
                self.assertEqual(diagnostic.requested_name, "calculator")

    def test_parse_reports_malformed_call_payloads(self):
        cases = (
            (ToolFormat.JSON, '{"tool": "calculator", "arguments": '),
            (
                ToolFormat.REACT,
                'Action: calculator\nAction Input: {"expression": ',
            ),
            (ToolFormat.OPENAI, '{"name": 3, "arguments": {}}'),
            (
                None,
                '<tool_call>{"name": "calculator", "arguments": }</tool_call>',
            ),
        )

        for tool_format, text in cases:
            with self.subTest(tool_format=tool_format):
                outcome = ToolCallParser(tool_format=tool_format).parse(text)

                self.assertEqual(outcome.calls, [])
                self.assertEqual(len(outcome.diagnostics), 1)
                diagnostic = outcome.diagnostics[0]
                self.assertEqual(
                    diagnostic.code,
                    ToolCallDiagnosticCode.MALFORMED_CALL,
                )
                self.assertEqual(
                    diagnostic.stage, ToolCallDiagnosticStage.PARSE
                )

    def test_parse_reports_text_size_limit(self):
        parser = ToolCallParser(
            tool_format=ToolFormat.JSON,
            maximum_text_size=4,
        )

        outcome = parser.parse('{"tool": "calculator", "arguments": {}}')

        self.assertEqual(outcome.calls, [])
        self.assertEqual(len(outcome.diagnostics), 1)
        diagnostic = outcome.diagnostics[0]
        self.assertIs(diagnostic.code, ToolCallDiagnosticCode.MAXIMUM_SIZE)
        self.assertIs(diagnostic.stage, ToolCallDiagnosticStage.PARSE)
        self.assertEqual(diagnostic.details["limit"], 4)

    def test_parse_reports_argument_depth_limit(self):
        cases = (
            (
                ToolFormat.JSON,
                (
                    '{"tool": "calculator", "arguments": '
                    '{"payload": {"value": 1}}}'
                ),
            ),
            (
                None,
                (
                    '<tool_call>{"name": "calculator", "arguments": '
                    '{"payload": {"value": 1}}}</tool_call>'
                ),
            ),
        )

        for tool_format, text in cases:
            with self.subTest(tool_format=tool_format):
                parser = ToolCallParser(
                    tool_format=tool_format,
                    maximum_payload_depth=1,
                )

                outcome = parser.parse(text)

                self.assertEqual(outcome.calls, [])
                self.assertEqual(len(outcome.diagnostics), 1)
                diagnostic = outcome.diagnostics[0]
                self.assertIs(
                    diagnostic.code, ToolCallDiagnosticCode.MAXIMUM_DEPTH
                )
                self.assertIs(diagnostic.stage, ToolCallDiagnosticStage.PARSE)
                self.assertEqual(diagnostic.requested_name, "calculator")
                self.assertEqual(diagnostic.details["limit"], 1)

    def test_parse_reports_argument_size_limit(self):
        parser = ToolCallParser(
            tool_format=ToolFormat.JSON,
            maximum_payload_size=8,
        )

        outcome = parser.parse(
            '{"tool": "calculator", "arguments": {"payload": "large"}}'
        )

        self.assertEqual(outcome.calls, [])
        self.assertEqual(len(outcome.diagnostics), 1)
        diagnostic = outcome.diagnostics[0]
        self.assertIs(diagnostic.code, ToolCallDiagnosticCode.MAXIMUM_SIZE)
        self.assertIs(diagnostic.stage, ToolCallDiagnosticStage.PARSE)
        self.assertEqual(diagnostic.requested_name, "calculator")
        self.assertEqual(diagnostic.details["limit"], 8)

    def test_parse_returns_empty_outcome_for_plain_text(self):
        for tool_format in (None, ToolFormat.JSON):
            with self.subTest(tool_format=tool_format):
                outcome = ToolCallParser(tool_format=tool_format).parse(
                    "plain assistant content"
                )

                self.assertEqual(outcome.calls, [])
                self.assertEqual(outcome.diagnostics, [])


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

    def test_commentary_channel_after_to(self):
        parser = ToolCallParser(tool_format=ToolFormat.HARMONY)
        text = (
            "<|start|>assistant<|channel|>commentary"
            " to=functions.database.tables<|channel|>commentary"
            " <|constrain|>json<|message|>{}<|call|>"
        )
        call_id = _uuid4()
        with patch("avalan.tool.parser.uuid4", return_value=call_id):
            expected = [
                ToolCall(
                    id=call_id,
                    name="database.tables",
                    arguments={},
                )
            ]
            self.assertEqual(parser(text), expected)

    def test_empty_message(self):
        parser = ToolCallParser(tool_format=ToolFormat.HARMONY)
        text = (
            "<|channel|>commentary to=functions.database.ping "
            "<|message|><|call|>"
        )
        call_id = _uuid4()
        with patch("avalan.tool.parser.uuid4", return_value=call_id):
            expected = [
                ToolCall(id=call_id, name="database.ping", arguments={})
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

    def test_single_preserves_embedded_id(self):
        text = (
            '<tool_call>{"name": "calculator", '
            '"arguments": {"expression": "1 + 1"}, '
            '"id": "call_123"}</tool_call>'
        )
        with patch("avalan.tool.parser.uuid4") as uuid4_mock:
            expected = [
                ToolCall(
                    id="call_123",
                    name="calculator",
                    arguments={"expression": "1 + 1"},
                )
            ]
            self.assertEqual(self.parser(text), expected)
        uuid4_mock.assert_not_called()

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
