from unittest import TestCase, main
from unittest.mock import patch
from uuid import uuid4 as _uuid4

from avalan.backends.ds4_native import ThinkMode
from avalan.entities import MessageRole, MessageToolCall, ToolCall, ToolFormat
from avalan.tool.dsml import DsmlPromptMessage, DsmlTools
from avalan.tool.parser import ToolCallParser


class DsmlToolsTestCase(TestCase):
    def test_render_tool_schemas_uses_function_payload(self):
        schemas = [
            {
                "type": "function",
                "function": {
                    "name": "math.calculator",
                    "description": "Evaluate an expression.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string"},
                        },
                    },
                },
            }
        ]

        self.assertEqual(
            DsmlTools.render_tool_schemas(schemas),
            '{"name":"math.calculator","description":"Evaluate an '
            'expression.","parameters":{"type":"object","properties":'
            '{"expression":{"type":"string"}}}}',
        )

    def test_render_prompt_includes_tool_result_and_replayed_calls(self):
        call = MessageToolCall(
            id="call_1",
            name="math.calculator",
            arguments={"expression": "2 + 2"},
        )
        raw_dsml = (
            "\n\n<DSML｜tool_calls>\n"
            '<DSML｜invoke name="math.calculator"></DSML｜invoke>\n'
            "</DSML｜tool_calls>"
        )

        rendered = DsmlTools.render_prompt(
            "System",
            [
                DsmlPromptMessage(role=MessageRole.USER, content="calculate"),
                DsmlPromptMessage(
                    role=MessageRole.ASSISTANT,
                    content="",
                    tool_calls=(call,),
                ),
                DsmlPromptMessage(
                    role=MessageRole.TOOL,
                    content="value <4> & done",
                ),
            ],
            '{"name":"math.calculator"}',
            ThinkMode.NONE,
            lambda calls: raw_dsml if calls == (call,) else None,
        )

        self.assertIn("## Tools", rendered)
        self.assertIn(raw_dsml, rendered)
        self.assertIn(
            "<tool_result>value &lt;4&gt; &amp; done</tool_result>",
            rendered,
        )

    def test_render_tool_calls_escapes_values_selectively(self):
        rendered = DsmlTools.render_tool_calls(
            (
                MessageToolCall(
                    name='math."<calc>"',
                    arguments={
                        "text": "a & b </｜DSML｜parameter>",
                        "count": 2,
                    },
                ),
            )
        )

        self.assertIn(
            '<｜DSML｜invoke name="math.&quot;&lt;calc&gt;&quot;">',
            rendered,
        )
        self.assertIn(
            '<｜DSML｜parameter name="text" string="true">'
            "a & b &lt;/｜DSML｜parameter>",
            rendered,
        )
        self.assertIn(
            '<｜DSML｜parameter name="count" string="false">2',
            rendered,
        )

    def test_render_empty_tool_inputs_return_none_or_empty(self):
        self.assertIsNone(DsmlTools.render_tool_schemas(None))
        self.assertIsNone(DsmlTools.render_tool_schemas([]))
        self.assertEqual(DsmlTools.render_tool_calls(()), "")

    def test_render_prompt_with_thinking_handles_intermediate_assistant(self):
        rendered = DsmlTools.render_prompt(
            None,
            [
                DsmlPromptMessage(role=MessageRole.USER, content="one"),
                DsmlPromptMessage(
                    role=MessageRole.ASSISTANT,
                    content="two",
                    reasoning="hidden",
                ),
                DsmlPromptMessage(role=MessageRole.USER, content="three"),
                DsmlPromptMessage(
                    role=MessageRole.ASSISTANT,
                    content="four",
                    reasoning="final",
                ),
            ],
            None,
            ThinkMode.HIGH,
        )

        self.assertIn("<｜Assistant｜></think>two", rendered)
        self.assertIn("<｜Assistant｜><think>final</think>four", rendered)

    def test_parse_generated_message_returns_reasoning_and_raw_dsml(self):
        text = (
            "<think>Use calculator.</think>I will calculate.\n\n"
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="math.calculator">\n'
            '<｜DSML｜parameter name="expression" string="true">'
            "2 + 2"
            "</｜DSML｜parameter>\n"
            '<｜DSML｜parameter name="precision" string="false">'
            "2"
            "</｜DSML｜parameter>\n"
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>"
        )
        call_id = _uuid4()

        with patch("avalan.tool.dsml.uuid4", return_value=call_id):
            parsed = DsmlTools.parse_generated_message(text)

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.content, "I will calculate.")
        self.assertEqual(parsed.reasoning, "Use calculator.")
        self.assertEqual(
            parsed.raw_dsml,
            text.split("I will calculate.", 1)[1].rstrip(),
        )
        self.assertEqual(
            parsed.calls,
            (
                ToolCall(
                    id=f"ds4_tool_{call_id.hex}",
                    name="math.calculator",
                    arguments={"expression": "2 + 2", "precision": 2},
                ),
            ),
        )

    def test_parse_accepts_short_marker_and_plain_xml_fallback(self):
        short = (
            "<DSML｜tool_calls>"
            '<DSML｜invoke name="math.calculator">'
            '<DSML｜parameter name="expression">2 + 2'
            "</DSML｜parameter>"
            "</DSML｜invoke>"
            "</DSML｜tool_calls>"
        )
        plain = (
            "<tool_calls>"
            '<invoke name="math.calculator">'
            '<parameter name="expression">2 + 2</parameter>'
            "</invoke>"
            "</tool_calls>"
        )

        self.assertEqual(
            DsmlTools.parse_tool_calls(short)[0].name, "math.calculator"
        )
        self.assertEqual(
            DsmlTools.parse_tool_calls(plain)[0].arguments,
            {"expression": "2 + 2"},
        )

    def test_parse_malformed_block_returns_none(self):
        self.assertIsNone(
            DsmlTools.parse_generated_message("<｜DSML｜tool_calls>")
        )

    def test_parse_generated_message_handles_unclosed_invoke_and_invalid_json(
        self,
    ):
        malformed = '<tool_calls><invoke name="broken"></tool_calls>'
        parsed = DsmlTools.parse_generated_message(malformed)

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.calls, ())

        invalid_json = (
            "<tool_calls>"
            '<invoke name="math.calculator">'
            '<parameter name="value" string="false">not-json</parameter>'
            "</invoke>"
            "</tool_calls>"
        )
        parsed = DsmlTools.parse_generated_message(invalid_json)

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.calls[0].arguments, {"value": None})

    def test_parameter_end_after_returns_original_index_for_unknown_marker(
        self,
    ):
        self.assertEqual(DsmlTools._parameter_end_after("abc", 1), 1)

    def test_stream_argument_deltas_omits_dsml_tags(self):
        raw = (
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="math.calculator">\n'
            '<｜DSML｜parameter name="expression" string="true">2 + '
        )
        deltas, offset = DsmlTools.stream_argument_deltas(raw, 0)

        self.assertEqual(deltas, ())
        self.assertEqual(offset, 0)

        raw += "2</｜DSML｜parameter>\n"
        raw += (
            '<｜DSML｜parameter name="precision" string="false">2'
            "</｜DSML｜parameter>\n"
        )
        deltas, offset = DsmlTools.stream_argument_deltas(raw, offset)

        self.assertEqual(deltas, ("2 + 2", "2"))
        self.assertGreater(offset, 0)

        raw += "</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
        deltas, next_offset = DsmlTools.stream_argument_deltas(raw, offset)

        self.assertEqual(deltas, ())
        self.assertEqual(next_offset, offset)


class ToolCallParserDsmlTestCase(TestCase):
    def test_tool_format_dsml_parses_calls(self):
        parser = ToolCallParser(tool_format=ToolFormat.DSML)
        text = (
            "<｜DSML｜tool_calls>"
            '<｜DSML｜invoke name="math.calculator">'
            '<｜DSML｜parameter name="expression">2 + 2'
            "</｜DSML｜parameter>"
            "</｜DSML｜invoke>"
            "</｜DSML｜tool_calls>"
        )

        calls = parser(text)

        self.assertIsInstance(calls, list)
        assert isinstance(calls, list)
        self.assertEqual(calls[0].name, "math.calculator")

    def test_message_tool_calls_detects_dsml_without_configured_format(self):
        parser = ToolCallParser()
        text = (
            "<tool_calls>"
            '<invoke name="math.calculator">'
            '<parameter name="expression">2 + 2</parameter>'
            "</invoke>"
            "</tool_calls>"
        )

        self.assertEqual(
            parser.message_tool_calls(text)[0]["name"],
            "math.calculator",
        )

    def test_dsml_tool_call_status_reports_prefix_open_and_closed(self):
        parser = ToolCallParser(tool_format=ToolFormat.DSML)

        self.assertIs(
            parser.tool_call_status("<｜DSM"),
            ToolCallParser.ToolCallBufferStatus.PREFIX,
        )
        self.assertIs(
            parser.tool_call_status("<｜DSML｜tool_calls>"),
            ToolCallParser.ToolCallBufferStatus.OPEN,
        )
        self.assertIs(
            parser.tool_call_status(
                "<｜DSML｜tool_calls></｜DSML｜tool_calls>"
            ),
            ToolCallParser.ToolCallBufferStatus.CLOSED,
        )


if __name__ == "__main__":
    main()
