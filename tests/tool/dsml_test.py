import subprocess
import sys
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch
from uuid import uuid4 as _uuid4

import pytest

from avalan.backends.ds4_native import ThinkMode
from avalan.entities import MessageRole, MessageToolCall, ToolCall, ToolFormat
from avalan.tool.dsml import DsmlPromptMessage, DsmlTools
from avalan.tool.parser import ToolCallParser

pytest.importorskip(
    "pyds4",
    reason="pyds4 is not installed; install the test group to run DS4 tests.",
)


class DsmlToolsTestCase(TestCase):
    def test_importing_dsml_module_does_not_import_pyds4(self):
        env = {
            "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
        }
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "import avalan.tool.dsml as dsml; "
                    "print(hasattr(dsml, 'DsmlTools')); "
                    "print('pyds4' in sys.modules)"
                ),
            ],
            check=True,
            env=env,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.stdout.splitlines(), ["True", "False"])

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

    def test_render_tool_result_escapes_content(self):
        self.assertEqual(
            DsmlTools.render_tool_result("value <4> & done"),
            "<tool_result>value &lt;4&gt; &amp; done</tool_result>",
        )

    def test_split_reasoning_returns_visible_content(self):
        self.assertEqual(
            DsmlTools.split_reasoning("<think>hidden</think>visible"),
            ("visible", "hidden"),
        )

    def test_tools_prompt_rejects_empty_schema_text(self):
        with self.assertRaises(ValueError):
            DsmlTools.tools_prompt("")

    def test_tools_prompt_rejects_none_from_dsml_module(self):
        fake_dsml = type(
            "FakeDsml",
            (),
            {"tools_prompt": staticmethod(lambda _: None)},
        )()

        with patch.object(DsmlTools, "_pyds4_dsml", return_value=fake_dsml):
            with self.assertRaises(ValueError):
                DsmlTools.tools_prompt("schema")

    def test_tools_prompt_returns_rendered_prompt(self):
        fake_dsml = type(
            "FakeDsml",
            (),
            {"tools_prompt": staticmethod(lambda _: "prompt")},
        )()

        with patch.object(DsmlTools, "_pyds4_dsml", return_value=fake_dsml):
            self.assertEqual(DsmlTools.tools_prompt("schema"), "prompt")

    def test_missing_pyds4_module_raises_runtime_error(self):
        with patch(
            "avalan.tool.dsml.import_module",
            side_effect=ModuleNotFoundError("pyds4"),
        ):
            with self.assertRaises(RuntimeError):
                DsmlTools.render_tool_result("content")

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

    def test_parse_tool_calls_returns_none_without_calls(self):
        self.assertIsNone(DsmlTools.parse_tool_calls("visible content"))

    def test_to_tool_call_defaults_non_dict_arguments(self):
        call_id = _uuid4()
        native_call = type(
            "NativeCall",
            (),
            {"name": "math.calculator", "arguments": "bad"},
        )()

        with patch("avalan.tool.dsml.uuid4", return_value=call_id):
            call = DsmlTools._to_tool_call(native_call)

        self.assertEqual(call.id, f"ds4_tool_{call_id.hex}")
        self.assertEqual(call.name, "math.calculator")
        self.assertEqual(call.arguments, {})

    def test_tool_call_start_suffix_length_tracks_marker_variants(self):
        self.assertEqual(
            DsmlTools.tool_call_start_suffix_length(
                "visible\n\n<｜DSML｜tool"
            ),
            len("\n\n<｜DSML｜tool"),
        )
        self.assertEqual(
            DsmlTools.tool_call_start_suffix_length("visible\n<DSML｜tool"),
            len("\n<DSML｜tool"),
        )
        self.assertEqual(
            DsmlTools.tool_call_start_suffix_length("visible<tool_calls>"),
            len("<tool_calls>"),
        )
        self.assertEqual(
            DsmlTools.tool_call_start_suffix_length("visible only"),
            0,
        )

    def test_tool_call_start_suffix_length_rejects_invalid_text(self):
        with self.assertRaises(TypeError):
            DsmlTools.tool_call_start_suffix_length(  # type: ignore[arg-type]
                object()
            )

    def test_parse_generated_message_rejects_malformed_dsml(
        self,
    ):
        malformed = '<tool_calls><invoke name="broken"></tool_calls>'
        invalid_json = (
            "<tool_calls>"
            '<invoke name="math.calculator">'
            '<parameter name="value" string="false">not-json</parameter>'
            "</invoke>"
            "</tool_calls>"
        )

        self.assertIsNone(DsmlTools.parse_generated_message(malformed))
        self.assertIsNone(DsmlTools.parse_generated_message(invalid_json))

    def test_stream_argument_deltas_rejects_invalid_offsets(
        self,
    ):
        with self.assertRaises(ValueError):
            DsmlTools.stream_argument_deltas("", -1)

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

    def test_message_tool_calls_uses_dsml_start_helper(self):
        parser = ToolCallParser()
        text = "generated tool_calls message"
        call = ToolCall(
            id="call_1",
            name="math.calculator",
            arguments={"expression": "2 + 2"},
        )

        with (
            patch.object(
                DsmlTools,
                "tool_call_start_span",
                return_value=(0, len(text)),
            ) as start_span,
            patch.object(
                DsmlTools,
                "parse_tool_calls",
                return_value=[call],
            ) as parse_tool_calls,
        ):
            calls = parser.message_tool_calls(text)

        start_span.assert_called_once_with(text)
        parse_tool_calls.assert_called_once_with(text)
        self.assertEqual(calls[0]["name"], "math.calculator")
        self.assertEqual(calls[0]["arguments"], {"expression": "2 + 2"})

    def test_message_tool_calls_skips_dsml_helper_for_plain_text(self):
        parser = ToolCallParser()

        with patch.object(
            DsmlTools,
            "tool_call_start_span",
            side_effect=AssertionError("unexpected DSML import"),
        ):
            self.assertEqual(parser.message_tool_calls("plain text"), [])

    def test_message_tool_calls_ignores_missing_pyds4_auto_probe(self):
        parser = ToolCallParser()

        with patch.object(
            DsmlTools,
            "tool_call_start_span",
            side_effect=RuntimeError("pyds4 missing"),
        ):
            self.assertEqual(
                parser.message_tool_calls("mentions tool_calls only"),
                [],
            )

    def test_message_tool_calls_reraises_missing_pyds4_for_dsml_format(self):
        parser = ToolCallParser(tool_format=ToolFormat.DSML)

        with patch.object(
            DsmlTools,
            "tool_call_start_span",
            side_effect=RuntimeError("pyds4 missing"),
        ):
            with self.assertRaises(RuntimeError):
                parser.message_tool_calls("mentions tool_calls only")

    def test_dsml_tool_call_status_reports_prefix_open_and_closed(self):
        parser = ToolCallParser(tool_format=ToolFormat.DSML)

        cases = (
            ("plain text", ToolCallParser.ToolCallBufferStatus.NONE),
            ("<｜DSM", ToolCallParser.ToolCallBufferStatus.PREFIX),
            (
                "<｜DSML｜tool_calls>",
                ToolCallParser.ToolCallBufferStatus.OPEN,
            ),
            (
                "<｜DSML｜tool_calls></｜DSML｜tool_calls>",
                ToolCallParser.ToolCallBufferStatus.CLOSED,
            ),
            ("<DSML｜tool_calls>", ToolCallParser.ToolCallBufferStatus.OPEN),
            (
                "<DSML｜tool_calls></DSML｜tool_calls>",
                ToolCallParser.ToolCallBufferStatus.CLOSED,
            ),
            ("<tool_calls>", ToolCallParser.ToolCallBufferStatus.OPEN),
            (
                "<tool_calls></tool_calls>",
                ToolCallParser.ToolCallBufferStatus.CLOSED,
            ),
        )

        for text, expected in cases:
            with self.subTest(text=text):
                self.assertIs(parser.tool_call_status(text), expected)


if __name__ == "__main__":
    main()
