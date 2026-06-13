from typing import Any, cast
from unittest import TestCase, main
from unittest.mock import patch
from uuid import uuid4 as _uuid4
from xml.etree import ElementTree

from avalan.entities import (
    Message,
    MessageContentText,
    MessageRole,
    ToolCall,
    ToolCallDiagnosticCode,
    ToolCallDiagnosticStage,
    ToolCallParseOutcome,
    ToolCallRecoveryFormat,
    ToolFormat,
)
from avalan.tool.parser import ToolCallParser


class ToolCallParserFullCoverageTestCase(TestCase):
    def test_parser_properties_and_stream_token_detection(self):
        parser = ToolCallParser(tool_format=ToolFormat.JSON)

        self.assertEqual(parser.tool_format, ToolFormat.JSON)
        parser.set_eos_token("<eos>")
        self.assertFalse(parser.is_potential_tool_call("", ""))
        self.assertFalse(parser.is_potential_tool_call("buffer", "   "))
        self.assertTrue(parser.is_potential_tool_call("buffer", "<"))

    def test_prepare_message_for_template_leaves_existing_list_without_source(
        self,
    ):
        parser = ToolCallParser()
        message = Message(role=MessageRole.ASSISTANT, content=None)
        message_dict: dict[str, object] = {"tool_calls": []}

        prepared = parser.prepare_message_for_template(message, message_dict)

        self.assertIsNone(prepared.template_content)
        self.assertEqual(prepared.message_dict, {"tool_calls": []})

    def test_prepare_message_for_template_extracts_structured_harmony(self):
        parser = ToolCallParser()
        message = Message(
            role=MessageRole.ASSISTANT,
            content=(
                "<|channel|>analysis<|message|>Inspect inputs<|call|>"
                "<|channel|>commentary to=functions.calculator"
                '<|message|>{"expression": "2 + 2"}<|call|>'
                "<|channel|>final<|message|>Result ready<|end|>"
            ),
        )
        message_dict: dict[str, object] = {"tool_calls": None}
        call_id = _uuid4()

        with patch("avalan.tool.parser.uuid4", return_value=call_id):
            prepared = parser.prepare_message_for_template(
                message, message_dict
            )

        self.assertEqual(prepared.template_content, "Result ready")
        self.assertEqual(
            prepared.message_dict,
            {
                "content": "Result ready",
                "thinking": "Inspect inputs",
                "tool_calls": [
                    {
                        "id": str(call_id),
                        "name": "calculator",
                        "arguments": {"expression": "2 + 2"},
                        "content_type": "json",
                    }
                ],
            },
        )

    def test_prepare_message_for_template_ignores_unstructured_source(self):
        parser = ToolCallParser()
        message = Message(
            role=MessageRole.ASSISTANT,
            content="plain text",
        )
        message_dict: dict[str, object] = {"tool_calls": []}

        prepared = parser.prepare_message_for_template(message, message_dict)

        self.assertEqual(prepared.template_content, "plain text")
        self.assertEqual(prepared.message_dict, {"tool_calls": []})

    def test_prepare_message_for_template_preserves_existing_tool_calls(self):
        parser = ToolCallParser()
        message = Message(
            role=MessageRole.ASSISTANT,
            content=(
                "<|channel|>analysis<|message|>Inspect inputs<|call|>"
                "<|channel|>commentary to=functions.calculator"
                '<|message|>{"expression": "2 + 2"}<|call|>'
                "<|channel|>final<|message|>Result ready<|end|>"
            ),
        )
        message_dict: dict[str, object] = {
            "tool_calls": [
                {
                    "id": "existing",
                    "name": "existing",
                    "arguments": {},
                    "content_type": "json",
                }
            ]
        }

        prepared = parser.prepare_message_for_template(message, message_dict)

        self.assertEqual(prepared.template_content, "Result ready")
        self.assertEqual(
            prepared.message_dict,
            {
                "content": "Result ready",
                "thinking": "Inspect inputs",
                "tool_calls": [
                    {
                        "id": "existing",
                        "name": "existing",
                        "arguments": {},
                        "content_type": "json",
                    }
                ],
            },
        )

    def test_extract_structured_message_requires_channel_marker(self):
        parser = ToolCallParser()

        self.assertIsNone(parser.extract_structured_message("plain text"))

    def test_has_dsml_tool_call_start_handles_lazy_dependency_errors(self):
        parser = ToolCallParser()

        with patch(
            "avalan.tool.parser.DsmlTools.tool_call_start_span",
            return_value=(0, 4),
        ):
            self.assertTrue(parser._has_dsml_tool_call_start("tool_calls"))

        with patch(
            "avalan.tool.parser.DsmlTools.tool_call_start_span",
            side_effect=RuntimeError("pyds4 unavailable"),
        ):
            self.assertFalse(parser._has_dsml_tool_call_start("tool_calls"))

        dsml_parser = ToolCallParser(tool_format=ToolFormat.DSML)
        with patch(
            "avalan.tool.parser.DsmlTools.tool_call_start_span",
            side_effect=RuntimeError("pyds4 unavailable"),
        ):
            with self.assertRaises(RuntimeError):
                dsml_parser._has_dsml_tool_call_start("tool_calls")

    def test_message_tool_calls_returns_empty_without_configured_parser(self):
        parser = ToolCallParser()

        self.assertEqual(parser.message_tool_calls("plain text"), [])

    def test_message_tool_calls_extracts_dsml_payload(self):
        parser = ToolCallParser()
        call = ToolCall(
            id="call-1",
            name="calculator",
            arguments={"expression": "2 + 2"},
        )

        with (
            patch(
                "avalan.tool.parser.DsmlTools.tool_call_start_span",
                return_value=(0, 12),
            ),
            patch(
                "avalan.tool.parser.DsmlTools.parse_tool_calls",
                return_value=[call],
            ),
        ):
            tool_calls = parser.message_tool_calls("tool_calls")

        self.assertEqual(
            tool_calls,
            [
                {
                    "id": "call-1",
                    "name": "calculator",
                    "arguments": {"expression": "2 + 2"},
                    "content_type": "json",
                }
            ],
        )

    def test_message_tool_calls_returns_empty_for_unexpected_shape(self):
        parser = ToolCallParser(tool_format=ToolFormat.JSON)

        with patch.object(
            ToolCallParser,
            "__call__",
            return_value=("calculator", {}, "extra"),
        ):
            self.assertEqual(parser.message_tool_calls("ignored"), [])

    def test_message_tool_calls_extracts_tuple_payload(self):
        parser = ToolCallParser(tool_format=ToolFormat.JSON)

        self.assertEqual(
            parser.message_tool_calls(
                '{"tool": "calculator", "arguments": {"expression": "2"}}'
            ),
            [
                {
                    "id": None,
                    "name": "calculator",
                    "arguments": {"expression": "2"},
                    "content_type": "json",
                }
            ],
        )

    def test_message_tool_calls_rejects_malformed_names(self):
        parser = ToolCallParser(tool_format=ToolFormat.JSON)
        invalid_call = ToolCall(
            id="call-1",
            name="math..calculator",
            arguments={},
        )

        self.assertEqual(
            parser.message_tool_calls(
                '{"tool": "math..calculator", "arguments": {}}'
            ),
            [],
        )
        with patch.object(
            ToolCallParser, "__call__", return_value=[invalid_call]
        ):
            self.assertEqual(parser.message_tool_calls("ignored"), [])

    def test_extract_harmony_content_collects_analysis_and_final_messages(
        self,
    ):
        parser = ToolCallParser()
        thinking, content = parser.extract_harmony_content(
            "<|channel|>analysis<|message|>step one<|call|>"
            "<|channel|>analysis<|message|>step two<|call|>"
            "<|channel|>final<|message|>final one<|end|>"
            "<|channel|>final<|message|>final two<|end|>"
        )

        self.assertEqual(thinking, "step one\n\nstep two")
        self.assertEqual(content, "final one\n\nfinal two")

    def test_extract_harmony_content_returns_empty_content_for_analysis_only(
        self,
    ):
        parser = ToolCallParser()
        thinking, content = parser.extract_harmony_content(
            "<|channel|>analysis<|message|>only analysis<|call|>"
        )

        self.assertEqual(thinking, "only analysis")
        self.assertEqual(content, "")

    def test_extract_harmony_content_skips_empty_segments_and_falls_back(
        self,
    ):
        parser = ToolCallParser()
        thinking, content = parser.extract_harmony_content(
            "<|channel|>analysis<|message|>   <|call|>"
            "<|channel|>final<|message|>done<|end|>"
        )

        self.assertIsNone(thinking)
        self.assertEqual(content, "done")

        thinking, content = parser.extract_harmony_content("plain\n\n\ntext")

        self.assertIsNone(thinking)
        self.assertEqual(content, "plain\n\ntext")

    def test_resolve_text_source_prefers_template_content_variants(self):
        parser = ToolCallParser()

        self.assertEqual(
            parser._resolve_text_source("template", None), "template"
        )
        self.assertEqual(
            parser._resolve_text_source(
                MessageContentText(type="text", text="direct"),
                None,
            ),
            "direct",
        )
        self.assertEqual(
            parser._resolve_text_source(
                [MessageContentText(type="text", text="listed")],
                None,
            ),
            "listed",
        )

    def test_resolve_text_source_accepts_serialized_content_variants(self):
        parser = ToolCallParser()

        self.assertEqual(
            parser._resolve_text_source(None, "serialized"), "serialized"
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

    def test_resolve_text_source_returns_none_for_invalid_serialized_text(
        self,
    ):
        parser = ToolCallParser()

        self.assertIsNone(
            parser._resolve_text_source(
                None,
                {"type": "text", "text": 1},
            )
        )
        self.assertIsNone(
            parser._resolve_text_source(
                None,
                [{"type": "text", "text": 1}],
            )
        )

    def test_merge_thinking_preserves_existing_when_new_is_none(self):
        parser = ToolCallParser()
        message_dict: dict[str, object] = {"thinking": "existing"}

        parser._merge_thinking(message_dict, None)

        self.assertEqual(message_dict["thinking"], "existing")

    def test_merge_thinking_normalizes_empty_and_combines_existing(self):
        parser = ToolCallParser()
        message_dict: dict[str, object] = {"thinking": ""}

        parser._merge_thinking(message_dict, None)

        self.assertIsNone(message_dict["thinking"])

        message_dict = {"thinking": "existing"}

        parser._merge_thinking(message_dict, "fresh")

        self.assertEqual(message_dict["thinking"], "existing\n\nfresh")

    def test_merge_thinking_replaces_blank_existing_value(self):
        parser = ToolCallParser()
        message_dict: dict[str, object] = {"thinking": "   "}

        parser._merge_thinking(message_dict, "fresh")

        self.assertEqual(message_dict["thinking"], "fresh")

    def test_tool_call_status_reports_standard_states(self):
        parser = ToolCallParser()

        self.assertEqual(
            parser.tool_call_status("plain text"),
            ToolCallParser.ToolCallBufferStatus.NONE,
        )
        for text in (
            "before <tool_callout> text",
            "before <tool_callbacks are text",
            "<tool_calls>not a standard tool tag</tool_calls>",
        ):
            with self.subTest(text=text):
                self.assertEqual(
                    parser.tool_call_status(text),
                    ToolCallParser.ToolCallBufferStatus.NONE,
                )
                self.assertEqual(
                    parser.tool_call_status(text, final=True),
                    ToolCallParser.ToolCallBufferStatus.NONE,
                )
                outcome = parser.parse(text)
                self.assertEqual(outcome.calls, [])
                self.assertEqual(outcome.diagnostics, [])
                self.assertEqual(parser.stream_buffer_diagnostics(text), [])
        self.assertEqual(
            parser.tool_call_status("<too"),
            ToolCallParser.ToolCallBufferStatus.PREFIX,
        )
        self.assertEqual(
            parser.tool_call_status("<tool_call"),
            ToolCallParser.ToolCallBufferStatus.PREFIX,
        )
        self.assertEqual(
            parser.tool_call_status("<tool_call", final=True),
            ToolCallParser.ToolCallBufferStatus.UNTERMINATED,
        )
        self.assertEqual(
            parser.tool_call_status('<tool_call name="calculator"'),
            ToolCallParser.ToolCallBufferStatus.OPEN,
        )
        self.assertEqual(
            parser.tool_call_status("<tool_call></tool_call>"),
            ToolCallParser.ToolCallBufferStatus.CLOSED,
        )
        self.assertEqual(
            parser.tool_call_status(
                '<tool_call>{"name": "calculator", '
                '"arguments": {"text": "/>"}}'
            ),
            ToolCallParser.ToolCallBufferStatus.OPEN,
        )
        self.assertEqual(
            parser.tool_call_status(
                '<tool_call>{"name": "calculator", '
                '"arguments": {"text": "<tool_call name=\\"inner\\"/>"}}'
            ),
            ToolCallParser.ToolCallBufferStatus.OPEN,
        )
        self.assertEqual(
            parser.tool_call_status(
                '<tool_call>{"name": "calculator", '
                '"arguments": {"text": "<tool_call></tool_call>"}}'
            ),
            ToolCallParser.ToolCallBufferStatus.OPEN,
        )
        self.assertEqual(
            parser.tool_call_status(
                '<tool_call>{"name": "calculator", '
                '"arguments": {"text": "</tool_call>"}}'
            ),
            ToolCallParser.ToolCallBufferStatus.OPEN,
        )
        self.assertEqual(
            parser.tool_call_status(
                '<tool_call>{"name": "calculator", '
                '"arguments": {"text": "escaped \\" </tool_call>"}}'
            ),
            ToolCallParser.ToolCallBufferStatus.OPEN,
        )
        self.assertEqual(
            parser.tool_call_status(
                '<tool_call name="first"/>'
                '<tool_call>{"name": "calculator", '
                '"arguments": {"text": "<tool_call name=\\"inner\\"/>"}}'
            ),
            ToolCallParser.ToolCallBufferStatus.OPEN,
        )
        self.assertEqual(
            parser.tool_call_status('<tool_call name="calculator"/>'),
            ToolCallParser.ToolCallBufferStatus.CLOSED,
        )
        self.assertEqual(
            parser.tool_call_status(
                '<tool_call name="calculator" arguments=\'{"expression": '
                '"1 > 0"}\'/>'
            ),
            ToolCallParser.ToolCallBufferStatus.CLOSED,
        )
        self.assertEqual(
            parser.tool_call_status(
                '<tool_call>{"name": "first", "arguments": {}}</tool_call>'
                '<tool_call name="second"/>'
            ),
            ToolCallParser.ToolCallBufferStatus.CLOSED,
        )

    def test_tool_call_status_uses_latest_mixed_tag_start(self) -> None:
        parser = ToolCallParser()
        open_sibling = (
            '<tool_call>{"name": "first", "arguments": {}}</tool_call>'
            ' between <tool name="second">{"value": 2}'
        )
        closed_sibling = open_sibling + "</tool>"

        self.assertEqual(
            parser.tool_call_status(open_sibling),
            ToolCallParser.ToolCallBufferStatus.OPEN,
        )
        self.assertEqual(
            parser.tool_call_status(open_sibling, final=True),
            ToolCallParser.ToolCallBufferStatus.UNTERMINATED,
        )
        self.assertEqual(
            parser.tool_call_status(closed_sibling),
            ToolCallParser.ToolCallBufferStatus.CLOSED,
        )
        self.assertEqual(
            parser._latest_tool_start(
                '<tool_call name="first"/><tool name="second">',
                ["<tool_call", "<tool ", "<tool>"],
            ),
            (25, "<tool "),
        )

    def test_tool_call_status_ignores_quoted_self_close_marker(self) -> None:
        parser = ToolCallParser()
        text = (
            '<tool_call name="calculator" arguments=\'{"expression": "/>"}\'>'
        )

        self.assertEqual(
            parser.tool_call_status(text),
            ToolCallParser.ToolCallBufferStatus.OPEN,
        )
        self.assertEqual(
            parser.tool_call_status(text, final=True),
            ToolCallParser.ToolCallBufferStatus.UNTERMINATED,
        )
        diagnostics = parser.stream_buffer_diagnostics(text)
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(
            diagnostics[0].code,
            ToolCallDiagnosticCode.MALFORMED_CALL,
        )
        self.assertEqual(
            diagnostics[0].details["stream_status"], "unterminated"
        )

    def test_tool_call_status_handles_visible_quote_text_before_tag(
        self,
    ) -> None:
        parser = ToolCallParser()
        apostrophe_text = (
            'don\'t <tool_call name="calculator" arguments=\'{"value": 2}\'/>'
        )
        quoted_prefix_text = (
            '"<tool_call" '
            '<tool_call name="calculator" arguments=\'{"value": 2}\'/>'
        )

        for text in (apostrophe_text, quoted_prefix_text):
            with self.subTest(text=text):
                self.assertEqual(
                    parser.tool_call_status(text),
                    ToolCallParser.ToolCallBufferStatus.CLOSED,
                )
                self.assertEqual(
                    parser.tool_call_status(text, final=True),
                    ToolCallParser.ToolCallBufferStatus.CLOSED,
                )
                outcome = parser.parse(text)
                self.assertEqual(len(outcome.calls), 1)
                self.assertEqual(outcome.calls[0].name, "calculator")
                self.assertEqual(outcome.calls[0].arguments, {"value": 2})
                self.assertEqual(outcome.diagnostics, [])

        quoted_only = '"<tool_call name=\\"calculator\\" arguments=\'{}\'/>"'
        self.assertEqual(
            parser.tool_call_status(quoted_only),
            ToolCallParser.ToolCallBufferStatus.NONE,
        )
        self.assertEqual(parser.parse(quoted_only), ToolCallParseOutcome())

    def test_tool_call_status_ignores_markdown_fenced_markers(self) -> None:
        parser = ToolCallParser()
        text = (
            "```xml\n"
            '<tool_call>{"name": "calculator", "arguments": {}}'
            "</tool_call>\n"
            "```"
        )

        self.assertEqual(
            parser.tool_call_status(""),
            ToolCallParser.ToolCallBufferStatus.NONE,
        )
        self.assertEqual(
            parser.tool_call_status(text),
            ToolCallParser.ToolCallBufferStatus.NONE,
        )
        self.assertEqual(parser.stream_buffer_diagnostics(text), [])

        unterminated = (
            '```xml\n<tool_call>{"name": "calculator", "arguments": {}}'
        )
        self.assertEqual(
            parser.tool_call_status(unterminated, final=True),
            ToolCallParser.ToolCallBufferStatus.NONE,
        )
        self.assertEqual(parser.stream_buffer_diagnostics(unterminated), [])

    def test_tool_call_status_detects_marker_after_markdown_fence(
        self,
    ) -> None:
        parser = ToolCallParser()
        fenced_text = "```xml\n<tool_call></tool_call>\n```\n"

        self.assertEqual(
            parser.tool_call_status(
                fenced_text
                + '<tool_call>{"name": "calculator", "arguments": {}}'
                + "</tool_call>",
                final=True,
            ),
            ToolCallParser.ToolCallBufferStatus.CLOSED,
        )

    def test_tool_end_helpers_ignore_quoted_markers(self) -> None:
        parser = ToolCallParser()

        self.assertFalse(
            parser._has_unquoted_tool_end(
                '{"text": "</tool_call>"}', ["</tool_call>"]
            )
        )
        self.assertTrue(
            parser._has_unquoted_tool_end(
                '{"text": "</tool_call>"}</tool_call>', ["</tool_call>"]
            )
        )
        text = (
            '<tool_call>{"text": "</tool_call>"}</tool_call>'
            '<tool_call name="second"/>'
        )
        second_start = text.find('<tool_call name="second"')
        self.assertEqual(
            parser._last_tool_end_before(
                text,
                second_start,
                ["</tool_call>"],
            ),
            second_start,
        )
        text_without_real_close = (
            '<tool_call>{"text": "</tool_call>"}<tool_call name="second"/>'
        )
        self.assertEqual(
            parser._last_tool_end_before(
                text_without_real_close,
                text_without_real_close.find('<tool_call name="second"'),
                ["</tool_call>"],
            ),
            -1,
        )
        two_self_closing_tags = (
            '<tool_call name="first"/>'
            "<tool_call name=\"second\" arguments='{}'/>"
        )
        self.assertFalse(
            parser._has_unclosed_tool_start_before(
                two_self_closing_tags,
                two_self_closing_tags.find('<tool_call name="second"'),
                ["<tool_call", "<tool ", "<tool>"],
                ["</tool_call>", "</tool>", "<|call|>"],
            )
        )
        open_then_self_closing = (
            '<tool_call>{"name": "first"}'
            "<tool_call name=\"second\" arguments='{}'/>"
        )
        self.assertTrue(
            parser._has_unclosed_tool_start_before(
                open_then_self_closing,
                open_then_self_closing.find('<tool_call name="second"'),
                ["<tool_call", "<tool ", "<tool>"],
                ["</tool_call>", "</tool>", "<|call|>"],
            )
        )
        self.assertFalse(
            parser._opening_tool_tag_is_self_closing("<tool>", "/>")
        )
        tag_text = '<tool_call arguments=\'{"expression": "1 > 0"}\'/>'
        self.assertEqual(
            parser._tag_end_index(tag_text, 0),
            tag_text.rindex(">"),
        )
        escaped_tag_text = "<tool_call arguments='escaped \\' > value'>"
        self.assertEqual(
            parser._tag_end_index(escaped_tag_text, 0),
            escaped_tag_text.rindex(">"),
        )
        self.assertEqual(
            parser._tag_end_index(
                '<tool_call arguments=\'{"expression": "unterminated"}',
                0,
            ),
            -1,
        )

    def test_tool_call_status_reports_dsml_states(self):
        parser = ToolCallParser(tool_format=ToolFormat.DSML)
        cases = (
            ("prefix", ToolCallParser.ToolCallBufferStatus.PREFIX),
            ("open", ToolCallParser.ToolCallBufferStatus.OPEN),
            ("closed", ToolCallParser.ToolCallBufferStatus.CLOSED),
            ("none", ToolCallParser.ToolCallBufferStatus.NONE),
        )

        for native_status, expected in cases:
            with self.subTest(native_status=native_status):
                with patch(
                    "avalan.tool.parser.DsmlTools.tool_call_buffer_status",
                    return_value=native_status,
                ):
                    self.assertEqual(
                        parser.tool_call_status("tool_calls"),
                        expected,
                    )

    def test_tool_call_status_reports_final_terminal_states(self):
        parser = ToolCallParser()

        self.assertEqual(
            parser.tool_call_status("<tool_call>", final=True),
            ToolCallParser.ToolCallBufferStatus.UNTERMINATED,
        )
        self.assertEqual(
            parser.tool_call_status("<tool_call></tool_call>", final=True),
            ToolCallParser.ToolCallBufferStatus.MALFORMED,
        )
        self.assertEqual(
            parser.tool_call_status(
                '<tool_call>{"name": "calculator", "arguments": {}}'
                "</tool_call>",
                final=True,
            ),
            ToolCallParser.ToolCallBufferStatus.CLOSED,
        )
        self.assertEqual(
            parser.tool_call_status(
                '<tool_call>{"name": "calculator", '
                '"arguments": {"text": "<tool_call name=\\"inner\\"/>"}}',
                final=True,
            ),
            ToolCallParser.ToolCallBufferStatus.UNTERMINATED,
        )

    def test_stream_buffer_diagnostics_reports_unterminated_buffer(self):
        parser = ToolCallParser()

        diagnostics = parser.stream_buffer_diagnostics(
            '<tool_call>{"name": "calculator", "arguments": {}}'
        )

        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(
            diagnostics[0].code,
            ToolCallDiagnosticCode.MALFORMED_CALL,
        )
        self.assertEqual(
            diagnostics[0].details["stream_status"], "unterminated"
        )

    def test_stream_buffer_diagnostics_reports_malformed_closed_buffer(self):
        parser = ToolCallParser()

        diagnostics = parser.stream_buffer_diagnostics(
            '<tool_call>{"name": "calculator", "arguments": }</tool_call>'
        )

        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(
            diagnostics[0].code,
            ToolCallDiagnosticCode.MALFORMED_CALL,
        )
        self.assertEqual(diagnostics[0].details["stream_status"], "malformed")

    def test_stream_buffer_diagnostics_reports_malformed_status_fallback(self):
        parser = ToolCallParser()

        with patch.object(
            ToolCallParser, "parse", return_value=ToolCallParseOutcome()
        ):
            diagnostics = parser.stream_buffer_diagnostics(
                "<tool_call></tool_call>"
            )

        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(
            diagnostics[0].code,
            ToolCallDiagnosticCode.MALFORMED_CALL,
        )
        self.assertEqual(diagnostics[0].details["stream_status"], "malformed")

    def test_stream_buffer_diagnostics_returns_parse_diagnostics(self):
        parser = ToolCallParser(tool_format=ToolFormat.JSON)

        diagnostics = parser.stream_buffer_diagnostics(
            '{"tool": "calculator", "arguments": []}'
        )

        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(
            diagnostics[0].code,
            ToolCallDiagnosticCode.MALFORMED_ARGUMENTS,
        )

    def test_stream_buffer_diagnostics_returns_empty_for_valid_buffer(self):
        parser = ToolCallParser()

        diagnostics = parser.stream_buffer_diagnostics(
            '<tool_call>{"name": "calculator", "arguments": {}}</tool_call>'
        )

        self.assertEqual(diagnostics, [])

    def test_tool_call_status_reports_harmony_closure(self):
        parser = ToolCallParser(tool_format=ToolFormat.HARMONY)

        self.assertEqual(
            parser.tool_call_status(
                "<|channel|>analysis to=functions.calculator"
                "<|message|>{}<|channel|>final<|message|>"
            ),
            ToolCallParser.ToolCallBufferStatus.CLOSED,
        )

    def test_openai_json_rejects_non_mapping_arguments(self):
        parser = ToolCallParser(tool_format=ToolFormat.OPENAI)

        self.assertIsNone(
            parser('{"name": "calculator", "arguments": ["bad"]}')
        )

    def test_tag_parser_skips_invalid_payload_before_valid_payload(self):
        parser = ToolCallParser()
        call_id = _uuid4()
        text = (
            '<tool_call>{"name": 1, "arguments": {}}</tool_call>'
            '<tool_call>{"name": "calculator", "arguments": {"value": 2}}'
            "</tool_call>"
        )

        with patch("avalan.tool.parser.uuid4", return_value=call_id):
            self.assertEqual(
                parser(text),
                [
                    ToolCall(
                        id=call_id,
                        name="calculator",
                        arguments={"value": 2},
                    )
                ],
            )

    def test_tag_parser_regex_fallback_skips_invalid_payload_shape(self):
        parser = ToolCallParser()
        text = '<tool_call>{"name": 1, "arguments": {}}</tool_call></extra>'

        self.assertIsNone(parser(text))

    def test_tag_parser_ignores_unexpected_deserialization_errors(self):
        parser = ToolCallParser()
        call_id = _uuid4()
        text = (
            '<tool_call>{"name": "broken", "arguments": {}}</tool_call>'
            '<tool_call>{"name": "calculator", "arguments": {"value": 2}}'
            "</tool_call>"
        )

        with (
            patch(
                "avalan.tool.parser.loads",
                side_effect=[
                    RuntimeError("boom"),
                    {"name": "calculator", "arguments": {"value": 2}},
                ],
            ),
            patch("avalan.tool.parser.uuid4", return_value=call_id),
        ):
            self.assertEqual(
                parser(text),
                [
                    ToolCall(
                        id=call_id,
                        name="calculator",
                        arguments={"value": 2},
                    )
                ],
            )

    def test_tag_payload_helpers_cover_xml_tool_edges(self):
        parser = ToolCallParser()

        self.assertEqual(
            parser._tag_payloads("<tool name=\"calculator\" arguments='{}'/>"),
            [{"name": "calculator", "arguments": {}}],
        )
        self.assertEqual(
            parser._tag_attribute(
                '<tool_call tool_name="bad" name="calculator"/>',
                "name",
            ),
            "calculator",
        )
        self.assertEqual(
            parser._tag_attribute(
                '<tool_call name="cal\\"culator"/>',
                "name",
            ),
            'cal\\"culator',
        )
        self.assertEqual(
            parser._tag_attribute_close_index(
                '"unterminated',
                1,
                '"',
            ),
            -1,
        )
        self.assertEqual(parser._tag_payloads("<tool />"), [])
        self.assertEqual(parser._tag_body_payloads("<tool_call"), [])
        self.assertEqual(parser._tag_body_payloads("<tool_call name"), [])
        self.assertEqual(parser._self_closing_tag_payloads("<tool_call"), [])
        self.assertEqual(
            parser._self_closing_tag_payloads("<tool_call name"),
            [],
        )

        named = ElementTree.fromstring(
            '<tool_call name="calculator">{"value": 2}</tool_call>'
        )
        self.assertEqual(
            parser._tag_payload_from_element(named),
            {"name": "calculator", "arguments": {"value": 2}},
        )

        unnamed = ElementTree.fromstring(
            '<tool_call>{"name": "calculator", "arguments": {}}</tool_call>'
        )
        with patch(
            "avalan.tool.parser.loads",
            side_effect=RuntimeError("boom"),
        ):
            self.assertIsNone(parser._tag_payload_from_element(unnamed))
            self.assertEqual(
                parser._self_closing_tag_payloads(
                    "<tool_call name=\"calculator\" arguments='{}'/>"
                ),
                [{"name": "calculator", "arguments": None}],
            )

    def test_tool_call_from_payload_rejects_invalid_inputs(self):
        self.assertIsNone(ToolCallParser._tool_call_from_payload("bad"))
        self.assertIsNone(
            ToolCallParser._tool_call_from_payload(
                {
                    "name": "calculator",
                    "arguments": [],
                }
            )
        )
        self.assertIsNone(
            ToolCallParser._tool_call_from_payload(
                {
                    "name": "math..calculator",
                    "arguments": {},
                }
            )
        )

    def test_parse_reports_diagnostic_from_invalid_list_call(self):
        parser = ToolCallParser()
        call = ToolCall(
            id="call-1",
            name="calculator",
            arguments=cast(Any, ["bad"]),
        )

        with patch.object(ToolCallParser, "__call__", return_value=[call]):
            outcome = parser.parse("ignored")

        self.assertEqual(outcome.calls, [])
        self.assertEqual(len(outcome.diagnostics), 1)
        self.assertEqual(outcome.diagnostics[0].call_id, "call-1")
        self.assertEqual(
            outcome.diagnostics[0].code,
            ToolCallDiagnosticCode.MALFORMED_ARGUMENTS,
        )

    def test_parse_reports_diagnostic_from_invalid_list_call_name(self):
        parser = ToolCallParser()
        call = ToolCall(
            id="call-1",
            name="math..calculator",
            arguments={},
        )

        with patch.object(ToolCallParser, "__call__", return_value=[call]):
            outcome = parser.parse("ignored")

        self.assertEqual(outcome.calls, [])
        self.assertEqual(len(outcome.diagnostics), 1)
        diagnostic = outcome.diagnostics[0]
        self.assertEqual(diagnostic.call_id, "call-1")
        self.assertEqual(diagnostic.requested_name, "math..calculator")
        self.assertEqual(
            diagnostic.code,
            ToolCallDiagnosticCode.MALFORMED_CALL,
        )

    def test_parse_reports_diagnostic_from_invalid_tuple_name(self):
        parser = ToolCallParser()

        with patch.object(
            ToolCallParser,
            "__call__",
            return_value=cast(Any, (" ", {})),
        ):
            outcome = parser.parse("ignored")

        self.assertEqual(outcome.calls, [])
        self.assertEqual(len(outcome.diagnostics), 1)
        self.assertEqual(
            outcome.diagnostics[0].code,
            ToolCallDiagnosticCode.MALFORMED_CALL,
        )

    def test_json_failure_diagnostics_cover_valid_and_top_level_array(self):
        parser = ToolCallParser(tool_format=ToolFormat.JSON)

        self.assertEqual(
            parser._json_failure_diagnostics(
                '{"tool": "calculator", "arguments": {}}',
                name_field="tool",
            ),
            [],
        )

        diagnostics = parser._json_failure_diagnostics(
            '["calculator"]',
            name_field="tool",
        )

        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(
            diagnostics[0].code,
            ToolCallDiagnosticCode.MALFORMED_CALL,
        )

    def test_react_failure_diagnostics_cover_absent_valid_and_invalid(self):
        parser = ToolCallParser(tool_format=ToolFormat.REACT)

        self.assertEqual(parser._react_failure_diagnostics("plain"), [])
        self.assertEqual(
            parser._react_failure_diagnostics(
                'Action: calculator\nAction Input: {"expression": "2"}'
            ),
            [],
        )
        self.assertEqual(
            ToolCallParser._decode_react_arguments(
                '{"expression": "2"}\nObservation: ok'
            ),
            ({"expression": "2"}, False),
        )

        diagnostics = parser._react_failure_diagnostics(
            'Action: calculator\nAction Input: {"expression": }'
        )

        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(
            diagnostics[0].code,
            ToolCallDiagnosticCode.MALFORMED_CALL,
        )
        self.assertEqual(diagnostics[0].requested_name, "calculator")

        for text in (
            "Action: calculator",
            'Action Input: {"expression": "2"}',
            "Action: calculator\nAction Input: ",
        ):
            with self.subTest(text=text):
                diagnostics = parser._react_failure_diagnostics(text)

                self.assertEqual(len(diagnostics), 1)
                self.assertEqual(
                    diagnostics[0].code,
                    ToolCallDiagnosticCode.MALFORMED_CALL,
                )

    def test_harmony_failure_diagnostics_cover_empty_and_invalid_messages(
        self,
    ):
        parser = ToolCallParser(tool_format=ToolFormat.HARMONY)

        self.assertEqual(
            parser._harmony_failure_diagnostics("<|channel|>commentary"),
            [],
        )
        self.assertEqual(
            parser._harmony_failure_diagnostics(
                "<|channel|>commentary to=functions.calculator"
                "<|message|><|call|>"
            ),
            [],
        )

        diagnostics = parser._harmony_failure_diagnostics(
            "<|channel|>commentary to=functions.calculator"
            '<|message|>{"expression": }<|call|>'
        )

        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(
            diagnostics[0].code,
            ToolCallDiagnosticCode.MALFORMED_CALL,
        )
        self.assertEqual(diagnostics[0].requested_name, "calculator")

    def test_tag_diagnostics_cover_xml_name_attributes(self):
        parser = ToolCallParser()

        tool_call_outcome = parser.parse(
            '<tool_call name="calculator">["bad"]</tool_call>'
        )
        self.assertEqual(tool_call_outcome.calls, [])
        self.assertEqual(len(tool_call_outcome.diagnostics), 1)
        self.assertEqual(
            tool_call_outcome.diagnostics[0].code,
            ToolCallDiagnosticCode.MALFORMED_ARGUMENTS,
        )

        tool_outcome = parser.parse('<tool name="calculator">["bad"]</tool>')
        self.assertEqual(tool_outcome.calls, [])
        self.assertEqual(len(tool_outcome.diagnostics), 1)
        self.assertEqual(
            tool_outcome.diagnostics[0].code,
            ToolCallDiagnosticCode.MALFORMED_ARGUMENTS,
        )

    def test_tag_diagnostics_cover_missing_self_closing_arguments(self):
        parser = ToolCallParser()
        text = (
            '<tool_call name="broken"/>'
            '<tool_call>{"name": "calculator", "arguments": {}}</tool_call>'
        )
        call_id = _uuid4()
        diagnostic_id = _uuid4()

        with patch(
            "avalan.tool.parser.uuid4",
            side_effect=[call_id, diagnostic_id],
        ):
            outcome = parser.parse(text)

        self.assertEqual(
            outcome.calls,
            [ToolCall(id=call_id, name="calculator", arguments={})],
        )
        self.assertEqual(len(outcome.diagnostics), 1)
        self.assertEqual(outcome.diagnostics[0].id, diagnostic_id)
        self.assertEqual(outcome.diagnostics[0].requested_name, "broken")
        self.assertEqual(
            outcome.diagnostics[0].code,
            ToolCallDiagnosticCode.MALFORMED_ARGUMENTS,
        )

    def test_tag_diagnostics_cover_malformed_self_closing_arguments(self):
        parser = ToolCallParser()
        outcome = parser.parse(
            "<tool_call name=\"calculator\" arguments='{' />"
        )

        self.assertEqual(outcome.calls, [])
        self.assertEqual(len(outcome.diagnostics), 1)
        diagnostic = outcome.diagnostics[0]
        self.assertEqual(diagnostic.requested_name, "calculator")
        self.assertEqual(
            diagnostic.code,
            ToolCallDiagnosticCode.MALFORMED_ARGUMENTS,
        )

    def test_tag_diagnostics_cover_regex_fallback_payloads(self):
        parser = ToolCallParser()

        cases = (
            (
                '<tool_call>{"name": "calculator", "arguments": '
                '["bad"]}</tool_call></extra>'
            ),
            '<tool_call name="calculator">["bad"]</tool_call></extra>',
            '<tool name="calculator">["bad"]</tool></extra>',
            '<tool_call name="calculator" arguments=\'["bad"]\'/></extra>',
        )

        for text in cases:
            with self.subTest(text=text):
                outcome = parser.parse(text)

                self.assertEqual(outcome.calls, [])
                self.assertEqual(len(outcome.diagnostics), 1)
                self.assertEqual(
                    outcome.diagnostics[0].code,
                    ToolCallDiagnosticCode.MALFORMED_ARGUMENTS,
                )

    def test_payload_diagnostic_returns_none_for_valid_payload(self):
        parser = ToolCallParser()

        self.assertIsNone(
            parser._payload_diagnostic({"name": "calculator", "arguments": {}})
        )

    def test_payload_diagnostic_rejects_invalid_name(self):
        parser = ToolCallParser()
        diagnostic = parser._payload_diagnostic({"name": " ", "arguments": {}})

        self.assertIsNotNone(diagnostic)
        assert diagnostic is not None
        self.assertEqual(
            diagnostic.code,
            ToolCallDiagnosticCode.MALFORMED_CALL,
        )

        diagnostic = parser._payload_diagnostic(
            {
                "name": "math..calculator",
                "arguments": {},
            }
        )

        self.assertIsNotNone(diagnostic)
        assert diagnostic is not None
        self.assertEqual(diagnostic.requested_name, "math..calculator")
        self.assertEqual(
            diagnostic.code,
            ToolCallDiagnosticCode.MALFORMED_CALL,
        )

    def test_payload_diagnostic_applies_resource_limits(self):
        parser = ToolCallParser(maximum_payload_size=4)

        diagnostic = parser._payload_diagnostic(
            {
                "id": "call-1",
                "name": "calculator",
                "arguments": {"expression": "large"},
            }
        )

        self.assertIsNotNone(diagnostic)
        assert diagnostic is not None
        self.assertEqual(diagnostic.call_id, "call-1")
        self.assertEqual(diagnostic.requested_name, "calculator")
        self.assertEqual(
            diagnostic.code,
            ToolCallDiagnosticCode.MAXIMUM_SIZE,
        )

    def test_resource_limit_helper_accepts_payload_within_limits(self):
        diagnostic = ToolCallParser.resource_limit_diagnostic(
            value={"items": [1, None, True, False, "ok"]},
            maximum_depth=2,
            maximum_size=32,
            stage=ToolCallDiagnosticStage.VALIDATE,
            call_id="call-1",
            requested_name="calculator",
            canonical_name="math.calculator",
        )

        self.assertIsNone(diagnostic)

    def test_resource_limit_helper_counts_empty_containers(self):
        self.assertEqual(ToolCallParser._value_depth({}), 1)
        self.assertEqual(ToolCallParser._value_depth([]), 1)

    def test_resource_limit_helper_rejects_invalid_stage(self):
        with self.assertRaises(AssertionError):
            ToolCallParser.resource_limit_diagnostic(
                value={},
                maximum_depth=None,
                maximum_size=None,
                stage=cast(Any, "validate"),
            )

    def test_resource_limit_helper_rejects_invalid_limits(self):
        invalid_cases = (
            {"maximum_depth": 0, "maximum_size": None},
            {"maximum_depth": None, "maximum_size": True},
        )

        for kwargs in invalid_cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(AssertionError):
                    ToolCallParser.resource_limit_diagnostic(
                        value={},
                        stage=ToolCallDiagnosticStage.VALIDATE,
                        **kwargs,
                    )

    def test_recovery_marker_helpers_cover_all_formats(self):
        cases = (
            (
                ToolCallRecoveryFormat.TOOL_CALL_BLOCK,
                "[TOOL_CALL]",
                "plain",
            ),
            (
                ToolCallRecoveryFormat.MINIMAX_XML,
                "<invoke",
                "plain",
            ),
            (
                ToolCallRecoveryFormat.TOOL_CODE,
                "<tool_code>",
                "plain",
            ),
            (
                ToolCallRecoveryFormat.BROAD_XML,
                "<function",
                "plain",
            ),
            (
                ToolCallRecoveryFormat.DSML_LEAKAGE,
                "<DSML:invoke",
                "invoke",
            ),
            (
                ToolCallRecoveryFormat.FENCED,
                "```json\n{}\n```",
                "plain",
            ),
        )

        for recovery_format, positive, negative in cases:
            with self.subTest(recovery_format=recovery_format):
                self.assertTrue(
                    ToolCallParser._has_recovery_marker(
                        positive, recovery_format
                    )
                )
                self.assertFalse(
                    ToolCallParser._has_recovery_marker(
                        negative, recovery_format
                    )
                )

    def test_recovery_marker_without_payload_reports_diagnostic(self):
        parser = ToolCallParser(
            recovery_formats=[ToolCallRecoveryFormat.TOOL_CALL_BLOCK]
        )

        outcome = parser.parse("[TOOL_CALL]{bad")

        self.assertEqual(outcome.calls, [])
        self.assertEqual(len(outcome.diagnostics), 1)
        self.assertEqual(
            outcome.diagnostics[0].details["source_format"],
            ToolCallRecoveryFormat.TOOL_CALL_BLOCK.value,
        )

    def test_recovery_payloads_skip_duplicate_formats(self):
        parser = ToolCallParser(
            recovery_formats=[
                ToolCallRecoveryFormat.MINIMAX_XML,
                ToolCallRecoveryFormat.BROAD_XML,
            ]
        )
        text = (
            '<invoke name="calculator"><parameter name="expression">'
            "1 + 1</parameter></invoke>"
        )

        recovered = parser._recovery_payloads(text)

        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0].payload["name"], "calculator")

    def test_recovery_payloads_skip_overlapping_duplicate_payloads(self):
        parser = ToolCallParser(
            recovery_formats=[
                ToolCallRecoveryFormat.MINIMAX_XML,
                ToolCallRecoveryFormat.FENCED,
            ]
        )
        text = (
            "```xml\n"
            '<invoke name="calculator"><parameter name="expression">'
            "1 + 1</parameter></invoke>\n"
            "```"
        )

        recovered = parser._recovery_payloads(text)

        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0].payload["name"], "calculator")

    def test_recovery_keeps_non_overlapping_duplicate_payloads(self):
        first_id = _uuid4()
        second_id = _uuid4()
        parser = ToolCallParser(
            recovery_formats=[ToolCallRecoveryFormat.MINIMAX_XML]
        )
        text = (
            '<invoke name="calculator"><parameter name="expression">'
            "1 + 1</parameter></invoke>"
            '<invoke name="calculator"><parameter name="expression">'
            "1 + 1</parameter></invoke>"
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
                    arguments={"expression": "1 + 1"},
                ),
                ToolCall(
                    id=second_id,
                    name="calculator",
                    arguments={"expression": "1 + 1"},
                ),
            ],
        )
        self.assertEqual(outcome.diagnostics, [])

    def test_fenced_recovery_accepts_direct_json_payload(self):
        call_id = _uuid4()
        parser = ToolCallParser(
            recovery_formats=[ToolCallRecoveryFormat.FENCED]
        )

        with patch("avalan.tool.parser.uuid4", return_value=call_id):
            outcome = parser.parse(
                '```json\n{"name": "calculator", "arguments": {}}\n```'
            )

        self.assertEqual(
            outcome.calls,
            [ToolCall(id=call_id, name="calculator", arguments={})],
        )
        self.assertEqual(outcome.diagnostics, [])

    def test_xml_recovery_helper_rejects_malformed_xml(self):
        parser = ToolCallParser()

        self.assertIsNone(parser._xml_payload("<invoke>"))
        self.assertIsNone(parser._xml_payload("<unknown></unknown>"))

    def test_xml_recovery_parses_tool_call_payload_shapes(self):
        parser = ToolCallParser()

        self.assertEqual(
            parser._xml_payload(
                '<tool_call>{"name": "calculator", "arguments": {}}'
                "</tool_call>"
            ),
            {"name": "calculator", "arguments": {}},
        )
        self.assertEqual(
            parser._xml_payload(
                '<tool_call name="calculator">{"expression": "2"}</tool_call>'
            ),
            {"name": "calculator", "arguments": {"expression": "2"}},
        )
        self.assertEqual(
            parser._xml_payload(
                "<tool_call><name>calculator</name><arguments>{}"
                "</arguments></tool_call>"
            ),
            {"name": "calculator", "arguments": {}},
        )
        self.assertIsNone(
            parser._xml_payload(
                "<tool_call><arguments>{}</arguments></tool_call>"
            )
        )
        self.assertIsNone(
            parser._xml_payload(
                "<tool_call><name>calculator</name><arguments>[]"
                "</arguments></tool_call>"
            )
        )
        self.assertIsNone(
            parser._xml_payload(
                "<tool_call name='calculator'><parameter>bad</parameter>"
                "</tool_call>"
            )
        )

    def test_xml_recovery_rejects_invalid_named_payloads(self):
        parser = ToolCallParser()

        self.assertIsNone(
            parser._xml_payload(
                "<invoke><parameter name='value'>1</parameter></invoke>"
            )
        )
        self.assertIsNone(
            parser._xml_payload(
                "<invoke name='calculator'><parameter>1</parameter></invoke>"
            )
        )

    def test_xml_recovery_handles_parameters_and_non_parameters(self):
        parser = ToolCallParser()

        payload = parser._xml_payload(
            "<invoke name='calculator'>"
            "<description>ignored</description>"
            "<parameter name='expression'>2 + 2</parameter>"
            "<parameter name='precision' string='false'>2</parameter>"
            "<parameter name='fallback' string='false'>not-json</parameter>"
            "</invoke>"
        )

        self.assertEqual(
            payload,
            {
                "name": "calculator",
                "arguments": {
                    "expression": "2 + 2",
                    "precision": 2,
                    "fallback": "not-json",
                },
            },
        )

    def test_function_call_payload_rejects_non_object_arguments(self):
        parser = ToolCallParser()

        self.assertIsNone(parser._function_call_payload("calculator({1})"))


if __name__ == "__main__":
    main()
