from typing import Any, cast
from unittest import TestCase, main
from unittest.mock import patch
from uuid import uuid4 as _uuid4

from avalan.entities import (
    Message,
    MessageContentText,
    MessageRole,
    ToolCall,
    ToolCallDiagnosticCode,
    ToolCallDiagnosticStage,
    ToolFormat,
)
from avalan.tool.parser import ToolCallParser


class ToolCallParserFullCoverageTestCase(TestCase):
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

    def test_message_tool_calls_returns_empty_without_configured_parser(self):
        parser = ToolCallParser()

        self.assertEqual(parser.message_tool_calls("plain text"), [])

    def test_message_tool_calls_returns_empty_for_unexpected_shape(self):
        parser = ToolCallParser(tool_format=ToolFormat.JSON)

        with patch.object(
            ToolCallParser,
            "__call__",
            return_value=("calculator", {}, "extra"),
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
        self.assertEqual(
            parser.tool_call_status("<too"),
            ToolCallParser.ToolCallBufferStatus.PREFIX,
        )
        self.assertEqual(
            parser.tool_call_status('<tool_call name="calculator"'),
            ToolCallParser.ToolCallBufferStatus.OPEN,
        )
        self.assertEqual(
            parser.tool_call_status("<tool_call></tool_call>"),
            ToolCallParser.ToolCallBufferStatus.CLOSED,
        )

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

        diagnostics = parser._react_failure_diagnostics(
            'Action: calculator\nAction Input: {"expression": }'
        )

        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(
            diagnostics[0].code,
            ToolCallDiagnosticCode.MALFORMED_CALL,
        )
        self.assertEqual(diagnostics[0].requested_name, "calculator")

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


if __name__ == "__main__":
    main()
