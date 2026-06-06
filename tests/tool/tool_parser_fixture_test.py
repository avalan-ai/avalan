from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch
from uuid import uuid4 as _uuid4

from avalan.entities import (
    ToolCall,
    ToolCallDiagnosticCode,
    ToolCallDiagnosticStage,
    ToolFormat,
)
from avalan.tool.parser import ToolCallParser

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "tool_parsing"


def read_fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


class ToolCallParserFixtureTestCase(TestCase):
    def test_malformed_envelope_reports_parse_diagnostic(self):
        outcome = ToolCallParser().parse(
            read_fixture("tag_malformed_envelope.txt")
        )

        self.assertEqual(outcome.calls, [])
        self.assertEqual(len(outcome.diagnostics), 1)
        diagnostic = outcome.diagnostics[0]
        self.assertEqual(
            diagnostic.code,
            ToolCallDiagnosticCode.MALFORMED_CALL,
        )
        self.assertEqual(diagnostic.stage, ToolCallDiagnosticStage.PARSE)

    def test_malformed_tag_before_valid_call_keeps_valid_call(self):
        call_id = _uuid4()

        with patch("avalan.tool.parser.uuid4", return_value=call_id):
            outcome = ToolCallParser().parse(
                read_fixture("tag_malformed_then_valid.txt")
            )

        self.assertEqual(
            outcome.calls,
            [
                ToolCall(
                    id=call_id,
                    name="calculator",
                    arguments={"expression": "2 + 2"},
                )
            ],
        )
        self.assertEqual(len(outcome.diagnostics), 1)
        self.assertEqual(
            outcome.diagnostics[0].code,
            ToolCallDiagnosticCode.MALFORMED_CALL,
        )

    def test_malformed_harmony_before_valid_call_keeps_valid_call(self):
        call_id = _uuid4()

        with patch("avalan.tool.parser.uuid4", return_value=call_id):
            outcome = ToolCallParser(tool_format=ToolFormat.HARMONY).parse(
                read_fixture("harmony_malformed_then_valid.txt")
            )

        self.assertEqual(
            outcome.calls,
            [
                ToolCall(
                    id=call_id,
                    name="calculator",
                    arguments={"expression": "4 + 4"},
                )
            ],
        )
        self.assertEqual(len(outcome.diagnostics), 1)
        diagnostic = outcome.diagnostics[0]
        self.assertEqual(diagnostic.requested_name, "broken")
        self.assertEqual(
            diagnostic.code,
            ToolCallDiagnosticCode.MALFORMED_CALL,
        )

    def test_adjacent_calls_preserve_order(self):
        first_id = _uuid4()
        second_id = _uuid4()

        with patch(
            "avalan.tool.parser.uuid4",
            side_effect=[first_id, second_id],
        ):
            outcome = ToolCallParser().parse(
                read_fixture("tag_adjacent_calls.txt")
            )

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
                    name="database.run",
                    arguments={"sql": "SELECT 1"},
                ),
            ],
        )
        self.assertEqual(outcome.diagnostics, [])

    def test_fenced_tool_call_example_does_not_parse_by_default(self):
        outcome = ToolCallParser().parse(
            read_fixture("markdown_fenced_tool_call_example.txt")
        )

        self.assertEqual(outcome.calls, [])
        self.assertEqual(outcome.diagnostics, [])

    def test_fenced_malformed_example_does_not_report_diagnostic(self):
        outcome = ToolCallParser().parse(
            read_fixture("markdown_fenced_malformed_example.txt")
        )

        self.assertEqual(outcome.calls, [])
        self.assertEqual(outcome.diagnostics, [])


if __name__ == "__main__":
    main()
