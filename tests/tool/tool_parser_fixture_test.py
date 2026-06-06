from json import loads
from pathlib import Path
from typing import cast
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

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "tool_parsing"


def read_fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def read_case_fixture(name: str) -> dict[str, str]:
    data = loads(read_fixture(name))
    assert isinstance(data, dict)
    for key, value in data.items():
        assert isinstance(key, str)
        assert isinstance(value, str)
    return cast(dict[str, str], data)


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

    def test_recovery_depth_limits_apply_to_all_formats(self):
        cases = read_case_fixture("recovery_nested_payloads.json")
        self.assertEqual(
            set(cases),
            {
                recovery_format.value
                for recovery_format in ToolCallRecoveryFormat
            },
        )

        for recovery_format in ToolCallRecoveryFormat:
            with self.subTest(recovery_format=recovery_format):
                parser = ToolCallParser(
                    recovery_formats=[recovery_format],
                    maximum_payload_depth=2,
                )

                outcome = parser.parse(cases[recovery_format.value])

                self.assertEqual(outcome.calls, [])
                self.assertEqual(len(outcome.diagnostics), 1)
                diagnostic = outcome.diagnostics[0]
                self.assertEqual(
                    diagnostic.code,
                    ToolCallDiagnosticCode.MAXIMUM_DEPTH,
                )
                self.assertEqual(
                    diagnostic.stage,
                    ToolCallDiagnosticStage.PARSE,
                )
                self.assertEqual(diagnostic.requested_name, "calculator")
                self.assertEqual(diagnostic.details["limit"], 2)

    def test_recovery_size_limits_apply_to_all_formats(self):
        cases = read_case_fixture("recovery_large_payloads.json")
        self.assertEqual(
            set(cases),
            {
                recovery_format.value
                for recovery_format in ToolCallRecoveryFormat
            },
        )

        for recovery_format in ToolCallRecoveryFormat:
            with self.subTest(recovery_format=recovery_format):
                parser = ToolCallParser(
                    recovery_formats=[recovery_format],
                    maximum_payload_size=12,
                )

                outcome = parser.parse(cases[recovery_format.value])

                self.assertEqual(outcome.calls, [])
                self.assertEqual(len(outcome.diagnostics), 1)
                diagnostic = outcome.diagnostics[0]
                self.assertEqual(
                    diagnostic.code,
                    ToolCallDiagnosticCode.MAXIMUM_SIZE,
                )
                self.assertEqual(
                    diagnostic.stage,
                    ToolCallDiagnosticStage.PARSE,
                )
                self.assertEqual(diagnostic.requested_name, "calculator")
                self.assertEqual(diagnostic.details["limit"], 12)

    def test_malformed_recovery_segments_keep_later_valid_calls(self):
        parser = ToolCallParser(
            recovery_formats=[
                ToolCallRecoveryFormat.TOOL_CALL_BLOCK,
                ToolCallRecoveryFormat.TOOL_CODE,
                ToolCallRecoveryFormat.MINIMAX_XML,
                ToolCallRecoveryFormat.FENCED,
            ]
        )

        outcome = parser.parse(
            read_fixture("recovery_malformed_segments_then_valid.txt")
        )

        self.assertEqual(
            [(call.name, call.arguments) for call in outcome.calls],
            [
                ("calculator", {"expression": "1 + 1"}),
                ("database.run", {"sql": "SELECT 1"}),
                ("search", {"query": "avalan docs"}),
                ("browser.open", {"url": "https://example.invalid"}),
            ],
        )
        self.assertEqual(
            [diagnostic.code for diagnostic in outcome.diagnostics],
            [ToolCallDiagnosticCode.MALFORMED_CALL] * 4,
        )
        self.assertEqual(
            [
                diagnostic.details["source_format"]
                for diagnostic in outcome.diagnostics
            ],
            [
                ToolCallRecoveryFormat.TOOL_CALL_BLOCK.value,
                ToolCallRecoveryFormat.TOOL_CODE.value,
                ToolCallRecoveryFormat.MINIMAX_XML.value,
                ToolCallRecoveryFormat.FENCED.value,
            ],
        )

    def test_fenced_code_example_does_not_parse_by_default(self):
        outcome = ToolCallParser().parse(
            read_fixture("recovery_fenced_code_example.txt")
        )

        self.assertEqual(outcome.calls, [])
        self.assertEqual(outcome.diagnostics, [])

    def test_fenced_code_recovery_fails_closed(self):
        outcome = ToolCallParser(
            recovery_formats=[ToolCallRecoveryFormat.FENCED]
        ).parse(read_fixture("recovery_fenced_code_example.txt"))

        self.assertEqual(outcome.calls, [])
        self.assertEqual(len(outcome.diagnostics), 1)
        diagnostic = outcome.diagnostics[0]
        self.assertEqual(
            diagnostic.code,
            ToolCallDiagnosticCode.MALFORMED_CALL,
        )
        self.assertEqual(
            diagnostic.details["source_format"],
            ToolCallRecoveryFormat.FENCED.value,
        )


if __name__ == "__main__":
    main()
