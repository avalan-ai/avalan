from json import loads
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from avalan.entities import (
    ToolCallDiagnosticCode,
    ToolCallToken,
    ToolFormat,
    ToolManagerSettings,
)
from avalan.event import EventType
from avalan.model.response.parsers.tool import ToolCallResponseParser
from avalan.tool.manager import ToolManager

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "tool_parsing"


def read_json_fixture(name: str) -> Any:
    return loads((FIXTURES / name).read_text(encoding="utf-8"))


class ToolCallParserFixtureTestCase(IsolatedAsyncioTestCase):
    async def test_marker_stream_boundaries(self) -> None:
        cases = read_json_fixture("stream_marker_boundaries.json")

        for case in cases:
            with self.subTest(case=case["name"]):
                manager = ToolManager.create_instance(enable_tools=[])
                parser = ToolCallResponseParser(manager, None)
                items: list[Any] = []

                for token in case["tokens"]:
                    items.extend(await parser.push(token))
                if case.get("flush"):
                    items.extend(await parser.flush())

                event_name = case["event"]
                if event_name is None:
                    self.assertEqual(items, case["strings"])
                    continue

                event_type = (
                    EventType.TOOL_PROCESS
                    if event_name == "tool_process"
                    else EventType.TOOL_DIAGNOSTIC
                )
                event = next(
                    item
                    for item in items
                    if getattr(item, "type", None) is event_type
                )
                self.assertEqual(event.type, event_type)

                strings = [item for item in items if isinstance(item, str)]
                self.assertEqual(strings, case.get("strings", []))

                if event_type is EventType.TOOL_PROCESS:
                    call = event.payload[0]
                    self.assertEqual(call.name, case["call"]["name"])
                    self.assertEqual(
                        call.arguments,
                        case["call"]["arguments"],
                    )
                    continue

                diagnostics = event.payload["diagnostics"]
                self.assertEqual(len(diagnostics), 1)
                self.assertEqual(
                    diagnostics[0].code,
                    ToolCallDiagnosticCode[case["diagnostic_code"]],
                )
                if "stream_status" in case:
                    self.assertEqual(
                        diagnostics[0].details["stream_status"],
                        case["stream_status"],
                    )

    async def test_harmony_stream_suppresses_call_bytes(self) -> None:
        fixture = read_json_fixture("stream_harmony_split_call.json")
        manager = ToolManager.create_instance(
            enable_tools=[],
            settings=ToolManagerSettings(tool_format=ToolFormat.HARMONY),
        )
        parser = ToolCallResponseParser(manager, None)
        items: list[Any] = []

        for token in fixture["tokens"]:
            items.extend(await parser.push(token))

        event = next(
            item
            for item in items
            if getattr(item, "type", None) is EventType.TOOL_PROCESS
        )
        call = event.payload[0]

        self.assertEqual(call.name, fixture["call"]["name"])
        self.assertEqual(call.arguments, fixture["call"]["arguments"])
        self.assertEqual(
            [item for item in items if isinstance(item, str)],
            fixture["strings"],
        )
        self.assertTrue(any(isinstance(item, ToolCallToken) for item in items))
