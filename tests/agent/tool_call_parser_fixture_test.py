from json import loads
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from avalan.entities import (
    ToolCallDiagnosticCode,
    ToolFormat,
    ToolManagerSettings,
)
from avalan.model.response.parsers.tool import ToolCallResponseParser
from avalan.model.stream import StreamItemKind, StreamProviderEvent
from avalan.tool.manager import ToolManager

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "tool_parsing"


def read_json_fixture(name: str) -> Any:
    return loads((FIXTURES / name).read_text(encoding="utf-8"))


class ToolCallParserFixtureTestCase(IsolatedAsyncioTestCase):
    @staticmethod
    def _answer_texts(items: list[Any]) -> list[str | None]:
        return [
            item.text_delta
            for item in items
            if (
                isinstance(item, StreamProviderEvent)
                and item.kind is StreamItemKind.ANSWER_DELTA
            )
        ]

    @staticmethod
    def _provider_events(
        items: list[Any],
        kind: StreamItemKind,
    ) -> list[StreamProviderEvent]:
        return [
            item
            for item in items
            if isinstance(item, StreamProviderEvent) and item.kind is kind
        ]

    async def test_marker_stream_boundaries(self) -> None:
        cases = read_json_fixture("stream_marker_boundaries.json")

        for case in cases:
            with self.subTest(case=case["name"]):
                manager = ToolManager.create_instance(enable_tools=[])
                parser = ToolCallResponseParser(
                    manager, None, legacy_fixture=True
                )
                items: list[Any] = []

                for token in case["tokens"]:
                    items.extend(await parser.push(token))
                if case.get("flush"):
                    items.extend(await parser.flush())

                event_name = case["event"]
                if event_name is None:
                    self.assertEqual(
                        self._answer_texts(items),
                        case["strings"],
                    )
                    continue

                self.assertEqual(
                    self._answer_texts(items),
                    case.get("strings", []),
                )

                if event_name == "tool_process":
                    ready = self._provider_events(
                        items,
                        StreamItemKind.TOOL_CALL_READY,
                    )[0]
                    assert isinstance(ready.data, dict)
                    self.assertEqual(ready.data["name"], case["call"]["name"])
                    self.assertEqual(
                        ready.data["arguments"],
                        case["call"]["arguments"],
                    )
                    continue

                diagnostic = self._provider_events(
                    items,
                    StreamItemKind.STREAM_DIAGNOSTIC,
                )[0]
                assert isinstance(diagnostic.data, dict)
                diagnostics = diagnostic.data["diagnostics"]
                self.assertEqual(len(diagnostics), 1)
                self.assertEqual(
                    diagnostics[0]["code"],
                    ToolCallDiagnosticCode[case["diagnostic_code"]].value,
                )
                if "stream_status" in case:
                    self.assertEqual(
                        diagnostics[0]["details"]["stream_status"],
                        case["stream_status"],
                    )

    async def test_harmony_stream_suppresses_call_bytes(self) -> None:
        fixture = read_json_fixture("stream_harmony_split_call.json")
        manager = ToolManager.create_instance(
            enable_tools=[],
            settings=ToolManagerSettings(tool_format=ToolFormat.HARMONY),
        )
        parser = ToolCallResponseParser(manager, None, legacy_fixture=True)
        items: list[Any] = []

        for token in fixture["tokens"]:
            items.extend(await parser.push(token))

        ready = self._provider_events(
            items,
            StreamItemKind.TOOL_CALL_READY,
        )[0]
        assert isinstance(ready.data, dict)

        self.assertEqual(ready.data["name"], fixture["call"]["name"])
        self.assertEqual(ready.data["arguments"], fixture["call"]["arguments"])
        self.assertEqual(
            self._answer_texts(items),
            fixture["strings"],
        )
        self.assertTrue(
            self._provider_events(
                items, StreamItemKind.TOOL_CALL_ARGUMENT_DELTA
            )
        )
