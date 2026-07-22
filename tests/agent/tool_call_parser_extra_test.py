from collections.abc import Iterable
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

from avalan.entities import (
    ToolCall,
    ToolCallDiagnostic,
    ToolCallDiagnosticCode,
    ToolCallDiagnosticStage,
    ToolCallParseOutcome,
    ToolCallToken,
    ToolFormat,
    ToolManagerSettings,
)
from avalan.event import Event
from avalan.model.capability import ModelCapabilityCatalog
from avalan.model.response.parsers.tool import (
    ToolCallResponseParser,
    _MarkdownFenceState,
    _VisibleQuoteState,
    _VisibleTextState,
)
from avalan.model.stream import StreamItemKind, StreamProviderEvent
from avalan.tool.manager import ToolManager
from avalan.tool.parser import ToolCallParser

_PARSER_CAPABILITY_NAMES = (
    "calc",
    "calculator",
    "browser.open",
    "call",
    "expression",
    "first",
    "functions.browser.open",
    "functions.db.run",
    "db.run",
    "inner",
    "math.calculator",
    "second",
    "x",
)


def _catalog(
    source: ToolManager | ModelCapabilityCatalog,
) -> ModelCapabilityCatalog:
    if isinstance(source, ModelCapabilityCatalog):
        return source
    assert isinstance(source, ToolManager)

    seed = source.export_model_capability_seed()
    descriptors = list(seed["descriptors"])
    existing_names = {
        descriptor["canonical_name"]
        for descriptor in descriptors
        if isinstance(descriptor, dict)
    }
    descriptors.extend(
        {
            "canonical_name": name,
            "description": f"Invoke {name}.",
            "aliases": [],
            "parameter_schema": {
                "type": "object",
                "additionalProperties": True,
            },
            "result_schema": None,
        }
        for name in _PARSER_CAPABILITY_NAMES
        if name not in existing_names
    )
    seed["descriptors"] = descriptors
    catalog = ModelCapabilityCatalog.create(seed)
    if type(source) is ToolManager:
        return catalog

    capability = MagicMock(spec=ModelCapabilityCatalog)
    capability.tool_format = catalog.tool_format
    capability.recovery_formats = catalog.recovery_formats
    capability.is_potential_tool_call.side_effect = (
        catalog.is_potential_tool_call
    )
    capability.tool_call_status.side_effect = catalog.tool_call_status
    if isinstance(source, CapturingToolManager):

        def parse_calls(text: str) -> ToolCallParseOutcome:
            source.parsed_texts.append(text)
            return catalog.parse_calls(text)

        capability.parse_calls.side_effect = parse_calls
        capability.stream_buffer_diagnostics.side_effect = (
            catalog.stream_buffer_diagnostics
        )
    else:
        capability.parse_calls.side_effect = source.parse_calls
        capability.stream_buffer_diagnostics.side_effect = (
            source.stream_buffer_diagnostics
        )
    return cast(ModelCapabilityCatalog, capability)


def _canonical_items(items: Iterable[Any]) -> list[StreamProviderEvent]:
    events = list(items)
    assert all(isinstance(item, StreamProviderEvent) for item in events)
    return events


def _events(
    items: Iterable[Any],
    kind: StreamItemKind,
) -> list[StreamProviderEvent]:
    return [item for item in _canonical_items(items) if item.kind is kind]


def _answer_texts(items: Iterable[Any]) -> list[str | None]:
    return [
        item.text_delta for item in _events(items, StreamItemKind.ANSWER_DELTA)
    ]


def _argument_texts(items: Iterable[Any]) -> list[str | None]:
    return [
        item.text_delta
        for item in _events(items, StreamItemKind.TOOL_CALL_ARGUMENT_DELTA)
    ]


def _ready_events(items: Iterable[Any]) -> list[StreamProviderEvent]:
    return _events(items, StreamItemKind.TOOL_CALL_READY)


def _ready_data(item: StreamProviderEvent) -> dict[str, Any]:
    assert item.kind is StreamItemKind.TOOL_CALL_READY
    assert isinstance(item.data, dict)
    return item.data


def _ready_names(items: Iterable[Any]) -> list[str | None]:
    return [
        data.get("name") if isinstance(data.get("name"), str) else None
        for data in (_ready_data(item) for item in _ready_events(items))
    ]


def _ready_arguments(items: Iterable[Any]) -> list[Any]:
    return [
        _ready_data(item).get("arguments") for item in _ready_events(items)
    ]


def _diagnostic_events(items: Iterable[Any]) -> list[StreamProviderEvent]:
    return _events(items, StreamItemKind.STREAM_DIAGNOSTIC)


def _diagnostic_data(item: StreamProviderEvent) -> list[dict[str, Any]]:
    assert item.kind is StreamItemKind.STREAM_DIAGNOSTIC
    assert isinstance(item.data, dict)
    diagnostics = item.data.get("diagnostics")
    assert isinstance(diagnostics, list)
    assert all(isinstance(diagnostic, dict) for diagnostic in diagnostics)
    return diagnostics


def _first_ready_data(items: Iterable[Any]) -> dict[str, Any]:
    return _ready_data(_ready_events(items)[0])


def _first_diagnostic_data(items: Iterable[Any]) -> list[dict[str, Any]]:
    return _diagnostic_data(_diagnostic_events(items)[0])


class DiagnosticFallbackToolManager(ToolManager):
    def parse_calls(self, text: str) -> ToolCallParseOutcome:
        return ToolCallParseOutcome()

    def stream_buffer_diagnostics(
        self, buffer: str
    ) -> list[ToolCallDiagnostic]:
        return [
            ToolCallDiagnostic(
                id="diagnostic-1",
                code=ToolCallDiagnosticCode.MALFORMED_CALL,
                stage=ToolCallDiagnosticStage.PARSE,
                message="Tool call stream is malformed.",
            )
        ]


class NoDiagnosticToolManager(ToolManager):
    def parse_calls(self, text: str) -> ToolCallParseOutcome:
        return ToolCallParseOutcome()

    def stream_buffer_diagnostics(
        self, buffer: str
    ) -> list[ToolCallDiagnostic]:
        return []


class CapturingToolManager(ToolManager):
    parsed_texts: list[str]


class ToolCallParserExtraTestCase(IsolatedAsyncioTestCase):
    async def test_tag_buffer_trim_no_check(self):
        manager = MagicMock(spec=ModelCapabilityCatalog)
        manager.is_potential_tool_call.return_value = False
        manager.tool_call_status.return_value = (
            ToolCallParser.ToolCallBufferStatus.NONE
        )
        parser = ToolCallResponseParser(_catalog(manager), None)

        long_token = "a" * 65
        result = await parser.push(long_token)

        self.assertEqual(_answer_texts(result), [long_token])
        self.assertEqual(len(parser._tag_buffer), 64)

    async def test_trigger_and_event(self):
        manager = MagicMock(spec=ModelCapabilityCatalog)
        manager.is_potential_tool_call.return_value = True
        manager.parse_calls.return_value = ToolCallParseOutcome(
            calls=[ToolCall(id=None, name="")]
        )
        base_parser = ToolCallParser()
        manager.tool_call_status.side_effect = base_parser.tool_call_status
        event_manager = MagicMock()
        event_manager.trigger = AsyncMock()

        parser = ToolCallResponseParser(_catalog(manager), event_manager)
        items = await parser.push("<tool_call></tool_call>")

        self.assertEqual(
            [item.kind for item in _canonical_items(items)],
            [StreamItemKind.TOOL_CALL_READY, StreamItemKind.TOOL_CALL_DONE],
        )
        event_manager.trigger.assert_not_awaited()
        manager.parse_calls.assert_called_once_with("<tool_call></tool_call>")
        self.assertEqual(parser._buffer.getvalue(), "")
        self.assertFalse(parser._inside_call)

        self.assertEqual(await parser.flush(), [])

    async def test_trigger_without_calls(self):
        manager = MagicMock(spec=ModelCapabilityCatalog)
        manager.is_potential_tool_call.return_value = True
        manager.parse_calls.return_value = ToolCallParseOutcome()
        manager.tool_call_status.return_value = (
            ToolCallParser.ToolCallBufferStatus.NONE
        )
        event_manager = MagicMock()
        event_manager.trigger = AsyncMock()

        parser = ToolCallResponseParser(_catalog(manager), event_manager)
        items = await parser.push("no_call")

        self.assertEqual(_answer_texts(items), ["no_call"])
        event_manager.trigger.assert_not_awaited()
        manager.parse_calls.assert_called_once_with("no_call")

    async def test_self_closing_tag(self):
        manager = MagicMock(spec=ModelCapabilityCatalog)
        manager.is_potential_tool_call.return_value = True
        manager.parse_calls.return_value = ToolCallParseOutcome(
            calls=[ToolCall(id=None, name="")]
        )
        base_parser = ToolCallParser()
        manager.tool_call_status.side_effect = base_parser.tool_call_status

        parser = ToolCallResponseParser(_catalog(manager), None)
        items = await parser.push('<tool_call name="calc"/>')

        self.assertEqual(
            [item.kind for item in _canonical_items(items)],
            [StreamItemKind.TOOL_CALL_READY, StreamItemKind.TOOL_CALL_DONE],
        )
        manager.parse_calls.assert_called_once_with('<tool_call name="calc"/>')
        self.assertFalse(parser._inside_call)
        self.assertEqual(parser._buffer.getvalue(), "")

    async def test_self_closing_tag_with_gt_attribute_value(self) -> None:
        manager = ToolManager.create_instance(enable_tools=[])
        parser = ToolCallResponseParser(_catalog(manager), None)

        items = await parser.push(
            '<tool_call name="math.calculator" arguments=\'{"expression": '
            '"1 > 0"}\'/> tail'
        )

        self.assertEqual(
            _argument_texts(items),
            ['{"expression":"1 > 0"}'],
        )
        ready_data = _first_ready_data(items)
        self.assertEqual(ready_data["name"], "math.calculator")
        self.assertEqual(
            ready_data["arguments"],
            {"expression": "1 > 0"},
        )
        self.assertEqual(_answer_texts(items), [" tail"])
        self.assertFalse(parser._inside_call)
        self.assertEqual(parser._buffer.getvalue(), " tail")

    async def test_self_closing_tag_with_inner_attribute_quote(self) -> None:
        manager = ToolManager.create_instance(enable_tools=[])
        parser = ToolCallResponseParser(_catalog(manager), None)

        items = await parser.push(
            '<tool_call name="math.calculator" arguments=\'{"phrase": '
            '"rock \', roll", "expression": "1 > 0"}\'/> tail'
        )

        self.assertEqual(
            _argument_texts(items),
            ['{"phrase":"rock \', roll","expression":"1 > 0"}'],
        )
        ready_data = _first_ready_data(items)
        self.assertEqual(ready_data["name"], "math.calculator")
        self.assertEqual(
            ready_data["arguments"],
            {"phrase": "rock ', roll", "expression": "1 > 0"},
        )
        self.assertEqual(_answer_texts(items), [" tail"])
        self.assertFalse(parser._inside_call)
        self.assertEqual(parser._buffer.getvalue(), " tail")

    async def test_quoted_self_close_marker_keeps_tag_open(self) -> None:
        manager = ToolManager.create_instance(enable_tools=[])
        parser = ToolCallResponseParser(_catalog(manager), None)

        pushed = await parser.push(
            '<tool_call name="math.calculator" arguments=\'{"expression": '
            '"/>"}\'>'
        )
        flushed = await parser.flush()

        self.assertEqual(pushed, [])
        self.assertGreaterEqual(len(flushed), 1)
        diagnostics = _diagnostic_data(_diagnostic_events(flushed)[-1])
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(
            diagnostics[0]["code"],
            ToolCallDiagnosticCode.MALFORMED_CALL.value,
        )
        self.assertEqual(
            diagnostics[0]["details"]["stream_status"], "unterminated"
        )
        self.assertFalse(parser._inside_call)

    async def test_harmony_long_call_followed_by_final_channel(self):
        manager = MagicMock(spec=ModelCapabilityCatalog)
        manager.is_potential_tool_call.return_value = True

        def _parse_calls(text: str) -> ToolCallParseOutcome:
            return ToolCallParseOutcome(
                calls=(
                    [ToolCall(id=None, name="", arguments={})]
                    if "<|call|>" in text
                    else []
                )
            )

        manager.parse_calls.side_effect = _parse_calls
        manager.tool_format = ToolFormat.HARMONY
        base_parser = ToolCallParser(tool_format=ToolFormat.HARMONY)
        manager.tool_call_status.side_effect = base_parser.tool_call_status

        parser = ToolCallResponseParser(_catalog(manager), None)
        long_content = "x" * 100
        parts = [
            "<|channel|>commentary to=functions.db.run code<|message|>{"
            + long_content,
            "}<|call|>",
            "<|channel|>final<|message|>",
            "done",
        ]
        tokens: list = []
        for part in parts:
            tokens.extend(await parser.push(part))

        self.assertTrue(_ready_events(tokens))
        self.assertEqual(
            _answer_texts(tokens),
            ["<|channel|>final<|message|>", "done"],
        )
        self.assertFalse(parser._inside_call)

    async def test_dsml_streaming_emits_tool_call_tokens_and_event(self):
        manager = MagicMock(spec=ModelCapabilityCatalog)
        manager.is_potential_tool_call.side_effect = lambda _buffer, token: (
            bool(token.strip())
        )
        manager.tool_format = ToolFormat.DSML
        base_parser = ToolCallParser(tool_format=ToolFormat.DSML)
        manager.tool_call_status.side_effect = base_parser.tool_call_status
        manager.parse_calls.side_effect = base_parser.parse

        parser = ToolCallResponseParser(_catalog(manager), None)
        tokens: list = []
        for part in (
            "<｜DSML｜tool_calls>",
            '<｜DSML｜invoke name="math.calculator">',
            '<｜DSML｜parameter name="expression">2 + 2',
            "</｜DSML｜parameter>",
            "</｜DSML｜invoke>",
            "</｜DSML｜tool_calls>",
        ):
            tokens.extend(await parser.push(part))

        self.assertEqual(_argument_texts(tokens), ["2 + 2"])
        self.assertTrue(_ready_events(tokens))
        self.assertFalse(parser._inside_call)

    async def test_harmony_flush_emits_event_for_missing_call_closure(self):
        manager = MagicMock(spec=ModelCapabilityCatalog)
        manager.is_potential_tool_call.return_value = True
        manager.tool_format = ToolFormat.HARMONY
        base_parser = ToolCallParser(tool_format=ToolFormat.HARMONY)
        manager.tool_call_status.side_effect = base_parser.tool_call_status
        manager.parse_calls.side_effect = base_parser.parse
        event_manager = MagicMock()
        event_manager.trigger = AsyncMock()

        parser = ToolCallResponseParser(_catalog(manager), event_manager)
        text = (
            "<|start|>assistant<|channel|>commentary "
            "to=functions.browser.open <|constrain|>json<|message|>"
            '{"url":"https://example.com"}'
        )
        tokens = await parser.push(text)
        flushed = await parser.flush()

        self.assertEqual(tokens, [])
        ready_data = _first_ready_data(flushed)
        self.assertEqual(ready_data["name"], "browser.open")
        self.assertEqual(
            ready_data["arguments"], {"url": "https://example.com"}
        )
        manager.parse_calls.assert_any_call(text + "<|call|>")
        event_manager.trigger.assert_not_awaited()
        self.assertFalse(parser._inside_call)

    async def test_harmony_final_channel_closes_missing_call_marker(
        self,
    ) -> None:
        manager = ToolManager.create_instance(
            enable_tools=[],
            settings=ToolManagerSettings(tool_format=ToolFormat.HARMONY),
        )
        parser = ToolCallResponseParser(_catalog(manager), None)
        text = (
            "<|start|>assistant<|channel|>commentary "
            "to=functions.browser.open <|constrain|>json<|message|>"
            '{"url":"https://example.com"}'
        )

        first = await parser.push(text)
        second = await parser.push("<|channel|>final<|message|>")
        third = await parser.push("done")

        self.assertEqual(first, [])
        ready_data = _first_ready_data(second)
        self.assertEqual(ready_data["name"], "browser.open")
        self.assertEqual(
            ready_data["arguments"], {"url": "https://example.com"}
        )
        self.assertEqual(_answer_texts(third), ["done"])
        self.assertEqual(parser._buffer.getvalue(), "done")
        self.assertFalse(parser._inside_call)

    async def test_harmony_final_channel_splits_from_whole_token(
        self,
    ) -> None:
        manager = ToolManager.create_instance(
            enable_tools=[],
            settings=ToolManagerSettings(tool_format=ToolFormat.HARMONY),
        )
        parser = ToolCallResponseParser(_catalog(manager), None)
        text = (
            "<|start|>assistant<|channel|>commentary "
            "to=functions.browser.open <|constrain|>json<|message|>"
            '{"url":"https://example.com"}'
        )

        items = await parser.push(text + "<|channel|>final<|message|>done")

        ready_data = _first_ready_data(items)
        self.assertEqual(ready_data["name"], "browser.open")
        self.assertEqual(
            ready_data["arguments"],
            {"url": "https://example.com"},
        )
        self.assertEqual(_answer_texts(items), ["done"])
        self.assertEqual(parser._buffer.getvalue(), "done")

    async def test_harmony_final_channel_diagnoses_malformed_call(
        self,
    ) -> None:
        manager = ToolManager.create_instance(
            enable_tools=[],
            settings=ToolManagerSettings(tool_format=ToolFormat.HARMONY),
        )
        parser = ToolCallResponseParser(_catalog(manager), None)
        text = (
            "<|start|>assistant<|channel|>commentary "
            "to=functions.browser.open <|constrain|>json<|message|>"
            '{"url":}'
        )

        await parser.push(text)
        items = await parser.push("<|channel|>final<|message|>")

        diagnostics = _first_diagnostic_data(items)
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(
            diagnostics[0]["code"],
            ToolCallDiagnosticCode.MALFORMED_CALL.value,
        )
        self.assertNotIn("requested_name", diagnostics[0])
        self.assertEqual(
            diagnostics[0]["details"]["stream_status"], "malformed"
        )
        self.assertFalse(parser._inside_call)

    def test_split_current_call_close_skips_preexisting_harmony_final(
        self,
    ) -> None:
        manager = ToolManager.create_instance(
            enable_tools=[],
            settings=ToolManagerSettings(tool_format=ToolFormat.HARMONY),
        )
        parser = ToolCallResponseParser(_catalog(manager), None)
        text = (
            "<|channel|>commentary to=functions.browser.open"
            '<|message|>{"url":"https://example.com"}'
            "<|channel|>final<|message|>done"
        )

        self.assertIsNone(parser._split_current_call_close(text, len(text)))

    async def test_catalog_mock_uses_parse_outcome(self) -> None:
        manager = MagicMock(spec=ModelCapabilityCatalog)
        manager.is_potential_tool_call.return_value = True
        manager.tool_format = ToolFormat.HARMONY
        base_parser = ToolCallParser(tool_format=ToolFormat.HARMONY)
        manager.tool_call_status.side_effect = base_parser.tool_call_status
        manager.parse_calls.side_effect = base_parser.parse
        manager.stream_buffer_diagnostics.return_value = []

        parser = ToolCallResponseParser(_catalog(manager), None)
        await parser.push(
            "<|start|>assistant<|channel|>commentary "
            "to=functions.browser.open <|constrain|>json<|message|>"
            '{"url":"https://example.com"}'
        )
        flushed = await parser.flush()

        ready_data = _first_ready_data(flushed)
        self.assertEqual(ready_data["name"], "browser.open")
        self.assertGreaterEqual(manager.parse_calls.call_count, 1)

    async def test_pending_tokens_flushed_on_status_none(self):
        manager = MagicMock(spec=ModelCapabilityCatalog)
        manager.is_potential_tool_call.return_value = False
        base_parser = ToolCallParser()
        manager.tool_call_status.side_effect = base_parser.tool_call_status

        parser = ToolCallResponseParser(_catalog(manager), None)
        await parser.push("<to")
        result = await parser.push("x")

        self.assertEqual(_answer_texts(result), ["<to", "x"])
        self.assertEqual(parser._pending_tokens, [])
        self.assertEqual(parser._pending_str, "")

    async def test_visible_text_before_split_tool_marker_is_preserved(
        self,
    ) -> None:
        manager = ToolManager.create_instance(enable_tools=[])
        parser = ToolCallResponseParser(_catalog(manager), None)

        first = await parser.push("answer <to")
        second = await parser.push(
            'ol_call>{"name": "math.calculator", '
            '"arguments": {"x": 1}}</tool_call>'
        )

        self.assertEqual(_answer_texts(first), ["answer "])
        self.assertEqual(_ready_names(second), ["math.calculator"])
        self.assertEqual(_ready_arguments(second), [{"x": 1}])
        self.assertEqual(parser._buffer.getvalue(), "")

    async def test_angle_text_before_closed_tool_marker_stays_visible(
        self,
    ) -> None:
        manager = CapturingToolManager.create_instance(enable_tools=[])
        assert isinstance(manager, CapturingToolManager)
        manager.parsed_texts = []
        parser = ToolCallResponseParser(_catalog(manager), None)
        tool_text = (
            '<tool_call>{"name": "math.calculator", '
            '"arguments": {"x": 1}}</tool_call>'
        )

        items = await parser.push("answer <x " + tool_text)

        self.assertEqual(_answer_texts(items), ["answer <x "])
        self.assertEqual(_ready_names(items), ["math.calculator"])
        self.assertEqual(manager.parsed_texts[-1], tool_text)

    async def test_quoted_visible_marker_before_tool_stays_visible(
        self,
    ) -> None:
        manager = CapturingToolManager.create_instance(enable_tools=[])
        assert isinstance(manager, CapturingToolManager)
        manager.parsed_texts = []
        parser = ToolCallResponseParser(_catalog(manager), None)
        tool_text = (
            '<tool_call name="math.calculator" arguments=\'{"x": 1}\'/>'
        )

        items = await parser.push('answer "<tool_call" ' + tool_text + " done")

        self.assertEqual(
            _answer_texts(items),
            ['answer "<tool_call" ', " done"],
        )
        self.assertEqual(_ready_names(items), ["math.calculator"])
        self.assertEqual(_ready_arguments(items), [{"x": 1}])
        self.assertEqual(manager.parsed_texts[-1], tool_text)

    async def test_split_quoted_visible_marker_stays_visible(
        self,
    ) -> None:
        manager = CapturingToolManager.create_instance(enable_tools=[])
        assert isinstance(manager, CapturingToolManager)
        manager.parsed_texts = []
        parser = ToolCallResponseParser(_catalog(manager), None)
        tool_text = (
            '<tool_call name="math.calculator" arguments=\'{"x": 1}\'/>'
        )

        first = await parser.push('answer "<to')
        second = await parser.push('ol_call" ')
        third = await parser.push(tool_text)

        self.assertEqual(_answer_texts(first), ['answer "<to'])
        self.assertEqual(_answer_texts(second), ['ol_call" '])
        self.assertEqual(_ready_arguments(third), [{"x": 1}])
        self.assertEqual(manager.parsed_texts[-1], tool_text)

    async def test_visible_apostrophe_before_tool_does_not_block_call(
        self,
    ) -> None:
        manager = CapturingToolManager.create_instance(enable_tools=[])
        assert isinstance(manager, CapturingToolManager)
        manager.parsed_texts = []
        parser = ToolCallResponseParser(_catalog(manager), None)
        tool_text = (
            '<tool_call name="math.calculator" arguments=\'{"x": 1}\'/>'
        )

        items = await parser.push("don't " + tool_text)

        self.assertEqual(_answer_texts(items), ["don't "])
        self.assertEqual(_ready_arguments(items), [{"x": 1}])
        self.assertEqual(manager.parsed_texts[-1], tool_text)

    async def test_only_quoted_visible_marker_does_not_trigger_tool_event(
        self,
    ) -> None:
        manager = MagicMock(spec=ModelCapabilityCatalog)
        manager.is_potential_tool_call.return_value = True
        base_parser = ToolCallParser()
        manager.tool_call_status.side_effect = base_parser.tool_call_status
        manager.parse_calls.side_effect = AssertionError(
            "quoted marker should stay visible"
        )
        parser = ToolCallResponseParser(_catalog(manager), None)

        items = await parser.push('answer "<tool_call" only')
        flushed = await parser.flush()

        self.assertEqual(_answer_texts(items), ['answer "<tool_call" only'])
        self.assertEqual(flushed, [])
        manager.parse_calls.assert_not_called()

    async def test_next_line_tool_after_unclosed_quote_executes(self) -> None:
        manager = ToolManager.create_instance(enable_tools=[])
        parser = ToolCallResponseParser(_catalog(manager), None)

        first = await parser.push('Visible "quoted text\n')
        second = await parser.push(
            '<tool_call>{"name": "math.calculator", '
            '"arguments": {"expression": "1 + 1"}}</tool_call>'
        )

        process_event = next(
            item
            for item in second
            if item.kind is StreamItemKind.TOOL_CALL_READY
        )

        self.assertEqual(_answer_texts(first), ['Visible "quoted text\n'])
        ready_data = _ready_data(process_event)
        self.assertEqual(ready_data["name"], "math.calculator")
        self.assertEqual(
            ready_data["arguments"],
            {"expression": "1 + 1"},
        )
        self.assertEqual(await parser.flush(), [])

    async def test_same_line_tool_after_unclosed_quote_stays_visible(
        self,
    ) -> None:
        manager = ToolManager.create_instance(enable_tools=[])
        parser = ToolCallResponseParser(_catalog(manager), None)
        text = (
            'Visible "quoted <tool_call>{"name": "math.calculator", '
            '"arguments": {}}</tool_call>'
        )

        items = await parser.push(text)

        self.assertEqual(_answer_texts(items), [text])
        self.assertEqual(await parser.flush(), [])

    async def test_text_after_closed_tool_marker_stays_visible(
        self,
    ) -> None:
        manager = CapturingToolManager.create_instance(enable_tools=[])
        assert isinstance(manager, CapturingToolManager)
        manager.parsed_texts = []
        parser = ToolCallResponseParser(_catalog(manager), None)
        tool_text = (
            '<tool_call>{"name": "math.calculator", '
            '"arguments": {"x": 1}}</tool_call>'
        )

        items = await parser.push("answer " + tool_text + " done")

        self.assertEqual(_answer_texts(items), ["answer ", " done"])
        self.assertEqual(_ready_names(items), ["math.calculator"])
        self.assertEqual(_ready_arguments(items), [{"x": 1}])
        self.assertEqual(manager.parsed_texts[-1], tool_text)
        self.assertEqual(parser._buffer.getvalue(), " done")

    async def test_slash_close_text_after_tool_marker_stays_visible(
        self,
    ) -> None:
        manager = CapturingToolManager.create_instance(enable_tools=[])
        assert isinstance(manager, CapturingToolManager)
        manager.parsed_texts = []
        parser = ToolCallResponseParser(_catalog(manager), None)
        tool_text = (
            '<tool_call>{"name": "math.calculator", '
            '"arguments": {}}</tool_call>'
        )

        items = await parser.push(tool_text + " visible /> tail")

        self.assertEqual(_ready_names(items), ["math.calculator"])
        self.assertEqual(_answer_texts(items), [" visible /> tail"])
        self.assertEqual(manager.parsed_texts[-1], tool_text)

    async def test_slash_close_inside_json_string_keeps_stream_open(
        self,
    ) -> None:
        manager = CapturingToolManager.create_instance(enable_tools=[])
        assert isinstance(manager, CapturingToolManager)
        manager.parsed_texts = []
        parser = ToolCallResponseParser(_catalog(manager), None)

        first = await parser.push(
            '<tool_call>{"name": "math.calculator", "arguments": {"x": "/>"}}'
        )
        self.assertTrue(parser._inside_call)
        second = await parser.push("</tool_call> done")

        self.assertEqual(first, [])
        self.assertEqual(_ready_arguments(second), [{"x": "/>"}])
        self.assertEqual(_answer_texts(second), [" done"])

    async def test_self_closing_marker_inside_json_string_keeps_stream_open(
        self,
    ) -> None:
        manager = CapturingToolManager.create_instance(enable_tools=[])
        assert isinstance(manager, CapturingToolManager)
        manager.parsed_texts = []
        parser = ToolCallResponseParser(_catalog(manager), None)

        first = await parser.push(
            '<tool_call>{"name": "math.calculator", "arguments": '
            '{"text": "<tool_call name=\\"inner\\"/>"}}'
        )
        self.assertTrue(parser._inside_call)
        second = await parser.push("</tool_call> done")

        self.assertEqual(first, [])
        self.assertEqual(
            _ready_arguments(second),
            [{"text": '<tool_call name="inner"/>'}],
        )
        self.assertEqual(_answer_texts(second), [" done"])
        self.assertEqual(
            _argument_texts(second),
            ['{"text":"<tool_call name=\\"inner\\"/>"}'],
        )
        self.assertEqual(
            manager.parsed_texts[-1],
            '<tool_call>{"name": "math.calculator", "arguments": '
            '{"text": "<tool_call name=\\"inner\\"/>"}}</tool_call>',
        )

    async def test_nested_marker_string_keeps_stream_open(self) -> None:
        manager = CapturingToolManager.create_instance(enable_tools=[])
        assert isinstance(manager, CapturingToolManager)
        manager.parsed_texts = []
        parser = ToolCallResponseParser(_catalog(manager), None)

        first = await parser.push(
            '<tool_call>{"name": "math.calculator", "arguments": '
            '{"text": "<tool_call></tool_call>"}}'
        )
        self.assertTrue(parser._inside_call)
        second = await parser.push("</tool_call> done")

        self.assertEqual(
            first,
            [],
        )
        self.assertEqual(
            _ready_arguments(second),
            [{"text": "<tool_call></tool_call>"}],
        )
        self.assertEqual(_answer_texts(second), [" done"])
        self.assertEqual(
            manager.parsed_texts[-1],
            '<tool_call>{"name": "math.calculator", "arguments": '
            '{"text": "<tool_call></tool_call>"}}</tool_call>',
        )

    async def test_close_marker_inside_json_string_keeps_stream_open(
        self,
    ) -> None:
        manager = CapturingToolManager.create_instance(enable_tools=[])
        assert isinstance(manager, CapturingToolManager)
        manager.parsed_texts = []
        parser = ToolCallResponseParser(_catalog(manager), None)

        first = await parser.push(
            '<tool_call>{"name": "math.calculator", "arguments": '
            '{"text": "</tool_call>"}}'
        )
        self.assertTrue(parser._inside_call)
        second = await parser.push("</tool_call> done")

        self.assertEqual(first, [])
        self.assertEqual(
            _ready_arguments(second),
            [{"text": "</tool_call>"}],
        )
        self.assertEqual(_answer_texts(second), [" done"])
        self.assertEqual(
            manager.parsed_texts[-1],
            '<tool_call>{"name": "math.calculator", "arguments": '
            '{"text": "</tool_call>"}}</tool_call>',
        )

    async def test_split_close_marker_inside_json_string_keeps_stream_open(
        self,
    ) -> None:
        manager = CapturingToolManager.create_instance(enable_tools=[])
        assert isinstance(manager, CapturingToolManager)
        manager.parsed_texts = []
        parser = ToolCallResponseParser(_catalog(manager), None)

        first = await parser.push(
            '<tool_call>{"name": "math.calculator", "arguments": {"text": "'
        )
        self.assertTrue(parser._inside_call)
        second = await parser.push('</tool_call>"}}</tool_call> done')

        self.assertEqual(first, [])
        self.assertEqual(
            _ready_arguments(second),
            [{"text": "</tool_call>"}],
        )
        self.assertEqual(_answer_texts(second), [" done"])
        self.assertEqual(
            manager.parsed_texts[-1],
            '<tool_call>{"name": "math.calculator", "arguments": '
            '{"text": "</tool_call>"}}</tool_call>',
        )

    async def test_split_quoted_close_marker_without_close_diagnoses_stream(
        self,
    ) -> None:
        manager = ToolManager.create_instance(enable_tools=[])
        parser = ToolCallResponseParser(_catalog(manager), None)

        pushed = []
        pushed.extend(
            await parser.push(
                '<tool_call>{"name": "math.calculator", "arguments": '
                '{"text": "'
            )
        )
        pushed.extend(await parser.push('</tool_call>"}}'))
        flushed = await parser.flush()

        self.assertEqual(pushed, [])
        self.assertGreaterEqual(len(flushed), 1)
        diagnostics = _diagnostic_data(_diagnostic_events(flushed)[-1])
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(
            diagnostics[0]["code"],
            ToolCallDiagnosticCode.MALFORMED_CALL.value,
        )
        self.assertEqual(
            diagnostics[0]["details"]["stream_status"], "unterminated"
        )

    async def test_quoted_close_marker_without_close_reports_diagnostic(
        self,
    ) -> None:
        manager = ToolManager.create_instance(enable_tools=[])
        parser = ToolCallResponseParser(_catalog(manager), None)

        pushed = await parser.push(
            '<tool_call>{"name": "math.calculator", "arguments": '
            '{"text": "</tool_call>"}}'
        )
        flushed = await parser.flush()

        self.assertEqual(pushed, [])
        self.assertGreaterEqual(len(flushed), 1)
        diagnostics = _diagnostic_data(_diagnostic_events(flushed)[-1])
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(
            diagnostics[0]["code"],
            ToolCallDiagnosticCode.MALFORMED_CALL.value,
        )
        self.assertEqual(
            diagnostics[0]["details"]["stream_status"], "unterminated"
        )

    async def test_closed_tool_marker_without_suffix_has_no_text_tail(
        self,
    ) -> None:
        manager = CapturingToolManager.create_instance(enable_tools=[])
        assert isinstance(manager, CapturingToolManager)
        manager.parsed_texts = []
        parser = ToolCallResponseParser(_catalog(manager), None)
        tool_text = (
            '<tool_call>{"name": "math.calculator", '
            '"arguments": {"x": 1}}</tool_call>'
        )

        items = await parser.push(tool_text)

        self.assertEqual(_ready_names(items), ["math.calculator"])
        self.assertEqual(_ready_arguments(items), [{"x": 1}])
        self.assertEqual(manager.parsed_texts[-1], tool_text)

    async def test_angle_text_before_split_tool_marker_stays_visible(
        self,
    ) -> None:
        manager = CapturingToolManager.create_instance(enable_tools=[])
        assert isinstance(manager, CapturingToolManager)
        manager.parsed_texts = []
        parser = ToolCallResponseParser(_catalog(manager), None)
        tool_remainder = (
            'ol_call>{"name": "math.calculator", '
            '"arguments": {"x": 1}}</tool_call>'
        )

        first = await parser.push("answer <x <to")
        second = await parser.push(tool_remainder)

        self.assertEqual(_answer_texts(first), ["answer <x "])
        self.assertEqual(_ready_arguments(second), [{"x": 1}])
        self.assertEqual(
            manager.parsed_texts[-1],
            "<tool_call>"
            '{"name": "math.calculator", "arguments": {"x": 1}}'
            "</tool_call>",
        )

    async def test_stale_pending_prefix_before_tool_marker_stays_visible(
        self,
    ) -> None:
        manager = CapturingToolManager.create_instance(enable_tools=[])
        assert isinstance(manager, CapturingToolManager)
        manager.parsed_texts = []
        parser = ToolCallResponseParser(_catalog(manager), None)

        first = await parser.push("<to")
        second = await parser.push(
            '<tool_call>{"name": "math.calculator", '
            '"arguments": {"x": 1}}</tool_call>'
        )

        self.assertEqual(first, [])
        self.assertEqual(_answer_texts(second), ["<to"])
        self.assertEqual(_ready_names(second), ["math.calculator"])
        self.assertEqual(_ready_arguments(second), [{"x": 1}])
        self.assertEqual(
            manager.parsed_texts[-1],
            '<tool_call>{"name": "math.calculator", '
            '"arguments": {"x": 1}}</tool_call>',
        )

    async def test_stale_pending_prefix_before_new_partial_marker_flushes_text(
        self,
    ) -> None:
        manager = CapturingToolManager.create_instance(enable_tools=[])
        assert isinstance(manager, CapturingToolManager)
        manager.parsed_texts = []
        parser = ToolCallResponseParser(_catalog(manager), None)

        first = await parser.push("<to")
        second = await parser.push(" visible <tool")
        third = await parser.push(
            '_call>{"name": "math.calculator", '
            '"arguments": {"x": 1}}</tool_call>'
        )

        self.assertEqual(first, [])
        self.assertEqual(_answer_texts(second), ["<to visible "])
        self.assertEqual(_ready_arguments(third), [{"x": 1}])
        self.assertEqual(
            manager.parsed_texts[-1],
            '<tool_call>{"name": "math.calculator", '
            '"arguments": {"x": 1}}</tool_call>',
        )

    async def test_non_marker_text_with_angle_bracket_is_not_split(
        self,
    ) -> None:
        manager = ToolManager.create_instance(enable_tools=[])
        parser = ToolCallResponseParser(_catalog(manager), None)

        result = await parser.push("answer <x")

        self.assertEqual(_answer_texts(result), ["answer <x"])
        self.assertEqual(parser._pending_tokens, [])
        self.assertEqual(parser._pending_str, "")

    async def test_markdown_fenced_tool_call_stays_visible(self) -> None:
        manager = ToolManager.create_instance(enable_tools=[])
        event_manager = MagicMock()
        event_manager.trigger = AsyncMock()
        parser = ToolCallResponseParser(_catalog(manager), event_manager)
        text = (
            "```xml\n"
            '<tool_call>{"name": "math.calculator", "arguments": {}}'
            "</tool_call>\n"
            "```"
        )

        pushed = await parser.push(text)
        flushed = await parser.flush()

        self.assertEqual(_answer_texts(pushed), [text])
        self.assertEqual(flushed, [])
        event_manager.trigger.assert_not_awaited()
        self.assertEqual(parser._pending_tokens, [])
        self.assertEqual(parser._pending_str, "")

    async def test_split_markdown_fenced_tool_call_stays_visible(self) -> None:
        manager = ToolManager.create_instance(enable_tools=[])
        event_manager = MagicMock()
        event_manager.trigger = AsyncMock()
        parser = ToolCallResponseParser(_catalog(manager), event_manager)

        first = await parser.push("```xml\n<to")
        second = await parser.push(
            'ol_call>{"name": "math.calculator", "arguments": {}}'
            "</tool_call>\n```"
        )
        flushed = await parser.flush()

        self.assertEqual(_answer_texts(first), ["```xml\n<to"])
        self.assertEqual(
            _answer_texts(second),
            [
                'ol_call>{"name": "math.calculator", "arguments": {}}'
                "</tool_call>\n```"
            ],
        )
        self.assertEqual(flushed, [])
        event_manager.trigger.assert_not_awaited()

    async def test_tool_call_after_closed_markdown_fence_executes(
        self,
    ) -> None:
        manager = ToolManager.create_instance(enable_tools=[])
        parser = ToolCallResponseParser(_catalog(manager), None)
        tool_text = (
            '<tool_call>{"name": "math.calculator", '
            '"arguments": {"x": 1}}</tool_call>'
        )

        items = await parser.push("```xml\n<tool_call></tool_call>\n```\n")
        items.extend(await parser.push(tool_text))

        self.assertEqual(
            _answer_texts(items),
            ["```xml\n<tool_call></tool_call>\n```\n"],
        )
        self.assertEqual(_ready_names(items), ["math.calculator"])
        self.assertEqual(_ready_arguments(items), [{"x": 1}])

    async def test_streaming_hot_path_uses_incremental_marker_state(
        self,
    ) -> None:
        manager = ToolManager.create_instance(enable_tools=[])
        parser = ToolCallResponseParser(_catalog(manager), None)
        items: list = []

        with (
            patch.object(
                ToolCallResponseParser,
                "_markdown_fence_is_open",
                side_effect=AssertionError("full fence scan used"),
            ),
            patch.object(
                ToolCallResponseParser,
                "_index_is_inside_visible_quote",
                side_effect=AssertionError("full quote scan used"),
            ),
            patch.object(
                ToolCallResponseParser,
                "_split_current_call_close",
                side_effect=AssertionError("full close scan used"),
            ),
        ):
            for token in (
                "```xml\n",
                (
                    '<tool_call>{"name": "ignored", "arguments": {}}'
                    "</tool_call>\n"
                ),
                "```\n",
                'visible "<tool_call" marker\n',
                (
                    '<tool_call>{"name": "math.calculator", '
                    '"arguments": {"x": 1}}'
                ),
                "</tool",
                "_call> tail",
            ):
                items.extend(await parser.push(token))

        process_event = next(
            item
            for item in items
            if item.kind is StreamItemKind.TOOL_CALL_READY
        )

        self.assertEqual(_ready_data(process_event)["name"], "math.calculator")
        self.assertEqual(_ready_data(process_event)["arguments"], {"x": 1})
        self.assertEqual(
            _answer_texts(items),
            [
                "```xml\n",
                (
                    '<tool_call>{"name": "ignored", "arguments": {}}'
                    "</tool_call>\n"
                ),
                "```\n",
                'visible "<tool_call" marker\n',
                " tail",
            ],
        )

    def test_incremental_fence_state_preserves_fence_lengths(self) -> None:
        state = _MarkdownFenceState()

        state.push("````python\n")
        self.assertTrue(state.is_open)
        state.push("```\n")
        self.assertTrue(state.is_open)
        state.push("````\n")

        self.assertFalse(state.is_open)

    def test_incremental_visible_quote_state_delays_apostrophes(
        self,
    ) -> None:
        quote = _VisibleQuoteState()
        quote.push_character("a")
        quote.push_character("'")

        self.assertTrue(quote.is_open_before(None))

        quote.push_character(" ")
        self.assertTrue(quote.is_open_before("<"))
        quote.push_character("\\")
        quote.push_character("'")
        self.assertTrue(quote.is_open_before("<"))
        quote.push_character("'")

        self.assertFalse(quote.is_open_before("<"))

        leading_quote = _VisibleQuoteState()
        leading_quote.push_character("'")

        self.assertTrue(leading_quote.is_open_before("<"))

    def test_incremental_visible_text_state_handles_empty_indexes(
        self,
    ) -> None:
        state = _VisibleTextState()

        self.assertEqual(state.executable_marker_indexes("plain", []), [])

    def test_visible_marker_indexes_supports_explicit_buffer_prefix(
        self,
    ) -> None:
        manager = MagicMock(spec=ModelCapabilityCatalog)
        parser = ToolCallResponseParser(_catalog(manager), None)

        self.assertEqual(
            parser._visible_marker_indexes(" before <tool_call>", "prefix "),
            [8],
        )

    def test_marker_helper_branches(self) -> None:
        manager = ToolManager.create_instance(enable_tools=[])
        parser = ToolCallResponseParser(_catalog(manager), None)

        self.assertFalse(
            ToolCallResponseParser._markdown_fence_is_open(
                "````python\n````\n"
            )
        )
        self.assertFalse(
            ToolCallResponseParser._index_is_inside_visible_quote(
                '"closed" <tool_call', len('"closed" ')
            )
        )
        self.assertFalse(
            ToolCallResponseParser._index_is_inside_visible_quote(
                '"open\n<tool_call', len('"open\n')
            )
        )
        self.assertIsNone(parser._self_closing_tool_close_span_at("plain", 0))
        self.assertIsNone(
            parser._self_closing_tool_close_span_at("<tool_call", 0)
        )
        self.assertIsNone(
            parser._self_closing_tool_close_span_at("<tool_call>", 0)
        )
        self.assertEqual(
            parser._self_closing_tool_close_span_at(
                '<tool_call name="x"/>', 0
            ),
            (0, len('<tool_call name="x"/>')),
        )
        self.assertEqual(
            parser._tool_start_marker_at("<tool_call", 0),
            "<tool_call",
        )
        self.assertIsNone(parser._tool_start_marker_at("<tool_callout", 0))
        self.assertEqual(
            parser._tool_marker_at("</tool>", 0, ("</tool>",)),
            "</tool>",
        )
        self.assertIsNone(parser._tool_marker_at("plain", 0, ("</tool>",)))
        self.assertIsNone(parser._tool_close_split("tool", "", "text", None))

    async def test_fence_opened_after_tool_suffix_blocks_next_call(
        self,
    ) -> None:
        manager = ToolManager.create_instance(enable_tools=[])
        event_manager = MagicMock()
        event_manager.trigger = AsyncMock()
        parser = ToolCallResponseParser(_catalog(manager), event_manager)
        first_tool = (
            '<tool_call>{"name": "math.calculator", '
            '"arguments": {"x": 1}}</tool_call>'
        )
        fenced_tool = (
            '<tool_call>{"name": "should.not_run", '
            '"arguments": {}}</tool_call>\n```'
        )

        first = await parser.push(first_tool + "\n```xml\n")
        second = await parser.push(fenced_tool)

        self.assertEqual(_ready_names(first), ["math.calculator"])
        self.assertEqual(_answer_texts(first), ["\n```xml\n"])
        self.assertEqual(_answer_texts(second), [fenced_tool])
        event_manager.trigger.assert_not_awaited()

    def test_split_visible_prefix_rejects_unconfirmed_marker(self) -> None:
        manager = MagicMock(spec=ModelCapabilityCatalog)
        manager.tool_call_status.return_value = (
            ToolCallParser.ToolCallBufferStatus.NONE
        )
        parser = ToolCallResponseParser(_catalog(manager), None)

        result = parser._split_visible_prefix(
            "answer <tool_call>",
            ToolCallParser.ToolCallBufferStatus.NONE,
        )

        self.assertEqual(result, (None, "answer <tool_call>"))

    def test_split_closed_visible_suffix_keeps_adjacent_tool_marker(
        self,
    ) -> None:
        manager = ToolManager.create_instance(enable_tools=[])
        parser = ToolCallResponseParser(_catalog(manager), None)
        text = '<tool_call name="first"/> <tool_call'

        self.assertEqual(
            parser._split_closed_visible_suffix(text),
            (text, ""),
        )

    def test_split_closed_visible_suffix_handles_dsml_suffix(self) -> None:
        manager = MagicMock(spec=ModelCapabilityCatalog)
        manager.tool_format = ToolFormat.DSML
        parser = ToolCallResponseParser(_catalog(manager), None)

        self.assertEqual(
            parser._split_closed_visible_suffix(
                "<tool_calls></tool_calls> done"
            ),
            ("<tool_calls></tool_calls>", " done"),
        )

    def test_split_closed_visible_suffix_ignores_slash_close_in_text(
        self,
    ) -> None:
        manager = ToolManager.create_instance(enable_tools=[])
        parser = ToolCallResponseParser(_catalog(manager), None)
        text = '<tool_call name="first"></tool_call> visible /> tail'

        self.assertEqual(
            parser._split_closed_visible_suffix(text),
            ('<tool_call name="first"></tool_call>', " visible /> tail"),
        )

    def test_split_closed_visible_suffix_handles_self_closing_tag(
        self,
    ) -> None:
        manager = ToolManager.create_instance(enable_tools=[])
        parser = ToolCallResponseParser(_catalog(manager), None)

        self.assertEqual(
            parser._split_closed_visible_suffix(
                '<tool_call name="first"/> done'
            ),
            ('<tool_call name="first"/>', " done"),
        )

    def test_split_closed_visible_suffix_skips_quoted_close_marker(
        self,
    ) -> None:
        manager = ToolManager.create_instance(enable_tools=[])
        parser = ToolCallResponseParser(_catalog(manager), None)

        self.assertEqual(
            parser._split_closed_visible_suffix(
                '<tool_call>{"name": "calculator", "arguments": '
                '{"text": "</tool_call>"}}</tool_call> done'
            ),
            (
                (
                    '<tool_call>{"name": "calculator", "arguments": '
                    '{"text": "</tool_call>"}}</tool_call>'
                ),
                " done",
            ),
        )

    def test_split_closed_visible_suffix_skips_escaped_marker(self) -> None:
        manager = ToolManager.create_instance(enable_tools=[])
        parser = ToolCallResponseParser(_catalog(manager), None)

        self.assertEqual(
            parser._split_closed_visible_suffix(
                '<tool_call>{"name": "calculator", "arguments": '
                '{"text": "\\</tool_call>"}}</tool_call> done'
            ),
            (
                (
                    '<tool_call>{"name": "calculator", "arguments": '
                    '{"text": "\\</tool_call>"}}</tool_call>'
                ),
                " done",
            ),
        )

    def test_split_current_call_close_skips_fenced_markers(self) -> None:
        manager = ToolManager.create_instance(enable_tools=[])
        parser = ToolCallResponseParser(_catalog(manager), None)

        self.assertIsNone(
            parser._split_current_call_close(
                '<tool_call>{"name": "calculator", "arguments": {}}\n'
                "```xml\n"
                "</tool_call>\n"
                "```",
                0,
            )
        )

    def test_split_current_call_close_handles_self_closing_tag(self) -> None:
        manager = ToolManager.create_instance(enable_tools=[])
        parser = ToolCallResponseParser(_catalog(manager), None)

        self.assertEqual(
            parser._split_current_call_close(
                '<tool_call name="first"/> tail',
                len("<tool_call"),
            ),
            ('<tool_call name="first"/>', " tail"),
        )

    def test_split_current_call_close_skips_preexisting_self_close(
        self,
    ) -> None:
        manager = ToolManager.create_instance(enable_tools=[])
        parser = ToolCallResponseParser(_catalog(manager), None)
        first = '<tool_call name="first"/>'
        text = first + '<tool_call name="second"/> tail'

        self.assertEqual(
            parser._split_current_call_close(text, len(first) + 1),
            (
                '<tool_call name="first"/><tool_call name="second"/>',
                " tail",
            ),
        )

    def test_self_closing_tool_end_indexes_handles_open_tag(self) -> None:
        manager = ToolManager.create_instance(enable_tools=[])
        parser = ToolCallResponseParser(_catalog(manager), None)

        self.assertEqual(
            list(parser._self_closing_tool_end_indexes("<tool_call")),
            [],
        )

    def test_self_closing_tool_end_indexes_include_close_positions(
        self,
    ) -> None:
        manager = ToolManager.create_instance(enable_tools=[])
        parser = ToolCallResponseParser(_catalog(manager), None)

        indexes = list(
            parser._self_closing_tool_end_indexes(
                '<tool_call name="first"/><tool name="second"/>'
            )
        )

        self.assertEqual(indexes, [25, 46])

    def test_marker_indexes_returns_all_matches(self) -> None:
        self.assertEqual(
            list(
                ToolCallResponseParser._marker_indexes(
                    "</tool> text </tool>", "</tool>"
                )
            ),
            [0, 13],
        )

    def test_visible_quote_detection_handles_escapes_and_single_quotes(
        self,
    ) -> None:
        parser = ToolCallResponseParser(
            MagicMock(spec=ModelCapabilityCatalog), None
        )
        escaped_quote = r'"quoted \" marker <tool_call'
        single_quote = "'<tool_call'"

        self.assertTrue(
            parser._index_is_inside_visible_quote(
                escaped_quote, escaped_quote.index("<tool_call")
            )
        )
        self.assertTrue(
            parser._index_is_inside_visible_quote(
                single_quote, single_quote.index("<tool_call")
            )
        )
        self.assertFalse(
            parser._index_is_inside_visible_quote(
                "don't <tool_call", len("don't ")
            )
        )

    def test_split_closed_visible_suffix_without_close_marker(
        self,
    ) -> None:
        manager = ToolManager.create_instance(enable_tools=[])
        parser = ToolCallResponseParser(_catalog(manager), None)

        self.assertEqual(
            parser._split_closed_visible_suffix("<tool_call>"),
            ("<tool_call>", ""),
        )

    async def test_flush_returns_pending_tokens(self):
        manager = MagicMock(spec=ModelCapabilityCatalog)
        manager.is_potential_tool_call.return_value = False
        base_parser = ToolCallParser()
        manager.tool_call_status.side_effect = base_parser.tool_call_status
        manager.stream_buffer_diagnostics.return_value = []

        parser = ToolCallResponseParser(_catalog(manager), None)
        await parser.push("<tool")

        self.assertEqual(_answer_texts(await parser.flush()), ["<tool"])
        self.assertEqual(parser._pending_tokens, [])
        self.assertEqual(parser._pending_str, "")

    async def test_flush_pending_tool_prefix_emits_diagnostic_event(self):
        manager = ToolManager.create_instance(enable_tools=[])
        parser = ToolCallResponseParser(_catalog(manager), None)

        pushed = await parser.push("<tool")
        flushed = await parser.flush()

        self.assertEqual(pushed, [])
        self.assertGreaterEqual(len(flushed), 1)
        diagnostics = _diagnostic_data(_diagnostic_events(flushed)[-1])
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(
            diagnostics[0]["code"],
            ToolCallDiagnosticCode.MALFORMED_CALL.value,
        )
        self.assertEqual(
            diagnostics[0]["details"]["stream_status"], "unterminated"
        )
        self.assertEqual(parser._pending_tokens, [])
        self.assertEqual(parser._pending_str, "")
        self.assertEqual(parser._buffer.getvalue(), "")

    async def test_flush_split_pending_tool_prefix_emits_diagnostic_event(
        self,
    ) -> None:
        manager = ToolManager.create_instance(enable_tools=[])
        parser = ToolCallResponseParser(_catalog(manager), None)

        pushed = await parser.push("answer <tool")
        flushed = await parser.flush()

        self.assertEqual(_answer_texts(pushed), ["answer "])
        self.assertGreaterEqual(len(flushed), 1)
        diagnostics = _diagnostic_data(_diagnostic_events(flushed)[-1])
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(
            diagnostics[0]["code"],
            ToolCallDiagnosticCode.MALFORMED_CALL.value,
        )
        self.assertEqual(
            diagnostics[0]["details"]["stream_status"], "unterminated"
        )
        self.assertEqual(parser._pending_tokens, [])
        self.assertEqual(parser._pending_str, "")
        self.assertEqual(parser._buffer.getvalue(), "")

    async def test_flush_unsplit_angle_text_returns_plain_text(
        self,
    ) -> None:
        manager = ToolManager.create_instance(enable_tools=[])
        parser = ToolCallResponseParser(_catalog(manager), None)

        pushed = await parser.push("answer <x")
        flushed = await parser.flush()

        self.assertEqual(_answer_texts(pushed), ["answer <x"])
        self.assertEqual(flushed, [])

    async def test_open_status_returns_empty_without_tokens(self):
        manager = MagicMock(spec=ModelCapabilityCatalog)
        manager.is_potential_tool_call.return_value = True
        manager.tool_call_status.return_value = (
            ToolCallParser.ToolCallBufferStatus.OPEN
        )

        parser = ToolCallResponseParser(_catalog(manager), None)

        class EmptyIterList(list):
            def __iter__(self):
                return iter([])

        parser._pending_tokens = EmptyIterList()

        self.assertEqual(await parser.push("<tool_call>"), [])
        self.assertTrue(parser._inside_call)

    async def test_closed_malformed_stream_emits_diagnostic_event(self):
        manager = ToolManager.create_instance(enable_tools=[])
        event_manager = MagicMock()
        event_manager.trigger = AsyncMock()
        parser = ToolCallResponseParser(_catalog(manager), event_manager)

        items: list = []
        for part in (
            "<tool_call>",
            '{"name": "calculator", "arguments": }',
            "</tool_call>",
        ):
            items.extend(await parser.push(part))

        diagnostics = _diagnostic_data(_diagnostic_events(items)[-1])
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(
            diagnostics[0]["code"],
            ToolCallDiagnosticCode.MALFORMED_CALL.value,
        )
        self.assertFalse(parser._inside_call)
        self.assertEqual(parser._buffer.getvalue(), "")
        event_manager.trigger.assert_not_awaited()

    async def test_malformed_tool_tag_stream_emits_diagnostic_event(self):
        manager = ToolManager.create_instance(enable_tools=[])
        parser = ToolCallResponseParser(_catalog(manager), None)

        items = await parser.push("<tool>not json</tool>")

        diagnostics = _first_diagnostic_data(items)
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(
            diagnostics[0]["code"],
            ToolCallDiagnosticCode.MALFORMED_CALL.value,
        )
        self.assertEqual(
            diagnostics[0]["details"]["stream_status"], "malformed"
        )
        self.assertFalse(parser._inside_call)
        self.assertEqual(parser._buffer.getvalue(), "")

    async def test_stream_with_call_and_diagnostic_emits_both_events(
        self,
    ) -> None:
        manager = ToolManager.create_instance(enable_tools=[])
        event_manager = MagicMock()
        event_manager.trigger = AsyncMock()
        parser = ToolCallResponseParser(_catalog(manager), event_manager)

        items = await parser.push(
            '<tool_call>{"name": "bad..name", "arguments": {}}</tool_call>'
            '<tool_call>{"name": "math.calculator", '
            '"arguments": {"x": 1}}</tool_call>'
        )

        diagnostic_event = next(
            item
            for item in items
            if item.kind is StreamItemKind.STREAM_DIAGNOSTIC
        )
        process_event = next(
            item
            for item in items
            if item.kind is StreamItemKind.TOOL_CALL_READY
        )
        diagnostics = _diagnostic_data(diagnostic_event)

        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(
            diagnostics[0]["code"],
            ToolCallDiagnosticCode.MALFORMED_CALL.value,
        )
        self.assertNotIn("requested_name", diagnostics[0])
        self.assertEqual(_ready_data(process_event)["name"], "math.calculator")
        self.assertEqual(_ready_data(process_event)["arguments"], {"x": 1})
        event_manager.trigger.assert_not_awaited()
        self.assertEqual(parser._buffer.getvalue(), "")

    async def test_stream_defers_complete_call_before_open_sibling(
        self,
    ) -> None:
        manager = ToolManager.create_instance(enable_tools=[])
        parser = ToolCallResponseParser(_catalog(manager), None)

        first = await parser.push(
            '<tool_call>{"name": "first", "arguments": {}}</tool_call>'
            ' between <tool_call>{"name": "second", "arguments": {}'
        )
        second = await parser.push("}</tool_call> after")

        self.assertFalse(
            any(item.kind is StreamItemKind.TOOL_CALL_READY for item in first)
        )
        self.assertEqual(_ready_names(second), ["first", "second"])
        self.assertEqual(_answer_texts(second), [" after"])
        self.assertFalse(parser._inside_call)
        self.assertEqual(parser._buffer.getvalue(), " after")

    async def test_stream_defers_complete_call_before_mixed_open_sibling(
        self,
    ) -> None:
        manager = ToolManager.create_instance(enable_tools=[])
        parser = ToolCallResponseParser(_catalog(manager), None)

        first = await parser.push(
            '<tool_call>{"name": "first", "arguments": {}}</tool_call>'
            ' between <tool name="second">{"value": '
        )
        second = await parser.push("2}</tool> after")

        self.assertFalse(
            any(item.kind is StreamItemKind.TOOL_CALL_READY for item in first)
        )
        self.assertEqual(_ready_names(second), ["first", "second"])
        self.assertEqual(_ready_arguments(second)[-1], {"value": 2})
        self.assertEqual(_answer_texts(second), [" after"])
        self.assertFalse(parser._inside_call)
        self.assertEqual(parser._buffer.getvalue(), " after")

    async def test_stream_processes_call_after_closed_call_suffix(
        self,
    ) -> None:
        manager = ToolManager.create_instance(enable_tools=[])
        parser = ToolCallResponseParser(_catalog(manager), None)

        self.assertEqual(
            await parser.push('<tool_call>{"name": "first", "arguments": {}}'),
            [],
        )
        second = await parser.push(
            '</tool_call> between <tool_call>{"name": "second", '
            '"arguments": {"value": '
        )
        third = await parser.push("2}}</tool_call> tail")

        self.assertEqual(_ready_names(second + third), ["first", "second"])
        self.assertEqual(_ready_arguments(second + third)[1], {"value": 2})
        self.assertIn(" between ", _answer_texts(second))
        self.assertEqual(_answer_texts(third), [" tail"])
        self.assertFalse(parser._inside_call)
        self.assertEqual(parser._buffer.getvalue(), " tail")

    async def test_stream_preserves_suffix_after_split_close_marker(
        self,
    ) -> None:
        manager = ToolManager.create_instance(enable_tools=[])
        parser = ToolCallResponseParser(_catalog(manager), None)

        self.assertEqual(
            await parser.push(
                '<tool_call>{"name": "first", "arguments": {}}</tool'
            ),
            [],
        )
        second = await parser.push("_call> tail")

        self.assertEqual(_ready_names(second), ["first"])
        self.assertEqual(_answer_texts(second), [" tail"])
        self.assertFalse(parser._inside_call)
        self.assertEqual(parser._buffer.getvalue(), " tail")

    async def test_split_harmony_final_marker_stays_visible_suffix(
        self,
    ) -> None:
        manager = ToolManager.create_instance(
            enable_tools=[],
            settings=ToolManagerSettings(tool_format=ToolFormat.HARMONY),
        )
        parser = ToolCallResponseParser(_catalog(manager), None)

        first = await parser.push(
            "<|start|>assistant<|channel|>commentary "
            "to=functions.browser.open <|constrain|>json<|message|>"
            '{"url":"https://example.com"}'
        )
        second = await parser.push("<|channel|>final<|mes")
        third = await parser.push("sage|>done")
        items = first + second + third

        process_event = next(
            item
            for item in items
            if item.kind is StreamItemKind.TOOL_CALL_READY
        )
        visible_items = _answer_texts(items)

        self.assertEqual(_ready_data(process_event)["name"], "browser.open")
        self.assertEqual(
            _ready_data(process_event)["arguments"],
            {"url": "https://example.com"},
        )
        self.assertEqual(visible_items, ["done"])
        self.assertFalse(
            any(
                item.kind is StreamItemKind.TOOL_CALL_ARGUMENT_DELTA
                and item.text_delta == "<|channel|>final<|message|>done"
                for item in items
            )
        )
        self.assertFalse(
            any(
                item.kind is StreamItemKind.STREAM_DIAGNOSTIC for item in items
            )
        )
        self.assertFalse(parser._inside_call)
        self.assertEqual(parser._buffer.getvalue(), "done")

    async def test_real_tool_manager_skips_status_rescan_for_open_chunks(
        self,
    ) -> None:
        manager = ToolManager.create_instance(enable_tools=[])
        catalog = _catalog(manager)
        original_status = catalog.tool_call_status
        status_lengths: list[int] = []

        def status(
            buffer: str, *, final: bool = False
        ) -> ToolCallParser.ToolCallBufferStatus:
            status_lengths.append(len(buffer))
            return original_status(buffer, final=final)

        with patch.object(
            ModelCapabilityCatalog,
            "tool_call_status",
            autospec=True,
            side_effect=lambda _catalog, buffer, *, final=False: status(
                buffer, final=final
            ),
        ):
            parser = ToolCallResponseParser(catalog, None)

            await parser.push(
                '<tool_call>{"name": "math.calculator", "arguments": '
                '{"expression": "'
            )
            status_lengths.clear()

            for token in ("1", " + ", "1", '"}}'):
                pushed = await parser.push(token)
                self.assertTrue(
                    all(
                        item.kind is StreamItemKind.TOOL_CALL_ARGUMENT_DELTA
                        for item in pushed
                    )
                )

        self.assertEqual(status_lengths, [])

        closed = await parser.push("</tool_call>")
        process_event = next(
            item
            for item in closed
            if item.kind is StreamItemKind.TOOL_CALL_READY
        )

        self.assertEqual(_ready_data(process_event)["name"], "math.calculator")
        self.assertEqual(
            _ready_data(process_event)["arguments"],
            {"expression": "1 + 1"},
        )
        self.assertFalse(parser._inside_call)

    async def test_real_manager_uses_incremental_tool_length_for_open_chunks(
        self,
    ) -> None:
        manager = ToolManager.create_instance(enable_tools=[])
        parser = ToolCallResponseParser(_catalog(manager), None)
        items: list[StreamProviderEvent] = []

        await parser.push(
            '<tool_call>{"name": "math.calculator", "arguments": '
            '{"expression": "'
        )

        with patch.object(
            parser._tool_buffer,
            "getvalue",
            side_effect=AssertionError("full tool buffer read"),
        ):
            for token in ("1", " + ", "1"):
                pushed = await parser.push(token)
                self.assertEqual(pushed, [])

        items.extend(await parser.push('"}}'))
        items.extend(await parser.push("</tool_call>"))
        process_event = next(
            item
            for item in items
            if item.kind is StreamItemKind.TOOL_CALL_READY
        )

        self.assertEqual(_ready_data(process_event)["name"], "math.calculator")
        self.assertEqual(
            _ready_data(process_event)["arguments"],
            {"expression": "1 + 1"},
        )
        self.assertFalse(parser._inside_call)

    async def test_split_quoted_marker_stays_visible_and_real_marker_executes(
        self,
    ) -> None:
        manager = ToolManager.create_instance(enable_tools=[])
        parser = ToolCallResponseParser(_catalog(manager), None)
        items: list[StreamProviderEvent] = []
        quoted_suffix = (
            'ol_call>{"name": "ignored", "arguments": {}}</tool_call>"\n'
        )

        for token in (
            'quoted "<to',
            quoted_suffix,
            "<tool",
            '_call>{"name": "math.calculator", "arguments": {"x": 2}}</tool',
            "_call> tail",
        ):
            items.extend(await parser.push(token))

        self.assertEqual(_ready_names(items), ["math.calculator"])
        self.assertEqual(
            _ready_arguments(items),
            [{"x": 2}],
        )
        self.assertEqual(
            _answer_texts(items),
            ['quoted "<to', quoted_suffix, " tail"],
        )
        self.assertFalse(
            any(isinstance(item, (Event, ToolCallToken)) for item in items)
        )
        self.assertFalse(_diagnostic_events(items))

    async def test_split_malformed_call_emits_canonical_diagnostic_only(
        self,
    ) -> None:
        manager = ToolManager.create_instance(enable_tools=[])
        parser = ToolCallResponseParser(_catalog(manager), None)
        items: list[StreamProviderEvent] = []

        for token in (
            "<tool_call>",
            '{"name": "math.calculator", "arguments": }',
            "</tool",
            "_call>",
        ):
            items.extend(await parser.push(token))

        diagnostics = _first_diagnostic_data(items)
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(
            diagnostics[0]["code"],
            ToolCallDiagnosticCode.MALFORMED_CALL.value,
        )
        self.assertFalse(
            any(isinstance(item, (Event, ToolCallToken)) for item in items)
        )
        self.assertFalse(_ready_events(items))

    async def test_stream_keeps_fenced_call_after_closed_call_visible(
        self,
    ) -> None:
        manager = ToolManager.create_instance(enable_tools=[])
        event_manager = MagicMock()
        event_manager.trigger = AsyncMock()
        parser = ToolCallResponseParser(_catalog(manager), event_manager)
        fenced_suffix = (
            "\n```xml\n"
            '<tool_call>{"name": "second", "arguments": {}}</tool_call>\n'
            "```"
        )

        await parser.push('<tool_call>{"name": "first", "arguments": {}}')
        second = await parser.push("</tool_call>" + fenced_suffix)

        process_events = [
            item
            for item in second
            if item.kind is StreamItemKind.TOOL_CALL_READY
        ]

        self.assertEqual(len(process_events), 1)
        self.assertEqual(_ready_data(process_events[0])["name"], "first")
        self.assertEqual(_answer_texts(second), [fenced_suffix])
        self.assertEqual(parser._buffer.getvalue(), fenced_suffix)
        event_manager.trigger.assert_not_awaited()

    async def test_stream_reports_malformed_open_sibling_after_valid_call(
        self,
    ) -> None:
        manager = ToolManager.create_instance(enable_tools=[])
        parser = ToolCallResponseParser(_catalog(manager), None)

        first = await parser.push(
            '<tool_call>{"name": "first", "arguments": {}}</tool_call>'
            ' between <tool_call>{"name": "bad..name", "arguments": {}'
        )
        second = await parser.push("}</tool_call> tail")

        self.assertFalse(
            any(item.kind is StreamItemKind.TOOL_CALL_READY for item in first)
        )
        diagnostic_event = next(
            item
            for item in second
            if item.kind is StreamItemKind.STREAM_DIAGNOSTIC
        )
        process_event = next(
            item
            for item in second
            if item.kind is StreamItemKind.TOOL_CALL_READY
        )
        diagnostics = _diagnostic_data(diagnostic_event)

        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(
            diagnostics[0]["code"],
            ToolCallDiagnosticCode.MALFORMED_CALL.value,
        )
        self.assertNotIn("requested_name", diagnostics[0])
        self.assertEqual(_ready_data(process_event)["name"], "first")
        self.assertEqual(_answer_texts(second), [" tail"])
        self.assertFalse(parser._inside_call)

    async def test_valid_stream_with_call_emits_no_diagnostic_event(
        self,
    ) -> None:
        manager = ToolManager.create_instance(enable_tools=[])
        event_manager = MagicMock()
        event_manager.trigger = AsyncMock()
        parser = ToolCallResponseParser(_catalog(manager), event_manager)

        items = await parser.push(
            '<tool_call>{"name": "math.calculator", '
            '"arguments": {"x": 1}}</tool_call>'
        )

        self.assertTrue(
            any(item.kind is StreamItemKind.TOOL_CALL_READY for item in items)
        )
        self.assertFalse(
            any(
                item.kind is StreamItemKind.STREAM_DIAGNOSTIC for item in items
            )
        )
        event_manager.trigger.assert_not_awaited()

    async def test_subclassed_manager_closed_stream_uses_diagnostics(
        self,
    ) -> None:
        manager = DiagnosticFallbackToolManager.create_instance(
            enable_tools=[]
        )
        parser = ToolCallResponseParser(_catalog(manager), None)

        items = await parser.push("<tool_call></tool_call>")

        diagnostics = _diagnostic_data(_diagnostic_events(items)[-1])
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(
            diagnostics[0]["code"],
            ToolCallDiagnosticCode.MALFORMED_CALL.value,
        )
        self.assertEqual(parser._buffer.getvalue(), "")

    async def test_closed_stream_without_diagnostics_returns_tokens(
        self,
    ) -> None:
        manager = NoDiagnosticToolManager.create_instance(enable_tools=[])
        parser = ToolCallResponseParser(_catalog(manager), None)

        items = await parser.push("<tool_call></tool_call>")

        self.assertEqual(items, [])
        self.assertFalse(parser._inside_call)

    async def test_flush_unterminated_stream_emits_diagnostic_event(self):
        manager = ToolManager.create_instance(enable_tools=[])
        parser = ToolCallResponseParser(_catalog(manager), None)

        pushed = await parser.push(
            '<tool_call>{"name": "calculator", "arguments": {}}'
        )
        flushed = await parser.flush()

        self.assertEqual(pushed, [])
        self.assertEqual(len(flushed), 1)
        diagnostics = _first_diagnostic_data(flushed)
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(
            diagnostics[0]["code"],
            ToolCallDiagnosticCode.MALFORMED_CALL.value,
        )
        self.assertEqual(
            diagnostics[0]["details"]["stream_status"], "unterminated"
        )
        self.assertFalse(parser._inside_call)
