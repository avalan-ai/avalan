from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock

from avalan.entities import (
    ToolCallDiagnosticCode,
    ToolCallToken,
    ToolFormat,
)
from avalan.event import EventType
from avalan.model.response.parsers.tool import ToolCallResponseParser
from avalan.tool.manager import ToolManager
from avalan.tool.parser import ToolCallParser


class ToolCallParserExtraTestCase(IsolatedAsyncioTestCase):
    async def test_tag_buffer_trim_no_check(self):
        manager = MagicMock()
        manager.is_potential_tool_call.return_value = False
        manager.tool_call_status.return_value = (
            ToolCallParser.ToolCallBufferStatus.NONE
        )
        parser = ToolCallResponseParser(manager, None)

        long_token = "a" * 65
        result = await parser.push(long_token)

        self.assertEqual(result, [long_token])
        self.assertEqual(len(parser._tag_buffer), 64)

    async def test_trigger_and_event(self):
        manager = MagicMock()
        manager.is_potential_tool_call.return_value = True
        manager.get_calls.return_value = [MagicMock()]
        base_parser = ToolCallParser()
        manager.tool_call_status.side_effect = base_parser.tool_call_status
        event_manager = MagicMock()
        event_manager.trigger = AsyncMock()

        parser = ToolCallResponseParser(manager, event_manager)
        items = await parser.push("<tool_call>")

        self.assertIsInstance(items[0], ToolCallToken)
        self.assertEqual(items[1].type, EventType.TOOL_PROCESS)
        event_manager.trigger.assert_awaited_once()
        manager.get_calls.assert_called_once_with("<tool_call>")
        self.assertEqual(parser._buffer.getvalue(), "")
        self.assertFalse(parser._inside_call)

        self.assertEqual(await parser.flush(), [])

    async def test_trigger_without_calls(self):
        manager = MagicMock()
        manager.is_potential_tool_call.return_value = True
        manager.get_calls.return_value = None
        manager.tool_call_status.return_value = (
            ToolCallParser.ToolCallBufferStatus.NONE
        )
        event_manager = MagicMock()
        event_manager.trigger = AsyncMock()

        parser = ToolCallResponseParser(manager, event_manager)
        items = await parser.push("no_call")

        self.assertEqual(items, ["no_call"])
        event_manager.trigger.assert_awaited_once()
        manager.get_calls.assert_called_once_with("no_call")

    async def test_self_closing_tag(self):
        manager = MagicMock()
        manager.is_potential_tool_call.return_value = True
        manager.get_calls.return_value = [MagicMock()]
        base_parser = ToolCallParser()
        manager.tool_call_status.side_effect = base_parser.tool_call_status

        parser = ToolCallResponseParser(manager, None)
        items = await parser.push('<tool_call name="calc"/>')

        self.assertIsInstance(items[0], ToolCallToken)
        self.assertEqual(items[1].type, EventType.TOOL_PROCESS)
        manager.get_calls.assert_called_once_with('<tool_call name="calc"/>')
        self.assertFalse(parser._inside_call)
        self.assertEqual(parser._buffer.getvalue(), "")

    async def test_harmony_long_call_followed_by_final_channel(self):
        manager = MagicMock()
        manager.is_potential_tool_call.return_value = True

        def _get_calls(text: str):
            return [MagicMock()] if "<|call|>" in text else None

        manager.get_calls.side_effect = _get_calls
        manager.tool_format = ToolFormat.HARMONY
        base_parser = ToolCallParser(tool_format=ToolFormat.HARMONY)
        manager.tool_call_status.side_effect = base_parser.tool_call_status

        parser = ToolCallResponseParser(manager, None)
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

        self.assertIsInstance(tokens[0], ToolCallToken)
        self.assertNotIsInstance(tokens[-2], ToolCallToken)
        self.assertNotIsInstance(tokens[-1], ToolCallToken)
        self.assertEqual(tokens[-2], "<|channel|>final<|message|>")
        self.assertEqual(tokens[-1], "done")
        self.assertFalse(parser._inside_call)

    async def test_dsml_streaming_emits_tool_call_tokens_and_event(self):
        manager = MagicMock()
        manager.is_potential_tool_call.side_effect = lambda _buffer, token: (
            bool(token.strip())
        )
        manager.tool_format = ToolFormat.DSML
        base_parser = ToolCallParser(tool_format=ToolFormat.DSML)
        manager.tool_call_status.side_effect = base_parser.tool_call_status
        manager.get_calls.side_effect = base_parser

        parser = ToolCallResponseParser(manager, None)
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

        self.assertTrue(
            all(isinstance(token, ToolCallToken) for token in tokens[:-1])
        )
        self.assertEqual(tokens[-1].type, EventType.TOOL_PROCESS)
        self.assertFalse(parser._inside_call)

    async def test_harmony_flush_emits_event_for_missing_call_closure(self):
        manager = MagicMock()
        manager.is_potential_tool_call.return_value = True
        manager.tool_format = ToolFormat.HARMONY
        base_parser = ToolCallParser(tool_format=ToolFormat.HARMONY)
        manager.tool_call_status.side_effect = base_parser.tool_call_status
        manager.get_calls.side_effect = base_parser
        event_manager = MagicMock()
        event_manager.trigger = AsyncMock()

        parser = ToolCallResponseParser(manager, event_manager)
        text = (
            "<|start|>assistant<|channel|>commentary "
            "to=functions.browser.open <|constrain|>json<|message|>"
            '{"url":"https://example.com"}'
        )
        tokens = await parser.push(text)
        flushed = await parser.flush()

        self.assertEqual(len(tokens), 1)
        self.assertIsInstance(tokens[0], ToolCallToken)
        self.assertEqual(len(flushed), 1)
        event = flushed[0]
        self.assertEqual(event.type, EventType.TOOL_PROCESS)
        call = event.payload[0]
        self.assertEqual(call.name, "browser.open")
        self.assertEqual(call.arguments, {"url": "https://example.com"})
        manager.get_calls.assert_any_call(text + "<|call|>")
        trigger_event = event_manager.trigger.await_args_list[0].args[0]
        self.assertEqual(trigger_event.type, EventType.TOOL_DETECT)
        self.assertFalse(parser._inside_call)

    async def test_pending_tokens_flushed_on_status_none(self):
        manager = MagicMock()
        manager.is_potential_tool_call.return_value = False
        base_parser = ToolCallParser()
        manager.tool_call_status.side_effect = base_parser.tool_call_status

        parser = ToolCallResponseParser(manager, None)
        await parser.push("<to")
        result = await parser.push("x")

        self.assertEqual(result, ["<to", "x"])
        self.assertEqual(parser._pending_tokens, [])
        self.assertEqual(parser._pending_str, "")

    async def test_flush_returns_pending_tokens(self):
        manager = MagicMock()
        manager.is_potential_tool_call.return_value = False
        base_parser = ToolCallParser()
        manager.tool_call_status.side_effect = base_parser.tool_call_status

        parser = ToolCallResponseParser(manager, None)
        await parser.push("<tool")

        self.assertEqual(await parser.flush(), ["<tool"])
        self.assertEqual(parser._pending_tokens, [])
        self.assertEqual(parser._pending_str, "")

    async def test_open_status_returns_empty_without_tokens(self):
        manager = MagicMock()
        manager.is_potential_tool_call.return_value = True
        manager.tool_call_status.return_value = (
            ToolCallParser.ToolCallBufferStatus.OPEN
        )

        parser = ToolCallResponseParser(manager, None)

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
        parser = ToolCallResponseParser(manager, event_manager)

        items: list = []
        for part in (
            "<tool_call>",
            '{"name": "calculator", "arguments": }',
            "</tool_call>",
        ):
            items.extend(await parser.push(part))

        diagnostic_event = items[-1]
        self.assertEqual(diagnostic_event.type, EventType.TOOL_DIAGNOSTIC)
        diagnostics = diagnostic_event.payload["diagnostics"]
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(
            diagnostics[0].code,
            ToolCallDiagnosticCode.MALFORMED_CALL,
        )
        self.assertFalse(parser._inside_call)
        self.assertEqual(parser._buffer.getvalue(), "")
        self.assertEqual(
            event_manager.trigger.await_args_list[0].args[0].type,
            EventType.TOOL_DETECT,
        )
        self.assertTrue(
            any(
                call.args[0].type == EventType.TOOL_DIAGNOSTIC
                for call in event_manager.trigger.await_args_list
            )
        )

    async def test_flush_unterminated_stream_emits_diagnostic_event(self):
        manager = ToolManager.create_instance(enable_tools=[])
        parser = ToolCallResponseParser(manager, None)

        pushed = await parser.push(
            '<tool_call>{"name": "calculator", "arguments": {}}'
        )
        flushed = await parser.flush()

        self.assertTrue(
            all(isinstance(token, ToolCallToken) for token in pushed)
        )
        self.assertEqual(len(flushed), 1)
        diagnostic_event = flushed[0]
        self.assertEqual(diagnostic_event.type, EventType.TOOL_DIAGNOSTIC)
        diagnostics = diagnostic_event.payload["diagnostics"]
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(
            diagnostics[0].code,
            ToolCallDiagnosticCode.MALFORMED_CALL,
        )
        self.assertEqual(
            diagnostics[0].details["stream_status"], "unterminated"
        )
        self.assertFalse(parser._inside_call)
