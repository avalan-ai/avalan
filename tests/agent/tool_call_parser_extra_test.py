from avalan.model.response.parsers.tool import ToolCallResponseParser
from avalan.entities import ToolCallToken, ToolFormat
from avalan.tool.parser import ToolCallParser
from avalan.event import EventType
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock


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
