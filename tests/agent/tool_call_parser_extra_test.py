from avalan.model.response.parsers.tool import ToolCallParser
from avalan.entities import ToolCallToken
from avalan.event import EventType
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock


class ToolCallParserExtraTestCase(IsolatedAsyncioTestCase):
    async def test_tag_buffer_trim_no_check(self):
        manager = MagicMock()
        manager.is_potential_tool_call.return_value = False
        parser = ToolCallParser(manager, None)

        long_token = "a" * 65
        result = await parser.push(long_token)

        self.assertEqual(result, [long_token])
        self.assertEqual(len(parser._tag_buffer), 64)

    async def test_trigger_and_event(self):
        manager = MagicMock()
        manager.is_potential_tool_call.return_value = True
        manager.get_calls.return_value = [MagicMock()]
        event_manager = MagicMock()
        event_manager.trigger = AsyncMock()

        parser = ToolCallParser(manager, event_manager)
        items = await parser.push("<tool_call>")

        self.assertIsInstance(items[0], ToolCallToken)
        self.assertEqual(items[1].type, EventType.TOOL_PROCESS)
        event_manager.trigger.assert_awaited_once()
        manager.get_calls.assert_called_once_with("<tool_call>")
        self.assertEqual(parser._buffer.getvalue(), "")
        self.assertFalse(parser._inside_call)

        self.assertEqual(await parser.flush(), [])
