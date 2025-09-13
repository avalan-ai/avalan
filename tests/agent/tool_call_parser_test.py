from avalan.model.response.parsers.tool import ToolCallResponseParser
from avalan.entities import ToolCallToken, ToolFormat
from avalan.tool.parser import ToolCallParser
from unittest import IsolatedAsyncioTestCase
from unittest.mock import MagicMock


class ToolCallParserTestCase(IsolatedAsyncioTestCase):
    async def test_with_tool_call_tags(self):
        manager = MagicMock()

        def _get_calls(text: str):
            return [MagicMock()] if "</tool_call>" in text else None

        manager.is_potential_tool_call.return_value = True
        manager.get_calls.side_effect = _get_calls
        base_parser = ToolCallParser()
        manager.tool_call_status.side_effect = base_parser.tool_call_status

        parser = ToolCallResponseParser(manager, None)
        tokens = []
        for t in ["<tool_call>", "x", "</tool_call>", "y"]:
            tokens.extend(await parser.push(t))

        self.assertIsInstance(tokens[0], ToolCallToken)
        self.assertIsInstance(tokens[1], ToolCallToken)
        self.assertIsInstance(tokens[2], ToolCallToken)
        self.assertEqual(tokens[-1], "y")

    async def test_harmony_format_tokens(self):
        manager = MagicMock()

        def _get_calls(text: str):
            return [MagicMock()] if "<|call|>" in text else None

        manager.is_potential_tool_call.return_value = True
        manager.get_calls.side_effect = _get_calls
        manager.tool_format = ToolFormat.HARMONY
        base_parser = ToolCallParser(tool_format=ToolFormat.HARMONY)
        manager.tool_call_status.side_effect = base_parser.tool_call_status

        parser = ToolCallResponseParser(manager, None)
        tokens: list = []
        parts = [
            "<|channel|>",
            "commentary to=functions.db.run code<|message|>{}",
            "<|call|>",
            "end",
        ]
        for part in parts:
            tokens.extend(await parser.push(part))

        self.assertIsInstance(tokens[0], ToolCallToken)
        self.assertIsInstance(tokens[1], ToolCallToken)
        self.assertIsInstance(tokens[2], ToolCallToken)
        self.assertEqual(tokens[-1], "end")

    async def test_harmony_format_tokens_analysis(self):
        manager = MagicMock()

        def _get_calls(text: str):
            return [MagicMock()] if "<|call|>" in text else None

        manager.is_potential_tool_call.return_value = True
        manager.get_calls.side_effect = _get_calls
        manager.tool_format = ToolFormat.HARMONY
        base_parser = ToolCallParser(tool_format=ToolFormat.HARMONY)
        manager.tool_call_status.side_effect = base_parser.tool_call_status

        parser = ToolCallResponseParser(manager, None)
        tokens: list = []
        parts = [
            "<|channel|>",
            "analysis to=functions.db.inspect code<|message|>{}",
            "<|call|>",
            "end",
        ]
        for part in parts:
            tokens.extend(await parser.push(part))

        self.assertIsInstance(tokens[0], ToolCallToken)
        self.assertIsInstance(tokens[1], ToolCallToken)
        self.assertIsInstance(tokens[2], ToolCallToken)
        self.assertEqual(tokens[-1], "end")

    async def test_harmony_format_tokens_with_prefix(self):
        manager = MagicMock()

        def _get_calls(text: str):
            return [MagicMock()] if "<|call|>" in text else None

        manager.is_potential_tool_call.return_value = True
        manager.get_calls.side_effect = _get_calls
        manager.tool_format = ToolFormat.HARMONY
        base_parser = ToolCallParser(tool_format=ToolFormat.HARMONY)
        manager.tool_call_status.side_effect = base_parser.tool_call_status

        parser = ToolCallResponseParser(manager, None)
        tokens: list = []
        parts = [
            "<|start|>",
            "assistant<|channel|>commentary to=functions.db.run code",
            "<|message|>{}",
            "<|call|>",
            "end",
        ]
        for part in parts:
            tokens.extend(await parser.push(part))

        self.assertIsInstance(tokens[0], ToolCallToken)
        self.assertIsInstance(tokens[1], ToolCallToken)
        self.assertIsInstance(tokens[2], ToolCallToken)
        self.assertEqual(tokens[-1], "end")
