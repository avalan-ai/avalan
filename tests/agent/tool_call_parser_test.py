from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import MagicMock

from avalan.entities import ToolFormat
from avalan.model.response.parsers.tool import ToolCallResponseParser
from avalan.model.stream import StreamItemKind, StreamProviderEvent
from avalan.tool.parser import ToolCallParser


def _call() -> SimpleNamespace:
    return SimpleNamespace(name="calc", arguments={})


def _kinds(items: list[object]) -> list[StreamItemKind]:
    return [
        item.kind for item in items if isinstance(item, StreamProviderEvent)
    ]


def _answer_text(items: list[object]) -> str:
    return "".join(
        item.text_delta or ""
        for item in items
        if (
            isinstance(item, StreamProviderEvent)
            and item.kind is StreamItemKind.ANSWER_DELTA
        )
    )


class ToolCallParserTestCase(IsolatedAsyncioTestCase):
    async def test_with_tool_call_tags(self):
        manager = MagicMock()

        def _get_calls(text: str):
            return [_call()] if "</tool_call>" in text else None

        manager.is_potential_tool_call.return_value = True
        manager.get_calls.side_effect = _get_calls
        base_parser = ToolCallParser()
        manager.tool_call_status.side_effect = base_parser.tool_call_status

        parser = ToolCallResponseParser(manager, None)
        tokens: list[object] = []
        for t in ["<tool_call>", "x", "</tool_call>", "y"]:
            tokens.extend(await parser.push(t))

        self.assertIn(StreamItemKind.TOOL_CALL_READY, _kinds(tokens))
        self.assertIn(StreamItemKind.TOOL_CALL_DONE, _kinds(tokens))
        self.assertEqual(_answer_text(tokens), "y")

    async def test_harmony_format_tokens(self):
        manager = MagicMock()

        def _get_calls(text: str):
            return [_call()] if "<|call|>" in text else None

        manager.is_potential_tool_call.return_value = True
        manager.get_calls.side_effect = _get_calls
        manager.tool_format = ToolFormat.HARMONY
        base_parser = ToolCallParser(tool_format=ToolFormat.HARMONY)
        manager.tool_call_status.side_effect = base_parser.tool_call_status

        parser = ToolCallResponseParser(manager, None)
        tokens: list[object] = []
        parts = [
            "<|channel|>",
            "commentary to=functions.db.run code<|message|>{}",
            "<|call|>",
            "end",
        ]
        for part in parts:
            tokens.extend(await parser.push(part))

        self.assertIn(StreamItemKind.TOOL_CALL_READY, _kinds(tokens))
        self.assertIn(StreamItemKind.TOOL_CALL_DONE, _kinds(tokens))
        self.assertEqual(_answer_text(tokens), "end")

    async def test_harmony_final_channel_marker_closes_without_visible_leak(
        self,
    ):
        manager = MagicMock()

        def _get_calls(text: str):
            return [_call()] if "<|call|>" in text else None

        manager.is_potential_tool_call.return_value = True
        manager.get_calls.side_effect = _get_calls
        manager.tool_format = ToolFormat.HARMONY
        base_parser = ToolCallParser(tool_format=ToolFormat.HARMONY)
        manager.tool_call_status.side_effect = base_parser.tool_call_status

        parser = ToolCallResponseParser(manager, None)
        tokens: list[object] = []
        parts = [
            "<|channel|>",
            "commentary to=functions.db.run code<|message|>{}",
            "<|channel|>final<|message|>done",
        ]
        for part in parts:
            tokens.extend(await parser.push(part))

        self.assertIn(StreamItemKind.TOOL_CALL_READY, _kinds(tokens))
        self.assertIn(StreamItemKind.TOOL_CALL_DONE, _kinds(tokens))
        self.assertEqual(_answer_text(tokens), "done")

    async def test_harmony_format_tokens_analysis(self):
        manager = MagicMock()

        def _get_calls(text: str):
            return [_call()] if "<|call|>" in text else None

        manager.is_potential_tool_call.return_value = True
        manager.get_calls.side_effect = _get_calls
        manager.tool_format = ToolFormat.HARMONY
        base_parser = ToolCallParser(tool_format=ToolFormat.HARMONY)
        manager.tool_call_status.side_effect = base_parser.tool_call_status

        parser = ToolCallResponseParser(manager, None)
        tokens: list[object] = []
        parts = [
            "<|channel|>",
            "analysis to=functions.db.inspect code<|message|>{}",
            "<|call|>",
            "end",
        ]
        for part in parts:
            tokens.extend(await parser.push(part))

        self.assertIn(StreamItemKind.TOOL_CALL_READY, _kinds(tokens))
        self.assertIn(StreamItemKind.TOOL_CALL_DONE, _kinds(tokens))
        self.assertEqual(_answer_text(tokens), "end")

    async def test_harmony_format_tokens_with_prefix(self):
        manager = MagicMock()

        def _get_calls(text: str):
            return [_call()] if "<|call|>" in text else None

        manager.is_potential_tool_call.return_value = True
        manager.get_calls.side_effect = _get_calls
        manager.tool_format = ToolFormat.HARMONY
        base_parser = ToolCallParser(tool_format=ToolFormat.HARMONY)
        manager.tool_call_status.side_effect = base_parser.tool_call_status

        parser = ToolCallResponseParser(manager, None)
        tokens: list[object] = []
        parts = [
            "<|start|>",
            "assistant<|channel|>commentary to=functions.db.run code",
            "<|message|>{}",
            "<|call|>",
            "end",
        ]
        for part in parts:
            tokens.extend(await parser.push(part))

        self.assertIn(StreamItemKind.TOOL_CALL_READY, _kinds(tokens))
        self.assertIn(StreamItemKind.TOOL_CALL_DONE, _kinds(tokens))
        self.assertEqual(_answer_text(tokens), "end")
