from logging import getLogger
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock

from avalan.entities import ReasoningSettings, ToolCallToken
from avalan.event import Event, EventType
from avalan.model.response.parsers.reasoning import (
    ReasoningParser,
    ReasoningToken,
    ReasoningTokenLimitExceeded,
)
from avalan.model.response.parsers.tool import ToolCallResponseParser
from avalan.tool.parser import ToolCallParser


class ReasoningParserAdditionalTestCase(IsolatedAsyncioTestCase):
    class _AlwaysStarts(str):
        def startswith(self, prefix):
            return True

    async def test_pending_tokens_and_budget(self) -> None:
        logger = getLogger("reasoning-test")
        settings = ReasoningSettings(
            max_new_tokens=2,
            stop_on_max_new_tokens=True,
        )
        parser = ReasoningParser(
            reasoning_settings=settings,
            logger=logger,
            bos_token="<|startoftext|>",
        )

        await parser.push("<|channel|>")
        await parser.push("analysis")
        flushed = await parser.push("noise")
        self.assertEqual(flushed, ["<|channel|>", "analysis", "noise"])

        for token in ("<|channel|>", "analysis", "<|message|>"):
            await parser.push(token)
        with self.assertRaises(ReasoningTokenLimitExceeded):
            await parser.push("thought")

    async def test_prefix_and_flush_behavior(self) -> None:
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(),
            logger=getLogger("reasoning-prefix"),
            prefixes=["Plan:"],
        )
        produced = await parser.push("Plan: consider")
        self.assertTrue(any(isinstance(t, ReasoningToken) for t in produced))

        parser._pending_tokens = [" keep", " going"]
        parser._pending_str = "keepgoing"
        parser._thinking = True
        flushed = await parser.flush()
        self.assertTrue(all(isinstance(t, ReasoningToken) for t in flushed))
        parser._thinking = False
        parser._pending_tokens = [" leftover"]
        parser._pending_str = "leftover"
        flushed_plain = await parser.flush()
        self.assertEqual(flushed_plain, [" leftover"])

    async def test_default_tag_allows_reasoning_tokens(self) -> None:
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(max_new_tokens=2),
            logger=getLogger("reasoning-default"),
        )
        await parser.push("<think>")
        tokens = await parser.push("thought")
        self.assertTrue(any(isinstance(t, ReasoningToken) for t in tokens))

    async def test_pending_str_trims_excess(self) -> None:
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(),
            logger=getLogger("reasoning-trim"),
        )
        parser._start_tag = self._AlwaysStarts(parser._start_tag)
        parser._pending_tokens = ["<think>", "extra"]
        parser._pending_str = "<think>extra"
        await parser.push("more")
        self.assertEqual(parser._pending_tokens, ["more"])

    async def test_budget_exceeded_without_stop(self) -> None:
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(
                max_new_tokens=0, stop_on_max_new_tokens=False
            ),
            logger=getLogger("reasoning-no-stop"),
        )
        parser.set_thinking(True)
        result = await parser.push("token")
        self.assertEqual(result, ["token"])

    async def test_pending_tokens_complete_start_tag(self) -> None:
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(),
            logger=getLogger("reasoning-complete"),
        )
        await parser.push("<think")
        tokens = await parser.push(">")
        self.assertIsInstance(tokens, list)

    async def test_pending_branch_trims_and_sets_thinking(self) -> None:
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(),
            logger=getLogger("reasoning-custom"),
        )

        class FakeTag:
            def __init__(self, value: str) -> None:
                self.value = value
                self._skip_first_equality = True

            def startswith(self, prefix: str) -> bool:
                return self.value.startswith(prefix)

            def __len__(self) -> int:
                return 1

            def __eq__(self, other: object) -> bool:
                if isinstance(other, FakeTag):
                    return self.value == other.value
                if isinstance(other, str) and other == self.value:
                    if self._skip_first_equality:
                        self._skip_first_equality = False
                        return False
                    return True
                return False

        parser._start_tag = FakeTag(parser._start_tag)
        result = await parser.push(parser._start_tag.value)
        self.assertEqual(result, [])
        self.assertTrue(parser.is_thinking)


class ToolCallResponseParserAdditionalTestCase(IsolatedAsyncioTestCase):
    async def test_emits_events_for_tool_calls(self) -> None:
        manager = MagicMock()
        manager.is_potential_tool_call.return_value = True
        statuses = iter(
            [
                ToolCallParser.ToolCallBufferStatus.PREFIX,
                ToolCallParser.ToolCallBufferStatus.OPEN,
                ToolCallParser.ToolCallBufferStatus.OPEN,
                ToolCallParser.ToolCallBufferStatus.CLOSED,
                ToolCallParser.ToolCallBufferStatus.CLOSED,
            ]
        )

        def status_side_effect(text: str):
            return next(statuses)

        manager.tool_call_status.side_effect = status_side_effect
        manager.get_calls.return_value = [SimpleNamespace(name="call")]
        event_manager = MagicMock()
        event_manager.trigger = AsyncMock()
        parser = ToolCallResponseParser(manager, event_manager)

        output = []
        output.extend(await parser.push("<"))
        output.extend(await parser.push("tool_call>"))
        output.extend(await parser.push('{"a":1}'))
        output.extend(await parser.push("</tool_call>"))

        self.assertTrue(any(isinstance(item, ToolCallToken) for item in output))
        event = next(item for item in output if isinstance(item, Event))
        self.assertEqual(event.type, EventType.TOOL_PROCESS)
        self.assertEqual(event.payload, [SimpleNamespace(name="call")])
        trigger_event = event_manager.trigger.await_args_list[0].args[0]
        self.assertEqual(trigger_event.type, EventType.TOOL_DETECT)

        parser._pending_tokens = ["rest"]
        parser._pending_str = "rest"
        flushed = await parser.flush()
        self.assertEqual(flushed, ["rest"])

    async def test_handles_non_matching_tokens(self) -> None:
        manager = MagicMock()
        manager.is_potential_tool_call.return_value = False
        manager.tool_call_status.return_value = ToolCallParser.ToolCallBufferStatus.NONE
        manager.get_calls.return_value = []
        parser = ToolCallResponseParser(manager, None)
        parser._pending_tokens = ["pending"]
        parser._pending_str = "pending"
        result = await parser.push("noise")
        self.assertEqual(result, ["pending", "noise"])
        self.assertEqual(parser._tag_buffer, "noise")
        result = await parser.push("a" * 70)
        self.assertEqual(result, ["a" * 70])
        self.assertEqual(len(parser._tag_buffer), 64)

    async def test_return_empty_result_when_list_ignores_appends(self) -> None:
        manager = MagicMock()
        manager.is_potential_tool_call.return_value = True
        manager.tool_call_status.return_value = ToolCallParser.ToolCallBufferStatus.OPEN

        class NonAppendingList(list):
            def append(self, value):  # type: ignore[override]
                return None

        parser = ToolCallResponseParser(manager, None)
        parser._pending_tokens = NonAppendingList()
        result = await parser.push("<tool_call>")
        self.assertEqual(result, [])
