from collections.abc import AsyncIterator, Iterable
from datetime import datetime, timezone
from logging import getLogger
from types import SimpleNamespace
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock

from avalan.entities import (
    ReasoningSettings,
    ReasoningToken,
    ToolCallDiagnostic,
    ToolCallDiagnosticCode,
    ToolCallDiagnosticStage,
    ToolCallToken,
    ToolFormat,
)
from avalan.event import Event
from avalan.model.response.parsers.reasoning import (
    ReasoningParser,
    ReasoningTokenLimitExceeded,
)
from avalan.model.response.parsers.tool import (
    ToolCallResponseParser,
    ToolCallResponseParserOutput,
)
from avalan.model.stream import (
    StreamItemKind,
    StreamProviderEvent,
    StreamReasoningRepresentation,
    StreamVisibility,
    normalize_provider_stream,
)
from avalan.tool.manager import ToolManager
from avalan.tool.parser import ToolCallParser


def _output_text(item: object) -> str | None:
    if isinstance(item, StreamProviderEvent):
        return item.text_delta
    assert isinstance(item, str)
    return item


def _is_reasoning_event(item: object) -> bool:
    return isinstance(item, StreamProviderEvent) and item.kind in {
        StreamItemKind.REASONING_DELTA,
        StreamItemKind.REASONING_DONE,
    }


def _is_reasoning_delta(item: object) -> bool:
    return (
        isinstance(item, StreamProviderEvent)
        and item.kind is StreamItemKind.REASONING_DELTA
    )


def _is_tool_event(item: object) -> bool:
    return isinstance(item, StreamProviderEvent) and item.kind in {
        StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
        StreamItemKind.TOOL_CALL_READY,
        StreamItemKind.TOOL_CALL_DONE,
        StreamItemKind.STREAM_DIAGNOSTIC,
    }


def _answer_delta_texts(items: Iterable[object]) -> list[str | None]:
    return [
        item.text_delta
        for item in items
        if (
            isinstance(item, StreamProviderEvent)
            and item.kind is StreamItemKind.ANSWER_DELTA
        )
    ]


async def _event_stream(
    events: list[StreamProviderEvent],
) -> AsyncIterator[StreamProviderEvent]:
    for event in events:
        yield event


class ReasoningParserAdditionalTestCase(IsolatedAsyncioTestCase):
    class _AlwaysStarts(str):
        def startswith(
            self, prefix: str | tuple[str, ...], start: int = 0, end: int = -1
        ) -> bool:
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
        self.assertTrue(any(_is_reasoning_delta(t) for t in produced))

        parser._pending_tokens = [" keep", " going"]
        parser._pending_str = "keepgoing"
        parser._thinking = True
        flushed = list(await parser.flush())
        self.assertTrue(all(_is_reasoning_event(t) for t in flushed))
        last_flushed = flushed[-1]
        self.assertIsInstance(last_flushed, StreamProviderEvent)
        assert isinstance(last_flushed, StreamProviderEvent)
        self.assertIs(last_flushed.kind, StreamItemKind.REASONING_DELTA)
        self.assertIs(
            last_flushed.reasoning_representation,
            StreamReasoningRepresentation.NATIVE_TEXT,
        )
        self.assertEqual(last_flushed.segment_instance_ordinal, 0)
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
        self.assertTrue(any(_is_reasoning_delta(t) for t in tokens))

    async def test_split_markers_preserve_surrounding_whitespace(
        self,
    ) -> None:
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(),
            logger=getLogger("reasoning-whitespace"),
        )
        output: list[Any] = []

        for token in (
            "lead ",
            " <thi",
            "nk> ",
            " private ",
            " </thi",
            "nk> ",
            "tail",
        ):
            output.extend(await parser.push(token))
        output.extend(await parser.flush())

        self.assertEqual(
            [_output_text(item) for item in output],
            [
                "lead ",
                " ",
                "<thi",
                "nk>",
                " ",
                " private ",
                " ",
                "</thi",
                "nk>",
                " ",
                "tail",
            ],
        )
        self.assertEqual(
            [_is_reasoning_event(item) for item in output],
            [
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
            ],
        )

    async def test_split_marker_false_positive_remains_visible(self) -> None:
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(),
            logger=getLogger("reasoning-false-positive"),
        )
        output: list[Any] = []

        for token in ("<", " think", "> visible"):
            output.extend(await parser.push(token))
        output.extend(await parser.flush())

        self.assertEqual(
            "".join(_output_text(item) or "" for item in output),
            "< think> visible",
        )
        self.assertFalse(any(_is_reasoning_event(item) for item in output))

    async def test_pending_marker_whitespace_flushes_on_current_side(
        self,
    ) -> None:
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(),
            logger=getLogger("reasoning-pending-space"),
        )

        await parser.push("<")
        visible = await parser.push(" ")

        self.assertEqual(visible, ["<", " "])
        self.assertFalse(parser.is_thinking)

        parser.set_thinking(True)
        await parser.push("<")
        private = await parser.push(" ")

        self.assertEqual(
            [_output_text(item) for item in private],
            ["<", " "],
        )
        self.assertTrue(all(_is_reasoning_delta(item) for item in private))

    async def test_embedded_markers_preserve_prefixes_and_suffixes(
        self,
    ) -> None:
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(),
            logger=getLogger("reasoning-embedded"),
        )

        output = await parser.push("lead <think>hidden</think> tail")

        self.assertEqual(
            [_output_text(item) for item in output],
            ["lead ", "<think>", "hidden", "</think>", " tail"],
        )
        self.assertEqual(
            [_is_reasoning_event(item) for item in output],
            [False, True, True, True, False],
        )
        self.assertFalse(parser.is_thinking)

    async def test_adjacent_embedded_reasoning_sections_do_not_leak(
        self,
    ) -> None:
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(),
            logger=getLogger("reasoning-adjacent"),
        )

        output = await parser.push("x<think>a</think><think>b</think>y")

        self.assertEqual(
            [_output_text(item) for item in output],
            [
                "x",
                "<think>",
                "a",
                "</think>",
                "<think>",
                "b",
                "</think>",
                "y",
            ],
        )
        self.assertEqual(
            [_is_reasoning_event(item) for item in output],
            [
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
            ],
        )
        self.assertEqual(
            "".join(item for item in output if isinstance(item, str)),
            "xy",
        )

    async def test_adjacent_split_reasoning_sections_do_not_leak(
        self,
    ) -> None:
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(),
            logger=getLogger("reasoning-adjacent-split"),
        )
        output: list[Any] = []

        for token in tuple("x<think>a</think><think>b</think>y"):
            output.extend(await parser.push(token))
        output.extend(await parser.flush())

        public = "".join(item for item in output if isinstance(item, str))
        private = "".join(
            item.text_delta or ""
            for item in output
            if _is_reasoning_delta(item)
        )

        self.assertEqual(public, "xy")
        self.assertEqual(private, "<think>a</think><think>b</think>")

    async def test_thinking_partial_end_marker_keeps_prefix_private(
        self,
    ) -> None:
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(),
            logger=getLogger("reasoning-partial-end"),
        )
        parser.set_thinking(True)

        output = await parser.push("private</thi")

        self.assertEqual(
            [_output_text(item) for item in output],
            ["private"],
        )
        self.assertTrue(all(_is_reasoning_delta(item) for item in output))
        self.assertEqual(parser._pending_tokens, ["</thi"])
        self.assertEqual(parser._pending_str, "</thi")

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

        fake_tag = FakeTag(parser._start_tag)
        cast(Any, parser)._start_tag = fake_tag
        result = await parser.push(fake_tag.value)
        self.assertEqual([_output_text(item) for item in result], ["<think>"])
        self.assertTrue(all(_is_reasoning_delta(item) for item in result))
        self.assertTrue(parser.is_thinking)

    async def test_default_reasoning_output_is_not_legacy_token(
        self,
    ) -> None:
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(),
            logger=getLogger("reasoning-canonical-negative"),
        )

        output: list[Any] = []
        output.extend(await parser.push("<think>"))
        output.extend(await parser.push("private"))
        output.extend(await parser.push("</think>"))

        self.assertTrue(any(_is_reasoning_delta(item) for item in output))
        self.assertTrue(
            all(
                item.visibility is StreamVisibility.PRIVATE
                for item in output
                if _is_reasoning_event(item)
            )
        )
        self.assertFalse(
            any(isinstance(item, ReasoningToken) for item in output)
        )
        self.assertFalse(any(isinstance(item, Event) for item in output))

    async def test_flush_open_reasoning_without_delta_is_invisible(
        self,
    ) -> None:
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(),
            logger=getLogger("reasoning-empty-done"),
        )
        parser.set_thinking(True)

        output = list(await parser.flush())

        self.assertEqual(output, [])


class ToolCallResponseParserAdditionalTestCase(IsolatedAsyncioTestCase):
    async def test_default_runtime_output_is_canonical_only(self) -> None:
        manager = ToolManager(parser=ToolCallParser())
        parser = ToolCallResponseParser(manager, None)

        output = list(
            await parser.push(
                '<tool_call>{"name":"calc","arguments":{"x":1}}</tool_call>'
            )
        )

        self.assertTrue(output)
        self.assertTrue(parser.canonicalizes_answer_deltas)
        self.assertTrue(
            all(
                isinstance(item, ToolCallResponseParserOutput)
                for item in output
            )
        )
        self.assertFalse(
            any(isinstance(item, (Event, ToolCallToken)) for item in output)
        )

    async def test_parser_still_emits_canonical_output(
        self,
    ) -> None:
        manager = ToolManager(parser=ToolCallParser())
        parser = ToolCallResponseParser(
            manager,
            None,
        )

        output = list(
            await parser.push(
                '<tool_call>{"name":"calc","arguments":{"x":1}}</tool_call>'
            )
        )

        self.assertTrue(parser.canonicalizes_answer_deltas)
        self.assertTrue(
            all(isinstance(item, StreamProviderEvent) for item in output)
        )
        self.assertFalse(
            any(isinstance(item, (Event, ToolCallToken)) for item in output)
        )

    async def test_emits_events_for_tool_calls(self) -> None:
        base_parser = ToolCallParser()
        manager = MagicMock()
        manager.is_potential_tool_call.return_value = True
        manager.tool_call_status.side_effect = base_parser.tool_call_status
        manager.get_calls.return_value = [
            SimpleNamespace(name="call", arguments={"a": 1})
        ]
        event_manager = MagicMock()
        event_manager.trigger = AsyncMock()
        parser = ToolCallResponseParser(manager, event_manager)

        output: list[Any] = []
        output.extend(await parser.push("<tool_call>"))
        output.extend(await parser.push('{"a":1}'))
        output.extend(await parser.push("</tool_call>"))

        self.assertFalse(
            any(isinstance(item, ToolCallToken) for item in output)
        )
        self.assertFalse(any(isinstance(item, Event) for item in output))
        self.assertEqual(
            [
                item.kind
                for item in output
                if isinstance(item, StreamProviderEvent)
                and _is_tool_event(item)
            ],
            [
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
            ],
        )
        argument_delta = next(
            item
            for item in output
            if (
                isinstance(item, StreamProviderEvent)
                and item.kind is StreamItemKind.TOOL_CALL_ARGUMENT_DELTA
            )
        )
        self.assertEqual(argument_delta.text_delta, '{"a":1}')
        tool_call_ids = {
            item.correlation.tool_call_id
            for item in output
            if isinstance(item, StreamProviderEvent) and _is_tool_event(item)
        }
        self.assertEqual(len(tool_call_ids), 1)
        ready = next(
            item
            for item in output
            if (
                isinstance(item, StreamProviderEvent)
                and item.kind is StreamItemKind.TOOL_CALL_READY
            )
        )
        self.assertIsInstance(ready, StreamProviderEvent)
        assert isinstance(ready, StreamProviderEvent)
        self.assertEqual(ready.data, {"name": "call", "arguments": {"a": 1}})
        event_manager.trigger.assert_not_awaited()

        parser._pending_tokens = ["rest"]
        parser._pending_str = "rest"
        flushed = await parser.flush()
        self.assertEqual(_answer_delta_texts(flushed), ["rest"])

    async def test_visible_text_around_tool_call_is_answer_delta_event(
        self,
    ) -> None:
        manager = ToolManager(parser=ToolCallParser())
        parser = ToolCallResponseParser(manager, None)
        text = (
            "visible before "
            '<tool_call>{"name":"calc","arguments":{"x":1}}</tool_call>'
            " visible after"
        )

        output = list(await parser.push(text))
        output.extend(await parser.flush())

        self.assertFalse(any(isinstance(item, str) for item in output))
        self.assertFalse(
            any(isinstance(item, ToolCallToken) for item in output)
        )
        self.assertEqual(
            _answer_delta_texts(output),
            ["visible before ", " visible after"],
        )
        self.assertNotIn("<tool_call", "".join(_answer_delta_texts(output)))
        self.assertEqual(
            [
                item.kind
                for item in output
                if isinstance(item, StreamProviderEvent)
                and _is_tool_event(item)
            ],
            [
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
            ],
        )

    async def test_valid_tool_call_preserves_explicit_id(self) -> None:
        manager = ToolManager(parser=ToolCallParser())
        parser = ToolCallResponseParser(manager, None)
        text = (
            '<tool_call>{"id":"model-call-1","name":"calc",'
            '"arguments":{"x":1}}</tool_call>'
        )

        output = list(await parser.push(text))

        provider_events = [
            event for event in output if isinstance(event, StreamProviderEvent)
        ]
        self.assertEqual(
            {
                event.correlation.tool_call_id
                for event in provider_events
                if event.correlation.tool_call_id is not None
            },
            {"model-call-1"},
        )
        argument_delta = next(
            event
            for event in provider_events
            if event.kind is StreamItemKind.TOOL_CALL_ARGUMENT_DELTA
        )
        self.assertEqual(argument_delta.text_delta, '{"x":1}')
        self.assertTrue(
            [
                item
                async for item in normalize_provider_stream(
                    _event_stream(provider_events),
                    stream_session_id="parser-stream",
                    run_id="parser-run",
                    turn_id="parser-turn",
                    provider_family="local",
                )
            ]
        )

    async def test_streams_argument_delta_before_close_marker(self) -> None:
        manager = ToolManager(parser=ToolCallParser())
        parser = ToolCallResponseParser(manager, None)

        first = list(
            await parser.push('<tool_call>{"name":"calc","arguments":')
        )
        second = list(await parser.push('{"x":1}}'))
        third = list(await parser.push("</tool_call>"))

        self.assertEqual(first, [])
        self.assertEqual(
            [
                event.kind
                for event in second
                if isinstance(event, StreamProviderEvent)
            ],
            [StreamItemKind.TOOL_CALL_ARGUMENT_DELTA],
        )
        argument_delta = second[0]
        self.assertIsInstance(argument_delta, StreamProviderEvent)
        assert isinstance(argument_delta, StreamProviderEvent)
        self.assertEqual(argument_delta.text_delta, '{"x":1}')
        self.assertNotIn(
            StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
            [
                event.kind
                for event in third
                if isinstance(event, StreamProviderEvent)
            ],
        )
        ready = next(
            event
            for event in third
            if (
                isinstance(event, StreamProviderEvent)
                and event.kind is StreamItemKind.TOOL_CALL_READY
            )
        )
        done = next(
            event
            for event in third
            if (
                isinstance(event, StreamProviderEvent)
                and event.kind is StreamItemKind.TOOL_CALL_DONE
            )
        )
        self.assertEqual(
            [
                argument_delta.correlation.tool_call_id,
                ready.correlation.tool_call_id,
                done.correlation.tool_call_id,
            ],
            ["parser-tool-call-1"] * 3,
        )

    async def test_tool_like_closed_suffix_remains_visible(self) -> None:
        base_parser = ToolCallParser()
        tool_text = '<tool_call>{"name":"calc","arguments":{}}</tool_call>'

        for visible_suffix in (
            " after <tool_callout> text",
            " after <tool_callout/> text",
        ):
            with self.subTest(visible_suffix=visible_suffix):
                manager = MagicMock()
                manager.tool_format = None
                manager.is_potential_tool_call.return_value = True
                manager.tool_call_status.side_effect = (
                    base_parser.tool_call_status
                )
                manager.get_calls.return_value = [
                    SimpleNamespace(name="call", arguments={})
                ]
                parser = ToolCallResponseParser(manager, None)

                output = list(await parser.push(tool_text + visible_suffix))

                self.assertIsInstance(output[0], StreamProviderEvent)
                self.assertIs(
                    output[0].kind,
                    StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                )
                self.assertEqual(output[0].text_delta, "{}")
                self.assertFalse(
                    any(isinstance(item, ToolCallToken) for item in output)
                )
                self.assertFalse(
                    any(isinstance(item, Event) for item in output)
                )
                self.assertTrue(
                    any(
                        isinstance(item, StreamProviderEvent)
                        and item.kind is StreamItemKind.TOOL_CALL_READY
                        for item in output
                    )
                )
                self.assertIsInstance(output[-1], StreamProviderEvent)
                suffix = cast(StreamProviderEvent, output[-1])
                self.assertIs(suffix.kind, StreamItemKind.ANSWER_DELTA)
                self.assertEqual(suffix.text_delta, visible_suffix)
                self.assertEqual(await parser.flush(), [])

    async def test_split_tool_like_prefix_remains_visible(self) -> None:
        base_parser = ToolCallParser()
        manager = MagicMock()
        manager.tool_format = None
        manager.is_potential_tool_call.return_value = True
        manager.tool_call_status.side_effect = base_parser.tool_call_status
        manager.get_calls.return_value = []
        parser = ToolCallResponseParser(manager, None)

        output: list[Any] = []
        for token in ("before ", "<tool_call", "out> text"):
            output.extend(await parser.push(token))
        output.extend(await parser.flush())

        self.assertEqual(
            _answer_delta_texts(output),
            ["before ", "<tool_call", "out> text"],
        )
        self.assertFalse(
            any(isinstance(item, ToolCallToken) for item in output)
        )
        self.assertEqual(
            list(parser._self_closing_tool_close_spans("<tool_call name")),
            [],
        )

    async def test_malformed_tool_call_emits_canonical_diagnostic(
        self,
    ) -> None:
        manager = ToolManager(parser=ToolCallParser())
        parser = ToolCallResponseParser(manager, None)

        output: list[Any] = []
        output.extend(await parser.push("<tool_call>"))
        output.extend(await parser.push('{"name": "calc", "arguments": []}'))
        output.extend(await parser.push("</tool_call>"))

        self.assertFalse(
            any(isinstance(item, ToolCallToken) for item in output)
        )
        self.assertFalse(any(isinstance(item, Event) for item in output))
        diagnostic = next(
            item
            for item in output
            if (
                isinstance(item, StreamProviderEvent)
                and item.kind is StreamItemKind.STREAM_DIAGNOSTIC
            )
        )
        tool_call_ids = {
            item.correlation.tool_call_id
            for item in output
            if isinstance(item, StreamProviderEvent)
            and item.correlation.tool_call_id is not None
        }
        self.assertEqual(len(tool_call_ids), 1)
        self.assertIsInstance(diagnostic.data, dict)
        assert isinstance(diagnostic.data, dict)
        self.assertEqual(diagnostic.data["code"], "tool_call.malformed")
        self.assertEqual(diagnostic.data["tool_call_id"], "parser-tool-call-1")
        self.assertIs(diagnostic.visibility, StreamVisibility.DIAGNOSTIC)

    async def test_malformed_and_valid_siblings_use_distinct_ids(
        self,
    ) -> None:
        manager = ToolManager(parser=ToolCallParser())
        parser = ToolCallResponseParser(manager, None)
        text = (
            '<tool_call>{"id":"bad-call-1","name":"bad",'
            '"arguments":[]}</tool_call>'
            '<tool_call>{"id":"good-call-1","name":"calc",'
            '"arguments":{"x":1}}</tool_call>'
        )

        events = list(await parser.push(text))
        provider_events = [
            event for event in events if isinstance(event, StreamProviderEvent)
        ]
        canonical_items = [
            item
            async for item in normalize_provider_stream(
                _event_stream(provider_events),
                stream_session_id="parser-stream",
                run_id="parser-run",
                turn_id="parser-turn",
                provider_family="local",
            )
        ]

        diagnostic_event = provider_events[0]
        ready_event = next(
            event
            for event in provider_events
            if event.kind is StreamItemKind.TOOL_CALL_READY
        )
        self.assertEqual(
            diagnostic_event.correlation.tool_call_id,
            "bad-call-1",
        )
        self.assertIsInstance(diagnostic_event.data, dict)
        assert isinstance(diagnostic_event.data, dict)
        self.assertEqual(
            diagnostic_event.data["tool_call_id"],
            "bad-call-1",
        )
        diagnostic_data = diagnostic_event.data["diagnostics"][0]
        self.assertEqual(diagnostic_data["call_id"], "bad-call-1")
        self.assertEqual(
            ready_event.correlation.tool_call_id,
            "good-call-1",
        )
        self.assertNotEqual(
            diagnostic_event.correlation.tool_call_id,
            ready_event.correlation.tool_call_id,
        )
        self.assertTrue(canonical_items)

    async def test_malformed_siblings_emit_separate_diagnostic_ids(
        self,
    ) -> None:
        manager = ToolManager(parser=ToolCallParser())
        parser = ToolCallResponseParser(manager, None)
        text = (
            '<tool_call>{"id":"bad-call-1","name":"bad",'
            '"arguments":[]}</tool_call>'
            '<tool_call>{"id":"bad-call-2","name":"bad",'
            '"arguments":[]}</tool_call>'
        )

        events = [
            event
            for event in await parser.push(text)
            if isinstance(event, StreamProviderEvent)
        ]
        diagnostics = [
            event
            for event in events
            if event.kind is StreamItemKind.STREAM_DIAGNOSTIC
        ]

        self.assertEqual(
            [event.correlation.tool_call_id for event in diagnostics],
            ["bad-call-1", "bad-call-2"],
        )
        for diagnostic in diagnostics:
            self.assertIsInstance(diagnostic.data, dict)
            assert isinstance(diagnostic.data, dict)
            self.assertEqual(
                diagnostic.data["tool_call_id"],
                diagnostic.correlation.tool_call_id,
            )
            nested = diagnostic.data["diagnostics"]
            self.assertEqual(len(nested), 1)
            self.assertEqual(
                nested[0]["call_id"], diagnostic.correlation.tool_call_id
            )

    async def test_malformed_tool_call_preserves_explicit_id(self) -> None:
        manager = ToolManager(parser=ToolCallParser())
        parser = ToolCallResponseParser(manager, None)
        text = (
            '<tool_call>{"id":"bad-call-1","name":"calc",'
            '"arguments":[]}</tool_call>'
        )

        output = list(await parser.push(text))

        diagnostic = next(
            item
            for item in output
            if (
                isinstance(item, StreamProviderEvent)
                and item.kind is StreamItemKind.STREAM_DIAGNOSTIC
            )
        )
        self.assertEqual(
            diagnostic.correlation.tool_call_id,
            "bad-call-1",
        )
        self.assertIsInstance(diagnostic.data, dict)
        assert isinstance(diagnostic.data, dict)
        self.assertEqual(diagnostic.data["tool_call_id"], "bad-call-1")
        nested_diagnostics = diagnostic.data["diagnostics"]
        self.assertEqual(nested_diagnostics[0]["call_id"], "bad-call-1")

    async def test_canonical_diagnostic_skips_event_manager(self) -> None:
        manager = ToolManager(parser=ToolCallParser())
        event_manager = MagicMock()
        event_manager.trigger = AsyncMock()
        parser = ToolCallResponseParser(manager, event_manager)

        output = await parser.push(
            '<tool_call>{"id":"bad-call-1","name":"calc",'
            '"arguments":[]}</tool_call>'
        )

        diagnostic_events = [
            event
            for event in output
            if event.kind is StreamItemKind.STREAM_DIAGNOSTIC
        ]
        event_manager.trigger.assert_not_awaited()
        self.assertEqual(len(diagnostic_events), 1)
        diagnostics = diagnostic_events[0].data["diagnostics"]
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0]["call_id"], "bad-call-1")

    async def test_canonical_diagnostic_includes_optional_fields(
        self,
    ) -> None:
        parser = ToolCallResponseParser(MagicMock(), None)
        started = datetime(2026, 1, 1, tzinfo=timezone.utc)
        finished = datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
        diagnostic = ToolCallDiagnostic(
            id="diagnostic-1",
            call_id="call-1",
            requested_name="bad",
            canonical_name="calc",
            code=ToolCallDiagnosticCode.UNKNOWN_TOOL,
            stage=ToolCallDiagnosticStage.RESOLVE,
            message="Unknown tool.",
            retryable=True,
            details={"reason": "missing"},
            started_at=started,
            finished_at=finished,
            duration_ms=7,
        )

        event = await parser._diagnostic_event([diagnostic])

        self.assertIsInstance(event, StreamProviderEvent)
        assert isinstance(event, StreamProviderEvent)
        self.assertIsInstance(event.data, dict)
        assert isinstance(event.data, dict)
        diagnostics = event.data["diagnostics"]
        self.assertIsInstance(diagnostics, list)
        diagnostic_data = diagnostics[0]
        self.assertEqual(diagnostic_data["call_id"], "call-1")
        self.assertEqual(diagnostic_data["requested_name"], "bad")
        self.assertEqual(diagnostic_data["canonical_name"], "calc")
        self.assertEqual(diagnostic_data["duration_ms"], 7)
        self.assertEqual(diagnostic_data["started_at"], started.isoformat())
        self.assertEqual(diagnostic_data["finished_at"], finished.isoformat())

    def test_empty_argument_delta_text_handles_missing_arguments(self) -> None:
        self.assertEqual(
            ToolCallResponseParser._tool_argument_delta_text(
                SimpleNamespace(arguments=None)
            ),
            "",
        )

    def test_stream_argument_helper_edges(self) -> None:
        parser = ToolCallResponseParser(MagicMock(), None)

        self.assertEqual(
            parser._tool_call_id_for_call(
                SimpleNamespace(id=None), first_call=True
            ),
            "parser-tool-call-1",
        )
        self.assertIsNone(parser._diagnostic_tool_call_id([]))
        self.assertEqual(
            parser._remaining_tool_argument_delta_text("call-1", ""), ""
        )
        self.assertIsNone(parser._current_explicit_tool_call_id("plain"))
        self.assertIsNone(parser._stream_tool_argument_text("plain"))
        self.assertIsNone(
            parser._stream_tool_argument_text(
                '<tool_call>{"name":"bad..name","arguments":{}}'
            )
        )
        self.assertIsNone(parser._decode_json_object(""))
        self.assertIsNone(parser._decode_json_object('{"ok": true} trailing'))

    def test_stream_payload_helper_edges(self) -> None:
        harmony_manager = MagicMock()
        harmony_manager.tool_format = ToolFormat.HARMONY
        harmony_parser = ToolCallResponseParser(harmony_manager, None)

        self.assertIsNone(harmony_parser._stream_tool_payload("analysis"))
        self.assertEqual(
            harmony_parser._stream_tool_payload(
                "<|channel|>commentary to=calc <|message|>"
                '{"x":1}<|call|>ignored'
            ),
            '{"x":1}',
        )

        parser = ToolCallResponseParser(MagicMock(), None)
        self.assertEqual(parser._stream_tool_payload("<tool>{}</tool>"), "{}")
        self.assertIsNone(parser._stream_tool_payload("<tool_call"))
        self.assertIsNone(parser._stream_tool_payload("plain text"))

        class MissingMarkerParser(ToolCallResponseParser):
            def _first_stream_tool_start_index(self, text: str) -> int | None:
                return 0

            def _tool_start_marker_at(
                self,
                text: str,
                index: int,
                markers: Iterable[str] | None = None,
            ) -> str | None:
                return None

        missing_marker_parser = MissingMarkerParser(MagicMock(), None)
        self.assertIsNone(missing_marker_parser._stream_tool_payload("plain"))

    async def test_dsml_tool_call_emits_ready_done(self) -> None:
        manager = ToolManager(
            parser=ToolCallParser(tool_format=ToolFormat.DSML)
        )
        parser = ToolCallResponseParser(manager, None)
        text = (
            "<｜DSML｜tool_calls>"
            '<｜DSML｜invoke name="math.calculator">'
            '<｜DSML｜parameter name="expression" string="true">2 + 2'
            "</｜DSML｜parameter>"
            "</｜DSML｜invoke>"
            "</｜DSML｜tool_calls>"
        )

        output: list[Any] = []
        output.extend(await parser.push(text[:20]))
        output.extend(await parser.push(text[20:80]))
        output.extend(await parser.push(text[80:]))

        kinds = [
            item.kind
            for item in output
            if isinstance(item, StreamProviderEvent)
        ]
        self.assertIn(StreamItemKind.TOOL_CALL_ARGUMENT_DELTA, kinds)
        self.assertIn(StreamItemKind.TOOL_CALL_READY, kinds)
        self.assertIn(StreamItemKind.TOOL_CALL_DONE, kinds)
        tool_call_ids = {
            item.correlation.tool_call_id
            for item in output
            if isinstance(item, StreamProviderEvent)
            and item.correlation.tool_call_id is not None
        }
        self.assertEqual(len(tool_call_ids), 1)

    async def test_handles_non_matching_tokens(self) -> None:
        manager = MagicMock()
        manager.is_potential_tool_call.return_value = False
        manager.tool_call_status.return_value = (
            ToolCallParser.ToolCallBufferStatus.NONE
        )
        manager.get_calls.return_value = []
        parser = ToolCallResponseParser(manager, None)
        parser._pending_tokens = ["pending"]
        parser._pending_str = "pending"
        result = await parser.push("noise")
        self.assertEqual(_answer_delta_texts(result), ["pending", "noise"])
        self.assertEqual(parser._tag_buffer, "noise")
        result = await parser.push("a" * 70)
        self.assertEqual(_answer_delta_texts(result), ["a" * 70])
        self.assertEqual(len(parser._tag_buffer), 64)

    async def test_return_empty_result_when_list_ignores_appends(self) -> None:
        manager = MagicMock()
        manager.is_potential_tool_call.return_value = True
        manager.tool_call_status.return_value = (
            ToolCallParser.ToolCallBufferStatus.OPEN
        )

        class NonAppendingList(list[str]):
            def append(self, value: str) -> None:
                return None

        parser = ToolCallResponseParser(manager, None)
        parser._pending_tokens = NonAppendingList()
        result = await parser.push("<tool_call>")
        self.assertEqual(result, [])
