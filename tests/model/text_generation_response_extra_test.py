from logging import getLogger
from unittest import IsolatedAsyncioTestCase
from unittest.mock import MagicMock

from avalan.entities import (
    GenerationSettings,
    ReasoningSettings,
    Token,
    TokenDetail,
    ToolCallToken,
)
from avalan.model.response.parsers.reasoning import ReasoningParser
from avalan.model.response.parsers.tool import ToolCallResponseParser
from avalan.model.response.text import TextGenerationResponse
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemKind,
    StreamTerminalOutcome,
    StreamValidationError,
)
from avalan.tool.parser import ToolCallParser


async def _complex_generator():
    rp = ReasoningParser(
        reasoning_settings=ReasoningSettings(),
        logger=getLogger(),
        legacy_fixture=True,
    )
    tm = MagicMock()
    tm.is_potential_tool_call.return_value = True
    tm.get_calls.return_value = None
    base_parser = ToolCallParser()
    tm.tool_call_status.side_effect = base_parser.tool_call_status
    tp = ToolCallResponseParser(tm, None, legacy_fixture=True)

    sequence = [
        "X",
        "<think>",
        "ra",
        "rb",
        "</think>",
        "Y",
        "<tool_call>",
        "foo",
        "bar",
        "</tool_call>",
        "Z",
    ]

    for s in sequence:
        items = await rp.push(s)
        for item in items:
            parsed = await tp.push(item) if isinstance(item, str) else [item]
            for p in parsed:
                if isinstance(p, str):
                    if p == "</think>":
                        yield TokenDetail(id=3, token=p, probability=0.5)
                    elif p in {"X", "Y"}:
                        yield Token(id=1, token=p)
                    else:
                        yield p
                elif isinstance(p, ToolCallToken):
                    if p.token == "</tool_call>":
                        yield TokenDetail(id=4, token=p.token, probability=0.5)
                    else:
                        yield p
                else:
                    yield p


class TextGenerationResponseParsersTestCase(IsolatedAsyncioTestCase):
    async def test_mixed_tokens(self):
        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: _complex_generator(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )

        with self.assertRaisesRegex(
            StreamValidationError,
            "unsupported legacy SDK response stream item",
        ):
            _ = [item async for item in resp]

    async def test_flush_pending_public_reasoning_prefix(self) -> None:
        async def gen():
            yield "<thi"

        settings = GenerationSettings()
        response = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )

        with self.assertRaisesRegex(
            StreamValidationError,
            "unsupported legacy SDK response stream item",
        ):
            _ = [item async for item in response]

    async def test_reasoning_done_precedes_following_answer_delta(
        self,
    ) -> None:
        async def gen():
            for item in (
                CanonicalStreamItem(
                    stream_session_id="response-stream",
                    run_id="response-run",
                    turn_id="response-turn",
                    sequence=0,
                    kind=StreamItemKind.STREAM_STARTED,
                    channel=StreamChannel.CONTROL,
                ),
                CanonicalStreamItem(
                    stream_session_id="response-stream",
                    run_id="response-run",
                    turn_id="response-turn",
                    sequence=1,
                    kind=StreamItemKind.ANSWER_DELTA,
                    channel=StreamChannel.ANSWER,
                    text_delta="x",
                ),
                CanonicalStreamItem(
                    stream_session_id="response-stream",
                    run_id="response-run",
                    turn_id="response-turn",
                    sequence=2,
                    kind=StreamItemKind.REASONING_DELTA,
                    channel=StreamChannel.REASONING,
                    text_delta="r",
                ),
                CanonicalStreamItem(
                    stream_session_id="response-stream",
                    run_id="response-run",
                    turn_id="response-turn",
                    sequence=3,
                    kind=StreamItemKind.REASONING_DONE,
                    channel=StreamChannel.REASONING,
                ),
                CanonicalStreamItem(
                    stream_session_id="response-stream",
                    run_id="response-run",
                    turn_id="response-turn",
                    sequence=4,
                    kind=StreamItemKind.ANSWER_DELTA,
                    channel=StreamChannel.ANSWER,
                    text_delta="y",
                ),
                CanonicalStreamItem(
                    stream_session_id="response-stream",
                    run_id="response-run",
                    turn_id="response-turn",
                    sequence=5,
                    kind=StreamItemKind.ANSWER_DONE,
                    channel=StreamChannel.ANSWER,
                ),
                CanonicalStreamItem(
                    stream_session_id="response-stream",
                    run_id="response-run",
                    turn_id="response-turn",
                    sequence=6,
                    kind=StreamItemKind.STREAM_COMPLETED,
                    channel=StreamChannel.CONTROL,
                    usage={},
                    terminal_outcome=StreamTerminalOutcome.COMPLETED,
                ),
            ):
                yield item

        settings = GenerationSettings()
        response = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )

        self.assertEqual(await response.to_str(), "xy")
        accumulator = response._stream_accumulator
        self.assertIsNotNone(accumulator)
        assert accumulator is not None
        items = accumulator.items
        reasoning_done_index = next(
            index
            for index, item in enumerate(items)
            if item.kind is StreamItemKind.REASONING_DONE
        )
        following_answer_index = next(
            index
            for index, item in enumerate(items)
            if (
                item.kind is StreamItemKind.ANSWER_DELTA
                and item.text_delta == "y"
            )
        )
        self.assertLess(reasoning_done_index, following_answer_index)

    async def test_adjacent_reasoning_whitespace_gap_stays_open(
        self,
    ) -> None:
        async def gen():
            for item in (
                CanonicalStreamItem(
                    stream_session_id="response-stream",
                    run_id="response-run",
                    turn_id="response-turn",
                    sequence=0,
                    kind=StreamItemKind.STREAM_STARTED,
                    channel=StreamChannel.CONTROL,
                ),
                CanonicalStreamItem(
                    stream_session_id="response-stream",
                    run_id="response-run",
                    turn_id="response-turn",
                    sequence=1,
                    kind=StreamItemKind.ANSWER_DELTA,
                    channel=StreamChannel.ANSWER,
                    text_delta="x",
                ),
                CanonicalStreamItem(
                    stream_session_id="response-stream",
                    run_id="response-run",
                    turn_id="response-turn",
                    sequence=2,
                    kind=StreamItemKind.REASONING_DELTA,
                    channel=StreamChannel.REASONING,
                    text_delta="<think>a</think><think>b</think>",
                ),
                CanonicalStreamItem(
                    stream_session_id="response-stream",
                    run_id="response-run",
                    turn_id="response-turn",
                    sequence=3,
                    kind=StreamItemKind.REASONING_DONE,
                    channel=StreamChannel.REASONING,
                ),
                CanonicalStreamItem(
                    stream_session_id="response-stream",
                    run_id="response-run",
                    turn_id="response-turn",
                    sequence=4,
                    kind=StreamItemKind.ANSWER_DELTA,
                    channel=StreamChannel.ANSWER,
                    text_delta=" \n y",
                ),
                CanonicalStreamItem(
                    stream_session_id="response-stream",
                    run_id="response-run",
                    turn_id="response-turn",
                    sequence=5,
                    kind=StreamItemKind.ANSWER_DONE,
                    channel=StreamChannel.ANSWER,
                ),
                CanonicalStreamItem(
                    stream_session_id="response-stream",
                    run_id="response-run",
                    turn_id="response-turn",
                    sequence=6,
                    kind=StreamItemKind.STREAM_COMPLETED,
                    channel=StreamChannel.CONTROL,
                    usage={},
                    terminal_outcome=StreamTerminalOutcome.COMPLETED,
                ),
            ):
                yield item

        settings = GenerationSettings()
        response = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )

        self.assertEqual(await response.to_str(), "x \n y")
        accumulator = response._stream_accumulator
        self.assertIsNotNone(accumulator)
        assert accumulator is not None
        self.assertEqual(
            accumulator.reasoning_text, "<think>a</think><think>b</think>"
        )
        done_items = [
            item
            for item in accumulator.items
            if item.kind is StreamItemKind.REASONING_DONE
        ]
        self.assertEqual(len(done_items), 1)

    async def test_pending_reasoning_done_without_parser_precedes_answer(
        self,
    ) -> None:
        async def gen():
            for item in (
                CanonicalStreamItem(
                    stream_session_id="response-stream",
                    run_id="response-run",
                    turn_id="response-turn",
                    sequence=0,
                    kind=StreamItemKind.STREAM_STARTED,
                    channel=StreamChannel.CONTROL,
                ),
                CanonicalStreamItem(
                    stream_session_id="response-stream",
                    run_id="response-run",
                    turn_id="response-turn",
                    sequence=1,
                    kind=StreamItemKind.REASONING_DELTA,
                    channel=StreamChannel.REASONING,
                    text_delta="private",
                ),
                CanonicalStreamItem(
                    stream_session_id="response-stream",
                    run_id="response-run",
                    turn_id="response-turn",
                    sequence=2,
                    kind=StreamItemKind.REASONING_DONE,
                    channel=StreamChannel.REASONING,
                ),
                CanonicalStreamItem(
                    stream_session_id="response-stream",
                    run_id="response-run",
                    turn_id="response-turn",
                    sequence=3,
                    kind=StreamItemKind.ANSWER_DELTA,
                    channel=StreamChannel.ANSWER,
                    text_delta="answer",
                ),
                CanonicalStreamItem(
                    stream_session_id="response-stream",
                    run_id="response-run",
                    turn_id="response-turn",
                    sequence=4,
                    kind=StreamItemKind.ANSWER_DONE,
                    channel=StreamChannel.ANSWER,
                ),
                CanonicalStreamItem(
                    stream_session_id="response-stream",
                    run_id="response-run",
                    turn_id="response-turn",
                    sequence=5,
                    kind=StreamItemKind.STREAM_COMPLETED,
                    channel=StreamChannel.CONTROL,
                    usage={},
                    terminal_outcome=StreamTerminalOutcome.COMPLETED,
                ),
            ):
                yield item

        response = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
        )
        items = [item async for item in response]

        reasoning_done_index = next(
            index
            for index, item in enumerate(items)
            if item.kind is StreamItemKind.REASONING_DONE
        )
        answer_index = next(
            index
            for index, item in enumerate(items)
            if item.kind is StreamItemKind.ANSWER_DELTA
        )
        self.assertLess(reasoning_done_index, answer_index)
