from avalan.model.response.text import TextGenerationResponse
from avalan.model.response.parsers.reasoning import ReasoningParser
from avalan.model.response.parsers.tool import ToolCallParser
from avalan.entities import ReasoningToken, ToolCallToken, Token, TokenDetail
from unittest.mock import MagicMock
from unittest import IsolatedAsyncioTestCase


async def _complex_generator():
    rp = ReasoningParser()
    tm = MagicMock()
    tm.is_potential_tool_call.return_value = True
    tm.get_calls.return_value = None
    tp = ToolCallParser(tm, None)

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
        resp = TextGenerationResponse(
            lambda: _complex_generator(), use_async_generator=True
        )

        tokens = []
        async for t in resp:
            tokens.append(t)

        self.assertEqual(
            len([t for t in tokens if isinstance(t, ReasoningToken)]),
            4,
        )
        self.assertEqual(
            len([t for t in tokens if isinstance(t, ToolCallToken)]),
            3,
        )
        self.assertEqual(
            len([t for t in tokens if isinstance(t, TokenDetail)]),
            1,
        )
        self.assertGreaterEqual(
            len([t for t in tokens if type(t) is Token]),
            2,
        )
        self.assertEqual(len([t for t in tokens if isinstance(t, str)]), 1)
