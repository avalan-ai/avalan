from asyncio import run
from json import loads
from logging import getLogger
from typing import cast
from unittest.mock import patch

import openai
from httpx import AsyncClient, MockTransport, Request, Response

from avalan.cli.__main__ import CLI
from avalan.entities import (
    GenerationSettings,
    Message,
    MessageRole,
    ReasoningSettings,
    ReasoningSummaryMode,
)
from avalan.model.nlp.text.vendor.openai import OpenAIClient

_REAL_ASYNC_OPENAI = openai.AsyncOpenAI
_PROMPT = "Keep this prompt byte-for-byte stable."


class _NonStreamingOpenAITransport:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []
        self.http_clients: list[AsyncClient] = []

    async def handle(self, request: Request) -> Response:
        self.requests.append(cast(dict[str, object], loads(request.content)))
        return Response(
            200,
            json={
                "id": f"response-{len(self.requests)}",
                "created_at": 0.0,
                "model": "gpt-5",
                "object": "response",
                "output": [],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "status": "completed",
                "usage": {
                    "input_tokens": 1,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 0,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 1,
                },
            },
        )

    def client_factory(self, **kwargs: object) -> openai.AsyncOpenAI:
        http_client = AsyncClient(transport=MockTransport(self.handle))
        self.http_clients.append(http_client)
        return _REAL_ASYNC_OPENAI(**kwargs, http_client=http_client)


async def _request_pair() -> tuple[dict[str, object], dict[str, object]]:
    transport = _NonStreamingOpenAITransport()
    with patch.object(
        openai,
        "AsyncOpenAI",
        side_effect=transport.client_factory,
    ):
        client = OpenAIClient(api_key="test-key", base_url=None)
        try:
            message = Message(role=MessageRole.USER, content=_PROMPT)
            await client(
                "gpt-5",
                [message],
                GenerationSettings(),
                use_async_generator=False,
            )
            await client(
                "gpt-5",
                [message],
                GenerationSettings(
                    reasoning=ReasoningSettings(
                        summary=ReasoningSummaryMode.DETAILED
                    )
                ),
                use_async_generator=False,
            )
        finally:
            await client.aclose()
    assert len(transport.requests) == 2
    return transport.requests[0], transport.requests[1]


def test_summary_request_is_prompt_independent() -> None:
    omitted, requested = run(_request_pair())

    assert omitted["input"] == requested["input"]
    assert _PROMPT in repr(omitted["input"])
    assert "reasoning" not in omitted
    assert requested["reasoning"] == {"summary": "detailed"}


def test_request_and_display_controls_are_independent() -> None:
    parser = CLI(getLogger("test.phase9.controls"))._parser
    cases = (
        ((), (None, False)),
        (("--reasoning-summary", "concise"), ("concise", False)),
        (("--display-reasoning",), (None, True)),
        (
            (
                "--reasoning-summary",
                "concise",
                "--display-reasoning",
            ),
            ("concise", True),
        ),
    )
    for options, expected in cases:
        parsed = parser.parse_args(["agent", "run", "agent.toml", *options])
        assert (
            parsed.run_reasoning_summary,
            parsed.display_reasoning,
        ) == expected

    omitted, requested = run(_request_pair())
    assert omitted["input"] == requested["input"]
    assert "reasoning" not in omitted
    assert requested["reasoning"] == {"summary": "detailed"}
