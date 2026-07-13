from asyncio import run
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
from json import dumps, loads
from logging import getLogger
from os import environ
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any, cast
from unittest.mock import patch

import openai
import pytest
from httpx import AsyncClient, MockTransport, Request, Response
from rich.console import Console

from avalan.cli.__main__ import CLI, _AnonymousHub
from avalan.cli.commands import agent as agent_commands
from avalan.cli.theme_registry import create_theme

_QUESTION = "What is (4 + 6) times 5, divided by 2?"
_FIRST_SUMMARY = ("Check the arithmetic.", "Use the calculator tool.")
_SECOND_SUMMARY = "Use the calculator result."
_ANSWER = {"answer": 25}
_REAL_ASYNC_OPENAI = openai.AsyncOpenAI
_FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "reasoning_summary"
    / "provider_traces"
)


def _response_usage() -> dict[str, object]:
    return {
        "input_tokens": 4,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 6,
        "output_tokens_details": {"reasoning_tokens": 2},
        "total_tokens": 10,
    }


def _phase9_provider_turns() -> tuple[list[dict[str, object]], ...]:
    multipart = loads((_FIXTURE_ROOT / "multipart.json").read_text())
    tools_answer = loads((_FIXTURE_ROOT / "tools_answer.json").read_text())
    multipart_events = deepcopy(multipart["responses"][0]["events"])
    tool_events = deepcopy(tools_answer["responses"][0]["events"])
    answer_events = deepcopy(tools_answer["responses"][1]["events"])

    first_events = multipart_events[:-1]
    first_reasoning = cast(dict[str, object], first_events[-1])["item"]
    assert isinstance(first_reasoning, dict)
    summary = cast(list[dict[str, object]], first_reasoning["summary"])
    summary[0]["text"], summary[1]["text"] = _FIRST_SUMMARY
    for event in first_events:
        if event.get("summary_index") in (0, 1):
            index = cast(int, event["summary_index"])
            if "delta" in event:
                event["delta"] = _FIRST_SUMMARY[index]
            if "text" in event:
                event["text"] = _FIRST_SUMMARY[index]
            part = event.get("part")
            if isinstance(part, dict) and part.get("text"):
                part["text"] = _FIRST_SUMMARY[index]

    function_events = tool_events[6:10]
    arguments = dumps({"expression": "(4 + 6) * 5 / 2"})
    for sequence, event in enumerate(function_events, len(first_events)):
        event["sequence_number"] = sequence
        item = event.get("item")
        if isinstance(item, dict):
            item["arguments"] = (
                arguments if item.get("status") == "completed" else ""
            )
        if "delta" in event:
            event["delta"] = arguments
        if "arguments" in event:
            event["arguments"] = arguments
    completed = deepcopy(tool_events[-1])
    completed["sequence_number"] = len(first_events) + len(function_events)
    response = cast(dict[str, object], completed["response"])
    function_item = cast(dict[str, object], function_events[-1]["item"])
    response["output"] = [first_reasoning, function_item]
    response["usage"] = _response_usage()
    first_events.extend((*function_events, completed))

    for event in answer_events:
        if event.get("type") == "response.reasoning_summary_text.delta":
            event["delta"] = _SECOND_SUMMARY
        if event.get("type") == "response.reasoning_summary_text.done":
            event["text"] = _SECOND_SUMMARY
        part = event.get("part")
        if (
            event.get("type") == "response.reasoning_summary_part.done"
            and isinstance(part, dict)
            and part.get("type") == "summary_text"
        ):
            part["text"] = _SECOND_SUMMARY
        item = event.get("item")
        if (
            event.get("type") == "response.output_item.done"
            and isinstance(item, dict)
            and item.get("type") == "reasoning"
        ):
            item["summary"] = [
                {"type": "summary_text", "text": _SECOND_SUMMARY}
            ]
        if event.get("type") == "response.output_text.delta":
            event["delta"] = dumps(_ANSWER, separators=(",", ":"))
        if event.get("type") == "response.output_text.done":
            event["text"] = dumps(_ANSWER, separators=(",", ":"))
        if (
            event.get("type") == "response.content_part.done"
            and isinstance(part, dict)
            and part.get("type") == "output_text"
        ):
            part["text"] = dumps(_ANSWER, separators=(",", ":"))
        if isinstance(item, dict) and item.get("type") == "message":
            content = cast(list[dict[str, object]], item["content"])
            if content:
                content[0]["text"] = dumps(_ANSWER, separators=(",", ":"))
        if event.get("type") == "response.completed":
            final_response = cast(dict[str, object], event["response"])
            final_response["usage"] = _response_usage()
            for output in cast(
                list[dict[str, object]], final_response["output"]
            ):
                if output.get("type") == "reasoning":
                    output["summary"] = [
                        {
                            "type": "summary_text",
                            "text": _SECOND_SUMMARY,
                        }
                    ]
                if output.get("type") == "message":
                    content = cast(list[dict[str, object]], output["content"])
                    content[0]["text"] = dumps(_ANSWER, separators=(",", ":"))

    return cast(
        tuple[list[dict[str, object]], ...],
        (first_events, answer_events),
    )


class _OpenAIMockTransport:
    def __init__(self) -> None:
        self.turns = _phase9_provider_turns()
        self.requests: list[dict[str, object]] = []
        self.http_clients: list[AsyncClient] = []

    async def handle(self, request: Request) -> Response:
        payload = cast(dict[str, object], loads(request.content))
        self.requests.append(payload)
        turn = len(self.requests) - 1
        if turn >= len(self.turns):
            return Response(500, json={"error": {"message": "extra call"}})
        events = deepcopy(self.turns[turn])
        if turn == 0:
            tools = cast(list[dict[str, object]], payload["tools"])
            provider_name = cast(str, tools[0]["name"])
            for event in events:
                item = event.get("item")
                if (
                    isinstance(item, dict)
                    and item.get("type") == "function_call"
                ):
                    item["name"] = provider_name
                if (
                    event.get("type")
                    == "response.function_call_arguments.done"
                ):
                    event["name"] = provider_name
                response = event.get("response")
                if isinstance(response, dict):
                    for output in cast(
                        list[dict[str, object]], response["output"]
                    ):
                        if output.get("type") == "function_call":
                            output["name"] = provider_name
        body = "".join(
            f"data: {dumps(event, separators=(',', ':'))}\n\n"
            for event in events
        )
        body += "data: [DONE]\n\n"
        return Response(
            200,
            content=body.encode(),
            headers={"content-type": "text/event-stream"},
        )

    def client_factory(self, **kwargs: object) -> openai.AsyncOpenAI:
        http_client = AsyncClient(transport=MockTransport(self.handle))
        self.http_clients.append(http_client)
        return _REAL_ASYNC_OPENAI(**kwargs, http_client=http_client)


def _agent_spec(provider: str = "openai") -> str:
    return f"""
[agent]
role = "assistant"
user = "{{{{ input }}}}"

[engine]
uri = "ai://env:PHASE9_OPENAI_KEY@{provider}/gpt-5"

[run]
maximum_tool_cycles = 2

[run.response_format]
type = "json_object"

[memory]
recent = true

[tool]
enable = ["math.calculator"]
"""


async def _run_positive_cli(
    theme_name: str,
) -> tuple[
    str,
    str,
    _OpenAIMockTransport,
    object,
]:
    transport = _OpenAIMockTransport()
    captured: list[object] = []
    original_from_file = agent_commands.OrchestratorLoader.from_file

    async def capture_from_file(
        loader: object, *args: object, **kwargs: object
    ) -> object:
        orchestrator = await original_from_file(
            cast(Any, loader), *args, **kwargs
        )
        captured.append(orchestrator)
        return orchestrator

    with TemporaryDirectory() as directory:
        spec = Path(directory) / "summary-agent.toml"
        spec.write_text(_agent_spec(), encoding="utf-8")
        cli = CLI(getLogger(f"test.phase9.cli.{theme_name}"))
        args = cli._parser.parse_args(
            [
                "agent",
                "run",
                str(spec),
                "--reasoning-summary",
                "detailed",
                "--display-reasoning",
                "--display-tools",
                "--skip-hub-access-check",
                "--no-repl",
                "--theme",
                theme_name,
            ]
        )
        theme = create_theme(
            theme_name,
            lambda value: value,
            lambda one, many, count: one if count == 1 else many,
        )
        stdout, stderr = StringIO(), StringIO()
        console = Console(
            file=stdout,
            force_terminal=False,
            color_system=None,
            width=160,
        )
        with NamedTemporaryFile("w+", encoding="utf-8") as input_stream:
            input_stream.write(_QUESTION)
            input_stream.flush()
            input_stream.seek(0)
            with (
                patch.dict(environ, {"PHASE9_OPENAI_KEY": "test-key"}),
                patch.object(
                    openai, "AsyncOpenAI", side_effect=transport.client_factory
                ),
                patch.object(
                    agent_commands.OrchestratorLoader,
                    "from_file",
                    capture_from_file,
                ),
                patch("avalan.cli.stdin", input_stream),
                redirect_stderr(stderr),
            ):
                await agent_commands.agent_run(
                    args,
                    console,
                    theme,
                    cast(Any, _AnonymousHub(args.cache_dir)),
                    getLogger(f"test.phase9.cli.{theme_name}"),
                    20,
                )
    assert len(captured) == 1
    return stdout.getvalue(), stderr.getvalue(), transport, captured[0]


@pytest.mark.parametrize("theme_name", ("basic", "fancy"))
def test_cli_summary_tool_continuation_is_credential_free(
    theme_name: str,
) -> None:
    stdout, stderr, transport, orchestrator = run(
        _run_positive_cli(theme_name)
    )

    assert loads(stdout) == _ANSWER
    assert len(transport.requests) == 2
    assert [request["reasoning"] for request in transport.requests] == [
        {"summary": "detailed"},
        {"summary": "detailed"},
    ]
    assert all(request.get("stream") is True for request in transport.requests)
    second_input = repr(transport.requests[1]["input"])
    assert "function_call_output" in second_input
    assert "25" in second_input
    assert "cipher-multi" in second_input
    assert "Reasoning summary" in stderr
    assert _FIRST_SUMMARY[0] in stderr
    assert _FIRST_SUMMARY[1] in stderr
    assert stderr.index(_FIRST_SUMMARY[0]) < stderr.index(_FIRST_SUMMARY[1])
    assert _SECOND_SUMMARY in stderr
    outward = stdout + repr(
        getattr(getattr(orchestrator, "memory"), "recent_message").data
    )
    assert _FIRST_SUMMARY[0] not in outward
    assert _FIRST_SUMMARY[1] not in outward
    assert _SECOND_SUMMARY not in outward
    assert "cipher-" not in outward


def test_cli_invalid_summary_value_is_zero_call_and_leak_free() -> None:
    cli = CLI(getLogger("test.phase9.cli.invalid"))
    stderr = StringIO()
    with redirect_stderr(stderr), pytest.raises(SystemExit) as error:
        cli._parser.parse_args(
            [
                "agent",
                "run",
                "agent.toml",
                "--reasoning-summary",
                "invalid",
            ]
        )
    assert error.value.code == 2
    assert "invalid choice" in stderr.getvalue()
    assert "summary-private" not in stderr.getvalue()


def test_cli_unsupported_adapter_is_zero_call_and_leak_free() -> None:
    async def exercise() -> tuple[
        int | str | None,
        str,
        str,
        _OpenAIMockTransport,
    ]:
        transport = _OpenAIMockTransport()
        with TemporaryDirectory() as directory:
            spec = Path(directory) / "unsupported-agent.toml"
            spec.write_text(_agent_spec("anyscale"), encoding="utf-8")
            stdout, stderr = StringIO(), StringIO()
            cli = CLI(getLogger("test.phase9.cli.unsupported"))
            argv = [
                "avalan",
                "agent",
                "run",
                str(spec),
                "--quiet",
                "--reasoning-summary",
                "detailed",
                "--skip-hub-access-check",
            ]
            with NamedTemporaryFile("w+", encoding="utf-8") as input_stream:
                input_stream.write(_QUESTION)
                input_stream.flush()
                input_stream.seek(0)
                with (
                    patch.dict(environ, {"PHASE9_OPENAI_KEY": "test-key"}),
                    patch("sys.argv", argv),
                    patch.object(
                        openai,
                        "AsyncOpenAI",
                        side_effect=transport.client_factory,
                    ),
                    patch("avalan.cli.stdin", input_stream),
                    redirect_stdout(stdout),
                    redirect_stderr(stderr),
                    pytest.raises(SystemExit) as error,
                ):
                    await cli()
        return (
            error.value.code,
            stdout.getvalue(),
            stderr.getvalue(),
            transport,
        )

    code, stdout, stderr, transport = run(exercise())
    assert code == 1
    assert "anyscale" in stderr
    assert "does not support reasoning summary" in stderr
    assert stdout == ""
    assert transport.requests == []
    assert "summary-private" not in stderr
