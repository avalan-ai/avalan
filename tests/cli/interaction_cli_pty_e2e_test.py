"""Exercise attached CLI input through real parser, pipes, and a PTY."""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import nullcontext
from datetime import UTC, datetime
from importlib.util import module_from_spec, spec_from_file_location
from json import dumps, loads
from logging import getLogger
from os import (
    WNOHANG,
    _exit,
    close,
    dup2,
    fork,
    kill,
    pipe,
    read,
    set_blocking,
    ttyname,
    waitpid,
    write,
)
from pathlib import Path
from pty import openpty
from select import select
from signal import SIGKILL
from sys import modules as system_modules
from time import monotonic
from traceback import print_exc
from tty import setraw
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

from avalan import cli as cli_package
from avalan.agent.execution import AttachedInteractionRuntime
from avalan.agent.loader import OrchestratorLoader
from avalan.cli import __main__ as cli_main
from avalan.cli.commands import agent as agent_cmds
from avalan.interaction import (
    AgentId,
    AnsweredResolution,
    BranchId,
    Choice,
    ChoiceValue,
    ConfirmationAnswer,
    ConfirmationQuestion,
    ContinuationId,
    DeclinedResolution,
    ExecutionDefinitionRef,
    ExecutionOrigin,
    FreeFormOther,
    InputRequest,
    InputRequestId,
    ModelCallId,
    MultilineTextAnswer,
    MultilineTextQuestion,
    MultipleSelectionAnswer,
    MultipleSelectionQuestion,
    PrincipalScope,
    QuestionId,
    RequestState,
    RequirementMode,
    RunId,
    SelectedChoice,
    StateRevision,
    StreamSessionId,
    TextAnswer,
    TextQuestion,
    TurnId,
)
from avalan.interaction.handler import (
    InputDisconnectReason,
    InputHandlerContext,
    InputHandlerDisconnected,
    InputHandlerResolution,
)
from avalan.model.response.text import TextGenerationResponse
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamConsumerProjection,
    StreamItemCorrelation,
    StreamItemKind,
    StreamTerminalOutcome,
    stream_channel_for_kind,
)


def _load_boundary_fixture() -> Any:
    path = (
        Path(__file__).parents[1]
        / "agent"
        / "execution_attached_boundaries_test.py"
    )
    spec = spec_from_file_location("_pty_boundary_fixture", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    system_modules[spec.name] = module
    spec.loader.exec_module(module)
    return cast(Any, module)


boundary_fixture = _load_boundary_fixture()


class _Response:
    input_token_count = 1
    can_think = False
    is_thinking = False

    def __init__(
        self,
        runtime: AttachedInteractionRuntime,
        owner: "_Orchestrator",
    ) -> None:
        self.runtime = runtime
        self.owner = owner
        self.cancellation_checker: Callable[[], Awaitable[None]] | None = None

    def set_cancellation_checker(
        self,
        checker: Callable[[], Awaitable[None]] | None,
    ) -> None:
        self.cancellation_checker = checker

    def consumer_projections(
        self,
        *,
        stream_session_id: str,
        run_id: str,
        turn_id: str,
    ) -> AsyncIterator[StreamConsumerProjection]:
        async def generate() -> AsyncIterator[StreamConsumerProjection]:
            self.owner.provider_calls += 1
            yield _projection(
                stream_session_id,
                run_id,
                turn_id,
                0,
                StreamItemKind.STREAM_STARTED,
            )
            outcome = await self.runtime.handler(
                InputHandlerContext(request=_request(self.owner.case))
            )
            if self.cancellation_checker is not None:
                await self.cancellation_checker()
            if self.owner.case == "cancel_then_text":
                assert isinstance(outcome, InputHandlerDisconnected)
                assert (
                    outcome.reason is InputDisconnectReason.HANDLER_CANCELLED
                )
                outcome = await self.runtime.handler(
                    InputHandlerContext(
                        request=_request("text", suffix="second")
                    )
                )
            assert isinstance(
                outcome,
                (InputHandlerResolution, InputHandlerDisconnected),
            )
            answer_text = _outcome_text(outcome)
            yield _projection(
                stream_session_id,
                run_id,
                turn_id,
                1,
                StreamItemKind.ANSWER_DELTA,
                text=answer_text,
            )
            yield _projection(
                stream_session_id,
                run_id,
                turn_id,
                2,
                StreamItemKind.ANSWER_DONE,
            )
            yield _projection(
                stream_session_id,
                run_id,
                turn_id,
                3,
                StreamItemKind.STREAM_COMPLETED,
            )

        return generate()


class _Orchestrator:
    id = "pty-agent"
    name = "PTY Agent"
    model_ids = ["fake-model"]
    _call_options = None

    def __init__(self, case: str) -> None:
        self.case = case
        self.calls: list[str] = []
        self.provider_calls = 0
        self.event_manager = SimpleNamespace(
            add_ui_listener=lambda _listener: None,
            remove_listener=lambda _listener: None,
        )
        self.memory = SimpleNamespace(
            has_recent_message=False,
            has_permanent_message=False,
            recent_message=SimpleNamespace(is_empty=True, size=0, data=[]),
        )
        self.engine = SimpleNamespace(
            model_id="fake-model",
            tokenizer_config=None,
            input_token_count=lambda *_args, **_kwargs: 1,
        )
        self.engine_agent = SimpleNamespace(
            engine_uri=SimpleNamespace(params={})
        )
        self.tool = SimpleNamespace(is_empty=True)

    async def __aenter__(self) -> "_Orchestrator":
        return self

    async def __aexit__(self, *_args: object) -> bool:
        return False

    async def __call__(self, input_text: str, **kwargs: object) -> _Response:
        self.calls.append(input_text)
        runtime = kwargs["interaction_runtime"]
        assert isinstance(runtime, AttachedInteractionRuntime)
        return _Response(runtime, self)


def _request(case: str, *, suffix: str = "first") -> InputRequest:
    question: (
        ConfirmationQuestion
        | TextQuestion
        | MultilineTextQuestion
        | MultipleSelectionQuestion
    )
    if case in {
        "confirmation",
        "decline",
        "cancel_input",
        "cancel_run",
        "disappear",
        "cancel_then_text",
    }:
        question = ConfirmationQuestion(
            question_id=QuestionId("confirm"),
            prompt="Proceed?",
            required=True,
        )
    elif case == "text":
        question = TextQuestion(
            question_id=QuestionId("text"),
            prompt="Name?",
            required=True,
        )
    elif case == "multiline":
        question = MultilineTextQuestion(
            question_id=QuestionId("multiline"),
            prompt="Details?",
            required=True,
        )
    else:
        assert case == "multiple_other"
        question = MultipleSelectionQuestion(
            question_id=QuestionId("multiple"),
            prompt="Modes?",
            required=True,
            choices=(
                Choice(value=ChoiceValue("fast"), label="Fast"),
                Choice(value=ChoiceValue("safe"), label="Safe"),
            ),
            allow_other=True,
        )
    return InputRequest(
        request_id=InputRequestId(f"pty-request-{suffix}"),
        continuation_id=ContinuationId(f"pty-continuation-{suffix}"),
        origin=ExecutionOrigin(
            run_id=RunId("pty-run"),
            turn_id=TurnId("pty-turn"),
            agent_id=AgentId("pty-agent"),
            branch_id=BranchId("pty-branch"),
            model_call_id=ModelCallId("pty-call"),
            stream_session_id=StreamSessionId("pty-stream"),
            definition=ExecutionDefinitionRef(
                agent_definition_locator="agent://pty",
                agent_definition_revision="r1",
                operation_id="operation",
                operation_index=0,
                model_config_reference="model-r1",
                tool_revision="tools-r1",
                capability_revision="capabilities-r1",
            ),
            principal=PrincipalScope(),
        ),
        mode=RequirementMode.REQUIRED,
        reason="Need confirmation.",
        questions=(question,),
        created_at=datetime(2026, 7, 24, tzinfo=UTC),
        state=RequestState.PENDING,
        state_revision=StateRevision(1),
    )


def _outcome_text(
    outcome: InputHandlerResolution | InputHandlerDisconnected,
) -> str:
    if isinstance(outcome, InputHandlerDisconnected):
        return f"disconnected:{outcome.reason.value}"
    resolution = outcome.resolution
    if isinstance(resolution, DeclinedResolution):
        return "declined"
    assert isinstance(resolution, AnsweredResolution)
    answer = resolution.answers[0]
    if isinstance(answer, ConfirmationAnswer):
        assert answer.value
        return "completed:yes"
    if isinstance(answer, TextAnswer):
        return f"completed:{answer.value}"
    if isinstance(answer, MultilineTextAnswer):
        return f"completed:{answer.value.replace(chr(10), '|')}"
    assert isinstance(answer, MultipleSelectionAnswer)
    values = [
        (
            value.value
            if isinstance(value, SelectedChoice)
            else value.text if isinstance(value, FreeFormOther) else ""
        )
        for value in answer.values
    ]
    assert all(values)
    return f"completed:{'|'.join(values)}"


def _projection(
    stream_id: str,
    run_id: str,
    turn_id: str,
    sequence: int,
    kind: StreamItemKind,
    *,
    text: str | None = None,
) -> StreamConsumerProjection:
    return StreamConsumerProjection(
        stream_session_id=stream_id,
        run_id=run_id,
        turn_id=turn_id,
        sequence=sequence,
        kind=kind,
        channel=stream_channel_for_kind(kind),
        correlation=StreamItemCorrelation(),
        text_delta=text,
        terminal_outcome=(
            StreamTerminalOutcome.COMPLETED
            if kind is StreamItemKind.STREAM_COMPLETED
            else None
        ),
        usage=(
            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
            if kind is StreamItemKind.STREAM_COMPLETED
            else None
        ),
    )


def _canonical_text_response(text: str) -> TextGenerationResponse:
    async def source() -> AsyncIterator[CanonicalStreamItem]:
        for sequence, kind, delta in (
            (0, StreamItemKind.STREAM_STARTED, None),
            (1, StreamItemKind.ANSWER_DELTA, text),
            (2, StreamItemKind.ANSWER_DONE, None),
            (3, StreamItemKind.STREAM_COMPLETED, None),
        ):
            yield CanonicalStreamItem(
                stream_session_id="pty-provider-stream",
                run_id="pty-provider-run",
                turn_id="pty-provider-turn",
                sequence=sequence,
                kind=kind,
                channel=stream_channel_for_kind(kind),
                text_delta=delta,
                terminal_outcome=(
                    StreamTerminalOutcome.COMPLETED
                    if kind is StreamItemKind.STREAM_COMPLETED
                    else None
                ),
                usage=(
                    {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "total_tokens": 2,
                    }
                    if kind is StreamItemKind.STREAM_COMPLETED
                    else None
                ),
                provider_family="openai",
            )

    return TextGenerationResponse(
        source,
        logger=getLogger(),
        use_async_generator=True,
    )


def _child(
    tty_path: str,
    stdin_fd: int,
    stdout_fd: int,
    stderr_fd: int,
    result_fd: int,
    *,
    real_orchestrator: bool,
    case: str,
) -> None:
    dup2(stdin_fd, 0)
    dup2(stdout_fd, 1)
    dup2(stderr_fd, 2)
    response_patch: Any
    enter_patch: Any
    text_response_patch: Any
    if real_orchestrator:
        manager = boundary_fixture._ModelManager()
        harness = boundary_fixture._Harness(
            broker=boundary_fixture._BoundaryBroker(),
            manager=manager,
        )
        orchestrator = harness.orchestrator
        response_patch = nullcontext()
        real_enter = boundary_fixture.Orchestrator.__aenter__

        async def enter_loaded_orchestrator(
            entered: object,
        ) -> object:
            result = await real_enter(entered)
            result._last_engine_agent = harness.agent
            return result

        enter_patch = patch.object(
            boundary_fixture.Orchestrator,
            "__aenter__",
            enter_loaded_orchestrator,
        )
        text_response_patch = patch.object(
            boundary_fixture,
            "_text_response",
            _canonical_text_response,
        )
    else:
        smoke_orchestrator = _Orchestrator(case)
        orchestrator = smoke_orchestrator
        response_patch = patch.object(
            agent_cmds,
            "OrchestratorResponse",
            _Response,
        )
        enter_patch = nullcontext()
        text_response_patch = nullcontext()
    child_stdin = open(0, closefd=False)
    child_stdout = open(1, "w", buffering=1, closefd=False)
    child_stderr = open(2, "w", buffering=1, closefd=False)
    child_argv = [
        "avalan",
        "agent",
        "run",
        "--engine-uri",
        "fake-model",
        "--quiet",
        "--no-repl",
        "--no-session",
        "--skip-hub-access-check",
        "--theme",
        "basic",
        "--tty",
        tty_path,
    ]
    try:
        with (
            patch.object(
                OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value=orchestrator),
            ),
            response_patch,
            enter_patch,
            text_response_patch,
            patch("sys.stdin", child_stdin),
            patch("sys.stdout", child_stdout),
            patch("sys.stderr", child_stderr),
            patch("sys.argv", child_argv),
            patch.object(cli_package, "stdin", child_stdin),
            patch.object(
                cli_main.CLI,
                "_needs_hf_token",
                new=AsyncMock(return_value=False),
            ),
            patch.object(
                cli_main,
                "_huggingface_hub_class",
                return_value=lambda *_args: object(),
            ),
        ):
            cli_main.main()
        result = (
            {
                "initial_prompt": boundary_fixture._user_prompt(
                    manager.calls[0].context.input
                ),
                "provider_calls": len(manager.calls),
            }
            if real_orchestrator
            else {
                "calls": smoke_orchestrator.calls,
                "provider_calls": smoke_orchestrator.provider_calls,
            }
        )
        write(result_fd, dumps(result).encode())
    except BaseException:
        print_exc(file=child_stderr)
        child_stderr.flush()
        _exit(1)
    else:
        _exit(0)


def _run_pty_case(
    *,
    real_orchestrator: bool,
    case: str = "confirmation",
    control_input: bytes | None = b"yes\n",
    prompt_marker: bytes = b"Answer yes or no:\n",
) -> tuple[int | None, dict[int, bytes], bytes]:
    master, slave = openpty()
    setraw(slave)
    tty_path = ttyname(slave)
    stdin_read, stdin_write = pipe()
    stdout_read, stdout_write = pipe()
    stderr_read, stderr_write = pipe()
    result_read, result_write = pipe()
    child = fork()
    if child == 0:
        for descriptor in (
            master,
            slave,
            stdin_write,
            stdout_read,
            stderr_read,
            result_read,
        ):
            close(descriptor)
        _child(
            tty_path,
            stdin_read,
            stdout_write,
            stderr_write,
            result_write,
            real_orchestrator=real_orchestrator,
            case=case,
        )

    status = None
    streams = {stdout_read: b"", stderr_read: b"", result_read: b""}
    control = b""
    try:
        for descriptor in (
            stdin_read,
            stdout_write,
            stderr_write,
            result_write,
        ):
            close(descriptor)
        write(stdin_write, b"initial prompt\n")
        close(stdin_write)
        for descriptor in streams:
            set_blocking(descriptor, False)
        deadline = monotonic() + 5
        while prompt_marker not in control and monotonic() < deadline:
            readable, _, _ = select([master, *streams], [], [], 0.05)
            for descriptor in readable:
                chunk = read(descriptor, 4096)
                if descriptor == master:
                    control += chunk
                else:
                    streams[descriptor] += chunk
            waited, status = waitpid(child, WNOHANG)
            if waited:
                break
        assert prompt_marker in control, streams[stderr_read].decode()
        if control_input is None:
            close(master)
            master = -1
        else:
            write(master, control_input)

        deadline = monotonic() + 5
        while monotonic() < deadline:
            monitored = [*streams]
            if master >= 0:
                monitored.append(master)
            readable, _, _ = select(monitored, [], [], 0.05)
            for descriptor in readable:
                chunk = read(descriptor, 4096)
                if descriptor == master:
                    control += chunk
                else:
                    streams[descriptor] += chunk
            waited, status = waitpid(child, WNOHANG)
            if waited:
                break
    finally:
        if status is None:
            kill(child, SIGKILL)
            waitpid(child, 0)
        for descriptor in (*streams, master, slave):
            if descriptor < 0:
                continue
            close(descriptor)

    return status, streams, control


def test_piped_prompt_and_pty_clarification_complete_one_run() -> None:
    status, streams, control = _run_pty_case(real_orchestrator=False)
    stdout, stderr, result = streams.values()
    assert status == 0, stderr.decode()
    assert stdout.decode() == "completed:yes\n"
    assert stderr == b""
    assert loads(result) == {
        "calls": ["initial prompt"],
        "provider_calls": 1,
    }
    assert (
        control.decode()
        == "Input required\n"
        "Reason: Need confirmation.\n"
        "Controls: :decline decline this request; :cancel cancel only this "
        "input; :cancel-run cancel the containing run; :steer TEXT send "
        "steering; :help show help. Prefix control-looking text with an "
        "extra ':'.\n"
        "\n"
        "Question 1/1 - Confirmation\n"
        "Proceed?\n"
        "Answer yes or no:\n"
    )


def test_semantic_text_multiline_and_multiple_other_rows() -> None:
    rows = (
        ("text", b"Ada\n", b"Enter one line:\n", "completed:Ada\n"),
        (
            "multiline",
            b"line one\n..\n.\n",
            (
                b"Enter text; finish with a line containing only '.'. "
                b"Enter '..' for a literal '.'.\n"
            ),
            "completed:line one|.\n",
        ),
        (
            "multiple_other",
            b"1,3\ncustom\n",
            b"Select numbers separated by commas, or enter 'none':\n",
            "completed:fast|custom\n",
        ),
    )
    for case, control_input, prompt_marker, expected_stdout in rows:
        status, streams, control = _run_pty_case(
            real_orchestrator=False,
            case=case,
            control_input=control_input,
            prompt_marker=prompt_marker,
        )
        stdout, stderr, result = streams.values()
        assert status == 0, f"{case}: {stderr.decode()}"
        assert stdout.decode() == expected_stdout, case
        assert stderr == b"", case
        assert loads(result) == {
            "calls": ["initial prompt"],
            "provider_calls": 1,
        }
        assert prompt_marker in control, case
    assert b"Enter the Other value:\n" in control


def test_decline_input_cancel_run_cancel_and_disappearance_are_distinct() -> (
    None
):
    rows = (
        ("decline", b":decline\n", "declined\n"),
        (
            "cancel_input",
            b":cancel\n",
            "disconnected:handler_cancelled\n",
        ),
        ("cancel_run", b":cancel-run\n", ""),
    )
    for case, control_input, expected_stdout in rows:
        status, streams, control = _run_pty_case(
            real_orchestrator=False,
            case=case,
            control_input=control_input,
        )
        stdout, stderr, result = streams.values()
        assert status == 0, f"{case}: {stderr.decode()}"
        assert stdout.decode() == expected_stdout, case
        assert stderr == b"", case
        assert loads(result) == {
            "calls": ["initial prompt"],
            "provider_calls": 1,
        }
        assert control.endswith(b"Answer yes or no:\n"), case


def test_terminal_disappearance_is_bounded_and_next_run_receives_bytes() -> (
    None
):
    status, streams, control = _run_pty_case(
        real_orchestrator=False,
        case="disappear",
        control_input=None,
    )
    stdout, stderr, result = streams.values()
    assert status == 0, stderr.decode()
    assert stdout.decode() == "disconnected:control_channel_closed\n"
    assert stderr == b""
    assert loads(result)["provider_calls"] == 1
    assert control.endswith(b"Answer yes or no:\n")

    status, streams, control = _run_pty_case(
        real_orchestrator=False,
        case="text",
        control_input=b"Ada\n",
        prompt_marker=b"Enter one line:\n",
    )
    stdout, stderr, result = streams.values()
    assert status == 0, stderr.decode()
    assert stdout.decode() == "completed:Ada\n"
    assert stderr == b""
    assert loads(result)["provider_calls"] == 1
    assert control.endswith(b"Enter one line:\n")


def test_cancelled_interaction_preserves_bytes_for_next_prompt() -> None:
    status, streams, control = _run_pty_case(
        real_orchestrator=False,
        case="cancel_then_text",
        control_input=b":cancel\nAda\n",
    )
    stdout, stderr, result = streams.values()
    assert status == 0, stderr.decode()
    assert stdout.decode() == "completed:Ada\n"
    assert stderr == b""
    assert loads(result) == {
        "calls": ["initial prompt"],
        "provider_calls": 1,
    }
    assert control.count(b"Input required\n") == 2
    assert control.endswith(b"Enter one line:\n")


def test_real_orchestrator_engine_agent_resumes_same_run() -> None:
    status, streams, control = _run_pty_case(real_orchestrator=True)
    stdout, stderr, result = streams.values()
    assert status == 0, stderr.decode()
    assert stdout.decode() == "done:initial prompt\n"
    assert stderr == b""
    assert loads(result) == {
        "initial_prompt": "initial prompt",
        "provider_calls": 2,
    }
    assert (
        control.decode()
        == "Input required\n"
        "Reason: Need one bounded decision.\n"
        "Controls: :decline decline this request; :cancel cancel only this "
        "input; :cancel-run cancel the containing run; :steer TEXT send "
        "steering; :help show help. Prefix control-looking text with an "
        "extra ':'.\n"
        "\n"
        "Question 1/1 - Confirmation\n"
        "Continue?\n"
        "Answer yes or no:\n"
    )


def test_real_orchestrator_run_cancel_owns_containing_run_cleanup() -> None:
    status, streams, control = _run_pty_case(
        real_orchestrator=True,
        control_input=b":cancel-run\n",
    )
    stdout, stderr, result = streams.values()
    assert status == 0, stderr.decode()
    assert stdout == b""
    assert stderr == b""
    assert loads(result) == {
        "initial_prompt": "initial prompt",
        "provider_calls": 1,
    }
    assert control.endswith(b"Answer yes or no:\n")
