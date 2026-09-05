"""Exercise attached CLI input through real parser, pipes, and a PTY."""

from asyncio import sleep as async_sleep
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import nullcontext
from datetime import UTC, datetime
from fcntl import ioctl
from importlib.util import module_from_spec, spec_from_file_location
from json import dumps, loads
from logging import CRITICAL, disable, getLogger
from os import (
    _exit,
    close,
    dup2,
    pipe,
    read,
    set_blocking,
    setsid,
    ttyname,
    waitstatus_to_exitcode,
    write,
)
from pathlib import Path
from pty import openpty
from select import select
from subprocess import Popen
from sys import argv, executable
from sys import modules as system_modules
from termios import TIOCSCTTY
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
    StreamReasoningRepresentation,
    StreamTerminalOutcome,
    StreamVisibility,
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

_CANCELLED_STDERR = (
    b'{"envelope_id": "cli.cancelled.v1", "payload": '
    b'{"channel": "control", "kind": "cancelled"}}\n'
)


class _Response:
    input_token_count = 1
    can_think: bool
    is_thinking: bool

    def __init__(
        self,
        runtime: AttachedInteractionRuntime,
        owner: "_Orchestrator",
    ) -> None:
        self.runtime = runtime
        self.owner = owner
        self.can_think = owner.case == "live_text"
        self.is_thinking = self.can_think
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
            try:
                self.owner.provider_calls += 1
                yield _projection(
                    stream_session_id,
                    run_id,
                    turn_id,
                    0,
                    StreamItemKind.STREAM_STARTED,
                )
                sequence = 1
                if self.owner.case == "live_text":
                    yield _projection(
                        stream_session_id,
                        run_id,
                        turn_id,
                        sequence,
                        StreamItemKind.REASONING_DELTA,
                        text="REASONING_ONCE",
                    )
                    sequence += 1
                    await async_sleep(0.25)
                outcome = await self.runtime.handler(
                    InputHandlerContext(request=_request(self.owner.case))
                )
                assert isinstance(
                    outcome,
                    (InputHandlerResolution, InputHandlerDisconnected),
                )
                self.owner.handler_outcomes.append(_outcome_text(outcome))
                if self.cancellation_checker is not None:
                    await self.cancellation_checker()
                if self.owner.case == "cancel_then_text":
                    assert isinstance(outcome, InputHandlerDisconnected)
                    assert (
                        outcome.reason
                        is InputDisconnectReason.HANDLER_CANCELLED
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
                    self.owner.handler_outcomes.append(_outcome_text(outcome))
                if self.owner.case == "live_text":
                    yield _projection(
                        stream_session_id,
                        run_id,
                        turn_id,
                        sequence,
                        StreamItemKind.REASONING_DELTA,
                        text="REASONING_AFTER",
                    )
                    sequence += 1
                    yield _projection(
                        stream_session_id,
                        run_id,
                        turn_id,
                        sequence,
                        StreamItemKind.REASONING_DONE,
                    )
                    sequence += 1
                answer_text = _outcome_text(outcome)
                yield _projection(
                    stream_session_id,
                    run_id,
                    turn_id,
                    sequence,
                    StreamItemKind.ANSWER_DELTA,
                    text=answer_text,
                )
                sequence += 1
                yield _projection(
                    stream_session_id,
                    run_id,
                    turn_id,
                    sequence,
                    StreamItemKind.ANSWER_DONE,
                )
                sequence += 1
                yield _projection(
                    stream_session_id,
                    run_id,
                    turn_id,
                    sequence,
                    StreamItemKind.STREAM_COMPLETED,
                )
                self.owner.stream_completed = True
            finally:
                self.owner.stream_cleanup_count += 1

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
        self.handler_outcomes: list[str] = []
        self.stream_cleanup_count = 0
        self.stream_completed = False
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
        "steer",
        "disappear",
        "cancel_then_text",
    }:
        question = ConfirmationQuestion(
            question_id=QuestionId("confirm"),
            prompt="Proceed?",
            required=True,
        )
    elif case in {"live_text", "text"}:
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
        visibility=(
            StreamVisibility.PRIVATE
            if kind is StreamItemKind.REASONING_DELTA
            else StreamVisibility.PUBLIC
        ),
        reasoning_representation=(
            StreamReasoningRepresentation.SUMMARY
            if kind is StreamItemKind.REASONING_DELTA
            else None
        ),
        segment_instance_ordinal=(
            0 if kind is StreamItemKind.REASONING_DELTA else None
        ),
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
        responses: list[Any] = []
        orchestrator_call = boundary_fixture.Orchestrator.__call__

        async def capture_response(
            loaded_orchestrator: object,
            *args: object,
            **kwargs: object,
        ) -> object:
            response = await orchestrator_call(
                loaded_orchestrator, *args, **kwargs
            )
            responses.append(response)
            return response

        response_patch = patch.object(
            boundary_fixture.Orchestrator,
            "__call__",
            capture_response,
        )
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
        "--no-repl",
        "--no-session",
        "--skip-hub-access-check",
        "--theme",
        "basic",
        "--tty",
        tty_path,
    ]
    if case == "live_text":
        child_argv.append("--display-reasoning")
    else:
        child_argv.append("--quiet")

    def execution_result() -> dict[str, object]:
        if real_orchestrator:
            initial_sources = tuple(manager.initial_sources.values())
            real_result = {
                "initial_prompt": boundary_fixture._user_prompt(
                    manager.calls[0].context.input
                ),
                "provider_calls": len(manager.calls),
                "initial_source_aclose_calls": sum(
                    source.aclose_calls for source in initial_sources
                ),
            }
            assert responses
            response = responses[-1]
            execution = response.execution
            assert execution is not None
            real_result.update(
                execution_status=execution.status.value,
                interaction_states=[
                    entry.request.state.value
                    for entry in execution.ledger
                    if entry.request is not None
                ],
                pending_request=execution.snapshot.pending_request is not None,
                cleanup_started=execution.snapshot.cleanup_started,
                interaction_cleanup_complete=(
                    response._interaction_cleanup_complete
                ),
                pending_interaction_task=(
                    response._pending_interaction_task is not None
                ),
                pending_tool_batch_task=(
                    response._pending_tool_batch_task is not None
                ),
            )
            return real_result
        fake_result: dict[str, object] = {
            "calls": smoke_orchestrator.calls,
            "provider_calls": smoke_orchestrator.provider_calls,
        }
        if case == "cancel_run":
            fake_result.update(
                handler_outcomes=smoke_orchestrator.handler_outcomes,
                stream_cleanup_count=(smoke_orchestrator.stream_cleanup_count),
                stream_completed=smoke_orchestrator.stream_completed,
            )
        return fake_result

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
            patch.object(cli_main, "_is_cuda_available", return_value=False),
            patch.object(cli_main, "_is_mps_available", return_value=False),
            patch.object(
                agent_cmds,
                "_agent_display_models",
                return_value=["fake-model"],
            ),
        ):
            cli_main.main()
        write(result_fd, dumps(execution_result()).encode())
    except SystemExit as error:
        write(result_fd, dumps(execution_result()).encode())
        child_stderr.flush()
        _exit(error.code if isinstance(error.code, int) else 1)
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
    control_input_chunks: tuple[bytes, ...] = (),
    prompt_marker: bytes = b"Answer yes or no:\n",
    attached_stdin: bool = False,
    control_observations: list[bytes] | None = None,
    raw_tty: bool = True,
    terminal_stdout: bool = False,
) -> tuple[int | None, dict[int, bytes], bytes]:
    master, slave = openpty()
    if raw_tty:
        setraw(slave)
    tty_path = ttyname(slave)
    stdin_read, stdin_write = pipe()
    stdout_read, stdout_write = pipe()
    stderr_read, stderr_write = pipe()
    result_read, result_write = pipe()
    process = Popen(
        (
            executable,
            str(Path(__file__).resolve()),
            "--pty-child",
            tty_path,
            str(slave),
            str(stdin_read),
            str(stdout_write),
            str(stderr_write),
            str(result_write),
            "1" if real_orchestrator else "0",
            case,
            "1" if attached_stdin else "0",
            "1" if terminal_stdout else "0",
        ),
        pass_fds=(
            slave,
            stdin_read,
            stdout_write,
            stderr_write,
            result_write,
        ),
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
        if attached_stdin:
            close(stdin_write)
        else:
            write(stdin_write, b"initial prompt\n")
            close(stdin_write)
        for descriptor in streams:
            set_blocking(descriptor, False)
        if attached_stdin:
            deadline = monotonic() + 5
            while not streams[stdout_read] and monotonic() < deadline:
                readable, _, _ = select(
                    [master, *streams],
                    [],
                    [],
                    0.05,
                )
                for descriptor in readable:
                    chunk = read(descriptor, 4096)
                    if descriptor == master:
                        control += chunk
                    else:
                        streams[descriptor] += chunk
                child_status = process.poll()
                if child_status is not None:
                    status = child_status
                    break
            assert streams[stdout_read], streams[stderr_read].decode()
            write(master, b"initial prompt\n")
        deadline = monotonic() + 5
        while prompt_marker not in control and monotonic() < deadline:
            readable, _, _ = select([master, *streams], [], [], 0.05)
            for descriptor in readable:
                chunk = read(descriptor, 4096)
                if descriptor == master:
                    control += chunk
                else:
                    streams[descriptor] += chunk
            child_status = process.poll()
            if child_status is not None:
                status = child_status
                break
        assert prompt_marker in control, {
            "stdout": streams[stdout_read].decode(),
            "stderr": streams[stderr_read].decode(),
            "control": control.decode(),
        }
        if control_input_chunks:
            for chunk in control_input_chunks:
                write(master, chunk)
                chunk_deadline = monotonic() + 0.25
                child_exited = False
                while monotonic() < chunk_deadline:
                    readable, _, _ = select(
                        [master, *streams],
                        [],
                        [],
                        0.05,
                    )
                    for descriptor in readable:
                        data = read(descriptor, 4096)
                        if descriptor == master:
                            control += data
                        else:
                            streams[descriptor] += data
                    child_status = process.poll()
                    if child_status is not None:
                        status = child_status
                        child_exited = True
                        break
                if child_exited:
                    break
                if control_observations is not None:
                    control_observations.append(control)
        elif control_input is None:
            close(master)
            master = -1
        else:
            write(master, control_input)

        deadline = monotonic() + 5
        while status is None and monotonic() < deadline:
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
            child_status = process.poll()
            if child_status is not None:
                status = child_status
                break
    finally:
        if status is None:
            process.kill()
            process.wait(timeout=5)
        for descriptor in (*streams, master, slave):
            if descriptor < 0:
                continue
            close(descriptor)

    raw_status = (
        None if status is None else status << 8 if status >= 0 else -status
    )
    return raw_status, streams, control


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


def test_attached_tty_prompt_and_clarification_complete_one_run() -> None:
    status, streams, control = _run_pty_case(
        real_orchestrator=False,
        attached_stdin=True,
    )
    stdout, stderr, result = streams.values()
    assert status == 0, stderr.decode()
    assert stdout.decode().endswith("completed:yes\n")
    assert stderr == b""
    assert loads(result) == {
        "calls": ["initial prompt"],
        "provider_calls": 1,
    }
    assert control.endswith(b"Answer yes or no:\n")


def test_live_reasoning_releases_terminal_while_text_is_typed() -> None:
    observations: list[bytes] = []
    status, streams, control = _run_pty_case(
        real_orchestrator=False,
        case="live_text",
        control_input_chunks=(b"A", b"d", b"a", b"\n"),
        prompt_marker=b"Enter one line:\r\n",
        control_observations=observations,
        raw_tty=False,
        terminal_stdout=True,
    )
    stdout, stderr, result = streams.values()
    assert status == 0, stderr.decode()
    assert stdout == b""
    assert stderr == b""
    assert loads(result) == {
        "calls": ["initial prompt"],
        "provider_calls": 1,
    }
    assert b"REASONING_ONCE" in control
    assert len(observations) >= 3
    pre_prompt_reasoning_count = observations[0].count(b"REASONING_ONCE")
    assert pre_prompt_reasoning_count > 0
    assert (
        len(
            {
                observation.count(b"REASONING_ONCE")
                for observation in observations[:3]
            }
        )
        == 1
    )
    assert control.count(b"REASONING_ONCE") == pre_prompt_reasoning_count
    assert all(
        b"REASONING_AFTER" not in observation
        for observation in observations[:3]
    )
    assert b"REASONING_AFTER" in control
    assert b"Enter one line:\r\nA" in observations[0]
    assert b"Enter one line:\r\nAd" in observations[1]
    assert b"Enter one line:\r\nAda" in observations[2]
    assert b"Enter one line:\r\nAda\r\n" in control


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
        ("decline", b":decline\n", "declined\n", 0, b""),
        (
            "cancel_input",
            b":cancel\n",
            "disconnected:handler_cancelled\n",
            0,
            b"",
        ),
        (
            "steer",
            b":steer redirect\nyes\n",
            "completed:yes\n",
            0,
            b"",
        ),
        (
            "cancel_run",
            b":cancel-run\n",
            "",
            130,
            _CANCELLED_STDERR,
        ),
    )
    for (
        case,
        control_input,
        expected_stdout,
        expected_exit,
        expected_stderr,
    ) in rows:
        status, streams, control = _run_pty_case(
            real_orchestrator=False,
            case=case,
            control_input=control_input,
        )
        stdout, stderr, result = streams.values()
        assert status is not None
        assert (
            waitstatus_to_exitcode(status) == expected_exit
        ), f"{case}: {stderr.decode()}"
        assert stdout.decode() == expected_stdout, case
        assert stderr == expected_stderr, case
        observed_result = loads(result)
        expected_result: dict[str, object] = {
            "calls": ["initial prompt"],
            "provider_calls": 1,
        }
        if case == "cancel_run":
            expected_result.update(
                handler_outcomes=["disconnected:handler_cancelled"],
                stream_cleanup_count=1,
                stream_completed=False,
            )
        assert observed_result == expected_result
        if case == "steer":
            assert (
                b"Invalid input: Steering is unavailable in this CLI "
                b"session.\n"
                in control
            )
            assert stdout == b"completed:yes\n"
        else:
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
        "initial_source_aclose_calls": 1,
        "execution_status": "completed",
        "interaction_states": ["pending", "answered"],
        "pending_request": False,
        "cleanup_started": False,
        "interaction_cleanup_complete": False,
        "pending_interaction_task": False,
        "pending_tool_batch_task": False,
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
    assert status is not None
    assert waitstatus_to_exitcode(status) == 130, stderr.decode()
    assert stdout == b""
    assert stderr == _CANCELLED_STDERR
    assert loads(result) == {
        "initial_prompt": "initial prompt",
        "provider_calls": 1,
        "initial_source_aclose_calls": 1,
        "execution_status": "cancelled",
        "interaction_states": ["pending"],
        "pending_request": False,
        "cleanup_started": True,
        "interaction_cleanup_complete": True,
        "pending_interaction_task": False,
        "pending_tool_batch_task": False,
    }
    assert control.endswith(b"Answer yes or no:\n")


def _run_pty_child() -> None:
    """Dispatch one fresh-interpreter CLI PTY child invocation."""
    assert len(argv) == 12
    assert argv[1] == "--pty-child"
    tty_path = argv[2]
    slave = int(argv[3])
    stdin_read = int(argv[4])
    stdout_write = int(argv[5])
    stderr_write = int(argv[6])
    result_write = int(argv[7])
    real_orchestrator = argv[8] == "1"
    case = argv[9]
    attached_stdin = argv[10] == "1"
    terminal_stdout = argv[11] == "1"
    child_stdin = slave if attached_stdin else stdin_read
    child_stdout = slave if terminal_stdout else stdout_write
    if attached_stdin:
        setsid()
        ioctl(slave, TIOCSCTTY, 0)
    for descriptor in (slave, stdin_read, stdout_write):
        if descriptor not in {child_stdin, child_stdout}:
            close(descriptor)
    disable(CRITICAL)
    _child(
        tty_path,
        child_stdin,
        child_stdout,
        stderr_write,
        result_write,
        real_orchestrator=real_orchestrator,
        case=case,
    )


if __name__ == "__main__":
    _run_pty_child()
