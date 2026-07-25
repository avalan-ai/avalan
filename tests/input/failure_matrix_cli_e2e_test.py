"""Exercise reachable CLI failure cells through the real main/PTY boundary."""

from collections.abc import Callable
from json import dumps, loads
from os import waitstatus_to_exitcode
from pathlib import Path
from sys import path as sys_path

import pytest

sys_path.append(str(Path(__file__).parents[1] / "cli"))

import interaction_cli_pty_e2e_test as cli_support  # noqa: E402

_ATTACHED_SURFACE = "cli-agent-run-attached-tty"
_PIPED_SURFACE = "cli-agent-run-piped-with-tty"
_CASES = (
    ("INPUT-F-02", _PIPED_SURFACE),
    ("INPUT-F-03", _PIPED_SURFACE),
    ("INPUT-F-04", _PIPED_SURFACE),
    ("INPUT-F-04", _ATTACHED_SURFACE),
    ("INPUT-F-05", _PIPED_SURFACE),
    ("INPUT-F-05", _ATTACHED_SURFACE),
    ("INPUT-F-06", _PIPED_SURFACE),
    ("INPUT-F-06", _ATTACHED_SURFACE),
    ("INPUT-F-10", _PIPED_SURFACE),
    ("INPUT-F-10", _ATTACHED_SURFACE),
)


def _evidence(
    condition_id: str,
    surface_id: str,
    transition: tuple[str, str],
    public_result_id: str,
    public_result: dict[str, object],
    status: tuple[str, str],
    provider_calls: int,
) -> dict[str, object]:
    return {
        "condition_id": condition_id,
        "surface_id": surface_id,
        "transition_from": transition[0],
        "transition_to": transition[1],
        "public_result_id": public_result_id,
        "public_result": public_result,
        "status_key": status[0],
        "status_value": status[1],
        "provider_call_count": provider_calls,
        "domain_side_effect_count": 0,
    }


def _run(
    condition_id: str,
    record_property: Callable[[str, object], None],
    surface_id: str,
) -> None:
    assert surface_id in {_ATTACHED_SURFACE, _PIPED_SURFACE}
    cases = {
        "INPUT-F-02": (True, b"yes\n", b"Answer yes or no:\n"),
        "INPUT-F-03": (True, None, b"Answer yes or no:\n"),
        "INPUT-F-04": (False, b"maybe\nyes\n", b"Answer yes or no:\n"),
        "INPUT-F-05": (
            False,
            b"9\n1\n",
            b"Select numbers separated by commas, or enter 'none':\n",
        ),
        "INPUT-F-06": (False, b"\nyes\n", b"Answer yes or no:\n"),
        "INPUT-F-10": (True, b":cancel-run\n", b"Answer yes or no:\n"),
    }
    real, control_input, marker = cases[condition_id]
    status, streams, control = cli_support._run_pty_case(
        real_orchestrator=real,
        case=(
            "multiple_other"
            if condition_id == "INPUT-F-05"
            else "confirmation"
        ),
        control_input=control_input,
        prompt_marker=marker,
        attached_stdin=surface_id == _ATTACHED_SURFACE,
    )
    stdout, stderr, result = streams.values()
    expected_exit = 130 if condition_id == "INPUT-F-10" else 0
    assert status is not None
    observed_exit = waitstatus_to_exitcode(status)
    assert observed_exit == expected_exit, stderr.decode()
    observed = loads(result)
    provider_calls = observed["provider_calls"]
    assert marker in control
    if condition_id != "INPUT-F-10":
        assert stderr == b""
    if condition_id == "INPUT-F-02":
        assert stdout == b"done:initial prompt\n" and provider_calls == 2
        assert observed["interaction_states"] == ["pending", "answered"]
        evidence = _evidence(
            condition_id,
            surface_id,
            ("pending", "answered"),
            "cli.process_exit.v1",
            {
                "exit_code": observed_exit,
                "stdout": stdout.decode(),
                "stderr": stderr.decode(),
            },
            ("exit", str(observed_exit)),
            provider_calls,
        )
    elif condition_id == "INPUT-F-03":
        assert stdout == b"done:initial prompt\n" and provider_calls == 2
        assert observed["interaction_states"] == ["pending", "unavailable"]
        evidence = _evidence(
            condition_id,
            surface_id,
            ("pending", "unavailable"),
            "cli.process_exit.v1",
            {
                "exit_code": observed_exit,
                "stdout": stdout.decode(),
                "stderr": stderr.decode(),
            },
            ("exit", str(observed_exit)),
            provider_calls,
        )
    elif condition_id == "INPUT-F-10":
        if surface_id == _PIPED_SURFACE:
            assert stdout == b""
        else:
            assert stdout and b"\n" not in stdout
        assert provider_calls == 1
        assert stderr == cli_support._CANCELLED_STDERR
        public_envelope = loads(stderr)
        assert public_envelope == {
            "envelope_id": "cli.cancelled.v1",
            "payload": {"channel": "control", "kind": "cancelled"},
        }
        assert observed["execution_status"] == "cancelled"
        assert observed["interaction_states"] == ["pending"]
        assert observed["pending_request"] is False
        assert observed["cleanup_started"] is True
        assert observed["interaction_cleanup_complete"] is True
        assert observed["pending_interaction_task"] is False
        assert observed["pending_tool_batch_task"] is False
        assert observed["initial_source_aclose_calls"] == 1
        evidence = _evidence(
            condition_id,
            surface_id,
            ("pending", "cancelled"),
            public_envelope["envelope_id"],
            public_envelope["payload"],
            ("exit", str(observed_exit)),
            provider_calls,
        )
    else:
        assert b"Invalid input" in control and provider_calls == 1
        invalid_lines = tuple(
            line
            for line in control.decode().splitlines()
            if line.startswith("Invalid input: ")
        )
        assert len(invalid_lines) == 1
        evidence = _evidence(
            condition_id,
            surface_id,
            ("pending", "pending"),
            "cli.terminal_validation.v1",
            {
                "stream": "control",
                "text": invalid_lines[0],
            },
            ("interaction_state", "pending"),
            provider_calls,
        )
    record_property(
        "failure_matrix_evidence", dumps([evidence], sort_keys=True)
    )


@pytest.mark.parametrize(
    ("condition_id", "surface_id"),
    _CASES,
    ids=tuple(
        f"{condition_id}|{surface_id}" for condition_id, surface_id in _CASES
    ),
)
def test_cli_agent_failure(
    condition_id: str,
    surface_id: str,
    record_property: Callable[[str, object], None],
) -> None:
    """Exercise one exact reachable agent CLI condition."""
    _run(condition_id, record_property, surface_id)
