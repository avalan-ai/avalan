"""Verify operation-specific rejected-result command typing."""

from os import pathsep
from pathlib import Path

from mypy import api as mypy_api
from pytest import MonkeyPatch


def test_rejected_results_reject_wrong_operation_commands(
    monkeypatch: MonkeyPatch,
) -> None:
    """Require one static argument-type error per hostile constructor."""
    root = Path(__file__).resolve().parents[2]
    fixture = (
        root
        / "tests"
        / "interaction_type_contracts"
        / "rejected_result_command_negative.py"
    )
    monkeypatch.setenv(
        "MYPYPATH",
        pathsep.join((str(root / "tests"), str(root / "src"))),
    )

    stdout, stderr, status = mypy_api.run(
        [
            "--strict",
            "--show-error-codes",
            "--no-error-summary",
            "--no-pretty",
            str(fixture),
        ]
    )

    diagnostics = tuple(
        line.removeprefix(f"{root}/")
        for line in stdout.splitlines()
        if ": error:" in line
    )
    assert status == 1
    assert stderr == ""
    assert diagnostics == (
        (
            "tests/interaction_type_contracts/"
            "rejected_result_command_negative.py:35:35: error: Argument "
            '"command" to "CreateInteractionRejected" has incompatible type '
            '"CancelInteractionCommand"; expected "CreateInteractionCommand"  '
            "[arg-type]"
        ),
        (
            "tests/interaction_type_contracts/"
            "rejected_result_command_negative.py:36:35: error: Argument "
            '"command" to "CancelInteractionRejected" has incompatible type '
            '"CreateInteractionCommand"; expected "CancelInteractionCommand"  '
            "[arg-type]"
        ),
        (
            "tests/interaction_type_contracts/"
            "rejected_result_command_negative.py:37:40: error: Argument "
            '"command" to "TerminalizeInteractionRejected" has incompatible '
            'type "CancelInteractionCommand"; expected '
            '"TerminalizeInteractionCommand"  [arg-type]'
        ),
        (
            "tests/interaction_type_contracts/"
            "rejected_result_command_negative.py:38:35: error: Argument "
            '"command" to "ScopeCancellationRejected" has incompatible type '
            '"SupersedeInteractionScopeCommand"; expected '
            '"TerminalizeInteractionScopeCommand"  [arg-type]'
        ),
        (
            "tests/interaction_type_contracts/"
            "rejected_result_command_negative.py:39:35: error: Argument "
            '"command" to "ScopeSupersessionRejected" has incompatible type '
            '"TerminalizeInteractionScopeCommand"; expected '
            '"SupersedeInteractionScopeCommand"  [arg-type]'
        ),
        (
            "tests/interaction_type_contracts/"
            "rejected_result_command_negative.py:40:33: error: Argument "
            '"command" to "DueInteractionsRejected" has incompatible type '
            '"CreateInteractionCommand"; expected '
            '"TerminalizeDueInteractionsCommand"  [arg-type]'
        ),
        (
            "tests/interaction_type_contracts/rejected_result_command_negative.py:41:41:"
            ' error: Argument "command" to "InteractionPresentationRejected"'
            ' has incompatible type "RecordControllerActivityCommand";'
            ' expected "PresentInteractionCommand | DetachInteractionCommand" '
            " [arg-type]"
        ),
        (
            "tests/interaction_type_contracts/"
            "rejected_result_command_negative.py:42:36: error: Argument "
            '"command" to "ControllerActivityRejected" has incompatible type '
            '"DetachInteractionCommand"; expected '
            '"RecordControllerActivityCommand"  [arg-type]'
        ),
    )
