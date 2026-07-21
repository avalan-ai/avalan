"""Verify rejected store trust-boundary constructions statically."""

from os import pathsep
from pathlib import Path

from mypy import api as mypy_api
from pytest import MonkeyPatch


def test_rejected_store_trust_boundary_has_exact_diagnostics(
    monkeypatch: MonkeyPatch,
) -> None:
    """Require one stable type error for every hostile construction."""
    root = Path(__file__).resolve().parents[2]
    fixture = (
        root
        / "tests"
        / "interaction_type_contracts"
        / "rejected_store_trust_boundary_negative.py"
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
            "rejected_store_trust_boundary_negative.py:5:1: error: "
            'Module "avalan.interaction" has no attribute '
            '"TrustedDefaultResolutionCommand"; maybe '
            '"TrustedDefaultResolutionApplied", '
            '"TrustedDefaultResolutionResult", or '
            '"TrustedDefaultResolutionRequest"?  [attr-defined]'
        ),
        (
            "tests/interaction_type_contracts/"
            "rejected_store_trust_boundary_negative.py:32:1: error: "
            '"type[TaskInputClassification]" has no attribute '
            '"_from_classifier"  [attr-defined]'
        ),
        (
            "tests/interaction_type_contracts/"
            "rejected_store_trust_boundary_negative.py:37:1: error: "
            'Unexpected keyword argument "proposed_resolution" for '
            '"TrustedDefaultResolutionRequest"  [call-arg]'
        ),
        (
            "tests/interaction_type_contracts/"
            "rejected_store_trust_boundary_negative.py:46:16: error: "
            'Argument "provenance" to "CancelInteractionCommand" has '
            'incompatible type "Literal[AnswerProvenance.POLICY]"; expected '
            '"Literal[AnswerProvenance.HUMAN, '
            'AnswerProvenance.EXTERNAL_CONTROLLER]"  [arg-type]'
        ),
        (
            "tests/interaction_type_contracts/"
            "rejected_store_trust_boundary_negative.py:52:16: error: "
            'Argument "provenance" to "TerminalizeInteractionCommand" has '
            "incompatible type "
            '"Literal[AnswerProvenance.TRUSTED_DEFAULT]"; expected '
            '"Literal[AnswerProvenance.HUMAN, '
            'AnswerProvenance.EXTERNAL_CONTROLLER]"  [arg-type]'
        ),
        (
            "tests/interaction_type_contracts/"
            "rejected_store_trust_boundary_negative.py:57:16: error: "
            'Argument "provenance" to "TerminalizeInteractionScopeCommand" '
            'has incompatible type "Literal[AnswerProvenance.POLICY]"; '
            'expected "Literal[AnswerProvenance.HUMAN, '
            'AnswerProvenance.EXTERNAL_CONTROLLER]"  [arg-type]'
        ),
        (
            "tests/interaction_type_contracts/"
            "rejected_store_trust_boundary_negative.py:62:16: error: "
            'Argument "provenance" to "SupersedeInteractionScopeCommand" '
            "has incompatible type "
            '"Literal[AnswerProvenance.TRUSTED_DEFAULT]"; expected '
            '"Literal[AnswerProvenance.HUMAN, '
            'AnswerProvenance.EXTERNAL_CONTROLLER]"  [arg-type]'
        ),
    )
