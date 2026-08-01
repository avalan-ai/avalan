#!/usr/bin/env python
"""Run sanitized exact-coverage and structured-input quality gates."""

from argparse import ArgumentParser, Namespace
from pathlib import Path
from subprocess import run
from sys import executable

from contract_gate import (
    POSTGRESQL_TEST_DSN_ENV,
    ContractGateError,
    exact_coverage_commands,
    isolated_subprocess_environment,
    remove_coverage_artifacts,
)


def repository_root() -> Path:
    """Return the repository root containing this script."""
    return Path(__file__).resolve().parents[1]


def run_coverage_gate(*, repo_root: Path | None = None) -> int:
    """Run exact source coverage in a sanitized subprocess environment."""
    root = (repo_root or repository_root()).resolve()
    _remove_coverage_artifacts(root, include_reports=True)
    commands = exact_coverage_commands()
    for command in commands:
        try:
            returncode = _run_isolated_command(root, command)
        except (ContractGateError, OSError):
            _remove_coverage_artifacts(root, include_reports=True)
            raise
        if returncode != 0:
            _remove_coverage_artifacts(root, include_reports=True)
            return returncode
    _remove_coverage_artifacts(root, include_reports=False)
    return 0


def run_gate(through_phase: int, *, repo_root: Path | None = None) -> int:
    """Run coverage and acceptance before the database harness exits."""
    root = (repo_root or repository_root()).resolve()
    coverage_exit = run_coverage_gate(repo_root=root)
    if coverage_exit != 0:
        return coverage_exit
    command = (
        executable,
        "scripts/verify_input_acceptance.py",
        "--through-phase",
        str(through_phase),
    )
    try:
        returncode = _run_isolated_command(root, command)
    except (ContractGateError, OSError):
        _remove_coverage_artifacts(root, include_reports=True)
        raise
    if returncode != 0:
        _remove_coverage_artifacts(root, include_reports=True)
    return returncode


def _run_isolated_command(root: Path, command: tuple[str, ...]) -> int:
    """Run one command in a fresh verified subprocess environment."""
    with isolated_subprocess_environment(
        root,
        inherited_names=(POSTGRESQL_TEST_DSN_ENV,),
        trusted_python_root=repository_root(),
    ) as environment:
        return run(
            command,
            cwd=root,
            check=False,
            env=environment,
        ).returncode


def _remove_coverage_artifacts(root: Path, *, include_reports: bool) -> None:
    remove_coverage_artifacts(root, include_reports=include_reports)


def _parse_args() -> Namespace:
    parser = ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--through-phase", type=int)
    mode.add_argument("--coverage-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run the selected exact gate from the command line."""
    args = _parse_args()
    if args.coverage_only:
        return run_coverage_gate()
    assert isinstance(args.through_phase, int)
    return run_gate(args.through_phase)


if __name__ == "__main__":
    raise SystemExit(main())
