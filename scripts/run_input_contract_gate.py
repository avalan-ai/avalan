#!/usr/bin/env python
"""Run sanitized exact-coverage and structured-input quality gates."""

from argparse import ArgumentParser, Namespace
from os import environ
from pathlib import Path
from subprocess import run
from sys import executable


def repository_root() -> Path:
    """Return the repository root containing this script."""
    return Path(__file__).resolve().parents[1]


def run_coverage_gate(*, repo_root: Path | None = None) -> int:
    """Run exact source coverage in a sanitized subprocess environment."""
    root = (repo_root or repository_root()).resolve()
    _remove_coverage_artifacts(root, include_reports=True)
    commands = (
        (
            executable,
            "-m",
            "pytest",
            "--verbose",
            "-s",
            "-o",
            "addopts=",
            "--cov=src/",
            "--cov-config=/dev/null",
            "--cov-report=xml",
            "--cov-report=json:coverage.json",
        ),
        (executable, "scripts/verify_src_coverage.py"),
        (
            "jq",
            "-r",
            (
                ".files | to_entries[] | select("
                ".value.summary.missing_lines != 0 or "
                ".value.summary.covered_lines != "
                ".value.summary.num_statements) | "
                '"\\(.key): " + '
                '"\\(.value.summary.percent_covered_display)%"'
            ),
            "coverage.json",
        ),
    )
    environment = _sanitized_environment()
    for command in commands:
        try:
            completed = run(command, cwd=root, check=False, env=environment)
        except OSError:
            _remove_coverage_artifacts(root, include_reports=True)
            raise
        if completed.returncode != 0:
            _remove_coverage_artifacts(root, include_reports=True)
            return completed.returncode
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
        completed = run(
            command,
            cwd=root,
            check=False,
            env=_sanitized_environment(),
        )
    except OSError:
        _remove_coverage_artifacts(root, include_reports=True)
        raise
    if completed.returncode != 0:
        _remove_coverage_artifacts(root, include_reports=True)
    return completed.returncode


def _sanitized_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in environ.items()
        if key.upper()
        not in {"PYTHONPATH", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"}
        and not key.upper().startswith(("COVERAGE_", "COV_CORE_"))
    }
    environment["PYTEST_ADDOPTS"] = ""
    environment["COVERAGE_RCFILE"] = "/dev/null"
    return environment


def _remove_coverage_artifacts(root: Path, *, include_reports: bool) -> None:
    artifacts = list(root.glob(".coverage.*"))
    artifacts.append(root / ".coverage")
    if include_reports:
        artifacts.extend((root / "coverage.json", root / "coverage.xml"))
    for artifact in artifacts:
        if artifact.is_file() or artifact.is_symlink():
            artifact.unlink()


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
