#!/usr/bin/env python
"""Run sealed exact-coverage and conversation-continuity gates."""

from argparse import ArgumentParser, Namespace
from collections.abc import Iterator
from contextlib import contextmanager
from os import environ
from pathlib import Path
from subprocess import run
from sys import executable, stderr

from contract_gate import (
    POSTGRESQL_TEST_DSN_ENV,
    ContractGateError,
    SealedArtifact,
    SealedInputInventory,
    StrictJsonError,
    capture_input_inventory,
    exact_coverage_commands,
    isolated_subprocess_environment,
    nonignored_execution_mirror,
    remove_coverage_artifacts,
    seal_artifacts,
    strict_json_path,
    verify_artifacts,
    verify_input_inventory,
    verify_pytest_module_name_uniqueness,
    verify_report_after_inventory,
)
from task_pgsql_test_database import (
    PostgreSQLTestDatabase,
    postgresql_test_database,
)

_INPUT_CURRENT_PHASE = 12
_CONVERSATION_CURRENT_PHASE = 2
_CONVERSATION_DATABASE_PHASE = 3
_COVERAGE_REPORTS = ("coverage.json", "coverage.xml")
_LEGACY_POSTGRESQL_LEASE_ENV = "AVALAN_TASK_TEST_POSTGRESQL_LEASE_SHA256"


def repository_root() -> Path:
    """Return the repository root containing this script."""
    return Path(__file__).resolve().parents[1]


def run_gate(
    through_phase: int,
    *,
    repo_root: Path | None = None,
) -> int:
    """Run one sealed full-suite and acceptance gate."""
    source_root = (repo_root or repository_root()).resolve()
    _validate_through_phase(source_root, through_phase)
    _reject_ambient_test_database()
    _remove_coverage_artifacts(source_root, include_reports=True)
    if through_phase >= _CONVERSATION_DATABASE_PHASE:
        with _owned_postgresql_database() as database:
            return _run_gate_with_database(
                source_root,
                through_phase,
                database,
            )
    return _run_gate_with_database(source_root, through_phase, None)


def _run_gate_with_database(
    source_root: Path,
    through_phase: int,
    database: PostgreSQLTestDatabase | None,
) -> int:
    """Run the gate while one optional outer database remains alive."""
    report_payloads: tuple[tuple[str, bytes], ...] = ()
    try:
        with nonignored_execution_mirror(source_root) as root:
            exit_code, inventory_sha256 = _run_mirrored_gate(
                root,
                through_phase,
                database,
            )
            if exit_code != 0:
                _remove_coverage_artifacts(
                    source_root,
                    include_reports=True,
                )
                return exit_code
            report_payloads = tuple(
                (report, (root / report).read_bytes())
                for report in _COVERAGE_REPORTS
            )
    except OSError:
        _remove_coverage_artifacts(source_root, include_reports=True)
        raise
    except ContractGateError:
        _remove_coverage_artifacts(source_root, include_reports=True)
        raise
    if len(report_payloads) != len(_COVERAGE_REPORTS):
        raise ContractGateError(
            "verified coverage report payloads are missing"
        )
    for report, payload in report_payloads:
        (source_root / report).write_bytes(payload)
    _remove_coverage_artifacts(source_root, include_reports=False)
    print(
        "conversation contract gate passed: "
        f"through_phase={through_phase} "
        f"input_inventory_sha256={inventory_sha256}"
    )
    return 0


def _run_mirrored_gate(
    root: Path,
    through_phase: int,
    database: PostgreSQLTestDatabase | None,
) -> tuple[int, str]:
    """Run the complete gate in one isolated nonignored snapshot."""
    _validate_through_phase(root, through_phase)
    verify_pytest_module_name_uniqueness(root)
    _remove_coverage_artifacts(root, include_reports=True)
    before = capture_input_inventory(root)
    _require_database_context(through_phase, database)
    coverage_commands = exact_coverage_commands()
    for command in coverage_commands:
        returncode = _run_isolated_command(root, command, database)
        _verify_mirrored_state(root, before)
        if returncode != 0:
            return returncode, before.sha256

    sealed_reports = seal_artifacts(root, _COVERAGE_REPORTS)
    conversation_command = (
        executable,
        "scripts/verify_conversation_acceptance.py",
        "--through-phase",
        str(through_phase),
    )
    conversation_result = _run_isolated_command(
        root,
        conversation_command,
        database,
    )
    _verify_mirrored_state(root, before, sealed_reports)
    if conversation_result != 0:
        return conversation_result, before.sha256

    if database is None:
        with _owned_postgresql_database() as input_database:
            input_result = _run_input_acceptance(root, input_database)
            _verify_mirrored_state(root, before, sealed_reports)
    else:
        input_result = _run_input_acceptance(root, database)
        _verify_mirrored_state(root, before, sealed_reports)
    if input_result != 0:
        return input_result, before.sha256

    verification_result = _run_isolated_command(
        root,
        coverage_commands[1],
        database,
    )
    _verify_mirrored_state(root, before, sealed_reports)
    if verification_result != 0:
        return verification_result, before.sha256
    for report in _COVERAGE_REPORTS:
        verify_report_after_inventory(root / report, before)
    _remove_coverage_artifacts(root, include_reports=False)
    return 0, before.sha256


def _run_input_acceptance(
    root: Path,
    database: PostgreSQLTestDatabase,
) -> int:
    """Run every current structured-input node under an owned database."""
    command = (
        executable,
        "scripts/verify_input_acceptance.py",
        "--through-phase",
        str(_INPUT_CURRENT_PHASE),
    )
    return _run_isolated_command(root, command, database)


def _run_isolated_command(
    root: Path,
    command: tuple[str, ...],
    database: PostgreSQLTestDatabase | None,
) -> int:
    """Run one command with a fresh verified startup environment."""
    with isolated_subprocess_environment(
        root,
        trusted_python_root=repository_root(),
    ) as base_environment:
        environment = _database_environment(base_environment, database)
        return run(
            command,
            cwd=root,
            check=False,
            env=environment,
        ).returncode


def _verify_mirrored_state(
    root: Path,
    expected: SealedInputInventory,
    sealed_reports: tuple[SealedArtifact, ...] = (),
) -> None:
    """Verify sealed inputs and optional reports before another command."""
    observed = capture_input_inventory(root)
    verify_input_inventory(expected, observed)
    if sealed_reports:
        verify_artifacts(root, sealed_reports)


def _database_environment(
    environment: dict[str, str],
    database: PostgreSQLTestDatabase | None,
) -> dict[str, str]:
    """Return child state containing only an internally owned database DSN."""
    child = dict(environment)
    if database is not None:
        child[POSTGRESQL_TEST_DSN_ENV] = database.dsn
    return child


def _require_database_context(
    through_phase: int,
    database: PostgreSQLTestDatabase | None,
) -> None:
    """Enforce phase-correct use of a typed internally owned database."""
    required = through_phase >= _CONVERSATION_DATABASE_PHASE
    supplied = isinstance(database, PostgreSQLTestDatabase)
    if required != supplied or (database is not None and not supplied):
        raise ContractGateError(
            "conversation database ownership does not match the selected phase"
        )


def _reject_ambient_test_database() -> None:
    """Reject caller-supplied task database state at the public runner."""
    if any(
        name in environ
        for name in (POSTGRESQL_TEST_DSN_ENV, _LEGACY_POSTGRESQL_LEASE_ENV)
    ):
        raise ContractGateError(
            "conversation runner rejects caller-supplied task PostgreSQL state"
        )


@contextmanager
def _owned_postgresql_database() -> Iterator[PostgreSQLTestDatabase]:
    """Yield one database whose full lifetime belongs to this runner."""
    admin_dsn = environ.get("AVALAN_TASK_TEST_POSTGRESQL_ADMIN_DSN")
    prefix = environ.get(
        "AVALAN_TASK_TEST_POSTGRESQL_DATABASE_PREFIX",
        "avalan_conversation_test",
    )
    image = environ.get(
        "AVALAN_TASK_TEST_POSTGRESQL_DOCKER_IMAGE",
        "postgres:16-alpine",
    )
    try:
        timeout_seconds = float(
            environ.get(
                "AVALAN_TASK_TEST_POSTGRESQL_DOCKER_TIMEOUT_SECONDS",
                "60",
            )
        )
    except ValueError as exc:
        raise ContractGateError(
            "PostgreSQL Docker timeout must be numeric"
        ) from exc
    with postgresql_test_database(
        admin_dsn=admin_dsn,
        database_prefix=prefix,
        docker=admin_dsn is None,
        image=image,
        timeout_seconds=timeout_seconds,
    ) as database:
        yield database


def _validate_through_phase(root: Path, through_phase: int) -> None:
    """Reject a phase outside the closed implemented manifest range."""
    path = root / "tests/fixtures/conversation/acceptance_manifest.phase2.json"
    try:
        payload = strict_json_path(path)
    except StrictJsonError as exc:
        raise ContractGateError(
            "cannot validate the conversation acceptance phase"
        ) from exc
    if not isinstance(payload, dict):
        raise ContractGateError("conversation acceptance manifest is invalid")
    current_phase = payload.get("current_phase")
    if (
        type(current_phase) is not int
        or current_phase != _CONVERSATION_CURRENT_PHASE
    ):
        raise ContractGateError(
            "conversation acceptance manifest current phase differs from "
            "the gate anchor"
        )
    if (
        type(through_phase) is not int
        or through_phase < 0
        or through_phase > current_phase
    ):
        raise ContractGateError(
            "through-phase must be within the implemented conversation "
            f"range 0..{current_phase}"
        )


def _remove_coverage_artifacts(
    root: Path,
    *,
    include_reports: bool,
) -> None:
    remove_coverage_artifacts(root, include_reports=include_reports)


def _parse_args() -> Namespace:
    parser = ArgumentParser(
        description=(
            "Run full exact coverage and lifecycle-aware conversation "
            "acceptance against one sealed input inventory."
        )
    )
    parser.add_argument("--through-phase", required=True, type=int)
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run the sealed conversation gate from the command line."""
    args = _parse_args()
    try:
        if args.preflight:
            _validate_through_phase(repository_root(), args.through_phase)
            print(
                "conversation contract preflight passed: "
                f"through_phase={args.through_phase}"
            )
            return 0
        return run_gate(args.through_phase)
    except ContractGateError as exc:
        print(f"conversation contract gate failed: {exc}", file=stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
