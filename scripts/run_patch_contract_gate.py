#!/usr/bin/env python
"""Run the sealed, current-phase patch contract gate."""

from argparse import ArgumentParser, Namespace
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from os import environ, pathsep
from pathlib import Path, PurePosixPath
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
from verify_patch_acceptance import load_phase0_contracts
from verify_patch_types import (
    discover_repository_python_paths,
    repository_python_ownership_environment,
)
from verify_patch_types import (
    load_manifest as load_patch_type_manifest,
)

_PATCH_CURRENT_PHASE = 3
_PATCH_DATABASE_PHASE = 8
_INPUT_MANIFEST = "tests/fixtures/input/acceptance_manifest.json"
_CONVERSATION_MANIFEST = (
    "tests/fixtures/conversation/acceptance_manifest.phase11.json"
)
_PATCH_MANIFEST = "tests/fixtures/patch/acceptance_manifest.json"
_COVERAGE_REPORTS = ("coverage.json", "coverage.xml")
_PYTEST_FACTS = ".patch-contract-pytest-facts.json"
_SEALED_ARTIFACTS = (*_COVERAGE_REPORTS, _PYTEST_FACTS)
_PYTEST_FACTS_ENV = "AVALAN_PATCH_CONTRACT_PYTEST_FACTS_PATH"
_PATCH_ARTIFACT_ROOT_ENV = "AVALAN_PATCH_CONTRACT_ARTIFACT_ROOT"
_PATCH_FORBIDDEN_ARTIFACTS_ENV = "AVALAN_PATCH_CONTRACT_FORBIDDEN_ARTIFACTS"
_PATCH_FORBIDDEN_ARTIFACTS_SHA256_ENV = (
    "AVALAN_PATCH_CONTRACT_FORBIDDEN_ARTIFACTS_SHA256"
)
_PATCH_FORBIDDEN_ARTIFACTS = ("specs/PATCH.md", "specs/PATCH-agenda.md")
_PHASE_EVIDENCE_ARTIFACT_ENVS = (
    "AVALAN_PATCH_PHASE_EVIDENCE_COVERAGE_JSON",
    "AVALAN_PATCH_PHASE_EVIDENCE_COVERAGE_XML",
    "AVALAN_PATCH_PHASE_EVIDENCE_PYTEST_FACTS",
)
_LEGACY_POSTGRESQL_LEASE_ENV = "AVALAN_TASK_TEST_POSTGRESQL_LEASE_SHA256"


class PatchContractGateError(RuntimeError):
    """Report a failed patch-specific exact gate condition."""


@dataclass(frozen=True, kw_only=True, slots=True)
class PytestFacts:
    """Store full-suite outcome facts emitted by the gate-owned plugin."""

    collected: int
    passed: int
    failed: int
    errors: int
    skipped: int
    collection_skipped: int
    xfailed: int
    xpassed: int
    deselected: int
    warnings: int
    leak_warnings: int
    exitstatus: int


def repository_root() -> Path:
    """Return the repository root containing this gate."""
    return Path(__file__).resolve().parents[1]


def preflight(through_phase: int, *, repo_root: Path | None = None) -> None:
    """Validate all gate phases and patch fixtures without invoking pytest."""
    root = (repo_root or repository_root()).resolve()
    _validate_patch_phase(root, through_phase)
    _current_phase(root, _INPUT_MANIFEST, "structured-input")
    _current_phase(root, _CONVERSATION_MANIFEST, "conversation")
    _load_patch_contracts(root)
    _load_patch_type_manifest(root)
    _reject_invalid_database_context(through_phase)


def run_gate(through_phase: int, *, repo_root: Path | None = None) -> int:
    """Run one fresh full suite and all current contract validators."""
    source_root = (repo_root or repository_root()).resolve()
    preflight(through_phase, repo_root=source_root)
    _remove_gate_artifacts(source_root, include_reports=True)
    external_database = _external_database(through_phase)
    try:
        if external_database is not None:
            return _run_gate_with_database(
                source_root,
                through_phase,
                external_database,
            )
        with _owned_postgresql_database() as database:
            return _run_gate_with_database(
                source_root,
                through_phase,
                database,
            )
    except (ContractGateError, OSError):
        _remove_gate_artifacts(source_root, include_reports=True)
        raise


def _run_gate_with_database(
    source_root: Path,
    through_phase: int,
    database: PostgreSQLTestDatabase,
) -> int:
    """Keep one internally owned database alive across every gate stage."""
    try:
        with nonignored_execution_mirror(source_root) as root:
            python_ownership = discover_repository_python_paths(source_root)
            return _run_mirrored_gate(
                source_root,
                root,
                through_phase,
                database,
                python_ownership,
            )
    except (ContractGateError, OSError):
        _remove_gate_artifacts(source_root, include_reports=True)
        raise


def _run_mirrored_gate(
    source_root: Path,
    root: Path,
    through_phase: int,
    external_database: PostgreSQLTestDatabase | None,
    python_ownership: tuple[PurePosixPath, ...],
) -> int:
    """Run and seal all verification inside one nonignored snapshot."""
    _validate_patch_phase(root, through_phase)
    verify_pytest_module_name_uniqueness(root)
    _remove_gate_artifacts(root, include_reports=True)
    before = capture_input_inventory(root)
    coverage_commands = exact_coverage_commands()
    facts_path = root / _PYTEST_FACTS
    coverage_command = _coverage_command(coverage_commands[0])
    coverage_result = _run_isolated_command(
        root,
        coverage_command,
        external_database,
        facts_path=facts_path,
    )
    if coverage_result != 0:
        _remove_gate_artifacts(root, include_reports=True)
        return coverage_result
    _verify_mirrored_state(root, before)
    facts = _read_pytest_facts(facts_path)
    _verify_pytest_facts(facts)

    for command in coverage_commands[1:]:
        result = _run_isolated_command(root, command, external_database)
        _verify_mirrored_state(root, before)
        if result != 0:
            _remove_gate_artifacts(root, include_reports=True)
            return result

    sealed = seal_artifacts(root, _SEALED_ARTIFACTS)
    _run_current_contract_verifiers(
        root,
        through_phase,
        external_database,
        sealed,
        before,
        python_ownership,
    )
    for report in _COVERAGE_REPORTS:
        verify_report_after_inventory(root / report, before)
    payloads = tuple(
        (name, (root / name).read_bytes()) for name in _SEALED_ARTIFACTS
    )
    print(
        "patch contract gate passed: "
        f"through_phase={through_phase} collected={facts.collected} "
        f"passed={facts.passed} skipped={facts.skipped} "
        f"warnings={facts.warnings} leak_warnings={facts.leak_warnings}"
    )
    _copy_artifacts_to_source(source_root, root, payloads)
    _remove_gate_artifacts(root, include_reports=False)
    return 0


def _run_current_contract_verifiers(
    root: Path,
    through_phase: int,
    external_database: PostgreSQLTestDatabase | None,
    sealed: tuple[SealedArtifact, ...],
    before: SealedInputInventory,
    python_ownership: tuple[PurePosixPath, ...],
) -> None:
    """Run every current contract family without another coverage run."""
    input_phase = _current_phase(root, _INPUT_MANIFEST, "structured-input")
    conversation_phase = _current_phase(
        root, _CONVERSATION_MANIFEST, "conversation"
    )
    commands = (
        (
            executable,
            "scripts/verify_input_types.py",
            "--through-phase",
            str(input_phase),
        ),
        (
            executable,
            "scripts/verify_conversation_types.py",
            "--through-phase",
            str(conversation_phase),
        ),
        (
            executable,
            "scripts/verify_patch_types.py",
            "--through-phase",
            str(through_phase),
        ),
    )
    for command in commands:
        _run_checked_command(
            root,
            command,
            external_database,
            sealed,
            before,
            python_ownership=python_ownership,
        )

    database_context = external_database or _owned_postgresql_database()
    if isinstance(database_context, PostgreSQLTestDatabase):
        _run_acceptance_verifiers(
            root,
            through_phase,
            database_context,
            sealed,
            before,
        )
        return
    with database_context as database:
        _run_acceptance_verifiers(
            root,
            through_phase,
            database,
            sealed,
            before,
        )


def _run_acceptance_verifiers(
    root: Path,
    through_phase: int,
    database: PostgreSQLTestDatabase,
    sealed: tuple[SealedArtifact, ...],
    before: SealedInputInventory,
) -> None:
    """Run current acceptance validators under one owned database lifetime."""
    input_phase = _current_phase(root, _INPUT_MANIFEST, "structured-input")
    conversation_phase = _current_phase(
        root, _CONVERSATION_MANIFEST, "conversation"
    )
    commands = (
        (
            executable,
            "scripts/verify_input_acceptance.py",
            "--through-phase",
            str(input_phase),
        ),
        (
            executable,
            "scripts/verify_conversation_acceptance.py",
            "--through-phase",
            str(conversation_phase),
        ),
        (
            executable,
            "scripts/verify_patch_acceptance.py",
            "--through-phase",
            str(through_phase),
        ),
    )
    for command in commands:
        _run_checked_command(root, command, database, sealed, before)


def _run_checked_command(
    root: Path,
    command: tuple[str, ...],
    database: PostgreSQLTestDatabase | None,
    sealed: tuple[SealedArtifact, ...],
    before: SealedInputInventory,
    *,
    python_ownership: tuple[PurePosixPath, ...] | None = None,
) -> None:
    """Run one verifier and reject any input or report replacement."""
    result = _run_isolated_command(
        root,
        command,
        database,
        evidence_artifacts=(
            root / "coverage.json",
            root / "coverage.xml",
            root / _PYTEST_FACTS,
        ),
        python_ownership=python_ownership,
    )
    _verify_mirrored_state(root, before, sealed)
    if result != 0:
        raise PatchContractGateError(
            "current contract verifier failed: " + " ".join(command)
        )


def _coverage_command(command: tuple[str, ...]) -> tuple[str, ...]:
    """Add the gate-owned outcome plugin to the one full pytest command."""
    assert command[-1] == "."
    return (*command[:-1], "-p", "patch_contract_gate_plugin", ".")


def _run_isolated_command(
    root: Path,
    command: tuple[str, ...],
    database: PostgreSQLTestDatabase | None,
    *,
    facts_path: Path | None = None,
    evidence_artifacts: tuple[Path, Path, Path] | None = None,
    python_ownership: tuple[PurePosixPath, ...] | None = None,
) -> int:
    """Run one verifier in a sealed environment with optional owned state."""
    with isolated_subprocess_environment(
        root,
        trusted_python_root=repository_root(),
    ) as base_environment:
        environment = dict(base_environment)
        environment.update(_patch_artifact_guard_environment(root))
        if python_ownership is not None:
            environment.update(
                repository_python_ownership_environment(python_ownership)
            )
        if database is not None:
            environment[POSTGRESQL_TEST_DSN_ENV] = database.dsn
        if facts_path is not None:
            environment[_PYTEST_FACTS_ENV] = str(facts_path)
        if evidence_artifacts is not None:
            for name, path in zip(
                _PHASE_EVIDENCE_ARTIFACT_ENVS,
                evidence_artifacts,
                strict=True,
            ):
                environment[name] = str(path)
        return run(command, cwd=root, check=False, env=environment).returncode


def _patch_artifact_guard_environment(root: Path) -> dict[str, str]:
    """Return exact absolute ignored-artifact paths owned by this gate."""
    resolved_root = root.resolve()
    forbidden = tuple(
        str((resolved_root / relative).resolve())
        for relative in _PATCH_FORBIDDEN_ARTIFACTS
    )
    forbidden_value = pathsep.join(forbidden)
    return {
        _PATCH_ARTIFACT_ROOT_ENV: str(resolved_root),
        _PATCH_FORBIDDEN_ARTIFACTS_ENV: forbidden_value,
        _PATCH_FORBIDDEN_ARTIFACTS_SHA256_ENV: (
            sha256(forbidden_value.encode("utf-8")).hexdigest()
        ),
    }


def _read_pytest_facts(path: Path) -> PytestFacts:
    """Load and validate complete facts from the sole covered test process."""
    if path.is_symlink() or not path.is_file():
        raise PatchContractGateError(
            "full suite did not emit pytest outcome facts"
        )
    try:
        raw = strict_json_path(path)
    except StrictJsonError as exc:
        raise PatchContractGateError(str(exc)) from exc
    if not isinstance(raw, dict):
        raise PatchContractGateError("pytest outcome facts must be an object")
    expected = {
        "schema_version",
        "collected",
        "passed",
        "failed",
        "errors",
        "skipped",
        "collection_skipped",
        "xfailed",
        "xpassed",
        "deselected",
        "warnings",
        "leak_warnings",
        "exitstatus",
    }
    if set(raw) != expected or raw.get("schema_version") != 1:
        raise PatchContractGateError("pytest outcome facts have invalid keys")
    values = {
        name: _nonnegative_fact(raw.get(name), name)
        for name in expected - {"schema_version"}
    }
    return PytestFacts(
        collected=values["collected"],
        passed=values["passed"],
        failed=values["failed"],
        errors=values["errors"],
        skipped=values["skipped"],
        collection_skipped=values["collection_skipped"],
        xfailed=values["xfailed"],
        xpassed=values["xpassed"],
        deselected=values["deselected"],
        warnings=values["warnings"],
        leak_warnings=values["leak_warnings"],
        exitstatus=values["exitstatus"],
    )


def _nonnegative_fact(value: object, name: str) -> int:
    """Return one non-negative integer from sealed pytest facts."""
    if type(value) is not int or value < 0:
        raise PatchContractGateError(
            f"pytest outcome fact must be a non-negative integer: {name}"
        )
    return value


def _verify_pytest_facts(facts: PytestFacts) -> None:
    """Reject non-passing, uncollected, or leak-bearing full-suite evidence."""
    if not facts.collected or facts.exitstatus != 0:
        raise PatchContractGateError(
            "full suite did not complete successfully"
        )
    if facts.failed or facts.errors:
        raise PatchContractGateError(
            "full suite recorded failed or errored tests"
        )
    if facts.xfailed or facts.xpassed or facts.deselected:
        raise PatchContractGateError(
            "full suite recorded xfailed, xpassed, or deselected tests"
        )
    if facts.leak_warnings:
        raise PatchContractGateError(
            "full suite recorded resource leak warnings"
        )
    if facts.collection_skipped > facts.skipped:
        raise PatchContractGateError(
            "collection skip facts exceed total skipped tests"
        )
    accounted = facts.passed + facts.skipped + facts.xfailed + facts.xpassed
    expected = facts.collected + facts.collection_skipped
    if accounted != expected:
        raise PatchContractGateError("pytest outcome facts do not balance")


def _verify_mirrored_state(
    root: Path,
    expected: SealedInputInventory,
    sealed: tuple[SealedArtifact, ...] = (),
) -> None:
    """Reject mutable inputs or generated evidence between validators."""
    verify_input_inventory(expected, capture_input_inventory(root))
    if sealed:
        verify_artifacts(root, sealed)


def _validate_patch_phase(root: Path, through_phase: int) -> None:
    """Require exactly the frozen patch phase supplied by the public target."""
    if type(through_phase) is not int or through_phase < 0:
        raise PatchContractGateError(
            "through-phase must be a non-negative integer"
        )
    current = _current_phase(root, _PATCH_MANIFEST, "patch")
    if current != _PATCH_CURRENT_PHASE or through_phase != current:
        raise PatchContractGateError(
            "patch acceptance phase is not implemented"
        )


def _current_phase(root: Path, relative: str, label: str) -> int:
    """Return one validated current phase from its tracked manifest."""
    try:
        payload = strict_json_path(root / relative)
    except StrictJsonError as exc:
        raise PatchContractGateError(
            f"cannot read {label} acceptance manifest"
        ) from exc
    if not isinstance(payload, dict):
        raise PatchContractGateError(
            f"{label} acceptance manifest phase is invalid"
        )
    current_phase = payload.get("current_phase")
    if type(current_phase) is not int:
        raise PatchContractGateError(
            f"{label} acceptance manifest phase is invalid"
        )
    return current_phase


def _load_patch_contracts(root: Path) -> None:
    """Validate the complete patch fixture graph without test execution."""
    load_phase0_contracts(root / "tests/fixtures/patch", repo_root=root)


def _load_patch_type_manifest(root: Path) -> None:
    """Validate the patch type fixture graph without invoking mypy."""
    load_patch_type_manifest(
        root / "tests/fixtures/patch/type_contract_manifest.json"
    )


def _reject_invalid_database_context(through_phase: int) -> None:
    """Reject ambient database state before durable patch phases exist."""
    supplied = any(
        name in environ
        for name in (POSTGRESQL_TEST_DSN_ENV, _LEGACY_POSTGRESQL_LEASE_ENV)
    )
    if supplied and through_phase < _PATCH_DATABASE_PHASE:
        raise PatchContractGateError(
            "patch phases 0 through 7 reject caller-supplied PostgreSQL state"
        )


def _external_database(through_phase: int) -> PostgreSQLTestDatabase | None:
    """Return the isolated harness database supplied only at durable phases."""
    dsn = environ.get(POSTGRESQL_TEST_DSN_ENV)
    if dsn is None:
        return None
    if through_phase < _PATCH_DATABASE_PHASE:
        raise PatchContractGateError("patch database phase is not active")
    return PostgreSQLTestDatabase(dsn=dsn, name="external-patch-gate")


@contextmanager
def _owned_postgresql_database() -> Iterator[PostgreSQLTestDatabase]:
    """Yield an isolated database for current contract acceptance only."""
    admin_dsn = environ.get("AVALAN_TASK_TEST_POSTGRESQL_ADMIN_DSN")
    prefix = environ.get(
        "AVALAN_TASK_TEST_POSTGRESQL_DATABASE_PREFIX",
        "avalan_patch_contract",
    )
    image = environ.get(
        "AVALAN_TASK_TEST_POSTGRESQL_DOCKER_IMAGE",
        "postgres:16-alpine",
    )
    try:
        timeout_seconds = float(
            environ.get(
                "AVALAN_TASK_TEST_POSTGRESQL_DOCKER_TIMEOUT_SECONDS", "60"
            )
        )
    except ValueError as exc:
        raise PatchContractGateError(
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


def _copy_artifacts_to_source(
    source_root: Path,
    root: Path,
    payloads: tuple[tuple[str, bytes], ...],
) -> None:
    """Copy only sealed generated evidence back to the caller's checkout."""
    if root == source_root:
        raise PatchContractGateError(
            "patch gate source and mirror are identical"
        )
    for name, payload in payloads:
        (source_root / name).write_bytes(payload)


def _remove_gate_artifacts(root: Path, *, include_reports: bool) -> None:
    """Delete stale coverage and outcome evidence before or after failure."""
    remove_coverage_artifacts(root, include_reports=include_reports)
    facts = root / _PYTEST_FACTS
    if facts.is_file() or facts.is_symlink():
        facts.unlink()


def _parse_args() -> Namespace:
    parser = ArgumentParser(
        description=(
            "Run one fresh full-suite patch contract gate and current "
            "contract validators."
        )
    )
    parser.add_argument("--through-phase", type=int, required=True)
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run the requested patch gate mode from the command line."""
    args = _parse_args()
    try:
        if args.preflight:
            preflight(args.through_phase)
            print(
                "patch contract preflight passed: "
                f"through_phase={args.through_phase}"
            )
            return 0
        return run_gate(args.through_phase)
    except (ContractGateError, PatchContractGateError) as exc:
        print(f"patch contract gate failed: {exc}", file=stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
