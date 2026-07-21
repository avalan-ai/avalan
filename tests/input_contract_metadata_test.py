"""Freeze project metadata for the structured-input quality gate."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from subprocess import CompletedProcess
from sys import modules
from types import ModuleType

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load_gate() -> ModuleType:
    """Return the common input-contract gate module."""
    name = "_input_contract_metadata_gate"
    spec = spec_from_file_location(
        name, _ROOT / "scripts" / "run_input_contract_gate.py"
    )
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    modules[name] = module
    spec.loader.exec_module(module)
    return module


_GATE = _load_gate()


def test_exact_make_target_fails_closed_on_pytest_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stop at pytest failure and remove stale or partial artifacts."""
    artifacts = (
        tmp_path / ".coverage",
        tmp_path / ".coverage.worker",
        tmp_path / "coverage.json",
        tmp_path / "coverage.xml",
    )
    for artifact in artifacts:
        artifact.write_text("stale", encoding="utf-8")
    calls: list[tuple[tuple[str, ...], dict[str, str]]] = []

    def fail_pytest(
        command: tuple[str, ...],
        *,
        cwd: Path,
        check: bool,
        env: dict[str, str],
    ) -> CompletedProcess[str]:
        assert cwd == tmp_path
        assert check is False
        assert not any(artifact.exists() for artifact in artifacts)
        calls.append((command, env))
        for artifact in artifacts:
            artifact.write_text("partial", encoding="utf-8")
        return CompletedProcess(command, 9)

    monkeypatch.setattr(_GATE, "run", fail_pytest)
    monkeypatch.setenv("PYTHONPATH", "ambient-path")
    monkeypatch.setenv("PYTEST_ADDOPTS", "--maxfail=1")
    monkeypatch.setenv("COVERAGE_PROCESS_START", "ambient-coveragerc")
    assert _GATE.run_coverage_gate(repo_root=tmp_path) == 9
    assert len(calls) == 1
    command, environment = calls[0]
    assert command[1:5] == ("-m", "pytest", "--verbose", "-s")
    assert ("-o", "addopts=") == command[5:7]
    assert "--cov-config=/dev/null" in command
    assert environment["PYTEST_ADDOPTS"] == ""
    assert environment["COVERAGE_RCFILE"] == "/dev/null"
    assert "PYTHONPATH" not in environment
    assert "COVERAGE_PROCESS_START" not in environment
    assert not any(artifact.exists() for artifact in artifacts)
    makefile = (_ROOT / "Makefile").read_text(encoding="utf-8")
    assert (
        "test-coverage-exact:\nifeq ($(filter no-install,$(TEST_ARGS)),)"
        in makefile
    )
    assert (
        "poetry run python scripts/run_input_contract_gate.py --coverage-only"
        in makefile
    )


def test_project_metadata_pins_complete_common_gate() -> None:
    """Require every script in lint, Make, and continuous integration."""
    makefile = (_ROOT / "Makefile").read_text(encoding="utf-8")
    workflow = (_ROOT / ".github" / "workflows" / "test.yml").read_text(
        encoding="utf-8"
    )
    expected_scripts = (
        "scripts/input_contract_json.py",
        "scripts/run_input_contract_gate.py",
        "scripts/task_pgsql_test_database.py",
        "scripts/verify_input_acceptance.py",
        "scripts/verify_input_types.py",
        "scripts/verify_src_coverage.py",
    )
    assignment = next(
        line
        for line in makefile.splitlines()
        if line.startswith("INPUT_CONTRACT_SCRIPTS")
    )
    assert assignment == "INPUT_CONTRACT_SCRIPTS := " + " ".join(
        expected_scripts
    )
    assert "LINT_PATHS := src/ tests/ $(INPUT_CONTRACT_SCRIPTS)" in makefile
    assert "poetry run ruff format --preview $(LINT_PATHS)" in makefile
    assert "poetry run black --preview" in makefile
    assert "poetry run ruff check --fix $(LINT_PATHS)" in makefile
    assert "poetry run mypy $(INPUT_CONTRACT_SCRIPTS)" in makefile
    assert (
        "--docker --runner-script scripts/run_input_contract_gate.py -- "
        "--through-phase $(INPUT_PHASE)"
        in makefile
    )
    assert (
        "poetry run python scripts/verify_input_types.py --through-phase "
        "$(INPUT_PHASE)"
        in makefile
    )
    assert "make lint" in workflow
    assert "make typecheck-input-contract INPUT_PHASE=2" in workflow
    assert "make test-pgsql-exact no-install INPUT_PHASE=2" in workflow
    assert "make test-coverage-exact no-install" in workflow
    assert (
        "poetry run python scripts/verify_input_acceptance.py"
        " --through-phase 2"
        in workflow
    )
    assert "git diff --check" in workflow
