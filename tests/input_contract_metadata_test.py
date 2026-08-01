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
    monkeypatch.setenv("HOME", "ambient-home")
    monkeypatch.setenv("OLDPWD", "ambient-oldpwd")
    monkeypatch.setenv("PWD", "ambient-pwd")
    monkeypatch.setenv("PYTHONPATH", "ambient-path")
    monkeypatch.setenv("PYTEST_ADDOPTS", "--maxfail=1")
    monkeypatch.setenv("COVERAGE_PROCESS_START", "ambient-coveragerc")
    assert _GATE.run_coverage_gate(repo_root=tmp_path) == 9
    assert len(calls) == 1
    command, environment = calls[0]
    assert command[1:3] == ("-m", "pytest")
    assert command[3:9] == (
        "-p",
        "no:cacheprovider",
        "-p",
        "pytest_cov",
        "-p",
        "anyio.pytest_plugin",
    )
    assert command[9:13] == ("--verbose", "-s", "-o", "addopts=")
    assert "--cov-config=/dev/null" in command
    assert environment["PYTEST_ADDOPTS"] == ""
    assert environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert environment["COVERAGE_RCFILE"] == "/dev/null"
    python_paths = environment["PYTHONPATH"].split(":")
    assert len(python_paths) == 1
    assert Path(python_paths[0]).name == "python-startup"
    assert (
        environment["AVALAN_CONTRACT_ALLOWED_PYTHONPATH"]
        == environment["PYTHONPATH"]
    )
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONSAFEPATH"] == "1"
    assert environment["PWD"] == str(tmp_path.resolve())
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    runtime_home = Path(environment["HOME"])
    assert runtime_home.name == "home"
    assert runtime_home.parent.name.startswith("avalan-contract-runtime-")
    assert not runtime_home.is_relative_to(tmp_path.resolve())
    assert "OLDPWD" not in environment
    assert "COVERAGE_PROCESS_START" not in environment
    assert not any(artifact.exists() for artifact in artifacts)
    makefile = (_ROOT / "Makefile").read_text(encoding="utf-8")
    assert (
        "test-coverage-exact:\nifeq ($(filter no-install,$(TEST_ARGS)),)"
        in makefile
    )
    assert (
        "poetry run env $(CONTRACT_PYTHON_ENV) python "
        "scripts/run_input_contract_gate.py --coverage-only"
        in makefile
    )
    assert "CONTRACT_PYTHON_ENV := PYTHONSAFEPATH=1" in makefile


def test_project_metadata_pins_complete_common_gate() -> None:
    """Require every script in lint, Make, and continuous integration."""
    makefile = (_ROOT / "Makefile").read_text(encoding="utf-8")
    workflow = (_ROOT / ".github" / "workflows" / "test.yml").read_text(
        encoding="utf-8"
    )
    coverage_workflow = (
        _ROOT / ".github" / "workflows" / "code-coverage.yml"
    ).read_text(encoding="utf-8")
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
    database_command = (
        "--docker --runner-script scripts/run_input_contract_gate.py -- "
        "--through-phase $(INPUT_PHASE)"
    )
    assert database_command in makefile
    assert (
        "poetry run -- env $(CONTRACT_PYTHON_ENV) python "
        f"scripts/task_pgsql_test_database.py {database_command}"
        in makefile
    )
    typecheck_command = (
        "scripts/verify_input_types.py --through-phase $(INPUT_PHASE)"
    )
    assert (
        f"poetry run env $(CONTRACT_PYTHON_ENV) python {typecheck_command}"
        in makefile
    )
    assert "make lint" in workflow
    assert "make typecheck-input-contract INPUT_PHASE=5" in workflow
    assert "sudo systemctl start postgresql.service" in workflow
    assert (
        "matrix.target.os == 'ubuntu-latest' && matrix.python == '3.11'"
        in workflow
    )
    admin_dsn = (
        "AVALAN_TASK_TEST_POSTGRESQL_ADMIN_DSN: "
        "postgresql://postgres:postgres@127.0.0.1:5432/postgres"
    )
    assert admin_dsn in workflow
    assert "run: make test no-install" in workflow
    assert "AVALAN_TASK_TEST_POSTGRESQL_DOCKER" not in workflow
    assert "make test-pgsql" not in workflow
    assert "run: make test coverage" in coverage_workflow
    assert "--through-phase 5" not in workflow
    assert "git diff --check" in workflow
