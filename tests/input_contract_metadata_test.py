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


def _lint_contract_uses_black_before_ruff(makefile: str) -> bool:
    """Return whether both lint targets use the canonical formatter policy."""
    lint = makefile.split("lint:\n", maxsplit=1)[1].split(
        "\n\nlint-check:\n",
        maxsplit=1,
    )[0]
    lint_check = makefile.split("lint-check:\n", maxsplit=1)[1].split(
        "\n\ntest:\n",
        maxsplit=1,
    )[0]
    black = (
        "poetry run black --preview "
        "--enable-unstable-feature=string_processing $(LINT_PATHS)"
    )
    black_check = (
        "poetry run black --check --preview "
        "--enable-unstable-feature=string_processing $(LINT_PATHS)"
    )
    ruff_fix = "poetry run ruff check --fix $(LINT_PATHS)"
    ruff_check = "poetry run ruff check --no-fix $(LINT_PATHS)"
    return (
        "ruff format" not in lint
        and "ruff format" not in lint_check
        and black in lint
        and ruff_fix in lint
        and lint.index(black) < lint.index(ruff_fix)
        and black_check in lint_check
        and ruff_check in lint_check
        and "--fix" not in lint_check
        and lint_check.index(black_check) < lint_check.index(ruff_check)
    )


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


def test_project_metadata_pins_input_gate_and_standard_coverage() -> None:
    """Require the input gate and standard coverage command in CI."""
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
    assert _lint_contract_uses_black_before_ruff(makefile)
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
    install = (
        "      - name: Install project dependencies\n"
        "        run: poetry sync --all-extras --with test\n"
    )
    assert install in coverage_workflow
    assert coverage_workflow.count("poetry sync --all-extras --with test") == 1
    assert "run: make test no-install coverage" in coverage_workflow
    assert "run: make test coverage" not in coverage_workflow
    assert "sudo apt-get install --yes bubblewrap libacl1" in coverage_workflow
    assert (
        "kernel.apparmor_restrict_unprivileged_userns=0" in coverage_workflow
    )
    smoke = (
        "      - name: Verify native sandbox and type-contract smoke\n"
        "        run: |\n"
        "          poetry run pytest -q "
        "tests/patch/phase_10_contract_test.py \\\n"
        "            tests/project_metadata_test.py::"
        "test_test_workflow_covers_supported_matrix_and_build_gates\n"
        "          make typecheck-input-contract INPUT_PHASE=12\n"
        "          make typecheck-conversation-contract "
        "CONVERSATION_PHASE=11\n"
    )
    assert smoke in coverage_workflow
    assert (
        "tests/interaction/rejected_result_type_contract_test.py"
        in coverage_workflow
    )
    assert (
        "tests/interaction/rejected_store_trust_boundary_type_contract_test.py"
        in coverage_workflow
    )
    assert coverage_workflow.index(smoke) < coverage_workflow.index(
        "      - name: Run coverage\n"
    )
    assert coverage_workflow.index(install) < coverage_workflow.index(smoke)
    assert "kernel.unprivileged_userns_clone=1" in coverage_workflow
    assert "bwrap --die-with-parent --unshare-user --uid 0 --gid 0" in (
        coverage_workflow
    )
    assert "run: make test-conversation-current-exact" not in coverage_workflow
    assert "make test-conversation-exact" not in coverage_workflow
    assert "make test-conversation-pgsql-exact" not in coverage_workflow
    pyproject = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    lockfile = (_ROOT / "poetry.lock").read_text(encoding="utf-8")
    assert '"opencv-python-headless==4.12.0.88",' in pyproject
    assert '"opencv-python>=4.12.0.88,<5.0.0",' not in pyproject
    assert '\nname = "opencv-python"\n' not in lockfile
    assert 'name = "opencv-python-headless"\nversion = "4.12.0.88"' in lockfile
    assert "sudo systemctl start postgresql.service" in coverage_workflow
    assert (
        "AVALAN_TASK_TEST_POSTGRESQL_ADMIN_DSN: "
        "postgresql://postgres:postgres@127.0.0.1:5432/postgres"
        in coverage_workflow
    )
    assert "--through-phase 5" not in workflow
    assert "git diff --check" in workflow


def test_project_metadata_rejects_ruff_formatter_in_lint() -> None:
    """Reject a second formatter in the mutating lint target."""
    makefile = (
        (_ROOT / "Makefile")
        .read_text(encoding="utf-8")
        .replace(
            "lint:\n",
            "lint:\n\tpoetry run ruff format --preview $(LINT_PATHS)\n",
            1,
        )
    )

    assert not _lint_contract_uses_black_before_ruff(makefile)


def test_project_metadata_rejects_ruff_before_black_in_lint() -> None:
    """Keep Black before the potentially mutating Ruff check."""
    makefile = (
        (_ROOT / "Makefile")
        .read_text(encoding="utf-8")
        .replace(
            "poetry run black --preview "
            "--enable-unstable-feature=string_processing $(LINT_PATHS)\n\t"
            "poetry run ruff check --fix $(LINT_PATHS)",
            "poetry run ruff check --fix $(LINT_PATHS)\n\tpoetry run black"
            " --preview --enable-unstable-feature=string_processing"
            " $(LINT_PATHS)",
            1,
        )
    )

    assert not _lint_contract_uses_black_before_ruff(makefile)


def test_project_metadata_rejects_mutating_or_reordered_lint_check() -> None:
    """Keep CI lint checks non-mutating and Black-first."""
    makefile = (_ROOT / "Makefile").read_text(encoding="utf-8")
    mutating = makefile.replace(
        "poetry run ruff check --no-fix $(LINT_PATHS)",
        "poetry run ruff check --fix $(LINT_PATHS)",
        1,
    )
    reordered = makefile.replace(
        "poetry run black --check --preview "
        "--enable-unstable-feature=string_processing $(LINT_PATHS)\n\t"
        "poetry run ruff check --no-fix $(LINT_PATHS)",
        "poetry run ruff check --no-fix $(LINT_PATHS)\n\tpoetry run black "
        "--check --preview --enable-unstable-feature=string_processing "
        "$(LINT_PATHS)",
        1,
    )

    assert not _lint_contract_uses_black_before_ruff(mutating)
    assert not _lint_contract_uses_black_before_ruff(reordered)
