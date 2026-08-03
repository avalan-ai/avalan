"""Exercise the sealed conversation contract harness."""

from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from json import loads
from os import environ, pathsep
from pathlib import Path
from subprocess import CompletedProcess
from subprocess import run as subprocess_run
from sys import executable, modules
from sys import path as sys_path
from types import ModuleType
from typing import cast

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_DSN_ENV = "AVALAN_TASK_TEST_POSTGRESQL_DSN"
_LEGACY_LEASE_ENV = "AVALAN_TASK_TEST_POSTGRESQL_LEASE_SHA256"
_GENERATED_DATABASE = "avalan_task_test_0123456789abcdef0123456789abcdef"
_GENERATED_DSN = f"postgresql://test/{_GENERATED_DATABASE}"


def _load_runner() -> ModuleType:
    """Return the conversation gate runner module."""
    scripts = str(_ROOT / "scripts")
    if scripts not in sys_path:
        sys_path.insert(0, scripts)
    name = "_conversation_contract_runner"
    spec = spec_from_file_location(
        name, _ROOT / "scripts" / "run_conversation_contract_gate.py"
    )
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    modules[name] = module
    spec.loader.exec_module(module)
    return module


_RUNNER = _load_runner()


@pytest.fixture(autouse=True)
def _without_ambient_task_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep public runner tests free of caller-supplied task state."""
    monkeypatch.delenv(_DSN_ENV, raising=False)
    monkeypatch.delenv(_LEGACY_LEASE_ENV, raising=False)


def _repository(tmp_path: Path) -> Path:
    """Create the minimum sealed repository inventory."""
    for relative, content in (
        ("src/sample.py", "VALUE = 1\n"),
        ("tests/sample_test.py", "def test_value() -> None:\n    pass\n"),
        (
            "tests/fixtures/conversation/acceptance_manifest.phase5.json",
            '{"current_phase":5}\n',
        ),
        (
            "tests/fixtures/conversation/acceptance_manifest.phase6.json",
            '{"current_phase":6}\n',
        ),
        (
            "tests/fixtures/conversation/acceptance_manifest.phase9.json",
            '{"current_phase":9}\n',
        ),
        ("scripts/gate.py", "VALUE = 1\n"),
        ("Makefile", "test:\n\ttrue\n"),
        ("pyproject.toml", "[project]\nname = 'sample'\n"),
        ("poetry.lock", "# lock\n"),
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return tmp_path


def _initialize_git(root: Path) -> None:
    """Initialize one repository for nonignored mirror enumeration."""
    subprocess_run(("git", "init", "-q"), cwd=root, check=True)


def test_runner_seals_inputs_and_preserves_fresh_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run every gate command against one unchanged input inventory."""
    root = _repository(tmp_path)
    _initialize_git(root)
    calls: list[tuple[tuple[str, ...], dict[str, str]]] = []
    events: list[str] = []
    mirror_roots: list[Path] = []

    def run_command(
        command: tuple[str, ...],
        *,
        cwd: Path,
        check: bool,
        env: dict[str, str],
    ) -> CompletedProcess[str]:
        assert cwd != root
        assert (cwd / "src" / "sample.py").is_file()
        assert check is False
        assert env["PYTEST_ADDOPTS"] == ""
        assert env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
        assert env["COVERAGE_RCFILE"] == "/dev/null"
        mirror_roots.append(cwd)
        calls.append((command, env))
        events.append(f"run-{len(calls)}")
        if len(calls) == 1:
            (cwd / ".coverage").write_text("data", encoding="utf-8")
            (cwd / "coverage.json").write_text("{}\n", encoding="utf-8")
            (cwd / "coverage.xml").write_text(
                "<coverage/>\n", encoding="utf-8"
            )
        return CompletedProcess(command, 0)

    @contextmanager
    def owned_database() -> Iterator[object]:
        events.append("database-enter")
        try:
            yield _RUNNER.PostgreSQLTestDatabase(
                dsn=_GENERATED_DSN,
                name=_GENERATED_DATABASE,
            )
        finally:
            events.append("database-exit")

    monkeypatch.setattr(_RUNNER, "run", run_command)
    monkeypatch.setattr(_RUNNER, "_owned_postgresql_database", owned_database)

    assert _RUNNER.run_gate(0, repo_root=root) == 0
    assert len(calls) == 6
    commands = tuple(command for command, _environment in calls)
    assert commands[0][1:3] == ("-m", "pytest")
    assert commands[1][1:] == ("scripts/verify_src_coverage.py",)
    assert commands[2][0] == "jq"
    assert commands[3][1:] == (
        "scripts/verify_conversation_acceptance.py",
        "--through-phase",
        "0",
    )
    assert commands[4][1:] == (
        "scripts/verify_input_acceptance.py",
        "--through-phase",
        "12",
    )
    assert commands[5][1:] == ("scripts/verify_src_coverage.py",)
    assert all(_DSN_ENV not in environment for _, environment in calls[:4])
    assert calls[4][1][_DSN_ENV] == _GENERATED_DSN
    assert _DSN_ENV not in calls[5][1]
    startup_roots = tuple(
        Path(environment["PYTHONPATH"].split(pathsep)[0])
        for _command, environment in calls
    )
    assert len(set(startup_roots)) == len(calls)
    assert not any(path.exists() for path in startup_roots)
    assert events == [
        "run-1",
        "run-2",
        "run-3",
        "run-4",
        "database-enter",
        "run-5",
        "database-exit",
        "run-6",
    ]
    assert len(set(mirror_roots)) == 1
    assert not mirror_roots[0].exists()
    assert not (root / ".coverage").exists()
    assert (root / "coverage.json").is_file()
    assert (root / "coverage.xml").is_file()


def test_runner_stops_and_cleans_after_command_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop after the first failed command and remove partial reports."""
    root = _repository(tmp_path)
    _initialize_git(root)
    calls: list[tuple[str, ...]] = []

    def fail_command(
        command: tuple[str, ...],
        *,
        cwd: Path,
        check: bool,
        env: dict[str, str],
    ) -> CompletedProcess[str]:
        assert cwd != root
        assert check is False
        assert env
        calls.append(command)
        for name in (".coverage", "coverage.json", "coverage.xml"):
            (cwd / name).write_text("partial", encoding="utf-8")
        return CompletedProcess(command, 7)

    monkeypatch.setattr(_RUNNER, "run", fail_command)

    assert _RUNNER.run_gate(0, repo_root=root) == 7
    assert len(calls) == 1
    assert not any(
        (root / name).exists()
        for name in (".coverage", "coverage.json", "coverage.xml")
    )


def test_runner_rejects_mutation_even_when_command_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prefer sealed-input failure over a concurrent command failure."""
    root = _repository(tmp_path)
    _initialize_git(root)

    def mutate_input(
        command: tuple[str, ...],
        *,
        cwd: Path,
        check: bool,
        env: dict[str, str],
    ) -> CompletedProcess[str]:
        assert cwd != root
        assert check is False
        assert env
        (cwd / "tests" / "sample_test.py").write_text(
            "def test_value() -> None:\n    assert True\n",
            encoding="utf-8",
        )
        return CompletedProcess(command, 9)

    monkeypatch.setattr(_RUNNER, "run", mutate_input)

    with pytest.raises(
        _RUNNER.ContractGateError,
        match="measured gate inputs changed",
    ):
        _RUNNER.run_gate(0, repo_root=root)


def test_runner_rejects_input_mutation_before_next_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify repository inputs before creating another child runtime."""
    root = _repository(tmp_path)
    _initialize_git(root)
    calls = 0

    def mutate_input(
        command: tuple[str, ...],
        *,
        cwd: Path,
        check: bool,
        env: dict[str, str],
    ) -> CompletedProcess[str]:
        nonlocal calls
        calls += 1
        assert check is False
        assert env
        (cwd / "tests" / "sample_test.py").write_text(
            "def test_value() -> None:\n    assert True\n",
            encoding="utf-8",
        )
        return CompletedProcess(command, 0)

    monkeypatch.setattr(_RUNNER, "run", mutate_input)

    with pytest.raises(
        _RUNNER.ContractGateError,
        match="measured gate inputs changed",
    ):
        _RUNNER.run_gate(0, repo_root=root)
    assert calls == 1


def test_runner_rejects_runtime_startup_tamper_before_next_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop when one command poisons its disposable startup copy."""
    root = _repository(tmp_path)
    _initialize_git(root)
    calls = 0

    def tamper_runtime(
        command: tuple[str, ...],
        *,
        cwd: Path,
        check: bool,
        env: dict[str, str],
    ) -> CompletedProcess[str]:
        nonlocal calls
        calls += 1
        assert check is False
        startup = Path(env["PYTHONPATH"].split(pathsep)[0])
        (startup / "sitecustomize.py").write_text(
            "import os\nos._exit(0)\n",
            encoding="utf-8",
        )
        return CompletedProcess(command, 0)

    monkeypatch.setattr(_RUNNER, "run", tamper_runtime)

    with pytest.raises(
        _RUNNER.ContractGateError,
        match="runtime Python startup assets changed",
    ):
        _RUNNER.run_gate(0, repo_root=root)
    assert calls == 1


def test_runner_rejects_trusted_startup_tamper_before_next_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop before copying a mutated trusted startup source again."""
    root = _repository(tmp_path)
    trusted_startup = root / "scripts" / "contract_startup"
    trusted_startup.mkdir(parents=True)
    for name in ("sitecustomize.py", "avalan_contract_gate_plugin.py"):
        (trusted_startup / name).write_bytes(
            (_ROOT / "scripts" / "contract_startup" / name).read_bytes()
        )
    _initialize_git(root)
    calls = 0

    def tamper_source(
        command: tuple[str, ...],
        *,
        cwd: Path,
        check: bool,
        env: dict[str, str],
    ) -> CompletedProcess[str]:
        nonlocal calls
        calls += 1
        assert cwd != root
        assert check is False
        assert env
        (trusted_startup / "sitecustomize.py").write_text(
            "VALUE = 1\n",
            encoding="utf-8",
        )
        return CompletedProcess(command, 0)

    monkeypatch.setattr(_RUNNER, "repository_root", lambda: root)
    monkeypatch.setattr(_RUNNER, "run", tamper_source)

    with pytest.raises(
        _RUNNER.ContractGateError,
        match="trusted Python startup assets changed",
    ):
        _RUNNER.run_gate(0, repo_root=root)
    assert calls == 1


def test_runner_rejects_coverage_tampering_after_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject byte replacement of reports after their exact verification."""
    root = _repository(tmp_path)
    _initialize_git(root)
    calls = 0

    def tamper_after_seal(
        command: tuple[str, ...],
        *,
        cwd: Path,
        check: bool,
        env: dict[str, str],
    ) -> CompletedProcess[str]:
        nonlocal calls
        assert check is False
        assert env
        calls += 1
        if calls == 1:
            (cwd / "coverage.json").write_text("{}\n", encoding="utf-8")
            (cwd / "coverage.xml").write_text(
                "<coverage/>\n",
                encoding="utf-8",
            )
        if calls == 4:
            (cwd / "coverage.json").write_text(
                '{"fresh":true}\n',
                encoding="utf-8",
            )
        return CompletedProcess(command, 0)

    monkeypatch.setattr(_RUNNER, "run", tamper_after_seal)

    with pytest.raises(
        _RUNNER.ContractGateError,
        match="changed after exact verification",
    ):
        _RUNNER.run_gate(0, repo_root=root)
    assert calls == 4
    assert not (root / "coverage.json").exists()
    assert not (root / "coverage.xml").exists()


def test_runner_rejects_source_mutation_during_mirrored_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a concurrent worktree change made after mirror creation."""
    root = _repository(tmp_path)
    _initialize_git(root)
    calls = 0

    def mutate_source_after_snapshot(
        command: tuple[str, ...],
        *,
        cwd: Path,
        check: bool,
        env: dict[str, str],
    ) -> CompletedProcess[str]:
        nonlocal calls
        assert check is False
        assert env
        calls += 1
        if calls == 1:
            (cwd / "coverage.json").write_text("{}\n", encoding="utf-8")
            (cwd / "coverage.xml").write_text(
                "<coverage/>\n",
                encoding="utf-8",
            )
        if calls == 4:
            (root / "src" / "sample.py").write_text(
                "VALUE = 2\n",
                encoding="utf-8",
            )
        return CompletedProcess(command, 0)

    monkeypatch.setattr(_RUNNER, "run", mutate_source_after_snapshot)

    @contextmanager
    def owned_database() -> Iterator[object]:
        yield _RUNNER.PostgreSQLTestDatabase(
            dsn=_GENERATED_DSN,
            name=_GENERATED_DATABASE,
        )

    monkeypatch.setattr(_RUNNER, "_owned_postgresql_database", owned_database)

    with pytest.raises(
        _RUNNER.ContractGateError,
        match="nonignored repository inputs changed during mirrored execution",
    ):
        _RUNNER.run_gate(0, repo_root=root)


def test_input_acceptance_command_owns_exact_current_inventory() -> None:
    """Pin every active structured-input node selected by the common gate."""
    payload = cast(
        dict[str, object],
        loads(
            (
                _ROOT / "tests/fixtures/input/acceptance_manifest.json"
            ).read_text(encoding="utf-8")
        ),
    )
    assert payload["current_phase"] == _RUNNER._INPUT_CURRENT_PHASE == 12
    nodes = cast(list[dict[str, object]], payload["nodes"])
    node_ids = tuple(
        cast(str, node["node_id"])
        for node in nodes
        if node["lifecycle"] == "active"
        and cast(int, node["active_from_phase"])
        <= _RUNNER._INPUT_CURRENT_PHASE
    )
    assert len(node_ids) == len(frozenset(node_ids)) == 896
    assert (
        sha256("\n".join(node_ids).encode()).hexdigest()
        == "c4937f64415d1f604419d3c4b46157fe462fbfd03b139386f2560645602c5717"
    )


def test_phase_five_owns_database_across_entire_mirrored_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep one runner-created database alive around every mirrored command."""
    root = _repository(tmp_path)
    _initialize_git(root)
    manifest = (
        root / "tests/fixtures/conversation/acceptance_manifest.phase5.json"
    )
    manifest.write_text('{"current_phase":5}\n', encoding="utf-8")
    calls: list[tuple[tuple[str, ...], dict[str, str]]] = []
    events: list[str] = []

    def run_command(
        command: tuple[str, ...],
        *,
        cwd: Path,
        check: bool,
        env: dict[str, str],
    ) -> CompletedProcess[str]:
        assert cwd != root
        assert check is False
        assert env[_DSN_ENV] == _GENERATED_DSN
        calls.append((command, env))
        events.append(f"run-{len(calls)}")
        if len(calls) == 1:
            (cwd / "coverage.json").write_text("{}\n", encoding="utf-8")
            (cwd / "coverage.xml").write_text(
                "<coverage/>\n",
                encoding="utf-8",
            )
        return CompletedProcess(command, 0)

    @contextmanager
    def owned_database() -> Iterator[object]:
        events.append("database-enter")
        try:
            yield _RUNNER.PostgreSQLTestDatabase(
                dsn=_GENERATED_DSN,
                name=_GENERATED_DATABASE,
            )
        finally:
            assert (root / "coverage.json").is_file()
            assert (root / "coverage.xml").is_file()
            events.append("database-exit")

    monkeypatch.setattr(_RUNNER, "run", run_command)
    monkeypatch.setattr(_RUNNER, "_owned_postgresql_database", owned_database)
    assert _RUNNER.run_gate(5, repo_root=root) == 0
    assert len(calls) == 6
    assert events == [
        "database-enter",
        "run-1",
        "run-2",
        "run-3",
        "run-4",
        "run-5",
        "run-6",
        "database-exit",
    ]


@pytest.mark.parametrize("through_phase", (-1, 10))
def test_runner_rejects_unimplemented_phase_before_cleanup_or_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    through_phase: int,
) -> None:
    """Reject the closed phase range before touching reports or pytest."""
    root = _repository(tmp_path)
    _initialize_git(root)
    report = root / "coverage.json"
    report.write_text("stale\n", encoding="utf-8")
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        _RUNNER,
        "run",
        lambda *args, **kwargs: calls.append(args[0]),
    )

    with pytest.raises(_RUNNER.ContractGateError, match="range 0..9"):
        _RUNNER.run_gate(through_phase, repo_root=root)
    assert report.read_text(encoding="utf-8") == "stale\n"
    assert calls == []


@pytest.mark.parametrize("name", (_DSN_ENV, _LEGACY_LEASE_ENV))
def test_public_runner_rejects_all_ambient_task_database_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    """Reject even plausible caller task state before cleanup or execution."""
    root = _repository(tmp_path)
    _initialize_git(root)
    report = root / "coverage.json"
    report.write_text("stale\n", encoding="utf-8")
    value = _GENERATED_DSN if name == _DSN_ENV else "0" * 64
    monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        _RUNNER,
        "_owned_postgresql_database",
        lambda: pytest.fail(
            "ambient state must fail before database creation"
        ),
    )

    with pytest.raises(
        _RUNNER.ContractGateError,
        match="rejects caller-supplied task PostgreSQL state",
    ):
        _RUNNER.run_gate(0, repo_root=root)

    assert report.read_text(encoding="utf-8") == "stale\n"


def test_database_context_is_phase_correct_and_not_digest_claimed() -> None:
    """Accept only a typed internal context for the selected phase."""
    database = _RUNNER.PostgreSQLTestDatabase(
        dsn=_GENERATED_DSN,
        name=_GENERATED_DATABASE,
    )
    _RUNNER._require_database_context(0, None)
    _RUNNER._require_database_context(3, database)
    for phase, candidate in ((0, database), (3, None), (3, object())):
        with pytest.raises(
            _RUNNER.ContractGateError,
            match="database ownership does not match",
        ):
            _RUNNER._require_database_context(phase, candidate)
    assert not hasattr(_RUNNER, "postgresql_test_lease_sha256")


def test_makefile_exposes_explicit_conversation_gates() -> None:
    """Pin separate static, non-database, and PostgreSQL gate targets."""
    makefile = (_ROOT / "Makefile").read_text(encoding="utf-8")
    assert (
        "CONVERSATION_CONTRACT_SCRIPTS := scripts/contract_gate.py" in makefile
    )
    assert (
        "scripts/contract_startup/avalan_contract_gate_plugin.py" in makefile
    )
    assert "scripts/contract_startup/sitecustomize.py" in makefile
    assert "CONTRACT_PYTHON_ENV := PYTHONSAFEPATH=1" in makefile
    assert "PYTHONNOUSERSITE=1" in makefile
    assert "AVALAN_CONTRACT_ALLOWED_PYTHONPATH=" in makefile
    assert "test-conversation-exact:" in makefile
    assert "test-conversation-pgsql-exact:" in makefile
    assert "typecheck-conversation-contract:" in makefile
    runner = "scripts/run_conversation_contract_gate.py"
    phase = "$(CONVERSATION_PHASE)"
    exact_command = f"{runner} --through-phase {phase}"
    preflight_command = f"{runner} --preflight --through-phase {phase}"
    assert exact_command in makefile
    assert (
        "poetry run env $(CONTRACT_PYTHON_ENV) python " + exact_command
        in makefile
    )
    assert makefile.count(preflight_command) == 2
    exact_target = makefile.split("test-conversation-exact:", 1)[1].split(
        "test-conversation-pgsql-exact:", 1
    )[0]
    exact_guard = 'test "$(CONVERSATION_PHASE)" -lt 3'
    assert exact_guard in exact_target
    assert "poetry run python -m pip install $(TASK_PGSQL_TEST_DEPS)" in (
        exact_target
    )
    assert exact_target.index(exact_guard) < exact_target.index(
        preflight_command
    )
    assert exact_target.index(preflight_command) < exact_target.index(
        "poetry sync"
    )
    assert exact_target.index(preflight_command) < exact_target.index(
        "python -m pip install"
    )
    pgsql_section = makefile.split("test-conversation-pgsql-exact:", 1)[1]
    pgsql_target = pgsql_section.split("typecheck-conversation-contract:", 1)[
        0
    ]
    pgsql_guard = 'test "$(CONVERSATION_PHASE)" -ge 3'
    assert pgsql_guard in pgsql_target
    assert "task_pgsql_test_database.py" not in pgsql_target
    assert pgsql_target.index(pgsql_guard) < pgsql_target.index(
        preflight_command
    )
    assert pgsql_target.index(preflight_command) < pgsql_target.index(
        "poetry sync"
    )
    assert pgsql_target.index(preflight_command) < pgsql_target.index(
        "python -m pip install"
    )


@pytest.mark.parametrize(
    ("target", "phase"),
    (
        ("test-conversation-exact", "-1"),
        ("test-conversation-pgsql-exact", "10"),
    ),
)
def test_make_preflight_rejects_closed_phase_before_installation(
    tmp_path: Path,
    target: str,
    phase: str,
) -> None:
    """Run the lightweight closed-range check before dependency commands."""
    executable_path = tmp_path / "poetry"
    log = tmp_path / "poetry.log"
    executable_path.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$AVALAN_PREFLIGHT_LOG"\n'
        'if [ "$1" = "run" ] && [ "$2" = "env" ]; then\n'
        "  shift 2\n"
        '  while [ "$1" != "python" ]; do\n'
        '    export "$1"\n'
        "    shift\n"
        "  done\n"
        "  shift\n"
        '  exec "$AVALAN_PREFLIGHT_PYTHON" "$@"\n'
        "fi\n"
        "exit 97\n",
        encoding="utf-8",
    )
    executable_path.chmod(0o755)
    child_environment = environ.copy()
    child_environment["PATH"] = f"{tmp_path}:/usr/bin:/bin"
    child_environment["AVALAN_PREFLIGHT_LOG"] = str(log)
    child_environment["AVALAN_PREFLIGHT_PYTHON"] = executable

    completed = subprocess_run(
        (
            "/usr/bin/make",
            target,
            f"CONVERSATION_PHASE={phase}",
        ),
        cwd=_ROOT,
        capture_output=True,
        check=False,
        env=child_environment,
        text=True,
    )

    assert completed.returncode != 0
    invocations = log.read_text(encoding="utf-8").splitlines()
    assert len(invocations) == 1
    invocation = invocations[0]
    assert invocation.startswith(
        "run env PYTHONSAFEPATH=1 PYTHONNOUSERSITE=1 "
        "PYTHONDONTWRITEBYTECODE=1 "
    )
    assert "AVALAN_CONTRACT_ALLOWED_PYTHONPATH=" in invocation
    assert invocation.endswith(
        "python scripts/run_conversation_contract_gate.py "
        f"--preflight --through-phase {phase}"
    )
