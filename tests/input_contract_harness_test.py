"""Exercise the database and common structured-input gate harnesses."""

from importlib.util import module_from_spec, spec_from_file_location
from os import environ, pathsep
from pathlib import Path
from subprocess import CompletedProcess
from sys import executable, modules
from sys import path as sys_path
from types import ModuleType
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_GENERATED_DATABASE = "contract_0123456789abcdef0123456789abcdef"


def _load_script(name: str) -> ModuleType:
    """Return one repository harness script as a module."""
    scripts = str(_ROOT / "scripts")
    if scripts not in sys_path:
        sys_path.insert(0, scripts)
    module_name = f"_input_contract_harness_{name}"
    spec = spec_from_file_location(
        module_name, _ROOT / "scripts" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_DATABASE = _load_script("task_pgsql_test_database")
_GATE = _load_script("run_input_contract_gate")


def test_admin_dsn_runner_script_uses_ephemeral_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward the runner and generated test DSN through admin mode."""
    operations: list[tuple[str, str, str | None]] = []
    observed: dict[str, Any] = {}
    lease_env = "AVALAN_TASK_TEST_POSTGRESQL_LEASE_SHA256"
    monkeypatch.setenv(lease_env, "ambient-marker")
    monkeypatch.setattr(_DATABASE, "_require_runtime_modules", lambda: None)
    monkeypatch.setattr(
        _DATABASE, "_database_name", lambda prefix: _GENERATED_DATABASE
    )
    monkeypatch.setattr(
        _DATABASE,
        "_create_database",
        lambda dsn, name: operations.append(("create", dsn, name)),
    )
    monkeypatch.setattr(
        _DATABASE,
        "_drop_database",
        lambda dsn, name: operations.append(("drop", dsn, name)),
    )

    def run_child(
        command: tuple[str, ...],
        *,
        check: bool,
        env: dict[str, str],
    ) -> CompletedProcess[str]:
        observed["command"] = command
        observed["check"] = check
        observed["dsn"] = env["AVALAN_TASK_TEST_POSTGRESQL_DSN"]
        observed["lease_present"] = lease_env in env
        operations.append(("run", observed["dsn"], None))
        return CompletedProcess(command, 0)

    monkeypatch.setattr(_DATABASE, "run", run_child)
    code = _DATABASE._run_with_admin_dsn(
        "postgresql://admin:secret@db.example/postgres?sslmode=require",
        "contract",
        ("--through-phase", "0"),
        runner_script="scripts/run_input_contract_gate.py",
    )
    assert code == 0
    assert observed["command"] == (
        executable,
        "scripts/run_input_contract_gate.py",
        "--through-phase",
        "0",
    )
    assert observed["check"] is False
    expected_dsn = (
        "postgresql://admin:secret@db.example/"
        f"{_GENERATED_DATABASE}?sslmode=require"
    )
    assert observed["dsn"] == expected_dsn
    assert observed["lease_present"] is False
    assert environ[lease_env] == "ambient-marker"
    assert operations == [
        (
            "create",
            "postgresql://admin:secret@db.example/postgres?sslmode=require",
            _GENERATED_DATABASE,
        ),
        (
            "run",
            expected_dsn,
            None,
        ),
        (
            "drop",
            "postgresql://admin:secret@db.example/postgres?sslmode=require",
            _GENERATED_DATABASE,
        ),
    ]


@pytest.mark.parametrize(
    "sslmode",
    ("disable", "allow", "prefer", "require", "verify-ca", "verify-full"),
)
def test_database_dsn_accepts_only_supported_sslmode_values(
    sslmode: str,
) -> None:
    """Preserve each exact lowercase libpq sslmode in the test DSN."""
    admin_dsn = (
        f"postgresql://admin:secret@db.example/postgres?sslmode={sslmode}"
    )

    assert (
        _DATABASE._database_dsn(admin_dsn, _GENERATED_DATABASE)
        == "postgresql://admin:secret@db.example/"
        f"{_GENERATED_DATABASE}?sslmode={sslmode}"
    )


@pytest.mark.parametrize(
    "admin_dsn",
    (
        "http://admin:secret@db.example/postgres",
        "postgresql:///postgres",
        "postgresql://:secret@db.example/postgres",
        "postgresql://admin:@db.example/postgres",
        "postgresql://db.example:invalid/postgres",
        "postgresql://db.example/postgres?dbname=other",
        "postgresql://db.example/postgres?hostaddr=127.0.0.1",
        "postgresql://db.example/postgres?service=production",
        "postgresql://db.example/postgres?servicefile=pg_service.conf",
        "postgresql://db.example/postgres?passfile=pgpass",
        "postgresql://db.example/postgres?application_name=contract",
        "postgresql://db.example/postgres?SSLMode=require",
        "postgresql://db.example/postgres?%73slmode=require",
        "postgresql://db.example/postgres?sslmode=REQUIRE",
        "postgresql://db.example/postgres?sslmode=%72equire",
        "postgresql://db.example/postgres?sslmode=invalid",
        "postgresql://db.example/postgres?sslmode=",
        "postgresql://db.example/postgres?=require",
        "postgresql://db.example/postgres?sslmode=require&sslmode=disable",
        "postgresql://db.example/postgres?sslmode=require&sslmode=require",
        "postgresql://db.example/postgres#fragment",
    ),
)
def test_database_dsn_rejects_ambiguous_admin_url(admin_dsn: str) -> None:
    """Use the same strict PostgreSQL URL boundary for admin DSNs."""
    with pytest.raises(SystemExit, match="unambiguous PostgreSQL URL"):
        _DATABASE._database_dsn(admin_dsn, _GENERATED_DATABASE)


def test_main_selects_admin_dsn_and_runner_script(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prefer the explicit admin DSN and preserve child arguments."""
    observed: dict[str, object] = {}

    def run_admin(
        dsn: str,
        prefix: str,
        child_args: tuple[str, ...],
        *,
        runner_script: str | None,
    ) -> int:
        observed.update(
            dsn=dsn,
            prefix=prefix,
            child_args=child_args,
            runner_script=runner_script,
        )
        return 4

    monkeypatch.setattr(_DATABASE, "_run_with_admin_dsn", run_admin)
    monkeypatch.setattr(
        _DATABASE,
        "_run_with_docker",
        lambda *args, **kwargs: pytest.fail("Docker must not be selected"),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "task_pgsql_test_database.py",
            "--admin-dsn",
            "postgresql://explicit/postgres",
            "--database-prefix",
            "selected",
            "--runner-script",
            "scripts/run_input_contract_gate.py",
            "--",
            "--through-phase",
            "0",
        ],
    )
    assert _DATABASE.main() == 4
    assert observed == {
        "dsn": "postgresql://explicit/postgres",
        "prefix": "selected",
        "child_args": ("--through-phase", "0"),
        "runner_script": "scripts/run_input_contract_gate.py",
    }


def test_docker_forwards_runner_and_always_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward the runner through Docker and stop the started container."""
    events: list[tuple[object, ...]] = []
    lease_env = "AVALAN_TASK_TEST_POSTGRESQL_LEASE_SHA256"
    monkeypatch.setenv(lease_env, "ambient-marker")
    monkeypatch.setattr(_DATABASE, "_require_runtime_modules", lambda: None)
    monkeypatch.setattr(
        _DATABASE, "_docker_container_name", lambda: "container"
    )
    monkeypatch.setattr(_DATABASE, "token_urlsafe", lambda size: "password")
    monkeypatch.setattr(_DATABASE, "_free_tcp_port", lambda: 55432)
    monkeypatch.setattr(
        _DATABASE,
        "_database_name",
        lambda prefix: _GENERATED_DATABASE,
    )
    monkeypatch.setattr(
        _DATABASE,
        "_start_docker_postgres",
        lambda **values: events.append(("start", values)),
    )
    monkeypatch.setattr(
        _DATABASE,
        "_wait_for_database",
        lambda dsn, timeout: events.append(("wait", dsn, timeout)),
    )

    monkeypatch.setattr(
        _DATABASE,
        "_create_database",
        lambda dsn, name: events.append(("create", dsn, name)),
    )
    monkeypatch.setattr(
        _DATABASE,
        "_drop_database",
        lambda dsn, name: events.append(("drop", dsn, name)),
    )

    def run_child(
        command: tuple[str, ...],
        *,
        check: bool,
        env: dict[str, str],
    ) -> CompletedProcess[str]:
        assert check is False
        assert lease_env not in env
        event = (
            "run",
            command,
            env["AVALAN_TASK_TEST_POSTGRESQL_DSN"],
        )
        events.append(event)
        return CompletedProcess(command, 6)

    monkeypatch.setattr(_DATABASE, "run", run_child)
    monkeypatch.setattr(
        _DATABASE,
        "_stop_docker_container",
        lambda name: events.append(("stop", name)),
    )
    code = _DATABASE._run_with_docker(
        "contract",
        ("--through-phase", "0"),
        image="postgres:16-alpine",
        timeout_seconds=12.0,
        runner_script="scripts/run_input_contract_gate.py",
    )
    assert code == 6
    assert events[0] == (
        "start",
        {
            "image": "postgres:16-alpine",
            "name": "container",
            "password": "password",
            "port": 55432,
        },
    )
    assert events[1] == (
        "wait",
        "postgresql://postgres:password@127.0.0.1:55432/postgres",
        12.0,
    )
    admin_dsn = "postgresql://postgres:password@127.0.0.1:55432/postgres"
    test_dsn = (
        f"postgresql://postgres:password@127.0.0.1:55432/{_GENERATED_DATABASE}"
    )
    assert events[2] == (
        "create",
        admin_dsn,
        _GENERATED_DATABASE,
    )
    assert events[3] == (
        "run",
        (
            executable,
            "scripts/run_input_contract_gate.py",
            "--through-phase",
            "0",
        ),
        test_dsn,
    )
    assert events[4] == (
        "drop",
        admin_dsn,
        _GENERATED_DATABASE,
    )
    assert events[5] == ("stop", "container")


def test_owned_database_drops_after_body_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drop the process-owned database when its enclosed gate raises."""
    events: list[tuple[str, str]] = []
    monkeypatch.setattr(_DATABASE, "_require_runtime_modules", lambda: None)
    monkeypatch.setattr(
        _DATABASE,
        "_database_name",
        lambda prefix: _GENERATED_DATABASE,
    )
    monkeypatch.setattr(
        _DATABASE,
        "_create_database",
        lambda dsn, name: events.append(("create", name)),
    )
    monkeypatch.setattr(
        _DATABASE,
        "_drop_database",
        lambda dsn, name: events.append(("drop", name)),
    )

    with pytest.raises(RuntimeError, match="gate failed"):
        with _DATABASE.postgresql_test_database(
            admin_dsn="postgresql://admin/postgres",
            database_prefix="contract",
            docker=False,
        ) as database:
            assert database.name == _GENERATED_DATABASE
            raise RuntimeError("gate failed")

    assert events == [
        ("create", _GENERATED_DATABASE),
        ("drop", _GENERATED_DATABASE),
    ]


def test_owned_database_fails_closed_when_drop_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Turn cleanup failure into a closed contract-gate failure."""
    monkeypatch.setattr(_DATABASE, "_require_runtime_modules", lambda: None)
    monkeypatch.setattr(
        _DATABASE,
        "_database_name",
        lambda prefix: _GENERATED_DATABASE,
    )
    monkeypatch.setattr(_DATABASE, "_create_database", lambda dsn, name: None)

    def fail_drop(dsn: str, name: str) -> None:
        raise RuntimeError("drop failed")

    monkeypatch.setattr(_DATABASE, "_drop_database", fail_drop)

    with pytest.raises(
        _DATABASE.ContractGateError,
        match="Unable to drop PostgreSQL test database",
    ):
        with _DATABASE.postgresql_test_database(
            admin_dsn="postgresql://admin/postgres",
            database_prefix="contract",
            docker=False,
        ):
            pass


def test_common_gate_cleans_partial_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Delete stale input and partial output when verification fails."""
    stale = tmp_path / ".coverage.stale"
    stale.write_text("stale", encoding="utf-8")
    calls: list[tuple[str, ...]] = []
    startup_roots: list[Path] = []

    def run_command(
        command: tuple[str, ...],
        *,
        cwd: Path,
        check: bool,
        env: dict[str, str],
    ) -> CompletedProcess[str]:
        assert cwd == tmp_path
        assert check is False
        python_paths = env["PYTHONPATH"].split(":")
        assert len(python_paths) == 1
        assert Path(python_paths[0]).name == "python-startup"
        assert env["AVALAN_CONTRACT_ALLOWED_PYTHONPATH"] == env["PYTHONPATH"]
        assert env["PYTHONSAFEPATH"] == "1"
        assert env["PYTHONNOUSERSITE"] == "1"
        assert env["PWD"] == str(tmp_path)
        assert not stale.exists()
        startup_roots.append(Path(env["PYTHONPATH"].split(pathsep)[0]))
        calls.append(command)
        if len(calls) == 1:
            (tmp_path / ".coverage").write_text("partial", encoding="utf-8")
            (tmp_path / "coverage.json").write_text(
                "partial", encoding="utf-8"
            )
            (tmp_path / "coverage.xml").write_text("partial", encoding="utf-8")
            return CompletedProcess(command, 0)
        return CompletedProcess(command, 5)

    monkeypatch.setattr(_GATE, "run", run_command)
    monkeypatch.setenv("PYTHONPATH", "ambient")
    assert _GATE.run_coverage_gate(repo_root=tmp_path) == 5
    assert len(calls) == 2
    assert len(set(startup_roots)) == len(calls)
    assert not any(path.exists() for path in startup_roots)
    assert calls[0][1:3] == ("-m", "pytest")
    assert calls[1][1:] == ("scripts/verify_src_coverage.py",)
    assert not any(
        (tmp_path / name).exists()
        for name in (".coverage", "coverage.json", "coverage.xml")
    )


def test_common_gate_rejects_runtime_tamper_and_cleans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop before a poisoned input-gate runtime can be reused."""
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
        assert cwd == tmp_path
        assert check is False
        startup = Path(env["PYTHONPATH"].split(pathsep)[0])
        (startup / "sitecustomize.py").write_text(
            "import os\nos._exit(0)\n",
            encoding="utf-8",
        )
        (tmp_path / "coverage.json").write_text("partial\n", encoding="utf-8")
        return CompletedProcess(command, 0)

    monkeypatch.setattr(_GATE, "run", tamper_runtime)

    with pytest.raises(
        _GATE.ContractGateError,
        match="runtime Python startup assets changed",
    ):
        _GATE.run_coverage_gate(repo_root=tmp_path)
    assert calls == 1
    assert not (tmp_path / "coverage.json").exists()


def test_common_gate_skips_acceptance_after_coverage_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Return the coverage failure without starting acceptance."""
    monkeypatch.setattr(_GATE, "run_coverage_gate", lambda *, repo_root: 11)
    monkeypatch.setattr(
        _GATE,
        "run",
        lambda *args, **kwargs: pytest.fail("acceptance must not run"),
    )
    assert _GATE.run_gate(0, repo_root=tmp_path) == 11
